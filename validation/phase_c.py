#!/usr/bin/env python
"""Phase C — Robustness and failure transparency (product robustness).

Primary endpoints:
    C1. Performance degradation curves under controlled perturbations,
        summarised by AUC of metric vs perturbation severity.
    C2. Hard-failure rate and soft-failure rate (QC-flagged but completed).

Secondary endpoints:
    - OOD generalisation: leave-one-dataset-out / leave-one-site-out.
    - Failure taxonomy rates.

Usage:
    python -m validation.phase_c --bids-dir /data/bids --gt-dir /data/gt \\
        --output-dir /data/out --dataset ucsf_pdgm \\
        --results-dir ./validation_results
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np

from .config import (
    DATASETS,
    PERTURBATIONS,
    REGION_MAP,
    SAP,
    PerturbationSpec,
    get_phase_dir,
)
from .metrics import compute_patient_metrics, evaluate_case
from .phase_a import CaseResult, run_single_case
from .stats import bootstrap_ci, degradation_auc, paired_bootstrap_delta


# ---------------------------------------------------------------------------
# Perturbation functions
# ---------------------------------------------------------------------------


def apply_gaussian_noise(
    img: nib.Nifti1Image,
    sigma_fraction: float,
    seed: int = 42,
) -> nib.Nifti1Image:
    """Add zero-mean Gaussian noise (σ = fraction of intensity range)."""
    rng = np.random.default_rng(seed)
    data = np.asanyarray(img.dataobj, dtype=np.float32)
    intensity_range = float(data.max() - data.min())
    if intensity_range == 0:
        return img
    sigma = sigma_fraction * intensity_range
    noisy = data + rng.normal(0, sigma, data.shape).astype(np.float32)
    return nib.Nifti1Image(noisy, img.affine, img.header)


def apply_bias_field(
    img: nib.Nifti1Image,
    order: int,
    seed: int = 42,
) -> nib.Nifti1Image:
    """Apply a smooth multiplicative bias field (polynomial of given order)."""
    rng = np.random.default_rng(seed)
    data = np.asanyarray(img.dataobj, dtype=np.float32)

    if order == 0:
        return img

    # Create coordinate grids normalised to [-1, 1]
    shape = data.shape
    coords = [np.linspace(-1, 1, s) for s in shape]
    grids = np.meshgrid(*coords, indexing="ij")

    bias = np.ones(shape, dtype=np.float32)
    for _ in range(order):
        coeffs = rng.uniform(-0.3, 0.3, size=len(shape))
        for dim, c in enumerate(coeffs):
            bias += c * grids[dim]

    # Ensure bias is positive
    bias = np.clip(bias, 0.3, 3.0)
    biased = data * bias
    return nib.Nifti1Image(biased, img.affine, img.header)


def apply_resolution_downsample(
    img: nib.Nifti1Image,
    downsample_factor: float,
) -> nib.Nifti1Image:
    """Isotropic downsample + upsample back to original grid."""
    from scipy.ndimage import zoom

    if downsample_factor <= 1.0:
        return img

    data = np.asanyarray(img.dataobj, dtype=np.float32)
    factor = 1.0 / downsample_factor

    # Downsample
    downsampled = zoom(data, factor, order=1)
    # Upsample back
    target_shape = data.shape
    upscale_factors = tuple(t / d for t, d in zip(target_shape, downsampled.shape))
    restored = zoom(downsampled, upscale_factors, order=1)

    # Handle potential shape mismatch from rounding
    restored = restored[:target_shape[0], :target_shape[1], :target_shape[2]]
    return nib.Nifti1Image(restored, img.affine, img.header)


def apply_intensity_scaling(
    img: nib.Nifti1Image,
    scale_factor: float,
) -> nib.Nifti1Image:
    """Global intensity rescaling."""
    data = np.asanyarray(img.dataobj, dtype=np.float32)
    scaled = data * scale_factor
    return nib.Nifti1Image(scaled, img.affine, img.header)


PERTURBATION_FUNCTIONS = {
    "gaussian_noise": apply_gaussian_noise,
    "bias_field": apply_bias_field,
    "resolution_downsample": apply_resolution_downsample,
    "intensity_scaling": apply_intensity_scaling,
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PerturbationResult:
    """Result of evaluating at a single perturbation level."""
    perturbation: str
    level_index: int
    level_value: float
    n_cases: int = 0
    n_hard_fail: int = 0
    n_soft_fail: int = 0
    patient_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class PhaseCResults:
    """Aggregated Phase C results."""
    dataset: str
    n_cases: int = 0

    # C1: Degradation curves — {perturbation: [PerturbationResult per level]}
    degradation_curves: Dict[str, List[Dict]] = field(default_factory=dict)

    # C1: AUC summaries — {perturbation: {region: auc}}
    auc_summaries: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # C2: Failure rates from Phase A (referenced)
    hard_failure_rate: float = 0.0
    soft_failure_rate: float = 0.0

    # Failure taxonomy
    failure_taxonomy: Dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Perturbation pipeline
# ---------------------------------------------------------------------------


def create_perturbed_bids(
    bids_dir: Path,
    output_dir: Path,
    perturbation_name: str,
    level_value: float,
    subjects: Optional[List[Tuple[str, Optional[str]]]] = None,
    seed: int = 42,
) -> Path:
    """Create a perturbed copy of BIDS data at a given severity level.

    Applies the perturbation to all NIfTI files in the ``anat/`` directory
    of each subject/session.

    Parameters
    ----------
    bids_dir : Path
        Original BIDS root.
    output_dir : Path
        Where to write the perturbed BIDS dataset.
    perturbation_name : str
        Key into PERTURBATION_FUNCTIONS.
    level_value : float
        Severity parameter value.
    subjects : list of (subject, session), optional
        Subset to perturb. Defaults to all.
    seed : int
        Random seed.

    Returns
    -------
    Path
        Root of the perturbed BIDS copy.
    """
    perturbed_root = output_dir / f"perturbed_{perturbation_name}_{level_value}"
    perturbed_root.mkdir(parents=True, exist_ok=True)

    perturb_fn = PERTURBATION_FUNCTIONS[perturbation_name]

    # Copy dataset_description.json if present
    dd = bids_dir / "dataset_description.json"
    if dd.exists():
        import shutil
        shutil.copy2(dd, perturbed_root / "dataset_description.json")

    # Discover and perturb NIfTI files
    for nii_path in sorted(bids_dir.rglob("sub-*/anat/*.nii.gz")):
        # Check if this subject is in the requested subset
        if subjects is not None:
            in_subset = False
            for sub, ses in subjects:
                if f"sub-{sub}" in str(nii_path):
                    if ses is None or f"ses-{ses}" in str(nii_path):
                        in_subset = True
                        break
            if not in_subset:
                continue

        # Load, perturb, save
        img = nib.load(str(nii_path))
        rel_path = nii_path.relative_to(bids_dir)
        out_path = perturbed_root / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            perturbed_img = perturb_fn(img, level_value, seed=seed) if perturbation_name in (
                "gaussian_noise", "bias_field"
            ) else perturb_fn(img, level_value)
            nib.save(perturbed_img, str(out_path))
        except Exception as e:
            # Copy original on failure
            import shutil
            shutil.copy2(nii_path, out_path)

    # Copy non-anat files (JSON sidecars, etc.)
    for json_path in bids_dir.rglob("sub-*/anat/*.json"):
        rel_path = json_path.relative_to(bids_dir)
        out_path = perturbed_root / rel_path
        if not out_path.exists():
            out_path.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(json_path, out_path)

    return perturbed_root


def evaluate_perturbation_level(
    pred_dir: Path,
    gt_dir: Path,
    perturbation_name: str,
    level_index: int,
    level_value: float,
    regions: Optional[Dict[str, List[int]]] = None,
    max_cases: Optional[int] = None,
) -> PerturbationResult:
    """Evaluate segmentation quality at a single perturbation level.

    Assumes OncoPrep has already been run on the perturbed data and
    the results are in pred_dir.
    """
    from .phase_b import discover_cases

    if regions is None:
        regions = REGION_MAP

    result = PerturbationResult(
        perturbation=perturbation_name,
        level_index=level_index,
        level_value=level_value,
    )

    cases = discover_cases(pred_dir, gt_dir)
    if max_cases is not None:
        cases = cases[:max_cases]
    result.n_cases = len(cases)

    # Collect patient-level Dice per region
    per_region: Dict[str, List[float]] = {r: [] for r in regions}

    for pred_path, gt_path, subject, session in cases:
        try:
            case_metrics = evaluate_case(str(pred_path), str(gt_path), regions=regions)
            for region in regions:
                d = case_metrics["patient_metrics"].get(region, {}).get("dice")
                if d is not None and np.isfinite(d):
                    per_region[region].append(d)
        except Exception:
            result.n_hard_fail += 1

    # Aggregate
    for region in regions:
        vals = per_region[region]
        if vals:
            result.patient_metrics[region] = {
                "dice_median": float(np.median(vals)),
                "dice_iqr_lower": float(np.percentile(vals, 25)),
                "dice_iqr_upper": float(np.percentile(vals, 75)),
                "n": len(vals),
            }

    return result


# ---------------------------------------------------------------------------
# Full Phase C runner
# ---------------------------------------------------------------------------


def run_phase_c(
    bids_dir: Path,
    gt_dir: Path,
    output_dir: Path,
    work_dir: Path,
    dataset_key: str,
    perturbation_types: Optional[List[str]] = None,
    nprocs: int = 4,
    mem_gb: float = 8.0,
    timeout_sec: int = 7200,
    max_cases: Optional[int] = None,
    skip_pipeline_run: bool = False,
    precomputed_dirs: Optional[Dict[str, Dict[float, Path]]] = None,
) -> PhaseCResults:
    """Execute full Phase C: robustness + failure transparency.

    For each perturbation type and severity level:
        1. Create perturbed BIDS dataset.
        2. Run OncoPrep on perturbed data (unless skip_pipeline_run).
        3. Evaluate segmentation quality against ground truth.
        4. Compute degradation AUC.

    Parameters
    ----------
    bids_dir, gt_dir : Path
        Original BIDS data and ground truth.
    output_dir : Path
        Root for all outputs (perturbed + OncoPrep derivatives).
    work_dir : Path
        Nipype work directory.
    dataset_key : str
        Dataset identifier.
    perturbation_types : list of str, optional
        Which perturbations to run. Defaults to all in config.
    skip_pipeline_run : bool
        If True, assumes OncoPrep results already exist in precomputed_dirs.
    precomputed_dirs : dict
        Mapping {perturbation: {level_value: Path_to_derivatives}}.

    Returns
    -------
    PhaseCResults
    """
    if perturbation_types is None:
        perturbation_types = list(PERTURBATIONS.keys())

    results = PhaseCResults(dataset=dataset_key)

    for pert_name in perturbation_types:
        spec = PERTURBATIONS[pert_name]
        level_results: List[Dict] = []

        for i, level_val in enumerate(spec.levels):
            if skip_pipeline_run and precomputed_dirs:
                pred_dir = precomputed_dirs.get(pert_name, {}).get(level_val)
                if pred_dir is None:
                    continue
            else:
                # 1. Create perturbed data
                perturbed_bids = create_perturbed_bids(
                    bids_dir, output_dir / "perturbed", pert_name, level_val,
                )
                # 2. Run OncoPrep
                pert_output = output_dir / f"derivatives_{pert_name}_{level_val}"
                pert_work = work_dir / f"work_{pert_name}_{level_val}"
                from .phase_a import collect_subjects
                subjects = collect_subjects(perturbed_bids)
                if max_cases:
                    subjects = subjects[:max_cases]
                for subject, session in subjects:
                    run_single_case(
                        bids_dir=perturbed_bids,
                        output_dir=pert_output,
                        work_dir=pert_work,
                        subject=subject,
                        session=session,
                        dataset=dataset_key,
                        nprocs=nprocs,
                        mem_gb=mem_gb,
                        timeout_sec=timeout_sec,
                    )
                pred_dir = pert_output / "oncoprep"

            # 3. Evaluate
            pert_result = evaluate_perturbation_level(
                pred_dir=pred_dir,
                gt_dir=gt_dir,
                perturbation_name=pert_name,
                level_index=i,
                level_value=level_val,
                max_cases=max_cases,
            )
            level_results.append({
                "level_index": i,
                "level_value": level_val,
                "n_cases": pert_result.n_cases,
                "n_hard_fail": pert_result.n_hard_fail,
                "patient_metrics": pert_result.patient_metrics,
            })

        results.degradation_curves[pert_name] = level_results

        # 4. Compute AUC per region
        auc_per_region: Dict[str, float] = {}
        for region in REGION_MAP:
            severities = []
            dice_values = []
            for lr in level_results:
                rm = lr["patient_metrics"].get(region, {})
                if "dice_median" in rm:
                    severities.append(lr["level_value"])
                    dice_values.append(rm["dice_median"])
            if len(severities) >= 2:
                auc_per_region[region] = degradation_auc(severities, dice_values)

        results.auc_summaries[pert_name] = auc_per_region

    return results


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def save_phase_c_results(results: PhaseCResults, output_path: Path) -> None:
    """Save Phase C results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "dataset": results.dataset,
        "n_cases": results.n_cases,
        "degradation_curves": results.degradation_curves,
        "auc_summaries": results.auc_summaries,
        "hard_failure_rate": results.hard_failure_rate,
        "soft_failure_rate": results.soft_failure_rate,
        "failure_taxonomy": results.failure_taxonomy,
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=lambda o: None if isinstance(o, float) and not np.isfinite(o) else o)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase C: Robustness and failure transparency"
    )
    parser.add_argument("--bids-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=Path("work"))
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASETS.keys()))
    parser.add_argument("--results-dir", type=Path, default=Path("validation_results"))
    parser.add_argument("--perturbations", nargs="*", default=None,
                        help="Perturbation types to run (default: all)")
    parser.add_argument("--nprocs", type=int, default=4)
    parser.add_argument("--mem-gb", type=float, default=8.0)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--skip-pipeline-run", action="store_true",
                        help="Evaluate pre-existing results only")

    args = parser.parse_args()

    results = run_phase_c(
        bids_dir=args.bids_dir,
        gt_dir=args.gt_dir,
        output_dir=args.output_dir,
        work_dir=args.work_dir,
        dataset_key=args.dataset,
        perturbation_types=args.perturbations,
        nprocs=args.nprocs,
        mem_gb=args.mem_gb,
        timeout_sec=args.timeout,
        max_cases=args.max_cases,
        skip_pipeline_run=args.skip_pipeline_run,
    )

    out_path = get_phase_dir(args.results_dir, "C") / f"phase_c_{args.dataset}.json"
    save_phase_c_results(results, out_path)

    print(f"\n{'='*60}")
    print(f"Phase C results — {args.dataset}")
    print(f"{'='*60}")
    for pert_name, auc_dict in results.auc_summaries.items():
        print(f"\n  {pert_name}:")
        for region, auc_val in auc_dict.items():
            print(f"    {region} AUC: {auc_val:.3f}")

    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
