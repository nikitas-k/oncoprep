#!/usr/bin/env python
"""Phase B — Segmentation accuracy (scientific efficacy).

Primary endpoints:
    B1. Lesion-wise Dice and lesion-wise HD95, median [IQR] + 95% bootstrap CI.
    B2. Patient-level Dice + HD95 (union of lesions), for comparability.

Secondary endpoints:
    - Surface Dice at 1–2 mm tolerance.
    - Small-lesion sensitivity stratified by volume bins (<1cc, 1–5cc, >5cc).

Usage:
    python -m validation.phase_b --pred-dir /data/oncoprep_output \\
        --gt-dir /data/ground_truth --dataset ucsf_pdgm \\
        --results-dir ./validation_results
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import (
    DATASETS,
    REGION_MAP,
    SAP,
    SURFACE_DICE_TOLERANCES,
    VOLUME_BINS,
    get_phase_dir,
)
from .metrics import (
    evaluate_case,
)
from .stats import bootstrap_ci, paired_bootstrap_delta


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CaseMetrics:
    """All metrics for a single case."""
    subject: str
    session: str
    dataset: str
    patient_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    lesion_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    volumes_pred: Dict[str, float] = field(default_factory=dict)
    volumes_gt: Dict[str, float] = field(default_factory=dict)


@dataclass
class PhaseBResults:
    """Aggregated Phase B results for a dataset."""
    dataset: str
    n_cases: int = 0

    # Patient-level (B2): per-region {metric: {point, ci_lower, ci_upper}}
    patient_level: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)

    # Lesion-wise (B1): per-region {metric: {point, ci_lower, ci_upper}}
    lesion_wise: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)

    # Volume-bin stratified sensitivity
    volume_bin_sensitivity: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Comparator deltas (if available)
    comparator_deltas: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)

    cases: List[CaseMetrics] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Case discovery
# ---------------------------------------------------------------------------


def discover_cases(
    pred_dir: Path,
    gt_dir: Path,
    pred_pattern: str = "*_dseg.nii.gz",
    gt_pattern: str = "*_dseg.nii.gz",
) -> List[Tuple[Path, Path, str, str]]:
    """Match predicted segmentations to ground-truth files.

    Expects BIDS-like directory structure:
        pred_dir/sub-XXX/[ses-YY/]anat/sub-XXX[_ses-YY]_*_dseg.nii.gz
        gt_dir/sub-XXX/[ses-YY/]anat/sub-XXX[_ses-YY]_*_dseg.nii.gz

    Returns
    -------
    list of (pred_path, gt_path, subject, session)
    """
    pairs = []
    for pred_file in sorted(pred_dir.rglob(pred_pattern)):
        # Extract subject/session from path
        parts = pred_file.parts
        subject = ""
        session = ""
        for part in parts:
            if part.startswith("sub-"):
                subject = part.removeprefix("sub-")
            elif part.startswith("ses-"):
                session = part.removeprefix("ses-")

        if not subject:
            continue

        # Find matching GT
        gt_candidates = list(gt_dir.rglob(f"sub-{subject}/**/{gt_pattern}"))
        if session:
            gt_candidates = [
                g for g in gt_candidates
                if f"ses-{session}" in str(g)
            ]

        if gt_candidates:
            pairs.append((pred_file, gt_candidates[0], subject, session))

    return pairs


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------


def evaluate_dataset(
    pred_dir: Path,
    gt_dir: Path,
    dataset_key: str,
    pred_pattern: str = "*_dseg.nii.gz",
    gt_pattern: str = "*_dseg.nii.gz",
    regions: Optional[Dict[str, List[int]]] = None,
    max_cases: Optional[int] = None,
) -> PhaseBResults:
    """Run full Phase B evaluation on a dataset.

    Parameters
    ----------
    pred_dir : Path
        OncoPrep derivatives root containing predicted segmentations.
    gt_dir : Path
        Ground truth directory (same BIDS-like layout).
    dataset_key : str
        Dataset identifier.
    pred_pattern, gt_pattern : str
        Glob patterns for segmentation files.
    regions : dict, optional
        Region definitions. Defaults to REGION_MAP.
    max_cases : int, optional
        Limit for debugging.

    Returns
    -------
    PhaseBResults
    """
    if regions is None:
        regions = REGION_MAP

    cases = discover_cases(pred_dir, gt_dir, pred_pattern, gt_pattern)
    if max_cases is not None:
        cases = cases[:max_cases]

    results = PhaseBResults(dataset=dataset_key, n_cases=len(cases))

    # Collect per-case metrics
    all_patient: Dict[str, List[Dict[str, float]]] = {r: [] for r in regions}
    all_lesion_dice: Dict[str, List[float]] = {r: [] for r in regions}
    all_lesion_hd95: Dict[str, List[float]] = {r: [] for r in regions}
    all_lesion_records: Dict[str, List[Dict]] = {r: [] for r in regions}

    for pred_path, gt_path, subject, session in cases:
        case_result = evaluate_case(str(pred_path), str(gt_path), regions=regions)

        cm = CaseMetrics(
            subject=subject,
            session=session,
            dataset=dataset_key,
            patient_metrics=case_result["patient_metrics"],
            lesion_metrics=case_result["lesion_metrics"],
            volumes_pred=case_result["volumes_pred"],
            volumes_gt=case_result["volumes_gt"],
        )
        results.cases.append(cm)

        for region in regions:
            all_patient[region].append(case_result["patient_metrics"][region])
            lm = case_result["lesion_metrics"][region]
            for gl in lm["gt_lesions"]:
                all_lesion_dice[region].append(gl["dice"])
                all_lesion_hd95[region].append(gl["hd95"])
                all_lesion_records[region].append(gl)

    # --- B2: Patient-level aggregation ---
    for region in regions:
        if not all_patient[region]:
            continue
        region_results: Dict[str, Dict[str, float]] = {}
        for metric_name in ["dice", "hd95"] + [f"surface_dice_{t}mm" for t in SURFACE_DICE_TOLERANCES]:
            values = [pm[metric_name] for pm in all_patient[region] if metric_name in pm]
            valid = [v for v in values if np.isfinite(v)]
            if valid:
                pt, lo, hi = bootstrap_ci(
                    np.array(valid), "median", SAP.n_bootstrap, SAP.ci_level, SAP.random_seed
                )
                region_results[metric_name] = {"median": pt, "ci_lower": lo, "ci_upper": hi}
        results.patient_level[region] = region_results

    # --- B1: Lesion-wise aggregation ---
    for region in regions:
        region_lw: Dict[str, Dict[str, float]] = {}
        dice_arr = np.array([d for d in all_lesion_dice[region] if np.isfinite(d)])
        hd95_arr = np.array([h for h in all_lesion_hd95[region] if np.isfinite(h)])

        if len(dice_arr) > 0:
            pt, lo, hi = bootstrap_ci(dice_arr, "median", SAP.n_bootstrap, SAP.ci_level, SAP.random_seed)
            region_lw["dice"] = {"median": pt, "ci_lower": lo, "ci_upper": hi}
        if len(hd95_arr) > 0:
            pt, lo, hi = bootstrap_ci(hd95_arr, "median", SAP.n_bootstrap, SAP.ci_level, SAP.random_seed)
            region_lw["hd95"] = {"median": pt, "ci_lower": lo, "ci_upper": hi}

        results.lesion_wise[region] = region_lw

    # --- Volume-bin stratified sensitivity ---
    for region in regions:
        records = all_lesion_records[region]
        if not records:
            continue
        bin_results: Dict[str, float] = {}
        for lo_vol, hi_vol in VOLUME_BINS:
            if hi_vol == float("inf"):
                bin_label = f">{lo_vol}cc"
            else:
                bin_label = f"{lo_vol}-{hi_vol}cc"
            in_bin = [r for r in records if lo_vol <= r["volume_cc"] < hi_vol]
            if in_bin:
                n_matched = sum(1 for r in in_bin if r["matched"])
                bin_results[bin_label] = n_matched / len(in_bin)
            else:
                bin_results[bin_label] = float("nan")
        results.volume_bin_sensitivity[region] = bin_results

    return results


# ---------------------------------------------------------------------------
# Comparator evaluation
# ---------------------------------------------------------------------------


def compare_pipelines(
    results_a: PhaseBResults,
    results_b: PhaseBResults,
    metric: str = "dice",
    statistic: str = "median",
) -> Dict[str, Dict[str, float]]:
    """Paired bootstrap comparison of two pipelines on matching cases.

    Returns per-region ``{delta, ci_lower, ci_upper}``.
    """
    # Index results_b by (subject, session)
    b_index = {(c.subject, c.session): c for c in results_b.cases}
    deltas: Dict[str, Dict[str, float]] = {}

    for region in REGION_MAP:
        vals_a, vals_b = [], []
        for ca in results_a.cases:
            cb = b_index.get((ca.subject, ca.session))
            if cb is None:
                continue
            va = ca.patient_metrics.get(region, {}).get(metric)
            vb = cb.patient_metrics.get(region, {}).get(metric)
            if va is not None and vb is not None and np.isfinite(va) and np.isfinite(vb):
                vals_a.append(va)
                vals_b.append(vb)

        if vals_a:
            delta, lo, hi = paired_bootstrap_delta(
                np.array(vals_a), np.array(vals_b),
                statistic, SAP.n_bootstrap, SAP.ci_level, SAP.random_seed,
            )
            deltas[region] = {"delta": delta, "ci_lower": lo, "ci_upper": hi, "n_pairs": len(vals_a)}

    return deltas


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def save_phase_b_results(results: PhaseBResults, output_path: Path) -> None:
    """Save Phase B results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "dataset": results.dataset,
        "n_cases": results.n_cases,
        "patient_level": results.patient_level,
        "lesion_wise": results.lesion_wise,
        "volume_bin_sensitivity": results.volume_bin_sensitivity,
        "comparator_deltas": results.comparator_deltas,
        "cases": [
            {
                "subject": c.subject,
                "session": c.session,
                "patient_metrics": c.patient_metrics,
                "volumes_pred": c.volumes_pred,
                "volumes_gt": c.volumes_gt,
                # lesion_metrics omitted for brevity; add if needed
            }
            for c in results.cases
        ],
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)


def _json_default(obj: Any) -> Any:
    """Handle non-serialisable types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase B: Segmentation accuracy validation"
    )
    parser.add_argument("--pred-dir", type=Path, required=True,
                        help="OncoPrep derivatives root with predicted segmentations")
    parser.add_argument("--gt-dir", type=Path, required=True,
                        help="Ground truth directory (BIDS-like layout)")
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASETS.keys()))
    parser.add_argument("--results-dir", type=Path, default=Path("validation_results"))
    parser.add_argument("--pred-pattern", default="*_dseg.nii.gz")
    parser.add_argument("--gt-pattern", default="*_dseg.nii.gz")
    parser.add_argument("--max-cases", type=int, default=None)

    # Optional comparator
    parser.add_argument("--comparator-dir", type=Path, default=None,
                        help="Comparator pipeline output directory")

    args = parser.parse_args()

    print(f"Running Phase B evaluation on {args.dataset}...")
    results = evaluate_dataset(
        pred_dir=args.pred_dir,
        gt_dir=args.gt_dir,
        dataset_key=args.dataset,
        pred_pattern=args.pred_pattern,
        gt_pattern=args.gt_pattern,
        max_cases=args.max_cases,
    )

    # Comparator evaluation
    if args.comparator_dir:
        print(f"Evaluating comparator from {args.comparator_dir}...")
        comparator_results = evaluate_dataset(
            pred_dir=args.comparator_dir,
            gt_dir=args.gt_dir,
            dataset_key=args.dataset,
            pred_pattern=args.pred_pattern,
            gt_pattern=args.gt_pattern,
            max_cases=args.max_cases,
        )
        deltas = compare_pipelines(results, comparator_results)
        results.comparator_deltas = deltas

    out_path = get_phase_dir(args.results_dir, "B") / f"phase_b_{args.dataset}.json"
    save_phase_b_results(results, out_path)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Phase B results — {args.dataset} (n={results.n_cases})")
    print(f"{'='*60}")
    for region, metrics in results.patient_level.items():
        dice_info = metrics.get("dice", {})
        hd95_info = metrics.get("hd95", {})
        print(f"  {region}:")
        if dice_info:
            print(f"    Patient Dice: {dice_info['median']:.3f} "
                  f"[{dice_info['ci_lower']:.3f}, {dice_info['ci_upper']:.3f}]")
        if hd95_info:
            print(f"    Patient HD95: {hd95_info['median']:.2f} "
                  f"[{hd95_info['ci_lower']:.2f}, {hd95_info['ci_upper']:.2f}]")

    if results.lesion_wise:
        print("\n  Lesion-wise:")
        for region, metrics in results.lesion_wise.items():
            dice_info = metrics.get("dice", {})
            if dice_info:
                print(f"    {region} Dice: {dice_info['median']:.3f} "
                      f"[{dice_info['ci_lower']:.3f}, {dice_info['ci_upper']:.3f}]")

    if results.volume_bin_sensitivity:
        print("\n  Volume-bin sensitivity:")
        for region, bins in results.volume_bin_sensitivity.items():
            for bin_label, sens in bins.items():
                if np.isfinite(sens):
                    print(f"    {region} [{bin_label}]: {sens:.1%}")

    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
