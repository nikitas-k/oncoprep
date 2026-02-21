#!/usr/bin/env python
"""Phase D — Quantitative stability (downstream utility).

Primary endpoints:
    D1. Volume agreement vs ground truth — Bland–Altman + ICC, per region.
    D2. Longitudinal plausibility — rate of implausible jumps flagged by QC.

Secondary endpoints:
    - Optional radiomics stability (ICC of selected features).

Usage:
    python -m validation.phase_d --pred-dir /data/oncoprep_output \\
        --gt-dir /data/ground_truth --dataset mu_glioma_post \\
        --results-dir ./validation_results
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np

from .config import (
    DATASETS,
    REGION_MAP,
    get_phase_dir,
)
from .metrics import extract_volumes_cc
from .stats import bland_altman, icc_31


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CaseVolumeRecord:
    """Volume measurements for a single case."""
    subject: str
    session: str
    timepoint_index: int  # ordinal within patient timeline
    volumes_pred: Dict[str, float] = field(default_factory=dict)
    volumes_gt: Dict[str, float] = field(default_factory=dict)


@dataclass
class PhaseDResults:
    """Aggregated Phase D results."""
    dataset: str
    n_cases: int = 0

    # D1: Bland–Altman per region
    bland_altman: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # D1: ICC per region (pred vs GT as two "raters")
    icc: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # D2: Longitudinal plausibility
    n_patients_longitudinal: int = 0
    n_implausible_jumps: int = 0
    implausible_jump_rate: float = 0.0
    implausible_cases: List[Dict[str, Any]] = field(default_factory=list)

    # D3: Radiomic feature stability (native-first vs atlas-first)
    radiomics_stability: Optional[Dict[str, Any]] = None

    # Per-case records
    cases: List[CaseVolumeRecord] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Volume extraction
# ---------------------------------------------------------------------------


def extract_case_volumes(
    pred_path: str,
    gt_path: str,
    regions: Optional[Dict[str, List[int]]] = None,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Load a pred/GT pair and extract regional volumes in cc.

    Returns
    -------
    (volumes_pred, volumes_gt)
    """
    if regions is None:
        regions = REGION_MAP

    pred_img = nib.load(pred_path)
    gt_img = nib.load(gt_path)

    pred_data = np.asanyarray(pred_img.dataobj).astype(int)
    gt_data = np.asanyarray(gt_img.dataobj).astype(int)
    voxel_spacing = tuple(float(v) for v in pred_img.header.get_zooms()[:3])

    vp = extract_volumes_cc(pred_data, voxel_spacing, regions)
    vg = extract_volumes_cc(gt_data, voxel_spacing, regions)
    return vp, vg


# ---------------------------------------------------------------------------
# D1: Volume agreement
# ---------------------------------------------------------------------------


def compute_volume_agreement(
    cases: List[CaseVolumeRecord],
    regions: Optional[Dict[str, List[int]]] = None,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """Compute Bland–Altman and ICC for each region.

    Returns
    -------
    ba_results : dict[region, Bland–Altman stats]
    icc_results : dict[region, {icc, ci_lower, ci_upper}]
    """
    if regions is None:
        regions = REGION_MAP

    ba_results: Dict[str, Dict[str, float]] = {}
    icc_results: Dict[str, Dict[str, float]] = {}

    for region in regions:
        preds = [c.volumes_pred.get(region, 0.0) for c in cases]
        gts = [c.volumes_gt.get(region, 0.0) for c in cases]

        preds_arr = np.array(preds)
        gts_arr = np.array(gts)

        # Bland–Altman
        ba = bland_altman(preds_arr, gts_arr)
        ba_results[region] = ba

        # ICC: treat pred & GT as two "raters"
        if len(preds) >= 2:
            measurements = np.column_stack([preds_arr, gts_arr])
            icc_val, icc_lo, icc_hi = icc_31(measurements)
            icc_results[region] = {
                "icc": icc_val,
                "ci_lower": icc_lo,
                "ci_upper": icc_hi,
            }

    return ba_results, icc_results


# ---------------------------------------------------------------------------
# D2: Longitudinal plausibility
# ---------------------------------------------------------------------------


def detect_implausible_jumps(
    cases: List[CaseVolumeRecord],
    max_volume_change_cc_per_day: float = 5.0,
    assumed_days_between_sessions: float = 90.0,
    regions: Optional[Dict[str, List[int]]] = None,
) -> Tuple[int, List[Dict[str, Any]]]:
    """Detect implausible volume jumps in longitudinal data.

    Heuristic: flag if volume changes by more than
    ``max_volume_change_cc_per_day * assumed_days_between_sessions`` cc
    between consecutive timepoints.

    Parameters
    ----------
    cases : list of CaseVolumeRecord
        Must include ``timepoint_index`` for ordering.
    max_volume_change_cc_per_day : float
        Maximum plausible volume change per day (cc).
    assumed_days_between_sessions : float
        Assumed inter-session interval in days.
    regions : dict, optional
        Region definitions.

    Returns
    -------
    n_implausible : int
    implausible_details : list of dicts
    """
    if regions is None:
        regions = REGION_MAP

    max_delta = max_volume_change_cc_per_day * assumed_days_between_sessions

    # Group by patient
    patient_timelines: Dict[str, List[CaseVolumeRecord]] = defaultdict(list)
    for c in cases:
        patient_timelines[c.subject].append(c)

    # Sort each timeline by timepoint_index
    for subj in patient_timelines:
        patient_timelines[subj].sort(key=lambda x: x.timepoint_index)

    n_implausible = 0
    details: List[Dict[str, Any]] = []

    for subj, timeline in patient_timelines.items():
        if len(timeline) < 2:
            continue
        for i in range(1, len(timeline)):
            prev = timeline[i - 1]
            curr = timeline[i]
            for region in regions:
                vol_prev = curr.volumes_pred.get(region, 0.0)
                vol_curr = prev.volumes_pred.get(region, 0.0)
                delta = abs(vol_curr - vol_prev)
                if delta > max_delta:
                    n_implausible += 1
                    details.append({
                        "subject": subj,
                        "session_from": prev.session,
                        "session_to": curr.session,
                        "region": region,
                        "volume_from_cc": vol_prev,
                        "volume_to_cc": vol_curr,
                        "delta_cc": delta,
                        "threshold_cc": max_delta,
                    })

    return n_implausible, details


# ---------------------------------------------------------------------------
# Full Phase D runner
# ---------------------------------------------------------------------------


def run_phase_d(
    pred_dir: Path,
    gt_dir: Path,
    dataset_key: str,
    pred_pattern: str = "*_dseg.nii.gz",
    gt_pattern: str = "*_dseg.nii.gz",
    regions: Optional[Dict[str, List[int]]] = None,
    max_cases: Optional[int] = None,
    comparator_dir: Optional[Path] = None,
    radiomics_pattern: str = "**/radiomics_features.json",
) -> PhaseDResults:
    """Execute Phase D on a dataset.

    Parameters
    ----------
    pred_dir, gt_dir : Path
        Prediction and ground truth roots.
    dataset_key : str
        Dataset identifier.
    comparator_dir : Path, optional
        Root of the atlas-first (comparator) derivatives. When provided,
        radiomic feature stability analysis (D3) is performed comparing
        native-first (pred_dir) against atlas-first (comparator_dir).
    radiomics_pattern : str
        Glob pattern for locating per-subject radiomics JSON files.

    Returns
    -------
    PhaseDResults
    """
    from .phase_b import discover_cases

    if regions is None:
        regions = REGION_MAP

    matched = discover_cases(pred_dir, gt_dir, pred_pattern, gt_pattern)
    if max_cases is not None:
        matched = matched[:max_cases]

    results = PhaseDResults(dataset=dataset_key, n_cases=len(matched))

    # Build volume records
    # Track per-patient session ordering for longitudinal analysis
    patient_sessions: Dict[str, List[str]] = defaultdict(list)
    for pred_path, gt_path, subject, session in matched:
        patient_sessions[subject].append(session)

    # Assign timepoint indices
    for subj in patient_sessions:
        patient_sessions[subj] = sorted(patient_sessions[subj])

    for pred_path, gt_path, subject, session in matched:
        vp, vg = extract_case_volumes(str(pred_path), str(gt_path), regions)
        tp_idx = patient_sessions[subject].index(session) if session in patient_sessions[subject] else 0
        record = CaseVolumeRecord(
            subject=subject,
            session=session,
            timepoint_index=tp_idx,
            volumes_pred=vp,
            volumes_gt=vg,
        )
        results.cases.append(record)

    # --- D1: Volume agreement ---
    ba_results, icc_results = compute_volume_agreement(results.cases, regions)
    results.bland_altman = ba_results
    results.icc = icc_results

    # --- D2: Longitudinal plausibility ---
    patients_with_multiple = set()
    for subj, sessions in patient_sessions.items():
        if len(sessions) >= 2:
            patients_with_multiple.add(subj)
    results.n_patients_longitudinal = len(patients_with_multiple)

    if patients_with_multiple:
        longitudinal_cases = [c for c in results.cases if c.subject in patients_with_multiple]
        n_jumps, jump_details = detect_implausible_jumps(longitudinal_cases, regions=regions)
        results.n_implausible_jumps = n_jumps
        results.implausible_cases = jump_details

        total_transitions = sum(
            len(patient_sessions[s]) - 1
            for s in patients_with_multiple
        ) * len(regions)
        results.implausible_jump_rate = n_jumps / max(total_transitions, 1)

    # --- D3: Radiomic feature stability (optional, requires comparator) ---
    if comparator_dir is not None:
        try:
            from .radiomics_stability import (
                compute_radiomics_stability,
                save_radiomics_stability_results,
            )

            rad_results = compute_radiomics_stability(
                native_dir=pred_dir,
                atlas_dir=comparator_dir,
                dataset_key=dataset_key,
                radiomics_pattern=radiomics_pattern,
                max_cases=max_cases,
            )
            # Store summary in PhaseDResults
            results.radiomics_stability = {
                "n_subjects": rad_results.n_subjects,
                "n_features_total": rad_results.n_features_total,
                "n_features_stable": rad_results.n_features_stable,
                "pct_features_stable": rad_results.pct_features_stable,
                "class_summary": rad_results.class_summary,
                "wilcoxon_statistic": rad_results.wilcoxon_statistic,
                "wilcoxon_p_value": rad_results.wilcoxon_p_value,
                "wilcoxon_fdr_rejected": rad_results.wilcoxon_fdr_rejected,
            }

            # Also save detailed per-feature results separately
            rad_out = get_phase_dir(
                pred_dir.parent if pred_dir.name == "oncoprep" else pred_dir,
                "D",
            ) / f"radiomics_stability_{dataset_key}.json"
            save_radiomics_stability_results(rad_results, rad_out)
        except Exception as exc:
            # Graceful degradation — radiomics stability is a secondary endpoint
            import warnings
            warnings.warn(
                f"D3 radiomics stability analysis failed: {exc}",
                stacklevel=2,
            )

    return results


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def save_phase_d_results(results: PhaseDResults, output_path: Path) -> None:
    """Save Phase D results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "dataset": results.dataset,
        "n_cases": results.n_cases,
        "bland_altman": results.bland_altman,
        "icc": results.icc,
        "n_patients_longitudinal": results.n_patients_longitudinal,
        "n_implausible_jumps": results.n_implausible_jumps,
        "implausible_jump_rate": results.implausible_jump_rate,
        "implausible_cases": results.implausible_cases,
        "radiomics_stability": results.radiomics_stability,
        "cases": [
            {
                "subject": c.subject,
                "session": c.session,
                "timepoint_index": c.timepoint_index,
                "volumes_pred": c.volumes_pred,
                "volumes_gt": c.volumes_gt,
            }
            for c in results.cases
        ],
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase D: Quantitative stability validation"
    )
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASETS.keys()))
    parser.add_argument("--results-dir", type=Path, default=Path("validation_results"))
    parser.add_argument("--pred-pattern", default="*_dseg.nii.gz")
    parser.add_argument("--gt-pattern", default="*_dseg.nii.gz")
    parser.add_argument("--comparator-dir", type=Path, default=None,
                        help="Atlas-first comparator derivatives root (enables D3 radiomics stability)")
    parser.add_argument("--radiomics-pattern", default="**/radiomics_features.json")
    parser.add_argument("--max-cases", type=int, default=None)

    args = parser.parse_args()

    print(f"Running Phase D evaluation on {args.dataset}...")
    results = run_phase_d(
        pred_dir=args.pred_dir,
        gt_dir=args.gt_dir,
        dataset_key=args.dataset,
        pred_pattern=args.pred_pattern,
        gt_pattern=args.gt_pattern,
        max_cases=args.max_cases,
        comparator_dir=args.comparator_dir,
        radiomics_pattern=args.radiomics_pattern,
    )

    out_path = get_phase_dir(args.results_dir, "D") / f"phase_d_{args.dataset}.json"
    save_phase_d_results(results, out_path)

    print(f"\n{'='*60}")
    print(f"Phase D results — {args.dataset} (n={results.n_cases})")
    print(f"{'='*60}")

    print("\n  Bland–Altman (volume agreement):")
    for region, ba in results.bland_altman.items():
        print(f"    {region}: mean diff = {ba['mean_diff']:.2f} cc, "
              f"LoA = [{ba['loa_lower']:.2f}, {ba['loa_upper']:.2f}]")

    print("\n  ICC (pred vs GT):")
    for region, ic in results.icc.items():
        print(f"    {region}: ICC = {ic['icc']:.3f} "
              f"[{ic['ci_lower']:.3f}, {ic['ci_upper']:.3f}]")

    if results.n_patients_longitudinal > 0:
        print(f"\n  Longitudinal plausibility ({results.n_patients_longitudinal} patients):")
        print(f"    Implausible jumps: {results.n_implausible_jumps}")
        print(f"    Jump rate: {results.implausible_jump_rate:.1%}")

    if results.radiomics_stability is not None:
        rs = results.radiomics_stability
        print("\n  Radiomic feature stability (D3):")
        print(f"    Subjects matched: {rs['n_subjects']}")
        print(f"    Features evaluated: {rs['n_features_total']}")
        print(f"    Highly stable: {rs['n_features_stable']} "
              f"({rs['pct_features_stable']:.1f}%)")
        w_p = rs.get("wilcoxon_p_value")
        if w_p is not None:
            sig = "significant" if rs.get("wilcoxon_fdr_rejected") else "not significant"
            print(f"    Wilcoxon CV native vs atlas: p = {w_p:.4f} ({sig})")

    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
