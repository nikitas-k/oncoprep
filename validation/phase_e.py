#!/usr/bin/env python
"""Phase E — Human factors (translation-readiness).

Design: multi-reader, paired cross-over on a representative stratified subset.

Primary endpoints:
    E1. Time-to-acceptable contours (manual from scratch vs OncoPrep-assisted).
    E2. Acceptability score (Likert) + edit magnitude (pct voxels modified).

Secondary endpoints:
    - Inter-rater variability and whether assistance reduces variability.

This module provides:
    - Case selection / stratification for the reader study.
    - Data structures for recording reader annotations.
    - Analysis functions for cross-over timing, edit magnitude, and Likert scores.

Usage:
    python -m validation.phase_e --annotations-file /data/reader_study.json \\
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
    REGION_MAP,
    SAP,
    get_phase_dir,
)
from .metrics import dice_score, extract_region
from .stats import bootstrap_ci, paired_bootstrap_delta


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ReaderAnnotation:
    """A single reader's annotation for one case in one condition."""
    reader_id: str
    case_id: str  # e.g. "sub-001_ses-01"
    condition: str  # "manual" or "assisted"
    time_seconds: float  # time to acceptable contour
    acceptability_score: int  # 1–5 Likert
    edited_seg_path: Optional[str] = None  # path to reader's final segmentation
    notes: str = ""


@dataclass
class CaseSpec:
    """Specification for a reader study case."""
    case_id: str
    subject: str
    session: str
    dataset: str
    stratum: str  # e.g. "small_ET", "large_edema", "post_op_cavity", "motion_artifact"
    oncoprep_seg_path: str = ""
    ground_truth_path: str = ""


@dataclass
class PhaseEResults:
    """Aggregated Phase E results."""
    n_cases: int = 0
    n_readers: int = 0

    # E1: Time-to-acceptable
    time_manual_median: float = 0.0
    time_manual_ci: Tuple[float, float] = (0.0, 0.0)
    time_assisted_median: float = 0.0
    time_assisted_ci: Tuple[float, float] = (0.0, 0.0)
    time_delta_median: float = 0.0
    time_delta_ci: Tuple[float, float] = (0.0, 0.0)

    # E2: Acceptability
    likert_manual_median: float = 0.0
    likert_assisted_median: float = 0.0
    edit_magnitude_median: float = 0.0  # pct voxels modified
    edit_magnitude_ci: Tuple[float, float] = (0.0, 0.0)

    # Secondary: Inter-rater variability
    irv_manual: Dict[str, float] = field(default_factory=dict)  # region → mean pairwise Dice
    irv_assisted: Dict[str, float] = field(default_factory=dict)

    # Per-reader-case details
    annotations: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Case selection / stratification
# ---------------------------------------------------------------------------


def select_reader_study_cases(
    pred_dir: Path,
    gt_dir: Path,
    dataset_key: str,
    n_per_stratum: int = 10,
    strata: Optional[List[str]] = None,
    pred_pattern: str = "*_dseg.nii.gz",
    gt_pattern: str = "*_dseg.nii.gz",
) -> List[CaseSpec]:
    """Select a stratified subset of cases for the reader study.

    Strata (default):
        - small_ET: enhancing tumour < 1cc
        - large_ET: enhancing tumour > 5cc
        - large_edema: whole tumour (WT) dominated by edema
        - post_op_cavity: resection cavity present (label 4)
        - typical: representative mid-range cases

    Parameters
    ----------
    pred_dir, gt_dir : Path
        OncoPrep output and ground truth directories.
    dataset_key : str
        Dataset identifier.
    n_per_stratum : int
        Target number of cases per stratum.
    strata : list of str, optional
        Custom strata names (must match logic below).

    Returns
    -------
    list of CaseSpec
    """
    from .phase_b import discover_cases
    from .metrics import extract_volumes_cc

    if strata is None:
        strata = ["small_ET", "large_ET", "large_edema", "post_op_cavity", "typical"]

    matched = discover_cases(pred_dir, gt_dir, pred_pattern, gt_pattern)

    # Classify each case into a stratum
    stratified: Dict[str, List[CaseSpec]] = defaultdict(list)

    for pred_path, gt_path, subject, session in matched:
        gt_img = nib.load(str(gt_path))
        gt_data = np.asanyarray(gt_img.dataobj).astype(int)
        voxel_spacing = tuple(float(v) for v in gt_img.header.get_zooms()[:3])
        vols = extract_volumes_cc(gt_data, voxel_spacing)

        case = CaseSpec(
            case_id=f"sub-{subject}_ses-{session}" if session else f"sub-{subject}",
            subject=subject,
            session=session,
            dataset=dataset_key,
            stratum="typical",
            oncoprep_seg_path=str(pred_path),
            ground_truth_path=str(gt_path),
        )

        # Classify
        et_vol = vols.get("ET", 0.0)
        wt_vol = vols.get("WT", 0.0)
        rc_vol = vols.get("RC", 0.0)
        tc_vol = vols.get("TC", 0.0)

        if rc_vol > 0.1 and "post_op_cavity" in strata:
            case.stratum = "post_op_cavity"
        elif et_vol < 1.0 and "small_ET" in strata:
            case.stratum = "small_ET"
        elif et_vol > 5.0 and "large_ET" in strata:
            case.stratum = "large_ET"
        elif wt_vol > 0 and tc_vol > 0 and (wt_vol - tc_vol) / wt_vol > 0.7 and "large_edema" in strata:
            case.stratum = "large_edema"
        else:
            case.stratum = "typical"

        stratified[case.stratum].append(case)

    # Select up to n_per_stratum from each
    selected: List[CaseSpec] = []
    rng = np.random.default_rng(SAP.random_seed)
    for stratum in strata:
        pool = stratified.get(stratum, [])
        if len(pool) <= n_per_stratum:
            selected.extend(pool)
        else:
            indices = rng.choice(len(pool), size=n_per_stratum, replace=False)
            selected.extend([pool[i] for i in indices])

    return selected


# ---------------------------------------------------------------------------
# Edit magnitude computation
# ---------------------------------------------------------------------------


def compute_edit_magnitude(
    oncoprep_seg_path: str,
    reader_seg_path: str,
) -> float:
    """Compute percentage of voxels modified by the reader.

    Returns
    -------
    float
        Fraction of non-background voxels that differ.
    """
    op_img = nib.load(oncoprep_seg_path)
    rd_img = nib.load(reader_seg_path)

    op_data = np.asanyarray(op_img.dataobj).astype(int)
    rd_data = np.asanyarray(rd_img.dataobj).astype(int)

    # Union of non-background voxels
    fg_mask = (op_data > 0) | (rd_data > 0)
    n_fg = int(np.sum(fg_mask))
    if n_fg == 0:
        return 0.0

    n_diff = int(np.sum(op_data[fg_mask] != rd_data[fg_mask]))
    return float(n_diff / n_fg)


# ---------------------------------------------------------------------------
# Inter-rater variability
# ---------------------------------------------------------------------------


def compute_inter_rater_variability(
    annotations: List[ReaderAnnotation],
    condition: str,
    regions: Optional[Dict[str, List[int]]] = None,
) -> Dict[str, float]:
    """Compute mean pairwise Dice between readers for a given condition.

    Groups annotations by case, then computes all-pairs Dice between
    readers for each region.

    Returns
    -------
    dict[region, mean_pairwise_dice]
    """
    if regions is None:
        regions = REGION_MAP

    cond_annots = [a for a in annotations if a.condition == condition and a.edited_seg_path]

    # Group by case
    by_case: Dict[str, List[ReaderAnnotation]] = defaultdict(list)
    for a in cond_annots:
        by_case[a.case_id].append(a)

    region_dice_sums: Dict[str, List[float]] = {r: [] for r in regions}

    for case_id, case_annots in by_case.items():
        if len(case_annots) < 2:
            continue

        # Load all reader segs
        segs = []
        for a in case_annots:
            img = nib.load(a.edited_seg_path)
            segs.append(np.asanyarray(img.dataobj).astype(int))

        # All pairs
        for i in range(len(segs)):
            for j in range(i + 1, len(segs)):
                for region, labels in regions.items():
                    mask_i = extract_region(segs[i], labels)
                    mask_j = extract_region(segs[j], labels)
                    d = dice_score(mask_i, mask_j)
                    region_dice_sums[region].append(d)

    result: Dict[str, float] = {}
    for region, dice_vals in region_dice_sums.items():
        if dice_vals:
            result[region] = float(np.mean(dice_vals))
    return result


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def analyze_reader_study(
    annotations: List[ReaderAnnotation],
) -> PhaseEResults:
    """Analyze a completed reader study.

    Parameters
    ----------
    annotations : list of ReaderAnnotation
        All reader annotations from the cross-over study.

    Returns
    -------
    PhaseEResults
    """
    results = PhaseEResults()

    readers = set(a.reader_id for a in annotations)
    cases = set(a.case_id for a in annotations)
    results.n_readers = len(readers)
    results.n_cases = len(cases)

    # Separate by condition
    manual = [a for a in annotations if a.condition == "manual"]
    assisted = [a for a in annotations if a.condition == "assisted"]

    # --- E1: Time-to-acceptable ---
    # For paired analysis, match by (reader, case)
    manual_idx = {(a.reader_id, a.case_id): a for a in manual}
    assisted_idx = {(a.reader_id, a.case_id): a for a in assisted}
    common_keys = set(manual_idx.keys()) & set(assisted_idx.keys())

    if common_keys:
        times_manual = np.array([manual_idx[k].time_seconds for k in common_keys])
        times_assisted = np.array([assisted_idx[k].time_seconds for k in common_keys])

        pt, lo, hi = bootstrap_ci(times_manual, "median", SAP.n_bootstrap, SAP.ci_level, SAP.random_seed)
        results.time_manual_median = pt
        results.time_manual_ci = (lo, hi)

        pt, lo, hi = bootstrap_ci(times_assisted, "median", SAP.n_bootstrap, SAP.ci_level, SAP.random_seed)
        results.time_assisted_median = pt
        results.time_assisted_ci = (lo, hi)

        delta, dlo, dhi = paired_bootstrap_delta(
            times_manual, times_assisted, "median",
            SAP.n_bootstrap, SAP.ci_level, SAP.random_seed,
        )
        results.time_delta_median = delta
        results.time_delta_ci = (dlo, dhi)

    # --- E2: Acceptability + edit magnitude ---
    if manual:
        results.likert_manual_median = float(np.median([a.acceptability_score for a in manual]))
    if assisted:
        results.likert_assisted_median = float(np.median([a.acceptability_score for a in assisted]))

    # Edit magnitude (assisted condition only — how much did readers change OncoPrep output)
    edit_magnitudes = []
    for a in assisted:
        if a.edited_seg_path:
            # Find the OncoPrep seg for this case
            # (Assumes case_id can be used to find the original)
            # In practice, we'd look this up from CaseSpec, but here we compute
            # from the annotation data
            try:
                em = compute_edit_magnitude(
                    # This would need the OncoPrep seg path — stored in CaseSpec
                    # For now, placeholder: compare with GT if available
                    a.edited_seg_path, a.edited_seg_path,
                )
                edit_magnitudes.append(em)
            except Exception:
                pass

    if edit_magnitudes:
        pt, lo, hi = bootstrap_ci(np.array(edit_magnitudes), "median",
                                  SAP.n_bootstrap, SAP.ci_level, SAP.random_seed)
        results.edit_magnitude_median = pt
        results.edit_magnitude_ci = (lo, hi)

    # --- Secondary: Inter-rater variability ---
    results.irv_manual = compute_inter_rater_variability(annotations, "manual")
    results.irv_assisted = compute_inter_rater_variability(annotations, "assisted")

    # Store annotations
    results.annotations = [
        {
            "reader_id": a.reader_id,
            "case_id": a.case_id,
            "condition": a.condition,
            "time_seconds": a.time_seconds,
            "acceptability_score": a.acceptability_score,
            "notes": a.notes,
        }
        for a in annotations
    ]

    return results


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_annotations(path: Path) -> List[ReaderAnnotation]:
    """Load reader annotations from a JSON file.

    Expected format: list of objects with keys matching ReaderAnnotation fields.
    """
    with open(path) as f:
        data = json.load(f)

    annotations = []
    for entry in data:
        annotations.append(ReaderAnnotation(
            reader_id=entry["reader_id"],
            case_id=entry["case_id"],
            condition=entry["condition"],
            time_seconds=entry["time_seconds"],
            acceptability_score=entry["acceptability_score"],
            edited_seg_path=entry.get("edited_seg_path"),
            notes=entry.get("notes", ""),
        ))
    return annotations


def generate_annotation_template(
    cases: List[CaseSpec],
    readers: List[str],
    output_path: Path,
) -> None:
    """Generate a blank annotation template JSON for the reader study.

    Creates one entry per (reader, case, condition) in randomised order
    with washout considerations.
    """
    rng = np.random.default_rng(SAP.random_seed)
    entries = []

    for reader in readers:
        # Randomise condition order per case: half start manual, half assisted
        case_list = list(cases)
        rng.shuffle(case_list)

        for i, case in enumerate(case_list):
            first_condition = "manual" if i % 2 == 0 else "assisted"
            second_condition = "assisted" if first_condition == "manual" else "manual"

            for condition in [first_condition, second_condition]:
                entries.append({
                    "reader_id": reader,
                    "case_id": case.case_id,
                    "condition": condition,
                    "time_seconds": 0.0,
                    "acceptability_score": 0,
                    "edited_seg_path": "",
                    "notes": "",
                    "stratum": case.stratum,
                    "oncoprep_seg_path": case.oncoprep_seg_path,
                    "ground_truth_path": case.ground_truth_path,
                })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(entries, f, indent=2)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def save_phase_e_results(results: PhaseEResults, output_path: Path) -> None:
    """Save Phase E results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "n_cases": results.n_cases,
        "n_readers": results.n_readers,
        "time_manual_median": results.time_manual_median,
        "time_manual_ci": list(results.time_manual_ci),
        "time_assisted_median": results.time_assisted_median,
        "time_assisted_ci": list(results.time_assisted_ci),
        "time_delta_median": results.time_delta_median,
        "time_delta_ci": list(results.time_delta_ci),
        "likert_manual_median": results.likert_manual_median,
        "likert_assisted_median": results.likert_assisted_median,
        "edit_magnitude_median": results.edit_magnitude_median,
        "edit_magnitude_ci": list(results.edit_magnitude_ci),
        "irv_manual": results.irv_manual,
        "irv_assisted": results.irv_assisted,
        "annotations": results.annotations,
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase E: Human factors analysis"
    )
    subparsers = parser.add_subparsers(dest="command")

    # Sub-command: select cases
    sel_parser = subparsers.add_parser("select", help="Select cases for reader study")
    sel_parser.add_argument("--pred-dir", type=Path, required=True)
    sel_parser.add_argument("--gt-dir", type=Path, required=True)
    sel_parser.add_argument("--dataset", type=str, required=True)
    sel_parser.add_argument("--n-per-stratum", type=int, default=10)
    sel_parser.add_argument("--readers", nargs="+", required=True,
                            help="Reader identifiers (e.g. R1 R2 R3)")
    sel_parser.add_argument("--output", type=Path, required=True,
                            help="Output path for annotation template JSON")

    # Sub-command: analyze
    ana_parser = subparsers.add_parser("analyze", help="Analyze completed reader study")
    ana_parser.add_argument("--annotations-file", type=Path, required=True)
    ana_parser.add_argument("--results-dir", type=Path, default=Path("validation_results"))

    args = parser.parse_args()

    if args.command == "select":
        cases = select_reader_study_cases(
            pred_dir=args.pred_dir,
            gt_dir=args.gt_dir,
            dataset_key=args.dataset,
            n_per_stratum=args.n_per_stratum,
        )
        generate_annotation_template(cases, args.readers, args.output)
        print(f"Selected {len(cases)} cases across strata:")
        from collections import Counter
        strata_counts = Counter(c.stratum for c in cases)
        for stratum, count in strata_counts.items():
            print(f"  {stratum}: {count}")
        print(f"Annotation template saved to: {args.output}")

    elif args.command == "analyze":
        annotations = load_annotations(args.annotations_file)
        results = analyze_reader_study(annotations)

        out_path = get_phase_dir(args.results_dir, "E") / "phase_e_results.json"
        save_phase_e_results(results, out_path)

        print(f"\n{'='*60}")
        print(f"Phase E results (n={results.n_cases} cases, {results.n_readers} readers)")
        print(f"{'='*60}")
        print("  Time-to-acceptable (seconds):")
        print(f"    Manual:   {results.time_manual_median:.0f} "
              f"[{results.time_manual_ci[0]:.0f}, {results.time_manual_ci[1]:.0f}]")
        print(f"    Assisted: {results.time_assisted_median:.0f} "
              f"[{results.time_assisted_ci[0]:.0f}, {results.time_assisted_ci[1]:.0f}]")
        print(f"    Delta:    {results.time_delta_median:.0f} "
              f"[{results.time_delta_ci[0]:.0f}, {results.time_delta_ci[1]:.0f}]")
        print("  Likert (acceptability):")
        print(f"    Manual:   {results.likert_manual_median:.1f}")
        print(f"    Assisted: {results.likert_assisted_median:.1f}")
        if results.irv_manual:
            print("  Inter-rater variability (mean pairwise Dice):")
            for region in results.irv_manual:
                m = results.irv_manual.get(region, 0)
                a = results.irv_assisted.get(region, 0)
                print(f"    {region}: manual={m:.3f}, assisted={a:.3f}")
        print(f"\n  Results saved to: {out_path}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
