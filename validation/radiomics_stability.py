#!/usr/bin/env python
"""Radiomic feature stability analysis — native-first vs atlas-first.

Assesses preservation of micro-architectural signal fidelity by comparing
IBSI-compliant radiomic features extracted under two preprocessing
architectures (OncoPrep native-first and a baseline atlas-first pipeline).

Metrics
-------
- **Coefficient of Variation (CV)**: normalised dispersion per feature
  across subjects, :math:`CV_i = (\\sigma_i / \\mu_i) \\times 100\\%`.
- **Intra-class Correlation Coefficient ICC(3,1)**: two-way mixed-effects
  absolute agreement isolating variance introduced by the preprocessing
  architecture.
- **Stability classification**: features with ICC ≥ 0.85 *and* CV ≤ 10 %
  are classified as *highly stable*.
- **Wilcoxon signed-rank test** (paired, per-feature CV) with
  Benjamini–Hochberg FDR correction for multiple comparisons.

Usage (standalone)::

    python -m validation.radiomics_stability \\
        --native-dir /data/oncoprep_output/radiomics \\
        --atlas-dir /data/atlas_pipeline_output/radiomics \\
        --results-dir ./validation_results \\
        --dataset ucsf_pdgm
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import DATASETS, get_phase_dir
from .stats import (
    benjamini_hochberg,
    coefficient_of_variation,
    icc_31,
    wilcoxon_signed_rank,
)


# ---------------------------------------------------------------------------
# Stability thresholds (pre-specified SAP)
# ---------------------------------------------------------------------------
ICC_THRESHOLD: float = 0.85
CV_THRESHOLD: float = 10.0  # percent


# ---------------------------------------------------------------------------
# Feature class groupings (IBSI-compliant categories)
# ---------------------------------------------------------------------------
FEATURE_CLASSES = OrderedDict([
    ("firstorder", "First-Order Statistics"),
    ("shape", "Shape"),
    ("glcm", "GLCM"),
    ("glrlm", "GLRLM"),
    ("glszm", "GLSZM"),
    ("gldm", "GLDM"),
    ("ngtdm", "NGTDM"),
])


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class FeatureStabilityRecord:
    """Stability metrics for a single radiomic feature."""

    feature_name: str
    feature_class: str
    region: str

    # CV across subjects — native-first pipeline
    cv_native: float = float("nan")
    # CV across subjects — atlas-first pipeline
    cv_atlas: float = float("nan")

    # ICC(3,1) between the two architectures
    icc: float = float("nan")
    icc_ci_lower: float = float("nan")
    icc_ci_upper: float = float("nan")

    # Stability classification
    is_highly_stable: bool = False


@dataclass
class RadiomicsStabilityResults:
    """Aggregated results from the radiomic stability analysis."""

    dataset: str
    n_subjects: int = 0
    n_features_total: int = 0
    n_features_stable: int = 0
    pct_features_stable: float = 0.0

    # Per-feature records
    features: List[FeatureStabilityRecord] = field(default_factory=list)

    # Per-class summary:  class → {n_total, n_stable, pct_stable, median_icc, median_cv_native, median_cv_atlas}
    class_summary: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Wilcoxon test on CV distributions (native vs atlas)
    wilcoxon_statistic: float = float("nan")
    wilcoxon_p_value: float = float("nan")
    wilcoxon_fdr_rejected: bool = False

    # Per-feature Wilcoxon p-values (one per feature, comparing CV
    # computed from paired differences — stored for transparency)
    fdr_n_rejected: int = 0
    fdr_alpha: float = 0.05


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------


def _classify_feature(feature_name: str) -> str:
    """Assign a feature to its IBSI class based on naming convention.

    PyRadiomics names features as ``original_<class>_<FeatureName>``.
    """
    lower = feature_name.lower()
    for cls in FEATURE_CLASSES:
        if f"_{cls}_" in lower or lower.startswith(f"{cls}_"):
            return cls
    # Fallback
    if "shape" in lower:
        return "shape"
    return "unknown"


def load_radiomics_json(path: Path) -> Dict[str, Any]:
    """Load a single OncoPrep radiomics features JSON.

    Expected structure (as produced by ``PyRadiomicsFeatureExtraction``)::

        {
            "<region_abbrev>": {
                "label": <int or list>,
                "name": "<human-readable>",
                "features": {
                    "original_firstorder_Mean": 123.4,
                    ...
                }
            },
            ...
        }

    Returns
    -------
    dict
        Region → {feature_name: value} mapping.
    """
    with open(path) as f:
        raw = json.load(f)

    result: Dict[str, Dict[str, float]] = {}
    for region_key, region_data in raw.items():
        features = region_data.get("features", {})
        if not features:
            continue
        # Keep only numeric feature values (skip diagnostics, metadata)
        numeric_feats: Dict[str, float] = {}
        for fname, fval in features.items():
            try:
                numeric_feats[fname] = float(fval)
            except (TypeError, ValueError):
                continue
        if numeric_feats:
            result[region_key] = numeric_feats
    return result


def discover_radiomics_cases(
    radiomics_dir: Path,
    pattern: str = "**/radiomics_features.json",
) -> List[Tuple[str, Path]]:
    """Discover (subject_id, radiomics_json_path) pairs.

    Attempts to extract a subject identifier from the BIDS-style path.

    Parameters
    ----------
    radiomics_dir : Path
        Root directory to search.
    pattern : str
        Glob pattern for radiomics JSON files.

    Returns
    -------
    list of (subject_id, path)
    """
    cases: List[Tuple[str, Path]] = []
    for p in sorted(radiomics_dir.rglob(pattern.lstrip("**/"))):
        # Try extracting sub-XXX from path components
        subject = "unknown"
        for part in p.parts:
            if part.startswith("sub-"):
                subject = part
                break
        cases.append((subject, p))
    return cases


def _match_cases(
    native_cases: List[Tuple[str, Path]],
    atlas_cases: List[Tuple[str, Path]],
) -> List[Tuple[str, Path, Path]]:
    """Match native and atlas cases by subject ID.

    Returns
    -------
    list of (subject, native_path, atlas_path)
    """
    atlas_map = {subj: p for subj, p in atlas_cases}
    matched: List[Tuple[str, Path, Path]] = []
    for subj, native_path in native_cases:
        if subj in atlas_map:
            matched.append((subj, native_path, atlas_map[subj]))
    return matched


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------


def compute_radiomics_stability(
    native_dir: Path,
    atlas_dir: Path,
    dataset_key: str = "unknown",
    radiomics_pattern: str = "**/radiomics_features.json",
    icc_threshold: float = ICC_THRESHOLD,
    cv_threshold: float = CV_THRESHOLD,
    fdr_alpha: float = 0.05,
    max_cases: Optional[int] = None,
) -> RadiomicsStabilityResults:
    """Run the full radiomic feature stability analysis.

    Parameters
    ----------
    native_dir : Path
        Root of the native-first (OncoPrep) derivatives containing
        per-subject ``radiomics_features.json`` files.
    atlas_dir : Path
        Root of the atlas-first (comparator) derivatives.
    dataset_key : str
        Dataset identifier for provenance.
    radiomics_pattern : str
        Glob pattern for locating radiomics JSON files.
    icc_threshold : float
        ICC threshold for "highly stable" classification (default 0.85).
    cv_threshold : float
        CV threshold (%) for "highly stable" classification (default 10.0).
    fdr_alpha : float
        FDR level for Benjamini–Hochberg correction (default 0.05).
    max_cases : int, optional
        Limit the number of matched cases for quick testing.

    Returns
    -------
    RadiomicsStabilityResults
    """
    # 1. Discover and match cases
    native_cases = discover_radiomics_cases(native_dir, radiomics_pattern)
    atlas_cases = discover_radiomics_cases(atlas_dir, radiomics_pattern)
    matched = _match_cases(native_cases, atlas_cases)

    if max_cases is not None:
        matched = matched[:max_cases]

    n_subjects = len(matched)
    results = RadiomicsStabilityResults(dataset=dataset_key, n_subjects=n_subjects)

    if n_subjects < 2:
        return results

    # 2. Load all feature vectors
    # Structure: {region: {feature_name: [val_subj1, val_subj2, ...]}}
    native_features: Dict[str, Dict[str, List[float]]] = {}
    atlas_features: Dict[str, Dict[str, List[float]]] = {}

    for subj, native_path, atlas_path in matched:
        native_data = load_radiomics_json(native_path)
        atlas_data = load_radiomics_json(atlas_path)

        # Accumulate per-region features
        all_regions = set(native_data.keys()) | set(atlas_data.keys())
        for region in all_regions:
            if region not in native_features:
                native_features[region] = {}
            if region not in atlas_features:
                atlas_features[region] = {}

            n_feats = native_data.get(region, {})
            a_feats = atlas_data.get(region, {})
            all_feat_names = set(n_feats.keys()) | set(a_feats.keys())

            for fname in all_feat_names:
                if fname not in native_features[region]:
                    native_features[region][fname] = []
                if fname not in atlas_features[region]:
                    atlas_features[region][fname] = []

                # Use NaN for missing values
                native_features[region][fname].append(
                    n_feats.get(fname, float("nan"))
                )
                atlas_features[region][fname].append(
                    a_feats.get(fname, float("nan"))
                )

    # 3. Compute per-feature stability metrics
    all_cv_native: List[float] = []
    all_cv_atlas: List[float] = []

    for region in sorted(native_features.keys()):
        n_region = native_features[region]
        a_region = atlas_features.get(region, {})

        all_feature_names = sorted(
            set(n_region.keys()) | set(a_region.keys())
        )

        for fname in all_feature_names:
            native_vals = np.array(n_region.get(fname, []), dtype=float)
            atlas_vals = np.array(a_region.get(fname, []), dtype=float)

            # Drop subjects with NaN in either pipeline
            valid = ~(np.isnan(native_vals) | np.isnan(atlas_vals))
            native_valid = native_vals[valid]
            atlas_valid = atlas_vals[valid]

            if len(native_valid) < 2:
                continue

            # CV per pipeline
            cv_n = coefficient_of_variation(native_valid)
            cv_a = coefficient_of_variation(atlas_valid)

            # ICC(3,1): subjects × 2 raters (native, atlas)
            measurements = np.column_stack([native_valid, atlas_valid])
            icc_val, icc_lo, icc_hi = icc_31(measurements)

            # Stability classification
            is_stable = (
                not np.isnan(icc_val)
                and not np.isnan(cv_n)
                and icc_val >= icc_threshold
                and cv_n <= cv_threshold
            )

            feat_class = _classify_feature(fname)

            record = FeatureStabilityRecord(
                feature_name=fname,
                feature_class=feat_class,
                region=region,
                cv_native=cv_n,
                cv_atlas=cv_a,
                icc=icc_val,
                icc_ci_lower=icc_lo,
                icc_ci_upper=icc_hi,
                is_highly_stable=is_stable,
            )
            results.features.append(record)

            # Collect CVs for Wilcoxon test
            if not np.isnan(cv_n) and not np.isnan(cv_a):
                all_cv_native.append(cv_n)
                all_cv_atlas.append(cv_a)

    # 4. Aggregate totals
    results.n_features_total = len(results.features)
    results.n_features_stable = sum(1 for f in results.features if f.is_highly_stable)
    results.pct_features_stable = (
        results.n_features_stable / max(results.n_features_total, 1) * 100.0
    )

    # 5. Per-class summary
    class_groups: Dict[str, List[FeatureStabilityRecord]] = {}
    for rec in results.features:
        cls = rec.feature_class
        if cls not in class_groups:
            class_groups[cls] = []
        class_groups[cls].append(rec)

    for cls, recs in sorted(class_groups.items()):
        n_total = len(recs)
        n_stable = sum(1 for r in recs if r.is_highly_stable)
        icc_vals = [r.icc for r in recs if not np.isnan(r.icc)]
        cv_n_vals = [r.cv_native for r in recs if not np.isnan(r.cv_native)]
        cv_a_vals = [r.cv_atlas for r in recs if not np.isnan(r.cv_atlas)]

        results.class_summary[cls] = {
            "n_total": n_total,
            "n_stable": n_stable,
            "pct_stable": n_stable / max(n_total, 1) * 100.0,
            "median_icc": float(np.median(icc_vals)) if icc_vals else float("nan"),
            "median_cv_native": float(np.median(cv_n_vals)) if cv_n_vals else float("nan"),
            "median_cv_atlas": float(np.median(cv_a_vals)) if cv_a_vals else float("nan"),
        }

    # 6. Wilcoxon signed-rank test on CV distributions (paired by feature)
    if len(all_cv_native) >= 10:
        cv_native_arr = np.array(all_cv_native)
        cv_atlas_arr = np.array(all_cv_atlas)
        w_stat, w_p = wilcoxon_signed_rank(cv_native_arr, cv_atlas_arr)
        results.wilcoxon_statistic = w_stat
        results.wilcoxon_p_value = w_p

        # Apply BH-FDR (single test here, but framework supports expansion)
        if not np.isnan(w_p):
            rejected, adjusted = benjamini_hochberg(
                np.array([w_p]), alpha=fdr_alpha,
            )
            results.wilcoxon_fdr_rejected = bool(rejected[0])
        results.fdr_alpha = fdr_alpha

    return results


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def save_radiomics_stability_results(
    results: RadiomicsStabilityResults,
    output_path: Path,
) -> None:
    """Save radiomics stability results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "dataset": results.dataset,
        "n_subjects": results.n_subjects,
        "n_features_total": results.n_features_total,
        "n_features_stable": results.n_features_stable,
        "pct_features_stable": results.pct_features_stable,
        "icc_threshold": ICC_THRESHOLD,
        "cv_threshold": CV_THRESHOLD,
        "wilcoxon_statistic": _safe_float(results.wilcoxon_statistic),
        "wilcoxon_p_value": _safe_float(results.wilcoxon_p_value),
        "wilcoxon_fdr_rejected": results.wilcoxon_fdr_rejected,
        "fdr_alpha": results.fdr_alpha,
        "fdr_n_rejected": results.fdr_n_rejected,
        "class_summary": {
            cls: {k: _safe_float(v) for k, v in sums.items()}
            for cls, sums in results.class_summary.items()
        },
        "features": [
            {
                "feature_name": f.feature_name,
                "feature_class": f.feature_class,
                "region": f.region,
                "cv_native": _safe_float(f.cv_native),
                "cv_atlas": _safe_float(f.cv_atlas),
                "icc": _safe_float(f.icc),
                "icc_ci_lower": _safe_float(f.icc_ci_lower),
                "icc_ci_upper": _safe_float(f.icc_ci_upper),
                "is_highly_stable": f.is_highly_stable,
            }
            for f in results.features
        ],
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def _safe_float(v: float) -> Any:
    """Convert NaN/Inf to JSON-safe null."""
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    return v


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Radiomic feature stability: native-first vs atlas-first"
    )
    parser.add_argument("--native-dir", type=Path, required=True,
                        help="OncoPrep (native-first) derivatives root")
    parser.add_argument("--atlas-dir", type=Path, required=True,
                        help="Atlas-first (comparator) derivatives root")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=list(DATASETS.keys()),
                        help="Dataset identifier")
    parser.add_argument("--results-dir", type=Path,
                        default=Path("validation_results"))
    parser.add_argument("--radiomics-pattern", default="**/radiomics_features.json",
                        help="Glob pattern for radiomics JSONs")
    parser.add_argument("--max-cases", type=int, default=None)

    args = parser.parse_args()

    print(f"Running radiomic feature stability analysis on {args.dataset}...")
    results = compute_radiomics_stability(
        native_dir=args.native_dir,
        atlas_dir=args.atlas_dir,
        dataset_key=args.dataset,
        radiomics_pattern=args.radiomics_pattern,
        max_cases=args.max_cases,
    )

    out_path = (
        get_phase_dir(args.results_dir, "D")
        / f"radiomics_stability_{args.dataset}.json"
    )
    save_radiomics_stability_results(results, out_path)

    print(f"\n{'=' * 60}")
    print(f"Radiomic Feature Stability — {args.dataset} (n={results.n_subjects})")
    print(f"{'=' * 60}")
    print(f"  Total features evaluated: {results.n_features_total}")
    print(f"  Highly stable (ICC≥{ICC_THRESHOLD}, CV≤{CV_THRESHOLD}%): "
          f"{results.n_features_stable} ({results.pct_features_stable:.1f}%)")

    print("\n  Per-class summary:")
    for cls, sums in results.class_summary.items():
        label = FEATURE_CLASSES.get(cls, cls)
        n_s = sums["n_stable"]
        n_t = sums["n_total"]
        med_icc = sums["median_icc"]
        print(f"    {label}: {n_s}/{n_t} stable, median ICC = "
              f"{med_icc:.3f}" if med_icc is not None else "N/A")

    if not np.isnan(results.wilcoxon_p_value):
        sig = "significant" if results.wilcoxon_fdr_rejected else "not significant"
        print(f"\n  Wilcoxon signed-rank (CV native vs atlas): "
              f"W = {results.wilcoxon_statistic:.1f}, "
              f"p = {results.wilcoxon_p_value:.4f} ({sig} at FDR α={results.fdr_alpha})")

    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
