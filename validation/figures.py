#!/usr/bin/env python
"""Figure generation for the OncoPrep validation manuscript.

Generates Figures 1–7 from the blueprint (or subsets thereof) using
matplotlib/seaborn, reading Phase A–E result JSONs.

Usage:
    python -m validation.figures --results-dir ./validation_results \\
        --output-dir ./figures --format pdf --dpi 300
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.gridspec as gridspec  # noqa: E402
import numpy as np  # noqa: E402

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from .config import DATASETS, PERTURBATIONS, REGION_MAP  # noqa: E402


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
REGION_COLORS = {
    "ET": "#e41a1c",
    "TC": "#ff7f00",
    "WT": "#377eb8",
    "RC": "#984ea3",
}

STATUS_COLORS = {
    "success": "#4daf4a",
    "soft_fail": "#ff7f00",
    "hard_fail": "#e41a1c",
}


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _save_fig(fig: plt.Figure, path: Path, fmt: str = "pdf", dpi: int = 300) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), format=fmt, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 3 — Execution efficacy + compliance
# ---------------------------------------------------------------------------


def figure3_execution_efficacy(
    phase_a_paths: List[Path],
    output_dir: Path,
    fmt: str = "pdf",
    dpi: int = 300,
) -> None:
    """Figure 3: completion rate, failure taxonomy, runtime distribution.

    (A) Completion rate by dataset (stacked bar).
    (B) Failure taxonomy bar chart.
    (C) Runtime distribution (violin/box) by dataset.
    """
    fig = plt.figure(figsize=(14, 4.5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.2], wspace=0.35)

    datasets_data: List[dict] = []
    for p in phase_a_paths:
        datasets_data.append(_load_json(p))

    # --- Panel A: Completion rate stacked bar ---
    ax_a = fig.add_subplot(gs[0])
    names = [d.get("dataset", "?") for d in datasets_data]
    short_names = [DATASETS.get(n, type("", (), {"short_name": n})()).short_name
                   if n in DATASETS else n for n in names]

    success_counts = [d["n_success"] for d in datasets_data]
    soft_counts = [d["n_soft_fail"] for d in datasets_data]
    hard_counts = [d["n_hard_fail"] for d in datasets_data]

    x = np.arange(len(names))
    bar_w = 0.5
    ax_a.bar(x, success_counts, bar_w, label="Success", color=STATUS_COLORS["success"])
    ax_a.bar(x, soft_counts, bar_w, bottom=success_counts, label="Soft fail",
             color=STATUS_COLORS["soft_fail"])
    bottoms = [s + sf for s, sf in zip(success_counts, soft_counts)]
    ax_a.bar(x, hard_counts, bar_w, bottom=bottoms, label="Hard fail",
             color=STATUS_COLORS["hard_fail"])

    # Add completion rate text
    for i, d in enumerate(datasets_data):
        rate = d["completion_rate"]
        ci = d["completion_ci"]
        ax_a.text(i, d["n_total"] + 2, f"{rate:.0%}\n[{ci[0]:.0%}, {ci[1]:.0%}]",
                  ha="center", va="bottom", fontsize=7)

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(short_names, fontsize=8)
    ax_a.set_ylabel("Cases")
    ax_a.set_title("(A) Completion rate", fontsize=10, fontweight="bold")
    ax_a.legend(fontsize=7, loc="upper right")

    # --- Panel B: Failure taxonomy ---
    ax_b = fig.add_subplot(gs[1])
    all_failures: Dict[str, int] = {}
    for d in datasets_data:
        for cat, count in d.get("failure_taxonomy", {}).items():
            all_failures[cat] = all_failures.get(cat, 0) + count

    if all_failures:
        cats = sorted(all_failures.keys(), key=lambda k: all_failures[k], reverse=True)
        counts = [all_failures[c] for c in cats]
        ax_b.barh(range(len(cats)), counts, color="#e41a1c", alpha=0.7)
        ax_b.set_yticks(range(len(cats)))
        ax_b.set_yticklabels([c.replace("_", " ").title() for c in cats], fontsize=8)
        ax_b.invert_yaxis()
    ax_b.set_xlabel("Count")
    ax_b.set_title("(B) Failure taxonomy", fontsize=10, fontweight="bold")

    # --- Panel C: Runtime distribution ---
    ax_c = fig.add_subplot(gs[2])
    all_runtimes: List[List[float]] = []
    for d in datasets_data:
        case_runtimes = [c["runtime_sec"] for c in d.get("cases", []) if c.get("runtime_sec", 0) > 0]
        all_runtimes.append(case_runtimes)

    if HAS_SEABORN:
        parts = ax_c.violinplot(all_runtimes, positions=x, widths=0.6, showmedians=True)
        for pc in parts.get("bodies", []):
            pc.set_facecolor("#377eb8")
            pc.set_alpha(0.6)
    else:
        ax_c.boxplot(all_runtimes, positions=x, widths=0.5)

    ax_c.set_xticks(x)
    ax_c.set_xticklabels(short_names, fontsize=8)
    ax_c.set_ylabel("Runtime (seconds)")
    ax_c.set_title("(C) Runtime distribution", fontsize=10, fontweight="bold")

    _save_fig(fig, output_dir / f"figure3_execution_efficacy.{fmt}", fmt, dpi)


# ---------------------------------------------------------------------------
# Figure 4 — Segmentation performance
# ---------------------------------------------------------------------------


def figure4_segmentation_performance(
    phase_b_paths: List[Path],
    output_dir: Path,
    fmt: str = "pdf",
    dpi: int = 300,
) -> None:
    """Figure 4: patient-wise + lesion-wise metrics.

    (A) Violin/box of Dice, HD95 per region (patient-level).
    (B) Lesion-wise Dice & HD95.
    (C) Stratified by lesion volume bin.
    """
    fig = plt.figure(figsize=(16, 5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.2, 1, 1], wspace=0.3)

    # Pool all datasets
    all_data: List[dict] = [_load_json(p) for p in phase_b_paths]

    # --- Panel A: Patient-level Dice ---
    ax_a = fig.add_subplot(gs[0])
    regions = list(REGION_MAP.keys())
    region_dice_values: Dict[str, List[float]] = {r: [] for r in regions}

    for d in all_data:
        for c in d.get("cases", []):
            pm = c.get("patient_metrics", {})
            for r in regions:
                val = pm.get(r, {}).get("dice")
                if val is not None:
                    region_dice_values[r].append(val)

    positions = np.arange(len(regions))
    box_data = [region_dice_values[r] for r in regions]
    bp = ax_a.boxplot(box_data, positions=positions, widths=0.5, patch_artist=True,
                      showfliers=True, flierprops=dict(markersize=2))
    for i, (patch, region) in enumerate(zip(bp["boxes"], regions)):
        patch.set_facecolor(REGION_COLORS.get(region, "#999999"))
        patch.set_alpha(0.6)

    ax_a.set_xticks(positions)
    ax_a.set_xticklabels(regions)
    ax_a.set_ylabel("Dice coefficient")
    ax_a.set_title("(A) Patient-level Dice", fontsize=10, fontweight="bold")
    ax_a.set_ylim(-0.05, 1.05)

    # --- Panel B: Lesion-wise summary ---
    ax_b = fig.add_subplot(gs[1])
    for d in all_data:
        lw = d.get("lesion_wise", {})
        labels_used = set()
        for i, region in enumerate(regions):
            info = lw.get(region, {}).get("dice", {})
            if info:
                median = info.get("median", 0)
                ci_lo = info.get("ci_lower", median)
                ci_hi = info.get("ci_upper", median)
                color = REGION_COLORS.get(region, "#999999")
                label = region if region not in labels_used else None
                labels_used.add(region)
                ax_b.errorbar(i, median, yerr=[[median - ci_lo], [ci_hi - median]],
                              fmt="o", color=color, capsize=4, markersize=8, label=label)

    ax_b.set_xticks(range(len(regions)))
    ax_b.set_xticklabels(regions)
    ax_b.set_ylabel("Lesion-wise Dice")
    ax_b.set_title("(B) Lesion-wise Dice", fontsize=10, fontweight="bold")
    ax_b.set_ylim(-0.05, 1.05)
    ax_b.legend(fontsize=7)

    # --- Panel C: Volume-bin stratified sensitivity ---
    ax_c = fig.add_subplot(gs[2])
    for d in all_data:
        vbs = d.get("volume_bin_sensitivity", {})
        for region in regions:
            bins_data = vbs.get(region, {})
            if bins_data:
                bin_labels = list(bins_data.keys())
                sens_vals = [bins_data[b] for b in bin_labels]
                valid = [(lbl, v) for lbl, v in zip(bin_labels, sens_vals) if v is not None]
                if valid:
                    labels, vals = zip(*valid)
                    ax_c.plot(range(len(labels)), vals, "o-",
                              color=REGION_COLORS.get(region, "#999999"),
                              label=region, markersize=6)
                    ax_c.set_xticks(range(len(labels)))
                    ax_c.set_xticklabels(labels, fontsize=8)

    ax_c.set_ylabel("Detection sensitivity")
    ax_c.set_xlabel("Lesion volume bin")
    ax_c.set_title("(C) Size-stratified sensitivity", fontsize=10, fontweight="bold")
    ax_c.set_ylim(-0.05, 1.05)
    ax_c.legend(fontsize=7)

    _save_fig(fig, output_dir / f"figure4_segmentation.{fmt}", fmt, dpi)


# ---------------------------------------------------------------------------
# Figure 5 — Robustness / stress testing
# ---------------------------------------------------------------------------


def figure5_robustness(
    phase_c_paths: List[Path],
    output_dir: Path,
    fmt: str = "pdf",
    dpi: int = 300,
) -> None:
    """Figure 5: Degradation curves + AUC summary.

    Left panel: performance vs perturbation severity with CI ribbons.
    Right panel: AUC bar chart comparing perturbation types.
    """
    all_data = [_load_json(p) for p in phase_c_paths]

    perturbation_types = set()
    for d in all_data:
        perturbation_types.update(d.get("degradation_curves", {}).keys())
    perturbation_types = sorted(perturbation_types)
    n_perts = max(len(perturbation_types), 1)

    fig = plt.figure(figsize=(5 * n_perts + 4, 4.5))
    gs = gridspec.GridSpec(1, n_perts + 1, width_ratios=[1] * n_perts + [0.8], wspace=0.3)

    regions = list(REGION_MAP.keys())

    # --- Degradation curves per perturbation type ---
    for pi, pert_name in enumerate(perturbation_types):
        ax = fig.add_subplot(gs[pi])
        for d in all_data:
            curve = d.get("degradation_curves", {}).get(pert_name, [])
            if not curve:
                continue
            for region in regions:
                levels = []
                medians = []
                for level_data in curve:
                    rm = level_data.get("patient_metrics", {}).get(region, {})
                    if "dice_median" in rm:
                        levels.append(level_data["level_value"])
                        medians.append(rm["dice_median"])

                if levels:
                    color = REGION_COLORS.get(region, "#999999")
                    ax.plot(levels, medians, "o-", color=color, label=region, markersize=4)

        spec = PERTURBATIONS.get(pert_name)
        xlabel = spec.unit if spec else "Severity"
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("Dice" if pi == 0 else "")
        ax.set_title(pert_name.replace("_", " ").title(), fontsize=9, fontweight="bold")
        ax.set_ylim(-0.05, 1.05)
        if pi == 0:
            ax.legend(fontsize=6)

    # --- AUC summary bar chart ---
    ax_auc = fig.add_subplot(gs[-1])
    for d in all_data:
        auc_sums = d.get("auc_summaries", {})
        for region in regions:
            auc_vals = []
            pert_labels = []
            for pert_name in perturbation_types:
                val = auc_sums.get(pert_name, {}).get(region)
                if val is not None:
                    auc_vals.append(val)
                    pert_labels.append(pert_name[:6])

            if auc_vals:
                x_pos = np.arange(len(auc_vals))
                offset = list(regions).index(region) * 0.15
                color = REGION_COLORS.get(region, "#999999")
                ax_auc.bar(x_pos + offset, auc_vals, 0.12,
                           label=region if region not in
                           [r for r in regions[:list(regions).index(region)]] else None,
                           color=color, alpha=0.7)

    if perturbation_types:
        ax_auc.set_xticks(np.arange(len(perturbation_types)) + 0.2)
        ax_auc.set_xticklabels([p[:8] for p in perturbation_types], fontsize=7, rotation=30)
    ax_auc.set_ylabel("AUC")
    ax_auc.set_title("AUC summary", fontsize=9, fontweight="bold")
    ax_auc.legend(fontsize=6)

    _save_fig(fig, output_dir / f"figure5_robustness.{fmt}", fmt, dpi)


# ---------------------------------------------------------------------------
# Figure 6 — Quantitative stability / biomarker readiness
# ---------------------------------------------------------------------------


def figure6_quantitative_stability(
    phase_d_paths: List[Path],
    output_dir: Path,
    fmt: str = "pdf",
    dpi: int = 300,
) -> None:
    """Figure 6: Bland–Altman, ICC, longitudinal plausibility.

    (A) Bland–Altman plots per region.
    (B) ICC bar chart.
    (C) Longitudinal trajectory examples (if available).
    """
    all_data = [_load_json(p) for p in phase_d_paths]
    regions = list(REGION_MAP.keys())

    fig = plt.figure(figsize=(14, 5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.5, 0.8, 1.2], wspace=0.35)

    # --- Panel A: Bland–Altman ---
    n_regions = len(regions)
    gs_inner = gridspec.GridSpecFromSubplotSpec(1, n_regions, subplot_spec=gs[0], wspace=0.3)

    for ri, region in enumerate(regions):
        ax = fig.add_subplot(gs_inner[ri])
        for d in all_data:
            cases = d.get("cases", [])
            preds = [c["volumes_pred"].get(region, 0) for c in cases]
            gts = [c["volumes_gt"].get(region, 0) for c in cases]
            preds_arr = np.array(preds)
            gts_arr = np.array(gts)

            means = (preds_arr + gts_arr) / 2
            diffs = preds_arr - gts_arr

            color = REGION_COLORS.get(region, "#999999")
            ax.scatter(means, diffs, s=8, alpha=0.5, color=color)

            # Mean diff and LoA lines
            ba_stats = d.get("bland_altman", {}).get(region, {})
            if ba_stats:
                md = ba_stats.get("mean_diff", 0)
                loa_lo = ba_stats.get("loa_lower", 0)
                loa_hi = ba_stats.get("loa_upper", 0)
                ax.axhline(md, color="k", linestyle="-", linewidth=0.8)
                ax.axhline(loa_lo, color="k", linestyle="--", linewidth=0.5)
                ax.axhline(loa_hi, color="k", linestyle="--", linewidth=0.5)

        ax.axhline(0, color="gray", linestyle=":", linewidth=0.5)
        ax.set_xlabel("Mean (cc)", fontsize=7)
        if ri == 0:
            ax.set_ylabel("Pred − GT (cc)", fontsize=7)
        ax.set_title(region, fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=6)

    # --- Panel B: ICC ---
    ax_icc = fig.add_subplot(gs[1])
    for d in all_data:
        icc_data = d.get("icc", {})
        icc_vals = []
        icc_cis = []
        region_labels = []
        for region in regions:
            info = icc_data.get(region, {})
            if info:
                icc_vals.append(info["icc"])
                icc_cis.append((info["icc"] - info["ci_lower"], info["ci_upper"] - info["icc"]))
                region_labels.append(region)

        if icc_vals:
            x_pos = np.arange(len(icc_vals))
            colors = [REGION_COLORS.get(r, "#999999") for r in region_labels]
            yerr = np.array(icc_cis).T
            ax_icc.bar(x_pos, icc_vals, 0.6, color=colors, alpha=0.7, yerr=yerr, capsize=3)
            ax_icc.set_xticks(x_pos)
            ax_icc.set_xticklabels(region_labels, fontsize=8)

    ax_icc.set_ylabel("ICC(3,1)")
    ax_icc.set_title("(B) ICC", fontsize=10, fontweight="bold")
    ax_icc.set_ylim(0, 1.05)

    # --- Panel C: Longitudinal plausibility ---
    ax_long = fig.add_subplot(gs[2])
    for d in all_data:
        cases = d.get("cases", [])
        # Group by patient, plot trajectories for a few patients
        from collections import defaultdict
        patient_data: Dict[str, List[tuple]] = defaultdict(list)
        for c in cases:
            patient_data[c["subject"]].append((c["timepoint_index"], c["volumes_pred"]))

        # Plot up to 10 patients for WT region
        n_plotted = 0
        for subj, tps in sorted(patient_data.items()):
            if len(tps) < 2:
                continue
            tps.sort(key=lambda x: x[0])
            tp_indices = [t[0] for t in tps]
            wt_vols = [t[1].get("WT", 0) for t in tps]
            ax_long.plot(tp_indices, wt_vols, "o-", alpha=0.4, markersize=3, linewidth=0.8)
            n_plotted += 1
            if n_plotted >= 15:
                break

        # Mark implausible jumps
        jumps = d.get("implausible_cases", [])
        for j in jumps[:10]:
            ax_long.annotate("!", xy=(0, 0), fontsize=6, color="red")

    ax_long.set_xlabel("Timepoint index")
    ax_long.set_ylabel("WT volume (cc)")
    ax_long.set_title("(C) Longitudinal trajectories", fontsize=10, fontweight="bold")

    _save_fig(fig, output_dir / f"figure6_quantitative_stability.{fmt}", fmt, dpi)


# ---------------------------------------------------------------------------
# Figure 6b — Radiomic feature stability (native-first vs atlas-first)
# ---------------------------------------------------------------------------

# Colour palette for feature classes
FEATURE_CLASS_COLORS = {
    "firstorder": "#1f77b4",
    "shape": "#ff7f0e",
    "glcm": "#2ca02c",
    "glrlm": "#d62728",
    "glszm": "#9467bd",
    "gldm": "#8c564b",
    "ngtdm": "#e377c2",
    "unknown": "#999999",
}


def figure6b_radiomics_stability(
    radiomics_paths: List[Path],
    output_dir: Path,
    fmt: str = "pdf",
    dpi: int = 300,
) -> None:
    """Figure 6b: Radiomic feature stability across preprocessing architectures.

    (A) Per-feature ICC scatter (native CV vs atlas CV, coloured by class).
    (B) Per-class stacked bar (stable vs unstable features).
    (C) CV distribution comparison (native vs atlas, violin/box).
    """
    all_data = [_load_json(p) for p in radiomics_paths]

    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.2, 0.8, 1.0], wspace=0.35)

    # Gather feature-level records across all datasets
    all_features: List[Dict] = []
    for d in all_data:
        all_features.extend(d.get("features", []))

    icc_thresh = all_data[0].get("icc_threshold", 0.85) if all_data else 0.85
    cv_thresh = all_data[0].get("cv_threshold", 10.0) if all_data else 10.0

    # --- Panel A: ICC vs CV scatter ---
    ax_a = fig.add_subplot(gs[0])
    for feat in all_features:
        icc_val = feat.get("icc")
        cv_n = feat.get("cv_native")
        if icc_val is None or cv_n is None:
            continue
        cls = feat.get("feature_class", "unknown")
        color = FEATURE_CLASS_COLORS.get(cls, "#999999")
        ax_a.scatter(cv_n, icc_val, s=12, alpha=0.5, color=color, edgecolors="none")

    # Threshold lines
    ax_a.axhline(icc_thresh, color="k", linestyle="--", linewidth=0.8, alpha=0.6)
    ax_a.axvline(cv_thresh, color="k", linestyle="--", linewidth=0.8, alpha=0.6)
    ax_a.set_xlabel("CV (%) — Native-first", fontsize=9)
    ax_a.set_ylabel("ICC(3,1)", fontsize=9)
    ax_a.set_title("(A) Feature stability landscape", fontsize=10, fontweight="bold")
    ax_a.set_ylim(-0.1, 1.05)
    ax_a.set_xlim(left=0)

    # Legend for feature classes
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=6, label=cls)
        for cls, c in FEATURE_CLASS_COLORS.items()
        if cls != "unknown"
    ]
    ax_a.legend(handles=handles, fontsize=6, loc="lower left", ncol=2, framealpha=0.8)

    # --- Panel B: Per-class stacked bar ---
    ax_b = fig.add_subplot(gs[1])

    # Aggregate class summary from all datasets
    combined_class: Dict[str, Dict[str, int]] = {}
    for d in all_data:
        for cls, sums in d.get("class_summary", {}).items():
            if cls not in combined_class:
                combined_class[cls] = {"n_stable": 0, "n_unstable": 0}
            combined_class[cls]["n_stable"] += int(sums.get("n_stable", 0))
            n_t = int(sums.get("n_total", 0))
            n_s = int(sums.get("n_stable", 0))
            combined_class[cls]["n_unstable"] += max(n_t - n_s, 0)

    if combined_class:
        classes = sorted(combined_class.keys())
        n_stable = [combined_class[c]["n_stable"] for c in classes]
        n_unstable = [combined_class[c]["n_unstable"] for c in classes]
        x_pos = np.arange(len(classes))

        ax_b.bar(x_pos, n_stable, 0.6, label="Stable", color="#4daf4a", alpha=0.8)
        ax_b.bar(x_pos, n_unstable, 0.6, bottom=n_stable,
                 label="Unstable", color="#e41a1c", alpha=0.6)
        ax_b.set_xticks(x_pos)
        ax_b.set_xticklabels(classes, fontsize=7, rotation=45, ha="right")
        ax_b.legend(fontsize=7)

    ax_b.set_ylabel("Feature count")
    ax_b.set_title("(B) Stability by class", fontsize=10, fontweight="bold")

    # --- Panel C: CV distribution comparison ---
    ax_c = fig.add_subplot(gs[2])

    cv_native = [f["cv_native"] for f in all_features
                 if f.get("cv_native") is not None]
    cv_atlas = [f["cv_atlas"] for f in all_features
                if f.get("cv_atlas") is not None]

    if cv_native and cv_atlas:
        bp = ax_c.boxplot(
            [cv_native, cv_atlas],
            labels=["Native-first", "Atlas-first"],
            widths=0.5,
            patch_artist=True,
            medianprops=dict(color="black"),
        )
        colors_box = ["#377eb8", "#ff7f00"]
        for patch, color in zip(bp["boxes"], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        # annotate Wilcoxon result if available
        for d in all_data:
            w_p = d.get("wilcoxon_p_value")
            if w_p is not None:
                sig_str = f"p = {w_p:.4f}"
                if d.get("wilcoxon_fdr_rejected"):
                    sig_str += " *"
                ax_c.annotate(
                    sig_str, xy=(1.5, max(max(cv_native), max(cv_atlas)) * 0.95),
                    ha="center", fontsize=8,
                )
                break

    ax_c.set_ylabel("CV (%)")
    ax_c.set_title("(C) CV distribution", fontsize=10, fontweight="bold")

    _save_fig(fig, output_dir / f"figure6b_radiomics_stability.{fmt}", fmt, dpi)


# ---------------------------------------------------------------------------
# Figure 7 — Human factors
# ---------------------------------------------------------------------------


def figure7_human_factors(
    phase_e_path: Path,
    output_dir: Path,
    fmt: str = "pdf",
    dpi: int = 300,
) -> None:
    """Figure 7 (optional): Time-to-acceptable, edit magnitude, QC examples.

    (A) Time-to-acceptable: manual vs assisted (paired).
    (B) Edit magnitude distribution.
    (C) Placeholder for QC report panels.
    """
    d = _load_json(phase_e_path)

    fig = plt.figure(figsize=(12, 4))
    gs = gridspec.GridSpec(1, 3, wspace=0.35)

    # --- Panel A: Time comparison ---
    ax_a = fig.add_subplot(gs[0])
    conditions = ["Manual", "Assisted"]
    medians = [d.get("time_manual_median", 0), d.get("time_assisted_median", 0)]
    cis_lo = [d.get("time_manual_ci", [0, 0])[0], d.get("time_assisted_ci", [0, 0])[0]]
    cis_hi = [d.get("time_manual_ci", [0, 0])[1], d.get("time_assisted_ci", [0, 0])[1]]

    yerr_lo = [m - lo for m, lo in zip(medians, cis_lo)]
    yerr_hi = [hi - m for m, hi in zip(medians, cis_hi)]

    colors = ["#e41a1c", "#4daf4a"]
    ax_a.bar(range(2), medians, 0.5, color=colors, alpha=0.7,
             yerr=[yerr_lo, yerr_hi], capsize=5)
    ax_a.set_xticks(range(2))
    ax_a.set_xticklabels(conditions)
    ax_a.set_ylabel("Time to acceptable (sec)")
    ax_a.set_title("(A) Time-to-acceptable", fontsize=10, fontweight="bold")

    # --- Panel B: Edit magnitude ---
    ax_b = fig.add_subplot(gs[1])
    # We'd plot distribution of edit magnitudes here
    em = d.get("edit_magnitude_median", 0)
    em_ci = d.get("edit_magnitude_ci", [0, 0])
    ax_b.bar([0], [em * 100], 0.4, color="#377eb8", alpha=0.7,
             yerr=[[em * 100 - em_ci[0] * 100], [em_ci[1] * 100 - em * 100]], capsize=5)
    ax_b.set_xticks([0])
    ax_b.set_xticklabels(["Assisted"])
    ax_b.set_ylabel("Voxels modified (%)")
    ax_b.set_title("(B) Edit magnitude", fontsize=10, fontweight="bold")

    # --- Panel C: Likert scores ---
    ax_c = fig.add_subplot(gs[2])
    lm = d.get("likert_manual_median", 0)
    la = d.get("likert_assisted_median", 0)
    ax_c.bar([0, 1], [lm, la], 0.5, color=colors, alpha=0.7)
    ax_c.set_xticks([0, 1])
    ax_c.set_xticklabels(["Manual", "Assisted"])
    ax_c.set_ylabel("Acceptability (Likert 1–5)")
    ax_c.set_title("(C) Acceptability", fontsize=10, fontweight="bold")
    ax_c.set_ylim(0, 5.5)

    _save_fig(fig, output_dir / f"figure7_human_factors.{fmt}", fmt, dpi)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate validation manuscript figures")
    parser.add_argument("--results-dir", type=Path, default=Path("validation_results"),
                        help="Root of phase result JSONs")
    parser.add_argument("--output-dir", type=Path, default=Path("figures"),
                        help="Output directory for figure files")
    parser.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--figures", nargs="*", default=["3", "4", "5", "6", "6b", "7"],
                        help="Which figures to generate (3–7, 6b)")

    args = parser.parse_args()
    results_dir = args.results_dir

    if "3" in args.figures:
        phase_a_files = sorted(results_dir.glob("phase_a/phase_a_*.json"))
        if phase_a_files:
            figure3_execution_efficacy(phase_a_files, args.output_dir, args.format, args.dpi)
        else:
            print("  Skipping Figure 3: no Phase A results found")

    if "4" in args.figures:
        phase_b_files = sorted(results_dir.glob("phase_b/phase_b_*.json"))
        if phase_b_files:
            figure4_segmentation_performance(phase_b_files, args.output_dir, args.format, args.dpi)
        else:
            print("  Skipping Figure 4: no Phase B results found")

    if "5" in args.figures:
        phase_c_files = sorted(results_dir.glob("phase_c/phase_c_*.json"))
        if phase_c_files:
            figure5_robustness(phase_c_files, args.output_dir, args.format, args.dpi)
        else:
            print("  Skipping Figure 5: no Phase C results found")

    if "6" in args.figures:
        phase_d_files = sorted(results_dir.glob("phase_d/phase_d_*.json"))
        if phase_d_files:
            figure6_quantitative_stability(phase_d_files, args.output_dir, args.format, args.dpi)
        else:
            print("  Skipping Figure 6: no Phase D results found")

    if "6b" in args.figures:
        rad_stability_files = sorted(results_dir.glob("phase_d/radiomics_stability_*.json"))
        if rad_stability_files:
            figure6b_radiomics_stability(rad_stability_files, args.output_dir, args.format, args.dpi)
        else:
            print("  Skipping Figure 6b: no radiomics stability results found")

    if "7" in args.figures:
        phase_e_file = results_dir / "phase_e" / "phase_e_results.json"
        if phase_e_file.exists():
            figure7_human_factors(phase_e_file, args.output_dir, args.format, args.dpi)
        else:
            print("  Skipping Figure 7: no Phase E results found")


if __name__ == "__main__":
    main()
