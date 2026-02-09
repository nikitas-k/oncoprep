#!/usr/bin/env python
"""Table generation for the OncoPrep validation manuscript.

Generates Tables 1–4 from the blueprint as publication-ready LaTeX and CSV,
reading Phase A–E result JSONs.

Usage:
    python -m validation.tables --results-dir ./validation_results \\
        --output-dir ./tables --format latex
"""

from __future__ import annotations

import argparse
import csv
import json
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .config import DATASETS, PERTURBATIONS, REGION_MAP


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _save_table(content: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Table 1 — Cohort summary
# ---------------------------------------------------------------------------


def table1_cohort_summary(output_dir: Path, fmt: str = "latex") -> None:
    """Table 1: Dataset/cohort summary.

    Columns: Dataset, N subjects, N sessions, Vendor/field, Label set,
             GT type, Split method.
    """
    rows = []
    for key, ds in DATASETS.items():
        labels_str = ", ".join(str(l) for l in ds.label_set)
        rows.append([
            ds.short_name,
            str(ds.n_subjects),
            str(ds.n_sessions),
            f"{ds.vendor} / {ds.field_strength}",
            labels_str,
            ds.ground_truth_type,
            ds.split_method,
        ])

    headers = ["Dataset", "N subj.", "N sess.", "Vendor / Field", "Labels", "GT type", "Split"]

    if fmt == "latex":
        content = _to_latex(headers, rows, caption="Cohort summary", label="tab:cohort")
    else:
        content = _to_csv(headers, rows)

    ext = "tex" if fmt == "latex" else "csv"
    _save_table(content, output_dir / f"table1_cohort.{ext}")


# ---------------------------------------------------------------------------
# Table 2 — Primary endpoints (headline results)
# ---------------------------------------------------------------------------


def table2_primary_endpoints(
    phase_b_paths: List[Path],
    output_dir: Path,
    fmt: str = "latex",
) -> None:
    """Table 2: Segmentation metrics per dataset × region.

    Columns: Dataset, Region, LW-Dice, LW-HD95, Pt-Dice, Pt-HD95,
             SurfDice 1mm, Comparator Δ Dice.
    """
    rows: List[List[str]] = []
    regions = list(REGION_MAP.keys())

    for p in phase_b_paths:
        d = _load_json(p)
        ds_name = d.get("dataset", "?")
        short = DATASETS.get(ds_name, type("", (), {"short_name": ds_name})()).short_name \
            if ds_name in DATASETS else ds_name

        for region in regions:
            # Patient-level
            pt = d.get("patient_level", {}).get(region, {})
            pt_dice = _fmt_ci(pt.get("dice", {}))
            pt_hd95 = _fmt_ci(pt.get("hd95", {}), decimals=1)
            pt_sdice = _fmt_ci(pt.get("surface_dice_1.0mm", {}))

            # Lesion-wise
            lw = d.get("lesion_wise", {}).get(region, {})
            lw_dice = _fmt_ci(lw.get("dice", {}))
            lw_hd95 = _fmt_ci(lw.get("hd95", {}), decimals=1)

            # Comparator delta
            cd = d.get("comparator_deltas", {}).get(region, {})
            delta_str = _fmt_delta(cd) if cd else "—"

            rows.append([short, region, lw_dice, lw_hd95, pt_dice, pt_hd95, pt_sdice, delta_str])

    headers = ["Dataset", "Region", "LW Dice", "LW HD95", "Pt Dice", "Pt HD95",
               "Surf Dice 1mm", "Δ Dice (vs C1)"]

    if fmt == "latex":
        content = _to_latex(headers, rows, caption="Primary segmentation endpoints", label="tab:endpoints")
    else:
        content = _to_csv(headers, rows)

    ext = "tex" if fmt == "latex" else "csv"
    _save_table(content, output_dir / f"table2_endpoints.{ext}")


# ---------------------------------------------------------------------------
# Table 3 — Reliability & robustness
# ---------------------------------------------------------------------------


def table3_reliability(
    phase_a_paths: List[Path],
    phase_c_paths: List[Path],
    output_dir: Path,
    fmt: str = "latex",
) -> None:
    """Table 3: Completion, compliance, stress-test AUC, top failure modes."""
    rows: List[List[str]] = []

    # Phase A rows
    for p in phase_a_paths:
        d = _load_json(p)
        ds = d.get("dataset", "?")
        short = DATASETS.get(ds, type("", (), {"short_name": ds})()).short_name \
            if ds in DATASETS else ds

        compl = f"{d['completion_rate']:.1%} [{d['completion_ci'][0]:.1%}, {d['completion_ci'][1]:.1%}]"
        comply = f"{d['compliance_rate']:.1%} [{d['compliance_ci'][0]:.1%}, {d['compliance_ci'][1]:.1%}]"

        # Top failure modes
        ftax = d.get("failure_taxonomy", {})
        top_failures = ", ".join(f"{k}: {v}" for k, v in sorted(ftax.items(), key=lambda x: -x[1])[:3])
        if not top_failures:
            top_failures = "—"

        # Stress-test AUC (from Phase C)
        auc_str = "—"
        for cp in phase_c_paths:
            cd = _load_json(cp)
            if cd.get("dataset") == ds:
                auc_parts = []
                for pert, regions in cd.get("auc_summaries", {}).items():
                    wt = regions.get("WT")
                    if wt is not None:
                        auc_parts.append(f"{pert[:8]}={wt:.2f}")
                if auc_parts:
                    auc_str = "; ".join(auc_parts)
                break

        rows.append([short, compl, comply, auc_str, top_failures])

    headers = ["Dataset", "Completion", "Compliance", "Stress AUC (WT)", "Top failures"]

    if fmt == "latex":
        content = _to_latex(headers, rows, caption="Reliability and robustness", label="tab:reliability")
    else:
        content = _to_csv(headers, rows)

    ext = "tex" if fmt == "latex" else "csv"
    _save_table(content, output_dir / f"table3_reliability.{ext}")


# ---------------------------------------------------------------------------
# Table 4 — Compute & reproducibility
# ---------------------------------------------------------------------------


def table4_compute(
    phase_a_paths: List[Path],
    output_dir: Path,
    fmt: str = "latex",
) -> None:
    """Table 4: Runtime, memory, determinism, container versions."""
    rows: List[List[str]] = []

    for p in phase_a_paths:
        d = _load_json(p)
        ds = d.get("dataset", "?")
        short = DATASETS.get(ds, type("", (), {"short_name": ds})()).short_name \
            if ds in DATASETS else ds

        rt_med = f"{d.get('runtime_median_sec', 0):.0f}s"
        rt_p95 = f"{d.get('runtime_p95_sec', 0):.0f}s"
        iqr = d.get("runtime_iqr_sec", [0, 0])
        rt_iqr = f"[{iqr[0]:.0f}, {iqr[1]:.0f}]s"
        mem = f"{d.get('peak_rss_median_mb', 0):.0f} MB"

        # Determinism: count unique hashes
        hashes = [c.get("output_hash", "") for c in d.get("cases", []) if c.get("output_hash")]
        n_unique = len(set(hashes))
        n_total = len(hashes)
        determ_str = f"{n_unique}/{n_total} unique hashes" if n_total > 0 else "—"

        rows.append([short, rt_med, rt_iqr, rt_p95, mem, determ_str])

    headers = ["Dataset", "Median RT", "IQR RT", "p95 RT", "Peak RSS", "Determinism"]

    if fmt == "latex":
        content = _to_latex(headers, rows, caption="Compute and reproducibility", label="tab:compute")
    else:
        content = _to_csv(headers, rows)

    ext = "tex" if fmt == "latex" else "csv"
    _save_table(content, output_dir / f"table4_compute.{ext}")


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _fmt_ci(info: Dict[str, float], decimals: int = 3) -> str:
    """Format a {median, ci_lower, ci_upper} dict as 'X.XXX [X.XXX, X.XXX]'."""
    if not info:
        return "—"
    m = info.get("median", 0)
    lo = info.get("ci_lower", m)
    hi = info.get("ci_upper", m)
    return f"{m:.{decimals}f} [{lo:.{decimals}f}, {hi:.{decimals}f}]"


def _fmt_delta(info: Dict[str, float]) -> str:
    """Format a comparator delta dict."""
    delta = info.get("delta", 0)
    lo = info.get("ci_lower", delta)
    hi = info.get("ci_upper", delta)
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta:.3f} [{lo:.3f}, {hi:.3f}]"


def _to_latex(
    headers: List[str],
    rows: List[List[str]],
    caption: str = "",
    label: str = "",
) -> str:
    """Generate a LaTeX table string."""
    n_cols = len(headers)
    col_spec = "l" + "c" * (n_cols - 1)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(f"\\textbf{{{h}}}" for h in headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def _to_csv(headers: List[str], rows: List[List[str]]) -> str:
    """Generate CSV string."""
    buf = StringIO()
    writer = csv.writer(buf)
    writer.writerow(headers)
    writer.writerows(rows)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate validation manuscript tables")
    parser.add_argument("--results-dir", type=Path, default=Path("validation_results"))
    parser.add_argument("--output-dir", type=Path, default=Path("tables"))
    parser.add_argument("--format", default="latex", choices=["latex", "csv"])
    parser.add_argument("--tables", nargs="*", default=["1", "2", "3", "4"],
                        help="Which tables to generate (1–4)")

    args = parser.parse_args()
    results_dir = args.results_dir

    if "1" in args.tables:
        table1_cohort_summary(args.output_dir, args.format)

    if "2" in args.tables:
        phase_b_files = sorted(results_dir.glob("phase_b/phase_b_*.json"))
        if phase_b_files:
            table2_primary_endpoints(phase_b_files, args.output_dir, args.format)
        else:
            print("  Skipping Table 2: no Phase B results")

    if "3" in args.tables:
        phase_a_files = sorted(results_dir.glob("phase_a/phase_a_*.json"))
        phase_c_files = sorted(results_dir.glob("phase_c/phase_c_*.json"))
        if phase_a_files:
            table3_reliability(phase_a_files, phase_c_files, args.output_dir, args.format)
        else:
            print("  Skipping Table 3: no Phase A results")

    if "4" in args.tables:
        phase_a_files = sorted(results_dir.glob("phase_a/phase_a_*.json"))
        if phase_a_files:
            table4_compute(phase_a_files, args.output_dir, args.format)
        else:
            print("  Skipping Table 4: no Phase A results")


if __name__ == "__main__":
    main()
