#!/usr/bin/env python
"""Main runner — execute the full OncoPrep validation protocol.

Orchestrates Phases A–E, figure generation, and table generation.

Usage examples:

    # Run full protocol on a dataset
    python -m validation.run_all --bids-dir /data/bids --gt-dir /data/gt \\
        --output-dir /data/out --dataset ucsf_pdgm --phases A B C D

    # Generate figures + tables from existing results
    python -m validation.run_all --results-dir ./validation_results \\
        --phases figures tables

    # Dry run: just validate config
    python -m validation.run_all --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

from .config import (
    COMPARATORS,
    DATASETS,
    PERTURBATIONS,
    SAP,
    get_output_root,
)


def run_all(
    bids_dir: Optional[Path] = None,
    gt_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    work_dir: Path = Path("work"),
    results_dir: Path = Path("validation_results"),
    dataset_key: Optional[str] = None,
    phases: Optional[List[str]] = None,
    nprocs: int = 4,
    mem_gb: float = 8.0,
    timeout_sec: int = 7200,
    max_cases: Optional[int] = None,
    comparator_dir: Optional[Path] = None,
    figure_format: str = "pdf",
    figure_dpi: int = 300,
    dry_run: bool = False,
) -> None:
    """Orchestrate the complete validation protocol."""
    if phases is None:
        phases = ["A", "B", "C", "D", "figures", "tables"]

    if dry_run:
        _print_config(dataset_key, phases)
        return

    ts = time.strftime("%Y%m%d_%H%M%S")
    print(f"OncoPrep Validation Protocol — run started at {ts}")
    print(f"  Dataset: {dataset_key}")
    print(f"  Phases:  {', '.join(phases)}")
    print(f"  Results: {results_dir}")

    # --- Phase A ---
    if "A" in phases:
        print(f"\n{'='*60}")
        print("Phase A: Standards + execution efficacy")
        print(f"{'='*60}")
        if bids_dir is None or output_dir is None:
            print("  SKIP: --bids-dir and --output-dir required for Phase A")
        else:
            from .phase_a import run_phase_a, save_phase_a_results
            results_a = run_phase_a(
                bids_dir=bids_dir,
                output_dir=output_dir,
                work_dir=work_dir,
                dataset_key=dataset_key or "unknown",
                nprocs=nprocs,
                mem_gb=mem_gb,
                timeout_sec=timeout_sec,
                max_cases=max_cases,
            )
            out = results_dir / "phase_a" / f"phase_a_{dataset_key}.json"
            save_phase_a_results(results_a, out)
            print(f"  Completion: {results_a.completion_rate:.1%}")

    # --- Phase B ---
    if "B" in phases:
        print(f"\n{'='*60}")
        print("Phase B: Segmentation accuracy")
        print(f"{'='*60}")
        pred_dir = (output_dir / "oncoprep") if output_dir else None
        if pred_dir is None or gt_dir is None:
            print("  SKIP: --output-dir and --gt-dir required for Phase B")
        else:
            from .phase_b import evaluate_dataset, save_phase_b_results, compare_pipelines
            results_b = evaluate_dataset(
                pred_dir=pred_dir,
                gt_dir=gt_dir,
                dataset_key=dataset_key or "unknown",
                max_cases=max_cases,
            )
            if comparator_dir:
                comparator_results = evaluate_dataset(
                    pred_dir=comparator_dir,
                    gt_dir=gt_dir,
                    dataset_key=dataset_key or "unknown",
                    max_cases=max_cases,
                )
                results_b.comparator_deltas = compare_pipelines(results_b, comparator_results)

            out = results_dir / "phase_b" / f"phase_b_{dataset_key}.json"
            save_phase_b_results(results_b, out)
            print(f"  Evaluated {results_b.n_cases} cases")

    # --- Phase C ---
    if "C" in phases:
        print(f"\n{'='*60}")
        print("Phase C: Robustness and failure transparency")
        print(f"{'='*60}")
        if bids_dir is None or gt_dir is None or output_dir is None:
            print("  SKIP: --bids-dir, --gt-dir, and --output-dir required for Phase C")
        else:
            from .phase_c import run_phase_c, save_phase_c_results
            results_c = run_phase_c(
                bids_dir=bids_dir,
                gt_dir=gt_dir,
                output_dir=output_dir / "phase_c_work",
                work_dir=work_dir,
                dataset_key=dataset_key or "unknown",
                nprocs=nprocs,
                mem_gb=mem_gb,
                timeout_sec=timeout_sec,
                max_cases=max_cases,
            )
            out = results_dir / "phase_c" / f"phase_c_{dataset_key}.json"
            save_phase_c_results(results_c, out)
            print(f"  Completed {len(results_c.degradation_curves)} perturbation types")

    # --- Phase D ---
    if "D" in phases:
        print(f"\n{'='*60}")
        print("Phase D: Quantitative stability")
        print(f"{'='*60}")
        pred_dir = (output_dir / "oncoprep") if output_dir else None
        if pred_dir is None or gt_dir is None:
            print("  SKIP: --output-dir and --gt-dir required for Phase D")
        else:
            from .phase_d import run_phase_d, save_phase_d_results
            results_d = run_phase_d(
                pred_dir=pred_dir,
                gt_dir=gt_dir,
                dataset_key=dataset_key or "unknown",
                max_cases=max_cases,
            )
            out = results_dir / "phase_d" / f"phase_d_{dataset_key}.json"
            save_phase_d_results(results_d, out)
            print(f"  Evaluated {results_d.n_cases} cases")

    # --- Figures ---
    if "figures" in phases:
        print(f"\n{'='*60}")
        print("Generating figures")
        print(f"{'='*60}")
        from .figures import main as figures_main
        sys.argv = [
            "figures",
            "--results-dir", str(results_dir),
            "--output-dir", str(results_dir / "figures"),
            "--format", figure_format,
            "--dpi", str(figure_dpi),
        ]
        figures_main()

    # --- Tables ---
    if "tables" in phases:
        print(f"\n{'='*60}")
        print("Generating tables")
        print(f"{'='*60}")
        from .tables import main as tables_main
        sys.argv = [
            "tables",
            "--results-dir", str(results_dir),
            "--output-dir", str(results_dir / "tables"),
            "--format", "latex",
        ]
        tables_main()

    print(f"\n{'='*60}")
    print(f"Validation run complete. Results in: {results_dir}")
    print(f"{'='*60}")


def _print_config(dataset_key: Optional[str], phases: List[str]) -> None:
    """Print configuration summary for dry run."""
    print("OncoPrep Validation Protocol — DRY RUN")
    print(f"\n  Phases requested: {', '.join(phases)}")
    print(f"\n  Available datasets:")
    for key, ds in DATASETS.items():
        marker = " ← selected" if key == dataset_key else ""
        print(f"    {key}: {ds.short_name} (n={ds.n_subjects}){marker}")
    print(f"\n  Comparators:")
    for key, comp in COMPARATORS.items():
        print(f"    {key}: {comp.description}")
    print(f"\n  SAP config:")
    print(f"    Bootstrap iterations: {SAP.n_bootstrap}")
    print(f"    CI level: {SAP.ci_level}")
    print(f"    Random seed: {SAP.random_seed}")
    print(f"    Perturbation types: {SAP.perturbation_types}")
    print(f"\n  Perturbation specs:")
    for name, spec in PERTURBATIONS.items():
        print(f"    {name}: {spec.levels} ({spec.unit})")
    print(f"\n  Sample size heuristics:")
    from .stats import sample_size_ci_proportion, sample_size_paired_continuous
    n_prop = sample_size_ci_proportion(target_width=2 * SAP.target_ci_width_proportion)
    n_cont = sample_size_paired_continuous(SAP.detectable_effect, SAP.assumed_sigma_delta)
    print(f"    Phase A (±{SAP.target_ci_width_proportion:.0%} CI): ~{n_prop} cases")
    print(f"    Phase B (δ={SAP.detectable_effect}, σ={SAP.assumed_sigma_delta}): ~{n_cont} cases")
    print(f"    Phase E: ≥{SAP.min_cases} cases × ≥{SAP.min_readers} readers")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OncoPrep validation protocol — full orchestration",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--bids-dir", type=Path, help="BIDS dataset root")
    parser.add_argument("--gt-dir", type=Path, help="Ground truth directory")
    parser.add_argument("--output-dir", type=Path, help="Pipeline output root")
    parser.add_argument("--work-dir", type=Path, default=Path("work"))
    parser.add_argument("--results-dir", type=Path, default=Path("validation_results"))
    parser.add_argument("--dataset", type=str, choices=list(DATASETS.keys()))
    parser.add_argument("--phases", nargs="*",
                        default=["A", "B", "C", "D", "figures", "tables"],
                        help="Phases to run: A B C D E figures tables")
    parser.add_argument("--comparator-dir", type=Path)
    parser.add_argument("--nprocs", type=int, default=4)
    parser.add_argument("--mem-gb", type=float, default=8.0)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--figure-format", default="pdf", choices=["pdf", "png", "svg"])
    parser.add_argument("--figure-dpi", type=int, default=300)
    parser.add_argument("--dry-run", action="store_true", help="Print config and exit")

    args = parser.parse_args()

    run_all(
        bids_dir=args.bids_dir,
        gt_dir=args.gt_dir,
        output_dir=args.output_dir,
        work_dir=args.work_dir,
        results_dir=args.results_dir,
        dataset_key=args.dataset,
        phases=args.phases,
        nprocs=args.nprocs,
        mem_gb=args.mem_gb,
        timeout_sec=args.timeout,
        max_cases=args.max_cases,
        comparator_dir=args.comparator_dir,
        figure_format=args.figure_format,
        figure_dpi=args.figure_dpi,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
