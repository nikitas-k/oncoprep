#!/usr/bin/env python
"""Phase A — Standards + execution efficacy (engineering efficacy).

Primary endpoints:
    A1. Completion rate (overall and per dataset) with 95% Wilson CI.
    A2. Standards compliance — proportion of BIDS Derivatives conforming outputs.

Secondary endpoints:
    - Runtime distribution (median, IQR, p95)
    - Memory footprint (peak RSS)
    - Deterministic reproducibility across platforms/containers

Usage:
    python -m validation.phase_a --bids-dir /data/bids --output-dir /data/out \\
        --dataset ucsf_pdgm --results-dir ./validation_results
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import (
    DATASETS,
    SAP,
    get_phase_dir,
)
from .stats import wilson_ci


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    """Result of running OncoPrep on a single subject/session."""
    subject: str
    session: str
    dataset: str
    status: str  # "success", "soft_fail", "hard_fail"
    runtime_sec: float = 0.0
    peak_rss_mb: float = 0.0
    bids_compliant: bool = False
    bids_issues: List[str] = field(default_factory=list)
    failure_category: str = ""  # e.g. "registration", "skull_strip", "t1ce_ambiguity"
    error_message: str = ""
    output_hash: str = ""


@dataclass
class PhaseAResults:
    """Aggregated Phase A results for a single dataset."""
    dataset: str
    n_total: int = 0
    n_success: int = 0
    n_soft_fail: int = 0
    n_hard_fail: int = 0
    completion_rate: float = 0.0
    completion_ci_lower: float = 0.0
    completion_ci_upper: float = 0.0
    compliance_rate: float = 0.0
    compliance_ci_lower: float = 0.0
    compliance_ci_upper: float = 0.0
    runtime_median: float = 0.0
    runtime_iqr_lower: float = 0.0
    runtime_iqr_upper: float = 0.0
    runtime_p95: float = 0.0
    peak_rss_median_mb: float = 0.0
    failure_taxonomy: Dict[str, int] = field(default_factory=dict)
    cases: List[CaseResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# BIDS Derivatives compliance checker
# ---------------------------------------------------------------------------

FAILURE_CATEGORIES = [
    "registration",
    "skull_strip",
    "t1ce_ambiguity",
    "missing_modality",
    "docker_timeout",
    "segmentation_empty",
    "output_write",
    "unknown",
]


def check_bids_compliance(
    output_dir: Path,
    subject: str,
    session: Optional[str] = None,
) -> Tuple[bool, List[str]]:
    """Check whether OncoPrep derivatives conform to BIDS Derivatives conventions.

    Checks:
        - Presence of _mask.nii.gz or _dseg.nii.gz
        - Sidecar JSON with ``Sources`` provenance
        - Optional _dseg.tsv label look-up table
        - Optional _probseg.nii.gz

    Parameters
    ----------
    output_dir : Path
        BIDS derivatives root (e.g. ``output_dir/oncoprep/``).
    subject : str
        Subject label (without ``sub-`` prefix).
    session : str, optional
        Session label (without ``ses-`` prefix).

    Returns
    -------
    is_compliant : bool
    issues : list of str
    """
    issues: List[str] = []

    # Build expected path
    sub_dir = output_dir / f"sub-{subject}"
    if session:
        sub_dir = sub_dir / f"ses-{session}"
    anat_dir = sub_dir / "anat"

    if not anat_dir.exists():
        issues.append(f"Missing anat directory: {anat_dir}")
        return False, issues

    # Check for required derivative files
    files_in_anat = list(anat_dir.iterdir())
    filenames = [f.name for f in files_in_anat]

    has_dseg = any("_dseg.nii" in fn for fn in filenames)
    has_mask = any("_mask.nii" in fn for fn in filenames)

    if not has_dseg and not has_mask:
        issues.append("No _dseg.nii.gz or _mask.nii.gz found in anat/")

    # Check sidecar JSON with Sources
    json_files = [f for f in files_in_anat if f.suffix == ".json"]
    has_sources = False
    for jf in json_files:
        try:
            with open(jf) as fh:
                meta = json.load(fh)
            if "Sources" in meta:
                has_sources = True
                break
        except (json.JSONDecodeError, IOError):
            issues.append(f"Invalid JSON sidecar: {jf.name}")

    if not has_sources and json_files:
        issues.append("No JSON sidecar contains 'Sources' provenance key")

    # Check label LUT (_dseg.tsv)
    if has_dseg:
        has_lut = any("_dseg.tsv" in fn for fn in filenames)
        if not has_lut:
            issues.append("_dseg.nii.gz present but _dseg.tsv label LUT missing")

    is_compliant = len(issues) == 0
    return is_compliant, issues


# ---------------------------------------------------------------------------
# Single-case runner
# ---------------------------------------------------------------------------


def run_single_case(
    bids_dir: Path,
    output_dir: Path,
    work_dir: Path,
    subject: str,
    session: Optional[str],
    dataset: str,
    nprocs: int = 4,
    mem_gb: float = 8.0,
    timeout_sec: int = 7200,
) -> CaseResult:
    """Run OncoPrep on a single subject/session and collect Phase A metrics.

    Parameters
    ----------
    bids_dir : Path
        BIDS dataset root.
    output_dir : Path
        Derivative output directory.
    work_dir : Path
        Nipype working directory.
    subject, session : str
        Subject and optional session labels (without prefixes).
    dataset : str
        Dataset identifier for grouping results.
    nprocs : int
        Number of parallel processes for Nipype.
    mem_gb : float
        Memory limit in GB.
    timeout_sec : int
        Wall-clock timeout.

    Returns
    -------
    CaseResult
    """
    result = CaseResult(
        subject=subject,
        session=session or "",
        dataset=dataset,
    )

    cmd = [
        "oncoprep",
        str(bids_dir),
        str(output_dir),
        "participant",
        "--participant-label", subject,
        "--nprocs", str(nprocs),
        "--mem-gb", str(mem_gb),
        "--run-segmentation",
    ]
    if session:
        cmd.extend(["--session-label", session])

    start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            cwd=str(bids_dir.parent),
        )
        result.runtime_sec = time.monotonic() - start

        if proc.returncode == 0:
            # Check BIDS compliance
            deriv_dir = output_dir / "oncoprep"
            compliant, issues = check_bids_compliance(deriv_dir, subject, session)
            result.bids_compliant = compliant
            result.bids_issues = issues

            if compliant:
                result.status = "success"
            else:
                result.status = "soft_fail"
                result.failure_category = "output_write"
                result.error_message = "; ".join(issues)
        else:
            result.status = "hard_fail"
            result.error_message = proc.stderr[-2000:] if proc.stderr else "Non-zero exit"
            result.failure_category = _classify_failure(proc.stderr or "")

    except subprocess.TimeoutExpired:
        result.runtime_sec = time.monotonic() - start
        result.status = "hard_fail"
        result.failure_category = "docker_timeout"
        result.error_message = f"Timed out after {timeout_sec}s"

    except Exception:
        result.runtime_sec = time.monotonic() - start
        result.status = "hard_fail"
        result.failure_category = "unknown"
        result.error_message = traceback.format_exc()[-2000:]

    # Attempt to capture peak RSS (macOS / Linux)
    try:
        rusage = resource.getrusage(resource.RUSAGE_CHILDREN)
        # macOS returns bytes, Linux returns KB
        if sys.platform == "darwin":
            result.peak_rss_mb = rusage.ru_maxrss / (1024 ** 2)
        else:
            result.peak_rss_mb = rusage.ru_maxrss / 1024
    except Exception:
        pass

    # Compute output hash for reproducibility checks
    result.output_hash = _hash_outputs(output_dir, subject, session)

    return result


def _classify_failure(stderr: str) -> str:
    """Classify a failure from stderr into a taxonomy category."""
    stderr_lower = stderr.lower()
    if "registration" in stderr_lower or "ants" in stderr_lower:
        return "registration"
    if "skull" in stderr_lower or "bet" in stderr_lower or "strip" in stderr_lower:
        return "skull_strip"
    if "t1ce" in stderr_lower or "contrast" in stderr_lower:
        return "t1ce_ambiguity"
    if "missing" in stderr_lower or "not found" in stderr_lower:
        return "missing_modality"
    if "docker" in stderr_lower or "container" in stderr_lower:
        return "docker_timeout"
    if "empty" in stderr_lower or "no voxel" in stderr_lower:
        return "segmentation_empty"
    return "unknown"


def _hash_outputs(
    output_dir: Path,
    subject: str,
    session: Optional[str],
) -> str:
    """SHA256 hash of all NIfTI outputs for reproducibility checks."""
    sub_dir = output_dir / "oncoprep" / f"sub-{subject}"
    if session:
        sub_dir = sub_dir / f"ses-{session}"

    h = hashlib.sha256()
    if sub_dir.exists():
        for nii in sorted(sub_dir.rglob("*.nii.gz")):
            h.update(nii.read_bytes())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Dataset-level aggregation
# ---------------------------------------------------------------------------


def collect_subjects(bids_dir: Path) -> List[Tuple[str, Optional[str]]]:
    """Discover (subject, session) pairs from a BIDS directory.

    Returns
    -------
    list of (subject_label, session_label_or_None)
    """
    pairs: List[Tuple[str, Optional[str]]] = []
    for sub_dir in sorted(bids_dir.glob("sub-*")):
        subject = sub_dir.name.removeprefix("sub-")
        ses_dirs = sorted(sub_dir.glob("ses-*"))
        if ses_dirs:
            for ses_dir in ses_dirs:
                session = ses_dir.name.removeprefix("ses-")
                pairs.append((subject, session))
        else:
            pairs.append((subject, None))
    return pairs


def run_phase_a(
    bids_dir: Path,
    output_dir: Path,
    work_dir: Path,
    dataset_key: str,
    nprocs: int = 4,
    mem_gb: float = 8.0,
    timeout_sec: int = 7200,
    max_cases: Optional[int] = None,
) -> PhaseAResults:
    """Execute Phase A on a full dataset.

    Parameters
    ----------
    bids_dir : Path
        BIDS root for this dataset.
    output_dir : Path
        Derivative output root.
    work_dir : Path
        Working directory.
    dataset_key : str
        Key into ``DATASETS`` dict.
    max_cases : int, optional
        Cap on number of cases (for debugging / pilot).

    Returns
    -------
    PhaseAResults
    """
    subjects = collect_subjects(bids_dir)
    if max_cases is not None:
        subjects = subjects[:max_cases]

    results = PhaseAResults(dataset=dataset_key, n_total=len(subjects))
    failure_counts: Dict[str, int] = {cat: 0 for cat in FAILURE_CATEGORIES}

    for subject, session in subjects:
        case = run_single_case(
            bids_dir=bids_dir,
            output_dir=output_dir,
            work_dir=work_dir,
            subject=subject,
            session=session,
            dataset=dataset_key,
            nprocs=nprocs,
            mem_gb=mem_gb,
            timeout_sec=timeout_sec,
        )
        results.cases.append(case)
        if case.status == "success":
            results.n_success += 1
        elif case.status == "soft_fail":
            results.n_soft_fail += 1
        else:
            results.n_hard_fail += 1
            if case.failure_category in failure_counts:
                failure_counts[case.failure_category] += 1

    results.failure_taxonomy = {k: v for k, v in failure_counts.items() if v > 0}

    # --- A1: Completion rate + Wilson CI ---
    n_completed = results.n_success + results.n_soft_fail
    rate, ci_lo, ci_hi = wilson_ci(n_completed, results.n_total, SAP.wilson_ci_level)
    results.completion_rate = rate
    results.completion_ci_lower = ci_lo
    results.completion_ci_upper = ci_hi

    # --- A2: BIDS compliance rate + Wilson CI ---
    n_compliant = sum(1 for c in results.cases if c.bids_compliant)
    c_rate, c_lo, c_hi = wilson_ci(n_compliant, results.n_total, SAP.wilson_ci_level)
    results.compliance_rate = c_rate
    results.compliance_ci_lower = c_lo
    results.compliance_ci_upper = c_hi

    # --- Secondary: runtime stats ---
    runtimes = np.array([c.runtime_sec for c in results.cases if c.runtime_sec > 0])
    if len(runtimes) > 0:
        results.runtime_median = float(np.median(runtimes))
        results.runtime_iqr_lower = float(np.percentile(runtimes, 25))
        results.runtime_iqr_upper = float(np.percentile(runtimes, 75))
        results.runtime_p95 = float(np.percentile(runtimes, 95))

    rss_vals = np.array([c.peak_rss_mb for c in results.cases if c.peak_rss_mb > 0])
    if len(rss_vals) > 0:
        results.peak_rss_median_mb = float(np.median(rss_vals))

    return results


# ---------------------------------------------------------------------------
# Reproducibility check
# ---------------------------------------------------------------------------


def check_reproducibility(
    results_run1: PhaseAResults,
    results_run2: PhaseAResults,
) -> Dict[str, object]:
    """Compare output hashes between two independent runs.

    Returns
    -------
    dict with ``n_compared``, ``n_match``, ``n_mismatch``, ``mismatched_cases``.
    """
    hash1 = {(c.subject, c.session): c.output_hash for c in results_run1.cases}
    hash2 = {(c.subject, c.session): c.output_hash for c in results_run2.cases}
    common_keys = set(hash1.keys()) & set(hash2.keys())

    mismatches = []
    for key in sorted(common_keys):
        if hash1[key] and hash2[key] and hash1[key] != hash2[key]:
            mismatches.append({
                "subject": key[0],
                "session": key[1],
                "hash_run1": hash1[key],
                "hash_run2": hash2[key],
            })

    return {
        "n_compared": len(common_keys),
        "n_match": len(common_keys) - len(mismatches),
        "n_mismatch": len(mismatches),
        "mismatched_cases": mismatches,
    }


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def save_phase_a_results(results: PhaseAResults, output_path: Path) -> None:
    """Save Phase A results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Convert dataclass to dict, handling nested dataclasses
    data = {
        "dataset": results.dataset,
        "n_total": results.n_total,
        "n_success": results.n_success,
        "n_soft_fail": results.n_soft_fail,
        "n_hard_fail": results.n_hard_fail,
        "completion_rate": results.completion_rate,
        "completion_ci": [results.completion_ci_lower, results.completion_ci_upper],
        "compliance_rate": results.compliance_rate,
        "compliance_ci": [results.compliance_ci_lower, results.compliance_ci_upper],
        "runtime_median_sec": results.runtime_median,
        "runtime_iqr_sec": [results.runtime_iqr_lower, results.runtime_iqr_upper],
        "runtime_p95_sec": results.runtime_p95,
        "peak_rss_median_mb": results.peak_rss_median_mb,
        "failure_taxonomy": results.failure_taxonomy,
        "cases": [asdict(c) for c in results.cases],
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def load_phase_a_results(path: Path) -> dict:
    """Load Phase A results JSON."""
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase A: Standards + execution efficacy validation"
    )
    parser.add_argument("--bids-dir", type=Path, required=True, help="BIDS dataset root")
    parser.add_argument("--output-dir", type=Path, required=True, help="Derivatives output root")
    parser.add_argument("--work-dir", type=Path, default=Path("work"), help="Working directory")
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASETS.keys()),
                        help="Dataset identifier")
    parser.add_argument("--results-dir", type=Path, default=Path("validation_results"),
                        help="Where to save Phase A results JSON")
    parser.add_argument("--nprocs", type=int, default=4)
    parser.add_argument("--mem-gb", type=float, default=8.0)
    parser.add_argument("--timeout", type=int, default=7200, help="Per-case timeout (sec)")
    parser.add_argument("--max-cases", type=int, default=None, help="Cap for debugging")

    args = parser.parse_args()

    results = run_phase_a(
        bids_dir=args.bids_dir,
        output_dir=args.output_dir,
        work_dir=args.work_dir,
        dataset_key=args.dataset,
        nprocs=args.nprocs,
        mem_gb=args.mem_gb,
        timeout_sec=args.timeout,
        max_cases=args.max_cases,
    )

    out_path = get_phase_dir(args.results_dir, "A") / f"phase_a_{args.dataset}.json"
    save_phase_a_results(results, out_path)

    print(f"\n{'='*60}")
    print(f"Phase A results — {args.dataset}")
    print(f"{'='*60}")
    print(f"  Total cases:       {results.n_total}")
    print(f"  Success:           {results.n_success}")
    print(f"  Soft failures:     {results.n_soft_fail}")
    print(f"  Hard failures:     {results.n_hard_fail}")
    print(f"  Completion rate:   {results.completion_rate:.1%} "
          f"[{results.completion_ci_lower:.1%}, {results.completion_ci_upper:.1%}]")
    print(f"  BIDS compliance:   {results.compliance_rate:.1%} "
          f"[{results.compliance_ci_lower:.1%}, {results.compliance_ci_upper:.1%}]")
    print(f"  Runtime median:    {results.runtime_median:.0f}s")
    print(f"  Runtime p95:       {results.runtime_p95:.0f}s")
    if results.failure_taxonomy:
        print(f"  Failure taxonomy:  {results.failure_taxonomy}")
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
