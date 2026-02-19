#!/usr/bin/env python
"""Convert ACRIN-DSC-MR-Brain (ACRIN-6677) DICOM data to BIDS layout via dcm2niix.

Source layout (TCIA):
    ACRIN-DSC-MR-Brain/{SubjectID}/{Date-Desc-UID}/{SeriesNum-SeriesDesc-UID}/
        1-01.dcm, 1-02.dcm, ...

BIDS output:
    sub-{NNN}/ses-{MM}/anat/  (T1w, ce-gadolinium_T1w, T2w, FLAIR)
    sub-{NNN}/ses-{MM}/dwi/   (DWI, DTI)
    sub-{NNN}/ses-{MM}/perf/  (DSC, DCE)

Only 3 subjects have ROIs — those are placed in derivatives/ground-truth/.

Usage:
    python scripts/convert_acrin_dsc_to_bids.py \\
        --source "/Volumes/MHFCBCR/imaging_datasets/ACRIN-DSC-MR-Brain" \\
        --output /data/bids/acrin_dsc \\
        [--skip-existing]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Series description → BIDS suffix + datatype
SERIES_RULES: List[Tuple[str, str, str]] = [
    # ROIs (very sparse — only 3 subjects)
    (r"\broi\b|ROICBV|tumor", "__roi__", "roi"),
    # Structural
    (r"T1.*(?:pre|SE)\b(?!.*GAD)(?!.*POST)(?!.*FLAIR)", "anat", "T1w"),
    (r"T1.*(?:post|GAD)|POST.?GAD.*T1|SPGR.*GAD|IRSPGR", "anat", "ce-gadolinium_T1w"),
    (r"FLAIR", "anat", "FLAIR"),
    (r"T2(?!.*STAR)(?!.*GRE)(?!.*FLAIR)", "anat", "T2w"),
    (r"T2.*(?:STAR|GRE)|SUSCEPT", "anat", "T2starw"),
    # Diffusion
    (r"DWI|DTI|DIFFUSION|TENSOR", "dwi", "dwi"),
    (r"ADC|Apparent.Diffusion", "dwi", "adc"),
    (r"Fractional.Anisotropy|FA\b", "dwi", "fa"),
    # Perfusion
    (r"DSC|Perfusion", "perf", "dsc"),
    (r"DCE", "perf", "dce"),
    # MR Spec
    (r"MRS|CSI|SPECTRO", "mrs", "mrs"),
]


def _classify_series(series_dirname: str) -> Tuple[str, str]:
    for pattern, dtype, suffix in SERIES_RULES:
        if re.search(pattern, series_dirname, re.IGNORECASE):
            return dtype, suffix
    return "other", "unknown"


def _run_dcm2niix(dicom_dir: Path, output_dir: Path, fname_pattern: str) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "dcm2niix",
        "-z", "y",
        "-b", "y",
        "-f", fname_pattern,
        "-o", str(output_dir),
        str(dicom_dir),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    except subprocess.CalledProcessError as exc:
        print(f"  WARN: dcm2niix failed on {dicom_dir.name}: {exc.stderr.decode()[:200]}")
        return []
    except subprocess.TimeoutExpired:
        print(f"  WARN: dcm2niix timed out on {dicom_dir.name}")
        return []
    return sorted(output_dir.glob(f"{fname_pattern}*.nii.gz"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ACRIN-DSC-MR-Brain DICOM to BIDS")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-existing", action="store_true", default=False)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    deriv_dir = output / "derivatives" / "ground-truth"

    data_root = None
    for mdir in source.iterdir():
        candidate = mdir / "ACRIN-DSC-MR-Brain"
        if candidate.is_dir():
            data_root = candidate
            break
    if data_root is None:
        sys.exit(f"ERROR: Cannot find ACRIN-DSC-MR-Brain/ under {source}")

    subject_dirs = sorted(d for d in data_root.iterdir() if d.is_dir())
    print(f"Found {len(subject_dirs)} subjects under {data_root.name}")

    rows: List[Dict[str, str]] = []

    for sub_dir in subject_dirs:
        m = re.search(r"(\d{3})$", sub_dir.name)
        if not m:
            print(f"  SKIP: cannot parse {sub_dir.name}")
            continue

        sub_num = m.group(1)
        sub_label = f"ACRINDSC{sub_num}"
        sub_str = f"sub-{sub_label}"

        if args.skip_existing and (output / sub_str).exists():
            print(f"  SKIP (exists): {sub_str}")
            continue

        session_dirs = sorted(d for d in sub_dir.iterdir() if d.is_dir())
        if not session_dirs:
            continue
        print(f"\n{sub_str}: {len(session_dirs)} sessions")

        for ses_idx, ses_dir in enumerate(session_dirs, start=1):
            ses_str = f"ses-{ses_idx:02d}"
            series_dirs = sorted(d for d in ses_dir.iterdir() if d.is_dir())

            for sr_dir in series_dirs:
                dtype, suffix = _classify_series(sr_dir.name)
                if dtype == "other":
                    continue

                if dtype.startswith("__"):
                    dest = deriv_dir / sub_str / ses_str / "anat"
                    fname = f"{sub_str}_{ses_str}_{suffix}"
                else:
                    dest = output / sub_str / ses_str / dtype
                    fname = f"{sub_str}_{ses_str}_{suffix}"

                niftis = _run_dcm2niix(sr_dir, dest, fname)
                if niftis:
                    existing = sorted(dest.glob(f"{sub_str}_{ses_str}_{suffix}*.nii.gz"))
                    if len(existing) > 1:
                        for run_idx, ef in enumerate(existing, start=1):
                            new_name = ef.name.replace(
                                f"_{suffix}",
                                f"_run-{run_idx:02d}_{suffix}",
                            )
                            if ef.name != new_name:
                                ef.rename(ef.parent / new_name)
                                json_old = ef.with_suffix("").with_suffix(".json")
                                if json_old.exists():
                                    json_old.rename(
                                        ef.parent / new_name.replace(".nii.gz", ".json")
                                    )

            rows.append({
                "participant_id": sub_str,
                "session_id": ses_str,
                "source_session": ses_dir.name,
            })

    # BIDS metadata
    desc = {
        "Name": "ACRIN-6677 DSC-MR Perfusion for Recurrent GBM",
        "BIDSVersion": "1.9.0",
        "DatasetType": "raw",
        "GeneratedBy": [{"Name": "convert_acrin_dsc_to_bids.py"}],
        "License": "TCIA Restricted License",
        "Authors": ["ACRIN-6677 Investigators"],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "dataset_description.json").write_text(json.dumps(desc, indent=2) + "\n")

    if deriv_dir.exists():
        dd = {
            "Name": "ACRIN-DSC ground-truth ROIs",
            "BIDSVersion": "1.9.0",
            "DatasetType": "derivative",
        }
        (deriv_dir / "dataset_description.json").write_text(json.dumps(dd, indent=2) + "\n")

    # participants.tsv
    unique = {}
    for r in rows:
        if r["participant_id"] not in unique:
            unique[r["participant_id"]] = r["participant_id"]
    lines = ["participant_id"]
    for pid in unique:
        lines.append(pid)
    (output / "participants.tsv").write_text("\n".join(lines) + "\n")

    print(f"\nDone. Processed {len(rows)} subject-sessions from {len(unique)} subjects.")


if __name__ == "__main__":
    main()
