#!/usr/bin/env python
"""Convert ACRIN-FMISO (ACRIN-6684) DICOM data to BIDS layout using dcm2niix.

Source layout (TCIA):
    ACRIN-FMISO-Brain/{SubjectID}/{Date-Desc-UID}/{SeriesNum-SeriesDesc-UID}/
        1-01.dcm, 1-02.dcm, ...

BIDS output:
    sub-{NNN}/ses-{MM}/anat/  (T1w, ce-gadolinium_T1w, T2w, FLAIR)
    sub-{NNN}/ses-{MM}/dwi/   (DWI)
    sub-{NNN}/ses-{MM}/perf/  (DSC, DCE)
    sub-{NNN}/ses-{MM}/pet/   (FMISO PET)
    derivatives/ground-truth/sub-{NNN}/ses-{MM}/anat/ (HVC masks, DeltaT1ROI)

Sessions are numbered chronologically per subject (ses-01, ses-02, …).

Usage:
    python scripts/convert_acrin_fmiso_to_bids.py \\
        --source "/Volumes/MHFCBCR/imaging_datasets/ACRIN-FMISO" \\
        --output /data/bids/acrin_fmiso \\
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
# Patterns are matched case-insensitively against the series dir name
SERIES_RULES: List[Tuple[str, str, str]] = [
    # (regex pattern on series dirname, BIDS datatype, BIDS suffix)
    # ROIs / masks → derivatives (handled separately)
    (r"HVC-Mask", "__mask__", "mask"),
    (r"DeltaT1ROI", "__roi__", "DeltaT1ROI"),
    (r"TUMOR.?PERFUSION", "__roi__", "TumorPerfusion"),
    # Structural
    (r"T1.*(PRE|SE)\b(?!.*GAD)(?!.*POST)", "anat", "T1w"),
    (r"T1.*(POST|GAD)", "anat", "ce-gadolinium_T1w"),
    (r"MPRAGE.*POST", "anat", "ce-gadolinium_T1w"),
    (r"FLAIR", "anat", "FLAIR"),
    (r"T2(?!.*STAR)", "anat", "T2w"),
    # Diffusion
    (r"DWI|DIFFUSION|DTI", "dwi", "dwi"),
    (r"ADC", "dwi", "adc"),
    # Perfusion
    (r"DSC|DUAL.*DSC", "perf", "dsc"),
    (r"DCE|T1.*MAP|pCASL", "perf", "dce"),
    # PET
    (r"FMISO|F-MISO|SUV", "pet", "pet"),
    # BOLD
    (r"BOLD", "func", "bold"),
]


def _classify_series(series_dirname: str) -> Tuple[str, str]:
    """Return (datatype, suffix) for a DICOM series directory name."""
    for pattern, dtype, suffix in SERIES_RULES:
        if re.search(pattern, series_dirname, re.IGNORECASE):
            return dtype, suffix
    return "other", "unknown"


def _run_dcm2niix(dicom_dir: Path, output_dir: Path, fname_pattern: str) -> List[Path]:
    """Run dcm2niix on a DICOM directory, return list of created NIfTI files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "dcm2niix",
        "-z", "y",           # gzip compress
        "-b", "y",           # generate BIDS sidecar
        "-f", fname_pattern,  # filename pattern
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
    parser = argparse.ArgumentParser(description="Convert ACRIN-FMISO DICOM to BIDS")
    parser.add_argument("--source", type=Path, required=True,
                        help="Root of ACRIN-FMISO download")
    parser.add_argument("--output", type=Path, required=True,
                        help="BIDS output directory")
    parser.add_argument("--skip-existing", action="store_true", default=False,
                        help="Skip subjects that already have output")
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    deriv_dir = output / "derivatives" / "ground-truth"

    # Find the data directory (inside manifest-*/ )
    data_root = None
    for mdir in source.iterdir():
        candidate = mdir / "ACRIN-FMISO-Brain"
        if candidate.is_dir():
            data_root = candidate
            break
    if data_root is None:
        sys.exit(f"ERROR: Cannot find ACRIN-FMISO-Brain/ under {source}")

    subject_dirs = sorted(d for d in data_root.iterdir() if d.is_dir())
    print(f"Found {len(subject_dirs)} subjects under {data_root.name}")

    rows: List[Dict[str, str]] = []

    for sub_dir in subject_dirs:
        # Extract subject number: ACRIN-FMISO-Brain-001 → 001
        m = re.search(r"(\d{3})$", sub_dir.name)
        if not m:
            print(f"  SKIP: cannot parse {sub_dir.name}")
            continue

        sub_num = m.group(1)
        sub_label = f"ACRINFMISO{sub_num}"
        sub_str = f"sub-{sub_label}"

        if args.skip_existing and (output / sub_str).exists():
            print(f"  SKIP (exists): {sub_str}")
            continue

        # Sessions sorted by directory name (date-based → chronological)
        session_dirs = sorted(d for d in sub_dir.iterdir() if d.is_dir())
        print(f"\n{sub_str}: {len(session_dirs)} sessions")

        for ses_idx, ses_dir in enumerate(session_dirs, start=1):
            ses_str = f"ses-{ses_idx:02d}"

            # Each session has series subdirectories
            series_dirs = sorted(d for d in ses_dir.iterdir() if d.is_dir())

            for sr_dir in series_dirs:
                dtype, suffix = _classify_series(sr_dir.name)

                if dtype == "other":
                    continue  # skip unrecognised series

                # Handle ROI/mask series → derivatives
                if dtype.startswith("__"):
                    dest = deriv_dir / sub_str / ses_str / "anat"
                    fname = f"{sub_str}_{ses_str}_{suffix}"
                else:
                    dest = output / sub_str / ses_str / dtype
                    fname = f"{sub_str}_{ses_str}_{suffix}"

                niftis = _run_dcm2niix(sr_dir, dest, fname)
                if niftis:
                    # If multiple runs of same suffix, rename with run entity
                    existing = sorted(dest.glob(f"{sub_str}_{ses_str}_{suffix}*.nii.gz"))
                    if len(existing) > 1:
                        for run_idx, ef in enumerate(existing, start=1):
                            new_name = ef.name.replace(
                                f"_{suffix}",
                                f"_run-{run_idx:02d}_{suffix}",
                            )
                            if ef.name != new_name:
                                ef.rename(ef.parent / new_name)
                                # Rename matching JSON sidecar too
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
        "Name": "ACRIN-6684 FMISO-PET/MR for Newly Diagnosed GBM",
        "BIDSVersion": "1.9.0",
        "DatasetType": "raw",
        "GeneratedBy": [{"Name": "convert_acrin_fmiso_to_bids.py"}],
        "License": "TCIA Restricted License",
        "Authors": ["ACRIN-6684 Investigators"],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "dataset_description.json").write_text(json.dumps(desc, indent=2) + "\n")

    if deriv_dir.exists():
        dd = {
            "Name": "ACRIN-FMISO ground-truth ROIs (HVC-Mask, DeltaT1ROI)",
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
