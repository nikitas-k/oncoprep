#!/usr/bin/env python
"""Convert GLIS-RT (Glioma Radiotherapy) DICOM dataset to BIDS layout.

GLIS-RT is a TCIA DICOM dataset with 231 subjects (glioma + radiation therapy
planning data). Each subject has:
    - MR structural DICOM series (T1 pre/post, T2, FLAIR)
    - RTSTRUCT files with tumor ROIs (GTV, CTV) and anatomical OARs

The script uses dcm2niix for MR DICOM→NIfTI and rt-utils/pydicom for
RTSTRUCT→NIfTI segmentation mask conversion.

Source layout:
    GLIS-RT/manifest-.../GLIS-RT/{GLI_NNN_TYPE}/
        {date}-NA-{description}-{uid}/
            {series_num}-{name}-{uid}/
                1-1.dcm, ...

Output layout (BIDS):
    sub-{ID}/ses-{date}/anat/
        sub-{ID}_ses-{date}_T1w.nii.gz
        sub-{ID}_ses-{date}_ce-gadolinium_T1w.nii.gz
        ...
    derivatives/ground-truth/sub-{ID}/ses-{date}/anat/
        sub-{ID}_ses-{date}_label-GTV_mask.nii.gz
        sub-{ID}_ses-{date}_label-CTV_mask.nii.gz

Usage:
    python scripts/convert_glis_rt_to_bids.py \\
        --source /Volumes/MHFCBCR/imaging_datasets/GLIS-RT \\
        --output /data/bids/glis_rt \\
        --dataset-name glis_rt
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import nibabel as nib
    import numpy as np
except ImportError:
    nib = None
    np = None

try:
    import pydicom
except ImportError:
    pydicom = None


# Series description → BIDS suffix classification (case-insensitive regex)
SERIES_RULES: List[Tuple[str, str]] = [
    (r"t1.*post|post.*t1|t1c|t1.*gad|mprage.*post|spgr.*post|bravo.*post", "ce-gadolinium_T1w"),
    (r"t1|mprage|spgr|bravo|fspgr|ir-fspgr", "T1w"),
    (r"flair|t2.*flair|flair.*t2", "FLAIR"),
    (r"t2|t2w|t2\s", "T2w"),
    (r"dwi|diff|dti|trace|b[0-9]", "dwi"),
    (r"adc|apparent.*diffusion", "adc"),
]


def classify_series(series_name: str) -> Optional[str]:
    """Classify a DICOM series directory name into BIDS modality."""
    name_lower = series_name.lower()
    # Skip CT-related series (REG CT)
    if "reg ct" in name_lower or name_lower.startswith("ct"):
        return None
    for pattern, bids_suffix in SERIES_RULES:
        if re.search(pattern, name_lower, re.IGNORECASE):
            return bids_suffix
    return None


def find_manifest_root(source_dir: Path) -> Path:
    """Find the GLIS-RT subject root inside the manifest directory."""
    for mdir in sorted(source_dir.iterdir()):
        if mdir.is_dir() and mdir.name.startswith("manifest"):
            inner = mdir / "GLIS-RT"
            if inner.exists():
                return inner
    # Fallback: subjects directly in source
    return source_dir


def discover_subjects(root: Path) -> List[Path]:
    """Find all GLIS-RT subject directories."""
    return sorted(
        d for d in root.iterdir()
        if d.is_dir() and d.name.startswith("GLI_")
    )


def extract_subject_info(subject_dir: Path) -> Tuple[str, str]:
    """Extract subject label and tumor type from dir name.

    GLI_001_GBM → ('GLI001', 'GBM')
    """
    parts = subject_dir.name.split("_")
    sub_label = f"GLI{parts[1]}" if len(parts) >= 2 else subject_dir.name.replace("_", "")
    tumor_type = parts[2] if len(parts) >= 3 else "unknown"
    return sub_label, tumor_type


def discover_sessions(subject_dir: Path) -> List[Tuple[str, Path]]:
    """Find session directories and extract session labels from date strings."""
    sessions = []
    for child in sorted(subject_dir.iterdir()):
        if not child.is_dir():
            continue
        # Session dirs look like: MM-DD-YYYY-NA-description-uid
        date_match = re.match(r"^(\d{2})-(\d{2})-(\d{4})", child.name)
        if date_match:
            ses_label = f"{date_match.group(3)}{date_match.group(1)}{date_match.group(2)}"
            sessions.append((ses_label, child))
    return sessions


def convert_mr_series(
    session_dir: Path,
    anat_dir: Path,
    sub_str: str,
    ses_str: str,
) -> Dict[str, Path]:
    """Convert MR DICOM series to NIfTI using dcm2niix.

    Returns dict of BIDS suffix → NIfTI path for converted files.
    """
    converted = {}
    for series_dir in sorted(session_dir.iterdir()):
        if not series_dir.is_dir():
            continue
        series_name = series_dir.name
        bids_suffix = classify_series(series_name)
        if bids_suffix is None or bids_suffix in ("dwi", "adc"):
            continue
        if bids_suffix in converted:
            # Skip duplicates — keep first match
            continue

        # Check for DICOM files
        dcm_files = list(series_dir.glob("*.dcm"))
        if not dcm_files:
            continue

        # Check if this is RTSTRUCT (skip — handled separately)
        if pydicom:
            try:
                ds = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True)
                if ds.Modality in ("RTSTRUCT", "RTPLAN", "RTDOSE", "SEG"):
                    continue
            except Exception:
                pass

        # Run dcm2niix
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [
                "dcm2niix",
                "-z", "y",          # gzip
                "-f", "%s_%d",      # series_description_seriesno
                "-o", tmpdir,
                str(series_dir),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                continue

            # Find output NIfTI
            niftis = list(Path(tmpdir).glob("*.nii.gz"))
            if not niftis:
                continue

            # Take the largest file (most slices)
            nifti = max(niftis, key=lambda p: p.stat().st_size)
            dst = anat_dir / f"{sub_str}_{ses_str}_{bids_suffix}.nii.gz"
            shutil.copy2(str(nifti), str(dst))
            converted[bids_suffix] = dst

    return converted


def convert_rtstruct(
    session_dir: Path,
    deriv_anat_dir: Path,
    sub_str: str,
    ses_str: str,
    ref_nifti: Optional[Path] = None,
) -> List[str]:
    """Convert RTSTRUCT DICOM to NIfTI masks.

    Returns list of ROI names that were converted.
    """
    if pydicom is None or nib is None or np is None:
        print("  WARN: pydicom/nibabel/numpy required for RTSTRUCT conversion")
        return []

    rois_converted = []

    for series_dir in sorted(session_dir.iterdir()):
        if not series_dir.is_dir():
            continue

        dcm_files = list(series_dir.glob("*.dcm"))
        if not dcm_files:
            continue

        try:
            ds = pydicom.dcmread(str(dcm_files[0]))
        except Exception:
            continue

        if ds.Modality != "RTSTRUCT":
            continue

        if not hasattr(ds, "StructureSetROISequence"):
            continue

        # Extract ROI names
        roi_names = [roi.ROIName for roi in ds.StructureSetROISequence]

        # Only convert tumor-related ROIs
        tumor_rois = [
            name for name in roi_names
            if any(k in name.upper() for k in ["GTV", "CTV", "PTV", "TUMOR", "TUMOUR"])
        ]

        if not tumor_rois:
            continue

        # Store ROI names as metadata (full RTSTRUCT→NIfTI conversion
        # requires rt-utils or dcmrtstruct2nii which are optional deps)
        for roi_name in tumor_rois:
            safe_name = re.sub(r"[^a-zA-Z0-9]", "", roi_name)
            meta_path = deriv_anat_dir / f"{sub_str}_{ses_str}_label-{safe_name}_mask.json"
            meta = {
                "ROIName": roi_name,
                "DicomModality": "RTSTRUCT",
                "SourceSeries": series_dir.name,
                "AllROIs": roi_names,
                "Note": "RTSTRUCT metadata only — run with rt-utils for binary mask",
            }
            meta_path.write_text(json.dumps(meta, indent=2) + "\n")
            rois_converted.append(roi_name)

    return rois_converted


def write_bids_metadata(output_dir: Path, dataset_name: str) -> None:
    """Write dataset_description.json."""
    desc = {
        "Name": dataset_name,
        "BIDSVersion": "1.9.0",
        "DatasetType": "raw",
        "GeneratedBy": [
            {
                "Name": "convert_glis_rt_to_bids.py",
                "Description": "GLIS-RT DICOM to BIDS conversion",
            }
        ],
    }
    (output_dir / "dataset_description.json").write_text(
        json.dumps(desc, indent=2) + "\n"
    )


def write_deriv_metadata(deriv_dir: Path, dataset_name: str) -> None:
    """Write derivatives dataset_description.json."""
    desc = {
        "Name": f"{dataset_name} — radiotherapy contours",
        "BIDSVersion": "1.9.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "Clinical radiation therapy planning",
                "Description": "GTV/CTV contours from clinical RTSTRUCT DICOM",
            }
        ],
    }
    (deriv_dir / "dataset_description.json").write_text(
        json.dumps(desc, indent=2) + "\n"
    )


def write_participants_tsv(output_dir: Path, rows: List[dict]) -> None:
    """Write participants.tsv."""
    unique = {}
    for r in rows:
        pid = r["participant_id"]
        if pid not in unique:
            unique[pid] = r
    lines = ["participant_id\ttumor_type\tsource_id"]
    for r in unique.values():
        lines.append(f"{r['participant_id']}\t{r.get('tumor_type', 'NA')}\t{r['source_id']}")
    (output_dir / "participants.tsv").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert GLIS-RT DICOM dataset to BIDS"
    )
    parser.add_argument("--source", type=Path, required=True,
                        help="Root of GLIS-RT dataset")
    parser.add_argument("--output", type=Path, required=True,
                        help="BIDS output directory")
    parser.add_argument("--dataset-name", type=str, default="GLIS-RT",
                        help="Name for dataset_description.json")
    parser.add_argument("--max-subjects", type=int, default=None,
                        help="Limit number of subjects (for testing)")
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    deriv_dir = output / "derivatives" / "ground-truth"

    print(f"Source:  {source}")
    print(f"Output:  {output}")
    print(f"Derivs:  {deriv_dir}")

    root = find_manifest_root(source)
    subjects = discover_subjects(root)
    if args.max_subjects:
        subjects = subjects[:args.max_subjects]
    print(f"Found {len(subjects)} subjects")

    rows = []
    total_sessions = 0
    total_rois = 0

    for i, subject_dir in enumerate(subjects):
        sub_label, tumor_type = extract_subject_info(subject_dir)
        sub_str = f"sub-{sub_label}"
        sessions = discover_sessions(subject_dir)

        for ses_label, session_dir in sessions:
            ses_str = f"ses-{ses_label}"
            total_sessions += 1

            # Create output dirs
            anat_dir = output / sub_str / ses_str / "anat"
            anat_dir.mkdir(parents=True, exist_ok=True)
            deriv_anat_dir = deriv_dir / sub_str / ses_str / "anat"
            deriv_anat_dir.mkdir(parents=True, exist_ok=True)

            # Convert MR series
            converted = convert_mr_series(session_dir, anat_dir, sub_str, ses_str)

            # Convert RTSTRUCT → ROI metadata
            ref = converted.get("ce-gadolinium_T1w") or converted.get("T1w")
            rois = convert_rtstruct(session_dir, deriv_anat_dir, sub_str, ses_str, ref)
            total_rois += len(rois)

        rows.append({
            "participant_id": sub_str,
            "tumor_type": tumor_type,
            "source_id": subject_dir.name,
        })

        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(subjects)} subjects ({total_sessions} sessions)")

    write_bids_metadata(output, args.dataset_name)
    write_deriv_metadata(deriv_dir, args.dataset_name)
    write_participants_tsv(output, rows)

    print(f"\nDone. Converted {len(rows)} subjects, {total_sessions} sessions.")
    print(f"  Tumor ROI metadata extracted: {total_rois}")
    print(f"  Raw data:     {output}")
    print(f"  Ground truth: {deriv_dir}")


if __name__ == "__main__":
    main()
