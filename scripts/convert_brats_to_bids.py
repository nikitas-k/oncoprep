#!/usr/bin/env python
"""Convert BraTS-style challenge datasets to BIDS layout.

Supports all BraTS challenge datasets that share the naming convention:
    {SubjectID}-t1c.nii.gz  →  sub-{ID}/anat/sub-{ID}_ce-T1w.nii.gz
    {SubjectID}-t1n.nii.gz  →  sub-{ID}/anat/sub-{ID}_T1w.nii.gz
    {SubjectID}-t2f.nii.gz  →  sub-{ID}/anat/sub-{ID}_FLAIR.nii.gz
    {SubjectID}-t2w.nii.gz  →  sub-{ID}/anat/sub-{ID}_T2w.nii.gz
    {SubjectID}-seg.nii.gz  →  derivatives/ground-truth/sub-{ID}/anat/sub-{ID}_dseg.nii.gz

Supported datasets:
    - BraTS-GLI-2024 (adult glioma)
    - BraTS-MEN-2024 (meningioma)
    - BraTS-MET-2024 (brain metastases)
    - BraTS-SSA-2023 (sub-Saharan Africa glioma)
    - BraTS2025-GLI-PRE (pre-treatment glioma 2025)
    - BraTS2025-MET (metastases 2025, incl. UCSD split)
    - BraTS-GoAT-2024 (generalizable across tumours)

Subject IDs follow either:
    BraTS-{TYPE}-{5digit}-{3digit}  (GLI, MEN, MET, SSA — 3-digit session suffix)
    BraTS-{TYPE}-{5digit}           (GoAT — no session suffix)

Usage:
    python scripts/convert_brats_to_bids.py \\
        --source /Volumes/MHFCBCR/imaging_datasets/BraTS-GLI-2024 \\
        --output /data/bids/brats_gli \\
        --dataset-name brats_gli

    python scripts/convert_brats_to_bids.py \\
        --source /Volumes/MHFCBCR/imaging_datasets/ASNR-MICCAI-BraTS2023-SSA-Challenge-TrainingData_V2 \\
        --output /data/bids/brats_ssa \\
        --dataset-name brats_ssa

    python scripts/convert_brats_to_bids.py \\
        --source /Volumes/MHFCBCR/imaging_datasets/MICCAI2024-BraTS-GoAT-TrainingData-With-GroundTruth \\
        --output /data/bids/brats_goat \\
        --dataset-name brats_goat

    python scripts/convert_brats_to_bids.py \\
        --source /Volumes/MHFCBCR/imaging_datasets/BraTS2025-GLI-PRE-Challenge-TrainingData \\
        --output /data/bids/brats_gli_pre \\
        --dataset-name brats_gli_pre

    python scripts/convert_brats_to_bids.py \\
        --source /Volumes/MHFCBCR/imaging_datasets/MICCAI-LH-BraTS2025-MET-Challenge-Training \\
        --output /data/bids/brats_met_2025 \\
        --dataset-name brats_met_2025

    # Optional: apply corrected labels to BraTS2025-MET
    python scripts/convert_brats_to_bids.py \\
        --source /Volumes/MHFCBCR/imaging_datasets/MICCAI-LH-BraTS2025-MET-Challenge-Training \\
        --output /data/bids/brats_met_2025 \\
        --dataset-name brats_met_2025 \\
        --corrected-labels /Volumes/MHFCBCR/imaging_datasets/MICCAI-LH-BraTS2025-MET-Challenge-corrected-labels
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import List, Optional, Tuple


# BraTS modality → BIDS suffix mapping
MODALITY_MAP = {
    "t1c": "ce-T1w",      # contrast-enhanced T1
    "t1n": "T1w",          # native T1
    "t2f": "FLAIR",        # T2 FLAIR
    "t2w": "T2w",          # T2 weighted
}

# BraTS subject ID regex — handles two patterns:
#   BraTS-{TYPE}-{5digit}-{3digit}  (GLI, MEN, MET, SSA)
#   BraTS-{TYPE}-{5digit}           (GoAT)
BRATS_ID_RE = re.compile(
    r"^(BraTS-(?:GLI|MEN|MET|SSA|GoAT)-\d{5})(?:-(\d{3}))?$"
)


def parse_brats_id(folder_name: str) -> Optional[Tuple[str, str, str]]:
    """Parse a BraTS folder name into (sub_label, ses_label, raw_id).

    Returns None if the folder name doesn't match the expected pattern.
    """
    m = BRATS_ID_RE.match(folder_name)
    if not m:
        return None
    base_id = m.group(1)   # e.g. BraTS-GLI-00005 or BraTS-GoAT-00003
    timepoint = m.group(2) or "000"  # Default to 000 for GoAT (no session)

    # Sanitise for BIDS (no hyphens in labels)
    sub_label = base_id.replace("-", "")        # BraTSGLI00005
    ses_label = timepoint                        # 100

    return sub_label, ses_label, folder_name


def discover_subject_folders(source_dir: Path) -> List[Path]:
    """Find all subject folders across training splits (skip validation)."""
    folders = []
    for subdir in sorted(source_dir.iterdir()):
        if not subdir.is_dir():
            continue
        # Skip validation data (no segmentations)
        if "validation" in subdir.name.lower():
            continue
        # Check if it's a split directory (training_data1, UCSD - Training, etc.)
        # or directly a subject folder
        children = list(subdir.iterdir())
        if any(c.name.endswith("-seg.nii.gz") for c in children if c.is_file()):
            # This is itself a subject folder
            folders.append(subdir)
        elif subdir.name.startswith("BraTS-"):
            # Direct subject folder without segmentation (unusual, but include)
            folders.append(subdir)
        else:
            # It's a split directory containing subject folders
            for child in sorted(subdir.iterdir()):
                if child.is_dir():
                    folders.append(child)
    return folders


def convert_subject(
    subject_dir: Path,
    output_dir: Path,
    deriv_dir: Path,
    use_symlinks: bool = True,
) -> Optional[dict]:
    """Convert a single BraTS subject folder to BIDS layout.

    Returns a participants.tsv row dict, or None on failure.
    """
    parsed = parse_brats_id(subject_dir.name)
    if parsed is None:
        print(f"  SKIP: cannot parse {subject_dir.name}")
        return None

    sub_label, ses_label, raw_id = parsed
    sub_str = f"sub-{sub_label}"
    ses_str = f"ses-{ses_label}"

    # Create output directories
    anat_dir = output_dir / sub_str / ses_str / "anat"
    anat_dir.mkdir(parents=True, exist_ok=True)

    deriv_anat_dir = deriv_dir / sub_str / ses_str / "anat"
    deriv_anat_dir.mkdir(parents=True, exist_ok=True)

    copy_fn = _symlink if use_symlinks else shutil.copy2

    # Process modalities
    for brats_suffix, bids_suffix in MODALITY_MAP.items():
        src = subject_dir / f"{raw_id}-{brats_suffix}.nii.gz"
        if src.exists():
            dst = anat_dir / f"{sub_str}_{ses_str}_{bids_suffix}.nii.gz"
            copy_fn(src, dst)
        else:
            print(f"  WARN: missing {src.name}")

    # Process segmentation → derivatives
    seg_src = subject_dir / f"{raw_id}-seg.nii.gz"
    if seg_src.exists():
        seg_dst = deriv_anat_dir / f"{sub_str}_{ses_str}_dseg.nii.gz"
        copy_fn(seg_src, seg_dst)
    else:
        print(f"  WARN: missing segmentation for {raw_id}")

    return {
        "participant_id": sub_str,
        "session_id": ses_str,
        "source_id": raw_id,
    }


def _symlink(src: Path, dst: Path) -> None:
    """Create a symlink, removing existing."""
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def write_bids_metadata(output_dir: Path, dataset_name: str) -> None:
    """Write dataset_description.json and other BIDS boilerplate."""
    desc = {
        "Name": dataset_name,
        "BIDSVersion": "1.9.0",
        "DatasetType": "raw",
        "GeneratedBy": [
            {
                "Name": "convert_brats_to_bids.py",
                "Description": "BraTS to BIDS conversion script",
            }
        ],
    }
    (output_dir / "dataset_description.json").write_text(
        json.dumps(desc, indent=2) + "\n"
    )


def write_deriv_metadata(deriv_dir: Path, dataset_name: str) -> None:
    """Write derivatives dataset_description.json."""
    desc = {
        "Name": f"{dataset_name} — ground truth segmentations",
        "BIDSVersion": "1.9.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "BraTS challenge organisers",
                "Description": "Expert-refined segmentation labels",
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
    lines = ["participant_id\tsource_id"]
    for r in unique.values():
        lines.append(f"{r['participant_id']}\t{r['source_id']}")
    (output_dir / "participants.tsv").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert BraTS-style challenge datasets to BIDS"
    )
    parser.add_argument("--source", type=Path, required=True,
                        help="Root of the BraTS dataset (e.g. BraTS-GLI-2024/)")
    parser.add_argument("--output", type=Path, required=True,
                        help="BIDS output directory")
    parser.add_argument("--dataset-name", type=str, default="BraTS",
                        help="Name for dataset_description.json")
    parser.add_argument("--copy", action="store_true", default=False,
                        help="Copy files instead of symlinking")
    parser.add_argument("--corrected-labels", type=Path, default=None,
                        help="Directory with corrected seg files to overlay "
                             "(e.g. BraTS2025-MET corrected-labels)")
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    deriv_dir = output / "derivatives" / "ground-truth"

    print(f"Source:  {source}")
    print(f"Output:  {output}")
    print(f"Derivs:  {deriv_dir}")

    folders = discover_subject_folders(source)
    print(f"Found {len(folders)} subject folders (with segmentation)")

    rows = []
    for i, folder in enumerate(folders):
        row = convert_subject(
            folder, output, deriv_dir,
            use_symlinks=not args.copy,
        )
        if row:
            rows.append(row)
        if (i + 1) % 200 == 0:
            print(f"  Processed {i + 1}/{len(folders)}")

    write_bids_metadata(output, args.dataset_name)
    write_deriv_metadata(deriv_dir, args.dataset_name)
    write_participants_tsv(output, rows)

    # Apply corrected labels if provided
    n_corrected = 0
    if args.corrected_labels and args.corrected_labels.exists():
        print(f"\nApplying corrected labels from {args.corrected_labels}")
        copy_fn = _symlink if not args.copy else shutil.copy2
        for seg_file in sorted(args.corrected_labels.glob("*-seg.nii.gz")):
            # Parse subject ID from filename: BraTS-MET-01094-003-seg.nii.gz
            stem = seg_file.name.replace("-seg.nii.gz", "")
            parsed = parse_brats_id(stem)
            if parsed is None:
                print(f"  SKIP corrected label: {seg_file.name}")
                continue
            sub_label, ses_label, _ = parsed
            sub_str = f"sub-{sub_label}"
            ses_str = f"ses-{ses_label}"
            dst = deriv_dir / sub_str / ses_str / "anat" / f"{sub_str}_{ses_str}_dseg.nii.gz"
            dst.parent.mkdir(parents=True, exist_ok=True)
            copy_fn(seg_file, dst)
            n_corrected += 1
            print(f"  Replaced: {seg_file.name} → {dst.relative_to(deriv_dir)}")

    print(f"\nDone. Converted {len(rows)}/{len(folders)} subjects to BIDS.")
    if n_corrected:
        print(f"  Corrected labels applied: {n_corrected}")
    print(f"  Raw data:     {output}")
    print(f"  Ground truth: {deriv_dir}")


if __name__ == "__main__":
    main()
