#!/usr/bin/env python
"""Convert UCSF-PDGM-v3 dataset to BIDS layout.

Source layout:
    UCSF-PDGM-v3/UCSF-PDGM-{NNNN}/UCSF-PDGM-{NNNN}_{suffix}.nii.gz

BIDS mapping (structural → rawdata, segs → derivatives/ground-truth):
    T1.nii.gz           → sub-{ID}/anat/sub-{ID}_T1w.nii.gz
    T1c.nii.gz          → sub-{ID}/anat/sub-{ID}_ce-T1w.nii.gz
    T2.nii.gz           → sub-{ID}/anat/sub-{ID}_T2w.nii.gz
    FLAIR.nii.gz        → sub-{ID}/anat/sub-{ID}_FLAIR.nii.gz
    SWI.nii.gz          → sub-{ID}/anat/sub-{ID}_T2starw.nii.gz
    tumor_segmentation  → derivatives/.../sub-{ID}/anat/sub-{ID}_desc-tumorOld_dseg.nii.gz
    brain_segmentation  → derivatives/.../sub-{ID}/anat/sub-{ID}_desc-brain_mask.nii.gz

Usage:
    python scripts/convert_ucsf_pdgm_to_bids.py \\
        --source /Volumes/MHFCBCR/imaging_datasets/UCSF-PDGM-v3-20230111 \\
        --output /data/bids/ucsf_pdgm
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional


# Source suffix → (BIDS directory, BIDS filename suffix)
# Only map the modalities OncoPrep can use + segmentations
MODALITY_MAP: Dict[str, str] = {
    "T1":    "T1w",
    "T1c":   "ce-T1w",
    "T2":    "T2w",
    "FLAIR": "FLAIR",
    "SWI":   "T2starw",
}

# Files that go into derivatives/ground-truth
SEGMENTATION_MAP: Dict[str, str] = {
    "tumor_segmentation":              "desc-tumorOld_dseg",
    "tumor_segmentation_et":           "label-ET_mask",
    "tumor_core":                      "label-TC_mask",
    "whole_tumor":                     "label-WT_mask",
    "whole_tumor_and_edema":           "label-WTedema_mask",
    "non_tumor":                       "label-nontumor_mask",
    "brain_segmentation":              "desc-brain_mask",
    "brain_parenchyma_segmentation":   "desc-brainparenchyma_mask",
}


def _symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def convert_subject(
    subject_dir: Path,
    output_dir: Path,
    deriv_dir: Path,
    use_symlinks: bool = True,
) -> Optional[dict]:
    """Convert one UCSF-PDGM subject to BIDS."""
    raw_id = subject_dir.name  # e.g. UCSF-PDGM-0004
    # Extract numeric ID for BIDS label
    parts = raw_id.split("-")
    if len(parts) != 3:
        print(f"  SKIP: unexpected name {raw_id}")
        return None
    sub_label = f"UCSFPDGM{parts[2]}"
    sub_str = f"sub-{sub_label}"

    anat_dir = output_dir / sub_str / "anat"
    anat_dir.mkdir(parents=True, exist_ok=True)

    deriv_anat = deriv_dir / sub_str / "anat"
    deriv_anat.mkdir(parents=True, exist_ok=True)

    copy_fn = _symlink if use_symlinks else shutil.copy2

    # Map structural modalities
    for src_suffix, bids_suffix in MODALITY_MAP.items():
        src = subject_dir / f"{raw_id}_{src_suffix}.nii.gz"
        if src.exists():
            dst = anat_dir / f"{sub_str}_{bids_suffix}.nii.gz"
            copy_fn(src, dst)

    # Map segmentations to derivatives
    for src_suffix, bids_suffix in SEGMENTATION_MAP.items():
        src = subject_dir / f"{raw_id}_{src_suffix}.nii.gz"
        if src.exists():
            dst = deriv_anat / f"{sub_str}_{bids_suffix}.nii.gz"
            copy_fn(src, dst)

    return {"participant_id": sub_str, "source_id": raw_id}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert UCSF-PDGM-v3 to BIDS"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--copy", action="store_true", default=False)
    args = parser.parse_args()

    source = args.source.resolve()
    data_dir = source / "UCSF-PDGM-v3"
    output = args.output.resolve()
    deriv_dir = output / "derivatives" / "ground-truth"

    if not data_dir.exists():
        print(f"ERROR: {data_dir} not found")
        return

    subjects = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    print(f"Found {len(subjects)} subjects")

    rows: List[dict] = []
    for i, sub_dir in enumerate(subjects):
        row = convert_subject(sub_dir, output, deriv_dir, not args.copy)
        if row:
            rows.append(row)
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(subjects)}")

    # Write BIDS metadata
    desc = {
        "Name": "UCSF Preoperative Diffuse Glioma MRI (UCSF-PDGM-v3)",
        "BIDSVersion": "1.9.0",
        "DatasetType": "raw",
        "GeneratedBy": [{"Name": "convert_ucsf_pdgm_to_bids.py"}],
    }
    (output / "dataset_description.json").write_text(json.dumps(desc, indent=2) + "\n")

    deriv_desc = {
        "Name": "UCSF-PDGM-v3 — ground truth segmentations",
        "BIDSVersion": "1.9.0",
        "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "UCSF-PDGM dataset authors"}],
    }
    (deriv_dir / "dataset_description.json").write_text(json.dumps(deriv_desc, indent=2) + "\n")

    # participants.tsv
    lines = ["participant_id\tsource_id"]
    for r in rows:
        lines.append(f"{r['participant_id']}\t{r['source_id']}")
    (output / "participants.tsv").write_text("\n".join(lines) + "\n")

    print(f"\nDone. Converted {len(rows)} subjects.")


if __name__ == "__main__":
    main()
