#!/usr/bin/env python
"""Convert UCSD-PTGBM-v1 (post-treatment GBM) to BIDS layout.

Source layout:
    UCSD-PTGBM/UCSD-PTGBM-{NNNN}_{SS}/{ID}_{suffix}.nii.gz
    where SS = session code (01, 02, …)

BIDS mapping:
    T1pre   → sub-{ID}/ses-{SS}/anat/sub-{ID}_ses-{SS}_T1w.nii.gz
    T1post  → sub-{ID}/ses-{SS}/anat/sub-{ID}_ses-{SS}_ce-T1w.nii.gz
    T2      → sub-{ID}/ses-{SS}/anat/sub-{ID}_ses-{SS}_T2w.nii.gz
    FLAIR   → sub-{ID}/ses-{SS}/anat/sub-{ID}_ses-{SS}_FLAIR.nii.gz
    BraTS_tumor_seg → derivatives/ground-truth/.../sub-{ID}_ses-{SS}_dseg.nii.gz

Usage:
    python scripts/convert_ucsd_ptgbm_to_bids.py \\
        --source "/Volumes/MHFCBCR/imaging_datasets/PKG - UCSD-PTGBM-v1" \\
        --output /data/bids/ucsd_ptgbm
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Dict, List


MODALITY_MAP: Dict[str, str] = {
    "T1pre":  "T1w",
    "T1post": "ce-T1w",
    "T2":     "T2w",
    "FLAIR":  "FLAIR",
    "SWAN":   "T2starw",
}

SEGMENTATION_MAP: Dict[str, str] = {
    "BraTS_tumor_seg":                  "dseg",
    "enhancing_cellular_tumor_seg":     "label-enhancingcellular_mask",
    "non_enhancing_cellular_tumor_seg": "label-nonenhancingcellular_mask",
    "total_cellular_tumor_seg":         "label-totalcellular_mask",
}

# UCSD-PTGBM-0002_01
UCSD_ID_RE = re.compile(r"^UCSD-PTGBM-(\d{4})_(\d{2})$")


def _symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert UCSD-PTGBM-v1 to BIDS"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--copy", action="store_true", default=False)
    args = parser.parse_args()

    source = args.source.resolve()
    data_dir = source / "UCSD-PTGBM"
    output = args.output.resolve()
    deriv_dir = output / "derivatives" / "ground-truth"

    if not data_dir.exists():
        print(f"ERROR: {data_dir} not found")
        return

    copy_fn = _symlink if not args.copy else shutil.copy2

    subject_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    print(f"Found {len(subject_dirs)} subject-session dirs")

    rows: List[dict] = []
    for sub_dir in subject_dirs:
        m = UCSD_ID_RE.match(sub_dir.name)
        if not m:
            print(f"  SKIP: cannot parse {sub_dir.name}")
            continue

        num_id = m.group(1)   # 0002
        ses_id = m.group(2)   # 01
        raw_id = sub_dir.name  # UCSD-PTGBM-0002_01

        sub_label = f"UCSDPTGBM{num_id}"
        sub_str = f"sub-{sub_label}"
        ses_str = f"ses-{ses_id}"

        anat_dir = output / sub_str / ses_str / "anat"
        anat_dir.mkdir(parents=True, exist_ok=True)

        deriv_anat = deriv_dir / sub_str / ses_str / "anat"
        deriv_anat.mkdir(parents=True, exist_ok=True)

        # Structural modalities
        for src_suffix, bids_suffix in MODALITY_MAP.items():
            src = sub_dir / f"{raw_id}_{src_suffix}.nii.gz"
            if src.exists():
                dst = anat_dir / f"{sub_str}_{ses_str}_{bids_suffix}.nii.gz"
                copy_fn(src, dst)

        # Segmentations → derivatives
        for src_suffix, bids_suffix in SEGMENTATION_MAP.items():
            src = sub_dir / f"{raw_id}_{src_suffix}.nii.gz"
            if src.exists():
                dst = deriv_anat / f"{sub_str}_{ses_str}_{bids_suffix}.nii.gz"
                copy_fn(src, dst)

        rows.append({
            "participant_id": sub_str,
            "session_id": ses_str,
            "source_id": raw_id,
        })

    # BIDS metadata
    desc = {
        "Name": "UCSD Post-Treatment GBM (UCSD-PTGBM-v1)",
        "BIDSVersion": "1.9.0",
        "DatasetType": "raw",
        "GeneratedBy": [{"Name": "convert_ucsd_ptgbm_to_bids.py"}],
    }
    (output / "dataset_description.json").write_text(json.dumps(desc, indent=2) + "\n")

    dd = {
        "Name": "UCSD-PTGBM — ground truth segmentations",
        "BIDSVersion": "1.9.0",
        "DatasetType": "derivative",
    }
    deriv_dir.mkdir(parents=True, exist_ok=True)
    (deriv_dir / "dataset_description.json").write_text(json.dumps(dd, indent=2) + "\n")

    # participants.tsv
    unique = {}
    for r in rows:
        if r["participant_id"] not in unique:
            unique[r["participant_id"]] = r["source_id"].rsplit("_", 1)[0]
    lines = ["participant_id\tsource_id"]
    for pid, sid in unique.items():
        lines.append(f"{pid}\t{sid}")
    (output / "participants.tsv").write_text("\n".join(lines) + "\n")

    print(f"\nDone. {len(rows)} subject-sessions from {len(unique)} subjects.")


if __name__ == "__main__":
    main()
