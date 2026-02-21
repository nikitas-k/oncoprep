#!/usr/bin/env python
"""Convert UPENN-GBM dataset to BIDS layout.

Source layout:
    NIfTI-files/images_structural/{ID}_{11}/  → T1, T1GD, T2, FLAIR
    NIfTI-files/automated_segm/{ID}_automated_approx_segm.nii.gz
    NIfTI-files/images_segm/{ID}_segm.nii.gz  (expert, 147 subjects only)

BIDS mapping:
    T1.nii.gz    → sub-{ID}/anat/sub-{ID}_T1w.nii.gz
    T1GD.nii.gz  → sub-{ID}/anat/sub-{ID}_ce-gadolinium_T1w.nii.gz
    T2.nii.gz    → sub-{ID}/anat/sub-{ID}_T2w.nii.gz
    FLAIR.nii.gz → sub-{ID}/anat/sub-{ID}_FLAIR.nii.gz
    *_segm       → derivatives/ground-truth/.../sub-{ID}_desc-tumorOld_dseg.nii.gz
    *_auto_segm  → derivatives/automated-segm/.../sub-{ID}_desc-tumorOld_dseg.nii.gz

Usage:
    python scripts/convert_upenn_gbm_to_bids.py \\
        --source /Volumes/MHFCBCR/imaging_datasets/UPENN-GBM \\
        --output /data/bids/upenn_gbm
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Dict, List


MODALITY_MAP: Dict[str, str] = {
    "T1":    "T1w",
    "T1GD":  "ce-gadolinium_T1w",
    "T2":    "T2w",
    "FLAIR": "FLAIR",
}

# Regex for subject IDs: UPENN-GBM-00001_11
UPENN_ID_RE = re.compile(r"^(UPENN-GBM-(\d{5}))_(\d{2})$")


def _symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert UPENN-GBM to BIDS"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--copy", action="store_true", default=False)
    parser.add_argument("--prefer-manual", action="store_true", default=True,
                        help="Prefer expert segm over automated when both exist")
    args = parser.parse_args()

    source = args.source.resolve()
    struct_dir = source / "NIfTI-files" / "images_structural"
    auto_seg_dir = source / "NIfTI-files" / "automated_segm"
    manual_seg_dir = source / "NIfTI-files" / "images_segm"
    output = args.output.resolve()
    gt_deriv = output / "derivatives" / "ground-truth"
    auto_deriv = output / "derivatives" / "automated-segm"

    if not struct_dir.exists():
        print(f"ERROR: {struct_dir} not found")
        return

    copy_fn = _symlink if not args.copy else shutil.copy2

    # Discover subjects from structural directories
    subject_dirs = sorted([d for d in struct_dir.iterdir() if d.is_dir()])
    print(f"Found {len(subject_dirs)} structural subject dirs")

    # Build manual seg lookup
    manual_segs = {}
    if manual_seg_dir.exists():
        for f in manual_seg_dir.glob("*_segm.nii.gz"):
            # e.g. UPENN-GBM-00002_11_segm.nii.gz
            key = f.name.replace("_segm.nii.gz", "")
            manual_segs[key] = f
    print(f"  Manual segmentations: {len(manual_segs)}")

    # Build automated seg lookup
    auto_segs = {}
    if auto_seg_dir.exists():
        for f in auto_seg_dir.glob("*_automated_approx_segm.nii.gz"):
            key = f.name.replace("_automated_approx_segm.nii.gz", "")
            auto_segs[key] = f
    print(f"  Automated segmentations: {len(auto_segs)}")

    rows: List[dict] = []
    n_with_seg = 0

    for sub_dir in subject_dirs:
        m = UPENN_ID_RE.match(sub_dir.name)
        if not m:
            print(f"  SKIP: cannot parse {sub_dir.name}")
            continue

        num_id = m.group(2)       # 00001
        raw_id = sub_dir.name     # UPENN-GBM-00001_11

        sub_label = f"UPENNGBM{num_id}"
        sub_str = f"sub-{sub_label}"

        anat_dir = output / sub_str / "anat"
        anat_dir.mkdir(parents=True, exist_ok=True)

        # Copy structural modalities
        for src_suffix, bids_suffix in MODALITY_MAP.items():
            src = sub_dir / f"{raw_id}_{src_suffix}.nii.gz"
            if src.exists():
                dst = anat_dir / f"{sub_str}_{bids_suffix}.nii.gz"
                copy_fn(src, dst)

        # Segmentation: prefer manual, fall back to automated
        has_seg = False
        if raw_id in manual_segs:
            seg_out = gt_deriv / sub_str / "anat"
            seg_out.mkdir(parents=True, exist_ok=True)
            dst = seg_out / f"{sub_str}_desc-tumorOld_dseg.nii.gz"
            copy_fn(manual_segs[raw_id], dst)
            has_seg = True

        if raw_id in auto_segs:
            seg_out = auto_deriv / sub_str / "anat"
            seg_out.mkdir(parents=True, exist_ok=True)
            dst = seg_out / f"{sub_str}_desc-tumorOld_dseg.nii.gz"
            copy_fn(auto_segs[raw_id], dst)
            if not has_seg:
                # Also place in primary ground-truth as fallback
                gt_out = gt_deriv / sub_str / "anat"
                gt_out.mkdir(parents=True, exist_ok=True)
                gt_dst = gt_out / f"{sub_str}_desc-tumorOld_dseg.nii.gz"
                copy_fn(auto_segs[raw_id], gt_dst)
            has_seg = True

        if has_seg:
            n_with_seg += 1

        rows.append({
            "participant_id": sub_str,
            "source_id": raw_id,
            "has_manual_seg": raw_id in manual_segs,
            "has_auto_seg": raw_id in auto_segs,
        })

    # Write BIDS metadata
    desc = {
        "Name": "University of Pennsylvania GBM Collection (UPENN-GBM)",
        "BIDSVersion": "1.9.0",
        "DatasetType": "raw",
        "GeneratedBy": [{"Name": "convert_upenn_gbm_to_bids.py"}],
    }
    (output / "dataset_description.json").write_text(json.dumps(desc, indent=2) + "\n")

    for ddir, dname in [
        (gt_deriv, "UPENN-GBM — ground truth segmentations"),
        (auto_deriv, "UPENN-GBM — automated approximate segmentations"),
    ]:
        ddir.mkdir(parents=True, exist_ok=True)
        dd = {
            "Name": dname,
            "BIDSVersion": "1.9.0",
            "DatasetType": "derivative",
        }
        (ddir / "dataset_description.json").write_text(json.dumps(dd, indent=2) + "\n")

    # participants.tsv
    lines = ["participant_id\tsource_id\thas_manual_seg\thas_auto_seg"]
    for r in rows:
        lines.append(
            f"{r['participant_id']}\t{r['source_id']}\t"
            f"{r['has_manual_seg']}\t{r['has_auto_seg']}"
        )
    (output / "participants.tsv").write_text("\n".join(lines) + "\n")

    print(f"\nDone. {len(rows)} subjects, {n_with_seg} with segmentation.")


if __name__ == "__main__":
    main()
