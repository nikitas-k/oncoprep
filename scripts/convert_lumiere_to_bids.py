#!/usr/bin/env python
"""Convert LUMIERE longitudinal glioma dataset to BIDS layout.

LUMIERE is a longitudinal brain tumour monitoring dataset with 91 patients
and multiple time-points (weeks). Each session contains:
    - CT1.nii.gz (contrast-enhanced T1)
    - T1.nii.gz (native T1)
    - T2.nii.gz (T2 weighted)
    - FLAIR.nii.gz (FLAIR)
    - DeepBraTumIA-segmentation/native/segmentation/{mod}_seg_mask.nii.gz
    - HD-GLIO-AUTO-segmentation/native/segmentation_{mod}_origspace.nii.gz

Source layout:
    LUMIERE/Imaging/Patient-{NNN}/week-{NNN}[-{rep}]/
        CT1.nii.gz
        T1.nii.gz
        T2.nii.gz
        FLAIR.nii.gz
        DeepBraTumIA-segmentation/native/segmentation/ct1_seg_mask.nii.gz
        HD-GLIO-AUTO-segmentation/native/segmentation_CT1_origspace.nii.gz

Output layout (BIDS):
    sub-{NNN}/ses-week{NNN}{rep}/anat/
        sub-{NNN}_ses-week{NNN}{rep}_ce-T1w.nii.gz
        sub-{NNN}_ses-week{NNN}{rep}_T1w.nii.gz
        sub-{NNN}_ses-week{NNN}{rep}_T2w.nii.gz
        sub-{NNN}_ses-week{NNN}{rep}_FLAIR.nii.gz
    derivatives/ground-truth/sub-{NNN}/ses-week{NNN}{rep}/anat/
        sub-{NNN}_ses-week{NNN}{rep}_desc-DeepBraTumIA_dseg.nii.gz
        sub-{NNN}_ses-week{NNN}{rep}_desc-HDGLIO_dseg.nii.gz

Usage:
    python scripts/convert_lumiere_to_bids.py \\
        --source /Volumes/MHFCBCR/imaging_datasets/LUMIERE \\
        --output /data/bids/lumiere \\
        --dataset-name lumiere
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import List, Optional, Tuple


# Source filename → BIDS suffix mapping
MODALITY_MAP = {
    "CT1.nii.gz": "ce-T1w",       # contrast-enhanced T1
    "T1.nii.gz": "T1w",           # native T1
    "T2.nii.gz": "T2w",           # T2 weighted
    "FLAIR.nii.gz": "FLAIR",      # FLAIR
}

# Segmentation sources
SEG_SOURCES = {
    "deepbratumia": {
        "dir": "DeepBraTumIA-segmentation",
        "files": {
            "native/segmentation/ct1_seg_mask.nii.gz": "desc-DeepBraTumIA_dseg",
        },
        "bids_desc": "DeepBraTumIA",
    },
    "hdglio": {
        "dir": "HD-GLIO-AUTO-segmentation",
        "files": {
            "native/segmentation_CT1_origspace.nii.gz": "desc-HDGLIO_dseg",
        },
        "bids_desc": "HDGLIO",
    },
}


def parse_patient_id(patient_dir: str) -> Optional[str]:
    """Extract patient number from Patient-NNN."""
    m = re.match(r"^Patient-(\d+)$", patient_dir)
    return m.group(1) if m else None


def parse_session_id(session_dir: str) -> Optional[str]:
    """Extract session label from week-NNN[-rep].

    week-000-1 → week0001
    week-044   → week044
    """
    m = re.match(r"^week-(\d+)(?:-(\d+))?$", session_dir)
    if not m:
        return None
    week = m.group(1)
    rep = m.group(2) or ""
    return f"week{week}{rep}"


def discover_patients(source_dir: Path) -> List[Path]:
    """Find all patient directories."""
    imaging_dir = source_dir / "Imaging"
    if not imaging_dir.exists():
        imaging_dir = source_dir
    return sorted(
        d for d in imaging_dir.iterdir()
        if d.is_dir() and d.name.startswith("Patient-")
    )


def discover_sessions(patient_dir: Path) -> List[Tuple[str, Path]]:
    """Find all session directories for a patient."""
    sessions = []
    for child in sorted(patient_dir.iterdir()):
        if not child.is_dir():
            continue
        ses_label = parse_session_id(child.name)
        if ses_label is not None:
            sessions.append((ses_label, child))
    return sessions


def _symlink(src: Path, dst: Path) -> None:
    """Create a symlink, removing existing."""
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def convert_session(
    session_dir: Path,
    anat_dir: Path,
    deriv_anat_dir: Path,
    sub_str: str,
    ses_str: str,
    use_symlinks: bool = True,
) -> dict:
    """Convert a single LUMIERE session to BIDS.

    Returns a dict with conversion stats.
    """
    copy_fn = _symlink if use_symlinks else shutil.copy2
    stats = {"modalities": 0, "segs": 0}

    # Convert structural modalities
    for src_name, bids_suffix in MODALITY_MAP.items():
        src = session_dir / src_name
        if src.exists():
            dst = anat_dir / f"{sub_str}_{ses_str}_{bids_suffix}.nii.gz"
            copy_fn(src, dst)
            stats["modalities"] += 1

    # Convert segmentations → derivatives
    for seg_key, seg_spec in SEG_SOURCES.items():
        seg_base = session_dir / seg_spec["dir"]
        if not seg_base.exists():
            continue
        for rel_path, bids_suffix in seg_spec["files"].items():
            src = seg_base / rel_path
            if src.exists():
                dst = deriv_anat_dir / f"{sub_str}_{ses_str}_{bids_suffix}.nii.gz"
                copy_fn(src, dst)
                stats["segs"] += 1

    return stats


def write_bids_metadata(output_dir: Path, dataset_name: str) -> None:
    """Write dataset_description.json."""
    desc = {
        "Name": dataset_name,
        "BIDSVersion": "1.9.0",
        "DatasetType": "raw",
        "GeneratedBy": [
            {
                "Name": "convert_lumiere_to_bids.py",
                "Description": "LUMIERE to BIDS conversion",
            }
        ],
    }
    (output_dir / "dataset_description.json").write_text(
        json.dumps(desc, indent=2) + "\n"
    )


def write_deriv_metadata(deriv_dir: Path, dataset_name: str) -> None:
    """Write derivatives dataset_description.json."""
    desc = {
        "Name": f"{dataset_name} — automated tumour segmentations",
        "BIDSVersion": "1.9.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "DeepBraTumIA / HD-GLIO-AUTO",
                "Description": "Automated brain tumour segmentation models",
            }
        ],
    }
    (deriv_dir / "dataset_description.json").write_text(
        json.dumps(desc, indent=2) + "\n"
    )


def write_participants_tsv(
    output_dir: Path,
    rows: List[dict],
) -> None:
    """Write participants.tsv."""
    lines = ["participant_id\tn_sessions\tsource_id"]
    for r in rows:
        lines.append(
            f"{r['participant_id']}\t{r['n_sessions']}\t{r['source_id']}"
        )
    (output_dir / "participants.tsv").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert LUMIERE longitudinal glioma dataset to BIDS"
    )
    parser.add_argument("--source", type=Path, required=True,
                        help="Root of LUMIERE dataset")
    parser.add_argument("--output", type=Path, required=True,
                        help="BIDS output directory")
    parser.add_argument("--dataset-name", type=str, default="LUMIERE",
                        help="Name for dataset_description.json")
    parser.add_argument("--copy", action="store_true", default=False,
                        help="Copy files instead of symlinking")
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    deriv_dir = output / "derivatives" / "ground-truth"

    print(f"Source:  {source}")
    print(f"Output:  {output}")
    print(f"Derivs:  {deriv_dir}")

    patients = discover_patients(source)
    print(f"Found {len(patients)} patients")

    rows = []
    total_sessions = 0
    total_mods = 0
    total_segs = 0

    for i, patient_dir in enumerate(patients):
        pat_id = parse_patient_id(patient_dir.name)
        if pat_id is None:
            print(f"  SKIP: cannot parse {patient_dir.name}")
            continue

        sub_str = f"sub-{pat_id}"
        sessions = discover_sessions(patient_dir)
        n_sessions = 0

        for ses_label, session_dir in sessions:
            ses_str = f"ses-{ses_label}"

            anat_dir = output / sub_str / ses_str / "anat"
            anat_dir.mkdir(parents=True, exist_ok=True)
            deriv_anat_dir = deriv_dir / sub_str / ses_str / "anat"
            deriv_anat_dir.mkdir(parents=True, exist_ok=True)

            stats = convert_session(
                session_dir, anat_dir, deriv_anat_dir,
                sub_str, ses_str,
                use_symlinks=not args.copy,
            )
            total_mods += stats["modalities"]
            total_segs += stats["segs"]
            n_sessions += 1

        total_sessions += n_sessions
        rows.append({
            "participant_id": sub_str,
            "n_sessions": n_sessions,
            "source_id": patient_dir.name,
        })

        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(patients)} patients")

    write_bids_metadata(output, args.dataset_name)
    write_deriv_metadata(deriv_dir, args.dataset_name)
    write_participants_tsv(output, rows)

    print(f"\nDone. Converted {len(rows)} patients, {total_sessions} sessions.")
    print(f"  Modality files: {total_mods}")
    print(f"  Segmentation files: {total_segs}")
    print(f"  Raw data:     {output}")
    print(f"  Ground truth: {deriv_dir}")


if __name__ == "__main__":
    main()
