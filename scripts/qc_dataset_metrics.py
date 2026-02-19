#!/usr/bin/env python
"""Quality-control metrics for each discovered imaging dataset.

Reports per dataset:
  - Number of subjects / subject-sessions
  - Per-modality completeness (missing volumes)
  - Voxel dimensions and grid shapes (consistency check)
  - SNR estimate (foreground mean / background std, estimated via Otsu mask)
  - Dropout detection (slices with ≥ 50 % zeros inside brain mask)
  - Segmentation label inventory and non-empty counts

Usage (targets the raw source data on the volume):
    python scripts/qc_dataset_metrics.py \\
        --volume /Volumes/MHFCBCR/imaging_datasets \\
        --output validation/qc_report.json \\
        [--max-subjects 50]

The --max-subjects flag limits per-dataset sampling for faster runs.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import nibabel as nib
except ImportError:
    sys.exit("nibabel is required.  pip install nibabel")


# ---------------------------------------------------------------------------
# Dataset discovery helpers
# ---------------------------------------------------------------------------

DATASET_SPECS: Dict[str, Dict[str, Any]] = {
    "brats_gli": {
        "display": "BraTS-GLI-2024",
        "subdir": "BraTS-GLI-2024",
        "modality_suffixes": {"t1c": "ce-gadolinium_T1w", "t1n": "T1w", "t2f": "FLAIR", "t2w": "T2w"},
        "seg_suffix": "seg",
        "layout": "brats",  # {ID}/{ID}-{suffix}.nii.gz
    },
    "brats_men": {
        "display": "BraTS-MEN-2024",
        "subdir": "BraTS-MEN-2024",
        "modality_suffixes": {"t1c": "ce-gadolinium_T1w", "t1n": "T1w", "t2f": "FLAIR", "t2w": "T2w"},
        "seg_suffix": "seg",
        "layout": "brats",
    },
    "brats_met": {
        "display": "BraTS-MET-2024",
        "subdir": "BraTS-MET-2024",
        "modality_suffixes": {"t1c": "ce-gadolinium_T1w", "t1n": "T1w", "t2f": "FLAIR", "t2w": "T2w"},
        "seg_suffix": "seg",
        "layout": "brats",
    },
    "ucsf_pdgm": {
        "display": "UCSF-PDGM-v3",
        "subdir": "UCSF-PDGM-v3-20230111",  # outer download folder
        "modality_suffixes": {"T1": "T1w", "T1c": "ce-gadolinium_T1w", "T2": "T2w", "FLAIR": "FLAIR"},
        "seg_suffix": "tumor_segmentation",
        "layout": "ucsf",  # nested: outer/UCSF-PDGM-v3/{ID}/{ID}_{suffix}.nii.gz
    },
    "upenn_gbm": {
        "display": "UPENN-GBM",
        "subdir": "UPENN-GBM",
        "modality_suffixes": {"T1": "T1w", "T1GD": "ce-gadolinium_T1w", "T2": "T2w", "FLAIR": "FLAIR"},
        "seg_suffix": None,  # separate dirs: automated_segm / images_segm
        "layout": "upenn",
    },
    "ucsd_ptgbm": {
        "display": "PKG-UCSD-PTGBM-v1",
        "subdir": "PKG - UCSD-PTGBM-v1",
        "modality_suffixes": {"T1pre": "T1w", "T1post": "ce-gadolinium_T1w", "T2": "T2w", "FLAIR": "FLAIR"},
        "seg_suffix": "BraTS_tumor_seg",
        "layout": "ucsd",  # UCSD-PTGBM/{ID}/{ID}_{suffix}.nii.gz
    },
    "acrin_fmiso": {
        "display": "ACRIN-FMISO (ACRIN-6684)",
        "subdir": "ACRIN-FMISO",
        "modality_suffixes": {},  # DICOM — no NIfTI suffixes
        "seg_suffix": None,
        "layout": "acrin",  # DICOM via TCIA manifest
    },
    "acrin_dsc": {
        "display": "ACRIN-DSC-MR-Brain (ACRIN-6677)",
        "subdir": "ACRIN-DSC-MR-Brain",
        "modality_suffixes": {},  # DICOM
        "seg_suffix": None,
        "layout": "acrin",
    },
    # ---- Additional BraTS challenge datasets ----
    "brats_ssa": {
        "display": "BraTS-SSA-2023",
        "subdir": "ASNR-MICCAI-BraTS2023-SSA-Challenge-TrainingData_V2",
        "modality_suffixes": {"t1c": "ce-gadolinium_T1w", "t1n": "T1w", "t2f": "FLAIR", "t2w": "T2w"},
        "seg_suffix": "seg",
        "layout": "brats",
    },
    "brats_gli_pre": {
        "display": "BraTS2025-GLI-PRE",
        "subdir": "BraTS2025-GLI-PRE-Challenge-TrainingData",
        "modality_suffixes": {"t1c": "ce-gadolinium_T1w", "t1n": "T1w", "t2f": "FLAIR", "t2w": "T2w"},
        "seg_suffix": "seg",
        "layout": "brats",
    },
    "brats_met_2025": {
        "display": "BraTS2025-MET",
        "subdir": "MICCAI-LH-BraTS2025-MET-Challenge-Training",
        "modality_suffixes": {"t1c": "ce-gadolinium_T1w", "t1n": "T1w", "t2f": "FLAIR", "t2w": "T2w"},
        "seg_suffix": "seg",
        "layout": "brats",
    },
    "brats_goat": {
        "display": "BraTS-GoAT-2024",
        "subdir": "MICCAI2024-BraTS-GoAT-TrainingData-With-GroundTruth",
        "modality_suffixes": {"t1c": "ce-gadolinium_T1w", "t1n": "T1w", "t2f": "FLAIR", "t2w": "T2w"},
        "seg_suffix": "seg",
        "layout": "brats",
    },
    # ---- TCIA / public datasets with segmentations ----
    "glis_rt": {
        "display": "GLIS-RT",
        "subdir": "GLIS-RT",
        "modality_suffixes": {},  # DICOM
        "seg_suffix": None,  # RTSTRUCT, not voxel labels
        "layout": "glis_rt",  # Custom: manifest/GLIS-RT/GLI_NNN_TYPE/date/series/
    },
    "lumiere": {
        "display": "LUMIERE",
        "subdir": "LUMIERE",
        "modality_suffixes": {"CT1": "ce-gadolinium_T1w", "T1": "T1w", "T2": "T2w", "FLAIR": "FLAIR"},
        "seg_suffix": None,  # Multiple seg sources in subdirs
        "layout": "lumiere",  # Custom: Imaging/Patient-NNN/week-NNN/
    },
}


def _find_brats_subjects(root: Path) -> List[Path]:
    """Collect BraTS subject dirs from training* / Train subdirs.

    Handles:
    - All subjects directly under root (e.g. BraTS-GoAT, BraTS-SSA)
    - Subjects in training split subdirs (e.g. training_data1/)
    - Mixed layout with subjects at root + sub-splits (BraTS2025-MET)
    """
    subjects = []
    split_subjects = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        name_lower = name.lower()
        # Skip validation directories (no ground truth labels)
        if "validation" in name_lower:
            continue
        # Check if this is a split/container directory (not a subject itself)
        is_split = any(k in name_lower for k in ("train", "challenge"))
        if name.startswith("BraTS-") and not is_split:
            # Direct BraTS subject folder at root level
            subjects.append(child)
        elif is_split:
            # Split subdirectory containing subject folders
            split_subjects.extend(sorted(d for d in child.iterdir() if d.is_dir()))
    # Combine both sources (dedup by name)
    seen = {s.name for s in subjects}
    for s in split_subjects:
        if s.name not in seen:
            subjects.append(s)
            seen.add(s.name)
    return sorted(subjects, key=lambda p: p.name)


def _find_upenn_subjects(root: Path) -> List[Path]:
    """UPENN-GBM: subject dirs sit under NIfTI-files/images_structural/."""
    for prefix in [root / "NIfTI-files", root]:
        struct = prefix / "images_structural"
        if struct.exists():
            return sorted(d for d in struct.iterdir() if d.is_dir())
    return sorted(d for d in root.iterdir() if d.is_dir() and d.name.startswith("UPENN"))


def _find_ucsd_subjects(root: Path) -> List[Path]:
    """UCSD-PTGBM: inside UCSD-PTGBM/ subdir."""
    inner = root / "UCSD-PTGBM"
    if inner.exists():
        return sorted(d for d in inner.iterdir() if d.is_dir())
    return []


def _find_ucsf_subjects(root: Path) -> List[Path]:
    """UCSF-PDGM-v3: subjects live under <root>/UCSF-PDGM-v3/{ID}/."""
    inner = root / "UCSF-PDGM-v3"
    if inner.exists():
        return sorted(
            d for d in inner.iterdir()
            if d.is_dir() and d.name.startswith("UCSF-PDGM-")
        )
    return sorted(
        d for d in root.iterdir()
        if d.is_dir() and d.name.startswith("UCSF-PDGM-")
    )


def _find_flat_subjects(root: Path) -> List[Path]:
    return sorted(d for d in root.iterdir() if d.is_dir() and not d.name.startswith("."))


def _find_acrin_subjects(root: Path) -> List[Path]:
    """ACRIN TCIA datasets: subjects live under manifest-*/ACRIN-*/."""
    for mdir in sorted(root.iterdir()):
        if not mdir.is_dir() or not mdir.name.startswith("manifest"):
            continue
        for inner in sorted(mdir.iterdir()):
            if inner.is_dir() and inner.name.startswith("ACRIN"):
                return sorted(d for d in inner.iterdir() if d.is_dir())
    return []


def _find_glis_rt_subjects(root: Path) -> List[Path]:
    """GLIS-RT TCIA dataset: subjects under manifest-*/GLIS-RT/GLI_NNN_TYPE."""
    for mdir in sorted(root.iterdir()):
        if not mdir.is_dir() or not mdir.name.startswith("manifest"):
            continue
        inner = mdir / "GLIS-RT"
        if inner.exists():
            return sorted(
                d for d in inner.iterdir()
                if d.is_dir() and d.name.startswith("GLI_")
            )
    return sorted(
        d for d in root.iterdir()
        if d.is_dir() and d.name.startswith("GLI_")
    )


def _find_lumiere_subjects(root: Path) -> List[Path]:
    """LUMIERE: patients under Imaging/Patient-NNN/."""
    imaging = root / "Imaging"
    if imaging.exists():
        return sorted(
            d for d in imaging.iterdir()
            if d.is_dir() and d.name.startswith("Patient-")
        )
    return sorted(
        d for d in root.iterdir()
        if d.is_dir() and d.name.startswith("Patient-")
    )


def discover_subjects(ds_key: str, volume: Path) -> List[Path]:
    spec = DATASET_SPECS[ds_key]
    root = volume / spec["subdir"]
    if not root.exists():
        return []
    layout = spec["layout"]
    if layout == "brats":
        return _find_brats_subjects(root)
    elif layout == "upenn":
        return _find_upenn_subjects(root)
    elif layout == "ucsd":
        return _find_ucsd_subjects(root)
    elif layout == "ucsf":
        return _find_ucsf_subjects(root)
    elif layout == "acrin":
        return _find_acrin_subjects(root)
    elif layout == "glis_rt":
        return _find_glis_rt_subjects(root)
    elif layout == "lumiere":
        return _find_lumiere_subjects(root)
    else:
        return _find_flat_subjects(root)


# ---------------------------------------------------------------------------
# Per-image QC
# ---------------------------------------------------------------------------

def _otsu_threshold(data: np.ndarray) -> float:
    """Simple Otsu for foreground/background split."""
    hist, edges = np.histogram(data[data > 0].ravel(), bins=256)
    bin_centers = (edges[:-1] + edges[1:]) / 2
    total = hist.sum()
    if total == 0:
        return 0.0
    w0, w1 = np.cumsum(hist).astype(float), np.cumsum(hist[::-1])[::-1].astype(float)
    mu0 = np.cumsum(hist * bin_centers) / np.maximum(w0, 1)
    mu1 = (np.cumsum((hist * bin_centers)[::-1]) / np.maximum(w1[::-1], 1))[::-1]
    variance = w0[:-1] * w1[1:] * (mu0[:-1] - mu1[1:]) ** 2
    if variance.size == 0:
        return 0.0
    idx = int(np.argmax(variance))
    return float(bin_centers[idx])


def compute_image_qc(filepath: Path) -> Dict[str, Any]:
    """Compute lightweight QC on a single NIfTI."""
    try:
        img = nib.load(str(filepath))
    except Exception as exc:
        return {"error": str(exc)}

    hdr = img.header
    shape = tuple(int(s) for s in img.shape[:3])
    zooms = tuple(round(float(z), 3) for z in hdr.get_zooms()[:3])
    dtype = str(hdr.get_data_dtype())

    result: Dict[str, Any] = {
        "shape": shape,
        "voxel_mm": zooms,
        "dtype": dtype,
    }

    # Load data (float32 to save memory)
    try:
        data = np.asanyarray(img.dataobj, dtype=np.float32)
    except Exception:
        result["error"] = "could not load dataobj"
        return result

    if data.ndim > 3:
        data = data[..., 0]

    result["data_range"] = [float(np.nanmin(data)), float(np.nanmax(data))]

    # Foreground mask via Otsu
    thresh = _otsu_threshold(data)
    fg_mask = data > thresh
    bg_mask = (data > 0) & (~fg_mask)

    fg_mean = float(np.mean(data[fg_mask])) if fg_mask.any() else 0.0
    bg_std = float(np.std(data[bg_mask])) if bg_mask.any() else 1e-9
    snr = fg_mean / max(bg_std, 1e-9)
    result["snr_otsu"] = round(snr, 2)

    # Dropout: slices where brain-mask coverage drops dramatically
    # Only count slices that SHOULD have brain (middle 80%) but have <10% nonzero
    n_slices = data.shape[2]
    lo = int(n_slices * 0.1)
    hi = int(n_slices * 0.9)
    dropout_slices = 0
    brain_slices = 0
    for z in range(lo, hi):
        sl = data[:, :, z]
        if sl.size == 0:
            continue
        brain_slices += 1
        nonzero_frac = np.count_nonzero(sl) / sl.size
        if nonzero_frac < 0.10:
            dropout_slices += 1
    result["dropout_frac"] = round(dropout_slices / max(brain_slices, 1), 3)

    return result


def compute_seg_qc(filepath: Path) -> Dict[str, Any]:
    """QC on segmentation: label inventory, non-empty volume."""
    try:
        img = nib.load(str(filepath))
        data = np.asanyarray(img.dataobj)
    except Exception as exc:
        return {"error": str(exc)}

    if data.ndim > 3:
        data = data[..., 0]

    labels = sorted(int(v) for v in np.unique(data) if v != 0)
    total = int(np.count_nonzero(data))
    zooms = img.header.get_zooms()[:3]
    voxvol = float(np.prod(zooms))
    vol_cc = round(total * voxvol / 1000.0, 2)

    return {
        "labels_present": labels,
        "nonzero_voxels": total,
        "volume_cc": vol_cc,
        "shape": tuple(int(s) for s in img.shape[:3]),
    }


# ---------------------------------------------------------------------------
# Dataset-level QC
# ---------------------------------------------------------------------------

def _resolve_modality_path(
    sub_dir: Path, sub_id: str, suffix: str, layout: str,
) -> Optional[Path]:
    """Guess path for a modality file."""
    if layout in ("brats",):
        p = sub_dir / f"{sub_id}-{suffix}.nii.gz"
    elif layout in ("upenn",):
        # UPENN has underscore separator and subject dir name == sub_id
        p = sub_dir / f"{sub_id}_{suffix}.nii.gz"
    else:
        p = sub_dir / f"{sub_id}_{suffix}.nii.gz"
    return p if p.exists() else None


def _resolve_seg_path(
    sub_dir: Path, sub_id: str, spec: dict, volume: Path,
) -> Optional[Path]:
    layout = spec["layout"]
    seg_suf = spec["seg_suffix"]

    if layout == "upenn":
        # Manual first, then automated; check NIfTI-files/ subdir first
        root = volume / spec["subdir"]
        for prefix in [root / "NIfTI-files", root]:
            for sdir, suf in [("images_segm", "_segm"), ("automated_segm", "_automated_approx_segm")]:
                p = prefix / sdir / f"{sub_id}{suf}.nii.gz"
                if p.exists():
                    return p
        return None

    if seg_suf is None:
        return None

    if layout == "brats":
        p = sub_dir / f"{sub_id}-{seg_suf}.nii.gz"
    else:
        p = sub_dir / f"{sub_id}_{seg_suf}.nii.gz"
    return p if p.exists() else None


# ---------------------------------------------------------------------------
# DICOM-specific QC (ACRIN datasets)
# ---------------------------------------------------------------------------

def _run_dicom_dataset_qc(
    sample: List[Path],
    result: Dict[str, Any],
    spec: Dict[str, Any],
) -> Dict[str, Any]:
    """QC for DICOM-only datasets: count sessions, series, classify by name."""
    import re

    total_sessions = 0
    total_series = 0
    series_types: Dict[str, int] = defaultdict(int)
    roi_count = 0
    manufacturers: Dict[str, int] = defaultdict(int)
    dcm_read_attempted = False

    for sub_dir in sample:
        session_dirs = sorted(d for d in sub_dir.iterdir() if d.is_dir())
        total_sessions += len(session_dirs)

        for ses_dir in session_dirs:
            series_dirs = sorted(d for d in ses_dir.iterdir() if d.is_dir())
            total_series += len(series_dirs)

            for sr_dir in series_dirs:
                sr_name = sr_dir.name
                # Classify series by name
                if re.search(r"Mask|ROI|tumor", sr_name, re.IGNORECASE):
                    roi_count += 1
                    series_types["ROI/Mask"] += 1
                elif re.search(r"T1.*(?:POST|GAD)", sr_name, re.IGNORECASE):
                    series_types["ce-gadolinium_T1w"] += 1
                elif re.search(r"T1.*(?:PRE|SE)\b", sr_name, re.IGNORECASE):
                    series_types["T1w"] += 1
                elif re.search(r"FLAIR", sr_name, re.IGNORECASE):
                    series_types["FLAIR"] += 1
                elif re.search(r"T2(?!.*STAR)(?!.*GRE)", sr_name, re.IGNORECASE):
                    series_types["T2w"] += 1
                elif re.search(r"DWI|DTI|DIFF|TENSOR", sr_name, re.IGNORECASE):
                    series_types["DWI/DTI"] += 1
                elif re.search(r"DSC|Perf", sr_name, re.IGNORECASE):
                    series_types["DSC"] += 1
                elif re.search(r"DCE|pCASL", sr_name, re.IGNORECASE):
                    series_types["DCE"] += 1
                elif re.search(r"FMISO|SUV|PET", sr_name, re.IGNORECASE):
                    series_types["PET"] += 1
                elif re.search(r"BOLD", sr_name, re.IGNORECASE):
                    series_types["BOLD"] += 1
                else:
                    series_types["other"] += 1

                # Try to read one DICOM header for matrix/manufacturer info
                if not dcm_read_attempted:
                    dcm_files = sorted(sr_dir.glob("*.dcm"))[:1]
                    if dcm_files:
                        dcm_read_attempted = True
                        try:
                            import pydicom
                            ds = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True)
                            rows = getattr(ds, "Rows", None)
                            cols = getattr(ds, "Columns", None)
                            if rows and cols:
                                sh_key = f"{rows}×{cols}"
                                result["shapes"][sh_key] = 1
                            mfr = getattr(ds, "Manufacturer", "unknown")
                            if mfr:
                                manufacturers[str(mfr)] += 1
                        except ImportError:
                            result["errors"].append(
                                "pydicom not installed — DICOM header QC skipped"
                            )
                        except Exception as exc:
                            result["errors"].append(f"DICOM read error: {exc}")

    result["dicom_summary"] = {
        "total_sessions": total_sessions,
        "total_series": total_series,
        "avg_sessions_per_subject": round(total_sessions / max(len(sample), 1), 1),
        "series_type_distribution": dict(series_types),
        "roi_mask_series": roi_count,
    }
    if manufacturers:
        result["dicom_summary"]["manufacturers"] = dict(manufacturers)

    # Convert defaultdicts for JSON
    result["shapes"] = dict(result["shapes"])
    result["seg_label_counts"] = dict(result["seg_label_counts"])
    result["missing_modality_subjects"] = dict(result["missing_modality_subjects"])
    del result["snr_values"]
    del result["dropout_values"]
    del result["seg_volumes_cc"]

    return result


# ---------------------------------------------------------------------------
# Dataset-level QC (NIfTI datasets)
# ---------------------------------------------------------------------------

def _run_lumiere_qc(
    sample: List[Path],
    result: Dict[str, Any],
    spec: Dict[str, Any],
) -> Dict[str, Any]:
    """QC for LUMIERE: NIfTI multi-session patients with subdirectory segs."""
    mod_suffixes = spec["modality_suffixes"]
    modality_found = {m: 0 for m in mod_suffixes}
    total_sessions = 0

    for patient_dir in sample:
        # Each patient has week-NNN[-rep] session dirs
        session_dirs = sorted(
            d for d in patient_dir.iterdir()
            if d.is_dir() and d.name.startswith("week-")
        )
        total_sessions += len(session_dirs)

        for ses_dir in session_dirs:
            # Check structural modalities
            for src_name, bids_name in mod_suffixes.items():
                fpath = ses_dir / f"{src_name}.nii.gz"
                if fpath.exists():
                    modality_found[src_name] += 1
                    qc = compute_image_qc(fpath)
                    if "error" not in qc:
                        sh_key = "×".join(str(s) for s in qc["shape"])
                        result["shapes"][sh_key] = result["shapes"].get(sh_key, 0) + 1
                        if "snr_otsu" in qc:
                            result["snr_values"].append(qc["snr_otsu"])
                        if "dropout_frac" in qc:
                            result["dropout_values"].append(qc["dropout_frac"])
                else:
                    result["missing_modality_subjects"][bids_name] = (
                        result["missing_modality_subjects"].get(bids_name, 0) + 1
                    )

            # Check segmentations in subdirs
            for seg_key in ["HD-GLIO-AUTO-segmentation", "DeepBraTumIA-segmentation"]:
                seg_base = ses_dir / seg_key
                if not seg_base.exists():
                    continue
                # HD-GLIO: native/segmentation_CT1_origspace.nii.gz
                # DeepBraTumIA: native/segmentation/ct1_seg_mask.nii.gz
                seg_candidates = list(seg_base.rglob("*.nii.gz"))
                for seg_path in seg_candidates[:1]:  # Just QC first seg per method
                    sqc = compute_seg_qc(seg_path)
                    if "error" not in sqc:
                        for lbl in sqc.get("labels_present", []):
                            result["seg_label_counts"][str(lbl)] = (
                                result["seg_label_counts"].get(str(lbl), 0) + 1
                            )
                        if sqc.get("volume_cc") is not None:
                            result["seg_volumes_cc"].append(sqc["volume_cc"])

    # Summary
    total_mod_checks = total_sessions  # Each session should have each modality
    modality_comp = {}
    for src_name, bids_name in mod_suffixes.items():
        modality_comp[bids_name] = round(
            modality_found[src_name] / max(total_mod_checks, 1) * 100, 1
        )
    result["modality_completeness"] = modality_comp
    result["lumiere_summary"] = {
        "total_sessions": total_sessions,
        "avg_sessions_per_patient": round(total_sessions / max(len(sample), 1), 1),
    }

    snrs = result["snr_values"]
    if snrs:
        result["snr_summary"] = {
            "median": round(float(np.median(snrs)), 2),
            "iqr": [round(float(np.percentile(snrs, 25)), 2),
                    round(float(np.percentile(snrs, 75)), 2)],
            "min": round(float(np.min(snrs)), 2),
            "max": round(float(np.max(snrs)), 2),
        }
    drops = result["dropout_values"]
    if drops:
        result["dropout_summary"] = {
            "median": round(float(np.median(drops)), 3),
            "max": round(float(np.max(drops)), 3),
            "n_above_20pct": sum(1 for d in drops if d > 0.2),
        }
    segs = result["seg_volumes_cc"]
    if segs:
        result["seg_volume_summary_cc"] = {
            "median": round(float(np.median(segs)), 2),
            "iqr": [round(float(np.percentile(segs, 25)), 2),
                    round(float(np.percentile(segs, 75)), 2)],
            "min": round(float(np.min(segs)), 2),
            "max": round(float(np.max(segs)), 2),
        }

    # Convert for JSON
    result["shapes"] = dict(result["shapes"])
    result["seg_label_counts"] = dict(result["seg_label_counts"])
    result["missing_modality_subjects"] = dict(result["missing_modality_subjects"])
    del result["snr_values"]
    del result["dropout_values"]
    del result["seg_volumes_cc"]

    return result


def run_dataset_qc(
    ds_key: str,
    volume: Path,
    max_subjects: int = 0,
) -> Dict[str, Any]:
    spec = DATASET_SPECS[ds_key]
    subjects = discover_subjects(ds_key, volume)

    result: Dict[str, Any] = {
        "dataset": spec["display"],
        "n_subjects_total": len(subjects),
        "subjects_sampled": 0,
        "modality_completeness": {},
        "shapes": defaultdict(int),
        "snr_values": [],
        "dropout_values": [],
        "seg_label_counts": defaultdict(int),
        "seg_volumes_cc": [],
        "missing_modality_subjects": defaultdict(int),
        "errors": [],
    }

    if not subjects:
        result["errors"].append(f"no subjects found for {ds_key}")
        return result

    sample = subjects
    if max_subjects and max_subjects < len(subjects):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(subjects), size=max_subjects, replace=False)
        sample = [subjects[i] for i in sorted(idx)]

    result["subjects_sampled"] = len(sample)

    # ---- DICOM-only datasets (ACRIN, GLIS-RT) ----
    if spec["layout"] in ("acrin", "glis_rt"):
        return _run_dicom_dataset_qc(sample, result, spec)

    # ---- LUMIERE: NIfTI but multi-session per patient ----
    if spec["layout"] == "lumiere":
        return _run_lumiere_qc(sample, result, spec)

    mod_suffixes = spec["modality_suffixes"]
    modality_found = {m: 0 for m in mod_suffixes}

    for sub_dir in sample:
        sub_id = sub_dir.name

        # Modality QC
        for raw_suf, bids_name in mod_suffixes.items():
            fpath = _resolve_modality_path(sub_dir, sub_id, raw_suf, spec["layout"])
            if fpath is None:
                result["missing_modality_subjects"][bids_name] = (
                    result["missing_modality_subjects"].get(bids_name, 0) + 1
                )
                continue
            modality_found[raw_suf] += 1

            qc = compute_image_qc(fpath)
            if "error" in qc:
                result["errors"].append(f"{sub_id}/{raw_suf}: {qc['error']}")
                continue

            sh_key = "×".join(str(s) for s in qc["shape"])
            result["shapes"][sh_key] = result["shapes"].get(sh_key, 0) + 1

            if "snr_otsu" in qc:
                result["snr_values"].append(qc["snr_otsu"])
            if "dropout_frac" in qc:
                result["dropout_values"].append(qc["dropout_frac"])

        # Seg QC
        seg_path = _resolve_seg_path(sub_dir, sub_id, spec, volume)
        if seg_path is not None:
            sqc = compute_seg_qc(seg_path)
            if "error" in sqc:
                result["errors"].append(f"{sub_id}/seg: {sqc['error']}")
            else:
                for lbl in sqc.get("labels_present", []):
                    result["seg_label_counts"][str(lbl)] = (
                        result["seg_label_counts"].get(str(lbl), 0) + 1
                    )
                if sqc.get("volume_cc") is not None:
                    result["seg_volumes_cc"].append(sqc["volume_cc"])

    # Compute summary statistics
    modality_comp = {}
    for raw_suf, bids_name in mod_suffixes.items():
        modality_comp[bids_name] = round(modality_found[raw_suf] / len(sample) * 100, 1)
    result["modality_completeness"] = modality_comp

    snrs = result["snr_values"]
    if snrs:
        result["snr_summary"] = {
            "median": round(float(np.median(snrs)), 2),
            "iqr": [round(float(np.percentile(snrs, 25)), 2),
                    round(float(np.percentile(snrs, 75)), 2)],
            "min": round(float(np.min(snrs)), 2),
            "max": round(float(np.max(snrs)), 2),
        }
    drops = result["dropout_values"]
    if drops:
        result["dropout_summary"] = {
            "median": round(float(np.median(drops)), 3),
            "max": round(float(np.max(drops)), 3),
            "n_above_20pct": sum(1 for d in drops if d > 0.2),
        }
    segs = result["seg_volumes_cc"]
    if segs:
        result["seg_volume_summary_cc"] = {
            "median": round(float(np.median(segs)), 2),
            "iqr": [round(float(np.percentile(segs, 25)), 2),
                    round(float(np.percentile(segs, 75)), 2)],
            "min": round(float(np.min(segs)), 2),
            "max": round(float(np.max(segs)), 2),
        }

    # Convert defaultdicts for JSON
    result["shapes"] = dict(result["shapes"])
    result["seg_label_counts"] = dict(result["seg_label_counts"])
    result["missing_modality_subjects"] = dict(result["missing_modality_subjects"])

    # Remove raw lists from output
    del result["snr_values"]
    del result["dropout_values"]
    del result["seg_volumes_cc"]

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="QC metrics for imaging datasets")
    parser.add_argument(
        "--volume", type=Path, required=True,
        help="Root of external volume containing dataset folders",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("validation/qc_report.json"),
        help="Path for JSON report (default: validation/qc_report.json)",
    )
    parser.add_argument(
        "--max-subjects", type=int, default=50,
        help="Max subjects to sample per dataset (0 = all, default: 50)",
    )
    parser.add_argument(
        "--datasets", nargs="*", default=None,
        help="Subset of dataset keys to process (default: all)",
    )
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=FutureWarning)
    volume = args.volume.resolve()
    if not volume.exists():
        sys.exit(f"ERROR: volume {volume} not found")

    keys = args.datasets or list(DATASET_SPECS.keys())
    report: Dict[str, Any] = {}

    for ds_key in keys:
        if ds_key not in DATASET_SPECS:
            print(f"WARN: unknown dataset key '{ds_key}', skipping")
            continue
        print(f"\n{'='*60}")
        print(f"Processing: {DATASET_SPECS[ds_key]['display']}")
        print(f"{'='*60}")
        report[ds_key] = run_dataset_qc(ds_key, volume, max_subjects=args.max_subjects)

        # Print quick summary
        r = report[ds_key]
        print(f"  Subjects total: {r['n_subjects_total']}")
        print(f"  Sampled:        {r['subjects_sampled']}")
        print(f"  Completeness:   {r['modality_completeness']}")
        print(f"  Shapes:         {r['shapes']}")
        if "snr_summary" in r:
            print(f"  SNR:            {r['snr_summary']}")
        if "dropout_summary" in r:
            print(f"  Dropout:        {r['dropout_summary']}")
        if "seg_volume_summary_cc" in r:
            print(f"  Seg volumes:    {r['seg_volume_summary_cc']}")
        if r["seg_label_counts"]:
            print(f"  Seg labels:     {r['seg_label_counts']}")
        if r["errors"]:
            print(f"  Errors ({len(r['errors'])}): {r['errors'][:5]}")

    # Write report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n\nReport written to {args.output}")


if __name__ == "__main__":
    main()
