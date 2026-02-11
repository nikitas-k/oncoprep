#!/usr/bin/env python
"""Generate synthetic example data for OncoPrep tests.

Creates:
  1. examples/bids/ — A minimal BIDS dataset with synthetic NIfTI volumes and
     JSON sidecars for two subjects (sub-001, sub-002), including T1w, T1ce,
     T2w, FLAIR, perfusion, and ROI masks.
  2. examples/bids/derivatives/oncoprep/ — Synthetic derivative outputs
     (preprocessed images, brain masks, segmentations, tissue priors, transforms).
  3. examples/data/ — Minimal synthetic DICOM-like stub files organised into
     series directories so that tests exercising DICOM discovery still pass.

All image data is randomly generated (no patient content). JSON sidecars
contain plausible but entirely fabricated acquisition parameters.

Usage
-----
    python tests/generate_synthetic_data.py          # from repo root
    python -m tests.generate_synthetic_data          # alternative

Requirements: numpy, nibabel (both are project dependencies).
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nb
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
BIDS_DIR = EXAMPLES / "bids"
DATA_DIR = EXAMPLES / "data"

SHAPE_3D = (32, 32, 32)        # Small 3-D volume for fast I/O
SHAPE_4D = (32, 32, 16, 5)     # 4-D perfusion (fewer slices, 5 time-points)
VOXEL_SIZE = 1.0                # 1 mm isotropic

SUBJECTS = ["001", "002"]

# Realistic-ish MR acquisition parameters (completely fabricated)
T1W_JSON: Dict = {
    "Modality": "MR",
    "MagneticFieldStrength": 3,
    "ImagingFrequency": 123.22,
    "Manufacturer": "SyntheticVendor",
    "ManufacturersModelName": "PhantomScanner",
    "BodyPart": "BRAIN",
    "SoftwareVersions": "synthetic v1.0",
    "MRAcquisitionType": "3D",
    "SeriesDescription": "t1_mprage_sag_p2_1.0_iso",
    "ProtocolName": "t1_mprage_sag_p2_1.0_iso",
    "ScanningSequence": "GR\\IR",
    "SequenceVariant": "SK\\SP\\MP",
    "ScanOptions": "IR",
    "SequenceName": "*tfl3d1_16ns",
    "ImageType": ["ORIGINAL", "PRIMARY", "M", "ND", "NORM"],
    "SeriesNumber": 32,
    "SliceThickness": 1,
    "EchoTime": 0.0035,
    "RepetitionTime": 2.0,
    "InversionTime": 1.1,
    "FlipAngle": 7,
    "BaseResolution": 256,
    "PixelBandwidth": 190,
    "ConversionSoftware": "synthetic-gen",
    "ConversionSoftwareVersion": "v0.1.0",
}

T1CE_JSON: Dict = {
    **T1W_JSON,
    "SeriesDescription": "t1_mprage_sag_p2_1.0_iso_POST",
    "ProtocolName": "t1_mprage_sag_p2_1.0_iso_POST",
    "SeriesNumber": 71,
}

T2W_JSON: Dict = {
    **T1W_JSON,
    "SeriesDescription": "t2_spc_da-fl_sag_p2_1.0",
    "ProtocolName": "t2_spc_da-fl_sag_p2_1.0",
    "ScanningSequence": "SE\\IR",
    "SequenceName": "*spcir_278ns",
    "EchoTime": 0.386,
    "RepetitionTime": 5.0,
    "InversionTime": 1.8,
    "FlipAngle": 120,
    "SeriesNumber": 12,
}

FLAIR_JSON: Dict = {
    **T2W_JSON,
    "SeriesDescription": "COR FLAIR",
    "ImageType": ["DERIVED", "SECONDARY", "MPR"],
    "SeriesNumber": 103,
    "SliceThickness": 3,
}

PERF_JSON: Dict = {
    **T1W_JSON,
    "MRAcquisitionType": "2D",
    "SeriesDescription": "ep2d_perf_p2_MoCo",
    "ProtocolName": "ep2d_perf_p2",
    "ScanningSequence": "EP",
    "SequenceName": "*epfid2d1_128",
    "ImageType": ["DERIVED", "PRIMARY", "PERFUSION", "ND", "MOCO"],
    "EchoTime": 0.03,
    "RepetitionTime": 1.7,
    "FlipAngle": 90,
    "BaseResolution": 128,
    "SeriesNumber": 73,
}

# DICOM-like series to create under examples/data/<subject>/
DICOM_SERIES: Dict[str, Dict] = {
    "001": {
        "COR_FLAIR_0103": 10,
        "EP2D_PERF_P2_0072": 5,
        "EP2D_PERF_P2_MOCO_0073": 5,
        "T1_MPRAGE_SAG_P2_1_0_ISO_0032": 10,
        "T1_MPRAGE_SAG_P2_1_0_ISO_POST_0071": 10,
        "T2_SPC_DA-FL_SAG_P2_1_0_0012": 10,
    },
    "002": {
        "AX_FLAIR_0100": 10,
        "AX_T1_0102": 10,
        "AX_T1_POST_0106": 10,
        "T2_SPC_DA-FL_SAG_P2_1_0_0012": 10,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _affine(voxel_size: float = VOXEL_SIZE) -> np.ndarray:
    """Return a simple diagonal affine."""
    aff = np.eye(4)
    aff[:3, :3] *= voxel_size
    return aff


def _random_volume(
    shape: Tuple[int, ...] = SHAPE_3D,
    low: int = 50,
    high: int = 250,
    dtype: type = np.int16,
) -> np.ndarray:
    """Generate a random integer volume."""
    rng = np.random.default_rng()
    return rng.integers(low, high, size=shape, dtype=dtype)


def _brain_like_volume(shape: Tuple[int, ...] = SHAPE_3D) -> np.ndarray:
    """Generate a rough brain-shaped mask (ellipsoid) as uint8."""
    z, y, x = np.ogrid[
        -1 : 1 : complex(shape[0]),
        -1 : 1 : complex(shape[1]),
        -1 : 1 : complex(shape[2]),
    ]
    # Slightly elongated ellipsoid
    mask = (x ** 2 / 0.7 ** 2 + y ** 2 / 0.85 ** 2 + z ** 2 / 0.9 ** 2) <= 1.0
    return mask.astype(np.uint8)


def _tumor_seg(shape: Tuple[int, ...] = SHAPE_3D) -> np.ndarray:
    """Generate a synthetic multi-label tumor segmentation.

    Labels: 0=background, 1=necrotic core, 2=edema, 4=enhancing tumour
    (following BraTS convention).
    """
    seg = np.zeros(shape, dtype=np.uint8)
    c = [s // 2 for s in shape]
    r = min(shape) // 6

    z, y, x = np.ogrid[
        -1 : 1 : complex(shape[0]),
        -1 : 1 : complex(shape[1]),
        -1 : 1 : complex(shape[2]),
    ]
    # Shift centre slightly
    dist = np.sqrt(
        ((x - 0.15) / 0.22) ** 2 + ((y + 0.1) / 0.28) ** 2 + ((z - 0.05) / 0.25) ** 2
    )
    seg[dist <= 1.0] = 2   # edema (largest)
    seg[dist <= 0.6] = 4   # enhancing
    seg[dist <= 0.3] = 1   # necrotic core
    return seg


def _tissue_priors(brain_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create fake WM / GM / CSF probability maps that sum to 1 inside mask."""
    rng = np.random.default_rng(42)
    raw = rng.random((3, *brain_mask.shape), dtype=np.float32)
    raw[:, brain_mask == 0] = 0
    total = raw.sum(axis=0, keepdims=True)
    total[total == 0] = 1  # avoid /0
    normed = raw / total
    return normed[0], normed[1], normed[2]  # WM, GM, CSF


def _save_nifti(data: np.ndarray, path: Path, affine: Optional[np.ndarray] = None) -> None:
    """Save a numpy array as a compressed NIfTI file."""
    if affine is None:
        affine = _affine()
    img = nb.Nifti1Image(data, affine)
    path.parent.mkdir(parents=True, exist_ok=True)
    nb.save(img, str(path))


def _save_json(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


# ---------------------------------------------------------------------------
# Raw BIDS generation
# ---------------------------------------------------------------------------


def generate_bids_raw(bids_dir: Path) -> None:
    """Create a synthetic raw BIDS dataset."""

    # dataset_description.json
    _save_json(
        {
            "Name": "OncoPrep Synthetic Test Dataset",
            "BIDSVersion": "1.9.0",
            "DatasetType": "raw",
            "License": "CC0",
            "Authors": [{"name": "OncoPrep Synthetic Data Generator"}],
            "Acknowledgements": "Synthetic data – no real patients",
            "HowToAcknowledge": "Please cite OncoPrep and the BIDS specification",
            "Funding": [],
            "EthicsApprovals": [],
            "ReferencesAndLinks": [],
            "DatasetLinks": {},
            "Keywords": ["neuro-oncology", "MRI", "synthetic"],
            "SourceDatasets": [],
            "ConsentLinks": [],
        },
        bids_dir / "dataset_description.json",
    )

    aff = _affine()

    for sub in SUBJECTS:
        prefix = f"sub-{sub}"
        anat_dir = bids_dir / prefix / "anat"
        anat_dir.mkdir(parents=True, exist_ok=True)

        # --- T1w ---
        vol_t1 = _random_volume()
        _save_nifti(vol_t1, anat_dir / f"{prefix}_T1w.nii.gz", aff)
        _save_json(T1W_JSON, anat_dir / f"{prefix}_T1w.json")

        # --- T1ce (contrast-enhanced) ---
        vol_t1ce = _random_volume(low=60, high=255)
        _save_nifti(vol_t1ce, anat_dir / f"{prefix}_ce-gadovist_T1w.nii.gz", aff)
        _save_json(T1CE_JSON, anat_dir / f"{prefix}_ce-gadovist_T1w.json")

        # --- T2w ---
        vol_t2 = _random_volume(low=40, high=200)
        _save_nifti(vol_t2, anat_dir / f"{prefix}_T2w.nii.gz", aff)
        _save_json(T2W_JSON, anat_dir / f"{prefix}_T2w.json")

        # --- FLAIR ---
        vol_flair = _random_volume(low=30, high=180)
        _save_nifti(vol_flair, anat_dir / f"{prefix}_FLAIR.nii.gz", aff)
        _save_json(FLAIR_JSON, anat_dir / f"{prefix}_FLAIR.json")

        # --- ROI masks (binary) ---
        roi = _brain_like_volume()
        _save_nifti(roi, anat_dir / f"{prefix}_FLAIR_ROI1.nii.gz", aff)
        if sub == "002":
            _save_nifti(roi, anat_dir / f"{prefix}_T1w_ROI1.nii.gz", aff)
            _save_nifti(roi, anat_dir / f"{prefix}_ce-gadovist_T1w_ROI1.nii.gz", aff)

        # --- Perfusion (sub-001 only) ---
        if sub == "001":
            perf_dir = bids_dir / prefix / "perf"
            perf_dir.mkdir(parents=True, exist_ok=True)
            vol_perf = _random_volume(shape=SHAPE_4D, low=20, high=150)
            _save_nifti(vol_perf, perf_dir / f"{prefix}_perf.nii.gz", aff)
            _save_json(PERF_JSON, perf_dir / f"{prefix}_perf.json")

    print(f"  [BIDS raw] Generated in {bids_dir}")


# ---------------------------------------------------------------------------
# Derivatives generation
# ---------------------------------------------------------------------------


def generate_derivatives(bids_dir: Path) -> None:
    """Create synthetic OncoPrep derivative outputs."""
    deriv_root = bids_dir / "derivatives" / "oncoprep"
    aff = _affine()

    for sub in SUBJECTS:
        prefix = f"sub-{sub}"
        anat = deriv_root / prefix / "anat"
        anat.mkdir(parents=True, exist_ok=True)

        brain_mask = _brain_like_volume()
        tumor_seg = _tumor_seg()
        wm, gm, csf = _tissue_priors(brain_mask)

        # ----- Native-space derivatives -----
        _save_nifti(_random_volume(), anat / f"{prefix}_desc-preproc_T1w.nii.gz", aff)
        _save_json({"SkullStripped": False}, anat / f"{prefix}_desc-preproc_T1w.json")

        _save_nifti(_random_volume(), anat / f"{prefix}_acq-ce_desc-preproc_T1w.nii.gz", aff)
        _save_json(
            {"Resolution": "native", "SkullStripped": True},
            anat / f"{prefix}_acq-ce_desc-preproc_T1w.json",
        )

        _save_nifti(_random_volume(), anat / f"{prefix}_desc-preproc_T2w.nii.gz", aff)
        _save_json(
            {"Resolution": "native", "SkullStripped": True},
            anat / f"{prefix}_desc-preproc_T2w.json",
        )

        _save_nifti(_random_volume(), anat / f"{prefix}_desc-preproc_FLAIR.nii.gz", aff)
        _save_json(
            {"Resolution": "native", "SkullStripped": True},
            anat / f"{prefix}_desc-preproc_FLAIR.json",
        )

        # Brain mask
        _save_nifti(brain_mask, anat / f"{prefix}_desc-brain_mask.nii.gz", aff)
        _save_json(
            {"RawSources": [f"bids:raw:{prefix}/anat/{prefix}_T1w.nii.gz"], "Type": "Brain"},
            anat / f"{prefix}_desc-brain_mask.json",
        )

        # Tissue segmentation + priors
        tissue_dseg = np.zeros(SHAPE_3D, dtype=np.uint8)
        tissue_dseg[wm > gm] = 3
        tissue_dseg[gm >= wm] = 1
        tissue_dseg[(csf > wm) & (csf > gm)] = 2
        tissue_dseg[brain_mask == 0] = 0
        _save_nifti(tissue_dseg, anat / f"{prefix}_dseg.nii.gz", aff)
        _save_json({"out_path_base": "oncoprep"}, anat / f"{prefix}_dseg.json")

        _save_nifti(wm, anat / f"{prefix}_label-WM_probseg.nii.gz", aff)
        _save_nifti(gm, anat / f"{prefix}_label-GM_probseg.nii.gz", aff)
        _save_nifti(csf, anat / f"{prefix}_label-CSF_probseg.nii.gz", aff)

        # Tumor segmentations
        _save_nifti(tumor_seg, anat / f"{prefix}_desc-tumor_dseg.nii.gz", aff)
        _save_json({"out_path_base": "oncoprep"}, anat / f"{prefix}_desc-tumor_dseg.json")
        _save_nifti(tumor_seg, anat / f"{prefix}_desc-tumorOld_dseg.nii.gz", aff)
        _save_json({"out_path_base": "oncoprep"}, anat / f"{prefix}_desc-tumorOld_dseg.json")
        _save_nifti(tumor_seg, anat / f"{prefix}_desc-tumorNew_dseg.nii.gz", aff)
        _save_json({"out_path_base": "oncoprep"}, anat / f"{prefix}_desc-tumorNew_dseg.json")

        # ----- Template-space derivatives -----
        mni = "MNI152NLin2009cAsym"
        _save_nifti(
            _random_volume(), anat / f"{prefix}_space-{mni}_desc-preproc_T1w.nii.gz", aff
        )
        _save_json({"SkullStripped": True}, anat / f"{prefix}_space-{mni}_desc-preproc_T1w.json")

        _save_nifti(
            _random_volume(), anat / f"{prefix}_acq-ce_space-{mni}_desc-preproc_T1w.nii.gz", aff
        )
        _save_json(
            {"SkullStripped": True}, anat / f"{prefix}_acq-ce_space-{mni}_desc-preproc_T1w.json"
        )

        _save_nifti(
            _random_volume(), anat / f"{prefix}_space-{mni}_desc-preproc_T2w.nii.gz", aff
        )
        _save_json({"SkullStripped": True}, anat / f"{prefix}_space-{mni}_desc-preproc_T2w.json")

        _save_nifti(
            _random_volume(), anat / f"{prefix}_space-{mni}_desc-preproc_FLAIR.nii.gz", aff
        )
        _save_json(
            {"SkullStripped": True}, anat / f"{prefix}_space-{mni}_desc-preproc_FLAIR.json"
        )

        _save_nifti(brain_mask, anat / f"{prefix}_space-{mni}_desc-brain_mask.nii.gz", aff)
        _save_json({"Type": "Brain"}, anat / f"{prefix}_space-{mni}_desc-brain_mask.json")

        _save_nifti(tissue_dseg, anat / f"{prefix}_space-{mni}_dseg.nii.gz", aff)
        _save_nifti(wm, anat / f"{prefix}_space-{mni}_label-WM_probseg.nii.gz", aff)
        _save_nifti(gm, anat / f"{prefix}_space-{mni}_label-GM_probseg.nii.gz", aff)
        _save_nifti(csf, anat / f"{prefix}_space-{mni}_label-CSF_probseg.nii.gz", aff)

        # Transforms (write small dummy HDF5-ish files — just need to exist)
        for direction in [
            f"{prefix}_from-T1w_to-{mni}_mode-image_xfm.h5",
            f"{prefix}_from-{mni}_to-T1w_mode-image_xfm.h5",
        ]:
            xfm_path = anat / direction
            xfm_path.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00" * 64)  # HDF5 magic + padding

        # Figures
        fig_dir = deriv_root / prefix / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        for fig_name in [
            f"{prefix}_desc-about_T1w.html",
            f"{prefix}_desc-conform_T1w.html",
            f"{prefix}_desc-summary_T1w.html",
            f"{prefix}_desc-tumor_dseg.svg",
            f"{prefix}_dseg.svg",
            f"{prefix}_space-{mni}_FLAIR.svg",
            f"{prefix}_space-{mni}_T1w.svg",
            f"{prefix}_space-{mni}_T2w.svg",
            f"{prefix}_space-{mni}_acq-ce_T1w.svg",
        ]:
            if fig_name.endswith(".html"):
                (fig_dir / fig_name).write_text(
                    f"<html><body><h1>Synthetic report — {fig_name}</h1></body></html>\n"
                )
            else:
                (fig_dir / fig_name).write_text(
                    '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">'
                    '<rect width="100" height="100" fill="#ccc"/>'
                    f'<text x="10" y="50" font-size="8">synthetic {fig_name}</text></svg>\n'
                )

    print(f"  [Derivatives] Generated in {deriv_root}")


# ---------------------------------------------------------------------------
# Synthetic DICOM-like stubs
# ---------------------------------------------------------------------------


def generate_dicom_stubs(data_dir: Path) -> None:
    """Create minimal DICOM-like stub files (128 bytes each, not real DICOM).

    These have the .IMA extension so glob-based discovery finds them,
    but contain only the DICOM preamble + magic number and padding.
    They are NOT valid DICOM and cannot be read by pydicom, but they
    satisfy file-existence and extension checks in the test suite.
    """
    # DICOM preamble (128 zero bytes) + "DICM" magic
    preamble = b"\x00" * 128 + b"DICM"
    padding = b"\x00" * 64

    for sub_id, series_map in DICOM_SERIES.items():
        for series_name, n_files in series_map.items():
            series_dir = data_dir / sub_id / series_name
            series_dir.mkdir(parents=True, exist_ok=True)
            for i in range(1, n_files + 1):
                fname = (
                    f"SYNTHETIC.MR.ONCOPREP_TEST."
                    f"{series_name.split('_')[-1]}."
                    f"{i:04d}.2025.01.01.00.00.00.000000.{100000 + i}.IMA"
                )
                (series_dir / fname).write_bytes(preamble + padding)

    print(f"  [DICOM stubs] Generated in {data_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("OncoPrep — generating synthetic example data …")

    # Wipe existing example data
    for d in [BIDS_DIR, DATA_DIR]:
        if d.exists():
            shutil.rmtree(d)
            print(f"  Removed {d}")

    generate_bids_raw(BIDS_DIR)
    generate_derivatives(BIDS_DIR)
    generate_dicom_stubs(DATA_DIR)

    # Re-create mapping template
    mapping = EXAMPLES / "mapping_template.txt"
    mapping.write_text(
        "# Example DICOM series-to-BIDS modality mapping\n"
        "# Format: <dicom_series_description>=<bids_modality>\n"
        "#\n"
        "# Update the left-hand side to match your DICOM SeriesDescription\n"
        "# and the right-hand side to match the desired BIDS label.\n"
        "\n"
        "T1_MPRAGE=T1\n"
        "T1_POST=T1ce\n"
        "T2_SPC=T2\n"
        "T2_FLAIR=FLAIR\n"
    )

    print("\nDone. All example data is now synthetic (no patient content).")


if __name__ == "__main__":
    main()
