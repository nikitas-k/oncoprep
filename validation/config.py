"""Shared configuration for the OncoPrep validation protocol.

Centralises dataset definitions, label mappings, metric specifications,
and statistical analysis plan (SAP) parameters so every phase script
imports a single source of truth.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Label definitions (BraTS convention)
# ---------------------------------------------------------------------------
# BraTS 2023 mapping used in OncoPrep
LABEL_MAP: Dict[int, str] = {
    0: "BG",        # Background
    1: "NETC",      # Non-enhancing tumour core (NCR in older BraTS)
    2: "ED",        # Peritumoral edema / invaded tissue
    3: "ET",        # Enhancing tumour
    4: "RC",        # Resection cavity (post-treatment only)
}

# Composite "regions" used for BraTS-style reporting
REGION_MAP: Dict[str, List[int]] = {
    "ET":  [3],
    "TC":  [1, 3],       # Tumour core = NETC + ET
    "WT":  [1, 2, 3],    # Whole tumour = NETC + ED + ET
    "RC":  [4],           # Cavity (MU-Glioma Post only)
}

# Volume bins for lesion-stratified analysis (in cm³ / cc)
VOLUME_BINS: List[Tuple[float, float]] = [
    (0.0, 1.0),
    (1.0, 5.0),
    (5.0, float("inf")),
]

# Surface Dice tolerance(s) in mm
SURFACE_DICE_TOLERANCES: List[float] = [1.0, 2.0]

# ---------------------------------------------------------------------------
# Dataset specifications
# ---------------------------------------------------------------------------

@dataclass
class DatasetSpec:
    """Specification for a single validation dataset."""
    name: str
    short_name: str
    n_subjects: int
    n_sessions: int
    modalities: List[str]
    label_set: List[int]
    ground_truth_type: str
    field_strength: str  # e.g. "3T", "1.5T/3T", "mixed"
    vendor: str          # e.g. "Siemens", "multi", "mixed"
    doi_or_url: str
    bids_root: Optional[str] = None  # Filled at runtime
    split_method: str = "5-fold-CV"
    notes: str = ""


DATASETS: Dict[str, DatasetSpec] = {
    # ---- BraTS 2024 challenge datasets ----
    "brats_gli": DatasetSpec(
        name="BraTS-GLI-2024: Adult Glioma",
        short_name="BraTS-GLI",
        n_subjects=1621,
        n_sessions=1621,
        modalities=["T1w", "ce-T1w", "T2w", "FLAIR"],
        label_set=[1, 2, 3, 4],
        ground_truth_type="Expert annotations (BraTS challenge, training set with labels)",
        field_strength="mixed",
        vendor="mixed",
        doi_or_url="https://www.synapse.org/Synapse:syn53708249",
        split_method="5-fold-CV",
        notes=(
            "1350 subjects in training_data1 + 271 in training_data_additional. "
            "Grid 182\u00d7218\u00d7182 (SRI24 atlas space). Seg dtype float32. "
            "Labels: 1=NETC, 2=ED, 3=ET, 4=RC (4 present in ~80% of subjects)."
        ),
    ),
    "brats_men": DatasetSpec(
        name="BraTS-MEN-2024: Meningioma",
        short_name="BraTS-MEN",
        n_subjects=1000,
        n_sessions=1000,
        modalities=["T1w", "ce-T1w", "T2w", "FLAIR"],
        label_set=[1, 2, 3],
        ground_truth_type="Expert annotations (BraTS challenge, training set with labels)",
        field_strength="mixed",
        vendor="mixed",
        doi_or_url="https://www.synapse.org/Synapse:syn53708249",
        split_method="5-fold-CV",
        notes="Grid 240×240×155. Seg dtype uint16. Meningioma-specific labels.",
    ),
    "brats_met": DatasetSpec(
        name="BraTS-MET-2024: Brain Metastases",
        short_name="BraTS-MET",
        n_subjects=652,
        n_sessions=652,
        modalities=["T1w", "ce-T1w", "T2w", "FLAIR"],
        label_set=[1, 2, 3],
        ground_truth_type="Expert annotations (BraTS challenge, training set with labels)",
        field_strength="mixed",
        vendor="mixed",
        doi_or_url="https://www.synapse.org/Synapse:syn53708249",
        split_method="5-fold-CV",
        notes=(
            "328 subjects in Training_1 + 324 in Training_2. "
            "Grid 240×240×155. Seg dtype float64."
        ),
    ),
    # ---- Institutional / public datasets ----
    "ucsf_pdgm": DatasetSpec(
        name="UCSF Preoperative Diffuse Glioma MRI (UCSF-PDGM)",
        short_name="UCSF-PDGM",
        n_subjects=496,
        n_sessions=496,
        modalities=["T1w", "ce-T1w", "T2w", "FLAIR"],
        label_set=[1, 2, 4],
        ground_truth_type="Expert-refined BraTS-workflow segmentations",
        field_strength="3T",
        vendor="mixed",
        doi_or_url="https://doi.org/10.1148/ryai.220058",
        split_method="5-fold stratified by tumour grade",
        notes=(
            "Pre-operative, standardised protocol. 496 subjects (was 501, QC removed 5). "
            "Grid 240×240×155. Old BraTS labelling (1=NCR, 2=ED, 4=ET). "
            "Also has DTI, ASL, SWI and brain parcellations."
        ),
    ),
    "upenn_gbm": DatasetSpec(
        name="UPENN-GBM: University of Pennsylvania Glioblastoma",
        short_name="UPENN-GBM",
        n_subjects=671,
        n_sessions=671,
        modalities=["T1w", "ce-T1w", "T2w", "FLAIR"],
        label_set=[1, 2, 4],
        ground_truth_type="Automated + manually-refined segmentations (147 manual, 611 automated),",
        field_strength="mixed",
        vendor="mixed",
        doi_or_url="https://doi.org/10.7937/TCIA.709X-DN49",
        split_method="5-fold-CV",
        notes=(
            "611 automated segmentations, 147 manual. Grid 240×240×155. "
            "Old BraTS labelling (1=NCR, 2=ED, 4=ET). "
            "Structural in NIfTI-files/images_structural/, segs in images_segm/ and automated_segm/."
        ),
    ),
    "ucsd_ptgbm": DatasetSpec(
        name="UCSD Post-Treatment GBM (UCSD-PTGBM-v1)",
        short_name="UCSD-PTGBM",
        n_subjects=184,
        n_sessions=184,
        modalities=["T1w", "ce-T1w", "T2w", "FLAIR"],
        label_set=[1, 2, 3, 4],
        ground_truth_type="BraTS-style tumor segmentation + cellular tumor maps",
        field_strength="mixed",
        vendor="mixed",
        doi_or_url="https://doi.org/10.7937/k6s5-mq14",
        split_method="Patient-level 5-fold (multi-session patients exist)",
        notes=(
            "Post-treatment GBM, some patients have multiple sessions. "
            "Grid 256×256×256, oblique affine. Also has DWI, RSI, DSC, ASL, SWAN."
        ),
    ),
    "mu_glioma_post": DatasetSpec(
        name="MU-Glioma Post: longitudinal post-treatment glioma",
        short_name="MU-Glioma Post",
        n_subjects=203,
        n_sessions=594,
        modalities=["T1w", "ce-T1w", "T2w", "FLAIR"],
        label_set=[1, 2, 3, 4],
        ground_truth_type="Automatic segmentations manually refined by neuroradiologists",
        field_strength="mixed",
        vendor="mixed",
        doi_or_url="https://doi.org/10.1038/s41597-025-06011-7",
        split_method="Patient-level 5-fold (no timepoint leakage)",
        notes="Hardest domain shift — post-treatment, includes resection cavity (label 4)",
    ),
    # ---- ACRIN clinical trial datasets (DICOM source, require dcm2niix) ----
    "acrin_fmiso": DatasetSpec(
        name="ACRIN-6684: FMISO-PET/MR for newly diagnosed GBM",
        short_name="ACRIN-FMISO",
        n_subjects=46,
        n_sessions=430,
        modalities=["T1w", "ce-T1w", "T2w", "FLAIR", "DWI", "DSC", "DCE", "PET"],
        label_set=[],  # ROIs are binary masks (HVC-Mask), not BraTS-style labels
        ground_truth_type="Hand-drawn visit masks (HVC-Mask) + parametric ROIs (DeltaT1ROI)",
        field_strength="mixed",
        vendor="mixed",
        doi_or_url="https://doi.org/10.7937/K9/TCIA.2018.7ICE1GHK",
        split_method="leave-one-out (small N)",
        notes=(
            "DICOM source — requires dcm2niix conversion. "
            "46 subjects (of 50 enrolled), ~9 sessions/subject. "
            "103 ROI dirs: 51 HVC-Mask, 50 DeltaT1ROI, 20 TUMOR PERFUSION. "
            "Multi-modal: MR + FMISO-PET + CT. Rich clinical metadata (29 CSVs). "
            "256×256 matrix, 5 mm slice thickness. Siemens/GE/Philips multi-site."
        ),
    ),
    "acrin_dsc": DatasetSpec(
        name="ACRIN-6677: DSC-MR Perfusion for recurrent GBM",
        short_name="ACRIN-DSC",
        n_subjects=124,
        n_sessions=566,
        modalities=["T1w", "ce-T1w", "T2w", "FLAIR", "DWI", "DSC", "DCE"],
        label_set=[],  # Only 3 subjects have any ROIs
        ground_truth_type="Sparse ROIs (3 subjects only — not systematic)",
        field_strength="mixed",
        vendor="mixed",
        doi_or_url="https://doi.org/10.7937/K9/TCIA.2016.JQEJZZQ8",
        split_method="5-fold-CV",
        notes=(
            "DICOM source — requires dcm2niix conversion. "
            "124 subjects, ~4.6 sessions/subject. Only 3 subjects have tumor ROIs. "
            "Primarily useful for robustness/OOD testing (Phase C), not accuracy (Phase B). "
            "256×256 matrix, 5 mm slice thickness. Predominantly GE, some Siemens."
        ),
    ),
    # ---- Additional BraTS challenge datasets ----
    "brats_ssa": DatasetSpec(
        name="BraTS-SSA-2023: Sub-Saharan Africa Glioma",
        short_name="BraTS-SSA",
        n_subjects=60,
        n_sessions=60,
        modalities=["T1w", "ce-T1w", "T2w", "FLAIR"],
        label_set=[1, 2, 3],
        ground_truth_type="Expert annotations (BraTS challenge, training set with labels)",
        field_strength="mixed",
        vendor="mixed",
        doi_or_url="https://www.synapse.org/Synapse:syn51156910",
        split_method="leave-one-out (small N)",
        notes=(
            "60 subjects from sub-Saharan African sites. BraTS naming convention. "
            "Important for geographic diversity / OOD testing. "
            "Grid sizes likely heterogeneous across sites."
        ),
    ),
    "brats_gli_pre": DatasetSpec(
        name="BraTS2025-GLI-PRE: Pre-treatment Glioma 2025",
        short_name="BraTS-GLI-PRE",
        n_subjects=1251,
        n_sessions=1251,
        modalities=["T1w", "ce-T1w", "T2w", "FLAIR"],
        label_set=[1, 2, 3],
        ground_truth_type="Expert annotations (BraTS 2025 challenge, training set with labels)",
        field_strength="mixed",
        vendor="mixed",
        doi_or_url="https://www.synapse.org/Synapse:syn64233756",
        split_method="5-fold-CV",
        notes=(
            "1251 subjects, pre-treatment only (no resection cavity). "
            "Overlapping subject IDs with BraTS-GLI-2024 but pre-treatment subset. "
            "BraTS naming convention (t1c/t1n/t2f/t2w/seg)."
        ),
    ),
    "brats_met_2025": DatasetSpec(
        name="BraTS2025-MET: Brain Metastases 2025",
        short_name="BraTS-MET-25",
        n_subjects=1296,
        n_sessions=1296,
        modalities=["T1w", "ce-T1w", "T2w", "FLAIR"],
        label_set=[1, 2, 3],
        ground_truth_type="Expert annotations (BraTS 2025 challenge, training set with labels)",
        field_strength="mixed",
        vendor="mixed",
        doi_or_url="https://www.synapse.org/Synapse:syn64233756",
        split_method="5-fold-CV",
        notes=(
            "650 direct BraTS-MET subjects + 646 UCSD subjects in subdirectory. "
            "2 corrected segmentation labels available. "
            "BraTS naming convention. Superset of BraTS-MET-2024."
        ),
    ),
    "brats_goat": DatasetSpec(
        name="BraTS-GoAT-2024: Generalizable Across Tumours",
        short_name="BraTS-GoAT",
        n_subjects=1351,
        n_sessions=1351,
        modalities=["T1w", "ce-T1w", "T2w", "FLAIR"],
        label_set=[1, 2, 3],
        ground_truth_type="Expert annotations (BraTS GoAT challenge, training with GT)",
        field_strength="mixed",
        vendor="mixed",
        doi_or_url="https://www.synapse.org/Synapse:syn53708249",
        split_method="5-fold-CV",
        notes=(
            "1351 subjects with ground truth (+1138 without GT, excluded). "
            "Mixed tumour types (glioma, meningioma, metastases). "
            "Subject IDs have no session suffix (BraTS-GoAT-NNNNN). "
            "BraTS naming convention."
        ),
    ),
    # ---- TCIA datasets with segmentations ----
    "glis_rt": DatasetSpec(
        name="GLIS-RT: Glioma Radiotherapy",
        short_name="GLIS-RT",
        n_subjects=231,
        n_sessions=231,
        modalities=["T1w", "ce-T1w", "T2w", "FLAIR"],
        label_set=[],  # RTSTRUCT contours (GTV/CTV), not voxel labels
        ground_truth_type="Clinical RTSTRUCT contours (GTV, CTV) from radiation therapy planning",
        field_strength="mixed",
        vendor="mixed",
        doi_or_url="https://doi.org/10.7937/E7TG-PG24",
        split_method="5-fold-CV",
        notes=(
            "DICOM source with RTSTRUCT. 231 glioma subjects with GBM/AAC subtypes. "
            "Tumor ROIs: GTV (gross tumour volume), CTV (clinical target volume). "
            "Also has OARs (brainstem, eyes, optic nerves, etc.) and anatomical barriers. "
            "Requires dcm2niix + RTSTRUCT→NIfTI conversion."
        ),
    ),
    "lumiere": DatasetSpec(
        name="LUMIERE: Longitudinal Brain Tumour Monitoring",
        short_name="LUMIERE",
        n_subjects=91,
        n_sessions=300,  # approximate, multi-timepoint
        modalities=["T1w", "ce-T1w", "T2w", "FLAIR"],
        label_set=[0, 1, 2, 3],  # DeepBraTumIA labels; HD-GLIO uses [0,1,2]
        ground_truth_type="Automated segmentations (DeepBraTumIA + HD-GLIO-AUTO)",
        field_strength="mixed",
        vendor="mixed",
        doi_or_url="https://doi.org/10.7937/3RAG-D070",
        split_method="Patient-level 5-fold (no timepoint leakage)",
        notes=(
            "91 patients, multi-timepoint longitudinal monitoring. "
            "NIfTI source with two automated segmentation methods per session. "
            "DeepBraTumIA: labels [0,1,2,3], HD-GLIO: labels [0,1,2]. "
            "Grid 256×256×192. Expert ratings available in CSV."
        ),
    ),
}


# ---------------------------------------------------------------------------
# Comparator specifications
# ---------------------------------------------------------------------------

@dataclass
class ComparatorSpec:
    """Specification for a comparator pipeline."""
    class_id: str        # C1 or C2
    description: str
    executable: str      # Shell command / Docker image
    version_pin: str     # Frozen at study start
    notes: str = ""


COMPARATORS: Dict[str, ComparatorSpec] = {
    "C1": ComparatorSpec(
        class_id="C1",
        description="Strong segmentation baseline + minimal standardization (e.g. nnU-Net BraTS)",
        executable="nnunet_brats:latest",
        version_pin="TBD — freeze before first run",
    ),
    "C2": ComparatorSpec(
        class_id="C2",
        description="Alternative integrated workflow (e.g. institutional pipeline or GLISTRboost)",
        executable="TBD",
        version_pin="TBD — freeze before first run",
    ),
}

# ---------------------------------------------------------------------------
# Phase-specific endpoint definitions
# ---------------------------------------------------------------------------

PHASE_A_ENDPOINTS = {
    "primary": {
        "A1": "Completion rate (overall and per dataset) with 95% Wilson CI",
        "A2": "Standards compliance — proportion of BIDS Derivatives conforming outputs (_mask, _dseg, _probseg, sidecars)",
    },
    "secondary": [
        "Runtime distribution (median, IQR, p95)",
        "Memory footprint (peak RSS)",
        "Deterministic reproducibility (hash-match across platforms/containers)",
    ],
}

PHASE_B_ENDPOINTS = {
    "primary": {
        "B1": "Lesion-wise Dice and lesion-wise HD95, median [IQR] + 95% bootstrap CI",
        "B2": "Patient-level Dice + HD95 (union of lesions), for comparability",
    },
    "secondary": [
        "Surface Dice at 1–2 mm tolerance",
        "Small-lesion sensitivity stratified by volume bins (<1cc, 1–5cc, >5cc)",
    ],
}

PHASE_C_ENDPOINTS = {
    "primary": {
        "C1": "Performance degradation curves under perturbations — AUC of metric vs severity",
        "C2": "Hard-failure rate and soft-failure rate (QC-flagged but completed)",
    },
    "secondary": [
        "OOD generalisation — leave-one-dataset-out / leave-one-site-out",
        "Failure taxonomy rates (registration fail, skull-strip fail, T1ce ambiguity, etc.)",
    ],
}

PHASE_D_ENDPOINTS = {
    "primary": {
        "D1": "Volume agreement vs ground truth — Bland–Altman + ICC, per region",
        "D2": "Longitudinal plausibility — rate of implausible jumps flagged by QC",
    },
    "secondary": [
        "D3: Radiomic feature stability — CV and ICC(3,1) across native-first vs "
        "atlas-first preprocessing; features with ICC≥0.85 and CV≤10% classified as "
        "highly stable; paired Wilcoxon signed-rank with Benjamini–Hochberg FDR",
    ],
}

PHASE_E_ENDPOINTS = {
    "primary": {
        "E1": "Time-to-acceptable contours (manual from scratch vs OncoPrep-assisted)",
        "E2": "Acceptability score (Likert) + edit magnitude (pct voxels modified)",
    },
    "secondary": [
        "Inter-rater variability reduction",
    ],
}

# ---------------------------------------------------------------------------
# SAP (Statistical Analysis Plan) parameters
# ---------------------------------------------------------------------------

@dataclass
class SAPConfig:
    """Pre-specified statistical analysis plan parameters."""
    # Bootstrap
    n_bootstrap: int = 10_000
    ci_level: float = 0.95
    random_seed: int = 42

    # Phase A
    wilson_ci_level: float = 0.95

    # Phase B
    mixed_effects_formula: str = "metric ~ pipeline + (1 | dataset) + (1 | patient)"

    # Phase C — robustness
    perturbation_types: List[str] = field(default_factory=lambda: [
        "gaussian_noise",
        "bias_field",
        "resolution_downsample",
        "intensity_scaling",
    ])
    perturbation_levels: int = 5  # severity levels 0..4

    # Phase D — volumes
    icc_type: str = "ICC(3,1)"  # two-way mixed, single measures, consistency

    # Phase E — reader study
    min_readers: int = 3
    min_cases: int = 30
    washout_weeks: int = 4

    # Sample size heuristics
    target_ci_width_proportion: float = 0.05  # ±5% for Phase A
    assumed_sigma_delta: float = 0.06          # for Phase B power approximation
    detectable_effect: float = 0.02            # δ for paired metric comparison


SAP = SAPConfig()


# ---------------------------------------------------------------------------
# BIDS Derivatives compliance checklist
# ---------------------------------------------------------------------------
BIDS_DERIVATIVES_REQUIRED = [
    "_mask.nii.gz",           # Binary masks
    "_dseg.nii.gz",           # Discrete segmentation
    "_dseg.tsv",              # Label look-up table
    ".json",                  # Sidecar with Sources provenance
]

BIDS_DERIVATIVES_OPTIONAL = [
    "_probseg.nii.gz",        # Probabilistic segmentation
]

# ---------------------------------------------------------------------------
# Perturbation definitions (Phase C)
# ---------------------------------------------------------------------------

@dataclass
class PerturbationSpec:
    """Specification for a single perturbation type."""
    name: str
    param_name: str
    levels: List[float]
    unit: str
    description: str


PERTURBATIONS: Dict[str, PerturbationSpec] = {
    "gaussian_noise": PerturbationSpec(
        name="Additive Gaussian noise",
        param_name="sigma_fraction",
        levels=[0.0, 0.02, 0.05, 0.10, 0.20],
        unit="fraction of image intensity range",
        description="Zero-mean Gaussian noise added to all modalities",
    ),
    "bias_field": PerturbationSpec(
        name="Multiplicative bias field",
        param_name="order",
        levels=[0, 1, 2, 3, 4],
        unit="polynomial order",
        description="Smooth multiplicative bias field (simulated B1 inhomogeneity)",
    ),
    "resolution_downsample": PerturbationSpec(
        name="Resolution degradation",
        param_name="downsample_factor",
        levels=[1.0, 1.5, 2.0, 3.0, 4.0],
        unit="×",
        description="Isotropic downsample + upsample back to original grid",
    ),
    "intensity_scaling": PerturbationSpec(
        name="Global intensity shift",
        param_name="scale_factor",
        levels=[1.0, 0.8, 0.6, 1.2, 1.5],
        unit="multiplicative factor",
        description="Global intensity rescaling across all modalities",
    ),
}

# ---------------------------------------------------------------------------
# Compute / reproducibility spec
# ---------------------------------------------------------------------------

@dataclass
class ComputeSpec:
    """Target compute environments for reproducibility testing."""
    name: str
    gpus: str
    ram_gb: int
    container_runtime: str


COMPUTE_ENVIRONMENTS: List[ComputeSpec] = [
    ComputeSpec("workstation", "1× NVIDIA RTX 4090", 64, "docker"),
    ComputeSpec("hpc_node", "2× NVIDIA A100-80GB", 256, "singularity"),
    ComputeSpec("cloud_cpu", "none (CPU only)", 32, "docker"),
]

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def get_output_root(base: Path) -> Path:
    """Return ``<base>/validation_results/`` and create it."""
    out = base / "validation_results"
    out.mkdir(parents=True, exist_ok=True)
    return out


def get_phase_dir(base: Path, phase: str) -> Path:
    """Return ``<base>/validation_results/phase_<X>/`` and create it."""
    d = get_output_root(base) / f"phase_{phase.lower()}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_docker_config(config_dir: Optional[Path] = None) -> dict:
    """Load OncoPrep dockers.json."""
    if config_dir is None:
        config_dir = Path(__file__).parent.parent / "src" / "oncoprep" / "config"
    with open(config_dir / "dockers.json") as f:
        return json.load(f)
