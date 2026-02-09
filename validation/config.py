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
    "ucsf_pdgm": DatasetSpec(
        name="UCSF Preoperative Diffuse Glioma MRI (UCSF-PDGM)",
        short_name="UCSF-PDGM",
        n_subjects=501,
        n_sessions=501,
        modalities=["T1w", "ce-T1w", "T2w", "FLAIR"],
        label_set=[1, 2, 3],
        ground_truth_type="Expert-refined BraTS-workflow segmentations",
        field_strength="3T",
        vendor="mixed",
        doi_or_url="https://doi.org/10.1148/ryai.220058",
        split_method="5-fold stratified by tumour grade",
        notes="Pre-operative, standardised protocol, 501 after QC",
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
        "Optional radiomics stability (ICC of selected features)",
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
