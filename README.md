# oncoprep

[![Documentation](https://readthedocs.org/projects/oncoprep/badge/?version=latest)](https://oncoprep.readthedocs.io)
[![License: GPL3.0](https://img.shields.io/badge/License-GPL3.0-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)


A toolbox for preprocessing and analyzing neuro-oncology MRI using standardized, reproducible pipelines. The toolbox is centered on Nipype workflows, with fMRIprep-style preprocessing, automated tumor segmentation, and radiomics, plus utilities for DICOM→BIDS conversion, BIDS Apps execution, and report generation.

## Scope

**In-scope**
- DICOM→BIDS conversion (heuristic + mapping workflows) with correct sidecars/headers
- BIDS App runner for preprocessing and tumor segmentation (BIDS Derivatives outputs)
- fMRIPrep-style HTML reports
- Multi-site robustness features (sequence missingness, vendor variability, optional defacing)

**Out-of-scope (initially)**
- Regulatory/clinical certification (THIS IS NOT A CLINICAL TOOL!)
- PACS/REST integration (can be a later adapter)
- Non-MRI modalities unless explicitly added later (PET, perfusion, etc.)

## Architecture

OncoPrep is structured as a three-layer Nipype workflow system following [nipreps](https://www.nipreps.org/) conventions (fMRIPrep, sMRIPrep):

```
┌─────────────────────────────────────────────────────────────────────┐
│                          CLI / BIDS App                             │
│                  oncoprep <bids_dir> <out_dir> ...                  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼─────────────────────────────────────┐
│                     Orchestration Layer                            │
│                   init_oncoprep_wf()  (base.py)                    │
│       iterates subjects/sessions, manages memory & logging         │
└───────┬──────────┬───────────┬──────────┬──────────────────────────┘
        │          │           │          │
        ▼          ▼           ▼          ▼
┌───────────┐┌──────────┐┌──────────┐┌────────┐
│ Anatomical││Segmentat.││  Fusion  ││Radiom. │
│    WF     ││    WF    ││    WF    ││   WF   │
│           ││          ││          ││        │
│ •register ││ •Docker  ││ •MAV     ││•Hist   │
│ •skull-   ││  models  ││ •SIMPLE  ││ norm   │
│  strip    ││ •nnInter-││ •BraTS   ││•PyRad  │
│ •deface   ││  active  ││  fusion  ││ feat.  │
│ •template ││  ensemb. ││          ││•reports│
└─────┬─────┘└────┬─────┘└────┬─────┘└───┬────┘
      │           │           │          │
      └───────────┴───────────┴──────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                        Outputs Layer                                │
│              DerivativesDataSink → BIDS Derivatives                 │
│       sub-XXX/anat/  •NIfTI  •JSON  •TSV  •HTML reports             │
└─────────────────────────────────────────────────────────────────────┘
```

### Data flow

```
BIDS input ─► Anatomical WF ─► registered T1w/T1ce/T2w/FLAIR
                 │
                 ├──► Segmentation WF ─► N tumor predictions
                 │         │
                 │         └──► Fusion WF ─► consensus segmentation
                 │                  │
                 │                  └──► Radiomics WF ─► features JSON + report
                 │
                 └──► DerivativesDataSink ─► BIDS-compliant derivatives/
```

## Features

| Feature | Description |
|---------|-------------|
| **BIDS-native** | Full [BIDS](https://bids-specification.readthedocs.io/) and BIDS Derivatives compliance for inputs, outputs, and file naming (via PyBIDS + niworkflows `DerivativesDataSink`). |
| **Nipype workflows** | Composable Nipype workflow graphs — parallel execution, HPC plugin support (SGE, PBS, SLURM), provenance tracking, and crash recovery. |
| **Container-based segmentation** | 14 BraTS-challenge Docker models in isolated containers; supports Docker and Singularity/Apptainer runtimes with GPU passthrough. |
| **nnInteractive segmentation** | Zero-shot 3D promptable segmentation (Isensee et al., 2025) — no Docker needed, CPU or GPU, ~400 MB model weights from HuggingFace. |
| **Ensemble fusion** | Three fusion algorithms (majority vote, SIMPLE, BraTS-specific) combine predictions from multiple models for robust consensus labels. |
| **IBSI-compliant radiomics** | Intensity normalization (z-score, Nyul, WhiteStripe) before PyRadiomics feature extraction; reproducible across scanners and sites. |
| **Multi-modal support** | Joint processing of T1w, T1ce, T2w, and FLAIR with automatic handling of missing modalities. |
| **fMRIPrep-style reports** | Per-subject HTML reports with registration overlays, tumor ROI contour plots, radiomics summary tables, and methods boilerplate. |
| **HPC-ready** | Singularity/Apptainer support with pre-downloadable model caches; PBS/SLURM job script patterns included. |
| **Portable & reproducible** | Docker image with all neuroimaging dependencies (ANTs, FSL, FreeSurfer, dcm2niix) pinned; deterministic workflow hashing for cache reuse. |

## Installation

```bash
pip install oncoprep
```

Optional extras:

```bash
pip install "oncoprep[radiomics]"   # PyRadiomics feature extraction
pip install "oncoprep[dev]"         # development (pytest, ruff)
```

Docker:

```bash
docker pull nko11/oncoprep:latest
```

## Quick start

Convert DICOMs to BIDS:

```bash
oncoprep-convert /path/to/dicoms /path/to/bids --subject 001
```

Run preprocessing:

```bash
oncoprep /path/to/bids /path/to/derivatives participant \
  --participant-label 001
```

Run with segmentation (nnInteractive, no Docker needed):

```bash
oncoprep /path/to/bids /path/to/derivatives participant \
  --participant-label 001 --run-segmentation --default-seg
```

Run with radiomics:

```bash
oncoprep /path/to/bids /path/to/derivatives participant \
  --participant-label 001 --run-radiomics --default-seg
```

Generate reports from existing outputs:

```bash
oncoprep /path/to/bids /path/to/derivatives participant \
  --participant-label 001 --reports-only
```

## Documentation

Full documentation — including tutorials, CLI reference, Docker/HPC usage,
segmentation details, radiomics configuration, and Python API — is available at:

**https://oncoprep.readthedocs.io/en/latest**

## Development

```bash
git clone https://github.com/nikitas-k/oncoprep.git
cd oncoprep
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```
