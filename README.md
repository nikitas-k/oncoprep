# oncoprep
A toolbox for analyzing neuro-oncology MRI using standardized, reproducible pipelines. The toolbox is centered on Nipype workflows, with BraTS-style preprocessing and tumor segmentation, plus utilities for DICOM→BIDS conversion, BIDS Apps execution, and report generation.

## Scope

**In-scope**
- DICOM→BIDS conversion (heuristic + mapping workflows) with correct sidecars/headers
- BIDS App runner for preprocessing and tumor segmentation (BIDS Derivatives outputs)
- fMRIPrep-style HTML reports
- Multi-site robustness features (sequence missingness, vendor variability, optional defacing)

**Out-of-scope (initially)**
- Regulatory/clinical certification
- PACS/REST integration (can be a later adapter)
- Non-MRI modalities unless explicitly added later (PET, perfusion, etc.)

## Quick start

```bash
pip install oncoprep
```

Run DICOM->BIDS conversion with dcm2niix:

```bash
oncoprep-convert /path/to/dicoms /path/to/bids
```

Run preprocessing pipeline

```bash
oncoprep /path/to/bids /path/to/derivatives participant --participant-label sub-001
```

Generate a report stub:

```bash
oncoprep /path/to/bids /path/to/derivatives participant --participant-label sub-001 --reports-only
```

## Segmentation

Run preprocessing pipeline and segmentation (with ensemble consensus voting)

```bash
# Ensemble mode, slower, needs GPU (all models + fusion)
oncoprep /path/to/bids /path/to/derivatives participant --participant-label sub-001 --run-segmentation

# Default model, CPU only, faster (econib 2018)
oncoprep /path/to/bids /path/to/derivatives participant --participant-label sub-001 --run-segmentation --default-seg

# Custom model path
oncoprep /path/to/bids /path/to/derivatives participant --participant-label sub-001 --run-segmentation --seg-model-path /path/to/model
```

## Development

```bash
python -m venv .venv
source .venv/bin/activate
git clone https://github.com/nikitas-k/oncoprep.git
pip install -e .[dev]
pytest
```
