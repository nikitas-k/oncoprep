# oncoprep
A toolbox for analysing neuro-oncology MRI

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

Run DICOM->BIDS conversion with dcm2niix and BIDScoin:

```bash
oncoprep bids-convert /path/to/dicom /path/to/bids --backend dcm2niix --mapping mapping.txt
```

Generate a report stub:

```bash
oncoprep report --bids-root /path/to/bids --output-dir /path/to/derivatives/reports
```

## Development

```bash
python -m venv .venv
source .venv/bin/activate
git clone https://github.com/nikitas-k/oncoprep.git
pip install -e .[dev]
pytest
```