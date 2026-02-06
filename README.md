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

<<<<<<< Updated upstream
=======
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

## Docker

OncoPrep is available as a Docker image with all neuroimaging dependencies (ANTs, FSL, dcm2niix) pre-installed.

### Pull

```bash
docker pull nko11/oncoprep:latest
```

### Run preprocessing

```bash
docker run --platform linux/amd64 --rm \
  -v /path/to/bids:/data/bids:ro \
  -v /path/to/output:/data/output \
  -v /path/to/work:/data/work \
  nko11/oncoprep:latest \
  /data/bids /data/output participant --participant-label sub-001
```

### Run with segmentation

Segmentation launches Docker containers for each model, so the host Docker socket must be mounted:

```bash
docker run --platform linux/amd64 --rm \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /path/to/bids:/data/bids:ro \
  -v /path/to/output:/data/output \
  -v /path/to/work:/data/work \
  nko11/oncoprep:latest \
  /data/bids /data/output participant \
  --participant-label sub-001 --run-segmentation --default-seg
```

On a Linux host with GPU support (required for ensemble mode):

```bash
docker run --rm --gpus all \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /path/to/bids:/data/bids:ro \
  -v /path/to/output:/data/output \
  -v /path/to/work:/data/work \
  nko11/oncoprep:latest \
  /data/bids /data/output participant \
  --participant-label sub-001 --run-segmentation --use-gpu
```

### DICOM conversion

```bash
docker run --platform linux/amd64 --rm \
  -v /path/to/dicoms:/data/dicom:ro \
  -v /path/to/bids:/data/bids \
  --entrypoint oncoprep-convert \
  nko11/oncoprep:latest \
  /data/dicom /data/bids --subject 001
```

### Build from source

```bash
docker build --platform linux/amd64 -t oncoprep:latest .
```

> **Note:** The image targets `linux/amd64`. On Apple Silicon Macs, Docker Desktop will use Rosetta emulation automatically, but ANTs binaries may hit `Illegal instruction` errors due to AVX instruction limitations. Use the native pip install for local development on ARM Macs.

>>>>>>> Stashed changes
## Development

```bash
python -m venv .venv
source .venv/bin/activate
git clone https://github.com/nikitas-k/oncoprep.git
pip install -e .[dev]
pytest
```
