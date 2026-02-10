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

## HPC / Singularity

On HPC systems (e.g. NCI Gadi), Docker is unavailable. OncoPrep supports Singularity/Apptainer. All steps below run directly on the **login node** — no need to exec into any container.

### 1. Build the OncoPrep SIF

```bash
module load singularity
singularity pull oncoprep.sif docker://nko11/oncoprep:latest
```

### 2. Pre-download segmentation models

The segmentation models are Docker images that need to be converted to SIF files. A standalone script is included in the repo — no pip install needed:

```bash
# Download the script (or copy from the repo's scripts/ directory)
wget -O pull_seg_models.sh https://raw.githubusercontent.com/nikitas-k/oncoprep/main/scripts/pull_seg_models.sh

# Pull all models
bash pull_seg_models.sh /scratch/$PROJECT/$USER/seg_cache

# Or just CPU models (smaller, no GPU needed)
bash pull_seg_models.sh /scratch/$PROJECT/$USER/seg_cache --cpu-only
```

### 3. Run on a compute node

```bash
singularity run --nv \
  --bind /scratch/$PROJECT/$USER/seg_cache:/seg_cache \
  oncoprep.sif \
  /data/bids /data/output participant \
  --participant-label sub-001 \
  --run-segmentation \
  --container-runtime singularity \
  --seg-cache-dir /seg_cache
```

CPU-only (single model):

```bash
singularity run \
  --bind /scratch/$PROJECT/$USER/seg_cache:/seg_cache \
  oncoprep.sif \
  /data/bids /data/output participant \
  --participant-label sub-001 \
  --run-segmentation --default-seg \
  --container-runtime singularity \
  --seg-cache-dir /seg_cache
```

### PBS job script example (NCI Gadi)

```bash
#!/bin/bash
#PBS -l ncpus=12,mem=48GB,walltime=04:00:00,jobfs=100GB
#PBS -l storage=scratch/$PROJECT
#PBS -l ngpus=1
#PBS -l wd

module load singularity

SEG_CACHE=/scratch/$PROJECT/$USER/seg_cache

singularity run --nv \
  --bind $SEG_CACHE:/seg_cache \
  --bind /scratch/$PROJECT/$USER/bids:/data/bids:ro \
  --bind /scratch/$PROJECT/$USER/output:/data/output \
  --bind $PBS_JOBFS:/work \
  /scratch/$PROJECT/$USER/oncoprep.sif \
  /data/bids /data/output participant \
  --participant-label sub-001 \
  --run-segmentation \
  --container-runtime singularity \
  --seg-cache-dir /seg_cache \
  --work-dir /work
```

## Pre-downloading models (Docker)

If using Docker and you want to pre-cache models (e.g. for offline use):

```bash
oncoprep-models pull -o /path/to/seg_cache --runtime docker
oncoprep /data/bids /data/output participant \
  --run-segmentation --seg-cache-dir /path/to/seg_cache
```

## Development

```bash
python -m venv .venv
source .venv/bin/activate
git clone https://github.com/nikitas-k/oncoprep.git
pip install -e .[dev]
pytest
```
