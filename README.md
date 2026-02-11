# oncoprep
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

OncoPrep supports multi-model ensemble tumor segmentation using containerized BraTS challenge models. The model list includes top-performing solutions from the BraTS challenge (2018-2024).

### Supported Models

| Key | Year | Rank | Task | Architecture | Authors | Docker Image |
|-----|------|------|------|-------------|---------|--------------|
| [`econib`](https://link.springer.com/chapter/10.1007/978-3-030-11726-9_2) | 2018 | — | Adult Glioma | 3D U-Net | M. Marcinkiewicz | `econib/brats-2018` |
| [`mic-dkfz`](https://link.springer.com/chapter/10.1007/978-3-030-11726-9_21) | 2018 | **1st** | Adult Glioma | nnU-Net | F. Isensee | `fabianisensee/isen2018` |
| [`scan`](https://link.springer.com/chapter/10.1007/978-3-030-11726-9_40) | 2018 | — | Adult Glioma | DeepSCAN | R. McKinley | `mckinleyscan/brats:v1` |
| [`xfeng`](https://link.springer.com/chapter/10.1007/978-3-030-11726-9_25) | 2018 | — | Adult Glioma | 3D U-Net | X. Feng | `xf4j/brats18` |
| [`lfb_rwth`](https://link.springer.com/chapter/10.1007/978-3-030-11726-9_1) | 2018 | — | Adult Glioma | 3D U-Net | L. Weninger | `leonweninger/brats18_segmentation` |
| [`gbmnet`](https://link.springer.com/chapter/10.1007/978-3-030-11726-9_22) | 2018 | — | Adult Glioma | GBMNet | N. Nuechterlein | `nknuecht/gbmnet18` |
| [`zyx_2019`](https://link.springer.com/chapter/10.1007/978-3-030-32248-9_29) | 2019 | — | Adult Glioma | — | Y. Zhao | `jiaocha/zyxbrats` |
| [`scan_2019`](https://link.springer.com/chapter/10.1007/978-3-030-46640-4_36) | 2019 | — | Adult Glioma | DeepSCAN | R. McKinley | `scan/brats2019` |
| [`isen-20`](https://doi.org/10.1007/978-3-030-72087-2_11) | 2020 | **1st** | Adult Glioma | nnU-Net | F. Isensee (DKFZ) | `brats/isen-20` |
| [`hnfnetv1-20`](https://doi.org/10.1007/978-3-030-72087-2_6) | 2020 | 2nd | Adult Glioma | HNFNet | H. Jia | `brats/hnfnetv1-20` |
| [`yixinmpl-20`](https://doi.org/10.1007/978-3-030-72084-1_21) | 2020 | 2nd | Adult Glioma | — | Y. Wang | `brats/yixinmpl-20` |
| [`sanet0-20`](https://doi.org/10.1007/978-3-030-72084-1_26) | 2020 | 3rd | Adult Glioma | SANet | Y. Yuan | `brats/sanet0-20` |
| [`scan-20`](https://doi.org/10.1007/978-3-030-72084-1_36) | 2020 | — | Adult Glioma | DeepSCAN | R. McKinley | `brats/scan-20` |
| [`kaist-21`](https://link.springer.com/chapter/10.1007/978-3-031-09002-8_16) | 2021 | **1st** | Adult Glioma | Extended nnU-Net + Axial Attention | H.M. Luu, S.-H. Park | `rixez/brats21nnunet` |

### Running segmentation

Run preprocessing pipeline and segmentation (with ensemble consensus voting)

```bash
# Ensemble mode, slower, needs GPU (all models + fusion)
oncoprep /path/to/bids /path/to/derivatives participant --participant-label sub-001 --run-segmentation

# Default model, CPU only, faster (econib 2018)
oncoprep /path/to/bids /path/to/derivatives participant --participant-label sub-001 --run-segmentation --default-seg

# Custom model path
oncoprep /path/to/bids /path/to/derivatives participant --participant-label sub-001 --run-segmentation --seg-model-path /path/to/model
```

## Radiomics

OncoPrep includes a radiomics feature extraction pipeline built on [PyRadiomics](https://pyradiomics.readthedocs.io/), integrated as a Nipype workflow (`init_anat_radiomics_wf`). It computes quantitative imaging features from preprocessed anatomical images using the tumor segmentation masks produced by the segmentation step.

### Install

PyRadiomics is an optional dependency:

```bash
pip install "oncoprep[radiomics]"
```

### Running radiomics

The `--run-radiomics` flag enables feature extraction after segmentation. It implies `--run-segmentation`, so you don't need to pass both:

```bash
# Run preprocessing + segmentation + radiomics (default single model, CPU)
oncoprep /path/to/bids /path/to/derivatives participant \
  --participant-label sub-001 --run-radiomics --default-seg

# Run with GPU ensemble segmentation + radiomics
oncoprep /path/to/bids /path/to/derivatives participant \
  --participant-label sub-001 --run-radiomics
```

### What gets extracted

Features are computed for each **tumor sub-region** and **composite region**:

| Region | Label | Description |
|--------|-------|-------------|
| NCR | 1 | Necrotic Core |
| ED | 2 | Peritumoral Edema |
| ET | 3 | Enhancing Tumor |
| RC | 4 | Resection Cavity |
| WT | 1+2+3+4 | Whole Tumor (composite) |
| TC | 1+3+4 | Tumor Core (composite) |

For each region, the following feature classes are extracted (all enabled by default):

| Feature Class | Description | Example Features |
|---------------|-------------|------------------|
| **Shape** | 3D morphological descriptors | Volume, Surface Area, Sphericity, Elongation, Flatness |
| **First Order** | Intensity histogram statistics | Mean, Median, Std Dev, Skewness, Kurtosis, Entropy, Energy |
| **GLCM** | Gray-Level Co-occurrence Matrix | Contrast, Correlation, Homogeneity, Entropy, Cluster Shade |
| **GLRLM** | Gray-Level Run Length Matrix | Short/Long Run Emphasis, Gray Level Non-Uniformity |
| **GLSZM** | Gray-Level Size Zone Matrix | Small/Large Area Emphasis, Zone Entropy |
| **GLDM** | Gray-Level Dependence Matrix | Dependence Entropy, Non-Uniformity |
| **NGTDM** | Neighbouring Gray Tone Difference Matrix | Coarseness, Contrast, Busyness, Complexity |

### Configuration

The radiomics extraction can be configured through the Python API by passing parameters to the workflow or interface directly.

#### Feature class selection

Toggle individual feature classes when building the workflow:

```python
from oncoprep.workflows.radiomics import init_anat_radiomics_wf

wf = init_anat_radiomics_wf(
    output_dir='/path/to/derivatives',
    extract_shape=True,       # 3D shape features
    extract_firstorder=True,  # intensity statistics
    extract_glcm=True,        # co-occurrence matrix
    extract_glrlm=False,      # run length matrix (disabled)
    extract_glszm=False,      # size zone matrix (disabled)
    extract_gldm=False,       # dependence matrix (disabled)
    extract_ngtdm=False,      # grey tone difference (disabled)
)
```

#### Custom PyRadiomics settings

Pass a settings dict to the `PyRadiomicsFeatureExtraction` interface for fine-grained control over PyRadiomics internals (bin width, resampling, image filters, etc.):

```python
from oncoprep.interfaces.radiomics import PyRadiomicsFeatureExtraction

extract = PyRadiomicsFeatureExtraction(
    settings={
        'binWidth': 25,
        'resampledPixelSpacing': [1, 1, 1],
        'interpolator': 'sitkBSpline',
        'normalizeScale': 1,
        'normalize': True,
    },
)
```

See the [PyRadiomics documentation](https://pyradiomics.readthedocs.io/en/latest/customization.html) for all available settings.

#### Custom label definitions

Override the default BraTS label map for non-standard segmentation masks:

```python
from oncoprep.interfaces.radiomics import PyRadiomicsFeatureExtraction

extract = PyRadiomicsFeatureExtraction(
    label_map={1: 'tumor', 2: 'edema'},
    label_names={1: 'Tumor', 2: 'Edema'},
    composites={
        'ALL': {'name': 'All Abnormal', 'labels': [1, 2]},
    },
)
```

#### Multi-modal extraction

Extract features across all four MRI modalities (T1w, T1ce, T2w, FLAIR) using the same segmentation mask:

```python
from oncoprep.workflows.radiomics import init_multimodal_radiomics_wf

wf = init_multimodal_radiomics_wf(
    output_dir='/path/to/derivatives',
    modalities=['t1w', 't1ce', 't2w', 'flair'],
)
```

### Outputs

Radiomics produces two BIDS derivative outputs per subject:

| File | Description |
|------|-------------|
| `sub-XXX_desc-radiomics_features.json` | Full feature set as structured JSON, keyed by region then feature class |
| `sub-XXX.html` (Radiomics section) | Summary and detailed feature tables embedded in the subject HTML report |

The JSON output structure:

```json
{
  "NCR": {
    "label": 1,
    "name": "Necrotic Core (NCR)",
    "features": {
      "shape": { "MeshVolume": 1234.5, "Sphericity": 0.82, ... },
      "firstorder": { "Mean": 45.2, "Entropy": 3.1, ... },
      "glcm": { "Contrast": 12.3, ... }
    }
  },
  "WT": {
    "label": [1, 2, 3, 4],
    "name": "Whole Tumor (WT)",
    "features": { ... }
  }
}
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

On HPC systems, Docker can be unavailable, so OncoPrep supports Singularity/Apptainer for HPC deployment. All steps below run directly on the **login node** — no need to exec into any container.

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
  /data/bids /data/bids/derivatives participant \
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
  /data/bids /data/bids/derivatives participant \
  --participant-label sub-001 \
  --run-segmentation --default-seg \
  --container-runtime singularity \
  --seg-cache-dir /seg_cache
```

### PBS job script example

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
  --bind /scratch/$PROJECT/$USER/bids/derivatives:/data/bids/derivatives \
  --bind $PBS_JOBFS:/work \
  /scratch/$PROJECT/$USER/oncoprep.sif \
  /data/bids /data/bids/derivatives participant \
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
oncoprep /data/bids /data/bids/derivatives participant \
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
