# Tutorial: End-to-End Neuro-Oncology Preprocessing

This tutorial walks through a complete OncoPrep workflow — from raw DICOMs
to preprocessed derivatives and tumor segmentation — using a single subject.

## Prerequisites

```bash
pip install oncoprep
```

You will also need:
- A directory of DICOM files for one or more subjects
- Docker installed (for the segmentation step)
- ~4 GB free disk space for intermediate files

## Step 1 — DICOM to BIDS conversion

Organize your raw data so each subject has its own directory:

```
raw_dicoms/
└── 001/
    ├── T1_MPRAGE_SAG_P2_1_0_ISO_0032/
    ├── T1_MPRAGE_SAG_P2_1_0_ISO_POST_0071/
    ├── T2_SPC_DA-FL_SAG_P2_1_0_0012/
    └── COR_FLAIR_0103/
```

Run the conversion:

```bash
oncoprep-convert raw_dicoms/ bids_output/ --subject 001
```

The output is a valid BIDS dataset:

```
bids_output/
├── dataset_description.json
└── sub-001/
    └── anat/
        ├── sub-001_T1w.nii.gz
        ├── sub-001_T1w.json
        ├── sub-001_ce-gadolinium_T1w.nii.gz
        ├── sub-001_T2w.nii.gz
        └── sub-001_FLAIR.nii.gz
```

:::{tip}
For batch conversion of many subjects, use the `--batch` flag:
```bash
oncoprep-convert raw_dicoms/ bids_output/ --batch
```
:::

## Step 2 — Preprocessing

Run the anatomical preprocessing pipeline:

```bash
oncoprep bids_output/ derivatives/ participant \
  --participant-label 001 \
  --nprocs 4
```

This will:
1. **Validate** the BIDS dataset structure
2. **Conform** images to 1 mm isotropic resolution
3. **Register** T1ce, T2w, and FLAIR to the T1w reference image
4. **Skull-strip** using ANTs brain extraction
5. **Normalize** to MNI152NLin2009cAsym template space
6. Write all outputs as BIDS derivatives

:::{note}
Processing a single subject typically takes 15–30 minutes on 4 cores.
:::

### Choosing a skull-stripping backend

OncoPrep supports three backends:

```bash
# ANTs brain extraction (default)
oncoprep ... --skull-strip-backend ants

# HD-BET (GPU-accelerated, requires pip install "oncoprep[hd-bet]")
oncoprep ... --skull-strip-backend hdbet

# FreeSurfer SynthStrip
oncoprep ... --skull-strip-backend synthstrip
```

### Choosing a registration backend

```bash
# ANTs SyN (default, slower, more accurate)
oncoprep ... --registration-backend ants

# PICSL Greedy (faster)
oncoprep ... --registration-backend greedy
```

## Step 3 — Tumor segmentation

Segmentation requires Docker. Make sure the Docker daemon is running.

### Single model (fast, CPU)

```bash
oncoprep bids_output/ derivatives/ participant \
  --participant-label 001 \
  --run-segmentation --default-seg
```

This runs the default model (`econib/brats-2018`) on CPU. Takes ~5 minutes.

### Multi-model ensemble (GPU)

```bash
oncoprep bids_output/ derivatives/ participant \
  --participant-label 001 \
  --run-segmentation
```

This runs all 14 BraTS models and fuses their predictions using majority
voting. Requires a GPU with CUDA support. Takes ~30–60 minutes.

The output is a discrete segmentation label map:

```
derivatives/oncoprep/sub-001/anat/sub-001_desc-tumor_dseg.nii.gz
```

With BraTS labels:
| Label | Region                |
|-------|-----------------------|
| 1     | Necrotic Core (NCR)   |
| 2     | Peritumoral Edema (ED)|
| 3     | Enhancing Tumor (ET)  |
| 4     | Resection Cavity (RC) |

## Step 4 — Radiomics (optional)

Extract quantitative imaging features from the segmentation:

```bash
pip install "oncoprep[radiomics]"

oncoprep bids_output/ derivatives/ participant \
  --participant-label 001 \
  --run-radiomics --default-seg
```

`--run-radiomics` implies `--run-segmentation`, so you don't need both flags.

Output:

```
derivatives/oncoprep/sub-001/anat/sub-001_desc-radiomics_features.json
```

The JSON contains features for each tumor region (NCR, ED, ET, WT, TC)
across feature classes (shape, first-order, GLCM, GLRLM, GLSZM, GLDM,
NGTDM).

## Step 5 — Quality control with MRIQC (optional)

Run [MRIQC](https://mriqc.readthedocs.io/) to compute no-reference image
quality metrics (IQMs) on the raw BIDS data. This runs *in parallel* with
preprocessing and can flag unusable scans early:

```bash
pip install "oncoprep[mriqc]"

oncoprep bids_output/ derivatives/ participant \
  --participant-label 001 \
  --run-qc
```

Quality metrics are written to `derivatives/mriqc/` and include per-subject
HTML reports and JSON files with IQMs such as SNR, CNR, CJV, EFC, and FBER.

## Step 6 — Reports

Generate an HTML quality-assurance report:

```bash
oncoprep bids_output/ derivatives/ participant \
  --participant-label 001 --reports-only
```

Open the generated `sub-001.html` in your browser for a summary of
preprocessing steps, registration quality, and segmentation overlays.

## Using the Python API

For scripting and integration, you can build workflows directly:

```python
from pathlib import Path
from oncoprep.workflows.base import init_oncoprep_wf

wf = init_oncoprep_wf(
    bids_dir=Path("bids_output"),
    output_dir=Path("derivatives"),
    subject_session_list=[("001", None)],
    work_dir=Path("work"),
    run_segmentation=True,
    default_seg=True,
)

# Run with 4 parallel processes
wf.run(plugin="MultiProc", plugin_args={"n_procs": 4})
```

### Running individual workflows

You can also run sub-workflows in isolation:

```python
from oncoprep.workflows.anatomical import init_anat_preproc_wf

anat_wf = init_anat_preproc_wf(
    bids_dir="/path/to/bids",
    output_dir="/path/to/derivatives",
    omp_nthreads=4,
    skull_strip_backend="ants",
)
anat_wf.run()
```

```python
from oncoprep.workflows.radiomics import init_anat_radiomics_wf

radio_wf = init_anat_radiomics_wf(
    output_dir="/path/to/derivatives",
    extract_shape=True,
    extract_firstorder=True,
    extract_glcm=True,
    extract_glrlm=False,
    extract_glszm=False,
    extract_gldm=False,
    extract_ngtdm=False,
)
radio_wf.run()
```

## End-to-End Reference

The following sections document **every feature** available through the
`oncoprep` CLI, from minimal invocations to fully-loaded commands
combining all pipeline stages.

### Minimal invocation (preprocessing only)

```bash
oncoprep /path/to/bids /path/to/derivatives participant \
  --participant-label 001
```

### Full-featured invocation (all features enabled)

This command runs the complete pipeline with every major feature flag:

```bash
oncoprep /path/to/bids /path/to/derivatives participant \
  --participant-label 001 002 \
  --session-label 01 02 \
  --bids-filter-file /path/to/filters.json \
  --subject-anatomical-reference first-lex \
  --output-spaces MNI152NLin2009cAsym \
  --skull-strip-template OASIS30ANTs \
  --skull-strip-backend ants \
  --skull-strip-mode auto \
  --skull-strip-fixed-seed \
  --registration-backend ants \
  --longitudinal \
  --deface \
  --run-segmentation \
  --run-radiomics \
  --run-qc \
  --container-runtime auto \
  --seg-cache-dir /path/to/seg_cache \
  --templateflow-home /path/to/templateflow \
  --work-dir /path/to/work \
  --nprocs 8 \
  --omp-nthreads 4 \
  --mem-gb 32 \
  --resource-monitor \
  --stop-on-first-crash \
  --write-graph \
  -vvv
```

### Feature-by-feature breakdown

#### BIDS filtering

Select specific participants, sessions, or apply custom pybids filters:

```bash
# Multiple participants
oncoprep /data/bids /data/out participant \
  --participant-label 001 002 003

# Specific sessions
oncoprep /data/bids /data/out participant \
  --participant-label 001 \
  --session-label 01 02

# Custom pybids filter file (JSON)
oncoprep /data/bids /data/out participant \
  --bids-filter-file my_filters.json

# Session-wise independent processing (each session treated separately)
oncoprep /data/bids /data/out participant \
  --participant-label 001 \
  --subject-anatomical-reference sessionwise
```

Example `my_filters.json`:
```json
{
  "t1w": {"datatype": "anat", "suffix": "T1w", "extension": ".nii.gz"},
  "flair": {"datatype": "anat", "suffix": "FLAIR", "extension": ".nii.gz"}
}
```

#### Skull-stripping options

Three backends and multiple control modes:

```bash
# ANTs brain extraction (default)
oncoprep ... --skull-strip-backend ants

# HD-BET GPU-accelerated (requires pip install "oncoprep[hd-bet]")
oncoprep ... --skull-strip-backend hdbet

# FreeSurfer SynthStrip
oncoprep ... --skull-strip-backend synthstrip

# Force skull-stripping even on pre-stripped inputs
oncoprep ... --skull-strip-mode force

# Skip skull-stripping entirely
oncoprep ... --skull-strip-mode skip

# Fixed seed for deterministic reproduction (combine with --omp-nthreads 1)
oncoprep ... --skull-strip-fixed-seed --omp-nthreads 1

# Use a different skull-stripping template
oncoprep ... --skull-strip-template OASIS30ANTs
```

#### Registration options

```bash
# ANTs SyN registration (default, more accurate)
oncoprep ... --registration-backend ants

# PICSL Greedy (faster)
oncoprep ... --registration-backend greedy

# Register to a custom output space
oncoprep ... --output-spaces MNI152NLin2009cAsym

# Longitudinal mode (builds unbiased within-subject template)
oncoprep ... --longitudinal
```

#### Segmentation options

```bash
# Full multi-model ensemble (GPU required, 14 BraTS models)
oncoprep /data/bids /data/out participant \
  --participant-label 001 \
  --run-segmentation

# Single default model (CPU-friendly, faster)
oncoprep /data/bids /data/out participant \
  --participant-label 001 \
  --run-segmentation --default-seg

# Custom model path
oncoprep /data/bids /data/out participant \
  --participant-label 001 \
  --run-segmentation --seg-model-path /path/to/model

# Force CPU-only (disable GPU)
oncoprep /data/bids /data/out participant \
  --participant-label 001 \
  --run-segmentation --no-gpu

# Use pre-cached model images (.sif or .tar)
oncoprep /data/bids /data/out participant \
  --participant-label 001 \
  --run-segmentation \
  --seg-cache-dir /path/to/seg_cache

# Choose container runtime explicitly
oncoprep /data/bids /data/out participant \
  --participant-label 001 \
  --run-segmentation \
  --container-runtime singularity
```

#### Radiomics feature extraction

```bash
# Run radiomics (implies --run-segmentation automatically)
oncoprep /data/bids /data/out participant \
  --participant-label 001 \
  --run-radiomics

# Radiomics with single default model (CPU, fast)
oncoprep /data/bids /data/out participant \
  --participant-label 001 \
  --run-radiomics --default-seg
```

#### Quality control with MRIQC

```bash
# Run MRIQC on raw BIDS data in parallel with preprocessing
oncoprep /data/bids /data/out participant \
  --participant-label 001 \
  --run-qc
```

IQM reports (SNR, CNR, CJV, EFC, FBER) and per-subject HTML reports are
written to `<output_dir>/mriqc/`.

#### Privacy (defacing)

```bash
# Remove facial features from anatomical images
oncoprep /data/bids /data/out participant \
  --participant-label 001 \
  --deface
```

#### TemplateFlow & offline mode

```bash
# Pre-fetch templates on a login node (for HPC)
oncoprep --fetch-templates \
  --templateflow-home /scratch/templateflow \
  --output-spaces MNI152NLin2009cAsym \
  --skull-strip-template OASIS30ANTs

# Run on a compute node without internet
oncoprep /data/bids /data/out participant \
  --participant-label 001 \
  --templateflow-home /scratch/templateflow \
  --offline
```

#### Performance tuning

```bash
# Parallel processing
oncoprep ... --nprocs 12 --omp-nthreads 4

# Memory limit
oncoprep ... --mem-gb 48

# Low-memory mode (trades disk I/O for RAM)
oncoprep ... --low-mem

# Custom Nipype plugin (YAML config for SGE, PBS, SLURM)
oncoprep ... --use-plugin my_plugin.yml

# Resource monitoring (memory + CPU tracking)
oncoprep ... --resource-monitor

# Debug verbosity (-v = verbose, -vv = more, -vvv = debug)
oncoprep ... -vvv
```

#### Reports & debugging

```bash
# Generate only HTML reports (skip processing)
oncoprep /data/bids /data/out participant \
  --participant-label 001 \
  --reports-only

# Include logs from a previous failed run in reports
oncoprep /data/bids /data/out participant \
  --participant-label 001 \
  --reports-only --run-uuid 20260210-143022_abc123

# Export workflow graph as SVG
oncoprep /data/bids /data/out participant \
  --participant-label 001 \
  --write-graph

# Stop immediately on first crash (for debugging)
oncoprep ... --stop-on-first-crash

# Generate boilerplate methods text only
oncoprep /data/bids /data/out participant \
  --participant-label 001 \
  --boilerplate

# Skip re-running existing derivatives
oncoprep ... --fast-track
```

### Docker invocations

#### Basic preprocessing in Docker

```bash
docker run --platform linux/amd64 --rm \
  -v /path/to/bids:/data/bids:ro \
  -v /path/to/output:/data/output \
  -v /path/to/work:/data/work \
  nko11/oncoprep:latest \
  /data/bids /data/output participant \
  --participant-label 001
```

#### Docker with all features (GPU + segmentation + radiomics + QC)

```bash
docker run --rm --gpus all \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /path/to/bids:/data/bids:ro \
  -v /path/to/output:/data/output \
  -v /path/to/work:/data/work \
  nko11/oncoprep:latest \
  /data/bids /data/output participant \
  --participant-label 001 002 \
  --session-label 01 \
  --run-segmentation \
  --run-radiomics \
  --run-qc \
  --deface \
  --nprocs 8 \
  --mem-gb 32 \
  --work-dir /data/work \
  -vv
```

### HPC / Singularity invocations

#### Full-featured Singularity run (PBS)

```bash
#!/bin/bash
#PBS -l ncpus=12,mem=48GB,walltime=04:00:00,jobfs=100GB
#PBS -l storage=gdata/$PROJECT+scratch/$PROJECT
#PBS -l ngpus=1
#PBS -l wd

module load singularity

SEG_CACHE=/scratch/$PROJECT/$USER/seg_cache
TF_HOME=/scratch/$PROJECT/$USER/templateflow

singularity run --nv \
  --bind $SEG_CACHE:/seg_cache \
  --bind $TF_HOME:/templateflow \
  --bind /scratch/$PROJECT/$USER/bids:/data/bids:ro \
  --bind /scratch/$PROJECT/$USER/derivatives:/data/output \
  --bind $PBS_JOBFS:/work \
  /scratch/$PROJECT/$USER/oncoprep.sif \
  /data/bids /data/output participant \
  --participant-label 001 \
  --run-segmentation \
  --run-radiomics \
  --run-qc \
  --deface \
  --container-runtime singularity \
  --seg-cache-dir /seg_cache \
  --templateflow-home /templateflow \
  --offline \
  --work-dir /work \
  --nprocs $PBS_NCPUS \
  --mem-gb 48 \
  --omp-nthreads 4 \
  --stop-on-first-crash \
  -vv
```

#### Full-featured Singularity run (SLURM)

```bash
#!/bin/bash
#SBATCH --job-name=oncoprep
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1

module load singularity

SEG_CACHE=/scratch/$USER/seg_cache
TF_HOME=/scratch/$USER/templateflow

singularity run --nv \
  --bind $SEG_CACHE:/seg_cache \
  --bind $TF_HOME:/templateflow \
  --bind /scratch/$USER/bids:/data/bids:ro \
  --bind /scratch/$USER/derivatives:/data/output \
  --bind $TMPDIR:/work \
  /scratch/$USER/oncoprep.sif \
  /data/bids /data/output participant \
  --participant-label 001 \
  --run-segmentation \
  --run-radiomics \
  --run-qc \
  --deface \
  --container-runtime singularity \
  --seg-cache-dir /seg_cache \
  --templateflow-home /templateflow \
  --offline \
  --work-dir /work \
  --nprocs $SLURM_CPUS_PER_TASK \
  --mem-gb 48 \
  --omp-nthreads 4 \
  -vv
```

### Python API: full-featured invocation

```python
from pathlib import Path
from oncoprep.workflows.base import init_oncoprep_wf

wf = init_oncoprep_wf(
    bids_dir=Path("/data/bids"),
    output_dir=Path("/data/derivatives"),
    subject_session_list=[("001", ["01", "02"]), ("002", None)],
    work_dir=Path("/data/work"),
    run_uuid="20260211-120000_manual",
    omp_nthreads=4,
    nprocs=12,
    mem_gb=48,
    skull_strip_template="OASIS30ANTs",
    skull_strip_fixed_seed=True,
    skull_strip_mode="auto",
    skull_strip_backend="ants",
    registration_backend="ants",
    longitudinal=False,
    output_spaces=["MNI152NLin2009cAsym"],
    use_gpu=True,
    deface=True,
    run_segmentation=True,
    run_radiomics=True,
    run_qc=True,
    default_seg=False,
    seg_model_path=None,
    sloppy=False,
    container_runtime="auto",
    seg_cache_dir=Path("/data/seg_cache"),
)

wf.run(plugin="MultiProc", plugin_args={"n_procs": 12})
```

### Output structure (all features enabled)

```
derivatives/
├── oncoprep/
│   ├── dataset_description.json
│   ├── logs/
│   │   └── CITATION.md
│   └── sub-001/
│       └── ses-01/
│           └── anat/
│               ├── sub-001_ses-01_desc-preproc_T1w.nii.gz
│               ├── sub-001_ses-01_desc-preproc_T1w.json
│               ├── sub-001_ses-01_desc-preproc_T1ce.nii.gz
│               ├── sub-001_ses-01_desc-preproc_T2w.nii.gz
│               ├── sub-001_ses-01_desc-preproc_FLAIR.nii.gz
│               ├── sub-001_ses-01_desc-brain_mask.nii.gz
│               ├── sub-001_ses-01_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz
│               ├── sub-001_ses-01_desc-tumor_dseg.nii.gz
│               ├── sub-001_ses-01_desc-radiomics_features.json
│               └── sub-001_ses-01_desc-defaced_T1w.nii.gz
└── mriqc/
    ├── sub-001_ses-01_T1w.html
    └── sub-001_ses-01_T1w.json
```

### Quick reference: all CLI flags

| Flag | Purpose | Default |
|------|---------|---------|
| `--participant-label` | Subject IDs | All subjects |
| `--session-label` | Session IDs | All sessions |
| `--bids-filter-file` | Custom pybids filters | None |
| `--subject-anatomical-reference` | `first-lex` / `unbiased` / `sessionwise` | `first-lex` |
| `--output-spaces` | Template space(s) | `MNI152NLin2009cAsym` |
| `--skull-strip-template` | Skull-strip atlas | `OASIS30ANTs` |
| `--skull-strip-backend` | `ants` / `hdbet` / `synthstrip` | `ants` |
| `--skull-strip-mode` | `auto` / `skip` / `force` | `auto` |
| `--skull-strip-fixed-seed` | Deterministic seed | Off |
| `--registration-backend` | `ants` / `greedy` | `ants` |
| `--longitudinal` | Multi-session template | Off |
| `--deface` | Remove facial features | Off |
| `--run-segmentation` | Enable tumor segmentation | Off |
| `--default-seg` | Single model (CPU) | Off |
| `--seg-model-path` | Custom model path | None |
| `--no-gpu` | Force CPU-only | Off |
| `--container-runtime` | `auto` / `docker` / `singularity` / `apptainer` | `auto` |
| `--seg-cache-dir` | Pre-downloaded model cache | Auto |
| `--run-radiomics` | Feature extraction | Off |
| `--run-qc` | MRIQC quality control | Off |
| `--nprocs` | CPU count | All available |
| `--omp-nthreads` | Threads per process | Auto |
| `--mem-gb` | Memory limit (GB) | Unlimited |
| `--low-mem` | Trade disk for memory | Off |
| `--use-plugin` | Custom Nipype plugin YAML | MultiProc |
| `--work-dir` | Working directory | `./work` |
| `--templateflow-home` | Template cache path | `$TEMPLATEFLOW_HOME` |
| `--offline` | Disable network access | Off |
| `--fast-track` | Reuse existing derivatives | Off |
| `--reports-only` | Skip processing | Off |
| `--run-uuid` | Previous run UUID for reports | None |
| `--write-graph` | Export DAG as SVG | Off |
| `--boilerplate` | Generate methods text | Off |
| `--stop-on-first-crash` | Abort on error | Off |
| `--resource-monitor` | Track CPU/memory | Off |
| `--notrack` | Disable telemetry | Off |
| `--sloppy` | Low-quality (testing only) | Off |
| `-v` / `-vv` / `-vvv` | Verbosity level | Standard |

## Troubleshooting

### Docker "permission denied"

Make sure your user is in the `docker` group or run with `sudo`:

```bash
sudo usermod -aG docker $USER
# Log out and back in
```

### "Illegal instruction" on Apple Silicon

The Docker image targets `linux/amd64`. On ARM Macs, AVX instructions may
fail under Rosetta emulation. Use the native pip install for local
development:

```bash
pip install -e ".[dev]"
```

### Out of memory during segmentation

Reduce parallel processes or increase available memory:

```bash
oncoprep ... --nprocs 2 --mem-gb 8 --low-mem
```
