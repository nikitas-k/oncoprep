# HPC / Singularity Deployment

On HPC systems where Docker is unavailable, OncoPrep supports
Singularity and Apptainer.

## 1. Build the SIF image

```bash
module load singularity
singularity pull oncoprep.sif docker://nko11/oncoprep:latest
```

## 2. Pre-download segmentation models

Segmentation models are Docker images that need to be converted to `.sif`
files. A helper script is included in the repository:

```bash
# Download the script
wget -O pull_seg_models.sh \
  https://raw.githubusercontent.com/nikitas-k/oncoprep/main/scripts/pull_seg_models.sh

# Pull all models
bash pull_seg_models.sh /scratch/$PROJECT/$USER/seg_cache

# CPU-only models (smaller, no GPU needed)
bash pull_seg_models.sh /scratch/$PROJECT/$USER/seg_cache --cpu-only
```

Or use the CLI:

```bash
oncoprep-models pull -o /scratch/$PROJECT/$USER/seg_cache \
  --runtime singularity
```

## 3. Run on a compute node

### GPU mode

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

### CPU-only (single model)

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

## 4. Direct Execution Mode (Recommended for HPC)

When running inside a Singularity container on HPC, nested container
invocation (singularity inside singularity) may not be supported. In this
case, use the `direct` runtime to execute segmentation models directly
from pre-extracted SIF files.

### Step 1: Pull and extract models (on login node)

**IMPORTANT:** This step must be done on the HPC **login node** (not inside
a container) where the `singularity` or `apptainer` command is available.

```bash
# Load singularity module (if needed)
module load singularity  # or: module load apptainer

# Pull models as SIF files
oncoprep-models pull -o /scratch/$PROJECT/$USER/seg_cache

# Extract SIF files for direct execution
# This uses 'singularity build --sandbox' to extract the container filesystem
oncoprep-models extract --cache-dir /scratch/$PROJECT/$USER/seg_cache
```

### Step 2: Run with direct runtime (in job script)

```bash
singularity run --nv \
  --bind /scratch/$PROJECT/$USER/seg_cache:/seg_cache \
  oncoprep.sif \
  /data/bids /data/bids/derivatives participant \
  --participant-label sub-001 \
  --run-segmentation \
  --container-runtime direct \
  --seg-cache-dir /seg_cache
```

The `direct` runtime:
- Runs model scripts directly from pre-extracted SIF filesystems
- Works when neither Docker nor nested Singularity is available
- Auto-detects when running inside a Singularity container
- Requires models to be pre-extracted on the login node

## PBS job script example

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
  --container-runtime direct \
  --seg-cache-dir /seg_cache \
  --work-dir /work
```

## SLURM job script example

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

singularity run --nv \
  --bind $SEG_CACHE:/seg_cache \
  --bind /scratch/$USER/bids:/data/bids:ro \
  --bind /scratch/$USER/derivatives:/data/derivatives \
  --bind $TMPDIR:/work \
  /scratch/$USER/oncoprep.sif \
  /data/bids /data/derivatives participant \
  --participant-label sub-001 \
  --run-segmentation \
  --container-runtime direct \
  --seg-cache-dir /seg_cache \
  --work-dir /work \
  --nprocs $SLURM_CPUS_PER_TASK
```
