# Installation

## Requirements

- Python 3.9 or later
- [Docker](https://docs.docker.com/get-docker/) (for tumor segmentation)
- GPU with CUDA support (optional — required for ensemble segmentation mode)

## Quick install

```bash
pip install oncoprep
```

## Optional extras

### Radiomics

```bash
pip install "oncoprep[radiomics]"
```

### HD-BET skull stripping

```bash
pip install "oncoprep[hd-bet]"
```

### Development

```bash
pip install "oncoprep[dev]"
```

## From source

```bash
git clone https://github.com/nikitas-k/oncoprep.git
cd oncoprep
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Docker

If you prefer a fully self-contained environment with ANTs, FSL, and dcm2niix
pre-installed:

```bash
docker pull nko11/oncoprep:latest
```

See {doc}`usage/docker` for detailed Docker usage.

## Singularity / Apptainer (HPC)

```bash
module load singularity
singularity pull oncoprep.sif docker://nko11/oncoprep:latest
```

See {doc}`usage/hpc` for HPC deployment instructions.

## Verifying the installation

```bash
oncoprep --version
```
