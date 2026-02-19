# Installation

## Requirements

- Python 3.9 or later
- [ANTs](https://github.com/ANTsX/ANTs) (provides `DenoiseImage`, `ImageMath`, `antsRegistration`, etc.)
- [Docker](https://docs.docker.com/get-docker/) (for tumor segmentation)
- GPU with CUDA support (optional — required for ensemble segmentation mode)

ANTs is a **required** system dependency for anatomical preprocessing
(denoising, skull-stripping, registration). Install it via your system
package manager, an HPC module (`module load ants`), or
[build from source](https://github.com/ANTsX/ANTs/wiki/Compiling-ANTs-on-Linux-and-Mac-OS).
Ensure the ANTs `bin/` directory is on your `$PATH`.

## Quick install

```bash
pip install oncoprep
```

## Optional extras

### Radiomics

```bash
pip install "oncoprep[radiomics]"
```

### VASARI feature extraction

```bash
pip install "oncoprep[vasari]"
```

Requires `vasari-auto` from GitHub (no PyPI package). The extra installs
directly from `git+https://github.com/nikitas-k/vasari-auto.git`.

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
