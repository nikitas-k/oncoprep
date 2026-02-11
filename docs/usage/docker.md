# Docker Usage

OncoPrep is available as a Docker image with all neuroimaging dependencies
(ANTs, FSL, dcm2niix) pre-installed.

## Pull the image

```bash
docker pull nko11/oncoprep:latest
```

## Basic preprocessing

```bash
docker run --platform linux/amd64 --rm \
  -v /path/to/bids:/data/bids:ro \
  -v /path/to/output:/data/output \
  -v /path/to/work:/data/work \
  nko11/oncoprep:latest \
  /data/bids /data/output participant \
  --participant-label sub-001
```

## With segmentation

Segmentation launches Docker containers for each model, so the host Docker
socket must be mounted:

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

### GPU ensemble mode (Linux)

```bash
docker run --rm --gpus all \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /path/to/bids:/data/bids:ro \
  -v /path/to/output:/data/output \
  -v /path/to/work:/data/work \
  nko11/oncoprep:latest \
  /data/bids /data/output participant \
  --participant-label sub-001 --run-segmentation
```

## DICOM conversion

```bash
docker run --platform linux/amd64 --rm \
  -v /path/to/dicoms:/data/dicom:ro \
  -v /path/to/bids:/data/bids \
  --entrypoint oncoprep-convert \
  nko11/oncoprep:latest \
  /data/dicom /data/bids --subject 001
```

## Build from source

```bash
docker build --platform linux/amd64 -t oncoprep:latest .
```

:::{warning}
The image targets `linux/amd64`. On Apple Silicon Macs, Docker Desktop uses
Rosetta emulation automatically, but ANTs binaries may hit *Illegal
instruction* errors. Use the native `pip install` for local development on
ARM Macs.
:::
