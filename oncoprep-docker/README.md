# oncoprep-docker

A lightweight Docker/Podman wrapper for the
[OncoPrep](https://github.com/nikitas-k/oncoprep) neuro-oncology MRI
preprocessing pipeline.

This package provides the `oncoprep-docker` command, which transparently
maps host paths into the container, handles GPU pass-through, TemplateFlow
caching, and forwards all unknown flags directly to the containerised
`oncoprep` entrypoint.

## Installation

```bash
pip install oncoprep-docker
```

## Requirements

- Python ≥ 3.9
- [Docker](https://docs.docker.com/get-docker/) or
  [Podman](https://podman.io/) installed and running

## Usage

Run `oncoprep-docker` exactly as you would run `oncoprep` on a bare-metal
installation:

```bash
oncoprep-docker /path/to/bids /path/to/output participant \
    --participant-label 001 --run-segmentation
```

For full usage information:

```bash
oncoprep-docker --help
```

## License

Apache 2.0 — see [LICENSE](https://github.com/nikitas-k/oncoprep/blob/main/LICENSE).
