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
