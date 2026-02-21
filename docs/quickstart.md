# Quick Start

This page covers the three most common commands to get you up and running.

## 1. Convert DICOMs to BIDS

```bash
oncoprep-convert /path/to/dicoms /path/to/bids --subject 001
```

This converts raw DICOM directories into a
[BIDS-valid](https://bids.neuroimaging.io/) dataset using `dcm2niix`.

## 2. Run preprocessing

```bash
oncoprep /path/to/bids /path/to/derivatives participant \
  --participant-label sub-001
```

This runs the full anatomical preprocessing pipeline:

1. Validates the BIDS dataset
2. Collects T1w, T1ce, T2w, and FLAIR images
3. Conforms images to 1 mm isotropic resolution
4. Co-registers all modalities to the T1w reference
5. Skull-strips the brain
6. Normalizes to MNI152NLin2009cAsym template space
7. Writes BIDS derivative outputs

When segmentation is enabled (`--run-segmentation`), template registration
is **deferred** until after the tumor mask is available, so it can be used
as a cost-function exclusion mask for ANTs SyN. The pipeline becomes:

1. Anatomical preprocessing (conform, co-register, skull-strip, N4)
2. Tumor segmentation (native space)
3. Radiomics extraction (native-space mask, optional)
4. Template registration with dilated tumor mask as exclusion region
5. VASARI feature extraction (template-space mask, optional)
6. Writes BIDS derivative outputs

## 3. Run with quality control

> **Note:** MRIQC integration is temporarily disabled in this release.
> The `--run-qc` flag is accepted but ignored. It will be re-enabled in a
> future version.

```bash
# TEMPORARILY DISABLED — flag is accepted but has no effect
oncoprep /path/to/bids /path/to/derivatives participant \
  --participant-label sub-001 \
  --run-qc
```

## 4. Run with segmentation

```bash
# Default: nnInteractive zero-shot model (no Docker needed)
oncoprep /path/to/bids /path/to/derivatives participant \
  --participant-label sub-001 \
  --run-segmentation --default-seg

# Full Docker ensemble (GPU required)
oncoprep /path/to/bids /path/to/derivatives participant \
  --participant-label sub-001 \
  --run-segmentation
```

## Output structure

After processing, derivatives follow BIDS conventions:

```
derivatives/
└── oncoprep/
    └── sub-001/
        └── anat/
            ├── sub-001_desc-preproc_T1w.nii.gz
            ├── sub-001_desc-preproc_T1w.json
            ├── sub-001_desc-brain_mask.nii.gz
            ├── sub-001_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz
            └── sub-001_desc-tumor_dseg.nii.gz   # if segmentation was run
```

## Next steps

- Read the full {doc}`tutorial` for a worked example
- See {doc}`usage/cli` for all command-line options
- Check the {doc}`api/workflows` reference for Python API usage
