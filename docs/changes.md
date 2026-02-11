# Changelog

## 0.2.0 (2025-02-11)

### Features

- Integrated MRIQC for no-reference image quality control (`--run-qc` CLI flag)
- New `init_mriqc_wf()` and `init_mriqc_group_wf()` Nipype workflow factories
- New `MRIQC` and `MRIQCGroup` Nipype `CommandLine` interfaces
- Automatic QC gating with configurable IQM thresholds (SNR, CNR, CJV, EFC, FBER, QI1)
- Optional `mriqc` extra dependency (`pip install oncoprep[mriqc]`)

### Deprecations

- `oncoprep.workflows.metrics` module deprecated in favour of `oncoprep.workflows.mriqc`
  - `init_qa_metrics_wf()`, `init_snr_metrics_wf()`, `init_coverage_metrics_wf()`,
    `init_tissue_stats_wf()`, and `init_registration_quality_wf()` emit
    `DeprecationWarning` and return stub workflows
  - Will be removed in 0.3.0

### Changes

- CLI flag renamed from `--run-mriqc` to `--run-qc`
- Updated documentation: README, quickstart, tutorial, CLI reference, and API docs

## 0.1.0 (Unreleased)

Initial release.

### Features

- BIDS-compliant anatomical preprocessing (T1w, T1ce, T2w, FLAIR)
- Multi-model ensemble tumor segmentation (14 BraTS Docker models)
- Segmentation fusion (majority voting, SIMPLE, BraTS-specific)
- Radiomics feature extraction via PyRadiomics
- FreeSurfer surface processing with GIFTI/CIFTI-2 output *(planned)*
- DICOM → BIDS conversion (`oncoprep-convert`)
- fMRIPrep-style HTML quality-assurance reports
- Docker and Singularity/Apptainer support
- PBS and SLURM job script compatibility
