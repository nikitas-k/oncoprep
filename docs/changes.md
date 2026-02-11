# Changelog

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
