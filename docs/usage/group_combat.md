# Group-Level ComBat Harmonization

Multi-site and multi-scanner studies introduce systematic batch effects in
radiomics features that can confound downstream analyses.  OncoPrep provides
**group-level ComBat harmonization** — a cohort-wide post-processing step
that removes scanner/site effects from radiomics features while preserving
biological covariates of interest (age, sex, etc.).

This page covers the design, CLI flags, batch CSV format, longitudinal
support, and Python API.

## Background

ComBat (Johnson et al., *Biostatistics* 2007) was originally developed for
removing batch effects from microarray gene expression data.  Fortin et al.
(*NeuroImage* 2018) adapted it for neuroimaging, showing it effectively
harmonizes cortical thickness and other structural metrics across scanners
and sites.  Pati et al. (*AJNR* 2024) demonstrated its value for
reproducible tumor-habitat radiomics in neuro-oncology.

OncoPrep implements ComBat as a **group-level** (cohort-wide) step — not
per-subject.  This is essential because ComBat requires observations from
multiple scanner batches in a single matrix to estimate and remove the
batch effect.

### Two-Stage Architecture

```
Stage 1: participant-level
  oncoprep <bids> <out> participant --run-radiomics
  → per-subject radiomics JSON (no harmonization)

Stage 2: group-level
  oncoprep <bids> <out> group --generate-combat-batch
  → cohort-wide ComBat harmonization
  → per-subject harmonized JSON + HTML report
```

## CLI Usage

### Minimal: Auto-generate batch labels from BIDS metadata

```bash
oncoprep /path/to/bids /path/to/derivatives group \
  --generate-combat-batch
```

This will:

1. Scan BIDS JSON sidecars for `Manufacturer`, `ManufacturerModelName`,
   and `MagneticFieldStrength` (fields that survive anonymization)
2. Generate a batch CSV at `<output_dir>/oncoprep/combat_batch.csv`
3. Run ComBat harmonization across all participants

### Provide a custom batch CSV

```bash
oncoprep /path/to/bids /path/to/derivatives group \
  --combat-batch /path/to/site_labels.csv
```

### Subset participants

```bash
oncoprep /path/to/bids /path/to/derivatives group \
  --generate-combat-batch \
  --participant-label 001 002 003 004 005
```

### Non-parametric ComBat

```bash
oncoprep /path/to/bids /path/to/derivatives group \
  --generate-combat-batch \
  --combat-nonparametric
```

## Batch CSV Format

The batch CSV is the key input.  It maps observations (subjects or
subject × session) to scanner batches, and optionally provides biological
covariates.

### Cross-sectional (one scan per subject)

```csv
subject_id,batch,age,sex
sub-001,Siemens_Prisma_3T,45,M
sub-002,GE_SIGNA_15T,52,F
sub-003,Siemens_Prisma_3T,60,M
sub-004,GE_SIGNA_15T,38,F
```

### Longitudinal (multiple sessions per subject)

```csv
subject_id,batch,age,sex
sub-001_ses-01,Siemens_Prisma_3T,45,M
sub-001_ses-02,Siemens_Prisma_3T,46,M
sub-002_ses-01,GE_SIGNA_15T,52,F
sub-002_ses-02,GE_SIGNA_15T,53,F
```

### Column reference

| Column | Required | Description |
|--------|----------|-------------|
| `subject_id` | **Yes** | Subject label (`sub-001`) or observation label (`sub-001_ses-01`) |
| `batch` | **Yes** | Scanner/site identifier (e.g. `Siemens_Prisma_3T`) |
| `age` | No | Age (continuous covariate — preserved by ComBat) |
| `sex` | No | Sex (categorical covariate — preserved by ComBat) |
| Any other column | No | Additional covariates — string columns treated as categorical, numeric as continuous |

### Auto-generated columns

When using `--generate-combat-batch`, the CSV also contains these
informational columns (not used by ComBat itself):

| Column | Description |
|--------|-------------|
| `Manufacturer` | Scanner manufacturer (e.g. `Siemens`) |
| `ManufacturerModelName` | Scanner model (e.g. `Prisma`) |
| `MagneticFieldStrength` | Field strength in Tesla |

## Age and Sex Extraction

OncoPrep attempts to extract age and sex from two sources (in priority
order):

1. **JSON sidecar** — fields `PatientAge` / `Age`, `PatientSex` / `Sex`
2. **`participants.tsv`** at the BIDS root — columns `age`, `sex`

These are included in the auto-generated batch CSV when found, and
forwarded to ComBat as biological covariates of interest (preserved, not
removed).

:::{note}
Standard DICOM anonymization removes `PatientAge` and `PatientSex`.
If your data has been anonymized, add a `participants.tsv` to the BIDS
root with `age` and `sex` columns, or supply a custom batch CSV with
these columns.
:::

## Longitudinal Support

OncoPrep automatically detects longitudinal datasets (multiple sessions
per subject) and adjusts the ComBat strategy accordingly.

### How detection works

After collecting radiomics JSON files, OncoPrep checks whether the number
of unique subjects is less than the number of observations.  If so, the
dataset is longitudinal.

### Nested vs. crossing subjects

Two scenarios arise in longitudinal multi-site data:

**Subjects nested within batches** (common — each subject scanned at one
site across all sessions):

```
sub-001 → siteA  (ses-01, ses-02)
sub-002 → siteA  (ses-01, ses-02)
sub-003 → siteB  (ses-01, ses-02)
sub-004 → siteB  (ses-01, ses-02)
```

In this case, subject indicators are perfectly collinear with batch.
OncoPrep treats each session as an independent observation — the
within-subject variance is naturally preserved because all of a subject's
sessions share the same batch adjustment.

**Subjects crossing batches** (rare — a subject scanned at different sites
across sessions):

```
sub-001 → siteA (ses-01), siteB (ses-02)
sub-002 → siteB (ses-01), siteA (ses-02)
```

Here, OncoPrep adds subject identity as a categorical covariate to
neuroCombat, explicitly preserving within-subject variance while removing
batch effects.

:::{note}
There is no native Python implementation of longitudinal ComBat
(longCombat, Beer et al., *NeuroImage* 2020 — R only).  OncoPrep's
approach follows the standard recommendation for using neuroCombat with
repeated measures.
:::

### Per-session output files

In longitudinal mode, harmonized files are written next to each session's
original radiomics output:

```
derivatives/oncoprep/sub-001/ses-01/anat/
  sub-001_ses-01_desc-radiomics_features.json        ← original
  sub-001_ses-01_desc-radiomicsCombat_features.json   ← harmonized

derivatives/oncoprep/sub-001/ses-02/anat/
  sub-001_ses-02_desc-radiomics_features.json        ← original
  sub-001_ses-02_desc-radiomicsCombat_features.json   ← harmonized
```

## Output Files

| File | Description |
|------|-------------|
| `sub-XXX_desc-radiomicsCombat_features.json` | Per-subject (or per-session) harmonized features in the same nested JSON structure as the original |
| `combat_batch.csv` | Auto-generated batch CSV (when `--generate-combat-batch` is used) |
| `group_combat_report.html` | HTML report with summary statistics, batch distribution, variance reduction, and per-subject output listing |

## HTML Report

The group-level ComBat report includes:

- **Summary cards**: number of observations, scanner batches, features
  harmonized, and mean variance change
- **Unique subjects card** (longitudinal only)
- **Configuration table**: parametric vs. non-parametric, longitudinal
  mode, algorithm and implementation references
- **Biological covariates**: which covariates were preserved
- **Batch distribution**: per-batch observation counts
- **Output file listing**: every harmonized JSON file written

## CLI Reference

| Flag | Description | Default |
|------|-------------|---------|
| `--combat-batch CSV` | Custom batch CSV (columns: `subject_id`, `batch`, optional covariates) | None |
| `--combat-parametric` | Parametric empirical Bayes priors | `True` |
| `--combat-nonparametric` | Non-parametric empirical Bayes | `False` |
| `--generate-combat-batch` | Auto-generate batch CSV from BIDS sidecars | `False` |

## Python API

```python
from pathlib import Path
from oncoprep.workflows.group import (
    generate_combat_batch_csv,
    run_group_analysis,
)

# Auto-generate batch CSV
generate_combat_batch_csv(
    bids_dir=Path("/data/bids"),
    output_csv=Path("/data/derivatives/oncoprep/combat_batch.csv"),
)

# Run group-level ComBat
retcode = run_group_analysis(
    output_dir=Path("/data/derivatives"),
    bids_dir=Path("/data/bids"),
    generate_batch_csv=True,
    combat_parametric=True,
)
```

### Lower-level functions

```python
from oncoprep.workflows.group import (
    _collect_radiomics_jsons,
    _run_combat_harmonization,
    _flatten_features,
    _unflatten_features,
)

# Collect all per-observation radiomics JSONs
obs_files = _collect_radiomics_jsons(
    output_dir=Path("/data/derivatives"),
    participant_label=["sub-001", "sub-002"],
)

# Run ComBat directly
_run_combat_harmonization(
    output_dir=Path("/data/derivatives"),
    batch_file="/data/derivatives/oncoprep/combat_batch.csv",
    parametric=True,
)
```

## Requirements

ComBat harmonization requires the `radiomics` optional dependency group:

```bash
pip install "oncoprep[radiomics]"
```

This installs `neuroCombat>=0.2.12` along with PyRadiomics and other
radiomics dependencies.

## References

- Johnson, W. E., Li, C., & Rabinovic, A. (2007). Adjusting batch effects
  in microarray expression data using empirical Bayes methods.
  *Biostatistics*, 8(1), 118–127.
- Fortin, J.-P. et al. (2018). Harmonization of cortical thickness
  measurements across scanners and sites. *NeuroImage*, 167, 104–120.
- Pati, S. et al. (2024). Reproducibility of the tumor-habitat MRI
  biomarker DESMOND. *AJNR Am J Neuroradiol*, 45(9), 1291–1298.
- Beer, J. C. et al. (2020). Longitudinal ComBat: A method for
  harmonizing longitudinal multi-scanner imaging data. *NeuroImage*,
  220, 117129.
