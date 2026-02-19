# Command-Line Interface

OncoPrep provides four CLI commands:

| Command | Purpose |
|---------|---------|
| `oncoprep` | Main preprocessing + segmentation pipeline |
| `oncoprep-convert` | DICOM → BIDS conversion |
| `oncoprep-docker` | Run OncoPrep inside a Docker container |
| `oncoprep-models` | Manage segmentation model images |

## `oncoprep`

The main pipeline command. Follows the
[BIDS-Apps](https://bids-apps.neuroimaging.io/) convention:

```bash
oncoprep <bids_dir> <output_dir> <analysis_level> [options]
```

### Positional arguments

| Argument | Description |
|----------|-------------|
| `bids_dir` | Root folder of a valid BIDS dataset |
| `output_dir` | Output path for derivatives and reports |
| `analysis_level` | `participant` (subject-level) or `group` (cohort-level ComBat) |

### Key options

#### BIDS filtering

```
--participant-label LABEL [LABEL ...]
    Subject identifiers (sub- prefix optional)

--session-label LABEL [LABEL ...]
    Session identifiers (ses- prefix optional)

--bids-filter-file PATH
    JSON file with custom pybids query filters
```

#### Performance

```
--nprocs N          Number of CPUs (default: all available)
--omp-nthreads N    Threads per process (default: auto)
--mem-gb N          Memory limit in GB
--low-mem           Trade disk for memory
```

#### Workflow configuration

```
--output-spaces SPACE [SPACE ...]
    Template spaces (default: MNI152NLin2009cAsym)

--skull-strip-backend {ants,hdbet,synthstrip}
    Skull-stripping method (default: ants)

--registration-backend {ants,greedy}
    Registration method (default: ants)

--deface
    Remove facial features for privacy
```

#### Segmentation

```
--run-segmentation    Enable tumor segmentation
--default-seg         Use single default model (CPU-friendly)
--seg-model-path PATH Custom model path
--no-gpu              Force CPU-only execution
--container-runtime {auto,docker,singularity,apptainer}
    Container runtime (default: auto)
--seg-cache-dir PATH  Pre-downloaded model cache directory
```

#### Radiomics

```
--run-radiomics       Enable feature extraction (implies --run-segmentation)
```

#### ComBat Harmonization (group-level)

These flags are used with `analysis_level = group`.  See
{doc}`group_combat` for a full guide.

```
--combat-batch CSV    CSV mapping subjects to scanner/site batches
                      (columns: subject_id, batch, optional covariates)

--combat-parametric   Use parametric empirical Bayes priors (default)

--combat-nonparametric
                      Use non-parametric empirical Bayes

--generate-combat-batch
                      Auto-generate batch CSV from BIDS JSON sidecars
                      (Manufacturer, ManufacturerModelName,
                      MagneticFieldStrength, age, sex)
```

#### VASARI

```
--run-vasari          Enable automated VASARI feature extraction and
                      radiology report generation
                      (requires vasari-auto; implies --run-segmentation)
```

#### Quality control (temporarily disabled)

```
--run-qc              [TEMPORARILY DISABLED] MRIQC quality control
                      integration is non-functional in this release.
                      This flag is accepted but ignored.
```

#### Other

```
--work-dir PATH       Working directory (default: ./work)
--reports-only        Generate reports without running workflows
--write-graph         Export workflow DAG as SVG
--stop-on-first-crash Abort on first error
-v / -vv / -vvv      Increase verbosity (debug: -vvv)
```

## `oncoprep-convert`

DICOM to BIDS conversion using `dcm2niix`:

```bash
oncoprep-convert <dicom_dir> <bids_dir> [options]
```

```
--subject ID        Subject identifier
--session ID        Session identifier
--batch             Process all subject directories in dicom_dir
```

## `oncoprep-docker`

Wrapper to run OncoPrep inside Docker with automatic volume binding and GPU
detection:

```bash
oncoprep-docker <bids_dir> <output_dir> participant [oncoprep options]
```

## `oncoprep-models`

Manage pre-downloaded segmentation model container images:

```bash
# List available models
oncoprep-models list

# Download all models
oncoprep-models pull -o /path/to/cache --runtime docker

# Download CPU-only models
oncoprep-models pull -o /path/to/cache --runtime docker --cpu-only

# Check download status
oncoprep-models status -o /path/to/cache
```
