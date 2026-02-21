# Changelog

## 0.2.4 (2026-02-21)

### Validation: Radiomic Feature Stability Analysis (D3)

- **New `validation/radiomics_stability.py` module** — quantitative
  assessment of micro-architectural signal fidelity across preprocessing
  architectures (native-first vs atlas-first).  Implements IBSI-compliant
  radiomic feature extraction comparison using:
  - **Coefficient of Variation (CV)**: per-feature normalised dispersion
    across subjects, $CV_i = (\sigma_i / \mu_i) \times 100\%$.
  - **ICC(3,1)**: two-way mixed-effects intra-class correlation to isolate
    variance introduced by the preprocessing architecture.
  - **Stability classification**: features with ICC ≥ 0.85 and CV ≤ 10%
    are classified as highly stable.
  - **Paired Wilcoxon signed-rank test** on CV distributions with
    Benjamini–Hochberg FDR correction for multiple comparisons.
- **New statistical utilities in `validation/stats.py`**:
  `coefficient_of_variation()`, `wilcoxon_signed_rank()`,
  `benjamini_hochberg()`.
- **Phase D extended with D3 endpoint** — `run_phase_d()` now accepts
  `comparator_dir` to trigger radiomics stability analysis.  Results are
  stored in `PhaseDResults.radiomics_stability` and serialised to a
  separate `radiomics_stability_<dataset>.json`.
- **Figure 6b** (`figure6b_radiomics_stability`): ICC vs CV scatter
  coloured by IBSI feature class, per-class stacked bar (stable vs
  unstable), and CV distribution box plots with Wilcoxon annotation.
- **Table 5** (`table5_radiomics_stability`): per-class summary with
  N features, % stable, median ICC, and median CV for both pipelines.
- Updated `PHASE_D_ENDPOINTS` in `validation/config.py` to formally
  describe D3.
- `run_all.py` now passes `comparator_dir` through to Phase D.

### Packaging

- **`vasari-auto` is now on PyPI** (`vasari-auto>=0.1.0`).  The `[vasari]`
  optional dependency now installs from PyPI instead of requiring a direct
  git URL.  `vasari-auto` is a fork of the
  [original by Ruffle et al. (2024)](https://doi.org/10.1016/j.nicl.2024.103668),
  maintained for OncoPrep integration.
- **`pyradiomics` dependency switched from git URL to PyPI** release
  (`pyradiomics>=3.0`), fixing PyPI upload rejection for direct references.
- Fixed `_pop()` in `workflows/outputs.py` — now handles plain strings
  (not just lists), preventing a `TraitError` when `CopyXForm` flattens
  single-element lists in the N4/ATROPOS sub-workflow.

### Bug Fixes

- Fixed `test_nninteractive::test_workflow_graph_is_connected` — updated
  expected node count from 6 to 5 after `resample_seg_to_std` was removed
  in the deferred registration refactor (0.2.3).

## 0.2.3 (2025-02-21)

### Pipeline Reordering: Deferred Template Registration

- **Template registration is now deferred until after segmentation** when
  `--run-segmentation` is enabled.  The whole-tumor mask (binarized, dilated
  by 4 voxels with a 6-connected structuring element) is passed to ANTs SyN
  as a cost-function exclusion mask (`-x`), preventing pathological tissue
  from distorting the diffeomorphic warp.  This preserves global geometry
  for atlas-based analyses (VASARI) and template-space derivatives.
- **Tumor segmentation no longer resamples to template space internally.**
  The `tumor_seg_std` output has been removed from both `init_anat_seg_wf()`
  and `init_nninteractive_seg_wf()`.  Template-space resampling is now
  performed in `base.py` after the deferred registration, using the
  topology-preserving transform.
- **VASARI now receives the template-space segmentation from the deferred
  registration block** (`resample_seg_to_std` in `base.py`) instead of from
  the segmentation workflow's `outputnode.tumor_seg_std`.
- **Radiomics operates on the native-space segmentation**, consistent with
  Pati et al. (*AJNR* 2024) — intensity normalization and SUSAN denoising
  are applied inside the radiomics workflow, after segmentation.
- Added `skip_registration` parameter to `init_anat_preproc_wf()` and
  `init_anat_fit_wf()`.  When `True`, Stage 5 (template registration) is
  skipped entirely, and buffer nodes emit empty lists for transforms.
- Added `_create_dilated_tumor_mask()` helper function in `base.py` for
  generating the cost-function exclusion mask.

### Pipeline Order (with segmentation enabled)

1. Anatomical preprocessing (N4, skull-strip, co-registration) — *no
   template registration*
2. Tumor segmentation (native space)
3. Radiomics feature extraction (native-space mask, with histogram
   normalization + SUSAN denoising)
4. Deferred template registration (ANTs SyN with dilated tumor mask as
   cost-function exclusion)
5. Resample tumor segmentation to template space (nearest-neighbor)
6. VASARI feature extraction (template-space mask)

## 0.2.2 (2025-02-19)

### Features

- **VASARI section in subject HTML report** — the master subject report
  (`sub-XXX.html`) now includes a "VASARI" section after Radiomics,
  containing the automated VASARI feature assessment table and the
  AI-generated radiology report.  Both are collated from figures-directory
  reportlets and conditionally rendered with a navbar link.
- **Group-level ComBat harmonization** — new `analysis_level = group` stage
  removes scanner/site batch effects from radiomics features across an entire
  cohort using neuroCombat (Fortin et al., *NeuroImage* 2018), following the
  methodology of Pati et al. (*AJNR* 2024).
  - Operates on participant-level radiomics JSON outputs; writes harmonized
    per-subject (or per-session) `*_desc-radiomicsCombat_features.json` files.
  - Auto-generates batch CSV from BIDS sidecars (`--generate-combat-batch`)
    using `Manufacturer`, `ManufacturerModelName`, and
    `MagneticFieldStrength` (fields that survive anonymization).
  - Extracts age and sex from JSON sidecars (`PatientAge`/`Age`,
    `PatientSex`/`Sex`) with fallback to `participants.tsv`; forwards them
    as biological covariates preserved by ComBat.
  - **Longitudinal auto-detection**: when multiple sessions per subject are
    found, each session is treated as a separate observation.  If subjects
    cross batches (scanned at different sites), subject identity is added as
    a categorical covariate; otherwise, sessions are treated as independent
    observations with within-subject variance naturally preserved.
  - Group-level HTML report (`group_combat_report.html`) with summary cards
    (observations, batches, features, variance change), batch distribution
    table, longitudinal mode indicator, and output file listing.
  - New CLI flags: `--combat-batch`, `--combat-parametric`,
    `--combat-nonparametric`, `--generate-combat-batch`.
  - New module: `oncoprep.workflows.group` with public API
    `run_group_analysis()` and `generate_combat_batch_csv()`.
- **SUSAN denoising** — pure-Python SUSAN non-linear noise reduction
  (`SUSANDenoising` interface) applied after histogram normalization in
  the radiomics workflow.  Edge-preserving smoothing reduces noise while
  maintaining tumor boundary detail.
- **Template-space tumor segmentation resampling** — the native-space
  tumor segmentation is resampled into the chosen template space (MNI152 or
  SRI24) using ANTs `ApplyTransforms` with nearest-neighbor interpolation.
  *Note: as of 0.2.3, this resampling moved from the segmentation workflows
  to `base.py` as part of the deferred registration block.*
- **VASARI atlas space selection** — `init_vasari_wf()` accepts an
  `atlas_space` parameter (`'mni152'`, `'MNI152NLin2009cAsym'`, or `'SRI24'`)
  and passes the pre-resampled segmentation directly to vasari-auto, skipping
  its internal ANTs SyN registration entirely.
- **Bundled atlas masks** — anatomical atlas ROI masks for MNI152 and SRI24
  are now shipped inside OncoPrep at `data/atlas_masks/{mni152,sri24}/`.
  Helper functions `get_atlas_dir()` and `get_atlas_reference()` resolve
  TemplateFlow-style space names to the correct local directory.

### vasari-auto upstream fixes

- Fixed `ATLAS_AFFINE` crash: vasari-auto loaded the MNI reference brain
  from a CWD-relative path (`atlas_masks/MNI152_T1_1mm_brain.nii.gz`) at
  module-import time, causing `FileNotFoundError` when run from any other
  directory.  Now resolved via `__file__`-relative path.
- Added `template_space` parameter to `get_vasari_features()` — accepted
  values: `'mni152'`, `'MNI152NLin2009cAsym'`, `'MNI152NLin6Asym'`,
  `'SRI24'`.  Defaults to `'mni152'`.
- Atlas masks are now organised into `atlas_masks/{mni152,sri24}/`
  subdirectories with per-space reference brains.
- `pyproject.toml` updated to include `atlas_masks/**/*.nii.gz` in the
  package distribution (previously missing from installs).
- `__init__.py` switched to lazy import via `__getattr__` to avoid loading
  heavy dependencies (ANTs, scipy, sklearn) at package-import time.
- `utils.py` `register_to_mni()` now auto-detects the reference brain
  filename (MNI152 or SRI24) in the atlas directory.

### Documentation

- New docs page: `usage/group_combat.md` — full reference for batch CSV
  format, age/sex extraction, longitudinal handling, output files, CLI
  flags, and Python API.
- Tutorial Step 5 added for group-level ComBat harmonization (quick start,
  custom batch CSV with biological covariates, longitudinal datasets,
  report inspection, Python API).
- README architecture diagram updated to show two-stage
  participant/group pipeline with group-level ComBat stage.
- README features table updated with ComBat harmonization and SUSAN
  denoising entries.
- CLI reference updated with group-level ComBat flags.
- Sphinx toctree updated to include `usage/group_combat`.

### Tests

- `TestGroupComBat` — 8 tests for cross-sectional ComBat (collection,
  filtering, Combat-file exclusion, harmonization, site-effect reduction,
  error handling, flatten/unflatten roundtrip).
- `TestLongitudinalComBat` — 5 tests for longitudinal ComBat (per-session
  collection, participant filtering, harmonization run, report content,
  cross-sectional report verification).
- `TestBatchCsvGeneration` — 4 tests for batch CSV generation (basic output,
  age/sex from sidecars, `participants.tsv` fallback, per-session rows).

### Bug Fixes

- **MPS crash in nnInteractive subprocess** — Apple Metal/MPS device
  initialization aborts when called inside a Nipype `forkserver` worker
  process (triggered by a macOS update).  Fixed via three layers:
  (1) `run_without_submitting=True` on the segmentation node keeps it in
  the main process where MPS works;
  (2) `_in_subprocess()` CPU-fallback safety net in the interface;
  (3) `--nprocs 1` automatically selects the `Linear` plugin.
- **VASARI radiology report DerivativesDataSink path error** — `Could not
  build path with entities` for `suffix='report'` with `.html`/`.txt`
  extensions.  Added `suffix<report>` to `_ONCOPREP_EXTRA_PATTERNS` in the
  figures BIDS pattern and set `datatype='figures'` on the radiology sinks.
- **Ambiguous radiology sink entities** — `ds_radiology_html` and
  `ds_radiology_txt` previously shared the same `desc='vasariRadiology'`;
  the text sink now uses `desc='vasariRadiologyText'` to prevent path
  collisions.
- Fixed `_import_vasari_auto()` in OncoPrep — no longer requires
  CWD manipulation or temporary symlinks; uses a straightforward import
  now that vasari-auto resolves paths correctly.

### Housekeeping

- Added VASARI references to `REFERENCES` string: Ruffle et al. 2024
  (VASARI-auto) and Gutman et al. 2013 (original VASARI feature set).
- Removed dead code: `src/oncoprep/reports/` package (unused `report.py`
  and `template.html.j2` — report generation uses `utils/collate.py`).
- Consolidated `base.py` imports at top of file, removed unused
  `init_anat_reports_wf` import — fixed all 11 ruff errors.
- Removed unused imports in `interfaces/nninteractive.py` (`Optional`)
  and `workflows/nninteractive.py` (`Path`).

## 0.2.1 (2025-02-18)

### Features

- **nnInteractive default segmentation** — the `--default-seg` flag now uses
  nnInteractive (Isensee et al., 2025; arXiv:2503.08373), a zero-shot 3D
  promptable foundation model trained on 120+ volumetric datasets.  No Docker
  containers are required; model weights (~400 MB) download automatically from
  HuggingFace on first use.
- Fully-automated seed-point generation via multi-modal anomaly scoring
  (T1ce enhancement × T2 anomaly × FLAIR hyperintensity) with adaptive
  percentile thresholding.
- Three-step segmentation: ET on T1ce, NCR via hole-filling, WT on FLAIR.
- White-matter negative prompt heuristic for reducing ET false positives.
- New documentation page: `usage/segmentation.md`.
- **Methods boilerplate** — `--boilerplate` now writes `CITATION.md` and
  `CITATION.html` to the logs directory and prints the full methods text to
  stdout.  Uses `LiterateWorkflow.visit_desc()` to recursively aggregate
  descriptions from every sub-workflow.
- **Separate references section** in HTML reports — references are rendered
  in their own `<div id="references">` section with a dedicated nav link,
  keeping the Methods section concise.
- `--reports-only` now builds the workflow graph (without heavy template
  fetching) so that `visit_desc()` produces the full boilerplate, then
  generates per-subject HTML reports with figures, methods, and references.
- Added workflow description (`__desc__`) to the nnInteractive segmentation
  workflow, citing Isensee et al. (2025).
- All `@citation_key` references used in workflow descriptions are now
  resolved to full bibliographic entries (13 references covering Nipype,
  ANTs, N4, HD-BET, SynthStrip, FSL FAST, TemplateFlow, PyRadiomics, and
  more).

### Temporarily Disabled

- **MRIQC integration** (`--run-qc`) is temporarily disabled in this release.
  The CLI flag is accepted but ignored with a deprecation warning.
  Tests for MRIQC have been removed. Integration will be re-enabled in a
  future version once upstream compatibility issues are resolved.

### Bug Fixes

- Fixed broken Nipype connections for the Docker-based segmentation path
  (`--default-seg` not set): `base.py` now conditionally wires preprocessed
  modalities (`t1w_preproc`, `t1ce_preproc`, etc.) to `init_anat_seg_wf` and
  raw BIDS images to `init_nninteractive_seg_wf`, instead of applying
  nnInteractive-style field names to both paths.
- Fixed stale `inputnode` field names (`t1w_preproc` → `t1w`, etc.) in the
  integration test `test_workflow_runs_end_to_end`.
- Moved `_pick_first` helper below all top-level imports in `base.py` to fix
  E402 (module-level import not at top of file).
- Removed `os.environ['PYTORCH_ENABLE_MPS_FALLBACK']` side-effect at
  `interfaces/nninteractive.py` import time; the variable is now set only
  inside `_init_session()`.
- Removed stale `brain_mask` entry from `init_nninteractive_seg_wf` docstring
  (the field is not part of the workflow's inputnode).
- Replaced unused `from nipype import Workflow` with
  `from niworkflows.engine.workflows import LiterateWorkflow as Workflow` in
  `workflows/nninteractive.py`.
- Fixed single-subject workflow description mentioning "functional data"
  (OncoPrep processes anatomical data only).
- Suppressed misleading "ANAT Stage X" logger messages during
  `--reports-only` and `--boilerplate` (no actual processing occurs).

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
