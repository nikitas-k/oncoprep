# OncoPrep — Pull Request: Tumor Segmentation Pipeline & Workflow Improvements

## Summary

This PR delivers the end-to-end tumor segmentation pipeline, refactors the base orchestration workflow, overhauls spatial normalization, and adds fMRIPrep-style HTML report collation. Together these changes bring OncoPrep from a preprocessing-only tool to a fully integrated preprocessing + segmentation pipeline.

**Diff stats:** 19 files changed, +2,649 / −905 lines

---

## New Features

### 1. Tumor Segmentation Workflow (`workflows/segment.py`)

Replaced the stub `build_segmentation_workflow()` with a production-ready `init_anat_seg_wf()` that follows nipreps conventions:

- **Three execution modes:**
  - `--default-seg` — single-model CPU-friendly segmentation using the econib (Marcinkiewicz, BraTS 2018) Docker container
  - `--run-segmentation` (without `--default-seg`) — multi-model ensemble using all available Docker images, fused with BraTS-specific SIMPLE algorithm
  - `--seg-model-path` — arbitrary custom container path
- **Sloppy/testing mode** returns a zero-filled dummy segmentation matching T1w dimensions — no Docker needed
- **Skull-stripping of inputs:** all modalities are masked with the brain mask before being passed to Docker containers, as required by BraTS models
- **ARM64/Apple Silicon support:** automatically adds `--platform linux/amd64` for QEMU x86 emulation when running on Apple Silicon
- **Cache-safe result handling:** segmentation results are copied to each node's own working directory so they survive Nipype cache invalidation of upstream nodes
- **Dual label conversion nodes:**
  - Old BraTS labels (2017–2020): 1=NCR, 2=ED, 3=ET, 4=RC
  - New derived labels (2021+): 1=ET, 2=TC, 3=WT, 4=NETC, 5=SNFH, 6=RC

### 2. Segmentation Fusion Workflow (`workflows/fusion.py`)

- Added `init_anat_seg_fuse_wf()` — a standalone fusion sub-workflow for multi-model ensemble integration
- Refactored `_fuse_segmentations()` to be fully self-contained (all helper functions inlined for Nipype Function-node serialization)
- Added `_convert_to_old_brats_labels()` and `_convert_to_new_brats_labels()` conversion functions
- Fusion methods: MAV (majority voting), SIMPLE (iterative DICE-based), BraTS-specific hierarchical fusion

### 3. BraTS Derivatives Output (`workflows/brats_outputs.py`)

- `init_ds_tumor_seg_wf()` now saves **three** segmentation variants via `DerivativesDataSink`:
  - Raw model output (`label-tumor`)
  - Old BraTS labels (`label-tumorOld`)
  - New derived labels (`label-tumorNew`)
- Added `_check_seg_file()` validation gate — raises `RuntimeError` for None/missing files so downstream nodes fail gracefully
- Switched from `niworkflows.interfaces.bids.DerivativesDataSink` to the project's own `..interfaces.DerivativesDataSink`

### 4. Segmentation Utility Module (`utils/segment.py` — new)

- `check_gpu_available()` — probes `nvidia-smi` then Docker `--gpus` fallback
- `check_docker_image()` / `pull_docker_image()` — local image inspection and registry pull
- `ensure_docker_images()` — batch availability check with automatic download for missing models
- `BRATS_OLD_LABELS` / `BRATS_NEW_LABELS` constants

### 5. Label Splitting Utility (`utils/labels.py` — new)

- `split_seg_labels()` — splits a multi-label NIfTI into per-label binary masks so each BraTS region gets its own contour color in reports

### 6. HTML Report Collation (`utils/collate.py` — new)

- `collate_subject_report()` — assembles all per-subject reportlets (SVGs, HTML fragments) into a single fMRIPrep-style XHTML report with Bootstrap 4 navbar
- Sections: Summary, Anatomical (conformance, tissue segmentation, spatial normalization per modality), Tumor Segmentation, About, Methods (boilerplate), Errors
- Automatically discovers figures under `<derivatives>/oncoprep/<sub>/figures/`

### 7. Tumor ROI Overlay Report (`interfaces/reports.py`)

- Added `TumorROIsPlot` — extends niworkflows' `ROIsPlot` with an inline SVG color legend for tumor segmentation regions (NCR, ED, ET, RC)
- Custom `_TumorROIsPlotInputSpec` with color, level, and legend-label traits

### 8. Greedy Registration Backend (`workflows/fit/registration.py`)

- Added `_greedy_registration()` — full PICSL Greedy diffeomorphic registration (affine → deformable → reslice) as an alternative to ANTs SyN
- `init_multimodal_template_registration_wf()` now accepts `registration_backend='ants'|'greedy'`
- Boilerplate description dynamically reflects the chosen backend

---

## Refactors & Improvements

### Base Orchestration (`workflows/base.py`)

- **CLI flag rename:** `--skip-segmentation` → `--run-segmentation` (opt-in instead of opt-out)
- Added `--seg-model-path` and `--default-seg` CLI arguments
- Integrated `init_anat_seg_wf()` + `init_ds_tumor_seg_wf()` + `TumorROIsPlot` into the single-subject workflow when `run_segmentation=True`
- Added report collation node (`collate_subject_report`) gated behind a sentinel merge so it runs only after all reportlets are written
- Removed imports of deleted `build_preproc_workflow`, `init_ds_mask_wf`, `init_ds_modalities_wf`, `init_ds_template_wf`, and metrics workflows

### Spatial Normalization (`workflows/fit/registration.py`)

- Replaced hand-rolled `ants.Registration` with niworkflows' `SpatialNormalization` (better parameter defaults, sloppy mode support)
- Added `TemplateDesc` + `TemplateFlowSelect` for proper template resolution via TemplateFlow
- Added `_set_reference()` to select T1w vs T2w template reference with correct histogram-matching settings
- Added `_fmt_cohort()` for cohort-aware template naming
- Added `_make_outputnode()` factory with `JoinNode` support for template iterables
- Apply-transform nodes use `_apply_transform_if_exists()` which gracefully handles None inputs (missing modalities)
- Rename nodes use `_rename_output_safe()` instead of `_rename_output()` to handle None paths
- Added `inputnode` fields for `t1w_mask` and `lesion_mask`

### Anatomical Workflow (`workflows/anatomical.py`)

- Added `t1w_brain` to outputnode (skull-stripped T1w for segmentation input)
- Added `tumor_dseg` to inputnode/outputnode for passing segmentation results back
- Renumbered stage logging: Stage 4a → Stage 4, Stage 4b → Stage 5, Stage 4.5 → Stage 6 (commented defacing block)

### Report Workflows (`workflows/outputs.py`)

- Each modality (T1w, T1ce, T2w, FLAIR) now gets its own spatial-normalization reportlet with dedicated `norm_msk_*`, `norm_rpt_*`, and `ds_std_*_report` nodes — previously all four modalities shared a single T1w reportlet
- T1ce report uses `acq='ce'` BIDS entity to distinguish from pre-contrast T1w
- Fixed `spaces._cached` attribute access → `spaces.references` 
- Replaced `init_multimodal_template_iterator_wf` → `init_template_iterator_wf`
- Removed FreeSurfer surface report section (not yet supported)

### Report Generation (`workflows/reports.py`)

- Set default values for optional inputnode fields (`anat_dseg`, `report_dict`, `bids_metadata`, `conversion_dict`) to prevent Nipype trait errors
- `_generate_report_metrics()` now handles `anat_dseg=None` gracefully — tissue volume statistics are skipped when no segmentation is available
- Fixed bare `LOGGER` reference → uses `get_logger(__name__)`

### Metrics (`workflows/metrics.py`)

- Set default `None` values for optional QA metrics inputs (`anat_dseg`, `anat2std_xfm`)

### TemplateFlow Interface (`interfaces/templateflow.py`)

- Added `t1ce_file` and `flair_file` output fields to `TemplateFlowSelect`

---

## Removed

- **`workflows/preproc.py`** — deleted entirely (−248 lines). The `build_preproc_workflow()` function was a wrapper around `init_anat_preproc_wf()` with redundant BIDS querying. This logic now lives in `init_single_subject_wf()` in `base.py`.
- **`conversion_recovery.py`** — deleted (−197 lines). Unused conversion recovery module.

---

## Configuration & Build

- **`pyproject.toml`**: Added `picsl-greedy` dependency for optional Greedy registration backend
- **`pytest.ini`** (new): Test configuration with markers (`slow`, `fast`, `integration`, `conversion`, `preprocessing`, `requires_dcm2niix`, `requires_nibabel`)
- **`.gitignore`**: Added `*.build`, `*.egg-info/`, `*work/`, `*.DS_Store`
- **`README.md`**: Added segmentation usage examples (ensemble, default model, custom model)
- **`.github/copilot-instructions.md`** (new): AI agent guidelines covering architecture, conventions, and integration points

---

## CLI Changes

| Old Flag | New Flag | Notes |
|----------|----------|-------|
| `--skip-segmentation` | `--run-segmentation` | Now opt-in (default: off) |
| — | `--default-seg` | Use single econib model (CPU-friendly) |
| — | `--seg-model-path PATH` | Custom Docker model path |

---

## Testing Notes

- The pipeline has been tested end-to-end with `--run-segmentation --default-seg` on the example BIDS dataset (`sub-001`, `sub-002`)
- Sloppy mode (`--sloppy`) creates zero-filled dummy segmentations without Docker
- ARM64 (Apple Silicon) x86 emulation works but is significantly slower than native x86
- Multi-model ensemble mode requires Docker with internet access for initial image pulls
