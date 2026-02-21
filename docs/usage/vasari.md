# VASARI Feature Extraction & Radiology Reports

OncoPrep can automatically derive
[VASARI](https://wiki.cancerimagingarchive.net/display/Public/VASARI+Research+Project)
(Visually AcceSAble Rembrandt Images) MRI features from tumor segmentation
masks and generate structured radiology reports.

This functionality is powered by
[vasari-auto](https://github.com/nikitas-k/vasari-auto) (Ruffle et al.,
*NeuroImage: Clinical*, 2024), which computes 25 standardised VASARI features
from multi-label glioma segmentation masks without requiring source imaging
data.

Both OncoPrep and vasari-auto bundle anatomical atlas masks for MNI152 and
SRI24 template spaces.  vasari-auto accepts a `template_space` parameter
(default `'mni152'`) and resolves atlas paths from its own package data.
The tumor segmentation is automatically resampled from native T1w space into
the chosen template space (using the `anat2std_xfm` from the deferred
template registration) **before** being passed to vasari-auto, so no additional
ANTs SyN registration is performed at VASARI time.

:::{note}
**This is NOT a clinical tool.** Automated VASARI reports are intended for
research use only and should not be used for clinical decision-making.
:::

## Quick start

```bash
oncoprep /path/to/bids /path/to/derivatives participant \
  --participant-label 001 \
  --run-vasari
```

`--run-vasari` implies `--run-segmentation` — a tumor segmentation is required
to compute VASARI features.

By default the segmentation is resampled to **MNI152** space (matching the
first entry in `--output-spaces`).  Both MNI152 and SRI24 atlas spaces are
supported.

## What VASARI features are computed

VASARI-auto derives the following features from the segmentation mask by
evaluating its overlap with anatomical atlas ROIs in template space.  Because
Because OncoPrep resamples the segmentation into the template space during the
deferred registration block in `base.py`, vasari-auto skips its internal ANTs
SyN registration and uses the OncoPrep-bundled atlas masks directly:

| Feature | Description | Method |
|---------|-------------|--------|
| **F1** Tumour Location | Predominant anatomical region | Atlas ROI overlap (frontal, temporal, parietal, occipital, insula, brainstem, thalamus, corpus callosum) |
| **F2** Laterality | Right / Left / Bilateral | Voxel count per hemisphere |
| **F4** Enhancement Quality | None / Mild / Marked | Proportion of enhancing voxels |
| **F5** Proportion Enhancing | Categorical (≤5%, 6–33%, 34–67%, >68%) | Enhancing vs total lesion volume |
| **F6** Proportion nCET | Categorical (6 bins) | Non-enhancing vs total volume |
| **F7** Proportion Necrosis | Categorical | Necrosis proportion |
| **F9** Multifocality | Unifocal / Multifocal | Connected component analysis |
| **F11** Enhancing Margin | Thin / Thick / Solid | Skeletonisation ratio |
| **F14** Proportion Oedema | Categorical (4 bins) | Oedema vs total volume |
| **F19** Ependymal Invasion | Absent / Present | Overlap with ventricle atlas |
| **F20** Cortical Involvement | Absent / Present | Overlap with cortex atlas |
| **F21** Deep WM Invasion | Absent / Present | Overlap with brainstem / CC / IC |
| **F22** nCET Midline Crossing | Yes / No | nCET voxels in both hemispheres |
| **F23** CET Midline Crossing | Yes / No | CET voxels in both hemispheres |
| **F24** Satellites | Absent / Present | CET connected components |

Features requiring source imaging data (F3, F8, F10, F12, F13, F16–F18, F25)
are reported as **unsupported** and return `null`.

## Outputs

When `--run-vasari` is enabled, the following derivatives are produced for each
subject:

### VASARI features JSON

```
<output_dir>/sub-<label>/anat/sub-<label>_desc-vasari_features.json
```

A structured JSON file containing all 25 VASARI features with integer codes,
human-readable labels, and metadata:

```json
{
  "metadata": {
    "reporter": "VASARI-auto",
    "time_taken_seconds": 3.2,
    "software_note": "Automated VASARI features derived from tumor segmentation..."
  },
  "features": {
    "F1": {
      "name": "Tumour Location",
      "code": 1,
      "label": "Frontal Lobe"
    },
    "F2": {
      "name": "Side of Tumour Epicenter",
      "code": 1,
      "label": "Right"
    }
  }
}
```

### Radiology report (HTML + text)

```
<output_dir>/sub-<label>/anat/sub-<label>_desc-vasariRadiology_report.html
<output_dir>/sub-<label>/anat/sub-<label>_desc-vasariRadiology_report.txt
```

A structured radiology report summarising the VASARI assessment. Three report
styles are available via the Python API:

- **`structured`** (default) — tabular format with feature codes and
  descriptions
- **`narrative`** — prose-style report with clinical language
- **`brief`** — key findings only (location, enhancement, midline)

Example narrative excerpt:

> There is a mass lesion centred in the frontal lobe with right predominance.
> Enhancement quality is marked/avid, with the enhancing component comprising
> 6–33% of the lesion. The enhancing margin is thick (>3×). There is evidence
> of cortical involvement. [...]
>
> **Impression:** Right frontal lobe mass lesion, with marked/avid enhancement
> demonstrating deep white matter invasion. Automated VASARI assessment
> consistent with high-grade glioma characteristics. Clinical and
> histopathological correlation recommended.

### VASARI HTML report fragment

```
<output_dir>/sub-<label>/figures/sub-<label>_desc-vasari_features.html
```

An HTML fragment automatically included in the OncoPrep subject-level QA
report, showing a summary table of all VASARI features.

## Template-space resampling

Before VASARI feature extraction, the native-space tumor segmentation is
resampled into the chosen template space using ANTs `ApplyTransforms` with
**nearest-neighbor interpolation** (preserving discrete labels).  This step
happens in `base.py` as part of the **deferred template registration** block,
not inside the segmentation workflow.

When `--run-segmentation` is enabled, template registration is deferred until
after the tumor mask is available.  The whole-tumor mask is dilated by 4 voxels
and used as a cost-function exclusion region (`-x`) for ANTs SyN, preventing
pathological tissue from distorting the diffeomorphic warp.  The resulting
transform is then used to resample the native-space segmentation into the
template space.

The transform used is `anat2std_xfm` (native → template) produced by the
deferred registration workflow.  The reference image is the bundled
template brain for the chosen atlas space:

| Atlas Space | Reference Image |
|-------------|----------------|
| `mni152` / `MNI152NLin2009cAsym` / `MNI152NLin6Asym` | `MNI152_T1_1mm_brain.nii.gz` |
| `sri24` / `SRI24` | `MNI152_in_SRI24_T1_1mm_brain.nii.gz` |

The resampled segmentation is produced by the deferred registration block in
`base.py` and is passed directly to the VASARI workflow.

## Python API

```python
from oncoprep.workflows.vasari import init_vasari_wf

vasari_wf = init_vasari_wf(
    output_dir='/path/to/derivatives',
    atlas_space='mni152',         # 'mni152', 'MNI152NLin2009cAsym', or 'SRI24'
    report_template='narrative',  # 'structured', 'narrative', or 'brief'
)
```

### Workflow inputs

| Field | Description |
|-------|-------------|
| `source_file` | Source BIDS file for derivatives naming |
| `tumor_seg_std` | Multi-label segmentation **already in template space** (old BraTS: 1=NCR/nCET, 2=ED, 3=ET) |
| `subject_id` | Subject identifier for report header |

### Workflow outputs

| Field | Description |
|-------|-------------|
| `out_features` | Path to VASARI features JSON |
| `out_report` | Path to HTML report fragment |
| `out_radiology_report` | Path to HTML radiology report |
| `out_radiology_text` | Path to plain-text radiology report |

## Using the interface directly

For standalone feature extraction outside a Nipype workflow:

```python
from oncoprep.interfaces.vasari import (
    VASARIFeatureExtraction,
    get_atlas_dir,
    get_atlas_reference,
)

# Preferred: segmentation already in template space, use bundled atlases
vasari = VASARIFeatureExtraction(
    in_seg='/path/to/tumor_seg_std.nii.gz',  # already in MNI/SRI24
    atlas_dir=get_atlas_dir('mni152'),
)
result = vasari.run()
print(result.outputs.out_features)  # JSON path
print(result.outputs.out_report)    # HTML path

# Atlas utilities
print(get_atlas_dir('mni152'))      # path to MNI152 atlas masks directory
print(get_atlas_dir('SRI24'))       # path to SRI24 atlas masks directory
print(get_atlas_reference('mni152'))  # path to MNI152_T1_1mm_brain.nii.gz
```

:::{tip}
When the segmentation is already in template space, do **not** pass `in_anat`.
Setting `in_anat` causes vasari-auto to re-register the segmentation to MNI
with its own ANTs SyN, which is redundant and slow.
:::

For report generation from existing VASARI features:

```python
from oncoprep.interfaces.vasari import VASARIRadiologyReport

report = VASARIRadiologyReport(
    in_features='/path/to/vasari_features.json',
    patient_id='sub-001',
    template='narrative',  # or 'structured' or 'brief'
)
result = report.run()
print(result.outputs.out_report)  # HTML
print(result.outputs.out_text)    # plain text
```

## Atlas masks

OncoPrep bundles anatomical atlas masks for two template spaces under
`src/oncoprep/data/atlas_masks/`:

| Space | Directory | ROI masks |
|-------|-----------|----------|
| MNI152 | `atlas_masks/mni152/` | brainstem, corpus callosum, cortex, frontal/occipital/parietal/temporal lobes, insula, internal capsule, thalamus, ventricles |
| SRI24 | `atlas_masks/sri24/` | same ROIs in SRI24 coordinates |

Each directory also contains a reference brain image used by
`ApplyTransforms` when resampling segmentations into template space.

The `get_atlas_dir()` and `get_atlas_reference()` helpers in
`oncoprep.interfaces.vasari` resolve TemplateFlow-style space names
(e.g. `MNI152NLin2009cAsym`) to the bundled atlas directory.

## Requirements

VASARI feature extraction requires the `vasari-auto` package:

```bash
pip install "oncoprep[vasari]"
```

`vasari-auto` is a fork of the
[original by Ruffle et al. (2024)](https://doi.org/10.1016/j.nicl.2024.103668),
available on [PyPI](https://pypi.org/project/vasari-auto/).  It ships its own
atlas masks and resolves them from package data (no CWD dependency).  It
depends on `antspyx`, `nibabel`, `scipy`, `scikit-image`, `pandas`, and
`seaborn`.  When OncoPrep resamples the segmentation to template space
(via the deferred registration block),
the `antspyx` ANTs SyN registration step
inside vasari-auto is bypassed entirely.

### Using vasari-auto standalone

vasari-auto can also be used independently of OncoPrep:

```python
from vasari_auto.vasari_auto import get_vasari_features

# Default: MNI152 atlas, expects seg already in MNI space
df = get_vasari_features('tumor_seg_mni.nii.gz', template_space='mni152')

# SRI24 atlas
df = get_vasari_features('tumor_seg_sri24.nii.gz', template_space='SRI24')

# With anatomical image (triggers internal ANTs SyN registration)
df = get_vasari_features('tumor_seg_native.nii.gz', anat_img='t1w.nii.gz')
```

## References

- J. Ruffle et al., "VASARI-auto: Equitable, efficient, and economical
  featurisation of glioma MRI," *NeuroImage: Clinical*, 2024.
- VASARI Research Project, The Cancer Imaging Archive:
  <https://wiki.cancerimagingarchive.net/display/Public/VASARI+Research+Project>
