# Segmentation

OncoPrep supports two tumor segmentation backends:

1. **nnInteractive** (default) — a promptable 3D foundation model that runs
   natively in Python, no Docker required
2. **Docker ensemble** — 14 BraTS-challenge Docker containers with majority-vote
   fusion

## nnInteractive (default)

### Background

**nnInteractive** is an open-set 3D interactive segmentation model developed by
the German Cancer Research Center (DKFZ):

> Isensee, F.\*, Rokuss, M.\*, Krämer, L.\*, Dinkelacker, S., Ravindran, A.,
> Stritzke, F., Hamm, B., Wald, T., Langenberg, M., Ulrich, C., Deissler, J.,
> Floca, R., & Maier-Hein, K. (2025). *nnInteractive: Redefining 3D Promptable
> Segmentation.* arXiv:2503.08373. <https://arxiv.org/abs/2503.08373>

nnInteractive is the first comprehensive 3D interactive segmentation model
supporting diverse prompt types — points, scribbles, bounding boxes, and a
novel lasso prompt — while leveraging intuitive 2D interactions to produce full
3D segmentations.  Trained on **120+ diverse volumetric 3D datasets** spanning
CT, MRI, PET, and 3D microscopy, the model achieves state-of-the-art zero-shot
generalisation across modalities and anatomies it has never seen during training.

Key properties relevant to OncoPrep:

- **Zero-shot inference** — the model was *not* fine-tuned on glioma or BraTS
  data, yet produces clinically plausible delineations from just a few
  point/bounding-box prompts.
- **No Docker dependency** — runs as a native Python module with SimpleITK and
  PyTorch; model weights (~400 MB) are downloaded automatically from
  HuggingFace on first use.
- **Raw image input** — nnInteractive was explicitly trained on unprocessed
  images ("DO NOT preprocess ... give it to nnInteractive as it is"), so
  OncoPrep passes raw T1w, T1ce, and T2w directly from the BIDS source.
- **GPU optional** — runs on CUDA, Apple Silicon (MPS), or CPU.  A GPU is
  recommended; on CPU a single subject takes ~15–30 minutes.
- **CVPR 2025 winner** — nnInteractive won 1st place in the CVPR 2025
  Foundation Models for Interactive 3D Biomedical Image Segmentation Challenge.


### Usage

```bash
# Default segmentation with nnInteractive (no Docker needed)
oncoprep /path/to/bids /path/to/derivatives participant \
  --participant-label 001 \
  --run-segmentation --default-seg
```

Model weights are resolved in this order:

1. `--seg-model-path` CLI argument (if given)
2. `/tmp/nnInteractive_v1.0`
3. `~/.cache/oncoprep/nninteractive/nnInteractive_v1.0`
4. Automatic download from HuggingFace (`nnInteractive/nnInteractive`)


### Three-Step Segmentation

The interface performs fully-automated multi-compartment BraTS-style tumor
delineation in three sequential passes:

| Step | Modality | Target | Prompts |
|------|----------|--------|---------|
| 1 | T1ce | Enhancing tumor (ET) | Positive point + axial bbox + WM negative points |
| 2 | — (derived) | Necrotic core (NCR) | Morphological hole-filling of ET ring |
| 3 | FLAIR | Whole tumor (WT) | Positive point + axial bbox |

The final label map combines all three into the standard BraTS convention:

| Label | Region | Colour (ITK-SNAP) |
|-------|--------|-------------------|
| 1 | NCR — Necrotic Core | Red |
| 2 | ED — Peritumoral Edema | Green |
| 4 | ET — Enhancing Tumor | Yellow |


### Automated Seed Point Detection

Because nnInteractive is a *promptable* model (not fully automatic), it
requires spatial prompts — point coordinates and bounding boxes — to know
*where* and *what* to segment.  In a typical interactive session a human
operator provides these prompts through a GUI.  OncoPrep instead derives them
automatically from the multi-modal MRI intensities, enabling fully hands-free
pipeline execution.

#### Multi-Modal Anomaly Scoring

The seed detection algorithm fuses information from three modalities to
identify the tumour without any prior segmentation:

```
anomaly_score(v) = enhancement(v) × T2_anomaly(v) × FLAIR_anomaly(v)
```

where, for each brain voxel *v*:

- **Enhancement map** = `clip(T1ce_norm − T1w_norm, 0, ∞)` —
  gadolinium-enhancing tissue appears bright on T1ce but not on pre-contrast
  T1w.  The difference isolates contrast-enhancing tumour and suppresses normal
  white matter.

- **T2 anomaly** = `clip((T2w_norm − median) / std, 0, ∞)` —
  tumour tissue (both enhancing and necrotic) typically shows elevated T2
  signal relative to normal brain parenchyma.

- **FLAIR anomaly** = `clip((FLAIR_norm − median) / std, 0, ∞)` —
  peritumoral oedema and infiltrating tumour are hyperintense on FLAIR while
  CSF is suppressed, making FLAIR the standard clinical sequence for
  delineating the non-enhancing tumour extent.

All intensity maps are percentile-normalised (1st–99th) within these brain mask
before scoring.

**Rationale:** The *product* of three independent anomaly signals is
deliberately aggressive — a voxel must be simultaneously enhancing *and*
T2-bright *and* FLAIR-hyperintense to score highly.  This triple-conjunction
virtually eliminates false positives from normal tissue (white matter is not
T2-bright; CSF is not FLAIR-bright; cortex does not enhance) while reliably
highlighting the tumour core where all three pathologies overlap.

The score is Gaussian-smoothed (σ = 3 mm) to bridge small gaps and create a
contiguous tumour candidate region.

#### Adaptive Thresholding

A single fixed threshold cannot accommodate the wide range of tumour sizes,
enhancement strengths, and scanner contrasts encountered in clinical practice.
Instead, the algorithm searches downward through a set of decreasing
percentiles (99, 97, 95, 93, 90, 85) of the anomaly score, at each threshold
applying morphological opening + closing and connected-component analysis.  The
search stops as soon as the **largest connected component** falls within a
plausible tumour volume range (500–50 000 voxels ≈ 0.5–50 cm³ at 1 mm
isotropic).

This adaptive strategy means:

- **Large, strongly-enhancing tumours** are captured at the 99th percentile
  (tight threshold, small but confident blob).
- **Smaller or weakly-enhancing tumours** are captured at lower percentiles as
  the threshold relaxes.
- **No tumour** is gracefully handled: if no blob in the acceptable size range
  is found at any threshold, an empty segmentation is returned.

#### ET Seed Point

The enhancing-tumour seed is placed at the **centre of mass of the
highest-enhancement sub-region** within the anomaly blob:

1. Compute a voxel-wise "ET plausibility" score = `enhancement × T1ce_norm`
   within the anomaly region.
2. Threshold at the 80th percentile of non-zero values to isolate the brightest
   enhancement core.
3. Place the seed at the centroid of this bright core.

**Rationale:** Enhancement strength correlates with active tumour vascularity.
By restricting the seed to the brightest enhancement within the already-detected
anomaly, the positive prompt guides nnInteractive toward the most unambiguous
enhancing tumour tissue, maximising the chance that the initial segmentation
captures true ET rather than adjacent necrosis or oedema.

#### WT Seed Point

The whole-tumour seed is placed at the **centre of mass of the full anomaly
blob**, which represents the geometric centre of the combined tumour signal
across all modalities.

**Rationale:** The WT seed needs to anchor the FLAIR-based segmentation
roughly in the middle of the lesion.  Because the anomaly blob already
integrates enhancement, T2, and FLAIR signals, its centroid is a natural
summary location that avoids being biased toward any single compartment.

#### White-Matter Negative Prompts

nnInteractive supports *negative* prompts — points that tell the model "do not
include this".  Normal-appearing white matter (NAWM) near the tumour often has
similar T1ce intensity to enhancing tumour, so without negative guidance the
model may "leak" into NAWM.

Selection criteria for NAWM negative points:

1. **High T1ce intensity** (`T1ce_norm > 0.5`) — the voxel looks bright on
   contrast-enhanced imaging, mimicking enhancement.
2. **Low enhancement** (`enhancement < 0.05`) — despite being bright, there is
   negligible difference from pre-contrast T1w, ruling out true gadolinium
   uptake.
3. **Near the tumour** — restricted to within 10 voxels of the anomaly bounding
   box to provide locally relevant counter-examples.
4. **Morphologically eroded** (3 iterations) — ensures the selected WM region
   is deep within normal tissue, not at an ambiguous tumour boundary.
5. **Outside the anomaly** — explicitly excluded from the detected tumour
   region.

Up to three negative points (centroids of the three largest qualifying WM
clusters) are passed to nnInteractive alongside the positive ET prompt.

**Rationale:** Negative prompts are the most effective way to communicate
"specificity" to a promptable model.  By placing 2–3 negative points in bright
NAWM near the tumour, the model learns the local intensity boundary between
enhancing tumour and normal tissue, dramatically reducing false-positive ET
leakage into white matter tracts.

#### Axial Bounding Boxes

Each prompt set includes a 2D axial bounding box (required by nnInteractive's
pre-trained model for best results).  The box is derived from the anomaly
blob's 3D bounding box with a 15-voxel margin, collapsed to a single axial
slice at the seed point's z-coordinate.

**Rationale:** nnInteractive was trained with 2D bounding boxes (one dimension
= 1 slice) to enable the "3D from 2D" paradigm.  The axial projection
provides spatial extent information that a single-point prompt cannot convey,
helping the model estimate tumour size and shape from the first interaction.


### Post-Processing

#### Enhancement-Based ET Filtering

After the raw ET segmentation, voxels with enhancement below the median
enhancement of the segmented region are removed.  This filters out
false-positive inclusions (e.g., grey matter or vessels) that were captured by
the model but lack true contrast enhancement.  The result is additionally
constrained to within 5 voxels of the anomaly region and pruned to the largest
connected component.

#### NCR from Hole-Filling

Necrotic core is not segmented directly but derived via morphological
hole-filling of the ET mask:

1. Fill holes in the 3D ET volume.
2. Fill holes in each 2D axial slice of the ET mask.
3. Union the two filled volumes.
4. Subtract the original ET mask to obtain NCR.

**Rationale:** Necrosis in glioblastoma typically appears as a non-enhancing
region *enclosed* by the enhancing rim.  Hole-filling captures this "ring
topology" robustly without requiring a separate segmentation pass, and the
combination of 3D + slice-wise filling handles both thick and thin-wall
enhancement patterns.

#### WT Spatial Constraint

The FLAIR-based whole-tumour mask is constrained to within 12 voxels (dilated)
of the anomaly region and pruned to the largest connected component.  This
prevents the model from including distant FLAIR hyperintensities (e.g.,
periventricular caps, white matter lesions of ageing) that are unrelated to the
tumour.


### Image Input Strategy

| Modality | Source | Rationale |
|----------|--------|-----------|
| T1w | Raw BIDS | Reference grid; nnInteractive trained on raw |
| T1ce | Raw BIDS | Enhancement detection needs intact intensities |
| T2w | Raw BIDS | Same 1 mm grid as T1w, no resampling needed |
| FLAIR | Preprocessed (registered to T1w) | Raw FLAIR typically has thick slices (3 mm+) that produce comb artefacts after simple nearest-neighbour resampling; the ANTs registration in `anat_preproc_wf` produces a high-quality 1 mm FLAIR in T1w space |

The brain mask is always derived as `T1w > 0` rather than using the
preprocessing skull-strip mask.  This ensures the normalisation percentiles
used for seed detection are computed over the full head volume, matching the
conditions under which the heuristics were developed and validated.


### Python API

```python
from oncoprep.workflows.nninteractive import init_nninteractive_seg_wf

seg_wf = init_nninteractive_seg_wf(
    model_dir="/path/to/nnInteractive_v1.0",  # optional
    device="auto",                             # cuda > mps > cpu
)
seg_wf.inputs.inputnode.t1w = "sub-001_T1w.nii.gz"
seg_wf.inputs.inputnode.t1ce = "sub-001_ce-gadolinium_T1w.nii.gz"
seg_wf.inputs.inputnode.t2w = "sub-001_T2w.nii.gz"
seg_wf.inputs.inputnode.flair = "sub-001_desc-preproc_FLAIR.nii.gz"
seg_wf.run()
```


## Docker Ensemble

The original segmentation backend runs 14 BraTS-challenge Docker model
containers and fuses their predictions.  This requires Docker and (for
reasonable speed) a CUDA GPU.

```bash
# Full ensemble (GPU required, ~30–60 min)
oncoprep /path/to/bids /path/to/derivatives participant \
  --participant-label 001 \
  --run-segmentation
```

See {doc}`docker` for container runtime configuration and `oncoprep-models`
for managing model images.


## Template-Space Resampling

Both segmentation backends (nnInteractive and Docker ensemble) produce tumor
segmentation masks in the **native T1w space**.  When a downstream workflow
requires the segmentation in a standard template space (e.g. VASARI feature
extraction), the segmentation workflow automatically resamples the native-space
labels into the template using the `anat2std_xfm` transform from the
anatomical preprocessing workflow.

The resampling is performed with ANTs `ApplyTransforms` using
**nearest-neighbor interpolation** (to preserve discrete label values) and is
exposed on the segmentation workflow's `outputnode` as `tumor_seg_std`.

### Supported template spaces

| OncoPrep `--output-spaces` | Atlas Space | Reference Image |
|-----------------------------|-------------|-----------------|
| `MNI152NLin2009cAsym` | `mni152` | `MNI152_T1_1mm_brain.nii.gz` |
| `MNI152NLin6Asym` | `mni152` | `MNI152_T1_1mm_brain.nii.gz` |
| `SRI24` | `sri24` | `MNI152_in_SRI24_T1_1mm_brain.nii.gz` |

### Segmentation workflow outputs

| Field | Description |
|-------|-------------|
| `tumor_seg` | Multi-label segmentation in new BraTS convention (native space) |
| `tumor_seg_old` | Multi-label segmentation in old BraTS convention (native space) |
| `tumor_seg_std` | Multi-label segmentation in old BraTS convention, **resampled to template space** |
| `tumor_mask` | Binary whole-tumor mask (native space) |
