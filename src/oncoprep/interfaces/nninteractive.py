"""nnInteractive promptable 3D segmentation interface for Nipype.

Wraps the **nnInteractive** model [1]_ to perform fully-automated
multi-compartment BraTS-style tumor segmentation using multi-modal anomaly
detection for automatic seed-point generation.

nnInteractive is a 3D open-set promptable foundation model trained on 120+
diverse volumetric datasets (CT, MRI, PET, microscopy).  It achieves
state-of-the-art *zero-shot* segmentation — the model has never seen glioma
or BraTS training data, yet produces clinically plausible delineations from
a handful of point and bounding-box prompts.  It won 1st place at the
CVPR 2025 Foundation Models for Interactive 3D Biomedical Image Segmentation
Challenge.

OncoPrep eliminates the need for interactive prompts by deriving seed points
automatically from multi-modal intensity anomalies:

  * **Enhancement map** (T1ce − T1w) isolates gadolinium uptake.
  * **T2 anomaly** flags voxels above the brain-wide T2 median.
  * **FLAIR anomaly** flags FLAIR hyperintensity.
  * The *product* of these three maps identifies the tumour core where all
    three signals overlap, virtually eliminating false positives.

The interface performs three sequential segmentation passes:

  1. **ET** — enhancing tumour on T1ce, guided by positive and negative
     (white-matter) point prompts plus an axial bounding box.
  2. **NCR** — necrotic core, derived by morphological hole-filling of the
     ET ring (no additional model call).
  3. **WT** — whole tumour on FLAIR, anchored by the anomaly centroid.

Edema (label 2) is assigned to WT voxels outside the tumour core.

Output labels follow the BraTS convention:
  1 = NCR (necrotic core — holes inside ET rim)
  2 = ED  (peritumoral edema — FLAIR hyperintensity outside tumour core)
  4 = ET  (enhancing tumour — bright on T1ce, filtered by enhancement map)

References
----------
.. [1] Isensee, F.*, Rokuss, M.*, Krämer, L.*, Dinkelacker, S.,
   Ravindran, A., Stritzke, F., Hamm, B., Wald, T., Langenberg, M.,
   Ulrich, C., Deissler, J., Floca, R., & Maier-Hein, K. (2025).
   nnInteractive: Redefining 3D Promptable Segmentation.
   *arXiv:2503.08373*. https://arxiv.org/abs/2503.08373
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
)

LOGGER = logging.getLogger('nipype.interface')

# Default HuggingFace repo for model weights
_NNINTERACTIVE_HF_REPO = 'nnInteractive/nnInteractive'
_NNINTERACTIVE_HF_PATTERN = 'nnInteractive_v1.0/*'


class _NNInteractiveSegmentationInputSpec(BaseInterfaceInputSpec):
    t1w = File(exists=True, mandatory=True, desc='T1w image (reference grid)')
    t1ce = File(exists=True, mandatory=True, desc='T1ce image')
    t2w = File(exists=True, mandatory=True, desc='T2w image')
    flair = File(exists=True, mandatory=True,
                 desc='FLAIR image (preferably preprocessed/registered to T1w space)')
    model_dir = traits.Directory(
        desc='Path to nnInteractive model weights directory. '
             'If not provided, downloads from HuggingFace to ~/.cache/oncoprep/nninteractive.',
    )
    device = traits.Str(
        'auto',
        usedefault=True,
        desc="Torch device: 'auto' (MPS > CUDA > CPU), 'cuda', 'mps', or 'cpu'",
    )


class _NNInteractiveSegmentationOutputSpec(TraitedSpec):
    tumor_seg = File(exists=True, desc='Multi-class tumor segmentation (BraTS labels 1/2/4)')


class NNInteractiveSegmentation(SimpleInterface):
    """Run nnInteractive promptable segmentation on multi-modal brain MRI.

    Uses SimpleITK for all image I/O so that numpy arrays are in the same
    (z, y, x) axis convention that nnInteractive was trained with — no
    transpose or axis-reordering required.

    Performs fully-automated tumor delineation in three steps:
      1. Detect tumor seed points from multi-modal intensity anomalies
      2. Segment enhancing tumor (ET) on T1ce, then derive necrosis (NET)
         from holes inside the ET ring
      3. Segment whole tumor (WT) on FLAIR, assign edema outside tumor core

    The output is a single NIfTI with BraTS labels (1=NCR, 2=ED, 4=ET).
    """

    input_spec = _NNInteractiveSegmentationInputSpec
    output_spec = _NNInteractiveSegmentationOutputSpec

    def _run_interface(self, runtime):
        import numpy as np
        import SimpleITK as sitk
        from scipy import ndimage

        # -- Load images via SimpleITK ----------------------------------
        # SimpleITK's GetArrayFromImage returns arrays in (z,y,x) order,
        # which is the native convention nnInteractive was trained with.
        # T1w defines the reference geometry; other modalities are
        # resampled to match if their grids differ.
        t1w_sitk = sitk.ReadImage(self.inputs.t1w)
        t1w_arr = sitk.GetArrayFromImage(t1w_sitk).astype(np.float32)
        ref_shape = t1w_arr.shape  # (z, y, x)

        t1ce_arr = self._conform(self.inputs.t1ce, t1w_sitk, 'T1ce')
        t2w_arr = self._conform(self.inputs.t2w, t1w_sitk, 'T2w')
        flair_arr = self._conform(self.inputs.flair, t1w_sitk, 'FLAIR')

        LOGGER.info('nnInteractive: reference shape (z,y,x) = %s', ref_shape)

        # Always use t1w > 0 as brain mask for seed detection.
        # This matches test_interactive.py and avoids the tighter
        # preprocessing brain mask changing normalization / WM detection
        # in ways that degrade seed quality.
        brain_mask = t1w_arr > 0

        # -- Normalize --------------------------------------------------
        def _norm(arr, mask):
            vals = arr[mask]
            mn, mx = np.percentile(vals, [1, 99])
            out = np.clip((arr - mn) / (mx - mn + 1e-8), 0, 1)
            out[~mask] = 0
            return out

        t1w_n = _norm(t1w_arr, brain_mask)
        t1ce_n = _norm(t1ce_arr, brain_mask)
        t2w_n = _norm(t2w_arr, brain_mask)
        flair_n = _norm(flair_arr, brain_mask)

        # -- Multi-modal anomaly detection ------------------------------
        enhancement = np.clip(t1ce_n - t1w_n, 0, None)
        enhancement[~brain_mask] = 0

        t2_med = np.median(t2w_n[brain_mask])
        t2_std = np.std(t2w_n[brain_mask])
        t2_anomaly = np.clip((t2w_n - t2_med) / (t2_std + 1e-8), 0, None)
        t2_anomaly[~brain_mask] = 0

        fl_med = np.median(flair_n[brain_mask])
        fl_std = np.std(flair_n[brain_mask])
        fl_anomaly = np.clip((flair_n - fl_med) / (fl_std + 1e-8), 0, None)
        fl_anomaly[~brain_mask] = 0

        combined_score = enhancement * t2_anomaly * fl_anomaly
        combined_score[~brain_mask] = 0
        combined_smooth = ndimage.gaussian_filter(combined_score, sigma=3)
        combined_smooth[~brain_mask] = 0

        # Adaptive thresholding -- find a blob of tumor-plausible size
        tumor_region = None
        for pct in [99, 97, 95, 93, 90, 85]:
            nonzero = combined_smooth[combined_smooth > 0]
            if len(nonzero) == 0:
                continue
            thr = np.percentile(nonzero, pct)
            am = combined_smooth > thr
            am = ndimage.binary_opening(am, iterations=1)
            am = ndimage.binary_closing(am, iterations=2)
            labs, nc = ndimage.label(am)
            if nc > 0:
                sizes = ndimage.sum(am, labs, range(1, nc + 1))
                biggest_idx = int(np.argmax(sizes)) + 1
                biggest_size = int(sizes[biggest_idx - 1])
                if 500 <= biggest_size <= 50000:
                    tumor_region = labs == biggest_idx
                    LOGGER.info(
                        'nnInteractive: anomaly detected (pct=%d, thr=%.3f, size=%d vox)',
                        pct, thr, biggest_size,
                    )
                    break

        if tumor_region is None:
            nonzero = combined_smooth[combined_smooth > 0]
            if len(nonzero) == 0:
                LOGGER.warning('nnInteractive: no anomaly detected -- returning empty seg')
                out_path = self._save_empty(t1w_sitk, ref_shape, runtime)
                self._results['tumor_seg'] = out_path
                return runtime
            thr = np.percentile(nonzero, 95)
            am = combined_smooth > thr
            am = ndimage.binary_opening(am, iterations=1)
            labs, nc = ndimage.label(am)
            if nc == 0:
                out_path = self._save_empty(t1w_sitk, ref_shape, runtime)
                self._results['tumor_seg'] = out_path
                return runtime
            sizes = ndimage.sum(am, labs, range(1, nc + 1))
            biggest_idx = int(np.argmax(sizes)) + 1
            tumor_region = labs == biggest_idx
            LOGGER.warning(
                'nnInteractive: fallback anomaly blob (%d vox)', int(sizes[biggest_idx - 1])
            )

        # -- Seed points ------------------------------------------------
        # ET seed: peak enhancement within anomaly
        et_score = enhancement.copy()
        et_score[~tumor_region] = 0
        et_score *= t1ce_n
        et_vals = et_score[tumor_region]
        et_thr = np.percentile(et_vals[et_vals > 0], 80) if np.sum(et_vals > 0) > 100 else 0
        et_submask = (et_score > et_thr) & tumor_region
        et_center = tuple(np.array(
            ndimage.center_of_mass(et_submask if np.any(et_submask) else tumor_region)
        ).astype(int))

        # WT seed: centroid of full anomaly
        wt_center = tuple(np.array(ndimage.center_of_mass(tumor_region)).astype(int))

        # Anomaly bounding box (with 15-vox margin)
        tumor_slices = ndimage.find_objects(tumor_region.astype(int))[0]
        anomaly_bbox = [
            [max(0, s.start - 15), min(tumor_region.shape[i], s.stop + 15)]
            for i, s in enumerate(tumor_slices)
        ]

        # WM negative prompt points (bright T1ce, low enhancement, near tumor)
        wm_like = (t1ce_n > 0.5) & (enhancement < 0.05) & brain_mask
        wm_like = ndimage.binary_erosion(wm_like, iterations=3)
        wm_near = wm_like.copy()
        wm_near[:max(0, anomaly_bbox[0][0] - 10), :, :] = False
        wm_near[anomaly_bbox[0][1] + 10:, :, :] = False
        wm_near[:, :max(0, anomaly_bbox[1][0] - 10), :] = False
        wm_near[:, anomaly_bbox[1][1] + 10:, :] = False
        wm_near[:, :, :max(0, anomaly_bbox[2][0] - 10)] = False
        wm_near[:, :, anomaly_bbox[2][1] + 10:] = False
        wm_near[tumor_region] = False

        wm_labs, wm_n = ndimage.label(wm_near)
        wm_neg_points = []
        if wm_n > 0:
            wm_sizes = ndimage.sum(wm_near, wm_labs, range(1, wm_n + 1))
            for idx in np.argsort(wm_sizes)[::-1][:3]:
                wm_neg_points.append(
                    tuple(np.array(ndimage.center_of_mass(wm_labs == (idx + 1))).astype(int))
                )

        LOGGER.info(
            'nnInteractive seeds -- ET: %s, WT: %s, WM neg: %s',
            et_center, wt_center, wm_neg_points,
        )

        # Free large intermediate arrays (keep flair_arr and t1ce_arr for inference)
        del t1w_arr, t2w_arr, t2w_n, flair_n
        del combined_score, combined_smooth, t2_anomaly, fl_anomaly

        # -- Initialize nnInteractive session ---------------------------
        model_dir = self._resolve_model_dir()
        session = self._init_session(model_dir)

        # -- STEP 1: Enhancing Tumor on T1ce ----------------------------
        LOGGER.info('nnInteractive STEP 1: ET segmentation on T1ce')

        et_prompts = [
            {'type': 'point', 'coord': et_center, 'include': True},
            {'type': 'bbox', 'bbox': self._axial_bbox(anomaly_bbox, et_center[0]), 'include': True},
        ]
        for wm_pt in wm_neg_points:
            et_prompts.append({'type': 'point', 'coord': wm_pt, 'include': False})

        et_mask_raw = self._run_prompts(session, t1ce_arr, et_prompts)

        # Enhancement-based post-processing
        enh_in_mask = enhancement[et_mask_raw > 0]
        if len(enh_in_mask) > 0 and np.sum(enh_in_mask > 0) > 50:
            enh_thr = np.median(enh_in_mask[enh_in_mask > 0])
            LOGGER.info('nnInteractive: enhancement threshold = %.3f', enh_thr)
        else:
            enh_thr = 0.05

        et_mask = et_mask_raw.copy()
        et_mask[enhancement < enh_thr] = 0

        # Spatial constraint: within dilated anomaly
        et_mask[~ndimage.binary_dilation(tumor_region, iterations=5)] = 0

        # Keep largest component
        et_mask = self._keep_largest(et_mask)
        LOGGER.info('nnInteractive: ET = %d voxels', np.sum(et_mask > 0))

        # -- STEP 2: Necrosis from ET hole-filling ----------------------
        LOGGER.info('nnInteractive STEP 2: NET derivation (hole-filling)')

        et_filled_3d = ndimage.binary_fill_holes(et_mask > 0)
        et_filled_slice = np.zeros_like(et_mask, dtype=bool)
        for z in range(et_mask.shape[0]):
            if np.any(et_mask[z] > 0):
                et_filled_slice[z] = ndimage.binary_fill_holes(et_mask[z] > 0)
        et_filled = et_filled_3d | et_filled_slice

        net_mask = (et_filled & ~(et_mask > 0)).astype(np.uint8)
        tc_mask = et_filled.astype(np.uint8)
        LOGGER.info(
            'nnInteractive: NET = %d voxels, TC = %d voxels',
            np.sum(net_mask > 0), np.sum(tc_mask > 0),
        )

        # -- STEP 3: Whole Tumor on FLAIR -------------------------------
        LOGGER.info('nnInteractive STEP 3: WT segmentation on FLAIR')

        wt_prompts = [
            {'type': 'point', 'coord': wt_center, 'include': True},
            {'type': 'bbox', 'bbox': self._axial_bbox(anomaly_bbox, wt_center[0]), 'include': True},
        ]
        wt_mask_raw = self._run_prompts(session, flair_arr, wt_prompts)

        # Spatial constraint: within generously dilated anomaly
        wt_mask = wt_mask_raw.copy()
        wt_mask[~ndimage.binary_dilation(tumor_region, iterations=12)] = 0
        wt_mask = self._keep_largest(wt_mask)
        LOGGER.info('nnInteractive: WT = %d voxels', np.sum(wt_mask > 0))

        # -- STEP 4: Combine into BraTS label map -----------------------
        combined = np.zeros(et_mask.shape, dtype=np.uint8)
        combined[wt_mask > 0] = 2   # Edema
        combined[net_mask > 0] = 1  # Necrosis
        combined[et_mask > 0] = 4   # Enhancing tumor
        # Ensure TC voxels outside WT are still labeled
        combined[(et_mask > 0) & (wt_mask == 0)] = 4
        combined[(net_mask > 0) & (wt_mask == 0)] = 1

        LOGGER.info(
            'nnInteractive combined: ET=%d, NCR=%d, ED=%d, WT=%d',
            np.sum(combined == 4),
            np.sum(combined == 1),
            np.sum(combined == 2),
            np.sum(combined > 0),
        )

        # -- Save via SimpleITK (preserves original NIfTI geometry) -----
        out_path = os.path.join(runtime.cwd, 'nninteractive_tumor_seg.nii.gz')
        combined_sitk = sitk.GetImageFromArray(combined)
        combined_sitk.CopyInformation(t1w_sitk)
        sitk.WriteImage(combined_sitk, out_path)
        self._results['tumor_seg'] = out_path

        return runtime

    # -- Private helpers ------------------------------------------------

    @staticmethod
    def _conform(img_path, ref_sitk, label):
        """Load an image via SimpleITK; resample to the reference grid if needed.

        SimpleITK's ``Resample`` uses the header geometry for alignment --
        no iterative registration, just spatial resampling.  This is
        appropriate for images from the same scanning session.
        """
        import numpy as np
        import SimpleITK as sitk

        img = sitk.ReadImage(img_path)
        if img.GetSize() == ref_sitk.GetSize():
            LOGGER.info('nnInteractive: %s conforms (%s)', label, img.GetSize())
            return sitk.GetArrayFromImage(img).astype(np.float32)

        LOGGER.info(
            'nnInteractive: %s size %s != ref %s -- resampling',
            label, img.GetSize(), ref_sitk.GetSize(),
        )
        resampled = sitk.Resample(
            img, ref_sitk,
            sitk.Transform(), sitk.sitkLinear, 0.0,
            img.GetPixelID(),
        )
        return sitk.GetArrayFromImage(resampled).astype(np.float32)

    @staticmethod
    def _save_empty(ref_sitk, ref_shape, runtime):
        """Save empty segmentation when no tumor is detected."""
        import numpy as np
        import SimpleITK as sitk

        empty = np.zeros(ref_shape, dtype=np.uint8)
        out_path = os.path.join(runtime.cwd, 'nninteractive_tumor_seg.nii.gz')
        empty_sitk = sitk.GetImageFromArray(empty)
        empty_sitk.CopyInformation(ref_sitk)
        sitk.WriteImage(empty_sitk, out_path)
        LOGGER.warning('nnInteractive: no tumor found -- saving empty segmentation')
        return out_path

    def _resolve_model_dir(self) -> str:
        """Resolve or download nnInteractive model weights."""
        if self.inputs.model_dir:
            d = str(self.inputs.model_dir)
            if os.path.isdir(d):
                return d

        # Check common cache locations
        for candidate in [
            '/tmp/nnInteractive_v1.0',
            os.path.expanduser('~/.cache/oncoprep/nninteractive/nnInteractive_v1.0'),
        ]:
            if os.path.isdir(candidate):
                LOGGER.info('nnInteractive model found at %s', candidate)
                return candidate

        # Download from HuggingFace
        LOGGER.info('Downloading nnInteractive model weights from HuggingFace ...')
        cache_dir = os.path.expanduser('~/.cache/oncoprep/nninteractive')
        os.makedirs(cache_dir, exist_ok=True)
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=_NNINTERACTIVE_HF_REPO,
            allow_patterns=[_NNINTERACTIVE_HF_PATTERN],
            local_dir=cache_dir,
        )
        model_dir = os.path.join(cache_dir, 'nnInteractive_v1.0')
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(
                f'nnInteractive model not found after download at {model_dir}'
            )
        return model_dir

    def _init_session(self, model_dir: str):
        """Initialize an nnInteractive inference session."""
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

        import torch
        from nnInteractive.inference.inference_session import nnInteractiveInferenceSession

        device_str = self.inputs.device
        if device_str == 'auto':
            if torch.cuda.is_available():
                device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')
        else:
            device = torch.device(device_str)

        LOGGER.info('nnInteractive device: %s', device)

        session = nnInteractiveInferenceSession(
            device=device,
            use_torch_compile=False,
            verbose=False,
            torch_n_threads=os.cpu_count() or 4,
            do_autozoom=True,
            use_pinned_memory=False,
        )
        session.initialize_from_trained_model_folder(model_dir)

        # MPS monkey-patch for _detect_change_at_border
        if device.type == 'mps':
            _orig = session._detect_change_at_border

            def _border_cpu(pred, prev_pred,
                            abs_th=1500, rel_th=0.2, min_th=100):
                saved = session.device
                session.device = torch.device('cpu')
                try:
                    return _orig(pred.cpu(), prev_pred.cpu(), abs_th, rel_th, min_th)
                finally:
                    session.device = saved

            session._detect_change_at_border = _border_cpu

        return session

    @staticmethod
    def _run_prompts(session, image_arr, prompts):
        """Run nnInteractive with a list of prompts on a single image."""
        import numpy as np
        import torch

        img_4d = image_arr[np.newaxis].copy()
        session.set_image(img_4d)
        target = torch.zeros(image_arr.shape, dtype=torch.uint8)
        session.set_target_buffer(target)

        for p in prompts:
            if p['type'] == 'point':
                coord = tuple(
                    min(max(c, 0), s - 1)
                    for c, s in zip(p['coord'], image_arr.shape)
                )
                session.add_point_interaction(coord, include_interaction=p.get('include', True))
            elif p['type'] == 'bbox':
                session.add_bbox_interaction(p['bbox'], include_interaction=p.get('include', True))

        result = session.target_buffer.clone().numpy()
        session.reset_interactions()
        return result

    @staticmethod
    def _axial_bbox(bbox_3d, center_z):
        """Convert 3D bbox to 2D axial bbox at a specific z slice."""
        return [[center_z, center_z + 1], bbox_3d[1], bbox_3d[2]]

    @staticmethod
    def _keep_largest(mask_arr):
        """Keep only the largest connected component."""
        import numpy as np
        from scipy import ndimage

        binary = mask_arr > 0
        binary = ndimage.binary_opening(binary, iterations=1)
        labs, nc = ndimage.label(binary)
        if nc > 1:
            sizes = ndimage.sum(binary, labs, range(1, nc + 1))
            biggest = int(np.argmax(sizes)) + 1
            return (labs == biggest).astype(np.uint8)
        return binary.astype(np.uint8)
