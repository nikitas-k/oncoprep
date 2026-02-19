# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2025 The OncoPrep Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Nipype interfaces for PyRadiomics feature extraction."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict

from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
    isdefined,
)


# ---------------------------------------------------------------------------
# BraTS label definitions
# ---------------------------------------------------------------------------

BRATS_OLD_LABEL_MAP = {
    1: 'NCR',   # Necrotic Core
    2: 'ED',    # Peritumoral Edema
    3: 'ET',    # Enhancing Tumor
    4: 'RC',    # Resection Cavity
}

BRATS_OLD_LABEL_NAMES = {
    1: 'Necrotic Core (NCR)',
    2: 'Peritumoral Edema (ED)',
    3: 'Enhancing Tumor (ET)',
    4: 'Resection Cavity (RC)',
}

# Composite regions derived from old labels
COMPOSITE_REGIONS = {
    'WT': {
        'name': 'Whole Tumor (WT)',
        'labels': [1, 2, 3, 4],
    },
    'TC': {
        'name': 'Tumor Core (TC)',
        'labels': [1, 3, 4],
    },
}


# ---------------------------------------------------------------------------
# Histogram Normalization Interface
# ---------------------------------------------------------------------------

class _HistogramNormInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True,
                   desc='Input image to normalize')
    in_mask = File(exists=True, mandatory=True,
                   desc='Brain or tumor mask defining the ROI for statistics')
    method = traits.Enum(
        'zscore', 'nyul', 'whitestripe',
        usedefault=True,
        desc='Normalization method: zscore (default), nyul, or whitestripe',
    )
    percentile_lower = traits.Float(
        1.0, usedefault=True,
        desc='Lower percentile for Winsorization / outlier clipping',
    )
    percentile_upper = traits.Float(
        99.0, usedefault=True,
        desc='Upper percentile for Winsorization / outlier clipping',
    )
    target_mean = traits.Float(
        0.0, usedefault=True,
        desc='Target mean for z-score normalization (default: 0)',
    )
    target_std = traits.Float(
        1.0, usedefault=True,
        desc='Target standard deviation for z-score normalization (default: 1)',
    )


class _HistogramNormOutputSpec(TraitedSpec):
    out_file = File(exists=True,
                    desc='Intensity-normalized image')


class HistogramNormalization(SimpleInterface):
    """Brain-masked intensity normalization for radiomics reproducibility.

    Standardizes image intensities prior to radiomics feature extraction
    following IBSI (Image Biomarker Standardisation Initiative) best practices.
    This reduces scanner- and protocol-dependent intensity variation that
    confounds texture and first-order features.

    Three methods are available:

    ``zscore`` (default)
        Compute mean and standard deviation within the brain mask, then
        transform: ``(x − μ) / σ × target_std + target_mean``.  Outlier
        intensities are Winsorized at ``percentile_lower`` and
        ``percentile_upper`` before computing statistics.  This is the
        simplest and most widely used approach in the radiomics literature
        [@shinohara2014; @um2019].

    ``nyul``
        Nyul–Udupa piecewise-linear histogram standardization.  Maps
        percentile landmarks of the brain-masked histogram to a standard
        scale.  Requires no training data when used subject-by-subject
        (internal landmark mapping).

    ``whitestripe``
        Estimates normal-appearing white matter peak via smoothed
        histogram mode and normalizes to that peak.  Robust for T1w images.
    """

    input_spec = _HistogramNormInputSpec
    output_spec = _HistogramNormOutputSpec

    def _run_interface(self, runtime):
        import nibabel as nib
        import numpy as np

        img = nib.load(self.inputs.in_file)
        data = np.array(img.get_fdata(), dtype=np.float64)

        mask_img = nib.load(self.inputs.in_mask)
        mask_data = np.asarray(mask_img.dataobj)

        # Ensure shapes match
        if mask_data.shape != data.shape:
            raise RuntimeError(
                f'Image shape {data.shape} does not match mask shape '
                f'{mask_data.shape}. Ensure inputs are in the same space.'
            )

        # Binarise multi-label masks (e.g. segmentation) before bool cast
        if mask_data.max() > 1:
            mask = mask_data > 0
        else:
            mask = mask_data.astype(bool)

        # Replace NaN/Inf with 0 to prevent propagation through statistics
        nan_mask = ~np.isfinite(data)
        if nan_mask.any():
            data[nan_mask] = 0.0
            # Also exclude non-finite voxels from the brain mask
            mask = mask & ~nan_mask

        method = self.inputs.method

        if method == 'zscore':
            data = self._zscore_normalize(data, mask)
        elif method == 'nyul':
            data = self._nyul_normalize(data, mask)
        elif method == 'whitestripe':
            data = self._whitestripe_normalize(data, mask)

        out_img = nib.Nifti1Image(data.astype(np.float32),
                                  img.affine, img.header)

        # Derive a unique output name from the input to avoid collisions
        # in parallel (multimodal) workflows sharing a working directory.
        in_name = Path(self.inputs.in_file).name
        # Strip .nii.gz / .nii in one step so e.g. 'sub-01_T1w.nii.gz' → 'sub-01_T1w'
        for ext in ('.nii.gz', '.nii'):
            if in_name.endswith(ext):
                in_stem = in_name[:-len(ext)]
                break
        else:
            in_stem = Path(in_name).stem
        out_path = os.path.join(runtime.cwd, f'{in_stem}_normalized.nii.gz')
        nib.save(out_img, out_path)
        self._results['out_file'] = out_path

        return runtime

    def _zscore_normalize(self, data, mask):
        """Z-score normalization with Winsorization.

        Only brain-masked voxels contribute to μ and σ.  The
        transformation is applied to masked voxels only; voxels outside
        the mask retain their original intensities so that downstream
        ROI extraction (e.g. tumor mask at brain boundary) is not
        affected by artificial zeroing.
        """
        import numpy as np

        brain_vals = data[mask]
        if brain_vals.size == 0:
            return data

        plow = np.percentile(brain_vals, self.inputs.percentile_lower)
        phigh = np.percentile(brain_vals, self.inputs.percentile_upper)

        # Winsorize to remove outliers before computing statistics
        clipped = np.clip(brain_vals, plow, phigh)
        mu = clipped.mean()
        sigma = clipped.std()

        if sigma < 1e-8:
            return data

        # Normalize in-mask voxels; leave non-brain intensities untouched
        # so that tumour ROI voxels at the brain-mask boundary are not
        # replaced with zeros.
        normed = data.copy()
        normed[mask] = (
            (data[mask] - mu) / sigma
        ) * self.inputs.target_std + self.inputs.target_mean
        return normed

    def _nyul_normalize(self, data, mask):
        """Nyul–Udupa piecewise-linear histogram standardization.

        Uses internal landmarks (deciles of the brain-masked histogram)
        and maps them to a standard scale [0, 100].
        """
        import numpy as np

        brain_vals = data[mask]
        if brain_vals.size == 0:
            return data

        # Compute decile landmarks from the input histogram
        landmarks_pct = np.arange(0, 101, 10)  # 0, 10, 20, ..., 100
        src_landmarks = np.percentile(brain_vals, landmarks_pct)
        tgt_landmarks = landmarks_pct.astype(np.float64)  # Map to [0, 100]

        # np.interp requires strictly increasing xp.  If brain values
        # are constant (or nearly so), multiple landmarks collapse to
        # the same value.  Deduplicate to keep only unique source values.
        unique_mask = np.concatenate(
            ([True], np.diff(src_landmarks) > 0)
        )
        src_landmarks = src_landmarks[unique_mask]
        tgt_landmarks = tgt_landmarks[unique_mask]

        if len(src_landmarks) < 2:
            # All brain voxels have the same intensity — nothing to map
            return data

        # Apply piecewise-linear mapping only to masked voxels
        normed = data.copy()
        normed[mask] = np.interp(data[mask], src_landmarks, tgt_landmarks)
        return normed

    def _whitestripe_normalize(self, data, mask):
        """WhiteStripe-like normalization using histogram mode estimation.

        Estimates the dominant tissue peak (typically WM for T1w) from a
        smoothed histogram of brain-masked intensities, then normalizes
        the entire volume so that peak equals 1.0.
        """
        import numpy as np

        brain_vals = data[mask]
        if brain_vals.size == 0:
            return data

        # Smoothed histogram for mode estimation
        plow = np.percentile(brain_vals, self.inputs.percentile_lower)
        phigh = np.percentile(brain_vals, self.inputs.percentile_upper)
        clipped = brain_vals[(brain_vals >= plow) & (brain_vals <= phigh)]

        if clipped.size < 2:
            # Not enough data to estimate a mode
            return data

        n_bins = min(256, max(64, int(np.sqrt(clipped.size))))
        hist, bin_edges = np.histogram(clipped, bins=n_bins)

        # Simple smoothing with uniform kernel
        kernel_size = max(3, n_bins // 20)
        kernel = np.ones(kernel_size) / kernel_size
        smoothed = np.convolve(hist.astype(float), kernel, mode='same')

        # Mode = bin centre of maximum count
        mode_idx = np.argmax(smoothed)
        mode_val = (bin_edges[mode_idx] + bin_edges[mode_idx + 1]) / 2.0

        if abs(mode_val) < 1e-8:
            return data

        # Normalize in-mask voxels only; preserve original outside the mask
        normed = data.copy()
        normed[mask] = data[mask] / mode_val
        return normed


# ---------------------------------------------------------------------------
# SUSAN Denoising Interface
# ---------------------------------------------------------------------------

class _SUSANDenoisingInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True,
                   desc='Input image to denoise')
    in_mask = File(exists=True, mandatory=True,
                   desc='Brain mask for estimating brightness threshold')
    fwhm = traits.Float(
        2.0, usedefault=True,
        desc='Full-width half-maximum of the Gaussian smoothing kernel in mm '
             '(default: 2.0, as recommended for radiomics preprocessing)',
    )
    brightness_threshold_pct = traits.Float(
        75.0, usedefault=True,
        desc='Percentile of brain-masked intensities to use as the SUSAN '
             'brightness threshold.  The default (75th percentile) produces '
             'edge-preserving smoothing that reduces noise while retaining '
             'tumor boundaries, following the approach described in '
             'Pati et al., AJNR 2024; 45: 1291–1298.',
    )


class _SUSANDenoisingOutputSpec(TraitedSpec):
    out_file = File(exists=True,
                    desc='Denoised image')


class SUSANDenoising(SimpleInterface):
    """SUSAN edge-preserving denoising for radiomics reproducibility.

    Applies FSL SUSAN (Smallest Univalue Segment Assimilating Nucleus)
    noise reduction to pre-normalised brain images.  SUSAN smooths
    homogeneous regions while preserving edges, making it ideal for
    pre-processing prior to texture-feature extraction.

    The brightness threshold is derived automatically from the brain-masked
    intensity distribution (default: 75th percentile), following the
    preprocessing pipeline described in:

        S. Pati et al., "Reproducibility of the Tumor-Habitat MRI
        Biomarker DESMOND," *AJNR Am J Neuroradiol*, vol. 45, no. 9,
        pp. 1291–1298, 2024.

    This interface is a pure-Python re-implementation that does **not**
    require FSL to be installed.  It uses a Gaussian-weighted local-mean
    filter that skips voxels whose intensity differs from the centre by
    more than the brightness threshold.
    """

    input_spec = _SUSANDenoisingInputSpec
    output_spec = _SUSANDenoisingOutputSpec

    def _run_interface(self, runtime):
        import nibabel as nib
        import numpy as np

        img = nib.load(self.inputs.in_file)
        data = np.array(img.get_fdata(), dtype=np.float64)

        mask_img = nib.load(self.inputs.in_mask)
        mask_data = np.asarray(mask_img.dataobj)

        if mask_data.shape != data.shape:
            raise RuntimeError(
                f'Image shape {data.shape} does not match mask shape '
                f'{mask_data.shape}. Ensure inputs are in the same space.'
            )

        # Binarise multi-label masks
        mask = mask_data > 0 if mask_data.max() > 1 else mask_data.astype(bool)

        # Derive brightness threshold from brain-masked intensities
        brain_vals = data[mask]
        if brain_vals.size == 0:
            # Nothing to denoise
            out_path = os.path.join(runtime.cwd, 'susan_denoised.nii.gz')
            nib.save(img, out_path)
            self._results['out_file'] = out_path
            return runtime

        bt = float(np.percentile(
            brain_vals, self.inputs.brightness_threshold_pct,
        ))

        fwhm = self.inputs.fwhm
        denoised = self._susan_smooth(data, mask, bt, fwhm, img.header.get_zooms()[:3])

        out_img = nib.Nifti1Image(
            denoised.astype(np.float32), img.affine, img.header,
        )

        # Derive unique output name from input
        in_name = Path(self.inputs.in_file).name
        for ext in ('.nii.gz', '.nii'):
            if in_name.endswith(ext):
                in_stem = in_name[:-len(ext)]
                break
        else:
            in_stem = Path(in_name).stem
        out_path = os.path.join(runtime.cwd, f'{in_stem}_susan.nii.gz')
        nib.save(out_img, out_path)
        self._results['out_file'] = out_path
        return runtime

    @staticmethod
    def _susan_smooth(data, mask, bt, fwhm, voxel_sizes):
        """Pure-Python SUSAN-style edge-preserving smoothing.

        Parameters
        ----------
        data : ndarray
            3-D image volume.
        mask : ndarray (bool)
            Brain mask.
        bt : float
            Brightness threshold — voxels with intensity difference
            greater than this from the centre voxel are excluded from
            the local weighted average.
        fwhm : float
            Gaussian kernel FWHM in mm.
        voxel_sizes : tuple of float
            Voxel dimensions in mm (used to convert *fwhm* to voxels).

        Returns
        -------
        ndarray
            Smoothed volume (unmasked voxels are left unchanged).
        """
        import numpy as np

        sigma_mm = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        voxel_sizes = np.asarray(voxel_sizes, dtype=np.float64)
        sigma_vox = sigma_mm / voxel_sizes  # per-axis sigma in voxels

        # Kernel half-width (3-sigma, at least 1 voxel)
        hw = np.maximum(np.ceil(3.0 * sigma_vox).astype(int), 1)

        result = data.copy()

        # Pre-compute Gaussian kernel weights (separable)
        kernels = []
        for ax in range(3):
            r = np.arange(-hw[ax], hw[ax] + 1, dtype=np.float64)
            k = np.exp(-0.5 * (r / max(sigma_vox[ax], 1e-8)) ** 2)
            kernels.append(k)

        # Build 3-D weight kernel
        kx, ky, kz = np.meshgrid(
            kernels[0], kernels[1], kernels[2], indexing='ij',
        )
        gauss_kernel = kx * ky * kz

        # Pad data for boundary handling
        pad_widths = [(h, h) for h in hw]
        padded = np.pad(data, pad_widths, mode='reflect')
        pad_mask = np.pad(mask, pad_widths, mode='constant', constant_values=False)

        # Iterate only over masked voxels
        coords = np.argwhere(mask)
        for idx in coords:
            i, j, k = idx
            # Extract local patch from padded volume
            pi, pj, pk = i + hw[0], j + hw[1], k + hw[2]
            patch = padded[
                pi - hw[0]:pi + hw[0] + 1,
                pj - hw[1]:pj + hw[1] + 1,
                pk - hw[2]:pk + hw[2] + 1,
            ]
            patch_mask = pad_mask[
                pi - hw[0]:pi + hw[0] + 1,
                pj - hw[1]:pj + hw[1] + 1,
                pk - hw[2]:pk + hw[2] + 1,
            ]

            centre_val = data[i, j, k]
            # Brightness gate: keep only voxels within bt of centre
            intensity_gate = np.abs(patch - centre_val) <= bt
            combined_gate = intensity_gate & patch_mask & (gauss_kernel > 0)

            if combined_gate.any():
                weights = gauss_kernel[combined_gate]
                values = patch[combined_gate]
                result[i, j, k] = np.dot(weights, values) / weights.sum()

        return result


# ---------------------------------------------------------------------------
# ComBat Harmonization Interface
# ---------------------------------------------------------------------------

class _ComBatHarmonizationInputSpec(BaseInterfaceInputSpec):
    in_features = File(
        exists=True, mandatory=True,
        desc='JSON file with extracted radiomics features '
             '(single-subject, output of PyRadiomicsFeatureExtraction)',
    )
    batch_file = File(
        exists=True,
        desc='CSV file mapping subjects to scanner/site batches. '
             'Must have columns: subject_id, batch.  '
             'Optional covariate columns (e.g. age, sex) are also '
             'passed to ComBat as biological covariates.',
    )
    subject_id = traits.Str(
        mandatory=True,
        desc='Current subject identifier (must match a row in batch_file)',
    )
    reference_features_dir = traits.Directory(
        desc='Directory containing per-subject radiomics JSON files '
             '(same schema as in_features) for the harmonization '
             'reference cohort.  All JSON files matching '
             '"*radiomics*.json" will be loaded.',
    )
    parametric = traits.Bool(
        True, usedefault=True,
        desc='Use parametric priors for ComBat (default: True). '
             'Set to False for non-parametric empirical Bayes.',
    )


class _ComBatHarmonizationOutputSpec(TraitedSpec):
    out_features = File(
        exists=True,
        desc='JSON file with ComBat-harmonized radiomics features',
    )
    out_report = File(
        exists=True,
        desc='HTML fragment summarising harmonization statistics',
    )


class ComBatHarmonization(SimpleInterface):
    """ComBat harmonization of radiomics features across scanner sites.

    Applies the ComBat batch-effect correction algorithm (Johnson et al.,
    Biostatistics 2007) to radiomics features to reduce inter-scanner
    variability while preserving biological signal.

    This follows the methodology described in:

        S. Pati et al., "Reproducibility of the Tumor-Habitat MRI
        Biomarker DESMOND," *AJNR Am J Neuroradiol*, vol. 45, no. 9,
        pp. 1291–1298, 2024.

    The implementation uses the *neuroCombat* library
    (Fortin et al., NeuroImage 2018).

    To harmonize a single subject's features, a reference cohort must
    be provided via ``reference_features_dir`` (directory of per-subject
    JSON files) and a ``batch_file`` (CSV mapping each subject to its
    scanner/site batch).
    """

    input_spec = _ComBatHarmonizationInputSpec
    output_spec = _ComBatHarmonizationOutputSpec

    def _run_interface(self, runtime):
        import numpy as np
        import pandas as pd

        # --- Load current subject features ---
        with open(self.inputs.in_features) as f:
            subj_features = json.load(f)

        # If no batch file provided, skip harmonization
        if not isdefined(self.inputs.batch_file) or not self.inputs.batch_file:
            out_json = os.path.abspath('radiomics_combat.json')
            with open(out_json, 'w') as f:
                json.dump(subj_features, f, indent=2, default=str)
            self._results['out_features'] = out_json

            out_html = os.path.abspath('combat_report.html')
            with open(out_html, 'w') as f:
                f.write(
                    '<div class="combat-report">'
                    '<p>ComBat harmonization skipped — '
                    'no batch file provided.</p></div>'
                )
            self._results['out_report'] = out_html
            return runtime

        try:
            from neuroCombat import neuroCombat
        except ImportError:
            raise ImportError(
                'neuroCombat is required for ComBat harmonization. '
                'Install with: pip install neuroCombat'
            )

        # --- Load batch info ---
        batch_df = pd.read_csv(self.inputs.batch_file)
        if 'subject_id' not in batch_df.columns or 'batch' not in batch_df.columns:
            raise ValueError(
                'batch_file must contain columns: subject_id, batch'
            )

        # --- Load reference cohort features ---
        ref_dir = self.inputs.reference_features_dir
        if not isdefined(ref_dir) or not ref_dir:
            raise ValueError(
                'reference_features_dir is required for ComBat harmonization'
            )

        ref_dir_path = Path(ref_dir)
        ref_files = sorted(ref_dir_path.glob('*radiomics*.json'))
        if len(ref_files) < 2:
            raise ValueError(
                f'Need at least 2 reference feature files in '
                f'{ref_dir}, found {len(ref_files)}'
            )

        # --- Build feature matrix ---
        # Flatten features from all subjects into a matrix
        all_subjects = {}  # subject_id → {feature_name: value}
        subject_id = self.inputs.subject_id

        # Add current subject
        flat_current = _flatten_features(subj_features)
        all_subjects[subject_id] = flat_current

        # Add reference subjects
        for ref_file in ref_files:
            ref_subj_id = ref_file.stem.split('_')[0]  # e.g. sub-001
            if ref_subj_id == subject_id:
                continue
            with open(ref_file) as f:
                ref_data = json.load(f)
            all_subjects[ref_subj_id] = _flatten_features(ref_data)

        # Build aligned feature matrix (subjects × features)
        subj_ids = list(all_subjects.keys())
        feature_names = sorted(flat_current.keys())

        # Only keep numeric features present in all subjects
        valid_features = []
        for fn in feature_names:
            try:
                vals = [float(all_subjects[s].get(fn, np.nan)) for s in subj_ids]
                if not any(np.isnan(v) for v in vals):
                    valid_features.append(fn)
            except (TypeError, ValueError):
                continue

        if not valid_features:
            # No features to harmonize
            out_json = os.path.abspath('radiomics_combat.json')
            with open(out_json, 'w') as f:
                json.dump(subj_features, f, indent=2, default=str)
            self._results['out_features'] = out_json

            out_html = os.path.abspath('combat_report.html')
            with open(out_html, 'w') as f:
                f.write(
                    '<div class="combat-report">'
                    '<p>ComBat harmonization skipped — '
                    'no valid numeric features found.</p></div>'
                )
            self._results['out_report'] = out_html
            return runtime

        # features × subjects matrix (neuroCombat expects this orientation)
        data_matrix = np.array([
            [float(all_subjects[s][fn]) for s in subj_ids]
            for fn in valid_features
        ])

        # Build batch vector aligned with subjects
        batch_dict = dict(zip(
            batch_df['subject_id'].astype(str),
            batch_df['batch'].astype(str),
        ))

        batch_vector = []
        keep_idx = []
        for i, sid in enumerate(subj_ids):
            # Try both with and without 'sub-' prefix
            b = batch_dict.get(sid) or batch_dict.get(sid.replace('sub-', ''))
            if b is not None:
                batch_vector.append(b)
                keep_idx.append(i)

        if len(set(batch_vector)) < 2:
            # Need at least 2 batches for ComBat
            out_json = os.path.abspath('radiomics_combat.json')
            with open(out_json, 'w') as f:
                json.dump(subj_features, f, indent=2, default=str)
            self._results['out_features'] = out_json

            out_html = os.path.abspath('combat_report.html')
            with open(out_html, 'w') as f:
                f.write(
                    '<div class="combat-report">'
                    '<p>ComBat harmonization skipped — '
                    'fewer than 2 scanner batches found.</p></div>'
                )
            self._results['out_report'] = out_html
            return runtime

        # Subset to subjects with batch info
        data_matrix = data_matrix[:, keep_idx]
        subj_ids_filtered = [subj_ids[i] for i in keep_idx]

        # Build covars DataFrame
        covars = pd.DataFrame({
            'batch': batch_vector,
        }, index=subj_ids_filtered)

        # Add optional biological covariates
        bio_cols = [c for c in batch_df.columns
                    if c not in ('subject_id', 'batch')]
        for col in bio_cols:
            col_dict = dict(zip(
                batch_df['subject_id'].astype(str),
                batch_df[col],
            ))
            covars[col] = [
                col_dict.get(sid, col_dict.get(sid.replace('sub-', ''), np.nan))
                for sid in subj_ids_filtered
            ]

        # Run ComBat
        combat_result = neuroCombat(
            dat=data_matrix,
            covars=covars,
            batch_col='batch',
            categorical_cols=[c for c in bio_cols
                              if batch_df[c].dtype == object],
            continuous_cols=[c for c in bio_cols
                            if batch_df[c].dtype != object],
        )
        harmonized = combat_result['data']  # features × subjects

        # Find column index for current subject
        try:
            subj_col = subj_ids_filtered.index(subject_id)
        except ValueError:
            # Subject not in batch file — return unharmonized
            out_json = os.path.abspath('radiomics_combat.json')
            with open(out_json, 'w') as f:
                json.dump(subj_features, f, indent=2, default=str)
            self._results['out_features'] = out_json

            out_html = os.path.abspath('combat_report.html')
            with open(out_html, 'w') as f:
                f.write(
                    '<div class="combat-report">'
                    '<p>ComBat harmonization: current subject '
                    'not found in batch file.</p></div>'
                )
            self._results['out_report'] = out_html
            return runtime

        # Replace features in the original structure
        harmonized_flat = {
            fn: float(harmonized[i, subj_col])
            for i, fn in enumerate(valid_features)
        }
        harmonized_features = _unflatten_features(
            subj_features, harmonized_flat,
        )

        # --- Write outputs ---
        out_json = os.path.abspath('radiomics_combat.json')
        with open(out_json, 'w') as f:
            json.dump(harmonized_features, f, indent=2, default=str)
        self._results['out_features'] = out_json

        # Report
        n_features = len(valid_features)
        n_subjects = data_matrix.shape[1]
        n_batches = len(set(batch_vector))

        out_html = os.path.abspath('combat_report.html')
        with open(out_html, 'w') as f:
            f.write(
                '<div class="combat-report">'
                '<h4>ComBat Harmonization Summary</h4>'
                '<table class="combat-summary">'
                '<tbody>'
                f'<tr><td>Features harmonized</td><td>{n_features}</td></tr>'
                f'<tr><td>Reference cohort size</td><td>{n_subjects}</td></tr>'
                f'<tr><td>Scanner batches</td><td>{n_batches}</td></tr>'
                f'<tr><td>Parametric</td><td>{self.inputs.parametric}</td></tr>'
                '</tbody></table></div>'
            )
        self._results['out_report'] = out_html
        return runtime


def _flatten_features(features_dict):
    """Flatten nested radiomics JSON to ``{region__category__feature: value}``.

    Parameters
    ----------
    features_dict : dict
        Nested dict as produced by ``PyRadiomicsFeatureExtraction``:
        ``{region: {features: {category: {feature: value}}}}``.

    Returns
    -------
    dict
        Flat ``{key: value}`` mapping.
    """
    flat = {}
    for region, rdata in features_dict.items():
        feats = rdata.get('features', {})
        for category, cat_feats in feats.items():
            for feat_name, feat_val in cat_feats.items():
                key = f'{region}__{category}__{feat_name}'
                try:
                    flat[key] = float(feat_val)
                except (TypeError, ValueError):
                    pass
    return flat


def _unflatten_features(original, harmonized_flat):
    """Replace values in nested features dict with harmonized values.

    Parameters
    ----------
    original : dict
        Original nested features dict.
    harmonized_flat : dict
        Flat ``{region__category__feature: value}`` from ComBat.

    Returns
    -------
    dict
        Updated nested dict with harmonized values.
    """
    import copy
    result = copy.deepcopy(original)
    for region, rdata in result.items():
        feats = rdata.get('features', {})
        for category, cat_feats in feats.items():
            for feat_name in list(cat_feats.keys()):
                key = f'{region}__{category}__{feat_name}'
                if key in harmonized_flat:
                    cat_feats[feat_name] = harmonized_flat[key]
    return result


# ---------------------------------------------------------------------------
# Input / Output Specs
# ---------------------------------------------------------------------------

class _PyRadiomicsInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True,
                   desc='Input image (e.g. preprocessed T1w)')
    in_mask = File(exists=True, mandatory=True,
                   desc='Tumor segmentation mask (multi-label)')
    label_map = traits.Dict(
        key_trait=traits.Int(),
        value_trait=traits.Str(),
        desc='Mapping of label integer → region abbreviation',
    )
    label_names = traits.Dict(
        key_trait=traits.Int(),
        value_trait=traits.Str(),
        desc='Mapping of label integer → human-readable name',
    )
    composites = traits.Dict(
        desc='Composite region definitions: {abbrev: {name, labels}}',
    )
    settings = traits.Dict(
        desc='PyRadiomics extraction settings (optional)',
    )
    extract_shape = traits.Bool(True, usedefault=True,
                                desc='Extract shape features')
    extract_firstorder = traits.Bool(True, usedefault=True,
                                     desc='Extract first-order features')
    extract_glcm = traits.Bool(True, usedefault=True,
                               desc='Extract GLCM texture features')
    extract_glrlm = traits.Bool(True, usedefault=True,
                                desc='Extract GLRLM texture features')
    extract_glszm = traits.Bool(True, usedefault=True,
                                desc='Extract GLSZM texture features')
    extract_gldm = traits.Bool(True, usedefault=True,
                               desc='Extract GLDM texture features')
    extract_ngtdm = traits.Bool(True, usedefault=True,
                                desc='Extract NGTDM texture features')


class _PyRadiomicsOutputSpec(TraitedSpec):
    out_features = File(exists=True,
                        desc='JSON file with extracted radiomics features')
    out_report = File(exists=True,
                      desc='HTML fragment with radiomics feature tables')


# ---------------------------------------------------------------------------
# Interface implementation
# ---------------------------------------------------------------------------

class PyRadiomicsFeatureExtraction(SimpleInterface):
    """Extract radiomics features from an image using a tumor segmentation mask.

    Uses the PyRadiomics library to compute shape, first-order, and texture
    features for each label in the segmentation mask plus user-defined
    composite regions (e.g. whole tumor, tumor core).

    The interface outputs:
    - A JSON file containing all extracted features keyed by region.
    - An HTML report fragment with styled feature tables for embedding
      in the OncoPrep subject report.
    """

    input_spec = _PyRadiomicsInputSpec
    output_spec = _PyRadiomicsOutputSpec

    def _run_interface(self, runtime):
        import nibabel as nib
        import numpy as np

        try:
            from radiomics import featureextractor
        except ImportError:
            raise ImportError(
                'pyradiomics is required for radiomics feature extraction. '
                'Install with: pip install pyradiomics'
            )

        in_file = self.inputs.in_file
        in_mask = self.inputs.in_mask

        # Label definitions
        label_map = (
            dict(self.inputs.label_map)
            if isdefined(self.inputs.label_map)
            else dict(BRATS_OLD_LABEL_MAP)
        )
        label_names = (
            dict(self.inputs.label_names)
            if isdefined(self.inputs.label_names)
            else dict(BRATS_OLD_LABEL_NAMES)
        )
        composites = (
            dict(self.inputs.composites)
            if isdefined(self.inputs.composites)
            else dict(COMPOSITE_REGIONS)
        )

        # Build extractor settings
        settings = {}
        if isdefined(self.inputs.settings) and self.inputs.settings:
            settings = dict(self.inputs.settings)

        # Determine which feature classes to enable
        feature_classes = []
        if self.inputs.extract_shape:
            feature_classes.append('shape')
        if self.inputs.extract_firstorder:
            feature_classes.append('firstorder')
        if self.inputs.extract_glcm:
            feature_classes.append('glcm')
        if self.inputs.extract_glrlm:
            feature_classes.append('glrlm')
        if self.inputs.extract_glszm:
            feature_classes.append('glszm')
        if self.inputs.extract_gldm:
            feature_classes.append('gldm')
        if self.inputs.extract_ngtdm:
            feature_classes.append('ngtdm')

        # Initialize extractor
        extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
        extractor.disableAllFeatures()
        for fc in feature_classes:
            extractor.enableFeatureClassByName(fc)

        # Load mask to discover available labels
        mask_img = nib.load(in_mask)
        mask_data = np.asarray(mask_img.dataobj)
        available_labels = set(np.unique(mask_data).astype(int)) - {0}

        all_features = {}

        # --- Per-label extraction ---
        for label_int, abbrev in sorted(label_map.items()):
            if label_int not in available_labels:
                continue
            region_name = label_names.get(label_int, abbrev)
            try:
                result = extractor.execute(in_file, in_mask, label=label_int)
                features = _parse_radiomics_result(result)
                all_features[abbrev] = {
                    'label': label_int,
                    'name': region_name,
                    'features': features,
                }
            except Exception as exc:
                all_features[abbrev] = {
                    'label': label_int,
                    'name': region_name,
                    'features': {},
                    'error': str(exc),
                }

        # --- Composite region extraction ---
        for comp_abbrev, comp_def in composites.items():
            comp_labels = comp_def['labels']
            comp_name = comp_def['name']
            # Create binary mask for composite region
            comp_mask_data = np.zeros_like(mask_data, dtype=np.uint8)
            for lbl in comp_labels:
                comp_mask_data[mask_data == lbl] = 1

            if comp_mask_data.sum() == 0:
                continue

            comp_mask_path = os.path.abspath(f'composite_{comp_abbrev}.nii.gz')
            comp_mask_img = nib.Nifti1Image(comp_mask_data, mask_img.affine,
                                            mask_img.header)
            nib.save(comp_mask_img, comp_mask_path)

            try:
                result = extractor.execute(in_file, comp_mask_path, label=1)
                features = _parse_radiomics_result(result)
                all_features[comp_abbrev] = {
                    'label': comp_labels,
                    'name': comp_name,
                    'features': features,
                }
            except Exception as exc:
                all_features[comp_abbrev] = {
                    'label': comp_labels,
                    'name': comp_name,
                    'features': {},
                    'error': str(exc),
                }

        # --- Write JSON ---
        out_json = os.path.abspath('radiomics_features.json')
        with open(out_json, 'w') as f:
            json.dump(all_features, f, indent=2, default=str)
        self._results['out_features'] = out_json

        # --- Write HTML report fragment ---
        out_html = os.path.abspath('radiomics_report.html')
        html = _render_radiomics_html(all_features)
        with open(out_html, 'w') as f:
            f.write(html)
        self._results['out_report'] = out_html

        return runtime


# ---------------------------------------------------------------------------
# Helpers (module-level so Nipype Function nodes can use them too)
# ---------------------------------------------------------------------------

def _parse_radiomics_result(result: dict) -> Dict[str, Dict[str, float]]:
    """Parse pyradiomics result dict into categorised feature dict.

    Parameters
    ----------
    result : dict
        Raw output from ``extractor.execute()``.

    Returns
    -------
    dict
        ``{category: {feature_name: value}}`` where *category* is e.g.
        ``'shape'``, ``'firstorder'``, ``'glcm'``.
    """
    features = {}  # type: Dict[str, Dict[str, float]]
    for key, val in result.items():
        # Skip diagnostic / metadata keys
        if key.startswith('diagnostics_') or key.startswith('general_'):
            continue
        # Keys look like: "original_shape_Elongation"
        parts = key.split('_', 2)
        if len(parts) < 3:
            continue
        _image_type, category, feat_name = parts
        category_lower = category.lower()
        if category_lower not in features:
            features[category_lower] = {}
        # Convert numpy types to float
        try:
            features[category_lower][feat_name] = float(val)
        except (TypeError, ValueError):
            features[category_lower][feat_name] = str(val)
    return features


def _render_radiomics_html(all_features: dict) -> str:
    """Render radiomics features as an HTML fragment with collapsible tables.

    Parameters
    ----------
    all_features : dict
        Output from ``PyRadiomicsFeatureExtraction``, keyed by region
        abbreviation.

    Returns
    -------
    str
        HTML fragment string.
    """
    # Category display order & pretty names
    CATEGORY_NAMES = {
        'shape': 'Shape',
        'firstorder': 'First Order',
        'glcm': 'GLCM',
        'glrlm': 'GLRLM',
        'glszm': 'GLSZM',
        'gldm': 'GLDM',
        'ngtdm': 'NGTDM',
    }

    lines = []
    lines.append('<div class="radiomics-report">')

    # --- Summary table: key metrics across all regions ---
    lines.append('<h4>Summary</h4>')
    lines.append(
        '<table class="radiomics-summary">'
        '<thead><tr>'
        '<th>Region</th>'
        '<th>Volume (mm³)</th>'
        '<th>Surface Area (mm²)</th>'
        '<th>Sphericity</th>'
        '<th>Mean Intensity</th>'
        '<th>Std Dev</th>'
        '</tr></thead><tbody>'
    )

    for abbrev, region_data in all_features.items():
        name = region_data.get('name', abbrev)
        feats = region_data.get('features', {})
        error = region_data.get('error')

        if error:
            lines.append(
                f'<tr><td>{name}</td>'
                f'<td colspan="5" style="color:#c0392b;">Error: {error}</td>'
                '</tr>'
            )
            continue

        shape = feats.get('shape', {})
        fo = feats.get('firstorder', {})

        vol = shape.get('MeshVolume', shape.get('VoxelVolume', '—'))
        sa = shape.get('SurfaceArea', '—')
        sph = shape.get('Sphericity', '—')
        mean_i = fo.get('Mean', '—')
        std_i = fo.get('StandardDeviation', '—')

        def _fmt(v):
            if isinstance(v, (int, float)):
                return f'{v:.4f}' if abs(v) < 1e4 else f'{v:.2e}'
            return str(v)

        lines.append(
            f'<tr><td><strong>{name}</strong></td>'
            f'<td>{_fmt(vol)}</td>'
            f'<td>{_fmt(sa)}</td>'
            f'<td>{_fmt(sph)}</td>'
            f'<td>{_fmt(mean_i)}</td>'
            f'<td>{_fmt(std_i)}</td>'
            '</tr>'
        )

    lines.append('</tbody></table>')

    # --- Detailed per-region tables ---
    lines.append('<h4>Detailed Features</h4>')

    for abbrev, region_data in all_features.items():
        name = region_data.get('name', abbrev)
        feats = region_data.get('features', {})
        error = region_data.get('error')

        if error or not feats:
            continue

        region_id = f'radiomics-{abbrev}'
        lines.append(
            f'<details class="radiomics-region" id="{region_id}">'
            f'<summary><strong>{name}</strong></summary>'
        )

        for cat_key in CATEGORY_NAMES:
            if cat_key not in feats:
                continue
            cat_name = CATEGORY_NAMES[cat_key]
            cat_feats = feats[cat_key]

            lines.append(
                f'<table class="radiomics-detail">'
                f'<caption>{cat_name} Features</caption>'
                '<thead><tr><th>Feature</th><th>Value</th></tr></thead>'
                '<tbody>'
            )

            for feat_name, feat_val in sorted(cat_feats.items()):
                if isinstance(feat_val, float):
                    display = (
                        f'{feat_val:.6f}' if abs(feat_val) < 1e6
                        else f'{feat_val:.4e}'
                    )
                else:
                    display = str(feat_val)
                lines.append(
                    f'<tr><td>{feat_name}</td><td>{display}</td></tr>'
                )

            lines.append('</tbody></table>')

        lines.append('</details>')

    lines.append('</div>')
    return '\n'.join(lines)
