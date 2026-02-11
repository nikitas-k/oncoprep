"""Unit tests for HistogramNormalization and radiomics workflow instantiation."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_dir():
    """Provide a temporary directory cleaned up after each test."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture()
def synthetic_image(tmp_dir):
    """Create a synthetic 3-D NIfTI image with known intensities."""
    rng = np.random.default_rng(42)
    shape = (32, 32, 32)
    data = rng.normal(loc=100.0, scale=20.0, size=shape).astype(np.float32)
    affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)
    path = str(tmp_dir / 'image.nii.gz')
    nib.save(img, path)
    return path, data


@pytest.fixture()
def synthetic_brain_mask(tmp_dir, synthetic_image):
    """Binary brain mask covering the central cube of the image."""
    _, data = synthetic_image
    mask = np.zeros(data.shape, dtype=np.uint8)
    mask[8:24, 8:24, 8:24] = 1
    img = nib.Nifti1Image(mask, np.eye(4))
    path = str(tmp_dir / 'brain_mask.nii.gz')
    nib.save(img, path)
    return path, mask.astype(bool)


@pytest.fixture()
def multilabel_mask(tmp_dir, synthetic_image):
    """Multi-label segmentation mask (labels 1-4) that should be binarised."""
    _, data = synthetic_image
    mask = np.zeros(data.shape, dtype=np.uint8)
    mask[10:14, 10:14, 10:14] = 1
    mask[14:18, 14:18, 14:18] = 2
    mask[18:22, 18:22, 18:22] = 3
    mask[22:24, 22:24, 22:24] = 4
    img = nib.Nifti1Image(mask, np.eye(4))
    path = str(tmp_dir / 'seg_mask.nii.gz')
    nib.save(img, path)
    return path


# ---------------------------------------------------------------------------
# _run_interface-level tests  (calls through Nipype SimpleInterface)
# ---------------------------------------------------------------------------

class TestHistogramNormalization:
    """Tests for the HistogramNormalization interface."""

    def _run(self, in_file, in_mask, method='zscore', **kwargs):
        """Helper to instantiate and execute the interface."""
        from oncoprep.interfaces.radiomics import HistogramNormalization

        node = HistogramNormalization(
            in_file=in_file,
            in_mask=in_mask,
            method=method,
            **kwargs,
        )
        result = node.run()
        return result

    # -- zscore ----------------------------------------------------------

    def test_zscore_produces_output(self, synthetic_image, synthetic_brain_mask):
        """zscore should write a NIfTI file and set out_file."""
        in_file, _ = synthetic_image
        in_mask, _ = synthetic_brain_mask
        result = self._run(in_file, in_mask, method='zscore')
        assert os.path.isfile(result.outputs.out_file)

    def test_zscore_mean_and_std(self, synthetic_image, synthetic_brain_mask):
        """Masked voxels should have approximately zero mean / unit std."""
        in_file, _ = synthetic_image
        in_mask, mask_arr = synthetic_brain_mask
        result = self._run(in_file, in_mask, method='zscore')

        out = nib.load(result.outputs.out_file).get_fdata()
        brain_vals = out[mask_arr]
        assert abs(brain_vals.mean()) < 0.15, f'mean={brain_vals.mean():.4f}'
        assert abs(brain_vals.std() - 1.0) < 0.15, f'std={brain_vals.std():.4f}'

    def test_zscore_preserves_outside_mask(
        self, synthetic_image, synthetic_brain_mask
    ):
        """Voxels outside the brain mask must keep their original values."""
        in_file, orig_data = synthetic_image
        in_mask, mask_arr = synthetic_brain_mask
        result = self._run(in_file, in_mask, method='zscore')

        out = nib.load(result.outputs.out_file).get_fdata()
        np.testing.assert_array_almost_equal(
            out[~mask_arr], orig_data[~mask_arr], decimal=4,
            err_msg='Non-brain voxels were modified',
        )

    def test_zscore_constant_image(self, tmp_dir, synthetic_brain_mask):
        """A constant image should be returned unchanged (sigma ~0)."""
        in_mask, mask_arr = synthetic_brain_mask
        data = np.full((32, 32, 32), 42.0, dtype=np.float32)
        img = nib.Nifti1Image(data, np.eye(4))
        in_file = str(tmp_dir / 'const.nii.gz')
        nib.save(img, in_file)

        result = self._run(in_file, in_mask, method='zscore')
        out = nib.load(result.outputs.out_file).get_fdata()
        np.testing.assert_array_almost_equal(out, data, decimal=4)

    # -- nyul ------------------------------------------------------------

    def test_nyul_produces_output(self, synthetic_image, synthetic_brain_mask):
        """Nyul normalisation should produce a valid NIfTI."""
        in_file, _ = synthetic_image
        in_mask, _ = synthetic_brain_mask
        result = self._run(in_file, in_mask, method='nyul')
        assert os.path.isfile(result.outputs.out_file)

    def test_nyul_range(self, synthetic_image, synthetic_brain_mask):
        """Masked voxels should be mapped to approximately [0, 100]."""
        in_file, _ = synthetic_image
        in_mask, mask_arr = synthetic_brain_mask
        result = self._run(in_file, in_mask, method='nyul')

        out = nib.load(result.outputs.out_file).get_fdata()
        brain_vals = out[mask_arr]
        assert brain_vals.min() >= -1.0     # slight float tolerance
        assert brain_vals.max() <= 101.0

    def test_nyul_constant_image(self, tmp_dir, synthetic_brain_mask):
        """Nyul on a constant image should return it unchanged."""
        in_mask, _ = synthetic_brain_mask
        data = np.full((32, 32, 32), 7.0, dtype=np.float32)
        img = nib.Nifti1Image(data, np.eye(4))
        in_file = str(tmp_dir / 'const.nii.gz')
        nib.save(img, in_file)

        result = self._run(in_file, in_mask, method='nyul')
        out = nib.load(result.outputs.out_file).get_fdata()
        np.testing.assert_array_almost_equal(out, data, decimal=4)

    def test_nyul_preserves_outside_mask(
        self, synthetic_image, synthetic_brain_mask
    ):
        """Voxels outside the mask should retain original intensities."""
        in_file, orig_data = synthetic_image
        in_mask, mask_arr = synthetic_brain_mask
        result = self._run(in_file, in_mask, method='nyul')

        out = nib.load(result.outputs.out_file).get_fdata()
        np.testing.assert_array_almost_equal(
            out[~mask_arr], orig_data[~mask_arr], decimal=4,
        )

    # -- whitestripe -----------------------------------------------------

    def test_whitestripe_produces_output(
        self, synthetic_image, synthetic_brain_mask
    ):
        """WhiteStripe normalisation should produce a valid NIfTI."""
        in_file, _ = synthetic_image
        in_mask, _ = synthetic_brain_mask
        result = self._run(in_file, in_mask, method='whitestripe')
        assert os.path.isfile(result.outputs.out_file)

    def test_whitestripe_preserves_outside_mask(
        self, synthetic_image, synthetic_brain_mask
    ):
        """Voxels outside the mask should retain original intensities."""
        in_file, orig_data = synthetic_image
        in_mask, mask_arr = synthetic_brain_mask
        result = self._run(in_file, in_mask, method='whitestripe')

        out = nib.load(result.outputs.out_file).get_fdata()
        np.testing.assert_array_almost_equal(
            out[~mask_arr], orig_data[~mask_arr], decimal=4,
        )

    # -- Cross-cutting ---------------------------------------------------

    def test_multilabel_mask_binarised(
        self, synthetic_image, multilabel_mask
    ):
        """A multi-label mask (max > 1) should still produce valid output."""
        in_file, _ = synthetic_image
        result = self._run(in_file, multilabel_mask, method='zscore')
        assert os.path.isfile(result.outputs.out_file)

    def test_shape_mismatch_raises(self, tmp_dir, synthetic_image):
        """Mismatched image / mask shapes should raise RuntimeError."""
        in_file, _ = synthetic_image
        # Create a mask with a different shape
        wrong_mask = nib.Nifti1Image(
            np.ones((16, 16, 16), dtype=np.uint8), np.eye(4)
        )
        mask_path = str(tmp_dir / 'wrong_mask.nii.gz')
        nib.save(wrong_mask, mask_path)

        with pytest.raises(RuntimeError, match='does not match mask shape'):
            self._run(in_file, mask_path, method='zscore')

    def test_nan_in_input_handled(self, tmp_dir, synthetic_brain_mask):
        """NaNs in the input image should not propagate to output stats."""
        in_mask, mask_arr = synthetic_brain_mask
        rng = np.random.default_rng(99)
        data = rng.normal(100.0, 20.0, (32, 32, 32)).astype(np.float32)
        # Inject NaNs inside the mask
        data[12, 12, 12] = np.nan
        data[15, 15, 15] = np.inf
        img = nib.Nifti1Image(data, np.eye(4))
        in_file = str(tmp_dir / 'nan_image.nii.gz')
        nib.save(img, in_file)

        result = self._run(in_file, in_mask, method='zscore')
        out = nib.load(result.outputs.out_file).get_fdata()
        assert np.all(np.isfinite(out)), 'Output contains NaN or Inf'

    def test_unique_output_filename(self, synthetic_image, synthetic_brain_mask):
        """Output filename should be derived from the input stem."""
        in_file, _ = synthetic_image
        in_mask, _ = synthetic_brain_mask
        result = self._run(in_file, in_mask, method='zscore')
        out_name = Path(result.outputs.out_file).name
        assert 'image' in out_name, (
            f'Output name {out_name} does not contain input stem'
        )
        assert out_name != 'normalized.nii.gz', (
            'Output still uses generic name — collision risk in multimodal'
        )

    def test_empty_mask(self, tmp_dir, synthetic_image):
        """An all-zero mask should return the image unchanged."""
        in_file, orig_data = synthetic_image
        empty_mask = nib.Nifti1Image(
            np.zeros((32, 32, 32), dtype=np.uint8), np.eye(4)
        )
        mask_path = str(tmp_dir / 'empty_mask.nii.gz')
        nib.save(empty_mask, mask_path)

        result = self._run(in_file, mask_path, method='zscore')
        out = nib.load(result.outputs.out_file).get_fdata()
        np.testing.assert_array_almost_equal(out, orig_data, decimal=4)


# ---------------------------------------------------------------------------
# Workflow instantiation smoke tests
# ---------------------------------------------------------------------------

class TestRadiomicsWorkflows:
    """Smoke tests for radiomics workflow factory functions."""

    def test_anat_workflow_has_hist_norm(self):
        from oncoprep.workflows.radiomics import init_anat_radiomics_wf

        wf = init_anat_radiomics_wf(output_dir='/tmp/out')
        nodes = wf.list_node_names()
        assert 'hist_norm' in nodes
        assert 'radiomics_extract' in nodes
        assert 'inputnode' in nodes
        assert 'outputnode' in nodes

    def test_anat_workflow_inputnode_has_brain_mask(self):
        from oncoprep.workflows.radiomics import init_anat_radiomics_wf

        wf = init_anat_radiomics_wf(output_dir='/tmp/out')
        infields = wf.get_node('inputnode').outputs.copyable_trait_names()
        assert 'brain_mask' in infields

    def test_multimodal_workflow_has_per_modality_norm(self):
        from oncoprep.workflows.radiomics import init_multimodal_radiomics_wf

        wf = init_multimodal_radiomics_wf(
            output_dir='/tmp/out',
            modalities=['t1w', 'flair'],
        )
        nodes = wf.list_node_names()
        assert 'hist_norm_t1w' in nodes
        assert 'hist_norm_flair' in nodes
        assert 'radiomics_t1w' in nodes
        assert 'radiomics_flair' in nodes

    def test_multimodal_workflow_inputnode_has_brain_mask(self):
        from oncoprep.workflows.radiomics import init_multimodal_radiomics_wf

        wf = init_multimodal_radiomics_wf(
            output_dir='/tmp/out',
            modalities=['t1w'],
        )
        infields = wf.get_node('inputnode').outputs.copyable_trait_names()
        assert 'brain_mask' in infields
