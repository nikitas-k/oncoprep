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
# SUSAN denoising tests
# ---------------------------------------------------------------------------

class TestSUSANDenoising:
    """Tests for the SUSANDenoising interface."""

    def _run(self, in_file, in_mask, fwhm=2.0, **kwargs):
        """Helper to instantiate and execute the interface."""
        from oncoprep.interfaces.radiomics import SUSANDenoising

        node = SUSANDenoising(
            in_file=in_file,
            in_mask=in_mask,
            fwhm=fwhm,
            **kwargs,
        )
        result = node.run()
        return result

    def test_produces_output(self, synthetic_image, synthetic_brain_mask):
        """SUSAN should write a denoised NIfTI file."""
        in_file, _ = synthetic_image
        in_mask, _ = synthetic_brain_mask
        result = self._run(in_file, in_mask)
        assert os.path.isfile(result.outputs.out_file)

    def test_output_has_correct_shape(self, synthetic_image, synthetic_brain_mask):
        """Output should have the same shape as the input."""
        in_file, orig_data = synthetic_image
        in_mask, _ = synthetic_brain_mask
        result = self._run(in_file, in_mask)
        out = nib.load(result.outputs.out_file).get_fdata()
        assert out.shape == orig_data.shape

    def test_preserves_outside_mask(self, synthetic_image, synthetic_brain_mask):
        """Voxels outside the brain mask should keep their original values."""
        in_file, orig_data = synthetic_image
        in_mask, mask_arr = synthetic_brain_mask
        result = self._run(in_file, in_mask)
        out = nib.load(result.outputs.out_file).get_fdata()
        np.testing.assert_array_almost_equal(
            out[~mask_arr], orig_data[~mask_arr], decimal=4,
            err_msg='Non-brain voxels were modified by SUSAN',
        )

    def test_reduces_noise(self, synthetic_image, synthetic_brain_mask):
        """SUSAN should reduce variance (smooth) within the brain mask."""
        in_file, orig_data = synthetic_image
        in_mask, mask_arr = synthetic_brain_mask
        result = self._run(in_file, in_mask, fwhm=4.0)
        out = nib.load(result.outputs.out_file).get_fdata()
        # Smoothing should reduce standard deviation of masked voxels
        orig_std = orig_data[mask_arr].std()
        out_std = out[mask_arr].std()
        assert out_std <= orig_std * 1.05, (
            f'SUSAN did not smooth: orig_std={orig_std:.4f}, out_std={out_std:.4f}'
        )

    def test_unique_output_filename(self, synthetic_image, synthetic_brain_mask):
        """Output filename should contain the input stem + '_susan'."""
        in_file, _ = synthetic_image
        in_mask, _ = synthetic_brain_mask
        result = self._run(in_file, in_mask)
        out_name = Path(result.outputs.out_file).name
        assert 'susan' in out_name

    def test_empty_mask(self, tmp_dir, synthetic_image):
        """An all-zero mask should return the image unchanged."""
        in_file, orig_data = synthetic_image
        empty_mask = nib.Nifti1Image(
            np.zeros((32, 32, 32), dtype=np.uint8), np.eye(4),
        )
        mask_path = str(tmp_dir / 'empty_mask.nii.gz')
        nib.save(empty_mask, mask_path)

        result = self._run(in_file, mask_path)
        assert os.path.isfile(result.outputs.out_file)

    def test_shape_mismatch_raises(self, tmp_dir, synthetic_image):
        """Mismatched image / mask shapes should raise RuntimeError."""
        in_file, _ = synthetic_image
        wrong_mask = nib.Nifti1Image(
            np.ones((16, 16, 16), dtype=np.uint8), np.eye(4),
        )
        mask_path = str(tmp_dir / 'wrong_mask.nii.gz')
        nib.save(wrong_mask, mask_path)

        with pytest.raises(RuntimeError, match='does not match mask shape'):
            self._run(in_file, mask_path)


# ---------------------------------------------------------------------------
# ComBat harmonization tests
# ---------------------------------------------------------------------------

class TestComBatHarmonization:
    """Tests for the ComBatHarmonization interface."""

    @pytest.fixture()
    def sample_features(self, tmp_dir):
        """Create a sample radiomics features JSON."""
        features = {
            'NCR': {
                'label': 1,
                'name': 'Necrotic Core (NCR)',
                'features': {
                    'firstorder': {
                        'Mean': 100.5,
                        'StandardDeviation': 20.3,
                        'Entropy': 4.2,
                    },
                    'shape': {
                        'MeshVolume': 1500.0,
                        'Sphericity': 0.85,
                    },
                },
            },
            'ED': {
                'label': 2,
                'name': 'Peritumoral Edema (ED)',
                'features': {
                    'firstorder': {
                        'Mean': 80.2,
                        'StandardDeviation': 15.1,
                        'Entropy': 3.8,
                    },
                    'shape': {
                        'MeshVolume': 5000.0,
                        'Sphericity': 0.6,
                    },
                },
            },
        }
        path = str(tmp_dir / 'sub-001_radiomics.json')
        import json
        with open(path, 'w') as f:
            json.dump(features, f, indent=2)
        return path, features

    def test_skip_without_batch_file(self, sample_features):
        """Without batch_file, ComBat should return features unchanged."""
        from oncoprep.interfaces.radiomics import ComBatHarmonization

        feat_path, orig = sample_features
        node = ComBatHarmonization(
            in_features=feat_path,
            subject_id='sub-001',
        )
        result = node.run()
        assert os.path.isfile(result.outputs.out_features)
        assert os.path.isfile(result.outputs.out_report)

        import json
        with open(result.outputs.out_features) as f:
            out = json.load(f)
        # Features should be identical since ComBat was skipped
        assert out['NCR']['features']['firstorder']['Mean'] == 100.5

    def test_flatten_unflatten_roundtrip(self, sample_features):
        """Flatten + unflatten should preserve the data structure."""
        from oncoprep.interfaces.radiomics import (
            _flatten_features,
            _unflatten_features,
        )

        _, features = sample_features
        flat = _flatten_features(features)

        # Should have keys like 'NCR__firstorder__Mean'
        assert 'NCR__firstorder__Mean' in flat
        assert 'ED__shape__Sphericity' in flat

        # Unflatten back
        restored = _unflatten_features(features, flat)
        assert restored['NCR']['features']['firstorder']['Mean'] == 100.5
        assert restored['ED']['features']['shape']['Sphericity'] == 0.6


# ---------------------------------------------------------------------------
# Workflow instantiation smoke tests
# ---------------------------------------------------------------------------

class TestRadiomicsWorkflows:
    """Smoke tests for radiomics workflow factory functions."""

    def test_anat_workflow_has_hist_norm_and_susan(self):
        from oncoprep.workflows.radiomics import init_anat_radiomics_wf

        wf = init_anat_radiomics_wf(output_dir='/tmp/out')
        nodes = wf.list_node_names()
        assert 'hist_norm' in nodes
        assert 'susan_denoise' in nodes
        assert 'radiomics_extract' in nodes
        assert 'inputnode' in nodes
        assert 'outputnode' in nodes

    def test_anat_workflow_inputnode_has_brain_mask(self):
        from oncoprep.workflows.radiomics import init_anat_radiomics_wf

        wf = init_anat_radiomics_wf(output_dir='/tmp/out')
        infields = wf.get_node('inputnode').outputs.copyable_trait_names()
        assert 'brain_mask' in infields

    def test_anat_workflow_no_combat_nodes(self):
        """Participant-level workflow should NOT have any ComBat nodes."""
        from oncoprep.workflows.radiomics import init_anat_radiomics_wf

        wf = init_anat_radiomics_wf(output_dir='/tmp/out')
        nodes = wf.list_node_names()
        assert 'combat_harmonize' not in nodes
        assert 'get_subject_id' not in nodes
        assert 'ds_combat_json' not in nodes

    def test_multimodal_workflow_has_per_modality_susan(self):
        from oncoprep.workflows.radiomics import init_multimodal_radiomics_wf

        wf = init_multimodal_radiomics_wf(
            output_dir='/tmp/out',
            modalities=['t1w', 'flair'],
        )
        nodes = wf.list_node_names()
        assert 'hist_norm_t1w' in nodes
        assert 'hist_norm_flair' in nodes
        assert 'susan_denoise_t1w' in nodes
        assert 'susan_denoise_flair' in nodes
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


# ---------------------------------------------------------------------------
# Group-level ComBat harmonization tests
# ---------------------------------------------------------------------------

class TestGroupComBat:
    """Tests for group-level ComBat harmonization."""

    @pytest.fixture()
    def group_derivatives(self, tmp_dir):
        """Create a mock OncoPrep derivatives directory with
        per-subject radiomics JSONs from multiple scanner sites.
        """
        import csv
        import json as json_lib

        deriv = tmp_dir / 'oncoprep'
        rng = np.random.default_rng(123)

        # Create 6 subjects across 2 sites (3 per site)
        subjects = ['sub-001', 'sub-002', 'sub-003',
                    'sub-004', 'sub-005', 'sub-006']
        sites = ['siteA', 'siteA', 'siteA',
                 'siteB', 'siteB', 'siteB']

        for i, subj in enumerate(subjects):
            anat_dir = deriv / subj / 'anat'
            anat_dir.mkdir(parents=True)

            bias = 50.0 if sites[i] == 'siteB' else 0.0
            features = {
                'NCR': {
                    'label': 1,
                    'name': 'Necrotic Core (NCR)',
                    'features': {
                        'firstorder': {
                            'Mean': float(100.0 + bias + rng.normal(0, 5)),
                            'StandardDeviation': float(20.0 + bias * 0.2 + rng.normal(0, 2)),
                            'Entropy': float(4.0 + rng.normal(0, 0.3)),
                        },
                        'shape': {
                            'MeshVolume': float(1500.0 + rng.normal(0, 100)),
                            'Sphericity': float(0.85 + rng.normal(0, 0.05)),
                        },
                    },
                },
                'ED': {
                    'label': 2,
                    'name': 'Peritumoral Edema (ED)',
                    'features': {
                        'firstorder': {
                            'Mean': float(80.0 + bias + rng.normal(0, 5)),
                            'StandardDeviation': float(15.0 + bias * 0.15 + rng.normal(0, 2)),
                        },
                    },
                },
            }
            feat_path = anat_dir / f'{subj}_desc-radiomics_features.json'
            with open(feat_path, 'w') as f:
                json_lib.dump(features, f, indent=2)

        # Create batch CSV
        batch_file = tmp_dir / 'batch.csv'
        with open(batch_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['subject_id', 'batch'])
            for subj, site in zip(subjects, sites):
                writer.writerow([subj, site])

        return tmp_dir, str(batch_file), subjects, sites

    def test_collect_radiomics_jsons(self, group_derivatives):
        """Should find all per-subject radiomics JSON files."""
        from oncoprep.workflows.group import _collect_radiomics_jsons

        deriv_root, _, subjects, _ = group_derivatives
        found = _collect_radiomics_jsons(deriv_root)
        assert len(found) == 6
        assert set(found.keys()) == set(subjects)

    def test_collect_with_participant_filter(self, group_derivatives):
        """Should respect participant_label filter."""
        from oncoprep.workflows.group import _collect_radiomics_jsons

        deriv_root, _, _, _ = group_derivatives
        found = _collect_radiomics_jsons(
            deriv_root, participant_label=['sub-001', 'sub-003'],
        )
        assert len(found) == 2
        assert 'sub-001' in found
        assert 'sub-003' in found

    def test_collect_skips_combat_files(self, group_derivatives):
        """Should not include previously harmonized *Combat* files."""
        import json as json_lib
        from oncoprep.workflows.group import _collect_radiomics_jsons

        deriv_root, _, _, _ = group_derivatives
        # Create a fake Combat output
        combat_path = (
            deriv_root / 'oncoprep' / 'sub-001' / 'anat'
            / 'sub-001_desc-radiomicsCombat_features.json'
        )
        with open(combat_path, 'w') as f:
            json_lib.dump({}, f)

        found = _collect_radiomics_jsons(deriv_root)
        paths = [str(p) for p in found.values()]
        assert not any('Combat' in p for p in paths)

    def test_run_combat_harmonization(self, group_derivatives):
        """Full group-level ComBat run should produce harmonized JSONs."""
        import json as json_lib
        from oncoprep.workflows.group import _run_combat_harmonization

        deriv_root, batch_file, subjects, _ = group_derivatives
        _run_combat_harmonization(
            output_dir=deriv_root,
            batch_file=batch_file,
        )

        # Check that harmonized files were written
        for subj in subjects:
            combat_path = (
                deriv_root / 'oncoprep' / subj / 'anat'
                / f'{subj}_desc-radiomicsCombat_features.json'
            )
            assert combat_path.exists(), f'Missing harmonized file for {subj}'
            with open(combat_path) as f:
                data = json_lib.load(f)
            # Should still have the nested structure
            assert 'NCR' in data
            assert 'features' in data['NCR']

        # Check group report
        report_path = deriv_root / 'oncoprep' / 'group_combat_report.html'
        assert report_path.exists()
        html = report_path.read_text()
        assert 'ComBat Harmonization Report' in html
        assert 'siteA' in html
        assert 'siteB' in html

    def test_combat_reduces_site_effect(self, group_derivatives):
        """ComBat should reduce the between-site mean difference."""
        import json as json_lib
        from oncoprep.workflows.group import _run_combat_harmonization

        deriv_root, batch_file, subjects, sites = group_derivatives

        # Compute raw site-mean difference for NCR Mean
        raw_vals = {}
        for subj in subjects:
            feat_path = (
                deriv_root / 'oncoprep' / subj / 'anat'
                / f'{subj}_desc-radiomics_features.json'
            )
            with open(feat_path) as f:
                data = json_lib.load(f)
            raw_vals[subj] = data['NCR']['features']['firstorder']['Mean']

        siteA_raw = np.mean([raw_vals[s] for s, si in
                            zip(subjects, sites) if si == 'siteA'])
        siteB_raw = np.mean([raw_vals[s] for s, si in
                            zip(subjects, sites) if si == 'siteB'])
        raw_diff = abs(siteB_raw - siteA_raw)

        # Run ComBat
        _run_combat_harmonization(
            output_dir=deriv_root,
            batch_file=batch_file,
        )

        # Compute harmonized site-mean difference
        harm_vals = {}
        for subj in subjects:
            combat_path = (
                deriv_root / 'oncoprep' / subj / 'anat'
                / f'{subj}_desc-radiomicsCombat_features.json'
            )
            with open(combat_path) as f:
                data = json_lib.load(f)
            harm_vals[subj] = data['NCR']['features']['firstorder']['Mean']

        siteA_harm = np.mean([harm_vals[s] for s, si in
                             zip(subjects, sites) if si == 'siteA'])
        siteB_harm = np.mean([harm_vals[s] for s, si in
                             zip(subjects, sites) if si == 'siteB'])
        harm_diff = abs(siteB_harm - siteA_harm)

        # ComBat should substantially reduce the site effect
        assert harm_diff < raw_diff, (
            f'ComBat did not reduce site effect: '
            f'raw_diff={raw_diff:.2f}, harm_diff={harm_diff:.2f}'
        )

    def test_run_group_analysis_no_batch_file(self, group_derivatives):
        """run_group_analysis without batch file should warn and return 0."""
        from oncoprep.workflows.group import run_group_analysis

        deriv_root, _, _, _ = group_derivatives
        retcode = run_group_analysis(output_dir=deriv_root)
        assert retcode == 0

    def test_run_group_analysis_too_few_subjects(self, tmp_dir):
        """ComBat should fail if fewer than 3 subjects."""
        import csv
        import json as json_lib
        from oncoprep.workflows.group import run_group_analysis

        deriv = tmp_dir / 'oncoprep' / 'sub-001' / 'anat'
        deriv.mkdir(parents=True)
        feat = {'NCR': {'label': 1, 'features': {'firstorder': {'Mean': 1.0}}}}
        with open(deriv / 'sub-001_desc-radiomics_features.json', 'w') as f:
            json_lib.dump(feat, f)

        batch_file = str(tmp_dir / 'batch.csv')
        with open(batch_file, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['subject_id', 'batch'])
            w.writerow(['sub-001', 'siteA'])

        retcode = run_group_analysis(
            output_dir=tmp_dir,
            combat_batch_file=batch_file,
        )
        assert retcode == 1  # should fail gracefully

    def test_flatten_unflatten_roundtrip(self):
        """Group-level flatten/unflatten should round-trip correctly."""
        from oncoprep.workflows.group import (
            _flatten_features,
            _unflatten_features,
        )

        features = {
            'NCR': {
                'label': 1,
                'features': {
                    'firstorder': {'Mean': 42.0, 'Std': 5.0},
                    'shape': {'Volume': 100.0},
                },
            },
        }
        flat = _flatten_features(features)
        assert flat['NCR__firstorder__Mean'] == 42.0

        restored = _unflatten_features(features, flat)
        assert restored['NCR']['features']['firstorder']['Mean'] == 42.0
        assert restored['NCR']['features']['shape']['Volume'] == 100.0


class TestLongitudinalComBat:
    """Tests for longitudinal ComBat harmonization and age/sex batch CSV."""

    @pytest.fixture()
    def longitudinal_derivatives(self, tmp_dir):
        """Create a mock longitudinal OncoPrep derivatives directory.

        8 observations from 4 subjects × 2 sessions, across 2 sites
        (2 subjects per site to avoid collinearity).
        """
        import csv
        import json as json_lib

        deriv = tmp_dir / 'oncoprep'
        rng = np.random.default_rng(456)

        subjects = ['sub-001', 'sub-002', 'sub-003', 'sub-004']
        sessions = ['ses-01', 'ses-02']
        # 2 subjects per site to avoid collinearity with subject covariate
        site_map = {
            'sub-001': 'siteA', 'sub-002': 'siteA',
            'sub-003': 'siteB', 'sub-004': 'siteB',
        }

        obs_ids = []
        obs_sites = []
        for subj in subjects:
            for ses in sessions:
                obs_id = f'{subj}_{ses}'
                obs_ids.append(obs_id)
                obs_sites.append(site_map[subj])

                anat_dir = deriv / subj / ses / 'anat'
                anat_dir.mkdir(parents=True, exist_ok=True)

                bias = 50.0 if site_map[subj] == 'siteB' else 0.0
                features = {
                    'NCR': {
                        'label': 1,
                        'name': 'Necrotic Core (NCR)',
                        'features': {
                            'firstorder': {
                                'Mean': float(
                                    100.0 + bias + rng.normal(0, 5)
                                ),
                                'StandardDeviation': float(
                                    20.0 + bias * 0.2 + rng.normal(0, 2)
                                ),
                                'Entropy': float(4.0 + rng.normal(0, 0.3)),
                            },
                            'shape': {
                                'MeshVolume': float(
                                    1500.0 + rng.normal(0, 100)
                                ),
                                'Sphericity': float(
                                    0.85 + rng.normal(0, 0.05)
                                ),
                            },
                        },
                    },
                    'ED': {
                        'label': 2,
                        'name': 'Peritumoral Edema (ED)',
                        'features': {
                            'firstorder': {
                                'Mean': float(
                                    80.0 + bias + rng.normal(0, 5)
                                ),
                                'StandardDeviation': float(
                                    15.0 + bias * 0.15 + rng.normal(0, 2)
                                ),
                            },
                        },
                    },
                }
                fname = f'{subj}_{ses}_desc-radiomics_features.json'
                with open(anat_dir / fname, 'w') as f:
                    json_lib.dump(features, f, indent=2)

        # Create batch CSV keyed by observation ID
        batch_file = tmp_dir / 'batch_long.csv'
        with open(batch_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['subject_id', 'batch'])
            for obs_id, site in zip(obs_ids, obs_sites):
                writer.writerow([obs_id, site])

        return tmp_dir, str(batch_file), obs_ids, obs_sites, subjects

    def test_collect_longitudinal_obs(self, longitudinal_derivatives):
        """_collect_radiomics_jsons returns per-session observation keys."""
        from oncoprep.workflows.group import _collect_radiomics_jsons

        deriv_root, _, obs_ids, _, _ = longitudinal_derivatives
        found = _collect_radiomics_jsons(deriv_root)
        assert len(found) == 8
        assert set(found.keys()) == set(obs_ids)

    def test_collect_longitudinal_with_filter(self, longitudinal_derivatives):
        """Participant filter applies to subject-level, not observation."""
        from oncoprep.workflows.group import _collect_radiomics_jsons

        deriv_root, _, _, _, _ = longitudinal_derivatives
        found = _collect_radiomics_jsons(
            deriv_root, participant_label=['sub-001'],
        )
        assert len(found) == 2
        assert all(k.startswith('sub-001') for k in found)

    def test_longitudinal_combat_runs(self, longitudinal_derivatives):
        """Full longitudinal ComBat run should produce harmonized JSONs."""
        import json as json_lib
        from oncoprep.workflows.group import _run_combat_harmonization

        deriv_root, batch_file, obs_ids, _, subjects = (
            longitudinal_derivatives
        )
        _run_combat_harmonization(
            output_dir=deriv_root,
            batch_file=batch_file,
        )

        # Harmonized file should exist for each observation
        for obs_id in obs_ids:
            subj, ses = obs_id.split('_', 1)
            combat_path = (
                deriv_root / 'oncoprep' / subj / ses / 'anat'
                / f'{subj}_{ses}_desc-radiomicsCombat_features.json'
            )
            assert combat_path.exists(), (
                f'Missing harmonized file for {obs_id}'
            )
            with open(combat_path) as f:
                data = json_lib.load(f)
            assert 'NCR' in data

    def test_longitudinal_report_mentions_longitudinal(
        self, longitudinal_derivatives,
    ):
        """Report HTML should mention longitudinal mode."""
        from oncoprep.workflows.group import _run_combat_harmonization

        deriv_root, batch_file, _, _, _ = longitudinal_derivatives
        _run_combat_harmonization(
            output_dir=deriv_root,
            batch_file=batch_file,
        )

        report = deriv_root / 'oncoprep' / 'group_combat_report.html'
        assert report.exists()
        html = report.read_text()
        assert 'Longitudinal' in html
        assert 'Unique Subjects' in html

    def test_cross_sectional_no_longitudinal_note(self, tmp_dir):
        """Cross-sectional run should NOT mention longitudinal in report."""
        import csv
        import json as json_lib
        from oncoprep.workflows.group import _run_combat_harmonization

        # Create a small cross-sectional derivatives directory
        deriv = tmp_dir / 'xsect' / 'oncoprep'
        rng = np.random.default_rng(789)
        subjects = ['sub-001', 'sub-002', 'sub-003',
                    'sub-004', 'sub-005', 'sub-006']
        sites = ['siteA', 'siteA', 'siteA',
                 'siteB', 'siteB', 'siteB']
        for i, subj in enumerate(subjects):
            anat = deriv / subj / 'anat'
            anat.mkdir(parents=True)
            bias = 40.0 if sites[i] == 'siteB' else 0.0
            feat = {
                'R': {
                    'label': 1,
                    'features': {
                        'first': {
                            'Mean': float(90 + bias + rng.normal(0, 5)),
                            'Std': float(10 + rng.normal(0, 2)),
                        },
                    },
                },
            }
            with open(anat / f'{subj}_desc-radiomics_features.json', 'w') as f:
                json_lib.dump(feat, f)

        batch_file = tmp_dir / 'xsect_batch.csv'
        with open(batch_file, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['subject_id', 'batch'])
            for subj, site in zip(subjects, sites):
                w.writerow([subj, site])

        deriv_root = tmp_dir / 'xsect'
        _run_combat_harmonization(
            output_dir=deriv_root,
            batch_file=str(batch_file),
        )

        report = deriv_root / 'oncoprep' / 'group_combat_report.html'
        html = report.read_text()
        assert 'Unique Subjects' not in html
        assert 'No (cross-sectional)' in html


class TestBatchCsvGeneration:
    """Tests for generate_combat_batch_csv including age/sex extraction."""

    @pytest.fixture()
    def bids_with_metadata(self, tmp_dir):
        """Create a minimal BIDS directory with scanner and demographic
        metadata in JSON sidecars.
        """
        import json as json_lib

        bids = tmp_dir / 'bids_root'
        subjects = ['sub-001', 'sub-002', 'sub-003']
        age_sex = {
            'sub-001': {'PatientAge': '45', 'PatientSex': 'M'},
            'sub-002': {'Age': '52', 'Sex': 'F'},
            'sub-003': {},  # no demographics
        }
        scanners = {
            'sub-001': {
                'Manufacturer': 'Siemens',
                'ManufacturerModelName': 'Prisma',
                'MagneticFieldStrength': 3,
            },
            'sub-002': {
                'Manufacturer': 'GE',
                'ManufacturerModelName': 'SIGNA',
                'MagneticFieldStrength': 1.5,
            },
            'sub-003': {
                'Manufacturer': 'Siemens',
                'ManufacturerModelName': 'Prisma',
                'MagneticFieldStrength': 3,
            },
        }

        for subj in subjects:
            anat_dir = bids / subj / 'anat'
            anat_dir.mkdir(parents=True)
            meta = {}
            meta.update(scanners[subj])
            meta.update(age_sex[subj])
            with open(anat_dir / f'{subj}_T1w.json', 'w') as f:
                json_lib.dump(meta, f)

        return bids, subjects

    @pytest.fixture()
    def bids_with_participants_tsv(self, bids_with_metadata):
        """Extend bids_with_metadata with a participants.tsv that
        provides age/sex for subject 3 (which lacks sidecar data).
        """
        bids, subjects = bids_with_metadata
        tsv = bids / 'participants.tsv'
        tsv.write_text(
            'participant_id\tage\tsex\n'
            'sub-001\t45\tM\n'
            'sub-002\t52\tF\n'
            'sub-003\t60\tM\n'
        )
        return bids, subjects

    def test_batch_csv_basic(self, bids_with_metadata, tmp_dir):
        """Should produce a CSV with subject_id and batch columns."""
        import csv

        from oncoprep.workflows.group import generate_combat_batch_csv

        bids, _ = bids_with_metadata
        out_csv = tmp_dir / 'batch.csv'
        generate_combat_batch_csv(bids, out_csv)

        assert out_csv.exists()
        with open(out_csv) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 3
        assert 'subject_id' in rows[0]
        assert 'batch' in rows[0]

    def test_age_sex_from_sidecar(self, bids_with_metadata, tmp_dir):
        """Should extract age/sex from JSON sidecars when present."""
        import csv

        from oncoprep.workflows.group import generate_combat_batch_csv

        bids, _ = bids_with_metadata
        out_csv = tmp_dir / 'batch.csv'
        generate_combat_batch_csv(bids, out_csv)

        with open(out_csv) as f:
            reader = csv.DictReader(f)
            rows = {r['subject_id']: r for r in reader}

        assert rows['sub-001']['age'] == '45'
        assert rows['sub-001']['sex'] == 'M'
        assert rows['sub-002']['age'] == '52'
        assert rows['sub-002']['sex'] == 'F'

    def test_age_sex_fallback_to_participants_tsv(
        self, bids_with_participants_tsv, tmp_dir,
    ):
        """Subject 3 has no sidecar demographics; should fall back to
        participants.tsv.
        """
        import csv

        from oncoprep.workflows.group import generate_combat_batch_csv

        bids, _ = bids_with_participants_tsv
        out_csv = tmp_dir / 'batch.csv'
        generate_combat_batch_csv(bids, out_csv)

        with open(out_csv) as f:
            reader = csv.DictReader(f)
            rows = {r['subject_id']: r for r in reader}

        assert rows['sub-003']['age'] == '60'
        assert rows['sub-003']['sex'] == 'M'

    def test_longitudinal_batch_csv(self, tmp_dir):
        """Multi-session BIDS should produce one row per session."""
        import csv
        import json as json_lib

        from oncoprep.workflows.group import generate_combat_batch_csv

        bids = tmp_dir / 'bids_long'
        for subj in ['sub-001', 'sub-002']:
            for ses in ['ses-01', 'ses-02']:
                anat = bids / subj / ses / 'anat'
                anat.mkdir(parents=True)
                meta = {
                    'Manufacturer': 'Siemens',
                    'ManufacturerModelName': 'Prisma',
                    'MagneticFieldStrength': 3,
                }
                with open(anat / f'{subj}_{ses}_T1w.json', 'w') as f:
                    json_lib.dump(meta, f)

        out_csv = tmp_dir / 'batch_long.csv'
        generate_combat_batch_csv(bids, out_csv)

        with open(out_csv) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 4
        ids = {r['subject_id'] for r in rows}
        assert 'sub-001_ses-01' in ids
        assert 'sub-002_ses-02' in ids
