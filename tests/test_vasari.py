"""Unit tests for VASARI interfaces and workflow instantiation."""

from __future__ import annotations

import json
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
def synthetic_segmentation(tmp_dir):
    """Create a synthetic multi-label tumor segmentation (old BraTS convention).

    Labels: 1=nCET, 2=ED (oedema), 3=ET (enhancing tumor).
    Uses MNI152 1mm affine to match atlas expectations.
    """
    shape = (182, 218, 182)  # MNI152 1mm shape
    data = np.zeros(shape, dtype=np.uint8)

    # Place a small tumor in the right frontal region
    # nCET core
    data[70:80, 120:130, 90:100] = 1
    # Oedema surrounding
    data[65:85, 115:135, 85:105] = 2
    # Overwrite core with nCET again
    data[70:80, 120:130, 90:100] = 1
    # ET enhancing rim
    data[68:82, 118:132, 88:102] = 3
    data[70:80, 120:130, 90:100] = 1  # nCET in the centre

    affine = np.eye(4)
    affine[0, 3] = -90  # approximate MNI origin
    affine[1, 3] = -126
    affine[2, 3] = -72

    img = nib.Nifti1Image(data, affine)
    path = str(tmp_dir / 'tumor_seg.nii.gz')
    nib.save(img, path)
    return path, data


@pytest.fixture()
def synthetic_t1w(tmp_dir):
    """Create a synthetic T1w image for MNI registration testing."""
    shape = (182, 218, 182)
    rng = np.random.default_rng(42)
    data = rng.normal(loc=100.0, scale=20.0, size=shape).astype(np.float32)
    affine = np.eye(4)
    affine[0, 3] = -90
    affine[1, 3] = -126
    affine[2, 3] = -72
    img = nib.Nifti1Image(data, affine)
    path = str(tmp_dir / 't1w.nii.gz')
    nib.save(img, path)
    return path


# ---------------------------------------------------------------------------
# Helper conversion tests
# ---------------------------------------------------------------------------

class TestVASARIHelpers:
    """Tests for VASARI helper/conversion functions."""

    def test_dataframe_to_vasari_dict(self):
        """Test conversion of vasari-auto DataFrame to structured dict."""
        import pandas as pd
        from oncoprep.interfaces.vasari import _dataframe_to_vasari_dict

        # Simulate a typical vasari-auto output row
        row = {
            'filename': 'test_seg.nii.gz',
            'reporter': 'VASARI-auto',
            'time_taken_seconds': 2.5,
            'F1 Tumour Location': 1,  # Frontal Lobe
            'F2 Side of Tumour Epicenter': 1,  # Right
            'F3 Eloquent Brain': np.nan,
            'F4 Enhancement Quality': 3,  # Marked
            'F5 Proportion Enhancing': 4,  # 6-33%
            'F6 Proportion nCET': 3,  # ≤5%
            'F7 Proportion Necrosis': 3,
            'F8 Cyst(s)': np.nan,
            'F9 Multifocal or Multicentric': 1,  # Not multifocal
            'F10 T1/FLAIR Ratio': np.nan,
            'F11 Thickness of enhancing margin': 4,  # Thick
            'F12 Definition of the Enhancing margin': np.nan,
            'F13 Definition of the non-enhancing tumour margin': np.nan,
            'F14 Proportion of Oedema': 4,  # 6-33%
            'F16 haemorrhage': np.nan,
            'F17 Diffusion': np.nan,
            'F18 Pial invasion': np.nan,
            'F19 Ependymal Invasion': 1,  # Absent
            'F20 Cortical involvement': 2,  # Present
            'F21 Deep WM invasion': 1,  # Absent
            'F22 nCET Crosses Midline': 2,  # Does not cross
            'F23 CET Crosses midline': 2,  # Does not cross
            'F24 satellites': 1,  # Absent
            'F25 Calvarial modelling': np.nan,
            'COMMENTS': 'test',
        }

        df = pd.DataFrame([row])
        result = _dataframe_to_vasari_dict(df)

        # Check structure
        assert 'metadata' in result
        assert 'features' in result
        assert result['metadata']['reporter'] == 'VASARI-auto'
        assert result['metadata']['time_taken_seconds'] == 2.5

        # Check feature mapping
        feats = result['features']
        assert feats['F1']['label'] == 'Frontal Lobe'
        assert feats['F1']['code'] == 1
        assert feats['F2']['label'] == 'Right'
        assert feats['F3']['code'] is None  # NaN → None
        assert feats['F3']['label'] == 'Unsupported (requires source imaging)'
        assert feats['F4']['label'] == 'Marked/Avid'
        assert feats['F19']['label'] == 'Absent'
        assert feats['F20']['label'] == 'Present'

    def test_render_vasari_html(self):
        """Test that HTML rendering produces valid output."""
        from oncoprep.interfaces.vasari import _render_vasari_html

        features = {
            'metadata': {
                'reporter': 'VASARI-auto',
                'time_taken_seconds': 1.0,
            },
            'features': {
                'F1': {'name': 'Tumour Location', 'code': 1, 'label': 'Frontal Lobe'},
                'F2': {'name': 'Side of Tumour Epicenter', 'code': 1, 'label': 'Right'},
                'F3': {'name': 'Eloquent Brain', 'code': None, 'label': 'Unsupported'},
            },
        }

        html = _render_vasari_html(features)
        assert '<div class="vasari-report">' in html
        assert 'Frontal Lobe' in html
        assert 'Right' in html
        assert 'Unsupported' in html

    def test_generate_impression(self):
        """Test clinical impression generation."""
        from oncoprep.interfaces.vasari import _generate_impression

        feat_dict = {
            'F1': {'name': 'Tumour Location', 'code': 1, 'label': 'Frontal Lobe'},
            'F2': {'name': 'Side', 'code': 1, 'label': 'Right'},
            'F4': {'name': 'Enhancement', 'code': 3, 'label': 'Marked/Avid'},
            'F9': {'name': 'Multifocal', 'code': 1, 'label': 'Not multifocal'},
            'F19': {'name': 'Ependymal', 'code': 1, 'label': 'Absent'},
            'F21': {'name': 'Deep WM', 'code': 2, 'label': 'Present'},
            'F22': {'name': 'nCET Midline', 'code': 2, 'label': 'Does not cross'},
            'F23': {'name': 'CET Midline', 'code': 2, 'label': 'Does not cross'},
        }

        impression = _generate_impression(feat_dict)
        assert 'Frontal Lobe' in impression.lower() or 'frontal lobe' in impression
        assert 'deep white matter invasion' in impression
        assert len(impression) > 20


# ---------------------------------------------------------------------------
# Report generation tests
# ---------------------------------------------------------------------------

class TestRadiologyReport:
    """Tests for report generation interfaces."""

    @pytest.fixture()
    def sample_features_json(self, tmp_dir):
        """Create a sample VASARI features JSON file."""
        features = {
            'metadata': {
                'reporter': 'VASARI-auto',
                'time_taken_seconds': 1.5,
                'filename': 'test.nii.gz',
                'software_note': 'test note',
            },
            'features': {
                'F1': {'name': 'Tumour Location', 'code': 1, 'label': 'Frontal Lobe'},
                'F2': {'name': 'Side of Tumour Epicenter', 'code': 1, 'label': 'Right'},
                'F3': {'name': 'Eloquent Brain', 'code': None, 'label': 'Unsupported (requires source imaging)'},
                'F4': {'name': 'Enhancement Quality', 'code': 3, 'label': 'Marked/Avid'},
                'F5': {'name': 'Proportion Enhancing', 'code': 4, 'label': '6–33%'},
                'F6': {'name': 'Proportion nCET', 'code': 3, 'label': '≤5%'},
                'F7': {'name': 'Proportion Necrosis', 'code': 3, 'label': '≤5%'},
                'F8': {'name': 'Cysts', 'code': None, 'label': 'Unsupported'},
                'F9': {'name': 'Multifocal or Multicentric', 'code': 1, 'label': 'Not multifocal'},
                'F10': {'name': 'T1/FLAIR Ratio', 'code': None, 'label': 'Unsupported'},
                'F11': {'name': 'Thickness of Enhancing Margin', 'code': 4, 'label': 'Thick (>3×)'},
                'F12': {'name': 'Definition of Enhancing Margin', 'code': None, 'label': 'Unsupported'},
                'F13': {'name': 'Definition of Non-Enhancing Margin', 'code': None, 'label': 'Unsupported'},
                'F14': {'name': 'Proportion of Oedema', 'code': 4, 'label': '6–33%'},
                'F16': {'name': 'Haemorrhage', 'code': None, 'label': 'Unsupported'},
                'F17': {'name': 'Diffusion', 'code': None, 'label': 'Unsupported'},
                'F18': {'name': 'Pial Invasion', 'code': None, 'label': 'Unsupported'},
                'F19': {'name': 'Ependymal Invasion', 'code': 1, 'label': 'Absent'},
                'F20': {'name': 'Cortical Involvement', 'code': 2, 'label': 'Present'},
                'F21': {'name': 'Deep WM Invasion', 'code': 1, 'label': 'Absent'},
                'F22': {'name': 'nCET Crosses Midline', 'code': 2, 'label': 'Does not cross midline'},
                'F23': {'name': 'CET Crosses Midline', 'code': 2, 'label': 'Does not cross midline'},
                'F24': {'name': 'Satellites', 'code': 1, 'label': 'Absent'},
                'F25': {'name': 'Calvarial Modelling', 'code': None, 'label': 'Unsupported'},
            },
        }
        path = tmp_dir / 'vasari_features.json'
        with open(path, 'w') as f:
            json.dump(features, f, indent=2)
        return str(path)

    def test_structured_report(self, sample_features_json):
        """Test structured report generation via the interface."""
        from oncoprep.interfaces.vasari import VASARIRadiologyReport

        node = VASARIRadiologyReport(
            in_features=sample_features_json,
            patient_id='sub-001',
            template='structured',
        )
        result = node.run()

        assert os.path.isfile(result.outputs.out_report)
        assert os.path.isfile(result.outputs.out_text)

        with open(result.outputs.out_report) as f:
            html = f.read()
        assert 'VASARI' in html
        assert 'Frontal Lobe' in html

        with open(result.outputs.out_text) as f:
            text = f.read()
        assert 'VASARI' in text
        assert 'sub-001' in text

    def test_narrative_report(self, sample_features_json):
        """Test narrative report generation."""
        from oncoprep.interfaces.vasari import VASARIRadiologyReport

        node = VASARIRadiologyReport(
            in_features=sample_features_json,
            patient_id='sub-002',
            template='narrative',
        )
        result = node.run()

        with open(result.outputs.out_text) as f:
            text = f.read()
        assert 'mass lesion' in text.lower()
        assert 'sub-002' in text

    def test_brief_report(self, sample_features_json):
        """Test brief report generation."""
        from oncoprep.interfaces.vasari import VASARIRadiologyReport

        node = VASARIRadiologyReport(
            in_features=sample_features_json,
            template='brief',
        )
        result = node.run()

        with open(result.outputs.out_text) as f:
            text = f.read()
        assert 'KEY FINDINGS' in text
        assert 'Location: Frontal Lobe' in text


# ---------------------------------------------------------------------------
# Workflow instantiation tests
# ---------------------------------------------------------------------------

class TestVASARIWorkflow:
    """Test VASARI workflow instantiation."""

    def test_init_vasari_wf_creates_workflow(self, tmp_dir):
        """Test that the workflow factory creates a valid workflow."""
        from oncoprep.workflows.vasari import init_vasari_wf

        wf = init_vasari_wf(output_dir=str(tmp_dir))
        assert wf is not None
        assert wf.name == 'vasari_wf'

        node_names = wf.list_node_names()
        assert 'inputnode' in node_names
        assert 'outputnode' in node_names
        assert 'vasari_extract' in node_names
        assert 'vasari_report' in node_names
        assert 'ds_vasari_json' in node_names
        assert 'ds_vasari_report' in node_names
        assert 'ds_radiology_html' in node_names
        assert 'ds_radiology_txt' in node_names

    def test_init_vasari_wf_custom_name(self, tmp_dir):
        """Test workflow with custom name."""
        from oncoprep.workflows.vasari import init_vasari_wf

        wf = init_vasari_wf(output_dir=str(tmp_dir), name='custom_vasari')
        assert wf.name == 'custom_vasari'

    def test_init_vasari_wf_node_count(self, tmp_dir):
        """Test that expected number of nodes are created."""
        from oncoprep.workflows.vasari import init_vasari_wf

        wf = init_vasari_wf(output_dir=str(tmp_dir))
        node_names = wf.list_node_names()
        # inputnode, outputnode, vasari_extract, vasari_report,
        # ds_vasari_json, ds_vasari_report, ds_radiology_html, ds_radiology_txt
        assert len(node_names) >= 8

    def test_init_vasari_wf_atlas_space_mni(self, tmp_dir):
        """Test workflow with explicit MNI152 atlas space."""
        from oncoprep.workflows.vasari import init_vasari_wf

        wf = init_vasari_wf(
            output_dir=str(tmp_dir),
            atlas_space='MNI152NLin2009cAsym',
        )
        assert wf is not None
        # Verify vasari_extract has atlas_dir set
        extract_node = wf.get_node('vasari_extract')
        assert 'mni152' in extract_node.inputs.atlas_dir

    def test_init_vasari_wf_atlas_space_sri24(self, tmp_dir):
        """Test workflow with SRI24 atlas space."""
        from oncoprep.workflows.vasari import init_vasari_wf

        wf = init_vasari_wf(
            output_dir=str(tmp_dir),
            atlas_space='SRI24',
        )
        assert wf is not None
        extract_node = wf.get_node('vasari_extract')
        assert 'sri24' in extract_node.inputs.atlas_dir

    def test_init_vasari_wf_inputnode_has_tumor_seg_std(self, tmp_dir):
        """Test that inputnode expects template-space segmentation."""
        from oncoprep.workflows.vasari import init_vasari_wf

        wf = init_vasari_wf(output_dir=str(tmp_dir))
        inputnode = wf.get_node('inputnode')
        assert 'tumor_seg_std' in inputnode.inputs.copyable_trait_names()


class TestAtlasHelpers:
    """Tests for atlas directory resolution."""

    #: ROI masks that must be present in every atlas space directory
    EXPECTED_ROIS = [
        'brainstem',
        'corpus_callosum',
        'cortex',
        'frontal_lobe',
        'insula',
        'internal_capsule',
        'occipital',
        'parietal',
        'temporal',
        'thalamus',
        'ventricles',
    ]

    @pytest.mark.parametrize('space', ['mni152', 'sri24'])
    def test_atlas_masks_exist(self, space):
        """All expected ROI NIfTI masks are present on disk."""
        from oncoprep.interfaces.vasari import get_atlas_dir

        atlas_dir = get_atlas_dir(space)
        for roi in self.EXPECTED_ROIS:
            mask_path = os.path.join(atlas_dir, f'{roi}.nii.gz')
            assert os.path.isfile(mask_path), (
                f"Missing atlas mask: {mask_path}"
            )

    @pytest.mark.parametrize('space', ['mni152', 'sri24'])
    def test_atlas_reference_is_valid_nifti(self, space):
        """Reference brain image loads as a valid NIfTI with 3-D shape."""
        import nibabel as nib
        from oncoprep.interfaces.vasari import get_atlas_reference

        ref_path = get_atlas_reference(space)
        img = nib.load(ref_path)
        assert len(img.shape) == 3, (
            f"Expected 3-D image, got shape {img.shape}"
        )
        assert all(d > 0 for d in img.shape)

    @pytest.mark.parametrize('space', ['mni152', 'sri24'])
    def test_atlas_masks_are_valid_nifti(self, space):
        """Every ROI mask loads as a valid NIfTI."""
        import nibabel as nib
        from oncoprep.interfaces.vasari import get_atlas_dir

        atlas_dir = get_atlas_dir(space)
        for roi in self.EXPECTED_ROIS:
            img = nib.load(os.path.join(atlas_dir, f'{roi}.nii.gz'))
            assert len(img.shape) == 3, (
                f"{roi} has unexpected shape {img.shape}"
            )

    def test_get_atlas_dir_mni152(self):
        """Test MNI152 atlas directory resolution."""
        from oncoprep.interfaces.vasari import get_atlas_dir

        d = get_atlas_dir('mni152')
        assert d.endswith('/')
        assert 'mni152' in d
        assert os.path.isdir(d.rstrip('/'))

    def test_get_atlas_dir_mni152nlin2009casym(self):
        """Test MNI152NLin2009cAsym maps to mni152 atlas."""
        from oncoprep.interfaces.vasari import get_atlas_dir

        d = get_atlas_dir('MNI152NLin2009cAsym')
        assert 'mni152' in d

    def test_get_atlas_dir_sri24(self):
        """Test SRI24 atlas directory resolution."""
        from oncoprep.interfaces.vasari import get_atlas_dir

        d = get_atlas_dir('SRI24')
        assert 'sri24' in d
        assert os.path.isdir(d.rstrip('/'))

    def test_get_atlas_dir_invalid(self):
        """Test that invalid space raises ValueError."""
        from oncoprep.interfaces.vasari import get_atlas_dir

        with pytest.raises(ValueError, match='Unsupported'):
            get_atlas_dir('invalid_space')

    def test_get_atlas_reference_mni152(self):
        """Test MNI152 reference image resolution."""
        from oncoprep.interfaces.vasari import get_atlas_reference

        ref = get_atlas_reference('mni152')
        assert ref.endswith('.nii.gz')
        assert os.path.isfile(ref)
        assert 'MNI152_T1_1mm_brain' in ref

    def test_get_atlas_reference_sri24(self):
        """Test SRI24 reference image resolution."""
        from oncoprep.interfaces.vasari import get_atlas_reference

        ref = get_atlas_reference('SRI24')
        assert ref.endswith('.nii.gz')
        assert os.path.isfile(ref)
        assert 'SRI24' in ref
