"""Integration tests for DICOM to BIDS conversion."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from oncoprep.workflows.dicom_conversion import (
    infer_modality_from_series,
    create_bids_dataset_description,
    create_bids_sidecar,
)


class TestModalityInference:
    """Test modality inference from series names."""

    def test_t1_variants(self) -> None:
        """Test T1 modality detection."""
        assert infer_modality_from_series('T1')[0] == 'T1w'
        assert infer_modality_from_series('T1_MPRAGE')[0] == 'T1w'
        assert infer_modality_from_series('T1W')[0] == 'T1w'
        assert infer_modality_from_series('T1_MPRAGE_SAG_P2_1_0_ISO_0032')[0] == 'T1w'

    def test_t1ce_variants(self) -> None:
        """Test T1 contrast-enhanced modality detection."""
        mod, ce, _agent = infer_modality_from_series('T1CE')
        assert mod == 'T1w'
        assert ce is True
        mod, ce, _ = infer_modality_from_series('T1_CE')
        assert mod == 'T1w'
        assert ce is True
        mod, ce, _ = infer_modality_from_series('T1 CE')
        assert mod == 'T1w'
        assert ce is True
        # POST in a T1 series is treated as contrast-enhanced
        mod, ce, _ = infer_modality_from_series('T1_MPRAGE_POST')
        assert mod == 'T1w'
        assert ce is True
        mod, ce, _ = infer_modality_from_series('T1CE_POST')
        assert mod == 'T1w'
        assert ce is True

    def test_t2_variants(self) -> None:
        """Test T2 modality detection."""
        assert infer_modality_from_series('T2')[0] == 'T2w'
        assert infer_modality_from_series('T2W')[0] == 'T2w'
        assert infer_modality_from_series('T2_SPC')[0] == 'T2w'
        assert infer_modality_from_series('T2_SPC_DA-FL_SAG_P2_1_0_0012')[0] == 'T2w'

    def test_flair(self) -> None:
        """Test FLAIR modality detection."""
        assert infer_modality_from_series('FLAIR')[0] == 'FLAIR'
        assert infer_modality_from_series('COR_FLAIR')[0] == 'FLAIR'
        assert infer_modality_from_series('FLAIR_3D')[0] == 'FLAIR'

    def test_unknown_modality(self) -> None:
        """Test unknown modality returns None as first element."""
        assert infer_modality_from_series('UNKNOWN_SEQUENCE')[0] is None
        assert infer_modality_from_series('LOCALIZER')[0] is None


class TestBIDSDatasetDescription:
    """Test BIDS dataset description creation."""

    def test_create_dataset_description(self) -> None:
        """Test dataset_description.json creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bids_dir = Path(tmpdir)
            desc_file = create_bids_dataset_description(bids_dir)

            assert desc_file.exists()
            
            with open(desc_file) as f:
                desc = json.load(f)
            
            assert desc['Name'] == 'OncoPrep DICOM Conversion Dataset'
            assert desc['BIDSVersion'] == '1.9.0'
            assert desc['DatasetType'] == 'raw'
            assert 'License' in desc
            assert 'Authors' in desc

    def test_dataset_description_not_overwritten(self) -> None:
        """Test that existing dataset_description.json is not overwritten."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bids_dir = Path(tmpdir)
            
            # Create initial description
            desc_file = create_bids_dataset_description(bids_dir)
            
            # Try to create again
            desc_file2 = create_bids_dataset_description(bids_dir)
            
            # Should be the same file
            assert desc_file == desc_file2


class TestBIDSSidecar:
    """Test BIDS JSON sidecar creation."""

    def test_create_sidecar_with_metadata(self) -> None:
        """Test sidecar creation with custom metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bids_dir = Path(tmpdir)
            nifti_path = bids_dir / 'sub-001_T1w.nii.gz'
            nifti_path.touch()
            
            metadata = {
                'RepetitionTime': 2.3,
                'EchoTime': 0.00456,
                'FlipAngle': 9,
            }
            
            json_path = create_bids_sidecar(nifti_path, metadata)
            
            assert json_path.exists()
            
            with open(json_path) as f:
                sidecar = json.load(f)
            
            assert sidecar['RepetitionTime'] == 2.3
            assert sidecar['EchoTime'] == 0.00456
            assert sidecar['FlipAngle'] == 9

    def test_create_sidecar_default_values(self) -> None:
        """Test sidecar creation with default values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bids_dir = Path(tmpdir)
            nifti_path = bids_dir / 'sub-001_T1w.nii.gz'
            nifti_path.touch()
            
            json_path = create_bids_sidecar(nifti_path)
            
            with open(json_path) as f:
                sidecar = json.load(f)
            
            # Should have default values
            assert 'RepetitionTime' in sidecar
            assert 'EchoTime' in sidecar


class TestConversionExampleData:
    """Test conversion with example data if available."""

    @pytest.mark.skipif(
        not Path('examples/data').exists(),
        reason='Example DICOM data not available'
    )
    def test_example_data_discovery(self) -> None:
        """Test that example DICOM data can be discovered."""
        example_data = Path('examples/data')
        
        dicom_files = list(example_data.glob('**/*.IMA')) + list(example_data.glob('**/*.dcm'))
        assert len(dicom_files) > 0, "No DICOM files found in examples/data"
        
        # Check for series directories
        series_dirs = [d for d in example_data.iterdir() if d.is_dir() and not d.name.startswith('.')]
        assert len(series_dirs) > 0, "No DICOM series directories found"

    @pytest.mark.skipif(
        not Path('examples/data').exists(),
        reason='Example DICOM data not available'
    )
    def test_example_data_modality_inference(self) -> None:
        """Test modality inference on example DICOM series."""
        example_data = Path('examples/data')

        # Series directories are nested under subject directories
        series_dirs = [
            d
            for subj in example_data.iterdir()
            if subj.is_dir() and not subj.name.startswith('.')
            for d in subj.iterdir()
            if d.is_dir()
        ]

        # Should be able to infer modality for at least one series
        modalities = [infer_modality_from_series(d.name)[0] for d in series_dirs]
        assert any(m is not None for m in modalities), "Could not infer any modalities"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
