"""DICOM to BIDS conversion workflow."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import nibabel as nb

from oncoprep.utils.logging import get_logger

LOGGER = get_logger(__name__)

# Mapping of DICOM series descriptions to BIDS modalities
MODALITY_MAPPING = {
    'T1CE': 'T1ce',
    'T1_CE': 'T1ce',
    'T1 CE': 'T1ce',
    'T1W': 'T1w',
    'T1': 'T1w',
    'T2W': 'T2w',
    'T2': 'T2w',
    'FLAIR': 'FLAIR',
    'DWI': 'dwi',
    'DTI': 'dwi',
    'ADC': 'dwi',
    'BOLD': 'bold',
    'fMRI': 'bold',
    'FMRI': 'bold',
    'rsfMRI': 'bold',
    'RSFMRI': 'bold',
    'PERF': 'perf',
    'DSC': 'perf',
    'ASL': 'perf',
    'SWI': 'swi',
    'GRE': 'swi',
    'PD': 'PD',
    'PDFS': 'PD',
}


def infer_modality_from_series(series_name: str) -> Optional[str]:
    """
    Infer BIDS modality from DICOM series description.
    
    Parameters
    ----------
    series_name : str
        DICOM series description or directory name
        
    Returns
    -------
    Optional[str]
        BIDS modality suffix (T1w, T2w, T1ce, FLAIR) or None
    """
    upper_name = series_name.upper()
    
    # Check for T1CE/T1 CE before T1 to avoid false matches
    for key in ['T1CE', 'T1_CE', 'T1 CE']:
        if key in upper_name:
            return 'T1ce'
    
    # Check for other exact matches
    for key, modality in MODALITY_MAPPING.items():
        if key in upper_name and key not in ['T1CE', 'T1_CE', 'T1 CE']:
            return modality
    
    # Fallback to default
    return None


def convert_dicom_series_to_nifti(
    dicom_dir: Path,
    output_nifti: Path,
    tool: str = 'dcm2niix',
) -> bool:
    """
    Convert DICOM series to NIfTI format.
    
    Parameters
    ----------
    dicom_dir : Path
        Directory containing DICOM files
    output_nifti : Path
        Output path for NIfTI file
    tool : str
        Conversion tool to use ('dcm2niix' or 'nibabel')
        
    Returns
    -------
    bool
        True if conversion successful
    """
    if tool == 'dcm2niix':
        return _convert_with_dcm2niix(dicom_dir, output_nifti)
    elif tool == 'nibabel':
        return _convert_with_nibabel(dicom_dir, output_nifti)
    else:
        raise ValueError(f"Unknown conversion tool: {tool}")


def _convert_with_dcm2niix(dicom_dir: Path, output_nifti: Path) -> bool:
    """Convert DICOM to NIfTI using dcm2niix."""
    try:
        output_dir = output_nifti.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get the stem (filename without extension)
        output_prefix = output_nifti.stem.rsplit('.', 1)[0]  # Remove .nii or .nii.gz
        
        cmd = [
            'dcm2niix',
            '-ba', 'y',  # Save BIDS sidecar
            '-z', 'y',   # Compress output
            '-o', str(output_dir),
            '-f', output_prefix,
            str(dicom_dir),
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        
        if result.returncode != 0:
            LOGGER.warning(f"dcm2niix conversion failed: {result.stderr}")
            return False
        
        # Verify output exists
        if output_nifti.exists():
            LOGGER.info(f"✓ Converted DICOM to NIfTI: {output_nifti.name}")
            return True
        else:
            # Check if output was generated with different naming
            nifti_files = list(output_dir.glob(f"{output_prefix}*.nii*"))
            if nifti_files:
                generated_file = nifti_files[0]
                generated_file.rename(output_nifti)
                LOGGER.info(f"✓ Converted DICOM to NIfTI: {output_nifti.name}")
                return True
        
        return False
        
    except FileNotFoundError:
        LOGGER.error("dcm2niix not found. Install with: conda install dcm2niix")
        return False


def _convert_with_nibabel(dicom_dir: Path, output_nifti: Path) -> bool:
    """Convert DICOM to NIfTI using nibabel (basic fallback)."""
    try:
        import dicom
        
        # Find first DICOM file
        dicom_files = sorted(dicom_dir.glob('*.dcm')) + sorted(dicom_dir.glob('*.IMA'))
        
        if not dicom_files:
            LOGGER.warning(f"No DICOM files found in {dicom_dir}")
            return False
        
        # Load DICOM
        dcm = dicom.dcmread(str(dicom_files[0]))
        
        # Extract pixel array
        pixel_array = dcm.pixel_array
        
        # Create NIfTI image
        img = nb.Nifti1Image(pixel_array, affine=None)
        
        # Save
        output_nifti.parent.mkdir(parents=True, exist_ok=True)
        nb.save(img, str(output_nifti))
        
        LOGGER.info(f"✓ Converted DICOM to NIfTI (nibabel): {output_nifti.name}")
        return True
        
    except Exception as e:
        LOGGER.warning(f"nibabel conversion failed: {e}")
        return False


def extract_dicom_metadata(dicom_dir: Path) -> dict:
    """
    Extract key DICOM metadata from series for BIDS sidecar.
    
    Parameters
    ----------
    dicom_dir : Path
        Directory containing DICOM files
        
    Returns
    -------
    dict
        Extracted metadata for BIDS JSON sidecar
    """
    metadata = {}
    
    try:
        import dicom
        
        # Find first DICOM file
        dicom_files = sorted(dicom_dir.glob('*.dcm')) + sorted(dicom_dir.glob('*.IMA'))
        if not dicom_files:
            return metadata
        
        dcm = dicom.dcmread(str(dicom_files[0]))
        
        # Extract key parameters for BIDS
        key_mappings = {
            'RepetitionTime': ('0018', '0088'),      # TR in ms
            'EchoTime': ('0018', '0081'),           # TE in ms
            'FlipAngle': ('0018', '1314'),          # FA in degrees
            'MagneticFieldStrength': ('0018', '0087'), # Field strength
            'EchoTrainLength': ('0018', '0091'),    # ETL
            'SeriesNumber': ('0020', '0011'),       # Series number
            'SeriesDescription': ('0008', '103e'),  # Series description
            'SequenceName': ('0018', '0024'),       # Sequence name
            'SequenceVariant': ('0018', '0021'),    # Sequence variant
            'InversionTime': ('0018', '0082'),      # TI in ms
            'PatientPosition': ('0018', '5100'),    # Patient position
        }
        
        for bids_key, (group, element) in key_mappings.items():
            try:
                tag = int(group, 16) * 65536 + int(element, 16)
                if tag in dcm:
                    value = dcm[tag].value
                    # Convert to appropriate type
                    if isinstance(value, (int, float)):
                        metadata[bids_key] = float(value) if '.' in str(value) else int(value)
                    else:
                        metadata[bids_key] = str(value).strip()
            except (KeyError, ValueError, TypeError):
                pass
        
        LOGGER.debug(f"Extracted metadata: {metadata}")
        
    except Exception as e:
        LOGGER.debug(f"Could not extract DICOM metadata: {e}")
    
    return metadata


def create_bids_sidecar(
    output_nifti: Path,
    metadata: Optional[dict] = None,
) -> Path:
    """
    Create BIDS JSON sidecar for NIfTI file.
    
    Parameters
    ----------
    output_nifti : Path
        Path to NIfTI file
    metadata : Optional[dict]
        Metadata to include in sidecar
        
    Returns
    -------
    Path
        Path to created JSON sidecar
    """
    json_path = output_nifti.with_suffix('').with_suffix('.json')
    
    sidecar = metadata or {}
    
    # Add required BIDS fields if not present
    if 'RepetitionTime' not in sidecar:
        sidecar['RepetitionTime'] = 2.0  # Default
    if 'EchoTime' not in sidecar:
        sidecar['EchoTime'] = 0.00456  # Default
    
    with open(json_path, 'w') as f:
        json.dump(sidecar, f, indent=4)
    
    LOGGER.info(f"✓ Created BIDS sidecar: {json_path.name}")
    return json_path


def convert_subject_dicoms_to_bids(
    source_dir: Path,
    bids_dir: Path,
    subject: str,
    session: Optional[str] = None,
    conversion_tool: str = 'dcm2niix',
) -> bool:
    """
    Convert all DICOM series in a subject directory to BIDS format.
    
    Parameters
    ----------
    source_dir : Path
        Directory containing subject DICOM series subdirectories
    bids_dir : Path
        BIDS dataset root directory
    subject : str
        Subject identifier (without 'sub-' prefix)
    session : Optional[str]
        Session identifier (without 'ses-' prefix)
    conversion_tool : str
        Tool to use for conversion ('dcm2niix' or 'nibabel')
        
    Returns
    -------
    bool
        True if at least one series was converted successfully
    """
    # Create BIDS subject/session structure
    if session:
        sub_bids_dir = bids_dir / f'sub-{subject}' / f'ses-{session}'
    else:
        sub_bids_dir = bids_dir / f'sub-{subject}'
    
    converted_count = 0
    
    # Find all DICOM series directories
    for series_dir in sorted(source_dir.iterdir()):
        if not series_dir.is_dir():
            continue
        
        # Skip hidden directories
        if series_dir.name.startswith('.'):
            continue
        
        # Check if directory contains DICOM files
        dicom_files = list(series_dir.glob('*.dcm')) + list(series_dir.glob('*.IMA'))
        if not dicom_files:
            continue
        
        LOGGER.info(f"Processing DICOM series: {series_dir.name}")
        
        # Infer modality
        modality = infer_modality_from_series(series_dir.name)
        if not modality:
            LOGGER.warning(f"Could not infer modality from series: {series_dir.name}")
            modality = 'anat'  # Default
        
        # Determine datatype directory based on modality
        if modality in ['dwi', 'perf', 'swi']:
            datatype = modality
        elif modality in ['bold']:
            datatype = 'func'
        elif modality == 'PD':
            datatype = 'anat'
        else:
            datatype = 'anat'
        
        # Create datatype directory
        datatype_dir = sub_bids_dir / datatype
        datatype_dir.mkdir(parents=True, exist_ok=True)
        
        # Construct BIDS filename
        if session:
            bids_basename = f'sub-{subject}_ses-{session}_{modality}'
        else:
            bids_basename = f'sub-{subject}_{modality}'
        
        output_nifti = datatype_dir / f'{bids_basename}.nii.gz'
        
        # Skip if already exists
        if output_nifti.exists():
            LOGGER.info(f"Output file already exists, skipping: {output_nifti.name}")
            converted_count += 1
            continue
        
        # Convert DICOM to NIfTI
        success = convert_dicom_series_to_nifti(
            series_dir,
            output_nifti,
            tool=conversion_tool,
        )
        
        if success:
            # Only create sidecar for non-dcm2niix tools
            # (dcm2niix creates JSON automatically with -ba y flag)
            if conversion_tool != 'dcm2niix':
                metadata = extract_dicom_metadata(series_dir)
                create_bids_sidecar(output_nifti, metadata)
            converted_count += 1
        else:
            LOGGER.warning(f"Failed to convert series: {series_dir.name}")
    
    if converted_count == 0:
        LOGGER.warning(f"No DICOM series found in {source_dir}")
        return False
    
    LOGGER.info(f"✓ Converted {converted_count} DICOM series for sub-{subject}")
    return True


def create_bids_dataset_description(bids_dir: Path) -> Path:
    """
    Create BIDS dataset_description.json if it doesn't exist.
    
    Parameters
    ----------
    bids_dir : Path
        BIDS dataset root directory
        
    Returns
    -------
    Path
        Path to dataset_description.json
    """
    desc_file = bids_dir / 'dataset_description.json'
    
    if desc_file.exists():
        LOGGER.info("dataset_description.json already exists")
        return desc_file
    
    description = {
        "Name": "OncoPrep DICOM Conversion Dataset",
        "BIDSVersion": "1.9.0",
        "DatasetType": "raw",
        "License": "CC0",
        "Authors": [
            {
                "name": "OncoPrep Contributors",
            }
        ],
        "Acknowledgements": "Converted using OncoPrep",
        "HowToAcknowledge": "Please cite OncoPrep and the BIDS specification",
        "Funding": [],
        "EthicsApprovals": [],
        "ReferencesAndLinks": [],
        "DatasetLinks": {},
        "Keywords": ["neuro-oncology", "MRI", "DICOM"],
        "SourceDatasets": [],
        "ConsentLinks": [],
    }
    
    bids_dir.mkdir(parents=True, exist_ok=True)
    
    with open(desc_file, 'w') as f:
        json.dump(description, f, indent=4)
    
    LOGGER.info(f"✓ Created BIDS dataset_description.json")
    return desc_file
