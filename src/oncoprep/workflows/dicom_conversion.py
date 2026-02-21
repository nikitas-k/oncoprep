"""DICOM to BIDS conversion workflow."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import nibabel as nb

from oncoprep.utils.logging import get_logger

LOGGER = get_logger(__name__)

# DICOM fields containing Protected Health Information (PHI) that should be removed
# Based on HIPAA Safe Harbor de-identification requirements
PHI_DICOM_FIELDS = [
    # Patient identifiers
    'PatientName',
    'PatientID',
    'PatientBirthDate',
    'PatientBirthTime',
    'PatientSex',
    'PatientAge',
    'PatientWeight',
    'PatientSize',
    'PatientAddress',
    'PatientTelephoneNumbers',
    'PatientMotherBirthName',
    'OtherPatientIDs',
    'OtherPatientNames',
    'EthnicGroup',
    'PatientReligiousPreference',
    'PatientComments',
    'PatientState',
    # Study identifiers
    'StudyID',
    'AccessionNumber',
    'StudyDate',
    'StudyTime',
    'AcquisitionDate',
    'AcquisitionTime',
    'ContentDate',
    'ContentTime',
    'SeriesDate',
    'SeriesTime',
    'InstanceCreationDate',
    'InstanceCreationTime',
    # Institution identifiers
    'InstitutionName',
    'InstitutionAddress',
    'InstitutionalDepartmentName',
    'StationName',
    # Physician identifiers
    'ReferringPhysicianName',
    'PerformingPhysicianName',
    'NameOfPhysiciansReadingStudy',
    'OperatorsName',
    'PhysiciansOfRecord',
    # Other identifiers
    'RequestingPhysician',
    'RequestAttributesSequence',
    'DeviceSerialNumber',
    'PlateID',
    'CassetteID',
    'GantryID',
    # UIDs that could be used for re-identification
    'StudyInstanceUID',
    'SeriesInstanceUID',
    'SOPInstanceUID',
    'FrameOfReferenceUID',
    'MediaStorageSOPInstanceUID',
    # Private tags and comments
    'ImageComments',
    'AdditionalPatientHistory',
    'RequestedProcedureDescription',
    'PerformedProcedureStepDescription',
]

# Fields to anonymize in BIDS JSON sidecars
PHI_BIDS_FIELDS = [
    'InstitutionName',
    'InstitutionAddress',
    'InstitutionalDepartmentName',
    'StationName',
    'DeviceSerialNumber',
    'PatientPosition',  # Keep but check if identifiable
    'AcquisitionTime',
    'StudyDescription',
    'ProcedureStepDescription',
]

# Mapping of DICOM series descriptions to BIDS modalities
# Note: T1CE is detected via ceagent entity, not suffix
MODALITY_MAPPING = {
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

# Keywords that indicate contrast enhancement in series descriptions
CONTRAST_KEYWORDS = [
    'POST', '+C', 'CONTRAST', 'GAD', 'GADOLINIUM', 'CE', 'C+',
    'ENHANCED', 'POST-CONTRAST', 'POST_CONTRAST', 'POSTCONTRAST',
    'T1CE', 'T1_CE', 'T1 CE', 'T1+C', 'T1C', 'T1POST',
]


def detect_contrast_enhancement(
    series_name: str,
    dicom_dir: Optional[Path] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Detect if a series has contrast enhancement.
    
    Checks both series description and DICOM metadata (ContrastBolusAgent).
    
    Parameters
    ----------
    series_name : str
        DICOM series description or directory name
    dicom_dir : Optional[Path]
        Directory containing DICOM files for metadata extraction
        
    Returns
    -------
    Tuple[bool, Optional[str]]
        (is_contrast_enhanced, contrast_agent_name)
        contrast_agent_name is normalized for BIDS ceagent entity
    """
    upper_name = series_name.upper()
    
    # Check series description for contrast keywords
    for keyword in CONTRAST_KEYWORDS:
        if keyword in upper_name:
            # Try to determine contrast agent from DICOM metadata
            agent = _extract_contrast_agent(dicom_dir) if dicom_dir else None
            return True, agent or 'gd'  # Default to 'gd' (gadolinium)
    
    # Check DICOM metadata for ContrastBolusAgent
    if dicom_dir:
        agent = _extract_contrast_agent(dicom_dir)
        if agent:
            return True, agent
    
    return False, None


def _extract_contrast_agent(dicom_dir: Path) -> Optional[str]:
    """
    Extract contrast agent name from DICOM metadata.
    
    Parameters
    ----------
    dicom_dir : Path
        Directory containing DICOM files
        
    Returns
    -------
    Optional[str]
        Normalized contrast agent name for BIDS ceagent entity
    """
    try:
        import pydicom
        
        # Find first DICOM file
        dicom_files = sorted(dicom_dir.glob('*.dcm')) + sorted(dicom_dir.glob('*.IMA'))
        if not dicom_files:
            # Try files without extensions (common in some DICOM archives)
            dicom_files = [f for f in dicom_dir.iterdir() if f.is_file() and not f.name.startswith('.')]
        
        if not dicom_files:
            return None
        
        dcm = pydicom.dcmread(str(dicom_files[0]), stop_before_pixels=True)
        
        # Check ContrastBolusAgent (0018, 0010)
        if hasattr(dcm, 'ContrastBolusAgent') and dcm.ContrastBolusAgent:
            agent = str(dcm.ContrastBolusAgent).strip()
            return _normalize_contrast_agent(agent)
        
        # Check if contrast was used from ContrastBolusVolume or Route
        if hasattr(dcm, 'ContrastBolusVolume') and dcm.ContrastBolusVolume:
            return 'gd'  # Assume gadolinium if contrast was given but agent not specified
        
        if hasattr(dcm, 'ContrastBolusRoute') and dcm.ContrastBolusRoute:
            return 'gd'  # Assume gadolinium if contrast route specified
            
    except ImportError:
        LOGGER.debug("pydicom not available for contrast agent detection")
    except Exception as e:
        LOGGER.debug(f"Could not extract contrast agent: {e}")
    
    return None


def _normalize_contrast_agent(agent: str) -> str:
    """
    Normalize contrast agent name for BIDS ceagent entity.
    
    BIDS entity values must be alphanumeric.
    
    Parameters
    ----------
    agent : str
        Raw contrast agent name from DICOM
        
    Returns
    -------
    str
        Normalized agent name (alphanumeric, lowercase)
    """
    # Common gadolinium-based agents
    gd_agents = [
        'gadolinium', 'gadovist', 'dotarem', 'prohance', 'magnevist',
        'omniscan', 'multihance', 'eovist', 'gadavist', 'clariscan',
        'gd-dtpa', 'gd-dota', 'gd-bopta',
    ]
    
    agent_lower = agent.lower()
    
    # Check if it's a known gadolinium agent
    for gd_name in gd_agents:
        if gd_name in agent_lower or agent_lower in gd_name:
            # Use specific agent name if recognizable
            agent_clean = re.sub(r'[^a-zA-Z0-9]', '', agent_lower)
            if len(agent_clean) > 2:
                return agent_clean
            return 'gd'
    
    # Generic normalization: remove non-alphanumeric, lowercase
    normalized = re.sub(r'[^a-zA-Z0-9]', '', agent).lower()
    
    # If result is too short or empty, default to 'gd'
    if len(normalized) < 2:
        return 'gd'
    
    return normalized


def anonymize_bids_sidecar(json_path: Path, remove_fields: bool = True) -> None:
    """
    Remove PHI (Protected Health Information) from BIDS JSON sidecar.
    
    Parameters
    ----------
    json_path : Path
        Path to BIDS JSON sidecar file
    remove_fields : bool
        If True, remove PHI fields entirely. If False, replace with placeholder.
    """
    if not json_path.exists():
        return
    
    try:
        with open(json_path, 'r') as f:
            sidecar = json.load(f)
        
        modified = False
        for field in PHI_BIDS_FIELDS:
            if field in sidecar:
                if remove_fields:
                    del sidecar[field]
                else:
                    sidecar[field] = 'ANONYMIZED'
                modified = True
        
        if modified:
            with open(json_path, 'w') as f:
                json.dump(sidecar, f, indent=4)
            LOGGER.debug(f"Anonymized BIDS sidecar: {json_path.name}")
            
    except Exception as e:
        LOGGER.warning(f"Could not anonymize sidecar {json_path}: {e}")


def anonymize_dicom_series(dicom_dir: Path) -> bool:
    """
    Anonymize DICOM files in a directory by removing PHI fields.
    
    WARNING: This modifies the original DICOM files. Make a backup first.
    
    Parameters
    ----------
    dicom_dir : Path
        Directory containing DICOM files
        
    Returns
    -------
    bool
        True if anonymization was successful
    """
    try:
        import pydicom
        
        dicom_files = sorted(dicom_dir.glob('*.dcm')) + sorted(dicom_dir.glob('*.IMA'))
        if not dicom_files:
            dicom_files = [f for f in dicom_dir.iterdir() 
                          if f.is_file() and not f.name.startswith('.')]
        
        for dcm_file in dicom_files:
            try:
                dcm = pydicom.dcmread(str(dcm_file))
                
                for field in PHI_DICOM_FIELDS:
                    if hasattr(dcm, field):
                        delattr(dcm, field)
                
                # Generate new anonymous UIDs
                dcm.StudyInstanceUID = pydicom.uid.generate_uid()
                dcm.SeriesInstanceUID = pydicom.uid.generate_uid()
                dcm.SOPInstanceUID = pydicom.uid.generate_uid()
                
                dcm.save_as(str(dcm_file))
                
            except Exception as e:
                LOGGER.warning(f"Could not anonymize {dcm_file.name}: {e}")
                
        LOGGER.info(f"✓ Anonymized {len(dicom_files)} DICOM files in {dicom_dir.name}")
        return True
        
    except ImportError:
        LOGGER.error("pydicom not available for DICOM anonymization")
        return False
    except Exception as e:
        LOGGER.error(f"DICOM anonymization failed: {e}")
        return False


def infer_modality_from_series(
    series_name: str,
    dicom_dir: Optional[Path] = None,
) -> Tuple[Optional[str], bool, Optional[str]]:
    """
    Infer BIDS modality from DICOM series description.
    
    Parameters
    ----------
    series_name : str
        DICOM series description or directory name
    dicom_dir : Optional[Path]
        Directory containing DICOM files for metadata extraction
        
    Returns
    -------
    Tuple[Optional[str], bool, Optional[str]]
        (BIDS modality suffix, is_contrast_enhanced, contrast_agent)
        For contrast-enhanced T1w: returns ('T1w', True, 'gd')
    """
    return _infer_modality_impl(series_name, dicom_dir)


def _infer_modality_impl(
    series_name: str,
    dicom_dir: Optional[Path] = None,
) -> Tuple[Optional[str], bool, Optional[str]]:
    """Implementation of modality inference."""
    upper_name = series_name.upper()
    
    # Check for contrast enhancement first (for T1w images)
    is_contrast, agent = detect_contrast_enhancement(series_name, dicom_dir)
    
    # Check for T1CE/T1 CE keywords - these are contrast-enhanced T1w
    for key in ['T1CE', 'T1_CE', 'T1 CE', 'T1+C', 'T1POST', 'T1_POST']:
        if key in upper_name:
            return 'T1w', True, agent or 'gd'
    
    # Check for T1w with contrast enhancement
    if any(k in upper_name for k in ['T1W', 'T1', 'MPRAGE', 'SPGR', 'BRAVO']):
        if is_contrast:
            return 'T1w', True, agent
        # Check for POST in T1 series name (common convention)
        if 'POST' in upper_name:
            return 'T1w', True, agent or 'gd'
        return 'T1w', False, None
    
    # Check for other modalities
    for key, modality in MODALITY_MAPPING.items():
        if key in upper_name:
            return modality, False, None
    
    # Fallback to default
    return None, False, None


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
        import pydicom
        
        # Find first DICOM file
        dicom_files = sorted(dicom_dir.glob('*.dcm')) + sorted(dicom_dir.glob('*.IMA'))
        
        if not dicom_files:
            # Try files without extensions
            dicom_files = [f for f in dicom_dir.iterdir() 
                          if f.is_file() and not f.name.startswith('.')]
        
        if not dicom_files:
            LOGGER.warning(f"No DICOM files found in {dicom_dir}")
            return False
        
        # Load DICOM
        dcm = pydicom.dcmread(str(dicom_files[0]))
        
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
        import pydicom
        
        # Find first DICOM file
        dicom_files = sorted(dicom_dir.glob('*.dcm')) + sorted(dicom_dir.glob('*.IMA'))
        if not dicom_files:
            # Try files without extensions
            dicom_files = [f for f in dicom_dir.iterdir() 
                          if f.is_file() and not f.name.startswith('.')]
        if not dicom_files:
            return metadata
        
        dcm = pydicom.dcmread(str(dicom_files[0]), stop_before_pixels=True)
        
        # Extract key parameters for BIDS using attribute names
        key_mappings = {
            'RepetitionTime': 'RepetitionTime',
            'EchoTime': 'EchoTime',
            'FlipAngle': 'FlipAngle',
            'MagneticFieldStrength': 'MagneticFieldStrength',
            'EchoTrainLength': 'EchoTrainLength',
            'SeriesNumber': 'SeriesNumber',
            'SeriesDescription': 'SeriesDescription',
            'SequenceName': 'SequenceName',
            'InversionTime': 'InversionTime',
            'PatientPosition': 'PatientPosition',
            'ContrastBolusAgent': 'ContrastBolusAgent',
            'Manufacturer': 'Manufacturer',
            'ManufacturerModelName': 'ManufacturerModelName',
        }
        
        for bids_key, dicom_attr in key_mappings.items():
            try:
                if hasattr(dcm, dicom_attr):
                    value = getattr(dcm, dicom_attr)
                    # Convert to appropriate type
                    if isinstance(value, (int, float)):
                        metadata[bids_key] = float(value) if '.' in str(value) else int(value)
                    elif value:
                        metadata[bids_key] = str(value).strip()
            except (KeyError, ValueError, TypeError):
                pass
        
        LOGGER.debug(f"Extracted metadata: {metadata}")
        
    except ImportError:
        LOGGER.debug("pydicom not available for metadata extraction")
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
    anonymize: bool = True,
    overwrite: bool = False,
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
    anonymize : bool
        If True, remove PHI from BIDS JSON sidecars (default: True)
    overwrite : bool
        If True, overwrite existing output files (default: False)
        
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
            # Try files without extensions (common in some DICOM archives)
            potential_dicoms = [f for f in series_dir.iterdir() 
                               if f.is_file() and not f.name.startswith('.')]
            if not potential_dicoms:
                continue
        
        LOGGER.info(f"Processing DICOM series: {series_dir.name}")
        
        # Infer modality with contrast enhancement detection
        modality, is_contrast, contrast_agent = infer_modality_from_series(
            series_dir.name, 
            dicom_dir=series_dir,
        )
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
        
        # Construct BIDS filename with ce- entity for contrast-enhanced images
        # BIDS uses 'ce-' as the entity key for contrast agent (not 'ceagent-')
        if session:
            if is_contrast and contrast_agent:
                # Use ce entity: sub-XX_ses-XX_ce-gd_T1w.nii.gz
                bids_basename = f'sub-{subject}_ses-{session}_ce-{contrast_agent}_{modality}'
                LOGGER.info(f"  → Detected contrast enhancement: ce-{contrast_agent}")
            else:
                bids_basename = f'sub-{subject}_ses-{session}_{modality}'
        else:
            if is_contrast and contrast_agent:
                # Use ce entity: sub-XX_ce-gd_T1w.nii.gz
                bids_basename = f'sub-{subject}_ce-{contrast_agent}_{modality}'
                LOGGER.info(f"  → Detected contrast enhancement: ce-{contrast_agent}")
            else:
                bids_basename = f'sub-{subject}_{modality}'
        
        output_nifti = datatype_dir / f'{bids_basename}.nii.gz'
        
        # Handle existing files
        if output_nifti.exists():
            if overwrite:
                LOGGER.info(f"Overwriting existing file: {output_nifti.name}")
                output_nifti.unlink()
                # Also remove associated JSON sidecar if exists
                json_sidecar = output_nifti.with_suffix('').with_suffix('.json')
                if json_sidecar.exists():
                    json_sidecar.unlink()
            else:
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
                # Add contrast agent info to sidecar
                if is_contrast and contrast_agent:
                    metadata['ContrastBolusAgent'] = contrast_agent
                create_bids_sidecar(output_nifti, metadata)
            
            # Anonymize BIDS sidecar (remove PHI fields)
            if anonymize:
                json_path = output_nifti.with_suffix('').with_suffix('.json')
                anonymize_bids_sidecar(json_path)
            
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
    
    LOGGER.info("✓ Created BIDS dataset_description.json")
    return desc_file
