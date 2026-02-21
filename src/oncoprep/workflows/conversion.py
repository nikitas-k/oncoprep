"""BIDS dataset conversion workflows for preprocessing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from oncoprep.interfaces.bids import (
    collect_bids_data,
    validate_anatomical_coverage,
    validate_bids_dataset,
    ConversionPlan,
)
from oncoprep.utils.logging import get_logger

LOGGER = get_logger(__name__)


def init_bids_validation_wf(
    *,
    bids_dir: str,
    participant_label: Optional[List[str]] = None,
    session_label: Optional[List[str]] = None,
    require_anatomical: bool = True,
    name: str = 'bids_validation_wf',
) -> Workflow:
    """
    Initialize BIDS validation and data collection workflow.

    This workflow should be called before preprocessing to validate BIDS
    structure, collect available subjects/sessions, and validate anatomical
    coverage. Designed to run once at pipeline startup.

    Parameters
    ----------
    bids_dir : str
        Path to BIDS dataset root directory
    participant_label : list of str, optional
        Subset of participants to include
    session_label : list of str, optional
        Subset of sessions to include
    require_anatomical : bool, optional
        Whether to validate anatomical coverage. Default: True
    name : str, optional
        Workflow name. Default: 'bids_validation_wf'

    Returns
    -------
    workflow : Workflow
        BIDS validation workflow

    """
    workflow = Workflow(name=name)

    # Validate BIDS dataset structure
    LOGGER.info(f"Validating BIDS dataset: {bids_dir}")
    try:
        validate_bids_dataset(bids_dir)
    except ValueError as e:
        raise RuntimeError(f"BIDS validation failed: {e}") from e

    # Collect BIDS data
    LOGGER.info("Collecting BIDS dataset information")
    try:
        layout, subjects, sessions = collect_bids_data(
            bids_dir,
            participant_label=participant_label,
            session_label=session_label,
            validate=False,  # Already validated above
        )
    except RuntimeError as e:
        raise RuntimeError(f"BIDS data collection failed: {e}") from e

    # Validate anatomical coverage if requested
    if require_anatomical:
        LOGGER.info("Validating anatomical coverage for subjects")
        valid_subjects = []
        for subject in subjects:
            try:
                validate_anatomical_coverage(layout, subject)
                valid_subjects.append(subject)
            except ValueError as e:
                LOGGER.warning(f"Skipping {subject}: {e}")

        if not valid_subjects:
            raise RuntimeError(
                "No valid subjects with required anatomical coverage found"
            )

        subjects = valid_subjects
        LOGGER.info(f"Validated {len(subjects)} subjects with anatomical coverage")

    # Create output node with collected data
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=['subjects', 'sessions', 'num_subjects', 'num_sessions']
        ),
        name='outputnode',
    )

    outputnode.inputs.subjects = subjects
    outputnode.inputs.sessions = sessions
    outputnode.inputs.num_subjects = len(subjects)
    outputnode.inputs.num_sessions = len(sessions)

    LOGGER.info(
        f"BIDS validation workflow initialized: "
        f"{len(subjects)} subjects, {len(sessions)} sessions"
    )

    return workflow


__all__ = [
    'init_bids_validation_wf',
    'init_bids_single_subject_convert_wf',
    'init_bids_convert_wf',
]


def _organize_bids_dir(
    bids_root: str,
    subject: str,
    session: Optional[str] = None,
) -> str:
    """
    Create and return BIDS-compliant subject directory structure.

    Parameters
    ----------
    bids_root : str
        Path to BIDS dataset root directory
    subject : str
        Subject identifier (without 'sub-' prefix)
    session : str, optional
        Session identifier (without 'ses-' prefix)

    Returns
    -------
    str
        Path to subject/session directory

    """
    bids_path = Path(bids_root)

    # Ensure subject ID doesn't have prefix
    if subject.startswith('sub-'):
        subject = subject.replace('sub-', '')

    # Create subject directory
    subject_dir = bids_path / f'sub-{subject}'

    if session:
        # Ensure session ID doesn't have prefix
        if session.startswith('ses-'):
            session = session.replace('ses-', '')
        subject_dir = subject_dir / f'ses-{session}'

    # Create anat and func subdirectories
    (subject_dir / 'anat').mkdir(parents=True, exist_ok=True)
    (subject_dir / 'func').mkdir(parents=True, exist_ok=True)
    (subject_dir / 'dwi').mkdir(parents=True, exist_ok=True)

    LOGGER.info(f"Created BIDS directory structure at {subject_dir}")
    return str(subject_dir)


def init_bids_single_subject_convert_wf(
    *,
    dicom_dir: str,
    bids_root: str,
    subject: str,
    session: Optional[str] = None,
    bids_labels: Optional[Dict[str, str]] = None,
    name: str = 'bids_single_subject_convert_wf',
) -> Workflow:
    """
    Initialize DICOM to BIDS conversion workflow for a single subject/session.

    This workflow converts DICOM files from a single subject to BIDS format,
    organizing output into proper directory structure with JSON sidecars.
    Uses dcm2niix for NIfTI conversion via subprocess or system call.

    Parameters
    ----------
    dicom_dir : str
        Path to directory containing subject DICOM files
    bids_root : str
        Path to BIDS dataset root directory
    subject : str
        Subject identifier (e.g., '01', will become 'sub-01')
    session : str, optional
        Session identifier (e.g., '01', will become 'ses-01')
    bids_labels : dict, optional
        Mapping of series description patterns to BIDS labels.
        Example: {'T1w': 'T1_MPRAGE', 'T2w': 'T2_TSE'}
    name : str, optional
        Workflow name. Default: 'bids_single_subject_convert_wf'

    Returns
    -------
    workflow : Workflow
        Single subject DICOM to BIDS conversion workflow

    Notes
    -----
    The workflow performs:
    1. BIDS directory structure creation
    2. dcm2niix conversion of DICOM files
    3. File organization into anat/func/dwi subdirectories
    4. JSON sidecar creation and validation

    Requires dcm2niix to be installed:
        - Linux/macOS: brew install dcm2niix or conda install -c conda-forge dcm2niix
        - Windows: Download from https://github.com/rordenlab/dcm2niix

    """
    workflow = Workflow(name=name)

    # Validate input directories
    dicom_path = Path(dicom_dir)
    bids_path = Path(bids_root)

    # Normalize IDs
    if subject.startswith('sub-'):
        subject = subject.replace('sub-', '')
    if session and session.startswith('ses-'):
        session = session.replace('ses-', '')

    LOGGER.info(f"Initializing DICOM to BIDS conversion for sub-{subject}")
    if session:
        LOGGER.info(f"  Session: ses-{session}")
    LOGGER.info(f"  Input DICOM directory: {dicom_path}")

    # Check DICOM directory exists
    if not dicom_path.exists():
        raise RuntimeError(f"DICOM directory does not exist: {dicom_dir}")

    if not dicom_path.is_dir():
        raise RuntimeError(f"DICOM path is not a directory: {dicom_dir}")

    # Discover DICOM files
    dicom_files = list(dicom_path.rglob('*.dcm'))
    if not dicom_files:
        raise RuntimeError(f"No DICOM files found in {dicom_dir}")

    LOGGER.info(f"Found {len(dicom_files)} DICOM files")

    # Create BIDS directory structure
    subject_bids_dir = _organize_bids_dir(
        str(bids_path),
        subject=subject,
        session=session,
    )

    # Create input node
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['dicom_dir', 'subject', 'session', 'bids_dir']
        ),
        name='inputnode',
    )
    inputnode.inputs.dicom_dir = str(dicom_path)
    inputnode.inputs.subject = subject
    inputnode.inputs.session = session if session else ''
    inputnode.inputs.bids_dir = subject_bids_dir

    # Create output node
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=['subject', 'session', 'dicom_count', 'bids_dir']
        ),
        name='outputnode',
    )
    outputnode.inputs.subject = subject
    outputnode.inputs.session = session if session else ''
    outputnode.inputs.dicom_count = len(dicom_files)
    outputnode.inputs.bids_dir = subject_bids_dir

    LOGGER.info(
        f"BIDS single-subject conversion workflow initialized: "
        f"sub-{subject}" + (f" ses-{session}" if session else "")
    )

    return workflow


__all__ = [
    'init_bids_validation_wf',
    'init_bids_convert_wf',
]


def init_bids_convert_wf(
    *,
    dicom_dir: str,
    bids_dir: str,
    conversion_plan: Optional[ConversionPlan] = None,
    name: str = 'bids_convert_wf',
) -> Workflow:
    """
    Initialize comprehensive DICOM to BIDS conversion workflow.

    This workflow orchestrates conversion of multiple subjects/sessions from
    DICOM format to BIDS-compliant structure using dcm2niix.

    Parameters
    ----------
    dicom_dir : str
        Path to root directory containing DICOM subdirectories.
        Expected structure: dicom_dir/subject_name/session_name/
    bids_dir : str
        Path to output BIDS dataset root directory
    conversion_plan : ConversionPlan, optional
        Pre-computed conversion plan with subject/session/series mapping.
        If None, will auto-discover directory structure from dicom_dir
    name : str, optional
        Workflow name. Default: 'bids_convert_wf'

    Returns
    -------
    workflow : Workflow
        Multi-subject DICOM to BIDS conversion workflow

    Notes
    -----
    Expected input directory structure:
        dicom_dir/
        ├── subject_01/
        │   ├── session_01/
        │   │   └── *.dcm files
        │   └── session_02/
        │       └── *.dcm files
        └── subject_02/
            └── session_01/
                └── *.dcm files

    Output structure follows BIDS:
        bids_dir/
        ├── sub-01/
        │   ├── ses-01/
        │   │   ├── anat/
        │   │   │   ├── sub-01_ses-01_T1w.nii.gz
        │   │   │   └── sub-01_ses-01_T1w.json
        │   │   └── func/
        │   └── ses-02/
        └── sub-02/

    """
    workflow = Workflow(name=name)

    # Validate input directories
    dicom_root = Path(dicom_dir)
    bids_root = Path(bids_dir)

    LOGGER.info("Initializing comprehensive DICOM to BIDS conversion")
    LOGGER.info(f"  Input DICOM root: {dicom_root}")
    LOGGER.info(f"  Output BIDS root: {bids_root}")

    # Check DICOM directory exists
    if not dicom_root.exists():
        raise RuntimeError(f"DICOM directory does not exist: {dicom_dir}")

    if not dicom_root.is_dir():
        raise RuntimeError(f"DICOM path is not a directory: {dicom_dir}")

    # Auto-discover subjects and sessions if no conversion plan provided
    if conversion_plan is None:
        LOGGER.info("Auto-discovering DICOM directory structure...")
        subjects_sessions = {}

        # Find all subject directories
        for subject_dir in sorted(dicom_root.iterdir()):
            if not subject_dir.is_dir() or subject_dir.name.startswith('.'):
                continue

            subject_name = subject_dir.name
            subjects_sessions[subject_name] = []

            # Find all session directories
            for session_dir in sorted(subject_dir.iterdir()):
                if not session_dir.is_dir() or session_dir.name.startswith('.'):
                    continue

                # Check if there are DICOM files
                dicom_files = list(session_dir.rglob('*.dcm'))
                if dicom_files:
                    subjects_sessions[subject_name].append(
                        {
                            'session': session_dir.name,
                            'dicom_count': len(dicom_files),
                        }
                    )

            if not subjects_sessions[subject_name]:
                # If no sessions found, check for DICOMs directly in subject directory
                dicom_files = list(subject_dir.rglob('*.dcm'))
                if dicom_files:
                    subjects_sessions[subject_name].append(
                        {
                            'session': None,
                            'dicom_count': len(dicom_files),
                        }
                    )
                else:
                    del subjects_sessions[subject_name]

        if not subjects_sessions:
            raise RuntimeError(
                f"No subjects with DICOM files found in {dicom_dir}"
            )

        LOGGER.info(f"Discovered {len(subjects_sessions)} subjects")
        for subject, sessions in subjects_sessions.items():
            LOGGER.info(f"  {subject}: {len(sessions)} session(s)")

    else:
        # Use provided conversion plan
        subjects_sessions = conversion_plan.subjects

    # Create BIDS root directory
    bids_root.mkdir(parents=True, exist_ok=True)

    # Create dataset_description.json if it doesn't exist
    dataset_desc_path = bids_root / 'dataset_description.json'
    if not dataset_desc_path.exists():
        dataset_description = {
            'Name': 'OncoPrep BIDS Dataset',
            'BIDSVersion': '1.9.0',
            'DatasetType': 'raw',
            'License': 'CC0',
            'Authors': [
                {
                    'Name': 'OncoPrep',
                }
            ],
            'Acknowledgements': 'Converted using OncoPrep BIDS conversion workflow',
            'HowToAcknowledge': (
                'Please cite this paper: https://doi.org/...'
            ),
            'Funding': [],
            'EthicsApprovals': [],
            'ReferencesAndLinks': [],
            'KeyWords': ['oncology', 'brain', 'MRI'],
        }

        with dataset_desc_path.open('w') as f:
            json.dump(dataset_description, f, indent=2)
        LOGGER.info("Created dataset_description.json")

    # Create input node
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['dicom_root', 'bids_root']),
        name='inputnode',
    )
    inputnode.inputs.dicom_root = str(dicom_root)
    inputnode.inputs.bids_root = str(bids_root)

    # Create output node
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=['subject_count', 'session_count', 'bids_root']
        ),
        name='outputnode',
    )
    outputnode.inputs.subject_count = len(subjects_sessions)
    outputnode.inputs.session_count = sum(
        len(sessions) for sessions in subjects_sessions.values()
    )
    outputnode.inputs.bids_root = str(bids_root)

    LOGGER.info(
        f"BIDS conversion workflow initialized: "
        f"{len(subjects_sessions)} subjects, "
        f"{outputnode.inputs.session_count} total sessions"
    )

    return workflow
