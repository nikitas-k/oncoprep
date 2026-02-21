"""BIDS dataset utilities and interfaces."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from bids.layout import BIDSLayout as BIDSLayoutBackend
from oncoprep.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class DICOMSeries:
    """Class representing a DICOM series."""

    series_instance_uid: str
    series_description: str
    file_paths: List[Path]
    modality: Optional[str] = None


@dataclass(frozen=True)
class ConversionPlan:
    """Class representing a DICOM to BIDS conversion plan."""

    dicom_root: Path
    bids_root: Path
    subjects: dict[str, dict[str, List[DICOMSeries]]]
    unique_series: Optional[List] = None


# BIDS dataset validation and data collection utilities


def validate_bids_dataset(bids_dir: str) -> bool:
    """
    Validate BIDS dataset structure and required files.

    Checks for:
    - Valid BIDS directory structure
    - dataset_description.json file
    - At least one valid subject directory
    - Required JSON fields in descriptor

    Parameters
    ----------
    bids_dir : str
        Path to BIDS dataset root directory

    Returns
    -------
    bool
        True if dataset is valid BIDS structure

    Raises
    ------
    ValueError
        If dataset is not valid BIDS format

    """
    bids_path = Path(bids_dir)

    # Check BIDS root exists
    if not bids_path.exists():
        raise ValueError(f"BIDS directory does not exist: {bids_dir}")

    if not bids_path.is_dir():
        raise ValueError(f"BIDS path is not a directory: {bids_dir}")

    # Check for dataset_description.json
    desc_file = bids_path / 'dataset_description.json'
    if not desc_file.exists():
        raise ValueError(f"Missing dataset_description.json in {bids_dir}")

    try:
        with desc_file.open() as f:
            desc = json.load(f)
        if 'Name' not in desc or 'BIDSVersion' not in desc:
            raise ValueError("dataset_description.json missing required fields")
    except (json.JSONDecodeError, OSError) as e:
        raise ValueError(f"Invalid dataset_description.json: {e}") from e

    # Check for valid subject directories
    subject_dirs = [d for d in bids_path.glob('sub-*') if d.is_dir()]
    if not subject_dirs:
        raise ValueError(f"No subject directories (sub-*) found in {bids_dir}")

    LOGGER.info(f"BIDS dataset validation passed: {len(subject_dirs)} subjects found")
    return True


def collect_bids_data(
    bids_dir: str,
    participant_label: Optional[List[str]] = None,
    session_label: Optional[List[str]] = None,
    validate: bool = True,
) -> Tuple[BIDSLayoutBackend, List[str], List[str]]:
    """
    Collect BIDS dataset information and initialize layout.

    Performs BIDS validation and collects available subjects/sessions.
    Returns a BIDSLayout object for data queries and lists of available subjects
    and sessions for filtering and workflow initialization.

    Parameters
    ----------
    bids_dir : str
        Path to BIDS dataset root directory
    participant_label : list of str, optional
        Subset of participants to include. If None, all subjects included
    session_label : list of str, optional
        Subset of sessions to include. If None, all sessions included
    validate : bool, optional
        Whether to validate BIDS structure first. Default: True

    Returns
    -------
    layout : BIDSLayout
        BIDS layout object for querying dataset
    subjects : list of str
        List of subject identifiers to process
    sessions : list of str
        List of session identifiers (empty if none in dataset)

    Raises
    ------
    ValueError
        If dataset is not valid BIDS format
    RuntimeError
        If no data found for specified participants/sessions

    """
    if validate:
        validate_bids_dataset(bids_dir)

    LOGGER.info(f"Initializing BIDS layout for {bids_dir}")
    layout = BIDSLayoutBackend(bids_dir, validate=False)

    # Get all available subjects
    all_subjects = sorted(layout.get_subjects())
    if not all_subjects:
        raise RuntimeError(f"No subjects found in BIDS dataset: {bids_dir}")

    # Filter to requested subjects
    if participant_label:
        # Remove 'sub-' prefix if present for comparison
        requested = [p.replace('sub-', '') for p in participant_label]
        subjects = [s for s in all_subjects if s in requested]
        if not subjects:
            raise RuntimeError(
                f"None of the requested participants found: {participant_label}. "
                f"Available: {all_subjects}"
            )
    else:
        subjects = all_subjects

    # Get all available sessions
    all_sessions = sorted(layout.get_sessions())
    if session_label:
        # Remove 'ses-' prefix if present for comparison
        requested = [s.replace('ses-', '') for s in session_label]
        sessions = [s for s in all_sessions if s in requested]
        if not sessions and all_sessions:
            LOGGER.warning(
                f"None of the requested sessions found: {session_label}. "
                f"Available: {all_sessions}"
            )
            sessions = []
    else:
        sessions = all_sessions

    LOGGER.info(
        f"BIDS data collection complete: {len(subjects)} subjects, "
        f"{len(sessions)} sessions"
    )

    return layout, subjects, sessions


def get_anatomical_files(
    layout: BIDSLayoutBackend,
    subject: str,
    session: Optional[str] = None,
    modalities: Optional[List[str]] = None,
) -> dict:
    """
    Get anatomical files for a subject/session.

    Queries BIDS layout for anatomical images (T1w, T2w, FLAIR, etc.).
    Returns a dictionary with available modalities and their file paths.

    For T1ce (contrast-enhanced T1w), uses the BIDS ce (ceagent) entity
    (e.g., sub-01_ce-gadolinium_T1w.nii.gz).

    Parameters
    ----------
    layout : BIDSLayout
        BIDS layout object
    subject : str
        Subject identifier (with or without 'sub-' prefix)
    session : str, optional
        Session identifier (with or without 'ses-' prefix)
    modalities : list of str, optional
        Specific modalities to query. Default: ['T1w', 'T1ce', 'T2w', 'FLAIR']
        Note: 'T1ce' queries T1w files WITH ceagent entity

    Returns
    -------
    anat_files : dict
        Dictionary mapping modality to list of file paths
        Example: {'T1w': ['/path/sub-01/anat/sub-01_T1w.nii.gz'], ...}

    """
    from bids.layout import Query

    if modalities is None:
        modalities = ['T1w', 'T1ce', 'T2w', 'FLAIR']

    # Remove prefix if present
    subject = subject.replace('sub-', '')
    if session:
        session = session.replace('ses-', '')

    anat_files = {}
    for modality in modalities:
        # Special handling for T1ce: use ceagent entity
        if modality == 'T1ce':
            files = layout.get(
                subject=subject,
                session=session,
                datatype='anat',
                suffix='T1w',
                ceagent=Query.REQUIRED,
                extension='.nii.gz',
                return_type='filename',
            )
        elif modality == 'T1w':
            # Exclude T1w files with ceagent (those are T1ce)
            files = layout.get(
                subject=subject,
                session=session,
                datatype='anat',
                suffix='T1w',
                ceagent=Query.NONE,
                extension='.nii.gz',
                return_type='filename',
            )
        else:
            files = layout.get(
                subject=subject,
                session=session,
                datatype='anat',
                suffix=modality,
                extension='.nii.gz',
                return_type='filename',
            )
        if files:
            anat_files[modality] = sorted(files)

    return anat_files


def get_functional_files(
    layout: BIDSLayoutBackend,
    subject: str,
    session: Optional[str] = None,
    task: Optional[str] = None,
) -> dict:
    """
    Get functional files for a subject/session.

    Queries BIDS layout for fMRI data. Can filter by task if specified.

    Parameters
    ----------
    layout : BIDSLayout
        BIDS layout object
    subject : str
        Subject identifier
    session : str, optional
        Session identifier
    task : str, optional
        Task identifier to filter results

    Returns
    -------
    func_files : dict
        Dictionary with file paths and metadata

    """
    subject = subject.replace('sub-', '')
    if session:
        session = session.replace('ses-', '')

    files = layout.get(
        subject=subject,
        session=session,
        datatype='func',
        suffix='bold',
        task=task,
        extension='.nii.gz',
        return_type='filename',
    )

    return {'bold': sorted(files)} if files else {}


def validate_anatomical_coverage(
    layout: BIDSLayoutBackend,
    subject: str,
    session: Optional[str] = None,
    require_t1w: bool = True,
) -> bool:
    """
    Validate that subject has required anatomical coverage.

    Checks if subject has at least one T1w image and optionally T2w.

    Parameters
    ----------
    layout : BIDSLayout
        BIDS layout object
    subject : str
        Subject identifier
    session : str, optional
        Session identifier
    require_t1w : bool, optional
        Whether T1w is required. Default: True

    Returns
    -------
    bool
        True if anatomical coverage is sufficient

    Raises
    ------
    ValueError
        If required anatomical images are missing

    """
    subject = subject.replace('sub-', '')
    if session:
        session = session.replace('ses-', '')

    anat_files = get_anatomical_files(layout, subject, session)

    if require_t1w and 'T1w' not in anat_files:
        raise ValueError(
            f"Subject sub-{subject} is missing required T1w image"
            + (f" in session ses-{session}" if session else "")
        )

    if not anat_files:
        raise ValueError(
            f"Subject sub-{subject} has no anatomical images"
            + (f" in session ses-{session}" if session else "")
        )

    return True


def get_subjects_sessions(
    bids_dir: str,
    participant_label: Optional[List[str]] = None,
    session_label: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    """
    Get filtered list of subjects and sessions from BIDS dataset.

    Convenience function to quickly retrieve subject/session lists
    without full BIDSLayout initialization.

    Parameters
    ----------
    bids_dir : str
        Path to BIDS dataset
    participant_label : list of str, optional
        Subset of participants
    session_label : list of str, optional
        Subset of sessions

    Returns
    -------
    subjects : list of str
        Subject identifiers
    sessions : list of str
        Session identifiers

    """
    layout, subjects, sessions = collect_bids_data(
        bids_dir,
        participant_label=participant_label,
        session_label=session_label,
    )
    return subjects, sessions


# Custom Nipype interfaces for OncoPrep BIDS handling
from nipype.interfaces.base import (  # noqa: E402
    BaseInterfaceInputSpec,
    DynamicTraitedSpec,
    SimpleInterface,
    traits,
    File,
    InputMultiObject,
)


class _OncoprepBIDSDataGrabberInputSpec(BaseInterfaceInputSpec):
    """Input specification for OncoPrep BIDS data grabber."""
    subject_data = traits.Dict(desc='BIDS subject data dictionary')
    subject_id = traits.Str(desc='Subject identifier')


class _OncoprepBIDSDataGrabberOutputSpec(DynamicTraitedSpec):
    """Output specification for OncoPrep BIDS data grabber."""
    out_dict = traits.Dict(desc='Output data dictionary')
    t1w = InputMultiObject(File(exists=True), desc='T1-weighted images')
    t2w = InputMultiObject(File(exists=True), desc='T2-weighted images')
    t1ce = InputMultiObject(File(exists=True), desc='T1 contrast-enhanced images')
    flair = InputMultiObject(File(exists=True), desc='FLAIR images')
    bold = InputMultiObject(File(exists=True), desc='BOLD images')
    fmap = InputMultiObject(traits.Any, desc='Fieldmap data')
    roi = InputMultiObject(File(exists=True), desc='ROI mask images')
    dwi = InputMultiObject(File(exists=True), desc='Diffusion-weighted images')


class OncoprepBIDSDataGrabber(SimpleInterface):
    """
    Custom BIDS data grabber for OncoPrep that includes t1ce and flair.

    This extends the standard BIDSDataGrabber to include contrast-enhanced
    T1-weighted (t1ce) and FLAIR modalities commonly used in neuro-oncology.

    """

    input_spec = _OncoprepBIDSDataGrabberInputSpec
    output_spec = _OncoprepBIDSDataGrabberOutputSpec

    def _run_interface(self, runtime):
        """Run the interface to grab BIDS data."""
        subject_data = self.inputs.subject_data

        # Standard modalities
        for key in ['t1w', 't1ce', 't2w', 'flair', 'bold', 'fmap', 'roi', 'dwi']:
            self._results[key] = subject_data.get(key, [])

        # Store full dict for reference
        self._results['out_dict'] = {
            **subject_data,
        }

        return runtime


__all__ = [
    'validate_bids_dataset',
    'collect_bids_data',
    'get_anatomical_files',
    'get_functional_files',
    'validate_anatomical_coverage',
    'get_subjects_sessions',
    'DICOMSeries',
    'ConversionPlan',
    'OncoprepBIDSDataGrabber',
]
