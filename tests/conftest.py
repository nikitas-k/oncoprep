"""Pytest configuration and fixtures for OncoPrep tests."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Generator

import pytest
from oncoprep.utils.logging import get_logger

LOGGER = get_logger(__name__)


@pytest.fixture(scope="session")
def example_data_dir() -> Generator[Path, None, None]:
    """
    Get path to local example data (DICOM or NIfTI).

    Uses example data stored in ./examples/data directory when available.
    When the real data is absent (e.g. on CI), creates a temporary directory
    with synthetic DICOM-like stub files that satisfy the test expectations.

    Yields
    ------
    Path
        Path to the directory containing example data files
    """
    data_dir = Path(__file__).parent.parent / "examples" / "data"

    # Check whether *real* example data is usable
    has_real_data = False
    if data_dir.exists():
        dicom_files = list(data_dir.glob("**/*.IMA")) + list(data_dir.glob("**/*.dcm"))
        nifti_files = list(data_dir.glob("**/*.nii.gz"))
        if dicom_files or nifti_files:
            has_real_data = True

    if has_real_data:
        file_type = "DICOM" if dicom_files else "NIfTI"
        file_count = len(dicom_files) if dicom_files else len(nifti_files)
        LOGGER.info(f"Using local example data: {data_dir} ({file_count} {file_type} files)")
        yield data_dir
        return

    # ---- Spoof synthetic data for CI / environments without real data ----
    temp_root = Path(tempfile.mkdtemp(prefix="oncoprep_example_data_"))
    LOGGER.info(f"Real example data not found; creating synthetic data in {temp_root}")

    series_dirs = [
        "001/T1_MPRAGE_SAG_P2_1_0_ISO_0032",
        "001/T1_MPRAGE_SAG_P2_1_0_ISO_POST_0071",
        "001/T2_SPACE_SAG_P2_ISO_0013",
        "001/AX_FLAIR_0104",
    ]
    for rel in series_dirs:
        series_path = temp_root / rel
        series_path.mkdir(parents=True, exist_ok=True)
        # Write a small non-empty stub file so size > 0 checks pass
        for i in range(3):
            stub = series_path / f"stub_{i:04d}.dcm"
            stub.write_bytes(b"\x00" * 256)

    LOGGER.info(f"Synthetic example data ready ({len(series_dirs)} series)")

    try:
        yield temp_root
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


@pytest.fixture(scope="session")
def example_dicom_dir() -> Generator[Path, None, None]:
    """
    Download and extract example DICOM data from datalad.

    DEPRECATED: Use example_data_dir() instead for local NIfTI files.
    
    Uses https://github.com/datalad/example-dicom-structural as the data source.
    This fixture runs once per test session and cleans up afterward.

    Yields
    ------
    Path
        Path to the directory containing DICOM files
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="oncoprep_dicom_"))
    LOGGER.info(f"Setting up example DICOM data in {temp_dir}")

    try:
        # Clone the datalad example DICOM structural repository
        repo_url = "https://github.com/datalad/example-dicom-structural"
        repo_path = temp_dir / "example-dicom-structural"

        LOGGER.info(f"Cloning DICOM example repository from {repo_url}")
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(repo_path)],
            check=True,
            capture_output=True,
        )

        # Get the DICOM directory (typically dicomdir or similar)
        dicom_dir = repo_path / "dicomdir"
        if not dicom_dir.exists():
            # Fallback: look for any directory containing DICOM files
            for candidate in repo_path.rglob("*.dcm"):
                dicom_dir = candidate.parent
                break

        if not dicom_dir.exists() or not any(dicom_dir.glob("*.dcm")):
            raise RuntimeError(
                f"Could not find DICOM files in {repo_path}. "
                "Check repository structure at https://github.com/datalad/example-dicom-structural"
            )

        LOGGER.info(f"Found DICOM data at {dicom_dir}")
        yield dicom_dir

    finally:
        # Cleanup
        if temp_dir.exists():
            LOGGER.info(f"Cleaning up temporary DICOM directory: {temp_dir}")
            shutil.rmtree(temp_dir)


@pytest.fixture(scope="function")
def bids_dir(tmp_path: Path) -> Path:
    """
    Create a temporary BIDS directory for testing.

    Parameters
    ----------
    tmp_path : Path
        Pytest's temporary directory fixture

    Returns
    -------
    Path
        Path to the temporary BIDS directory
    """
    bids_root = tmp_path / "bids_dataset"
    bids_root.mkdir(exist_ok=True)

    # Create minimal BIDS structure
    dataset_description = {
        "Name": "Test OncoPrep Dataset",
        "BIDSVersion": "1.9.0",
        "DatasetType": "raw",
        "License": "CC0",
        "Authors": [{"name": "Test"}],
        "Acknowledgements": "OncoPrep Test Suite",
        "HowToAcknowledge": "Cite OncoPrep",
        "Funding": [],
        "EthicsApprovals": [],
        "ReferencesAndLinks": [],
        "DatasetLinks": {},
        "Keywords": ["neuro-oncology", "MRI"],
        "SourceDatasets": [],
        "ConsentLinks": [],
    }

    with open(bids_root / "dataset_description.json", "w") as f:
        json.dump(dataset_description, f, indent=2)

    LOGGER.info(f"Created BIDS directory at {bids_root}")
    return bids_root


@pytest.fixture(scope="function")
def output_dir(tmp_path: Path) -> Path:
    """
    Create a temporary output directory for derivatives.

    Parameters
    ----------
    tmp_path : Path
        Pytest's temporary directory fixture

    Returns
    -------
    Path
        Path to the temporary output directory
    """
    output_root = tmp_path / "derivatives"
    output_root.mkdir(exist_ok=True)
    LOGGER.info(f"Created output directory at {output_root}")
    return output_root


@pytest.fixture(scope="function")
def work_dir(tmp_path: Path) -> Path:
    """
    Create a temporary work directory for workflow execution.

    Parameters
    ----------
    tmp_path : Path
        Pytest's temporary directory fixture

    Returns
    -------
    Path
        Path to the temporary work directory
    """
    work_root = tmp_path / "work"
    work_root.mkdir(exist_ok=True)
    LOGGER.info(f"Created work directory at {work_root}")
    return work_root


@pytest.fixture
def nipype_config():
    """Configure Nipype for testing."""
    from nipype import config

    # Disable FSL warnings during testing
    config.set_default_config()
    config.update_config({"execution": {"remove_unnecessary_outputs": False}})

    yield config

    # Reset after test
    config.set_default_config()
