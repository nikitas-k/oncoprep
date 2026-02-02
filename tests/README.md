# OncoPrep Tests

Unit and integration tests for OncoPrep BIDS conversion and preprocessing workflows.

## Overview

This test suite demonstrates:

- **BIDS dataset structure** validation and creation
- **DICOM to NIfTI conversion** workflows (with example data from datalad)
- **Anatomical preprocessing** (T1w, T1ce, T2, FLAIR registration and skull-stripping)
- **End-to-end integration** testing with multi-subject datasets

## Test Organization

```
tests/
├── conftest.py              # Pytest configuration and fixtures
├── test_conversion.py       # BIDS conversion tests
├── test_preprocessing.py    # Anatomical preprocessing tests
├── test_integration.py      # End-to-end workflow tests
└── README.md                # This file
```

### conftest.py

Provides shared fixtures:

- `example_dicom_dir`: Clones DICOM example data from https://github.com/datalad/example-dicom-structural
- `bids_dir`: Creates temporary BIDS-compliant directory structure
- `output_dir`: Creates temporary derivatives output directory
- `work_dir`: Creates temporary workflow execution directory
- `nipype_config`: Configures Nipype for testing

### test_conversion.py

Tests for DICOM to BIDS conversion:

- `TestBIDSConversion`: BIDS directory structure, subject/session organization, file naming
- `TestDICOMToNIfTIConversion`: DICOM file discovery, NIfTI conversion (with dcm2niix)

### test_preprocessing.py

Tests for anatomical preprocessing:

- `TestAnatomicalPreprocessing`: Anatomical input preparation, brain masking, template registration setup
- Coverage of T1w, T1ce, T2w, FLAIR modalities
- Multimodal registration input validation
- Defacing option testing

### test_integration.py

End-to-end integration tests:

- `TestIntegrationWorkflow`: Single and multi-subject workflows
- Longitudinal dataset structures (multiple sessions)
- Derivative output validation
- Workflow setup and file collection

## Installation

### Prerequisites

```bash
# From project root directory
cd /Users/nk233/oncoprep

# Required
pip install -e ".[dev]"  # Installs pytest, ruff, and oncoprep

# Optional (for enhanced testing)
pip install nibabel        # NIfTI file I/O
pip install dcm2niix       # DICOM to NIfTI conversion (macOS: brew install dcm2niix)
pip install bids-validator # BIDS dataset validation
```

### Setup Environment

```bash
cd /Users/nk233/oncoprep

# Activate virtual environment
source .venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev]"

# Install optional test dependencies
pip install nibabel dcm2niix bids-validator
```

## Running Tests

### Run All Tests

```bash
# From project root
pytest tests/ -v

# With detailed output
pytest tests/ -v -s

# With coverage
pytest tests/ --cov=oncoprep --cov-report=html
```

### Run Specific Test Files

```bash
# BIDS conversion tests
pytest tests/test_conversion.py -v

# Preprocessing tests
pytest tests/test_preprocessing.py -v

# Integration tests
pytest tests/test_integration.py -v
```

### Run Specific Test Classes

```bash
# BIDS conversion class
pytest tests/test_conversion.py::TestBIDSConversion -v

# Anatomical preprocessing class
pytest tests/test_preprocessing.py::TestAnatomicalPreprocessing -v

# Integration workflow class
pytest tests/test_integration.py::TestIntegrationWorkflow -v
```

### Run Specific Tests

```bash
# Test BIDS directory structure
pytest tests/test_conversion.py::TestBIDSConversion::test_bids_directory_structure -v

# Test brain mask creation
pytest tests/test_preprocessing.py::TestAnatomicalPreprocessing::test_brain_mask_creation -v

# Test multi-subject structure
pytest tests/test_integration.py::TestIntegrationWorkflow::test_multisubject_bids_structure -v
```

### Run Tests with Markers

```bash
# Skip slow/network tests (datalad download)
pytest tests/ -v -m "not slow"

# Run only fast tests
pytest tests/ -v -m "fast"
```

### Run Tests with Options

```bash
# Stop on first failure
pytest tests/ -x

# Show print statements
pytest tests/ -s

# Verbose with full traceback
pytest tests/ -vv --tb=long

# Run in parallel (requires pytest-xdist)
pip install pytest-xdist
pytest tests/ -n auto
```

## Example Output

```
tests/test_conversion.py::TestBIDSConversion::test_bids_directory_structure PASSED
tests/test_conversion.py::TestBIDSConversion::test_create_subject_session_dirs PASSED
tests/test_conversion.py::TestBIDSConversion::test_create_dummy_nifti_files PASSED
tests/test_preprocessing.py::TestAnatomicalPreprocessing::test_create_anatomical_inputs PASSED
tests/test_preprocessing.py::TestAnatomicalPreprocessing::test_brain_mask_creation PASSED
tests/test_integration.py::TestIntegrationWorkflow::test_single_subject_minimal_workflow PASSED
tests/test_integration.py::TestIntegrationWorkflow::test_multisubject_bids_structure PASSED

=================== 7 passed in 3.24s ===================
```

## Test Features

### DICOM Example Data

The `example_dicom_dir` fixture automatically:

1. **Downloads** DICOM data from https://github.com/datalad/example-dicom-structural
2. **Extracts** DICOM files (looks for `.dcm` files)
3. **Cleans up** after test completion

This uses the datalad example dataset, which contains:

- Structural DICOM images (T1w, T2w, etc.)
- Single subject with realistic DICOM structure
- ~50-100 MB when downloaded

### BIDS Validation

Tests use two validation approaches:

1. **Minimal validation** (default): Checks required files and directory structure
2. **Full validation** (optional): Uses bids-validator if installed

## Notes for Developers

### Adding New Tests

Follow the pattern:

```python
def test_my_feature(self, bids_dir: Path, output_dir: Path) -> None:
    """
    Test description.
    
    Parameters
    ----------
    bids_dir : Path
        Fixture providing temporary BIDS directory
    output_dir : Path
        Fixture providing temporary output directory
    """
    # Setup
    # Execute
    # Assert
    LOGGER.info("✓ Test completed")
```

### Skipping Tests

```python
# Skip entire class
@pytest.mark.skip(reason="Requires dcm2niix")
class TestDICOMConversion:
    pass

# Skip single test
@pytest.mark.skip(reason="Requires dcm2niix installation")
def test_dcm2niix_conversion(self, ...):
    pass

# Skip with condition
@pytest.mark.skipif(not HAS_NIBABEL, reason="nibabel not installed")
def test_nifti_creation(self, ...):
    pass
```

### Debugging Tests

```bash
# Run with print statements visible
pytest examples/test_conversion.py::TestBIDSConversion::test_create_dummy_nifti_files -s

# Run with debugger on failure
pytest examples/test_preprocessing.py -x --pdb

# Show detailed logs
pytest examples/ -v --log-cli-level=DEBUG
```

## Known Limitations

1. **dcm2niix tests**: Skipped by default (requires dcm2niix installation)
2. **Complete workflow tests**: Skipped (requires full Docker/GPU setup)
3. **Surface processing**: Requires FreeSurfer (tested separately)

## CI/CD Integration

Example GitHub Actions workflow:

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - run: pip install -e ".[dev]"
      - run: pytest tests/ -v
```

## Performance Notes

- **Fast tests** (~1-5 seconds): BIDS structure, file naming validation
- **Medium tests** (~5-30 seconds): NIfTI file creation, workflow initialization
- **Slow tests** (30+ seconds): DICOM download, full workflow execution

Run with `-x` to stop on first failure and speed up debugging.

## References

- [BIDS Specification](https://bids-specification.readthedocs.io/)
- [datalad Example DICOM Structural](https://github.com/datalad/example-dicom-structural)
- [OncoPrep Documentation](../README.md)
- [Nipype Documentation](https://nipype.readthedocs.io/)

## Support

For issues or questions:

1. Check test output with `-vv` flag
2. Review fixture setup in `conftest.py`
3. Enable debug logging with `--log-cli-level=DEBUG`
4. Open an issue on the OncoPrep repository
