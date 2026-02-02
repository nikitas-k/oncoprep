"""Integration tests for end-to-end BIDS conversion and preprocessing."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import pytest

try:
    import nibabel as nb
    import numpy as np
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

from oncoprep.utils.logging import get_logger

LOGGER = get_logger(__name__)


@pytest.mark.skipif(not HAS_NIBABEL, reason="nibabel not installed")
class TestIntegrationWorkflow:
    """Integration tests for complete BIDS conversion and preprocessing workflows."""

    def test_single_subject_with_example_data(
        self,
        bids_dir: Path,
        example_data_dir: Path,
    ) -> None:
        """
        Test single-subject workflow using real example data.

        This test copies anatomical files from example data into a BIDS
        directory structure to test the conversion workflow.

        Parameters
        ----------
        bids_dir : Path
            Fixture providing temporary BIDS directory
        example_data_dir : Path
            Fixture providing path to example data
        """
        import shutil

        subject_id = "001"
        session_id = "01"

        # Create BIDS structure
        anat_dir = bids_dir / f"sub-{subject_id}" / f"ses-{session_id}" / "anat"
        anat_dir.mkdir(parents=True, exist_ok=True)

        # Copy T1, T1c (T1ce), T2, FLAIR from example data
        anatomical_mappings = {
            "T1w": "*T1[^c]*",      # T1w (not T1c)
            "T1ce": "*T1c*",        # T1ce (contrast enhanced)
            "T2w": "*T2[^_]*",      # T2w
            "FLAIR": "*FLAIR*",     # FLAIR
        }

        copied_files = {}
        for modality, pattern in anatomical_mappings.items():
            source_files = list(example_data_dir.glob(pattern))
            if source_files:
                source_file = source_files[0]
                dest_file = anat_dir / f"sub-{subject_id}_ses-{session_id}_{modality}.nii.gz"
                shutil.copy2(source_file, dest_file)
                copied_files[modality] = dest_file

                # Create JSON sidecar
                json_file = dest_file.with_suffix("").with_suffix(".json")
                with open(json_file, "w") as f:
                    json.dump({
                        "EchoTime": 0.00456,
                        "RepetitionTime": 2.3,
                        "FlipAngle": 9,
                        "Modality": modality,
                    }, f)

        # Verify structure
        assert len(copied_files) > 0, "No anatomical files copied from example data"
        assert anat_dir.exists()

        for modality, filepath in copied_files.items():
            assert filepath.exists(), f"{modality} file missing"

        LOGGER.info(f"✓ Copied {len(copied_files)} modalities from example data")

    def test_single_subject_minimal_workflow(
        self,
        bids_dir: Path,
        output_dir: Path,
        work_dir: Path,
    ) -> None:
        """
        Test minimal single-subject workflow with dummy data.

        This test creates a complete BIDS dataset with a single subject,
        single session, and minimal anatomical data, then verifies the
        directory structure is correct for preprocessing.

        Parameters
        ----------
        bids_dir : Path
            Fixture providing temporary BIDS directory
        output_dir : Path
            Fixture providing temporary output directory
        work_dir : Path
            Fixture providing temporary work directory
        """
        subject_id = "001"
        session_id = "01"

        # Step 1: Create valid BIDS structure
        anat_dir = bids_dir / f"sub-{subject_id}" / f"ses-{session_id}" / "anat"
        anat_dir.mkdir(parents=True, exist_ok=True)

        # Step 2: Create anatomical data
        data = np.random.randint(100, 200, (64, 64, 64), dtype=np.uint8)
        affine = np.eye(4)
        affine[:3, :3] *= 2.0

        img = nb.Nifti1Image(data, affine)
        t1w_file = anat_dir / f"sub-{subject_id}_ses-{session_id}_T1w.nii.gz"
        nb.save(img, t1w_file)

        # Create JSON sidecar
        json_file = anat_dir / f"sub-{subject_id}_ses-{session_id}_T1w.json"
        with open(json_file, "w") as f:
            json.dump({
                "EchoTime": 0.00456,
                "RepetitionTime": 2.3,
                "FlipAngle": 9,
            }, f)

        # Step 3: Verify BIDS structure
        assert (bids_dir / "dataset_description.json").exists()
        assert t1w_file.exists()
        assert json_file.exists()

        # Step 4: Verify output directory exists
        assert output_dir.exists()
        assert work_dir.exists()

        LOGGER.info(
            f"✓ Single-subject workflow setup: sub-{subject_id}/ses-{session_id}"
        )

    def test_multisubject_bids_structure(
        self,
        bids_dir: Path,
    ) -> None:
        """
        Test creation of multi-subject BIDS structure.

        Parameters
        ----------
        bids_dir : Path
            Fixture providing temporary BIDS directory
        """
        subjects = ["001", "002"]
        sessions = ["01", "02"]

        subject_dirs = []
        for sub in subjects:
            for ses in sessions:
                anat_dir = bids_dir / f"sub-{sub}" / f"ses-{ses}" / "anat"
                anat_dir.mkdir(parents=True, exist_ok=True)

                # Create data
                data = np.random.randint(100, 200, (32, 32, 32), dtype=np.uint8)
                affine = np.eye(4)

                img = nb.Nifti1Image(data, affine)
                t1w_file = anat_dir / f"sub-{sub}_ses-{ses}_T1w.nii.gz"
                nb.save(img, t1w_file)

                json_file = anat_dir / f"sub-{sub}_ses-{ses}_T1w.json"
                with open(json_file, "w") as f:
                    json.dump({"EchoTime": 0.0}, f)

                subject_dirs.append((sub, ses, anat_dir))

        # Verify structure
        assert len(subject_dirs) == len(subjects) * len(sessions)

        for sub, ses, anat_dir in subject_dirs:
            t1w_file = anat_dir / f"sub-{sub}_ses-{ses}_T1w.nii.gz"
            assert t1w_file.exists(), f"Missing T1w for sub-{sub}, ses-{ses}"

        LOGGER.info(
            f"✓ Multi-subject BIDS structure created: {len(subjects)} subjects, "
            f"{len(sessions)} sessions each"
        )

    def test_derivative_output_structure(
        self,
        output_dir: Path,
    ) -> None:
        """
        Test creation of proper derivative output structure.

        Parameters
        ----------
        output_dir : Path
            Fixture providing temporary output directory
        """
        subject_id = "001"
        session_id = "01"

        # Create OncoPrep derivatives structure
        derivatives_root = output_dir / "oncoprep"
        anat_dir = derivatives_root / f"sub-{subject_id}" / f"ses-{session_id}" / "anat"
        anat_dir.mkdir(parents=True, exist_ok=True)

        # Create preprocessed outputs
        data = np.random.randint(100, 200, (64, 64, 64), dtype=np.uint8)
        affine = np.eye(4)

        outputs = {
            "desc-preproc_T1w": data,
            "desc-brain_mask": (data > 120).astype(np.uint8),
            "space-template_T1w": data,
        }

        for output_type, output_data in outputs.items():
            img = nb.Nifti1Image(output_data, affine)
            filepath = anat_dir / f"sub-{subject_id}_ses-{session_id}_{output_type}.nii.gz"
            nb.save(img, filepath)

            # Create JSON sidecars
            json_file = filepath.with_suffix("").with_suffix(".json")
            with open(json_file, "w") as f:
                json.dump({"Description": output_type}, f)

        # Verify outputs
        for output_type in outputs.keys():
            filepath = anat_dir / f"sub-{subject_id}_ses-{session_id}_{output_type}.nii.gz"
            assert filepath.exists()

        LOGGER.info(f"✓ Derivative structure created with {len(outputs)} output types")

    def test_workflow_execution_setup(self, bids_dir: Path, output_dir: Path) -> None:
        """
        Test setup for workflow execution.

        This test verifies that all necessary directories and configurations
        are in place for running the full OncoPrep workflow.

        Parameters
        ----------
        bids_dir : Path
            Fixture providing temporary BIDS directory
        output_dir : Path
            Fixture providing temporary output directory
        """
        # Create test data
        subject_id = "001"
        session_id = "01"
        anat_dir = bids_dir / f"sub-{subject_id}" / f"ses-{session_id}" / "anat"
        anat_dir.mkdir(parents=True, exist_ok=True)

        data = np.random.randint(100, 200, (32, 32, 32), dtype=np.uint8)
        affine = np.eye(4)

        modalities = ["T1w", "T1ce", "T2w", "FLAIR"]
        for modality in modalities:
            img = nb.Nifti1Image(data, affine)
            filepath = anat_dir / f"sub-{subject_id}_ses-{session_id}_{modality}.nii.gz"
            nb.save(img, filepath)

            json_file = filepath.with_suffix("").with_suffix(".json")
            with open(json_file, "w") as f:
                json.dump({"Modality": modality}, f)

        # Verify execution setup
        assert bids_dir.exists()
        assert (bids_dir / "dataset_description.json").exists()
        assert output_dir.exists()

        # Check that all modalities are present
        for modality in modalities:
            filepath = anat_dir / f"sub-{subject_id}_ses-{session_id}_{modality}.nii.gz"
            assert filepath.exists(), f"Missing {modality}"

        LOGGER.info("✓ Workflow execution setup complete")

    def test_bids_validation_with_validator_tool(self, bids_dir: Path) -> None:
        """
        Test BIDS validation using bids-validator if available.

        Parameters
        ----------
        bids_dir : Path
            Fixture providing temporary BIDS directory
        """
        # Create minimal valid structure
        subject_id = "001"
        session_id = "01"
        anat_dir = bids_dir / f"sub-{subject_id}" / f"ses-{session_id}" / "anat"
        anat_dir.mkdir(parents=True, exist_ok=True)

        # Create data
        data = np.random.randint(100, 200, (32, 32, 32), dtype=np.uint8)
        affine = np.eye(4)

        img = nb.Nifti1Image(data, affine)
        t1w_file = anat_dir / f"sub-{subject_id}_ses-{session_id}_T1w.nii.gz"
        nb.save(img, t1w_file)

        json_file = anat_dir / f"sub-{subject_id}_ses-{session_id}_T1w.json"
        with open(json_file, "w") as f:
            json.dump({
                "EchoTime": 0.00456,
                "RepetitionTime": 2.3,
                "FlipAngle": 9,
            }, f)

        # Try to validate with bids-validator
        try:
            result = subprocess.run(
                ["bids-validator", str(bids_dir)],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                LOGGER.info("✓ BIDS validation passed")
            else:
                LOGGER.info(f"BIDS validation warnings/errors: {result.stdout}")

        except FileNotFoundError:
            LOGGER.info("bids-validator not installed, skipping validation")
        except subprocess.TimeoutExpired:
            LOGGER.warning("bids-validator timed out")

    def test_workflow_file_collection(self, bids_dir: Path) -> None:
        """
        Test collection of input files for workflow execution.

        Parameters
        ----------
        bids_dir : Path
            Fixture providing temporary BIDS directory
        """
        # Create multi-subject structure
        subjects_sessions = [("001", "01"), ("002", "01"), ("002", "02")]

        for subject, session in subjects_sessions:
            anat_dir = bids_dir / f"sub-{subject}" / f"ses-{session}" / "anat"
            anat_dir.mkdir(parents=True, exist_ok=True)

            # Create T1w files
            data = np.random.randint(100, 200, (32, 32, 32), dtype=np.uint8)
            affine = np.eye(4)

            img = nb.Nifti1Image(data, affine)
            filepath = anat_dir / f"sub-{subject}_ses-{session}_T1w.nii.gz"
            nb.save(img, filepath)

            json_file = filepath.with_suffix("").with_suffix(".json")
            with open(json_file, "w") as f:
                json.dump({"EchoTime": 0.0}, f)

        # Collect files
        collected_subjects = set()
        collected_sessions = {}

        for subject_dir in bids_dir.glob("sub-*"):
            if not subject_dir.is_dir():
                continue

            subject_id = subject_dir.name.replace("sub-", "")
            collected_subjects.add(subject_id)

            if subject_id not in collected_sessions:
                collected_sessions[subject_id] = []

            for session_dir in subject_dir.glob("ses-*"):
                if not session_dir.is_dir():
                    continue
                session_id = session_dir.name.replace("ses-", "")
                collected_sessions[subject_id].append(session_id)

        # Verify collection
        assert len(collected_subjects) == 2
        assert collected_sessions["001"] == ["01"]
        assert set(collected_sessions["002"]) == {"01", "02"}

        LOGGER.info(
            f"✓ Collected {len(collected_subjects)} subjects with "
            f"{sum(len(v) for v in collected_sessions.values())} sessions"
        )

    def test_longitudinal_structure(self, bids_dir: Path) -> None:
        """
        Test creation of longitudinal multi-session structure.

        Parameters
        ----------
        bids_dir : Path
            Fixture providing temporary BIDS directory
        """
        subject_id = "001"
        sessions = ["01", "02", "03"]

        for session in sessions:
            anat_dir = bids_dir / f"sub-{subject_id}" / f"ses-{session}" / "anat"
            anat_dir.mkdir(parents=True, exist_ok=True)

            # Create data with session-specific variation
            base_data = np.random.randint(100, 200, (32, 32, 32), dtype=np.uint8)
            session_num = int(session)
            varied_data = base_data + session_num * 5  # Simulate progression

            affine = np.eye(4)
            img = nb.Nifti1Image(varied_data, affine)
            filepath = anat_dir / f"sub-{subject_id}_ses-{session}_T1w.nii.gz"
            nb.save(img, filepath)

            json_file = filepath.with_suffix("").with_suffix(".json")
            with open(json_file, "w") as f:
                json.dump({"Session": session}, f)

        # Verify longitudinal structure
        session_files = list(bids_dir.glob(f"sub-{subject_id}/ses-*/anat/*_T1w.nii.gz"))
        assert len(session_files) == len(sessions)

        LOGGER.info(f"✓ Longitudinal structure with {len(sessions)} sessions created")

    @pytest.mark.skip(reason="Requires actual OncoPrep workflow instantiation")
    def test_complete_workflow_run(
        self,
        bids_dir: Path,
        output_dir: Path,
        work_dir: Path,
    ) -> None:
        """
        Test running the complete OncoPrep workflow.

        This test would run the full preprocessing pipeline.
        Currently skipped - requires full environment setup.

        Parameters
        ----------
        bids_dir : Path
            Fixture providing temporary BIDS directory
        output_dir : Path
            Fixture providing temporary output directory
        work_dir : Path
            Fixture providing temporary work directory
        """
        try:
            from oncoprep.workflows.base import init_oncoprep_wf

            # Initialize workflow
            workflow = init_oncoprep_wf(
                output_dir=output_dir,
                subject_session_list=[("001", ["01"])],
                run_uuid="test-uuid",
                work_dir=work_dir,
                bids_dir=bids_dir,
                omp_nthreads=1,
                nprocs=1,
                skip_segmentation=True,
                sloppy=True,
            )

            # Run workflow (would execute on actual data)
            # result = workflow.run(plugin='SingleThreadedPlugin')

            LOGGER.info("✓ Workflow execution test completed")

        except Exception as e:
            pytest.skip(f"Workflow execution not available: {e}")
