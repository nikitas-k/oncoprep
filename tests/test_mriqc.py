"""Tests for MRIQC quality control integration."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Interface tests
# ---------------------------------------------------------------------------


class TestMRIQCInterface:
    """Tests for the MRIQC Nipype CommandLine interface."""

    def test_import(self):
        from oncoprep.interfaces.mriqc import MRIQC, MRIQCGroup, check_mriqc_available

        assert MRIQC is not None
        assert MRIQCGroup is not None
        assert check_mriqc_available is not None

    def test_mriqc_cmdline(self, tmp_path):
        """MRIQC interface generates correct command-line string."""
        from oncoprep.interfaces.mriqc import MRIQC

        bids_dir = tmp_path / 'bids'
        out_dir = tmp_path / 'out'
        bids_dir.mkdir()
        out_dir.mkdir()

        mriqc = MRIQC()
        mriqc.inputs.bids_dir = str(bids_dir)
        mriqc.inputs.output_dir = str(out_dir)
        mriqc.inputs.analysis_level = 'participant'
        mriqc.inputs.participant_label = ['001']

        cmd = mriqc.cmdline
        assert 'mriqc' in cmd
        assert str(bids_dir) in cmd
        assert str(out_dir) in cmd
        assert 'participant' in cmd
        assert '--participant-label 001' in cmd
        assert '--no-sub' in cmd

    def test_mriqc_cmdline_with_options(self, tmp_path):
        """MRIQC interface includes optional parameters."""
        from oncoprep.interfaces.mriqc import MRIQC

        bids_dir = tmp_path / 'bids'
        out_dir = tmp_path / 'out'
        bids_dir.mkdir()
        out_dir.mkdir()

        mriqc = MRIQC()
        mriqc.inputs.bids_dir = str(bids_dir)
        mriqc.inputs.output_dir = str(out_dir)
        mriqc.inputs.participant_label = ['001']
        mriqc.inputs.modalities = ['T1w', 'T2w']
        mriqc.inputs.nprocs = 4
        mriqc.inputs.omp_nthreads = 2
        mriqc.inputs.mem_gb = 8.0

        cmd = mriqc.cmdline
        assert '--modalities T1w T2w' in cmd
        assert '--nprocs 4' in cmd
        assert '--omp-nthreads 2' in cmd
        assert '--mem-gb 8.00' in cmd

    def test_mriqc_group_cmdline(self, tmp_path):
        """MRIQCGroup generates correct group-level command."""
        from oncoprep.interfaces.mriqc import MRIQCGroup

        bids_dir = tmp_path / 'bids'
        out_dir = tmp_path / 'out'
        bids_dir.mkdir()
        out_dir.mkdir()

        mriqc_grp = MRIQCGroup()
        mriqc_grp.inputs.bids_dir = str(bids_dir)
        mriqc_grp.inputs.output_dir = str(out_dir)

        cmd = mriqc_grp.cmdline
        assert 'mriqc' in cmd
        assert 'group' in cmd
        assert '--no-sub' in cmd

    def test_check_mriqc_available_not_installed(self):
        """check_mriqc_available returns False when mriqc is absent."""
        from oncoprep.interfaces.mriqc import check_mriqc_available

        with patch('subprocess.run', side_effect=FileNotFoundError):
            assert check_mriqc_available() is False


# ---------------------------------------------------------------------------
# Workflow tests
# ---------------------------------------------------------------------------


class TestMRIQCWorkflow:
    """Tests for MRIQC Nipype workflow factories."""

    def test_init_mriqc_wf(self, tmp_path):
        """init_mriqc_wf creates a valid Nipype workflow."""
        from oncoprep.workflows.mriqc import init_mriqc_wf

        bids_dir = tmp_path / 'bids'
        bids_dir.mkdir()
        output_dir = tmp_path / 'out'
        output_dir.mkdir()

        wf = init_mriqc_wf(
            bids_dir=bids_dir,
            output_dir=output_dir,
            subject_id='001',
        )

        node_names = wf.list_node_names()
        assert 'inputnode' in node_names
        assert 'outputnode' in node_names
        assert 'run_mriqc' in node_names

    def test_init_mriqc_wf_with_group(self, tmp_path):
        """init_mriqc_wf with run_group=True adds group node."""
        from oncoprep.workflows.mriqc import init_mriqc_wf

        bids_dir = tmp_path / 'bids'
        bids_dir.mkdir()
        output_dir = tmp_path / 'out'
        output_dir.mkdir()

        wf = init_mriqc_wf(
            bids_dir=bids_dir,
            output_dir=output_dir,
            subject_id='001',
            run_group=True,
        )

        node_names = wf.list_node_names()
        assert 'run_mriqc_group' in node_names

    def test_init_mriqc_wf_custom_modalities(self, tmp_path):
        """init_mriqc_wf accepts custom modalities."""
        from oncoprep.workflows.mriqc import init_mriqc_wf

        bids_dir = tmp_path / 'bids'
        bids_dir.mkdir()
        output_dir = tmp_path / 'out'
        output_dir.mkdir()

        wf = init_mriqc_wf(
            bids_dir=bids_dir,
            output_dir=output_dir,
            subject_id='001',
            modalities=['T1w'],
            omp_nthreads=2,
            nprocs=4,
        )

        # Verify the run_mriqc node has the right inputs
        run_node = wf.get_node('run_mriqc')
        assert run_node.inputs.modalities == ['T1w']
        assert run_node.inputs.omp_nthreads == 2
        assert run_node.inputs.nprocs == 4

    def test_init_mriqc_group_wf(self, tmp_path):
        """init_mriqc_group_wf creates a valid group workflow."""
        from oncoprep.workflows.mriqc import init_mriqc_group_wf

        wf = init_mriqc_group_wf(
            bids_dir=tmp_path / 'bids',
            output_dir=tmp_path / 'out',
        )

        node_names = wf.list_node_names()
        assert 'outputnode' in node_names
        assert 'run_mriqc_group' in node_names
        assert 'parse_group_tsvs' in node_names

    def test_workflow_description(self, tmp_path):
        """Workflow contains boilerplate description."""
        from oncoprep.workflows.mriqc import init_mriqc_wf

        bids_dir = tmp_path / 'bids'
        bids_dir.mkdir()
        output_dir = tmp_path / 'out'
        output_dir.mkdir()

        wf = init_mriqc_wf(
            bids_dir=bids_dir,
            output_dir=output_dir,
            subject_id='001',
        )

        assert 'MRIQC' in wf.__desc__
        assert 'T1w' in wf.__desc__


# ---------------------------------------------------------------------------
# IQM extraction tests
# ---------------------------------------------------------------------------


class TestIQMExtraction:
    """Tests for the IQM summary extraction utility."""

    def test_extract_iqm_summary_good(self, tmp_path):
        """Good-quality IQMs pass QC gating."""
        from oncoprep.workflows.mriqc import _extract_iqm_summary

        iqm_data = {
            'snr_total': 15.5,
            'cnr': 3.2,
            'efc': 0.35,
            'fber': 1800.0,
            'inu_med': 0.02,
            'cjv': 0.3,
            'wm2max': 0.6,
            'qi_1': 0.001,
        }

        iqm_file = tmp_path / 'sub-001_T1w.json'
        iqm_file.write_text(json.dumps(iqm_data))

        result = _extract_iqm_summary(str(iqm_file))

        assert result['pass_qc'] is True
        assert result['snr_total'] == 15.5
        assert result['cnr'] == 3.2
        assert len(result['qc_flags']) == 0

    def test_extract_iqm_summary_low_snr(self, tmp_path):
        """Low SNR triggers QC failure."""
        from oncoprep.workflows.mriqc import _extract_iqm_summary

        iqm_data = {
            'snr_total': 1.5,
            'cnr': 0.5,
            'efc': 0.35,
            'cjv': 0.3,
            'qi_1': 0.001,
        }

        iqm_file = tmp_path / 'sub-002_T1w.json'
        iqm_file.write_text(json.dumps(iqm_data))

        result = _extract_iqm_summary(str(iqm_file))

        assert result['pass_qc'] is False
        assert any('SNR' in f for f in result['qc_flags'])

    def test_extract_iqm_summary_high_cjv(self, tmp_path):
        """High CJV triggers QC failure."""
        from oncoprep.workflows.mriqc import _extract_iqm_summary

        iqm_data = {
            'snr_total': 10.0,
            'cjv': 0.8,
            'efc': 0.3,
            'qi_1': 0.001,
        }

        iqm_file = tmp_path / 'sub-003_T1w.json'
        iqm_file.write_text(json.dumps(iqm_data))

        result = _extract_iqm_summary(str(iqm_file))

        assert result['pass_qc'] is False
        assert any('CJV' in f for f in result['qc_flags'])

    def test_extract_iqm_summary_missing_file(self):
        """Missing IQM file returns error dict."""
        from oncoprep.workflows.mriqc import _extract_iqm_summary

        result = _extract_iqm_summary('/nonexistent/path.json')

        assert result['pass_qc'] is False
        assert 'error' in result

    def test_extract_iqm_summary_with_warnings(self, tmp_path):
        """Elevated EFC and QI1 produce warnings but may still pass."""
        from oncoprep.workflows.mriqc import _extract_iqm_summary

        iqm_data = {
            'snr_total': 12.0,
            'cjv': 0.3,
            'efc': 0.65,   # above warning threshold
            'qi_1': 0.06,  # above warning threshold
        }

        iqm_file = tmp_path / 'sub-004_T1w.json'
        iqm_file.write_text(json.dumps(iqm_data))

        result = _extract_iqm_summary(str(iqm_file))

        # EFC and QI1 warnings don't fail QC by themselves
        assert result['pass_qc'] is True
        assert any('EFC' in f for f in result['qc_flags'])
        assert any('QI1' in f for f in result['qc_flags'])


# ---------------------------------------------------------------------------
# CLI integration test
# ---------------------------------------------------------------------------


class TestMRIQCCLI:
    """Test that --run-qc is recognized by the CLI parser."""

    def test_parser_accepts_run_qc(self):
        """CLI parser recognizes --run-qc flag."""
        from oncoprep.cli import get_parser

        parser = get_parser()
        args = parser.parse_args([
            '/data/bids',
            '/data/out',
            'participant',
            '--run-qc',
        ])

        assert args.run_qc is True

    def test_parser_default_no_qc(self):
        """--run-qc defaults to False."""
        from oncoprep.cli import get_parser

        parser = get_parser()
        args = parser.parse_args([
            '/data/bids',
            '/data/out',
            'participant',
        ])

        assert args.run_qc is False
