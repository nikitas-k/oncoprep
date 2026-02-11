# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2025 The OncoPrep Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Nipype interfaces for MRIQC (MRI Quality Control) integration."""

from typing import Optional

from nipype.interfaces.base import (
    CommandLine,
    CommandLineInputSpec,
    Directory,
    File,
    TraitedSpec,
    traits,
)


class _MRIQCInputSpec(CommandLineInputSpec):
    """Input specification for MRIQC command-line interface."""

    bids_dir = Directory(
        exists=True,
        mandatory=True,
        argstr='%s',
        position=0,
        desc='Root folder of the BIDS dataset.',
    )
    output_dir = Directory(
        mandatory=True,
        argstr='%s',
        position=1,
        desc='Output directory for MRIQC derivatives.',
    )
    analysis_level = traits.Enum(
        'participant',
        'group',
        usedefault=True,
        argstr='%s',
        position=2,
        desc='Processing level (participant or group).',
    )
    participant_label = traits.List(
        traits.Str,
        argstr='--participant-label %s',
        sep=' ',
        desc='Space-delimited list of participant identifiers.',
    )
    session_id = traits.List(
        traits.Str,
        argstr='--session-id %s',
        sep=' ',
        desc='Space-delimited list of session identifiers.',
    )
    modalities = traits.List(
        traits.Enum('T1w', 'T2w', 'bold', 'dwi'),
        argstr='-m %s',
        sep=' ',
        desc='Modalities to include (default: T1w T2w).',
    )
    nprocs = traits.Int(
        argstr='--nprocs %d',
        desc='Number of CPUs available.',
    )
    omp_nthreads = traits.Int(
        argstr='--omp-nthreads %d',
        desc='Maximum number of threads per process.',
    )
    mem_gb = traits.Float(
        argstr='--mem %.2f',
        desc='Upper bound memory limit (GB).',
    )
    work_dir = Directory(
        argstr='-w %s',
        desc='Path for intermediate results.',
    )
    no_sub = traits.Bool(
        True,
        usedefault=True,
        argstr='--no-sub',
        desc='Disable submission of anonymized quality metrics to MRIQC Web API.',
    )
    verbose_reports = traits.Bool(
        argstr='--verbose-reports',
        desc='Generate verbose reports with extra details.',
    )
    float32 = traits.Bool(
        argstr='--float32',
        desc='Cast input data to float32 to reduce memory footprint '
        '(may not be supported in all MRIQC versions).',
    )
    ica = traits.Bool(
        argstr='--ica',
        desc='Run ICA-based artifact detection on BOLD data (functional only).',
    )


class _MRIQCOutputSpec(TraitedSpec):
    """Output specification for MRIQC command-line interface."""

    out_dir = Directory(
        desc='Output directory containing MRIQC derivatives.',
    )
    group_tsv = File(
        desc='Path to group-level TSV file with image quality metrics.',
    )
    out_report = File(
        desc='Path to the participant-level HTML report.',
    )


class MRIQC(CommandLine):
    """Run MRIQC (MRI Quality Control) on a BIDS dataset.

    MRIQC extracts no-reference image quality metrics (IQMs) from
    structural (T1w/T2w) and functional (BOLD) MRI data. It produces
    individual visual reports and group-level summary statistics.

    For neuro-oncology data, MRIQC QC metrics can flag acquisition
    artifacts, motion corruption, and signal inhomogeneity *before*
    tumor segmentation, enabling early detection of unusable scans.

    Examples
    --------
    >>> from oncoprep.interfaces.mriqc import MRIQC
    >>> mriqc = MRIQC()
    >>> mriqc.inputs.bids_dir = '/data/bids'
    >>> mriqc.inputs.output_dir = '/data/derivatives/mriqc'
    >>> mriqc.inputs.participant_label = ['001']
    >>> mriqc.inputs.modalities = ['T1w', 'T2w']
    >>> mriqc.cmdline  # doctest: +SKIP
    'mriqc /data/bids /data/derivatives/mriqc participant ...'

    """

    _cmd = 'mriqc'
    input_spec = _MRIQCInputSpec
    output_spec = _MRIQCOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_dir'] = self.inputs.output_dir

        from pathlib import Path

        out_path = Path(self.inputs.output_dir)

        # Look for group-level TSV
        group_tsvs = list(out_path.glob('group_T1w.tsv'))
        if group_tsvs:
            outputs['group_tsv'] = str(group_tsvs[0])

        # Look for participant-level HTML report
        if self.inputs.participant_label:
            sub = self.inputs.participant_label[0]
            sub_prefix = sub if sub.startswith('sub-') else f'sub-{sub}'
            reports = list(out_path.glob(f'{sub_prefix}*.html'))
            if reports:
                outputs['out_report'] = str(reports[0])

        return outputs


class _MRIQCGroupInputSpec(CommandLineInputSpec):
    """Input specification for MRIQC group-level analysis."""

    bids_dir = Directory(
        exists=True,
        mandatory=True,
        argstr='%s',
        position=0,
        desc='Root folder of the BIDS dataset.',
    )
    output_dir = Directory(
        mandatory=True,
        argstr='%s',
        position=1,
        desc='Output directory (must contain participant-level results).',
    )
    analysis_level = traits.Enum(
        'group',
        usedefault=True,
        argstr='%s',
        position=2,
        desc='Processing level (always group for this interface).',
    )
    no_sub = traits.Bool(
        True,
        usedefault=True,
        argstr='--no-sub',
        desc='Disable anonymized quality metrics submission.',
    )


class _MRIQCGroupOutputSpec(TraitedSpec):
    """Output specification for MRIQC group-level analysis."""

    group_t1w_tsv = File(
        desc='Group-level T1w IQM summary.',
    )
    group_t2w_tsv = File(
        desc='Group-level T2w IQM summary.',
    )
    group_bold_tsv = File(
        desc='Group-level BOLD IQM summary.',
    )


class MRIQCGroup(CommandLine):
    """Run MRIQC group-level analysis.

    Aggregates per-participant image quality metrics (IQMs) into
    a group summary table, enabling outlier detection and batch
    quality assessment across subjects.

    """

    _cmd = 'mriqc'
    input_spec = _MRIQCGroupInputSpec
    output_spec = _MRIQCGroupOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        from pathlib import Path

        out_path = Path(self.inputs.output_dir)

        for modality, key in [
            ('T1w', 'group_t1w_tsv'),
            ('T2w', 'group_t2w_tsv'),
            ('bold', 'group_bold_tsv'),
        ]:
            tsv = out_path / f'group_{modality}.tsv'
            if tsv.exists():
                outputs[key] = str(tsv)

        return outputs


def check_mriqc_available() -> bool:
    """Check whether MRIQC is installed and callable.

    Returns
    -------
    bool
        True if ``mriqc --version`` succeeds, False otherwise.
    """
    import subprocess

    try:
        subprocess.run(
            ['mriqc', '--version'],
            check=True,
            capture_output=True,
            timeout=30,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


__all__ = [
    'MRIQC',
    'MRIQCGroup',
    'check_mriqc_available',
]
