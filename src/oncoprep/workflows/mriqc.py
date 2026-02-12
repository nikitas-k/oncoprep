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
"""MRIQC quality control workflow for OncoPrep.

This module provides Nipype workflows that invoke MRIQC to compute
image quality metrics (IQMs) on raw BIDS anatomical data before
preprocessing, enabling early detection of unusable scans.

The workflow runs **in parallel** with the anatomical preprocessing
pipeline — it operates on raw BIDS inputs and does not depend on
preprocessed outputs.

Key IQMs computed by MRIQC for structural data:

- **SNR** (signal-to-noise ratio within tissue masks)
- **CNR** (contrast-to-noise ratio between GM/WM)
- **EFC** (entropy focus criterion — ghosting/ringing)
- **FBER** (foreground-to-background energy ratio)
- **INU** (intensity non-uniformity metrics)
- **WM2MAX** (white-matter to maximum intensity ratio)
- **CJV** (coefficient of joint variation between GM/WM)

Outputs
-------
- Per-participant HTML reports (visual QC) in ``<output_dir>/mriqc/``
- Per-participant JSON IQM files
- Group-level TSV summary (when ``run_group=True``)
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Optional

from nipype import logging
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

LOGGER = logging.getLogger('nipype.workflow')


def init_mriqc_wf(
    *,
    bids_dir: Path,
    output_dir: Path,
    subject_id: str,
    session_id: Optional[str] = None,
    modalities: Optional[List[str]] = None,
    work_dir: Optional[Path] = None,
    omp_nthreads: int = 1,
    nprocs: int = 1,
    mem_gb: Optional[float] = None,
    run_group: bool = False,
    name: str = 'mriqc_wf',
) -> Workflow:
    """Create a workflow that runs MRIQC on raw BIDS data.

    This workflow wraps the MRIQC command-line tool to compute image
    quality metrics (IQMs) for structural anatomical images. It is
    designed to run **before or in parallel with** the main OncoPrep
    preprocessing, as it operates on the raw (unconverted) BIDS data.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from pathlib import Path
            from oncoprep.workflows.mriqc import init_mriqc_wf
            wf = init_mriqc_wf(
                bids_dir=Path('.'),
                output_dir=Path('derivatives'),
                subject_id='001',
            )

    Parameters
    ----------
    bids_dir : :obj:`Path`
        Root directory of the BIDS dataset.
    output_dir : :obj:`Path`
        Top-level output directory. MRIQC derivatives will be written
        to ``<output_dir>/mriqc/``.
    subject_id : :obj:`str`
        Subject identifier (without ``sub-`` prefix).
    session_id : :obj:`str` or None
        Session identifier (without ``ses-`` prefix), if applicable.
    modalities : list of str or None
        Modalities to process (default: ``['T1w', 'T2w']``).
    work_dir : :obj:`Path` or None
        Working directory for intermediate files. If ``None``, a
        ``mriqc_work`` subdirectory is created under ``output_dir``.
    omp_nthreads : :obj:`int`
        Maximum threads per process (default: 1).
    nprocs : :obj:`int`
        Number of parallel processes (default: 1).
    mem_gb : :obj:`float` or None
        Upper memory limit in GB.
    run_group : :obj:`bool`
        If True, also run MRIQC group-level analysis after the
        participant-level step (default: False).
    name : :obj:`str`
        Workflow name (default: ``mriqc_wf``).

    Inputs
    ------
    subject_id
        Subject identifier (can be overridden via connection).

    Outputs
    -------
    mriqc_dir
        Path to the MRIQC output directory.
    iqm_json
        Path to participant-level IQM JSON file.
    out_report
        Path to the participant-level HTML report.
    group_tsv
        Path to group-level TSV (only if ``run_group=True``).

    Returns
    -------
    Workflow
        Nipype workflow instance.

    """
    if modalities is None:
        modalities = ['T1w', 'T2w']

    if shutil.which('mriqc') is None:
        raise FileNotFoundError(
            'MRIQC executable not found on PATH. '
            'Install with: pip install mriqc  '
            '(or omit --run-qc to skip quality control)'
        )

    mriqc_output_dir = Path(output_dir) / 'mriqc'
    mriqc_work_dir = Path(work_dir) if work_dir else Path(output_dir) / 'mriqc_work'

    workflow = Workflow(name=name)
    workflow.__desc__ = f"""\
Image quality metrics were computed on raw anatomical images using
*MRIQC* (@mriqc1; @mriqc2; RRID:SCR_022942) prior to preprocessing.
Modalities assessed: {', '.join(modalities)}.
"""

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['subject_id']),
        name='inputnode',
    )
    inputnode.inputs.subject_id = subject_id

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'mriqc_dir',
                'iqm_json',
                'out_report',
                'group_tsv',
            ]
        ),
        name='outputnode',
    )

    # --- Participant-level MRIQC ---
    run_mriqc = pe.Node(
        niu.Function(
            function=_run_mriqc_participant,
            input_names=[
                'bids_dir',
                'output_dir',
                'subject_id',
                'session_id',
                'modalities',
                'work_dir',
                'omp_nthreads',
                'nprocs',
                'mem_gb',
            ],
            output_names=['mriqc_dir', 'iqm_json', 'out_report'],
        ),
        name='run_mriqc',
    )
    run_mriqc.inputs.bids_dir = str(bids_dir)
    run_mriqc.inputs.output_dir = str(mriqc_output_dir)
    run_mriqc.inputs.session_id = session_id
    run_mriqc.inputs.modalities = modalities
    run_mriqc.inputs.work_dir = str(mriqc_work_dir)
    run_mriqc.inputs.omp_nthreads = omp_nthreads
    run_mriqc.inputs.nprocs = nprocs
    run_mriqc.inputs.mem_gb = mem_gb

    workflow.connect([
        (inputnode, run_mriqc, [('subject_id', 'subject_id')]),
        (run_mriqc, outputnode, [
            ('mriqc_dir', 'mriqc_dir'),
            ('iqm_json', 'iqm_json'),
            ('out_report', 'out_report'),
        ]),
    ])

    # --- Optional group-level analysis ---
    if run_group:
        run_mriqc_group = pe.Node(
            niu.Function(
                function=_run_mriqc_group,
                input_names=['bids_dir', 'output_dir'],
                output_names=['group_tsv'],
            ),
            name='run_mriqc_group',
        )
        run_mriqc_group.inputs.bids_dir = str(bids_dir)
        run_mriqc_group.inputs.output_dir = str(mriqc_output_dir)

        workflow.connect([
            # Ensure group analysis runs after participant
            (run_mriqc, run_mriqc_group, [('mriqc_dir', 'bids_dir')]),
            (run_mriqc_group, outputnode, [('group_tsv', 'group_tsv')]),
        ])

        # Override bids_dir with the actual value (the connection above
        # is just for dependency ordering)
        run_mriqc_group.inputs.bids_dir = str(bids_dir)

    return workflow


def init_mriqc_group_wf(
    *,
    bids_dir: Path,
    output_dir: Path,
    name: str = 'mriqc_group_wf',
) -> Workflow:
    """Create a standalone MRIQC group-level workflow.

    This workflow aggregates participant-level IQMs into a group
    summary table, useful for outlier detection across subjects.

    Parameters
    ----------
    bids_dir : :obj:`Path`
        Root of the BIDS dataset.
    output_dir : :obj:`Path`
        Directory containing participant-level MRIQC results.
    name : :obj:`str`
        Workflow name (default: ``mriqc_group_wf``).

    Outputs
    -------
    group_t1w_tsv
        Group-level T1w IQM TSV.
    group_t2w_tsv
        Group-level T2w IQM TSV.

    Returns
    -------
    Workflow
        Nipype workflow instance.

    """
    mriqc_output_dir = Path(output_dir) / 'mriqc'

    workflow = Workflow(name=name)

    outputnode = pe.Node(
        niu.IdentityInterface(fields=['group_t1w_tsv', 'group_t2w_tsv']),
        name='outputnode',
    )

    run_group = pe.Node(
        niu.Function(
            function=_run_mriqc_group,
            input_names=['bids_dir', 'output_dir'],
            output_names=['group_tsv'],
        ),
        name='run_mriqc_group',
    )
    run_group.inputs.bids_dir = str(bids_dir)
    run_group.inputs.output_dir = str(mriqc_output_dir)

    # Parse group outputs
    parse_group = pe.Node(
        niu.Function(
            function=_parse_group_tsvs,
            input_names=['mriqc_dir'],
            output_names=['group_t1w_tsv', 'group_t2w_tsv'],
        ),
        name='parse_group_tsvs',
    )

    workflow.connect([
        (run_group, parse_group, [('group_tsv', 'mriqc_dir')]),
        (parse_group, outputnode, [
            ('group_t1w_tsv', 'group_t1w_tsv'),
            ('group_t2w_tsv', 'group_t2w_tsv'),
        ]),
    ])

    return workflow


# ---------------------------------------------------------------------------
# Private helper functions (executed inside Nipype Function nodes)
# ---------------------------------------------------------------------------

def _run_mriqc_participant(
    bids_dir,
    output_dir,
    subject_id,
    session_id,
    modalities,
    work_dir,
    omp_nthreads,
    nprocs,
    mem_gb,
):
    """Execute MRIQC participant-level analysis.

    Parameters
    ----------
    bids_dir : str
        Path to the BIDS dataset root.
    output_dir : str
        Path for MRIQC derivatives.
    subject_id : str
        Subject identifier (without sub- prefix).
    session_id : str or None
        Session identifier (without ses- prefix).
    modalities : list of str
        Modalities to process (e.g., ['T1w', 'T2w']).
    work_dir : str
        Working directory for intermediates.
    omp_nthreads : int
        Threads per process.
    nprocs : int
        Parallel processes.
    mem_gb : float or None
        Memory limit in GB.

    Returns
    -------
    mriqc_dir : str
        Path to MRIQC output directory.
    iqm_json : str or None
        Path to the first IQM JSON found, or None.
    out_report : str or None
        Path to the first HTML report found, or None.

    """
    import subprocess
    from pathlib import Path

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    work_path = Path(work_dir)
    work_path.mkdir(parents=True, exist_ok=True)

    # Strip 'sub-' prefix if present
    sub_id = subject_id.replace('sub-', '')

    cmd = [
        'mriqc',
        str(bids_dir),
        str(output_dir),
        'participant',
        '--participant-label', sub_id,
        '--no-sub',
        '-w', str(work_dir),
    ]

    if session_id:
        ses_id = session_id.replace('ses-', '')
        cmd.extend(['--session-id', ses_id])

    if modalities:
        cmd.extend(['-m'] + list(modalities))

    if omp_nthreads and omp_nthreads > 0:
        cmd.extend(['--omp-nthreads', str(omp_nthreads)])

    if nprocs and nprocs > 0:
        cmd.extend(['--nprocs', str(nprocs)])

    if mem_gb and mem_gb > 0:
        cmd.extend(['--mem', str(mem_gb)])

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        import warnings
        # Segfaults (signal 11) produce returncode -11 on Unix
        if exc.returncode and exc.returncode < 0:
            import signal
            try:
                sig_name = signal.Signals(-exc.returncode).name
            except (ValueError, AttributeError):
                sig_name = f'signal {-exc.returncode}'
            warnings.warn(
                f'MRIQC participant-level crashed with {sig_name} for sub-{sub_id}. '
                f'This is often caused by a Python/SQLAlchemy version incompatibility. '
                f'Try upgrading Python to 3.10+ or pinning mriqc<24. '
                f'Command: {" ".join(cmd)}',
                RuntimeWarning,
            )
        else:
            # Show the TAIL of stderr — argparse errors appear after the
            # usage block, so the last ~300 chars are most informative.
            if exc.stderr:
                stderr_tail = exc.stderr.strip().splitlines()
                # Grab last 10 lines max
                stderr_msg = '\n'.join(stderr_tail[-10:])
            else:
                stderr_msg = str(exc)
            warnings.warn(
                f'MRIQC participant-level failed for sub-{sub_id} '
                f'(exit code {exc.returncode}).\n'
                f'Command: {" ".join(cmd)}\n'
                f'Error: {stderr_msg}',
                RuntimeWarning,
            )
    except FileNotFoundError:
        raise FileNotFoundError(
            'MRIQC executable not found on PATH. '
            'Install with: pip install mriqc  '
            '(or omit --run-qc to skip quality control)'
        )

    # Locate outputs
    sub_prefix = f'sub-{sub_id}'
    iqm_json = None
    out_report = None

    # IQM JSON files
    jsons = sorted(out_path.glob(f'{sub_prefix}*_T1w.json'))
    if not jsons:
        jsons = sorted(out_path.glob(f'{sub_prefix}*.json'))
    if jsons:
        iqm_json = str(jsons[0])

    # HTML reports
    reports = sorted(out_path.glob(f'{sub_prefix}*.html'))
    if reports:
        out_report = str(reports[0])

    return str(out_path), iqm_json, out_report


def _run_mriqc_group(bids_dir, output_dir):
    """Execute MRIQC group-level analysis.

    Parameters
    ----------
    bids_dir : str
        Path to the BIDS dataset root.
    output_dir : str
        Path containing participant-level MRIQC results.

    Returns
    -------
    group_tsv : str or None
        Path to group-level TSV file, or None on failure.

    """
    import subprocess
    from pathlib import Path

    out_path = Path(output_dir)

    cmd = [
        'mriqc',
        str(bids_dir),
        str(output_dir),
        'group',
        '--no-sub',
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        import warnings
        warnings.warn(
            f'MRIQC group-level failed: {exc}',
            RuntimeWarning,
        )
        return None

    # Return the first group TSV found
    tsvs = sorted(out_path.glob('group_*.tsv'))
    return str(tsvs[0]) if tsvs else None


def _parse_group_tsvs(mriqc_dir):
    """Parse group-level TSV files from MRIQC output.

    Parameters
    ----------
    mriqc_dir : str
        Path to MRIQC output directory (may be the group_tsv path
        from previous step — we extract the parent directory).

    Returns
    -------
    group_t1w_tsv : str or None
        Path to group T1w TSV.
    group_t2w_tsv : str or None
        Path to group T2w TSV.

    """
    from pathlib import Path

    # mriqc_dir may be a file path (group_tsv) — get directory
    p = Path(mriqc_dir)
    if p.is_file():
        p = p.parent

    t1w = p / 'group_T1w.tsv'
    t2w = p / 'group_T2w.tsv'

    return (
        str(t1w) if t1w.exists() else None,
        str(t2w) if t2w.exists() else None,
    )


def _extract_iqm_summary(iqm_json_path):
    """Extract key IQMs from an MRIQC JSON file.

    This is a utility for downstream QC gating — it parses the
    MRIQC JSON output and returns a dictionary with the most
    relevant metrics for neuro-oncology preprocessing.

    Parameters
    ----------
    iqm_json_path : str
        Path to MRIQC IQM JSON file.

    Returns
    -------
    dict
        Dictionary with key IQMs:
        - ``snr_total``: Total SNR
        - ``cnr``: Contrast-to-noise ratio
        - ``efc``: Entropy focus criterion
        - ``fber``: Foreground-background energy ratio
        - ``inu_med``: Median intensity non-uniformity
        - ``cjv``: Coefficient of joint variation
        - ``wm2max``: WM-to-maximum intensity ratio
        - ``qi_1``: Proportion of artifact voxels (Mortamet QI1)
        - ``pass_qc``: Boolean flag (True if metrics seem acceptable)

    """
    import json
    from pathlib import Path

    path = Path(iqm_json_path)
    if not path.exists():
        return {'error': f'IQM file not found: {iqm_json_path}', 'pass_qc': False}

    with open(path) as f:
        iqm = json.load(f)

    # Extract key metrics with defaults
    summary = {
        'snr_total': iqm.get('snr_total', None),
        'cnr': iqm.get('cnr', None),
        'efc': iqm.get('efc', None),
        'fber': iqm.get('fber', None),
        'inu_med': iqm.get('inu_med', None),
        'cjv': iqm.get('cjv', None),
        'wm2max': iqm.get('wm2max', None),
        'qi_1': iqm.get('qi_1', None),
    }

    # Basic QC gating heuristic for structural T1w
    # These thresholds are conservative defaults
    pass_qc = True
    qc_flags = []

    snr = summary.get('snr_total')
    if snr is not None and snr < 3.0:
        pass_qc = False
        qc_flags.append(f'Low SNR ({snr:.2f} < 3.0)')

    cjv = summary.get('cjv')
    if cjv is not None and cjv > 0.6:
        pass_qc = False
        qc_flags.append(f'High CJV ({cjv:.3f} > 0.6)')

    efc = summary.get('efc')
    if efc is not None and efc > 0.6:
        qc_flags.append(f'High EFC ({efc:.3f} > 0.6) — possible ghosting')

    qi1 = summary.get('qi_1')
    if qi1 is not None and qi1 > 0.05:
        qc_flags.append(f'Elevated artifact ratio QI1 ({qi1:.4f} > 0.05)')

    summary['pass_qc'] = pass_qc
    summary['qc_flags'] = qc_flags

    return summary


__all__ = [
    'init_mriqc_wf',
    'init_mriqc_group_wf',
]
