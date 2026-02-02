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
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""OncoPrep base processing workflows for multi-subject execution."""

from __future__ import annotations

import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Union

from nipype import __version__ as nipype_ver
from nipype import logging
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from oncoprep import __version__
from oncoprep.workflows.preproc import build_preproc_workflow
from oncoprep.workflows.outputs import init_ds_mask_wf, init_ds_template_wf
from oncoprep.workflows.reports import init_report_wf
from oncoprep.workflows.metrics import (
    init_qa_metrics_wf,
    init_snr_metrics_wf,
)

LOGGER = logging.getLogger('nipype.workflow')


def init_oncoprep_wf(
    *,
    output_dir: Path,
    subject_session_list: list,
    run_uuid: str,
    work_dir: Path,
    bids_dir: Path,
    omp_nthreads: int = 1,
    nprocs: int = 1,
    mem_gb: Optional[float] = None,
    skull_strip_template: str = 'OASIS30ANTs',
    skull_strip_fixed_seed: bool = False,
    skull_strip_mode: str = 'auto',
    longitudinal: bool = False,
    output_spaces: Optional[list] = None,
    use_gpu: bool = False,
    deface: bool = False,
    skip_segmentation: bool = True,
    sloppy: bool = False,
) -> Workflow:
    """
    Create the execution graph of OncoPrep for multi-subject processing.

    Parameters
    ----------
    output_dir : Path
        Directory in which to save derivatives
    subject_session_list : list of tuple
        List of (subject_id, session_ids) tuples for processing
    run_uuid : str
        Unique identifier for execution instance
    work_dir : Path
        Directory for workflow execution state and temporary files
    bids_dir : Path
        Root directory of BIDS dataset
    omp_nthreads : int
        Maximum number of threads per process
    nprocs : int
        Number of parallel processes
    mem_gb : float | None
        Memory limit in GB
    skull_strip_template : str
        Template for skull stripping (default: OASIS30ANTs)
    skull_strip_fixed_seed : bool
        Use fixed random seed for reproducibility
    skull_strip_mode : str
        Skull stripping mode: 'auto', 'skip', or 'force'
    longitudinal : bool
        Treat as longitudinal dataset
    output_spaces : list | None
        Target template spaces (default: ['MNI152NLin2009cAsym'])
    use_gpu : bool
        Enable GPU acceleration if available
    deface : bool
        Apply mri_deface to remove facial features for privacy (default: False)
    skip_segmentation : bool
        Skip tumor segmentation step (default: True)
    sloppy : bool
        Use faster settings for testing

    Returns
    -------
    Workflow
        Top-level workflow for multi-subject processing
    """
    if output_spaces is None:
        output_spaces = ['MNI152NLin2009cAsym']

    oncoprep_wf = Workflow(name='oncoprep_wf')
    oncoprep_wf.base_dir = str(work_dir)

    oncoprep_wf.__desc__ = f"""
Results included in this manuscript come from preprocessing
performed using *OncoPrep* {__version__}
(https://github.com/nibabies/oncoprep),
which is based on *Nipype* {nipype_ver}
(@nipype1; @nipype2; RRID:SCR_002502).

"""
    oncoprep_wf.__postdesc__ = """

For more details of the pipeline, see [the section corresponding
to workflows in *OncoPrep*'s documentation]\
(https://github.com/nibabies/oncoprep \
"OncoPrep's documentation").

### References

"""

    # Create a single-subject workflow for each subject/session combination
    for subject_id, session_ids in subject_session_list:
        # Build workflow name
        # ('01', None) -> sub-01_wf
        # ('01', 'pre') -> sub-01_ses-pre_wf
        # ('01', ['pre', 'post']) -> sub-01_ses-pre-post_wf
        name = f'sub-{subject_id}_wf'
        if session_ids:
            ses_str = session_ids
            if isinstance(session_ids, list):
                ses_str = '_'.join(session_ids)

            name = f'sub-{subject_id}_ses-{ses_str}_wf'

        single_subject_wf = init_single_subject_wf(
            subject_id=subject_id,
            session_ids=session_ids,
            bids_dir=bids_dir,
            output_dir=output_dir,
            work_dir=work_dir,
            run_uuid=run_uuid,
            omp_nthreads=omp_nthreads,
            mem_gb=mem_gb,
            skull_strip_template=skull_strip_template,
            skull_strip_fixed_seed=skull_strip_fixed_seed,
            skull_strip_mode=skull_strip_mode,
            longitudinal=longitudinal,
            output_spaces=output_spaces,
            use_gpu=use_gpu,
            deface=deface,
            skip_segmentation=skip_segmentation,
            sloppy=sloppy,
            name=name,
        )

        # Configure crash dump directory
        single_subject_wf.config['execution']['crashdump_dir'] = os.path.join(
            output_dir, 'oncoprep', f'sub-{subject_id}', 'log', run_uuid
        )
        for node in single_subject_wf._get_all_nodes():
            node.config = deepcopy(single_subject_wf.config)

        oncoprep_wf.add_nodes([single_subject_wf])

    return oncoprep_wf


def init_single_subject_wf(
    *,
    subject_id: str,
    session_ids: Union[str, list, None],
    bids_dir: Path,
    output_dir: Path,
    work_dir: Path,
    run_uuid: str,
    omp_nthreads: int = 1,
    mem_gb: Optional[float] = None,
    skull_strip_template: str = 'OASIS30ANTs',
    skull_strip_fixed_seed: bool = False,
    skull_strip_mode: str = 'auto',
    longitudinal: bool = False,
    output_spaces: Optional[list] = None,
    use_gpu: bool = False,
    deface: bool = False,
    skip_segmentation: bool = True,
    sloppy: bool = False,
    name: str = 'single_subject_wf',
) -> Workflow:
    """
    Create a single subject preprocessing workflow.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from pathlib import Path
            from oncoprep.workflows.base import init_single_subject_wf
            wf = init_single_subject_wf(
                subject_id='01',
                session_ids=None,
                bids_dir=Path('.'),
                output_dir=Path('derivatives'),
                work_dir=Path('work'),
                run_uuid='testrun',
                omp_nthreads=1,
            )

    Parameters
    ----------
    subject_id : str
        Subject identifier (without 'sub-' prefix)
    session_ids : str, list, or None
        Session identifier(s)
    bids_dir : Path
        Root directory of BIDS dataset
    output_dir : Path
        Directory for saving derivatives
    work_dir : Path
        Working directory for execution
    run_uuid : str
        Unique run identifier
    omp_nthreads : int
        Maximum threads per process
    mem_gb : float | None
        Memory limit in GB
    skull_strip_template : str
        Template for skull stripping
    skull_strip_fixed_seed : bool
        Use fixed seed for reproducibility
    skull_strip_mode : str
        Skull stripping mode ('auto', 'skip', or 'force')
    longitudinal : bool
        Treat as longitudinal
    output_spaces : list | None
        Target template spaces
    use_gpu : bool
        Enable GPU acceleration
    deface : bool
        Apply mri_deface to remove facial features for privacy (default: False)
    skip_segmentation : bool
        Skip segmentation
    sloppy : bool
        Use faster settings
    name : str
        Workflow name

    Returns
    -------
    Workflow
        Single-subject preprocessing workflow
    """
    if output_spaces is None:
        output_spaces = ['MNI152NLin2009cAsym']

    workflow = Workflow(name=name)
    workflow.__desc__ = f"""
Preprocessing of anatomical and functional data for subject {subject_id}
was performed using OncoPrep.
"""

    # Input node - currently minimal, could be extended for derivatives
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['subject_id']),
        name='inputnode',
    )

    # Output node
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                't1w_preproc',
                't1w_mask',
                't1ce_preproc',
                't2w_preproc',
                'flair_preproc',
                'anat2std_xfm',
                'std2anat_xfm',
                'template',
            ]
        ),
        name='outputnode',
    )

    # Create preprocessing workflow
    anat_preproc_wf = build_preproc_workflow(
        bids_dir=bids_dir,
        output_dir=output_dir,
        participant_label=[subject_id],
        session_label=session_ids if isinstance(session_ids, list) else ([session_ids] if session_ids else None),
        nprocs=1,  # Single subject workflow
        omp_nthreads=omp_nthreads,
        mem_gb=mem_gb,
        skull_strip_template=skull_strip_template,
        skull_strip_fixed_seed=skull_strip_fixed_seed,
        skull_strip_mode=skull_strip_mode,
        longitudinal=longitudinal,
        output_spaces=output_spaces,
        use_gpu=use_gpu,
        deface=deface,
        skip_segmentation=skip_segmentation,
        sloppy=sloppy,
    )

    # Connect workflow
    workflow.connect([
        (anat_preproc_wf, outputnode, [
            ('outputnode.t1w_preproc', 't1w_preproc'),
            ('outputnode.t1w_mask', 't1w_mask'),
            ('outputnode.t1ce_preproc', 't1ce_preproc'),
            ('outputnode.t2w_preproc', 't2w_preproc'),
            ('outputnode.flair_preproc', 'flair_preproc'),
            ('outputnode.anat2std_xfm', 'anat2std_xfm'),
            ('outputnode.std2anat_xfm', 'std2anat_xfm'),
            ('outputnode.template', 'template'),
        ]),
    ])

    # Connect output workflows for BIDS derivatives
    ds_mask_wf = init_ds_mask_wf(
        bids_root=str(bids_dir),
        output_dir=str(output_dir),
        mask_type='brain',
    )
    ds_mask_wf.inputs.inputnode.source_files = [
        str(bids_dir / f'sub-{subject_id}' / 'anat' / f'sub-{subject_id}_T1w.nii.gz')
    ]

    ds_template_wf = init_ds_template_wf(
        num_anat=1,
        output_dir=str(output_dir),
        image_type='T1w',
    )

    # Report generation
    report_wf = init_report_wf(
        output_dir=str(output_dir),
        subject_label=subject_id,
        session_label=None if isinstance(session_ids, type(None)) else (
            session_ids if isinstance(session_ids, str) else session_ids[0]
        ),
    )

    # QA metrics
    qa_wf = init_qa_metrics_wf(
        output_dir=str(output_dir),
    )

    snr_wf = init_snr_metrics_wf(
        output_dir=str(output_dir),
    )

    workflow.connect([
        (anat_preproc_wf, ds_mask_wf, [
            ('outputnode.t1w_mask', 'inputnode.mask_file'),
        ]),
        (anat_preproc_wf, ds_template_wf, [
            ('outputnode.t1w_preproc', 'inputnode.anat_preproc'),
        ]),
        (anat_preproc_wf, report_wf, [
            ('outputnode.t1w_preproc', 'inputnode.anat_preproc'),
            ('outputnode.t1w_mask', 'inputnode.anat_mask'),
        ]),
        (anat_preproc_wf, qa_wf, [
            ('outputnode.t1w_preproc', 'inputnode.anat_preproc'),
            ('outputnode.t1w_mask', 'inputnode.anat_mask'),
        ]),
        (anat_preproc_wf, snr_wf, [
            ('outputnode.t1w_preproc', 'inputnode.anat_preproc'),
            ('outputnode.t1w_mask', 'inputnode.anat_mask'),
        ]),
    ])

    # Propagate preprocessing workflow description
    workflow.__desc__ += anat_preproc_wf.__desc__

    return workflow
