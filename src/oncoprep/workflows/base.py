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
from niworkflows.utils.bids import collect_data, BIDSLayout, DEFAULT_BIDS_QUERIES
from niworkflows.interfaces.bids import BIDSInfo
from niworkflows.utils.misc import fix_multi_T1w_source_name
from bids.layout import Query

# Custom BIDS queries for OncoPrep (adds t1ce for neuro-oncology)
# Uses ceagent entity per BIDS spec: T1ce = T1w with contrast agent (e.g., gadolinium)
import copy
ONCOPREP_BIDS_QUERIES = copy.deepcopy(DEFAULT_BIDS_QUERIES)
# T1ce: T1w images WITH ceagent entity (contrast-enhanced)
ONCOPREP_BIDS_QUERIES['t1ce'] = {
    'datatype': 'anat',
    'suffix': 'T1w',
    'ceagent': Query.REQUIRED,  # Must have ceagent entity
    'part': ['mag', None],
}
# Update T1w query to EXCLUDE images with ceagent (avoid duplicates)
ONCOPREP_BIDS_QUERIES['t1w'] = {
    'datatype': 'anat',
    'suffix': 'T1w',
    'ceagent': Query.NONE,  # Must NOT have ceagent entity
    'part': ['mag', None],
}

from oncoprep import __version__
from oncoprep.workflows.preproc import build_preproc_workflow
from oncoprep.workflows.outputs import init_ds_mask_wf, init_ds_modalities_wf, init_ds_template_wf
from oncoprep.workflows.reports import init_report_wf
from oncoprep.workflows.metrics import (
    init_qa_metrics_wf,
    init_snr_metrics_wf,
)
from ..interfaces import DerivativesDataSink, OncoprepBIDSDataGrabber
from ..interfaces.reports import SubjectSummary, AboutSummary
from .anatomical import init_anat_preproc_wf


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
    skull_strip_backend: str = 'ants',
    registration_backend: str = 'ants',
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
    skull_strip_backend : str
        Skull stripping backend: 'ants', 'hdbet', or 'synthstrip'
    registration_backend : str
        Registration backend: 'ants' (ANTs SyN) or 'greedy' (PICSL Greedy)
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
            session_id=session_ids,
            bids_dir=bids_dir,
            output_dir=output_dir,
            omp_nthreads=omp_nthreads,
            mem_gb=mem_gb,
            skull_strip_template=skull_strip_template,
            skull_strip_fixed_seed=skull_strip_fixed_seed,
            skull_strip_mode=skull_strip_mode,
            skull_strip_backend=skull_strip_backend,
            registration_backend=registration_backend,
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
    session_id: Union[str, list, None],
    bids_dir: Path,
    output_dir: Path,
    derivatives: List[Path] = [],
    layout: Optional[BIDSLayout] = None,
    bids_filters: Optional[dict] = None,
    omp_nthreads: int = 1,
    mem_gb: Optional[float] = None,
    skull_strip_template: str = 'OASIS30ANTs',
    skull_strip_fixed_seed: bool = False,
    skull_strip_mode: str = 'auto',
    skull_strip_backend: str = 'ants',
    registration_backend: str = 'ants',
    longitudinal: bool = False,
    output_spaces: Optional[list] = None,
    use_gpu: bool = False,
    deface: bool = False,
    skip_segmentation: bool = True,
    sloppy: bool = False,
    name: str = 'single_subject_wf',
    debug=False,
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
                session_id=None,
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
    skull_strip_backend : str
        Skull stripping backend ('ants', 'hdbet', or 'synthstrip')
    registration_backend : str
        Registration backend ('ants' or 'greedy')
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
    # Create layout if not provided
    if layout is None:
        layout = BIDSLayout(str(bids_dir), validate=False, derivatives=False)

    # Use custom queries that include t1ce for neuro-oncology data
    subject_data = collect_data(
        layout, subject_id, session_id=session_id, bids_filters=bids_filters,
        queries=ONCOPREP_BIDS_QUERIES,
    )[0]

    if output_spaces is None:
        output_spaces = ['MNI152NLin2009cAsym']

    workflow = Workflow(name=name)
    workflow.__desc__ = f"""
Preprocessing of anatomical and functional data for subject {subject_id}
was performed using OncoPrep {__version__}, which is based on Nipype {nipype_ver}.
(@nipype1; @nipype2; RRID:SCR_002502).
"""
    workflow.__postdesc__ = """

For more details of the pipeline, see [the section corresponding
to workflows in *OncoPrep*'s documentation]\
(https://oncoprep.readthedocs.io/en/latest/workflows.html \
"OncoPrep's documentation").

### References

"""

    from ..utils.bids import collect_derivatives

    deriv_cache = {}
    std_spaces = output_spaces.get_spaces(nonstandard=False) if hasattr(output_spaces, 'get_spaces') else output_spaces
    for deriv_dir in derivatives:
        deriv_cache.update(
            collect_derivatives(
                bids_dir=bids_dir,
                deriv_dir=deriv_dir,
                subject_id=subject_id,
                session_id=session_id,
                spaces=std_spaces,
            )
        )

    # Input node - currently minimal, could be extended for derivatives
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['subject_id', 'subjects_dir']),
        name='inputnode',
    )
    bidssrc = pe.Node(
        OncoprepBIDSDataGrabber(subject_data=subject_data),
        name='bidssrc',
    )

    bids_info = pe.Node(
        BIDSInfo(bids_dir=bids_dir),
        name='bids_info',
        run_without_submitting=True,
    )

    summary = pe.Node(
        SubjectSummary(output_spaces=std_spaces),
        name='summary',
        run_without_submitting=True,
    )

    about = pe.Node(
        AboutSummary(version=__version__, command=" ".join(sys.argv)),
        name='about',
        run_without_submitting=True,
    )

    dismiss_entities = ('session',) if session_id else None

    ds_report_summary = pe.Node(
        DerivativesDataSink(
            base_directory=str(output_dir),
            dismiss_entities=dismiss_entities,
            desc='summary',
            datatype='figures',
        ),
        name='ds_report_summary',
        run_without_submitting=True,
    )

    ds_report_about = pe.Node(
        DerivativesDataSink(
            base_directory=str(output_dir),
            dismiss_entities=dismiss_entities,
            desc='about',
            datatype='figures',
        ),
        name='ds_report_about',
        run_without_submitting=True,
    )

    # preprocessing of anat (includes registration to MNI)
    anat_preproc_wf = init_anat_preproc_wf(
        bids_dir=bids_dir,
        output_dir=output_dir,
        sloppy=sloppy,
        debug=debug,
        precomputed=deriv_cache,
        longitudinal=longitudinal,
        name='anat_preproc_wf',
        t1w=subject_data.get('t1w', None),
        t1ce=subject_data.get('t1ce', None),
        t2w=subject_data.get('t2w', None),
        flair=subject_data.get('flair', None),
        output_spaces=std_spaces,
        skull_strip_template=skull_strip_template,
        skull_strip_fixed_seed=skull_strip_fixed_seed,
        skull_strip_mode=skull_strip_mode,
        skull_strip_backend=skull_strip_backend,
        registration_backend=registration_backend,
        omp_nthreads=omp_nthreads,
    )
    
    workflow.connect([
        (inputnode, anat_preproc_wf, [('subjects_dir', 'inputnode.subjects_dir')]),
        (bidssrc, bids_info, [(('t1w', fix_multi_T1w_source_name), 'in_file')]),
        (inputnode, summary, [('subjects_dir', 'subjects_dir')]),
        (bidssrc, summary, [('t1w', 't1w'),
                            ('t1ce', 't1ce'),
                            ('t2w', 't2w'),
                            ('flair', 'flair')]),
        (bids_info, summary, [('subject', 'subject_id')]),
        (bids_info, anat_preproc_wf, [(('subject', _prefix, session_id), 'inputnode.subject_id')]),
        (bidssrc, anat_preproc_wf, [('t1w', 'inputnode.t1w'),
                                    ('t1ce', 'inputnode.t1ce'),
                                    ('t2w', 'inputnode.t2w'),
                                    ('flair', 'inputnode.flair')]),
        (bidssrc, ds_report_summary, [(('t1w', fix_multi_T1w_source_name), 'source_file')]),
        (summary, ds_report_summary, [('out_report', 'in_file')]),
        (bidssrc, ds_report_about, [(('t1w', fix_multi_T1w_source_name), 'source_file')]),
        (about, ds_report_about, [('out_report', 'in_file')]),
    ])

    # if not skip_segmentation:
    #     anat_seg_wf = init_anat_seg_wf(

    #     ) # TODO: segmentation workflow
    #     workflow.connect([
    #         (anat_preproc_wf, anat_seg_wf, [('outputnode.brain_mask', 'inputnode.brain_mask'),
    #                                     ('outputnode.t1w_preproc', 'inputnode.t1w_preproc')]),
    #     ])

    return workflow

def _prefix(subject_id, session_id):
    if not subject_id.startswith('sub-'):
        subject_id = f'sub-{subject_id}'

    if session_id:
        ses_str = session_id
        if isinstance(session_id, list):
            from ..utils.misc import stringify_sessions
            ses_str = stringify_sessions(session_id)
        if not ses_str.startswith('ses-'):
            ses_str = f'ses-{ses_str}'
        subject_id += f'_{ses_str}'

    return subject_id
