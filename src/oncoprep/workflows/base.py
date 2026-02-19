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

from .outputs import init_anat_reports_wf
from ..utils.labels import split_seg_labels


def _pick_first(val):
    """Extract the first element if *val* is a list, pass through otherwise."""
    if isinstance(val, (list, tuple)):
        return val[0] if val else None
    return val

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
from oncoprep.workflows.segment import init_anat_seg_wf
from oncoprep.workflows.nninteractive import init_nninteractive_seg_wf
from oncoprep.workflows.brats_outputs import init_ds_tumor_seg_wf
from oncoprep.workflows.radiomics import init_anat_radiomics_wf
from oncoprep.workflows.vasari import init_vasari_wf
# NOTE: MRIQC integration is temporarily disabled (non-functional).
# from oncoprep.workflows.mriqc import init_mriqc_wf
from ..interfaces import DerivativesDataSink, OncoprepBIDSDataGrabber
from ..interfaces.reports import SubjectSummary, AboutSummary
from .anatomical import init_anat_preproc_wf


LOGGER = logging.getLogger('nipype.workflow')

REFERENCES = """\
1. K. Gorgolewski et al., "Nipype: A Flexible, Lightweight and Extensible
   Neuroimaging Data Processing Framework in Python," *Front. Neuroinform.*,
   vol. 5, p. 13, 2011. https://doi.org/10.3389/fninf.2011.00013
2. K. Gorgolewski et al., "Nipype," *Software*.
   https://doi.org/10.5281/zenodo.596855
3. F. Isensee et al., "nnInteractive: Redefining 3D Promptable Segmentation,"
   *arXiv:2503.08373*, 2025. https://arxiv.org/abs/2503.08373
4. B.B. Avants et al., "A reproducible evaluation of ANTs similarity metric
   performance in brain image registration," *NeuroImage*, vol. 54, no. 3,
   pp. 2033–2044, 2011. https://doi.org/10.1016/j.neuroimage.2010.09.025
5. N.J. Tustison et al., "N4ITK: Improved N3 Bias Correction," *IEEE Trans.
   Med. Imaging*, vol. 29, no. 6, pp. 1310–1320, 2010.
   https://doi.org/10.1109/TMI.2010.2046908
6. F. Isensee et al., "Automated brain extraction of multisequence MRI using
   artificial neural networks," *Hum. Brain Mapp.*, vol. 40, no. 17,
   pp. 4952–4964, 2019. https://doi.org/10.1002/hbm.24750
7. A.M. Hoopes et al., "SynthStrip: Skull-Stripping for Any Brain Image,"
   *NeuroImage*, vol. 260, p. 119474, 2022.
   https://doi.org/10.1016/j.neuroimage.2022.119474
8. Y. Zhang, M. Brady, and S. Smith, "Segmentation of brain MR images through
   a hidden Markov random field model and the expectation-maximization
   algorithm," *IEEE Trans. Med. Imaging*, vol. 20, no. 1, pp. 45–57, 2001.
   https://doi.org/10.1109/42.906424
9. V. Fonov et al., "Unbiased average age-appropriate atlases for pediatric
   studies," *NeuroImage*, vol. 54, no. 1, pp. 313–327, 2011.
   https://doi.org/10.1016/j.neuroimage.2010.07.033
10. R. Ciric et al., "TemplateFlow: FAIR-sharing of multi-scale, multi-species
    brain models," *Nat. Methods*, vol. 19, pp. 1568–1571, 2022.
    https://doi.org/10.1038/s41592-022-01681-2
11. J.J.M. van Griethuysen et al., "Computational Radiomics System to Decode
    the Radiographic Phenotype," *Cancer Res.*, vol. 77, no. 21, pp. e104–e107,
    2017. https://doi.org/10.1158/0008-5472.CAN-17-0339
12. R.T. Shinohara et al., "Statistical normalization techniques for magnetic
    resonance imaging," *NeuroImage: Clinical*, vol. 6, pp. 9–19, 2014.
    https://doi.org/10.1016/j.nicl.2014.08.008
13. H. Um et al., "Impact of image preprocessing on the scanner dependence of
    multi-parametric MRI radiomic features and covariate shift in
    multi-institutional glioblastoma datasets," *Phys. Med. Biol.*, vol. 64,
    no. 16, p. 165011, 2019. https://doi.org/10.1088/1361-6560/ab2f44
"""


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
    use_gpu: bool = True,
    deface: bool = False,
    run_segmentation: bool = False,
    run_radiomics: bool = False,
    run_vasari: bool = False,
    run_qc: bool = False,
    seg_model_path: Optional[Path] = None,
    default_seg: bool = False,
    sloppy: bool = False,
    container_runtime: str = 'auto',
    seg_cache_dir: Optional[Path] = None,
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
        Enable GPU acceleration (default: True, use --no-gpu to disable)
    deface : bool
        Apply mri_deface to remove facial features for privacy (default: False)
    run_segmentation : bool
        Run tumor segmentation step (default: False, requires Docker; GPU used by default if available)
    run_vasari : bool
        Run automated VASARI feature extraction and radiology report
        generation (default: False, requires vasari-auto; implies
        --run-segmentation)
    run_qc : bool
        Run quality control on raw data using MRIQC (default: False)
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
(https://github.com/nikitas-k/oncoprep),
which is based on *Nipype* {nipype_ver}
(@nipype1; @nipype2; RRID:SCR_002502).

"""
    oncoprep_wf.__postdesc__ = """

For more details of the pipeline, see [the section corresponding
to workflows in *OncoPrep*'s documentation]\
(https://github.com/nikitas-k/oncoprep \
"OncoPrep's documentation").

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
            run_segmentation=run_segmentation,
            run_radiomics=run_radiomics,
            run_vasari=run_vasari,
            run_qc=run_qc,
            seg_model_path=seg_model_path,
            default_seg=default_seg,
            sloppy=sloppy,
            container_runtime=container_runtime,
            seg_cache_dir=seg_cache_dir,
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
    use_gpu: bool = True,
    deface: bool = False,
    run_segmentation: bool = False,
    run_radiomics: bool = False,
    run_vasari: bool = False,
    run_qc: bool = False,
    seg_model_path: Optional[Path] = None,
    default_seg: bool = False,
    sloppy: bool = False,
    container_runtime: str = 'auto',
    seg_cache_dir: Optional[Path] = None,
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
        Enable GPU acceleration (default: True, use --no-gpu to disable)
    deface : bool
        Apply mri_deface to remove facial features for privacy (default: False)
    run_segmentation : bool
        Run tumor segmentation (default: False, requires Docker; GPU used by default if available)
    run_qc : bool
        Run quality control on raw data using MRIQC (default: False)
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
Preprocessing of anatomical data for subject {subject_id}
was performed using OncoPrep {__version__}, which is based on Nipype {nipype_ver}
(@nipype1; @nipype2; RRID:SCR_002502).
"""
    workflow.__postdesc__ = """

For more details of the pipeline, see [the section corresponding
to workflows in *OncoPrep*'s documentation]\
(https://oncoprep.readthedocs.io/en/latest/workflows.html \
"OncoPrep's documentation").
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

    # Tumor segmentation workflow (optional)
    if run_segmentation:
        if default_seg:
            # Default segmentation: nnInteractive promptable model (no Docker required)
            LOGGER.info(
                'ANAT Stage 6: Initializing nnInteractive segmentation workflow '
                '(default_seg=True)'
            )
            anat_seg_wf = init_nninteractive_seg_wf(
                device='auto',
                name='anat_seg_wf',
            )
        else:
            # Ensemble / custom Docker-based segmentation
            LOGGER.info(
                'ANAT Stage 6: Initializing Docker-based segmentation workflow '
                '(default_seg=False)'
            )
            anat_seg_wf = init_anat_seg_wf(
                output_dir=output_dir,
                use_gpu=use_gpu,
                model_path=seg_model_path,
                default_model=False,
                sloppy=sloppy,
                container_runtime=container_runtime,
                seg_cache_dir=seg_cache_dir,
                name='anat_seg_wf',
            )
        
        # Datasink for tumor segmentation output
        ds_tumor_seg_wf = init_ds_tumor_seg_wf(
            output_dir=str(output_dir),
            name='ds_tumor_seg_wf',
        )

        if default_seg:
            # nnInteractive: raw BIDS images for T1w/T1ce/T2w (same 1mm grid).
            # FLAIR uses the preprocessed (registered) version because the
            # raw FLAIR typically has thick slices (e.g. 3mm) whose simple
            # affine resampling to the 1mm T1w grid is too lossy for the
            # WT segmentation step.
            workflow.connect([
                (bidssrc, anat_seg_wf, [
                    (('t1w', fix_multi_T1w_source_name), 'inputnode.source_file'),
                    (('t1w', fix_multi_T1w_source_name), 'inputnode.t1w'),
                    (('t1ce', _pick_first), 'inputnode.t1ce'),
                    (('t2w', _pick_first), 'inputnode.t2w'),
                ]),
                (anat_preproc_wf, anat_seg_wf, [
                    ('outputnode.flair_preproc', 'inputnode.flair'),
                    ('outputnode.anat2std_xfm', 'inputnode.anat2std_xfm'),
                ]),
            ])
        else:
            # Docker ensemble: preprocessed modalities
            workflow.connect([
                (bidssrc, anat_seg_wf, [
                    (('t1w', fix_multi_T1w_source_name), 'inputnode.source_file'),
                ]),
                (anat_preproc_wf, anat_seg_wf, [
                    ('outputnode.t1w_brain', 'inputnode.t1w_preproc'),
                    ('outputnode.t1ce_preproc', 'inputnode.t1ce_preproc'),
                    ('outputnode.t2w_preproc', 'inputnode.t2w_preproc'),
                    ('outputnode.flair_preproc', 'inputnode.flair_preproc'),
                    ('outputnode.t1w_mask', 'inputnode.brain_mask'),
                    ('outputnode.anat2std_xfm', 'inputnode.anat2std_xfm'),
                ]),
            ])

        # Resolve the template-space atlas reference image for resampling
        # the tumor segmentation. Uses the first output space to determine
        # which atlas set (MNI152 or SRI24) to use.
        from ..interfaces.vasari import get_atlas_reference
        _atlas_space = std_spaces[0] if std_spaces else 'MNI152NLin2009cAsym'
        _std_ref = get_atlas_reference(_atlas_space)
        anat_seg_wf.get_node('inputnode').inputs.std_reference = _std_ref

        workflow.connect([
            # Save tumor segmentation to BIDS derivatives
            (bidssrc, ds_tumor_seg_wf, [
                (('t1w', fix_multi_T1w_source_name), 'inputnode.source_file'),
            ]),
            (anat_seg_wf, ds_tumor_seg_wf, [
                ('outputnode.tumor_seg', 'inputnode.tumor_seg'),
                ('outputnode.tumor_seg_old', 'inputnode.tumor_seg_old'),
                ('outputnode.tumor_seg_new', 'inputnode.tumor_seg_new'),
            ]),
        ])

        # Tumor segmentation report (runs after segmentation, outside anat_preproc_wf to avoid cycle)
        split_seg = pe.Node(
            niu.Function(
                function=split_seg_labels,
                input_names=['seg_file'],
                output_names=['mask_files'],
            ),
            name='split_seg_labels',
        )

        from ..interfaces.reports import TumorROIsPlot
        tumor_rpt = pe.Node(
            TumorROIsPlot(
                colors=['red', 'gold', 'lime', 'cyan'],
                levels=[0.5],
                legend_labels=[
                    ('red', 'NCR \u2014 Necrotic Core'),
                    ('gold', 'ED \u2014 Peritumoral Edema'),
                    ('lime', 'ET \u2014 Enhancing Tumor'),
                    ('cyan', 'RC \u2014 Resection Cavity'),
                ],
            ),
            name='tumor_rpt',
        )

        ds_tumor_dseg_report = pe.Node(
            DerivativesDataSink(
                base_directory=str(output_dir),
                desc='tumor',
                suffix='dseg',
                datatype='figures',
            ),
            name='ds_tumor_dseg_report',
            run_without_submitting=True,
        )
        workflow.connect([
            (anat_preproc_wf, tumor_rpt, [
                ('outputnode.t1w_preproc', 'in_file'),
                ('outputnode.t1w_mask', 'in_mask'),
            ]),
            (anat_seg_wf, split_seg, [
                ('outputnode.tumor_seg_old', 'seg_file'),
            ]),
            (split_seg, tumor_rpt, [
                ('mask_files', 'in_rois'),
            ]),
            (bidssrc, ds_tumor_dseg_report, [
                (('t1w', fix_multi_T1w_source_name), 'source_file'),
            ]),
            (tumor_rpt, ds_tumor_dseg_report, [('out_report', 'in_file')]),
        ])


    # Radiomics feature extraction workflow (optional, requires segmentation)
    if run_radiomics and run_segmentation:
        try:
            import radiomics  # noqa: F401
        except ImportError:
            LOGGER.warning(
                'pyradiomics is not installed — skipping radiomics feature extraction. '
                'Install with: pip install pyradiomics  '
                '(Note: pyradiomics 3.0.1 requires Python <3.12; '
                'for 3.12+ install from git: pip install git+https://github.com/AIM-Harvard/pyradiomics.git)'
            )
            run_radiomics = False

    if run_radiomics and run_segmentation:
        LOGGER.info('ANAT Stage 8: Initializing radiomics feature extraction workflow (--run-radiomics=True requires --run-segmentation=True)')
        anat_radiomics_wf = init_anat_radiomics_wf(
            output_dir=str(output_dir),
            name='anat_radiomics_wf',
        )

        workflow.connect([
            (bidssrc, anat_radiomics_wf, [
                (('t1w', fix_multi_T1w_source_name), 'inputnode.source_file'),
            ]),
            (anat_preproc_wf, anat_radiomics_wf, [
                ('outputnode.t1w_preproc', 'inputnode.t1w_preproc'),
                ('outputnode.t1w_mask', 'inputnode.brain_mask'),
            ]),
            (anat_seg_wf, anat_radiomics_wf, [
                ('outputnode.tumor_seg_old', 'inputnode.tumor_seg'),
            ]),
        ])

    # VASARI feature extraction and radiology report (optional, requires segmentation)
    if run_vasari and run_segmentation:
        try:
            from ..interfaces.vasari import _import_vasari_auto  # noqa: F401
            _import_vasari_auto()  # validates vasari-auto is importable
        except ImportError:
            LOGGER.warning(
                'vasari-auto is not installed — skipping VASARI feature extraction. '
                'Install with: pip install vasari-auto  '
                'or: pip install -e /path/to/vasari-auto'
            )
            run_vasari = False

    if run_vasari and run_segmentation:
        LOGGER.info(
            'ANAT Stage 9: Initializing VASARI feature extraction workflow '
            '(--run-vasari=True requires --run-segmentation=True)'
        )
        anat_vasari_wf = init_vasari_wf(
            output_dir=str(output_dir),
            atlas_space=_atlas_space,
            name='anat_vasari_wf',
        )

        workflow.connect([
            (bidssrc, anat_vasari_wf, [
                (('t1w', fix_multi_T1w_source_name), 'inputnode.source_file'),
            ]),
            (anat_seg_wf, anat_vasari_wf, [
                ('outputnode.tumor_seg_std', 'inputnode.tumor_seg_std'),
            ]),
            (bids_info, anat_vasari_wf, [
                (('subject', _prefix, session_id), 'inputnode.subject_id'),
            ]),
        ])

    # MRIQC quality control workflow — TEMPORARILY DISABLED
    # The MRIQC integration is non-functional in this release.
    # It will be re-enabled in a future version.
    if run_qc:
        import warnings
        warnings.warn(
            '--run-qc / run_qc=True was requested but MRIQC integration is '
            'temporarily disabled in this release. The flag will be ignored. '
            'See https://github.com/nibabies/oncoprep for updates.',
            UserWarning,
            stacklevel=2,
        )
        run_qc = False

    # ---- Collate all figures into a single sub-<label>.html report ----
    from ..utils.collate import collate_subject_report as _collate_fn

    # Merge sentinel signals from every report-writing datasink so the
    # collation node runs only after ALL figures have been written.
    n_sentinels = 3
    if run_segmentation:
        n_sentinels += 1  # tumor dseg report
    if run_radiomics and run_segmentation:
        n_sentinels += 1  # radiomics report
    if run_vasari and run_segmentation:
        n_sentinels += 1  # vasari report
    # NOTE: MRIQC sentinel disabled (non-functional)
    # if run_qc:
    #     n_sentinels += 1  # mriqc report
    report_sentinel_merge = pe.Node(
        niu.Merge(n_sentinels, no_flatten=True),
        name='report_sentinel_merge',
    )

    collate_report = pe.Node(
        niu.Function(
            function=_collate_fn,
            input_names=['output_dir', 'subject_id', 'version',
                         'report_files', 'workflow_desc',
                         'references'],
            output_names=['out_report'],
        ),
        name='collate_report',
        run_without_submitting=True,
    )
    collate_report.inputs.output_dir = str(output_dir)
    collate_report.inputs.subject_id = f'sub-{subject_id}'
    collate_report.inputs.version = __version__
    collate_report.inputs.workflow_desc = workflow.visit_desc() or ''
    collate_report.inputs.references = REFERENCES

    workflow.connect([
        (ds_report_summary, report_sentinel_merge, [('out_file', 'in1')]),
        (ds_report_about, report_sentinel_merge, [('out_file', 'in2')]),
        (anat_preproc_wf, report_sentinel_merge, [
            ('outputnode.anat2std_xfm', 'in3'),
        ]),
    ])

    _sentinel_idx = 3
    if run_segmentation:
        _sentinel_idx += 1
        workflow.connect([
            (ds_tumor_dseg_report, report_sentinel_merge, [('out_file', f'in{_sentinel_idx}')]),
        ])

    if run_radiomics and run_segmentation:
        _sentinel_idx += 1
        workflow.connect([
            (anat_radiomics_wf, report_sentinel_merge, [
                ('ds_radiomics_report.out_file', f'in{_sentinel_idx}'),
            ]),
        ])

    if run_vasari and run_segmentation:
        _sentinel_idx += 1
        workflow.connect([
            (anat_vasari_wf, report_sentinel_merge, [
                ('ds_vasari_report.out_file', f'in{_sentinel_idx}'),
            ]),
        ])

    # NOTE: MRIQC sentinel disabled (non-functional)
    # if run_qc:
    #     _sentinel_idx += 1
    #     workflow.connect([
    #         (mriqc_wf, report_sentinel_merge, [
    #             ('outputnode.mriqc_dir', f'in{_sentinel_idx}'),
    #         ]),
    #     ])

    workflow.connect([
        (report_sentinel_merge, collate_report, [('out', 'report_files')]),
    ])

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
