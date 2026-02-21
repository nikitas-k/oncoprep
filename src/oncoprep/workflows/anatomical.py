# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Original sMRIprep header:
#
# Copyright 2025 The NiPreps Developers <nipreps@gmail.com>
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
# CHANGES FROM SMRIPREP v2.5.3 (by OncoPrep developers):
# - Adapted for OncoPrep and BraTS data
# - Unimplemented FreeSurfer surface reconstruction steps for simplicity
# - Included inputs/outputs to focus on T1w, T1ce, T2w, and FLAIR modalities
#
"""OncoPrep anatomical (T1w, T2w, FLAIR) preprocessing workflows for BraTS data."""

from pathlib import Path
from typing import List, Optional, Union

import nibabel as nb
import numpy as np
from nipype import logging
from nipype.interfaces import ants, utility as niu, image
from nipype.pipeline import engine as pe
from nipype.interfaces.ants.base import Info as ANTsInfo

from niworkflows.interfaces.header import ValidateImage
from niworkflows.interfaces.images import Conform, TemplateDimensions
from niworkflows.interfaces.nibabel import ApplyMask, Binarize
from niworkflows.interfaces.nitransforms import ConcatenateXFMs
from niworkflows.interfaces.freesurfer import (
    PatchedLTAConvert as LTAConvert,
    StructuralReference,
)

from oncoprep.workflows._compat import Workflow, tag
from niworkflows.anat.ants import init_brain_extraction_wf, init_n4_only_wf
from niworkflows.utils.spaces import Reference, SpatialReferences
from niworkflows.utils.misc import add_suffix

from .fit.registration import init_multimodal_template_registration_wf
from .outputs import (
    _pop,
    init_anat_reports_wf,
    init_ds_anat_volumes_wf,
    init_ds_dseg_wf,
    init_ds_mask_wf,
    init_ds_modalities_wf,
    init_ds_template_registration_wf,
    init_ds_template_wf,
    init_ds_tpms_wf,
    init_template_iterator_wf,
)
from ..interfaces.fsl import FAST
from ..data import load as load_data

from ..utils.misc import apply_lut as _apply_bids_lut
# from .surfaces import (
#     init_anat_ribbon_wf,
# ) # --- IGNORE --- (surface pipeline needs testing)

LOGGER = logging.getLogger('nipype.workflow')


def init_anat_preproc_wf(
    *,
    bids_dir: Union[Path, str],
    output_dir: Union[Path, str],
    t1w: list,
    t1ce: Optional[list] = None,
    t2w: Optional[list] = None,
    flair: Optional[list] = None,
    longitudinal: bool = False,
    output_spaces: Optional[SpatialReferences] = None,
    skull_strip_mode: str = 'auto',
    skull_strip_template: Union[str, Reference] = 'OASIS30ANTs',
    skull_strip_fixed_seed: bool = False,
    skull_strip_backend: str = 'ants',
    registration_backend: str = 'ants',
    omp_nthreads: int = 1,
    debug: bool = False,
    precomputed: dict = {},
    use_gpu: bool = False,
    defacing: bool = False,
    sloppy: bool = False,
    skip_registration: bool = False,
    name: str = 'anat_preproc_wf',
):
    """
    Stage the anatomical preprocessing steps of OncoPrep for BraTS data.

    This workflow handles:
      - T1w reference: averaging multiple T1w images if present
      - Brain extraction using ANTs-based methods
      - Intensity non-uniformity (INU) correction with N4
      - Multi-modal co-registration (align T1ce, T2w, FLAIR to T1w)
      - Optional defacing using mri_deface (for privacy)
      - Spatial normalization to standard templates

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from oncoprep.workflows.anatomical import init_anat_preproc_wf
            wf = init_anat_preproc_wf(
                t1w=['t1w.nii.gz'],
                t1ce=['t1ce.nii.gz'],
                t2w=['t2w.nii.gz'],
                flair=['flair.nii.gz'],
                output_spaces=['MNI152NLin2009cAsym'],
                omp_nthreads=4,
            )

    Parameters
    ----------
    t1w : :obj:`list`
        List of T1-weighted structural images
    t1ce : :obj:`list`, optional
        List of T1-weighted contrast-enhanced images
    t2w : :obj:`list`, optional
        List of T2-weighted images
    flair : :obj:`list`, optional
        List of FLAIR images
    output_spaces : :obj:`list`, optional
        Target template spaces for registration (default: ['MNI152NLin2009cAsym'])
    skull_strip_template : :obj:`str` or :class:`~niworkflows.utils.spaces.Reference`
        Template for skull stripping (default: 'OASIS30ANTs'). If a string is provided,
        it will be converted to a Reference object.
    skull_strip_fixed_seed : :obj:`bool`
        Use fixed seed for reproducibility in brain extraction (default: False)
    skull_strip_backend : :obj:`str`
        Backend for skull stripping: 'ants' (ANTs brain extraction),
        'hdbet' (HD-BET, GPU recommended), 'synthstrip' (FreeSurfer SynthStrip).
        Default: 'ants'
    omp_nthreads : :obj:`int`
        Maximum number of threads an individual process may use
    use_gpu : :obj:`bool`
        Enable GPU acceleration for HD-BET (default: False)
    defacing : :obj:`bool`
        Apply mri_deface to remove facial features for privacy (default: False)
    sloppy : :obj:`bool`
        Quick, imprecise operations for testing (default: False)
    name : :obj:`str`, optional
        Workflow name (default: anat_preproc_wf)

    Inputs
    ------
    t1w
        List of T1-weighted structural images
    t1ce
        List of T1-weighted contrast-enhanced images
    t2w
        List of T2-weighted images
    flair
        List of FLAIR images

    Outputs
    -------
    t1w_preproc
        T1w reference (brain-extracted, bias-corrected, averaged if multiple)
    t1w_mask
        Brain mask estimated from T1w
    t1w_defaced
        Defaced T1w image (only if defacing=True)
    t1ce_preproc
        T1ce image aligned to T1w space
    t1ce_defaced
        Defaced T1ce image (only if defacing=True)
    t2w_preproc
        T2w image aligned to T1w space
    t2w_defaced
        Defaced T2w image (only if defacing=True)
    flair_preproc
        FLAIR image aligned to T1w space
    flair_defaced
        Defaced FLAIR image (only if defacing=True)
    anat2std_xfm
        Nonlinear spatial transforms to standard template space
    std2anat_xfm
        Reverse transforms from standard to anatomical space
    template
        Name of template used for registration
    """
    if output_spaces is None or not isinstance(output_spaces, SpatialReferences):
        output_spaces = SpatialReferences(['MNI152NLin2009cAsym'])

    t1ce = t1ce or []
    t2w = t2w or []
    flair = flair or []

    workflow = Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['t1w', 't1ce', 't2w', 'flair', 'subjects_dir', 'subject_id', 'tumor_dseg']
        ),
        name='inputnode',
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'template',
                'subjects_dir',
                'subject_id',
                #'t1w_defaced',
                't1w_preproc',
                't1w_brain',
                't1w_mask',
                't1w_dseg',
                't1w_tpms',
                'anat2std_xfm',
                'std2anat_xfm',
                #'t1ce_defaced',
                't1ce_preproc',
                #'t1ce_tpms',
                #'t2w_defaced',
                't2w_preproc',
                #'t2w_tpms',
                #'flair_defaced',
                'flair_preproc',
                #'flair_tpms',
                'tumor_dseg',
            ]
        ),
        name='outputnode',
    )
    anat_fit_wf = init_anat_fit_wf(
        bids_dir=bids_dir,
        output_dir=output_dir,
        longitudinal=longitudinal,
        skull_strip_mode=skull_strip_mode,
        skull_strip_template=skull_strip_template,
        output_spaces=output_spaces,
        skull_strip_backend=skull_strip_backend,
        registration_backend=registration_backend,
        t1w=t1w,
        t1ce=t1ce,
        t2w=t2w,
        flair=flair,
        precomputed=precomputed,
        debug=debug,
        sloppy=sloppy,
        omp_nthreads=omp_nthreads,
        skull_strip_fixed_seed=skull_strip_fixed_seed,
        skip_registration=skip_registration,
    )

    workflow.connect([
        (inputnode, anat_fit_wf, [
            ('t1w', 'inputnode.t1w'),
            ('t1ce', 'inputnode.t1ce'),
            ('t2w', 'inputnode.t2w'),
            ('flair', 'inputnode.flair'),
            ('subjects_dir', 'inputnode.subjects_dir'),
            ('subject_id', 'inputnode.subject_id'),
        ]),
        (anat_fit_wf, outputnode, [
            ('outputnode.template', 'template'),
            ('outputnode.subjects_dir', 'subjects_dir'),
            ('outputnode.subject_id', 'subject_id'),
            ('outputnode.t1w_brain', 't1w_brain'),
            ('outputnode.t1w_defaced', 't1w_defaced'),
            ('outputnode.t1w_preproc', 't1w_preproc'),
            ('outputnode.t1w_mask', 't1w_mask'),
            ('outputnode.t1w_dseg', 't1w_dseg'),
            ('outputnode.t1w_tpms', 't1w_tpms'),
            ('outputnode.anat2std_xfm', 'anat2std_xfm'),
            ('outputnode.std2anat_xfm', 'std2anat_xfm'),
            ('outputnode.t1ce_defaced', 't1ce_defaced'),
            ('outputnode.t1ce_preproc', 't1ce_preproc'),
            ('outputnode.t2w_defaced', 't2w_defaced'),
            ('outputnode.t2w_preproc', 't2w_preproc'),
            ('outputnode.flair_defaced', 'flair_defaced'),
            ('outputnode.flair_preproc', 'flair_preproc'),
        ]),
    ])

    # Template-space volume outputs (only when registration runs inside this workflow)
    if not skip_registration:
        template_iterator_wf = init_template_iterator_wf(spaces=output_spaces, sloppy=sloppy)
        ds_std_volumes_wf = init_ds_anat_volumes_wf(
            bids_dir=bids_dir,
            output_dir=output_dir,
        )
        workflow.connect([
            (anat_fit_wf, template_iterator_wf, [
                ('outputnode.template', 'inputnode.template'),
                ('outputnode.anat2std_xfm', 'inputnode.anat2std_xfm'),
            ]),
            (anat_fit_wf, ds_std_volumes_wf, [
                ('outputnode.t1w_valid_list', 'inputnode.source_files'),
                ('outputnode.t1w_preproc', 'inputnode.anat_preproc'),
                ('outputnode.t1w_mask', 'inputnode.anat_mask'),
                ('outputnode.t1w_dseg', 'inputnode.anat_dseg'),
                ('outputnode.t1w_tpms', 'inputnode.anat_tpms'),
                ('outputnode.t1ce_preproc', 'inputnode.t1ce_preproc'),
                ('outputnode.t2w_preproc', 'inputnode.t2w_preproc'),
                ('outputnode.flair_preproc', 'inputnode.flair_preproc'),
            ]),
            (template_iterator_wf, ds_std_volumes_wf, [
                ('outputnode.std_t1w', 'inputnode.ref_file'),
                ('outputnode.anat2std_xfm', 'inputnode.anat2std_xfm'),
                ('outputnode.space', 'inputnode.space'),
                ('outputnode.cohort', 'inputnode.cohort'),
                ('outputnode.resolution', 'inputnode.resolution'),
            ]),
        ])
    workflow.__desc__ = anat_fit_wf.__desc__
    return workflow
    # TODO: add surface pipeline when tested

@tag('anat.fit')
def init_anat_fit_wf(
    *,
    bids_dir: Union[Path, str],
    output_dir: Union[Path, str],
    t1w: List,
    t1ce: Optional[List] = None,
    t2w: Optional[List] = None,
    flair: Optional[List] = None,
    longitudinal: bool = False,
    skull_strip_template: Union[str, Reference] = 'OASIS30ANTs',
    output_spaces: Optional[SpatialReferences] = None,
    omp_nthreads: int = 1,
    precomputed: dict = {},
    debug: bool = False,
    name='anat_fit_wf',
    skull_strip_mode: str = 'auto',
    skull_strip_fixed_seed: bool = False,
    skull_strip_backend: str = 'ants',
    registration_backend: str = 'ants',
    sloppy: bool = False,
    skip_registration: bool = False,
):
    """
    Stage the anatomical preprocessing steps of *OncoPrep*.
    Based on *sMRIprep*'s anatomical workflow, adapted for BraTS data.

    This includes:

      - T1w reference: realigning and then averaging T1w images.
      - Brain extraction and INU (bias field) correction.
      - Brain tissue segmentation.
      - Spatial normalization to standard spaces.
      - Surface reconstruction with FreeSurfer_ (NOT IMPLEMENTED).

    .. include:: ../links.rst

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from niworkflows.utils.spaces import SpatialReferences, Reference
            from smriprep.workflows.anatomical import init_anat_fit_wf
            wf = init_anat_fit_wf(
                bids_dir='.',
                output_dir='.',
                longitudinal=False,
                t1w=['t1w.nii.gz'],
                t1ce=['t1ce.nii.gz'],
                t2w=['t2w.nii.gz'],
                flair=['flair.nii.gz'],
                skull_strip_template=Reference('OASIS30ANTs'),
                spaces=SpatialReferences(spaces=['MNI152NLin2009cAsym', 'fsaverage5']),
                precomputed={},
                debug=False,
                sloppy=False,
                omp_nthreads=1,
                skull_strip_fixed_seed=False,
            )


    Parameters
    ----------
    bids_dir : :obj:`str`
        Path of the input BIDS dataset root
    output_dir : :obj:`str`
        Directory in which to save derivatives
    longitudinal : :obj:`bool`
        Create unbiased structural template, regardless of number of inputs
        (may increase runtime)
    t1w : :obj:`list`
        List of T1-weighted structural images.
    t1ce : :obj:`list`, optional
        List of T1-weighted contrast-enhanced images.
    t2w : :obj:`list`, optional
        List of T2-weighted images.
    flair : :obj:`list`, optional
        List of FLAIR images.
    skull_strip_mode : :obj:`str`
        Determiner for T1-weighted skull stripping (`force` ensures skull stripping,
        `skip` ignores skull stripping, and `auto` automatically ignores skull stripping
        if pre-stripped brains are detected).
    skull_strip_template : :py:class:`~niworkflows.utils.spaces.Reference`
        Spatial reference to use in atlas-based brain extraction.
    output_spaces : :py:class:`~niworkflows.utils.spaces.SpatialReferences`
        Object containing standard and nonstandard space specifications.
    precomputed : :obj:`dict`
        Dictionary mapping output specification attribute names and
        paths to precomputed derivatives.
    omp_nthreads : :obj:`int`
        Maximum number of threads an individual process may use
    debug : :obj:`bool`
        Enable debugging outputs
    sloppy: :obj:`bool`
        Quick, impercise operations. Used to decrease workflow duration.
    name : :obj:`str`, optional
        Workflow name (default: anat_fit_wf)
    skull_strip_fixed_seed : :obj:`bool`
        Do not use a random seed for skull-stripping - will ensure
        run-to-run replicability when used with --omp-nthreads 1
        (default: ``False``).

    Inputs
    ------
    t1w
        List of T1-weighted structural images
    t2w
        List of T2-weighted structural images
    t1ce
        List of T1-weighted contrast-enhanced images
    roi
        A mask to exclude regions during standardization
    flair
        List of FLAIR images
    subjects_dir
        FreeSurfer SUBJECTS_DIR
    subject_id
        FreeSurfer subject ID

    Outputs
    -------
    t1w_preproc
        The T1w reference map, which is calculated as the average of bias-corrected
        and preprocessed T1w images, defining the anatomical space.
    t1w_mask
        Brain (binary) mask estimated by brain extraction.
    t1w_dseg
        Brain tissue segmentation of the preprocessed structural image, including
        gray-matter (GM), white-matter (WM) and cerebrospinal fluid (CSF).
    t1w_tpms
        List of tissue probability maps corresponding to ``t1w_dseg``.
    t1w_valid_list
        List of input T1w images accepted for preprocessing. If t1w_preproc is
        precomputed, this is always a list containing that image.
    template
        List of template names to which the structural image has been registered
    anat2std_xfm
        List of nonlinear spatial transforms to resample data from subject
        anatomical space into standard template spaces. Collated with template.
    std2anat_xfm
        List of nonlinear spatial transforms to resample data from standard
        template spaces into subject anatomical space. Collated with template.
    subjects_dir
        FreeSurfer SUBJECTS_DIR; use as input to a node to ensure that it is run after
        FreeSurfer reconstruction is completed.
    subject_id
        FreeSurfer subject ID; use as input to a node to ensure that it is run after
        FreeSurfer reconstruction is completed.
    fsnative2t1w_xfm
        ITK-style affine matrix translating from FreeSurfer-conformed subject space to T1w

    See Also
    --------
    * :py:func:`~niworkflows.anat.ants.init_brain_extraction_wf`
    * :py:func:`~smriprep.workflows.surfaces.init_surface_recon_wf`

    """
    workflow = Workflow(name=name)
    num_t1w = len(t1w)

    # Convert skull_strip_template to Reference if it's a string
    if isinstance(skull_strip_template, str):
        skull_strip_template = Reference(skull_strip_template)

    desc = """
Anatomical data preprocessing

: A total of {num_t1w} T1-weighted (T1w) images were found within the input
BIDS dataset.
""".format(num_t1w=num_t1w)

    if t1ce:
        desc += f"Additionally, {len(t1ce)} T1-weighted contrast-enhanced (T1ce) images "
        desc += f"{len(t2w)} T2-weighted (T2w) images, and {len(flair)} FLAIR images were available.\n"

    if t2w:
        desc += f"Additionally, {len(t2w)} T2-weighted (T2w) images "
    
    if flair:
        desc += f"and {len(flair)} FLAIR images were available.\n"
    
    have_t1w = 't1w_preproc' in precomputed
    have_t2w = 't2w_preproc' in precomputed
    have_t1ce = 't1ce_preproc' in precomputed
    have_flair = 'flair_preproc' in precomputed
    have_mask = 't1w_mask' in precomputed
    have_dseg = 't1w_dseg' in precomputed
    have_tpms = 't1w_tpms' in precomputed

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['t1w', 't1ce', 't2w', 'flair', 'subjects_dir', 'subject_id']
        ),
        name='inputnode',
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                # Primary derivatives
                't1w_defaced',
                't1w_preproc',
                't1w_brain',
                't1ce_defaced',
                't1ce_preproc',
                't2w_defaced',
                't2w_preproc',
                'flair_defaced',
                'flair_preproc',
                't1w_mask',
                't1w_dseg',
                't1w_tpms',
                't1w_valid_list',
                'anat2std_xfm',
                'std2anat_xfm',
                # Metadata
                'template',
                'subjects_dir',
                'subject_id',
                't1w_valid_list',
                'bids_dir',
            ]
        ),
        name='outputnode',
    )
    # If all derivatives exist, inputnode could go unconnected, so add explicitly
    workflow.add_nodes([inputnode])

    # Stage 1 inputs (filtered)
    sourcefile_buffer = pe.Node(
        niu.IdentityInterface(fields=['source_files']),
        name='sourcefile_buffer',
    )

    # Stage 1.5 results (defacing if requested)
    deface_buffer = pe.Node(
        niu.IdentityInterface(
            fields=[
                't1w_defaced',
                't1ce_defaced',
                't2w_defaced',
                'flair_defaced',
            ]
        ),
        name='deface_buffer',
    )

    # Stage 2 results
    t1w_buffer = pe.Node(
        niu.IdentityInterface(fields=['t1w_preproc', 't1w_mask', 't1w_brain', 'ants_seg']),
        name='t1w_buffer',
    )
    t1ce_buffer = pe.Node(
        niu.IdentityInterface(fields=['t1ce_preproc', 't1ce_brain']),
        name='t1ce_buffer',
    )
    t2w_buffer = pe.Node(
        niu.IdentityInterface(fields=['t2w_preproc', 't2w_brain']),
        name='t2w_buffer',
    )
    flair_buffer = pe.Node(
        niu.IdentityInterface(fields=['flair_preproc', 'flair_brain']),
        name='flair_buffer',
    )

    # Stage 3 results
    seg_buffer = pe.Node(
        niu.IdentityInterface(fields=['t1w_dseg', 't1w_tpms']),
        name='seg_buffer',
    )
    # Stage 4 results: collated template names, forward and reverse transforms
    template_buffer = pe.Node(niu.Merge(2), name='template_buffer')
    anat2std_buffer = pe.Node(niu.Merge(2), name='anat2std_buffer')
    std2anat_buffer = pe.Node(niu.Merge(2), name='std2anat_buffer')

    # Stage 6 results: Refined stage 2 results; may be direct copy if no refinement
    refined_buffer = pe.Node(
        niu.IdentityInterface(fields=['t1w_mask', 't1w_brain']),
        name='refined_buffer',
    )

    # Reporting
    # When registration is deferred (skip_registration=True), pass empty
    # spaces so the reports workflow does not try to iterate over templates
    # and select transforms that don't exist yet.
    _report_spaces = [] if skip_registration else output_spaces
    anat_reports_wf = init_anat_reports_wf(
        spaces=_report_spaces,
        output_dir=output_dir,
        sloppy=sloppy,
        freesurfer=False,
    )
    
    # Extract first source file from list for reports
    def _get_first_file(files):
        """Get the first file from a list."""
        if isinstance(files, list):
            return files[0] if files else None
        return files
    
    source_file_select = pe.Node(
        niu.Function(
            function=_get_first_file,
            input_names=['files'],
            output_names=['out_file'],
        ),
        name='source_file_select',
        run_without_submitting=True,
    )
    
    workflow.connect([
        (seg_buffer, outputnode, [
            ('t1w_dseg', 't1w_dseg'),
            ('t1w_tpms', 't1w_tpms'),
        ]),
        (anat2std_buffer, outputnode, [
            ('out', 'anat2std_xfm'),
        ]),
        (std2anat_buffer, outputnode, [
            ('out', 'std2anat_xfm'),
        ]),
        (template_buffer, outputnode, [
            ('out', 'template'),
        ]),
        (sourcefile_buffer, outputnode, [
            ('source_files', 't1w_valid_list'),
        ]),
        (deface_buffer, outputnode, [
            ('t1w_defaced', 't1w_defaced'),
            ('t1ce_defaced', 't1ce_defaced'),
            ('t2w_defaced', 't2w_defaced'),
            ('flair_defaced', 'flair_defaced'),
        ]),
        (source_file_select, anat_reports_wf, [
            ('out_file', 'inputnode.source_file'),
        ]),
        (outputnode, anat_reports_wf, [
            ('t1w_preproc', 'inputnode.t1w_preproc'),
            ('t1w_mask', 'inputnode.t1w_mask'),
            ('t1w_dseg', 'inputnode.t1w_dseg'),
            ('t1ce_preproc', 'inputnode.t1ce_preproc'),
            ('t2w_preproc', 'inputnode.t2w_preproc'),
            ('flair_preproc', 'inputnode.flair_preproc'),
            ('template', 'inputnode.template'),
            ('anat2std_xfm', 'inputnode.anat2std_xfm'),
            ('subjects_dir', 'inputnode.subjects_dir'),
            ('subject_id', 'inputnode.subject_id'),
        ]),
    ])

    # Stage 1: Conform and validate T1w images
    # =========================================
    anat_validate = pe.Node(ValidateImage(), name='anat_validate', run_without_submitting=True)
    if not have_t1w:
        LOGGER.info('ANAT Stage 1: T1w conformance and averaging')
        ants_ver = ANTsInfo.version()
        LOGGER.info(f'Detected ANTs version: {ants_ver}')
        desc += f"""\
 {'Each' if num_t1w > 1 else 'The'} T1-weighted (T1w) image was corrected for intensity 
non-uniformity (INU) with `N4BiasFieldCorrection` [@n4], distributed with ANTs {ants_ver}
[@ants, RRID:SCR_004757]"""
        desc += ".\n" if num_t1w > 1 else ', and used as T1w-reference throughtout the workflow.\n'

        anat_template_wf = init_anat_template_wf(
            longitudinal=longitudinal,
            omp_nthreads=omp_nthreads,
            num_files=num_t1w,
            image_type='T1w',
            name='anat_template_wf',
        )
        ds_template_wf = init_ds_template_wf(
            output_dir=output_dir, num_anat=num_t1w, image_type='T1w'
        )

        workflow.connect([
            (inputnode, anat_template_wf, [('t1w', 'inputnode.anat_files')]),
            (anat_template_wf, anat_validate, [('outputnode.anat_ref', 'in_file')]),
            (anat_template_wf, sourcefile_buffer, [
                ('outputnode.anat_valid_list', 'source_files'),
            ]),
            (anat_template_wf, anat_reports_wf, [
                ('outputnode.out_report', 'inputnode.t1w_conform_report'),
            ]),
            (anat_template_wf, ds_template_wf, [
                ('outputnode.anat_realign_xfm', 'inputnode.anat_ref_xfms'),
            ]),
            (sourcefile_buffer, source_file_select, [('source_files', 'files')]),
            (source_file_select, ds_template_wf, [('out_file', 'inputnode.source_file')]),
            (t1w_buffer, ds_template_wf, [('t1w_preproc', 'inputnode.anat_preproc')]),
            (ds_template_wf, outputnode, [('outputnode.anat_preproc', 't1w_preproc')]),
        ])
    else:
        LOGGER.info('ANAT Found preprocessed T1w - skipping stage 1')
        desc += """\ A preprocessed T1-weighted (T1w) image was provided as a precomputed
        input and used as T1w-reference throughout the workflow.
        \n"""

        anat_validate.inputs.in_file = precomputed['t1w_preproc']
        sourcefile_buffer.inputs.source_files = [precomputed['t1w_preproc']]

        workflow.connect([
            (anat_validate, t1w_buffer, [('out_file', 't1w_preproc')]),
            (t1w_buffer, outputnode, [('t1w_preproc', 't1w_preproc')]),
        ])

    # Stage 2: INU correction and brain extraction
    # =======================================
    # We always need to generate t1w_brain; how to that depends on whether we have
    # a pre-corrected T1w or precomputed mask, or are given an already masked image
    if not have_mask:
        LOGGER.info('ANAT Stage 2: Preparing brain extraction workflow')
        if skull_strip_mode == 'auto':
            run_skull_strip = not all(_is_skull_stripped(img) for img in t1w)
        else:
            run_skull_strip = {'force': True, 'skip': False}[skull_strip_mode]

        if run_skull_strip:
            # brain extraction - select backend
            if skull_strip_backend == 'hdbet':
                desc += """\
    The T1w-reference was skull-stripped using HD-BET (High-resolution Brain Extraction Tool)
    [@hdbet], a deep learning-based brain extraction method.
    """
                brain_extraction_wf = init_hdbet_wf(
                    omp_nthreads=omp_nthreads,
                    name='brain_extraction_wf',
                )
            elif skull_strip_backend == 'synthstrip':
                desc += """\
    The T1w-reference was skull-stripped using SynthStrip [@synthstrip], 
    a robust, learning-based brain extraction tool from FreeSurfer.
    """
                brain_extraction_wf = init_synthstrip_wf(
                    omp_nthreads=omp_nthreads,
                    name='brain_extraction_wf',
                )
            else:  # default: ants
                desc += f"""\
    The T1w-reference was skull-stripped with a *Nipype* implementation of
    the `antsBrainExtraction.sh` workflow (from ANTs), using {skull_strip_template.fullname}
    as the target template.
    """
                brain_extraction_wf = init_brain_extraction_wf(
                    in_template=skull_strip_template.space,
                    template_spec=skull_strip_template.spec,
                    omp_nthreads=omp_nthreads,
                    atropos_use_random_seed=not skull_strip_fixed_seed,
                    normalization_quality='precise' if not sloppy else 'testing',
                )
            workflow.connect([
                (anat_validate, brain_extraction_wf, [('out_file', 'inputnode.in_files')]),
                (brain_extraction_wf, t1w_buffer, [
                    ('outputnode.out_mask', 't1w_mask'),
                    (('outputnode.out_file', _pop), 't1w_brain'),
                    ('outputnode.out_segm', 'ants_seg'),
                ]),
            ])
            if not have_t1w:
                workflow.connect([
                    (brain_extraction_wf, t1w_buffer, [
                        (('outputnode.bias_corrected', _pop), 't1w_preproc'),
                    ]),
                ])

        elif not have_t1w:
            LOGGER.info('ANAT Stage 2: Skipping brain extraction, INU-correction only for T1w')
            desc += """\
    The provided T1w image was previously skull-stripped; a brain mask was derived from the input image.
    """
            n4_only_wf = init_n4_only_wf(
                omp_nthreads=omp_nthreads,
                atropos_use_random_seed=not skull_strip_fixed_seed,
            )
            workflow.connect([
                (anat_validate, n4_only_wf, [('out_file', 'inputnode.in_files')]),
                (n4_only_wf, t1w_buffer, [
                    (('outputnode.bias_corrected', _pop), 't1w_preproc'),
                    ('outputnode.out_mask', 't1w_mask'),
                    (('outputnode.out_file', _pop), 't1w_brain'),
                    ('outputnode.out_segm', 'ants_seg'),
                ]),
            ])
        elif not have_t1ce:
            LOGGER.info('ANAT Stage 2: Skipping brain extraction, INU-correction only for T1ce')
            n4_only_t1ce_wf = init_n4_only_wf(
                omp_nthreads=omp_nthreads,
                atropos_use_random_seed=not skull_strip_fixed_seed,
                name='n4_only_t1ce_wf',
            )
            workflow.connect([
                (anat_validate, n4_only_t1ce_wf, [('out_file', 'inputnode.in_files')]),
                (n4_only_t1ce_wf, t1ce_buffer, [
                    (('outputnode.bias_corrected', _pop), 't1ce_preproc'),
                    ('outputnode.out_mask', 't1ce_mask'),
                    (('outputnode.out_file', _pop), 't1ce_brain'),
                    ('outputnode.out_segm', 'ants_seg'),
                ]),
            ])
        elif not have_t2w:
            LOGGER.info('ANAT Stage 2: Skipping brain extraction, INU-correction only for T2w')
            n4_only_t2w_wf = init_n4_only_wf(
                omp_nthreads=omp_nthreads,
                atropos_use_random_seed=not skull_strip_fixed_seed,
                name='n4_only_t2w_wf',
            )
            workflow.connect([
                (anat_validate, n4_only_t2w_wf, [('out_file', 'inputnode.in_files')]),
                (n4_only_t2w_wf, t2w_buffer, [
                    (('outputnode.bias_corrected', _pop), 't2w_preproc'),
                    ('outputnode.out_mask', 't2w_mask'),
                    (('outputnode.out_file', _pop), 't2w_brain'),
                    ('outputnode.out_segm', 'ants_seg'),
                ]),
            ])
        elif not have_flair:
            LOGGER.info('ANAT Stage 2: Skipping brain extraction, INU-correction only for FLAIR')
            n4_only_flair_wf = init_n4_only_wf(
                omp_nthreads=omp_nthreads,
                atropos_use_random_seed=not skull_strip_fixed_seed,
                name='n4_only_flair_wf',
            )
            workflow.connect([
                (anat_validate, n4_only_flair_wf, [('out_file', 'inputnode.in_files')]),
                (n4_only_flair_wf, flair_buffer, [
                    (('outputnode.bias_corrected', _pop), 'flair_preproc'),
                    ('outputnode.out_mask', 'flair_mask'),
                    (('outputnode.out_file', _pop), 'flair_brain'),
                    ('outputnode.out_segm', 'ants_seg'),
                ]),
            ])
        # Binarize the already uniformized image
        else:
            LOGGER.info('ANAT Stage 2: Skipping brain extraction, generating mask from input')
            desc += """\
    The provided T1w image was previously skull-stripped; a brain mask was derived from the input image.
    """
            binarize = pe.Node(Binarize(thresh_low=2), name='binarize')
            workflow.connect([
                (anat_validate, binarize, [('out_file', 'in_file')]),
                (anat_validate, t1w_buffer, [('out_file', 't1w_brain')]),
                (binarize, t1w_buffer, [('out_file', 't1w_mask')]),
            ])
    
        ds_t1w_mask_wf = init_ds_mask_wf(
            bids_dir=bids_dir,
            output_dir=output_dir,
            mask_type='brain',
            name='ds_t1w_mask_wf',
        )
        workflow.connect([
            (sourcefile_buffer, ds_t1w_mask_wf, [('source_files', 'inputnode.source_files')]),
            (refined_buffer, ds_t1w_mask_wf, [('t1w_mask', 'inputnode.mask_file')]),
            (ds_t1w_mask_wf, outputnode, [('outputnode.mask_file', 't1w_mask')]),
        ])
    else:
        LOGGER.info('ANAT Found precomputed T1w mask - skipping brain extraction')
        desc += """\ A precomputed brain mask was provided and used to derive the brain-extracted
        T1w image throughout the workflow.
        \n"""

        t1w_buffer.inputs.t1w_mask = precomputed['t1w_mask']
        # if we have a mask, always apply it
        apply_mask = pe.Node(ApplyMask(in_mask=precomputed['t1w_mask']), name='apply_mask')
        workflow.connect([
            (anat_validate, apply_mask, [('out_file', 'in_file')]),
        ])
        # run N4 if it hasn't been pre-run
        if not have_t1w:
            LOGGER.info('ANAT skipping brain extraction, INU-correction only')
            n4_only_wf = init_n4_only_wf(
                omp_nthreads=omp_nthreads,
                atropos_use_random_seed=not skull_strip_fixed_seed,
            )
            workflow.connect([
                (apply_mask, n4_only_wf, [('out_file', 'inputnode.in_files')]),
                (n4_only_wf, t1w_buffer, [
                    (('outputnode.bias_corrected', _pop), 't1w_preproc'),
                    (('outputnode.out_file', _pop), 't1w_brain'),
                ]),
            ])
        else:
            LOGGER.info('ANAT Skipping Stage 2')
            workflow.connect([(apply_mask, t1w_buffer, [('out_file', 't1w_brain')])])
        
        workflow.connect([refined_buffer, outputnode, [('t1w_mask', 't1w_mask')]])

    # Connect t1w_buffer results to refined_buffer for downstream processing (both branches)
    workflow.connect([
        (t1w_buffer, refined_buffer, [
            ('t1w_brain', 't1w_brain'),
            ('t1w_mask', 't1w_mask'),
        ]),
    ])

    # Stage 3: Tissue segmentation
    # ============================
    if not (have_dseg and have_tpms):
        LOGGER.info('ANAT Stage 3: Preparing tissue segmentation workflow')
        fsl_ver = FAST().version or '(version unknown)'
        desc += f"""\
Brain tissue segmentation of cerebrospinal fluid (CSF),
white-matter (WM) and gray-matter (GM) was performed on the
brain-extracted T1w using `fast` [FSL {fsl_ver}; RRID:SCR_002823, @fsl_fast].
"""
        fast = pe.Node(
            FAST(segments=True, no_bias=True, probability_maps=True, bias_iters=0),
            name='fast',
            mem_gb=3,
        )
        lut_t1w_dseg = pe.Node(niu.Function(function=_apply_bids_lut), name='lut_t1w_dseg')
        lut_t1w_dseg.inputs.lut = (0, 3, 1, 2) # maps: 0 -> 0, 3 -> 1, 1 -> 2, 2 -> 3.
        fast2bids = pe.Node(
            niu.Function(function=_probseg_fast2bids),
            name='fast2bids',
            run_without_submitting=True,
        )
        workflow.connect([
            (refined_buffer, fast, [('t1w_brain', 'in_files')])
        ])

        if not have_dseg:
            ds_dseg_wf = init_ds_dseg_wf(output_dir=output_dir)
            workflow.connect([
                (fast, lut_t1w_dseg, [('partial_volume_map', 'in_dseg')]),
                (sourcefile_buffer, ds_dseg_wf, [('source_files', 'inputnode.source_files')]),
                (lut_t1w_dseg, ds_dseg_wf, [('out', 'inputnode.anat_dseg')]),
                (ds_dseg_wf, seg_buffer, [('outputnode.anat_dseg', 't1w_dseg')]),
            ])
        if not have_tpms:
            ds_tpms_wf = init_ds_tpms_wf(output_dir=output_dir)
            workflow.connect([
                (fast, fast2bids, [('partial_volume_files', 'inlist')]),
                (sourcefile_buffer, ds_tpms_wf, [('source_files', 'inputnode.source_files')]),
                (fast2bids, ds_tpms_wf, [('out', 'inputnode.anat_tpms')]),
                (ds_tpms_wf, seg_buffer, [('outputnode.anat_tpms', 't1w_tpms')]),
            ])
    else:
        LOGGER.info('ANAT Skipping Stage 3')
    if have_dseg:
        LOGGER.info('ANAT Found discrete tissue segmentation')
        desc += "Precomputed discrete tissue segmentations were provided as inputs.\n"
        seg_buffer.inputs.t1w_dseg = precomputed['t1w_dseg']
    if have_tpms:
        LOGGER.info('ANAT Found tissue probability maps')
        desc += "Precomputed tissue probability maps were provided as inputs.\n"
        seg_buffer.inputs.t1w_tpms = precomputed['t1w_tpms']

    # Stage 4a: Multi-modal co-registration and preprocessing
    # ========================================================
    # Additional modalities (T1ce, T2w, FLAIR) are processed as follows:
    # 1. Register to raw T1w reference (anat_validate output)
    # 2. Apply T1w brain mask to skull-strip the registered modality  
    # 3. Run N4 INU correction on the skull-stripped image
    # 4. Standard-space warping is handled in outputs.py using anat2std_xfm
    
    coreg_buffer = pe.Node(
        niu.IdentityInterface(fields=['t1ce_preproc', 't2w_preproc', 'flair_preproc']),
        name='coreg_buffer',
    )

    if t1ce or t2w or flair:
        LOGGER.info('ANAT Stage 4: Multi-modal co-registration and preprocessing')
        desc += "\n\nAdditional modalities were rigidly registered to the T1w reference, "
        desc += "skull-stripped using the T1w brain mask, and bias-field corrected with N4.\n"

    if t1ce:
        # Step 1: Register T1ce to raw T1w
        coreg_t1ce = pe.Node(
            niu.Function(
                function=_register_modality,
                input_names=['moving', 'fixed', 'fixed_mask'],
                output_names=['registered'],
            ),
            name='coreg_t1ce',
        )
        # Step 2: Apply T1w brain mask to skull-strip
        mask_t1ce = pe.Node(ApplyMask(), name='mask_t1ce')
        # Step 3: N4 bias field correction
        n4_t1ce = pe.Node(
            ants.N4BiasFieldCorrection(
                dimension=3,
                copy_header=True,
                n_iterations=[50, 50, 30, 20],
                convergence_threshold=1e-6,
                shrink_factor=3,
                num_threads=omp_nthreads,
            ),
            name='n4_t1ce',
        )

        workflow.connect([
            # Register to raw T1w (anat_validate output)
            (inputnode, coreg_t1ce, [('t1ce', 'moving')]),
            (anat_validate, coreg_t1ce, [('out_file', 'fixed')]),
            (refined_buffer, coreg_t1ce, [('t1w_mask', 'fixed_mask')]),
            # Apply brain mask
            (coreg_t1ce, mask_t1ce, [('registered', 'in_file')]),
            (refined_buffer, mask_t1ce, [('t1w_mask', 'in_mask')]),
            # N4 correction
            (mask_t1ce, n4_t1ce, [('out_file', 'input_image')]),
            (n4_t1ce, coreg_buffer, [('output_image', 't1ce_preproc')]),
        ])
    else:
        coreg_buffer.inputs.t1ce_preproc = None

    if t2w:
        # Step 1: Register T2w to raw T1w
        coreg_t2w = pe.Node(
            niu.Function(
                function=_register_modality,
                input_names=['moving', 'fixed', 'fixed_mask'],
                output_names=['registered'],
            ),
            name='coreg_t2w',
        )
        # Step 2: Apply T1w brain mask to skull-strip
        mask_t2w = pe.Node(ApplyMask(), name='mask_t2w')
        # Step 3: N4 bias field correction
        n4_t2w = pe.Node(
            ants.N4BiasFieldCorrection(
                dimension=3,
                copy_header=True,
                n_iterations=[50, 50, 30, 20],
                convergence_threshold=1e-6,
                shrink_factor=3,
                num_threads=omp_nthreads,
            ),
            name='n4_t2w',
        )

        workflow.connect([
            # Register to raw T1w (anat_validate output)
            (inputnode, coreg_t2w, [('t2w', 'moving')]),
            (anat_validate, coreg_t2w, [('out_file', 'fixed')]),
            (refined_buffer, coreg_t2w, [('t1w_mask', 'fixed_mask')]),
            # Apply brain mask
            (coreg_t2w, mask_t2w, [('registered', 'in_file')]),
            (refined_buffer, mask_t2w, [('t1w_mask', 'in_mask')]),
            # N4 correction
            (mask_t2w, n4_t2w, [('out_file', 'input_image')]),
            (n4_t2w, coreg_buffer, [('output_image', 't2w_preproc')]),
        ])
    else:
        coreg_buffer.inputs.t2w_preproc = None

    if flair:
        # Step 1: Register FLAIR to raw T1w
        coreg_flair = pe.Node(
            niu.Function(
                function=_register_modality,
                input_names=['moving', 'fixed', 'fixed_mask'],
                output_names=['registered'],
            ),
            name='coreg_flair',
        )
        # Step 2: Apply T1w brain mask to skull-strip
        mask_flair = pe.Node(ApplyMask(), name='mask_flair')
        # Step 3: N4 bias field correction
        n4_flair = pe.Node(
            ants.N4BiasFieldCorrection(
                dimension=3,
                copy_header=True,
                n_iterations=[50, 50, 30, 20],
                convergence_threshold=1e-6,
                shrink_factor=3,
                num_threads=omp_nthreads,
            ),
            name='n4_flair',
        )

        workflow.connect([
            # Register to raw T1w (anat_validate output)
            (inputnode, coreg_flair, [('flair', 'moving')]),
            (anat_validate, coreg_flair, [('out_file', 'fixed')]),
            (refined_buffer, coreg_flair, [('t1w_mask', 'fixed_mask')]),
            # Apply brain mask
            (coreg_flair, mask_flair, [('registered', 'in_file')]),
            (refined_buffer, mask_flair, [('t1w_mask', 'in_mask')]),
            # N4 correction
            (mask_flair, n4_flair, [('out_file', 'input_image')]),
            (n4_flair, coreg_buffer, [('output_image', 'flair_preproc')]),
        ])
    else:
        coreg_buffer.inputs.flair_preproc = None

    # Save native-space preprocessed modalities BEFORE template registration
    # ======================================================================
    ds_modalities_wf = init_ds_modalities_wf(
        bids_dir=str(bids_dir),
        output_dir=str(output_dir),
        name='ds_modalities_wf',
    )
    workflow.connect([
        (sourcefile_buffer, ds_modalities_wf, [
            (('source_files', _pop), 'inputnode.source_file'),
        ]),
        (refined_buffer, ds_modalities_wf, [
            ('t1w_brain', 'inputnode.t1w_brain'),
        ]),
        (coreg_buffer, ds_modalities_wf, [
            ('t1ce_preproc', 'inputnode.t1ce_preproc'),
            ('t2w_preproc', 'inputnode.t2w_preproc'),
            ('flair_preproc', 'inputnode.flair_preproc'),
        ]),
    ])

    # Create a barrier node to ensure native-space modalities are saved BEFORE registration
    # This node passes through the T1w brain but waits for ds_modalities_wf to complete
    def _wait_for_save(t1w_brain, t1w_native, t1ce_native, t2w_native, flair_native):
        """Pass through t1w_brain after saving completes (dependency barrier)."""
        return t1w_brain

    wait_for_native_save = pe.Node(
        niu.Function(
            function=_wait_for_save,
            input_names=['t1w_brain', 't1w_native', 't1ce_native', 't2w_native', 'flair_native'],
            output_names=['t1w_brain'],
        ),
        name='wait_for_native_save',
        run_without_submitting=True,
    )

    workflow.connect([
        (refined_buffer, wait_for_native_save, [('t1w_brain', 't1w_brain')]),
        (ds_modalities_wf, wait_for_native_save, [
            ('outputnode.t1w_native', 't1w_native'),
            ('outputnode.t1ce_native', 't1ce_native'),
            ('outputnode.t2w_native', 't2w_native'),
            ('outputnode.flair_native', 'flair_native'),
        ]),
    ])

    # Stage 4: Normalization
    templates = []
    found_xfms = {}
    # Convert list to SpatialReferences if needed
    if output_spaces is not None and not isinstance(output_spaces, SpatialReferences):
        output_spaces = SpatialReferences(output_spaces)
    
    for template in output_spaces.get_spaces(nonstandard=False, dim=(3,)):
        xfms = precomputed.get('transforms', {}).get(template, {})
        if set(xfms) != {'forward', 'reverse'}:
            templates.append(template)
        else:
            found_xfms[template] = xfms

    template_buffer.inputs.in1 = list(found_xfms)
    anat2std_buffer.inputs.in1 = [xfms['forward'] for xfms in found_xfms.values()]
    std2anat_buffer.inputs.in1 = [xfms['reverse'] for xfms in found_xfms.values()]

    # Stage 5: Spatial normalization (if needed)
    # ============================================
    if skip_registration:
        LOGGER.info(
            'ANAT Stage 5: Skipping template registration '
            '(deferred until after segmentation for cost-function masking)'
        )
        desc += (
            "\n\nTemplate registration was deferred until after tumor "
            "segmentation to enable cost-function exclusion masking "
            "(see deferred registration section below).\n"
        )
        template_buffer.inputs.in2 = []
        anat2std_buffer.inputs.in2 = []
        std2anat_buffer.inputs.in2 = []
    elif templates:
        LOGGER.info(f'ANAT Stage 5: Registering to template(s): {templates}')
        register_template_wf = init_multimodal_template_registration_wf(
            sloppy=sloppy,
            omp_nthreads=omp_nthreads,
            templates=templates,
            registration_backend=registration_backend,
            name='register_template_wf',
        )
        ds_template_registration_wf = init_ds_template_registration_wf(
            output_dir=output_dir,
            image_type='T1w',
        )
        
        # Use JoinNodes to collect outputs from iterable registration workflow
        # join_template = pe.JoinNode(
        #     niu.IdentityInterface(fields=['template']),
        #     joinsource='register_template_wf.inputnode',
        #     joinfield=['template'],
        #     name='join_template',
        # )
        # join_anat2std = pe.JoinNode(
        #     niu.IdentityInterface(fields=['anat2std_xfm']),
        #     joinsource='register_template_wf.inputnode',
        #     joinfield=['anat2std_xfm'],
        #     name='join_anat2std',
        # )
        # join_std2anat = pe.JoinNode(
        #     niu.IdentityInterface(fields=['std2anat_xfm']),
        #     joinsource='register_template_wf.inputnode',
        #     joinfield=['std2anat_xfm'],
        #     name='join_std2anat',
        # )
        
        workflow.connect([
            # Connect T1w brain to registration via barrier (ensures native save completes first)
            (wait_for_native_save, register_template_wf, [('t1w_brain', 'inputnode.t1w')]),
            # Connect other modalities for warping to standard space
            (coreg_buffer, register_template_wf, [
                ('t1ce_preproc', 'inputnode.t1ce'),
                ('t2w_preproc', 'inputnode.t2w'),
                ('flair_preproc', 'inputnode.flair'),
            ]),
            (refined_buffer, register_template_wf, [
                ('t1w_mask', 'inputnode.t1w_mask'),
            ]),
            (sourcefile_buffer, ds_template_registration_wf, [
                ('source_files', 'inputnode.source_files'),
            ]),
            (register_template_wf, ds_template_registration_wf, [
                ('outputnode.template', 'inputnode.template'),
                ('outputnode.anat2std_xfm', 'inputnode.anat2std_xfm'),
                ('outputnode.std2anat_xfm', 'inputnode.std2anat_xfm'),
            ]),
            (register_template_wf, template_buffer, [('outputnode.template', 'in2')]),
            (ds_template_registration_wf, std2anat_buffer, [('outputnode.std2anat_xfm', 'in2')]),
            (ds_template_registration_wf, anat2std_buffer, [('outputnode.anat2std_xfm', 'in2')]),
         ])
    if found_xfms:
        LOGGER.info(f'ANAT Stage 5: Found precomputed transforms for {list(found_xfms)}')

    # Connect additional modality outputs to outputnode
    workflow.connect([
        (coreg_buffer, outputnode, [
            ('t1ce_preproc', 't1ce_preproc'),
            ('t2w_preproc', 't2w_preproc'),
            ('flair_preproc', 'flair_preproc'),
        ]),
        (refined_buffer, outputnode, [
            ('t1w_brain', 't1w_brain'),
        ]),
    ])

    # Note: Native-space modalities are saved by ds_modalities_wf (above, lines ~1100-1117)
    # before template registration

    workflow.__desc__ = desc
    return workflow

    # # Stage 6: Optional defacing with mri_deface
    # # ============================================
    # if defacing:
    #     LOGGER.info('ANAT Stage 6: Defacing anatomical images with mri_deface')

    #     # Deface T1w
    #     deface_t1w = pe.Node(
    #         niu.Function(
    #             function=_deface_anatomical,
    #             input_names=['in_file'],
    #             output_names=['out_file'],
    #         ),
    #         name='deface_t1w',
    #     )

    #     desc += "\nAnatomical images were defaced using mri_deface to protect participant privacy."

    #     # Create a buffer node for defaced outputs
    #     deface_buffer = pe.Node(
    #         niu.IdentityInterface(fields=['t1w_defaced', 't1ce_defaced', 't2w_defaced', 'flair_defaced']),
    #         name='deface_buffer',
    #     )

    #     workflow.connect([(t1w_buffer, deface_t1w, [('t1w_preproc', 'in_file')])])
    #     workflow.connect([(deface_t1w, deface_buffer, [('out_file', 't1w_defaced')])])

    #     # Deface T1ce if present
    #     if t1ce:
    #         deface_t1ce = pe.Node(
    #             niu.Function(
    #                 function=_deface_anatomical,
    #                 input_names=['in_file'],
    #                 output_names=['out_file'],
    #             ),
    #             name='deface_t1ce',
    #         )
    #         workflow.connect([(coreg_buffer, deface_t1ce, [('t1ce_preproc', 'in_file')])])
    #         workflow.connect([(deface_t1ce, deface_buffer, [('out_file', 't1ce_defaced')])])
    #     else:
    #         deface_buffer.inputs.t1ce_defaced = None

    #     # Deface T2w if present
    #     if t2w:
    #         deface_t2w = pe.Node(
    #             niu.Function(
    #                 function=_deface_anatomical,
    #                 input_names=['in_file'],
    #                 output_names=['out_file'],
    #             ),
    #             name='deface_t2w',
    #         )
    #         workflow.connect([(coreg_buffer, deface_t2w, [('t2w_preproc', 'in_file')])])
    #         workflow.connect([(deface_t2w, deface_buffer, [('out_file', 't2w_defaced')])])
    #     else:
    #         deface_buffer.inputs.t2w_defaced = None

    #     # Deface FLAIR if present
    #     if flair:
    #         deface_flair = pe.Node(
    #             niu.Function(
    #                 function=_deface_anatomical,
    #                 input_names=['in_file'],
    #                 output_names=['out_file'],
    #             ),
    #             name='deface_flair',
    #         )
    #         workflow.connect([(coreg_buffer, deface_flair, [('flair_preproc', 'in_file')])])
    #         workflow.connect([(deface_flair, deface_buffer, [('out_file', 'flair_defaced')])])
    #     else:
    #         deface_buffer.inputs.flair_defaced = None
    # else:
    #     LOGGER.info('ANAT Stage 6: Skipping defacing (defacing=False)')
    #     deface_buffer = pe.Node(
    #         niu.IdentityInterface(fields=['t1w_defaced', 't1ce_defaced', 't2w_defaced', 'flair_defaced']),
    #         name='deface_buffer',
    #     )
    #     deface_buffer.inputs.t1w_defaced = None
    #     deface_buffer.inputs.t1ce_defaced = None
    #     deface_buffer.inputs.t2w_defaced = None
    #     deface_buffer.inputs.flair_defaced = None


def _average_images(image_list):
    """Average a list of NIfTI images."""
    if isinstance(image_list, str):
        return image_list

    imgs = [nb.load(img) for img in image_list]
    data = np.stack([img.get_fdata() for img in imgs], axis=-1)
    mean_data = np.mean(data, axis=-1)

    out_img = nb.Nifti1Image(mean_data, imgs[0].affine, imgs[0].header)
    out_path = Path.cwd() / 't1w_reference.nii.gz'
    out_img.to_filename(out_path)
    return str(out_path)


def _register_modality(moving, fixed, fixed_mask):
    """
    Register a single modality to T1w using ANTs rigid registration.

    Parameters
    ----------
    moving : str or list
        Path(s) to moving image(s)
    fixed : str
        Path to fixed reference image
    fixed_mask : str
        Path to fixed image mask

    Returns
    -------
    registered : str
        Path to registered image
    """
    from pathlib import Path
    from nipype.interfaces.ants import Registration
    from nipype.interfaces.base import Undefined

    # Handle list input (use first image if multiple)
    if isinstance(moving, list):
        moving = moving[0] if moving else None
    
    # Handle undefined or None input
    if moving is None or moving == Undefined or isinstance(moving, type(Undefined)):
        raise ValueError("No moving image provided for registration")
    
    # Ensure moving is a valid path string
    moving = str(moving)
    if not Path(moving).exists():
        raise FileNotFoundError(f"Moving image not found: {moving}")

    # ANTs rigid registration
    ants_rigid = Registration(
        dimension=3,
        transforms=['Rigid'],
        transform_parameters=[(0.1,)],
        metric=['Mattes'],
        metric_weight=[1.0],
        radius_or_number_of_bins=[32],
        sampling_strategy=['Regular'],
        sampling_percentage=[0.25],
        number_of_iterations=[[1000, 500, 250, 0]],
        convergence_window_size=[10],
        smoothing_sigmas=[[3, 2, 1, 0]],
        shrink_factors=[[8, 4, 2, 1]],
        use_histogram_matching=[True],
        num_threads=1,
        output_warped_image=True,  # Ensure warped image is output
        output_transform_prefix='modality_to_t1w_',
    )

    ants_rigid.inputs.moving_image = moving
    ants_rigid.inputs.fixed_image = fixed
    # Only set fixed_image_masks if mask is provided (mask the fixed image only)
    if fixed_mask is not None:
        ants_rigid.inputs.fixed_image_masks = fixed_mask

    result = ants_rigid.run()
    registered_img = result.outputs.warped_image
    
    if registered_img is None or registered_img == Undefined:
        raise RuntimeError("Registration failed: no output warped image produced")

    return str(registered_img)


def _deface_anatomical(in_file):
    """
    Remove facial features from anatomical image using mri_deface.

    mri_deface is a tool for privacy-protecting defacing of structural MRI images.
    It automatically detects and removes the facial region while preserving
    the entire brain.

    Parameters
    ----------
    in_file : str
        Path to input anatomical image

    Returns
    -------
    out_file : str
        Path to defaced anatomical image
    """
    from pathlib import Path
    import subprocess

    try:
        import mri_deface  # noqa: F401
        from mri_deface.deface import run as deface_run  # noqa: F401
    except ImportError:
        LOGGER.warning(
            "mri_deface not available. Install with: pip install mri-deface. "
            "Returning original image without defacing."
        )
        return in_file

    try:
        in_file_path = Path(in_file)
        out_file_path = Path.cwd() / f"{in_file_path.stem}_defaced.nii.gz"

        # Use mri_deface via command line
        # mri_deface takes input and output paths
        cmd = ['mri_deface', str(in_file_path), str(out_file_path)]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            LOGGER.warning(
                f"mri_deface failed with return code {result.returncode}. "
                f"stderr: {result.stderr}. Returning original image."
            )
            return in_file

        if out_file_path.exists():
            return str(out_file_path)
        else:
            LOGGER.warning(
                f"mri_deface output not found at {out_file_path}. "
                "Returning original image."
            )
            return in_file

    except Exception as e:
        LOGGER.warning(
            f"Error during defacing: {e}. Returning original image without defacing."
        )
        return in_file

def init_anat_template_wf(
    *,
    longitudinal: bool = False,
    omp_nthreads: int = 1,
    num_files: int,
    image_type: str = 'T1w',
    name: str = 'anat_template_wf',
):
    """
    Generate a canonically-oriented, structural average from all input images.
    
    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from smriprep.workflows.anatomical import init_anat_template_wf
            wf = init_anat_template_wf(
                longitudinal=False, omp_nthreads=1, num_files=1, image_type="T1w"
            )

    Parameters
    ----------
    longitudinal : :obj:`bool`
        Create unbiased structural average, regardless of number of inputs
        (may increase runtime)
    omp_nthreads : :obj:`int`
        Maximum number of threads an individual process may use
    num_files : :obj:`int`
        Number of images
    image_type : :obj:`str`
       MR image type (T1w, T2w, etc.)
    name : :obj:`str`, optional
        Workflow name (default: anat_template_wf)

    Inputs
    ------
    anat_files
        List of structural images

    Outputs
    -------
    anat_ref
        Structural reference averaging input images
    anat_valid_list
        List of structural images accepted for combination
    anat_realign_xfm
        List of affine transforms to realign input images to final reference
    out_report
        Conformation report

    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['anat_files']),
        name='inputnode',
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=['anat_ref', 'anat_valid_list', 'anat_realign_xfm', 'out_report']
        ),
        name='outputnode',
    )

    # 0. Denoise and reorient T1w images(s) to RAS and resample to common voxel space
    anat_ref_dimensions = pe.Node(TemplateDimensions(), name='anat_ref_dimensions')
    denoise = pe.MapNode(
        ants.DenoiseImage(noise_model='Rician', num_threads=omp_nthreads),
        iterfield='input_image',
        name='denoise',
    )
    anat_conform = pe.MapNode(
        Conform(),
        iterfield='in_file',
        name='anat_conform',
    )

    workflow.connect([
        (inputnode, anat_ref_dimensions, [('anat_files', 't1w_list')]),
        (anat_ref_dimensions, denoise, [('t1w_valid_list', 'input_image')]),
        (anat_ref_dimensions, anat_conform, [
            ('target_zooms', 'target_zooms'),
            ('target_shape', 'target_shape'),
        ]),
        (denoise, anat_conform, [('output_image', 'in_file')]),
        (anat_ref_dimensions, outputnode, [
            ('out_report', 'out_report'),
            ('t1w_valid_list', 'anat_valid_list'),
        ]),
    ])

    if num_files == 1:
        get1st = pe.Node(niu.Select(index=[0]), name='get1st')
        outputnode.inputs.anat_realign_xfm = [str(load_data('itkIdentityTransform.txt'))]

        workflow.connect([
            (anat_conform, get1st, [('out_file', 'inlist')]),
            (get1st, outputnode, [('out', 'anat_ref')]),
        ])
        return workflow
    
    anat_conform_xfm = pe.MapNode(
        LTAConvert(in_lta='identity.nofile', out_lta=True),
        iterfield=['source_file', 'target_file'],
        name='anat_conform_xfm',
    )

    # 1. Template (only if several inputs or longitudinal)
    # 1a. Correct for bias field: the bias field is an additive factor
    #     in log-transformed intensity units. Therefore, it is not a linear
    #     combination of fields and N4 fails with merged images.
    # 1b. Align and merge if several T1w images are provided.
    n4_correct = pe.MapNode(
        ants.N4BiasFieldCorrection(dimension=3, copy_header=True, num_threads=omp_nthreads),
        iterfield='input_image',
        name='n4_correct',
        n_procs=1,
    ) # n_procs=1 for reproducibility
    # StructuralReference is fs.RobustTemplate if > 1 volume, copying otherwise
    anat_merge = pe.Node(
        StructuralReference(
            auto_detect_sensitivity=True,
            initial_timepoint=1, # for deterministic behavior
            intensity_scaling=True, # 7-DOF (rigid + intensity)
            subsample_threshold=200,
            fixed_timepoint=not longitudinal,
            no_iteration=not longitudinal,
            transform_outputs=True,
        ),
        mem_gb=2 * num_files - 1,
        name='anat_merge',
    )

    # 2. Reorient template to RAS, if needed (mri_robust_template may set to LIA)
    anat_reorient = pe.Node(image.Reorient(), name='anat_reorient')

    merge_xfm = pe.MapNode(
        niu.Merge(2),
        name='merge_xfm',
        iterfield=['in1', 'in2'],
        run_without_submitting=True,
    )
    concat_xfms = pe.MapNode(
        ConcatenateXFMs(inverse=True),
        name='concat_xfms',
        iterfield=['in_xfms'],
        run_without_submitting=True,
    )

    def _set_threads(in_list, maximum):
        return min(len(in_list), maximum)
    
    workflow.connect([
        (anat_ref_dimensions, anat_conform_xfm, [('anat_valid_list', 'source_file')]),
        (anat_conform, anat_conform_xfm, [('out_file', 'target_file')]),
        (anat_conform, n4_correct, [('out_file', 'input_image')]),
        (anat_conform, anat_merge, [
            (('out_file', _set_threads, omp_nthreads), 'num_threads'),
            (('out_file', add_suffix, '_template'), 'out_file')
        ]),
        (n4_correct, anat_merge, [('output_image', 'in_files')]),
        (anat_merge, anat_reorient, [('out_file', 'in_file')]),
        # combine orientation and template transforms
        (anat_conform_xfm, merge_xfm, [('out_lta', 'in1')]),
        (anat_merge, merge_xfm, [('transform_outputs', 'in2')]),
        (merge_xfm, concat_xfms, [('out', 'in_xfms')]),
        # outputs
        (anat_reorient, outputnode, [('out_file', 'anat_ref')]),
        (concat_xfms, outputnode, [('out_xfm', 'anat_realign_xfm')]),
    ])

    return workflow

def _probseg_fast2bids(inlist):
    """Reorder a list of probseg maps from FAST (CSF, WM, GM) to BIDS (GM, WM, CSF)"""
    return [inlist[2], inlist[1], inlist[0]]

def _is_skull_stripped(img):
    """Check if images are skull-stripped"""
    import nibabel as nb
    import numpy as np

    data = nb.load(img).dataobj
    sidevals = (
        np.abs(data[0, :, :]).sum()
        + np.abs(data[-1, :, :]).sum()
        + np.abs(data[:, 0, :]).sum()
        + np.abs(data[:, -1, :]).sum()
        + np.abs(data[:, :, 0]).sum()
        + np.abs(data[:, :, -1]).sum()
    )
    return sidevals < 10


def init_hdbet_wf(
    *,
    omp_nthreads: int = 1,
    use_gpu: bool = True,
    name: str = 'hdbet_wf',
) -> Workflow:
    """
    Build a workflow for brain extraction using HD-BET.
    
    HD-BET (High-resolution Brain Extraction Tool) is a deep learning-based
    brain extraction method that provides robust skull-stripping.
    
    Parameters
    ----------
    omp_nthreads : int
        Number of threads for parallel processing
    use_gpu : bool
        Use GPU acceleration (recommended)
    name : str
        Workflow name
        
    Returns
    -------
    Workflow
        Brain extraction workflow with inputnode.in_files and 
        outputnode.out_file, out_mask, bias_corrected
    """
    from nipype.interfaces import utility as niu
    from nipype.pipeline import engine as pe
    from nipype.interfaces.ants import N4BiasFieldCorrection
    
    workflow = Workflow(name=name)
    
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['in_files']),
        name='inputnode',
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['out_file', 'out_mask', 'bias_corrected', 'out_segm']),
        name='outputnode',
    )
    
    # HD-BET brain extraction
    hdbet = pe.Node(
        niu.Function(
            function=_run_hdbet,
            input_names=['in_file', 'use_gpu'],
            output_names=['out_file', 'out_mask'],
        ),
        name='hdbet',
    )
    hdbet.inputs.use_gpu = use_gpu
    
    # N4 bias field correction on brain-extracted image
    n4_correct = pe.Node(
        N4BiasFieldCorrection(
            dimension=3,
            copy_header=True,
            n_iterations=[50, 50, 30, 20],
            convergence_threshold=1e-6,
            shrink_factor=3,
            num_threads=omp_nthreads,
        ),
        name='n4_correct',
    )
    
    workflow.connect([
        (inputnode, hdbet, [(('in_files', _pop), 'in_file')]),
        (hdbet, n4_correct, [('out_file', 'input_image')]),
        (hdbet, outputnode, [
            ('out_file', 'out_file'),
            ('out_mask', 'out_mask'),
        ]),
        (n4_correct, outputnode, [('output_image', 'bias_corrected')]),
    ])
    
    # out_segm is not available from HD-BET, set to None
    outputnode.inputs.out_segm = None
    
    return workflow


def _run_hdbet(in_file: str, use_gpu: bool = True) -> tuple:
    """
    Run HD-BET brain extraction.
    
    Parameters
    ----------
    in_file : str
        Input NIfTI file path
    use_gpu : bool
        Use GPU acceleration
        
    Returns
    -------
    tuple
        (brain_extracted_file, brain_mask_file)
    """
    from pathlib import Path
    import subprocess
    
    in_path = Path(in_file)
    out_path = Path.cwd() / f'{in_path.stem.replace(".nii", "")}_hdbet.nii.gz'
    
    # Build HD-BET command
    cmd = ['hd-bet', '-i', str(in_path), '-o', str(out_path)]
    if not use_gpu:
        cmd.extend(['-device', 'cpu'])
    
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError:
        raise RuntimeError(
            "HD-BET not found. Install with: pip install hd-bet"
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"HD-BET failed: {e.stderr}")
    
    # HD-BET creates <output>_mask.nii.gz for the mask
    mask_path = out_path.with_name(out_path.stem.replace('.nii', '') + '_mask.nii.gz')
    
    if not out_path.exists():
        raise RuntimeError(f"HD-BET output not found: {out_path}")
    if not mask_path.exists():
        # Try alternate naming convention
        mask_path = out_path.with_suffix('').with_suffix('.nii.gz').with_name(
            out_path.stem + '_mask.nii.gz'
        )
        if not mask_path.exists():
            raise RuntimeError("HD-BET mask not found")
    
    return str(out_path), str(mask_path)


def init_synthstrip_wf(
    *,
    omp_nthreads: int = 1,
    name: str = 'synthstrip_wf',
) -> Workflow:
    """
    Build a workflow for brain extraction using SynthStrip.
    
    SynthStrip is a robust, learning-based brain extraction tool from FreeSurfer
    that works across MRI contrasts and resolutions.
    
    Parameters
    ----------
    omp_nthreads : int
        Number of threads for parallel processing
    name : str
        Workflow name
        
    Returns
    -------
    Workflow
        Brain extraction workflow with inputnode.in_files and 
        outputnode.out_file, out_mask, bias_corrected
    """
    from nipype.interfaces import utility as niu
    from nipype.pipeline import engine as pe
    from nipype.interfaces.ants import N4BiasFieldCorrection
    
    workflow = Workflow(name=name)
    
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['in_files']),
        name='inputnode',
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['out_file', 'out_mask', 'bias_corrected', 'out_segm']),
        name='outputnode',
    )
    
    # SynthStrip brain extraction
    synthstrip = pe.Node(
        niu.Function(
            function=_run_synthstrip,
            input_names=['in_file'],
            output_names=['out_file', 'out_mask'],
        ),
        name='synthstrip',
    )
    
    # N4 bias field correction on brain-extracted image
    n4_correct = pe.Node(
        N4BiasFieldCorrection(
            dimension=3,
            copy_header=True,
            n_iterations=[50, 50, 30, 20],
            convergence_threshold=1e-6,
            shrink_factor=3,
            num_threads=omp_nthreads,
        ),
        name='n4_correct',
    )
    
    workflow.connect([
        (inputnode, synthstrip, [(('in_files', _pop), 'in_file')]),
        (synthstrip, n4_correct, [('out_file', 'input_image')]),
        (synthstrip, outputnode, [
            ('out_file', 'out_file'),
            ('out_mask', 'out_mask'),
        ]),
        (n4_correct, outputnode, [('output_image', 'bias_corrected')]),
    ])
    
    # out_segm is not available from SynthStrip, set to None
    outputnode.inputs.out_segm = None
    
    return workflow


def _run_synthstrip(in_file: str) -> tuple:
    """
    Run SynthStrip brain extraction.
    
    Parameters
    ----------
    in_file : str
        Input NIfTI file path
        
    Returns
    -------
    tuple
        (brain_extracted_file, brain_mask_file)
    """
    from pathlib import Path
    import subprocess
    
    in_path = Path(in_file)
    out_path = Path.cwd() / f'{in_path.stem.replace(".nii", "")}_synthstrip.nii.gz'
    mask_path = Path.cwd() / f'{in_path.stem.replace(".nii", "")}_synthstrip_mask.nii.gz'
    
    # Build SynthStrip command (mri_synthstrip is the FreeSurfer command)
    cmd = [
        'mri_synthstrip',
        '-i', str(in_path),
        '-o', str(out_path),
        '-m', str(mask_path),
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError:
        # Try synthstrip as standalone command
        cmd[0] = 'synthstrip'
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except FileNotFoundError:
            raise RuntimeError(
                "SynthStrip not found. Install FreeSurfer or use: "
                "pip install synthstrip-cpu (or synthstrip for GPU)"
            )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"SynthStrip failed: {e.stderr}")
    
    if not out_path.exists():
        raise RuntimeError(f"SynthStrip output not found: {out_path}")
    if not mask_path.exists():
        raise RuntimeError(f"SynthStrip mask not found: {mask_path}")
    
    return str(out_path), str(mask_path)