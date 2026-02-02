# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2024 The OncoPrep Developers
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
"""Spatial normalization workflows for BraTS-style multi-modal registration."""

from collections import defaultdict
from pathlib import Path
from typing import List, Optional

from nipype.interfaces import ants, utility as niu
from nipype.interfaces.ants.base import Info as ANTsInfo
from nipype.pipeline import engine as pe
from niworkflows.engine import Workflow, tag
from templateflow import __version__ as tf_ver
from templateflow.api import get_metadata

from oncoprep.utils.logging import get_logger

LOGGER = get_logger(__name__)


@tag('anat.register-template')
def init_multimodal_template_registration_wf(
    *,
    sloppy: bool = False,
    omp_nthreads: int = 1,
    templates: Optional[List[str]] = None,
    modalities: Optional[List[str]] = None,
    name: str = 'multimodal_template_registration_wf',
) -> Workflow:
    """
    Build a multi-modal spatial normalization workflow using ANTs.

    Performs nonlinear registration of T1w to template space, then applies
    the same transformation to T1ce, T2w, and FLAIR modalities.

    Parameters
    ----------
    sloppy : bool
        Apply sloppy arguments to speed up processing. Use with caution,
        registration will be less accurate.
    omp_nthreads : int
        Maximum number of threads per process.
    templates : list[str] | None
        List of standard space names (e.g., 'MNI152NLin2009cAsym').
        Default: ['MNI152NLin2009cAsym']
    modalities : list[str] | None
        List of modalities to register (e.g., ['T1w', 'T1ce', 'T2w', 'FLAIR']).
        Default: ['T1w', 'T1ce', 'T2w', 'FLAIR']
    name : str
        Workflow name.

    Returns
    -------
    Workflow
        The workflow object.

    Inputs
    ------
    t1w
        T1-weighted anatomical image (used as reference).
    t1ce
        T1-weighted post-contrast image (optional).
    t2w
        T2-weighted image (optional).
    flair
        FLAIR image (optional).
    template
        Template name specification.

    Outputs
    -------
    t1w_std
        T1w image in template space.
    t1ce_std
        T1ce image in template space.
    t2w_std
        T2w image in template space.
    flair_std
        FLAIR image in template space.
    anat2std_xfm
        Composite transform (moving to fixed).
    std2anat_xfm
        Inverse composite transform (fixed to moving).
    template
        Template name.
    """
    if templates is None:
        templates = ['MNI152NLin2009cAsym']
    
    if modalities is None:
        modalities = ['T1w', 'T1ce', 'T2w', 'FLAIR']
    
    ntpls = len(templates)
    workflow = Workflow(name=name)
    
    # Generate workflow description for boilerplate
    if templates:
        workflow.__desc__ = """\
Volume-based spatial normalization to {targets} ({targets_id}) was performed through
nonlinear (SyN) registration with `antsRegistration` (ANTs {ants_ver}),
using brain-extracted and bias-corrected T1w reference.
All modalities (T1w, T1ce, T2w, FLAIR) were aligned to template space using the
T1w-to-template transformation to maintain inter-modal consistency.
The following template{tpls} were selected for spatial normalization
and accessed with *TemplateFlow* [{tf_ver}]:
""".format(
            ants_ver=ANTsInfo.version() or '(version unknown)',
            targets='{} standard space{}'.format(
                defaultdict('several'.format, {1: 'one', 2: 'two', 3: 'three', 4: 'four'})[ntpls],
                's' * (ntpls != 1),
            ),
            targets_id=', '.join(templates),
            tf_ver=tf_ver,
            tpls=(' was', 's were')[ntpls != 1],
        )
        
        # Append template citations to description
        for template in templates:
            try:
                template_meta = get_metadata(template.split(':')[0])
                template_refs = [f'@{template.split(":")[ 0].lower()}']
                
                if template_meta.get('RRID', None):
                    template_refs.append(f'RRID:{template_meta["RRID"]}')
                
                workflow.__desc__ += """\
*{template_name}* [{template_refs}; TemplateFlow ID: {template}]""".format(
                    template=template,
                    template_name=template_meta.get('Name', template),
                    template_refs=', '.join(template_refs),
                )
                workflow.__desc__ += '.\n' if template == templates[-1] else ', '
            except Exception as e:
                LOGGER.warning(f'Could not fetch metadata for template {template}: {e}')
                workflow.__desc__ += f'*{template}*.\n' if template == templates[-1] else f'*{template}*, '
    
    # Input node
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['t1w', 't1ce', 't2w', 'flair', 'template']
        ),
        name='inputnode',
    )
    inputnode.iterables = [('template', templates)]
    
    # Output node
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                't1w_std',
                't1ce_std',
                't2w_std',
                'flair_std',
                'anat2std_xfm',
                'std2anat_xfm',
                'template',
            ]
        ),
        name='outputnode',
    )
    
    # Get template from templateflow
    get_template = pe.Node(
        niu.Function(
            function=_get_template_image,
            input_names=['template_name'],
            output_names=['template_image'],
        ),
        name='get_template',
        run_without_submitting=True,
    )
    
    # Register T1w to template (nonlinear SyN)
    register_t1w = pe.Node(
        ants.Registration(
            dimension=3,
            float=True,
            output_transform_prefix='t1w2std_',
            transforms=['Affine', 'SyN'],
            transform_parameters=[(2.0,), (0.1, 3.0, 0.0)],
            metric=['Mattes', 'CC'],
            metric_weight=[1, 1],
            radius_or_number_of_bins=[32, 4],
            sampling_strategy=['Regular', 'Regular'],
            sampling_percentage=[0.3, None],
            number_of_iterations=[[1000, 500, 250, 100], [100, 70, 50, 20]],
            convergence_threshold=[1e-8, 1e-9],
            convergence_window_size=[10, 10],
            smoothing_sigmas=[[3, 2, 1, 0], [3, 2, 1, 0]],
            sigma_units=['vox', 'vox'],
            shrink_factors=[[8, 4, 2, 1], [8, 4, 2, 1]],
            use_estimate_learning_rate_once=[True, True],
            use_histogram_matching=[True, True],
            initial_moving_transform_com=True,
            verbose=True,
        ),
        name='register_t1w',
        n_procs=omp_nthreads,
        mem_gb=3,
    )
    
    # Apply transforms to other modalities
    apply_t1ce = pe.Node(
        ants.ApplyTransforms(
            dimension=3,
            interpolation='Linear',
            default_value=0,
        ),
        name='apply_t1ce',
        n_procs=omp_nthreads,
    )
    
    apply_t2w = pe.Node(
        ants.ApplyTransforms(
            dimension=3,
            interpolation='Linear',
            default_value=0,
        ),
        name='apply_t2w',
        n_procs=omp_nthreads,
    )
    
    apply_flair = pe.Node(
        ants.ApplyTransforms(
            dimension=3,
            interpolation='Linear',
            default_value=0,
        ),
        name='apply_flair',
        n_procs=omp_nthreads,
    )
    
    # Rename outputs to BIDS convention
    rename_t1w = pe.Node(
        niu.Function(
            function=_rename_output,
            input_names=['image_path', 'modality', 'space'],
            output_names=['output_path'],
        ),
        name='rename_t1w',
    )
    rename_t1w.inputs.modality = 'T1w'
    
    rename_t1ce = pe.Node(
        niu.Function(
            function=_rename_output,
            input_names=['image_path', 'modality', 'space'],
            output_names=['output_path'],
        ),
        name='rename_t1ce',
    )
    rename_t1ce.inputs.modality = 'T1w'  # T1ce is classified as T1w in BIDS
    
    rename_t2w = pe.Node(
        niu.Function(
            function=_rename_output,
            input_names=['image_path', 'modality', 'space'],
            output_names=['output_path'],
        ),
        name='rename_t2w',
    )
    rename_t2w.inputs.modality = 'T2w'
    
    rename_flair = pe.Node(
        niu.Function(
            function=_rename_output,
            input_names=['image_path', 'modality', 'space'],
            output_names=['output_path'],
        ),
        name='rename_flair',
    )
    rename_flair.inputs.modality = 'FLAIR'
    
    # Connect workflow
    workflow.connect([
        # Get template
        (inputnode, get_template, [('template', 'template_name')]),
        
        # Register T1w to template
        (inputnode, register_t1w, [('t1w', 'moving_image')]),
        (get_template, register_t1w, [('template_image', 'fixed_image')]),
        
        # Apply transforms to other modalities
        (inputnode, apply_t1ce, [('t1ce', 'input_image')]),
        (get_template, apply_t1ce, [('template_image', 'reference_image')]),
        (register_t1w, apply_t1ce, [('composite_transform', 'transforms')]),
        
        (inputnode, apply_t2w, [('t2w', 'input_image')]),
        (get_template, apply_t2w, [('template_image', 'reference_image')]),
        (register_t1w, apply_t2w, [('composite_transform', 'transforms')]),
        
        (inputnode, apply_flair, [('flair', 'input_image')]),
        (get_template, apply_flair, [('template_image', 'reference_image')]),
        (register_t1w, apply_flair, [('composite_transform', 'transforms')]),
        
        # Rename outputs
        (register_t1w, rename_t1w, [('warped_image', 'image_path')]),
        (inputnode, rename_t1w, [('template', 'space')]),
        
        (apply_t1ce, rename_t1ce, [('output_image', 'image_path')]),
        (inputnode, rename_t1ce, [('template', 'space')]),
        
        (apply_t2w, rename_t2w, [('output_image', 'image_path')]),
        (inputnode, rename_t2w, [('template', 'space')]),
        
        (apply_flair, rename_flair, [('output_image', 'image_path')]),
        (inputnode, rename_flair, [('template', 'space')]),
        
        # Output
        (rename_t1w, outputnode, [('output_path', 't1w_std')]),
        (rename_t1ce, outputnode, [('output_path', 't1ce_std')]),
        (rename_t2w, outputnode, [('output_path', 't2w_std')]),
        (rename_flair, outputnode, [('output_path', 'flair_std')]),
        (register_t1w, outputnode, [
            ('composite_transform', 'anat2std_xfm'),
            ('inverse_composite_transform', 'std2anat_xfm'),
        ]),
        (inputnode, outputnode, [('template', 'template')]),
    ])
    
    return workflow


def _get_template_image(template_name: str) -> str:
    """
    Get the template image path from templateflow.
    
    Parameters
    ----------
    template_name : str
        Template name (e.g., 'MNI152NLin2009cAsym')
    
    Returns
    -------
    str
        Path to the template T1w image
    """
    from templateflow.api import get as get_template
    from nipype import logging as nipype_logging
    
    logger = nipype_logging.getLogger('nipype.workflow')
    
    try:
        template_path = get_template(
            template_name,
            desc='brain',
            resolution=1,
            suffix='T1w',
            extension='.nii.gz',
        )
        logger.info(f'Using template: {template_path}')
        return str(template_path)
    except Exception as e:
        logger.error(f'Error fetching template {template_name}: {e}')
        raise


def _rename_output(image_path: str, modality: str, space: str) -> str:
    """
    Rename output image to BIDS convention.
    
    Parameters
    ----------
    image_path : str
        Path to the image
    modality : str
        Modality name (T1w, T2w, FLAIR)
    space : str
        Template space name
    
    Returns
    -------
    str
        New path with BIDS-convention naming
    """
    from pathlib import Path as PathlibPath
    
    p = PathlibPath(image_path)
    stem = p.stem.rsplit('.', 1)[0] if '.nii' in p.suffix else p.stem
    
    # Extract subject/session info if present
    parts = stem.split('_')
    base_parts = []
    for part in parts:
        if part.startswith('space-'):
            break
        base_parts.append(part)
    
    base = '_'.join(base_parts)
    new_name = f'{base}_space-{space}_{modality}.nii.gz'
    new_path = p.parent / new_name
    
    # Rename file
    if PathlibPath(image_path).exists():
        PathlibPath(image_path).rename(new_path)
    
    return str(new_path)
