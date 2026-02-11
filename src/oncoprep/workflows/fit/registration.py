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
from niworkflows.interfaces.norm import SpatialNormalization

from oncoprep.workflows._compat import Workflow, tag
from templateflow import __version__ as tf_ver
from templateflow.api import get_metadata

from oncoprep.interfaces.templateflow import TemplateDesc, TemplateFlowSelect
from oncoprep.utils.logging import get_logger

LOGGER = get_logger(__name__)


def _greedy_registration(moving_image, fixed_image, omp_nthreads=1, sloppy=False):
    """
    Perform diffeomorphic registration using PICSL Greedy.
    
    Parameters
    ----------
    moving_image : str
        Path to the moving (subject) image
    fixed_image : str
        Path to the fixed (template) image
    omp_nthreads : int
        Number of threads to use
    sloppy : bool
        Use faster but less accurate settings
        
    Returns
    -------
    forward_transforms : list
        List of transform files [affine, warp] for moving-to-fixed
    reverse_transforms : list
        List of transform files [warp, affine] for fixed-to-moving
    warped_image : str
        Path to the warped moving image in template space
    """
    import os
    import shutil
    import subprocess
    from pathlib import Path
    
    # Check if greedy is available
    if shutil.which('greedy') is None:
        raise FileNotFoundError(
            "PICSL Greedy is not installed or not in PATH. "
            "Install it from https://github.com/pyushkevich/greedy or use "
            "--registration-backend ants instead."
        )
    
    # Set thread count
    os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = str(omp_nthreads)
    
    work_dir = Path.cwd()
    prefix = 'greedy_'
    
    affine_out = work_dir / f'{prefix}affine.mat'
    warp_out = work_dir / f'{prefix}warp.nii.gz'
    inv_warp_out = work_dir / f'{prefix}inv_warp.nii.gz'
    warped_out = work_dir / f'{prefix}warped.nii.gz'
    
    # Step 1: Affine registration
    affine_cmd = [
        'greedy',
        '-d', '3',
        '-a',  # Affine mode
        '-i', fixed_image, moving_image,
        '-o', str(affine_out),
        '-ia-image-centers',  # Initialize with image centers
        '-n', '100x50x10' if sloppy else '200x100x50x25',
        '-m', 'NCC', '4x4x4',  # Normalized cross-correlation
        '-threads', str(omp_nthreads),
    ]
    
    print(f'Running Greedy affine registration: {" ".join(affine_cmd)}')
    subprocess.run(affine_cmd, check=True)
    
    # Step 2: Deformable registration
    deform_iter = '50x30x10' if sloppy else '100x70x50x20'
    deform_cmd = [
        'greedy',
        '-d', '3',
        '-i', fixed_image, moving_image,
        '-it', str(affine_out),
        '-o', str(warp_out),
        '-oinv', str(inv_warp_out),
        '-n', deform_iter,
        '-m', 'NCC', '4x4x4',
        '-s', '2.0vox', '0.5vox',  # Smoothing sigmas
        '-threads', str(omp_nthreads),
    ]
    
    print(f'Running Greedy deformable registration: {" ".join(deform_cmd)}')
    subprocess.run(deform_cmd, check=True)
    
    # Step 3: Apply transform to create warped image
    reslice_cmd = [
        'greedy',
        '-d', '3',
        '-rf', fixed_image,
        '-rm', moving_image, str(warped_out),
        '-r', str(warp_out), str(affine_out),
        '-threads', str(omp_nthreads),
    ]
    
    print(f'Running Greedy reslice: {" ".join(reslice_cmd)}')
    subprocess.run(reslice_cmd, check=True)
    
    # Return transforms in order for forward/inverse application
    forward_transforms = [str(warp_out), str(affine_out)]
    reverse_transforms = [str(affine_out), str(inv_warp_out)]  # Note: inverse order
    
    return forward_transforms, reverse_transforms, str(warped_out)


@tag('anat.register-template')
def init_multimodal_template_registration_wf(
    *,
    sloppy: bool = False,
    omp_nthreads: int = 1,
    templates: Optional[List[str]] = None,
    modalities: Optional[List[str]] = None,
    registration_backend: str = 'ants',
    name: str = 'multimodal_template_registration_wf',
) -> Workflow:
    """
    Build a multi-modal spatial normalization workflow.

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
    registration_backend : str
        Registration backend: 'ants' (ANTs SyN) or 'greedy' (PICSL Greedy).
        Default: 'ants'
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
        if registration_backend == 'greedy':
            reg_desc = "nonlinear (greedy diffeomorphic) registration with `greedy` (PICSL Greedy)"
        else:
            reg_desc = f"nonlinear (SyN) registration with `antsRegistration` (ANTs {ANTsInfo.version() or '(version unknown)'})"
        
        workflow.__desc__ = """\
Volume-based spatial normalization to {targets} ({targets_id}) was performed through
{reg_desc},
using brain-extracted and bias-corrected T1w reference.
All modalities (T1w, T1ce, T2w, FLAIR) were aligned to template space using the
T1w-to-template transformation to maintain inter-modal consistency.
The following template{tpls} were selected for spatial normalization
and accessed with *TemplateFlow* [{tf_ver}]:
""".format(
            reg_desc=reg_desc,
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
            fields=[
                't1w',
                't1w_mask',
                'lesion_mask',
                't1ce', 
                't2w', 
                'flair', 
                'template']
        ),
        name='inputnode',
    )
    inputnode.iterables = [('template', templates)]
    
    # Output node
    out_fields = [
                't1w_std',
                't1ce_std',
                't2w_std',
                'flair_std',
                'anat2std_xfm',
                'std2anat_xfm',
                'template',
            ]

    outputnode = _make_outputnode(
        workflow, out_fields, joinsource='inputnode',
    )

    # # Get template from templateflow
    # get_template = pe.Node(
    #     niu.Function(
    #         function=_get_template_image,
    #         input_names=['template_name'],
    #         output_names=['template_image'],
    #     ),
    #     name='get_template',
    #     run_without_submitting=True,
    # )

    split_desc = pe.Node(
        TemplateDesc(),
        run_without_submitting=True,
        name='split_desc',
    )

    tf_select = pe.Node(
        TemplateFlowSelect(resolution=1+sloppy),
        name='tf_select',
        run_without_submitting=True,
    )

    set_reference = pe.Node(
        niu.Function(
            function=_set_reference,
            output_names=['reference_type'],
        ),
        name='set_reference',
    )
    set_reference.inputs.image_type = 'T1w'

    trunc_mov = pe.Node(
        ants.ImageMath(operation='TruncateImageIntensity', op2='0.01 0.999 256'),
        name='trunc_mov',
    )

    register_T1w = pe.Node(
        SpatialNormalization(
            float=True,
            flavor=['precise', 'testing'][sloppy],
        ),
        name='register_T1w',
        n_procs=omp_nthreads,
        mem_gb=2,
    )

    fmt_cohort = pe.Node(
        niu.Function(
            function=_fmt_cohort, output_names=['template', 'spec']
            ),
        name='fmt_cohort',
        run_without_submitting=True,
    )
    
    workflow.connect([
        (inputnode, split_desc, [('template', 'template')]),
        (inputnode, trunc_mov, [('t1w', 'op1')]),
        (inputnode, register_T1w, [
            ('t1w_mask', 'moving_mask'),
            ('lesion_mask', 'lesion_mask'),
        ]),
        (split_desc, tf_select, [
            ('name', 'template'),
            ('spec', 'template_spec'),
        ]),
        (tf_select, set_reference, [
            ('t1w_file', 'template_t1w'),
            ('t2w_file', 'template_t2w'),
        ]),
        (set_reference, register_T1w, [
            ('reference_type', 'reference'),
        ]),
        (split_desc, register_T1w, [
            ('name', 'template'),
            ('spec', 'template_spec'),
        ]),
        (trunc_mov, register_T1w, [
            ('output_image', 'moving_image'),
        ]),
        (split_desc, fmt_cohort, [
            ('name', 'template'),
            ('spec', 'spec'),
        ]),
        (fmt_cohort, outputnode, [
            ('template', 'template'),
            ('spec', 'template_spec'),
        ]),
        (register_T1w, outputnode, [
            ('composite_transform', 'anat2std_xfm'),
            ('inverse_composite_transform', 'std2anat_xfm'),
        ]),
    ])    
    
    # Apply transforms to other modalities (only if inputs are provided)
    # These use Function nodes that handle None inputs gracefully
    apply_t1ce = pe.Node(
        niu.Function(
            function=_apply_transform_if_exists,
            input_names=['input_image', 'reference_image', 'transforms', 'backend'],
            output_names=['output_image'],
        ),
        name='apply_t1ce',
    )
    apply_t1ce.inputs.backend = registration_backend
    
    apply_t2w = pe.Node(
        niu.Function(
            function=_apply_transform_if_exists,
            input_names=['input_image', 'reference_image', 'transforms', 'backend'],
            output_names=['output_image'],
        ),
        name='apply_t2w',
    )
    apply_t2w.inputs.backend = registration_backend
    
    apply_flair = pe.Node(
        niu.Function(
            function=_apply_transform_if_exists,
            input_names=['input_image', 'reference_image', 'transforms', 'backend'],
            output_names=['output_image'],
        ),
        name='apply_flair',
    )
    apply_flair.inputs.backend = registration_backend
    
    # Rename outputs to BIDS convention (handles None input)
    rename_t1w = pe.Node(
        niu.Function(
            function=_rename_output_safe,
            input_names=['image_path', 'modality', 'space'],
            output_names=['output_path'],
        ),
        name='rename_t1w',
    )
    rename_t1w.inputs.modality = 'T1w'
    
    rename_t1ce = pe.Node(
        niu.Function(
            function=_rename_output_safe,
            input_names=['image_path', 'modality', 'space'],
            output_names=['output_path'],
        ),
        name='rename_t1ce',
    )
    rename_t1ce.inputs.modality = 'T1w'  # T1ce is classified as T1w in BIDS
    
    rename_t2w = pe.Node(
        niu.Function(
            function=_rename_output_safe,
            input_names=['image_path', 'modality', 'space'],
            output_names=['output_path'],
        ),
        name='rename_t2w',
    )
    rename_t2w.inputs.modality = 'T2w'
    
    rename_flair = pe.Node(
        niu.Function(
            function=_rename_output_safe,
            input_names=['image_path', 'modality', 'space'],
            output_names=['output_path'],
        ),
        name='rename_flair',
    )
    rename_flair.inputs.modality = 'FLAIR'
    
    # Connect apply transform nodes for other modalities
    # Uses composite_transform from register_T1w (SpatialNormalization uses ANTs internally)
    workflow.connect([
        # Apply transforms to T1ce
        (inputnode, apply_t1ce, [('t1ce', 'input_image')]),
        (tf_select, apply_t1ce, [('t1w_file', 'reference_image')]),
        (register_T1w, apply_t1ce, [('composite_transform', 'transforms')]),
        
        # Apply transforms to T2w
        (inputnode, apply_t2w, [('t2w', 'input_image')]),
        (tf_select, apply_t2w, [('t1w_file', 'reference_image')]),
        (register_T1w, apply_t2w, [('composite_transform', 'transforms')]),
        
        # Apply transforms to FLAIR
        (inputnode, apply_flair, [('flair', 'input_image')]),
        (tf_select, apply_flair, [('t1w_file', 'reference_image')]),
        (register_T1w, apply_flair, [('composite_transform', 'transforms')]),
    ])
    
    # Connect rename nodes and output
    workflow.connect([
        # Rename T1w (warped image from registration)
        (register_T1w, rename_t1w, [('warped_image', 'image_path')]),
        (inputnode, rename_t1w, [('template', 'space')]),
        
        # Rename T1ce
        (apply_t1ce, rename_t1ce, [('output_image', 'image_path')]),
        (inputnode, rename_t1ce, [('template', 'space')]),
        
        # Rename T2w
        (apply_t2w, rename_t2w, [('output_image', 'image_path')]),
        (inputnode, rename_t2w, [('template', 'space')]),
        
        # Rename FLAIR
        (apply_flair, rename_flair, [('output_image', 'image_path')]),
        (inputnode, rename_flair, [('template', 'space')]),
        
        # Connect renamed outputs to outputnode
        (rename_t1w, outputnode, [('output_path', 't1w_std')]),
        (rename_t1ce, outputnode, [('output_path', 't1ce_std')]),
        (rename_t2w, outputnode, [('output_path', 't2w_std')]),
        (rename_flair, outputnode, [('output_path', 'flair_std')]),
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


def _rename_output_safe(image_path, modality: str, space: str):
    """
    Rename output image to BIDS convention, handling None inputs.
    
    Parameters
    ----------
    image_path : str or None
        Path to the image (can be None if modality not available)
    modality : str
        Modality name (T1w, T2w, FLAIR)
    space : str
        Template space name
    
    Returns
    -------
    str or None
        New path with BIDS-convention naming, or None if input was None
    """
    if image_path is None:
        return None
    
    # Inline the rename logic (can't call _rename_output from isolated Function node)
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


def _apply_transform_if_exists(input_image, reference_image, transforms, backend='ants'):
    """
    Apply transforms to an image only if the input exists.
    
    Parameters
    ----------
    input_image : str or None
        Path to input image (can be None)
    reference_image : str
        Path to reference image
    transforms : list or str
        Transform(s) to apply
    backend : str
        Registration backend ('ants' or 'greedy')
        
    Returns
    -------
    str or None
        Path to output image, or None if input was None
    """
    if input_image is None:
        return None
    
    from pathlib import Path
    import subprocess
    
    input_path = Path(input_image)
    output_path = Path.cwd() / f'{input_path.stem.replace(".nii", "")}_std.nii.gz'
    
    # Build transform application command
    if isinstance(transforms, str):
        transforms = [transforms]
    
    if backend == 'greedy':
        # Use greedy to apply transforms
        # Greedy uses -r for reslice mode with -rf (reference) and -rm (moving)
        cmd = [
            'greedy', '-d', '3',
            '-rf', str(reference_image),
            '-rm', str(input_image), str(output_path),
        ]
        # Add transforms
        for t in transforms:
            cmd.extend(['-r', str(t)])
    else:
        # Use ANTs to apply transforms
        cmd = [
            'antsApplyTransforms',
            '-d', '3',
            '-i', str(input_image),
            '-r', str(reference_image),
            '-o', str(output_path),
            '-n', 'Linear',
        ]
        for t in transforms:
            cmd.extend(['-t', str(t)])
    
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    
    return str(output_path)

def _set_reference(image_type, template_t1w, template_t2w=None):
    """
    Determine the normalization reference and whether histogram matching will be used.

    Parameters
    ----------
    image_type : MR image type of anatomical reference (T1w, T2w)
    template_t1w : T1w file
    template_t2w : T2w file or undefined

    Returns
    -------
    reference_type : modality of template reference (T1w, T2w)
    """
    if image_type == 'T2w':
        if template_t2w:
            return 'T2w'
        return 'T1w'
    return 'T1w'

def _fmt_cohort(template, spec):
    cohort = spec.pop('cohort', None)
    if cohort is not None:
        template = f'{template}:cohort-{cohort}'
    return template, spec

def _make_outputnode(workflow, out_fields, joinsource):
    if joinsource:
        pout = pe.Node(niu.IdentityInterface(fields=out_fields), name='poutputnode')
        out = pe.JoinNode(
            niu.IdentityInterface(fields=out_fields), name='outputnode', joinsource=joinsource
        )
        workflow.connect([(pout, out, [(f, f) for f in out_fields])])
        return pout
    return pe.Node(niu.IdentityInterface(fields=out_fields), name='outputnode')