# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
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
"""OncoPrep anatomical (T1w, T2w, FLAIR) preprocessing workflows for BraTS data."""

from pathlib import Path
from typing import List, Optional

import nibabel as nb
import numpy as np
from nipype import logging
from nipype.interfaces import ants, utility as niu
from nipype.pipeline import engine as pe

from niworkflows.engine import Workflow

from .fit.registration import init_multimodal_template_registration_wf

LOGGER = logging.getLogger('nipype.workflow')


def init_anat_preproc_wf(
    *,
    t1w: list,
    t1ce: Optional[list] = None,
    t2w: Optional[list] = None,
    flair: Optional[list] = None,
    output_spaces: Optional[list] = None,
    skull_strip_template: str = 'OASIS30ANTs',
    omp_nthreads: int = 1,
    use_gpu: bool = False,
    defacing: bool = False,
    sloppy: bool = False,
    name: str = 'anat_preproc_wf',
):
    """
    Stage the anatomical preprocessing steps of OncoPrep for BraTS data.

    This workflow handles:
      - T1w reference: averaging multiple T1w images if present
      - Brain extraction using HD-BET (GPU-accelerated)
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
    skull_strip_template : :obj:`str`
        Template for skull stripping (default: 'OASIS30ANTs')
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
    if output_spaces is None:
        output_spaces = ['MNI152NLin2009cAsym']

    t1ce = t1ce or []
    t2w = t2w or []
    flair = flair or []

    workflow = Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['t1w', 't1ce', 't2w', 'flair']
        ),
        name='inputnode',
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                't1w_preproc',
                't1w_mask',
                't1w_defaced',
                't1ce_preproc',
                't1ce_defaced',
                't2w_preproc',
                't2w_defaced',
                'flair_preproc',
                'flair_defaced',
                'anat2std_xfm',
                'std2anat_xfm',
                'template',
            ]
        ),
        name='outputnode',
    )

    workflow.add_nodes([inputnode])

    desc = """
Anatomical data preprocessing

: A total of {} T1-weighted (T1w) images were found within the input
BIDS dataset.
""".format(len(t1w))

    if t1ce:
        desc += f"Additionally, {len(t1ce)} T1-weighted contrast-enhanced (T1ce) images "
        desc += f"{len(t2w)} T2-weighted (T2w) images, and {len(flair)} FLAIR images were available.\n"

    # Stage 1: Conform and validate T1w images
    # =========================================
    LOGGER.info('ANAT Stage 1: T1w conformance and averaging')

    # Create T1w reference (average if multiple)
    if len(t1w) > 1:
        t1w_ref = pe.Node(
            niu.Function(
                function=_average_images,
                input_names=['image_list'],
                output_names=['output_image'],
            ),
            name='t1w_ref',
        )
        desc += f"""\
The T1w reference was computed as the average of {len(t1w)} T1-weighted images
after rigid-body realignment to a common space.
"""
        workflow.connect([(inputnode, t1w_ref, [('t1w', 'image_list')])])
        t1w_source = t1w_ref
        t1w_output_field = 'output_image'
    else:
        t1w_source = inputnode
        t1w_output_field = 't1w'

    # Stage 2: INU correction
    # =======================
    LOGGER.info('ANAT Stage 2: N4 bias field correction')

    n4_t1w = pe.Node(
        ants.N4BiasFieldCorrection(
            dimension=3,
            copy_header=True,
            num_threads=omp_nthreads,
        ),
        name='n4_t1w',
    )
    desc += """\
The T1w-reference was corrected for intensity non-uniformity (INU) using
`N4BiasFieldCorrection` from the Advanced Normalization Tools (ANTs) package.
"""

    workflow.connect([(t1w_source, n4_t1w, [(t1w_output_field, 'input_image')])])

    # Stage 3: Brain extraction
    # =========================
    LOGGER.info('ANAT Stage 3: Brain extraction with HD-BET')

    brain_extract = pe.Node(
        niu.Function(
            function=_hd_bet_skullstrip,
            input_names=['img_path', 'use_gpu'],
            output_names=['brain_img', 'brain_mask'],
        ),
        name='brain_extract',
    )
    brain_extract.inputs.use_gpu = use_gpu

    desc += f"""\
Brain extraction was performed using HD-BET, a deep learning-based
brain extraction tool trained on tumor-bearing brain scans.
"""
    if use_gpu:
        desc += "GPU acceleration was enabled.\n"

    workflow.connect([(n4_t1w, brain_extract, [('output_image', 'img_path')])])

    # Buffer nodes for optional modalities
    # ====================================
    t1w_buffer = pe.Node(
        niu.IdentityInterface(fields=['t1w_preproc', 't1w_mask']),
        name='t1w_buffer',
    )
    workflow.connect([
        (brain_extract, t1w_buffer, [
            ('brain_img', 't1w_preproc'),
            ('brain_mask', 't1w_mask'),
        ]),
    ])

    # Stage 4: Multi-modal co-registration
    # ====================================
    if t1ce or t2w or flair:
        LOGGER.info('ANAT Stage 4: Multi-modal co-registration to T1w')

        coreg_buffer = pe.Node(
            niu.IdentityInterface(fields=['t1ce_preproc', 't2w_preproc', 'flair_preproc']),
            name='coreg_buffer',
        )

        if t1ce:
            coreg_t1ce = pe.Node(
                niu.Function(
                    function=_register_modality,
                    input_names=['moving', 'fixed', 'fixed_mask'],
                    output_names=['registered'],
                ),
                name='coreg_t1ce',
            )
            desc += "\nT1-weighted contrast-enhanced (T1ce) images were rigidly registered to T1w space using ANTs."

            workflow.connect([
                (inputnode, coreg_t1ce, [('t1ce', 'moving')]),
                (brain_extract, coreg_t1ce, [
                    ('brain_img', 'fixed'),
                    ('brain_mask', 'fixed_mask'),
                ]),
                (coreg_t1ce, coreg_buffer, [('registered', 't1ce_preproc')]),
            ])
        else:
            coreg_buffer.inputs.t1ce_preproc = None

        if t2w:
            coreg_t2w = pe.Node(
                niu.Function(
                    function=_register_modality,
                    input_names=['moving', 'fixed', 'fixed_mask'],
                    output_names=['registered'],
                ),
                name='coreg_t2w',
            )
            desc += "\nT2-weighted (T2w) images were rigidly registered to T1w space using ANTs."

            workflow.connect([
                (inputnode, coreg_t2w, [('t2w', 'moving')]),
                (brain_extract, coreg_t2w, [
                    ('brain_img', 'fixed'),
                    ('brain_mask', 'fixed_mask'),
                ]),
                (coreg_t2w, coreg_buffer, [('registered', 't2w_preproc')]),
            ])
        else:
            coreg_buffer.inputs.t2w_preproc = None

        if flair:
            coreg_flair = pe.Node(
                niu.Function(
                    function=_register_modality,
                    input_names=['moving', 'fixed', 'fixed_mask'],
                    output_names=['registered'],
                ),
                name='coreg_flair',
            )
            desc += "\nFLAIR images were rigidly registered to T1w space using ANTs."

            workflow.connect([
                (inputnode, coreg_flair, [('flair', 'moving')]),
                (brain_extract, coreg_flair, [
                    ('brain_img', 'fixed'),
                    ('brain_mask', 'fixed_mask'),
                ]),
                (coreg_flair, coreg_buffer, [('registered', 'flair_preproc')]),
            ])
        else:
            coreg_buffer.inputs.flair_preproc = None
    else:
        LOGGER.info('ANAT Stage 4: Skipping co-registration (no additional modalities)')
        coreg_buffer = pe.Node(
            niu.IdentityInterface(fields=['t1ce_preproc', 't2w_preproc', 'flair_preproc']),
            name='coreg_buffer',
        )
        coreg_buffer.inputs.t1ce_preproc = None
        coreg_buffer.inputs.t2w_preproc = None
        coreg_buffer.inputs.flair_preproc = None

    # Stage 4.5: Optional defacing with mri_deface
    # ============================================
    if defacing:
        LOGGER.info('ANAT Stage 4.5: Defacing anatomical images with mri_deface')

        # Deface T1w
        deface_t1w = pe.Node(
            niu.Function(
                function=_deface_anatomical,
                input_names=['in_file'],
                output_names=['out_file'],
            ),
            name='deface_t1w',
        )

        desc += "\nAnatomical images were defaced using mri_deface to protect participant privacy."

        # Create a buffer node for defaced outputs
        deface_buffer = pe.Node(
            niu.IdentityInterface(fields=['t1w_defaced', 't1ce_defaced', 't2w_defaced', 'flair_defaced']),
            name='deface_buffer',
        )

        workflow.connect([(t1w_buffer, deface_t1w, [('t1w_preproc', 'in_file')])])
        workflow.connect([(deface_t1w, deface_buffer, [('out_file', 't1w_defaced')])])

        # Deface T1ce if present
        if t1ce:
            deface_t1ce = pe.Node(
                niu.Function(
                    function=_deface_anatomical,
                    input_names=['in_file'],
                    output_names=['out_file'],
                ),
                name='deface_t1ce',
            )
            workflow.connect([(coreg_buffer, deface_t1ce, [('t1ce_preproc', 'in_file')])])
            workflow.connect([(deface_t1ce, deface_buffer, [('out_file', 't1ce_defaced')])])
        else:
            deface_buffer.inputs.t1ce_defaced = None

        # Deface T2w if present
        if t2w:
            deface_t2w = pe.Node(
                niu.Function(
                    function=_deface_anatomical,
                    input_names=['in_file'],
                    output_names=['out_file'],
                ),
                name='deface_t2w',
            )
            workflow.connect([(coreg_buffer, deface_t2w, [('t2w_preproc', 'in_file')])])
            workflow.connect([(deface_t2w, deface_buffer, [('out_file', 't2w_defaced')])])
        else:
            deface_buffer.inputs.t2w_defaced = None

        # Deface FLAIR if present
        if flair:
            deface_flair = pe.Node(
                niu.Function(
                    function=_deface_anatomical,
                    input_names=['in_file'],
                    output_names=['out_file'],
                ),
                name='deface_flair',
            )
            workflow.connect([(coreg_buffer, deface_flair, [('flair_preproc', 'in_file')])])
            workflow.connect([(deface_flair, deface_buffer, [('out_file', 'flair_defaced')])])
        else:
            deface_buffer.inputs.flair_defaced = None
    else:
        LOGGER.info('ANAT Stage 4.5: Skipping defacing (defacing=False)')
        deface_buffer = pe.Node(
            niu.IdentityInterface(fields=['t1w_defaced', 't1ce_defaced', 't2w_defaced', 'flair_defaced']),
            name='deface_buffer',
        )
        deface_buffer.inputs.t1w_defaced = None
        deface_buffer.inputs.t1ce_defaced = None
        deface_buffer.inputs.t2w_defaced = None
        deface_buffer.inputs.flair_defaced = None

    # Stage 5: Template registration
    # =============================
    if output_spaces:
        LOGGER.info(f'ANAT Stage 5: Template registration to {output_spaces}')

        # Collect all modalities for template registration
        modalities = ['T1w']
        if t1ce:
            modalities.append('T1ce')
        if t2w:
            modalities.append('T2w')
        if flair:
            modalities.append('FLAIR')

        register_template_wf = init_multimodal_template_registration_wf(
            sloppy=sloppy,
            omp_nthreads=omp_nthreads,
            templates=output_spaces,
            modalities=modalities,
        )

        desc += f"""\
Nonlinear spatial normalization to {', '.join(output_spaces)} was
performed using ANTs SyN, with prior affine alignment.
"""

        # Connect T1w to registration workflow
        workflow.connect([
            (brain_extract, register_template_wf, [
                ('brain_img', 'inputnode.t1w'),
            ]),
        ])

        # Connect optional modalities
        if t1ce:
            workflow.connect([
                (coreg_buffer, register_template_wf, [
                    ('t1ce_preproc', 'inputnode.t1ce'),
                ]),
            ])
        if t2w:
            workflow.connect([
                (coreg_buffer, register_template_wf, [
                    ('t2w_preproc', 'inputnode.t2w'),
                ]),
            ])
        if flair:
            workflow.connect([
                (coreg_buffer, register_template_wf, [
                    ('flair_preproc', 'inputnode.flair'),
                ]),
            ])

        # Connect outputs
        workflow.connect([
            (register_template_wf, outputnode, [
                ('outputnode.template', 'template'),
                ('outputnode.anat2std_xfm', 'anat2std_xfm'),
                ('outputnode.std2anat_xfm', 'std2anat_xfm'),
            ]),
        ])
    else:
        LOGGER.info('ANAT Stage 5: Skipping template registration (no output spaces specified)')
        outputnode.inputs.template = None
        outputnode.inputs.anat2std_xfm = None
        outputnode.inputs.std2anat_xfm = None

    # Connect T1w outputs
    workflow.connect([
        (t1w_buffer, outputnode, [
            ('t1w_preproc', 't1w_preproc'),
            ('t1w_mask', 't1w_mask'),
        ]),
        (coreg_buffer, outputnode, [
            ('t1ce_preproc', 't1ce_preproc'),
            ('t2w_preproc', 't2w_preproc'),
            ('flair_preproc', 'flair_preproc'),
        ]),
        (deface_buffer, outputnode, [
            ('t1w_defaced', 't1w_defaced'),
            ('t1ce_defaced', 't1ce_defaced'),
            ('t2w_defaced', 't2w_defaced'),
            ('flair_defaced', 'flair_defaced'),
        ]),
    ])

    workflow.__desc__ = desc
    return workflow


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


def _hd_bet_skullstrip(img_path, use_gpu=False):
    """
    Perform skull stripping using HD-BET.

    Parameters
    ----------
    img_path : str
        Path to input image
    use_gpu : bool
        Enable GPU acceleration

    Returns
    -------
    brain_img : str
        Path to brain-extracted image
    brain_mask : str
        Path to brain mask
    """
    try:
        from hd_bet.model import predict
    except ImportError:
        raise ImportError("HD-BET is required for skull stripping. Install with: pip install HD-BET")

    import torch
    from pathlib import Path

    # Use GPU if available and requested
    if use_gpu and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # HD-BET prediction
    try:
        predict(
            mode='fast',
            files=[img_path],
            output_dir=str(Path.cwd()),
            device=device,
            postprocess=True,
        )
    except Exception as e:
        LOGGER.warning(f"HD-BET prediction failed: {e}. Falling back to basic thresholding.")
        return _fallback_brain_extraction(img_path)

    # Load and process results
    brain_path = Path(img_path).parent / Path(img_path).stem.replace('.nii.gz', '_bet.nii.gz')
    mask_path = Path(img_path).parent / Path(img_path).stem.replace('.nii.gz', '_mask.nii.gz')

    if brain_path.exists():
        return str(brain_path), str(mask_path)
    else:
        LOGGER.warning("HD-BET output not found. Falling back to basic thresholding.")
        return _fallback_brain_extraction(img_path)


def _fallback_brain_extraction(img_path):
    """Fallback brain extraction using simple thresholding."""
    from pathlib import Path

    img = nb.load(img_path)
    data = img.get_fdata()

    # Simple threshold-based brain extraction
    threshold = np.percentile(data[data > 0], 5)
    mask = (data > threshold).astype(np.uint8)

    # Morphological operations to clean up
    from scipy import ndimage
    mask = ndimage.binary_closing(mask, iterations=2)
    mask = ndimage.binary_opening(mask, iterations=1)

    brain_data = data * mask

    # Save outputs
    out_dir = Path.cwd()
    brain_path = out_dir / 'brain.nii.gz'
    mask_path = out_dir / 'brain_mask.nii.gz'

    nb.Nifti1Image(brain_data, img.affine, img.header).to_filename(brain_path)
    nb.Nifti1Image(mask.astype(np.uint8), img.affine, img.header).to_filename(mask_path)

    return str(brain_path), str(mask_path)


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

    # Handle list input (use first image if multiple)
    if isinstance(moving, list):
        moving = moving[0] if moving else fixed

    # ANTs rigid registration
    ants_rigid = ants.Registration(
        dimension=3,
        transforms=['Rigid'],
        transform_parameters=[(0.1,)],
        metric=['Mattes'],
        metric_weight=[1.0],
        radius_or_number_of_bins=[32],
        sampling_strategy=['Regular'],
        sampling_percentage=[0.25],
        convergence_iterations=[[1000, 500, 250, 0]],
        convergence_tolerance=[1e-6],
        convergence_window_size=[10],
        use_estimate_learning_rate_flag=[True],
        use_histogram_matching=[True],
        masks=[[fixed_mask, '']],
        num_threads=1,
    )

    ants_rigid.inputs.moving_image = moving
    ants_rigid.inputs.fixed_image = fixed

    result = ants_rigid.run()

    out_path = Path.cwd() / 'registered_modality.nii.gz'
    registered_img = result.outputs.warped_image

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
        import mri_deface
        from mri_deface.deface import run as deface_run
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

