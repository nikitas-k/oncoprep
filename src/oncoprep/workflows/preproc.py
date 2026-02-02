# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""OncoPrep preprocessing workflow orchestration using nipreps patterns."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from nipype import logging, Workflow
from nipype.interfaces.utility import IdentityInterface
from nipype.pipeline import engine as pe

from oncoprep.workflows.anatomical import init_anat_preproc_wf

LOGGER = logging.getLogger('nipype.workflow')


def build_preproc_workflow(
    bids_dir: Path = None,
    output_dir: Path = None,
    participant_label: Optional[List[str]] = None,
    session_label: Optional[List[str]] = None,
    nprocs: int = 1,
    omp_nthreads: int = 1,
    mem_gb: Optional[float] = None,
    skull_strip_template: str = "OASIS30ANTs",
    skull_strip_fixed_seed: bool = False,
    skull_strip_mode: str = "auto",
    longitudinal: bool = False,
    output_spaces: Optional[List[str]] = None,
    use_gpu: bool = False,
    deface: bool = False,
    skip_segmentation: bool = True,
    sloppy: bool = False,
    name: str = "oncoprep_preproc",
) -> Workflow:
    """Build OncoPrep preprocessing workflow using anatomical modules.

    This workflow orchestrates BraTS-compatible preprocessing with nipreps patterns:
    
    1. **T1w Conformance**: Average multiple T1w images if present
    2. **N4 Bias Correction**: Intensity non-uniformity correction (ANTs)
    3. **Brain Extraction**: HD-BET with optional GPU acceleration
    4. **Multi-modal Co-registration**: Align T1ce, T2, FLAIR to T1w space (ANTs Rigid)
    5. **Template Registration**: Nonlinear registration to standard space (ANTs SyN)

    Parameters
    ----------
    bids_dir : Path
        Root directory of BIDS dataset
    output_dir : Path
        Output directory for results
    participant_label : list[str] | None
        List of participant IDs to process
    session_label : list[str] | None
        List of session IDs to process
    nprocs : int
        Number of parallel processes
    omp_nthreads : int
        Number of OpenMP threads per process
    mem_gb : float | None
        Memory limit in GB
    skull_strip_template : str
        Template for skull stripping (default: OASIS30ANTs)
    skull_strip_fixed_seed : bool
        Use fixed random seed for reproducibility
    skull_strip_mode : str
        Skull stripping mode: 'auto', 'skip', or 'force' (default: auto)
    longitudinal : bool
        Treat as longitudinal dataset (default: False)
    output_spaces : list[str] | None
        Target template spaces (default: ['MNI152NLin2009cAsym'])
    use_gpu : bool
        Enable GPU acceleration for HD-BET (default: False)
    deface : bool
        Apply mri_deface to remove facial features for privacy (default: False)
    skip_segmentation : bool
        Skip tumor segmentation (default: True)
    sloppy : bool
        Use faster settings for testing (default: False)
    name : str
        Workflow name (default: oncoprep_preproc)

    Returns
    -------
    Workflow
        Nipype workflow object for anatomical preprocessing

    Workflow Graph
    ---------------
    inputnode → anat_preproc_wf → outputnode

    Where anat_preproc_wf includes:
    - T1w reference creation (averaging/conformance)
    - N4 bias field correction
    - HD-BET brain extraction
    - ANTs rigid co-registration of modalities to T1w
    - ANTs SyN nonlinear template registration
    """
    if output_spaces is None:
        output_spaces = ["MNI152NLin2009cAsym"]

    # Get BIDS files for this participant first
    from bids.layout import BIDSLayout
    layout = BIDSLayout(str(bids_dir), validate=False)
    
    t1w_files = layout.get(
        subject=participant_label[0] if participant_label else None,
        session=session_label[0] if session_label else None,
        suffix='T1w',
        extension='nii.gz',
        return_type='filename',
    )
    
    t1ce_files = layout.get(
        subject=participant_label[0] if participant_label else None,
        session=session_label[0] if session_label else None,
        suffix='T1ce',
        extension='nii.gz',
        return_type='filename',
    )
    
    t2w_files = layout.get(
        subject=participant_label[0] if participant_label else None,
        session=session_label[0] if session_label else None,
        suffix='T2w',
        extension='nii.gz',
        return_type='filename',
    )
    
    flair_files = layout.get(
        subject=participant_label[0] if participant_label else None,
        session=session_label[0] if session_label else None,
        suffix='FLAIR',
        extension='nii.gz',
        return_type='filename',
    )

    workflow = Workflow(name=name)
    if output_dir:
        workflow.base_dir = str(output_dir)

    # Convert single-file lists to strings for proper handling
    # The anatomical workflow expects either a single file string or a list of files for averaging
    t1w_input = t1w_files[0] if (isinstance(t1w_files, list) and len(t1w_files) == 1) else t1w_files
    t1ce_input = t1ce_files[0] if (isinstance(t1ce_files, list) and len(t1ce_files) == 1) else t1ce_files
    t2w_input = t2w_files[0] if (isinstance(t2w_files, list) and len(t2w_files) == 1) else t2w_files
    flair_input = flair_files[0] if (isinstance(flair_files, list) and len(flair_files) == 1) else flair_files

    # Create nodes to provide file paths to the anatomical workflow
    t1w_source = pe.Node(
        IdentityInterface(fields=['t1w']),
        name='t1w_source',
    )
    t1w_source.inputs.t1w = t1w_input if t1w_input else []

    t1ce_source = pe.Node(
        IdentityInterface(fields=['t1ce']),
        name='t1ce_source',
    )
    t1ce_source.inputs.t1ce = t1ce_input if t1ce_input else []

    t2w_source = pe.Node(
        IdentityInterface(fields=['t2w']),
        name='t2w_source',
    )
    t2w_source.inputs.t2w = t2w_input if t2w_input else []

    flair_source = pe.Node(
        IdentityInterface(fields=['flair']),
        name='flair_source',
    )
    flair_source.inputs.flair = flair_input if flair_input else []

    # Main anatomical preprocessing workflow with integrated registration
    # Now built with actual file lists from BIDS
    anat_preproc_wf = init_anat_preproc_wf(
        t1w=t1w_files,
        t1ce=t1ce_files,
        t2w=t2w_files,
        flair=flair_files,
        output_spaces=output_spaces,
        skull_strip_template=skull_strip_template,
        omp_nthreads=omp_nthreads,
        use_gpu=use_gpu,
        defacing=deface,
        sloppy=sloppy,
    )

    # Output node
    outputnode = pe.Node(
        IdentityInterface(
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

    # Connect workflow - feed file paths from sources to anatomical workflow
    workflow.connect([
        (t1w_source, anat_preproc_wf, [
            ('t1w', 'inputnode.t1w'),
        ]),
        (t1ce_source, anat_preproc_wf, [
            ('t1ce', 'inputnode.t1ce'),
        ]),
        (t2w_source, anat_preproc_wf, [
            ('t2w', 'inputnode.t2w'),
        ]),
        (flair_source, anat_preproc_wf, [
            ('flair', 'inputnode.flair'),
        ]),
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

    # Add workflow description
    workflow.__desc__ = anat_preproc_wf.__desc__

    LOGGER.info(
        "Built OncoPrep preprocessing workflow: %s "
        "(participants: %s, sessions: %s, spaces: %s, GPU: %s)",
        name,
        participant_label or "all",
        session_label or "all",
        output_spaces,
        use_gpu,
    )

    return workflow

