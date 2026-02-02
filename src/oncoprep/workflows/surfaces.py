# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2021 The NiPreps Developers <nipreps@gmail.com>
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
"""
FreeSurfer surface processing workflows for OncoPrep.

Handles extraction and conversion of FreeSurfer-generated surface files
to BIDS-compliant GIFTI format.
"""

import typing as ty
from pathlib import Path

from nipype.interfaces import freesurfer as fs
from nipype.interfaces import io as nio
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine import Workflow, tag
from niworkflows.interfaces.patches import FreeSurferSource

if ty.TYPE_CHECKING:
    pass


@tag('anat.surfaces')
def init_gifti_surfaces_wf(
    *,
    surfaces: ty.List[str] = ('pial', 'white', 'inflated'),
    to_scanner: bool = True,
    name: str = 'gifti_surfaces_wf',
) -> Workflow:
    """
    Prepare GIFTI surfaces from FreeSurfer subjects directory.

    Converts FreeSurfer surface files to GIFTI format for BIDS compliance.
    Default surfaces are pial, white matter, and inflated surfaces.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from oncoprep.workflows.surfaces import init_gifti_surfaces_wf
            wf = init_gifti_surfaces_wf()

    Parameters
    ----------
    surfaces : :class:`list` of :class:`str`
        Surface names to extract (e.g., 'pial', 'white', 'inflated')
    to_scanner : :class:`bool`
        Convert coordinates to scanner space (default: True)
    name : :class:`str`
        Workflow name (default: 'gifti_surfaces_wf')

    Inputs
    ------
    subjects_dir
        FreeSurfer SUBJECTS_DIR
    subject_id
        FreeSurfer subject ID

    Outputs
    -------
    surfaces
        List of GIFTI surfaces for all requested surfaces
    ``<surface>``
        Left and right GIFTI for each surface passed to ``surfaces``

    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['subjects_dir', 'subject_id']),
        name='inputnode',
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['surfaces', *surfaces]),
        name='outputnode',
    )

    # Get surfaces from FreeSurfer directory
    get_surfaces = pe.Node(
        niu.Function(function=_get_surfaces, output_names=surfaces),
        name='get_surfaces',
    )
    get_surfaces.inputs.surfaces = surfaces

    # Merge surfaces for batch processing
    surface_list = pe.Node(
        niu.Merge(len(surfaces), ravel_inputs=True),
        name='surface_list',
        run_without_submitting=True,
    )

    # Convert FreeSurfer surfaces to GIFTI
    fs2gii = pe.MapNode(
        fs.MRIsConvert(out_datatype='gii', to_scanner=to_scanner),
        iterfield='in_file',
        name='fs2gii',
    )

    # Split surfaces by type for output
    surface_groups = pe.Node(
        niu.Split(splits=[2] * len(surfaces)),
        name='surface_groups',
        run_without_submitting=True,
    )

    # fmt:off
    workflow.connect([
        # Get surfaces from FreeSurfer
        (inputnode, get_surfaces, [
            ('subjects_dir', 'subjects_dir'),
            ('subject_id', 'subject_id'),
        ]),
        # Merge all surfaces for batch conversion
        (get_surfaces, surface_list, [
            (surf, f'in{i}') for i, surf in enumerate(surfaces, start=1)
        ]),
        # Convert to GIFTI
        (surface_list, fs2gii, [('out', 'in_file')]),
        # Output all surfaces as list
        (fs2gii, outputnode, [('converted', 'surfaces')]),
        # Split into individual surface types
        (fs2gii, surface_groups, [('converted', 'inlist')]),
        # Output individual surface types
        (surface_groups, outputnode, [
            (f'out{i}', surf) for i, surf in enumerate(surfaces, start=1)
        ]),
    ])
    # fmt:on

    return workflow


@tag('anat.morphometrics')
def init_gifti_morphometrics_wf(
    *,
    morphometrics: ty.List[str] = ('thickness', 'curv', 'sulc'),
    name: str = 'gifti_morphometrics_wf',
) -> Workflow:
    """
    Prepare GIFTI shape files from FreeSurfer morphometrics.

    Extracts morphometric data (thickness, curvature, sulcal depth) from
    FreeSurfer and converts to GIFTI format.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from oncoprep.workflows.surfaces import init_gifti_morphometrics_wf
            wf = init_gifti_morphometrics_wf()

    Parameters
    ----------
    morphometrics : :class:`list` of :class:`str`
        Morphometric names (e.g., 'thickness', 'curv', 'sulc')
    name : :class:`str`
        Workflow name (default: 'gifti_morphometrics_wf')

    Inputs
    ------
    subjects_dir
        FreeSurfer SUBJECTS_DIR
    subject_id
        FreeSurfer subject ID

    Outputs
    -------
    morphometrics
        List of GIFTI shape files for all requested morphometrics
    ``<morphometric>``
        Left and right GIFTIs for each morphometry type

    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['subjects_dir', 'subject_id']),
        name='inputnode',
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['morphometrics', *morphometrics]),
        name='outputnode',
    )

    # Get FreeSurfer subject data
    get_subject = pe.Node(FreeSurferSource(), name='get_subject')

    # Merge morphometrics for batch processing
    morphometry_list = pe.Node(
        niu.Merge(len(morphometrics), ravel_inputs=True),
        name='morphometry_list',
        run_without_submitting=True,
    )

    # Convert to GIFTI shape files using mris_convert
    morphs2gii = pe.MapNode(
        fs.MRIsConvert(out_datatype='gii'),
        iterfield='in_file',
        name='morphs2gii',
    )

    # Split morphometrics by type for output
    morph_groups = pe.Node(
        niu.Split(splits=[2] * len(morphometrics)),
        name='morph_groups',
        run_without_submitting=True,
    )

    # fmt:off
    workflow.connect([
        # Get FreeSurfer data
        (inputnode, get_subject, [
            ('subjects_dir', 'subjects_dir'),
            ('subject_id', 'subject_id'),
        ]),
        # Get morphometric files
        (get_subject, morphometry_list, [
            (morph, f'in{i}') for i, morph in enumerate(morphometrics, start=1)
        ]),
        # Convert to GIFTI
        (morphometry_list, morphs2gii, [('out', 'in_file')]),
        # Output all morphometrics as list
        (morphs2gii, outputnode, [('converted', 'morphometrics')]),
        # Split into individual types
        (morphs2gii, morph_groups, [('converted', 'inlist')]),
        # Output individual morphometric types
        (morph_groups, outputnode, [
            (f'out{i}', surf) for i, surf in enumerate(morphometrics, start=1)
        ]),
    ])
    # fmt:on

    return workflow


@tag('anat.surfaces-ds')
def init_surface_datasink_wf(
    *,
    output_dir: str,
    name: str = 'surface_datasink_wf',
) -> Workflow:
    """
    Save GIFTI surfaces and morphometrics to BIDS derivatives.

    Writes surface files to standardized derivative paths with proper
    BIDS entity naming.

    Parameters
    ----------
    output_dir : :obj:`str`
        Directory in which to save derivatives
    name : :obj:`str`
        Workflow name (default: 'surface_datasink_wf')

    Inputs
    ------
    source_file
        Source anatomical image for reference
    surfaces
        List of GIFTI surface files
    morphometrics
        List of GIFTI morphometric files
    subject_label
        Subject identifier
    session_label
        Session identifier (optional)

    Outputs
    -------
    surfaces_dir
        Directory containing saved surface files

    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'source_file',
                'surfaces',
                'morphometrics',
                'subject_label',
                'session_label',
            ]
        ),
        name='inputnode',
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['surfaces_dir']),
        name='outputnode',
    )

    # Save surfaces
    ds_surfaces = pe.MapNode(
        nio.DataSink(parameterization=False),
        iterfield=['in_file'],
        name='ds_surfaces',
    )
    ds_surfaces.inputs.base_directory = output_dir

    # Save morphometrics
    ds_morphs = pe.MapNode(
        nio.DataSink(parameterization=False),
        iterfield=['in_file'],
        name='ds_morphometrics',
    )
    ds_morphs.inputs.base_directory = output_dir

    # Get output directory
    get_outdir = pe.Node(
        niu.Function(
            function=_get_surface_outdir,
            output_names=['surfaces_dir'],
        ),
        name='get_outdir',
    )

    # fmt:off
    workflow.connect([
        (inputnode, ds_surfaces, [
            ('surfaces', 'in_file'),
        ]),
        (inputnode, ds_morphs, [
            ('morphometrics', 'in_file'),
        ]),
        (inputnode, get_outdir, [
            ('source_file', 'source_file'),
            ('subject_label', 'subject_label'),
            ('session_label', 'session_label'),
        ]),
        (get_outdir, outputnode, [('surfaces_dir', 'surfaces_dir')]),
    ])
    # fmt:on

    return workflow


def _get_surfaces(
    subjects_dir: str,
    subject_id: str,
    surfaces: ty.List[str],
) -> ty.Union[ty.List[str], ty.Tuple[ty.List[str], ...]]:
    """
    Get FreeSurfer surface files for a subject.

    Parameters
    ----------
    subjects_dir : :obj:`str`
        FreeSurfer SUBJECTS_DIR
    subject_id : :obj:`str`
        FreeSurfer subject ID
    surfaces : :class:`list` of :class:`str`
        Surface names to fetch

    Returns
    -------
    surfaces : :class:`list` of :class:`str` or tuple of lists
        Sorted list of surface files for each requested surface type.
        Returns tuple if multiple surface types requested, else list.

    """
    surf_dir = Path(subjects_dir) / subject_id / 'surf'

    all_surfs = {}
    for surface in surfaces:
        # Handle underscore to dot conversion (e.g., mid_thickness -> mid.thickness)
        surface_pattern = surface.replace('_', '.')
        surfs = sorted(str(fn) for fn in surf_dir.glob(f'[lr]h.{surface_pattern}'))
        if not surfs:
            raise FileNotFoundError(
                f'No surfaces matching "{surface}" found in {surf_dir}'
            )
        all_surfs[surface] = surfs

    ret = tuple(all_surfs[surface] for surface in surfaces)
    return ret if len(ret) > 1 else ret[0]


def _get_surface_outdir(
    source_file: str,
    subject_label: str,
    session_label: ty.Optional[str] = None,
) -> str:
    """
    Determine output directory for surface files.

    Parameters
    ----------
    source_file : :obj:`str`
        Source anatomical image
    subject_label : :obj:`str`
        Subject identifier
    session_label : :obj:`str` or None
        Session identifier

    Returns
    -------
    outdir : :obj:`str`
        Output directory path

    """
    from pathlib import Path

    # Get derivatives root (parent of derivatives)
    source_path = Path(source_file)
    bids_root = source_path.parent.parent.parent  # ../derivatives/sub-XX/ses-XX/anat

    # Build surface output path
    if session_label:
        surf_dir = (
            bids_root / 'derivatives' / f'sub-{subject_label}' / f'ses-{session_label}' / 'surf'
        )
    else:
        surf_dir = bids_root / 'derivatives' / f'sub-{subject_label}' / 'surf'

    surf_dir.mkdir(parents=True, exist_ok=True)
    return str(surf_dir)
