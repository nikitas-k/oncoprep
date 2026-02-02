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
"""Surface handling interfaces."""

import os
from pathlib import Path
from typing import Optional

import nibabel as nb
import nitransforms as nt
import numpy as np
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    InputMultiObject,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)


class _NormalizeSurfInputSpec(BaseInterfaceInputSpec):
    """Input specification for NormalizeSurf interface."""

    in_file = File(
        mandatory=True,
        exists=True,
        desc='FreeSurfer-generated GIFTI file',
    )
    transform_file = File(
        exists=True,
        desc='FSL, LTA or ITK affine transform file',
    )


class _NormalizeSurfOutputSpec(TraitedSpec):
    """Output specification for NormalizeSurf interface."""

    out_file = File(desc='output file with re-centered GIFTI coordinates')


class NormalizeSurf(SimpleInterface):
    """
    Normalize a FreeSurfer-generated GIFTI surface image.

    FreeSurfer includes volume geometry metadata that serves as an affine
    transformation for coordinates, respected by FreeSurfer tools but not
    by Connectome Workbench. This normalization removes the volume geometry
    metadata to ensure consistent interpretation across tools.

    Requires that the GIFTI surface be converted with ``mris_convert --to-scanner``,
    with which FreeSurfer applies the volume geometry. Since FreeSurfer does not
    update metadata, there is no programmatic way to detect the conversion method,
    and the user must ensure ``mris_convert --to-scanner`` was used.

    For midthickness/graymid surfaces, adds metadata entries::

        AnatomicalStructureSecondary: MidThickness
        GeometricType: Anatomical

    Intended for uniform application to surfaces from ``?h.white``/``?h.smoothwm``
    and ``?h.pial``, as well as externally-generated ``?h.midthickness``/``?h.graymid``.
    In principle, applies safely to any surface, though less relevant for non-anatomical.

    """

    input_spec = _NormalizeSurfInputSpec
    output_spec = _NormalizeSurfOutputSpec

    def _run_interface(self, runtime):
        """Run interface."""
        transform_file = self.inputs.transform_file
        if not isdefined(transform_file):
            transform_file = None
        self._results['out_file'] = normalize_surfs(
            self.inputs.in_file, transform_file, newpath=runtime.cwd
        )
        return runtime


class FixGiftiMetadataInputSpec(TraitedSpec):
    """Input specification for FixGiftiMetadata interface."""

    in_file = File(
        mandatory=True,
        exists=True,
        desc='FreeSurfer-generated GIFTI file',
    )


class FixGiftiMetadataOutputSpec(TraitedSpec):
    """Output specification for FixGiftiMetadata interface."""

    out_file = File(desc='output file with fixed metadata')


class FixGiftiMetadata(SimpleInterface):
    """
    Fix known incompatible metadata in GIFTI files.

    Currently resolves FreeSurfer setting GeometricType to 'Sphere' instead of 'Spherical'.
    This issue persists in FreeSurfer through version 7.4.0.

    """

    input_spec = FixGiftiMetadataInputSpec
    output_spec = FixGiftiMetadataOutputSpec

    def _run_interface(self, runtime):
        """Run interface."""
        self._results['out_file'] = fix_gifti_metadata(self.inputs.in_file, newpath=runtime.cwd)
        return runtime


class AggregateSurfacesInputSpec(TraitedSpec):
    """Input specification for AggregateSurfaces interface."""

    surfaces = InputMultiObject(File(exists=True), desc='Input surfaces')
    morphometrics = InputMultiObject(File(exists=True), desc='Input morphometrics')


class AggregateSurfacesOutputSpec(TraitedSpec):
    """Output specification for AggregateSurfaces interface."""

    pial = traits.List(File(), maxlen=2, desc='Pial surfaces')
    white = traits.List(File(), maxlen=2, desc='White surfaces')
    inflated = traits.List(File(), maxlen=2, desc='Inflated surfaces')
    midthickness = traits.List(File(), maxlen=2, desc='Midthickness (or graymid) surfaces')
    thickness = traits.List(File(), maxlen=2, desc='Cortical thickness maps')
    sulc = traits.List(File(), maxlen=2, desc='Sulcal depth maps')
    curv = traits.List(File(), maxlen=2, desc='Curvature maps')


class AggregateSurfaces(SimpleInterface):
    """
    Aggregate and group surfaces and morphometrics into left/right hemisphere pairs.

    Organizes surface files and morphometric maps by type and hemisphere,
    enabling efficient processing of surface data in workflows.

    """

    input_spec = AggregateSurfacesInputSpec
    output_spec = AggregateSurfacesOutputSpec

    def _run_interface(self, runtime):
        """Run interface."""
        import re
        from collections import defaultdict

        container = defaultdict(list)
        inputs = (self.inputs.surfaces or []) + (self.inputs.morphometrics or [])
        findre = re.compile(
            r'(?:^|[^d])(?P<name>white|pial|inflated|midthickness|thickness|sulc|curv)'
        )
        for surface in sorted(inputs, key=os.path.basename):
            match = findre.search(os.path.basename(surface))
            if match:
                container[match.group('name')].append(surface)
        for name, files in container.items():
            self._results[name] = files
        return runtime


class MakeRibbonInputSpec(TraitedSpec):
    """Input specification for MakeRibbon interface."""

    white_distvols = traits.List(
        File(exists=True),
        minlen=2,
        maxlen=2,
        desc='White matter distance volumes',
    )
    pial_distvols = traits.List(
        File(exists=True),
        minlen=2,
        maxlen=2,
        desc='Pial matter distance volumes',
    )


class MakeRibbonOutputSpec(TraitedSpec):
    """Output specification for MakeRibbon interface."""

    ribbon = File(desc='Binary ribbon mask')


class MakeRibbon(SimpleInterface):
    """
    Create a binary ribbon mask from white and pial distance volumes.

    The ribbon is computed as the intersection of positive white matter
    distances and negative pial surface distances for both hemispheres.

    """

    input_spec = MakeRibbonInputSpec
    output_spec = MakeRibbonOutputSpec

    def _run_interface(self, runtime):
        """Run interface."""
        self._results['ribbon'] = make_ribbon(
            self.inputs.white_distvols, self.inputs.pial_distvols, newpath=runtime.cwd
        )
        return runtime


def normalize_surfs(in_file: str, transform_file: Optional[str], newpath: Optional[str] = None) -> str:
    """
    Update GIFTI metadata and apply rigid coordinate correction.

    This function removes volume geometry metadata that FreeSurfer includes.
    Connectome Workbench does not respect this metadata, while FreeSurfer will
    apply it when converting with ``mris_convert --to-scanner`` and then again when
    reading with ``freeview``.

    For midthickness surfaces, add MidThickness metadata.

    Parameters
    ----------
    in_file : str
        Input FreeSurfer-generated GIFTI file
    transform_file : str or None
        Optional FSL (.mat), LTA (.lta), or ITK (.txt) affine transform file
    newpath : str, optional
        Output directory (defaults to current working directory)

    Returns
    -------
    str
        Path to normalized GIFTI file

    """
    img = nb.load(in_file)
    if transform_file is None:
        transform = nt.linear.Affine()
    else:
        xfm_fmt = {
            '.txt': 'itk',
            '.mat': 'fsl',
            '.lta': 'fs',
        }[Path(transform_file).suffix]
        transform = nt.linear.load(transform_file, fmt=xfm_fmt)
    pointset = img.get_arrays_from_intent('NIFTI_INTENT_POINTSET')[0]

    if not np.allclose(transform.matrix, np.eye(4)):
        pointset.data = transform.map(pointset.data, inverse=True)

    fname = os.path.basename(in_file)
    if 'graymid' in fname.lower():
        # Rename graymid to midthickness
        fname = fname.replace('graymid', 'midthickness')
    if 'midthickness' in fname.lower():
        pointset.meta.setdefault('AnatomicalStructureSecondary', 'MidThickness')
        pointset.meta.setdefault('GeometricType', 'Anatomical')

    # FreeSurfer incorrectly uses "Sphere" for spherical surfaces
    if pointset.meta.get('GeometricType') == 'Sphere':
        pointset.meta['GeometricType'] = 'Spherical'
    else:
        # mris_convert --to-scanner removes VolGeom transform from coordinates,
        # but not metadata. Following HCP pipelines, we only adjust coordinates
        # for anatomical surfaces. For spherical surfaces, metadata is left intact.
        for XYZC in 'XYZC':
            for RAS in 'RAS':
                pointset.meta.pop(f'VolGeom{XYZC}_{RAS}', None)

    if newpath is None:
        newpath = os.getcwd()
    out_file = os.path.join(newpath, fname)
    img.to_filename(out_file)
    return out_file


def fix_gifti_metadata(in_file: str, newpath: Optional[str] = None) -> str:
    """
    Fix known incompatible metadata in GIFTI files.

    Currently resolves FreeSurfer setting GeometricType to 'Sphere' instead of 'Spherical'.
    This is not fixed as of FreeSurfer 7.4.0
    (https://github.com/freesurfer/freesurfer/pull/1112)

    Parameters
    ----------
    in_file : str
        Input GIFTI file with incompatible metadata
    newpath : str, optional
        Output directory (defaults to current working directory)

    Returns
    -------
    str
        Path to fixed GIFTI file

    """
    img = nb.GiftiImage.from_filename(in_file)
    pointset = img.get_arrays_from_intent('NIFTI_INTENT_POINTSET')[0]

    if pointset.meta.get('GeometricType') == 'Sphere':
        pointset.meta['GeometricType'] = 'Spherical'

    if newpath is None:
        newpath = os.getcwd()
    out_file = os.path.join(newpath, os.path.basename(in_file))
    img.to_filename(out_file)
    return out_file


def make_ribbon(
    white_distvols: list,
    pial_distvols: list,
    newpath: Optional[str] = None,
) -> str:
    """
    Create binary ribbon mask from white and pial distance volumes.

    The ribbon is defined as regions where white matter distances are positive
    and pial surface distances are negative (interior to pial surface).

    Parameters
    ----------
    white_distvols : list of str
        White matter distance volume files (length 2)
    pial_distvols : list of str
        Pial surface distance volume files (length 2)
    newpath : str, optional
        Output directory (defaults to current working directory)

    Returns
    -------
    str
        Path to binary ribbon mask (nii.gz format)

    """
    base_img = nb.load(white_distvols[0])
    header = base_img.header
    header.set_data_dtype('uint8')

    ribbons = [
        (np.array(nb.load(white).dataobj) > 0) & (np.array(nb.load(pial).dataobj) < 0)
        for white, pial in zip(white_distvols, pial_distvols, strict=True)
    ]

    if newpath is None:
        newpath = os.getcwd()
    out_file = os.path.join(newpath, 'ribbon.nii.gz')

    ribbon_data = ribbons[0] | ribbons[1]
    ribbon = base_img.__class__(ribbon_data, base_img.affine, header)
    ribbon.to_filename(out_file)
    return out_file


__all__ = [
    'NormalizeSurf',
    'FixGiftiMetadata',
    'AggregateSurfaces',
    'MakeRibbon',
    'normalize_surfs',
    'fix_gifti_metadata',
    'make_ribbon',
]
