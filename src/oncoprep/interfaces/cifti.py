# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""CIFTI-2 image generation interfaces for OncoPrep."""

import json
from pathlib import Path
from typing import Optional, Tuple

import nibabel as nb
import numpy as np
from nibabel import cifti2 as ci
from nipype.interfaces.base import File, SimpleInterface, TraitedSpec, traits
from templateflow import api as tf


class _GenerateDScalarInputSpec(TraitedSpec):
    """Input specification for GenerateDScalar interface."""

    surface_target = traits.Enum(
        'fsLR',
        usedefault=True,
        desc='CIFTI surface target space',
    )
    grayordinates = traits.Enum(
        '91k',
        '170k',
        usedefault=True,
        desc='Final CIFTI grayordinates',
    )
    scalar_surfs = traits.List(
        File(exists=True),
        mandatory=True,
        desc='list of surface BOLD GIFTI files (length 2 with order [L,R])',
    )
    scalar_name = traits.Str(
        mandatory=True,
        desc='Name of scalar',
    )


class _GenerateDScalarOutputSpec(TraitedSpec):
    """Output specification for GenerateDScalar interface."""

    out_file = File(desc='generated CIFTI file')
    out_metadata = File(desc='CIFTI metadata JSON')


class GenerateDScalar(SimpleInterface):
    """
    Generate a HCP-style CIFTI-2 dscalar image from scalar surface files.

    This interface combines left and right hemisphere scalar surface files
    (in GIFTI format) into a single CIFTI-2 dscalar file with proper
    spatial structure and metadata.

    Inputs
    ------
    surface_target : str
        CIFTI surface target space (default: 'fsLR')
    grayordinates : {'91k', '170k'}
        Final CIFTI grayordinates density (default: '91k')
    scalar_surfs : list of str
        List of surface BOLD GIFTI files (length 2 with order [L, R])
    scalar_name : str
        Name of scalar map in CIFTI file

    Outputs
    -------
    out_file : str
        Path to generated CIFTI dscalar file
    out_metadata : str
        Path to CIFTI metadata JSON file

    """

    input_spec = _GenerateDScalarInputSpec
    output_spec = _GenerateDScalarOutputSpec

    def _run_interface(self, runtime):
        """Run the interface."""
        surface_labels, metadata = _prepare_cifti(self.inputs.grayordinates)
        self._results['out_file'] = _create_cifti_image(
            self.inputs.scalar_surfs,
            surface_labels,
            self.inputs.scalar_name,
            metadata,
        )
        metadata_file = Path('dscalar.json').absolute()
        metadata_file.write_text(json.dumps(metadata, indent=2))
        self._results['out_metadata'] = str(metadata_file)
        return runtime


def _prepare_cifti(grayordinates: str) -> Tuple:
    """
    Fetch the required templates needed for CIFTI-2 generation.

    Based on input surface density, retrieves the appropriate surface
    templates and metadata from templateflow.

    Parameters
    ----------
    grayordinates : {'91k', '170k'}
        Total CIFTI grayordinates

    Returns
    -------
    surface_labels : list of str
        Surface label files for vertex inclusion/exclusion (L, R)
    metadata : dict
        Dictionary with BIDS metadata

    Examples
    --------
    >>> surface_labels, metadata = _prepare_cifti('91k')
    >>> len(surface_labels)
    2
    >>> 'Density' in metadata
    True

    """
    grayord_key = {
        '91k': {
            'surface-den': '32k',
            'tf-res': '02',
            'grayords': '91,282',
            'res-mm': '2mm',
        },
        '170k': {
            'surface-den': '59k',
            'tf-res': '06',
            'grayords': '170,494',
            'res-mm': '1.6mm',
        },
    }
    if grayordinates not in grayord_key:
        raise NotImplementedError(f'Grayordinates {grayordinates} is not supported.')

    total_grayords = grayord_key[grayordinates]['grayords']
    res_mm = grayord_key[grayordinates]['res-mm']
    surface_density = grayord_key[grayordinates]['surface-den']

    # Fetch templates from templateflow
    surface_labels = [
        str(
            tf.get(
                'fsLR',
                density=surface_density,
                hemi=hemi,
                desc='nomedialwall',
                suffix='dparc',
                raise_empty=True,
            )
        )
        for hemi in ('L', 'R')
    ]

    tf_url = 'https://templateflow.s3.amazonaws.com'
    surfaces_url = (
        f'{tf_url}/tpl-fsLR/tpl-fsLR_den-{surface_density}_hemi-%s_midthickness.surf.gii'
    )
    metadata = {
        'Density': (
            f'{total_grayords} grayordinates corresponding to all of the grey matter sampled at a '
            f'{res_mm} average vertex spacing on the surface'
        ),
        'SpatialReference': {
            'CIFTI_STRUCTURE_CORTEX_LEFT': surfaces_url % 'L',
            'CIFTI_STRUCTURE_CORTEX_RIGHT': surfaces_url % 'R',
        },
    }
    return surface_labels, metadata


def _create_cifti_image(
    scalar_surfs: Tuple[str, str],
    surface_labels: Tuple[str, str],
    scalar_name: str,
    metadata: Optional[dict] = None,
) -> Path:
    """
    Generate CIFTI-2 dscalar image in target space.

    Combines scalar surface data from left and right hemispheres into
    a single CIFTI-2 dscalar file with proper brain model axes and metadata.

    Parameters
    ----------
    scalar_surfs : tuple of str
        Surface scalar files (L, R)
    surface_labels : tuple of str
        Surface label files used to remove medial wall (L, R)
    scalar_name : str
        Name to apply to scalar map
    metadata : dict, optional
        Metadata to include in CIFTI header

    Returns
    -------
    out_file : Path
        Path to saved CIFTI dscalar file

    """
    brainmodels = []
    arrays = []

    # Process each hemisphere
    for idx, hemi in enumerate(('left', 'right')):
        # Load surface labels and create mask
        labels = nb.load(surface_labels[idx])
        mask = np.bool_(labels.darrays[0].data)

        struct = f'cortex_{hemi}'
        brainmodels.append(
            ci.BrainModelAxis(
                struct,
                vertex=np.nonzero(mask)[0],
                nvertices={struct: len(mask)},
            )
        )

        # Load and mask scalar data
        morph_scalar = nb.load(scalar_surfs[idx])
        arrays.append(morph_scalar.darrays[0].data[mask].astype('float32'))

    # Provide default metadata if none given
    if not metadata:
        metadata = {
            'surface': 'fsLR',
        }

    # Generate CIFTI header
    hdr = ci.Cifti2Header.from_axes(
        (ci.ScalarAxis([scalar_name]), brainmodels[0] + brainmodels[1])
    )
    hdr.matrix.metadata = ci.Cifti2MetaData(metadata)

    # Create and save CIFTI image
    img = ci.Cifti2Image(
        dataobj=np.atleast_2d(np.concatenate(arrays)),
        header=hdr,
    )
    img.nifti_header.set_intent('NIFTI_INTENT_CONNECTIVITY_DENSE_SCALARS')

    # Generate output filename
    stem = Path(scalar_surfs[0]).name.split('.')[0]
    cifti_stem = '_'.join(ent for ent in stem.split('_') if not ent.startswith('hemi-'))
    out_file = Path.cwd() / f'{cifti_stem}.dscalar.nii'
    img.to_filename(out_file)

    return out_file


__all__ = ['GenerateDScalar']
