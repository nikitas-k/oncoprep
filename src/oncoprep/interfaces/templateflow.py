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
"""Interfaces to get templates from TemplateFlow."""

import logging
from typing import Optional

from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    InputMultiObject,
    SimpleInterface,
    TraitedSpec,
    Undefined,
    isdefined,
    traits,
)
from templateflow import api as tf

LOGGER = logging.getLogger('nipype.interface')


class _TemplateFlowSelectInputSpec(BaseInterfaceInputSpec):
    """Input specification for TemplateFlowSelect interface."""

    template = traits.Str(
        'MNI152NLin2009cAsym',
        mandatory=True,
        desc='Template ID',
    )
    atlas = InputMultiObject(traits.Str, desc='Specify an atlas')
    cohort = InputMultiObject(
        traits.Either(traits.Str, traits.Int),
        desc='Specify a cohort',
    )
    resolution = InputMultiObject(traits.Int, desc='Specify a template resolution index')
    template_spec = traits.Dict(
        traits.Str,
        value={'atlas': None, 'cohort': None},
        usedefault=True,
        desc='Template specifications',
    )


class _TemplateFlowSelectOutputSpec(TraitedSpec):
    """Output specification for TemplateFlowSelect interface."""

    t1w_file = File(exists=True, desc='T1w template')
    brain_mask = File(exists=True, desc="Template's brain mask")
    t2w_file = File(desc='T2w template')
    t1ce_file = File(desc='T1ce template (if available)')
    flair_file = File(desc='FLAIR template (if available)')


class TemplateFlowSelect(SimpleInterface):
    """
    Select TemplateFlow elements.

    Retrieves template images and masks from TemplateFlow with support for
    multiple template spaces, resolutions, atlases, and cohorts.

    Examples
    --------
    >>> from oncoprep.interfaces import TemplateFlowSelect
    >>> select = TemplateFlowSelect(resolution=1)
    >>> select.inputs.template = 'MNI152NLin2009cAsym'
    >>> result = select.run()
    >>> result.outputs.t1w_file  # doctest: +ELLIPSIS
    '.../tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz'

    >>> select = TemplateFlowSelect()
    >>> select.inputs.template = 'MNIPediatricAsym'
    >>> select.inputs.template_spec = {'cohort': 5, 'resolution': 1}
    >>> result = select.run()
    >>> result.outputs.t1w_file  # doctest: +ELLIPSIS
    '.../tpl-MNIPediatricAsym_cohort-5_res-1_T1w.nii.gz'

    """

    input_spec = _TemplateFlowSelectInputSpec
    output_spec = _TemplateFlowSelectOutputSpec

    def _run_interface(self, runtime):
        """Run interface."""
        specs = self.inputs.template_spec
        if isdefined(self.inputs.resolution):
            specs['resolution'] = self.inputs.resolution
        if isdefined(self.inputs.atlas):
            specs['atlas'] = self.inputs.atlas
        if isdefined(self.inputs.cohort):
            specs['cohort'] = self.inputs.cohort

        files = fetch_template_files(self.inputs.template, specs)
        self._results['t1w_file'] = files['t1w']
        self._results['brain_mask'] = files['mask']
        self._results['t2w_file'] = files['t2w']
        self._results['t1ce_file'] = files.get('t1ce', Undefined)
        self._results['flair_file'] = files.get('flair', Undefined)
        return runtime


class _TemplateDescInputSpec(BaseInterfaceInputSpec):
    """Input specification for TemplateDesc interface."""

    template = traits.Str(
        mandatory=True,
        desc='univocal template identifier',
    )


class _TemplateDescOutputSpec(TraitedSpec):
    """Output specification for TemplateDesc interface."""

    name = traits.Str(desc='template identifier')
    spec = traits.Dict(desc='template arguments')


class TemplateDesc(SimpleInterface):
    """
    Select template description and name pairs.

    Parses template identifiers and specifications to ensure proper functioning
    with iterables and JoinNodes. Supports colon-separated template specs.

    Examples
    --------
    >>> from oncoprep.interfaces import TemplateDesc
    >>> select = TemplateDesc(template='MNI152NLin2009cAsym')
    >>> result = select.run()
    >>> result.outputs.name
    'MNI152NLin2009cAsym'
    >>> result.outputs.spec
    {}

    >>> select = TemplateDesc(template='MNIPediatricAsym:cohort-2')
    >>> result = select.run()
    >>> result.outputs.name
    'MNIPediatricAsym'
    >>> result.outputs.spec
    {'cohort': '2'}

    """

    input_spec = _TemplateDescInputSpec
    output_spec = _TemplateDescOutputSpec

    def _run_interface(self, runtime):
        """Run interface."""
        _split = self.inputs.template.split(':')
        self._results['name'] = _split[0]

        self._results['spec'] = {}
        if len(_split) > 1:
            for desc in _split[1:]:
                descsplit = desc.split('-')
                self._results['spec'][descsplit[0]] = descsplit[1]
        return runtime


def fetch_template_files(
    template: str,
    specs: Optional[dict] = None,
    sloppy: bool = False,
) -> dict:
    """
    Fetch template files from TemplateFlow.

    Parameters
    ----------
    template : str
        Template identifier
    specs : dict, optional
        Template specifications (atlas, cohort, resolution, etc.)
    sloppy : bool, optional
        Use lower resolution (2) if specific resolution unavailable (default: False)

    Returns
    -------
    dict
        Dictionary with keys: 't1w' (T1w template), 'mask' (brain mask), 't2w' (T2w template)

    """
    if specs is None:
        specs = {}

    name = template.strip(':').split(':', 1)
    if len(name) > 1:
        specs.update(
            {
                k: v
                for modifier in name[1].split(':')
                for k, v in [tuple(modifier.split('-'))]
                if k not in specs
            }
        )

    if res := specs.pop('res', None):
        if res != 'native':
            specs['resolution'] = res

    if not specs.get('resolution'):
        specs['resolution'] = 2 if sloppy else 1

    if specs.get('resolution') and not isinstance(specs['resolution'], list):
        specs['resolution'] = [specs['resolution']]

    available_resolutions = tf.TF_LAYOUT.get_resolutions(template=name[0])
    if specs.get('resolution') and not set(specs['resolution']) & set(available_resolutions):
        fallback_res = available_resolutions[0] if available_resolutions else None
        LOGGER.warning(
            f'Template {name[0]} does not have resolution(s): {specs["resolution"]}.'
            f'Falling back to resolution: {fallback_res}.'
        )
        specs['resolution'] = fallback_res

    files = {}
    files['t1w'] = tf.get(name[0], desc=None, suffix='T1w', **specs)
    files['mask'] = tf.get(name[0], desc='brain', suffix='mask', **specs) or tf.get(
        name[0], label='brain', suffix='mask', **specs
    )
    # Not guaranteed to exist so add fallback
    files['t2w'] = tf.get(name[0], desc=None, suffix='T2w', **specs) or Undefined
    return files


__all__ = ['TemplateFlowSelect', 'TemplateDesc', 'fetch_template_files']
