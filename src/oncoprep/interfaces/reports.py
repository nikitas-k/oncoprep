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
"""Interfaces to generate reportlets."""

import time
from pathlib import Path

from nipype.interfaces import freesurfer as fs
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    Directory,
    File,
    InputMultiObject,
    SimpleInterface,
    Str,
    TraitedSpec,
    isdefined,
    traits,
)
from nipype.interfaces.io import FSSourceInputSpec as _FSSourceInputSpec
from nipype.interfaces.mixins import reporting
from niworkflows.interfaces.reportlets.base import _SVGReportCapableInputSpec

from niworkflows.interfaces.reportlets.masks import ROIsPlot as _ROIsPlot

SUBJECT_TEMPLATE = """\
\t<ul class="elem-desc">
\t\t<li>Subject ID: {subject_id}</li>
\t\t<li>Structural images: {n_t1s:d} T1-weighted {t2w}{t1ce}{flair_seg}</li>
\t\t<li>Standard spaces: {output_spaces}</li>
\t\t<li>FreeSurfer reconstruction: {freesurfer_status}</li>
\t</ul>
"""

ABOUT_TEMPLATE = """\t<ul>
\t\t<li>OncoPrep version: {version}</li>
\t\t<li>OncoPrep command: <code>{command}</code></li>
\t\t<li>Date preprocessed: {date}</li>
\t</ul>
</div>
"""


class _SummaryOutputSpec(TraitedSpec):
    """Output specification for summary reportlets."""

    out_report = File(exists=True, desc='HTML segment containing summary')


class SummaryInterface(SimpleInterface):
    """
    Base Nipype interface for HTML summary reportlets.

    This interface provides a foundation for generating HTML report segments,
    handling file I/O and path management for reportlet generation.

    """

    output_spec = _SummaryOutputSpec

    def _run_interface(self, runtime):
        """Run interface and save generated HTML segment."""
        segment = self._generate_segment()
        path = Path(runtime.cwd) / 'report.html'
        path.write_text(segment)
        self._results['out_report'] = str(path)
        return runtime

    def _generate_segment(self):
        """Generate HTML segment content."""
        raise NotImplementedError


class _SubjectSummaryInputSpec(BaseInterfaceInputSpec):
    """Input specification for subject summary reportlet."""

    t1w = InputMultiObject(File(exists=True), desc='T1w structural images')
    t2w = InputMultiObject(File(exists=True), desc='T2w structural images')
    t1ce = InputMultiObject(File(exists=True), desc='T1ce contrast-enhanced images')
    flair = InputMultiObject(File(exists=True), desc='FLAIR images')
    subjects_dir = Directory(desc='FreeSurfer subjects directory')
    subject_id = Str(desc='Subject ID')
    output_spaces = InputMultiObject(Str, desc='list of standard spaces')


class _SubjectSummaryOutputSpec(_SummaryOutputSpec):
    """Output specification for subject summary reportlet."""

    # Ensures summary runs before first ReconAll call
    subject_id = Str(desc='FreeSurfer subject ID')


class SubjectSummary(SummaryInterface):
    """
    Subject HTML summary reportlet.

    Generates a summary of subject-level preprocessing including structural images,
    output spaces, and FreeSurfer reconstruction status.

    """

    input_spec = _SubjectSummaryInputSpec
    output_spec = _SubjectSummaryOutputSpec

    def _run_interface(self, runtime):
        """Run interface, preserving subject_id in outputs."""
        if isdefined(self.inputs.subject_id):
            self._results['subject_id'] = self.inputs.subject_id
        return super()._run_interface(runtime)

    def _generate_segment(self):
        """Generate subject summary HTML segment."""
        if not isdefined(self.inputs.subjects_dir):
            freesurfer_status = 'Not run'
        else:
            recon = fs.ReconAll(
                subjects_dir=self.inputs.subjects_dir,
                subject_id=self.inputs.subject_id,
                T1_files=self.inputs.t1w,
                flags='-noskullstrip',
            )
            if recon.cmdline.startswith('echo'):
                freesurfer_status = 'Pre-existing directory'
            else:
                freesurfer_status = 'Run by OncoPrep'

        t2w_seg = ''
        if self.inputs.t2w:
            t2w_seg = f', {len(self.inputs.t2w):d} T2w'

        t1ce_seg = ''
        if isdefined(self.inputs.t1ce) and self.inputs.t1ce:
            t1ce_seg = f', {len(self.inputs.t1ce):d} T1ce'

        flair_seg = ''
        if isdefined(self.inputs.flair) and self.inputs.flair:
            flair_seg = f', {len(self.inputs.flair):d} FLAIR'

        output_spaces = self.inputs.output_spaces
        if not isdefined(output_spaces):
            output_spaces = '&lt;none given&gt;'
        else:
            output_spaces = ', '.join(output_spaces)

        return SUBJECT_TEMPLATE.format(
            subject_id=self.inputs.subject_id,
            n_t1s=len(self.inputs.t1w),
            t2w=t2w_seg,
            t1ce=t1ce_seg,
            flair_seg=flair_seg,
            output_spaces=output_spaces,
            freesurfer_status=freesurfer_status,
        )


class _AboutSummaryInputSpec(BaseInterfaceInputSpec):
    """Input specification for about summary reportlet."""

    version = Str(desc='OncoPrep version')
    command = Str(desc='OncoPrep command')


class AboutSummary(SummaryInterface):
    """
    About section reportlet.

    Generates an about section containing version, command, and processing date.

    """

    input_spec = _AboutSummaryInputSpec

    def _generate_segment(self):
        """Generate about section HTML segment."""
        return ABOUT_TEMPLATE.format(
            version=self.inputs.version,
            command=self.inputs.command,
            date=time.strftime('%Y-%m-%d %H:%M:%S %z'),
        )


class _FSSurfaceReportInputSpec(_SVGReportCapableInputSpec, _FSSourceInputSpec):
    """Input specification for FreeSurfer surface report."""

    pass


class _FSSurfaceReportOutputSpec(reporting.ReportCapableOutputSpec):
    """Output specification for FreeSurfer surface report."""

    pass


class FSSurfaceReport(SimpleInterface):
    """
    FreeSurfer surface report interface.

    Replaces ReconAllRPT without requiring recon-all execution. Generates
    SVG visualization of brain anatomy with ribbon overlay.

    """

    input_spec = _FSSurfaceReportInputSpec
    output_spec = _FSSurfaceReportOutputSpec

    def _run_interface(self, runtime):
        """Run interface and generate surface visualization report."""
        from nibabel import load

        from niworkflows.viz.utils import compose_view, cuts_from_bbox, plot_registration

        rootdir = Path(self.inputs.subjects_dir) / self.inputs.subject_id
        _anat_file = str(rootdir / 'mri' / 'brain.mgz')
        _contour_file = str(rootdir / 'mri' / 'ribbon.mgz')

        anat = load(_anat_file)
        contour_nii = load(_contour_file)

        n_cuts = 7
        cuts = cuts_from_bbox(contour_nii, cuts=n_cuts)

        self._results['out_report'] = str(Path(runtime.cwd) / self.inputs.out_report)

        # Call composer
        compose_view(
            plot_registration(
                anat,
                'fixed-image',
                estimate_brightness=True,
                cuts=cuts,
                contour=contour_nii,
                compress=self.inputs.compress_report,
            ),
            [],
            out_file=self._results['out_report'],
        )
        return runtime


class _TumorROIsPlotInputSpec(_SVGReportCapableInputSpec):
    in_file = File(exists=True, mandatory=True, desc='background volume (e.g. T1w)')
    in_rois = InputMultiObject(
        File(exists=True), mandatory=True, desc='tumor segmentation file(s)'
    )
    in_mask = File(exists=True, desc='brain mask')
    masked = traits.Bool(False, usedefault=True, desc='mask in_file prior to plotting')
    colors = traits.Either(
        None, traits.List(Str), usedefault=True, desc='contour colors per level'
    )
    levels = traits.Either(
        None, traits.List(traits.Float), usedefault=True, desc='contour levels'
    )
    mask_color = Str('r', usedefault=True, desc='color for brain mask contour')
    legend_labels = traits.List(
        traits.Tuple(Str, Str),
        desc='list of (color, label) tuples for the SVG legend',
    )


class TumorROIsPlot(_ROIsPlot):
    """ROIsPlot with an appended color legend for tumor segmentation regions."""

    input_spec = _TumorROIsPlotInputSpec

    def _generate_report(self):
        super()._generate_report()

        if not isdefined(self.inputs.legend_labels) or not self.inputs.legend_labels:
            return

        legend_items = list(self.inputs.legend_labels)
        svg_text = Path(self._out_report).read_text()

        box_h = 28 + 22 * len(legend_items)
        legend_svg = (
            '<g transform="translate(10, 10)">'
            f'<rect x="0" y="0" width="260" height="{box_h}" rx="6" ry="6" '
            'fill="white" fill-opacity="0.85" stroke="#ccc" stroke-width="0.5"/>'
            '<text x="10" y="20" font-family="Arial, sans-serif" font-size="11" '
            'font-weight="bold" fill="#333">Tumor Segmentation</text>'
        )
        for i, (color, label) in enumerate(legend_items):
            y = 38 + i * 22
            legend_svg += (
                f'<rect x="14" y="{y - 9}" width="14" height="14" rx="2" ry="2" '
                f'fill="none" stroke="{color}" stroke-width="2.5"/>'
                f'<text x="36" y="{y + 3}" font-family="Arial, sans-serif" '
                f'font-size="11" fill="#333">{label}</text>'
            )
        legend_svg += '</g>'

        svg_text = svg_text.replace('</svg>', legend_svg + '\n</svg>')
        Path(self._out_report).write_text(svg_text)


__all__ = ['SummaryInterface', 'SubjectSummary', 'AboutSummary', 'FSSurfaceReport', 'TumorROIsPlot']
