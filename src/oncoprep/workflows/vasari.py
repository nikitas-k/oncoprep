# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2025 The OncoPrep Developers
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
"""Automated VASARI feature extraction and radiology report generation workflow.

Derives VASARI MRI features from glioma tumor segmentation masks using the
*vasari-auto* library (Ruffle et al., NeuroImage: Clinical, 2024) and
generates structured radiology reports.

VASARI (Visually AcceSAble Rembrandt Images) features characterise glioma
morphology across 25+ imaging features including tumor location, enhancement
quality, proportion of sub-regions, midline crossing, and anatomical invasion
patterns.
"""

from __future__ import annotations

from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from oncoprep.utils.logging import get_logger

LOGGER = get_logger(__name__)


def init_vasari_wf(
    *,
    output_dir: str,
    atlas_space: str = 'mni152',
    report_template: str = 'structured',
    name: str = 'vasari_wf',
) -> Workflow:
    """Create a VASARI feature extraction and radiology report workflow.

    This workflow extracts automated VASARI MRI features from a glioma tumor
    segmentation mask and generates a structured radiology report.  It uses
    the *vasari-auto* library to compute 25 VASARI features from the
    segmentation, including tumor location, enhancement characteristics,
    sub-region proportions, midline crossing, and anatomical invasion patterns.

    The input segmentation must already be in template space (resampled via
    ``ApplyTransforms`` with the ``anat2std_xfm`` from the anatomical
    workflow).  The atlas masks used for location derivation are bundled
    inside OncoPrep under ``data/atlas_masks/{mni152,sri24}/``.

    Features requiring source imaging data (F3, F8, F10, F12, F13, F16–F18,
    F25) are reported as unsupported.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from oncoprep.workflows.vasari import init_vasari_wf
            wf = init_vasari_wf(output_dir='/tmp')

    Parameters
    ----------
    output_dir : str
        Derivatives output directory for saving VASARI JSON and reports.
    atlas_space : str
        Template space for the atlas masks.  Must be one of ``'mni152'``,
        ``'MNI152NLin2009cAsym'``, or ``'SRI24'`` (default: ``'mni152'``).
    report_template : str
        Radiology report style: ``'structured'`` (tabular, default),
        ``'narrative'`` (prose), or ``'brief'`` (key findings only).
    name : str
        Workflow name (default: ``'vasari_wf'``).

    Inputs
    ------
    source_file : str
        Source BIDS file for derivatives naming (typically T1w).
    tumor_seg_std : str
        Path to tumor segmentation mask **already resampled to template
        space** (multi-label, old BraTS convention: 1=NCR/nCET, 2=ED/oedema,
        3=ET).
    subject_id : str
        Subject identifier for the report header.

    Outputs
    -------
    out_features : str
        Path to JSON file containing all VASARI features.
    out_report : str
        Path to HTML fragment for the OncoPrep subject report.
    out_radiology_report : str
        Path to HTML radiology report.
    out_radiology_text : str
        Path to plain-text radiology report.

    Returns
    -------
    Workflow
        Nipype workflow for VASARI feature extraction.

    References
    ----------
    J. Ruffle et al., "VASARI-auto: Equitable, efficient, and economical
    featurisation of glioma MRI," *NeuroImage: Clinical*, 2024.
    """
    from ..interfaces.vasari import (
        VASARIFeatureExtraction,
        VASARIRadiologyReport,
        get_atlas_dir,
    )
    from ..interfaces import DerivativesDataSink

    # Resolve atlas directory at workflow construction time
    resolved_atlas_dir = get_atlas_dir(atlas_space)
    LOGGER.info('VASARI atlas space: %s → %s', atlas_space, resolved_atlas_dir)

    workflow = Workflow(name=name)
    workflow.__desc__ = """\
Automated VASARI MRI features were derived from the tumor segmentation mask
using *vasari-auto* [@vasari_auto], which computes 25 standardised VASARI
features from multi-label glioma segmentation masks without requiring source
imaging data.  The tumor segmentation was resampled to template space using the
native-to-standard transform computed during anatomical preprocessing, then
evaluated against anatomical atlas ROIs (brainstem, frontal lobe, insula,
occipital, parietal, temporal, thalamus, corpus callosum, ventricles, internal
capsule, cortex).  Tumor morphology features (enhancement quality, sub-region
proportions, enhancing margin thickness, multifocality, satellite lesions,
midline crossing) are computed from voxel-level statistics of the segmentation
labels.
"""

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'source_file',      # BIDS source for derivatives naming
                'tumor_seg_std',    # Template-space tumor segmentation (old BraTS)
                'subject_id',       # Subject identifier for report header
            ],
        ),
        name='inputnode',
    )

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'out_features',          # JSON with VASARI features
                'out_report',            # HTML fragment for subject report
                'out_radiology_report',  # HTML radiology report
                'out_radiology_text',    # Plain-text radiology report
            ],
        ),
        name='outputnode',
    )

    # --- VASARI feature extraction ---
    # atlas_dir is set explicitly so vasari-auto uses our bundled atlases
    # in_anat is NOT connected — segmentation is already in template space
    vasari_extract = pe.Node(
        VASARIFeatureExtraction(
            atlas_dir=resolved_atlas_dir,
        ),
        name='vasari_extract',
        mem_gb=4,
    )

    # --- Radiology report generation ---
    vasari_report = pe.Node(
        VASARIRadiologyReport(
            template=report_template,
        ),
        name='vasari_report',
    )

    # --- Save VASARI features JSON as BIDS derivative ---
    ds_vasari_json = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            desc='vasari',
            suffix='features',
            compress=False,
        ),
        name='ds_vasari_json',
        run_without_submitting=True,
    )

    # --- Save VASARI HTML report fragment ---
    ds_vasari_report = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            desc='vasari',
            suffix='features',
            datatype='figures',
        ),
        name='ds_vasari_report',
        run_without_submitting=True,
    )

    # --- Save radiology report (HTML) ---
    ds_radiology_html = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            desc='vasariRadiology',
            suffix='report',
            datatype='figures',
            compress=False,
        ),
        name='ds_radiology_html',
        run_without_submitting=True,
    )

    # --- Save radiology report (plain text) ---
    ds_radiology_txt = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            desc='vasariRadiologyText',
            suffix='report',
            datatype='figures',
            compress=False,
        ),
        name='ds_radiology_txt',
        run_without_submitting=True,
    )

    # --- Connections ---
    workflow.connect([
        # Template-space segmentation → VASARI feature extraction
        # (no in_anat — seg is already in template space, skips internal registration)
        (inputnode, vasari_extract, [
            ('tumor_seg_std', 'in_seg'),
        ]),
        # VASARI features JSON → radiology report
        (vasari_extract, vasari_report, [
            ('out_features', 'in_features'),
        ]),
        (inputnode, vasari_report, [
            ('subject_id', 'patient_id'),
        ]),
        # Extraction → outputs
        (vasari_extract, outputnode, [
            ('out_features', 'out_features'),
            ('out_report', 'out_report'),
        ]),
        (vasari_report, outputnode, [
            ('out_report', 'out_radiology_report'),
            ('out_text', 'out_radiology_text'),
        ]),
        # Derivatives sinks — VASARI features JSON
        (inputnode, ds_vasari_json, [
            ('source_file', 'source_file'),
        ]),
        (vasari_extract, ds_vasari_json, [
            ('out_features', 'in_file'),
        ]),
        # Derivatives sinks — VASARI HTML report fragment
        (inputnode, ds_vasari_report, [
            ('source_file', 'source_file'),
        ]),
        (vasari_extract, ds_vasari_report, [
            ('out_report', 'in_file'),
        ]),
        # Derivatives sinks — radiology report HTML
        (inputnode, ds_radiology_html, [
            ('source_file', 'source_file'),
        ]),
        (vasari_report, ds_radiology_html, [
            ('out_report', 'in_file'),
        ]),
        # Derivatives sinks — radiology report text
        (inputnode, ds_radiology_txt, [
            ('source_file', 'source_file'),
        ]),
        (vasari_report, ds_radiology_txt, [
            ('out_text', 'in_file'),
        ]),
    ])

    return workflow
