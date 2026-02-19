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
"""Radiomics feature extraction workflow using PyRadiomics.

Extracts shape, first-order, and texture (GLCM, GLRLM, GLSZM, GLDM, NGTDM)
features from preprocessed anatomical images using tumor segmentation masks.
Features are computed per tumor sub-region (NCR, ED, ET, RC) and for composite
regions (Whole Tumor, Tumor Core).
"""

from __future__ import annotations

from typing import List, Optional

from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from oncoprep.utils.logging import get_logger

LOGGER = get_logger(__name__)


def init_anat_radiomics_wf(
    *,
    output_dir: str,
    extract_shape: bool = True,
    extract_firstorder: bool = True,
    extract_glcm: bool = True,
    extract_glrlm: bool = True,
    extract_glszm: bool = True,
    extract_gldm: bool = True,
    extract_ngtdm: bool = True,
    susan_fwhm: float = 2.0,
    name: str = 'anat_radiomics_wf',
) -> Workflow:
    """Create a radiomics feature extraction workflow.

    This workflow extracts quantitative radiomics features from preprocessed
    anatomical images using tumor segmentation masks. It uses PyRadiomics
    to compute shape-based, first-order intensity, and texture features
    for each tumor sub-region and composite regions.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from oncoprep.workflows.radiomics import init_anat_radiomics_wf
            wf = init_anat_radiomics_wf(output_dir='/tmp')

    Parameters
    ----------
    output_dir : str
        Derivatives output directory for saving radiomics JSON and reports.
    extract_shape : bool
        Extract shape-based features (volume, surface area, sphericity, etc.).
    extract_firstorder : bool
        Extract first-order intensity statistics (mean, variance, entropy, etc.).
    extract_glcm : bool
        Extract Gray-Level Co-occurrence Matrix texture features.
    extract_glrlm : bool
        Extract Gray-Level Run Length Matrix texture features.
    extract_glszm : bool
        Extract Gray-Level Size Zone Matrix texture features.
    extract_gldm : bool
        Extract Gray-Level Dependence Matrix texture features.
    extract_ngtdm : bool
        Extract Neighbouring Gray Tone Difference Matrix texture features.
    susan_fwhm : float
        FWHM for SUSAN edge-preserving denoising in mm (default: 2.0).
        Applied after histogram normalization but before feature extraction,
        following Pati et al., AJNR 2024; 45: 1291–1298.
    name : str
        Workflow name (default: ``'anat_radiomics_wf'``).

    Inputs
    ------
    source_file : str
        Source BIDS file for derivatives naming (typically T1w).
    t1w_preproc : str
        Path to preprocessed T1w image.
    tumor_seg : str
        Path to tumor segmentation mask (multi-label, old BraTS convention:
        1=NCR, 2=ED, 3=ET, 4=RC).

    Outputs
    -------
    out_features : str
        Path to JSON file containing all extracted radiomics features.
    out_report : str
        Path to HTML fragment for the OncoPrep subject report.

    Returns
    -------
    Workflow
        Nipype workflow for radiomics extraction.
    """
    from ..interfaces.radiomics import (
        HistogramNormalization,
        SUSANDenoising,
        PyRadiomicsFeatureExtraction,
        BRATS_OLD_LABEL_MAP,
        BRATS_OLD_LABEL_NAMES,
        COMPOSITE_REGIONS,
    )
    from ..interfaces import DerivativesDataSink

    workflow = Workflow(name=name)
    workflow.__desc__ = """\
Prior to feature extraction, the preprocessed T1-weighted image was
intensity-normalised using brain-masked z-score standardisation with
Winsorisation at the 1st and 99th percentiles, following IBSI best-practice
recommendations for reproducible radiomics [@shinohara2014; @um2019].
The normalised image was then denoised using SUSAN (Smallest Univalue
Segment Assimilating Nucleus) edge-preserving smoothing (FWHM = {fwhm} mm),
which reduces noise while preserving tumor boundaries [@smith1997].
This preprocessing order (intensity normalisation followed by SUSAN
denoising) follows the pipeline described in Pati et al., *AJNR* 2024; 45:
1291–1298.
Radiomics features were then extracted using *PyRadiomics* [@pyradiomics].
For each tumor sub-region defined by the BraTS segmentation labels (necrotic
core, peritumoral edema, enhancing tumor, resection cavity) and composite
regions (whole tumor, tumor core), shape-based, first-order intensity, and
texture features (GLCM, GLRLM, GLSZM, GLDM, NGTDM) were computed.
""".format(fwhm=susan_fwhm)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'source_file',   # BIDS source for derivatives naming
                't1w_preproc',   # Preprocessed T1w image
                'tumor_seg',     # Multi-label tumor segmentation
                'brain_mask',    # Brain mask for histogram normalization
            ],
        ),
        name='inputnode',
    )

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'out_features',  # JSON with all radiomics features
                'out_report',    # HTML fragment for report
            ],
        ),
        name='outputnode',
    )

    # --- Histogram normalization (IBSI best-practice) ---
    hist_norm = pe.Node(
        HistogramNormalization(
            method='zscore',
            percentile_lower=1.0,
            percentile_upper=99.0,
        ),
        name='hist_norm',
        mem_gb=2,
    )

    # --- SUSAN edge-preserving denoising (after intensity normalization) ---
    susan_denoise = pe.Node(
        SUSANDenoising(fwhm=susan_fwhm),
        name='susan_denoise',
        mem_gb=4,
    )

    # --- PyRadiomics extraction node ---
    radiomics_extract = pe.Node(
        PyRadiomicsFeatureExtraction(
            label_map=BRATS_OLD_LABEL_MAP,
            label_names=BRATS_OLD_LABEL_NAMES,
            composites=COMPOSITE_REGIONS,
            extract_shape=extract_shape,
            extract_firstorder=extract_firstorder,
            extract_glcm=extract_glcm,
            extract_glrlm=extract_glrlm,
            extract_glszm=extract_glszm,
            extract_gldm=extract_gldm,
            extract_ngtdm=extract_ngtdm,
        ),
        name='radiomics_extract',
        mem_gb=4,
    )

    # --- Save radiomics JSON as BIDS derivative ---
    ds_radiomics_json = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            desc='radiomics',
            suffix='features',
            compress=False,
        ),
        name='ds_radiomics_json',
        run_without_submitting=True,
    )

    # --- Save radiomics HTML report fragment ---
    ds_radiomics_report = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            desc='radiomics',
            suffix='features',
            datatype='figures',
        ),
        name='ds_radiomics_report',
        run_without_submitting=True,
    )

    # --- Connections ---
    workflow.connect([
        # Image + brain mask → histogram normalization
        (inputnode, hist_norm, [
            ('t1w_preproc', 'in_file'),
            ('brain_mask', 'in_mask'),
        ]),
        # Normalized image → SUSAN denoising
        (hist_norm, susan_denoise, [
            ('out_file', 'in_file'),
        ]),
        (inputnode, susan_denoise, [
            ('brain_mask', 'in_mask'),
        ]),
        # Denoised image + tumor seg → extraction
        (susan_denoise, radiomics_extract, [
            ('out_file', 'in_file'),
        ]),
        (inputnode, radiomics_extract, [
            ('tumor_seg', 'in_mask'),
        ]),
        # Extraction → outputs
        (radiomics_extract, outputnode, [
            ('out_features', 'out_features'),
            ('out_report', 'out_report'),
        ]),
        # Derivatives sinks
        (inputnode, ds_radiomics_json, [
            ('source_file', 'source_file'),
        ]),
        (radiomics_extract, ds_radiomics_json, [
            ('out_features', 'in_file'),
        ]),
        (inputnode, ds_radiomics_report, [
            ('source_file', 'source_file'),
        ]),
        (radiomics_extract, ds_radiomics_report, [
            ('out_report', 'in_file'),
        ]),
    ])

    return workflow


def init_multimodal_radiomics_wf(
    *,
    output_dir: str,
    modalities: Optional[List[str]] = None,
    extract_shape: bool = True,
    extract_firstorder: bool = True,
    extract_glcm: bool = True,
    extract_glrlm: bool = False,
    extract_glszm: bool = False,
    extract_gldm: bool = False,
    extract_ngtdm: bool = False,
    susan_fwhm: float = 2.0,
    name: str = 'multimodal_radiomics_wf',
) -> Workflow:
    """Create a multi-modal radiomics extraction workflow.

    Extracts radiomics features from multiple MRI modalities (T1w, T1ce, T2w,
    FLAIR) using the same tumor segmentation mask.  Each modality produces its
    own feature set which is merged into a single JSON file.

    This is useful for building radiomic signatures that leverage contrast
    differences across modalities.

    Parameters
    ----------
    output_dir : str
        Derivatives output directory.
    modalities : list of str, optional
        Modalities to process.  Default: ``['t1w', 't1ce', 't2w', 'flair']``.
    extract_shape : bool
        Extract shape features (computed once from the mask, same across
        modalities).
    extract_firstorder : bool
        Extract first-order intensity features per modality.
    extract_glcm : bool
        Extract GLCM texture features per modality.
    extract_glrlm : bool
        Extract GLRLM texture features per modality.
    extract_glszm : bool
        Extract GLSZM texture features per modality.
    extract_gldm : bool
        Extract GLDM texture features per modality.
    extract_ngtdm : bool
        Extract NGTDM texture features per modality.
    susan_fwhm : float
        FWHM for SUSAN edge-preserving denoising in mm (default: 2.0).
    name : str
        Workflow name.

    Inputs
    ------
    source_file : str
        Source BIDS file for derivatives naming.
    t1w_preproc : str
        Preprocessed T1w image.
    t1ce_preproc : str
        Preprocessed T1ce image.
    t2w_preproc : str
        Preprocessed T2w image.
    flair_preproc : str
        Preprocessed FLAIR image.
    tumor_seg : str
        Tumor segmentation mask.

    Outputs
    -------
    out_features : str
        Path to merged multi-modal JSON.
    out_report : str
        Path to HTML report fragment.

    Returns
    -------
    Workflow
        Nipype workflow.
    """
    from ..interfaces.radiomics import (
        HistogramNormalization,
        SUSANDenoising,
        PyRadiomicsFeatureExtraction,
        BRATS_OLD_LABEL_MAP,
        BRATS_OLD_LABEL_NAMES,
        COMPOSITE_REGIONS,
    )
    from ..interfaces import DerivativesDataSink

    if modalities is None:
        modalities = ['t1w', 't1ce', 't2w', 'flair']

    workflow = Workflow(name=name)
    workflow.__desc__ = """\
Multi-modal radiomics features were extracted from preprocessed {mods}
images using *PyRadiomics* [@pyradiomics]. Each modality was first
intensity-normalised using brain-masked z-score standardisation with
Winsorisation at the 1st and 99th percentiles, following IBSI best-practice
recommendations for reproducible radiomics. The normalised images were
then denoised using SUSAN edge-preserving smoothing (FWHM = {fwhm} mm),
following the preprocessing pipeline of Pati et al., *AJNR* 2024; 45:
1291–1298. Features were then computed for each tumor sub-region and
composite region using the BraTS segmentation mask.
""".format(
        mods=', '.join(m.upper() for m in modalities),
        fwhm=susan_fwhm,
    )

    input_fields = ['source_file', 'tumor_seg', 'brain_mask']
    for mod in modalities:
        input_fields.append(f'{mod}_preproc')

    inputnode = pe.Node(
        niu.IdentityInterface(fields=input_fields),
        name='inputnode',
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['out_features', 'out_report']),
        name='outputnode',
    )

    # Create one normalization + denoising + extraction node per modality
    extract_nodes = {}
    for mod in modalities:
        # Histogram normalization per modality
        norm_node = pe.Node(
            HistogramNormalization(
                method='zscore',
                percentile_lower=1.0,
                percentile_upper=99.0,
            ),
            name=f'hist_norm_{mod}',
            mem_gb=2,
        )

        # SUSAN denoising per modality (after normalization)
        susan_node = pe.Node(
            SUSANDenoising(fwhm=susan_fwhm),
            name=f'susan_denoise_{mod}',
            mem_gb=4,
        )

        node = pe.Node(
            PyRadiomicsFeatureExtraction(
                label_map=BRATS_OLD_LABEL_MAP,
                label_names=BRATS_OLD_LABEL_NAMES,
                composites=COMPOSITE_REGIONS,
                extract_shape=extract_shape,
                extract_firstorder=extract_firstorder,
                extract_glcm=extract_glcm,
                extract_glrlm=extract_glrlm,
                extract_glszm=extract_glszm,
                extract_gldm=extract_gldm,
                extract_ngtdm=extract_ngtdm,
            ),
            name=f'radiomics_{mod}',
            mem_gb=4,
        )
        extract_nodes[mod] = node
        workflow.connect([
            # Image + brain mask → normalization
            (inputnode, norm_node, [
                (f'{mod}_preproc', 'in_file'),
                ('brain_mask', 'in_mask'),
            ]),
            # Normalized image → SUSAN denoising
            (norm_node, susan_node, [
                ('out_file', 'in_file'),
            ]),
            (inputnode, susan_node, [
                ('brain_mask', 'in_mask'),
            ]),
            # Denoised image + tumor seg → extraction
            (susan_node, node, [
                ('out_file', 'in_file'),
            ]),
            (inputnode, node, [
                ('tumor_seg', 'in_mask'),
            ]),
        ])

    # Merge all per-modality JSONs into one
    merge_features = pe.Node(
        niu.Function(
            function=_merge_multimodal_features,
            input_names=['feature_files', 'modality_names'],
            output_names=['out_features', 'out_report'],
        ),
        name='merge_features',
    )
    merge_features.inputs.modality_names = modalities

    # Use a Merge node to collect all feature files
    feature_merge = pe.Node(
        niu.Merge(len(modalities)),
        name='feature_merge',
    )
    for idx, mod in enumerate(modalities, 1):
        workflow.connect([
            (extract_nodes[mod], feature_merge, [
                ('out_features', f'in{idx}'),
            ]),
        ])

    workflow.connect([
        (feature_merge, merge_features, [('out', 'feature_files')]),
        (merge_features, outputnode, [
            ('out_features', 'out_features'),
            ('out_report', 'out_report'),
        ]),
    ])

    # Derivative sinks
    ds_multimodal_json = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            desc='radiomicsMultimodal',
            suffix='features',
            compress=False,
        ),
        name='ds_multimodal_json',
        run_without_submitting=True,
    )
    ds_multimodal_report = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            desc='radiomicsMultimodal',
            suffix='features',
            datatype='figures',
        ),
        name='ds_multimodal_report',
        run_without_submitting=True,
    )

    workflow.connect([
        (inputnode, ds_multimodal_json, [('source_file', 'source_file')]),
        (merge_features, ds_multimodal_json, [('out_features', 'in_file')]),
        (inputnode, ds_multimodal_report, [('source_file', 'source_file')]),
        (merge_features, ds_multimodal_report, [('out_report', 'in_file')]),
    ])

    return workflow


def _extract_subject_id(source_file):
    """Extract BIDS subject ID from a source file path.

    Parameters
    ----------
    source_file : str
        BIDS-style file path (e.g. ``'.../sub-001_ses-01_T1w.nii.gz'``).

    Returns
    -------
    str
        Subject identifier (e.g. ``'sub-001'``).
    """
    import os
    import re

    basename = os.path.basename(source_file)
    match = re.match(r'(sub-[a-zA-Z0-9]+)', basename)
    if match:
        return match.group(1)
    return basename.split('_')[0]


# ---------------------------------------------------------------------------
# Helper functions for Nipype Function nodes
# ---------------------------------------------------------------------------

def _merge_multimodal_features(feature_files, modality_names):
    """Merge per-modality radiomics JSONs into one file and render report.

    This function is wrapped in a Nipype Function node.

    Parameters
    ----------
    feature_files : list of str
        Paths to per-modality JSON files.
    modality_names : list of str
        Modality labels matching ``feature_files`` order.

    Returns
    -------
    out_features : str
        Path to merged JSON.
    out_report : str
        Path to HTML report fragment.
    """
    import json
    import os

    merged = {}
    for mod, fpath in zip(modality_names, feature_files):
        with open(fpath) as f:
            merged[mod.upper()] = json.load(f)

    out_features = os.path.abspath('multimodal_radiomics.json')
    with open(out_features, 'w') as f:
        json.dump(merged, f, indent=2, default=str)

    # Render multi-modal HTML
    out_report = os.path.abspath('multimodal_radiomics_report.html')
    html = _render_multimodal_html(merged)
    with open(out_report, 'w') as f:
        f.write(html)

    return out_features, out_report


def _render_multimodal_html(merged_features):
    """Render multi-modal radiomics as HTML with per-modality tabs.

    Parameters
    ----------
    merged_features : dict
        ``{modality: {region: {features: ...}}}``.

    Returns
    -------
    str
        HTML fragment.
    """
    lines = ['<div class="radiomics-multimodal">']
    lines.append('<h4>Multi-Modal Radiomics Summary</h4>')

    # Summary table across modalities — key shape/intensity metrics
    lines.append(
        '<table class="radiomics-summary">'
        '<thead><tr>'
        '<th>Modality</th><th>Region</th>'
        '<th>Volume (mm³)</th><th>Mean</th><th>Std Dev</th>'
        '<th>Entropy</th><th>Sphericity</th>'
        '</tr></thead><tbody>'
    )

    for mod, regions in merged_features.items():
        for abbrev, rdata in regions.items():
            name = rdata.get('name', abbrev)
            feats = rdata.get('features', {})
            shape = feats.get('shape', {})
            fo = feats.get('firstorder', {})

            def _fmt(v):
                if isinstance(v, (int, float)):
                    return f'{v:.4f}' if abs(v) < 1e4 else f'{v:.2e}'
                return str(v)

            vol = shape.get('MeshVolume', shape.get('VoxelVolume', '—'))
            mean = fo.get('Mean', '—')
            std = fo.get('StandardDeviation', '—')
            entropy = fo.get('Entropy', '—')
            sph = shape.get('Sphericity', '—')

            lines.append(
                f'<tr><td><strong>{mod}</strong></td>'
                f'<td>{name}</td>'
                f'<td>{_fmt(vol)}</td>'
                f'<td>{_fmt(mean)}</td>'
                f'<td>{_fmt(std)}</td>'
                f'<td>{_fmt(entropy)}</td>'
                f'<td>{_fmt(sph)}</td></tr>'
            )

    lines.append('</tbody></table>')
    lines.append('</div>')
    return '\n'.join(lines)
