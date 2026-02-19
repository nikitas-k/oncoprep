# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2025 The OncoPrep Developers
# Based on vasari-auto (Ruffle et al., NeuroImage: Clinical, 2024)
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
"""Nipype interfaces for VASARI feature extraction and radiology report generation.

Wraps the *vasari-auto* library (Ruffle et al., NeuroImage: Clinical, 2024)
to derive VASARI MRI features from glioma tumor segmentation masks.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict

from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
    isdefined,
)

# ---------------------------------------------------------------------------
# Atlas directory resolution
# ---------------------------------------------------------------------------


def _import_vasari_auto():
    """Import ``get_vasari_features`` from *vasari-auto*.

    Returns
    -------
    callable
        ``vasari_auto.vasari_auto.get_vasari_features``

    Raises
    ------
    ImportError
        If *vasari-auto* is not installed.
    """
    try:
        from vasari_auto.vasari_auto import get_vasari_features
    except ImportError:
        raise ImportError(
            'vasari-auto is required for VASARI feature extraction. '
            'Install with: pip install vasari-auto  '
            'or: pip install "vasari-auto @ '
            'git+https://github.com/nikitas-k/vasari-auto.git"'
        )
    return get_vasari_features


#: Mapping from TemplateFlow / OncoPrep space names to local atlas sub-dir
_SPACE_TO_ATLAS_DIR = {
    'mni152': 'mni152',
    'mni152nlin2009casym': 'mni152',
    'mni152nlin6asym': 'mni152',
    'sri24': 'sri24',
}

#: Reference brain image per atlas sub-directory (used as ApplyTransforms ref)
_ATLAS_REFERENCE = {
    'mni152': 'MNI152_T1_1mm_brain.nii.gz',
    'sri24': 'MNI152_in_SRI24_T1_1mm_brain.nii.gz',
}


def get_atlas_dir(space: str = 'mni152') -> str:
    """Return the absolute path to the bundled atlas masks for *space*.

    Parameters
    ----------
    space : str
        Template space name (case-insensitive).  Accepted values include
        ``'MNI152'``, ``'MNI152NLin2009cAsym'``, ``'MNI152NLin6Asym'``, and
        ``'SRI24'``.

    Returns
    -------
    str
        Path to the atlas directory **with** a trailing ``/`` (required by
        *vasari-auto* which concatenates filenames directly).

    Raises
    ------
    ValueError
        If *space* cannot be mapped to a known atlas directory.
    """
    key = space.lower().replace('-', '').replace('_', '')
    subdir = _SPACE_TO_ATLAS_DIR.get(key)
    if subdir is None:
        raise ValueError(
            f"Unsupported atlas space '{space}'. "
            f"Accepted spaces: {list(_SPACE_TO_ATLAS_DIR.keys())}"
        )
    atlas_root = Path(__file__).resolve().parent.parent / 'data' / 'atlas_masks' / subdir
    if not atlas_root.is_dir():
        raise FileNotFoundError(
            f"Atlas directory not found: {atlas_root}. "
            f"Ensure the atlas_masks data is installed."
        )
    return str(atlas_root) + '/'


def get_atlas_reference(space: str = 'mni152') -> str:
    """Return the path to the reference brain image for *space*.

    This is the NIfTI file used as the ``reference_image`` for
    ``ApplyTransforms`` when resampling segmentations into template space.

    Parameters
    ----------
    space : str
        Template space name (same values as :func:`get_atlas_dir`).

    Returns
    -------
    str
        Absolute path to the 1 mm reference brain NIfTI.
    """
    key = space.lower().replace('-', '').replace('_', '')
    subdir = _SPACE_TO_ATLAS_DIR.get(key)
    if subdir is None:
        raise ValueError(
            f"Unsupported atlas space '{space}'. "
            f"Accepted spaces: {list(_SPACE_TO_ATLAS_DIR.keys())}"
        )
    ref_name = _ATLAS_REFERENCE[subdir]
    atlas_root = Path(__file__).resolve().parent.parent / 'data' / 'atlas_masks' / subdir
    ref_path = atlas_root / ref_name
    if not ref_path.is_file():
        raise FileNotFoundError(
            f"Atlas reference brain not found: {ref_path}. "
            f"Ensure the atlas_masks data is installed."
        )
    return str(ref_path)

# ---------------------------------------------------------------------------
# VASARI feature lookup dictionaries
# ---------------------------------------------------------------------------

#: VASARI F1 — Tumour Location codes
F1_LOCATION_MAP = {
    1: 'Frontal Lobe',
    2: 'Temporal Lobe',
    3: 'Insula',
    4: 'Parietal Lobe',
    5: 'Occipital Lobe',
    6: 'Brainstem',
    7: 'Corpus Callosum',
    8: 'Thalamus',
}

#: VASARI F2 — Laterality codes
F2_LATERALITY_MAP = {
    1: 'Right',
    2: 'Bilateral',
    3: 'Left',
}

#: VASARI F4 — Enhancement quality codes
F4_ENHANCEMENT_MAP = {
    1: 'None',
    2: 'Mild/Faint',
    3: 'Marked/Avid',
}

#: VASARI F5 — Proportion enhancing codes
F5_PROP_ENHANCING_MAP = {
    3: '≤5%',
    4: '6–33%',
    5: '34–67%',
    6: '>68%',
}

#: VASARI F6 — Proportion nCET codes
F6_PROP_NCET_MAP = {
    3: '≤5%',
    4: '6–33%',
    5: '34–67%',
    6: '68–95%',
    7: '96–99.5%',
    8: '>99.5% (nCET-predominant)',
}

#: VASARI F11 — Enhancing margin thickness codes
F11_THICKNESS_MAP = {
    3: 'Thin (≤3×)',
    4: 'Thick (>3×)',
    5: 'Solid (thick, no nCET)',
}

#: VASARI F19 — Ependymal invasion codes
F19_EPENDYMAL_MAP = {
    1: 'Absent',
    2: 'Present',
}

#: VASARI F20 — Cortical involvement codes
F20_CORTICAL_MAP = {
    1: 'Absent',
    2: 'Present',
}

#: VASARI F21 — Deep WM invasion codes
F21_DEEP_WM_MAP = {
    1: 'Absent',
    2: 'Present',
}

#: VASARI F22/F23 — Midline crossing codes
F22_F23_MIDLINE_MAP = {
    2: 'Does not cross midline',
    3: 'Crosses midline',
}

#: VASARI F24 — Satellites codes
F24_SATELLITES_MAP = {
    1: 'Absent',
    2: 'Present',
}


# ---------------------------------------------------------------------------
# VASARI Feature Extraction Interface
# ---------------------------------------------------------------------------

class _VASARIFeaturesInputSpec(BaseInterfaceInputSpec):
    in_seg = File(
        exists=True, mandatory=True,
        desc='Tumor segmentation mask (multi-label NIfTI). '
             'Labels: 1=nCET, 2=oedema, 3=ET (BraTS old convention).',
    )
    in_anat = File(
        exists=True,
        desc='Optional anatomical image (T1w) for MNI registration. '
             'If not provided, segmentation is assumed to be in MNI space.',
    )
    atlas_dir = traits.Str(
        desc='Path to atlas masks directory. If not set, uses the '
             'bundled vasari_auto atlas_masks directory.',
    )
    enhancing_label = traits.Int(
        3, usedefault=True,
        desc='Integer label for enhancing tumor in the segmentation.',
    )
    nonenhancing_label = traits.Int(
        1, usedefault=True,
        desc='Integer label for non-enhancing tumor in the segmentation.',
    )
    oedema_label = traits.Int(
        2, usedefault=True,
        desc='Integer label for perilesional oedema in the segmentation.',
    )
    verbose = traits.Bool(
        False, usedefault=True,
        desc='Enable verbose logging during feature extraction.',
    )


class _VASARIFeaturesOutputSpec(TraitedSpec):
    out_features = File(
        exists=True,
        desc='JSON file with all VASARI features.',
    )
    out_report = File(
        exists=True,
        desc='HTML fragment with VASARI feature summary for the subject report.',
    )


class VASARIFeatureExtraction(SimpleInterface):
    """Extract VASARI MRI features from a glioma tumor segmentation mask.

    Uses the *vasari-auto* library to compute 27 VASARI features from a
    multi-label segmentation mask (BraTS convention: 1=nCET, 2=ED, 3=ET).
    Optionally registers the segmentation to MNI152 space using an
    accompanying anatomical image.

    The interface outputs:
    - A JSON file containing all VASARI features and their human-readable
      labels.
    - An HTML report fragment with a styled feature table suitable for
      embedding in the OncoPrep subject report.

    References
    ----------
    J. Ruffle et al., "VASARI-auto: Equitable, efficient, and economical
    featurisation of glioma MRI," *NeuroImage: Clinical*, 2024.
    """

    input_spec = _VASARIFeaturesInputSpec
    output_spec = _VASARIFeaturesOutputSpec

    def _run_interface(self, runtime):

        get_vasari_features = _import_vasari_auto()

        seg_file = self.inputs.in_seg

        # Determine atlas directory
        # Priority: explicit atlas_dir → bundled OncoPrep atlases → vasari-auto bundled
        atlas_dir = None
        if isdefined(self.inputs.atlas_dir) and self.inputs.atlas_dir:
            atlas_dir = self.inputs.atlas_dir
        else:
            # Use bundled OncoPrep atlases (default: MNI152)
            try:
                atlas_dir = get_atlas_dir('mni152')
            except (ValueError, FileNotFoundError):
                # Fallback to vasari-auto bundled atlas
                from vasari_auto.vasari_auto import _resolve_atlas_dir
                atlas_dir = _resolve_atlas_dir('mni152')

        # Optional anatomical image for MNI registration.
        # When the segmentation is already in template space (pre-resampled),
        # in_anat should NOT be set — vasari-auto will skip its internal
        # registration and use the segmentation as-is.
        anat_img = None
        if isdefined(self.inputs.in_anat):
            anat_img = self.inputs.in_anat

        # Run vasari-auto feature extraction
        result_df = get_vasari_features(
            file=seg_file,
            anat_img=anat_img,
            atlases=atlas_dir,
            verbose=self.inputs.verbose,
            enhancing_label=self.inputs.enhancing_label,
            nonenhancing_label=self.inputs.nonenhancing_label,
            oedema_label=self.inputs.oedema_label,
        )

        # Convert DataFrame row to a structured dict
        features = _dataframe_to_vasari_dict(result_df)

        # Write JSON
        out_json = os.path.abspath('vasari_features.json')
        with open(out_json, 'w') as f:
            json.dump(features, f, indent=2, default=str)
        self._results['out_features'] = out_json

        # Render HTML report
        out_html = os.path.abspath('vasari_report.html')
        html = _render_vasari_html(features)
        with open(out_html, 'w') as f:
            f.write(html)
        self._results['out_report'] = out_html

        return runtime


# ---------------------------------------------------------------------------
# Radiology Report Generator Interface
# ---------------------------------------------------------------------------

class _RadiologyReportInputSpec(BaseInterfaceInputSpec):
    in_features = File(
        exists=True, mandatory=True,
        desc='JSON file with VASARI features (from VASARIFeatureExtraction).',
    )
    patient_id = traits.Str(
        desc='Patient/subject identifier for the report header.',
    )
    template = traits.Enum(
        'structured', 'narrative', 'brief',
        usedefault=True,
        desc='Report style: structured (tabular), narrative (prose), '
             'or brief (key findings only).',
    )


class _RadiologyReportOutputSpec(TraitedSpec):
    out_report = File(
        exists=True,
        desc='HTML radiology report.',
    )
    out_text = File(
        exists=True,
        desc='Plain-text radiology report.',
    )


class VASARIRadiologyReport(SimpleInterface):
    """Generate a radiology report from VASARI features.

    Produces a structured or narrative radiology report summarising the
    automated VASARI assessment of a glioma.  The report is generated in
    both HTML (for embedding in the OncoPrep visual report) and plain text
    (for clinical review or downstream NLP pipelines).

    Three report styles are supported:
    - ``structured``: Tabular format with feature codes and descriptions.
    - ``narrative``: Prose-style report suitable for clinical review.
    - ``brief``: Key findings only (location, enhancement, midline).
    """

    input_spec = _RadiologyReportInputSpec
    output_spec = _RadiologyReportOutputSpec

    def _run_interface(self, runtime):
        with open(self.inputs.in_features) as f:
            features = json.load(f)

        patient_id = ''
        if isdefined(self.inputs.patient_id):
            patient_id = self.inputs.patient_id

        template = self.inputs.template

        if template == 'narrative':
            html, text = _generate_narrative_report(features, patient_id)
        elif template == 'brief':
            html, text = _generate_brief_report(features, patient_id)
        else:
            html, text = _generate_structured_report(features, patient_id)

        out_html = os.path.abspath('vasari_radiology_report.html')
        with open(out_html, 'w') as f:
            f.write(html)
        self._results['out_report'] = out_html

        out_txt = os.path.abspath('vasari_radiology_report.txt')
        with open(out_txt, 'w') as f:
            f.write(text)
        self._results['out_text'] = out_txt

        return runtime


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _dataframe_to_vasari_dict(df) -> Dict:
    """Convert a vasari-auto DataFrame result to a structured dict.

    Parameters
    ----------
    df : pandas.DataFrame
        Single-row DataFrame from ``get_vasari_features()``.

    Returns
    -------
    dict
        Structured VASARI features with codes, labels, and metadata.
    """
    import math

    row = df.iloc[0].to_dict()

    def _safe(val):
        """Convert NaN to None for JSON serialisation."""
        if isinstance(val, float) and math.isnan(val):
            return None
        return val

    def _lookup(val, lookup_map):
        """Look up human-readable label for a VASARI code."""
        val = _safe(val)
        if val is None:
            return None
        try:
            return lookup_map.get(int(val), str(val))
        except (TypeError, ValueError):
            return str(val)

    features = {
        'metadata': {
            'filename': str(row.get('filename', '')),
            'reporter': str(row.get('reporter', 'VASARI-auto')),
            'time_taken_seconds': _safe(row.get('time_taken_seconds')),
            'software_note': (
                'Automated VASARI features derived from tumor segmentation '
                'mask using vasari-auto (Ruffle et al. 2024). Features '
                'requiring source imaging data are reported as unsupported.'
            ),
        },
        'features': {
            'F1': {
                'name': 'Tumour Location',
                'code': _safe(row.get('F1 Tumour Location')),
                'label': _lookup(row.get('F1 Tumour Location'), F1_LOCATION_MAP),
            },
            'F2': {
                'name': 'Side of Tumour Epicenter',
                'code': _safe(row.get('F2 Side of Tumour Epicenter')),
                'label': _lookup(row.get('F2 Side of Tumour Epicenter'), F2_LATERALITY_MAP),
            },
            'F3': {
                'name': 'Eloquent Brain',
                'code': None,
                'label': 'Unsupported (requires source imaging)',
            },
            'F4': {
                'name': 'Enhancement Quality',
                'code': _safe(row.get('F4 Enhancement Quality')),
                'label': _lookup(row.get('F4 Enhancement Quality'), F4_ENHANCEMENT_MAP),
            },
            'F5': {
                'name': 'Proportion Enhancing',
                'code': _safe(row.get('F5 Proportion Enhancing')),
                'label': _lookup(row.get('F5 Proportion Enhancing'), F5_PROP_ENHANCING_MAP),
            },
            'F6': {
                'name': 'Proportion nCET',
                'code': _safe(row.get('F6 Proportion nCET')),
                'label': _lookup(row.get('F6 Proportion nCET'), F6_PROP_NCET_MAP),
            },
            'F7': {
                'name': 'Proportion Necrosis',
                'code': _safe(row.get('F7 Proportion Necrosis')),
                'label': _lookup(row.get('F7 Proportion Necrosis'), F6_PROP_NCET_MAP),
            },
            'F8': {
                'name': 'Cysts',
                'code': None,
                'label': 'Unsupported (requires source imaging)',
            },
            'F9': {
                'name': 'Multifocal or Multicentric',
                'code': _safe(row.get('F9 Multifocal or Multicentric')),
                'label': _lookup(
                    row.get('F9 Multifocal or Multicentric'),
                    {1: 'Not multifocal', 2: 'Multifocal/Multicentric'},
                ),
            },
            'F10': {
                'name': 'T1/FLAIR Ratio',
                'code': None,
                'label': 'Unsupported (requires source imaging)',
            },
            'F11': {
                'name': 'Thickness of Enhancing Margin',
                'code': _safe(row.get('F11 Thickness of enhancing margin')),
                'label': _lookup(
                    row.get('F11 Thickness of enhancing margin'),
                    F11_THICKNESS_MAP,
                ),
            },
            'F12': {
                'name': 'Definition of the Enhancing Margin',
                'code': None,
                'label': 'Unsupported (requires source imaging)',
            },
            'F13': {
                'name': 'Definition of the Non-Enhancing Tumour Margin',
                'code': None,
                'label': 'Unsupported (requires source imaging)',
            },
            'F14': {
                'name': 'Proportion of Oedema',
                'code': _safe(row.get('F14 Proportion of Oedema')),
                'label': _lookup(
                    row.get('F14 Proportion of Oedema'),
                    {2: 'None', 3: '≤5%', 4: '6–33%', 5: '>33%'},
                ),
            },
            'F16': {
                'name': 'Haemorrhage',
                'code': None,
                'label': 'Unsupported (requires source imaging)',
            },
            'F17': {
                'name': 'Diffusion',
                'code': None,
                'label': 'Unsupported (requires source imaging)',
            },
            'F18': {
                'name': 'Pial Invasion',
                'code': None,
                'label': 'Unsupported (requires source imaging)',
            },
            'F19': {
                'name': 'Ependymal Invasion',
                'code': _safe(row.get('F19 Ependymal Invasion')),
                'label': _lookup(row.get('F19 Ependymal Invasion'), F19_EPENDYMAL_MAP),
            },
            'F20': {
                'name': 'Cortical Involvement',
                'code': _safe(row.get('F20 Cortical involvement')),
                'label': _lookup(row.get('F20 Cortical involvement'), F20_CORTICAL_MAP),
            },
            'F21': {
                'name': 'Deep WM Invasion',
                'code': _safe(row.get('F21 Deep WM invasion')),
                'label': _lookup(row.get('F21 Deep WM invasion'), F21_DEEP_WM_MAP),
            },
            'F22': {
                'name': 'nCET Crosses Midline',
                'code': _safe(row.get('F22 nCET Crosses Midline')),
                'label': _lookup(
                    row.get('F22 nCET Crosses Midline'), F22_F23_MIDLINE_MAP,
                ),
            },
            'F23': {
                'name': 'CET Crosses Midline',
                'code': _safe(row.get('F23 CET Crosses midline')),
                'label': _lookup(
                    row.get('F23 CET Crosses midline'), F22_F23_MIDLINE_MAP,
                ),
            },
            'F24': {
                'name': 'Satellites',
                'code': _safe(row.get('F24 satellites')),
                'label': _lookup(row.get('F24 satellites'), F24_SATELLITES_MAP),
            },
            'F25': {
                'name': 'Calvarial Modelling',
                'code': None,
                'label': 'Unsupported (requires source imaging)',
            },
        },
    }

    return features


def _render_vasari_html(features: Dict) -> str:
    """Render VASARI features as an HTML report fragment.

    Parameters
    ----------
    features : dict
        Structured VASARI feature dict from ``_dataframe_to_vasari_dict()``.

    Returns
    -------
    str
        HTML fragment string.
    """
    lines = ['<div class="vasari-report">']
    lines.append('<h4>Automated VASARI Assessment</h4>')

    # Metadata
    meta = features.get('metadata', {})
    lines.append(
        '<p class="vasari-meta">'
        f'Reporter: <strong>{meta.get("reporter", "VASARI-auto")}</strong> '
        f'| Processing time: {meta.get("time_taken_seconds", "—"):.1f}s'
        '</p>'
    )

    # Feature table
    lines.append(
        '<table class="vasari-features">'
        '<thead><tr>'
        '<th>Feature</th><th>Description</th><th>Code</th><th>Assessment</th>'
        '</tr></thead><tbody>'
    )

    feat_dict = features.get('features', {})
    for fkey in sorted(feat_dict.keys(), key=lambda k: int(k[1:])):
        feat = feat_dict[fkey]
        name = feat.get('name', fkey)
        code = feat.get('code')
        label = feat.get('label', '—')

        code_str = str(code) if code is not None else '—'
        css_class = 'vasari-unsupported' if code is None else ''

        lines.append(
            f'<tr class="{css_class}">'
            f'<td><strong>{fkey}</strong></td>'
            f'<td>{name}</td>'
            f'<td>{code_str}</td>'
            f'<td>{label}</td>'
            '</tr>'
        )

    lines.append('</tbody></table>')

    # Disclaimer
    lines.append(
        '<p class="vasari-disclaimer">'
        '<em>Note: VASARI features marked as "Unsupported" require source '
        'imaging data and cannot be derived from segmentation masks alone. '
        'This automated assessment is not intended for clinical use.</em>'
        '</p>'
    )

    lines.append('</div>')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Radiology report generators
# ---------------------------------------------------------------------------

def _generate_structured_report(features: Dict, patient_id: str = ''):
    """Generate a structured (tabular) radiology report.

    Parameters
    ----------
    features : dict
        VASARI feature dict.
    patient_id : str
        Patient identifier.

    Returns
    -------
    html : str
        HTML report.
    text : str
        Plain-text report.
    """
    feat_dict = features.get('features', {})

    # -- Plain text --
    text_lines = []
    text_lines.append('=' * 72)
    text_lines.append('AUTOMATED VASARI RADIOLOGY REPORT')
    text_lines.append('=' * 72)
    if patient_id:
        text_lines.append(f'Patient: {patient_id}')
    text_lines.append(f'Reporter: {features["metadata"].get("reporter", "VASARI-auto")}')
    text_lines.append('')
    text_lines.append('FINDINGS')
    text_lines.append('-' * 72)

    for fkey in sorted(feat_dict.keys(), key=lambda k: int(k[1:])):
        feat = feat_dict[fkey]
        name = feat.get('name', fkey)
        label = feat.get('label', '—')
        text_lines.append(f'  {fkey:>4s}  {name:<45s}  {label}')

    text_lines.append('')
    text_lines.append('IMPRESSION')
    text_lines.append('-' * 72)
    impression = _generate_impression(feat_dict)
    text_lines.append(impression)
    text_lines.append('')
    text_lines.append(
        'NOTE: This report was generated automatically using vasari-auto '
        '(Ruffle et al. 2024) and is not intended for clinical use.'
    )
    text = '\n'.join(text_lines)

    # -- HTML --
    html_lines = ['<div class="vasari-radiology-report">']
    html_lines.append('<h3>Automated VASARI Radiology Report</h3>')
    if patient_id:
        html_lines.append(f'<p><strong>Patient:</strong> {patient_id}</p>')
    html_lines.append(
        f'<p><strong>Reporter:</strong> '
        f'{features["metadata"].get("reporter", "VASARI-auto")}</p>'
    )

    html_lines.append('<h4>Findings</h4>')
    html_lines.append(
        '<table class="vasari-report-table">'
        '<thead><tr><th>Feature</th><th>Assessment</th></tr></thead><tbody>'
    )
    for fkey in sorted(feat_dict.keys(), key=lambda k: int(k[1:])):
        feat = feat_dict[fkey]
        name = feat.get('name', fkey)
        label = feat.get('label', '—')
        css = ' class="unsupported"' if feat.get('code') is None else ''
        html_lines.append(
            f'<tr{css}><td><strong>{fkey}</strong> {name}</td>'
            f'<td>{label}</td></tr>'
        )
    html_lines.append('</tbody></table>')

    html_lines.append('<h4>Impression</h4>')
    html_lines.append(f'<p>{impression}</p>')

    html_lines.append(
        '<p class="disclaimer"><em>This report was generated automatically '
        'using vasari-auto (Ruffle et al. 2024) and is not intended for '
        'clinical use.</em></p>'
    )
    html_lines.append('</div>')
    html = '\n'.join(html_lines)

    return html, text


def _generate_narrative_report(features: Dict, patient_id: str = ''):
    """Generate a narrative (prose) radiology report.

    Parameters
    ----------
    features : dict
        VASARI feature dict.
    patient_id : str
        Patient identifier.

    Returns
    -------
    html : str
        HTML report.
    text : str
        Plain-text report.
    """
    feat = features.get('features', {})

    # Build narrative paragraphs
    paragraphs = []

    # Location and laterality
    loc = feat.get('F1', {}).get('label', 'unknown location')
    side = feat.get('F2', {}).get('label', 'indeterminate laterality')
    paragraphs.append(
        f'There is a mass lesion centred in the {loc} with {side.lower()} '
        f'predominance.'
    )

    # Enhancement
    enh_qual = feat.get('F4', {}).get('label', 'unknown')
    enh_prop = feat.get('F5', {}).get('label')
    margin = feat.get('F11', {}).get('label')
    enh_desc = f'Enhancement quality is {enh_qual.lower()}'
    if enh_prop:
        enh_desc += f', with the enhancing component comprising {enh_prop} of the lesion'
    if margin:
        enh_desc += f'. The enhancing margin is {margin.lower()}'
    enh_desc += '.'
    paragraphs.append(enh_desc)

    # Non-enhancing and necrosis
    ncet_prop = feat.get('F6', {}).get('label')
    oedema_prop = feat.get('F14', {}).get('label')
    if ncet_prop:
        paragraphs.append(
            f'The non-enhancing tumor component comprises {ncet_prop} of the lesion.'
        )
    if oedema_prop:
        paragraphs.append(f'Perilesional oedema accounts for {oedema_prop}.')

    # Multifocality
    multifocal = feat.get('F9', {}).get('label')
    satellites = feat.get('F24', {}).get('label')
    if multifocal and 'Multifocal' in str(multifocal):
        paragraphs.append('The lesion demonstrates multifocal disease.')
    if satellites and 'Present' in str(satellites):
        paragraphs.append('Satellite lesions are present.')

    # Anatomical involvement
    involvement = []
    if feat.get('F19', {}).get('label') == 'Present':
        involvement.append('ependymal invasion')
    if feat.get('F20', {}).get('label') == 'Present':
        involvement.append('cortical involvement')
    if feat.get('F21', {}).get('label') == 'Present':
        involvement.append('deep white matter invasion')
    if involvement:
        paragraphs.append(
            f'There is evidence of {", ".join(involvement)}.'
        )

    # Midline crossing
    ncet_midline = feat.get('F22', {}).get('label', '')
    cet_midline = feat.get('F23', {}).get('label', '')
    if 'Crosses' in str(ncet_midline) or 'Crosses' in str(cet_midline):
        parts = []
        if 'Crosses' in str(ncet_midline):
            parts.append('non-enhancing tumor')
        if 'Crosses' in str(cet_midline):
            parts.append('enhancing tumor')
        paragraphs.append(
            f'The {" and ".join(parts)} crosses the midline.'
        )

    narrative = ' '.join(paragraphs)

    # Impression
    impression = _generate_impression(feat)

    # -- Plain text --
    text_lines = []
    text_lines.append('=' * 72)
    text_lines.append('AUTOMATED VASARI RADIOLOGY REPORT')
    text_lines.append('=' * 72)
    if patient_id:
        text_lines.append(f'Patient: {patient_id}')
    text_lines.append(f'Reporter: {features["metadata"].get("reporter", "VASARI-auto")}')
    text_lines.append('')
    text_lines.append('FINDINGS')
    text_lines.append('-' * 72)
    text_lines.append(narrative)
    text_lines.append('')
    text_lines.append('IMPRESSION')
    text_lines.append('-' * 72)
    text_lines.append(impression)
    text_lines.append('')
    text_lines.append(
        'NOTE: This report was generated automatically using vasari-auto '
        '(Ruffle et al. 2024) and is not intended for clinical use.'
    )
    text = '\n'.join(text_lines)

    # -- HTML --
    html_lines = ['<div class="vasari-radiology-report narrative">']
    html_lines.append('<h3>Automated VASARI Radiology Report</h3>')
    if patient_id:
        html_lines.append(f'<p><strong>Patient:</strong> {patient_id}</p>')
    html_lines.append(
        f'<p><strong>Reporter:</strong> '
        f'{features["metadata"].get("reporter", "VASARI-auto")}</p>'
    )
    html_lines.append('<h4>Findings</h4>')
    html_lines.append(f'<p>{narrative}</p>')
    html_lines.append('<h4>Impression</h4>')
    html_lines.append(f'<p>{impression}</p>')
    html_lines.append(
        '<p class="disclaimer"><em>This report was generated automatically '
        'using vasari-auto (Ruffle et al. 2024) and is not intended for '
        'clinical use.</em></p>'
    )
    html_lines.append('</div>')
    html = '\n'.join(html_lines)

    return html, text


def _generate_brief_report(features: Dict, patient_id: str = ''):
    """Generate a brief (key findings only) radiology report.

    Parameters
    ----------
    features : dict
        VASARI feature dict.
    patient_id : str
        Patient identifier.

    Returns
    -------
    html : str
        HTML report.
    text : str
        Plain-text report.
    """
    feat = features.get('features', {})

    key_findings = [
        f'Location: {feat.get("F1", {}).get("label", "—")}',
        f'Laterality: {feat.get("F2", {}).get("label", "—")}',
        f'Enhancement: {feat.get("F4", {}).get("label", "—")}',
        f'Enhancing proportion: {feat.get("F5", {}).get("label", "—")}',
        f'Enhancing margin: {feat.get("F11", {}).get("label", "—")}',
        f'Ependymal invasion: {feat.get("F19", {}).get("label", "—")}',
        f'Cortical involvement: {feat.get("F20", {}).get("label", "—")}',
        f'Deep WM invasion: {feat.get("F21", {}).get("label", "—")}',
        f'Midline crossing (nCET): {feat.get("F22", {}).get("label", "—")}',
        f'Midline crossing (CET): {feat.get("F23", {}).get("label", "—")}',
        f'Multifocal: {feat.get("F9", {}).get("label", "—")}',
        f'Satellites: {feat.get("F24", {}).get("label", "—")}',
    ]

    text_lines = ['VASARI KEY FINDINGS']
    if patient_id:
        text_lines.append(f'Patient: {patient_id}')
    text_lines.append('')
    for finding in key_findings:
        text_lines.append(f'  - {finding}')
    text_lines.append('')
    text_lines.append(
        'NOTE: Automated assessment, not for clinical use.'
    )
    text = '\n'.join(text_lines)

    html_lines = ['<div class="vasari-brief-report">']
    html_lines.append('<h4>VASARI Key Findings</h4>')
    html_lines.append('<ul>')
    for finding in key_findings:
        html_lines.append(f'<li>{finding}</li>')
    html_lines.append('</ul>')
    html_lines.append(
        '<p class="disclaimer"><em>Automated assessment, '
        'not for clinical use.</em></p>'
    )
    html_lines.append('</div>')
    html = '\n'.join(html_lines)

    return html, text


def _generate_impression(feat_dict: Dict) -> str:
    """Generate a clinical impression summary from VASARI features.

    Parameters
    ----------
    feat_dict : dict
        The ``features`` sub-dict from the structured VASARI output.

    Returns
    -------
    str
        Impression text.
    """
    parts = []

    # Location
    loc = feat_dict.get('F1', {}).get('label')
    side = feat_dict.get('F2', {}).get('label')
    if loc and side:
        parts.append(
            f'{side} {loc.lower()} mass lesion'
        )
    elif loc:
        parts.append(f'{loc} mass lesion')

    # Enhancement characteristics
    enh = feat_dict.get('F4', {}).get('label')
    if enh and enh != 'None':
        parts.append(f'with {enh.lower()} enhancement')

    # Aggressive features
    aggressive = []
    if feat_dict.get('F19', {}).get('label') == 'Present':
        aggressive.append('ependymal invasion')
    if feat_dict.get('F21', {}).get('label') == 'Present':
        aggressive.append('deep white matter invasion')
    ncet_ml = feat_dict.get('F22', {}).get('label', '')
    cet_ml = feat_dict.get('F23', {}).get('label', '')
    if 'Crosses' in str(cet_ml):
        aggressive.append('enhancing tumor crossing midline')
    elif 'Crosses' in str(ncet_ml):
        aggressive.append('non-enhancing tumor crossing midline')

    if aggressive:
        parts.append(f'demonstrating {", ".join(aggressive)}')

    # Multifocal
    if feat_dict.get('F9', {}).get('code') == 2:
        parts.append('with multifocal disease')

    if parts:
        impression = ', '.join(parts) + '.'
        impression = impression[0].upper() + impression[1:]
    else:
        impression = 'Glioma with indeterminate VASARI features.'

    impression += (
        ' Automated VASARI assessment consistent with high-grade glioma '
        'characteristics. Clinical and histopathological correlation '
        'recommended.'
    )

    return impression
