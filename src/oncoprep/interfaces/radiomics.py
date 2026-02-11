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
"""Nipype interfaces for PyRadiomics feature extraction."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
    isdefined,
)


# ---------------------------------------------------------------------------
# BraTS label definitions
# ---------------------------------------------------------------------------

BRATS_OLD_LABEL_MAP = {
    1: 'NCR',   # Necrotic Core
    2: 'ED',    # Peritumoral Edema
    3: 'ET',    # Enhancing Tumor
    4: 'RC',    # Resection Cavity
}

BRATS_OLD_LABEL_NAMES = {
    1: 'Necrotic Core (NCR)',
    2: 'Peritumoral Edema (ED)',
    3: 'Enhancing Tumor (ET)',
    4: 'Resection Cavity (RC)',
}

# Composite regions derived from old labels
COMPOSITE_REGIONS = {
    'WT': {
        'name': 'Whole Tumor (WT)',
        'labels': [1, 2, 3, 4],
    },
    'TC': {
        'name': 'Tumor Core (TC)',
        'labels': [1, 3, 4],
    },
}


# ---------------------------------------------------------------------------
# Input / Output Specs
# ---------------------------------------------------------------------------

class _PyRadiomicsInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True,
                   desc='Input image (e.g. preprocessed T1w)')
    in_mask = File(exists=True, mandatory=True,
                   desc='Tumor segmentation mask (multi-label)')
    label_map = traits.Dict(
        key_trait=traits.Int(),
        value_trait=traits.Str(),
        desc='Mapping of label integer → region abbreviation',
    )
    label_names = traits.Dict(
        key_trait=traits.Int(),
        value_trait=traits.Str(),
        desc='Mapping of label integer → human-readable name',
    )
    composites = traits.Dict(
        desc='Composite region definitions: {abbrev: {name, labels}}',
    )
    settings = traits.Dict(
        desc='PyRadiomics extraction settings (optional)',
    )
    extract_shape = traits.Bool(True, usedefault=True,
                                desc='Extract shape features')
    extract_firstorder = traits.Bool(True, usedefault=True,
                                     desc='Extract first-order features')
    extract_glcm = traits.Bool(True, usedefault=True,
                               desc='Extract GLCM texture features')
    extract_glrlm = traits.Bool(True, usedefault=True,
                                desc='Extract GLRLM texture features')
    extract_glszm = traits.Bool(True, usedefault=True,
                                desc='Extract GLSZM texture features')
    extract_gldm = traits.Bool(True, usedefault=True,
                               desc='Extract GLDM texture features')
    extract_ngtdm = traits.Bool(True, usedefault=True,
                                desc='Extract NGTDM texture features')


class _PyRadiomicsOutputSpec(TraitedSpec):
    out_features = File(exists=True,
                        desc='JSON file with extracted radiomics features')
    out_report = File(exists=True,
                      desc='HTML fragment with radiomics feature tables')


# ---------------------------------------------------------------------------
# Interface implementation
# ---------------------------------------------------------------------------

class PyRadiomicsFeatureExtraction(SimpleInterface):
    """Extract radiomics features from an image using a tumor segmentation mask.

    Uses the PyRadiomics library to compute shape, first-order, and texture
    features for each label in the segmentation mask plus user-defined
    composite regions (e.g. whole tumor, tumor core).

    The interface outputs:
    - A JSON file containing all extracted features keyed by region.
    - An HTML report fragment with styled feature tables for embedding
      in the OncoPrep subject report.
    """

    input_spec = _PyRadiomicsInputSpec
    output_spec = _PyRadiomicsOutputSpec

    def _run_interface(self, runtime):
        import nibabel as nib
        import numpy as np

        try:
            from radiomics import featureextractor
        except ImportError:
            raise ImportError(
                'pyradiomics is required for radiomics feature extraction. '
                'Install with: pip install pyradiomics'
            )

        in_file = self.inputs.in_file
        in_mask = self.inputs.in_mask

        # Label definitions
        label_map = (
            dict(self.inputs.label_map)
            if isdefined(self.inputs.label_map)
            else dict(BRATS_OLD_LABEL_MAP)
        )
        label_names = (
            dict(self.inputs.label_names)
            if isdefined(self.inputs.label_names)
            else dict(BRATS_OLD_LABEL_NAMES)
        )
        composites = (
            dict(self.inputs.composites)
            if isdefined(self.inputs.composites)
            else dict(COMPOSITE_REGIONS)
        )

        # Build extractor settings
        settings = {}
        if isdefined(self.inputs.settings) and self.inputs.settings:
            settings = dict(self.inputs.settings)

        # Determine which feature classes to enable
        feature_classes = []
        if self.inputs.extract_shape:
            feature_classes.append('shape')
        if self.inputs.extract_firstorder:
            feature_classes.append('firstorder')
        if self.inputs.extract_glcm:
            feature_classes.append('glcm')
        if self.inputs.extract_glrlm:
            feature_classes.append('glrlm')
        if self.inputs.extract_glszm:
            feature_classes.append('glszm')
        if self.inputs.extract_gldm:
            feature_classes.append('gldm')
        if self.inputs.extract_ngtdm:
            feature_classes.append('ngtdm')

        # Initialize extractor
        extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
        extractor.disableAllFeatures()
        for fc in feature_classes:
            extractor.enableFeatureClassByName(fc)

        # Load mask to discover available labels
        mask_img = nib.load(in_mask)
        mask_data = np.asarray(mask_img.dataobj)
        available_labels = set(np.unique(mask_data).astype(int)) - {0}

        all_features = {}

        # --- Per-label extraction ---
        for label_int, abbrev in sorted(label_map.items()):
            if label_int not in available_labels:
                continue
            region_name = label_names.get(label_int, abbrev)
            try:
                result = extractor.execute(in_file, in_mask, label=label_int)
                features = _parse_radiomics_result(result)
                all_features[abbrev] = {
                    'label': label_int,
                    'name': region_name,
                    'features': features,
                }
            except Exception as exc:
                all_features[abbrev] = {
                    'label': label_int,
                    'name': region_name,
                    'features': {},
                    'error': str(exc),
                }

        # --- Composite region extraction ---
        for comp_abbrev, comp_def in composites.items():
            comp_labels = comp_def['labels']
            comp_name = comp_def['name']
            # Create binary mask for composite region
            comp_mask_data = np.zeros_like(mask_data, dtype=np.uint8)
            for lbl in comp_labels:
                comp_mask_data[mask_data == lbl] = 1

            if comp_mask_data.sum() == 0:
                continue

            comp_mask_path = os.path.abspath(f'composite_{comp_abbrev}.nii.gz')
            comp_mask_img = nib.Nifti1Image(comp_mask_data, mask_img.affine,
                                            mask_img.header)
            nib.save(comp_mask_img, comp_mask_path)

            try:
                result = extractor.execute(in_file, comp_mask_path, label=1)
                features = _parse_radiomics_result(result)
                all_features[comp_abbrev] = {
                    'label': comp_labels,
                    'name': comp_name,
                    'features': features,
                }
            except Exception as exc:
                all_features[comp_abbrev] = {
                    'label': comp_labels,
                    'name': comp_name,
                    'features': {},
                    'error': str(exc),
                }

        # --- Write JSON ---
        out_json = os.path.abspath('radiomics_features.json')
        with open(out_json, 'w') as f:
            json.dump(all_features, f, indent=2, default=str)
        self._results['out_features'] = out_json

        # --- Write HTML report fragment ---
        out_html = os.path.abspath('radiomics_report.html')
        html = _render_radiomics_html(all_features)
        with open(out_html, 'w') as f:
            f.write(html)
        self._results['out_report'] = out_html

        return runtime


# ---------------------------------------------------------------------------
# Helpers (module-level so Nipype Function nodes can use them too)
# ---------------------------------------------------------------------------

def _parse_radiomics_result(result: dict) -> Dict[str, Dict[str, float]]:
    """Parse pyradiomics result dict into categorised feature dict.

    Parameters
    ----------
    result : dict
        Raw output from ``extractor.execute()``.

    Returns
    -------
    dict
        ``{category: {feature_name: value}}`` where *category* is e.g.
        ``'shape'``, ``'firstorder'``, ``'glcm'``.
    """
    features = {}  # type: Dict[str, Dict[str, float]]
    for key, val in result.items():
        # Skip diagnostic / metadata keys
        if key.startswith('diagnostics_') or key.startswith('general_'):
            continue
        # Keys look like: "original_shape_Elongation"
        parts = key.split('_', 2)
        if len(parts) < 3:
            continue
        _image_type, category, feat_name = parts
        category_lower = category.lower()
        if category_lower not in features:
            features[category_lower] = {}
        # Convert numpy types to float
        try:
            features[category_lower][feat_name] = float(val)
        except (TypeError, ValueError):
            features[category_lower][feat_name] = str(val)
    return features


def _render_radiomics_html(all_features: dict) -> str:
    """Render radiomics features as an HTML fragment with collapsible tables.

    Parameters
    ----------
    all_features : dict
        Output from ``PyRadiomicsFeatureExtraction``, keyed by region
        abbreviation.

    Returns
    -------
    str
        HTML fragment string.
    """
    # Category display order & pretty names
    CATEGORY_NAMES = {
        'shape': 'Shape',
        'firstorder': 'First Order',
        'glcm': 'GLCM',
        'glrlm': 'GLRLM',
        'glszm': 'GLSZM',
        'gldm': 'GLDM',
        'ngtdm': 'NGTDM',
    }

    lines = []
    lines.append('<div class="radiomics-report">')

    # --- Summary table: key metrics across all regions ---
    lines.append('<h4>Summary</h4>')
    lines.append(
        '<table class="radiomics-summary">'
        '<thead><tr>'
        '<th>Region</th>'
        '<th>Volume (mm³)</th>'
        '<th>Surface Area (mm²)</th>'
        '<th>Sphericity</th>'
        '<th>Mean Intensity</th>'
        '<th>Std Dev</th>'
        '</tr></thead><tbody>'
    )

    for abbrev, region_data in all_features.items():
        name = region_data.get('name', abbrev)
        feats = region_data.get('features', {})
        error = region_data.get('error')

        if error:
            lines.append(
                f'<tr><td>{name}</td>'
                f'<td colspan="5" style="color:#c0392b;">Error: {error}</td>'
                '</tr>'
            )
            continue

        shape = feats.get('shape', {})
        fo = feats.get('firstorder', {})

        vol = shape.get('MeshVolume', shape.get('VoxelVolume', '—'))
        sa = shape.get('SurfaceArea', '—')
        sph = shape.get('Sphericity', '—')
        mean_i = fo.get('Mean', '—')
        std_i = fo.get('StandardDeviation', '—')

        def _fmt(v):
            if isinstance(v, (int, float)):
                return f'{v:.4f}' if abs(v) < 1e4 else f'{v:.2e}'
            return str(v)

        lines.append(
            f'<tr><td><strong>{name}</strong></td>'
            f'<td>{_fmt(vol)}</td>'
            f'<td>{_fmt(sa)}</td>'
            f'<td>{_fmt(sph)}</td>'
            f'<td>{_fmt(mean_i)}</td>'
            f'<td>{_fmt(std_i)}</td>'
            '</tr>'
        )

    lines.append('</tbody></table>')

    # --- Detailed per-region tables ---
    lines.append('<h4>Detailed Features</h4>')

    for abbrev, region_data in all_features.items():
        name = region_data.get('name', abbrev)
        feats = region_data.get('features', {})
        error = region_data.get('error')

        if error or not feats:
            continue

        region_id = f'radiomics-{abbrev}'
        lines.append(
            f'<details class="radiomics-region" id="{region_id}">'
            f'<summary><strong>{name}</strong></summary>'
        )

        for cat_key in CATEGORY_NAMES:
            if cat_key not in feats:
                continue
            cat_name = CATEGORY_NAMES[cat_key]
            cat_feats = feats[cat_key]

            lines.append(
                f'<table class="radiomics-detail">'
                f'<caption>{cat_name} Features</caption>'
                '<thead><tr><th>Feature</th><th>Value</th></tr></thead>'
                '<tbody>'
            )

            for feat_name, feat_val in sorted(cat_feats.items()):
                if isinstance(feat_val, float):
                    display = (
                        f'{feat_val:.6f}' if abs(feat_val) < 1e6
                        else f'{feat_val:.4e}'
                    )
                else:
                    display = str(feat_val)
                lines.append(
                    f'<tr><td>{feat_name}</td><td>{display}</td></tr>'
                )

            lines.append('</tbody></table>')

        lines.append('</details>')

    lines.append('</div>')
    return '\n'.join(lines)
