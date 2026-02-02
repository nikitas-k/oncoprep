# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2024 The OncoPrep Developers <oncoprep@gmail.com>
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
"""Utilities to handle BIDS inputs."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from bids.layout import BIDSLayout
from niworkflows.data import load as nwf_load


def collect_derivatives(
    derivatives_dir: str,
    subject_id: str,
    std_spaces: List[str],
    spec: Optional[Dict[str, Any]] = None,
    patterns: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Gather existing derivatives and compose a cache.

    Parameters
    ----------
    derivatives_dir : str
        Path to the derivatives directory
    subject_id : str
        Subject identifier
    std_spaces : list of str
        Standard space names to query
    spec : dict, optional
        Derivative specification. If None, defaults to nipreps spec
    patterns : dict, optional
        BIDS filename patterns. If None, defaults to nipreps patterns
    session_id : str, optional
        Session identifier

    Returns
    -------
    derivs_cache : dict
        Dictionary of collected derivatives

    """
    if spec is None or patterns is None:
        deriv_config = nwf_load('nipreps.json')
        if spec is None:
            spec = deriv_config.get('spec', {})
        if patterns is None:
            patterns = deriv_config.get('patterns', {})

    deriv_config = nwf_load('nipreps.json')
    layout = BIDSLayout(derivatives_dir, config=deriv_config, validate=False)

    derivs_cache = {}

    # Subject and session (if available) will be added to all queries
    qry_base = {'subject': subject_id}
    if session_id:
        qry_base['session'] = session_id

    for key, qry in spec.get('baseline', {}).items():
        qry = {**qry, **qry_base}
        item = layout.get(**qry)
        if not item:
            continue

        # Respect label order in queries
        if 'label' in qry:
            item = sorted(item, key=lambda x: qry['label'].index(x.entities['label']))

        paths = [item.path for item in item]

        derivs_cache[f't1w_{key}'] = paths[0] if len(paths) == 1 else paths

    transforms = derivs_cache.setdefault('transforms', {})
    for _space in std_spaces:
        space = _space.replace(':cohort-', '+')
        for key, qry in spec.get('transforms', {}).items():
            qry = {**qry, **qry_base}
            qry['from'] = qry.get('from') or space
            qry['to'] = qry.get('to') or space
            item = layout.get(return_type='filename', **qry)
            if not item:
                continue
            transforms.setdefault(_space, {})[key] = item[0] if len(item) == 1 else item

    for key, qry in spec.get('surfaces', {}).items():
        qry = {**qry, **qry_base}
        item = layout.get(return_type='filename', **qry)
        if not item or len(item) != 2:
            continue

        derivs_cache[key] = sorted(item)

    return derivs_cache


def write_bidsignore(deriv_dir: str) -> None:
    """
    Write a ``.bidsignore`` file to the derivatives directory.

    Parameters
    ----------
    deriv_dir : str
        Path to derivatives directory

    """
    bids_ignore = [
        '*.html',
        'logs/',
        'figures/',  # Reports
        '*_xfm.*',  # Unspecified transform files
        '*.surf.gii',  # Unspecified structural outputs
    ]
    ignore_file = Path(deriv_dir) / '.bidsignore'

    ignore_file.write_text('\n'.join(bids_ignore) + '\n')


def write_derivative_description(
    bids_dir: str,
    deriv_dir: str,
) -> None:
    """
    Write a ``dataset_description.json`` for the derivatives folder.

    Parameters
    ----------
    bids_dir : str
        Path to the input BIDS dataset
    deriv_dir : str
        Path to the derivatives directory

    Examples
    --------
    >>> from pathlib import Path
    >>> from tempfile import TemporaryDirectory
    >>> tmpdir = TemporaryDirectory()
    >>> deriv_desc = Path(tmpdir.name) / 'dataset_description.json'
    >>> write_derivative_description('.', deriv_desc.parent)  # doctest: +SKIP
    >>> deriv_desc.is_file()  # doctest: +SKIP
    True

    """
    from oncoprep.__about__ import __version__

    bids_dir = Path(bids_dir)
    deriv_dir = Path(deriv_dir)

    desc = {
        'Name': 'OncoPrep - Oncology MRI PREProcessing workflow',
        'BIDSVersion': '1.9.0',
        'DatasetType': 'derivative',
        'GeneratedBy': [
            {
                'Name': 'OncoPrep',
                'Version': __version__,
                'CodeURL': 'https://github.com/oncoprep/oncoprep',
            }
        ],
        'HowToAcknowledge': 'Please cite the OncoPrep paper and include the generated '
        'citation boilerplate within the Methods section of the text.',
    }

    # Keys deriving from source dataset
    orig_desc = {}
    fname = bids_dir / 'dataset_description.json'
    if fname.exists():
        try:
            orig_desc = json.loads(fname.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            orig_desc = {}

    if 'DatasetDOI' in orig_desc:
        doi = orig_desc['DatasetDOI']
        desc['SourceDatasets'] = [
            {
                'URL': f'https://doi.org/{doi}',
                'DOI': doi,
            }
        ]
    if 'License' in orig_desc:
        desc['License'] = orig_desc['License']

    (deriv_dir / 'dataset_description.json').write_text(json.dumps(desc, indent=4) + '\n')


__all__ = [
    'collect_derivatives',
    'write_bidsignore',
    'write_derivative_description',
]
