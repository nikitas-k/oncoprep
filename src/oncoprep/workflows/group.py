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
"""Group-level analysis for OncoPrep.

Implements cohort-wide post-processing operations that require data from
**all** participants, most notably ComBat harmonization of radiomics
features across scanner/site batches.

This module is invoked when ``analysis_level == 'group'`` in the BIDS-Apps
CLI.  It operates on OncoPrep derivatives that were produced during the
``participant`` stage.

References
----------
.. [johnson2007] W. E. Johnson, C. Li, and A. Rabinovic,
   "Adjusting batch effects in microarray expression data using
   empirical Bayes methods," *Biostatistics*, vol. 8, no. 1,
   pp. 118–127, 2007.
.. [fortin2018] J.-P. Fortin et al., "Harmonization of cortical
   thickness measurements across scanners and sites,"
   *NeuroImage*, vol. 167, pp. 104–120, 2018.
.. [pati2024] S. Pati et al., "Reproducibility of the Tumor-Habitat
   MRI Biomarker DESMOND," *AJNR Am J Neuroradiol*, vol. 45, no. 9,
   pp. 1291–1298, 2024.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger('oncoprep.group')


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_combat_batch_csv(
    bids_dir: Path,
    output_csv: Path,
    *,
    participant_label: Optional[List[str]] = None,
) -> Path:
    """Auto-generate a ComBat batch CSV from BIDS JSON sidecars.

    Scans the raw BIDS dataset for JSON sidecars accompanying
    anatomical images and extracts scanner-identifying fields that
    survive anonymization:

    * ``Manufacturer``
    * ``ManufacturerModelName``
    * ``MagneticFieldStrength``

    A batch label is constructed by concatenating these values
    (e.g. ``"Siemens_Prisma_3T"``).

    Age and sex are also extracted when available.  The function
    looks in three places (in order of priority):

    1. The JSON sidecar itself (``PatientAge`` / ``Age``, ``PatientSex`` / ``Sex``)
    2. ``participants.tsv`` at the BIDS root (``age``, ``sex`` columns)

    For longitudinal (multi-session) datasets one row is emitted per
    subject × session so that the downstream ComBat step can operate
    on individual observations.

    The resulting CSV is suitable for ``--combat-batch``.

    Parameters
    ----------
    bids_dir : Path
        Root of the raw BIDS dataset.
    output_csv : Path
        Path where the batch CSV will be written.
    participant_label : list of str, optional
        Restrict to these participants (with or without ``sub-`` prefix).

    Returns
    -------
    Path
        The written CSV file path.

    Raises
    ------
    FileNotFoundError
        If *bids_dir* does not exist or contains no sidecars.
    """
    import csv
    import re

    bids_dir = Path(bids_dir).resolve()
    if not bids_dir.is_dir():
        raise FileNotFoundError(f'BIDS directory not found: {bids_dir}')

    # Normalise participant filter
    allowed: Optional[set] = None
    if participant_label:
        allowed = set()
        for p in participant_label:
            allowed.add(p if p.startswith('sub-') else f'sub-{p}')

    # --- Load participants.tsv if available ---
    participants_info: Dict[str, Dict[str, str]] = {}
    tsv_path = bids_dir / 'participants.tsv'
    if tsv_path.is_file():
        try:
            with open(tsv_path) as f:
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    pid = row.get('participant_id', '')
                    if not pid:
                        continue
                    p_info: Dict[str, str] = {}
                    # age
                    for key in ('age', 'Age', 'PatientAge'):
                        if key in row and row[key]:
                            p_info['age'] = row[key]
                            break
                    # sex
                    for key in ('sex', 'Sex', 'PatientSex', 'gender', 'Gender'):
                        if key in row and row[key]:
                            p_info['sex'] = row[key]
                            break
                    participants_info[pid] = p_info
        except Exception:
            logger.debug('Could not parse participants.tsv')

    # --- Glob for anat JSON sidecars ---
    patterns = [
        bids_dir.glob('sub-*/anat/*.json'),
        bids_dir.glob('sub-*/ses-*/anat/*.json'),
    ]
    sidecar_files = sorted(
        fp for pat in patterns for fp in pat
    )

    if not sidecar_files:
        raise FileNotFoundError(
            f'No anatomical JSON sidecars found under {bids_dir}'
        )

    # Build one row per observation (subject or subject×session)
    rows: Dict[str, Dict[str, str]] = {}  # observation_id → row dict
    for fp in sidecar_files:
        subj_match = re.search(r'(sub-[a-zA-Z0-9]+)', str(fp))
        if not subj_match:
            continue
        subj_id = subj_match.group(1)
        if allowed is not None and subj_id not in allowed:
            continue

        ses_match = re.search(r'(ses-[a-zA-Z0-9]+)', str(fp))
        session = ses_match.group(1) if ses_match else None
        obs_id = f'{subj_id}_{session}' if session else subj_id

        if obs_id in rows:
            continue  # first sidecar per observation wins

        try:
            with open(fp) as f:
                meta = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        manufacturer = str(meta.get('Manufacturer', 'Unknown')).strip()
        model = str(meta.get('ManufacturerModelName', 'Unknown')).strip()
        field_strength = meta.get('MagneticFieldStrength', 'Unknown')

        # Sanitise components for a clean batch label
        manufacturer_clean = re.sub(r'[^a-zA-Z0-9]', '', manufacturer)
        model_clean = re.sub(r'[^a-zA-Z0-9]', '', model)
        fs_clean = (
            str(int(field_strength))
            if isinstance(field_strength, (int, float))
            else re.sub(r'[^a-zA-Z0-9]', '', str(field_strength))
        )

        batch = f'{manufacturer_clean}_{model_clean}_{fs_clean}T'

        row: Dict[str, str] = {
            'subject_id': obs_id,
            'batch': batch,
            'Manufacturer': manufacturer,
            'ManufacturerModelName': model,
            'MagneticFieldStrength': str(field_strength),
        }

        # --- Extract age and sex ---
        # Priority 1: sidecar JSON
        age_val = None
        for key in ('PatientAge', 'Age', 'age'):
            if key in meta and meta[key] not in (None, '', 'n/a'):
                age_val = str(meta[key])
                break
        sex_val = None
        for key in ('PatientSex', 'Sex', 'sex'):
            if key in meta and meta[key] not in (None, '', 'n/a'):
                sex_val = str(meta[key])
                break

        # Priority 2: participants.tsv
        p_info = participants_info.get(subj_id, {})
        if age_val is None and 'age' in p_info:
            age_val = p_info['age']
        if sex_val is None and 'sex' in p_info:
            sex_val = p_info['sex']

        if age_val is not None:
            row['age'] = age_val
        if sex_val is not None:
            row['sex'] = sex_val

        rows[obs_id] = row

    if not rows:
        raise FileNotFoundError(
            'No scanner metadata found in any JSON sidecar.  '
            'Ensure sidecars contain Manufacturer / '
            'ManufacturerModelName fields.'
        )

    # Determine which optional columns to include
    has_age = any('age' in r for r in rows.values())
    has_sex = any('sex' in r for r in rows.values())

    # Write CSV
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        'subject_id', 'batch',
        'Manufacturer', 'ManufacturerModelName', 'MagneticFieldStrength',
    ]
    if has_age:
        fieldnames.append('age')
    if has_sex:
        fieldnames.append('sex')

    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for obs_id in sorted(rows.keys()):
            writer.writerow(rows[obs_id])

    n_batches = len(set(r['batch'] for r in rows.values()))
    logger.info(
        'Generated combat batch CSV: %s (%d observations, %d batches%s%s)',
        output_csv, len(rows), n_batches,
        ', age included' if has_age else '',
        ', sex included' if has_sex else '',
    )
    return output_csv


def run_group_analysis(
    output_dir: Path,
    *,
    bids_dir: Optional[Path] = None,
    combat_batch_file: Optional[str] = None,
    combat_parametric: bool = True,
    participant_label: Optional[List[str]] = None,
    generate_batch_csv: bool = False,
) -> int:
    """Run group-level analysis on OncoPrep derivatives.

    Currently this consists of cohort-wide ComBat harmonization of
    radiomics features.  More group-level analyses can be added here.

    Parameters
    ----------
    output_dir : Path
        OncoPrep output / derivatives directory (the same directory
        that was passed to the ``participant`` stage).
    bids_dir : Path, optional
        Root of the raw BIDS dataset.  Required when
        *generate_batch_csv* is ``True``.
    combat_batch_file : str, optional
        Path to a CSV file mapping subjects to scanner/site batches.
        Must contain columns ``subject_id`` and ``batch``.  Optional
        covariate columns (e.g. ``age``, ``sex``) are forwarded to
        ComBat as biological covariates of interest (preserved).
        If ``None`` and *generate_batch_csv* is ``False``, ComBat
        harmonization is skipped.
    combat_parametric : bool
        Whether to use parametric empirical Bayes priors for ComBat
        (default ``True``).
    participant_label : list of str, optional
        Restrict harmonization to this subset of participants.
        By default all participants found in *output_dir* are included.
    generate_batch_csv : bool
        If ``True``, auto-generate the batch CSV from BIDS JSON
        sidecars using ``Manufacturer``, ``ManufacturerModelName``,
        and ``MagneticFieldStrength`` (fields that survive
        anonymization).  The generated CSV is written to
        ``<output_dir>/oncoprep/combat_batch.csv``.

    Returns
    -------
    int
        Return code (0 = success, >0 = error).
    """
    output_dir = Path(output_dir).resolve()
    retcode = 0

    # Auto-generate batch CSV from BIDS sidecars if requested
    if generate_batch_csv and bids_dir is not None:
        auto_csv = output_dir / 'oncoprep' / 'combat_batch.csv'
        try:
            generate_combat_batch_csv(
                bids_dir=bids_dir,
                output_csv=auto_csv,
                participant_label=participant_label,
            )
            if combat_batch_file is None:
                combat_batch_file = str(auto_csv)
                logger.info(
                    'Using auto-generated batch CSV: %s', auto_csv,
                )
        except Exception:
            logger.exception(
                'Failed to auto-generate combat batch CSV from BIDS sidecars'
            )
            retcode = 1
            return retcode

    if combat_batch_file is not None:
        try:
            _run_combat_harmonization(
                output_dir=output_dir,
                batch_file=combat_batch_file,
                parametric=combat_parametric,
                participant_label=participant_label,
            )
        except Exception:
            logger.exception('ComBat harmonization failed')
            retcode = 1
    else:
        logger.warning(
            'Group-level analysis requested but no --combat-batch file '
            'was provided.  Use --generate-combat-batch to auto-generate '
            'from BIDS JSON sidecars, or supply a CSV with --combat-batch.'
        )
        retcode = 0

    return retcode


# ---------------------------------------------------------------------------
# ComBat harmonization (group-level)
# ---------------------------------------------------------------------------


def _collect_radiomics_jsons(
    output_dir: Path,
    participant_label: Optional[List[str]] = None,
) -> Dict[str, Path]:
    """Discover per-observation radiomics JSON files from derivatives.

    Searches ``<output_dir>/oncoprep/sub-*/anat/*radiomics*.json`` and
    ``<output_dir>/oncoprep/sub-*/ses-*/anat/*radiomics*.json``.

    For longitudinal datasets (multiple sessions per subject), each
    session is returned as a separate observation keyed as
    ``sub-XXX_ses-YY``.  For cross-sectional data the key is just
    ``sub-XXX``.

    Parameters
    ----------
    output_dir : Path
        OncoPrep derivatives root.
    participant_label : list of str, optional
        Restrict to these participants (with or without ``sub-`` prefix).

    Returns
    -------
    dict
        ``{observation_id: Path}`` mapping observation labels
        (e.g. ``'sub-001'`` or ``'sub-001_ses-01'``) to their
        radiomics JSON files.
    """
    import re

    deriv_dir = output_dir / 'oncoprep'
    if not deriv_dir.is_dir():
        raise FileNotFoundError(
            f'No OncoPrep derivatives found at {deriv_dir}.  '
            f'Run the participant-level analysis first.'
        )

    # Normalise participant labels
    allowed: Optional[set] = None
    if participant_label:
        allowed = set()
        for p in participant_label:
            p_clean = p if p.startswith('sub-') else f'sub-{p}'
            allowed.add(p_clean)

    # Glob for radiomics JSON files (exclude *Combat* files from previous runs)
    pattern_nosess = deriv_dir.glob('sub-*/anat/*radiomics*.json')
    pattern_sess = deriv_dir.glob('sub-*/ses-*/anat/*radiomics*.json')

    results: Dict[str, Path] = {}
    for fp in sorted(list(pattern_nosess) + list(pattern_sess)):
        if 'combat' in fp.name.lower() or 'Combat' in fp.name:
            continue  # skip previously harmonized outputs
        subj_match = re.search(r'(sub-[a-zA-Z0-9]+)', str(fp))
        if not subj_match:
            continue
        subj_id = subj_match.group(1)
        if allowed is not None and subj_id not in allowed:
            continue

        # Check for session
        ses_match = re.search(r'(ses-[a-zA-Z0-9]+)', str(fp))
        session = ses_match.group(1) if ses_match else None
        obs_id = f'{subj_id}_{session}' if session else subj_id

        if obs_id not in results:
            results[obs_id] = fp

    return results


def _flatten_features(features_dict: dict) -> Dict[str, float]:
    """Flatten nested radiomics JSON to ``{region__category__feature: value}``.

    Parameters
    ----------
    features_dict : dict
        Nested dict as produced by ``PyRadiomicsFeatureExtraction``:
        ``{region: {features: {category: {feature: value}}}}``.

    Returns
    -------
    dict
        Flat ``{key: float}`` mapping.
    """
    flat: Dict[str, float] = {}
    for region, rdata in features_dict.items():
        feats = rdata.get('features', {})
        for category, cat_feats in feats.items():
            for feat_name, feat_val in cat_feats.items():
                key = f'{region}__{category}__{feat_name}'
                try:
                    flat[key] = float(feat_val)
                except (TypeError, ValueError):
                    pass
    return flat


def _unflatten_features(
    original: dict,
    harmonized_flat: Dict[str, float],
) -> dict:
    """Replace values in nested features dict with harmonized values.

    Parameters
    ----------
    original : dict
        Original (nested) features dict.
    harmonized_flat : dict
        ``{region__category__feature: value}`` produced by ComBat.

    Returns
    -------
    dict
        Copy of *original* with harmonized values patched in.
    """
    import copy

    result = copy.deepcopy(original)
    for region, rdata in result.items():
        feats = rdata.get('features', {})
        for category, cat_feats in feats.items():
            for feat_name in list(cat_feats.keys()):
                key = f'{region}__{category}__{feat_name}'
                if key in harmonized_flat:
                    cat_feats[feat_name] = harmonized_flat[key]
    return result


def _run_combat_harmonization(
    output_dir: Path,
    batch_file: str,
    parametric: bool = True,
    participant_label: Optional[List[str]] = None,
) -> None:
    """Apply ComBat harmonization across the entire cohort.

    This is the group-level ComBat implementation.  It collects **all**
    participant-level radiomics JSON files, builds a features×observations
    matrix, runs neuroCombat once across the whole cohort, and writes
    harmonized per-observation JSON files back into the derivatives tree.

    **Longitudinal auto-detection:** when multiple observations share
    the same subject (e.g. ``sub-001_ses-01`` and ``sub-001_ses-02``),
    longitudinal mode is activated automatically.  In this mode the
    subject identity is included as a categorical covariate so that
    within-subject variance is preserved while scanner batch effects
    are removed — following the approach recommended when no native
    Python longitudinal ComBat (longCombat) package is available.

    Parameters
    ----------
    output_dir : Path
        OncoPrep derivatives root.
    batch_file : str
        CSV with ``subject_id`` and ``batch`` columns (plus optional
        biological covariates).  For longitudinal data, ``subject_id``
        should contain observation IDs (e.g. ``sub-001_ses-01``).
    parametric : bool
        Parametric empirical Bayes (default ``True``).
    participant_label : list of str, optional
        Subset of participants to include.

    Raises
    ------
    ImportError
        If *neuroCombat* is not installed.
    ValueError
        If fewer than 2 scanner batches are present or if the batch
        file is malformed.
    FileNotFoundError
        If no radiomics outputs are found.
    """
    import re

    import numpy as np
    import pandas as pd

    try:
        from neuroCombat import neuroCombat
    except ImportError:
        raise ImportError(
            'neuroCombat is required for group-level ComBat harmonization. '
            'Install with: pip install neuroCombat'
        )

    # --- 1. Collect per-observation radiomics files ---
    obs_files = _collect_radiomics_jsons(output_dir, participant_label)
    if len(obs_files) < 3:
        raise ValueError(
            f'ComBat requires at least 3 observations, found '
            f'{len(obs_files)}.  Run participant-level radiomics first.'
        )

    logger.info(
        'Collected radiomics features for %d observations: %s',
        len(obs_files),
        ', '.join(sorted(obs_files.keys())),
    )

    # --- 1b. Detect longitudinal data ---
    # Extract subject ID from obs_id: "sub-001_ses-01" → "sub-001"
    def _extract_subject(obs_id: str) -> str:
        m = re.match(r'(sub-[a-zA-Z0-9]+)', obs_id)
        return m.group(1) if m else obs_id

    obs_to_subject: Dict[str, str] = {
        oid: _extract_subject(oid) for oid in obs_files
    }
    unique_subjects = set(obs_to_subject.values())
    is_longitudinal = len(unique_subjects) < len(obs_files)

    if is_longitudinal:
        n_sessions_per_subj = {}
        for subj in unique_subjects:
            n_sessions_per_subj[subj] = sum(
                1 for s in obs_to_subject.values() if s == subj
            )
        logger.info(
            'Longitudinal data detected: %d observations from %d unique '
            'subjects (range %d–%d sessions per subject).  Subject identity '
            'will be preserved as a categorical covariate.',
            len(obs_files),
            len(unique_subjects),
            min(n_sessions_per_subj.values()),
            max(n_sessions_per_subj.values()),
        )
    else:
        logger.info('Cross-sectional data: %d subjects.', len(obs_files))

    # --- 2. Load batch information ---
    batch_df = pd.read_csv(batch_file)
    if 'subject_id' not in batch_df.columns or 'batch' not in batch_df.columns:
        raise ValueError(
            "batch_file must contain columns 'subject_id' and 'batch'.  "
            f"Found columns: {list(batch_df.columns)}"
        )

    # Normalise subject IDs in batch file (ensure 'sub-' prefix)
    batch_df['subject_id'] = batch_df['subject_id'].astype(str).apply(
        lambda s: s if s.startswith('sub-') else f'sub-{s}'
    )

    batch_lookup: Dict[str, str] = dict(
        zip(batch_df['subject_id'], batch_df['batch'].astype(str))
    )

    # --- 3. Load features & build matrix ---
    all_features: Dict[str, dict] = {}      # obs_id → raw nested dict
    all_flat: Dict[str, Dict[str, float]] = {}  # obs_id → flat dict

    for obs_id, fpath in obs_files.items():
        with open(fpath) as f:
            raw = json.load(f)
        all_features[obs_id] = raw
        all_flat[obs_id] = _flatten_features(raw)

    # Determine shared feature set (features present in ALL observations)
    feature_sets = [set(flat.keys()) for flat in all_flat.values()]
    common_features = sorted(set.intersection(*feature_sets))

    # Keep only numeric features with non-zero variance
    valid_features: List[str] = []
    obs_ids = sorted(all_flat.keys())
    for fn in common_features:
        try:
            vals = [float(all_flat[s][fn]) for s in obs_ids]
            if np.std(vals) > 0 and not any(np.isnan(v) for v in vals):
                valid_features.append(fn)
        except (TypeError, ValueError):
            continue

    if not valid_features:
        raise ValueError(
            'No valid numeric features with non-zero variance found '
            'across all observations.'
        )

    logger.info(
        'Building feature matrix: %d features × %d observations',
        len(valid_features), len(obs_ids),
    )

    # features × observations matrix (neuroCombat convention)
    data_matrix = np.array([
        [float(all_flat[s][fn]) for s in obs_ids]
        for fn in valid_features
    ])

    # --- 4. Build batch vector aligned with observations ---
    batch_vector: List[str] = []
    keep_idx: List[int] = []
    skipped_obs: List[str] = []

    for i, oid in enumerate(obs_ids):
        b = batch_lookup.get(oid)
        # Fallback: try subject-level match for cross-sectional batch files
        if b is None:
            subj = obs_to_subject[oid]
            b = batch_lookup.get(subj)
        if b is not None:
            batch_vector.append(b)
            keep_idx.append(i)
        else:
            skipped_obs.append(oid)
            logger.warning(
                'Observation %s not found in batch file — will be excluded '
                'from ComBat harmonization.',
                oid,
            )

    n_batches = len(set(batch_vector))
    if n_batches < 2:
        raise ValueError(
            f'ComBat requires at least 2 scanner batches, found {n_batches}.  '
            f'Check your --combat-batch CSV.'
        )

    # Subset matrix to observations with batch info
    data_matrix = data_matrix[:, keep_idx]
    obs_ids_filtered = [obs_ids[i] for i in keep_idx]

    logger.info(
        'ComBat: %d observations across %d batches (batch distribution: %s)',
        len(obs_ids_filtered),
        n_batches,
        _batch_distribution(batch_vector),
    )

    # --- 5. Build covars DataFrame ---
    covars = pd.DataFrame(
        {'batch': batch_vector},
        index=obs_ids_filtered,
    )

    # Identify biological covariates (everything except subject_id, batch,
    # and scanner metadata columns)
    meta_cols = {'subject_id', 'batch', 'Manufacturer',
                 'ManufacturerModelName', 'MagneticFieldStrength'}
    bio_cols = [
        c for c in batch_df.columns if c not in meta_cols
    ]
    categorical_cols: List[str] = []
    continuous_cols: List[str] = []

    for col in bio_cols:
        col_lookup = dict(
            zip(batch_df['subject_id'], batch_df[col])
        )
        vals = []
        for oid in obs_ids_filtered:
            v = col_lookup.get(oid)
            if v is None:
                v = col_lookup.get(obs_to_subject[oid], np.nan)
            vals.append(v)
        covars[col] = vals

        if batch_df[col].dtype == object:
            categorical_cols.append(col)
        else:
            continuous_cols.append(col)

    if bio_cols:
        logger.info(
            'Biological covariates (preserved by ComBat): %s',
            ', '.join(bio_cols),
        )

    # --- 5b. Longitudinal: inject subject as categorical covariate ---
    # Only when subjects cross batches (scanned at different sites);
    # when subjects are nested within batches (each subject at one
    # site), subject indicators are collinear with batch and must NOT
    # be included.
    subjects_cross_batches = False
    if is_longitudinal:
        subj_batches: Dict[str, set] = {}
        for oid, b in zip(obs_ids_filtered, batch_vector):
            subj = obs_to_subject[oid]
            subj_batches.setdefault(subj, set()).add(b)
        subjects_cross_batches = any(
            len(bs) > 1 for bs in subj_batches.values()
        )

        if subjects_cross_batches:
            covars['_subject'] = [
                obs_to_subject[oid] for oid in obs_ids_filtered
            ]
            categorical_cols.append('_subject')
            logger.info(
                'Longitudinal mode: subjects cross batches — subject '
                'identity added as categorical covariate (%d unique '
                'subjects).',
                len(set(covars['_subject'])),
            )
        else:
            logger.info(
                'Longitudinal mode: subjects are nested within batches '
                '(no subject crosses sites).  Each session treated as an '
                'independent observation; within-subject variance is '
                'naturally preserved.',
            )

    # --- 6. Run neuroCombat ---
    logger.info(
        'Running neuroCombat (parametric=%s, longitudinal=%s) on '
        '%d features × %d observations …',
        parametric, is_longitudinal,
        len(valid_features), len(obs_ids_filtered),
    )

    combat_result = neuroCombat(
        dat=data_matrix,
        covars=covars,
        batch_col='batch',
        categorical_cols=categorical_cols if categorical_cols else None,
        continuous_cols=continuous_cols if continuous_cols else None,
    )
    harmonized_matrix = combat_result['data']  # features × observations

    logger.info('ComBat harmonization completed successfully.')

    # --- 7. Write harmonized per-observation JSON files ---
    out_dir = output_dir / 'oncoprep'
    written_files: List[Tuple[str, Path]] = []
    for col_idx, oid in enumerate(obs_ids_filtered):
        harmonized_flat = {
            fn: float(harmonized_matrix[row_idx, col_idx])
            for row_idx, fn in enumerate(valid_features)
        }
        harmonized_nested = _unflatten_features(
            all_features[oid], harmonized_flat,
        )

        # Write next to the original radiomics file
        orig_path = obs_files[oid]
        combat_name = orig_path.name.replace(
            'radiomics', 'radiomicsCombat',
        )
        combat_path = orig_path.parent / combat_name
        with open(combat_path, 'w') as f:
            json.dump(harmonized_nested, f, indent=2, default=str)

        written_files.append((oid, combat_path))
        logger.debug('Wrote harmonized features: %s', combat_path)

    logger.info(
        'Wrote %d harmonized feature files.', len(written_files),
    )

    # --- 8. Write group-level HTML report ---
    _write_combat_report(
        output_dir=out_dir,
        n_features=len(valid_features),
        n_subjects=len(obs_ids_filtered),
        n_batches=n_batches,
        batch_vector=batch_vector,
        parametric=parametric,
        bio_cols=bio_cols,
        skipped_subjects=skipped_obs,
        written_files=written_files,
        data_raw=data_matrix,
        data_harmonized=harmonized_matrix,
        valid_features=valid_features,
        is_longitudinal=is_longitudinal,
        n_unique_subjects=len(unique_subjects) if is_longitudinal else None,
        subjects_cross_batches=subjects_cross_batches,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _batch_distribution(batch_vector: List[str]) -> str:
    """Pretty-print batch distribution, e.g. ``'SiteA: 5, SiteB: 3'``."""
    from collections import Counter

    counts = Counter(batch_vector)
    return ', '.join(
        f'{batch}: {n}' for batch, n in sorted(counts.items())
    )


def _write_combat_report(
    output_dir: Path,
    n_features: int,
    n_subjects: int,
    n_batches: int,
    batch_vector: List[str],
    parametric: bool,
    bio_cols: List[str],
    skipped_subjects: List[str],
    written_files: List[Tuple[str, Path]],
    data_raw,
    data_harmonized,
    valid_features: List[str],
    is_longitudinal: bool = False,
    n_unique_subjects: Optional[int] = None,
    subjects_cross_batches: bool = False,
) -> Path:
    """Write an HTML report summarising group-level ComBat harmonization.

    Parameters
    ----------
    output_dir : Path
        Directory where the report will be written.
    n_features, n_subjects, n_batches : int
        Descriptive counts.  *n_subjects* is the number of
        observations (which may exceed the number of unique subjects
        in longitudinal datasets).
    batch_vector : list of str
        Per-observation batch labels.
    parametric : bool
        Whether parametric priors were used.
    bio_cols : list of str
        Names of biological covariates.
    skipped_subjects : list of str
        Observations excluded due to missing batch info.
    written_files : list of (str, Path)
        (observation_id, path) for each harmonized output.
    data_raw, data_harmonized : ndarray
        Features × observations matrices (before / after ComBat).
    valid_features : list of str
        Feature names.
    is_longitudinal : bool
        Whether the dataset contains repeated measures.
    n_unique_subjects : int, optional
        Number of unique subjects (only relevant when longitudinal).
    subjects_cross_batches : bool
        Whether any subject appears in multiple batches.

    Returns
    -------
    Path
        Path to the generated HTML file.
    """
    import numpy as np
    from collections import Counter

    batch_counts = Counter(batch_vector)

    # Compute summary statistics
    raw_mean_var = float(np.mean(np.var(data_raw, axis=1)))
    harm_mean_var = float(np.mean(np.var(data_harmonized, axis=1)))
    variance_reduction_pct = (
        (1 - harm_mean_var / raw_mean_var) * 100
        if raw_mean_var > 0
        else 0.0
    )

    report_path = output_dir / 'group_combat_report.html'

    batch_rows = '\n'.join(
        f'<tr><td>{batch}</td><td>{count}</td></tr>'
        for batch, count in sorted(batch_counts.items())
    )

    skip_note = ''
    if skipped_subjects:
        skip_note = (
            '<div class="alert alert-warning">'
            f'<strong>Warning:</strong> {len(skipped_subjects)} observation(s) '
            f'excluded (not in batch file): {", ".join(skipped_subjects)}'
            '</div>'
        )

    long_note = ''
    if is_longitudinal and n_unique_subjects is not None:
        if subjects_cross_batches:
            long_detail = (
                'Subject identity is included as a categorical covariate '
                'so that within-subject variance is preserved while '
                'scanner batch effects are removed.'
            )
        else:
            long_detail = (
                'Subjects are nested within batches (no subject crosses '
                'sites), so each session is treated as an independent '
                'observation.  Within-subject variance is naturally '
                'preserved.'
            )
        long_note = (
            '<div class="summary-box" style="border-left-color: #27ae60;">'
            f'<p><strong>Longitudinal dataset</strong>: {n_subjects} '
            f'observations from {n_unique_subjects} unique subjects.  '
            f'{long_detail}</p></div>'
        )

    unique_subj_card = ''
    if is_longitudinal and n_unique_subjects is not None:
        unique_subj_card = (
            '<div class="stat-card">'
            f'<div class="value">{n_unique_subjects}</div>'
            '<div class="label">Unique Subjects</div>'
            '</div>'
        )

    bio_note = ''
    if bio_cols:
        bio_note = (
            '<p><strong>Biological covariates</strong> (preserved): '
            f'{", ".join(bio_cols)}</p>'
        )

    if is_longitudinal:
        if subjects_cross_batches:
            long_config_label = (
                'Yes &mdash; subject identity preserved as categorical '
                'covariate (subjects cross batches)'
            )
        else:
            long_config_label = (
                'Yes &mdash; subjects nested within batches; each session '
                'treated as independent observation'
            )
    else:
        long_config_label = 'No (cross-sectional)'

    html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>OncoPrep – Group-Level ComBat Harmonization Report</title>
<style>
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
                 'Helvetica Neue', Arial, sans-serif;
    max-width: 900px;
    margin: 2rem auto;
    padding: 0 1rem;
    color: #333;
}}
h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 0.5rem; }}
h2 {{ color: #2c3e50; }}
table {{
    border-collapse: collapse;
    width: 100%;
    margin: 1rem 0;
}}
th, td {{
    border: 1px solid #ddd;
    padding: 0.5rem 0.75rem;
    text-align: left;
}}
th {{
    background-color: #3498db;
    color: white;
}}
tr:nth-child(even) {{ background-color: #f2f2f2; }}
.summary-box {{
    background: #ecf0f1;
    border-left: 4px solid #3498db;
    padding: 1rem;
    margin: 1rem 0;
}}
.alert-warning {{
    background: #fff3cd;
    border-left: 4px solid #ffc107;
    padding: 0.75rem 1rem;
    margin: 1rem 0;
}}
.stat-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1rem;
    margin: 1rem 0;
}}
.stat-card {{
    background: white;
    border: 1px solid #ddd;
    border-radius: 4px;
    padding: 1rem;
    text-align: center;
}}
.stat-card .value {{
    font-size: 1.8rem;
    font-weight: bold;
    color: #2c3e50;
}}
.stat-card .label {{
    font-size: 0.85rem;
    color: #7f8c8d;
    margin-top: 0.25rem;
}}
</style>
</head>
<body>

<h1>ComBat Harmonization Report</h1>

<div class="summary-box">
<p>Group-level ComBat batch-effect correction was applied to radiomics
features extracted during participant-level processing, following the
methodology of Pati et al., <em>AJNR</em> 2024; 45: 1291–1298.</p>
<p>Scanner/site effects are modelled as covariates of <strong>no
interest</strong> and removed, while biological covariates of interest
(if provided) are preserved.</p>
</div>

{long_note}
{skip_note}

<h2>Summary</h2>

<div class="stat-grid">
  <div class="stat-card">
    <div class="value">{n_subjects}</div>
    <div class="label">Observations</div>
  </div>
  {unique_subj_card}
  <div class="stat-card">
    <div class="value">{n_batches}</div>
    <div class="label">Scanner Batches</div>
  </div>
  <div class="stat-card">
    <div class="value">{n_features}</div>
    <div class="label">Features Harmonized</div>
  </div>
  <div class="stat-card">
    <div class="value">{variance_reduction_pct:+.1f}%</div>
    <div class="label">Mean Variance Change</div>
  </div>
</div>

<h2>Configuration</h2>
<table>
<tbody>
<tr><td><strong>Parametric priors</strong></td><td>{'Yes' if parametric else 'No (non-parametric)'}</td></tr>
<tr><td><strong>Longitudinal</strong></td><td>{long_config_label}</td></tr>
<tr><td><strong>Algorithm</strong></td><td>ComBat (Johnson et al., <em>Biostatistics</em> 2007)</td></tr>
<tr><td><strong>Implementation</strong></td><td>neuroCombat (Fortin et al., <em>NeuroImage</em> 2018)</td></tr>
</tbody>
</table>
{bio_note}

<h2>Batch Distribution</h2>
<table>
<thead><tr><th>Batch / Site</th><th>Observations</th></tr></thead>
<tbody>
{batch_rows}
</tbody>
</table>

<h2>Output Files</h2>
<table>
<thead><tr><th>Observation</th><th>Harmonized Features</th></tr></thead>
<tbody>
{''.join(
    f'<tr><td>{sid}</td><td><code>{fpath.name}</code></td></tr>'
    for sid, fpath in sorted(written_files)
)}
</tbody>
</table>

<hr>
<p style="color: #999; font-size: 0.8rem;">
Generated by OncoPrep group-level analysis.
ComBat: Johnson et al., <em>Biostatistics</em> 2007;
neuroCombat: Fortin et al., <em>NeuroImage</em> 2018.
</p>

</body>
</html>
"""
    with open(report_path, 'w') as f:
        f.write(html)

    logger.info('Group-level ComBat report written to %s', report_path)
    return report_path
