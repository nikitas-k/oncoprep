"""Segmentation metrics: patient-level, lesion-level, and surface-based.

Implements all metric computations required by Phases B–D:
    - Dice coefficient (binary & multi-class)
    - Hausdorff distance 95th percentile (HD95)
    - Surface Dice at configurable tolerance
    - Lesion-wise metrics (connected-component level)
    - BraTS-2024 lesion-wise metrics (dilation-based matching, aggregated scores)
    - Volume extraction in cm³

BraTS-2024 lesion-wise metrics follow:
    Saluja et al., BraTS-2024-Metrics (https://github.com/rachitsaluja/BraTS-2024-Metrics)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from scipy import ndimage

from .config import REGION_MAP


# ---------------------------------------------------------------------------
# Core binary metrics
# ---------------------------------------------------------------------------


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Binary Dice coefficient."""
    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)
    intersection = np.sum(pred_bool & gt_bool)
    total = np.sum(pred_bool) + np.sum(gt_bool)
    if total == 0:
        return 1.0 if np.sum(pred_bool) == 0 and np.sum(gt_bool) == 0 else 0.0
    return float(2.0 * intersection / total)


def hausdorff_distance_95(
    pred: np.ndarray,
    gt: np.ndarray,
    voxel_spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
) -> float:
    """95th-percentile Hausdorff distance (mm).

    Returns ``inf`` if either mask is empty.
    """
    from scipy.ndimage import distance_transform_edt

    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)

    if not pred_bool.any() or not gt_bool.any():
        return float("inf")

    # Distance from pred boundary to nearest gt surface
    pred_boundary = pred_bool ^ ndimage.binary_erosion(pred_bool)
    gt_boundary = gt_bool ^ ndimage.binary_erosion(gt_bool)

    if not pred_boundary.any() or not gt_boundary.any():
        return float("inf")

    dt_gt = distance_transform_edt(~gt_boundary, sampling=voxel_spacing)
    dt_pred = distance_transform_edt(~pred_boundary, sampling=voxel_spacing)

    d_pred_to_gt = dt_gt[pred_boundary]
    d_gt_to_pred = dt_pred[gt_boundary]

    all_distances = np.concatenate([d_pred_to_gt, d_gt_to_pred])
    return float(np.percentile(all_distances, 95))


def surface_dice(
    pred: np.ndarray,
    gt: np.ndarray,
    tolerance_mm: float = 1.0,
    voxel_spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
) -> float:
    """Normalised surface Dice (NSD) at a given tolerance (mm).

    Returns NaN if both masks are empty.
    """
    from scipy.ndimage import distance_transform_edt

    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)

    if not pred_bool.any() and not gt_bool.any():
        return float("nan")
    if not pred_bool.any() or not gt_bool.any():
        return 0.0

    pred_boundary = pred_bool ^ ndimage.binary_erosion(pred_bool)
    gt_boundary = gt_bool ^ ndimage.binary_erosion(gt_bool)

    if not pred_boundary.any() or not gt_boundary.any():
        return 0.0

    dt_gt = distance_transform_edt(~gt_boundary, sampling=voxel_spacing)
    dt_pred = distance_transform_edt(~pred_boundary, sampling=voxel_spacing)

    pred_within = np.sum(dt_gt[pred_boundary] <= tolerance_mm)
    gt_within = np.sum(dt_pred[gt_boundary] <= tolerance_mm)

    total_surface = np.sum(pred_boundary) + np.sum(gt_boundary)
    if total_surface == 0:
        return 0.0
    return float((pred_within + gt_within) / total_surface)


# ---------------------------------------------------------------------------
# Region-level helpers
# ---------------------------------------------------------------------------


def extract_region(seg: np.ndarray, labels: List[int]) -> np.ndarray:
    """Create binary mask from a union of integer labels."""
    mask = np.zeros_like(seg, dtype=bool)
    for lbl in labels:
        mask |= (seg == lbl)
    return mask.astype(np.uint8)


def compute_patient_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    voxel_spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
    regions: Optional[Dict[str, List[int]]] = None,
    surface_tolerances: Optional[List[float]] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute patient-level metrics for all regions.

    Parameters
    ----------
    pred, gt : ndarray
        Multi-label segmentation arrays (integer labels).
    voxel_spacing : tuple of float
        Voxel dimensions in mm.
    regions : dict
        Map of region name → list of labels. Defaults to ``REGION_MAP``.
    surface_tolerances : list of float
        Tolerances for surface Dice (mm). Defaults to [1.0, 2.0].

    Returns
    -------
    dict[region_name, dict[metric_name, value]]
    """
    if regions is None:
        regions = REGION_MAP
    if surface_tolerances is None:
        surface_tolerances = [1.0, 2.0]

    results: Dict[str, Dict[str, float]] = {}
    for region_name, label_list in regions.items():
        pred_mask = extract_region(pred, label_list)
        gt_mask = extract_region(gt, label_list)

        metrics: Dict[str, float] = {
            "dice": dice_score(pred_mask, gt_mask),
            "hd95": hausdorff_distance_95(pred_mask, gt_mask, voxel_spacing),
        }
        for tol in surface_tolerances:
            metrics[f"surface_dice_{tol}mm"] = surface_dice(
                pred_mask, gt_mask, tol, voxel_spacing
            )
        results[region_name] = metrics

    return results


# ---------------------------------------------------------------------------
# Lesion-wise metrics (connected-component level)
# ---------------------------------------------------------------------------


def _get_connected_components(
    binary_mask: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """Label connected components, return (label_map, n_components)."""
    structure = ndimage.generate_binary_structure(3, 2)  # 18-connectivity
    labeled, n = ndimage.label(binary_mask.astype(bool), structure=structure)
    return labeled, n


def lesion_volume_cc(
    binary_mask: np.ndarray,
    voxel_spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
) -> float:
    """Volume of a binary mask in cm³ (cc)."""
    voxel_vol_mm3 = float(np.prod(voxel_spacing))
    return float(np.sum(binary_mask > 0) * voxel_vol_mm3 / 1000.0)


def compute_lesion_wise_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    voxel_spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
    overlap_threshold: float = 0.0,
) -> Dict[str, object]:
    """Lesion-wise evaluation following BraTS-2023-Metrics conventions.

    For each ground-truth connected component (lesion), finds the best
    overlapping predicted component and computes metrics.  Also reports
    false-positive predicted lesions (no GT overlap).

    Parameters
    ----------
    pred, gt : ndarray
        Binary masks (0/1).
    voxel_spacing : tuple
        Voxel size in mm.
    overlap_threshold : float
        Minimum overlap (Dice) to count as a match.

    Returns
    -------
    dict with keys:
        gt_lesions : list of dicts (per-GT-lesion metrics)
        fp_lesions : list of dicts (false-positive predicted lesions)
        detection_sensitivity : float
        detection_precision : float
    """
    gt_labeled, n_gt = _get_connected_components(gt)
    pred_labeled, n_pred = _get_connected_components(pred)

    gt_lesions = []
    matched_pred_ids = set()

    for gt_id in range(1, n_gt + 1):
        gt_mask = (gt_labeled == gt_id).astype(np.uint8)
        vol_cc = lesion_volume_cc(gt_mask, voxel_spacing)

        # Find overlapping predicted components
        overlapping_pred_ids = set(pred_labeled[gt_mask.astype(bool)]) - {0}

        if not overlapping_pred_ids:
            # Missed lesion
            gt_lesions.append({
                "gt_id": gt_id,
                "volume_cc": vol_cc,
                "matched": False,
                "dice": 0.0,
                "hd95": float("inf"),
            })
            continue

        # Merge all overlapping pred components
        merged_pred = np.zeros_like(pred, dtype=np.uint8)
        for pid in overlapping_pred_ids:
            merged_pred[pred_labeled == pid] = 1

        d = dice_score(merged_pred, gt_mask)
        if d <= overlap_threshold:
            gt_lesions.append({
                "gt_id": gt_id,
                "volume_cc": vol_cc,
                "matched": False,
                "dice": d,
                "hd95": hausdorff_distance_95(merged_pred, gt_mask, voxel_spacing),
            })
            continue

        matched_pred_ids.update(overlapping_pred_ids)
        gt_lesions.append({
            "gt_id": gt_id,
            "volume_cc": vol_cc,
            "matched": True,
            "dice": d,
            "hd95": hausdorff_distance_95(merged_pred, gt_mask, voxel_spacing),
        })

    # False-positive lesions
    fp_lesions = []
    for pred_id in range(1, n_pred + 1):
        if pred_id not in matched_pred_ids:
            pred_mask = (pred_labeled == pred_id).astype(np.uint8)
            fp_lesions.append({
                "pred_id": pred_id,
                "volume_cc": lesion_volume_cc(pred_mask, voxel_spacing),
            })

    n_detected = sum(1 for gl in gt_lesions if gl["matched"])
    sensitivity = n_detected / max(n_gt, 1)
    precision = n_detected / max(n_detected + len(fp_lesions), 1)

    return {
        "gt_lesions": gt_lesions,
        "fp_lesions": fp_lesions,
        "detection_sensitivity": sensitivity,
        "detection_precision": precision,
        "n_gt_lesions": n_gt,
        "n_pred_lesions": n_pred,
    }


# ---------------------------------------------------------------------------
# BraTS-2024 lesion-wise metrics
# (dilation-based matching, aggregated lesion-wise scores)
# Adapted from: https://github.com/rachitsaluja/BraTS-2024-Metrics
# ---------------------------------------------------------------------------

# Default lesion-volume thresholds (mm³) — lesions smaller than this
# are excluded from final aggregated counts.
BRATS2024_VOLUME_THRESH: Dict[str, float] = {
    "ET": 10.0,
    "TC": 20.0,
    "WT": 20.0,
    "RC": 20.0,
}

# Default dilation factor (iterations) used to merge nearby CCs
# and to create a tolerance zone for lesion matching.
BRATS2024_DILATION_FACTOR: int = 2


def _cc_26(binary_mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """Connected components with full 26-connectivity (3 × 3 × 3 cube).

    Uses ``scipy.ndimage.label`` with ``generate_binary_structure(3, 3)``
    to replicate the ``cc3d.connected_components(mask, connectivity=26)``
    behaviour from BraTS-2024-Metrics without requiring the cc3d package.
    """
    struct = ndimage.generate_binary_structure(3, 3)
    labeled, n = ndimage.label(binary_mask.astype(bool), structure=struct)
    return labeled.astype(np.int32), int(n)


def _combine_cc_by_dilation(
    mask: np.ndarray,
    dil_factor: int,
) -> np.ndarray:
    """Group nearby connected components by dilating then re-labelling.

    Reproduces ``get_GTseg_combinedByDilation`` / ``get_Predseg_combinedByDilation``
    from BraTS-2024-Metrics: dilate the binary mask → CC on dilated →
    for each dilated CC, assign original voxels the dilated CC id.
    This merges fragments that are within *dil_factor* voxels of each other.
    """
    if not mask.any():
        return np.zeros_like(mask, dtype=np.int32)

    dilation_struct = ndimage.generate_binary_structure(3, 1)  # 6-conn for dilation

    # Original CC labels (26-conn)
    cc_orig, _ = _cc_26(mask)

    # Dilated mask and its CC labels
    dilated = ndimage.binary_dilation(
        mask.astype(bool), structure=dilation_struct, iterations=dil_factor,
    )
    cc_dil, _ = _cc_26(dilated)

    # Build mapping: original CC id → dilated CC id (majority vote)
    combined = np.zeros_like(mask, dtype=np.int32)
    for orig_id in range(1, cc_orig.max() + 1):
        orig_voxels = cc_orig == orig_id
        dil_ids = cc_dil[orig_voxels]
        dil_ids = dil_ids[dil_ids > 0]
        if len(dil_ids) == 0:
            continue
        # Assign the most frequent dilated id
        vals, counts = np.unique(dil_ids, return_counts=True)
        best_dil = vals[np.argmax(counts)]
        combined[orig_voxels] = best_dil

    # Re-label from 1..N for cleanliness
    unique_ids = np.unique(combined[combined > 0])
    relabel = np.zeros(combined.max() + 1, dtype=np.int32)
    for new_id, old_id in enumerate(unique_ids, start=1):
        relabel[old_id] = new_id
    combined = relabel[combined]

    return combined


def _get_tissue_seg(
    pred: np.ndarray,
    gt: np.ndarray,
    tissue_type: str,
    regions: Optional[Dict[str, List[int]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract binary masks for a BraTS tissue type (ET, TC, WT, RC).

    Mirrors ``get_TissueWiseSeg`` from BraTS-2024-Metrics.
    """
    if regions is None:
        regions = REGION_MAP
    labels = regions[tissue_type]
    pred_bin = extract_region(pred, labels)
    gt_bin = extract_region(gt, labels)
    return pred_bin, gt_bin


def _image_diagonal_mm(shape: Tuple[int, ...], spacing: Tuple[float, ...]) -> float:
    """Compute the image diagonal in mm (used as max HD95 penalty for FPs)."""
    return float(math.sqrt(sum((s * d) ** 2 for s, d in zip(shape, spacing))))


def compute_brats2024_lesion_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    voxel_spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
    tissue_type: str = "WT",
    dil_factor: int = BRATS2024_DILATION_FACTOR,
    volume_thresh_mm3: Optional[float] = None,
    regions: Optional[Dict[str, List[int]]] = None,
) -> Dict[str, object]:
    """BraTS-2024 lesion-wise metrics for a single tissue type.

    Implements the full BraTS-2024-Metrics protocol:
        1. Tissue-wise binary segmentation extraction
        2. 26-connectivity CC with dilation-based grouping
        3. Dilation-based lesion matching (GT dilated → find overlapping pred CCs)
        4. Per-lesion Dice, HD95, NSD@0.5mm, NSD@1.0mm
        5. Aggregated LesionWise scores (FP-penalised denominator)
        6. Legacy (full-volume) metrics for comparison
        7. Volume-threshold filtering of small lesions

    Parameters
    ----------
    pred, gt : ndarray
        Multi-label segmentation arrays (integer labels).
    voxel_spacing : tuple of float
        Voxel dimensions in mm.
    tissue_type : str
        Region key (``ET``, ``TC``, ``WT``, ``RC``).
    dil_factor : int
        Dilation iterations for CC grouping and matching.
    volume_thresh_mm3 : float or None
        Minimum lesion volume (mm³) to include in aggregated scores.
        If *None*, uses ``BRATS2024_VOLUME_THRESH[tissue_type]``.
    regions : dict or None
        Custom region → label mapping.  Defaults to ``REGION_MAP``.

    Returns
    -------
    dict with keys:

        Per-lesion detail
        ~~~~~~~~~~~~~~~~~
        lesion_pairs : list of dict
            One entry per ground-truth lesion (post-grouping) with keys
            ``gt_id``, ``gt_vol_mm3``, ``matched``, ``pred_ids``,
            ``dice``, ``hd95``, ``nsd_05``, ``nsd_10``.
        fp_lesions : list of dict
            One entry per unmatched predicted lesion.

        Aggregated BraTS-2024 scores
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        lesionwise_dice : float
        lesionwise_hd95 : float
        lesionwise_nsd_05 : float
        lesionwise_nsd_10 : float

        Legacy (full-volume) metrics
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        legacy_dice : float
        legacy_hd95 : float
        legacy_nsd_05 : float
        legacy_nsd_10 : float

        Detection counts
        ~~~~~~~~~~~~~~~~
        num_tp, num_fp, num_fn : int
        sensitivity, specificity : float
        gt_total_volume_mm3, pred_total_volume_mm3 : float
    """
    if volume_thresh_mm3 is None:
        volume_thresh_mm3 = BRATS2024_VOLUME_THRESH.get(tissue_type, 20.0)

    # 1. Extract binary masks for the tissue type
    pred_bin, gt_bin = _get_tissue_seg(pred, gt, tissue_type, regions)

    # 2. Legacy (full-volume) metrics
    legacy_dice = dice_score(pred_bin, gt_bin)
    legacy_hd95 = hausdorff_distance_95(pred_bin, gt_bin, voxel_spacing)
    legacy_nsd_05 = surface_dice(pred_bin, gt_bin, 0.5, voxel_spacing)
    legacy_nsd_10 = surface_dice(pred_bin, gt_bin, 1.0, voxel_spacing)

    gt_total_vol = float(np.sum(gt_bin > 0)) * float(np.prod(voxel_spacing))
    pred_total_vol = float(np.sum(pred_bin > 0)) * float(np.prod(voxel_spacing))

    # Handle both-empty case
    if not gt_bin.any() and not pred_bin.any():
        return {
            "lesion_pairs": [],
            "fp_lesions": [],
            "lesionwise_dice": 1.0,
            "lesionwise_hd95": 0.0,
            "lesionwise_nsd_05": 1.0,
            "lesionwise_nsd_10": 1.0,
            "legacy_dice": 1.0,
            "legacy_hd95": 0.0,
            "legacy_nsd_05": 1.0,
            "legacy_nsd_10": 1.0,
            "num_tp": 0,
            "num_fp": 0,
            "num_fn": 0,
            "sensitivity": 1.0,
            "specificity": 1.0,
            "gt_total_volume_mm3": 0.0,
            "pred_total_volume_mm3": 0.0,
        }

    # 3. Dilation-based CC grouping (26-connectivity)
    dilation_struct = ndimage.generate_binary_structure(3, 1)

    gt_cc = _combine_cc_by_dilation(gt_bin, dil_factor)
    pred_cc, _ = _cc_26(pred_bin)

    # Also combine pred CCs by dilation (mirrors save_tmp_files logic)
    pred_cc_grouped = _combine_cc_by_dilation(pred_bin, dil_factor)

    # Remove tiny predicted CCs (volume threshold on prediction side)
    for pid in range(1, int(pred_cc_grouped.max()) + 1):
        vol_mm3 = float(np.sum(pred_cc_grouped == pid)) * float(np.prod(voxel_spacing))
        if vol_mm3 <= volume_thresh_mm3:
            pred_cc_grouped[pred_cc_grouped == pid] = 0

    # Relabel pred after filtering
    unique_pred = np.unique(pred_cc_grouped[pred_cc_grouped > 0])
    remap = np.zeros(int(pred_cc_grouped.max()) + 1, dtype=np.int32)
    for new_id, old_id in enumerate(unique_pred, start=1):
        remap[old_id] = new_id
    pred_label_cc = remap[pred_cc_grouped]

    gt_label_cc = gt_cc

    n_gt_cc = int(gt_label_cc.max()) if gt_label_cc.any() else 0
    n_pred_cc = int(pred_label_cc.max()) if pred_label_cc.any() else 0

    # 4. Lesion-by-lesion matching
    tp_pred_ids: List[int] = []
    gt_tp_ids: List[int] = []
    fn_ids: List[int] = []
    lesion_pairs: List[Dict] = []

    for gt_id in range(1, n_gt_cc + 1):
        gt_mask = (gt_label_cc == gt_id).astype(np.uint8)
        gt_vol_mm3 = float(np.sum(gt_mask > 0)) * float(np.prod(voxel_spacing))

        # Dilate GT lesion to create tolerance zone
        gt_dilated = ndimage.binary_dilation(
            gt_mask.astype(bool), structure=dilation_struct, iterations=dil_factor,
        )

        # Find predicted CCs overlapping with dilated GT lesion
        pred_in_dil = pred_label_cc[gt_dilated]
        intersecting = np.unique(pred_in_dil)
        intersecting = intersecting[intersecting != 0].tolist()

        if not intersecting:
            # Missed lesion (FN)
            fn_ids.append(gt_id)
            lesion_pairs.append({
                "gt_id": gt_id,
                "gt_vol_mm3": gt_vol_mm3,
                "matched": False,
                "pred_ids": [],
                "dice": 0.0,
                "hd95": float("inf"),
                "nsd_05": 0.0,
                "nsd_10": 0.0,
            })
            continue

        # Record TPs
        for cc in intersecting:
            tp_pred_ids.append(cc)
        gt_tp_ids.append(gt_id)

        # Isolate matched predicted components and binarise
        pred_matched = np.zeros_like(pred_label_cc, dtype=np.uint8)
        for pid in intersecting:
            pred_matched[pred_label_cc == pid] = 1

        # Per-lesion metrics
        d = dice_score(pred_matched, gt_mask)
        hd = hausdorff_distance_95(pred_matched, gt_mask, voxel_spacing)

        # NSD @ 0.5mm and 1.0mm
        if pred_matched.any() and gt_mask.any():
            nsd_05 = surface_dice(pred_matched, gt_mask, 0.5, voxel_spacing)
            nsd_10 = surface_dice(pred_matched, gt_mask, 1.0, voxel_spacing)
        elif not pred_matched.any() and not gt_mask.any():
            nsd_05, nsd_10 = 1.0, 1.0
        else:
            nsd_05, nsd_10 = 0.0, 0.0

        lesion_pairs.append({
            "gt_id": gt_id,
            "gt_vol_mm3": gt_vol_mm3,
            "matched": True,
            "pred_ids": intersecting,
            "dice": d,
            "hd95": hd,
            "nsd_05": nsd_05,
            "nsd_10": nsd_10,
        })

    # 5. Identify FP predicted lesions (not matched to any GT)
    tp_set = set(tp_pred_ids)
    fp_ids = [pid for pid in range(1, n_pred_cc + 1) if pid not in tp_set]
    fp_lesion_list = []
    for pid in fp_ids:
        pred_mask = (pred_label_cc == pid).astype(np.uint8)
        fp_lesion_list.append({
            "pred_id": pid,
            "volume_mm3": float(np.sum(pred_mask > 0)) * float(np.prod(voxel_spacing)),
        })

    # 6. Volume-threshold filtering for aggregated scores
    # Exclude small GT lesions from counts (matches BraTS-2024 logic)
    fn_sub = sum(
        1 for lp in lesion_pairs
        if not lp["matched"] and lp["gt_vol_mm3"] <= volume_thresh_mm3
    )
    gt_tp_sub = sum(
        1 for lp in lesion_pairs
        if lp["matched"] and lp["gt_vol_mm3"] <= volume_thresh_mm3
    )

    # Filter to only lesions above volume threshold for metric aggregation
    pairs_above_thresh = [
        lp for lp in lesion_pairs if lp["gt_vol_mm3"] > volume_thresh_mm3
    ]
    matched_above = [lp for lp in pairs_above_thresh if lp["matched"]]
    n_fp = len(fp_ids)

    # 7. Aggregated lesion-wise scores (BraTS-2024 formula)
    # LW_Dice = sum(dice_per_matched_gt) / (n_matched_above_thresh + n_fp)
    # LW_HD95 = (sum(hd95_per_matched) + n_fp * diag) / (n_matched + n_fp)
    diag_mm = _image_diagonal_mm(pred.shape[:3], voxel_spacing)

    denom = len(matched_above) + n_fp

    if denom == 0:
        lw_dice = 1.0
        lw_hd95 = 0.0
        lw_nsd_05 = 1.0
        lw_nsd_10 = 1.0
    else:
        sum_dice = sum(lp["dice"] for lp in matched_above)
        lw_dice = sum_dice / denom

        # Replace inf HD95 with image diagonal (BraTS-2024 convention)
        sum_hd95 = sum(
            lp["hd95"] if not math.isinf(lp["hd95"]) else diag_mm
            for lp in matched_above
        )
        sum_hd95 += n_fp * diag_mm
        lw_hd95 = sum_hd95 / denom

        sum_nsd_05 = sum(lp["nsd_05"] for lp in matched_above)
        lw_nsd_05 = sum_nsd_05 / denom

        sum_nsd_10 = sum(lp["nsd_10"] for lp in matched_above)
        lw_nsd_10 = sum_nsd_10 / denom

    # Handle NaN → default (BraTS-2024 convention)
    if math.isnan(lw_dice):
        lw_dice = 1.0
    if math.isnan(lw_nsd_05):
        lw_nsd_05 = 1.0
    if math.isnan(lw_nsd_10):
        lw_nsd_10 = 1.0
    if math.isnan(lw_hd95):
        lw_hd95 = 0.0

    # 8. Sensitivity / specificity
    num_tp = len(gt_tp_ids) - gt_tp_sub
    num_fn = len(fn_ids) - fn_sub
    full_sens = num_tp / max(num_tp + num_fn, 1)
    full_spec = _specificity(pred_bin, gt_bin)

    return {
        # Per-lesion detail
        "lesion_pairs": lesion_pairs,
        "fp_lesions": fp_lesion_list,
        # Aggregated BraTS-2024 scores
        "lesionwise_dice": lw_dice,
        "lesionwise_hd95": lw_hd95,
        "lesionwise_nsd_05": lw_nsd_05,
        "lesionwise_nsd_10": lw_nsd_10,
        # Legacy (full-volume) metrics
        "legacy_dice": legacy_dice,
        "legacy_hd95": legacy_hd95,
        "legacy_nsd_05": legacy_nsd_05,
        "legacy_nsd_10": legacy_nsd_10,
        # Detection counts
        "num_tp": num_tp,
        "num_fp": n_fp,
        "num_fn": num_fn,
        "sensitivity": full_sens,
        "specificity": full_spec,
        "gt_total_volume_mm3": gt_total_vol,
        "pred_total_volume_mm3": pred_total_vol,
    }


def _specificity(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    """Whole-volume specificity: TN / (TN + FP)."""
    pred_b = pred_bin.astype(bool)
    gt_b = gt_bin.astype(bool)
    tn = int(np.sum(~pred_b & ~gt_b))
    fp = int(np.sum(pred_b & ~gt_b))
    return tn / max(tn + fp, 1)


def compute_brats2024_all_regions(
    pred: np.ndarray,
    gt: np.ndarray,
    voxel_spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
    regions: Optional[Dict[str, List[int]]] = None,
    dil_factor: int = BRATS2024_DILATION_FACTOR,
    volume_thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Dict[str, object]]:
    """Run BraTS-2024 lesion-wise metrics across all tissue types.

    Parameters
    ----------
    pred, gt : ndarray
        Multi-label segmentation arrays.
    voxel_spacing : tuple
        Voxel dimensions in mm.
    regions : dict or None
        Region → label mapping.
    dil_factor : int
        Dilation iterations.
    volume_thresholds : dict or None
        Per-region volume thresholds (mm³).  Defaults to
        ``BRATS2024_VOLUME_THRESH``.

    Returns
    -------
    dict mapping region name → metrics dict (output of
    ``compute_brats2024_lesion_metrics``).
    """
    if regions is None:
        regions = REGION_MAP
    if volume_thresholds is None:
        volume_thresholds = BRATS2024_VOLUME_THRESH

    results: Dict[str, Dict[str, object]] = {}
    for tissue_type in regions:
        thresh = volume_thresholds.get(tissue_type, 20.0)
        results[tissue_type] = compute_brats2024_lesion_metrics(
            pred, gt, voxel_spacing,
            tissue_type=tissue_type,
            dil_factor=dil_factor,
            volume_thresh_mm3=thresh,
            regions=regions,
        )
    return results


# ---------------------------------------------------------------------------
# Volume extraction (Phase D)
# ---------------------------------------------------------------------------


def extract_volumes_cc(
    seg: np.ndarray,
    voxel_spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
    regions: Optional[Dict[str, List[int]]] = None,
) -> Dict[str, float]:
    """Extract region volumes in cm³ from a multi-label segmentation.

    Returns
    -------
    dict[region_name, volume_cc]
    """
    if regions is None:
        regions = REGION_MAP
    voxel_vol_mm3 = float(np.prod(voxel_spacing))

    volumes: Dict[str, float] = {}
    for region_name, label_list in regions.items():
        mask = extract_region(seg, label_list)
        volumes[region_name] = float(np.sum(mask) * voxel_vol_mm3 / 1000.0)
    return volumes


# ---------------------------------------------------------------------------
# Convenience: load NIfTI + compute all metrics
# ---------------------------------------------------------------------------


def evaluate_case(
    pred_path: str,
    gt_path: str,
    regions: Optional[Dict[str, List[int]]] = None,
) -> Dict:
    """Run full metric battery on a single case.

    Returns dict with keys: ``patient_metrics``, ``lesion_metrics``,
    ``brats2024_metrics``, ``volumes_pred``, ``volumes_gt``.
    """
    pred_img = nib.load(pred_path)
    gt_img = nib.load(gt_path)

    pred_data = np.asanyarray(pred_img.dataobj).astype(int)
    gt_data = np.asanyarray(gt_img.dataobj).astype(int)
    voxel_spacing = tuple(float(v) for v in pred_img.header.get_zooms()[:3])

    if regions is None:
        regions = REGION_MAP

    patient_metrics = compute_patient_metrics(
        pred_data, gt_data, voxel_spacing, regions=regions
    )

    # Lesion-wise for each region (original BraTS-2023 style)
    lesion_metrics: Dict[str, Dict] = {}
    for region_name, label_list in regions.items():
        pred_mask = extract_region(pred_data, label_list)
        gt_mask = extract_region(gt_data, label_list)
        lesion_metrics[region_name] = compute_lesion_wise_metrics(
            pred_mask, gt_mask, voxel_spacing
        )

    # BraTS-2024 lesion-wise metrics (dilation-based)
    brats2024_metrics = compute_brats2024_all_regions(
        pred_data, gt_data, voxel_spacing, regions=regions,
    )

    volumes_pred = extract_volumes_cc(pred_data, voxel_spacing, regions)
    volumes_gt = extract_volumes_cc(gt_data, voxel_spacing, regions)

    return {
        "patient_metrics": patient_metrics,
        "lesion_metrics": lesion_metrics,
        "brats2024_metrics": brats2024_metrics,
        "volumes_pred": volumes_pred,
        "volumes_gt": volumes_gt,
        "voxel_spacing": voxel_spacing,
    }
