"""Segmentation fusion workflows for multi-model ensemble integration."""

from __future__ import annotations

import itertools
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nb
import numpy as np
from nipype import logging as nipype_logging
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow

from oncoprep.utils.logging import get_logger

LOGGER = get_logger(__name__)
iflogger = nipype_logging.getLogger('nipype.interface')


def _binary_majority_vote(
    candidates: List[np.ndarray],
    weights: Optional[List[float]] = None,
    verbose: bool = True,
) -> np.ndarray:
    """Perform binary majority voting on segmentations.

    Fuses binary (0/1) segmentations using weighted majority voting.
    Each voxel is labeled as 1 if the weighted sum of votes exceeds 50%.

    Parameters
    ----------
    candidates : list of ndarray
        Binary segmentation arrays (0/1) with identical shape
    weights : list of float, optional
        Weights for each segmentation (default: equal weights of 1)
    verbose : bool
        Print detailed information (default: True)

    Returns
    -------
    ndarray
        Fused binary segmentation

    Raises
    ------
    ValueError
        If no candidates provided
    """
    num_cands = len(candidates)
    if num_cands == 0:
        raise ValueError("No segmentations provided for fusion")

    if weights is None:
        weights = list(itertools.repeat(1.0, num_cands))
    else:
        weights = list(weights)

    if num_cands == 1:
        if verbose:
            LOGGER.info("Single candidate provided, returning as-is")
        return candidates[0].astype(np.uint8)

    if verbose:
        LOGGER.info(f"Binary majority voting on {num_cands} candidates")
        for i, c in enumerate(candidates):
            LOGGER.debug(
                f"  Candidate {i}: shape={c.shape}, labels={np.unique(c)}, sum={np.sum(c)}"
            )

    template = candidates[0]
    result = np.zeros(template.shape, dtype=np.uint8)
    label_votes = np.zeros(template.shape, dtype=np.float32)

    # Tally votes
    for cand, weight in zip(candidates, weights):
        if cand.max() > 1 or cand.min() < 0:
            LOGGER.warning(
                f"Binary segmentation contains values outside [0,1]: "
                f"min={cand.min()}, max={cand.max()}"
            )
        label_votes[cand > 0] += float(weight)

    # Apply majority threshold
    total_weight = sum(weights)
    result[label_votes >= (total_weight / 2.0)] = 1

    if verbose:
        LOGGER.info(
            f"Fusion result: shape={result.shape}, "
            f"labels={np.unique(result)}, positive_voxels={np.sum(result)}"
        )

    return result


def _multiclass_majority_vote(
    candidates: List[np.ndarray],
    labels: Optional[List[int]] = None,
    weights: Optional[List[float]] = None,
    verbose: bool = True,
) -> np.ndarray:
    """Perform multi-class majority voting on segmentations.

    Fuses multi-label segmentations using weighted majority voting,
    processing each label independently.

    Parameters
    ----------
    candidates : list of ndarray
        Multi-label segmentation arrays with identical shape
    labels : list of int, optional
        Unique labels present in segmentations. If None, automatically detected.
    weights : list of float, optional
        Weights for each segmentation (default: equal weights of 1)
    verbose : bool
        Print detailed information (default: True)

    Returns
    -------
    ndarray
        Fused multi-label segmentation

    Raises
    ------
    ValueError
        If no candidates provided
    """
    num_cands = len(candidates)
    if num_cands == 0:
        raise ValueError("No segmentations provided for fusion")

    if weights is None:
        weights = list(itertools.repeat(1.0, num_cands))
    else:
        weights = list(weights)

    if num_cands == 1:
        if verbose:
            LOGGER.info("Single candidate provided, returning as-is")
        return candidates[0].astype(np.int16)

    # Detect labels if not provided
    if labels is None:
        labels_set = set()
        for c in candidates:
            labels_set.update(np.unique(c))
        labels = sorted(labels_set)
        if verbose:
            LOGGER.warning(
                f"Labels not provided, automatically detected: {labels}"
            )
    else:
        labels = sorted(labels)

    if verbose:
        LOGGER.info(
            f"Multi-class majority voting on {num_cands} candidates "
            f"with labels={labels}"
        )
        for i, c in enumerate(candidates):
            LOGGER.debug(
                f"  Candidate {i}: shape={c.shape}, "
                f"labels={np.unique(c)}, dtype={c.dtype}"
            )

    template = candidates[0]
    result = np.zeros(template.shape, dtype=np.int16)

    # Remove background label (0) from processing
    fg_labels = [l for l in labels if l != 0]

    # Process each label
    for label in reversed(fg_labels):  # Reverse order to avoid overwriting
        label_votes = np.zeros(template.shape, dtype=np.float32)

        # Tally votes for this label
        for cand, weight in zip(candidates, weights):
            label_votes[cand == label] += float(weight)

        # Apply majority threshold
        total_weight = sum(weights)
        result[label_votes >= (total_weight / 2.0)] = label

    if verbose:
        LOGGER.info(
            f"Fusion result: shape={result.shape}, "
            f"labels={np.unique(result)}, dtype={result.dtype}"
        )

    return result


def _compute_dice_score(
    segmentation: np.ndarray,
    reference: np.ndarray,
) -> float:
    """Compute DICE coefficient between two binary segmentations.

    Parameters
    ----------
    segmentation : ndarray
        Predicted binary segmentation
    reference : ndarray
        Reference binary segmentation

    Returns
    -------
    float
        DICE score in range [0, 1], where 1 is perfect match
    """
    tp = np.sum(np.logical_and(segmentation == 1, reference == 1))
    fp = np.sum(np.logical_and(segmentation == 1, reference == 0))
    fn = np.sum(np.logical_and(segmentation == 0, reference == 1))

    if (2 * tp + fp + fn) == 0:
        return 0.0

    dice = 2.0 * tp / (2.0 * tp + fp + fn)
    return float(np.clip(dice, 0.0, 1.0))


def _simple_fusion(
    candidates: List[np.ndarray],
    labels: Optional[List[int]] = None,
    weights: Optional[List[float]] = None,
    threshold: float = 0.05,
    convergence_threshold: int = 25,
    increment: float = 0.07,
    max_iterations: int = 25,
    verbose: bool = True,
) -> np.ndarray:
    """Perform SIMPLE (Staple-like) fusion with iterative refinement.

    Uses DICE-based scoring to iteratively assign weights to segmentations,
    removing low-quality estimates. Processes each label independently.

    Parameters
    ----------
    candidates : list of ndarray
        Segmentation arrays to fuse
    labels : list of int, optional
        Labels to process. If None, auto-detected.
    weights : list of float, optional
        Initial weights (default: equal)
    threshold : float
        Fraction of max score below which segmentations are dropped (default: 0.05)
    convergence_threshold : int
        Threshold for convergence detection (default: 25)
    increment : float
        Increment for threshold per iteration (default: 0.07)
    max_iterations : int
        Maximum iterations per label (default: 25)
    verbose : bool
        Print detailed information (default: True)

    Returns
    -------
    ndarray
        Fused multi-label segmentation
    """
    num_cands = len(candidates)
    if num_cands == 0:
        raise ValueError("No segmentations provided for fusion")

    if weights is None:
        weights = list(itertools.repeat(1.0, num_cands))
    else:
        weights = list(weights)

    if num_cands == 1:
        if verbose:
            LOGGER.info("Single candidate provided, returning as-is")
        return candidates[0].astype(np.int16)

    # Detect labels if not provided
    if labels is None:
        labels_set = set()
        for c in candidates:
            labels_set.update(np.unique(c))
        labels = sorted(labels_set)
        if verbose:
            LOGGER.warning(f"Labels not provided, auto-detected: {labels}")

    if verbose:
        LOGGER.info(f"SIMPLE fusion on {num_cands} candidates with labels={labels}")

    template = candidates[0]
    result = np.zeros(template.shape, dtype=np.int16)

    # Remove background label
    fg_labels = [l for l in labels if l != 0]
    backup_weights = weights.copy()

    # Process each label
    for label in sorted(fg_labels):
        if verbose:
            LOGGER.info(f"Fusing label {label}")

        # Binarize candidates for this label
        bin_cands = [(c == label).astype(np.uint8) for c in candidates]

        # Initial majority vote
        estimate = _binary_majority_vote(bin_cands, weights, verbose=False)
        convergence_baseline = np.sum(estimate)

        if np.sum(estimate) == 0:
            LOGGER.warning(f"Initial estimate for label {label} is empty")
            continue

        # Iterative refinement
        tau = threshold
        for iteration in range(max_iterations):
            iter_weights = []

            # Score each candidate
            for cand in bin_cands:
                dice = _compute_dice_score(cand, estimate)
                # Squared DICE for more aggressive filtering
                score = (dice + 1.0) ** 2
                iter_weights.append(score)

            max_score = max(iter_weights) if iter_weights else 1.0

            # Filter low-scoring candidates
            filtered_cands = [
                c
                for c, w in zip(bin_cands, iter_weights)
                if w > tau * max_score
            ]
            filtered_weights = [
                w for w in iter_weights if w > tau * max_score
            ]

            if not filtered_cands:
                if verbose:
                    LOGGER.warning(
                        f"Label {label} iteration {iteration}: "
                        f"all candidates filtered, using unfiltered"
                    )
                filtered_cands = bin_cands
                filtered_weights = iter_weights

            # Re-estimate
            estimate = _binary_majority_vote(
                filtered_cands, filtered_weights, verbose=False
            )

            # Check convergence
            new_convergence = np.sum(estimate)
            if abs(convergence_baseline - new_convergence) < convergence_threshold:
                if verbose:
                    LOGGER.info(
                        f"Label {label} converged after {iteration} iterations"
                    )
                break

            convergence_baseline = new_convergence
            tau += increment

        # Assign label to result
        result[estimate == 1] = label
        weights = backup_weights.copy()

    if verbose:
        LOGGER.info(
            f"SIMPLE fusion result: shape={result.shape}, "
            f"labels={np.unique(result)}"
        )

    return result


def _brats_fusion(
    candidates: List[np.ndarray],
    weights: Optional[List[float]] = None,
    threshold: float = 0.05,
    convergence_threshold: int = 25,
    increment: float = 0.07,
    max_iterations: int = 25,
    verbose: bool = True,
) -> np.ndarray:
    """Perform BRATS-specific SIMPLE fusion.

    Specialized for BraTS tumor labels (whole tumor, tumor core, active tumor).
    Uses SIMPLE algorithm with BRATS label hierarchy.

    Parameters
    ----------
    candidates : list of ndarray
        BRATS segmentations (labels: 0=bg, 1=core, 2=whole tumor, 4=active)
    weights : list of float, optional
        Initial weights (default: equal)
    threshold : float
        Dropout threshold (default: 0.05)
    convergence_threshold : int
        Convergence threshold (default: 25)
    increment : float
        Threshold increment per iteration (default: 0.07)
    max_iterations : int
        Maximum iterations (default: 25)
    verbose : bool
        Print detailed information (default: True)

    Returns
    -------
    ndarray
        Fused BRATS segmentation
    """
    num_cands = len(candidates)
    if num_cands == 0:
        raise ValueError("No segmentations provided for fusion")

    if weights is None:
        weights = list(itertools.repeat(1.0, num_cands))
    else:
        weights = list(weights)

    if num_cands == 1:
        if verbose:
            LOGGER.info("Single candidate provided, returning as-is")
        return candidates[0].astype(np.int16)

    if verbose:
        LOGGER.info(f"BRATS SIMPLE fusion on {num_cands} candidates")

    template = candidates[0]
    result = np.zeros(template.shape, dtype=np.int16)
    backup_weights = weights.copy()

    # BRATS label hierarchy: process in this order
    # 2: whole tumor (all non-background)
    # 1: tumor core (classes 1 and 4)
    # 4: active tumor (class 4 only)
    brats_labels = [2, 1, 4]

    for label_idx, brats_label in enumerate(brats_labels):
        if verbose:
            LOGGER.info(f"Processing BRATS label {brats_label}")

        # Define binarization rules for BRATS labels
        if brats_label == 2:
            # Whole tumor: any non-background
            bin_cands = [(c > 0).astype(np.uint8) for c in candidates]
        elif brats_label == 1:
            # Tumor core: classes 1 or 4
            bin_cands = [
                ((c == 1) | (c == 4)).astype(np.uint8) for c in candidates
            ]
        else:  # brats_label == 4
            # Active tumor: class 4 only
            bin_cands = [(c == 4).astype(np.uint8) for c in candidates]

        # Initial estimate
        estimate = _binary_majority_vote(bin_cands, weights, verbose=False)

        if np.sum(estimate) == 0:
            LOGGER.warning(f"Initial estimate for BRATS label {brats_label} empty")
            weights = backup_weights.copy()
            continue

        conv_baseline = np.sum(estimate)
        tau = threshold

        # Iterative refinement
        for iteration in range(max_iterations):
            iter_weights = []

            for cand in bin_cands:
                dice = _compute_dice_score(cand, estimate)
                score = (dice + 1.0) ** 2
                iter_weights.append(score)

            max_score = max(iter_weights) if iter_weights else 1.0

            # Filter candidates
            filtered_cands = [
                c
                for c, w in zip(bin_cands, iter_weights)
                if w > tau * max_score
            ]
            filtered_weights = [
                w for w in iter_weights if w > tau * max_score
            ]

            if not filtered_cands:
                filtered_cands = bin_cands
                filtered_weights = iter_weights

            # Re-estimate
            estimate = _binary_majority_vote(
                filtered_cands, filtered_weights, verbose=False
            )

            # Check convergence
            new_conv = np.sum(estimate)
            if abs(conv_baseline - new_conv) < convergence_threshold:
                if verbose:
                    LOGGER.info(
                        f"BRATS label {brats_label} converged after {iteration} iterations"
                    )
                break

            conv_baseline = new_conv
            tau += increment

        # Assign label
        result[estimate == 1] = brats_label
        weights = backup_weights.copy()

    if verbose:
        LOGGER.info(
            f"BRATS fusion result: shape={result.shape}, "
            f"labels={np.unique(result)}"
        )

    return result


def _fuse_segmentations(
    segmentation_files: List[str],
    output_path: str,
    method: str = "mav",
    labels: Optional[List[int]] = None,
    weights: Optional[List[float]] = None,
    verbose: bool = True,
) -> str:
    """Fuse multiple segmentation files into single output.

    Parameters
    ----------
    segmentation_files : list of str
        Paths to segmentation NIfTi files
    output_path : str
        Path for output fused segmentation
    method : str
        Fusion method: 'mav', 'simple', or 'brats' (default: 'mav')
    labels : list of int, optional
        Labels to process
    weights : list of float, optional
        Weights for each segmentation
    verbose : bool
        Print detailed information (default: True)

    Returns
    -------
    str
        Path to fused segmentation file
    """
    if not segmentation_files:
        raise ValueError("No segmentation files provided")

    if verbose:
        LOGGER.info(
            f"Loading {len(segmentation_files)} segmentations for fusion "
            f"(method={method})"
        )

    # Load segmentations
    candidates = []
    template_img = None

    for i, seg_path in enumerate(segmentation_files):
        try:
            img = nb.load(seg_path)
            candidates.append(img.get_fdata().astype(np.int16))
            if template_img is None:
                template_img = img
            if verbose:
                LOGGER.debug(f"Loaded segmentation {i}: {seg_path}")
        except Exception as e:
            LOGGER.error(f"Failed to load segmentation {seg_path}: {e}")
            raise

    if template_img is None:
        raise ValueError("No valid segmentation images loaded")

    # Perform fusion
    if method == "mav":
        fused = _multiclass_majority_vote(
            candidates, labels=labels, weights=weights, verbose=verbose
        )
    elif method == "simple":
        fused = _simple_fusion(
            candidates, labels=labels, weights=weights, verbose=verbose
        )
    elif method == "brats":
        fused = _brats_fusion(candidates, weights=weights, verbose=verbose)
    else:
        raise ValueError(f"Unknown fusion method: {method}")

    # Save result
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fused_img = nb.Nifti1Image(fused, template_img.affine, template_img.header)
    nb.save(fused_img, str(output_path))

    if verbose:
        LOGGER.info(f"Fused segmentation saved to {output_path}")

    return str(output_path)


def init_fusion_wf(
    *,
    output_dir: str,
    fusion_method: str = "mav",
    labels: Optional[List[int]] = None,
    weights: Optional[List[float]] = None,
    name: str = "fusion_wf",
) -> pe.Workflow:
    """Initialize segmentation fusion workflow.

    Creates a workflow for fusing multiple segmentation results using
    majority voting, SIMPLE algorithm, or BRATS-specific fusion.

    Parameters
    ----------
    output_dir : str
        Output directory for results
    fusion_method : str
        Fusion method: 'mav' (majority vote), 'simple', or 'brats' (default: 'mav')
    labels : list of int, optional
        Specific labels to process. If None, auto-detected.
    weights : list of float, optional
        Per-segmentation weights. If None, equal weighting.
    name : str
        Workflow name (default: 'fusion_wf')

    Returns
    -------
    Workflow
        Nipype fusion workflow

    Notes
    -----
    The workflow expects a list of segmentation file paths as input.
    Each file should be a NIfTi segmentation with compatible dimensions.

    **Input Fields:**
    - segmentation_files : list of str
        Paths to segmentation files to fuse
    - source_file : str (optional)
        BIDS source file for derivatives naming

    **Output Fields:**
    - fused_segmentation : str
        Path to fused segmentation result
    """
    workflow = LiterateWorkflow(name=name)

    # Input node
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "segmentation_files",
                "source_file",
            ]
        ),
        name="inputnode",
    )

    # Output node
    outputnode = pe.Node(
        niu.IdentityInterface(fields=["fused_segmentation"]),
        name="outputnode",
    )

    # Fusion node
    fuse_node = pe.Node(
        niu.Function(
            function=_fuse_segmentations,
            input_names=[
                "segmentation_files",
                "output_path",
                "method",
                "labels",
                "weights",
                "verbose",
            ],
            output_names=["fused_seg_path"],
        ),
        name="fuse_segmentations",
    )
    fuse_node.inputs.method = fusion_method
    fuse_node.inputs.labels = labels
    fuse_node.inputs.weights = weights
    fuse_node.inputs.verbose = True
    fuse_node.inputs.output_path = str(Path(output_dir) / "fused_segmentation.nii.gz")

    # Connect
    workflow.connect([
        (inputnode, fuse_node, [("segmentation_files", "segmentation_files")]),
        (fuse_node, outputnode, [("fused_seg_path", "fused_segmentation")]),
    ])

    # Workflow description
    workflow.__desc__ = f"""
## Segmentation Fusion

Multiple segmentation results are fused using {fusion_method.upper()} algorithm.
{
    "Majority voting assigns each voxel the label that appears in >50% of inputs."
    if fusion_method == "mav"
    else "SIMPLE algorithm iteratively refines segmentations based on DICE similarity."
    if fusion_method == "simple"
    else "BRATS-specific fusion processes tumor labels hierarchically."
}
"""

    return workflow
