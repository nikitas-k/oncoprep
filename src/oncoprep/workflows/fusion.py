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
    segmentation_files,
    output_path,
    method="mav",
    labels=None,
    weights=None,
    verbose=True,
):
    """Fuse multiple segmentation files into single output.

    This is a standalone function for use in Nipype Function nodes.
    All imports and helper functions are defined inside.

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
    import itertools
    import logging
    import math
    from pathlib import Path

    import nibabel as nb
    import numpy as np

    LOGGER = logging.getLogger('nipype.workflow')

    # === Helper functions (must be defined inside for Nipype serialization) ===

    def binary_majority_vote(candidates, weights=None):
        """Binary majority voting."""
        num_cands = len(candidates)
        if num_cands == 0:
            raise ValueError("No candidates")
        if weights is None:
            weights = list(itertools.repeat(1.0, num_cands))
        if num_cands == 1:
            return candidates[0].astype(np.uint8)

        template = candidates[0]
        label_votes = np.zeros(template.shape, dtype=np.float32)
        for cand, weight in zip(candidates, weights):
            label_votes[cand > 0] += float(weight)

        result = np.zeros(template.shape, dtype=np.uint8)
        total_weight = sum(weights)
        result[label_votes >= (total_weight / 2.0)] = 1
        return result

    def compute_dice(seg, ref):
        """Compute DICE coefficient."""
        tp = np.sum(np.logical_and(seg == 1, ref == 1))
        fp = np.sum(np.logical_and(seg == 1, ref == 0))
        fn = np.sum(np.logical_and(seg == 0, ref == 1))
        if (2 * tp + fp + fn) == 0:
            return 0.0
        return float(2.0 * tp / (2.0 * tp + fp + fn))

    def multiclass_majority_vote(candidates, labels=None, weights=None):
        """Multi-class majority voting."""
        num_cands = len(candidates)
        if num_cands == 0:
            raise ValueError("No candidates")
        if weights is None:
            weights = list(itertools.repeat(1.0, num_cands))
        if num_cands == 1:
            return candidates[0].astype(np.int16)

        if labels is None:
            labels_set = set()
            for c in candidates:
                labels_set.update(np.unique(c))
            labels = sorted(labels_set)

        template = candidates[0]
        result = np.zeros(template.shape, dtype=np.int16)
        fg_labels = [l for l in labels if l != 0]

        for label in reversed(fg_labels):
            label_votes = np.zeros(template.shape, dtype=np.float32)
            for cand, weight in zip(candidates, weights):
                label_votes[cand == label] += float(weight)
            total_weight = sum(weights)
            result[label_votes >= (total_weight / 2.0)] = label

        return result

    def brats_fusion(candidates, weights=None, threshold=0.05, max_iterations=25):
        """BraTS-specific SIMPLE fusion with DICE weighting."""
        num_cands = len(candidates)
        if num_cands == 0:
            raise ValueError("No candidates")
        if weights is None:
            weights = list(itertools.repeat(1.0, num_cands))
        if num_cands == 1:
            return candidates[0].astype(np.int16)

        template = candidates[0]
        result = np.zeros(template.shape, dtype=np.int16)
        backup_weights = list(weights)

        # BraTS label hierarchy: whole tumor, tumor core, active tumor
        brats_labels = [2, 1, 4]
        increment = 0.07
        convergence_threshold = 25

        for brats_label in brats_labels:
            if verbose:
                LOGGER.info(f"Processing BRATS label {brats_label}")

            # Binarize candidates
            if brats_label == 2:
                bin_cands = [(c > 0).astype(np.uint8) for c in candidates]
            elif brats_label == 1:
                bin_cands = [((c == 1) | (c == 4)).astype(np.uint8) for c in candidates]
            else:
                bin_cands = [(c == 4).astype(np.uint8) for c in candidates]

            # Initial estimate
            estimate = binary_majority_vote(bin_cands, weights)
            if np.sum(estimate) == 0:
                weights = list(backup_weights)
                continue

            conv_baseline = np.sum(estimate)
            tau = threshold

            # Iterative refinement
            for iteration in range(max_iterations):
                iter_weights = []
                for cand in bin_cands:
                    dice = compute_dice(cand, estimate)
                    iter_weights.append((dice + 1.0) ** 2)

                max_score = max(iter_weights) if iter_weights else 1.0
                filtered_cands = [c for c, w in zip(bin_cands, iter_weights) if w > tau * max_score]
                filtered_weights = [w for w in iter_weights if w > tau * max_score]

                if not filtered_cands:
                    filtered_cands = bin_cands
                    filtered_weights = iter_weights

                estimate = binary_majority_vote(filtered_cands, filtered_weights)

                new_conv = np.sum(estimate)
                if abs(conv_baseline - new_conv) < convergence_threshold:
                    if verbose:
                        LOGGER.info(f"BRATS label {brats_label} converged after {iteration} iterations")
                    break

                conv_baseline = new_conv
                tau += increment

            result[estimate == 1] = brats_label
            weights = list(backup_weights)

        return result

    # === Main fusion logic ===

    if not segmentation_files:
        raise ValueError("No segmentation files provided")

    # Filter out None values
    valid_files = [f for f in segmentation_files if f is not None]
    if not valid_files:
        raise ValueError("No valid segmentation files provided")

    if verbose:
        LOGGER.info(f"Loading {len(valid_files)} segmentations for fusion (method={method})")

    # Load segmentations
    candidates = []
    template_img = None

    for i, seg_path in enumerate(valid_files):
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
        fused = multiclass_majority_vote(candidates, labels=labels, weights=weights)
    elif method == "simple":
        # Use brats fusion for simple as well (it's more robust)
        fused = brats_fusion(candidates, weights=weights)
    elif method == "brats":
        fused = brats_fusion(candidates, weights=weights)
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


def init_anat_seg_fuse_wf(
    *,
    output_dir: str,
    fusion_method: str = "brats",
    name: str = "anat_seg_fuse_wf",
) -> pe.Workflow:
    """Initialize anatomical segmentation fusion workflow.

    This workflow receives multiple tumor segmentations from ensemble models
    and fuses them using consensus voting with DICE-based quality weighting.
    Designed for integration with the anatomical segmentation pipeline.

    The fusion algorithm performs:
    1. For each label, create binary masks from all candidate segmentations
    2. Compute initial consensus via majority voting
    3. Iteratively refine by scoring each candidate against current estimate
    4. Weight candidates by squared DICE score for robustness
    5. Drop low-quality candidates below threshold
    6. Repeat until convergence or max iterations

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from oncoprep.workflows.fusion import init_anat_seg_fuse_wf
            wf = init_anat_seg_fuse_wf(output_dir='/tmp')

    Parameters
    ----------
    output_dir : str
        Output directory for fused segmentation
    fusion_method : str
        Fusion algorithm to use:
        - 'mav': Multiclass majority voting (fast, simple)
        - 'simple': SIMPLE algorithm with iterative DICE-based refinement
        - 'brats': BraTS-specific hierarchical fusion (default)
    name : str
        Workflow name (default: 'anat_seg_fuse_wf')

    Inputs
    ------
    segmentation_files
        List of tumor segmentation files from ensemble models
    source_file
        Source anatomical image for BIDS derivatives naming
    t1w_preproc
        Preprocessed T1w for reference (affine, header)

    Outputs
    -------
    fused_seg
        Path to fused tumor segmentation
    fused_seg_old
        Path to fused segmentation with old BraTS labels
    fused_seg_new
        Path to fused segmentation with new derived labels

    Returns
    -------
    Workflow
        Nipype workflow for segmentation fusion

    Notes
    -----
    Fusion methods:
    - **mav (Majority Voting)**: Each voxel assigned the label appearing
      in >50% of inputs. Fast but may produce noisy boundaries.
    - **simple (SIMPLE Algorithm)**: Iteratively estimates segmentation
      quality using DICE similarity. Weights candidates by (DICE+1)^2
      and drops low-quality estimates. More robust to outliers.
    - **brats (BraTS-specific)**: Applies SIMPLE algorithm with BraTS
      label hierarchy: whole tumor → tumor core → active tumor.
      Recommended for brain tumor segmentation ensembles.

    References
    ----------
    Berger, C. et al. - BraTS Toolkit (Fusionator class)
    Warfield, S.K. et al. - STAPLE algorithm inspiration
    """
    from pathlib import Path

    from niworkflows.engine.workflows import LiterateWorkflow

    workflow = LiterateWorkflow(name=name)

    # Input node: receives list of segmentations from ensemble models
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'segmentation_files',  # List of paths to segmentations
                'source_file',  # BIDS source for derivatives naming
                't1w_preproc',  # Reference image for header/affine
            ]
        ),
        name='inputnode',
    )

    # Output node: provides fused segmentation
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'fused_seg',  # Fused tumor segmentation (raw labels)
                'fused_seg_old',  # Old BraTS labels
                'fused_seg_new',  # New derived labels
            ]
        ),
        name='outputnode',
    )

    # Main fusion node using _fuse_segmentations function
    fuse_node = pe.Node(
        niu.Function(
            function=_fuse_segmentations,
            input_names=[
                'segmentation_files',
                'output_path',
                'method',
                'labels',
                'weights',
                'verbose',
            ],
            output_names=['fused_seg_path'],
        ),
        name='fuse_segmentations',
    )
    fuse_node.inputs.method = fusion_method
    fuse_node.inputs.labels = None  # Auto-detect from input
    fuse_node.inputs.weights = None  # Equal weighting initially
    fuse_node.inputs.verbose = True

    # Output path will be in node's working directory
    def _get_output_path():
        """Generate output path in node's working directory."""
        import os
        work_dir = os.path.abspath('fusion_output')
        os.makedirs(work_dir, exist_ok=True)
        return os.path.join(work_dir, 'fused_tumor_seg.nii.gz')

    set_output_path = pe.Node(
        niu.Function(
            function=_get_output_path,
            input_names=[],
            output_names=['output_path'],
        ),
        name='set_output_path',
    )

    # Label conversion nodes for old and new BraTS label schemes
    convert_old = pe.Node(
        niu.Function(
            function=_convert_to_old_brats_labels,
            input_names=['seg_file'],
            output_names=['old_labels_file'],
        ),
        name='convert_to_old_labels',
    )

    convert_new = pe.Node(
        niu.Function(
            function=_convert_to_new_brats_labels,
            input_names=['seg_file'],
            output_names=['new_labels_file'],
        ),
        name='convert_to_new_labels',
    )

    # Connect workflow
    workflow.connect([
        # Set output path
        (set_output_path, fuse_node, [('output_path', 'output_path')]),
        # Perform fusion
        (inputnode, fuse_node, [('segmentation_files', 'segmentation_files')]),
        # Convert to old/new label schemes
        (fuse_node, convert_old, [('fused_seg_path', 'seg_file')]),
        (fuse_node, convert_new, [('fused_seg_path', 'seg_file')]),
        # Output connections
        (fuse_node, outputnode, [('fused_seg_path', 'fused_seg')]),
        (convert_old, outputnode, [('old_labels_file', 'fused_seg_old')]),
        (convert_new, outputnode, [('new_labels_file', 'fused_seg_new')]),
    ])

    # Workflow documentation
    method_desc = {
        'mav': 'majority voting (each voxel assigned the most common label)',
        'simple': 'SIMPLE algorithm with iterative DICE-based quality weighting',
        'brats': 'BraTS-specific hierarchical fusion with DICE weighting',
    }
    workflow.__desc__ = f"""
## Tumor Segmentation Fusion

Multiple tumor segmentation predictions from an ensemble of deep learning
models were fused using {method_desc.get(fusion_method, fusion_method)}.

The fusion algorithm:
1. Loads all candidate segmentations from ensemble models
2. For each tumor label, creates binary masks across candidates
3. Computes initial consensus via majority voting
4. Iteratively refines using DICE similarity scoring
5. Weights candidates by (DICE + 1)² for outlier robustness
6. Converges when estimate stabilizes or max iterations reached

Output labels follow BraTS convention:
- Raw: 1=necrotic core, 2=peritumoral edema, 4=enhancing tumor
- Old scheme: 1=NCR, 2=ED, 3=ET (remapped from 4), 4=RC
- New scheme: 1=ET, 2=TC, 3=WT, 4=NETC, 5=SNFH, 6=RC
"""

    return workflow


def _convert_to_old_brats_labels(seg_file):
    """Convert raw BraTS model output to old label scheme.

    Raw BraTS model outputs use label 4 for enhancing tumor.
    Old scheme remaps: 4 -> 3 for enhancing tumor.

    Parameters
    ----------
    seg_file : str or None
        Path to raw segmentation file

    Returns
    -------
    str or None
        Path to converted segmentation with old labels
    """
    import os
    from pathlib import Path

    import nibabel as nib
    import numpy as np

    if seg_file is None:
        return None

    img = nib.load(seg_file)
    data = np.asarray(img.dataobj)

    # Raw: 1=NCR, 2=ED, 4=ET -> Old: 1=NCR, 2=ED, 3=ET
    old_labels = np.zeros_like(data, dtype=np.uint8)
    old_labels[data == 1] = 1  # NCR stays 1
    old_labels[data == 2] = 2  # ED stays 2
    old_labels[data == 4] = 3  # ET becomes 3
    old_labels[data == 5] = 4  # RC if present

    out_dir = os.path.abspath('old_labels')
    os.makedirs(out_dir, exist_ok=True)
    out_path = str(Path(out_dir) / 'fused_tumor_seg_old_labels.nii.gz')

    out_img = nib.Nifti1Image(old_labels, img.affine, img.header)
    nib.save(out_img, out_path)

    return out_path


def _convert_to_new_brats_labels(seg_file):
    """Convert raw BraTS model output to new derived label scheme.

    Creates composite labels from raw BraTS segmentation:
    - ET (1): Enhancing tumor only
    - TC (2): Tumor core = NCR + ET
    - WT (3): Whole tumor = NCR + ED + ET
    - NETC (4): Non-enhancing tumor core = NCR only
    - SNFH (5): FLAIR hyperintensity = ED only
    - RC (6): Resection cavity (optional)

    Parameters
    ----------
    seg_file : str or None
        Path to raw segmentation file

    Returns
    -------
    str or None
        Path to converted segmentation with new labels
    """
    import os
    from pathlib import Path

    import nibabel as nib
    import numpy as np

    if seg_file is None:
        return None

    img = nib.load(seg_file)
    data = np.asarray(img.dataobj)

    # Extract raw labels
    ncr_mask = (data == 1)
    ed_mask = (data == 2)
    et_mask = (data == 4)
    rc_mask = (data == 5)

    # Create new derived labels
    new_labels = np.zeros_like(data, dtype=np.uint8)

    # WT (3) = NCR + ED + ET
    new_labels[ncr_mask | ed_mask | et_mask] = 3

    # TC (2) = NCR + ET
    new_labels[ncr_mask | et_mask] = 2

    # SNFH (5) = ED only
    new_labels[ed_mask & ~ncr_mask & ~et_mask] = 5

    # NETC (4) = NCR only
    new_labels[ncr_mask & ~et_mask] = 4

    # ET (1) = Enhancing tumor only
    new_labels[et_mask] = 1

    # RC (6) = Resection cavity
    new_labels[rc_mask] = 6

    out_dir = os.path.abspath('new_labels')
    os.makedirs(out_dir, exist_ok=True)
    out_path = str(Path(out_dir) / 'fused_tumor_seg_new_labels.nii.gz')

    out_img = nib.Nifti1Image(new_labels, img.affine, img.header)
    nib.save(out_img, out_path)

    return out_path
