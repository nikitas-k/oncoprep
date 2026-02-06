"""BraTS-specific BIDS derivatives for OncoPrep.

This module provides workflows for saving BraTS-specific outputs including
tumor-related segmentations, metrics, and custom derivatives.
"""

from typing import Dict, List, Optional
import json
from pathlib import Path

from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine import Workflow

from ..interfaces import DerivativesDataSink


def _check_seg_file(seg_file):
    """Check if segmentation file is valid and exists on disk.
    
    Returns the file path if valid and exists, raises RuntimeError otherwise.
    This allows the workflow to skip downstream nodes gracefully.
    """
    import os
    import logging
    LOGGER = logging.getLogger('nipype.workflow')
    
    if seg_file is None or seg_file == 'None' or str(seg_file) == 'None':
        LOGGER.warning("Segmentation file is None, skipping derivatives save")
        raise RuntimeError("Segmentation failed - no output file to save")
    
    if not os.path.isfile(seg_file):
        LOGGER.warning(f"Segmentation file does not exist: {seg_file}")
        raise RuntimeError(f"Segmentation file not found: {seg_file}")
    
    return seg_file


def init_ds_tumor_seg_wf(
    *,
    output_dir: str,
    name: str = 'ds_tumor_seg_wf',
) -> Workflow:
    """Save tumor segmentation in BIDS derivatives format.

    Saves three versions of the tumor segmentation:
    - Raw model output (label-tumor)
    - Old BraTS labels 2017-2020 (label-tumorOld): 1=NCR, 2=ED, 3=ET, 4=RC
    - New derived labels 2021+ (label-tumorNew): 1=ET, 2=TC, 3=WT, 4=NETC, 5=SNFH, 6=RC

    Parameters
    ----------
    output_dir : :obj:`str`
        Directory in which to save derivatives
    name : :obj:`str`
        Workflow name (default: ds_tumor_seg_wf)

    Inputs
    ------
    source_file
        Input anatomical image
    tumor_seg
        Raw tumor segmentation map (native space)
    tumor_seg_old
        Tumor segmentation with old BraTS labels
    tumor_seg_new
        Tumor segmentation with new derived labels

    Outputs
    -------
    tumor_seg_file
        Path to saved raw tumor segmentation
    tumor_seg_old_file
        Path to saved old labels segmentation
    tumor_seg_new_file
        Path to saved new labels segmentation

    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=[
            'source_file',
            'tumor_seg',
            'tumor_seg_old',
            'tumor_seg_new',
        ]),
        name='inputnode',
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=[
            'tumor_seg_file',
            'tumor_seg_old_file',
            'tumor_seg_new_file',
        ]),
        name='outputnode',
    )

    # Raw tumor segmentation (model output)
    ds_tumor_seg = pe.Node(
        DerivativesDataSink(
            check_hdr=False,
            desc='tumor',
            suffix='dseg',
            compress=True,
        ),
        name='ds_tumor_seg',
        run_without_submitting=True,
    )
    ds_tumor_seg.inputs.base_directory = output_dir
    ds_tumor_seg.inputs.out_path_base = 'oncoprep'
    ds_tumor_seg.inputs.label = 'tumor'

    # Old BraTS labels (2017-2020)
    ds_tumor_seg_old = pe.Node(
        DerivativesDataSink(
            check_hdr=False,
            desc='tumorOld',
            suffix='dseg',
            compress=True,
        ),
        name='ds_tumor_seg_old',
        run_without_submitting=True,
    )
    ds_tumor_seg_old.inputs.base_directory = output_dir
    ds_tumor_seg_old.inputs.out_path_base = 'oncoprep'
    ds_tumor_seg_old.inputs.label = 'tumorOld'

    # New derived labels (2021+)
    ds_tumor_seg_new = pe.Node(
        DerivativesDataSink(
            check_hdr=False,
            desc='tumorNew',
            suffix='dseg',
            compress=True,
        ),
        name='ds_tumor_seg_new',
        run_without_submitting=True,
    )
    ds_tumor_seg_new.inputs.base_directory = output_dir
    ds_tumor_seg_new.inputs.out_path_base = 'oncoprep'
    ds_tumor_seg_new.inputs.label = 'tumorNew'

    # Add filter nodes to validate segmentation files before saving
    # These will raise RuntimeError if seg_file is None, allowing graceful failure
    check_tumor_seg = pe.Node(
        niu.Function(
            function=_check_seg_file,
            input_names=['seg_file'],
            output_names=['seg_file'],
        ),
        name='check_tumor_seg',
    )

    check_tumor_seg_old = pe.Node(
        niu.Function(
            function=_check_seg_file,
            input_names=['seg_file'],
            output_names=['seg_file'],
        ),
        name='check_tumor_seg_old',
    )

    check_tumor_seg_new = pe.Node(
        niu.Function(
            function=_check_seg_file,
            input_names=['seg_file'],
            output_names=['seg_file'],
        ),
        name='check_tumor_seg_new',
    )

    workflow.connect([
        # Filter then save raw segmentation
        (inputnode, check_tumor_seg, [('tumor_seg', 'seg_file')]),
        (inputnode, ds_tumor_seg, [('source_file', 'source_file')]),
        (check_tumor_seg, ds_tumor_seg, [('seg_file', 'in_file')]),
        (ds_tumor_seg, outputnode, [('out_file', 'tumor_seg_file')]),
        # Filter then save old labels
        (inputnode, check_tumor_seg_old, [('tumor_seg_old', 'seg_file')]),
        (inputnode, ds_tumor_seg_old, [('source_file', 'source_file')]),
        (check_tumor_seg_old, ds_tumor_seg_old, [('seg_file', 'in_file')]),
        (ds_tumor_seg_old, outputnode, [('out_file', 'tumor_seg_old_file')]),
        # Filter then save new labels
        (inputnode, check_tumor_seg_new, [('tumor_seg_new', 'seg_file')]),
        (inputnode, ds_tumor_seg_new, [('source_file', 'source_file')]),
        (check_tumor_seg_new, ds_tumor_seg_new, [('seg_file', 'in_file')]),
        (ds_tumor_seg_new, outputnode, [('out_file', 'tumor_seg_new_file')]),
    ])

    return workflow


def init_ds_tumor_mask_wf(
    *,
    output_dir: str,
    name: str = 'ds_tumor_mask_wf',
) -> Workflow:
    """Save tumor mask in BIDS derivatives format.

    Parameters
    ----------
    output_dir : :obj:`str`
        Directory in which to save derivatives
    name : :obj:`str`
        Workflow name (default: ds_tumor_mask_wf)

    Inputs
    ------
    source_file
        Input anatomical image
    tumor_mask
        Binary tumor mask

    Outputs
    -------
    tumor_mask_file
        Path to saved tumor mask

    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['source_file', 'tumor_mask']),
        name='inputnode',
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['tumor_mask_file']),
        name='outputnode',
    )

    # Use DerivativesDataSink with BIDS mask specification
    ds_tumor_mask = pe.Node(
        DerivativesDataSink(
            check_hdr=False,
            desc='tumor',
            suffix='mask',
        ),
        name='ds_tumor_mask',
        run_without_submitting=True,
    )
    ds_tumor_mask.inputs.base_directory = output_dir

    workflow.connect([
        (inputnode, ds_tumor_mask, [
            ('source_file', 'source_file'),
            ('tumor_mask', 'in_file'),
        ]),
        (ds_tumor_mask, outputnode, [('out_file', 'tumor_mask_file')]),
    ])

    return workflow


def init_ds_tumor_metrics_wf(
    *,
    output_dir: str,
    name: str = 'ds_tumor_metrics_wf',
) -> Workflow:
    """Save tumor-derived metrics as BIDS JSON files.

    Parameters
    ----------
    output_dir : :obj:`str`
        Directory in which to save derivatives
    name : :obj:`str`
        Workflow name (default: ds_tumor_metrics_wf)

    Inputs
    ------
    source_files
        List of input anatomical images
    tumor_metrics
        Dictionary of tumor metrics (volume, location, etc.)

    Outputs
    -------
    metrics_file
        Path to saved JSON metrics file

    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['source_files', 'tumor_metrics']),
        name='inputnode',
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['metrics_file']),
        name='outputnode',
    )

    save_metrics = pe.Node(
        niu.Function(
            function=_save_tumor_metrics,
            input_names=['tumor_metrics', 'source_files', 'output_dir'],
            output_names=['metrics_file'],
        ),
        name='save_metrics',
        run_without_submitting=True,
    )
    save_metrics.inputs.output_dir = output_dir

    workflow.connect([
        (inputnode, save_metrics, [
            ('tumor_metrics', 'tumor_metrics'),
            ('source_files', 'source_files'),
        ]),
        (save_metrics, outputnode, [('metrics_file', 'metrics_file')]),
    ])

    return workflow


def init_ds_multimodal_tumor_wf(
    *,
    output_dir: str,
    modalities: Optional[List[str]] = None,
    name: str = 'ds_multimodal_tumor_wf',
) -> Workflow:
    """Save multi-modal tumor segmentations (T1w, T1ce, T2w, FLAIR).

    Parameters
    ----------
    output_dir : :obj:`str`
        Directory in which to save derivatives
    modalities : :obj:`list` of :obj:`str`, optional
        List of modalities to process (default: T1w, T1ce, T2w, FLAIR)
    name : :obj:`str`
        Workflow name (default: ds_multimodal_tumor_wf)

    Inputs
    ------
    source_file
        Input anatomical image
    tumor_seg_<modality>
        Tumor segmentation for each modality

    Outputs
    -------
    tumor_seg_files
        List of saved tumor segmentation files

    """
    if modalities is None:
        modalities = ['T1w', 'T1ce', 'T2w', 'FLAIR']

    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['source_file'] + [f'tumor_seg_{mod}' for mod in modalities]
        ),
        name='inputnode',
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['tumor_seg_files']),
        name='outputnode',
    )

    # Create datasink for each modality with DerivativesDataSink
    ds_nodes = {}
    for modality in modalities:
        ds_node = pe.Node(
            DerivativesDataSink(
                check_hdr=False,
                desc=f'tumor{modality}',
                suffix='dseg',
            ),
            name=f'ds_tumor_seg_{modality.lower()}',
            run_without_submitting=True,
        )
        ds_node.inputs.base_directory = output_dir
        ds_nodes[modality] = ds_node

        workflow.connect([
            (inputnode, ds_node, [
                ('source_file', 'source_file'),
                (f'tumor_seg_{modality}', 'in_file'),
            ]),
        ])

    # Merge all outputs
    merge_node = pe.Node(
        niu.Merge(len(modalities)),
        name='merge_outputs',
        run_without_submitting=True,
    )

    # Connect all datasinks to merge node
    for i, modality in enumerate(modalities, start=1):
        workflow.connect([
            (ds_nodes[modality], merge_node, [('out_file', f'in{i}')]),
        ])

    workflow.connect([
        (merge_node, outputnode, [('out', 'tumor_seg_files')]),
    ])

    return workflow


def init_ds_tumor_roi_stats_wf(
    *,
    output_dir: str,
    name: str = 'ds_tumor_roi_stats_wf',
) -> Workflow:
    """Save tumor ROI statistics (intensity, volume, location).

    Parameters
    ----------
    output_dir : :obj:`str`
        Directory in which to save derivatives
    name : :obj:`str`
        Workflow name (default: ds_tumor_roi_stats_wf)

    Inputs
    ------
    source_files
        List of input anatomical images
    anatomical_image
        Preprocessed anatomical image
    tumor_mask
        Binary tumor mask
    tumor_seg
        Labeled tumor segmentation

    Outputs
    -------
    stats_file
        Path to saved statistics JSON file

    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['source_files', 'anatomical_image', 'tumor_mask', 'tumor_seg']
        ),
        name='inputnode',
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['stats_file']),
        name='outputnode',
    )

    compute_stats = pe.Node(
        niu.Function(
            function=_compute_tumor_roi_stats, #PLACEHOLDER: replace with PyRadiomics-based functions
            input_names=['anatomical_image', 'tumor_mask', 'tumor_seg', 'source_files', 'output_dir'],
            output_names=['stats_file'],
        ),
        name='compute_stats',
        run_without_submitting=True,
    )
    compute_stats.inputs.output_dir = output_dir

    workflow.connect([
        (inputnode, compute_stats, [
            ('anatomical_image', 'anatomical_image'),
            ('tumor_mask', 'tumor_mask'),
            ('tumor_seg', 'tumor_seg'),
            ('source_files', 'source_files'),
        ]),
        (compute_stats, outputnode, [('stats_file', 'stats_file')]),
    ])

    return workflow


def _save_tumor_metrics(tumor_metrics, source_files, output_dir):
    """Save tumor metrics to JSON file.

    Parameters
    ----------
    tumor_metrics : :obj:`dict`
        Dictionary of tumor metrics
    source_files : :obj:`list` of :obj:`str`
        Source file paths
    output_dir : :obj:`str`
        Output directory

    Returns
    -------
    metrics_file : :obj:`str`
        Path to saved metrics file

    """
    import json
    from pathlib import Path

    metrics_dir = Path(output_dir) / 'derivatives'
    metrics_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = metrics_dir / 'tumor_metrics.json'
    metrics_file.write_text(json.dumps(tumor_metrics, indent=2))

    return str(metrics_file)


def _compute_tumor_roi_stats(anatomical_image, tumor_mask, tumor_seg, source_files, output_dir):
    """Compute statistics for tumor ROI.

    Parameters
    ----------
    anatomical_image : :obj:`str`
        Path to anatomical image
    tumor_mask : :obj:`str`
        Path to tumor mask
    tumor_seg : :obj:`str`
        Path to tumor segmentation
    source_files : :obj:`list` of :obj:`str`
        Source file paths
    output_dir : :obj:`str`
        Output directory

    Returns
    -------
    stats_file : :obj:`str`
        Path to saved statistics file

    """
    import json
    from pathlib import Path

    import nibabel as nb
    import numpy as np

    stats = {}

    try:
        anat_img = nb.load(anatomical_image)
        anat_data = anat_img.get_fdata()

        mask_img = nb.load(tumor_mask)
        mask_data = mask_img.get_fdata() > 0

        seg_img = nb.load(tumor_seg)
        seg_data = seg_img.get_fdata()

        voxel_vol = np.prod(anat_img.header.get_zooms()[:3])

        # Compute statistics
        stats['tumor_volume_mm3'] = float(np.sum(mask_data) * voxel_vol)
        stats['tumor_mean_intensity'] = float(np.mean(anat_data[mask_data]))
        stats['tumor_std_intensity'] = float(np.std(anat_data[mask_data]))
        stats['tumor_min_intensity'] = float(np.min(anat_data[mask_data]))
        stats['tumor_max_intensity'] = float(np.max(anat_data[mask_data]))

        # Get tumor centroid
        tumor_coords = np.where(mask_data)
        if len(tumor_coords[0]) > 0:
            centroid = [
                float(np.mean(tumor_coords[0])),
                float(np.mean(tumor_coords[1])),
                float(np.mean(tumor_coords[2])),
            ]
            stats['tumor_centroid'] = centroid

    except Exception as e:
        print(f"Warning: Could not compute tumor stats: {e}")

    # Save to JSON
    stats_dir = Path(output_dir) / 'derivatives'
    stats_dir.mkdir(parents=True, exist_ok=True)
    stats_file = stats_dir / 'tumor_roi_stats.json'
    stats_file.write_text(json.dumps(stats, indent=2))

    return str(stats_file)


__all__ = [
    'init_ds_tumor_seg_wf',
    'init_ds_tumor_mask_wf',
    'init_ds_tumor_metrics_wf',
    'init_ds_multimodal_tumor_wf',
    'init_ds_tumor_roi_stats_wf',
]
