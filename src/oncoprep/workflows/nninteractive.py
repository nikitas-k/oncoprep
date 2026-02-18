"""nnInteractive tumor segmentation workflow.

Provides ``init_nninteractive_seg_wf()`` — a Nipype workflow that wraps the
nnInteractive [1]_ promptable 3D foundation model for fully-automated
multi-compartment (BraTS-style) tumor delineation.

nnInteractive is trained on 120+ diverse 3D datasets and performs zero-shot
inference on glioma MRI without any BraTS-specific fine-tuning.  It is the
default segmentation backend (``--default-seg``) and requires **no Docker
containers** — only a ~400 MB model checkpoint downloaded from HuggingFace.

Seed points for the promptable model are derived automatically from
multi-modal intensity anomaly maps (T1ce enhancement × T2 anomaly × FLAIR
hyperintensity).  See ``oncoprep.interfaces.nninteractive`` for details on
the heuristic algorithm.

The workflow has the **same inputnode/outputnode contract** as
``init_anat_seg_wf()`` in ``workflows/segment.py``, so it can be swapped in
as a drop-in replacement for the Docker-based ensemble segmentation.

References
----------
.. [1] Isensee, F. et al. (2025). nnInteractive: Redefining 3D Promptable
   Segmentation. *arXiv:2503.08373*. https://arxiv.org/abs/2503.08373
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from oncoprep.utils.logging import get_logger

LOGGER = get_logger(__name__)


# ── Nipype Function-node helpers ──────────────────────────────────────────────
# These must live at module level so Nipype can serialise them with cloudpickle.

def _convert_to_old_labels(seg_file):
    """Convert raw BraTS labels (1/2/4) → old scheme (1/2/3).

    Mapping: 1=NCR→1, 2=ED→2, 4=ET→3, 5=RC→4.
    """
    import os
    import logging
    import nibabel as nib
    import numpy as np

    if seg_file is None or not os.path.isfile(str(seg_file)):
        logging.getLogger('nipype.workflow').warning(
            'nnInteractive seg file missing — skipping old-label conversion'
        )
        return None

    img = nib.load(seg_file)
    data = np.asarray(img.dataobj)
    old = np.zeros_like(data, dtype=np.uint8)
    old[data == 1] = 1
    old[data == 2] = 2
    old[data == 4] = 3
    old[data == 5] = 4

    out_dir = os.path.abspath('tumor_labels')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'tumor_seg_old_labels.nii.gz')
    nib.save(nib.Nifti1Image(old, img.affine, img.header), out_path)
    return out_path


def _convert_to_new_labels(seg_file):
    """Convert raw BraTS labels (1/2/4) → new derived scheme.

    New labels: 1=ET, 2=TC(NCR+ET), 3=WT(all), 4=NETC(NCR only),
    5=SNFH(ED only), 6=RC.
    """
    import os
    import logging
    import nibabel as nib
    import numpy as np

    if seg_file is None or not os.path.isfile(str(seg_file)):
        logging.getLogger('nipype.workflow').warning(
            'nnInteractive seg file missing — skipping new-label conversion'
        )
        return None

    img = nib.load(seg_file)
    data = np.asarray(img.dataobj)

    ncr = data == 1
    ed = data == 2
    et = data == 4
    rc = data == 5

    new = np.zeros_like(data, dtype=np.uint8)
    new[ncr | ed | et] = 3   # WT
    new[ncr | et] = 2        # TC
    new[ed & ~ncr & ~et] = 5 # SNFH
    new[ncr & ~et] = 4       # NETC
    new[et] = 1              # ET
    new[rc] = 6              # RC

    out_dir = os.path.abspath('tumor_labels')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'tumor_seg_new_labels.nii.gz')
    nib.save(nib.Nifti1Image(new, img.affine, img.header), out_path)
    return out_path


# ── Workflow factory ──────────────────────────────────────────────────────────

def init_nninteractive_seg_wf(
    *,
    model_dir: Optional[str] = None,
    device: str = 'auto',
    name: str = 'anat_seg_wf',
) -> Workflow:
    """Create an nnInteractive tumor segmentation workflow.

    This workflow uses the nnInteractive promptable 3D segmentation model to
    perform fully-automated multi-compartment BraTS-style tumor delineation.
    Seed points are derived from multi-modal intensity anomalies — no existing
    segmentation or Docker containers are required.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from oncoprep.workflows.nninteractive import init_nninteractive_seg_wf
            wf = init_nninteractive_seg_wf()

    Parameters
    ----------
    model_dir : str | None
        Path to nnInteractive v1.0 model weights.  When *None*, the interface
        checks ``/tmp/nnInteractive_v1.0`` and
        ``~/.cache/oncoprep/nninteractive/`` before downloading from
        HuggingFace.
    device : str
        Torch device string — ``'auto'`` (default), ``'cuda'``, ``'mps'``, or
        ``'cpu'``.
    name : str
        Workflow name (default: ``'anat_seg_wf'``).

    Inputs
    ------
    source_file
        Source file for BIDS derivatives (typically T1w).
    t1w
        Raw T1w image (defines the reference voxel grid).
    t1ce
        Raw T1ce (contrast-enhanced) image.
    t2w
        Raw T2w image.
    flair
        FLAIR image registered to T1w space (preprocessed; raw FLAIR
        typically has thick slices that degrade after simple resampling).

    Outputs
    -------
    tumor_seg
        Raw tumor segmentation (BraTS labels 1=NCR, 2=ED, 4=ET).
    tumor_seg_old
        Old BraTS labels (1=NCR, 2=ED, 3=ET, 4=RC).
    tumor_seg_new
        New derived labels (1=ET, 2=TC, 3=WT, 4=NETC, 5=SNFH, 6=RC).

    Returns
    -------
    Workflow
    """
    from oncoprep.interfaces.nninteractive import NNInteractiveSegmentation

    workflow = Workflow(name=name)
    workflow.__desc__ = """\
Tumor segmentation was performed using *nnInteractive* [@isensee2025nninteractive],
a zero-shot promptable 3D foundation model trained on 120+ volumetric datasets.
Seed points were generated automatically from multi-modal intensity anomaly
scoring (T1ce enhancement × T2 anomaly × FLAIR hyperintensity) with adaptive
percentile thresholding.  Segmentation proceeded in three steps: enhancing
tumor (ET) delineation on T1ce, necrotic core (NCR) via region-filling, and
whole tumor (WT) on FLAIR.  Model weights (~400 MB) were obtained from
HuggingFace.  No Docker containers or BraTS-specific fine-tuning were required.
"""

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'source_file',
                't1w',
                't1ce',
                't2w',
                'flair',
            ],
        ),
        name='inputnode',
    )

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=['tumor_seg', 'tumor_seg_old', 'tumor_seg_new'],
        ),
        name='outputnode',
    )

    # Core segmentation node
    seg_node = pe.Node(
        NNInteractiveSegmentation(device=device),
        name='nninteractive_seg',
        mem_gb=8,
    )
    if model_dir is not None:
        seg_node.inputs.model_dir = str(model_dir)

    # Label conversion nodes (same old/new scheme as econib path)
    convert_old = pe.Node(
        niu.Function(
            function=_convert_to_old_labels,
            input_names=['seg_file'],
            output_names=['old_labels_file'],
        ),
        name='convert_to_old_labels',
    )

    convert_new = pe.Node(
        niu.Function(
            function=_convert_to_new_labels,
            input_names=['seg_file'],
            output_names=['new_labels_file'],
        ),
        name='convert_to_new_labels',
    )

    # Wire up
    workflow.connect([
        # Feed raw images — nnInteractive was trained on unprocessed data.
        # The interface resamples non-conforming modalities to the T1w grid.
        (inputnode, seg_node, [
            ('t1w', 't1w'),
            ('t1ce', 't1ce'),
            ('t2w', 't2w'),
            ('flair', 'flair'),
        ]),
        # Raw output
        (seg_node, outputnode, [('tumor_seg', 'tumor_seg')]),
        # Label conversions
        (seg_node, convert_old, [('tumor_seg', 'seg_file')]),
        (seg_node, convert_new, [('tumor_seg', 'seg_file')]),
        (convert_old, outputnode, [('old_labels_file', 'tumor_seg_old')]),
        (convert_new, outputnode, [('new_labels_file', 'tumor_seg_new')]),
    ])

    return workflow
