from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from nipype import Workflow, logging as nipype_logging
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow

from oncoprep.utils.logging import get_logger
from oncoprep.utils.segment import (
    check_gpu_available,
    check_docker_image,
    pull_docker_image,
    ensure_docker_images,
    detect_container_runtime,
    _sif_path_for_image,
    _default_seg_cache_dir,
    BRATS_OLD_LABELS,
    BRATS_NEW_LABELS,
)

LOGGER = get_logger(__name__)
iflogger = nipype_logging.getLogger('nipype.interface')


# Nipype Function Node helpers
# These must remain in this file for Nipype's function serialization to work

def _prepare_segmentation_inputs(
    t1,
    t1ce,
    t2,
    flair,
    brain_mask,
    fileformats_config,
):
    """Prepare and save segmentation inputs to working directory.

    Creates a working directory in the current node's execution space and
    saves input images with standardized filenames for Docker container.
    Also creates symlinks for alternative naming conventions to support
    multiple models with different expected filenames.

    IMPORTANT: All images are skull-stripped using the brain mask to ensure
    background voxels are zero. BraTS containers (e.g., lfb_rwth) require
    this for proper processing.

    Parameters
    ----------
    t1, t1ce, t2, flair : str
        Paths to input MRI images
    brain_mask : str
        Path to brain mask (binary mask where 1=brain, 0=background)
    fileformats_config : dict
        File format mapping from fileformats.json for the specific model's format.
        Expected keys: 't1', 't1c', 't2', 'fla' with output filename values.

    Returns
    -------
    work_dir : str
        Path to working directory containing prepared files
    """
    import os
    import numpy as np
    import nibabel as nb
    import logging

    LOGGER = logging.getLogger('nipype.workflow')

    # Create working directory in current execution space
    work_dir = os.path.abspath('seg_inputs')
    os.makedirs(work_dir, exist_ok=True)

    # Load brain mask for skull-stripping
    mask_img = nb.load(brain_mask)
    mask_data = mask_img.get_fdata() > 0  # Binarize mask

    LOGGER.info(f"Loaded brain mask: {brain_mask}")
    LOGGER.info(f"Mask covers {100 * np.sum(mask_data) / mask_data.size:.1f}% of volume")

    # Map internal keys to fileformats.json keys
    # Internal: t1, t1ce, t2, flair -> Fileformat: t1, t1c, t2, fla
    key_mapping = {
        't1': 't1',
        't1ce': 't1c',
        't2': 't2',
        'flair': 'fla',
    }

    inputs = {
        't1': t1,
        't1ce': t1ce,
        't2': t2,
        'flair': flair,
    }

    # Track saved files for creating symlinks
    saved_files = {}

    for key, img_path in inputs.items():
        if img_path:
            # Load image
            img = nb.load(img_path)
            data = img.get_fdata()

            # Apply brain mask to ensure zero background
            # This is required by BraTS containers which assert background == 0
            masked_data = data * mask_data
            masked_img = nb.Nifti1Image(masked_data.astype(np.float32), img.affine, img.header)

            # Map key to fileformat key and get output filename
            fmt_key = key_mapping.get(key, key)
            out_name = fileformats_config.get(fmt_key, f'{key}.nii.gz')
            out_path = os.path.join(work_dir, out_name)
            nb.save(masked_img, out_path)
            saved_files[key] = out_name
            LOGGER.info(f"Prepared {key} -> {out_path} (skull-stripped)")

    # Create hard links or copies for alternative naming conventions
    # This allows multiple models with different expected filenames to work
    # from the same working directory.
    # Note: Using hard links (or copies) instead of symlinks because Docker
    # on macOS may not handle symlinks correctly when mounting volumes.
    import shutil
    alternative_names = {
        't1ce': ['t1c.nii.gz', 't1ce.nii.gz'],  # Some models expect t1c, others t1ce
        'flair': ['flair.nii.gz', 'fla.nii.gz'],  # Some models expect flair, others fla
    }

    for key, alt_names in alternative_names.items():
        if key in saved_files:
            primary_name = saved_files[key]
            primary_path = os.path.join(work_dir, primary_name)
            for alt_name in alt_names:
                if alt_name != primary_name:
                    alt_path = os.path.join(work_dir, alt_name)
                    # Remove existing file/symlink if present
                    if os.path.islink(alt_path) or os.path.exists(alt_path):
                        os.unlink(alt_path)
                    # Try hard link first, fall back to copy
                    try:
                        os.link(primary_path, alt_path)
                        LOGGER.info(f"Created hard link {alt_name} -> {primary_name}")
                    except (OSError, NotImplementedError):
                        # Hard links not supported, use copy
                        shutil.copy2(primary_path, alt_path)
                        LOGGER.info(f"Created copy {alt_name} from {primary_name}")

    LOGGER.info(f"Prepared input files in {work_dir}")
    return work_dir


def _run_segmentation_container(
    temp_dir,
    output_dir,
    container_id,
    config,
    gpu_id="0",
    use_new_docker=True,
    verbose=True,
    _inputs_ready=None,
    container_runtime="docker",
    seg_cache_dir=None,
):
    """Execute a segmentation container via Docker or Singularity/Apptainer.

    Parameters
    ----------
    temp_dir : str
        Input/output directory for container
    output_dir : str
        Final output directory
    container_id : str
        Container identifier
    config : dict
        Container configuration from config file
    gpu_id : str
        GPU device ID to use
    use_new_docker : bool
        Use new-style Docker GPU flags (--gpus vs --runtime=nvidia)
    verbose : bool
        Print execution details
    container_runtime : str
        Container runtime to use ("docker", "singularity", or "apptainer")
    seg_cache_dir : str or None
        Directory containing cached SIF files (Singularity/Apptainer only)

    Returns
    -------
    success : bool
        True if successful, False otherwise
    results_dir : str
        Path to results directory in the current node's working directory.
        This directory persists even when upstream nodes are cache-invalidated.
    """
    import os
    import platform
    import subprocess
    import logging
    LOGGER = logging.getLogger('nipype.workflow')

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    image_id = config['id']
    mountpoint = config.get("mountpoint", "/data")
    model_command = config.get('command', 'segment')

    if container_runtime in ("singularity", "apptainer"):
        # ---- Singularity / Apptainer path ----
        # Resolve the SIF file
        if seg_cache_dir is not None:
            from pathlib import Path as _Path
            safe_name = image_id.replace("/", "_").replace(":", "_")
            sif_path = str(_Path(seg_cache_dir) / f"{safe_name}.sif")
        else:
            # Fall back to default cache
            from pathlib import Path as _Path
            for var in ("ONCOPREP_SEG_CACHE", "SINGULARITY_CACHEDIR", "APPTAINER_CACHEDIR"):
                val = os.environ.get(var)
                if val:
                    cache = _Path(val) / "oncoprep_seg"
                    break
            else:
                cache = _Path.home() / ".cache" / "oncoprep" / "sif"
            safe_name = image_id.replace("/", "_").replace(":", "_")
            sif_path = str(cache / f"{safe_name}.sif")

        if not os.path.isfile(sif_path):
            LOGGER.error(
                f"SIF file not found: {sif_path}. "
                f"Run: {container_runtime} pull {sif_path} docker://{image_id}"
            )
            results_dst = os.path.abspath('results')
            os.makedirs(results_dst, exist_ok=True)
            return False, results_dst

        # Build Singularity/Apptainer command
        parts = [container_runtime, "exec"]

        # GPU support via --nv (NVIDIA) or --rocm (AMD)
        if config.get("runtime") == "nvidia":
            parts.append("--nv")

        # Writable tmpdir for containers that write to /tmp
        parts.extend(["--writable-tmpfs"])

        # Bind mount the data directory
        parts.extend(["--bind", f"{temp_dir}:{mountpoint}"])

        # Custom flags (adapt Docker flags → Singularity equivalents)
        # Singularity ignores Docker-only flags; we skip --user, --rm, etc.

        # SIF path
        parts.append(sif_path)

        # Execution command (split if needed)
        if model_command and model_command.strip():
            parts.extend(model_command.split())

        command = " ".join(parts)
    else:
        # ---- Docker path ----
        # If a seg-cache-dir is set, try loading image from a .tar file
        if seg_cache_dir is not None:
            from pathlib import Path as _Path
            safe_name = image_id.replace("/", "_").replace(":", "_")
            tar_path = _Path(seg_cache_dir) / f"{safe_name}.tar"
            if tar_path.is_file():
                LOGGER.info("Loading Docker image from cache: %s", tar_path)
                try:
                    subprocess.run(
                        ["docker", "load", "-i", str(tar_path)],
                        check=True, capture_output=True,
                    )
                except subprocess.CalledProcessError as e:
                    LOGGER.warning("docker load failed for %s: %s", tar_path, e)

        command = "docker run --rm"

        # Check if running on ARM64 (Apple Silicon) - add platform flag
        machine = platform.machine().lower()
        if machine in ('arm64', 'aarch64'):
            command += " --platform linux/amd64"
            LOGGER.warning(
                "Running x86_64 container on ARM64 via QEMU emulation. "
                "This may be slow. For faster processing, use an x86_64 machine."
            )

        # User mode flags
        if config.get("user_mode", False):
            command += " --user $(id -u):$(id -g)"

        # GPU flags
        if config.get("runtime") == "nvidia":
            if use_new_docker:
                command += f" --gpus device={gpu_id}"
            else:
                command += f" --runtime=nvidia -e CUDA_VISIBLE_DEVICES={gpu_id}"

        # Custom flags
        if "flags" in config:
            command += f" {config['flags']}"

        # Volume mapping
        command += f" -v {temp_dir}:{mountpoint}"

        # Container ID and execution command
        command += f" {image_id} {model_command}"

    if verbose:
        LOGGER.info(f"Executing container ({container_runtime}): {command}")

    try:
        subprocess.run(command, shell=True, check=True, capture_output=True)
        LOGGER.info(f"Container {container_id} completed successfully")
        success = True
    except subprocess.CalledProcessError as e:
        LOGGER.error(f"Container {container_id} failed: {e.stderr.decode()}")
        success = False

    # Copy results to node's own cwd so they survive prepare_inputs cache
    # invalidation.  When Nipype re-runs prepare_inputs (fresh directory)
    # but skips the Docker container (cached), the results directory inside
    # prepare_inputs is empty.  Keeping a copy here ensures extract_result
    # can still find them.
    import shutil as _shutil
    results_src = os.path.join(temp_dir, 'results')
    results_dst = os.path.abspath('results')
    if os.path.isdir(results_src) and os.path.abspath(results_src) != results_dst:
        if os.path.exists(results_dst):
            _shutil.rmtree(results_dst)
        _shutil.copytree(results_src, results_dst)
        LOGGER.info(f"Copied results to node cwd: {results_dst}")

    return success, results_dst


def _find_segmentation_result(results_dir, container_id, _wait=None):
    """Find segmentation output file in results directory.

    Copies the result to the current node's working directory so that
    the output path survives Nipype cache invalidation of upstream nodes
    (e.g. prepare_inputs being re-run without the Docker container
    re-running).

    Parameters
    ----------
    results_dir : str
        Directory containing results
    container_id : str
        Container identifier for matching
    _wait : any
        Dummy parameter to enforce dependency ordering

    Returns
    -------
    str or None
        Path to segmentation file (in node's own cwd), or None if not found
    """
    import os
    import glob
    import shutil
    import logging
    LOGGER = logging.getLogger('nipype.workflow')

    # Some containers (e.g. econib) write to a 'results' subdirectory
    results_subdir = os.path.join(results_dir, 'results')
    search_dirs = [results_dir]
    if os.path.isdir(results_subdir):
        search_dirs.insert(0, results_subdir)  # Search subdirectory first

    # Search patterns in priority order
    patterns = []
    for search_dir in search_dirs:
        patterns.extend([
            os.path.join(search_dir, f"tumor_{container_id}_class.nii*"),
            os.path.join(search_dir, "tumor_*_class.nii*"),
            os.path.join(search_dir, f"{container_id}*.nii*"),
            os.path.join(search_dir, "*tumor*.nii*"),
            # Match econib output pattern: tumor_fpeconib.nii.gz
            os.path.join(search_dir, "tumor_*.nii.gz"),
        ])

    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            # Filter out per-class segmentation files (e.g., tumor_*_1.nii.gz, tumor_*_2.nii.gz)
            # These are individual label files, not the combined segmentation
            import re
            combined_matches = [
                m for m in matches
                if not re.search(r'_[0-4]\.(nii|nii\.gz)$', m)
            ]
            # If we have combined files, use those; otherwise fall back to all matches
            matches = combined_matches if combined_matches else matches
            
            if len(matches) > 1:
                # Return file with most unique labels
                import nibabel as nb
                import numpy as np

                max_labels = 0
                best_file = matches[0]
                for match in matches:
                    img = nb.load(match)
                    unique_labels = len(np.unique(img.get_fdata()))
                    if unique_labels > max_labels:
                        max_labels = unique_labels
                        best_file = match
                LOGGER.warning(
                    f"Multiple segmentations found for {container_id}, "
                    f"selecting file with {max_labels} labels: {best_file}"
                )
                found = best_file
            else:
                found = matches[0]

            # Copy result into the current node's working directory so the
            # output path is independent of upstream nodes' directories.
            # This prevents "file not found" errors when Nipype invalidates
            # the prepare_inputs cache but skips the Docker container re-run.
            out_path = os.path.abspath(os.path.basename(found))
            if os.path.abspath(found) != out_path:
                shutil.copy2(found, out_path)
                LOGGER.info(f"Copied segmentation result to node cwd: {out_path}")
            return out_path

    LOGGER.error(f"No segmentation output found in {results_dir}")
    return None


def _fuse_segmentations(
    segmentation_files,
    output_path,
    method="majority",
):
    """Fuse multiple segmentation results.

    Parameters
    ----------
    segmentation_files : list of str
        Paths to segmentation files
    output_path : str
        Path for output fused segmentation
    method : str
        Fusion method: 'majority' or 'max'

    Returns
    -------
    str
        Path to fused segmentation
    """
    import nibabel as nb
    import numpy as np

    if len(segmentation_files) == 1:
        # Copy single result
        import shutil

        shutil.copy(segmentation_files[0], output_path)
        return output_path

    # Load all segmentations
    imgs = [nb.load(f).get_fdata() for f in segmentation_files]
    imgs_array = np.stack(imgs, axis=-1)

    # Apply fusion method
    if method == "majority":
        # Majority voting
        fused = np.squeeze(np.apply_along_axis(
            lambda x: np.bincount(x.astype(int), minlength=5).argmax(),
            axis=-1,
            arr=imgs_array,
        ))
    else:  # max
        fused = np.max(imgs_array, axis=-1)

    # Save result with same header as first image
    template_img = nb.load(segmentation_files[0])
    output_img = nb.Nifti1Image(fused, template_img.affine, template_img.header)
    nb.save(output_img, output_path)

    return output_path


# BraTS Label Definitions
# -----------------------
# See oncoprep.utils.segment for BRATS_OLD_LABELS and BRATS_NEW_LABELS dicts
#
# Old BraTS 2017-2020 labels (from raw model output):
#   1: Necrotic (NE) - necrotic tumor core
#   2: Edema (OE) - peritumoral edema
#   3: Enhancing (ET) - enhancing tumor (original label 4 mapped to 3)
#   4: Resection cavity (RC) - optional, post-operative
#
# New BraTS 2021+ derived labels:
#   1: Enhancing Tumor (ET) - label 4 from old
#   2: Tumor Core (TC) - labels 1 + 4 from old (NE + ET)
#   3: Whole Tumor (WT) - labels 1 + 2 + 4 from old (NE + OE + ET)
#   4: Non-Enhancing Tumor Core (NETC) - label 1 from old (NE only)
#   5: Surrounding Non-enhancing FLAIR Hyperintensity (SNFH) - label 2 from old (OE only)
#   6: Resection Cavity (RC) - optional, post-operative


def _convert_to_old_labels(seg_file):
    """Convert raw BraTS model output to old label scheme.
    
    Raw BraTS model outputs use label 4 for enhancing tumor.
    Old scheme remaps: 4 -> 3 for enhancing tumor.
    
    Parameters
    ----------
    seg_file : str or None
        Path to raw segmentation file, or None if segmentation failed
        
    Returns
    -------
    str or None
        Path to converted segmentation with old labels, or None if input is None
    """
    import os
    import nibabel as nib
    import numpy as np
    from pathlib import Path
    import logging
    
    if seg_file is None:
        logging.getLogger('nipype.workflow').warning(
            "Segmentation file is None, skipping old label conversion"
        )
        return None
    
    if not os.path.isfile(seg_file):
        logging.getLogger('nipype.workflow').warning(
            f"Segmentation file does not exist: {seg_file}, skipping old label conversion"
        )
        return None
    
    img = nib.load(seg_file)
    data = np.asarray(img.dataobj)
    
    # Create old label mapping
    # Raw: 1=NCR, 2=ED, 4=ET -> Old: 1=NCR, 2=ED, 3=ET
    old_labels = np.zeros_like(data, dtype=np.uint8)
    old_labels[data == 1] = 1  # NCR stays 1
    old_labels[data == 2] = 2  # ED stays 2
    old_labels[data == 4] = 3  # ET becomes 3
    # Resection cavity (if present as label 5 in raw) -> 4
    old_labels[data == 5] = 4
    
    # Save to node's working directory (not derivatives)
    out_dir = os.path.abspath('tumor_labels')
    os.makedirs(out_dir, exist_ok=True)
    out_path = str(Path(out_dir) / "tumor_seg_old_labels.nii.gz")
    
    out_img = nib.Nifti1Image(old_labels, img.affine, img.header)
    nib.save(out_img, out_path)
    
    return out_path


def _convert_to_new_labels(seg_file):
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
        Path to raw segmentation file, or None if segmentation failed
        
    Returns
    -------
    str or None
        Path to converted segmentation with new labels, or None if input is None
    """
    import os
    import nibabel as nib
    import numpy as np
    from pathlib import Path
    import logging
    
    if seg_file is None:
        logging.getLogger('nipype.workflow').warning(
            "Segmentation file is None, skipping new label conversion"
        )
        return None
    
    if not os.path.isfile(seg_file):
        logging.getLogger('nipype.workflow').warning(
            f"Segmentation file does not exist: {seg_file}, skipping new label conversion"
        )
        return None
    
    img = nib.load(seg_file)
    data = np.asarray(img.dataobj)
    
    # Extract raw labels
    # Raw BraTS: 1=NCR, 2=ED, 4=ET, 5=RC (optional)
    ncr_mask = (data == 1)
    ed_mask = (data == 2)
    et_mask = (data == 4)
    rc_mask = (data == 5)
    
    # Create new derived labels
    new_labels = np.zeros_like(data, dtype=np.uint8)
    
    # Priority order (lower labels overwritten by higher priority):
    # WT (3) = NCR + ED + ET - lowest priority for visualization
    new_labels[ncr_mask | ed_mask | et_mask] = 3
    
    # TC (2) = NCR + ET
    new_labels[ncr_mask | et_mask] = 2
    
    # SNFH (5) = ED only (peritumoral edema / FLAIR hyperintensity)
    new_labels[ed_mask & ~ncr_mask & ~et_mask] = 5
    
    # NETC (4) = NCR only (non-enhancing tumor core)
    new_labels[ncr_mask & ~et_mask] = 4
    
    # ET (1) = Enhancing tumor only - highest tumor priority
    new_labels[et_mask] = 1
    
    # RC (6) = Resection cavity (optional, post-op)
    new_labels[rc_mask] = 6
    
    # Save to node's working directory (not derivatives)
    out_dir = os.path.abspath('tumor_labels')
    os.makedirs(out_dir, exist_ok=True)
    out_path = str(Path(out_dir) / "tumor_seg_new_labels.nii.gz")
    
    out_img = nib.Nifti1Image(new_labels, img.affine, img.header)
    nib.save(out_img, out_path)
    
    return out_path


def init_anat_seg_wf(
    *,
    output_dir: Path,
    use_gpu: bool = True,
    default_model: bool = True,
    model_path: Optional[Path] = None,
    sloppy: bool = False,
    container_runtime: str = 'auto',
    seg_cache_dir: Optional[Path] = None,
    name: str = 'anat_seg_wf',
) -> Workflow:
    """
    Create tumor segmentation workflow for integration with anatomical preprocessing.

    This workflow receives preprocessed multi-modal MRI images from the anatomical
    preprocessing workflow and performs BraTS-style tumor segmentation using
    containerized deep learning models.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from oncoprep.workflows.segment import init_anat_seg_wf
            wf = init_anat_seg_wf(output_dir='/tmp')

    Parameters
    ----------
    output_dir : Path
        Output directory for derivatives
    use_gpu : bool
        Enable GPU acceleration for Docker containers (default: True)
    default_model : bool
        Use default segmentation model (default: True)
    model_path : Path | None
        Path to custom segmentation model
    sloppy : bool
        Use faster settings for testing (default: False)
    name : str
        Workflow name (default: anat_seg_wf)

    Inputs
    ------
    source_file
        Source file for BIDS derivatives (typically T1w)
    t1w_preproc
        Preprocessed T1w image
    t1ce_preproc
        Preprocessed T1ce image
    t2w_preproc
        Preprocessed T2w image
    flair_preproc
        Preprocessed FLAIR image
    brain_mask
        Brain mask (optional, for reference)

    Outputs
    -------
    tumor_seg
        Tumor segmentation in native anatomical space

    Returns
    -------
    Workflow
        Nipype workflow for tumor segmentation
    """
    from pathlib import Path
    from niworkflows.engine.workflows import LiterateWorkflow

    output_dir = Path(output_dir)
    workflow = LiterateWorkflow(name=name)

    # Input node: receives preprocessed modalities from anat_preproc_wf
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'source_file',    # For BIDS derivatives
                't1w_preproc',    # Preprocessed T1w
                't1ce_preproc',   # Preprocessed T1ce
                't2w_preproc',    # Preprocessed T2w
                'flair_preproc',  # Preprocessed FLAIR
                'brain_mask',     # Brain mask (optional reference)
            ]
        ),
        name='inputnode',
    )

    # Output node: provides segmentation results
    # tumor_seg: raw model output
    # tumor_seg_old: old BraTS labels (1=NCR, 2=ED, 3=ET, 4=RC)
    # tumor_seg_new: new derived labels (1=ET, 2=TC, 3=WT, 4=NETC, 5=SNFH, 6=RC)
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'tumor_seg',      # Raw tumor segmentation map
                'tumor_seg_old',  # Old BraTS labels
                'tumor_seg_new',  # New derived labels
            ]
        ),
        name='outputnode',
    )

    # Sloppy mode: skip actual segmentation and return dummy output
    if sloppy:
        LOGGER.warning(
            "SLOPPY MODE: Skipping actual tumor segmentation, "
            "returning empty mask for testing purposes"
        )

        def _create_dummy_segmentation(t1w_preproc, output_dir):
            """Create an empty segmentation mask matching T1w dimensions for testing."""
            import os
            import nibabel as nib
            import numpy as np
            from pathlib import Path

            # Load T1w to get dimensions and affine
            t1w_img = nib.load(t1w_preproc)
            data_shape = t1w_img.shape[:3]  # Get 3D shape

            # Create empty segmentation (all zeros)
            dummy_seg = np.zeros(data_shape, dtype=np.uint8)

            # Save to output directory
            out_dir = Path(output_dir) / "sloppy_seg"
            os.makedirs(out_dir, exist_ok=True)
            out_path = str(out_dir / "dummy_tumor_seg.nii.gz")

            seg_img = nib.Nifti1Image(dummy_seg, t1w_img.affine, t1w_img.header)
            nib.save(seg_img, out_path)

            return out_path

        dummy_seg_node = pe.Node(
            niu.Function(
                function=_create_dummy_segmentation,
                input_names=['t1w_preproc', 'output_dir'],
                output_names=['seg_file'],
            ),
            name='create_dummy_seg',
        )
        dummy_seg_node.inputs.output_dir = str(output_dir)

        workflow.connect([
            (inputnode, dummy_seg_node, [('t1w_preproc', 't1w_preproc')]),
            # In sloppy mode, all outputs are the same dummy segmentation
            (dummy_seg_node, outputnode, [
                ('seg_file', 'tumor_seg'),
                ('seg_file', 'tumor_seg_old'),
                ('seg_file', 'tumor_seg_new'),
            ]),
        ])

        return workflow

    # Detect container runtime (Docker vs Singularity/Apptainer)
    runtime = detect_container_runtime(container_runtime, seg_cache_dir=seg_cache_dir)
    LOGGER.info("Container runtime: %s", runtime)

    # Resolve SIF cache directory for Singularity/Apptainer
    if runtime != 'docker' and seg_cache_dir is not None:
        _sif_dir = Path(seg_cache_dir)
        _sif_dir.mkdir(parents=True, exist_ok=True)
    elif runtime != 'docker':
        _sif_dir = _default_seg_cache_dir()
    else:
        _sif_dir = None

    # Detect container runtime (Docker vs Singularity/Apptainer)
    runtime = detect_container_runtime(container_runtime, seg_cache_dir=seg_cache_dir)
    LOGGER.info("Container runtime: %s", runtime)

    # Resolve SIF cache directory for Singularity/Apptainer
    if runtime != 'docker' and seg_cache_dir is not None:
        _sif_dir = Path(seg_cache_dir)
        _sif_dir.mkdir(parents=True, exist_ok=True)
    elif runtime != 'docker':
        _sif_dir = _default_seg_cache_dir()
    else:
        _sif_dir = None

    # Check GPU availability and load appropriate config
    pkg_dir = Path(__file__).parent.parent
    config_dir = pkg_dir / "config"
    fileformats_path = config_dir / "fileformats.json"

    # Determine if GPU is available
    gpu_available = False
    if use_gpu:
        gpu_available = check_gpu_available()
        if not gpu_available:
            LOGGER.warning(
                "No GPU detected \u2014 falling back to CPU-only segmentation models. "
                "This is expected on CPU-only machines. To suppress this warning, "
                "pass --no-gpu."
            )
            print(
                "\n⚠️  No GPU detected. CPU-only segmentation models will be used.\n"
                "   To silence this warning, pass --no-gpu.\n"
            )

    # Load appropriate Docker config based on GPU availability
    if gpu_available:
        docker_config_path = config_dir / "gpu_dockers.json"
        if not docker_config_path.exists():
            # Fallback to main config
            docker_config_path = config_dir / "dockers.json"
        LOGGER.info("Using GPU-enabled segmentation models")
    else:
        docker_config_path = config_dir / "cpu_dockers.json"
        if not docker_config_path.exists():
            # Fallback to main config
            docker_config_path = config_dir / "dockers.json"
        LOGGER.info("Using CPU-only segmentation models")

    if not docker_config_path.exists():
        raise FileNotFoundError(f"Docker config not found: {docker_config_path}")

    with open(docker_config_path) as f:
        docker_config = json.load(f)

    fileformats = {}
    if fileformats_path.exists():
        with open(fileformats_path) as f:
            fileformats = json.load(f)

    # Check and download Docker images if necessary (skip for custom model_path)
    if model_path is not None:
        # Custom model path - skip Docker image checking
        LOGGER.info("Using custom model path, skipping Docker image checks")
        available_models = {}
    elif default_model:
        # Only check the default model
        model_keys_to_check = ["econib"]
        available_models = ensure_docker_images(
            docker_config,
            model_keys=model_keys_to_check,
            verbose=True,
            runtime=runtime,
            seg_cache_dir=_sif_dir,
        )
        if not available_models:
            raise RuntimeError(
                "Default segmentation model (econib/brats-2018) is not available. "
                "Please ensure Docker is installed and you have internet access."
            )
    else:
        # Check all models for ensemble
        model_keys_to_check = list(docker_config.keys())
        available_models = ensure_docker_images(
            docker_config,
            model_keys=model_keys_to_check,
            verbose=True,
            runtime=runtime,
            seg_cache_dir=_sif_dir,
        )
        if not available_models:
            raise RuntimeError(
                "No segmentation models are available. Please ensure Docker is "
                "installed and you have internet access to download models."
            )

    # LOGGER.info(
    #     "Initialized tumor segmentation workflow: %s (use_gpu=%s)",
    #     name,
    #     use_gpu,
    # )

    # Buffer node to gather all input modalities
    # The working directory will be created in the node's execution space
    inputbuffer = pe.Node(
        niu.IdentityInterface(
            fields=['t1', 't1ce', 't2', 'flair', 'brain_mask'],
        ),
        name='inputbuffer',
    )

    # Prepare inputs for Docker container
    # Creates working directory in node's execution space and returns path
    # Applies brain mask to all inputs to ensure zero background (required by BraTS containers)
    prepare_inputs = pe.Node(
        niu.Function(
            function=_prepare_segmentation_inputs,
            input_names=[
                "t1",
                "t1ce",
                "t2",
                "flair",
                "brain_mask",
                "fileformats_config",
            ],
            output_names=["work_dir"],
        ),
        name="prepare_inputs",
    )
    # Note: fileformats_config is set per-model below

    if model_path is not None:
        # Use custom segmentation model (single model, no fusion)
        LOGGER.info("ANAT Stage 7: Using custom segmentation model: %s", model_path)

        # Custom model uses default gz-b17 format
        model_format = fileformats.get("gz-b17", {})
        prepare_inputs.inputs.fileformats_config = model_format

        # Custom model configuration
        custom_model_cfg = {
            "id": str(model_path),
            "command": "",
            "mountpoint": "/data",
            "runtime": "nvidia" if use_gpu else "runc",
        }

        # Run custom segmentation container
        run_container = pe.Node(
            niu.Function(
                function=_run_segmentation_container,
                input_names=[
                    "temp_dir",
                    "output_dir",
                    "container_id",
                    "config",
                    "gpu_id",
                    "use_new_docker",
                    "verbose",
                    "_inputs_ready",
                    "container_runtime",
                    "seg_cache_dir",
                ],
                output_names=["success", "results_dir"],
            ),
            name="run_segmentation",
        )
        run_container.inputs.container_id = "custom"
        run_container.inputs.config = custom_model_cfg
        run_container.inputs.gpu_id = "0"  # Always set, used only if GPU enabled in config
        run_container.inputs.use_new_docker = True
        run_container.inputs.verbose = True
        run_container.inputs.output_dir = str(output_dir)
        run_container.inputs.container_runtime = runtime
        run_container.inputs.seg_cache_dir = str(_sif_dir) if _sif_dir else None

        # Extract segmentation result
        extract_result = pe.Node(
            niu.Function(
                function=_find_segmentation_result,
                input_names=["results_dir", "container_id", "_wait"],
                output_names=["seg_file"],
            ),
            name="extract_result",
        )
        extract_result.inputs.container_id = "custom"

        # Connect workflow for custom model
        workflow.connect([
            # Buffer inputs from preprocessed images
            (inputnode, inputbuffer, [
                ('t1w_preproc', 't1'),
                ('t1ce_preproc', 't1ce'),
                ('t2w_preproc', 't2'),
                ('flair_preproc', 'flair'),
                ('brain_mask', 'brain_mask'),
            ]),
            # Prepare inputs (creates work_dir in node's execution space)
            # Applies brain mask to ensure zero background for BraTS containers
            (inputbuffer, prepare_inputs, [
                ('t1', 't1'),
                ('t1ce', 't1ce'),
                ('t2', 't2'),
                ('flair', 'flair'),
                ('brain_mask', 'brain_mask'),
            ]),
            # Run container using prepare_inputs work_dir
            (prepare_inputs, run_container, [
                ('work_dir', 'temp_dir'),
                ('work_dir', '_inputs_ready'),
            ]),
            # Extract result from run_container's cwd AFTER container completes.
            # We use run_container.results_dir (not prepare_inputs.work_dir) so
            # the cached results copy survives prepare_inputs cache invalidation.
            (run_container, extract_result, [
                ('results_dir', 'results_dir'),
                ('success', '_wait'),
            ]),
            (extract_result, outputnode, [('seg_file', 'tumor_seg')]),
        ])

        # Add label conversion nodes
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

        workflow.connect([
            (extract_result, convert_old, [('seg_file', 'seg_file')]),
            (extract_result, convert_new, [('seg_file', 'seg_file')]),
            (convert_old, outputnode, [('old_labels_file', 'tumor_seg_old')]),
            (convert_new, outputnode, [('new_labels_file', 'tumor_seg_new')]),
        ])

        model_desc = f"a custom segmentation model ({model_path})"

    elif default_model:
        # Use econib BraTS 2018 as default model (CPU-compatible)
        container_key = "econib"  # Michal Marcinkiewicz's BraTS 2018 model
        if container_key not in docker_config:
            # Fallback to first available container
            container_key = list(docker_config.keys())[0]
            LOGGER.warning(
                "Default model 'econib' not found in config, using '%s'",
                container_key,
            )
        container_cfg = docker_config[container_key]

        # Get the model's file format from config
        model_fileformat = container_cfg.get("fileformat", "gz-b17")
        model_format = fileformats.get(model_fileformat, {})
        prepare_inputs.inputs.fileformats_config = model_format

        LOGGER.info("Using single model for segmentation: %s (format: %s)", container_key, model_fileformat)

        # Run segmentation container
        run_container = pe.Node(
            niu.Function(
                function=_run_segmentation_container,
                input_names=[
                    "temp_dir",
                    "output_dir",
                    "container_id",
                    "config",
                    "gpu_id",
                    "use_new_docker",
                    "verbose",
                    "_inputs_ready",
                    "container_runtime",
                    "seg_cache_dir",
                ],
                output_names=["success", "results_dir"],
            ),
            name="run_segmentation",
        )
        run_container.inputs.container_id = container_key
        run_container.inputs.config = container_cfg
        run_container.inputs.gpu_id = "0"  # Always set, used only if GPU enabled in config
        run_container.inputs.use_new_docker = True
        run_container.inputs.verbose = True
        run_container.inputs.output_dir = str(output_dir)
        run_container.inputs.container_runtime = runtime
        run_container.inputs.seg_cache_dir = str(_sif_dir) if _sif_dir else None

        # Extract segmentation result
        extract_result = pe.Node(
            niu.Function(
                function=_find_segmentation_result,
                input_names=["results_dir", "container_id", "_wait"],
                output_names=["seg_file"],
            ),
            name="extract_result",
        )
        extract_result.inputs.container_id = container_key

        # Connect workflow for single model
        workflow.connect([
            # Buffer inputs from preprocessed images
            (inputnode, inputbuffer, [
                ('t1w_preproc', 't1'),
                ('t1ce_preproc', 't1ce'),
                ('t2w_preproc', 't2'),
                ('flair_preproc', 'flair'),
                ('brain_mask', 'brain_mask'),
            ]),
            # Prepare inputs (creates work_dir in node's execution space)
            # Applies brain mask to ensure zero background for BraTS containers
            (inputbuffer, prepare_inputs, [
                ('t1', 't1'),
                ('t1ce', 't1ce'),
                ('t2', 't2'),
                ('flair', 'flair'),
                ('brain_mask', 'brain_mask'),
            ]),
            # Run container using prepare_inputs work_dir
            (prepare_inputs, run_container, [
                ('work_dir', 'temp_dir'),
                ('work_dir', '_inputs_ready'),
            ]),
            # Extract result from run_container's cwd AFTER container completes.
            # We use run_container.results_dir (not prepare_inputs.work_dir) so
            # the cached results copy survives prepare_inputs cache invalidation.
            (run_container, extract_result, [
                ('results_dir', 'results_dir'),
                ('success', '_wait'),
            ]),
            (extract_result, outputnode, [('seg_file', 'tumor_seg')]),
        ])

        # Add label conversion nodes
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

        workflow.connect([
            (extract_result, convert_old, [('seg_file', 'seg_file')]),
            (extract_result, convert_new, [('seg_file', 'seg_file')]),
            (convert_old, outputnode, [('old_labels_file', 'tumor_seg_old')]),
            (convert_new, outputnode, [('new_labels_file', 'tumor_seg_new')]),
        ])

        model_desc = f"the {container_key} model (Marcinkiewicz et al., BraTS 2018)"
    else:
        # Use all available models and fuse results
        model_keys = available_models  # Only use models that are available
        LOGGER.info(
            "ANAT Stage 7: Using multi-model ensemble for segmentation: %s",
            ", ".join(model_keys),
        )

        # For ensemble, use the first model's fileformat as default
        # (models should use consistent formats, or we'd need per-model prepare_inputs)
        if model_keys:
            first_model_cfg = docker_config[model_keys[0]]
            ensemble_fileformat = first_model_cfg.get("fileformat", "gz-b17")
            ensemble_format = fileformats.get(ensemble_fileformat, {})
            prepare_inputs.inputs.fileformats_config = ensemble_format
            LOGGER.info("Ensemble using file format: %s", ensemble_fileformat)
        else:
            # Fallback to default format
            prepare_inputs.inputs.fileformats_config = fileformats.get("gz-b17", {})

        # Create nodes for each model
        run_nodes = []
        extract_nodes = []
        for model_key in model_keys:
            model_cfg = docker_config[model_key]

            run_node = pe.Node(
                niu.Function(
                    function=_run_segmentation_container,
                    input_names=[
                        "temp_dir",
                        "output_dir",
                        "container_id",
                        "config",
                        "gpu_id",
                        "use_new_docker",
                        "verbose",
                        "_inputs_ready",
                        "container_runtime",
                        "seg_cache_dir",
                    ],
                    output_names=["success", "results_dir"],
                ),
                name=f"run_{model_key.replace('-', '_')}",
            )
            run_node.inputs.container_id = model_key
            run_node.inputs.config = model_cfg
            run_node.inputs.gpu_id = "0"  # Always set, used only if GPU enabled in config
            run_node.inputs.use_new_docker = True
            run_node.inputs.verbose = True
            run_node.inputs.output_dir = str(output_dir)
            run_node.inputs.container_runtime = runtime
            run_node.inputs.seg_cache_dir = str(_sif_dir) if _sif_dir else None
            run_nodes.append(run_node)

            extract_node = pe.Node(
                niu.Function(
                    function=_find_segmentation_result,
                    input_names=["results_dir", "container_id", "_wait"],
                    output_names=["seg_file"],
                ),
                name=f"extract_{model_key.replace('-', '_')}",
            )
            extract_node.inputs.container_id = model_key
            extract_nodes.append(extract_node)

        # Merge segmentation results
        merge_segs = pe.Node(
            niu.Merge(len(model_keys)),
            name="merge_segmentations",
        )

        # Initialize fusion sub-workflow using BraTS-specific SIMPLE algorithm
        from oncoprep.workflows.fusion import init_anat_seg_fuse_wf

        fusion_wf = init_anat_seg_fuse_wf(
            output_dir=str(output_dir),
            fusion_method="brats",  # BraTS-specific SIMPLE fusion with DICE weighting
            name="fusion_wf",
        )

        # Connect workflow for multi-model ensemble
        workflow.connect([
            # Buffer inputs from preprocessed images
            (inputnode, inputbuffer, [
                ('t1w_preproc', 't1'),
                ('t1ce_preproc', 't1ce'),
                ('t2w_preproc', 't2'),
                ('flair_preproc', 'flair'),
                ('brain_mask', 'brain_mask'),
            ]),
            # Prepare inputs (creates work_dir in node's execution space)
            # Applies brain mask to ensure zero background for BraTS containers
            (inputbuffer, prepare_inputs, [
                ('t1', 't1'),
                ('t1ce', 't1ce'),
                ('t2', 't2'),
                ('flair', 'flair'),
                ('brain_mask', 'brain_mask'),
            ]),
        ])

        # Connect each model's run and extract nodes with proper dependency
        for i, (run_node, extract_node) in enumerate(zip(run_nodes, extract_nodes)):
            workflow.connect([
                (prepare_inputs, run_node, [
                    ('work_dir', 'temp_dir'),
                    ('work_dir', '_inputs_ready'),
                ]),
                (run_node, extract_node, [
                    ('results_dir', 'results_dir'),
                    ('success', '_wait'),
                ]),  # extract gets results from run_node's cwd (cache-safe)
                (extract_node, merge_segs, [('seg_file', f'in{i+1}')]),
            ])

        # Connect fusion sub-workflow and outputs
        workflow.connect([
            # Pass merged segmentations to fusion workflow
            (merge_segs, fusion_wf, [('out', 'inputnode.segmentation_files')]),
            (inputnode, fusion_wf, [('t1w_preproc', 'inputnode.t1w_preproc')]),
            # Connect fusion outputs to main workflow outputs
            (fusion_wf, outputnode, [
                ('outputnode.fused_seg', 'tumor_seg'),
                ('outputnode.fused_seg_old', 'tumor_seg_old'),
                ('outputnode.fused_seg_new', 'tumor_seg_new'),
            ]),
        ])

        model_desc = f"an ensemble of {len(model_keys)} models with BraTS-specific SIMPLE fusion (consensus voting with DICE-based quality weighting)"

    workflow.__desc__ = f"""
## Tumor Segmentation

Brain tumor segmentation was performed using {model_desc}
{"with GPU acceleration" if use_gpu else "using CPU execution"}.
Multi-modal MRI inputs (T1w, T1ce, T2w, FLAIR) were used for segmentation.

### Output Labels

**Old BraTS Labels (2017-2020):**
- Label 1: Necrotic tumor (NT)
- Label 2: Peritumoral edema (OE)
- Label 3: Enhancing tumor (ET)
- Label 4: Resection cavity (RC, optional)

**New Derived Labels (2021+):**
- Label 1: Enhancing Tumor (ET)
- Label 2: Tumor Core (TC = NT + ET)
- Label 3: Whole Tumor (WT = NT + OE + ET)
- Label 4: Non-Enhancing Tumor Core (NETC)
- Label 5: Surrounding Non-enhancing FLAIR Hyperintensity (SNFH)
- Label 6: Resection Cavity (RC, optional)

"""

    return workflow
