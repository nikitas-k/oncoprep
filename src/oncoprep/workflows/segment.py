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

LOGGER = get_logger(__name__)
iflogger = nipype_logging.getLogger('nipype.interface')


def _prepare_segmentation_inputs(
    t1: str,
    t1ce: str,
    t2: str,
    flair: str,
    output_dir: str,
    temp_dir: str,
    fileformats_config: Dict[str, Dict[str, str]],
) -> Dict[str, str]:
    """Prepare and save segmentation inputs to temporary directory.

    Parameters
    ----------
    t1, t1ce, t2, flair : str
        Paths to input MRI images
    output_dir : str
        Output directory for results
    temp_dir : str
        Temporary directory for Docker inputs
    fileformats_config : dict
        File format mapping configuration

    Returns
    -------
    dict
        Mapping of prepared file paths
    """
    import nibabel as nb
    from nipype.utils.filemanip import fname_presuffix

    inputs = {
        't1': t1,
        't1ce': t1ce,
        't2': t2,
        'flair': flair,
    }

    prepared = {}
    for key, img_path in inputs.items():
        if img_path:
            # Load and save to temp directory with standard names
            img = nb.load(img_path)
            # Use format config if available
            out_name = fileformats_config.get(key, f'{key}.nii.gz')
            out_path = os.path.join(temp_dir, out_name)
            nb.save(img, out_path)
            prepared[key] = out_path

    return prepared


def _run_segmentation_container(
    temp_dir: str,
    output_dir: str,
    container_id: str,
    config: Dict,
    gpu_id: str = "0",
    use_new_docker: bool = True,
    verbose: bool = True,
) -> bool:
    """Execute Docker container for segmentation.

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

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    # Build Docker command
    command = "docker run --rm"

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
    mountpoint = config.get("mountpoint", "/data")
    command += f" -v {temp_dir}:{mountpoint}"

    # Container ID and execution command
    command += f" {config['id']} {config.get('command', 'segment')}"

    if verbose:
        LOGGER.info(f"Executing container: {command}")

    try:
        subprocess.run(command, shell=True, check=True, capture_output=True)
        LOGGER.info(f"Container {container_id} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        LOGGER.error(f"Container {container_id} failed: {e.stderr.decode()}")
        return False


def _find_segmentation_result(results_dir: str, container_id: str) -> Optional[str]:
    """Find segmentation output file in results directory.

    Parameters
    ----------
    results_dir : str
        Directory containing results
    container_id : str
        Container identifier for matching

    Returns
    -------
    str or None
        Path to segmentation file, or None if not found
    """
    import glob

    # Search patterns in priority order
    patterns = [
        os.path.join(results_dir, f"tumor_{container_id}_class.nii*"),
        os.path.join(results_dir, "tumor_*_class.nii*"),
        os.path.join(results_dir, f"{container_id}*.nii*"),
        os.path.join(results_dir, "*tumor*.nii*"),
    ]

    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
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
                return best_file
            return matches[0]

    LOGGER.error(f"No segmentation output found in {results_dir}")
    return None


def _fuse_segmentations(
    segmentation_files: List[str],
    output_path: str,
    method: str = "majority",
) -> str:
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


def build_segmentation_workflow(
    bids_dir: Path,
    output_dir: Path,
    participant_label: Optional[List[str]] = None,
    session_label: Optional[List[str]] = None,
    nprocs: int = 1,
    omp_nthreads: int = 1,
    mem_gb: Optional[float] = None,
    use_gpu: bool = False,
    default_model: bool = True,
    model_path: Optional[Path] = None,
    sloppy: bool = False,
    name: str = "brats_segment",
) -> Workflow:
    """Build a BraTS-style segmentation workflow with nipreps compatibility.

    Performs tumor segmentation on preprocessed multi-modal MRI data using either
    the default model or a custom model via Docker containers.

    This workflow follows nipreps conventions and can integrate with other
    OncoPrep preprocessing workflows. It supports:
    - Single or multi-modal MRI inputs (T1, T1ce, T2, FLAIR)
    - GPU-accelerated execution when available
    - Multiple segmentation models with fusion
    - Proper BIDS derivatives output

    Parameters
    ----------
    bids_dir : Path
        Root directory of BIDS dataset
    output_dir : Path
        Output directory for results
    participant_label : list[str] | None
        List of participant IDs to process. If None, processes all.
    session_label : list[str] | None
        List of session IDs to process. If None, processes all.
    nprocs : int
        Number of parallel processes (default: 1)
    omp_nthreads : int
        Number of OpenMP threads per process (default: 1)
    mem_gb : float | None
        Memory limit in GB. If None, uses system default.
    use_gpu : bool
        Enable GPU acceleration if available (default: False)
    default_model : bool
        Use default segmentation model (default: True)
    model_path : Path | None
        Path to custom segmentation model. Overrides default_model if provided.
    sloppy : bool
        Use faster settings for quick testing (default: False)
    name : str
        Workflow name (default: 'brats_segment')

    Returns
    -------
    Workflow
        Nipype workflow object with segmentation pipeline

    Notes
    -----
    The workflow expects Docker to be installed and available. It will:
    1. Collect multi-modal MRI inputs for each subject
    2. Prepare inputs for Docker containers
    3. Execute segmentation containers (supporting multiple models)
    4. Fuse results if multiple models are used
    5. Save outputs in BIDS derivatives format

    Requires:
    - Docker with GPU support (nvidia-docker or --gpus flag)
    - Segmentation model containers properly configured
    - Configuration files in package config directory
    """
    bids_dir = Path(bids_dir)
    output_dir = Path(output_dir)

    workflow = LiterateWorkflow(name=name)
    workflow.base_dir = str(output_dir / ".nipype")

    # Determine package config directory
    pkg_dir = Path(__file__).parent.parent
    config_dir = pkg_dir / "config"

    # Load configuration files
    docker_config_path = config_dir / "dockers.json"
    fileformats_path = config_dir / "fileformats.json"

    if not docker_config_path.exists():
        raise FileNotFoundError(f"Docker config not found: {docker_config_path}")

    with open(docker_config_path) as f:
        docker_config = json.load(f)

    fileformats = {}
    if fileformats_path.exists():
        with open(fileformats_path) as f:
            fileformats = json.load(f)

    LOGGER.info(
        "Initialized BraTS segmentation workflow: %s "
        "(participants: %s, sessions: %s, use_gpu=%s, nprocs=%d)",
        name,
        participant_label or "all",
        session_label or "all",
        use_gpu,
        nprocs,
    )

    # Input node: receives BIDS filenames and paths
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "source_file",  # BIDS source file for derivatives
                "t1_file",      # T1w image path
                "t1ce_file",    # T1w contrast-enhanced image path
                "t2_file",      # T2w image path
                "flair_file",   # FLAIR image path
            ]
        ),
        name="inputnode",
    )

    # Output node: provides results
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "tumor_seg",        # Tumor segmentation in native space
                "tumor_seg_report", # QC report
            ]
        ),
        name="outputnode",
    )

    # Prepare inputs for container
    prepare_inputs = pe.Node(
        niu.Function(
            function=_prepare_segmentation_inputs,
            input_names=[
                "t1",
                "t1ce",
                "t2",
                "flair",
                "output_dir",
                "temp_dir",
                "fileformats_config",
            ],
            output_names=["prepared"],
        ),
        name="prepare_inputs",
    )
    prepare_inputs.inputs.fileformats_config = fileformats

    # Create temporary working directory
    setup_tempdir = pe.Node(
        niu.Function(
            function=lambda output_dir: str(Path(output_dir) / "seg_work"),
            input_names=["output_dir"],
            output_names=["temp_dir"],
        ),
        name="setup_tempdir",
    )

    # Run segmentation container(s)
    # For default model or custom model
    if model_path or default_model:
        # Single model segmentation
        container_key = list(docker_config.keys())[0]  # Use first configured container
        container_cfg = docker_config[container_key]

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
                ],
                output_names=["success"],
            ),
            name="run_segmentation_container",
        )
        run_container.inputs.container_id = container_key
        run_container.inputs.config = container_cfg
        run_container.inputs.gpu_id = "0"
        run_container.inputs.use_new_docker = True
        run_container.inputs.verbose = True  # Always verbose for container execution

        # Extract results
        extract_results = pe.Node(
            niu.Function(
                function=_find_segmentation_result,
                input_names=["results_dir", "container_id"],
                output_names=["seg_file"],
            ),
            name="extract_segmentation_results",
        )
        extract_results.inputs.container_id = container_key

        workflow.connect(
            [
                (inputnode, prepare_inputs, [("t1_file", "t1")]),
                (inputnode, prepare_inputs, [("t1ce_file", "t1ce")]),
                (inputnode, prepare_inputs, [("t2_file", "t2")]),
                (inputnode, prepare_inputs, [("flair_file", "flair")]),
                (setup_tempdir, prepare_inputs, [("temp_dir", "temp_dir")]),
                (setup_tempdir, run_container, [("temp_dir", "temp_dir")]),
                (run_container, extract_results, [
                    ("temp_dir", "results_dir"),
                ]),
                (extract_results, outputnode, [("seg_file", "tumor_seg")]),
            ]
        )
    else:
        # Multi-model ensemble segmentation
        seg_files = []
        for container_key, container_cfg in docker_config.items():
            # Create separate container execution for each model
            run_container_node = pe.Node(
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
                    ],
                    output_names=["success"],
                ),
                name=f"run_container_{container_key}",
            )
            run_container_node.inputs.container_id = container_key
            run_container_node.inputs.config = container_cfg
            run_container_node.inputs.gpu_id = "0"
            run_container_node.inputs.use_new_docker = True
            run_container_node.inputs.verbose = True  # Always verbose for container execution

            extract_node = pe.Node(
                niu.Function(
                    function=_find_segmentation_result,
                    input_names=["results_dir", "container_id"],
                    output_names=["seg_file"],
                ),
                name=f"extract_{container_key}",
            )
            extract_node.inputs.container_id = container_key

            workflow.add_nodes([run_container_node, extract_node])
            workflow.connect(
                [
                    (setup_tempdir, run_container_node, [("temp_dir", "temp_dir")]),
                    (run_container_node, extract_node, [
                        ("temp_dir", "results_dir"),
                    ]),
                ]
            )
            seg_files.append(extract_node)

        # Fuse multiple segmentations
        fuse_segs = pe.Node(
            niu.Function(
                function=_fuse_segmentations,
                input_names=["segmentation_files", "output_path", "method"],
                output_names=["fused_seg"],
            ),
            name="fuse_segmentations",
        )
        fuse_segs.inputs.output_path = str(output_dir / "fused_segmentation.nii.gz")
        fuse_segs.inputs.method = "majority"

        workflow.connect(
            [
                (inputnode, prepare_inputs, [("t1_file", "t1")]),
                (inputnode, prepare_inputs, [("t1ce_file", "t1ce")]),
                (inputnode, prepare_inputs, [("t2_file", "t2")]),
                (inputnode, prepare_inputs, [("flair_file", "flair")]),
                (setup_tempdir, prepare_inputs, [("temp_dir", "temp_dir")]),
                (fuse_segs, outputnode, [("fused_seg", "tumor_seg")]),
            ]
        )

    # Setup workflow description
    workflow.__desc__ = f"""
## Segmentation

Brain tumor segmentation was performed using a DL-based model containerized
in Docker {"with GPU acceleration" if use_gpu else "using CPU execution"}.
The workflow supports multi-modal MRI inputs (T1, T1ce, T2, FLAIR) and
{"employs majority-voting ensemble fusion across multiple trained models." if not (model_path or default_model) else "uses a single trained segmentation model."}

"""

    return workflow
