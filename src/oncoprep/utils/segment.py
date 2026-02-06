"""Segmentation utility functions for OncoPrep.

This module provides helper functions for tumor segmentation workflows,
including Docker container management and GPU detection.
"""

from __future__ import annotations

import subprocess
from typing import Dict, List, Optional

from oncoprep.utils.logging import get_logger

LOGGER = get_logger(__name__)


def check_gpu_available() -> bool:
    """Check if NVIDIA GPU is available for Docker containers.

    Returns
    -------
    bool
        True if GPU is available, False otherwise
    """
    # Check 1: nvidia-smi available and returns success
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            check=False,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_names = result.stdout.decode().strip().split('\n')
            LOGGER.info(f"GPU(s) detected: {', '.join(gpu_names)}")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Check 2: Docker with --gpus works
    try:
        result = subprocess.run(
            ["docker", "run", "--rm", "--gpus", "all", "nvidia/cuda:11.0-base", "nvidia-smi"],
            capture_output=True,
            check=False,
            timeout=30,
        )
        if result.returncode == 0:
            LOGGER.info("GPU available via Docker nvidia runtime")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    LOGGER.info("No GPU detected, will use CPU-only models")
    return False


def check_docker_image(image_id: str) -> bool:
    """Check if a Docker image is available locally.

    Parameters
    ----------
    image_id : str
        Docker image ID (e.g., 'fabianisensee/isen2018')

    Returns
    -------
    bool
        True if image exists locally, False otherwise
    """
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image_id],
            capture_output=True,
            check=False,
        )
        return result.returncode == 0
    except FileNotFoundError:
        LOGGER.error("Docker is not installed or not in PATH")
        return False


def pull_docker_image(image_id: str, verbose: bool = True) -> bool:
    """Pull a Docker image from registry.

    Parameters
    ----------
    image_id : str
        Docker image ID to pull
    verbose : bool
        Print progress messages

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    if verbose:
        LOGGER.info(f"Pulling Docker image: {image_id}")
        print(f"  Downloading segmentation model: {image_id}...")

    try:
        subprocess.run(
            ["docker", "pull", image_id],
            check=True,
            capture_output=not verbose,
        )
        if verbose:
            LOGGER.info(f"Successfully pulled: {image_id}")
            print(f"  ✓ Downloaded: {image_id}")
        return True
    except subprocess.CalledProcessError as e:
        LOGGER.error(f"Failed to pull {image_id}: {e}")
        print(f"  ✗ Failed to download: {image_id}")
        return False


def ensure_docker_images(
    docker_config: Dict[str, Dict],
    model_keys: Optional[List[str]] = None,
    verbose: bool = True,
) -> List[str]:
    """Ensure Docker images are available, downloading if necessary.

    Parameters
    ----------
    docker_config : dict
        Docker configuration from dockers.json
    model_keys : list of str, optional
        Specific models to check. If None, checks all models.
    verbose : bool
        Print progress messages

    Returns
    -------
    list of str
        List of model keys that are available
    """
    if model_keys is None:
        model_keys = list(docker_config.keys())

    available_models = []
    missing_models = []

    if verbose:
        print("\nChecking segmentation model availability...")

    for key in model_keys:
        if key not in docker_config:
            LOGGER.warning(f"Model '{key}' not found in configuration")
            continue

        image_id = docker_config[key].get("id")
        if not image_id:
            LOGGER.warning(f"No image ID for model '{key}'")
            continue

        if check_docker_image(image_id):
            if verbose:
                LOGGER.debug(f"Image available: {image_id}")
            available_models.append(key)
        else:
            missing_models.append((key, image_id))

    # Pull missing images
    if missing_models:
        if verbose:
            print(f"\n{len(missing_models)} model(s) need to be downloaded:")
            for key, image_id in missing_models:
                print(f"  - {key}: {image_id}")
            print()

        for key, image_id in missing_models:
            if pull_docker_image(image_id, verbose=verbose):
                available_models.append(key)
            else:
                LOGGER.warning(
                    f"Could not download model '{key}' ({image_id}). "
                    "It will be skipped during segmentation."
                )

    if verbose:
        print(f"\n{len(available_models)}/{len(model_keys)} models available.\n")

    return available_models


# BraTS Label Definitions
# -----------------------
# Old BraTS 2017-2020 labels (from raw model output):
#   1: Necrotic (NCR) - necrotic tumor core
#   2: Edema (ED) - peritumoral edema
#   3: Enhancing (ET) - enhancing tumor (original label 4 mapped to 3)
#   4: Resection cavity (RC) - optional, post-operative
#
# New BraTS 2021+ derived labels:
#   1: Enhancing Tumor (ET) - label 4 from old
#   2: Tumor Core (TC) - labels 1 + 4 from old (NCR + ET)
#   3: Whole Tumor (WT) - labels 1 + 2 + 4 from old (NCR + ED + ET)
#   4: Non-Enhancing Tumor Core (NETC) - label 1 from old (NCR only)
#   5: Surrounding Non-enhancing FLAIR Hyperintensity (SNFH) - label 2 from old (ED)
#   6: Resection Cavity (RC) - optional, post-operative

BRATS_OLD_LABELS = {
    1: 'NCR',   # Necrotic
    2: 'ED',    # Edema
    3: 'ET',    # Enhancing (remapped from 4)
    4: 'RC',    # Resection cavity (optional)
}

BRATS_NEW_LABELS = {
    1: 'ET',    # Enhancing Tumor
    2: 'TC',    # Tumor Core (NCR + ET)
    3: 'WT',    # Whole Tumor (NCR + ED + ET)
    4: 'NETC',  # Non-Enhancing Tumor Core
    5: 'SNFH',  # Surrounding Non-enhancing FLAIR Hyperintensity
    6: 'RC',    # Resection Cavity (optional)
}
