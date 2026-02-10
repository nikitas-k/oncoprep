"""Segmentation utility functions for OncoPrep.

This module provides helper functions for tumor segmentation workflows,
including container runtime management (Docker and Singularity/Apptainer)
and GPU detection.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Dict, List, Literal, Optional

from oncoprep.utils.logging import get_logger

LOGGER = get_logger(__name__)

# Valid container runtime values
ContainerRuntime = Literal["docker", "singularity", "apptainer", "auto"]


# ---------------------------------------------------------------------------
# Container runtime detection
# ---------------------------------------------------------------------------

def _is_inside_singularity() -> bool:
    """Check if we are running inside a Singularity/Apptainer container."""
    return bool(
        os.environ.get("SINGULARITY_CONTAINER")
        or os.environ.get("APPTAINER_CONTAINER")
        or os.path.exists("/.singularity.d")
    )


def _find_singularity_cmd() -> Optional[str]:
    """Return the Singularity/Apptainer executable name, or None."""
    for cmd in ("apptainer", "singularity"):
        try:
            result = subprocess.run(
                [cmd, "--version"],
                capture_output=True,
                check=False,
                timeout=5,
            )
            if result.returncode == 0:
                return cmd
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None


def _docker_available() -> bool:
    """Return True if the Docker daemon is reachable."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            check=False,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def detect_container_runtime(
    requested: ContainerRuntime = "auto",
) -> str:
    """Determine which container runtime to use.

    Resolution order for ``"auto"``:
    1. If inside a Singularity/Apptainer container **and** Docker is not
       reachable, use whichever Singularity CLI is available.
    2. If Docker daemon is reachable, use ``"docker"``.
    3. If a Singularity/Apptainer CLI is found, use that.
    4. Raise :class:`RuntimeError`.

    Parameters
    ----------
    requested : {"docker", "singularity", "apptainer", "auto"}
        Explicit runtime choice, or ``"auto"`` to detect.

    Returns
    -------
    str
        One of ``"docker"``, ``"singularity"``, or ``"apptainer"``.
    """
    if requested in ("singularity", "apptainer"):
        cmd = _find_singularity_cmd()
        if cmd is None:
            raise RuntimeError(
                f"Requested container runtime '{requested}' but neither "
                "singularity nor apptainer were found on PATH."
            )
        # Honour user's preference (apptainer/singularity) if it exists;
        # fall back to whichever was actually found.
        if requested == "apptainer" and cmd == "apptainer":
            return "apptainer"
        if requested == "singularity" and cmd == "singularity":
            return "singularity"
        return cmd

    if requested == "docker":
        if not _docker_available():
            LOGGER.warning(
                "Requested Docker runtime but daemon is not reachable. "
                "Container operations may fail."
            )
        return "docker"

    # --- auto ---
    if _is_inside_singularity() and not _docker_available():
        cmd = _find_singularity_cmd()
        if cmd:
            LOGGER.info(
                "Running inside Singularity/Apptainer container and Docker "
                "is unavailable — using '%s' for segmentation models.",
                cmd,
            )
            return cmd

    if _docker_available():
        LOGGER.debug("Docker daemon reachable — using Docker runtime.")
        return "docker"

    cmd = _find_singularity_cmd()
    if cmd:
        LOGGER.info(
            "Docker not available — falling back to '%s' runtime.", cmd
        )
        return cmd

    raise RuntimeError(
        "No container runtime found. Install Docker, Singularity, or "
        "Apptainer to run segmentation models."
    )


# ---------------------------------------------------------------------------
# SIF cache helpers (Singularity/Apptainer)
# ---------------------------------------------------------------------------

def _default_sif_cache_dir() -> Path:
    """Return the default directory for cached SIF files.

    Respects ``ONCOPREP_SIF_CACHE``, ``SINGULARITY_CACHEDIR``, and
    ``APPTAINER_CACHEDIR`` environment variables (in that order).
    Falls back to ``~/.cache/oncoprep/sif``.
    """
    for var in ("ONCOPREP_SIF_CACHE", "SINGULARITY_CACHEDIR", "APPTAINER_CACHEDIR"):
        val = os.environ.get(var)
        if val:
            d = Path(val) / "oncoprep_sif"
            d.mkdir(parents=True, exist_ok=True)
            return d
    d = Path.home() / ".cache" / "oncoprep" / "sif"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sif_path_for_image(image_id: str, cache_dir: Optional[Path] = None) -> Path:
    """Return the expected SIF file path for a Docker image ID.

    Converts ``owner/image:tag`` → ``owner_image_tag.sif``.
    """
    if cache_dir is None:
        cache_dir = _default_sif_cache_dir()
    safe_name = image_id.replace("/", "_").replace(":", "_")
    return cache_dir / f"{safe_name}.sif"


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

def check_gpu_available() -> bool:
    """Check if NVIDIA GPU is available for containers.

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

    # Check 2: Docker with --gpus works (only if Docker available)
    if _docker_available():
        try:
            result = subprocess.run(
                ["docker", "run", "--rm", "--gpus", "all",
                 "nvidia/cuda:11.0-base", "nvidia-smi"],
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


# ---------------------------------------------------------------------------
# Image availability checks (Docker + Singularity/Apptainer)
# ---------------------------------------------------------------------------

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


def check_singularity_image(
    image_id: str,
    cache_dir: Optional[Path] = None,
) -> bool:
    """Check if a Singularity/Apptainer SIF file exists for the image.

    Parameters
    ----------
    image_id : str
        Docker-style image ID (e.g., 'fabianisensee/isen2018')
    cache_dir : Path, optional
        Directory containing cached SIF files

    Returns
    -------
    bool
        True if a SIF file exists, False otherwise
    """
    sif = _sif_path_for_image(image_id, cache_dir)
    return sif.is_file()


def check_container_image(
    image_id: str,
    runtime: str = "docker",
    sif_cache_dir: Optional[Path] = None,
) -> bool:
    """Check if a container image is available for the given runtime.

    Parameters
    ----------
    image_id : str
        Docker-style image ID
    runtime : str
        Container runtime ("docker", "singularity", or "apptainer")
    sif_cache_dir : Path, optional
        SIF cache directory (Singularity/Apptainer only)

    Returns
    -------
    bool
        True if the image is available
    """
    if runtime == "docker":
        return check_docker_image(image_id)
    return check_singularity_image(image_id, sif_cache_dir)


# ---------------------------------------------------------------------------
# Image pulling (Docker + Singularity/Apptainer)
# ---------------------------------------------------------------------------

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


def pull_singularity_image(
    image_id: str,
    cache_dir: Optional[Path] = None,
    verbose: bool = True,
) -> bool:
    """Pull a Docker image and convert it to a Singularity/Apptainer SIF.

    Parameters
    ----------
    image_id : str
        Docker-style image ID (pulled via ``docker://``)
    cache_dir : Path, optional
        Where to store the SIF file
    verbose : bool
        Print progress messages

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    cmd = _find_singularity_cmd()
    if cmd is None:
        LOGGER.error("Neither singularity nor apptainer found on PATH")
        return False

    sif = _sif_path_for_image(image_id, cache_dir)
    if sif.is_file():
        LOGGER.debug(f"SIF already exists: {sif}")
        return True

    docker_uri = f"docker://{image_id}"
    if verbose:
        LOGGER.info(f"Building SIF from {docker_uri} → {sif}")
        print(f"  Converting Docker image to SIF: {image_id}...")

    try:
        subprocess.run(
            [cmd, "pull", str(sif), docker_uri],
            check=True,
            capture_output=not verbose,
        )
        if verbose:
            LOGGER.info(f"Successfully built SIF: {sif}")
            print(f"  ✓ Downloaded: {image_id} → {sif.name}")
        return True
    except subprocess.CalledProcessError as e:
        LOGGER.error(f"Failed to build SIF for {image_id}: {e}")
        print(f"  ✗ Failed to download: {image_id}")
        # Clean up partial downloads
        if sif.exists():
            sif.unlink()
        return False


def pull_container_image(
    image_id: str,
    runtime: str = "docker",
    sif_cache_dir: Optional[Path] = None,
    verbose: bool = True,
) -> bool:
    """Pull a container image using the appropriate runtime.

    Parameters
    ----------
    image_id : str
        Docker-style image ID
    runtime : str
        Container runtime ("docker", "singularity", or "apptainer")
    sif_cache_dir : Path, optional
        SIF cache directory (Singularity/Apptainer only)
    verbose : bool
        Print progress messages

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    if runtime == "docker":
        return pull_docker_image(image_id, verbose=verbose)
    return pull_singularity_image(image_id, sif_cache_dir, verbose=verbose)


# ---------------------------------------------------------------------------
# Unified ensure routine
# ---------------------------------------------------------------------------

def ensure_docker_images(
    docker_config: Dict[str, Dict],
    model_keys: Optional[List[str]] = None,
    verbose: bool = True,
    runtime: str = "docker",
    sif_cache_dir: Optional[Path] = None,
) -> List[str]:
    """Ensure container images are available, downloading if necessary.

    Works with both Docker and Singularity/Apptainer runtimes.

    Parameters
    ----------
    docker_config : dict
        Docker configuration from dockers.json
    model_keys : list of str, optional
        Specific models to check. If None, checks all models.
    verbose : bool
        Print progress messages
    runtime : str
        Container runtime ("docker", "singularity", or "apptainer")
    sif_cache_dir : Path, optional
        SIF cache directory (Singularity/Apptainer only)

    Returns
    -------
    list of str
        List of model keys that are available
    """
    if model_keys is None:
        model_keys = list(docker_config.keys())

    available_models = []
    missing_models = []

    runtime_label = "Singularity/Apptainer" if runtime != "docker" else "Docker"
    if verbose:
        print(f"\nChecking segmentation model availability ({runtime_label})...")

    for key in model_keys:
        if key not in docker_config:
            LOGGER.warning(f"Model '{key}' not found in configuration")
            continue

        image_id = docker_config[key].get("id")
        if not image_id:
            LOGGER.warning(f"No image ID for model '{key}'")
            continue

        if check_container_image(image_id, runtime, sif_cache_dir):
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
            if pull_container_image(image_id, runtime, sif_cache_dir, verbose=verbose):
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
