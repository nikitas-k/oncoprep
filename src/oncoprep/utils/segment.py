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
    """Return the Singularity/Apptainer executable name, or None.

    Searches PATH first, then checks common HPC module-install locations
    (e.g. NCI Gadi ``/opt/singularity/bin``).  Inside a Singularity
    container the host PATH is normally passed through, but some
    containers override it.
    """
    # Common HPC locations where modules install singularity/apptainer
    _HPC_SEARCH_PATHS = [
        "/opt/singularity/bin",
        "/opt/apptainer/bin",
        "/usr/local/bin",
        "/apps/singularity/bin",
    ]

    for cmd in ("apptainer", "singularity"):
        # First try PATH (fast)
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
            pass

        # Then try common HPC absolute paths
        for search_dir in _HPC_SEARCH_PATHS:
            full_path = os.path.join(search_dir, cmd)
            if os.path.isfile(full_path) and os.access(full_path, os.X_OK):
                try:
                    result = subprocess.run(
                        [full_path, "--version"],
                        capture_output=True,
                        check=False,
                        timeout=5,
                    )
                    if result.returncode == 0:
                        LOGGER.debug(
                            "Found %s at %s (not on PATH)", cmd, full_path
                        )
                        return full_path
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
    seg_cache_dir: Optional[Path] = None,
) -> str:
    """Determine which container runtime to use.

    Resolution order for ``"auto"``:
    1. If inside a Singularity/Apptainer container **and** Docker is not
       reachable, use whichever Singularity CLI is available.
    2. If Docker daemon is reachable, use ``"docker"``.
    3. If a Singularity/Apptainer CLI is found, use that.
    4. If inside a Singularity container (env vars set) and a seg-cache
       directory exists with pre-downloaded images, assume ``"singularity"``
       — the host binary will be resolved at execution time.
    5. Raise :class:`RuntimeError`.

    Parameters
    ----------
    requested : {"docker", "singularity", "apptainer", "auto"}
        Explicit runtime choice, or ``"auto"`` to detect.
    seg_cache_dir : Path, optional
        Directory containing pre-downloaded model images.  For Singularity
        runtimes this holds ``.sif`` files; for Docker, ``.tar`` files.
        When provided inside a Singularity/Apptainer container, returns
        ``"singularity"`` even when no CLI is on PATH.

    Returns
    -------
    str
        One of ``"docker"``, ``"singularity"``, or ``"apptainer"``.
    """
    if requested in ("singularity", "apptainer"):
        cmd = _find_singularity_cmd()
        if cmd is None:
            # If inside a container with pre-cached SIFs, trust the user
            if _is_inside_singularity():
                LOGGER.warning(
                    "Requested '%s' but CLI was not found on PATH. "
                    "Proceeding because we are inside a Singularity/Apptainer "
                    "container.  Ensure the host binary is bind-mounted or "
                    "available via --bind at `singularity exec` time.",
                    requested,
                )
                return requested
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
        # No CLI found, but we ARE inside Singularity — check for cached
        # SIF files.  The host `singularity` binary may not be on the
        # container PATH but will be available if the user binds it in or
        # uses `singularity exec` from the host.
        cache = seg_cache_dir or _resolve_seg_cache_dir()
        if cache is not None and cache.is_dir() and any(cache.glob("*.sif")):
            LOGGER.info(
                "Inside Singularity container — no CLI found but SIF cache "
                "directory '%s' contains pre-downloaded images.  Using "
                "'singularity' runtime.  Ensure the host binary is "
                "bind-mounted or on PATH.",
                cache,
            )
            return "singularity"
        # Last resort: still inside Singularity, maybe the user just
        # forgot --container-runtime. Return "singularity" with a warning.
        LOGGER.warning(
            "Inside a Singularity/Apptainer container but no container "
            "runtime CLI was found on PATH and no pre-cached model images "
            "were detected.  Pre-download segmentation models with:\n"
            "  oncoprep-models pull --output-dir /path/to/seg_cache\n"
            "then re-run with:\n"
            "  --container-runtime singularity --seg-cache-dir /path/to/seg_cache\n"
            "Attempting to continue with 'singularity' runtime."
        )
        return "singularity"

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
        "Apptainer to run segmentation models.\n\n"
        "On HPC systems, pre-download segmentation models on a login "
        "node with:\n"
        "  oncoprep-models pull --output-dir /path/to/seg_cache\n"
        "then run with:\n"
        "  --container-runtime singularity --seg-cache-dir /path/to/seg_cache"
    )


# ---------------------------------------------------------------------------
# Segmentation model cache helpers
# ---------------------------------------------------------------------------

def _resolve_seg_cache_dir() -> Optional[Path]:
    """Return the seg-cache directory if it already exists, or None.

    Unlike :func:`_default_seg_cache_dir`, this does **not** create the
    directory — it is safe to call during auto-detection without side effects.

    Checks (in order): ``ONCOPREP_SEG_CACHE``, ``ONCOPREP_SIF_CACHE``
    (legacy), ``SINGULARITY_CACHEDIR``, ``APPTAINER_CACHEDIR``, then the
    default ``~/.cache/oncoprep/seg``.
    """
    for var in ("ONCOPREP_SEG_CACHE", "ONCOPREP_SIF_CACHE",
                "SINGULARITY_CACHEDIR", "APPTAINER_CACHEDIR"):
        val = os.environ.get(var)
        if val:
            # Try the new subdir first, fall back to legacy
            for subdir in ("oncoprep_seg", "oncoprep_sif"):
                d = Path(val) / subdir
                if d.is_dir():
                    return d
    # Check default paths (new then legacy)
    for subdir in ("seg", "sif"):
        d = Path.home() / ".cache" / "oncoprep" / subdir
        if d.is_dir():
            return d
    return None


def _default_seg_cache_dir() -> Path:
    """Return the default directory for cached segmentation model files.

    Respects ``ONCOPREP_SEG_CACHE``, ``ONCOPREP_SIF_CACHE`` (legacy),
    ``SINGULARITY_CACHEDIR``, and ``APPTAINER_CACHEDIR`` environment
    variables (in that order).  Falls back to ``~/.cache/oncoprep/seg``.
    """
    for var in ("ONCOPREP_SEG_CACHE", "ONCOPREP_SIF_CACHE",
                "SINGULARITY_CACHEDIR", "APPTAINER_CACHEDIR"):
        val = os.environ.get(var)
        if val:
            d = Path(val) / "oncoprep_seg"
            d.mkdir(parents=True, exist_ok=True)
            return d
    d = Path.home() / ".cache" / "oncoprep" / "seg"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sif_path_for_image(image_id: str, cache_dir: Optional[Path] = None) -> Path:
    """Return the expected SIF file path for a Docker image ID.

    Converts ``owner/image:tag`` → ``owner_image_tag.sif``.
    """
    if cache_dir is None:
        cache_dir = _default_seg_cache_dir()
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
    seg_cache_dir: Optional[Path] = None,
) -> bool:
    """Check if a container image is available for the given runtime.

    Parameters
    ----------
    image_id : str
        Docker-style image ID
    runtime : str
        Container runtime ("docker", "singularity", or "apptainer")
    seg_cache_dir : Path, optional
        Cache directory for model images

    Returns
    -------
    bool
        True if the image is available
    """
    if runtime == "docker":
        return check_docker_image(image_id)
    return check_singularity_image(image_id, seg_cache_dir)


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
    seg_cache_dir: Optional[Path] = None,
    verbose: bool = True,
) -> bool:
    """Pull a container image using the appropriate runtime.

    Parameters
    ----------
    image_id : str
        Docker-style image ID
    runtime : str
        Container runtime ("docker", "singularity", or "apptainer")
    seg_cache_dir : Path, optional
        Cache directory for model images
    verbose : bool
        Print progress messages

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    if runtime == "docker":
        return pull_docker_image(image_id, verbose=verbose)
    return pull_singularity_image(image_id, seg_cache_dir, verbose=verbose)


# ---------------------------------------------------------------------------
# Unified ensure routine
# ---------------------------------------------------------------------------

def ensure_docker_images(
    docker_config: Dict[str, Dict],
    model_keys: Optional[List[str]] = None,
    verbose: bool = True,
    runtime: str = "docker",
    seg_cache_dir: Optional[Path] = None,
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
    seg_cache_dir : Path, optional
        Cache directory for model images

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

        if check_container_image(image_id, runtime, seg_cache_dir):
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
            if pull_container_image(image_id, runtime, seg_cache_dir, verbose=verbose):
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


def get_model_image_ids(
    gpu: bool = True,
    cpu: bool = True,
) -> Dict[str, str]:
    """Return a ``{model_key: docker_image_id}`` mapping from config files.

    Parameters
    ----------
    gpu : bool
        Include GPU models (default True).
    cpu : bool
        Include CPU models (default True).

    Returns
    -------
    dict
        Mapping of model key → Docker image ID.
    """
    import json as _json

    config_dir = Path(__file__).parent.parent / "config"
    images: Dict[str, str] = {}

    configs_to_load: List[Path] = []
    if gpu:
        p = config_dir / "gpu_dockers.json"
        if p.exists():
            configs_to_load.append(p)
    if cpu:
        p = config_dir / "cpu_dockers.json"
        if p.exists():
            configs_to_load.append(p)

    # Fall back to main dockers.json if neither cpu/gpu found
    if not configs_to_load:
        p = config_dir / "dockers.json"
        if p.exists():
            configs_to_load.append(p)

    for cfg_path in configs_to_load:
        with open(cfg_path) as fh:
            cfg = _json.load(fh)
        for key, entry in cfg.items():
            image_id = entry.get("id")
            if image_id and key not in images:
                images[key] = image_id

    return images


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
