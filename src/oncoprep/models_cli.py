"""CLI for managing OncoPrep segmentation model container images.

Pre-download segmentation model images for offline use on HPC systems
or air-gapped environments.

- **Singularity/Apptainer**: converts Docker images to ``.sif`` files
- **Docker**: saves images as ``.tar`` files (via ``docker save``)

Usage (Singularity, e.g. on an HPC login node)::

    # Pull all models (GPU + CPU)
    oncoprep-models pull --output-dir /scratch/$USER/seg_cache

    # Pull only CPU models
    oncoprep-models pull --output-dir /scratch/$USER/seg_cache --cpu-only

    # List available models without downloading
    oncoprep-models list

    # Check which models are already cached
    oncoprep-models status --output-dir /scratch/$USER/seg_cache

Usage (Docker, e.g. on a build machine)::

    oncoprep-models pull --output-dir /data/seg_cache --runtime docker

Then run the pipeline pointing to the cache::

    oncoprep /data/bids /data/output participant \\
        --run-segmentation \\
        --seg-cache-dir /path/to/seg_cache
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict

try:
    from oncoprep import __version__
except ImportError:
    __version__ = "0.1.0"


def _load_model_configs(
    gpu: bool = True, cpu: bool = True,
) -> Dict[str, dict]:
    """Load model configurations from bundled JSON files.

    Returns
    -------
    dict
        ``{model_key: full_config_dict}`` for all matching models.
    """
    config_dir = Path(__file__).parent / "config"
    models: Dict[str, dict] = {}

    configs_to_load = []
    if gpu:
        p = config_dir / "gpu_dockers.json"
        if p.exists():
            configs_to_load.append(p)
    if cpu:
        p = config_dir / "cpu_dockers.json"
        if p.exists():
            configs_to_load.append(p)

    # Fallback
    if not configs_to_load:
        p = config_dir / "dockers.json"
        if p.exists():
            configs_to_load.append(p)

    for cfg_path in configs_to_load:
        with open(cfg_path) as fh:
            cfg = json.load(fh)
        for key, entry in cfg.items():
            if key not in models:
                models[key] = entry

    return models


def _sif_name(image_id: str) -> str:
    """Convert Docker image ID to SIF filename."""
    return image_id.replace("/", "_").replace(":", "_") + ".sif"


def _tar_name(image_id: str) -> str:
    """Convert Docker image ID to tar filename."""
    return image_id.replace("/", "_").replace(":", "_") + ".tar"


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def _cmd_list(args: argparse.Namespace) -> int:
    """List available segmentation models."""
    gpu = not args.cpu_only
    cpu = not args.gpu_only
    models = _load_model_configs(gpu=gpu, cpu=cpu)

    if not models:
        print("No models found in configuration files.")
        return 1

    print(f"\nOncoPrep segmentation models ({len(models)} total):\n")
    print(f"  {'Model Key':<20} {'Runtime':<10} {'Docker Image ID'}")
    print(f"  {'─' * 20} {'─' * 10} {'─' * 40}")
    for key, cfg in sorted(models.items()):
        rt = cfg.get("runtime", "runc")
        img = cfg.get("id", "?")
        label = "GPU" if rt == "nvidia" else "CPU"
        print(f"  {key:<20} {label:<10} {img}")
    print()
    return 0


def _find_singularity_or_apptainer() -> str:
    """Find singularity/apptainer binary, searching common HPC paths."""
    import subprocess

    # Check PATH first
    for cmd in ("apptainer", "singularity"):
        try:
            r = subprocess.run([cmd, "--version"], capture_output=True, timeout=5)
            if r.returncode == 0:
                return cmd
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    # Search common HPC installation paths
    hpc_paths = [
        "/opt/singularity/bin",
        "/opt/apptainer/bin",
        "/usr/local/bin",
        "/apps/singularity/bin",
        "/apps/apptainer/bin",
        "/opt/software/singularity/bin",
    ]
    for dirpath in hpc_paths:
        for binary in ("apptainer", "singularity"):
            full = Path(dirpath) / binary
            if full.is_file():
                try:
                    r = subprocess.run(
                        [str(full), "--version"],
                        capture_output=True, timeout=5,
                    )
                    if r.returncode == 0:
                        return str(full)
                except (OSError, subprocess.TimeoutExpired):
                    continue

    return ""


def _is_inside_container() -> bool:
    """Check if we're running inside a Singularity/Apptainer container."""
    return bool(
        os.environ.get("SINGULARITY_CONTAINER")
        or os.environ.get("APPTAINER_CONTAINER")
        or os.environ.get("SINGULARITY_NAME")
        or os.environ.get("APPTAINER_NAME")
    )


def _cmd_pull(args: argparse.Namespace) -> int:
    """Pull model images and cache them locally."""
    import subprocess

    gpu = not args.cpu_only
    cpu = not args.gpu_only
    models = _load_model_configs(gpu=gpu, cpu=cpu)

    if not models:
        print("No models found in configuration files.")
        return 1

    # Resolve output directory
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter to requested models
    if args.models:
        requested = set(args.models)
        missing = requested - set(models.keys())
        if missing:
            print(f"WARNING: Unknown model(s): {', '.join(sorted(missing))}")
        models = {k: v for k, v in models.items() if k in requested}

    dry_run = getattr(args, "dry_run", False)

    # Determine which runtime to use for pulling
    runtime = getattr(args, "runtime", "auto")
    if runtime == "auto":
        sing = _find_singularity_or_apptainer()
        if sing:
            runtime = sing
        else:
            try:
                r = subprocess.run(
                    ["docker", "info"], capture_output=True, timeout=10
                )
                if r.returncode == 0:
                    runtime = "docker"
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
    elif runtime in ("singularity", "apptainer"):
        sing = _find_singularity_or_apptainer()
        if sing:
            runtime = sing

    # If we still can't find a runtime, switch to dry-run mode
    # (print shell commands the user can paste on the host)
    if runtime == "auto" or (
        runtime in ("singularity", "apptainer") and not dry_run
    ):
        # Check if the bare command actually exists before we try it
        if runtime in ("singularity", "apptainer"):
            try:
                subprocess.run(
                    [runtime, "--version"],
                    capture_output=True, timeout=5,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired):
                runtime = "auto"  # mark as unavailable

    if runtime == "auto":
        if _is_inside_container():
            print(
                "NOTE: Running inside a Singularity/Apptainer container where\n"
                "no container runtime is available. Printing shell commands\n"
                "you can copy-paste and run on the HOST (login node).\n",
                file=sys.stderr,
            )
            dry_run = True
            runtime = "singularity"  # default for printed commands
        else:
            print(
                "ERROR: No container runtime found.\n"
                "Install Docker, Singularity, or Apptainer, or specify\n"
                "  --runtime docker|singularity|apptainer\n"
                "Or use --dry-run to print the pull commands.",
                file=sys.stderr,
            )
            return 1

    use_docker = runtime == "docker"
    pull_cmd = runtime
    fmt = "tar" if use_docker else "sif"
    name_fn = _tar_name if use_docker else _sif_name

    if dry_run:
        return _print_pull_commands(
            models, output_dir, pull_cmd, use_docker, name_fn, fmt,
        )

    print(f"\nUsing {pull_cmd} to pull model images ({fmt}) → {output_dir}\n")

    total = len(models)
    success = 0
    skipped = 0
    failed = 0

    for i, (key, cfg) in enumerate(sorted(models.items()), 1):
        image_id = cfg.get("id")
        if not image_id:
            print(f"  [{i}/{total}] {key}: no image ID in config, skipping")
            skipped += 1
            continue

        out_path = output_dir / name_fn(image_id)

        if out_path.is_file() and not args.force:
            size_mb = out_path.stat().st_size / (1024 * 1024)
            print(f"  [{i}/{total}] {key}: already cached ({size_mb:.0f} MB) ✓")
            skipped += 1
            success += 1
            continue

        if use_docker:
            print(f"  [{i}/{total}] {key}: docker pull {image_id} …")
            try:
                subprocess.run(["docker", "pull", image_id], check=True)
                subprocess.run(
                    ["docker", "save", "-o", str(out_path), image_id],
                    check=True,
                )
                size_mb = out_path.stat().st_size / (1024 * 1024)
                print(f"           → {out_path.name} ({size_mb:.0f} MB) ✓")
                success += 1
            except subprocess.CalledProcessError as exc:
                print(f"           ✗ FAILED: {exc}", file=sys.stderr)
                if out_path.exists():
                    out_path.unlink()
                failed += 1
        else:
            docker_uri = f"docker://{image_id}"
            print(f"  [{i}/{total}] {key}: pulling {docker_uri} …")
            try:
                subprocess.run(
                    [pull_cmd, "pull", "--force", str(out_path), docker_uri],
                    check=True,
                )
                size_mb = out_path.stat().st_size / (1024 * 1024)
                print(f"           → {out_path.name} ({size_mb:.0f} MB) ✓")
                success += 1
            except subprocess.CalledProcessError as exc:
                print(f"           ✗ FAILED: {exc}", file=sys.stderr)
                if out_path.exists():
                    out_path.unlink()
                failed += 1

    print(f"\nDone: {success} pulled, {skipped} skipped, {failed} failed.")
    if success > 0:
        print(
            f"\nTo use these models at runtime:\n"
            f"  oncoprep <bids_dir> <output_dir> participant \\"
            f"\n      --run-segmentation \\"
            f"\n      --seg-cache-dir {output_dir}\n"
        )

    return 1 if failed > 0 else 0


def _print_pull_commands(
    models: Dict[str, dict],
    output_dir: Path,
    pull_cmd: str,
    use_docker: bool,
    name_fn,
    fmt: str,
) -> int:
    """Print shell commands for the user to run on the host."""
    print("#!/usr/bin/env bash")
    print("# OncoPrep segmentation model pull commands")
    print(f"# Run these on a login node with '{pull_cmd}' available.")
    print("#")
    print(f"# Output directory: {output_dir}")
    print(f"# Format: {fmt}")
    print("")
    print("set -e")
    print(f"mkdir -p {output_dir}")
    print()

    for key, cfg in sorted(models.items()):
        image_id = cfg.get("id")
        if not image_id:
            continue
        out_path = output_dir / name_fn(image_id)
        if use_docker:
            print(f"# {key}")
            print(f"docker pull {image_id}")
            print(f"docker save -o {out_path} {image_id}")
            print()
        else:
            print(f"# {key}")
            print(f"{pull_cmd} pull --force {out_path} docker://{image_id}")
            print()

    print(f"echo 'Done. Use --seg-cache-dir {output_dir} when running oncoprep.'")
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    """Show which models are cached vs missing."""
    gpu = not args.cpu_only
    cpu = not args.gpu_only
    models = _load_model_configs(gpu=gpu, cpu=cpu)

    output_dir = Path(args.output_dir).resolve()

    if not output_dir.is_dir():
        print(f"Cache directory does not exist: {output_dir}")
        return 1

    print(f"\nModel cache: {output_dir}\n")
    print(f"  {'Model Key':<20} {'Image ID':<35} {'Status'}")
    print(f"  {'─' * 20} {'─' * 35} {'─' * 20}")

    cached = 0
    missing = 0
    for key, cfg in sorted(models.items()):
        image_id = cfg.get("id", "")
        # Check for both .sif and .tar
        sif_path = output_dir / _sif_name(image_id)
        tar_path = output_dir / _tar_name(image_id)
        if sif_path.is_file():
            size_mb = sif_path.stat().st_size / (1024 * 1024)
            status = f"✓ cached sif ({size_mb:.0f} MB)"
            cached += 1
        elif tar_path.is_file():
            size_mb = tar_path.stat().st_size / (1024 * 1024)
            status = f"✓ cached tar ({size_mb:.0f} MB)"
            cached += 1
        else:
            status = "✗ missing"
            missing += 1
        print(f"  {key:<20} {image_id:<35} {status}")

    print(f"\n{cached} cached, {missing} missing.\n")
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def get_parser() -> argparse.ArgumentParser:
    """Build the ``oncoprep-models`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="oncoprep-models",
        description=(
            "Manage OncoPrep segmentation model container images.\n\n"
            "Pre-download models as SIF files (Singularity/Apptainer) or\n"
            "tar files (Docker) for offline or cached use."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=f"oncoprep-models {__version__}"
    )

    sub = parser.add_subparsers(dest="command", help="Available commands")

    # -- list --
    p_list = sub.add_parser("list", help="List available segmentation models")
    p_list.add_argument("--gpu-only", action="store_true", help="Show only GPU models")
    p_list.add_argument("--cpu-only", action="store_true", help="Show only CPU models")

    # -- pull --
    p_pull = sub.add_parser(
        "pull",
        help="Pull model images and save locally",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  oncoprep-models pull --output-dir /scratch/$USER/seg_cache\n"
            "  oncoprep-models pull --output-dir ./cache --cpu-only\n"
            "  oncoprep-models pull --output-dir ./cache --runtime docker\n"
            "  oncoprep-models pull --output-dir ./cache --models econib mic-dkfz\n"
        ),
    )
    p_pull.add_argument(
        "--output-dir", "-o",
        type=Path,
        required=True,
        help="Directory to store downloaded model files",
    )
    p_pull.add_argument("--gpu-only", action="store_true", help="Pull only GPU models")
    p_pull.add_argument("--cpu-only", action="store_true", help="Pull only CPU models")
    p_pull.add_argument(
        "--runtime", "-r",
        choices=["auto", "docker", "singularity", "apptainer"],
        default="auto",
        help="Container runtime to use for pulling (default: auto-detect)",
    )
    p_pull.add_argument(
        "--models", "-m",
        nargs="+",
        metavar="KEY",
        help="Pull only specific models (by key name)",
    )
    p_pull.add_argument(
        "--force", "-f",
        action="store_true",
        help="Re-download even if cached file already exists",
    )
    p_pull.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Print shell commands instead of executing them "
        "(useful inside containers where singularity is unavailable)",
    )

    # -- status --
    p_status = sub.add_parser(
        "status", help="Check which models are cached vs missing"
    )
    p_status.add_argument(
        "--output-dir", "-o",
        type=Path,
        required=True,
        help="Model cache directory to check",
    )
    p_status.add_argument("--gpu-only", action="store_true", help="Check only GPU models")
    p_status.add_argument("--cpu-only", action="store_true", help="Check only CPU models")

    return parser


def main() -> None:
    """Entry point for ``oncoprep-models``."""
    parser = get_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "list": _cmd_list,
        "pull": _cmd_pull,
        "status": _cmd_status,
    }
    sys.exit(dispatch[args.command](args))


if __name__ == "__main__":
    raise RuntimeError(
        "oncoprep/models_cli.py should not be run directly;\n"
        "Please `pip install` oncoprep and use the `oncoprep-models` command"
    )
