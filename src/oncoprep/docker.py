#!/usr/bin/env python
"""
The *OncoPrep* on Docker wrapper.

This is a lightweight Python wrapper to run *OncoPrep* inside a Docker
container.  Docker (or Podman) must be installed and running.  This can
be checked running::

    docker info

The wrapper transparently maps host paths into the container, handles
GPU pass-through, TemplateFlow caching, and forwards all unknown flags
directly to the containerised ``oncoprep`` entrypoint.

Install the wrapper with pip::

    pip install oncoprep          # provides both `oncoprep` and `oncoprep-docker`

Then run ``oncoprep-docker`` exactly as you would run ``oncoprep`` on a
bare-metal installation::

    oncoprep-docker /path/to/bids /path/to/output participant \\
        --participant-label 001 --run-segmentation

Please report any feedback to https://github.com/nikitas-k/oncoprep/issues
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

try:
    from oncoprep import __version__
except ImportError:
    __version__ = '0.1.0'

__bugreports__ = 'https://github.com/nikitas-k/oncoprep/issues'

DEFAULT_IMAGE = 'nko11/oncoprep:{}'.format(__version__)

# Templates that ship inside the container — no extra bind-mount needed
TF_TEMPLATES = (
    'MNI152NLin2009cAsym',
    'OASIS30ANTs',
    'fsLR',
    'fsaverage',
    'fsaverage5',
)

MISSING = """
Image '{}' is missing.
Would you like to download it? [Y/n] """


# ---------------------------------------------------------------------------
# Docker sanity checks
# ---------------------------------------------------------------------------

def check_docker() -> int:
    """Verify Docker is installed and the user can talk to the daemon.

    Returns
    -------
    -1  Docker not found
     0  Docker found, cannot connect to daemon
     1  All OK
    """
    try:
        ret = subprocess.run(
            ['docker', 'version'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        from errno import ENOENT
        if exc.errno == ENOENT:
            return -1
        raise
    if ret.returncode != 0 or (
        ret.stderr and b'Cannot connect to the Docker daemon' in ret.stderr
    ):
        return 0
    return 1


def check_image(image: str) -> bool:
    """Return *True* if *image* exists locally."""
    ret = subprocess.run(
        ['docker', 'images', '-q', image],
        stdout=subprocess.PIPE,
    )
    return bool(ret.stdout.strip())


def check_memory(image: str) -> int:
    """Return total RAM (MB) visible inside the container, or -1 on error."""
    ret = subprocess.run(
        ['docker', 'run', '--rm', '--entrypoint=free', image, '-m'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if ret.returncode:
        return -1
    for line in ret.stdout.splitlines():
        if line.startswith(b'Mem:'):
            return int(line.split()[1])
    return -1


def _gpu_flags() -> list[str]:
    """Return Docker flags for NVIDIA GPU pass-through.

    Falls back to an empty list if ``nvidia-smi`` is not on PATH or the
    Docker ``nvidia`` runtime is unavailable.
    """
    try:
        ret = subprocess.run(
            ['nvidia-smi'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if ret.returncode == 0:
            return ['--gpus', 'all']
    except (OSError, FileNotFoundError):
        pass
    return []


def is_in_directory(filepath: str, directory: str) -> bool:
    return os.path.realpath(filepath).startswith(
        os.path.realpath(directory) + os.sep
    )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

class _ToDict(argparse.Action):
    """Parse ``KEY=VALUE`` pairs into a dict."""

    def __call__(self, parser, namespace, values, option_string=None):
        d = {}
        for kv in values:
            k, v = kv.split('=', 1)
            d[k] = os.path.abspath(v)
        setattr(namespace, self.dest, d)


def get_parser() -> argparse.ArgumentParser:
    """Build the ``oncoprep-docker`` argument parser.

    Only the arguments that need *special handling* (path mapping, GPU,
    developer options) are defined here.  Everything else is captured by
    ``parse_known_args`` and forwarded verbatim to the container.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
    )

    # ---- Standard BIDS-App positional arguments ----------------------------
    parser.add_argument(
        'bids_dir', nargs='?', type=os.path.abspath, default='',
    )
    parser.add_argument(
        'output_dir', nargs='?', type=os.path.abspath, default='',
    )
    parser.add_argument(
        'analysis_level', nargs='?',
        choices=['participant', 'group'], default='participant',
    )

    parser.add_argument(
        '-h', '--help', action='store_true',
        help='Show this help message and exit',
    )
    parser.add_argument(
        '--version', action='store_true',
        help="Show program's version number and exit",
    )

    # ---- Container selection -----------------------------------------------
    parser.add_argument(
        '-i', '--image', metavar='IMG', type=str, default=DEFAULT_IMAGE,
        help='Docker image to use (default: %(default)s)',
    )

    # ---- Wrapper-only options that require bind-mounts ---------------------
    g_wrap = parser.add_argument_group(
        'Wrapper options',
        'Standard options that require mapping files/directories into the '
        'container.  See ``oncoprep --help`` for full descriptions.',
    )
    g_wrap.add_argument(
        '-w', '--work-dir', action='store', type=os.path.abspath,
        help='Path where intermediate results should be stored',
    )
    g_wrap.add_argument(
        '-d', '--derivatives', nargs='+', metavar='PATH', action=_ToDict,
        help='Search PATH(s) for pre-computed derivatives (NAME=PATH …)',
    )
    g_wrap.add_argument(
        '--bids-filter-file', metavar='PATH', type=os.path.abspath,
        help='JSON file describing custom BIDS input filters',
    )
    g_wrap.add_argument(
        '--use-plugin', metavar='PATH', type=os.path.abspath,
        help='Nipype plugin configuration file',
    )
    g_wrap.add_argument(
        '--seg-model-path', metavar='PATH', type=os.path.abspath,
        help='Path to a custom segmentation model directory',
    )
    g_wrap.add_argument(
        '--output-spaces', nargs='*',
        help='Output spaces for template registration',
    )
    g_wrap.add_argument(
        '--templateflow-home', metavar='PATH', type=os.path.abspath,
        help='Path to a local TemplateFlow directory to bind-mount into the '
        'container (overrides the pre-fetched templates inside the image)',
    )

    # ---- GPU options -------------------------------------------------------
    g_gpu = parser.add_argument_group('GPU options')
    g_gpu.add_argument(
        '--no-gpu', action='store_true', default=False,
        help='Disable GPU pass-through (by default, GPUs are auto-detected)',
    )

    # ---- Developer / debugging options -------------------------------------
    g_dev = parser.add_argument_group(
        'Developer options', 'Tools for testing and debugging OncoPrep',
    )
    g_dev.add_argument(
        '--patch', nargs='+', metavar='PACKAGE=PATH', action=_ToDict,
        help='Sequence of PACKAGE=PATH to patch into the container Python env.',
    )
    g_dev.add_argument(
        '--shell', action='store_true',
        help='Open an interactive shell inside the container instead of '
        'running OncoPrep',
    )
    g_dev.add_argument(
        '--config', metavar='PATH', action='store', type=os.path.abspath,
        help='Use a custom nipype.cfg file',
    )
    g_dev.add_argument(
        '-e', '--env', action='append', nargs=2, metavar=('VAR', 'VALUE'),
        help='Set custom environment variables inside the container',
    )
    g_dev.add_argument(
        '-u', '--user', action='store',
        help='Run container as a given UID[:GID]',
    )
    g_dev.add_argument(
        '--network', action='store',
        help='Run container with a different network driver '
        '("none" to simulate no internet)',
    )
    g_dev.add_argument(
        '--no-tty', action='store_true',
        help='Run docker without the -it TTY flag',
    )

    return parser


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """Entry point for ``oncoprep-docker``."""
    parser = get_parser()
    opts, unknown_args = parser.parse_known_args()

    # Default to help if nothing provided
    if (opts.bids_dir, opts.output_dir, opts.version) == ('', '', False):
        opts.help = True

    # --- Docker checks ------------------------------------------------------
    check = check_docker()
    if check < 1:
        if opts.version:
            print('oncoprep-docker wrapper {}'.format(__version__))
        if opts.help:
            parser.print_help()
        if check == -1:
            print(
                'oncoprep-docker: Could not find the `docker` command. '
                'Is Docker installed?'
            )
        else:
            print(
                'oncoprep-docker: Cannot connect to the Docker daemon. '
                "Make sure you have permission to run 'docker'."
            )
        return 1

    # --- Pull image if missing ----------------------------------------------
    if not check_image(opts.image):
        resp = 'Y'
        if opts.version:
            print('oncoprep-docker wrapper {}'.format(__version__))
        if opts.help:
            parser.print_help()
        if opts.version or opts.help:
            try:
                resp = input(MISSING.format(opts.image))
            except KeyboardInterrupt:
                print()
                return 1
        if resp not in ('y', 'Y', ''):
            return 0
        print('Downloading image. This may take a while…')

    # --- Memory check -------------------------------------------------------
    mem_total = check_memory(opts.image)
    if mem_total == -1:
        print(
            'WARNING: Could not detect memory capacity of Docker container.\n'
            'Do you have permission to run docker?'
        )
    elif not (opts.help or opts.version) and mem_total < 8000:
        print(
            'WARNING: <8 GB of RAM is available inside the Docker environment.\n'
            'Some parts of OncoPrep may fail to complete.'
        )

    # --- Docker version string ----------------------------------------------
    ret = subprocess.run(
        ['docker', 'version', '--format', '{{.Server.Version}}'],
        stdout=subprocess.PIPE,
    )
    docker_version = ret.stdout.decode('ascii').strip()

    # --- Build docker run command -------------------------------------------
    command = [
        'docker', 'run', '--rm',
        '-e', 'DOCKER_VERSION_8395080871={}'.format(docker_version),
    ]

    # Interactive TTY
    if not opts.no_tty:
        command.append('-it' if not opts.help else '-i')

    # GPU pass-through
    if not opts.no_gpu:
        command.extend(_gpu_flags())

    # Developer patches (bind-mount source trees over pip-installed packages)
    pkg_site = '/usr/local/lib/python3.11/site-packages'
    if opts.patch:
        for pkg, repo_path in opts.patch.items():
            command.extend([
                '-v', '{}:{}/{}:ro'.format(repo_path, pkg_site, pkg),
            ])

    # Custom environment variables
    if opts.env:
        for var, value in opts.env:
            command.extend(['-e', '{}={}'.format(var, value)])

    # User mapping
    if opts.user:
        command.extend(['-u', opts.user])

    # ---- Bind-mount positional directories ---------------------------------
    main_args: list[str] = []
    if opts.bids_dir:
        command.extend(['-v', '{}:/data/bids:ro'.format(opts.bids_dir)])
        main_args.append('/data/bids')
    if opts.output_dir:
        os.makedirs(opts.output_dir, exist_ok=True)
        command.extend(['-v', '{}:/data/output'.format(opts.output_dir)])
        main_args.append('/data/output')
    main_args.append(opts.analysis_level)

    # ---- Work directory ----------------------------------------------------
    if opts.work_dir:
        if opts.bids_dir and is_in_directory(opts.work_dir, opts.bids_dir):
            print(
                'The selected working directory is a subdirectory of the '
                'input BIDS folder. Please choose a different work directory.'
            )
            return 1
        os.makedirs(opts.work_dir, exist_ok=True)
        command.extend(['-v', '{}:/data/work'.format(opts.work_dir)])
        unknown_args.extend(['-w', '/data/work'])

    # ---- Derivatives -------------------------------------------------------
    if opts.derivatives:
        unknown_args.append('--derivatives')
        for name, host_path in opts.derivatives.items():
            command.extend([
                '-v', '{}:/deriv/{}:ro'.format(host_path, name),
            ])
            unknown_args.append('/deriv/{}'.format(name))

    # ---- BIDS filter file --------------------------------------------------
    if opts.bids_filter_file:
        command.extend([
            '-v', '{}:/tmp/bids_filter.json:ro'.format(opts.bids_filter_file),
        ])
        unknown_args.extend(['--bids-filter-file', '/tmp/bids_filter.json'])

    # ---- Nipype plugin file ------------------------------------------------
    if opts.use_plugin:
        command.extend([
            '-v', '{}:/tmp/plugin.yml:ro'.format(opts.use_plugin),
        ])
        unknown_args.extend(['--use-plugin', '/tmp/plugin.yml'])

    # ---- Custom segmentation model -----------------------------------------
    if opts.seg_model_path:
        command.extend([
            '-v', '{}:/data/seg_model:ro'.format(opts.seg_model_path),
        ])
        unknown_args.extend(['--seg-model-path', '/data/seg_model'])

    # ---- Nipype config file ------------------------------------------------
    if opts.config:
        command.extend([
            '-v', '{}:/tmp/nipype.cfg:ro'.format(opts.config),
        ])

    # ---- TemplateFlow home -------------------------------------------------
    if opts.templateflow_home:
        command.extend([
            '-v', '{}:/opt/templateflow:ro'.format(opts.templateflow_home),
            '-e', 'TEMPLATEFLOW_HOME=/opt/templateflow',
        ])
    elif os.environ.get('TEMPLATEFLOW_HOME'):
        tf_host = os.environ['TEMPLATEFLOW_HOME']
        command.extend([
            '-v', '{}:/opt/templateflow:ro'.format(tf_host),
            '-e', 'TEMPLATEFLOW_HOME=/opt/templateflow',
        ])

    # ---- Output spaces (may include custom templates) ----------------------
    if opts.output_spaces:
        spaces: list[str] = []
        for space in opts.output_spaces:
            basename = os.path.basename(space).split(':')[0]
            if basename not in TF_TEMPLATES and os.path.isdir(space):
                tpl = basename
                if not tpl.startswith('tpl-'):
                    raise RuntimeError(
                        'Custom template {} must have a `tpl-` prefix'.format(
                            space
                        )
                    )
                target = '/opt/templateflow/{}'.format(tpl)
                command.extend([
                    '-v', '{}:{}:ro'.format(os.path.abspath(space), target),
                ])
                spaces.append(tpl[4:])
            else:
                spaces.append(space)
        unknown_args.extend(['--output-spaces'] + spaces)

    # ---- Interactive shell mode --------------------------------------------
    if opts.shell:
        command.append('--entrypoint=bash')

    # ---- Network -----------------------------------------------------------
    if opts.network:
        command.append('--network={}'.format(opts.network))

    # ---- Append image name -------------------------------------------------
    command.append(opts.image)

    # ---- Help / version overrides ------------------------------------------
    if opts.help:
        command.append('--help')
        try:
            target_help = subprocess.check_output(command).decode()
        except subprocess.CalledProcessError:
            target_help = ''
        if target_help:
            print(_merge_help(parser.format_help(), target_help))
        else:
            parser.print_help()
        return 0

    if opts.version:
        print('oncoprep-docker wrapper {}'.format(__version__))
        command.append('--version')
        ret = subprocess.run(command)
        return ret.returncode

    # ---- Run! --------------------------------------------------------------
    if not opts.shell:
        command.extend(main_args)
        command.extend(unknown_args)

    print('RUNNING: ' + ' '.join(command))
    ret = subprocess.run(command)
    if ret.returncode:
        print(
            'OncoPrep: Please report errors to {}'.format(__bugreports__)
        )
    return ret.returncode


# ---------------------------------------------------------------------------
# Help merging (combine wrapper + container help)
# ---------------------------------------------------------------------------

def _merge_help(wrapper_help: str, target_help: str) -> str:
    """Combine wrapper and target (container) help messages.

    Presents the *target* description and argument groups first, then
    appends wrapper-only sections.
    """
    sep = '\n' + '=' * 79 + '\n'
    return (
        target_help.rstrip()
        + sep
        + '\noncoprep-docker additional options\n'
        + '-' * 39
        + '\n\n'
        + wrapper_help.rstrip()
        + '\n'
    )


# ---------------------------------------------------------------------------
# Script entry
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    if '__main__.py' in sys.argv[0]:
        sys.argv[0] = '{} -m oncoprep.docker'.format(sys.executable)
    sys.exit(main())
