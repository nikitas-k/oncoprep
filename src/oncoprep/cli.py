"""Command-line interface for oncoprep workflows."""

from __future__ import annotations

import json
import logging
import os
import sys
import uuid
import warnings
from argparse import ArgumentParser, RawTextHelpFormatter
from multiprocessing import Manager, Process, set_start_method
from pathlib import Path
from time import strftime
from typing import List, Optional


def main():
    """Set an entrypoint for oncoprep."""
    # Handle --version early (before heavy imports that trigger TemplateFlow indexing)
    if '--version' in sys.argv:
        try:
            from oncoprep import __version__
            print(f"oncoprep {__version__}")
        except (ImportError, AttributeError):
            print("oncoprep 0.1.0")
        return

    # Handle --fetch-templates early (doesn't require positional args)
    if '--fetch-templates' in sys.argv:
        return _handle_fetch_templates()

    opts = get_parser().parse_args()
    return build_opts(opts)


def _handle_fetch_templates() -> int:
    """Handle --fetch-templates: download templates and exit."""
    # Parse only the relevant arguments
    parser = ArgumentParser(
        description='Fetch TemplateFlow templates for offline HPC use',
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        '--fetch-templates',
        action='store_true',
        required=True,
    )
    parser.add_argument(
        '--templateflow-home',
        metavar='PATH',
        type=Path,
        default=None,
        help='Template cache directory (default: ~/.cache/templateflow)',
    )
    parser.add_argument(
        '--output-spaces',
        nargs='*',
        default=['MNI152NLin2009cAsym'],
        help='Templates for output spaces (default: MNI152NLin2009cAsym)',
    )
    parser.add_argument(
        '--skull-strip-template',
        default='OASIS30ANTs',
        help='Skull-stripping template (default: OASIS30ANTs)',
    )
    parser.add_argument(
        '--sloppy',
        action='store_true',
        default=False,
        help='Fetch only res-02 templates (faster)',
    )

    args = parser.parse_args()

    # Set TEMPLATEFLOW_HOME if specified
    if args.templateflow_home is not None:
        tf_home = Path(args.templateflow_home).resolve()
        tf_home.mkdir(parents=True, exist_ok=True)
        os.environ['TEMPLATEFLOW_HOME'] = str(tf_home)
        print(f'TemplateFlow home: {tf_home}')
    else:
        tf_home = os.environ.get('TEMPLATEFLOW_HOME', '~/.cache/templateflow')
        print(f'TemplateFlow home: {tf_home}')

    # Collect templates to fetch
    templates = set()
    for space in args.output_spaces:
        template_name = space.split(':')[0].strip()
        if template_name:
            templates.add(template_name)
    if args.skull_strip_template:
        templates.add(args.skull_strip_template)

    print(f'Fetching templates: {sorted(templates)}')

    try:
        from templateflow import api as tf
    except ImportError:
        print('ERROR: templateflow package not installed.', file=sys.stderr)
        print('Install with: pip install templateflow', file=sys.stderr)
        return 1

    resolutions = [2] if args.sloppy else [1, 2]
    success_count = 0
    error_count = 0

    for template in sorted(templates):
        for res in resolutions:
            # T1w
            try:
                t1w = tf.get(template, desc=None, suffix='T1w', resolution=res)
                if t1w:
                    print(f'  ✓ {template} res-{res:02d} T1w')
                    success_count += 1
            except Exception as e:
                print(f'  ✗ {template} res-{res:02d} T1w: {e}')
                error_count += 1

            # Brain mask
            try:
                mask = tf.get(template, desc='brain', suffix='mask', resolution=res)
                if not mask:
                    mask = tf.get(template, label='brain', suffix='mask', resolution=res)
                if mask:
                    print(f'  ✓ {template} res-{res:02d} mask')
                    success_count += 1
            except Exception as e:
                print(f'  ✗ {template} res-{res:02d} mask: {e}')
                error_count += 1

            # T2w (optional)
            try:
                t2w = tf.get(template, desc=None, suffix='T2w', resolution=res)
                if t2w:
                    print(f'  ✓ {template} res-{res:02d} T2w')
                    success_count += 1
            except Exception:
                pass  # T2w is optional

    print(f'\n{success_count} files fetched, {error_count} errors.')
    print('\\nTemplates cached. Use --templateflow-home and --offline on compute nodes.')
    return 0 if error_count == 0 else 1


def get_parser():
    """Build parser object."""
    
    def _drop_ses(value):
        return value.removeprefix('ses-')
    
    parser = ArgumentParser(
        description='OncoPrep: BraTS-style preprocessing for neuro-oncology imaging',
        formatter_class=RawTextHelpFormatter,
    )
    
    # Positional arguments
    parser.add_argument(
        'bids_dir',
        action='store',
        type=Path,
        help='the root folder of a BIDS valid dataset (sub-XXXXX folders should '
        'be found at the top level in this folder).',
    )
    parser.add_argument(
        'output_dir',
        action='store',
        type=Path,
        help='the output path for the outcomes of preprocessing and visual reports',
    )
    parser.add_argument(
        'analysis_level',
        choices=['participant', 'group'],
        help='processing stage to be run, only "participant" in the case of '
        'OncoPrep (see BIDS-Apps specification).',
    )
    
    # Version
    parser.add_argument(
        '--version',
        action='store_true',
        help='show program\'s version number and exit',
    )
    
    # BIDS filtering options
    g_bids = parser.add_argument_group('Options for filtering BIDS queries')
    g_bids.add_argument(
        '--participant-label',
        '--participant_label',
        action='store',
        nargs='+',
        help='a space delimited list of participant identifiers or a single '
        'identifier (the sub- prefix can be removed)',
    )
    g_bids.add_argument(
        '--session-label',
        nargs='+',
        type=_drop_ses,
        help='A space delimited list of session identifiers or a single '
        'identifier (the ses- prefix can be removed)',
    )
    g_bids.add_argument(
        '-d',
        '--derivatives',
        action='store',
        metavar='PATH',
        type=Path,
        nargs='*',
        help='Search PATH(s) for pre-computed derivatives.',
    )
    g_bids.add_argument(
        '--bids-filter-file',
        action='store',
        type=Path,
        metavar='PATH',
        help='a JSON file describing custom BIDS input filters using pybids',
    )
    g_bids.add_argument(
        '--subject-anatomical-reference',
        choices=['first-lex', 'unbiased', 'sessionwise'],
        default='first-lex',
        help='Method to produce the reference anatomical space:\n'
        '\t"first-lex" will use the first image in lexicographical order\n'
        '\t"unbiased" will construct an unbiased template from all images\n'
        '\t"sessionwise" will independently process each session.',
    )
    
    # Performance options
    g_perfm = parser.add_argument_group('Options to handle performance')
    g_perfm.add_argument(
        '--nprocs',
        '--ncpus',
        '--nthreads',
        action='store',
        type=int,
        help='number of CPUs to be used.',
    )
    g_perfm.add_argument(
        '--omp-nthreads',
        action='store',
        type=int,
        default=0,
        help='maximum number of threads per-process',
    )
    g_perfm.add_argument(
        '--mem-gb',
        '--mem_gb',
        action='store',
        default=0,
        type=float,
        help='upper bound memory limit for OncoPrep processes (in GB).',
    )
    g_perfm.add_argument(
        '--low-mem',
        action='store_true',
        help='attempt to reduce memory usage (will increase disk usage in working directory)',
    )
    g_perfm.add_argument(
        '--use-plugin',
        action='store',
        default=None,
        help='nipype plugin configuration file',
    )
    g_perfm.add_argument('--boilerplate', action='store_true', help='generate boilerplate only')
    g_perfm.add_argument(
        '-v',
        '--verbose',
        dest='verbose_count',
        action='count',
        default=0,
        help='increases log verbosity for each occurrence, debug level is -vvv',
    )
    
    # Workflow configuration
    g_conf = parser.add_argument_group('Workflow configuration')
    g_conf.add_argument(
        '--output-spaces',
        nargs='*',
        default=['MNI152NLin2009cAsym'],
        help='output spaces for registration (default: MNI152NLin2009cAsym)',
    )
    g_conf.add_argument(
        '--longitudinal',
        action='store_true',
        help='treat as longitudinal dataset',
    )
    g_conf.add_argument(
        '--deface',
        action='store_true',
        help='remove facial features from anatomical images using mri_deface for privacy protection',
    )
    g_conf.add_argument(
        '--run-segmentation',
        action='store_true',
        help='run tumor segmentation step (requires Docker with GPU support)',
    )
    g_conf.add_argument(
        '--run-radiomics',
        action='store_true',
        help='run radiomics feature extraction on tumor segmentations '
        '(requires pyradiomics; implies --run-segmentation)',
    )
    g_conf.add_argument(
        '--run-qc',
        action='store_true',
        help='run quality control on raw BIDS data using MRIQC before '
        'preprocessing (requires mriqc; outputs to <output_dir>/mriqc/)',
    )
    
    # ANTs options
    g_ants = parser.add_argument_group('Specific options for ANTs registrations')
    g_ants.add_argument(
        '--skull-strip-template',
        default='OASIS30ANTs',
        help='select a template for skull-stripping (default: OASIS30ANTs)',
    )
    g_ants.add_argument(
        '--skull-strip-fixed-seed',
        action='store_true',
        help='do not use a random seed for skull-stripping - will ensure '
        'run-to-run replicability when used with --omp-nthreads 1',
    )
    g_ants.add_argument(
        '--skull-strip-mode',
        action='store',
        choices=('auto', 'skip', 'force'),
        default='auto',
        help='skull stripping mode: force ensures skull stripping, skip ignores it, '
        'auto automatically ignores if pre-stripped brains are detected.',
    )
    g_ants.add_argument(
        '--skull-strip-backend',
        action='store',
        choices=('ants', 'hdbet', 'synthstrip'),
        default='ants',
        help='skull stripping backend: ants (ANTs brain extraction), '
        'hdbet (HD-BET, requires GPU), synthstrip (FreeSurfer SynthStrip). '
        'default: ants',
    )
    g_ants.add_argument(
        '--registration-backend',
        action='store',
        choices=('ants', 'greedy'),
        default='ants',
        help='registration backend for template normalization: '
        'ants (ANTs SyN, default), greedy (PICSL Greedy, faster). '
        'default: ants',
    )
    
    # GPU and acceleration options
    g_accel = parser.add_argument_group('GPU and acceleration options')
    g_accel.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disable GPU acceleration and force CPU-only models',
    )
    g_accel.add_argument(
        '--container-runtime',
        choices=['auto', 'docker', 'singularity', 'apptainer'],
        default='auto',
        help='Container runtime for segmentation models. '
        '"auto" detects the best available runtime (default: auto). '
        'Use "singularity" or "apptainer" on HPC systems without Docker.',
    )
    g_accel.add_argument(
        '--seg-cache-dir',
        metavar='PATH',
        type=Path,
        default=None,
        help='Directory containing pre-downloaded segmentation model images. '
        'For Singularity/Apptainer: directory of .sif files. '
        'For Docker: directory of .tar files (from "docker save"). '
        'Use "oncoprep-models pull -o DIR" to download models. '
        'Defaults to $ONCOPREP_SEG_CACHE or ~/.cache/oncoprep/seg.',
    )
    
    # Segmentation options
    g_seg = parser.add_argument_group('Segmentation options')
    g_seg.add_argument(
        '--default-seg',
        action='store_true',
        help='use default segmentation model',
    )
    g_seg.add_argument(
        '--seg-model-path',
        metavar='PATH',
        type=Path,
        help='path to custom segmentation model',
    )
    
    # Other options
    g_other = parser.add_argument_group('Other options')
    g_other.add_argument(
        '-w',
        '--work-dir',
        action='store',
        type=Path,
        default=Path('work'),
        help='path where intermediate results should be stored',
    )
    g_other.add_argument(
        '--fast-track',
        action='store_true',
        default=False,
        help='fast-track the workflow by searching for existing derivatives.',
    )
    g_other.add_argument(
        '--resource-monitor',
        action='store_true',
        default=False,
        help="enable Nipype's resource monitoring to keep track of memory and CPU usage",
    )
    g_other.add_argument(
        '--reports-only',
        action='store_true',
        default=False,
        help="only generate reports, don't run workflows.",
    )
    g_other.add_argument(
        '--run-uuid',
        action='store',
        default=None,
        help='Specify UUID of previous run, to include error logs in report. '
        'No effect without --reports-only.',
    )
    g_other.add_argument(
        '--write-graph',
        action='store_true',
        default=False,
        help='Write workflow graph.',
    )
    g_other.add_argument(
        '--stop-on-first-crash',
        action='store_true',
        default=False,
        help='Force stopping on first crash, even if a work directory was specified.',
    )
    g_other.add_argument(
        '--notrack',
        action='store_true',
        default=False,
        help='Opt-out of sending tracking information of this run.',
    )
    g_other.add_argument(
        '--sloppy',
        action='store_true',
        default=False,
        help='Use low-quality tools for speed - TESTING ONLY',
    )
    
    # TemplateFlow options
    g_tf = parser.add_argument_group('TemplateFlow options')
    g_tf.add_argument(
        '--templateflow-home',
        metavar='PATH',
        type=Path,
        default=None,
        help='Path to TemplateFlow template cache directory. '
        'Overrides the TEMPLATEFLOW_HOME environment variable. '
        'Required for offline/HPC use - pre-fetch templates on a login node '
        'with internet access, then point to the cache on compute nodes.',
    )
    g_tf.add_argument(
        '--offline',
        action='store_true',
        default=False,
        help='Disable network access for TemplateFlow. '
        'Templates must already be cached in TEMPLATEFLOW_HOME. '
        'Use this on HPC compute nodes without internet access.',
    )
    g_tf.add_argument(
        '--fetch-templates',
        action='store_true',
        default=False,
        help='Download TemplateFlow templates and exit. '
        'Use on a login node with internet access to pre-cache templates '
        'for offline HPC use. Fetches templates for --output-spaces and '
        '--skull-strip-template.',
    )
    
    return parser


def build_opts(opts):
    """Trigger a new process that builds the workflow graph, based on the input options."""
    import gc

    from nipype import config as ncfg
    from nipype import logging as nlogging
    
    set_start_method('forkserver', force=True)
    
    logging.addLevelName(25, 'IMPORTANT')
    logging.addLevelName(15, 'VERBOSE')
    logger = logging.getLogger('cli')
    
    def _warn_redirect(message, category, filename, lineno, file=None, line=None):
        logger.warning('Captured warning (%s): %s', category, message)
    
    warnings.showwarning = _warn_redirect
    
    # Retrieve logging level
    log_level = int(max(25 - 5 * opts.verbose_count, logging.DEBUG))
    logger.setLevel(log_level)
    nlogging.getLogger('nipype.workflow').setLevel(log_level)
    nlogging.getLogger('nipype.interface').setLevel(log_level)
    nlogging.getLogger('nipype.utils').setLevel(log_level)
    
    errno = 0
    
    with Manager() as mgr:
        retval = mgr.dict()
        p = Process(target=build_workflow, args=(opts, retval))
        p.start()
        p.join()
        
        if p.exitcode != 0:
            sys.exit(p.exitcode)
        
        oncoprep_wf = retval['workflow']
        plugin_settings = retval['plugin_settings']
        bids_dir = retval['bids_dir']
        output_dir = retval['output_dir']
        subject_session_list = retval['subject_session_list']
        run_uuid = retval['run_uuid']
        retcode = retval['return_code']
    
    if oncoprep_wf is None:
        sys.exit(1)
    
    if opts.write_graph:
        oncoprep_wf.write_graph(graph2use='colored', format='svg', simple_form=True)
    
    if opts.reports_only:
        sys.exit(int(retcode > 0))
    
    if opts.boilerplate:
        sys.exit(int(retcode > 0))
    
    # Clean up master process before running workflow
    gc.collect()
    try:
        oncoprep_wf.run(**plugin_settings)
    except RuntimeError:
        errno = 1
    else:
        logger.log(25, 'OncoPrep finished without errors')
    finally:
        logger.log(
            25, 'OncoPrep processing complete. Results in %s', output_dir
        )
    
    sys.exit(int(errno > 0))


def build_workflow(opts, retval):
    """
    Create the Nipype Workflow that supports the whole execution graph, given the inputs.

    All the checks and the construction of the workflow are done
    inside this function that has pickleable inputs and output
    dictionary (``retval``) to allow isolation using a
    ``multiprocessing.Process`` that allows oncoprep to enforce
    a hard-limited memory-scope.
    """
    from os import cpu_count

    from bids.layout import BIDSLayout, Query
    from nipype import config as ncfg
    from nipype import logging as nlogging
    from niworkflows.utils.bids import collect_participants

    from oncoprep.workflows.base import init_oncoprep_wf
    
    logger = logging.getLogger('nipype.workflow')
    
    INIT_MSG = """
    Running OncoPrep version 0.1.0:
      * BIDS dataset path: {bids_dir}.
      * Participants & Sessions: {subject_session_list}.
      * Run identifier: {uuid}.
      * Output spaces: {spaces}.
    """
    
    # Set up instrumental utilities
    run_uuid = '{}_{}'.format(strftime('%Y%m%d-%H%M%S'), uuid.uuid4())
    
    # Validate BIDS directory
    bids_dir = opts.bids_dir.resolve()
    layout = BIDSLayout(str(bids_dir), validate=False)
    subject_list = collect_participants(layout, participant_label=opts.participant_label)
    session_list = opts.session_label or []
    
    subject_session_list = []
    for subject in subject_list:
        sessions = (
            layout.get_sessions(
                scope='raw',
                subject=subject,
                session=session_list or Query.OPTIONAL,
                suffix=['T1w', 'T2w', 'FLAIR'],
            )
            or None
        )
        
        if opts.subject_anatomical_reference == 'sessionwise':
            if not sessions:
                raise RuntimeError(
                    '--subject-anatomical-reference "sessionwise" was requested, but no sessions '
                    f'found for subject {subject}.'
                )
            for session in sessions:
                subject_session_list.append((subject, session))
        else:
            subject_session_list.append((subject, sessions))
    
    bids_filters = json.loads(opts.bids_filter_file.read_text()) if opts.bids_filter_file else None
    
    # Load plugin settings
    if opts.use_plugin is not None:
        from yaml import safe_load as loadyml
        
        with open(opts.use_plugin) as f:
            plugin_settings = loadyml(f)
        plugin_settings.setdefault('plugin_args', {})
    else:
        plugin_settings = {
            'plugin': 'MultiProc',
            'plugin_args': {
                'raise_insufficient': False,
                'maxtasksperchild': 1,
            },
        }
    
    # Resource management
    nprocs = plugin_settings['plugin_args'].get('n_procs')
    if nprocs is None or opts.nprocs is not None:
        nprocs = opts.nprocs
        if nprocs is None or nprocs < 1:
            nprocs = cpu_count()
        plugin_settings['plugin_args']['n_procs'] = nprocs
    
    if opts.mem_gb:
        plugin_settings['plugin_args']['memory_gb'] = opts.mem_gb
    
    omp_nthreads = opts.omp_nthreads
    if omp_nthreads == 0:
        omp_nthreads = min(nprocs - 1 if nprocs > 1 else cpu_count(), 8)
    
    if 1 < nprocs < omp_nthreads:
        logger.warning(
            'Per-process threads (--omp-nthreads=%d) exceed total '
            'available CPUs (--nprocs=%d)',
            omp_nthreads,
            nprocs,
        )
    
    # Set up directories
    output_dir = opts.output_dir.resolve()
    log_dir = output_dir / 'oncoprep' / 'logs'
    work_dir = opts.work_dir.resolve()
    
    log_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # Nipype configuration
    ncfg.update_config(
        {
            'logging': {'log_directory': str(log_dir), 'log_to_file': True},
            'execution': {
                'crashdump_dir': str(log_dir),
                'crashfile_format': 'txt',
                'get_linked_libs': False,
                'stop_on_first_crash': opts.stop_on_first_crash,
            },
            'monitoring': {
                'enabled': opts.resource_monitor,
                'sample_frequency': '0.5',
                'summary_append': True,
            },
        }
    )
    
    if opts.resource_monitor:
        ncfg.enable_resource_monitor()
    
    retval['return_code'] = 0
    retval['plugin_settings'] = plugin_settings
    retval['bids_dir'] = str(bids_dir)
    retval['output_dir'] = str(output_dir)
    retval['work_dir'] = str(work_dir)
    retval['subject_session_list'] = subject_session_list
    retval['run_uuid'] = run_uuid
    retval['workflow'] = None
    
    # Handle reports-only mode
    if opts.reports_only:
        logger.log(
            25, 'Running --reports-only on participants %s', 
            _pprint_subses(subject_session_list)
        )
        if opts.run_uuid is not None:
            run_uuid = opts.run_uuid
        return retval
    
    logger.log(
        25,
        INIT_MSG.format(
            bids_dir=bids_dir,
            subject_session_list=_pprint_subses(subject_session_list),
            uuid=run_uuid,
            spaces=', '.join(opts.output_spaces),
        ),
    )
    
    # Ensure TemplateFlow templates are available
    _ensure_templateflow_templates(
        output_spaces=opts.output_spaces,
        skull_strip_template=opts.skull_strip_template,
        templateflow_home=opts.templateflow_home,
        offline=opts.offline,
        sloppy=opts.sloppy,
        logger=logger,
    )

    # Build top-level workflow for multi-subject processing
    retval['workflow'] = init_oncoprep_wf(
        output_dir=output_dir,
        subject_session_list=subject_session_list,
        run_uuid=run_uuid,
        work_dir=work_dir,
        bids_dir=bids_dir,
        omp_nthreads=omp_nthreads,
        nprocs=nprocs,
        mem_gb=opts.mem_gb or (4 if opts.low_mem else None),
        skull_strip_template=opts.skull_strip_template,
        skull_strip_fixed_seed=opts.skull_strip_fixed_seed,
        skull_strip_mode=opts.skull_strip_mode,
        skull_strip_backend=opts.skull_strip_backend,
        registration_backend=opts.registration_backend,
        longitudinal=opts.longitudinal,
        output_spaces=opts.output_spaces,
        use_gpu=not opts.no_gpu,
        deface=opts.deface,
        run_segmentation=opts.run_segmentation or opts.run_radiomics,
        run_radiomics=opts.run_radiomics,
        run_qc=opts.run_qc,
        seg_model_path=opts.seg_model_path,
        default_seg=opts.default_seg,
        sloppy=opts.sloppy,
        container_runtime=opts.container_runtime,
        seg_cache_dir=opts.seg_cache_dir,
    )
    
    retval['return_code'] = 0
    return retval


def _pprint_subses(subses: list) -> str:
    """
    Pretty print a list of subjects and sessions.
    
    Example
    -------
    >>> _pprint_subses([('01', 'A'), ('02', ['A', 'B']), ('03', None), ('04', ['A'])])
    'sub-01 ses-A, sub-02 (2 sessions), sub-03, sub-04 ses-A'
    """
    output = []
    for subject, session in subses:
        if isinstance(session, list):
            if len(session) > 1:
                output.append(f'sub-{subject} ({len(session)} sessions)')
                continue
            session = session[0]
        if session is None:
            output.append(f'sub-{subject}')
        else:
            output.append(f'sub-{subject} ses-{session}')
    
    return ', '.join(output)


def _ensure_templateflow_templates(
    output_spaces: List[str],
    skull_strip_template: str,
    templateflow_home: Optional[Path] = None,
    offline: bool = False,
    sloppy: bool = False,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Ensure required TemplateFlow templates are available.

    This function pre-fetches templates before workflow construction to avoid
    network access during parallel execution (which can fail on HPC compute
    nodes without internet access).

    Parameters
    ----------
    output_spaces : list
        List of output space names (e.g., ['MNI152NLin2009cAsym'])
    skull_strip_template : str
        Name of skull-stripping template (e.g., 'OASIS30ANTs')
    templateflow_home : Path, optional
        Override TEMPLATEFLOW_HOME environment variable
    offline : bool
        If True, skip fetching and assume templates are already cached
    sloppy : bool
        If True, use lower resolution templates (res-02)
    logger : logging.Logger, optional
        Logger instance for status messages
    """
    if logger is None:
        logger = logging.getLogger('nipype.workflow')

    # Set TEMPLATEFLOW_HOME if specified
    if templateflow_home is not None:
        tf_home = Path(templateflow_home).resolve()
        tf_home.mkdir(parents=True, exist_ok=True)
        os.environ['TEMPLATEFLOW_HOME'] = str(tf_home)
        logger.info(f'TemplateFlow home set to: {tf_home}')

    if offline:
        # In offline mode, just verify TEMPLATEFLOW_HOME is set
        tf_home = os.environ.get('TEMPLATEFLOW_HOME')
        if not tf_home:
            logger.warning(
                '--offline specified but TEMPLATEFLOW_HOME is not set. '
                'Templates may not be found.'
            )
        else:
            logger.info(f'Offline mode: using cached templates from {tf_home}')
        return

    # Collect unique templates to fetch
    templates_to_fetch = set()

    # Add output spaces
    for space in output_spaces:
        # Parse space string (e.g., 'MNI152NLin2009cAsym:res-1')
        template_name = space.split(':')[0].strip()
        if template_name:
            templates_to_fetch.add(template_name)

    # Add skull-strip template
    if skull_strip_template:
        templates_to_fetch.add(skull_strip_template)

    if not templates_to_fetch:
        return

    logger.info(f'Ensuring TemplateFlow templates are cached: {sorted(templates_to_fetch)}')

    # Import templateflow (sets up TEMPLATEFLOW_HOME)
    try:
        from templateflow import api as tf
    except ImportError:
        logger.warning(
            'templateflow package not installed. '
            'Templates will be fetched on-demand during workflow execution.'
        )
        return

    resolutions = [2] if sloppy else [1, 2]

    for template in templates_to_fetch:
        for res in resolutions:
            try:
                # Fetch T1w template
                t1w = tf.get(template, desc=None, suffix='T1w', resolution=res)
                if t1w:
                    logger.debug(f'Cached: {template} res-{res:02d} T1w')
            except Exception as e:
                logger.debug(f'Could not fetch {template} res-{res:02d} T1w: {e}')

            try:
                # Fetch brain mask
                mask = tf.get(template, desc='brain', suffix='mask', resolution=res)
                if not mask:
                    mask = tf.get(template, label='brain', suffix='mask', resolution=res)
                if mask:
                    logger.debug(f'Cached: {template} res-{res:02d} mask')
            except Exception as e:
                logger.debug(f'Could not fetch {template} res-{res:02d} mask: {e}')

            try:
                # Fetch T2w if available (optional)
                t2w = tf.get(template, desc=None, suffix='T2w', resolution=res)
                if t2w:
                    logger.debug(f'Cached: {template} res-{res:02d} T2w')
            except Exception:
                pass  # T2w is optional

    logger.info('TemplateFlow templates ready')


if __name__ == '__main__':
    raise RuntimeError(
        'oncoprep/cli.py should not be run directly;\n'
        'Please `pip install` oncoprep and use the `oncoprep` command'
    )