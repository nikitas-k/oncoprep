"""Unified CLI for DICOM to BIDS conversion (single subject or batch)."""

from __future__ import annotations

import sys
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from multiprocessing import Pool
from typing import Tuple

from oncoprep.utils.logging import get_logger
from oncoprep.workflows.dicom_conversion import (
    create_bids_dataset_description,
    convert_subject_dicoms_to_bids,
)

LOGGER = get_logger(__name__)


def main():
    """Entry point for oncoprep-dicom command."""
    # Check for version flag before requiring positional arguments
    if '--version' in sys.argv:
        try:
            import oncoprep
            print(f"oncoprep {oncoprep.__version__}")
        except (ImportError, AttributeError):
            print("oncoprep 0.1.0")
        return 0
    
    opts = get_dicom_parser().parse_args()
    
    # Determine if single subject or batch mode
    if opts.subject:
        # Single subject mode
        return run_single_subject(opts)
    else:
        # Batch mode
        return run_batch(opts)


def get_dicom_parser():
    """Build unified parser for DICOM conversion CLI."""
    parser = ArgumentParser(
        description='OncoPrep DICOM to BIDS Conversion\n'
                    'Single subject: oncoprep-dicom DICOM_DIR BIDS_DIR --subject SUBID\n'
                    'Batch mode:     oncoprep-dicom DICOM_ROOT BIDS_DIR [--pattern "PATTERN"] [--n-procs N]',
        formatter_class=RawTextHelpFormatter,
    )
    
    # Positional arguments
    parser.add_argument(
        'dicom_dir',
        action='store',
        type=Path,
        help='directory containing DICOM files (single subject) or subject subdirectories (batch)',
    )
    parser.add_argument(
        'output_dir',
        action='store',
        type=Path,
        help='output BIDS dataset directory',
    )
    
    # Subject/session options
    g_bids = parser.add_argument_group('BIDS Options')
    g_bids.add_argument(
        '--subject',
        default=None,
        help='subject identifier for single subject conversion (without "sub-" prefix)\n'
             'If omitted, performs batch conversion on all subdirectories',
    )
    g_bids.add_argument(
        '--session',
        default=None,
        help='session identifier (without "ses-" prefix, optional)',
    )
    
    # Batch options
    g_batch = parser.add_argument_group('Batch Mode Options (when --subject is omitted)')
    g_batch.add_argument(
        '--pattern',
        default='*',
        help='pattern to match subject directories (e.g., "GBM*", "sub-*")\n'
             'default: * (matches all subdirectories)',
    )
    g_batch.add_argument(
        '--n-procs',
        type=int,
        default=1,
        help='number of parallel processes for batch conversion\n'
             'default: 1 (no parallelization)',
    )
    
    # Conversion options
    g_conv = parser.add_argument_group('Conversion Options')
    g_conv.add_argument(
        '--converter',
        choices=['dcm2niix', 'nibabel'],
        default='dcm2niix',
        help='DICOM to NIfTI conversion tool (default: dcm2niix)',
    )
    g_conv.add_argument(
        '--no-sidecar',
        action='store_true',
        help='do not create BIDS JSON sidecars',
    )
    g_conv.add_argument(
        '--overwrite',
        action='store_true',
        help='overwrite existing output files',
    )
    g_conv.add_argument(
        '--filter-file',
        type=Path,
        default=None,
        help='path to BIDS filter JSON file for selective conversion',
    )
    
    # Validation options
    g_valid = parser.add_argument_group('Validation Options')
    g_valid.add_argument(
        '--validate',
        action='store_true',
        help='run BIDS validation after conversion',
    )
    
    # Logging options
    g_log = parser.add_argument_group('Logging Options')
    g_log.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='logging verbosity (default: INFO)',
    )
    g_log.add_argument(
        '--version',
        action='store_true',
        help='show version number and exit',
    )
    
    return parser


def run_single_subject(opts) -> int:
    """Run single subject conversion."""
    LOGGER.info(f"Starting single subject conversion: sub-{opts.subject}")
    
    # Create BIDS dataset description
    if not opts.output_dir.exists():
        opts.output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        create_bids_dataset_description(opts.output_dir)
    except Exception as e:
        LOGGER.warning(f"Could not create dataset_description.json: {e}")
    
    # Run conversion
    try:
        result = convert_subject_dicoms_to_bids(
            source_dir=opts.dicom_dir,
            bids_dir=opts.output_dir,
            subject=opts.subject,
            session=opts.session,
            conversion_tool=opts.converter,
            overwrite=opts.overwrite,
        )
        
        if result:
            LOGGER.info(f"✓ Successfully converted sub-{opts.subject}")
            
            # Run validation if requested
            if opts.validate:
                _validate_dataset(opts.output_dir)
            
            return 0
        else:
            LOGGER.error(f"✗ Failed to convert sub-{opts.subject}")
            return 1
            
    except Exception as e:
        LOGGER.error(f"✗ Conversion failed: {e}")
        return 1


def run_batch(opts) -> int:
    """Run batch conversion."""
    
    LOGGER.info(f"Starting batch conversion from {opts.dicom_dir}")
    LOGGER.info(f"Pattern: {opts.pattern}, Processes: {opts.n_procs}")
    
    # Create BIDS dataset description
    if not opts.output_dir.exists():
        opts.output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        create_bids_dataset_description(opts.output_dir)
    except Exception as e:
        LOGGER.warning(f"Could not create dataset_description.json: {e}")
    
    # Find all matching subdirectories
    subject_dirs = sorted([d for d in opts.dicom_dir.glob(opts.pattern) if d.is_dir()])
    
    if not subject_dirs:
        LOGGER.warning(f"No directories matching pattern '{opts.pattern}' found in {opts.dicom_dir}")
        return 0
    
    LOGGER.info(f"Found {len(subject_dirs)} directories to process")
    
    # Prepare conversion tasks
    tasks = []
    for subject_dir in subject_dirs:
        subject_id = subject_dir.name
        tasks.append((subject_dir, opts.output_dir, subject_id, opts.session, opts.converter, opts.overwrite))
    
    # Run conversions
    successful = 0
    failed = 0
    
    if opts.n_procs > 1:
        # Parallel execution
        with Pool(processes=opts.n_procs) as pool:
            results = pool.map(_convert_subject_wrapper, tasks)
        
        for subject_id, success in results:
            if success:
                successful += 1
                LOGGER.info(f"✓ sub-{subject_id}")
            else:
                failed += 1
                LOGGER.warning(f"✗ sub-{subject_id}")
    else:
        # Sequential execution
        for subject_dir, bids_dir, subject_id, session, converter, overwrite in tasks:
            try:
                result = convert_subject_dicoms_to_bids(
                    source_dir=subject_dir,
                    bids_dir=bids_dir,
                    subject=subject_id,
                    session=session,
                    conversion_tool=converter,
                    overwrite=overwrite,
                )
                if result:
                    successful += 1
                    LOGGER.info(f"✓ sub-{subject_id}")
                else:
                    failed += 1
                    LOGGER.warning(f"✗ sub-{subject_id}")
            except Exception as e:
                failed += 1
                LOGGER.error(f"✗ sub-{subject_id}: {e}")
    
    # Print summary
    total = successful + failed
    LOGGER.info(f"\n{'='*50}")
    LOGGER.info("Batch Conversion Summary")
    LOGGER.info(f"{'='*50}")
    LOGGER.info(f"Total subjects:     {total}")
    LOGGER.info(f"Successful:         {successful}")
    LOGGER.info(f"Failed:             {failed}")
    LOGGER.info(f"Success rate:       {successful/total*100:.1f}%")
    LOGGER.info(f"{'='*50}")
    
    # Run validation if requested
    if opts.validate:
        _validate_dataset(opts.output_dir)
    
    return 0 if failed == 0 else 1


def _convert_subject_wrapper(args: Tuple) -> Tuple[str, bool]:
    """Wrapper for multiprocessing pool."""
    subject_dir, bids_dir, subject_id, session, converter, overwrite = args
    
    try:
        result = convert_subject_dicoms_to_bids(
            source_dir=subject_dir,
            bids_dir=bids_dir,
            subject=subject_id,
            session=session,
            conversion_tool=converter,
            overwrite=overwrite,
        )
        return (subject_id, result)
    except Exception as e:
        LOGGER.error(f"Failed to convert sub-{subject_id}: {e}")
        return (subject_id, False)


def _validate_dataset(bids_dir: Path) -> None:
    """Validate BIDS dataset."""
    try:
        from oncoprep.bids_validation import validate_bids_dataset, print_validation_report
        
        LOGGER.info(f"\nValidating BIDS dataset at {bids_dir}...")
        results = validate_bids_dataset(str(bids_dir))
        print_validation_report(results)
    except ImportError:
        LOGGER.warning("bids-validator not installed, skipping validation")
    except Exception as e:
        LOGGER.error(f"Validation failed: {e}")
