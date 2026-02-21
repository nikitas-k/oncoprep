"""Batch DICOM to BIDS conversion script."""

from __future__ import annotations

import logging
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


def convert_subject_wrapper(args: Tuple[Path, Path, str, str | None, str]) -> Tuple[str, bool]:
    """
    Wrapper for multiprocessing pool.

    Parameters
    ----------
    args : Tuple
        (source_dir, bids_dir, subject, session, converter)

    Returns
    -------
    Tuple[str, bool]
        Subject identifier and success status
    """
    source_dir, bids_dir, subject, session, converter = args

    try:
        success = convert_subject_dicoms_to_bids(
            source_dir=source_dir,
            bids_dir=bids_dir,
            subject=subject,
            session=session,
            conversion_tool=converter,
        )
        return (subject, success)
    except Exception as e:
        LOGGER.error(f"Failed to convert sub-{subject}: {e}")
        return (subject, False)


def batch_convert(
    dicom_root: Path,
    bids_dir: Path,
    session: str | None = None,
    converter: str = 'dcm2niix',
    n_procs: int = 1,
    pattern: str | None = None,
) -> dict:
    """
    Batch convert DICOM subjects to BIDS format.

    Parameters
    ----------
    dicom_root : Path
        Root directory containing subject subdirectories
    bids_dir : Path
        Output BIDS dataset directory
    session : Optional[str]
        Session identifier (applied to all subjects)
    converter : str
        Conversion tool ('dcm2niix' or 'nibabel')
    n_procs : int
        Number of parallel processes
    pattern : Optional[str]
        Pattern to match subject directories (e.g., 'sub-*' or 'GBM*')

    Returns
    -------
    dict
        Results dictionary with subject conversion status
    """
    # Find subject directories
    if pattern:
        subject_dirs = list(dicom_root.glob(pattern))
    else:
        subject_dirs = [d for d in dicom_root.iterdir() if d.is_dir() and not d.name.startswith('.')]

    if not subject_dirs:
        LOGGER.error(f"No subject directories found in {dicom_root}")
        return {}

    LOGGER.info(f"Found {len(subject_dirs)} subject directories")

    # Create BIDS dataset description
    create_bids_dataset_description(bids_dir)

    # Prepare conversion tasks
    tasks = []
    for subject_dir in sorted(subject_dirs):
        subject_id = subject_dir.name.replace('sub-', '').split('_')[0]
        tasks.append((subject_dir, bids_dir, subject_id, session, converter))

    results = {'total': len(tasks), 'successful': 0, 'failed': 0, 'subjects': {}}

    # Run conversions
    if n_procs == 1:
        LOGGER.info("Running conversions serially")
        for task in tasks:
            subject_id, success = convert_subject_wrapper(task)
            results['subjects'][subject_id] = success
            if success:
                results['successful'] += 1
            else:
                results['failed'] += 1
    else:
        LOGGER.info(f"Running conversions in parallel with {n_procs} processes")
        with Pool(processes=n_procs) as pool:
            for subject_id, success in pool.imap_unordered(convert_subject_wrapper, tasks):
                results['subjects'][subject_id] = success
                if success:
                    results['successful'] += 1
                else:
                    results['failed'] += 1

    # Summary
    LOGGER.info(f"\n{'=' * 70}")
    LOGGER.info("Batch Conversion Summary")
    LOGGER.info(f"{'=' * 70}")
    LOGGER.info(f"Total subjects: {results['total']}")
    LOGGER.info(f"Successful: {results['successful']}")
    LOGGER.info(f"Failed: {results['failed']}")
    LOGGER.info(f"Success rate: {100 * results['successful'] / results['total']:.1f}%")
    LOGGER.info(f"Output directory: {bids_dir}")
    LOGGER.info(f"{'=' * 70}\n")

    return results


def batch_conversion_main():
    """Entry point for batch conversion CLI."""
    parser = ArgumentParser(
        description='Batch convert DICOM subjects to BIDS format',
        formatter_class=RawTextHelpFormatter,
    )

    parser.add_argument(
        'dicom_root',
        type=Path,
        help='root directory containing subject subdirectories',
    )
    parser.add_argument(
        'output_dir',
        type=Path,
        help='output BIDS dataset directory',
    )

    parser.add_argument(
        '--session',
        default=None,
        help='session identifier to apply to all subjects (optional)',
    )
    parser.add_argument(
        '--converter',
        choices=['dcm2niix', 'nibabel'],
        default='dcm2niix',
        help='DICOM to NIfTI conversion tool (default: dcm2niix)',
    )
    parser.add_argument(
        '--nprocs',
        type=int,
        default=1,
        help='number of parallel processes (default: 1)',
    )
    parser.add_argument(
        '--pattern',
        default=None,
        help='glob pattern to match subject directories (e.g., "sub-*" or "GBM*")',
    )
    parser.add_argument(
        '-v',
        '--verbose',
        dest='verbose_count',
        action='count',
        default=0,
        help='increase verbosity',
    )
    parser.add_argument(
        '--version',
        action='store_true',
        help='show version and exit',
    )

    opts = parser.parse_args()

    if opts.version:
        try:
            import oncoprep
            print(f"oncoprep {oncoprep.__version__}")
        except (ImportError, AttributeError):
            print("oncoprep 0.1.0")
        return 0

    # Set logging level
    log_level = int(max(25 - 5 * opts.verbose_count, logging.DEBUG))
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')

    # Run batch conversion
    try:
        results = batch_convert(
            dicom_root=opts.dicom_root,
            bids_dir=opts.output_dir,
            session=opts.session,
            converter=opts.converter,
            n_procs=opts.nprocs,
            pattern=opts.pattern,
        )

        if results['failed'] == 0:
            return 0
        else:
            return 1

    except Exception as e:
        LOGGER.error(f"Batch conversion failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(batch_conversion_main())
