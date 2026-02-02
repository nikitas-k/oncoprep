"""BIDS validation integration for conversion output."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Optional

from oncoprep.utils.logging import get_logger

LOGGER = get_logger(__name__)


def validate_bids_dataset(bids_dir: Path, ignore_rules: Optional[list[str]] = None) -> dict:
    """
    Validate BIDS dataset using bids-validator.
    
    Parameters
    ----------
    bids_dir : Path
        Path to BIDS dataset
    ignore_rules : Optional[list[str]]
        List of validation rules to ignore
        
    Returns
    -------
    dict
        Validation results with issues, warnings, and summary
    """
    try:
        import subprocess
        
        # Build command
        cmd = ['bids-validator', str(bids_dir), '--json']
        
        # Add ignored rules if provided
        if ignore_rules:
            for rule in ignore_rules:
                cmd.extend(['--ignoreNiftiHeaders', rule])
        
        # Run validator
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        
        if result.returncode == 0 or result.stdout:
            try:
                output = json.loads(result.stdout)
                return {
                    'valid': result.returncode == 0,
                    'issues': output.get('issues', {}),
                    'output': output,
                }
            except json.JSONDecodeError:
                LOGGER.warning("Could not parse bids-validator output")
                return {
                    'valid': False,
                    'issues': {'error': [result.stdout]},
                    'output': {},
                }
        else:
            LOGGER.error(f"bids-validator failed: {result.stderr}")
            return {
                'valid': False,
                'issues': {'error': [result.stderr]},
                'output': {},
            }
            
    except FileNotFoundError:
        LOGGER.error("bids-validator not found. Install with: pip install bids-validator")
        return {
            'valid': False,
            'issues': {'error': ['bids-validator not installed']},
            'output': {},
        }
    except Exception as e:
        LOGGER.error(f"Validation error: {e}")
        return {
            'valid': False,
            'issues': {'error': [str(e)]},
            'output': {},
        }


def print_validation_report(bids_dir: Path, validation_result: dict) -> None:
    """
    Print validation report.
    
    Parameters
    ----------
    bids_dir : Path
        Path to BIDS dataset
    validation_result : dict
        Validation result from validate_bids_dataset()
    """
    print(f"\n{'=' * 70}")
    print(f"BIDS Validation Report")
    print(f"{'=' * 70}")
    print(f"Dataset: {bids_dir}")
    
    if validation_result['valid']:
        print(f"Status: ✓ VALID")
    else:
        print(f"Status: ✗ INVALID")
    
    issues = validation_result.get('issues', {})
    
    if 'errors' in issues:
        print(f"\nErrors ({len(issues['errors'])}):")
        for error in issues['errors'][:5]:
            print(f"  - {error}")
        if len(issues['errors']) > 5:
            print(f"  ... and {len(issues['errors']) - 5} more errors")
    
    if 'warnings' in issues:
        print(f"\nWarnings ({len(issues['warnings'])}):")
        for warning in issues['warnings'][:5]:
            print(f"  - {warning}")
        if len(issues['warnings']) > 5:
            print(f"  ... and {len(issues['warnings']) - 5} more warnings")
    
    print(f"\n{'=' * 70}\n")


def auto_validate_after_conversion(bids_dir: Path) -> bool:
    """
    Automatically validate BIDS dataset after conversion.
    
    Parameters
    ----------
    bids_dir : Path
        Path to BIDS dataset
        
    Returns
    -------
    bool
        True if validation passed
    """
    LOGGER.info(f"Validating BIDS dataset at {bids_dir}")
    
    result = validate_bids_dataset(bids_dir)
    print_validation_report(bids_dir, result)
    
    return result['valid']
