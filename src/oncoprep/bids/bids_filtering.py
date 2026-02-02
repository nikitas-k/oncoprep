"""Advanced BIDS filtering for selective conversion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from oncoprep.utils.logging import get_logger

LOGGER = get_logger(__name__)


def load_bids_filter_file(filter_file: Path) -> dict:
    """
    Load BIDS filter file (JSON format).
    
    Parameters
    ----------
    filter_file : Path
        Path to BIDS filter JSON file
        
    Returns
    -------
    dict
        Filter specification
        
    Example
    -------
    Filter file content (filter.json):
    {
        "datatype": "anat",
        "suffix": ["T1w", "T2w"],
        "modality": ["T1", "T2"],
        "exclude": {
            "acquisition": ["lowres"]
        }
    }
    """
    try:
        with open(filter_file) as f:
            filters = json.load(f)
        LOGGER.info(f"Loaded BIDS filter file: {filter_file}")
        return filters
    except (json.JSONDecodeError, FileNotFoundError) as e:
        LOGGER.warning(f"Could not load filter file: {e}")
        return {}


def matches_filter(series_name: str, filters: dict) -> bool:
    """
    Check if series name matches filter criteria.
    
    Parameters
    ----------
    series_name : str
        DICOM series name/description
    filters : dict
        Filter specification
        
    Returns
    -------
    bool
        True if series matches filter
    """
    if not filters:
        return True
    
    upper_name = series_name.upper()
    
    # Check include criteria (suffix)
    suffix_filter = filters.get('suffix', [])
    if suffix_filter:
        if isinstance(suffix_filter, list):
            if not any(s.upper() in upper_name for s in suffix_filter):
                return False
        elif isinstance(suffix_filter, str):
            if suffix_filter.upper() not in upper_name:
                return False
    
    # Check exclude criteria
    exclude = filters.get('exclude', {})
    for key, values in exclude.items():
        if isinstance(values, list):
            if any(v.upper() in upper_name for v in values):
                return False
        elif isinstance(values, str):
            if values.upper() in upper_name:
                return False
    
    return True


def create_example_filter_file(output_path: Path) -> None:
    """
    Create an example BIDS filter file.
    
    Parameters
    ----------
    output_path : Path
        Path to write example filter file
    """
    example_filter = {
        "description": "Example BIDS filter file",
        "include": {
            "suffix": ["T1w", "T2w", "FLAIR", "T1ce"],
            "notes": "Include T1/T2/FLAIR/T1ce anatomical images"
        },
        "exclude": {
            "acquisition": ["lowres", "scout"],
            "description": "Exclude low-resolution and scout images",
            "notes": "These are often localizer/reference images"
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(example_filter, f, indent=2)
    
    LOGGER.info(f"Created example BIDS filter file: {output_path}")
