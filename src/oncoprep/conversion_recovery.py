"""Error recovery and logging for conversion."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from oncoprep.utils.logging import get_logger

LOGGER = get_logger(__name__)


class ConversionLog:
    """Log conversion operations for error recovery."""
    
    def __init__(self, log_dir: Path):
        """
        Initialize conversion log.
        
        Parameters
        ----------
        log_dir : Path
            Directory to store log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.timestamp = datetime.now().isoformat()
        self.log_file = self.log_dir / f"conversion_{self.timestamp.replace(':', '-')}.json"
        
        self.entries = []
    
    def add_entry(
        self,
        subject: str,
        series: str,
        status: str,
        message: str = '',
        error: Optional[str] = None,
    ) -> None:
        """
        Add log entry.
        
        Parameters
        ----------
        subject : str
            Subject identifier
        series : str
            Series name
        status : str
            Status ('success', 'failed', 'skipped')
        message : str
            Optional message
        error : Optional[str]
            Optional error message
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'subject': subject,
            'series': series,
            'status': status,
            'message': message,
            'error': error,
        }
        
        self.entries.append(entry)
        
        # Log to file immediately for persistence
        self._write_log()
    
    def _write_log(self) -> None:
        """Write log to JSON file."""
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.entries, f, indent=2)
        except Exception as e:
            LOGGER.error(f"Could not write log file: {e}")
    
    def get_summary(self) -> dict:
        """
        Get log summary.
        
        Returns
        -------
        dict
            Summary statistics
        """
        total = len(self.entries)
        successful = sum(1 for e in self.entries if e['status'] == 'success')
        failed = sum(1 for e in self.entries if e['status'] == 'failed')
        skipped = sum(1 for e in self.entries if e['status'] == 'skipped')
        
        return {
            'total': total,
            'successful': successful,
            'failed': failed,
            'skipped': skipped,
            'log_file': str(self.log_file),
        }


def resume_partial_conversion(log_file: Path) -> dict:
    """
    Resume partially completed conversion from log file.
    
    Parameters
    ----------
    log_file : Path
        Path to conversion log file
        
    Returns
    -------
    dict
        Dictionary of completed conversions (subject -> list of series)
    """
    if not log_file.exists():
        return {}
    
    try:
        with open(log_file) as f:
            entries = json.load(f)
        
        completed = {}
        for entry in entries:
            if entry['status'] == 'success':
                subject = entry['subject']
                series = entry['series']
                
                if subject not in completed:
                    completed[subject] = []
                
                completed[subject].append(series)
        
        LOGGER.info(f"Loaded {len(completed)} previously completed subjects from log")
        return completed
        
    except (json.JSONDecodeError, KeyError) as e:
        LOGGER.warning(f"Could not load conversion log: {e}")
        return {}


def skip_if_completed(
    series_name: str,
    completed_series: dict,
    subject: str,
) -> bool:
    """
    Check if series was already converted.
    
    Parameters
    ----------
    series_name : str
        DICOM series name
    completed_series : dict
        Dictionary of completed conversions
    subject : str
        Subject identifier
        
    Returns
    -------
    bool
        True if series should be skipped
    """
    if subject not in completed_series:
        return False
    
    return series_name in completed_series[subject]


def create_recovery_checkpoint(
    bids_dir: Path,
    subject: str,
    checkpoint_data: dict,
) -> None:
    """
    Create recovery checkpoint for crash recovery.
    
    Parameters
    ----------
    bids_dir : Path
        BIDS dataset directory
    subject : str
        Subject identifier
    checkpoint_data : dict
        Checkpoint data (series converted, etc.)
    """
    checkpoint_file = bids_dir / '.oncoprep' / f'{subject}_checkpoint.json'
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
    except Exception as e:
        LOGGER.warning(f"Could not write checkpoint file: {e}")
