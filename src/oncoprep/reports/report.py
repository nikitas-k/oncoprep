from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from jinja2 import Environment, FileSystemLoader

from oncoprep.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class ReportContext:
    bids_root: Path
    derivatives_root: Path
    subject_id: str | None
    pipeline_name: str


def render_report(context: ReportContext, template_dir: Path, output_dir: Path) -> Path:
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    template = env.get_template("template.html.j2")

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.html"
    report_path.write_text(template.render(**context.__dict__))

    LOGGER.info("Rendered report to %s", report_path)
    return report_path


# ---- Filename patterns used by the collation function ----
# These mirror the BIDS entities written by DerivativesDataSink nodes.

_NORM_LABELS = {
    '_T1w.svg': 'T1w — Pre-contrast',
    '_acq-ce_T1w.svg': 'T1ce — Contrast-enhanced',
    '_T2w.svg': 'T2w',
    '_FLAIR.svg': 'FLAIR',
}


def collate_subject_report(
    output_dir: str,
    subject_id: str,
    version: str,
    report_files: list,
    workflow_desc: str = '',
) -> str:
    """Collate all figures into a single ``sub-<label>.html`` master report.

    This function is designed to be wrapped in a Nipype Function node.
    ``report_files`` carries data-dependency information so that Nipype
    schedules this node *after* every datasink has written its output;
    the actual file discovery is done by globbing the ``figures/`` directory
    so the function is robust to additional or missing reportlets.

    Parameters
    ----------
    output_dir : str
        Derivatives root (e.g. ``…/derivatives/oncoprep``).
    subject_id : str
        BIDS subject label **with** ``sub-`` prefix.
    version : str
        OncoPrep version string.
    report_files : list
        Sentinel list of written report file paths (used for DAG ordering).
    workflow_desc : str, optional
        Free-text workflow description to embed.

    Returns
    -------
    str
        Absolute path to the written HTML report.
    """
    from pathlib import Path as _Path

    from jinja2 import Environment, FileSystemLoader

    figures_dir = _Path(output_dir) / subject_id / 'figures'
    template_dir = _Path(__file__).resolve().parent  # src/oncoprep/reports/

    # --- Helper: safe-read a file ----------------------------------------
    def _read(path):
        try:
            return _Path(path).read_text(encoding='utf-8')
        except Exception:
            return ''

    # --- Discover and categorise figures ---------------------------------
    summary_html = ''
    about_html = ''
    conform_html = ''
    dseg_svg = ''
    tumor_svg = ''
    norm_figures: List[Tuple[str, str]] = []

    for fpath in sorted(figures_dir.glob('*')):
        fname = fpath.name
        if fname.endswith('_desc-summary_T1w.html'):
            summary_html = _read(fpath)
        elif fname.endswith('_desc-about_T1w.html'):
            about_html = _read(fpath)
        elif fname.endswith('_desc-conform_T1w.html'):
            conform_html = _read(fpath)
        elif fname.endswith('_desc-tumor_dseg.svg'):
            tumor_svg = _read(fpath)
        elif fname.endswith('_dseg.svg') and 'tumor' not in fname:
            dseg_svg = _read(fpath)
        elif fname.endswith('.svg') and 'space-' in fname:
            # Spatial normalization figure — determine modality label
            label = fname  # fallback
            for suffix, pretty in _NORM_LABELS.items():
                if fname.endswith(suffix):
                    # Extract space name
                    m = re.search(r'space-([A-Za-z0-9]+)', fname)
                    space = m.group(1) if m else 'standard'
                    label = f'{pretty} → {space}'
                    break
            norm_figures.append((label, _read(fpath)))

    # --- Render HTML report -----------------------------------------------
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=False,  # SVG content must be embedded raw
    )
    template = env.get_template('template.html.j2')

    html = template.render(
        subject_id=subject_id,
        version=version,
        workflow_desc=workflow_desc,
        summary_html=summary_html,
        conform_html=conform_html,
        dseg_svg=dseg_svg,
        norm_figures=norm_figures,
        tumor_svg=tumor_svg,
        about_html=about_html,
    )

    report_path = _Path(output_dir) / f'{subject_id}.html'
    report_path.write_text(html, encoding='utf-8')
    return str(report_path)