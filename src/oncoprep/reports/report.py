from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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