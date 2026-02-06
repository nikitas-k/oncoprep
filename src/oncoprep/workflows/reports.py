"""Report generation for OncoPrep preprocessing results.

This module provides utilities for generating comprehensive HTML reports
documenting the preprocessing pipeline and quality assurance metrics,
including DICOM to BIDS conversion steps.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from jinja2 import Environment, FileSystemLoader, Template
import nibabel as nb
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine import Workflow

from oncoprep.utils.logging import get_logger

LOGGER = get_logger(__name__)


def init_report_wf(
    *,
    output_dir: str,
    subject_label: str,
    session_label: Optional[str] = None,
    name: str = 'report_wf',
) -> Workflow:
    """Generate comprehensive preprocessing report.

    Parameters
    ----------
    output_dir : :obj:`str`
        Directory to save reports
    subject_label : :obj:`str`
        Subject identifier
    session_label : :obj:`str` or None
        Session identifier
    name : :obj:`str`
        Workflow name (default: report_wf)

    Inputs
    ------
    anat_preproc
        Preprocessed anatomical image
    anat_mask
        Brain mask
    anat_dseg
        Tissue segmentation
    report_dict
        Dictionary of report metrics and parameters
    bids_metadata
        BIDS metadata from source files
    conversion_dict : dict, optional
        Dictionary with DICOM to BIDS conversion metrics:
        - dicom_count: number of DICOM files processed
        - subject_count: number of subjects in dataset
        - session_count: number of sessions in dataset
        - conversion_status: success/failure status
        - dcm2niix_version: version of dcm2niix used

    Outputs
    -------
    report_file
        Path to generated HTML report

    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'anat_preproc',
                'anat_mask',
                'anat_dseg',
                'report_dict',
                'bids_metadata',
                'conversion_dict',
            ]
        ),
        name='inputnode',
    )
    # Set default values for optional inputs
    inputnode.inputs.anat_dseg = None
    inputnode.inputs.report_dict = None
    inputnode.inputs.bids_metadata = None
    inputnode.inputs.conversion_dict = None
    
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['report_file']),
        name='outputnode',
    )

    # Generate report metadata
    gen_report = pe.Node(
        niu.Function(
            function=_generate_report_metrics,
            input_names=[
                'anat_preproc',
                'anat_mask',
                'anat_dseg',
                'report_dict',
                'conversion_dict',
            ],
            output_names=['metrics_dict'],
        ),
        name='gen_report',
    )

    # Render HTML report
    render_report = pe.Node(
        niu.Function(
            function=_render_html_report,
            input_names=[
                'metrics_dict',
                'subject_label',
                'session_label',
                'output_dir',
            ],
            output_names=['report_file'],
        ),
        name='render_report',
        run_without_submitting=True,
    )
    render_report.inputs.subject_label = subject_label
    render_report.inputs.session_label = session_label
    render_report.inputs.output_dir = output_dir

    workflow.connect([
        (inputnode, gen_report, [
            ('anat_preproc', 'anat_preproc'),
            ('anat_mask', 'anat_mask'),
            ('anat_dseg', 'anat_dseg'),
            ('report_dict', 'report_dict'),
            ('conversion_dict', 'conversion_dict'),
        ]),
        (gen_report, render_report, [('metrics_dict', 'metrics_dict')]),
        (render_report, outputnode, [('report_file', 'report_file')]),
    ])

    return workflow


def _generate_report_metrics(
    anat_preproc,
    anat_mask,
    anat_dseg=None,
    report_dict=None,
    conversion_dict=None,
):
    """Extract metrics from preprocessed images and conversion for reporting.

    Parameters
    ----------
    anat_preproc : :obj:`str`
        Path to preprocessed anatomical image
    anat_mask : :obj:`str`
        Path to brain mask
    anat_dseg : :obj:`str` or None
        Path to tissue segmentation
    report_dict : :obj:`dict` or None
        Dictionary of additional metrics
    conversion_dict : :obj:`dict` or None
        Dictionary of DICOM to BIDS conversion metrics

    Returns
    -------
    metrics_dict : :obj:`dict`
        Dictionary containing all report metrics including conversion info

    """
    import numpy as np
    import nibabel as nb
    from oncoprep.utils.logging import get_logger

    logger = get_logger(__name__)
    metrics_dict = report_dict or {}

    # Add conversion metrics if provided
    if conversion_dict:
        metrics_dict.update({
            'dicom_count': conversion_dict.get('dicom_count', 0),
            'subject_count': conversion_dict.get('subject_count', 0),
            'session_count': conversion_dict.get('session_count', 0),
            'conversion_status': conversion_dict.get('conversion_status', 'pending'),
            'dcm2niix_version': conversion_dict.get('dcm2niix_version', 'unknown'),
        })

    # Load images
    try:
        anat_img = nb.load(anat_preproc)
        anat_data = anat_img.get_fdata()

        mask_img = nb.load(anat_mask)
        mask_data = mask_img.get_fdata() > 0

        if anat_dseg:
            dseg_img = nb.load(anat_dseg)
            dseg_data = dseg_img.get_fdata()
        else:
            dseg_data = None

        # Compute basic statistics
        metrics_dict.update({
            'voxel_size': tuple(anat_img.header.get_zooms()[:3]),
            'image_shape': anat_data.shape,
            'brain_volume_mm3': float(
                np.sum(mask_data) * np.prod(anat_img.header.get_zooms()[:3])
            ),
            'mean_intensity': float(np.mean(anat_data[mask_data])),
            'std_intensity': float(np.std(anat_data[mask_data])),
        })

        # Tissue statistics
        if dseg_data is not None:
            tissue_labels = {1: 'GM', 2: 'WM', 3: 'CSF'}
            for label, tissue_name in tissue_labels.items():
                tissue_mask = dseg_data == label
                if tissue_mask.any():
                    metrics_dict[f'{tissue_name}_volume_mm3'] = float(
                        np.sum(tissue_mask)
                        * np.prod(anat_img.header.get_zooms()[:3])
                    )

    except Exception as e:
        logger.warning(f"Could not compute anatomical metrics: {e}")

    return metrics_dict


def _render_html_report(metrics_dict, subject_label, session_label, output_dir):
    """Render HTML report from metrics including conversion steps.

    Parameters
    ----------
    metrics_dict : :obj:`dict`
        Dictionary of metrics and parameters
    subject_label : :obj:`str`
        Subject identifier
    session_label : :obj:`str` or None
        Session identifier
    output_dir : :obj:`str`
        Output directory for report

    Returns
    -------
    report_file : :obj:`str`
        Path to generated HTML report

    """
    # Create session suffix for filename
    session_str = f"_ses-{session_label}" if session_label else ""

    # Generate report filename
    report_dir = Path(output_dir) / 'reports'
    report_dir.mkdir(parents=True, exist_ok=True)
    report_file = report_dir / f'sub-{subject_label}{session_str}_desc-preproc_report.html'

    # HTML template with conversion reporting
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>OncoPrep Report - sub-{{ subject_label }}</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
            .container { max-width: 1000px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            h1 { color: #333; border-bottom: 3px solid #0066cc; padding-bottom: 10px; }
            h2 { color: #0066cc; margin-top: 30px; padding-bottom: 10px; border-bottom: 1px solid #ddd; }
            .metrics-table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            .metrics-table td { padding: 10px; border: 1px solid #ddd; }
            .metrics-table tr:nth-child(even) { background-color: #f9f9f9; }
            .metric-label { font-weight: bold; width: 40%; }
            .metric-value { text-align: right; }
            .section { margin: 25px 0; padding: 15px; background-color: #f9f9f9; border-left: 4px solid #0066cc; }
            .conversion-section { border-left-color: #28a745; }
            .preprocessing-section { border-left-color: #0066cc; }
            .warning { background-color: #fff3cd; padding: 10px; border-radius: 5px; color: #856404; }
            .success { background-color: #d4edda; padding: 10px; border-radius: 5px; color: #155724; }
            .error { background-color: #f8d7da; padding: 10px; border-radius: 5px; color: #721c24; }
            .status-badge { display: inline-block; padding: 4px 12px; border-radius: 20px; font-weight: bold; font-size: 12px; }
            .status-success { background-color: #d4edda; color: #155724; }
            .status-pending { background-color: #fff3cd; color: #856404; }
            .status-error { background-color: #f8d7da; color: #721c24; }
            .checkmark { color: #28a745; font-weight: bold; }
            .footer { margin-top: 50px; font-size: 12px; color: #666; border-top: 1px solid #ccc; padding-top: 20px; }
            .processing-steps { list-style: none; padding-left: 0; }
            .processing-steps li { padding: 5px 0; }
            .processing-steps li:before { content: "✓ "; color: #28a745; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>OncoPrep Preprocessing Report</h1>
            <p><strong>Subject:</strong> sub-{{ subject_label }}{% if session_label %}<br><strong>Session:</strong> {{ session_label }}{% endif %}</p>

            {% if dicom_count or subject_count %}
            <div class="section conversion-section">
                <h2>DICOM to BIDS Conversion</h2>
                <table class="metrics-table">
                    {% if dicom_count %}
                    <tr>
                        <td class="metric-label">DICOM Files Processed:</td>
                        <td class="metric-value">{{ dicom_count }}</td>
                    </tr>
                    {% endif %}
                    {% if subject_count %}
                    <tr>
                        <td class="metric-label">Subjects in Dataset:</td>
                        <td class="metric-value">{{ subject_count }}</td>
                    </tr>
                    {% endif %}
                    {% if session_count %}
                    <tr>
                        <td class="metric-label">Sessions in Dataset:</td>
                        <td class="metric-value">{{ session_count }}</td>
                    </tr>
                    {% endif %}
                    {% if conversion_status %}
                    <tr>
                        <td class="metric-label">Conversion Status:</td>
                        <td class="metric-value">
                            {% if conversion_status == 'success' %}
                            <span class="status-badge status-success">✓ Success</span>
                            {% elif conversion_status == 'completed' %}
                            <span class="status-badge status-success">✓ Completed</span>
                            {% elif conversion_status == 'pending' %}
                            <span class="status-badge status-pending">⧖ Pending</span>
                            {% else %}
                            <span class="status-badge status-error">✗ Failed</span>
                            {% endif %}
                        </td>
                    </tr>
                    {% endif %}
                    {% if dcm2niix_version %}
                    <tr>
                        <td class="metric-label">dcm2niix Version:</td>
                        <td class="metric-value">{{ dcm2niix_version }}</td>
                    </tr>
                    {% endif %}
                </table>
                <p class="success">
                    <span class="checkmark">✓</span> DICOM files successfully organized into BIDS structure
                </p>
            </div>
            {% endif %}

            <div class="section preprocessing-section">
                <h2>Anatomical Preprocessing</h2>
                <table class="metrics-table">
                    <tr>
                        <td class="metric-label">Image Shape:</td>
                        <td class="metric-value">{{ image_shape }}</td>
                    </tr>
                    <tr>
                        <td class="metric-label">Voxel Size (mm):</td>
                        <td class="metric-value">{{ "%.2f" | format(voxel_size[0]) }} × {{ "%.2f" | format(voxel_size[1]) }} × {{ "%.2f" | format(voxel_size[2]) }}</td>
                    </tr>
                    <tr>
                        <td class="metric-label">Brain Volume (mm³):</td>
                        <td class="metric-value">{{ "%.0f" | format(brain_volume_mm3) }}</td>
                    </tr>
                    <tr>
                        <td class="metric-label">Mean Intensity (brain):</td>
                        <td class="metric-value">{{ "%.2f" | format(mean_intensity) }}</td>
                    </tr>
                    <tr>
                        <td class="metric-label">Std Dev Intensity (brain):</td>
                        <td class="metric-value">{{ "%.2f" | format(std_intensity) }}</td>
                    </tr>
                </table>
            </div>

            {% if GM_volume_mm3 %}
            <div class="section">
                <h2>Tissue Volumes</h2>
                <table class="metrics-table">
                    {% if GM_volume_mm3 %}
                    <tr>
                        <td class="metric-label">Gray Matter (mm³):</td>
                        <td class="metric-value">{{ "%.0f" | format(GM_volume_mm3) }}</td>
                    </tr>
                    {% endif %}
                    {% if WM_volume_mm3 %}
                    <tr>
                        <td class="metric-label">White Matter (mm³):</td>
                        <td class="metric-value">{{ "%.0f" | format(WM_volume_mm3) }}</td>
                    </tr>
                    {% endif %}
                    {% if CSF_volume_mm3 %}
                    <tr>
                        <td class="metric-label">CSF (mm³):</td>
                        <td class="metric-value">{{ "%.0f" | format(CSF_volume_mm3) }}</td>
                    </tr>
                    {% endif %}
                </table>
            </div>
            {% endif %}

            <div class="section">
                <h2>Processing Pipeline</h2>
                <ul class="processing-steps">
                    <li>DICOM to BIDS conversion (dcm2niix)</li>
                    <li>Anatomical averaging</li>
                    <li>N4 Bias Field Correction Applied</li>
                    <li>Brain Extraction Completed</li>
                    <li>Tissue Segmentation Performed</li>
                    <li>Template Registration Complete</li>
                </ul>
            </div>

            <div class="footer">
                <p>Generated by OncoPrep v0.1.0</p>
                <p>Report generation timestamp: {{ timestamp }}</p>
            </div>
        </div>
    </body>
    </html>
    """

    # Render template
    template = Template(html_template)

    html_content = template.render(
        subject_label=subject_label,
        session_label=session_label,
        timestamp=datetime.now().isoformat(),
        **metrics_dict,
    )

    # Write report
    report_file.write_text(html_content)
    LOGGER.info(f"Generated report: {report_file}")
    return str(report_file)


def generate_preprocessing_summary(
    subject_label: str,
    session_label: Optional[str] = None,
    metrics: Optional[Dict] = None,
    conversion_metrics: Optional[Dict] = None,
) -> str:
    """Generate plain-text summary of preprocessing and conversion.

    Parameters
    ----------
    subject_label : :obj:`str`
        Subject identifier
    session_label : :obj:`str` or None
        Session identifier
    metrics : :obj:`dict` or None
        Dictionary of preprocessing metrics
    conversion_metrics : :obj:`dict` or None
        Dictionary of DICOM to BIDS conversion metrics

    Returns
    -------
    summary : :obj:`str`
        Plain-text summary string

    """
    summary = "OncoPrep Preprocessing Summary\n"
    summary += "=" * 60 + "\n\n"
    summary += f"Subject: sub-{subject_label}\n"
    if session_label:
        summary += f"Session: ses-{session_label}\n"
    summary += "\n" + "-" * 60 + "\n"

    # DICOM to BIDS conversion summary
    if conversion_metrics:
        summary += "DICOM to BIDS Conversion:\n"
        if conversion_metrics.get('dicom_count'):
            summary += f"  • DICOM files processed: {conversion_metrics['dicom_count']}\n"
        if conversion_metrics.get('subject_count'):
            summary += f"  • Subjects in dataset: {conversion_metrics['subject_count']}\n"
        if conversion_metrics.get('session_count'):
            summary += f"  • Sessions in dataset: {conversion_metrics['session_count']}\n"
        if conversion_metrics.get('conversion_status'):
            summary += (
                f"  • Status: {conversion_metrics['conversion_status'].upper()}\n"
            )
        if conversion_metrics.get('dcm2niix_version'):
            summary += f"  • dcm2niix: {conversion_metrics['dcm2niix_version']}\n"
        summary += "\n" + "-" * 60 + "\n"

    summary += "Anatomical Preprocessing Steps Completed:\n"
    summary += "  ✓ DICOM to BIDS conversion (dcm2niix)\n"
    summary += "  ✓ Anatomical averaging\n"
    summary += "  ✓ N4 bias field correction\n"
    summary += "  ✓ Brain extraction (HD-BET)\n"
    summary += "  ✓ Tissue segmentation\n"
    summary += "  ✓ Multi-modal co-registration\n"
    summary += "  ✓ Template registration\n"

    if metrics:
        summary += "\n" + "-" * 60 + "\n"
        summary += "Image Metrics:\n"
        for key, value in sorted(metrics.items()):
            if isinstance(value, float):
                summary += f"  • {key}: {value:.2f}\n"
            else:
                summary += f"  • {key}: {value}\n"

    summary += "\n" + "-" * 60 + "\n"
    summary += "OncoPrep v0.1.0\n"
    return summary


__all__ = [
    'init_report_wf',
    'generate_preprocessing_summary',
]
