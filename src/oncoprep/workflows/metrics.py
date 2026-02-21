"""Legacy quality metrics workflows (DEPRECATED).

.. deprecated:: 0.2.0
    This module is superseded by :mod:`oncoprep.workflows.mriqc`, which wraps
    `MRIQC <https://mriqc.readthedocs.io>`_ for standardised, no-reference
    image quality metric (IQM) extraction.  The custom QA workflow factories
    in this file are retained only for backwards compatibility and will be
    removed in a future release.

Migration
---------
Replace usage of any workflow factory in this module with the MRIQC-based
workflow::

    # Old
    from oncoprep.workflows.metrics import init_qa_metrics_wf

    # New
    from oncoprep.workflows.mriqc import init_mriqc_wf
"""

from __future__ import annotations

import warnings

from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

_DEPRECATION_MSG = (
    '{name}() is deprecated and will be removed in a future release. '
    'Use oncoprep.workflows.mriqc.init_mriqc_wf() instead, which wraps '
    'MRIQC for standardised image quality metrics.'
)


def init_qa_metrics_wf(
    *,
    output_dir: str,
    name: str = 'qa_metrics_wf',
) -> Workflow:
    """Compute quality assurance metrics for preprocessing.

    .. deprecated:: 0.2.0
        Use :func:`oncoprep.workflows.mriqc.init_mriqc_wf` instead.
    """
    warnings.warn(
        _DEPRECATION_MSG.format(name='init_qa_metrics_wf'),
        DeprecationWarning,
        stacklevel=2,
    )
    workflow = Workflow(name=name)
    workflow.add_nodes([
        pe.Node(niu.IdentityInterface(fields=['qa_report']), name='outputnode'),
    ])
    return workflow


def init_snr_metrics_wf(
    *,
    output_dir: str,
    name: str = 'snr_metrics_wf',
) -> Workflow:
    """Compute signal-to-noise ratio metrics.

    .. deprecated:: 0.2.0
        Use :func:`oncoprep.workflows.mriqc.init_mriqc_wf` instead.
    """
    warnings.warn(
        _DEPRECATION_MSG.format(name='init_snr_metrics_wf'),
        DeprecationWarning,
        stacklevel=2,
    )
    workflow = Workflow(name=name)
    workflow.add_nodes([
        pe.Node(niu.IdentityInterface(fields=['snr_report']), name='outputnode'),
    ])
    return workflow


def init_coverage_metrics_wf(
    *,
    output_dir: str,
    name: str = 'coverage_metrics_wf',
) -> Workflow:
    """Compute brain coverage and field-of-view metrics.

    .. deprecated:: 0.2.0
        Use :func:`oncoprep.workflows.mriqc.init_mriqc_wf` instead.
    """
    warnings.warn(
        _DEPRECATION_MSG.format(name='init_coverage_metrics_wf'),
        DeprecationWarning,
        stacklevel=2,
    )
    workflow = Workflow(name=name)
    workflow.add_nodes([
        pe.Node(niu.IdentityInterface(fields=['coverage_report']), name='outputnode'),
    ])
    return workflow


def init_tissue_stats_wf(
    *,
    output_dir: str,
    name: str = 'tissue_stats_wf',
) -> Workflow:
    """Compute detailed tissue-specific statistics.

    .. deprecated:: 0.2.0
        Use :func:`oncoprep.workflows.mriqc.init_mriqc_wf` instead.
    """
    warnings.warn(
        _DEPRECATION_MSG.format(name='init_tissue_stats_wf'),
        DeprecationWarning,
        stacklevel=2,
    )
    workflow = Workflow(name=name)
    workflow.add_nodes([
        pe.Node(niu.IdentityInterface(fields=['tissue_stats']), name='outputnode'),
    ])
    return workflow


def init_registration_quality_wf(
    *,
    output_dir: str,
    name: str = 'registration_quality_wf',
) -> Workflow:
    """Assess quality of template registration.

    .. deprecated:: 0.2.0
        Use :func:`oncoprep.workflows.mriqc.init_mriqc_wf` instead.
    """
    warnings.warn(
        _DEPRECATION_MSG.format(name='init_registration_quality_wf'),
        DeprecationWarning,
        stacklevel=2,
    )
    workflow = Workflow(name=name)
    workflow.add_nodes([
        pe.Node(
            niu.IdentityInterface(fields=['reg_quality_report']),
            name='outputnode',
        ),
    ])
    return workflow


__all__ = [
    'init_qa_metrics_wf',
    'init_snr_metrics_wf',
    'init_coverage_metrics_wf',
    'init_tissue_stats_wf',
    'init_registration_quality_wf',
]
