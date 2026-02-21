# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""OncoPrep interfaces and wrappers."""

try:
    from niworkflows.interfaces.bids import (
        BIDS_DERIV_PATTERNS,
        DerivativesDataSink as _NiworkflowsDerivativesDataSink,
    )

    # Custom path patterns for non-standard BIDS suffixes (e.g. "features")
    _ONCOPREP_EXTRA_PATTERNS = (
        'sub-{subject}[/ses-{session}]/{datatype<anat>|anat}/sub-{subject}'
        '[_ses-{session}][_acq-{acquisition}][_rec-{reconstruction}][_run-{run}]'
        '[_space-{space}][_cohort-{cohort}][_res-{resolution}][_desc-{desc}]'
        '_{suffix<features>}{extension<.json|.tsv>|.json}',
        'sub-{subject}/{datatype<figures>}/sub-{subject}'
        '[_ses-{session}][_acq-{acquisition}][_rec-{reconstruction}][_run-{run}]'
        '[_space-{space}][_cohort-{cohort}][_desc-{desc}]'
        '_{suffix<features|report>}{extension<.html|.svg|.txt>|.html}',
    )

    class DerivativesDataSink(_NiworkflowsDerivativesDataSink):
        """DerivativesDataSink with out_path_base set to 'oncoprep'."""
        out_path_base = 'oncoprep'
        _file_patterns = _ONCOPREP_EXTRA_PATTERNS + BIDS_DERIV_PATTERNS

    __all__ = ['DerivativesDataSink']
except ImportError:
    __all__ = []

try:
    from .cifti import GenerateDScalar  # noqa: F401
    __all__.append('GenerateDScalar')
except ImportError:
    pass

try:
    from .freesurfer import MakeMidthickness, MRIsConvertData, ReconAll  # noqa: F401
    __all__.extend(['ReconAll', 'MRIsConvertData', 'MakeMidthickness'])
except ImportError:
    pass

try:
    from .fsl import FAST  # noqa: F401
    __all__.append('FAST')
except ImportError:
    pass

try:
    from .gifti import MetricMath  # noqa: F401
    __all__.append('MetricMath')
except ImportError:
    pass

try:
    from .msm import MSM  # noqa: F401
    __all__.append('MSM')
except ImportError:
    pass

try:
    from .reports import SubjectSummary, AboutSummary, FSSurfaceReport  # noqa: F401
    __all__.extend(['SubjectSummary', 'AboutSummary', 'FSSurfaceReport'])
except ImportError:
    pass

try:
    from .radiomics import HistogramNormalization, PyRadiomicsFeatureExtraction  # noqa: F401
    __all__.extend(['HistogramNormalization', 'PyRadiomicsFeatureExtraction'])
except ImportError:
    pass

try:
    from .mriqc import MRIQC, MRIQCGroup, check_mriqc_available  # noqa: F401
    __all__.extend(['MRIQC', 'MRIQCGroup', 'check_mriqc_available'])
except ImportError:
    pass

try:
    from .surf import NormalizeSurf, FixGiftiMetadata, AggregateSurfaces, MakeRibbon  # noqa: F401
    __all__.extend(['NormalizeSurf', 'FixGiftiMetadata', 'AggregateSurfaces', 'MakeRibbon'])
except ImportError:
    pass

try:
    from .templateflow import TemplateFlowSelect, TemplateDesc  # noqa: F401
    __all__.extend(['TemplateFlowSelect', 'TemplateDesc'])
except ImportError:
    pass

try:
    from .workbench import (  # noqa: F401
        CreateSignedDistanceVolume,
        SurfaceAffineRegression,
        SurfaceApplyAffine,
        SurfaceApplyWarpfield,
        SurfaceModifySphere,
        SurfaceSphereProjectUnproject,
        SurfaceResample,
    )
    __all__.extend([
        'CreateSignedDistanceVolume',
        'SurfaceAffineRegression',
        'SurfaceApplyAffine',
        'SurfaceApplyWarpfield',
        'SurfaceModifySphere',
        'SurfaceSphereProjectUnproject',
        'SurfaceResample',
    ])
except ImportError:
    pass

try:
    from .bids import (  # noqa: F401
        validate_bids_dataset,
        collect_bids_data,
        get_anatomical_files,
        get_functional_files,
        validate_anatomical_coverage,
        get_subjects_sessions,
        OncoprepBIDSDataGrabber,
    )
    __all__.extend([
        'validate_bids_dataset',
        'collect_bids_data',
        'get_anatomical_files',
        'get_functional_files',
        'validate_anatomical_coverage',
        'get_subjects_sessions',
        'OncoprepBIDSDataGrabber',
    ])
except ImportError:
    pass
