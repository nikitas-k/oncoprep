# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""OncoPrep interfaces and wrappers."""

try:
    from niworkflows.interfaces.bids import DerivativesDataSink as _NiworkflowsDerivativesDataSink
    
    class DerivativesDataSink(_NiworkflowsDerivativesDataSink):
        """DerivativesDataSink with out_path_base set to 'oncoprep'."""
        out_path_base = 'oncoprep'
    
    __all__ = ['DerivativesDataSink']
except ImportError:
    __all__ = []

try:
    from .cifti import GenerateDScalar
    __all__.append('GenerateDScalar')
except ImportError:
    pass

try:
    from .freesurfer import MakeMidthickness, MRIsConvertData, ReconAll
    __all__.extend(['ReconAll', 'MRIsConvertData', 'MakeMidthickness'])
except ImportError:
    pass

try:
    from .fsl import FAST
    __all__.append('FAST')
except ImportError:
    pass

try:
    from .gifti import MetricMath
    __all__.append('MetricMath')
except ImportError:
    pass

try:
    from .msm import MSM
    __all__.append('MSM')
except ImportError:
    pass

try:
    from .reports import SubjectSummary, AboutSummary, FSSurfaceReport
    __all__.extend(['SubjectSummary', 'AboutSummary', 'FSSurfaceReport'])
except ImportError:
    pass

try:
    from .radiomics import PyRadiomicsFeatureExtraction
    __all__.append('PyRadiomicsFeatureExtraction')
except ImportError:
    pass

try:
    from .surf import NormalizeSurf, FixGiftiMetadata, AggregateSurfaces, MakeRibbon
    __all__.extend(['NormalizeSurf', 'FixGiftiMetadata', 'AggregateSurfaces', 'MakeRibbon'])
except ImportError:
    pass

try:
    from .templateflow import TemplateFlowSelect, TemplateDesc
    __all__.extend(['TemplateFlowSelect', 'TemplateDesc'])
except ImportError:
    pass

try:
    from .workbench import (
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
    from .bids import (
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