# Interfaces

Nipype interfaces wrap atomic processing steps. OncoPrep extends
`niworkflows` interfaces for BIDS-aware data sinking and custom processing.

## Core interfaces

```{eval-rst}
.. module:: oncoprep.interfaces
.. autosummary::
   :toctree: _generated
   :nosignatures:

   DerivativesDataSink
```

## BIDS utilities

```{eval-rst}
.. module:: oncoprep.interfaces.bids
.. autosummary::
   :toctree: _generated
   :nosignatures:

   OncoprepBIDSDataGrabber
   validate_bids_dataset
   collect_bids_data
   get_anatomical_files
   validate_anatomical_coverage
   get_subjects_sessions
```

## Radiomics

```{eval-rst}
.. module:: oncoprep.interfaces.radiomics
.. autosummary::
   :toctree: _generated
   :nosignatures:

   HistogramNormalization
   PyRadiomicsFeatureExtraction
```

## FreeSurfer *(planned)*

:::{note}
FreeSurfer, CIFTI, and GIFTI interfaces are scaffolded for future surface
processing support and are not yet functional.
:::

```{eval-rst}
.. module:: oncoprep.interfaces.freesurfer
.. autosummary::
   :toctree: _generated
   :nosignatures:

   ReconAll
   MRIsConvertData
   MakeMidthickness
```

## CIFTI *(planned)*

```{eval-rst}
.. module:: oncoprep.interfaces.cifti
.. autosummary::
   :toctree: _generated
   :nosignatures:

   GenerateDScalar
```

## GIFTI *(planned)*

```{eval-rst}
.. module:: oncoprep.interfaces.gifti
.. autosummary::
   :toctree: _generated
   :nosignatures:

   MetricMath
```

## Reports

```{eval-rst}
.. module:: oncoprep.interfaces.reports
.. autosummary::
   :toctree: _generated
   :nosignatures:

   SubjectSummary
   AboutSummary
   FSSurfaceReport
```
