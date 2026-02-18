OncoPrep Documentation
======================

**OncoPrep** is a neuro-oncology MRI preprocessing pipeline built on
`Nipype <https://nipype.readthedocs.io/>`_ workflows.  It follows
`NiPreps <https://www.nipreps.org/>`_ patterns (fMRIPrep, sMRIPrep) for
BIDS-Apps compatibility, supporting DICOM→BIDS conversion, tumor segmentation
with Docker-based multi-model ensembles, and BIDS derivative outputs.

.. note::

   **This is NOT a clinical tool.** OncoPrep is intended for research use only.

Key features
------------

- **BIDS-compliant preprocessing** — anatomical registration, skull-stripping,
  template normalization
- **nnInteractive default segmentation** — zero-shot 3D promptable foundation
  model (no Docker required; Isensee et al., 2025)
- **Multi-model ensemble segmentation** — 14 BraTS-challenge Docker models
- **Segmentation fusion** — majority voting, SIMPLE, and BraTS-specific fusion
- **Radiomics** — PyRadiomics-based quantitative feature extraction
- **Surface processing** *(planned)* — FreeSurfer + GIFTI/CIFTI-2 derivatives
- **HTML reports** — fMRIPrep-style quality-assurance reports


.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   tutorial
   usage/cli
   usage/segmentation
   usage/docker
   usage/hpc

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api/workflows
   api/interfaces
   api/io
   api/utils

.. toctree::
   :maxdepth: 1
   :caption: About

   changes
   contributing


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
