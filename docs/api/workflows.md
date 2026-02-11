# Workflows

OncoPrep's processing logic is organized as Nipype workflows.
Each workflow factory returns a `Workflow` instance with `inputnode` and
`outputnode` for composable integration.

## Orchestration

```{eval-rst}
.. module:: oncoprep.workflows.base
.. autosummary::
   :toctree: _generated
   :nosignatures:

   init_oncoprep_wf
   init_single_subject_wf
```

## Anatomical preprocessing

```{eval-rst}
.. module:: oncoprep.workflows.anatomical
.. autosummary::
   :toctree: _generated
   :nosignatures:

   init_anat_preproc_wf
   init_anat_fit_wf
   init_anat_template_wf
```

## Tumor segmentation

```{eval-rst}
.. module:: oncoprep.workflows.segment
.. autosummary::
   :toctree: _generated
   :nosignatures:

   init_anat_seg_wf
```

## Segmentation fusion

```{eval-rst}
.. module:: oncoprep.workflows.fusion
.. autosummary::
   :toctree: _generated
   :nosignatures:

   init_fusion_wf
   init_anat_seg_fuse_wf
```

## Radiomics

```{eval-rst}
.. module:: oncoprep.workflows.radiomics
.. autosummary::
   :toctree: _generated
   :nosignatures:

   init_anat_radiomics_wf
   init_multimodal_radiomics_wf
```

## Quality metrics

```{eval-rst}
.. module:: oncoprep.workflows.metrics
.. autosummary::
   :toctree: _generated
   :nosignatures:

   init_qa_metrics_wf
   init_snr_metrics_wf
   init_coverage_metrics_wf
   init_tissue_stats_wf
   init_registration_quality_wf
```

## Derivatives output

```{eval-rst}
.. module:: oncoprep.workflows.outputs
.. autosummary::
   :toctree: _generated
   :nosignatures:

   init_ds_template_wf
   init_ds_modalities_wf
   init_ds_mask_wf
   init_ds_dseg_wf
   init_ds_tpms_wf
   init_ds_template_registration_wf
   init_template_iterator_wf
```

## BraTS-specific outputs

```{eval-rst}
.. module:: oncoprep.workflows.brats_outputs
.. autosummary::
   :toctree: _generated
   :nosignatures:

   init_ds_tumor_seg_wf
   init_ds_tumor_mask_wf
   init_ds_tumor_metrics_wf
   init_ds_multimodal_tumor_wf
```

## Surface processing *(planned)*

:::{note}
Surface processing workflows are not yet implemented. The interfaces below
are scaffolded for future FreeSurfer + GIFTI/CIFTI-2 integration.
:::

```{eval-rst}
.. module:: oncoprep.workflows.surfaces
.. autosummary::
   :toctree: _generated
   :nosignatures:

   init_gifti_surfaces_wf
   init_gifti_morphometrics_wf
   init_surface_datasink_wf
```

## Reports

```{eval-rst}
.. module:: oncoprep.workflows.reports
.. autosummary::
   :toctree: _generated
   :nosignatures:

   init_report_wf
```

## BIDS conversion

```{eval-rst}
.. module:: oncoprep.workflows.conversion
.. autosummary::
   :toctree: _generated
   :nosignatures:

   init_bids_validation_wf
   init_bids_single_subject_convert_wf
   init_bids_convert_wf
```
