# OncoPrep Validation Protocol — Scripting Suite

> **Protocol v1.0 — 9 Feb 2026**
> Multi-phase product efficacy validation for OncoPrep neuro-oncology MRI preprocessing + tumour segmentation.

## Overview

This `validation/` directory implements the complete scripting infrastructure for the **OncoPrep formal validation protocol**, aligned with:

- **CLAIM** (Checklist for AI in Medical Imaging, 2024 update)
- **BIDS Derivatives** conventions for masks / segmentations / provenance
- **fMRIPrep-style** thesis: standards-native, analysis-agnostic, robust to heterogeneity, with visual QC checkpoints

### Phases

| Phase | Script | What it measures |
|-------|--------|-----------------|
| **A** | `phase_a.py` | Completion rate, BIDS compliance, runtime, reproducibility |
| **B** | `phase_b.py` | Patient-level & lesion-wise Dice, HD95, Surface Dice |
| **C** | `phase_c.py` | Robustness under perturbations (noise, bias, resolution, intensity) |
| **D** | `phase_d.py` | Volume agreement (Bland–Altman, ICC), longitudinal plausibility |
| **E** | `phase_e.py` | Reader study: time-to-acceptable, Likert, edit magnitude |

### Outputs

| Module | What it generates |
|--------|-------------------|
| `figures.py` | Figures 3–7 (matplotlib/seaborn, PDF/PNG/SVG) |
| `tables.py` | Tables 1–4 (LaTeX + CSV) |
| `claim_checklist.py` | Supplementary S1: filled CLAIM checklist |

### Shared modules

| Module | Purpose |
|--------|---------|
| `config.py` | Datasets, labels, regions, perturbations, SAP parameters, BIDS checks |
| `stats.py` | Bootstrap CI, Wilson CI, McNemar, ICC, Bland–Altman, degradation AUC |
| `metrics.py` | Dice, HD95, Surface Dice, lesion-wise metrics, volume extraction |

---

## Quick Start

### 1. Dry run — validate configuration

```bash
python -m validation.run_all --dry-run
```

### 2. Run Phase A (execution efficacy) on a dataset

```bash
python -m validation.phase_a \
    --bids-dir /data/ucsf_pdgm/bids \
    --output-dir /data/ucsf_pdgm/derivatives \
    --work-dir /tmp/work \
    --dataset ucsf_pdgm \
    --results-dir ./validation_results \
    --nprocs 8
```

### 3. Run Phase B (segmentation accuracy)

```bash
python -m validation.phase_b \
    --pred-dir /data/ucsf_pdgm/derivatives/oncoprep \
    --gt-dir /data/ucsf_pdgm/ground_truth \
    --dataset ucsf_pdgm \
    --results-dir ./validation_results
```

### 4. Run Phase C (robustness) — full perturbation sweep

```bash
python -m validation.phase_c \
    --bids-dir /data/ucsf_pdgm/bids \
    --gt-dir /data/ucsf_pdgm/ground_truth \
    --output-dir /data/ucsf_pdgm/phase_c_work \
    --dataset ucsf_pdgm \
    --results-dir ./validation_results \
    --perturbations gaussian_noise bias_field resolution_downsample intensity_scaling
```

### 5. Run Phase D (quantitative stability)

```bash
python -m validation.phase_d \
    --pred-dir /data/mu_glioma/derivatives/oncoprep \
    --gt-dir /data/mu_glioma/ground_truth \
    --dataset mu_glioma_post \
    --results-dir ./validation_results
```

### 6. Phase E — reader study setup + analysis

```bash
# Select cases + generate annotation template
python -m validation.phase_e select \
    --pred-dir /data/derivatives/oncoprep \
    --gt-dir /data/ground_truth \
    --dataset ucsf_pdgm \
    --n-per-stratum 10 \
    --readers R1 R2 R3 R4 R5 \
    --output ./reader_study/annotation_template.json

# After readers complete annotations:
python -m validation.phase_e analyze \
    --annotations-file ./reader_study/completed_annotations.json \
    --results-dir ./validation_results
```

### 7. Generate all figures + tables

```bash
python -m validation.figures \
    --results-dir ./validation_results \
    --output-dir ./figures \
    --format pdf --dpi 300

python -m validation.tables \
    --results-dir ./validation_results \
    --output-dir ./tables \
    --format latex
```

### 8. Full orchestration

```bash
python -m validation.run_all \
    --bids-dir /data/bids \
    --gt-dir /data/ground_truth \
    --output-dir /data/derivatives \
    --dataset ucsf_pdgm \
    --phases A B C D figures tables \
    --results-dir ./validation_results
```

---

## Directory Structure

```
validation/
├── __init__.py              # Package docstring
├── config.py                # Datasets, labels, SAP, perturbation specs
├── stats.py                 # Statistical analysis functions
├── metrics.py               # Segmentation metric computations
├── phase_a.py               # Phase A: execution efficacy
├── phase_b.py               # Phase B: segmentation accuracy
├── phase_c.py               # Phase C: robustness testing
├── phase_d.py               # Phase D: quantitative stability
├── phase_e.py               # Phase E: human factors
├── figures.py               # Figure 3–7 generation
├── tables.py                # Table 1–4 generation
├── claim_checklist.py       # CLAIM checklist (Supp S1)
├── run_all.py               # Full orchestration runner
└── README.md                # This file
```

## Results Structure

```
validation_results/
├── phase_a/
│   ├── phase_a_ucsf_pdgm.json
│   └── phase_a_mu_glioma_post.json
├── phase_b/
│   ├── phase_b_ucsf_pdgm.json
│   └── phase_b_mu_glioma_post.json
├── phase_c/
│   └── phase_c_ucsf_pdgm.json
├── phase_d/
│   └── phase_d_mu_glioma_post.json
├── phase_e/
│   └── phase_e_results.json
├── figures/
│   ├── figure3_execution_efficacy.pdf
│   ├── figure4_segmentation.pdf
│   ├── figure5_robustness.pdf
│   ├── figure6_quantitative_stability.pdf
│   └── figure7_human_factors.pdf
└── tables/
    ├── table1_cohort.tex
    ├── table2_endpoints.tex
    ├── table3_reliability.tex
    ├── table4_compute.tex
    └── S1_CLAIM.tex
```

---

## Datasets

| Key | Name | N | Modalities | Notes |
|-----|------|---|------------|-------|
| `ucsf_pdgm` | UCSF-PDGM | 501 subj | T1w, ce-T1w, T2w, FLAIR | Pre-operative, standardised 3T |
| `mu_glioma_post` | MU-Glioma Post | 203 pts / 594 tp | T1w, ce-T1w, T2w, FLAIR | Post-treatment, includes cavity |

---

## Statistical Analysis Plan (pre-specified)

- **Bootstrap**: 10,000 resamples, seed=42, 95% CI
- **Proportions**: Wilson score interval
- **Paired comparisons**: Bootstrap of median differences
- **ICC**: ICC(3,1) with F-test CI
- **Bland–Altman**: Mean diff ± 1.96 SD
- **Robustness AUC**: Trapezoid rule, normalised to [0,1]
- **Phase E**: Mixed-effects or Wilcoxon signed-rank on cross-over design

---

## Dependencies

Core (already in OncoPrep):
- `numpy`, `nibabel`, `scipy`

Additional for figures/tables:
- `matplotlib` (required)
- `seaborn` (optional, improves aesthetics)

---

## Manuscript Figure/Table Mapping

| Figure/Table | Module | Protocol Phase |
|-------------|--------|---------------|
| Figure 1 | (manual — pipeline diagram) | — |
| Figure 2 | (manual — dataset overview) | — |
| Figure 3 | `figures.figure3_execution_efficacy()` | A |
| Figure 4 | `figures.figure4_segmentation_performance()` | B |
| Figure 5 | `figures.figure5_robustness()` | C |
| Figure 6 | `figures.figure6_quantitative_stability()` | D |
| Figure 7 | `figures.figure7_human_factors()` | E |
| Table 1 | `tables.table1_cohort_summary()` | Config |
| Table 2 | `tables.table2_primary_endpoints()` | B |
| Table 3 | `tables.table3_reliability()` | A + C |
| Table 4 | `tables.table4_compute()` | A |
| Supp S1 | `claim_checklist.generate_claim_latex()` | All |
