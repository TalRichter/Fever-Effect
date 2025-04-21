<p align="center">
  <img src="https://github.com/TalRichter/Fever-Effect/blob/main/workflow.png" width="600"/>
</p>

<p align="center">
  <strong>Fever Effect in ASD</strong><br/>
  Machine learning pipeline for predicting fever-related behavioral responses in autism using transcriptomic and clinical data.
</p>

---

## Overview

This repository contains the full pipeline used in our study of the **Fever Effect in Autism Spectrum Disorder (ASD)** — a phenomenon where some autistic individuals show temporary behavioral improvements during febrile episodes.

We integrated:
- **Transcriptomic data** (RNA-seq),
- **Genomic data** (WES), and
- **Rich clinical metadata**

from the **Simons Simplex Collection (SSC)** to uncover molecular signatures and predictive markers associated with this effect.

---

## Highlights

- ⚙️ **Interpretable ML Pipeline**  
  Built an XGBoost classifier to predict fever-associated subgroups, with:
  - Bootstrap-based downsampling for imbalanced data
  - SHAP for model interpretation
  - Feature selection and performance tracking

- **Differential Expression Analysis**  
  Used DESeq2 and edgeR-limma to identify genes linked to fever response.

- **Pathway-Based Outlier Detection**  
  Detected immune-related transcriptomic outliers using Mahalanobis distance.

- **Subgroup Discovery**  
  Applied spectral biclustering to identify ASD subtypes based on expression profiles.

---

## Key Folders

- [`XGboostAnalysis/`](https://github.com/TalRichter/Fever-Effect/tree/main/XGboostAnalysis) — Full modeling pipeline and tutorial
- `DE_Analysis/` — DESeq2 and edgeR results
- `OutlierDetection/` — Pathway-based outlier detection
- `Biclustering/` — Spectral biclustering and subgroup analysis

---

## Publication

This project supports the manuscript:  
**"Molecular Markers of the Fever Effect in Autism Spectrum Disorder"**  
---

## 📜 License

This repository is shared for academic and non-commercial use.
