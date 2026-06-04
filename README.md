# Medical Imaging — Machine Learning Projects

Two medical-imaging ML projects from Signals, Software and Health (ELEC5622, University of Sydney, 2025).

## 1. Brain MRI Region Classification (SVM)

Classification on structural brain MRI data: AAL-atlas regional volume features (90 regions) extracted from T1-weighted MRI via a skull-stripping → tissue-segmentation → MNI152-registration pipeline, then SVM classification with feature scaling and cross-validated hyperparameter tuning.

*The FSL-based preprocessing pipeline was course-provided infrastructure; the feature analysis and classification work in the notebook is my own.*

## 2. HEp-2 Cell Image Classification (CNN)

Deep-learning classification of HEp-2 cell microscopy images (indirect immunofluorescence staining patterns, used in autoimmune-disease diagnostics): PyTorch training pipeline with structured experiment configs, augmentation, and validation tracking.

## Repo contents

| Path | Description |
|---|---|
| `notebooks/brain_mri_svm.ipynb` | MRI regional-volume SVM classification |
| `notebooks/hep2_cell_classification.ipynb` | HEp-2 CNN classification |

**Data not redistributed** (medical imaging data, course-supplied). Notebooks document the expected data layout.

## Context

Coursework projects, ELEC5622, University of Sydney, 2025.
