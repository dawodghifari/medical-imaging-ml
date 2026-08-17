# Medical Imaging — Machine Learning Projects

Two medical-imaging ML projects from Signals, Software and Health (ELEC5622, University of Sydney, 2025).

## 1. Brain MRI Region Classification (SVM)

Classification on structural brain MRI: AAL-atlas regional volume features across 90 regions, extracted from T1-weighted scans through skull stripping → tissue segmentation → MNI152 registration, then an SVM on the scaled features.

Linear and RBF kernels are compared under the same 5-fold stratified cross-validation, with `C` (and `gamma` for RBF) chosen by grid search inside the folds rather than on the whole set — 90 features against a small clinical cohort is exactly the shape where tuning on all the data produces a number that will not survive contact with a new scan. Model selection uses out-of-fold ROC-AUC.

*The FSL-based preprocessing pipeline was course-provided infrastructure; the feature analysis and classification work in the notebook is my own.*

## 2. HEp-2 Cell Image Classification (CNN)

Classification of HEp-2 cell microscopy images — indirect immunofluorescence staining patterns, read in autoimmune-disease diagnostics. PyTorch, 50 epochs, AdamW with `ReduceLROnPlateau` on the validation metric, 224px augmented inputs, structured experiment configs so runs are reproducible rather than reconstructed from memory.

## Repo contents

| Path | Description |
|---|---|
| `notebooks/brain_mri_svm.ipynb` | MRI regional-volume SVM classification |
| `notebooks/hep2_cell_classification.ipynb` | HEp-2 CNN classification |

**Data not redistributed** (medical imaging data, course-supplied). Notebooks document the expected data layout.

## No results table, and why

Both notebooks are committed with outputs stripped, and the data cannot be redistributed, so there is no figure here I can point at and defend. Quoting a remembered accuracy would be worse than quoting nothing. The notebooks show how the models were selected and evaluated; the numbers require a rerun against the course data.

## Limitations

- Small cohorts. Both datasets are teaching sets, and a cross-validated AUC on a few dozen subjects has wide error bars whatever the point estimate says.
- Regional volumes discard everything about shape and texture within a region. The atlas decides in advance what counts as a feature.
- HEp-2 staining patterns are read by trained humans with disagreement between them. A classifier trained on one lab's labels inherits that lab's conventions.
- Neither model was tested on data from a different scanner or a different site, which is the only test that would mean anything clinically.

## Context

Coursework projects, ELEC5622, University of Sydney, 2025.
