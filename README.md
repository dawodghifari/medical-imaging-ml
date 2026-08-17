# Medical Imaging — Machine Learning Projects

Two medical-imaging ML projects from Signals, Software and Health (ELEC5622, University of Sydney, 2025).

## 1. Brain MRI Region Classification (SVM)

Classification on structural brain MRI: AAL-atlas regional volume features across 90 regions, extracted from T1-weighted scans through skull stripping → tissue segmentation → MNI152 registration, then an SVM on the scaled features.

The task is Alzheimer's disease against normal control, roughly class-balanced, so `class_weight` is left alone. Linear and RBF kernels are compared under the same 5-fold stratified cross-validation, with `C` (and `gamma` for RBF) grid-searched inside the folds rather than on the whole set.

**Cross-validated ROC-AUC: 1.000 for the linear kernel, in every fold. 0.9875 for RBF.**

A perfect score is a warning, not a result. There are 40 samples against 90 features, so each fold holds out a handful of subjects and an SVM can separate almost anything in 90 dimensions at that sample size. What the number establishes is that the classes are linearly separable in this feature space on this cohort. It says nothing about a new scanner, a new site, or a new patient, and the RBF kernel scoring marginally *lower* is consistent with that reading: there was no non-linear structure left to find, so the extra flexibility only added variance.

*The FSL-based preprocessing pipeline was course-provided infrastructure; the feature analysis and classification work in the notebook is my own.*

## 2. HEp-2 Cell Image Classification (CNN)

Classification of HEp-2 cell microscopy images — indirect immunofluorescence staining patterns, read in autoimmune-disease diagnostics. Transfer learning with the classifier head replaced for the dataset's class count, trained 50 epochs at 224px with AdamW (lr 1e-4), batch 32, `ReduceLROnPlateau` scheduling on the validation metric, and resize-then-random-crop augmentation. Evaluation is a per-class `classification_report` and a confusion matrix rather than a single accuracy, since staining patterns are unevenly represented and one number would hide which pattern the model cannot read.

Config lives in a dataclass, so a run is reproducible from the file rather than reconstructed from memory.

## Repo contents

| Path | Description |
|---|---|
| `notebooks/brain_mri_svm.ipynb` | MRI regional-volume SVM classification |
| `notebooks/hep2_cell_classification.ipynb` | HEp-2 CNN classification |

**Data not redistributed** (medical imaging data, course-supplied). Notebooks document the expected data layout.

## No HEp-2 numbers here

Notebook outputs are stripped and the data cannot be redistributed. The MRI figures above are recorded in the notebook's own commentary, so they can be quoted; the HEp-2 run's per-class report is not, and it needs a rerun against the course data before any accuracy goes in this file. Quoting a remembered number would be worse than quoting none.

## Limitations

- **40 subjects.** Everything in the MRI section rests on a cohort small enough that a single mislabelled scan moves the result. Treat 1.000 as evidence about the feature space, not about the model.
- Regional volumes discard shape and texture within a region. The AAL atlas decides in advance what counts as a feature, and any signal it does not carve out is invisible to the classifier.
- HEp-2 staining patterns are read by trained humans who disagree with each other. A classifier trained on one lab's labels inherits that lab's conventions along with its errors.
- Neither model has seen data from a different scanner or a different site, which is the only test that would mean anything clinically.

## Context

Coursework projects, ELEC5622, University of Sydney, 2025.
