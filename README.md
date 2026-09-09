# UAV-Based-Aquatic-Plant-Classification

A deep learning framework for UAV-based aquatic plant classification using convolutional neural networks, transfer learning, robustness evaluation, uncertainty calibration, selective prediction, and explainable AI.

---

# Overview

Aquatic vegetation monitoring is essential for environmental assessment, ecosystem management, and water resource monitoring. Traditional field-based identification methods are time-consuming and require expert knowledge.

This project presents a UAV-based aquatic plant classification framework using deep learning models to automatically identify aquatic plant species from aerial images.

The framework investigates multiple deep learning architectures and evaluates their performance through:

- Classification performance analysis
- Robustness evaluation
- Uncertainty calibration
- Selective prediction
- Explainable AI analysis
- Failure analysis

---

# Project Pipeline

```text
Dataset Audit
      |
      ↓
Data Preprocessing
      |
      ↓
Model Training
      |
      ↓
Locked Test Evaluation
      |
      ↓
Robustness Evaluation
      |
      ↓
Calibration & Uncertainty Analysis
      |
      ↓
Selective Prediction
      |
      ↓
Explainable AI
      |
      ↓
Failure Analysis
```

---

# Repository Structure

```text
UAV-Based-Aquatic-Plant-Classification/

├── Dataset/
│   └── split_information/
│
├── Paper/
│
├── Results/
│   ├── Training/
│   ├── Evaluation/
│   ├── Robustness/
│   ├── Calibration/
│   ├── Selective Prediction/
│   ├── Explainability/
│   └── Failure Analysis/
│
├── src/
│   ├── phase_01_dataset_audit/
│   ├── phase_02_preprocessing/
│   ├── phase_03_model_training/
│   ├── phase_04_test_evaluation/
│   ├── phase_05_robustness/
│   ├── phase_06_calibration/
│   ├── phase_07_selective_prediction/
│   ├── phase_08_explainability/
│   └── phase_09_failure_analysis/
│
├── .gitattributes
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

# Dataset

The project uses a UAV-based aquatic plant image dataset for multi-class classification.

Dataset preparation includes:

- Dataset integrity verification
- Duplicate detection
- Near-duplicate analysis
- Data cleaning
- Train/validation/test split generation
- Image preprocessing

### Dataset Split Information

[View Dataset Split Information](Dataset/split_information/)

Available files:

- [Final Split Summary](Dataset/split_information/final_split_summary.csv)
- [Near-Duplicate Groups](Dataset/split_information/near_duplicate_groups.csv)
- [Verified Near-Duplicates](Dataset/split_information/verified_near_duplicates.csv)

---

# Deep Learning Models

The following deep learning architectures were implemented and evaluated:

| Model | Type |
|---|---|
| Custom CNN | Baseline Convolutional Neural Network |
| VGG16 | Transfer Learning |
| ResNet50 | Transfer Learning |
| EfficientNet-B0 | Transfer Learning |

Model training notebooks and trained checkpoints are available in:

[View Model Training & Checkpoints](src/phase_03_model_training/)

---

# Experimental Phases

## Phase A — Dataset Audit

Purpose:

- Verify dataset integrity
- Identify duplicate samples
- Analyze dataset distribution
- Reduce potential data leakage

[Open Phase A — Dataset Audit](src/phase_01_dataset_audit/)

---

## Phase B — Preprocessing

Includes:

- Image preprocessing
- Data augmentation
- Dataset preparation
- Input pipeline development

[Open Phase B — Preprocessing](src/phase_02_preprocessing/)

---

## Phase C — Model Training

Implemented models:

- Custom CNN
- VGG16
- ResNet50
- EfficientNet-B0

[Open Phase C — Model Training](src/phase_03_model_training/)

---

## Phase D — Locked Test Evaluation

Evaluation includes:

- Confusion matrices
- Classification metrics
- Prediction analysis
- Model comparison

[Open Phase D — Test Evaluation](src/phase_04_test_evaluation/)

---

## Phase E — Robustness Evaluation

The trained models were evaluated under image corruption conditions to analyze their robustness.

Includes:

- Corruption testing
- Performance degradation analysis
- Severity-based evaluation

[Open Phase E — Robustness Evaluation](src/phase_05_robustness/)

---

## Phase F — Calibration & Uncertainty Analysis

Prediction confidence and calibration were evaluated using:

- Reliability analysis
- Expected Calibration Error (ECE)
- Brier score analysis
- Uncertainty evaluation

[Open Phase F — Calibration & Uncertainty](src/phase_06_calibration/)

---

## Phase G — Selective Prediction

This phase investigates the ability of the models to reject uncertain predictions.

Includes:

- Confidence threshold analysis
- Risk-coverage analysis
- Error rejection analysis

[Open Phase G — Selective Prediction](src/phase_07_selective_prediction/)

---

## Phase H — Explainability

Model decisions were analyzed using explainability techniques.

Includes:

- Grad-CAM visualization
- Attention region analysis
- Error-case visualization

[Open Phase H — Explainability](src/phase_08_explainability/)

---

## Phase I — Failure Analysis

The final phase analyzes model failure patterns.

Includes:

- Misclassified samples
- Confusion patterns
- Cross-model errors
- Failure characteristics
- Model limitations

[Open Phase I — Failure Analysis](src/phase_09_failure_analysis/)

---

# Results

All experimental outputs are organized in the Results directory.

[View All Results](Results/)

## Training Results

Includes:

- Accuracy curves
- Loss curves
- Macro F1-score curves
- Training history

[View Training Results](Results/Training/)

---

## Evaluation Results

Includes:

- Confusion matrices
- Classification metrics
- Test predictions
- Model comparison

[View Evaluation Results](Results/Evaluation/)

---

## Robustness Results

Includes:

- Corruption evaluation
- Performance degradation analysis
- Severity-based analysis

[View Robustness Results](Results/Robustness/)

---

## Calibration Results

Includes:

- Reliability analysis
- Calibration metrics
- Expected Calibration Error
- Brier score analysis
- Uncertainty analysis

[View Calibration Results](Results/Calibration/)

---

## Selective Prediction Results

Includes:

- Risk-coverage curves
- Confidence threshold analysis
- Selective accuracy analysis
- Error rejection analysis

[View Selective Prediction Results](Results/Selective%20Prediction/)

---

## Explainability Results

Includes:

- Grad-CAM visualizations
- Model attention analysis
- Selected prediction cases

[View Explainability Results](Results/Explainability/)

---

## Failure Analysis Results

Includes:

- Error analysis
- Confusion patterns
- Cross-model failure analysis
- Priority failure cases

[View Failure Analysis Results](Results/Failure%20Analysis/)

---

# Trained Model Checkpoints

The trained model checkpoints are managed using Git LFS and stored with the model-training implementation.

Available architectures include:

- Custom CNN
- VGG16
- ResNet50
- EfficientNet-B0

[View Trained Models](src/phase_03_model_training/)

---

# Installation

Clone the repository:

```bash
git clone https://github.com/srabon-d-costa/UAV-Based-Aquatic-Plant-Classification.git
```

Move into the repository:

```bash
cd UAV-Based-Aquatic-Plant-Classification
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Project

The implementation is organized as Jupyter notebooks across nine experimental phases.

[Open Source Code](src/)

The recommended execution sequence is:

```text
Phase 01 → Phase 02 → Phase 03 → Phase 04 → Phase 05
    → Phase 06 → Phase 07 → Phase 08 → Phase 09
```

Each phase corresponds to a specific stage of the research pipeline.

---

# Reproducibility

To reproduce the experimental workflow:

1. Clone this repository.
2. Install the packages listed in `requirements.txt`.
3. Prepare the dataset according to the dataset split information.
4. Execute the notebooks sequentially from Phase 01 to Phase 09.
5. Compare the generated outputs with the provided experimental results.

Useful links:

- [Dataset Split Information](Dataset/split_information/)
- [Source Code](src/)
- [Experimental Results](Results/)
- [Model Training & Checkpoints](src/phase_03_model_training/)

---

# Paper

Research documents associated with this project are available in:

[Open Paper Folder](Paper/)

The folder contains the research manuscript and related project materials.

---

# Citation

If you use this repository or its implementation in your research, please cite:

```bibtex
@software{uav_aquatic_plant_classification,
  author = {Srabon D Costa},
  title = {UAV-Based-Aquatic-Plant-Classification},
  year = {2026},
  url = {https://github.com/srabon-d-costa/UAV-Based-Aquatic-Plant-Classification}
}
```

---

# License

This project is released under the MIT License.

[View License](LICENSE)

---

# Acknowledgement

This repository provides the implementation and experimental artifacts for a UAV-based aquatic plant classification workflow, covering dataset analysis, preprocessing, deep learning model development, locked test evaluation, robustness assessment, calibration and uncertainty analysis, selective prediction, explainability, and failure analysis.
