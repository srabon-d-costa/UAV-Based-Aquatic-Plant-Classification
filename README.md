# UAV-Based-Aquatic-Plant-Classification

A deep learning framework for UAV-based aquatic plant classification using convolutional neural networks, transfer learning, robustness evaluation, uncertainty calibration, selective prediction, and explainable AI.

---

# Overview

Aquatic vegetation monitoring is essential for environmental assessment, ecosystem management, and water resource monitoring. Traditional field-based identification methods are time-consuming and require expert knowledge.

This project presents a UAV-based aquatic plant classification framework using deep learning models to automatically identify aquatic plant species from aerial images.

The framework investigates multiple deep learning architectures and evaluates their performance under realistic conditions through:

- Classification performance analysis
- Robustness evaluation
- Uncertainty calibration
- Selective prediction
- Explainable AI analysis

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

```
UAV-Based-Aquatic-Plant-Classification/

│
├── Dataset/
├── Paper/
├── Results/
├── src/
├── requirements.txt
├── LICENSE
└── README.md
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

Dataset documentation and split information:

[View Dataset Folder](Dataset/)

---

# Deep Learning Models

The following deep learning architectures were implemented and evaluated:

| Model | Type |
|---|---|
| Custom CNN | Baseline Convolutional Neural Network |
| VGG16 | Transfer Learning |
| ResNet50 | Transfer Learning |
| EfficientNet-B0 | Transfer Learning |

---

# Experimental Phases

## Phase A — Dataset Audit

Purpose:

- Verify dataset integrity
- Identify duplicate samples
- Analyze dataset distribution
- Prevent data leakage

Implementation:

[Open Phase A - Dataset Audit](src/Phase_A_Dataset_Audit/)

---

## Phase B — Preprocessing

Includes:

- Image preprocessing
- Data augmentation
- Dataset preparation
- Input pipeline development

Implementation:

[Open Phase B - Preprocessing](src/Phase_B_Preprocessing/)

---

## Phase C — Model Training

Implemented models:

- Custom CNN
- VGG16
- ResNet50
- EfficientNet-B0

Implementation:

[Open Phase C - Model Training](src/Phase_C_Model_Training/)

---

## Phase D — Test Evaluation

Evaluation includes:

- Confusion matrices
- Classification metrics
- Prediction analysis
- Model comparison

Implementation:

[Open Phase D - Test Evaluation](src/Phase_D_Test_Evaluation/)

---

## Phase E — Robustness Evaluation

The trained models were evaluated under different image corruption conditions to analyze reliability.

Includes:

- Corruption testing
- Performance degradation analysis
- Severity analysis

Implementation:

[Open Phase E - Robustness Evaluation](src/Phase_E_Robustness/)

---

## Phase F — Calibration & Uncertainty Analysis

Prediction confidence reliability was evaluated using:

- Reliability diagrams
- Expected Calibration Error (ECE)
- Brier score analysis
- Uncertainty evaluation

Implementation:

[Open Phase F - Calibration Analysis](src/Phase_F_Calibration/)

---

## Phase G — Selective Prediction

This phase evaluates the capability of models to reject uncertain predictions.

Includes:

- Confidence threshold analysis
- Risk-coverage analysis
- Error rejection capability

Implementation:

[Open Phase G - Selective Prediction](src/Phase_G_Selective_Prediction/)

---

## Phase H — Explainable AI

Model decision analysis was performed using:

- Grad-CAM visualization
- Attention region analysis

Implementation:

[Open Phase H - Explainability](src/Phase_H_Explainability/)

---

## Phase I — Failure Analysis

Analysis of:

- Misclassified samples
- Confusion patterns
- Model limitations
- Error characteristics

Implementation:

[Open Phase I - Failure Analysis](src/Phase_I_Failure_Analysis/)

---

# Results

All experimental outputs are organized inside:

[View Results Folder](Results/)

The repository contains:

## Training Results

Includes:

- Training accuracy curves
- Loss curves
- Macro F1-score curves
- Training history files

---

## Evaluation Results

Includes:

- Confusion matrices
- Classification reports
- Prediction analysis
- Test performance comparison

---

## Robustness Results

Includes:

- Corruption evaluation
- Performance degradation analysis
- Severity-based analysis

---

## Calibration Results

Includes:

- Reliability diagrams
- Calibration metrics
- Uncertainty analysis

---

## Selective Prediction Results

Includes:

- Risk-coverage curves
- Confidence threshold analysis
- Error rejection analysis

---

## Explainability Results

Includes:

- Grad-CAM visualizations
- Model attention analysis

---

## Failure Analysis Results

Includes:

- Error analysis
- Confusion patterns
- Failure case investigation

---

# Trained Models

The trained model checkpoints are managed using Git LFS.

Available models:

[View Model Checkpoints](Models/)

Includes:

- Custom CNN
- VGG16
- ResNet50
- EfficientNet-B0

---

# Installation

Clone the repository:

```bash
git clone https://github.com/srabon-d-costa/UAV-Based-Aquatic-Plant-Classification.git
```

Install required dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Project

The implementation is provided through Jupyter notebooks.

Start Jupyter:

```bash
jupyter notebook
```

Navigate to:

[Open Source Code](src/)

Run the required experimental phase notebook.

Recommended execution order:

```
Phase A → Phase I
```

---

# Reproducibility

To reproduce the experiments:

1. Clone the repository.

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Prepare the dataset according to:

[Dataset Documentation](Dataset/)

4. Execute notebooks sequentially:

[Source Code](src/)

5. Generated outputs are available in:

[Results](Results/)

---

# Paper

Research documents are available in:

[Open Paper Folder](Paper/)

Including:

- Manuscript
- Conference paper version
- Poster presentation

---

# Citation

If you use this repository in your research, please cite:

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

See:

[LICENSE](LICENSE)

for details.

---

# Acknowledgement

This repository presents a complete deep learning workflow for UAV-based aquatic plant classification, including dataset analysis, model development, robustness evaluation, uncertainty analysis, selective prediction, explainable AI, and failure analysis.
