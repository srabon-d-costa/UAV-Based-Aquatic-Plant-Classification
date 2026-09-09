# UAV-Based-Aquatic-Plant-Classification

A reliability-oriented deep learning framework for UAV-based aquatic plant classification using convolutional neural networks, transfer learning, robustness evaluation, uncertainty calibration, selective prediction, explainable AI, and failure analysis.

---

## Quick Navigation

- [Dataset Information](Dataset/split_information/)
- [Source Code](src/)
- [Model Training & Checkpoints](src/phase_03_model_training/)
- [Experimental Results](Results/)
- [Research Paper & Materials](Paper/)
- [Requirements](requirements.txt)
- [License](LICENSE)

---

## Overview

Aquatic vegetation monitoring is important for environmental assessment, ecosystem management, biodiversity observation, and water resource monitoring. Traditional field-based identification methods can be time-consuming and often require specialist knowledge.

This project presents a UAV-based aquatic plant classification workflow using deep learning to identify aquatic plant classes from aerial imagery.

Rather than evaluating classification accuracy alone, the project follows a reliability-oriented experimental pipeline covering:

- Dataset integrity and leakage analysis
- Data preprocessing and augmentation
- Deep learning model training
- Locked test-set evaluation
- Robustness testing
- Calibration and uncertainty analysis
- Selective prediction
- Explainable AI using Grad-CAM
- Integrated failure analysis

---

## Research Workflow

```text
Dataset Audit
      │
      ▼
Data Preprocessing
      │
      ▼
Model Training
      │
      ▼
Locked Test Evaluation
      │
      ▼
Robustness Evaluation
      │
      ▼
Calibration & Uncertainty Analysis
      │
      ▼
Selective Prediction
      │
      ▼
Explainability Analysis
      │
      ▼
Failure Analysis
```

---

## Repository Structure

```text
UAV-Based-Aquatic-Plant-Classification/
│
├── Dataset/
│   └── split_information/
│
├── Paper/
│   ├── Aquatic Plant Poster.pdf
│   ├── Manuscript.pdf
│   └── Workflow.docx
│
├── Results/
│   ├── Calibration/
│   ├── Evaluation/
│   ├── Explainability/
│   ├── Failure Analysis/
│   ├── Robustness/
│   ├── Selective Prediction/
│   └── Training/
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

## Dataset

The classification workflow uses a UAV-based aquatic plant image dataset containing **14 aquatic plant classes**.

Following dataset auditing and leakage-aware preparation, the final experimental split contains **1,062 images**:

| Split | Images |
|---|---:|
| Training | 744 |
| Validation | 161 |
| Test | 157 |
| **Total** | **1,062** |

Dataset preparation includes:

- Dataset integrity verification
- Class-distribution analysis
- Corrupted-image checking
- Duplicate detection
- Near-duplicate analysis
- SSIM-based verification
- Group-aware dataset splitting
- Final leakage verification

The raw image dataset is not stored directly in this repository. Dataset split and audit information are provided here:

### Dataset Files

- [Final Split Summary](Dataset/split_information/final_split_summary.csv)
- [Near-Duplicate Groups](Dataset/split_information/near_duplicate_groups.csv)
- [Verified Near-Duplicates](Dataset/split_information/verified_near_duplicates.csv)
- [Open Dataset Information Folder](Dataset/split_information/)

---

## Deep Learning Models

Four deep learning architectures were investigated:

| Model | Approach |
|---|---|
| Custom CNN | Baseline convolutional neural network |
| VGG16 | Transfer learning |
| ResNet50 | Transfer learning |
| EfficientNet-B0 | Transfer learning |

### Training Notebooks and Checkpoints

| Model | Training Notebook | Checkpoint |
|---|---|---|
| Custom CNN | [Notebook](src/phase_03_model_training/aqua-plant-custom-cnn.ipynb) | [Best Model](src/phase_03_model_training/custom_cnn_best_f1.pth) |
| VGG16 | [Notebook](src/phase_03_model_training/aqua-plant-vgg16.ipynb) | [Best Model](src/phase_03_model_training/vgg16_stage1_best_f1.pth) |
| ResNet50 | [Notebook](src/phase_03_model_training/aqua-plant-resnet50-transfer-learning.ipynb) | [Stage 1](src/phase_03_model_training/resnet50_stage1_best.pth) / [Fine-tuned](src/phase_03_model_training/resnet50_finetuned_best.pth) |
| EfficientNet-B0 | [Notebook](src/phase_03_model_training/aqua-plant-efficientnetB0.ipynb) | [Frozen](src/phase_03_model_training/efficientnet_b0_frozen_best.pth) / [Fine-tuned](src/phase_03_model_training/efficientnet_b0_finetuned_best.pth) |

Model checkpoint files are managed using **Git LFS**.

[Open Complete Model Training Folder](src/phase_03_model_training/)

---

## Key Locked-Test Results

The locked clean-test evaluation stored in the repository reports the following results for the ResNet50 and EfficientNet-B0 configurations:

| Model Configuration | Test Accuracy | Macro-F1 | Macro ROC-AUC | Inference |
|---|---:|---:|---:|---:|
| **ResNet50 — Frozen** | **96.18%** | **95.90%** | 0.9994 | 2.98 ms/image |
| **ResNet50 — Fine-tuned** | **96.18%** | **95.90%** | **0.9995** | 2.82 ms/image |
| EfficientNet-B0 — Fine-tuned | 95.54% | 95.21% | 0.9974 | **1.42 ms/image** |
| EfficientNet-B0 — Frozen | 92.99% | 92.66% | 0.9953 | 1.46 ms/image |

The complete locked-test evaluation is available here:

- [Clean Test Summary](Results/Evaluation/Summary/phaseD_clean_test_summary.csv)
- [Fine-Tuning Ablation](Results/Evaluation/Summary/phaseD_finetuning_ablation.csv)
- [Model Comparison Figure](Results/Evaluation/Summary/phaseD_clean_test_comparison.png)
- [Open Evaluation Results](Results/Evaluation/)

---

# Experimental Phases

## Phase 01 — Dataset Audit

This phase verifies the reliability of the dataset before model development.

Main tasks include:

- Image counting
- Class-distribution analysis
- Image-resolution inspection
- Corrupted-image detection
- Duplicate checking
- Dataset cleaning
- Original split validation
- Near-duplicate auditing
- SSIM verification
- Near-duplicate grouping
- Group-aware splitting
- Final leakage verification

[Open Phase 01 — Dataset Audit](src/phase_01_dataset_audit/)

---

## Phase 02 — Preprocessing

This phase prepares the audited dataset for deep learning.

Main tasks include:

- Image preprocessing
- DataLoader preparation
- Data augmentation
- Augmentation visualization
- Class-balance analysis
- Final preprocessing configuration

[Open Phase 02 — Preprocessing](src/phase_02_preprocessing/)

---

## Phase 03 — Model Training

This phase trains and evaluates the candidate classification architectures.

Models include:

- Custom CNN
- VGG16
- ResNet50
- EfficientNet-B0

The transfer-learning experiments include frozen and fine-tuned configurations where applicable.

[Open Phase 03 — Model Training](src/phase_03_model_training/)

---

## Phase 04 — Locked Test Evaluation

This phase evaluates trained models on the locked test set.

Evaluation includes:

- Test accuracy
- Precision
- Recall
- Macro-F1
- Weighted-F1
- ROC-AUC
- Confusion matrices
- Class-wise metrics
- Prediction analysis
- Inference-time comparison
- Fine-tuning ablation

[Open Phase 04 — Locked Test Evaluation](src/phase_04_test_evaluation/)

[View Evaluation Results](Results/Evaluation/)

---

## Phase 05 — Robustness Evaluation

This phase measures model reliability under image corruption and degradation.

Analysis includes:

- Corruption-based testing
- Multiple corruption severities
- Macro-F1 degradation
- Strong-severity analysis
- Overall robustness comparison

[Open Phase 05 — Robustness](src/phase_05_robustness/)

[View Robustness Results](Results/Robustness/)

---

## Phase 06 — Calibration & Uncertainty Analysis

This phase examines whether model confidence corresponds to actual predictive correctness.

Analysis includes:

- Reliability diagrams
- Expected Calibration Error (ECE)
- Brier score
- Negative Log-Likelihood
- Temperature scaling
- Prediction uncertainty analysis

[Open Phase 06 — Calibration & Uncertainty](src/phase_06_calibration/)

[View Calibration Results](Results/Calibration/)

---

## Phase 07 — Selective Prediction

Selective prediction allows a model to abstain from predictions when confidence is insufficient.

Analysis includes:

- Confidence-threshold sweeps
- Risk-coverage analysis
- Selective accuracy
- Error-rejection efficiency
- Area Under the Risk-Coverage Curve
- High-confidence error analysis
- Low-confidence correct-case analysis

[Open Phase 07 — Selective Prediction](src/phase_07_selective_prediction/)

[View Selective Prediction Results](Results/Selective%20Prediction/)

---

## Phase 08 — Explainability

This phase analyzes which image regions influence model predictions.

Explainability analysis includes:

- Grad-CAM visualization
- Selected-case interpretation
- Misclassification visualization
- Model attention-region analysis

[Open Phase 08 — Explainability](src/phase_08_explainability/)

[View Explainability Results](Results/Explainability/)

---

## Phase 09 — Failure Analysis

The final phase integrates evidence from previous experiments to investigate model limitations.

Analysis includes:

- Unified error analysis
- Top confusion pairs
- Shared cross-model errors
- Model-specific errors
- Confidence and entropy behavior
- Error survival under selective prediction
- Strong-corruption comparison
- Priority failure cases
- Failure taxonomy

[Open Phase 09 — Failure Analysis](src/phase_09_failure_analysis/)

[View Failure Analysis Results](Results/Failure%20Analysis/)

---

# Experimental Results

All generated experimental outputs are organized inside the [`Results`](Results/) directory.

| Result Category | Contents | Link |
|---|---|---|
| Training | Accuracy, loss, Macro-F1 curves and training histories | [Open](Results/Training/) |
| Evaluation | Confusion matrices, metrics, predictions and test summaries | [Open](Results/Evaluation/) |
| Robustness | Corruption and severity-based robustness analysis | [Open](Results/Robustness/) |
| Calibration | Reliability, calibration and uncertainty analysis | [Open](Results/Calibration/) |
| Selective Prediction | Risk-coverage and confidence-threshold analysis | [Open](Results/Selective%20Prediction/) |
| Explainability | Grad-CAM and model-attention visualizations | [Open](Results/Explainability/) |
| Failure Analysis | Error patterns and integrated failure investigation | [Open](Results/Failure%20Analysis/) |

[View All Experimental Results](Results/)

---

## Installation

Clone the repository:

```bash
git clone https://github.com/srabon-d-costa/UAV-Based-Aquatic-Plant-Classification.git
```

Move into the project directory:

```bash
cd UAV-Based-Aquatic-Plant-Classification
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

## Running the Project

The complete implementation is organized into sequential experimental phases under:

[Open Source Code](src/)

Recommended execution order:

```text
Phase 01
   ↓
Phase 02
   ↓
Phase 03
   ↓
Phase 04
   ↓
Phase 05
   ↓
Phase 06
   ↓
Phase 07
   ↓
Phase 08
   ↓
Phase 09
```

Because later phases use trained checkpoints and outputs produced by earlier stages, the phases should normally be executed in sequence.

---

## Reproducibility

For reproducibility:

1. Clone the repository.
2. Install packages from [`requirements.txt`](requirements.txt).
3. Obtain and prepare the source dataset.
4. Follow the verified dataset split information in [`Dataset/split_information`](Dataset/split_information/).
5. Run the source-code phases sequentially from Phase 01 through Phase 09.
6. Compare generated outputs with the reference files provided in [`Results`](Results/).

### Important Resources

- [Dataset Split Information](Dataset/split_information/)
- [Source Code](src/)
- [Training Notebooks](src/phase_03_model_training/)
- [Evaluation Results](Results/Evaluation/)
- [All Results](Results/)
- [Research Materials](Paper/)

---

## Research Materials

The repository includes the research manuscript and supporting materials.

| File | Description |
|---|---|
| [Manuscript.pdf](Paper/Manuscript.pdf) | Research manuscript |
| [Aquatic Plant Poster.pdf](Paper/Aquatic%20Plant%20Poster.pdf) | Research poster |
| [Workflow.docx](Paper/Workflow.docx) | Research workflow document |

[Open Paper Folder](Paper/)

---

## Technologies and Libraries

The implementation primarily uses:

- Python
- PyTorch
- TorchVision
- NumPy
- Pandas
- SciPy
- OpenCV
- Pillow
- scikit-image
- scikit-learn
- Albumentations
- Matplotlib
- Grad-CAM
- TorchInfo
- Jupyter

The complete dependency list is available in:

[requirements.txt](requirements.txt)

---

## Citation

If you use this repository, its implementation, or its experimental resources in your research, please cite the repository:

```bibtex
@software{uav_aquatic_plant_classification_2026,
  author = {Srabon D Costa},
  title = {UAV-Based-Aquatic-Plant-Classification},
  year = {2026},
  url = {https://github.com/srabon-d-costa/UAV-Based-Aquatic-Plant-Classification}
}
```

---

## License

This project is distributed under the **MIT License**.

[View LICENSE](LICENSE)

---

## Acknowledgement

This repository provides a complete reliability-oriented experimental workflow for UAV-based aquatic plant classification. It combines dataset integrity analysis, preprocessing, deep learning model development, locked test evaluation, robustness testing, calibration and uncertainty analysis, selective prediction, explainability, and failure analysis within a reproducible research structure.
