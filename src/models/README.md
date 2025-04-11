# Models

This directory contains the production model implementations for evidence detection. Each model has its own approach, structure, and execution method.

## Model Overview

### SVM

- **Location**: `src/models/svm/`
- **Description**: A traditional machine learning approach using Support Vector Machines with GloVe embeddings and custom features.
- **Key Components**:
  - Feature extraction using GloVe embeddings with TF-IDF weighting
  - Custom feature engineering
  - Hyperparameter optimization with Optuna
  - Model evaluation with weighted metrics
- **NOTE**: this code was used to train the model locally, the notebook contains all of the code in one place and may not work.

### DeBERTa

- **Location**: `src/models/deberta/`
- **Description**: Implementation of the DeBERTa transformer model for evidence detection.
- **Key Components**:
  - Fine-tuning of pre-trained DeBERTa model
  - Parameter-efficient fine-tuning with LoRA
  - Advanced tokenization strategies
  - Training with weighted loss function

## Running the Models

You can run each model using either the traditional Python/pip approach or using UV (recommended).

### Using UV (Recommended)

UV provides isolated environments and faster dependency resolution:

#### SVM Model

```bash
uv run python -m src.models.svm.main
```

#### DeBERTa Model

```bash
uv run python -m src.models.deberta.main
```

### Using Python/pip

If you prefer using the traditional Python approach:

#### SVM Model

```bash
python -m src.models.svm.main
```

#### DeBERTa Model

```bash
python -m src.models.deberta.main
```

Note: For experimental models (T5, XGBoost, RoBERTa, and DeBERTa with EFL), please refer to the `src/experiments/` directory.
