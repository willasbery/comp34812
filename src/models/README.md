# Models

This directory contains various model implementations for evidence detection. Each model has its own approach, structure, and execution method.

## Model Overview

### SVM

- **Location**: `src/models/svm/`
- **Description**: A traditional machine learning approach using Support Vector Machines with GloVe embeddings and custom features.
- **Key Components**:
  - Feature extraction using GloVe embeddings with TF-IDF weighting
  - Custom feature engineering
  - Hyperparameter optimization with Optuna
  - Model evaluation with weighted metrics

### DeBERTa

- **Location**: `src/models/deberta/`
- **Description**: Implementation of the DeBERTa transformer model for evidence detection.
- **Key Components**:
  - Fine-tuning of pre-trained DeBERTa model
  - Parameter-efficient fine-tuning with LoRA
  - Advanced tokenization strategies
  - Training with weighted loss function

### DeBERTa with EFL

- **Location**: `src/models/deberta efl/`
- **Description**: Enhanced DeBERTa model using Evidence-Focused Learning (EFL).
- **Key Components**:
  - Integration of explicit evidence tagging
  - Attention mechanisms focused on evidence elements
  - Custom loss function for evidence-aware learning

### T5

- **Location**: `src/models/T5/`
- **Description**: Sequence-to-sequence approach using the T5 transformer for evidence detection.
- **Key Components**:
  - Text-to-text format for classification
  - Encoder-decoder architecture
  - Generation-based evidence detection

### XGBoost

- **Location**: `src/models/xgboost/`
- **Description**: Gradient boosting implementation using XGBoost for evidence classification.
- **Key Components**:
  - Feature extraction similar to SVM
  - Tree-based ensemble learning
  - Hyperparameter optimization

### RoBERTa

- **Location**: `src/models/roberta/`
- **Description**: Transformer-based approach using RoBERTa for evidence detection.
- **Key Components**:
  - Fine-tuning of pre-trained RoBERTa model
  - Custom head for classification
  - Training with learning rate scheduling

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

#### DeBERTa with EFL

```bash
uv run python -m src.models.deberta_efl.main
```

#### T5 Model

```bash
uv run python -m src.models.T5.main
```

#### XGBoost Model

```bash
uv run python -m src.models.xgboost.main
```

#### RoBERTa Model

```bash
uv run python -m src.models.roberta.main
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

#### DeBERTa with EFL

```bash
python -m src.models.deberta_efl.main
```

#### T5 Model

```bash
python -m src.models.T5.main
```

#### XGBoost Model

```bash
python -m src.models.xgboost.main
```

#### RoBERTa Model

```bash
python -m src.models.roberta.main
```

## Data Augmentation (takes >1h so run at your own risk)

To run data augmentation before training models:

```bash
# Using UV
uv run python -m src.augmentation.pipeline

# Using Python/pip
python -m src.augmentation.pipeline
```
