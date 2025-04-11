# Experimental Models

This directory contains experimental model implementations for evidence detection. These models were developed and tested during the research phase but are not part of the final production pipeline.

## Model Overview

### DeBERTa with EFL

- **Location**: `src/experiments/deberta efl/`
- **Description**: Enhanced DeBERTa model using Evidence-Focused Learning (EFL).
- **Key Components**:
  - Integration of explicit evidence tagging
  - Attention mechanisms focused on evidence elements
  - Custom loss function for evidence-aware learning
- **Status**: Experimental - Not used in final pipeline

### T5

- **Location**: `src/experiments/T5/`
- **Description**: Sequence-to-sequence approach using the T5 transformer for evidence detection.
- **Key Components**:
  - Text-to-text format for classification
  - Encoder-decoder architecture
  - Generation-based evidence detection
- **Status**: Experimental - Not used in final pipeline

### XGBoost

- **Location**: `src/experiments/xgboost/`
- **Description**: Gradient boosting implementation using XGBoost for evidence classification.
- **Key Components**:
  - Feature extraction similar to SVM
  - Tree-based ensemble learning
  - Hyperparameter optimization
- **Status**: Experimental - Not used in final pipeline

### RoBERTa

- **Location**: `src/experiments/roberta/`
- **Description**: Transformer-based approach using RoBERTa for evidence detection.
- **Key Components**:
  - Fine-tuning of pre-trained RoBERTa model
  - Custom head for classification
  - Training with learning rate scheduling
- **Status**: Experimental - Not used in final pipeline

## Running the Models

You can run each experimental model using either the traditional Python/pip approach or using UV (recommended).

### Using UV (Recommended)

UV provides isolated environments and faster dependency resolution:

#### DeBERTa with EFL

```bash
uv run python -m src.experiments.deberta_efl.main
```

#### T5 Model

```bash
uv run python -m src.experiments.T5.main
```

#### XGBoost Model

```bash
uv run python -m src.experiments.xgboost.main
```

#### RoBERTa Model

```bash
uv run python -m src.experiments.roberta.main
```

### Using Python/pip

If you prefer using the traditional Python approach:

#### DeBERTa with EFL

```bash
python -m src.experiments.deberta_efl.main
```

#### T5 Model

```bash
python -m src.experiments.T5.main
```

#### XGBoost Model

```bash
python -m src.experiments.xgboost.main
```

#### RoBERTa Model

```bash
python -m src.experiments.roberta.main
```

## Notes

- These models are experimental and may not be fully optimized or production-ready
- Some models may require additional setup or configuration
- Results and performance may vary from the production models
- Documentation and code quality may be less comprehensive than the production models

For production-ready models, please refer to the `src/models/` directory.
