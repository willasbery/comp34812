# Source Code Documentation

This directory contains the main source code for the COMP34812 NLU coursework. Below is a detailed breakdown of each component.

## Directory Structure

### `models/` - Model Implementations

Contains implementations of various models for evidence detection:

- **SVM (`svm/`)**

  - Traditional machine learning approach using Support Vector Machines
  - Includes feature extraction using GloVe embeddings
  - Contains training and evaluation scripts

- **DeBERTa (`deberta/` and `deberta efl/`)**

  - Transformer-based models for evidence detection
  - Standard DeBERTa and Evidence-Focused Learning (EFL) variants
  - Includes model configuration and training utilities
  - **NOTE**: the DeBERTa fine-tuning file is found in `notebooks/`, not `models/`

- **Other Models**
  - T5: Transformer-based text-to-text model
  - XGBoost: Gradient boosting implementation
  - RoBERTa: Robustly optimized BERT approach

### `notebooks/` - Jupyter Notebooks

Contains experimental notebooks for different models and analyses:

- **DeBERTa Notebooks**
  - Training and evaluation notebooks
  - Hyperparameter tuning experiments
  - Model analysis and visualization

### `augmentation/` - Data Augmentation

Utilities for data augmentation and preprocessing:

- Text augmentation techniques
- Data balancing utilities
- Preprocessing scripts

### `utils/` - Utility Functions

Common utilities used across the project:

- Data loading and processing helpers
- Evaluation metrics
- Visualization tools
- Common configuration utilities

### Configuration

- `config.py`: Central configuration file containing:
  - Model parameters
  - Training settings
  - Path configurations
  - Hyperparameter defaults

## Usage

1. **Model Training**

   - Each model directory contains its own README with specific instructions
   - Use the notebooks in `notebooks/` for interactive experimentation
   - Configuration can be modified in `config.py`

2. **Data Processing**

   - Use utilities in `augmentation/` for data preprocessing
   - Common data loading functions are in `utils/`

3. **Evaluation**
   - Each model includes evaluation scripts
   - Common metrics are implemented in `utils/`

## Best Practices

1. **Adding New Models**

   - Create a new directory in `models/`
   - Include a README with setup and usage instructions
   - Use common utilities from `utils/` where possible

2. **Data Processing**

   - Use the augmentation utilities for consistent preprocessing
   - Cache embeddings in the `cache/` directory
   - Document any new preprocessing steps

3. **Configuration**
   - Add new configuration parameters to `config.py`
   - Document any new settings
   - Maintain backward compatibility

## Dependencies

- Python 3.8+
- PyTorch
- Transformers library
- Scikit-learn
- Jupyter Notebook
- Other dependencies specified in the root requirements.txt
