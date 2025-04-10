# COMP34812 - Natural Language Understanding - Coursework

This repository contains code and models for the COMP34812 NLU coursework, focusing on evidence detection. The project implements various machine learning approaches including traditional approaches like SVM and XGBoost, and other transformer-based models like T5, RoBERTa and DeBERTa.

## Repository Structure

```
.
├── src/                   # Source code directory
│   ├── augmentation/      # Data augmentation utilities
│   ├── models/            # Model implementations
│   │   ├── svm/           # SVM model implementation
│   │   ├── deberta/       # DeBERTa model implementation
│   │   ├── deberta efl/   # DeBERTa with EFL implementation
│   │   ├── T5/            # T5 model implementation
│   │   ├── xgboost/       # XGBoost model implementation
│   │   └── roberta/       # RoBERTa model implementation
│   ├── notebooks/         # Jupyter notebooks for experiments
│   └── utils/             # Utility functions and helpers
├── data/                  # Data directory
├── cache/                 # Cache directory for GloVe embeddings
└── .venv/                 # Python virtual environment
```

## Models

The repository contains several model implementations:

### SVM Model

- Location: `src/models/svm/`
- A traditional machine learning approach using Support Vector Machines
- Includes feature extraction and model training utilities

### DeBERTa Models

- Location: `src/models/deberta/`
- Implementation of the DeBERTa transformer model
- Includes both standard and EFL (Evidence-Focused Learning) variants
- Notebooks for training and evaluation in `src/notebooks/deberta/`

### Other Models

- T5: `src/models/T5/`
- XGBoost: `src/models/xgboost/`
- RoBERTa: `src/models/roberta/`

## Getting Started

1. **Environment Setup**

   You can use either Python's built-in venv or UV package manager:

   **Option 1: Using Python venv**

   ```bash
   # Create and activate virtual environment
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

   # Install dependencies
   pip install -r requirements.txt
   ```

   **Option 2: Using UV (Recommended)**

   ```bash
   # Install UV if not already installed
   pip install uv

   # Create virtual environment and install dependencies
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -r requirements.txt
   ```

2. **Data Preparation**

   - Place your data files in the `data/` directory
   - Use the notebooks in `src/notebooks/` for data preprocessing

3. **Running Models**
   - For SVM: Use the scripts in `src/models/svm/`
     ```bash
     python -m src.models.svm.main
     or
     uv run python -m src.models.svm
     ```
   - For DeBERTa: Use the notebook in `src/notebooks/deberta/`
   - For data augmentation:
     ```bash
     python -m src.augmentation.pipeline.main
     or
     uv run python -m src.augmentation.pipeline
     ```
   - Each model directory contains specific instructions for training and evaluation

## Additional Information

- The project uses Python 3.8+ and PyTorch for deep learning models
- Data augmentation utilities are available in `src/augmentation/`
- Configuration settings can be found in `src/config.py`
- Cache directory is used for storing GloVe embeddings, speeding up training of the SVM model
