# COMP34812 - Natural Language Understanding - Coursework

This repository contains code and models for the COMP34812 NLU coursework, focusing on evidence detection. The project implements various machine learning approaches including traditional approaches like SVM and XGBoost, and other transformer-based models like T5, RoBERTa and DeBERTa.

## Repository Structure

```
.
├── src/                   # Source code directory
│   ├── augmentation/      # Data augmentation utilities
│   ├── experiments/       # Experimental model implementations
│   │   ├── xgboost/       # XGBoost experiments
│   │   ├── roberta/       # RoBERTa experiments
│   │   ├── deberta efl/   # DeBERTa with EFL experiments
│   │   └── T5/            # T5 experiments
│   ├── models/            # Production model implementations
│   │   ├── svm/           # SVM model implementation
│   │   └── deberta/       # DeBERTa model implementation
│   ├── notebooks/         # Jupyter notebooks for experiments
│   ├── utils/             # Utility functions and helpers
│   ├── config.py          # Configuration settings
│   └── README.md          # Source code documentation
├── data/                  # Data directory
├── cache/                 # Cache directory for GloVe embeddings
└── .venv/                 # Python virtual environment
```

## Where can I find the code for the training and demonstration?

The code can be found in both the src/models folder, or you can view the notebooks in the src/notebooks folder.

## Trained models

You can find the trained versions of the models [here](https://drive.google.com/drive/folders/1iPO9eOqhOcxakccri3-Thjm0wH09LKT4?usp=sharing).

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

- DeBERTa with EFL: `src/experiments/deberta efl`
- T5: `src/experiments/T5/`
- XGBoost: `src/experiments/xgboost/`
- RoBERTa: `src/experiments/roberta/`

## Getting Started

1. **Environment Setup**

   **Using UV**

   ```bash
   # Install UV if not already installed
   pip install uv
   # If your having troubles:
   ## Mac / Linux:
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ## Windows:
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"



   # Create virtual environment and install dependencies
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -r requirements.txt
   ```

2. **Data Preparation**

   - Place your data files in the `data/` directory

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
