# Model Card for SVM Evidence Detection Model

## Model Details

### Model Description

- **Developers**: Harvey Dennis and William Asbery
- **Model Type**: Supervised Learning
- **Model Architecture**: Support Vector Machine (SVM)
- **Language**: English
- **License**: cc-by-4.0
- **Repository**: https://github.com/willasbery/comp34812

### Model Summary

This is a classification model that was trained to detect whether the evidence provided supports the claim. The model uses a combination of GloVe embeddings and custom features to make predictions.

### Model Architecture

The model consists of a pipeline that includes:

1. Feature extraction using GloVe embeddings with optional TF-IDF weighting
2. Custom feature extraction
3. Standard scaling
4. Support Vector Machine classifier with RBF kernel

## Training Details

### Training Data

- 29K pairs of texts drawn from emails, news articles and blog posts
- 21.5K original samples
- 6.5K augmented samples
- Data includes both claim and evidence text pairs

### Hyperparameters

The model was optimized using Optuna with the following key hyperparameters:

- C (regularization parameter): Optimized between 0.01 and 100.0
- Kernel: RBF
- Gamma: 'auto'
- Class weight: Optimized between 'balanced' and None
- TF-IDF weighting: Enabled
- Feature selection: Disabled (found to be not beneficial)

### Training Process

- Number of optimization trials: 50
- Optimization method: Tree-structured Parzen Estimator (TPE)
- Pruning strategy: Median Pruner
- Training time: Varies based on hyperparameters
- Hardware: CPU-based training (scikit-learn implementation)

## Evaluation

### Testing Data

- Development dataset: 6K pairs
- Same distribution as training data
- Includes both original and augmented samples

### Testing Metrics

- Weighted Precision
- Weighted Recall
- Weighted Macro-F1
- Accuracy
- Matthews Correlation Coefficient (MCC)

### Results

The model's performance is evaluated using weighted metrics to account for class imbalance. The best performing configuration is selected based on the weighted Macro-F1 score.

## Technical Specifications

### Hardware Requirements

- RAM: At least 16GB recommended
- Storage: Minimal (model size is relatively small)
- CPU: Standard CPU sufficient (no GPU required)

### Software Requirements

- Python 3.x
- scikit-learn
- Optuna
- pandas
- numpy
- Standard Python scientific computing stack

## Limitations and Bias

### Known Limitations

- Performance may vary based on the domain of the input text
- Limited to English language text
- May not perform well on very short or very long text pairs
- Relies on pre-trained GloVe embeddings which may not capture domain-specific terminology

### Bias and Risks

- Model performance may be affected by biases present in the training data
- May not generalize well to domains not represented in the training data
- Performance may be affected by the quality of the text preprocessing

## Additional Information

### Model Optimization

The hyperparameters were determined through Bayesian optimization using Optuna with a TPE sampler. The optimization process considered various combinations of SVM parameters and feature extraction methods to find the optimal configuration.

### Feature Engineering

The model uses a combination of:

- GloVe embeddings with optional TF-IDF weighting
- Custom features extracted from the text
- Standard scaling for feature normalization

### Model Persistence

The final model is saved as a pickle file containing the complete pipeline, including all preprocessing steps and the trained SVM classifier.

## Usage Instructions

### Running the Model

To run the SVM model, you can use the `uv` command-line tool, which helps manage Python environments and dependencies:

```bash
uv run python -m src.models.svm.main
```

For data augmentation, you can use:

```bash
uv run python -m src.augmentation.pipeline
```

These commands ensure that all dependencies are properly managed and the correct Python environment is used. The SVM model can be found in the `src/models/svm/` directory, which contains all scripts necessary for training and evaluation.

### Inference

Once trained, the model can be loaded and used for inference:

```python
import pickle

# Load the model
with open('path/to/final_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Make predictions
predictions = model.predict(texts)
```
