# Utility Functions Documentation

This directory contains utility functions and classes that support the machine learning pipeline. Below is a detailed breakdown of each component.

## Core Utilities (`utils.py`)

Core utility functions used across the project:

- `get_device()`: Determines the best available device (CUDA, MPS, or CPU) for PyTorch operations
- `prepare_svm_data()`: Prepares data for SVM training by cleaning and formatting text
- `calculate_all_metrics()`: Computes comprehensive evaluation metrics (accuracy, precision, recall, F1, MCC)
- `get_memory_usage()`: Monitors memory usage during operations
- `timer()`: Context manager for timing code execution

## Word Embedding Vectorizer (`GloveVectorizer.py`)

Custom vectorizer that combines GloVe word embeddings with positional encoding:

- Caches GloVe embeddings for faster loading
- Supports TF-IDF weighting of word vectors
- Implements positional encoding for sequence information
- Computes interaction features between claim and evidence

Key methods:

- `fit()`: Prepares TF-IDF weights
- `transform()`: Converts text to feature vectors
- `_get_weighted_vector()`: Computes weighted word embeddings
- `_get_positional_encoding()`: Generates sinusoidal positional encodings

## Feature Extraction (`FeatureExtractor.py`)

Comprehensive feature extraction for evidence detection:

- Basic text statistics (lengths, word counts)
- Sentiment analysis using VADER
- Text characteristics (capitalization, punctuation)
- TF-IDF based similarity between claim and evidence

Key methods:

- `fit()`: Prepares TF-IDF vectorizer
- `transform()`: Computes all features for input texts

## Text Preprocessing (`TextPreprocessor.py`)

Simple text preprocessing class that:

- Converts text to lowercase
- Removes special characters
- Lemmatizes words

Implements scikit-learn interface with:

- `fit()`
- `transform()`
- `fit_transform()`

## Enhanced Pipeline (`LoggingPipeline.py`)

Extends scikit-learn's Pipeline with detailed logging:

- Logs progress of each pipeline step
- Tracks memory usage
- Records execution times
- Special handling for FeatureUnion components

Key methods:

- `fit()`: Logs fitting process
- `predict()`: Logs prediction process

## Data Processing (`extract_positive_examples.py`)

Utility script for data processing:

- Extracts positive examples from training data
- Filters for positive examples (label == 1)
- Saves to new CSV file
- Prints statistics about the data

## Usage Example

```python
from utils import get_device, calculate_all_metrics
from utils.GloveVectorizer import GloveVectorizer
from utils.FeatureExtractor import FeatureExtractor
from utils.TextPreprocessor import TextPreprocessor
from utils.LoggingPipeline import LoggingPipeline

# Initialize components
device = get_device()
vectorizer = GloveVectorizer()
feature_extractor = FeatureExtractor()
preprocessor = TextPreprocessor()

# Create pipeline with logging
pipeline = LoggingPipeline([
    ('preprocess', preprocessor),
    ('vectorize', vectorizer),
    ('features', feature_extractor),
    ('classifier', YourClassifier())
], logger=logging.getLogger())

# Fit and predict
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# Calculate metrics
metrics = calculate_all_metrics(y_test, y_pred)
```

## Best Practices

1. **Memory Management**

   - Use `get_memory_usage()` to monitor memory consumption
   - Use `timer()` context manager for performance profiling

2. **Feature Extraction**

   - Use `FeatureExtractor` for comprehensive feature engineering
   - Combine with `GloveVectorizer` for word embeddings
   - Apply `TextPreprocessor` for consistent text cleaning

3. **Pipeline Usage**

   - Use `LoggingPipeline` for detailed progress tracking
   - Monitor memory usage and execution times
   - Check logs for potential bottlenecks

4. **Evaluation**
   - Use `calculate_all_metrics()` for consistent evaluation
   - Compare different models using the same metrics

## Dependencies

- Python 3.8+
- PyTorch
- scikit-learn
- NLTK
- Gensim
- pandas
- numpy
