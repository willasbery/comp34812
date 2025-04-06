import pandas as pd # Much faster than pandas 
import numpy as np
import psutil
import time
import string
from nltk.corpus import stopwords
from contextlib import contextmanager
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    matthews_corrcoef
)
import torch

def get_device() -> torch.device:
    """Determine the device to use for computations."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def prepare_svm_data(train_df, dev_df, remove_stopwords: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Prepare data for SVM training by concatenating claim and evidence, converting text to lowercase, removing punctuation, normalizing whitespace, and optionally removing stopwords to maximize SVM performance."""
    translator = str.maketrans('', '', string.punctuation)

    def clean_text(text: str) -> str:
        text = text.lower().translate(translator)
        # Normalize whitespace
        text = " ".join(text.split())
        # Optionally remove stopwords
        if remove_stopwords:
            try:
                stopwords_set = set(stopwords.words("english"))
                text = " ".join([word for word in text.split() if word not in stopwords_set])
            except Exception:
                pass
        return text

    # Combine claim and evidence into a single text feature and clean the text
    train_df['text'] = ("Claim: " + train_df['Claim'].apply(clean_text) + " [SEP] " + "Evidence: " + train_df['Evidence'].apply(clean_text))
    dev_df['text'] = ("Claim: " + dev_df['Claim'].apply(clean_text) + " [SEP] " + "Evidence: " + dev_df['Evidence'].apply(clean_text))

    # Extract labels
    train_labels = train_df['label'].values
    dev_labels = dev_df['label'].values

    return train_df, dev_df, train_labels, dev_labels

def calculate_all_metrics(y_true, y_pred):
    """
    Calculate comprehensive evaluation metrics.
    
    Returns:
        dict: Dictionary containing all metrics
    """
    # Basic accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Calculate precision, recall, f1 (macro)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro'
    )
    
    # Calculate precision, recall, f1 (weighted)
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    
    # Matthews Correlation Coefficient
    mcc = matthews_corrcoef(y_true, y_pred)
    
    metrics = {
        'Accuracy': accuracy,
        'Macro-P': macro_precision,
        'Macro-R': macro_recall,
        'Macro-F1': macro_f1,
        'W Macro-P': weighted_precision,
        'W Macro-R': weighted_recall,
        'W Macro-F1': weighted_f1,
        'MCC': mcc
    }
    
    return metrics

# Memory monitoring
def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)

@contextmanager
def timer(name, logger):
    """Context manager for timing code execution."""
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        logger.info(f"{name} completed in {end_time - start_time:.2f} seconds")
