import pandas as pd # Much faster than pandas 
import numpy as np
import psutil
import time
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

def prepare_data(train_df, aug_train_df, dev_df) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Prepare data for XGBoost training."""
    # Combine claim and evidence into a single text feature for TF-IDF
    train_df['text'] = train_df['Claim'] + " [SEP] " + train_df['Evidence']
    aug_train_df['text'] = aug_train_df['Claim'] + " [SEP] " + aug_train_df['Evidence']
    dev_df['text'] = dev_df['Claim'] + " [SEP] " + dev_df['Evidence']
    
    # Extract labels
    train_labels = train_df['label'].values
    aug_train_labels = aug_train_df['label'].values
    dev_labels = dev_df['label'].values
    
    # Combine the augmented training data with the original training data
    train_df = pd.concat([train_df, aug_train_df])
    train_labels = np.concatenate([train_labels, aug_train_labels])
    
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
