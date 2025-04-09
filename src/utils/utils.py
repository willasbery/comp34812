import pandas as pd # Much faster than pandas 
import numpy as np
import psutil
import time
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from contextlib import contextmanager
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    matthews_corrcoef
)
import torch
from collections import Counter
from typing import Optional

def get_device() -> torch.device:
    """Determine the device to use for computations."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def prepare_svm_data(train_df, dev_df, remove_stopwords: bool = True, lemmatize: bool = True, 
                    min_freq: int = 2, vocab_size: Optional[int] = None) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Prepare data for SVM training with UNK replacement for rare words."""
    translator = str.maketrans('', '', string.punctuation)

    def clean_text(text: str) -> str:
        text = text.lower().translate(translator)
        # Normalize whitespace
        text = " ".join(text.split())
        # Optionally remove stopwords
        if remove_stopwords:
            try:
                # Keep important discourse markers and modal verbs
                keep_words = {
                    'because', 'since', 'therefore', 'hence', 'thus', 'although',
                    'however', 'but', 'not', 'should', 'must', 'might', 'may',
                    'could', 'would', 'against', 'between', 'before', 'after'
                }
                custom_stopwords = set(stopwords.words("english")) - keep_words
                
                text = " ".join([word for word in text.split() 
                               if word not in custom_stopwords])
            except Exception:
                pass
        # Optionally perform lemmatization
        if lemmatize:
            try:
                lemmatizer = WordNetLemmatizer()
                words = text.split()
                text = " ".join([lemmatizer.lemmatize(word) for word in words])
            except Exception:
                pass
        return text

    # First pass to build vocabulary from training data
    train_samples = pd.concat([train_df['Claim'], train_df['Evidence']]).apply(clean_text)
    all_words = [word for text in train_samples for word in text.split()]
    word_counts = Counter(all_words)

    # Filter by minimum frequency and sort
    filtered_words = [(word, count) for word, count in word_counts.items() if count >= min_freq]
    sorted_words = sorted(filtered_words, key=lambda x: (-x[1], x[0]))  # Sort by frequency then alphabetically
    
    # Apply vocabulary size limit
    if vocab_size is not None:
        sorted_words = sorted_words[:vocab_size]
    
    vocab = {word for word, _ in sorted_words}

    def replace_rare_words(text: str) -> str:
        return ' '.join([word if word in vocab else '<UNK>' for word in text.split()])

    # Second pass with UNK replacement
    train_df['text'] = ("Claim: " + train_df['Claim'].apply(clean_text).apply(replace_rare_words) + 
                       " [SEP] " + "Evidence: " + train_df['Evidence'].apply(clean_text).apply(replace_rare_words))
    dev_df['text'] = ("Claim: " + dev_df['Claim'].apply(clean_text).apply(replace_rare_words) + 
                     " [SEP] " + "Evidence: " + dev_df['Evidence'].apply(clean_text).apply(replace_rare_words))

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
