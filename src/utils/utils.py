import pandas as pd
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
    matthews_corrcoef,
    confusion_matrix
)
from matplotlib import pyplot as plt
import torch
from collections import Counter
from typing import Optional, Tuple, Dict, Set

def get_device() -> torch.device:
    """
    Determine the appropriate device for PyTorch computations.
    
    Returns:
        torch.device: The available computation device in order of preference:
                     CUDA GPU, Apple MPS, or CPU.
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def prepare_svm_data(data: pd.DataFrame, 
                    remove_stopwords: bool = True, 
                    lemmatize: bool = True, 
                    min_freq: int = 2, 
                    vocab_size: Optional[int] = None) -> Tuple[pd.DataFrame, np.ndarray, Set[str]]:
    """
    Prepare text data for SVM training by cleaning, normalizing and vocabulary management.
    
    Args:
        data: DataFrame containing 'Claim' and 'Evidence' columns
        remove_stopwords: Whether to remove common stopwords
        lemmatize: Whether to apply lemmatization
        min_freq: Minimum frequency for words to be included in vocabulary
        vocab_size: Maximum vocabulary size (most frequent words kept)
    
    Returns:
        Tuple containing:
            - Processed DataFrame with added 'text' column
            - NumPy array of labels
            - Set of vocabulary words
    """
    translator = str.maketrans('', '', string.punctuation)

    def clean_text(text: str) -> str:
        """
        Clean and normalize text by lowercasing, removing punctuation,
        and optionally removing stopwords and lemmatizing.
        """
        text = text.lower().translate(translator)
        # Normalize whitespace
        text = " ".join(text.split())
        
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
            
        if lemmatize:
            try:
                lemmatizer = WordNetLemmatizer()
                words = text.split()
                text = " ".join([lemmatizer.lemmatize(word) for word in words])
            except Exception:
                pass
        return text

    # Build vocabulary from training data
    train_samples = pd.concat([data['Claim'], data['Evidence']]).apply(clean_text)
    all_words = [word for text in train_samples for word in text.split()]
    word_counts = Counter(all_words)

    # Filter words by minimum frequency and sort by frequency
    filtered_words = [(word, count) for word, count in word_counts.items() if count >= min_freq]
    sorted_words = sorted(filtered_words, key=lambda x: (-x[1], x[0]))
    
    # Apply vocabulary size limit if specified
    if vocab_size is not None:
        sorted_words = sorted_words[:vocab_size]
    
    vocab = {word for word, _ in sorted_words}

    def replace_rare_words(text: str) -> str:
        """Replace words not in vocabulary with <UNK> token."""
        return ' '.join([word if word in vocab else '<UNK>' for word in text.split()])

    # Process the data with UNK replacement
    data['text'] = ("Claim: " + data['Claim'].apply(clean_text).apply(replace_rare_words) + 
                    " [SEP] " + "Evidence: " + data['Evidence'].apply(clean_text).apply(replace_rare_words))

    # Extract labels
    labels = data['label'].values

    return data, labels, vocab

def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics for classification.
    
    Args:
        y_true: Array of true labels
        y_pred: Array of predicted labels
    
    Returns:
        Dictionary containing accuracy, precision, recall, F1-score, and MCC metrics
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

def plot_confusion_matrix(y_true, y_pred, save_path):
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save the confusion matrix plot
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    classes = ['Negative', 'Positive']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, f'{cm[i, j]}\n({cm_norm[i, j]:.2f})',
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_path)
    plt.close()

def get_memory_usage() -> float:
    """
    Get current memory usage of the process.
    
    Returns:
        Memory usage in megabytes (MB)
    """
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)

@contextmanager
def timer(name: str, logger):
    """
    Context manager for timing code execution.
    
    Args:
        name: Descriptive name for the operation being timed
        logger: Logger object to output timing information
    
    Example:
        with timer("Data processing", logger):
            process_data()
    """
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        logger.info(f"{name} completed in {end_time - start_time:.2f} seconds")
