import pandas as pd # Much faster than pandas 
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    matthews_corrcoef
)

def prepare_data(train_df, aug_train_df, dev_df):
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