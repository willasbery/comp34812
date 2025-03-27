import logging
import pandas as pd
from pathlib import Path
import re
import json
import torch

# Hyperparameter tuning
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Text processing
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# SVM
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    f1_score,
    matthews_corrcoef
)
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler

from src.utils.utils import get_device, prepare_data, calculate_all_metrics
from src.utils.TextPreprocessor import TextPreprocessor
from src.utils.FeatureExtractor import FeatureExtractor
from src.config import config

# Set up logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)


NUM_TRIALS = 10

train_df = pd.read_csv(config.TRAIN_FILE)
dev_df = pd.read_csv(config.DEV_FILE)
train_aug_df = pd.read_csv(config.AUG_TRAIN_FILE)

train_df, dev_df, train_labels, dev_labels = prepare_data(train_df, train_aug_df, dev_df)

trial_number = 0

def objective(trial):
    """Optuna objective function for hyperparameter optimization."""
    # Load data
    global trial_number, train_df, dev_df, train_labels, dev_labels
    trial_number += 1
    
    # Suggest hyperparameters
    C = trial.suggest_float("C", 0.01, 100.0, log=True)
    kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly", "sigmoid"])
    gamma = trial.suggest_categorical("gamma", ["scale", "auto"]) if kernel in ["rbf", "poly", "sigmoid"] else "scale"
    
    if kernel == "poly":
        degree = trial.suggest_int("degree", 2, 5)
    else:
        degree = 3  # Default value
    
    # TF-IDF vectorizer parameters
    max_features = trial.suggest_categorical("max_features", [5000, 10000, 15000, 20000])
    min_df = trial.suggest_categorical("min_df", [1, 2, 3, 4, 5])
    max_df = trial.suggest_categorical("max_df", [0.5, 0.6, 0.7, 0.8, 0.9])
    ngram_range_str = trial.suggest_categorical("ngram_range", ["1,1", "1,2", "1,3"])
    ngram_range = tuple(map(int, ngram_range_str.split(",")))
    
    # Create pipeline
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_features', Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=max_features,
                    min_df=min_df,
                    max_df=max_df,
                    ngram_range=ngram_range,
                    stop_words='english',
                    analyzer='word',
                    token_pattern=r'\w+',
                    sublinear_tf=True
                ))
            ])),
            ('custom_features', FeatureExtractor())
        ])),
        ('scaler', StandardScaler(with_mean=False)),  # TF-IDF matrices are sparse
        ('svm', SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            degree=degree if kernel == "poly" else 3,
            probability=True
        ))
    ])
    
    # Train model
    logging.info(f"Training SVM with hyperparameters: C={C}, kernel={kernel}, gamma={gamma}")
    pipeline.fit(train_df['text'], train_labels)
    
    # Evaluate on dev set
    dev_preds = pipeline.predict(dev_df['text'])
    metrics = calculate_all_metrics(dev_labels, dev_preds)
    
    svm_dir = config.SAVE_DIR / "svm"
    svm_dir.mkdir(parents=True, exist_ok=True)
    with (svm_dir / f'svm_{trial_number}.json').open('w') as f:
        combined_results = {**metrics, **trial.params}
        json.dump(combined_results, f)
    
    return metrics["W Macro-F1"]

def main():
    print("\nHYPERPARAMETER TUNING")
    print("=====================")
    print(f"Running {NUM_TRIALS} trials...")
    
    # Check if GPU is available for NumPy/SciPy operations
    device = get_device()
    logging.info(f"Using device: {device} (Note: scikit-learn SVM implementation will utilize CPU)")
    
    # Create a study with TPE sampler and MedianPruner
    sampler = TPESampler(seed=42)  # TPE sampler as requested
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=2)
    
    study = optuna.create_study(
        direction='maximize',  # Maximize accuracy
        sampler=sampler,
        pruner=pruner,
        study_name='svm_evidence_detection'
    )
    
    try:
        study.optimize(objective, n_trials=NUM_TRIALS, n_jobs=-1)
    except KeyboardInterrupt:
        print("Hyperparameter tuning interrupted.")
    
    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value (Accuracy): {trial.value}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")