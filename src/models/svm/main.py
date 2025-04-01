"""
SVM-based evidence detection model with hyperparameter optimization using Optuna.
This module implements a Support Vector Machine classifier with various feature extraction
and selection methods, optimized through Bayesian hyperparameter tuning.
"""

import gc
import json
import logging
import time
from typing import Dict, List, Tuple, Union

import numpy as np
import optuna
import pandas as pd
import pickle
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, RFECV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import config
from src.utils.FeatureExtractor import FeatureExtractor
from src.utils.GloveVectorizer import GloveVectorizer
from src.utils.LoggingPipeline import LoggingPipeline
from src.utils.utils import (
    calculate_all_metrics,
    get_device,
    get_memory_usage,
    prepare_data,
    timer
)

# Configure logging
logger = logging.getLogger(__name__)

# Constants
NUM_TRIALS = 50

# Initialize memory tracking
initial_memory = get_memory_usage()
logger.info(f"Initial memory usage: {initial_memory:.2f} MB")

# Load and prepare data
train_df = pd.read_csv(config.TRAIN_FILE)
dev_df = pd.read_csv(config.DEV_FILE)
train_aug_df = pd.read_csv(config.AUG_TRAIN_FILE)

with timer("Data preparation", logger):
    train_df, dev_df, train_labels, dev_labels = prepare_data(train_df, train_aug_df, dev_df)
    logger.info(f"Prepared data: {len(train_df)} training samples, {len(dev_df)} validation samples")
    logger.info(f"Memory after data prep: {get_memory_usage():.2f} MB (+ {get_memory_usage() - initial_memory:.2f} MB)")


def objective(trial: optuna.Trial) -> float:
    """
    Optuna objective function for hyperparameter optimization.
    
    Args:
        trial: Optuna trial object containing hyperparameter suggestions
        
    Returns:
        float: Weighted Macro-F1 score for the trial
    """
    global train_df, dev_df, train_labels, dev_labels
    trial_number = trial.number
    
    logger.info(f"\n{'='*50}\nStarting trial {trial_number}/{NUM_TRIALS}\n{'='*50}")
    trial_start = time.time()
    
    # Hyperparameter suggestions
    params = {
        "C": trial.suggest_float("C", 0.01, 100.0, log=True),
        "kernel": trial.suggest_categorical("kernel", ["linear", "rbf", "poly", "sigmoid"]),
        "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]) 
                if trial.params["kernel"] in ["rbf", "poly", "sigmoid"] else "scale",
        "degree": trial.suggest_int("degree", 2, 5) 
                 if trial.params["kernel"] == "poly" else 3,
        "use_feature_selection": trial.suggest_categorical("use_feature_selection", [True, False]),
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
        "use_tfidf_weighting": trial.suggest_categorical("use_tfidf_weighting", [True, False])
    }
    
    if params["use_feature_selection"]:
        params["feature_selection_method"] = trial.suggest_categorical(
            "feature_selection_method", ["kbest", "model_based", "pca"]
        )
        if params["feature_selection_method"] == "kbest":
            params["k_best"] = trial.suggest_int("k_best", 10, 100)
        elif params["feature_selection_method"] == "pca":
            params["n_components"] = trial.suggest_float("n_components", 0.7, 0.99)
    
    logger.info(f"Trial {trial_number} hyperparameters:")
    logger.info(f"  SVM: C={params['C']}, kernel={params['kernel']}, "
               f"gamma={params['gamma']}, class_weight={params['class_weight']}")
    logger.info(f"  Feature selection: {params['use_feature_selection']} "
               f"(method={params.get('feature_selection_method', 'N/A')})")
    logger.info(f"  GloVe TF-IDF weighting: {params['use_tfidf_weighting']}")
    
    # Create and train pipeline
    pipeline = create_pipeline_from_params(params)
    
    with timer(f"Trial {trial_number} training", logger):
        pipeline.fit(train_df['text'], train_labels)
    
    with timer(f"Trial {trial_number} evaluation", logger):
        dev_preds = pipeline.predict(dev_df['text'])
        metrics = calculate_all_metrics(dev_labels, dev_preds)
    
    # Save trial results
    svm_dir = config.SAVE_DIR / "svm"
    svm_dir.mkdir(parents=True, exist_ok=True)
    with (svm_dir / f'svm_{trial_number}.json').open('w') as f:
        json.dump({**metrics, **params}, f)
    
    trial_duration = time.time() - trial_start
    logger.info(f"Trial {trial_number} completed in {trial_duration:.2f} seconds")
    logger.info(f"Trial {trial_number} results: W Macro-F1 = {metrics['W Macro-F1']:.4f}")
    
    gc.collect()
    return metrics["W Macro-F1"]


def create_pipeline_from_params(params: Dict) -> LoggingPipeline:
    """
    Create a pipeline from the given parameters.
    
    Args:
        params: Dictionary of hyperparameters
        
    Returns:
        LoggingPipeline: Configured pipeline with all components
    """
    # Feature pipeline components
    feature_pipeline = [
        ('text_features', Pipeline([
            ('glove', GloveVectorizer(use_tfidf_weighting=params['use_tfidf_weighting']))
        ])),
        ('custom_features', FeatureExtractor())
    ]
    
    # Main pipeline steps
    pipeline_steps = [
        ('features', FeatureUnion(feature_pipeline)),
        ('scaler', StandardScaler())
    ]
    
    # Add feature selection if enabled
    if params['use_feature_selection']:
        method = params['feature_selection_method']
        if method == "kbest":
            pipeline_steps.append(('feature_selection', 
                                 SelectKBest(f_classif, k=params['k_best'])))
        elif method == "model_based":
            pipeline_steps.append(('feature_selection', 
                                 SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))))
        elif method == "pca":
            pipeline_steps.append(('feature_selection', 
                                 PCA(n_components=params['n_components'])))
    
    # Add SVM classifier
    pipeline_steps.append(('svm', SVC(
        C=params['C'],
        kernel=params['kernel'],
        gamma=params['gamma'],
        degree=params['degree'],
        probability=True,
        class_weight=params['class_weight'],
        random_state=42
    )))
    
    return LoggingPipeline(pipeline_steps, logger=logger)


def main() -> None:
    """Main function to run the SVM model training and optimization process."""
    logger.info("\n" + "="*70)
    logger.info("EVIDENCE DETECTION SVM MODEL TRAINING")
    logger.info("="*70)
    logger.info(f"Running {NUM_TRIALS} hyperparameter optimization trials...")
    
    device = get_device()
    logger.info(f"Using device: {device} (Note: scikit-learn SVM implementation will utilize CPU)")
    
    # Configure Optuna study
    sampler = TPESampler(
        seed=42,
        n_startup_trials=int(NUM_TRIALS / 10),
        multivariate=True,
        constant_liar=True
    )
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=2)
    
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
        study_name='svm_evidence_detection'
    )
    
    try:
        with timer("Hyperparameter optimization", logger):
            study.optimize(objective, n_trials=NUM_TRIALS, n_jobs=5)
    except KeyboardInterrupt:
        logger.warning("Hyperparameter tuning interrupted by user.")
    
    # Log best trial results
    trial = study.best_trial
    logger.info("\nBest trial:")
    logger.info(f"  Value (W Macro-F1): {trial.value:.4f}")
    logger.info("  Params:")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")
    
    # Train final model with best parameters
    logger.info("\n" + "="*70)
    logger.info("TRAINING FINAL MODEL")
    logger.info("="*70)
    
    final_pipeline = create_pipeline_from_params(trial.params)
    combined_df = pd.concat([train_df, dev_df])
    combined_labels = np.concatenate([train_labels, dev_labels])
    
    logger.info(f"Training final model with best parameters on all data ({len(combined_df)} samples)...")
    
    with timer("Final model training", logger):
        final_pipeline.fit(combined_df['text'], combined_labels)
    
    # Save final model
    final_model_path = config.SAVE_DIR / "svm" / "final_model.pkl"
    with timer("Model saving", logger):
        with open(final_model_path, 'wb') as f:
            pickle.dump(final_pipeline, f)
    
    logger.info(f"Final model saved to {final_model_path}")
    logger.info(f"Final memory usage: {get_memory_usage():.2f} MB")
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
