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
from plotly.io import show
import pickle
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA

from src.config import config
from src.utils.GloveVectorizer import GloveVectorizer
from src.utils.FeatureExtractor import FeatureExtractor
from src.utils.LoggingPipeline import LoggingPipeline
from src.utils.utils import (
    calculate_all_metrics,
    get_device,
    get_memory_usage,
    prepare_svm_data,
    timer
)

# Configure logging
logger = logging.getLogger(__name__)

# Constants
NUM_TRIALS = 100
# VOCAB_SIZE = 1000 # No longer fixed, tuned by Optuna
TRAIN_SUBSET_FRACTION = 0.4 # Use 30% of training data for faster iteration (set to 1.0 for full training)

# Initialize memory tracking
initial_memory = get_memory_usage()
logger.info(f"Initial memory usage: {initial_memory:.2f} MB")

# Load data
train_df_raw = pd.read_csv(config.AUG_TRAIN_FILE)
dev_df_raw = pd.read_csv(config.DEV_FILE)

# Create a subset of the training data if needed
if TRAIN_SUBSET_FRACTION < 1.0:
    logger.info(f"Using {TRAIN_SUBSET_FRACTION*100:.0f}% of the training data ({int(len(train_df_raw) * TRAIN_SUBSET_FRACTION)} samples).")
    # Use stratified sampling to maintain label distribution
    train_df_subset, _ = train_test_split(
        train_df_raw, 
        train_size=TRAIN_SUBSET_FRACTION, 
        stratify=train_df_raw['label'], # Stratify based on the label column
        random_state=42
    )
    logger.info(f"Stratified subset created. Label distribution:\n{train_df_subset['label'].value_counts(normalize=True)}")
else:
    logger.info("Using 100% of the training data.")
    train_df_subset = train_df_raw

# NOTE: Feature pre-computation removed. Features are generated within each trial.

# --- Optuna Objective --- #
def objective(trial: optuna.Trial) -> float:
    """
    Optuna objective function - generates features within the trial.
    """
    # Access subsetted train/dev dataframes defined outside
    global train_df_subset, dev_df_raw
    trial_number = trial.number
    
    logger.info(f"\n{'='*50}\nStarting trial {trial_number}/{NUM_TRIALS}\n{'='*50}")
    trial_start = time.time()
    
    # Simplified hyperparameters for RBF Kernel SVM
    params = {
        "C": trial.suggest_float("C", 1.5, 2.5), # Focused range around C=2
        "vocab_size": trial.suggest_int("vocab_size", 10000, 20000, step=500), # Tune vocab size,
        "embedding_dim": trial.suggest_categorical ("embedding_dim", [100, 200, 300]), # Tune embedding dimension
        "pca_components": trial.suggest_int("pca_components", 100, 500, step=10), # Tune PCA components
        "ngram_range": trial.suggest_int("ngram_range", 1, 3), # Tune ngram range
        "min_df": trial.suggest_int("min_df", 1, 5),
        "max_df": trial.suggest_float("max_df", 0.85, 1.0, step=0.05),
    }
        
    # --- Data preparation for this trial --- #
    with timer(f"Trial {trial_number} Data Prep", logger):
        # Prepare train data for this trial using suggested vocab_size
        train_df_trial, train_labels_trial, trial_vocab = prepare_svm_data(
            train_df_subset.copy(), 
            remove_stopwords=True, lemmatize=True, min_freq=2, 
            vocab_size=params['vocab_size']
        )
        # Prepare dev data using the same vocab size
        dev_df_trial, dev_labels_trial, _ = prepare_svm_data(
            dev_df_raw.copy(), 
            remove_stopwords=True, lemmatize=True, min_freq=2, 
            vocab_size=params['vocab_size']
        ) 
        train_texts_trial = train_df_trial['text'].tolist()
        dev_texts_trial = dev_df_trial['text'].tolist()
        logger.info(f"  Trial vocab size: {len(trial_vocab)}")
    
    # Create the full pipeline for this trial
    pipeline = create_pipeline_from_params(params, trial_vocab)
    
    with timer(f"Trial {trial_number} training", logger):
        pipeline.fit(train_texts_trial, train_labels_trial)
    
    with timer(f"Trial {trial_number} evaluation", logger):
        dev_preds = pipeline.predict(dev_texts_trial)
        metrics = calculate_all_metrics(dev_labels_trial, dev_preds)
    
    # Save trial results (as before)
    svm_dir = config.SAVE_DIR / "svm"
    svm_dir.mkdir(parents=True, exist_ok=True)
    with (svm_dir / f'svm_{trial_number}.json').open('w') as f:
        # Convert numpy types for JSON serialization if necessary
        serializable_params = {k: (float(v) if isinstance(v, np.floating) else v) for k, v in params.items()}
        serializable_metrics = {k: (float(v) if isinstance(v, np.floating) else v) for k, v in metrics.items()}
        json.dump({**serializable_metrics, **serializable_params}, f)
    
    trial_duration = time.time() - trial_start
    logger.info(f"Trial {trial_number} completed in {trial_duration:.2f} seconds")
    logger.info(f"Trial {trial_number} results: W Macro-F1 = {metrics['W Macro-F1']:.4f}")
    
    # Explicit garbage collection might help Optuna with memory over many trials
    gc.collect()
    return metrics["W Macro-F1"]

# --- Pipeline Creation (Simplified for Trial) --- #
def create_pipeline_from_params(params: Dict, vocabulary: List[str]) -> Pipeline:
    """
    Create the full pipeline for a given trial's parameters.
    Includes GloveVectorizer -> optional Scaler -> SVM.
    """
    pipeline_steps = []
    
    # 1. GloveVectorizer
    pipeline_steps.append(('glove_feature_union', FeatureUnion([
        ('glove', GloveVectorizer(
            use_tfidf_weighting=True,
            vocabulary=vocabulary,
            embedding_dim=params['embedding_dim'],
            ngram_range=(1, params['ngram_range']),
            min_df=params['min_df'],
            max_df=params['max_df']
        )),
        ('feature_extractor', FeatureExtractor())
    ])))
    
    # RBF kernel: scaling, PCA for dimensionality reduction, then SVC
    pipeline_steps.append(('scaler', StandardScaler()))
    pipeline_steps.append(('pca', PCA(n_components=params['pca_components'])))
    pipeline_steps.append(('svm', SVC(
        C=params['C'],
        kernel='rbf',
        gamma='scale',
        probability=False,
        random_state=42
    )))
    
    return Pipeline(pipeline_steps)

def hyperparameter_tuning(show_plots: bool = False) -> None:
    """
    Perform hyperparameter tuning using Optuna.
    """
    logger.info(f"Running {NUM_TRIALS} hyperparameter optimization trials...")
    
    device = get_device()
    logger.info(f"Using device: {device} (Note: scikit-learn SVM implementation will utilize CPU)")
    
    # Configure Optuna study (as before)
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
        study_name='svm_evidence_detection_streamlined'
    )
    
    try:
        with timer("Hyperparameter optimization (on precomputed features)", logger):
            # n_jobs=1 because SVM fitting is CPU-bound and may not benefit from parallelism here
            # depending on the system. Can try increasing if CPU has many cores.
            study.optimize(objective, n_trials=NUM_TRIALS, n_jobs=8)
    except KeyboardInterrupt:
        logger.warning("Hyperparameter tuning interrupted by user.")
    
    # Log best trial results (as before)
    if not study.trials:
        logger.error("No trials completed. Exiting.")
        return
        
    trial = study.best_trial
    logger.info("\nBest trial:")
    logger.info(f"  Value (W Macro-F1): {trial.value:.4f}")
    logger.info("  Params:")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")

    # --- Plotting Optuna Trials --- //
    if show_plots:
        try:
            logger.info("Generating Optuna trial plots...")
            history_fig = optuna.visualization.plot_optimization_history(study)
            show(history_fig)
            importance_fig = optuna.visualization.plot_param_importances(study)
            show(importance_fig)
            heatmap_fig = optuna.visualization.plot_contour(study)
            show(heatmap_fig)
        except Exception as e:
            logger.error(f"Failed to generate plots: {e}")

    return trial.params

# --- Main Execution --- #
def main() -> None:
    global train_df_raw, dev_df_raw
    """Main function to run the SVM model training and optimization process."""
    logger.info("\n" + "="*70)
    logger.info("EVIDENCE DETECTION SVM MODEL TRAINING (Streamlined)")
    logger.info("="*70)
    
    params = hyperparameter_tuning(show_plots=True)

    train_df_raw, train_labels, best_vocab = prepare_svm_data(train_df_raw, remove_stopwords=True, lemmatize=True, min_freq=2, vocab_size=params['vocab_size'])
    dev_df_raw, dev_labels, _ = prepare_svm_data(dev_df_raw, remove_stopwords=True, lemmatize=True, min_freq=2, vocab_size=params['vocab_size'])


     # --- Train and Save Final Model --- //
    logger.info("\n" + "="*70)
    logger.info("TRAINING FINAL FULL MODEL")
    logger.info("="*70)

    pipeline = create_pipeline_from_params(params, best_vocab)
    pipeline.fit(train_df_raw['text'], train_labels)
    dev_preds = pipeline.predict(dev_df_raw['text'])
    metrics = calculate_all_metrics(dev_labels, dev_preds)
    logger.info(f"Final model evaluation: {metrics}")

    pipeline_pickle_path = config.SAVE_DIR / "svm" / "svm_pipeline.pkl"
    try:
        with open(pipeline_pickle_path, "wb") as f:
            pickle.dump(pipeline, f)
        logger.info(f"Pipeline successfully saved to {pipeline_pickle_path}")
    except Exception as e:
        logger.error(f"Error saving pipeline: {e}")



if __name__ == "__main__":
    main()
