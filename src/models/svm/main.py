"""
SVM-based evidence detection model with hyperparameter optimization using Optuna.

This module implements a Support Vector Machine classifier for evidence detection,
utilizing GloVe word embeddings and feature extraction methods. The model is optimized
through Bayesian hyperparameter tuning with Optuna.
"""

import gc
import json
import logging
import time
import pickle
from typing import Dict, List, Tuple, Union, Optional

import numpy as np
import optuna
import pandas as pd
from plotly.io import show
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA

from src.config import config
from src.utils.GloveVectorizer import GloveVectorizer
from src.utils.FeatureExtractor import FeatureExtractor
from src.utils.utils import (
    calculate_all_metrics,
    get_device,
    get_memory_usage,
    prepare_svm_data,
    timer
)

# Configure logging
logger = logging.getLogger(__name__)

# Configuration Constants
NUM_TRIALS = 100
TRAIN_SUBSET_FRACTION = 0.4  # Use 40% of training data for faster iteration

# Load and prepare initial data
initial_memory = get_memory_usage()
logger.info(f"Initial memory usage: {initial_memory:.2f} MB")

train_df_raw = pd.read_csv(config.AUG_TRAIN_FILE)
dev_df_raw = pd.read_csv(config.DEV_FILE)

# Create training subset for faster iteration if needed
if TRAIN_SUBSET_FRACTION < 1.0:
    train_samples = int(len(train_df_raw) * TRAIN_SUBSET_FRACTION)
    logger.info(f"Using {TRAIN_SUBSET_FRACTION*100:.0f}% of training data ({train_samples} samples)")
    
    train_df_subset, _ = train_test_split(
        train_df_raw, 
        train_size=TRAIN_SUBSET_FRACTION, 
        stratify=train_df_raw['label'],
        random_state=42
    )
    logger.info(f"Stratified subset created. Label distribution:\n{train_df_subset['label'].value_counts(normalize=True)}")
else:
    logger.info("Using 100% of the training data.")
    train_df_subset = train_df_raw


def objective(trial: optuna.Trial) -> float:
    """
    Optuna objective function for hyperparameter optimization.
    
    Generates features, trains SVM model, and evaluates performance for each trial.
    
    Args:
        trial: Current Optuna trial object
        
    Returns:
        float: Weighted Macro F1 score on development set
    """
    global train_df_subset, dev_df_raw
    trial_number = trial.number
    
    logger.info(f"\n{'='*50}\nStarting trial {trial_number}/{NUM_TRIALS}\n{'='*50}")
    trial_start = time.time()
    
    # Define hyperparameter search space
    params = {
        "C": trial.suggest_float("C", 1.5, 2.5), # Focused range around C=2
        "vocab_size": trial.suggest_int("vocab_size", 10000, 20000, step=500), # Tune vocab size,
        "embedding_dim": trial.suggest_categorical ("embedding_dim", [100, 200, 300]), # Tune embedding dimension
        "pca_components": trial.suggest_int("pca_components", 400, 600, step=10), # Tune PCA components
        # "min_df": trial.suggest_int("min_df", 1, 2),
        # "max_df": trial.suggest_float("max_df", 0.85, 1.0, step=0.05),
    }
        
    # Data preparation for this trial
    with timer(f"Trial {trial_number} Data Prep", logger):
        train_df_trial, train_labels_trial, trial_vocab = prepare_svm_data(
            train_df_subset.copy(), 
            remove_stopwords=True, 
            lemmatize=True, 
            min_freq=2, 
            vocab_size=params['vocab_size']
        )
        
        dev_df_trial, dev_labels_trial, _ = prepare_svm_data(
            dev_df_raw.copy(), 
            remove_stopwords=True, 
            lemmatize=True, 
            min_freq=2, 
            vocab_size=params['vocab_size']
        ) 
        
        train_texts_trial = train_df_trial['text'].tolist()
        dev_texts_trial = dev_df_trial['text'].tolist()
        logger.info(f"  Trial vocab size: {len(trial_vocab)}")
    
    # Create pipeline and train model
    pipeline = create_pipeline_from_params(params, trial_vocab)
    
    with timer(f"Trial {trial_number} training", logger):
        pipeline.fit(train_texts_trial, train_labels_trial)
    
    # Evaluate model
    with timer(f"Trial {trial_number} evaluation", logger):
        dev_preds = pipeline.predict(dev_texts_trial)
        metrics = calculate_all_metrics(dev_labels_trial, dev_preds)
    
    # Save trial results
    svm_dir = config.SAVE_DIR / "svm"
    svm_dir.mkdir(parents=True, exist_ok=True)
    
    with (svm_dir / f'svm_{trial_number}.json').open('w') as f:
        serializable_params = {k: (float(v) if isinstance(v, np.floating) else v) for k, v in params.items()}
        serializable_metrics = {k: (float(v) if isinstance(v, np.floating) else v) for k, v in metrics.items()}
        json.dump({**serializable_metrics, **serializable_params}, f)
    
    trial_duration = time.time() - trial_start
    logger.info(f"Trial {trial_number} completed in {trial_duration:.2f} seconds")
    logger.info(f"Trial {trial_number} results: W Macro-F1 = {metrics['W Macro-F1']:.4f}")
    
    # Free memory
    gc.collect()
    return metrics["W Macro-F1"]


def create_pipeline_from_params(params: Dict, vocabulary: List[str]) -> Pipeline:
    """
    Create scikit-learn pipeline for SVM classification.
    
    Builds a pipeline with feature extraction, scaling, dimensionality reduction,
    and SVM classification components based on specified hyperparameters.
    
    Args:
        params: Dictionary of hyperparameters
        vocabulary: List of vocabulary terms to use in vectorization
        
    Returns:
        Pipeline: Scikit-learn pipeline for text classification
    """
    pipeline_steps = []
    
    # Feature extraction component
    pipeline_steps.append(('glove_feature_union', FeatureUnion([
        ('glove', GloveVectorizer(
            use_tfidf_weighting=True,
            vocabulary=vocabulary,
            embedding_dim=params['embedding_dim'],
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )),
        ('feature_extractor', FeatureExtractor())
    ])))
    
    # Feature scaling and dimensionality reduction
    pipeline_steps.append(('scaler', StandardScaler()))
    pipeline_steps.append(('pca', PCA(n_components=params['pca_components'])))
    
    # SVM classifier with RBF kernel
    pipeline_steps.append(('svm', SVC(
        C=params['C'],
        kernel='rbf',
        gamma='scale',
        probability=False,
        random_state=42
    )))
    
    return Pipeline(pipeline_steps)


def hyperparameter_tuning(show_plots: bool = False) -> Dict:
    """
    Perform hyperparameter tuning using Optuna.
    
    Configures and runs an Optuna study to optimize hyperparameters for the SVM model.
    
    Args:
        show_plots: Whether to display optimization visualizations
        
    Returns:
        Dict: Best hyperparameters found during optimization
    """
    logger.info(f"Running {NUM_TRIALS} hyperparameter optimization trials...")
    
    # Configure Optuna sampler and pruner
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
        with timer("Hyperparameter optimization", logger):
            study.optimize(objective, n_trials=NUM_TRIALS, n_jobs=8)
    except KeyboardInterrupt:
        logger.warning("Hyperparameter tuning interrupted by user.")
    
    # Log best trial results
    if not study.trials:
        logger.error("No trials completed. Exiting.")
        return {}
        
    trial = study.best_trial
    logger.info("\nBest trial:")
    logger.info(f"  Value (W Macro-F1): {trial.value:.4f}")
    logger.info("  Params:")
    
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")

    # Generate and display plots
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


def main() -> None:
    """
    Main execution function for SVM model training and optimization.
    
    Performs hyperparameter tuning, trains the final model with optimal 
    parameters on full dataset, and saves the model.
    """
    global train_df_raw, dev_df_raw
    
    logger.info("\n" + "="*70)
    logger.info("EVIDENCE DETECTION SVM MODEL TRAINING")
    logger.info("="*70)
    
    # Find optimal hyperparameters
    params = hyperparameter_tuning(show_plots=True)
    
    # Process data with optimal parameters
    train_df_processed, train_labels, best_vocab = prepare_svm_data(
        train_df_raw, 
        remove_stopwords=True, 
        lemmatize=True, 
        min_freq=2, 
        vocab_size=params['vocab_size']
    )
    
    dev_df_processed, dev_labels, _ = prepare_svm_data(
        dev_df_raw, 
        remove_stopwords=True, 
        lemmatize=True, 
        min_freq=2, 
        vocab_size=params['vocab_size']
    )

    # Train and evaluate final model
    logger.info("\n" + "="*70)
    logger.info("TRAINING FINAL FULL MODEL")
    logger.info("="*70)

    pipeline = create_pipeline_from_params(params, best_vocab)
    pipeline.fit(train_df_processed['text'], train_labels)
    dev_preds = pipeline.predict(dev_df_processed['text'])
    metrics = calculate_all_metrics(dev_labels, dev_preds)
    logger.info(f"Final model evaluation: {metrics}")

    # Save the trained model
    pipeline_pickle_path = config.SAVE_DIR / "svm" / "svm_pipeline.pkl"
    try:
        with open(pipeline_pickle_path, "wb") as f:
            pickle.dump(pipeline, f)
        logger.info(f"Pipeline successfully saved to {pipeline_pickle_path}")
    except Exception as e:
        logger.error(f"Error saving pipeline: {e}")


if __name__ == "__main__":
    main()
