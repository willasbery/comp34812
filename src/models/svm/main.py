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
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import config
from src.utils.GloveVectorizer import GloveVectorizer
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
NUM_TRIALS = 50
# VOCAB_SIZE = 1000 # No longer fixed, tuned by Optuna
TRAIN_SUBSET_FRACTION = 0.3 # Use 50% of training data for faster iteration (set to 1.0 for full training)

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
<<<<<<< Updated upstream
<<<<<<< Updated upstream
        "C": trial.suggest_float("C", 0.1, 10.0, log=True), # Focused range around C=2
        "kernel": trial.suggest_categorical("kernel", ["linear", "rbf"]),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]), # None worked best
        "vocab_size": trial.suggest_int("vocab_size", 500, 10000, step=500), # Tune vocab size,
        "embedding_dim": trial.suggest_int("embedding_dim", 100, 300, step=50), # Tune embedding dimension
        "use_tfidf_weighting": True # Fixed based on previous findings
=======
=======
>>>>>>> Stashed changes
        "C": trial.suggest_float("C", 0.1, 10.0, log=True),
        "kernel": trial.suggest_categorical("kernel", ["rbf"]),
        "gamma": trial.suggest_categorical("gamma", ["auto"]) 
                if trial.params["kernel"] in ["rbf", "poly", "sigmoid"] else "scale",
        "degree": trial.suggest_int("degree", 2, 5) 
                 if trial.params["kernel"] == "poly" else 3,
        "use_feature_selection": trial.suggest_categorical("use_feature_selection", [False]), # Feature selection hasn't been beneficial
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
        "use_tfidf_weighting": trial.suggest_categorical("use_tfidf_weighting", [True]) # TF-IDF weighting has been beneficial
>>>>>>> Stashed changes
    }
    # Conditionally suggest gamma only for RBF kernel
    if params["kernel"] == "rbf":
        params["gamma"] = trial.suggest_categorical("gamma", ["scale", "auto"]) 
    
    logger.info(f"Trial {trial_number} hyperparameters:")
    log_str = f"  SVM: C={params['C']}, kernel={params['kernel']}, class_weight={params['class_weight']}"
    if params["kernel"] == "rbf":
        log_str += f", gamma={params['gamma']}"
    logger.info(log_str)
    logger.info(f"  Feature Params: vocab_size={params['vocab_size']}")
    
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
    pipeline_steps.append(('glove', GloveVectorizer(
        use_tfidf_weighting=params['use_tfidf_weighting'],
        vocabulary=vocabulary,
        embedding_dim=params['embedding_dim']
    )))
    
    if params['kernel'] == 'rbf':
        # RBF kernel needs scaling
        pipeline_steps.append(('scaler', StandardScaler()))
        pipeline_steps.append(('svm', SVC(
            C=params['C'],
            kernel='rbf',
            gamma=params['gamma'],
            probability=False,
            class_weight=params['class_weight'],
            random_state=42
        )))
    elif params['kernel'] == 'linear':
        # LinearSVC doesn't typically need scaling on these features
        pipeline_steps.append(('svm', LinearSVC(
            C=params['C'],
            dual='auto', # Automatically selects based on n_samples/n_features
            class_weight=params['class_weight'],
            random_state=42,
            max_iter=3000 # Increase iterations for convergence
        )))
    else:
        raise ValueError(f"Unsupported kernel: {params['kernel']}")
    
    return Pipeline(pipeline_steps)

# --- Main Execution --- #
def main() -> None:
    """Main function to run the SVM model training and optimization process."""
    logger.info("\n" + "="*70)
    logger.info("EVIDENCE DETECTION SVM MODEL TRAINING (Streamlined)")
    logger.info("="*70)
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
<<<<<<< Updated upstream
        with timer("Hyperparameter optimization (on precomputed features)", logger):
            # n_jobs=1 because SVM fitting is CPU-bound and may not benefit from parallelism here
            # depending on the system. Can try increasing if CPU has many cores.
            study.optimize(objective, n_trials=NUM_TRIALS, n_jobs=1) 
=======
        with timer("Hyperparameter optimization", logger):
            study.optimize(objective, n_trials=NUM_TRIALS, n_jobs=5)
<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
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
    
    # --- Train and Save Final Model --- #
    logger.info("\n" + "="*70)
    logger.info("TRAINING FINAL FULL MODEL")
    logger.info("="*70)
    
    # Get best hyperparameters
    best_params = trial.params
    best_vocab_size = best_params['vocab_size'] # Get best vocab size
    
    # Prepare combined data using the BEST vocab size found
    logger.info("Combining processed training subset and full dev set for final model training...")
    # Use the original subset and dev dataframes
    train_final_df, train_final_labels, final_train_vocab = prepare_svm_data(
        train_df_subset.copy(), remove_stopwords=True, lemmatize=True, 
        min_freq=2, vocab_size=best_vocab_size
    )
    dev_final_df, dev_final_labels, _ = prepare_svm_data(
        dev_df_raw.copy(), remove_stopwords=True, lemmatize=True, 
        min_freq=2, vocab_size=best_vocab_size
    )
    
    combined_df_processed = pd.concat([train_final_df, dev_final_df])
    combined_labels = np.concatenate([train_final_labels, dev_final_labels])
    combined_texts = combined_df_processed['text'].tolist()
    # Re-calculate combined vocabulary based on the actual combined data used for the final model
    combined_vocab = set([word for text in combined_texts for word in text.split() if word != '<UNK>']) # Use the derived combined vocab
    logger.info(f"Final combined data: {len(combined_texts)} samples. Final vocab size: {len(combined_vocab)}")
    
    logger.info(f"Training final model with best parameters on all data ({len(combined_texts)} samples)...")
    
    # Build final pipeline steps using best params and combined_vocab
    final_pipeline_steps = []
    final_pipeline_steps.append(('glove', GloveVectorizer(
        use_tfidf_weighting=best_params.get('use_tfidf_weighting', True), # Use best or default
        vocabulary=combined_vocab,
        embedding_dim=best_params['embedding_dim']
    )))

    if best_params['kernel'] == 'rbf':
        final_scaler = StandardScaler()
        final_svm = SVC(
            C=best_params['C'],
            kernel='rbf',
            gamma=best_params['gamma'],
            class_weight=best_params['class_weight'],
            probability=True, # Enable probability for potential later use
            random_state=42
        )
        final_pipeline_steps.extend([('scaler', final_scaler), ('svm', final_svm)])
    elif best_params['kernel'] == 'linear':
        final_svm = LinearSVC(
            C=best_params['C'],
            class_weight=best_params['class_weight'],
            dual='auto',
            random_state=42,
            max_iter=3000
        )
        final_pipeline_steps.append(('svm', final_svm))
    else:
        raise ValueError(f"Unsupported kernel in best params: {best_params['kernel']}")
    
    # Assemble the full final pipeline
    final_pipeline = Pipeline(final_pipeline_steps)
    
    # Use LoggingPipeline for final fit if desired, or standard Pipeline
    final_pipeline_logged = LoggingPipeline(final_pipeline.steps, logger=logger)

    with timer("Final full model training", logger):
        # Fit the full pipeline on the raw combined text data
        final_pipeline_logged.fit(combined_texts, combined_labels)
    
    # Save final model
    final_model_path = config.SAVE_DIR / "svm" / "final_model_streamlined.pkl"
    with timer("Model saving", logger):
        with open(final_model_path, 'wb') as f:
            pickle.dump(final_pipeline_logged, f)
    
    logger.info(f"Final model saved to {final_model_path}")
    logger.info(f"Final memory usage: {get_memory_usage():.2f} MB")
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
