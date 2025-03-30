import logging
import pandas as pd
import json
import numpy as np
from pathlib import Path
import pickle
import time
import psutil
import gc
from tqdm import tqdm
from contextlib import contextmanager

# Hyperparameter tuning
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# SVM
from sklearn.svm import SVC

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, RFECV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.metrics import make_scorer, f1_score
from sklearn.ensemble import RandomForestClassifier

from src.utils.utils import get_device, prepare_data, calculate_all_metrics
from src.utils.TextPreprocessor import TextPreprocessor
from src.utils.FeatureExtractor import FeatureExtractor
from src.utils.GloveVectorizer import GloveVectorizer
from src.config import config

# Set up logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# Memory monitoring
def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)

@contextmanager
def timer(name):
    """Context manager for timing code execution."""
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        logger.info(f"{name} completed in {end_time - start_time:.2f} seconds")


NUM_TRIALS = 50

# Store initial memory usage
initial_memory = get_memory_usage()
logger.info(f"Initial memory usage: {initial_memory:.2f} MB")

train_df = pd.read_csv(config.TRAIN_FILE)
dev_df = pd.read_csv(config.DEV_FILE)
train_aug_df = pd.read_csv(config.AUG_TRAIN_FILE)

with timer("Data preparation"):
    train_df, dev_df, train_labels, dev_labels = prepare_data(train_df, train_aug_df, dev_df)
    logger.info(f"Prepared data: {len(train_df)} training samples, {len(dev_df)} validation samples")
    logger.info(f"Memory after data prep: {get_memory_usage():.2f} MB (+ {get_memory_usage() - initial_memory:.2f} MB)")


def objective(trial):
    """Optuna objective function for hyperparameter optimization."""
    # Load data
    global train_df, dev_df, train_labels, dev_labels
    trial_number = trial.number
    
    logger.info(f"\n{'='*50}\nStarting trial {trial_number}/{NUM_TRIALS}\n{'='*50}")
    trial_start = time.time()
    
    # Suggest hyperparameters
    C = trial.suggest_float("C", 0.01, 100.0, log=True)
    kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly", "sigmoid"])
    gamma = trial.suggest_categorical("gamma", ["scale", "auto"]) if kernel in ["rbf", "poly", "sigmoid"] else "scale"
    
    if kernel == "poly":
        degree = trial.suggest_int("degree", 2, 5)
    else:
        degree = 3  # Default value
    
    # Feature selection parameters
    use_feature_selection = trial.suggest_categorical("use_feature_selection", [True, False])
    if use_feature_selection:
        feature_selection_method = trial.suggest_categorical("feature_selection_method", 
                                                           ["kbest", "model_based", "pca"])
        if feature_selection_method == "kbest":
            k_best = trial.suggest_int("k_best", 10, 100)
        elif feature_selection_method == "pca":
            n_components = trial.suggest_float("n_components", 0.7, 0.99)
    
    # Class weighting for imbalanced data
    class_weight = trial.suggest_categorical("class_weight", ["balanced", None])
    
    # TF-IDF weighting for GloVe
    use_tfidf_weighting = trial.suggest_categorical("use_tfidf_weighting", [True, False])
    
    logger.info(f"Trial {trial_number} hyperparameters:")
    logger.info(f"  SVM: C={C}, kernel={kernel}, gamma={gamma}, class_weight={class_weight}")
    logger.info(f"  Feature selection: {use_feature_selection} "
               f"(method={feature_selection_method if use_feature_selection else 'N/A'})")
    logger.info(f"  GloVe TF-IDF weighting: {use_tfidf_weighting}")
    
    # Create feature pipeline
    feature_pipeline = []
    
    # Add text features
    feature_pipeline.append(
        ('text_features', Pipeline([
            ('glove', GloveVectorizer(use_tfidf_weighting=use_tfidf_weighting))
        ]))
    )
    
    # Add custom features
    feature_pipeline.append(
        ('custom_features', FeatureExtractor())
    )
    
    # Create main pipeline
    pipeline_steps = [
        ('features', FeatureUnion(feature_pipeline)),
        ('scaler', StandardScaler())
    ]
    
    # Add feature selection if enabled
    if use_feature_selection:
        if feature_selection_method == "kbest":
            pipeline_steps.append(('feature_selection', SelectKBest(f_classif, k=k_best)))
        elif feature_selection_method == "model_based":
            pipeline_steps.append(('feature_selection', SelectFromModel(
                RandomForestClassifier(n_estimators=100, random_state=42)
            )))
        elif feature_selection_method == "pca":
            pipeline_steps.append(('feature_selection', PCA(n_components=n_components)))
    
    # Add SVM classifier
    pipeline_steps.append(('svm', SVC(
        C=C,
        kernel=kernel,
        gamma=gamma,
        degree=degree if kernel == "poly" else 3,
        probability=True,
        class_weight=class_weight,
        random_state=42
    )))
    
    # Create the pipeline with logging
    pipeline = Pipeline(pipeline_steps, verbose=True)
    
    # Train model
    with timer(f"Trial {trial_number} training"):
        pipeline.fit(train_df['text'], train_labels)
    
    # Evaluate on dev set
    with timer(f"Trial {trial_number} evaluation"):
        dev_preds = pipeline.predict(dev_df['text'])
        metrics = calculate_all_metrics(dev_labels, dev_preds)
    
    # Save trial results
    svm_dir = config.SAVE_DIR / "svm"
    svm_dir.mkdir(parents=True, exist_ok=True)
    with (svm_dir / f'svm_{trial_number}.json').open('w') as f:
        combined_results = {**metrics, **trial.params}
        json.dump(combined_results, f)
    
    trial_duration = time.time() - trial_start
    logger.info(f"Trial {trial_number} completed in {trial_duration:.2f} seconds")
    logger.info(f"Trial {trial_number} results: W Macro-F1 = {metrics['W Macro-F1']:.4f}")
    
    # Explicitly collect garbage to free memory
    gc.collect()
    
    return metrics["W Macro-F1"]

def main():
    logger.info("\n" + "="*70)
    logger.info("EVIDENCE DETECTION SVM MODEL TRAINING")
    logger.info("="*70)
    logger.info(f"Running {NUM_TRIALS} hyperparameter optimization trials...")
    
    # Check if GPU is available for NumPy/SciPy operations
    device = get_device()
    logger.info(f"Using device: {device} (Note: scikit-learn SVM implementation will utilize CPU)")
    
    # Create a study with TPE sampler and MedianPruner
    sampler = TPESampler(seed=42, 
                         n_startup_trials=int(NUM_TRIALS / 10), # First 10% of trials are random, then TPE
                         multivariate=True, 
                         constant_liar=True)  # TPE sampler as requested
    pruner = MedianPruner(n_startup_trials=5, 
                          n_warmup_steps=5, 
                          interval_steps=2)
    
    study = optuna.create_study(
        direction='maximize',  # Maximize F1 score
        sampler=sampler,
        pruner=pruner,
        study_name='svm_evidence_detection'
    )
    
    try:
        with timer("Hyperparameter optimization"):
            study.optimize(objective, n_trials=NUM_TRIALS, n_jobs=5)
    except KeyboardInterrupt:
        logger.warning("Hyperparameter tuning interrupted by user.")
    
    logger.info("\nBest trial:")
    trial = study.best_trial
    logger.info(f"  Value (W Macro-F1): {trial.value:.4f}")
    logger.info("  Params:")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")
        
    # Train final model with best parameters and save
    best_params = trial.params.copy()
    
    # Create final model pipeline with best parameters
    final_pipeline = create_pipeline_from_params(best_params)
    
    # Train on combined training + validation for final model
    combined_df = pd.concat([train_df, dev_df])
    combined_labels = np.concatenate([train_labels, dev_labels])
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING FINAL MODEL")
    logger.info("="*70)
    logger.info(f"Training final model with best parameters on all data ({len(combined_df)} samples)...")
    
    with timer("Final model training"):
        final_pipeline.fit(combined_df['text'], combined_labels)
    
    # Save the final model
    final_model_path = config.SAVE_DIR / "svm" / "final_model.pkl"
    with timer("Model saving"):
        with open(final_model_path, 'wb') as f:
            pickle.dump(final_pipeline, f)
    
    logger.info(f"Final model saved to {final_model_path}")
    logger.info(f"Final memory usage: {get_memory_usage():.2f} MB")
    logger.info("Training completed successfully!")

def create_pipeline_from_params(params):
    """Create a pipeline from the best parameters."""
    # Extract parameters
    C = params.get("C")
    kernel = params.get("kernel")
    gamma = params.get("gamma")
    degree = params.get("degree", 3) if kernel == "poly" else 3
    class_weight = params.get("class_weight")
    use_feature_selection = params.get("use_feature_selection", False)
    feature_selection_method = params.get("feature_selection_method", None)
    k_best = params.get("k_best", 50)
    n_components = params.get("n_components", 0.9)
    use_tfidf_weighting = params.get("use_tfidf_weighting", True)
    
    # Create feature pipeline
    feature_pipeline = []
    
    # Add text features
    feature_pipeline.append(
        ('text_features', Pipeline([
            ('glove', GloveVectorizer(use_tfidf_weighting=use_tfidf_weighting))
        ]))
    )
    
    # Add custom features
    feature_pipeline.append(
        ('custom_features', FeatureExtractor())
    )
    
    # Create main pipeline
    pipeline_steps = [
        ('features', FeatureUnion(feature_pipeline)),
        ('scaler', StandardScaler())
    ]
    
    # Add feature selection if enabled
    if use_feature_selection:
        if feature_selection_method == "kbest":
            pipeline_steps.append(('feature_selection', SelectKBest(f_classif, k=k_best)))
        elif feature_selection_method == "model_based":
            pipeline_steps.append(('feature_selection', SelectFromModel(
                RandomForestClassifier(n_estimators=100, random_state=42)
            )))
        elif feature_selection_method == "pca":
            pipeline_steps.append(('feature_selection', PCA(n_components=n_components)))
    
    # Add SVM classifier
    pipeline_steps.append(('svm', SVC(
        C=C,
        kernel=kernel,
        gamma=gamma,
        degree=degree,
        probability=True,
        class_weight=class_weight,
        random_state=42
    )))
    
    # Create the pipeline with logging capabilities
    return Pipeline(pipeline_steps)