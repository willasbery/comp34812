import logging
import pandas as pd
import json
import numpy as np
from pathlib import Path
import pickle

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
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)


NUM_TRIALS = 10

train_df = pd.read_csv(config.TRAIN_FILE)
dev_df = pd.read_csv(config.DEV_FILE)
train_aug_df = pd.read_csv(config.AUG_TRAIN_FILE)

train_df, dev_df, train_labels, dev_labels = prepare_data(train_df, train_aug_df, dev_df)

def objective(trial):
    """Optuna objective function for hyperparameter optimization."""
    # Load data
    global train_df, dev_df, train_labels, dev_labels
    trial_number = trial.number
    
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
    
    # Create the pipeline
    pipeline = Pipeline(pipeline_steps, verbose=True)
    
    # Create cross-validation strategy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Train model with cross-validation
    logging.info(f"Training SVM with hyperparameters: C={C}, kernel={kernel}, gamma={gamma}, class_weight={class_weight}")
    
    # Use cross-validation for more robust evaluation
    cv_scores = cross_val_score(
        pipeline, 
        train_df['text'], 
        train_labels, 
        cv=cv,
        scoring=make_scorer(f1_score, average='weighted'),
        n_jobs=-1
    )
    mean_cv_score = np.mean(cv_scores)
    
    # Also train on full training set and evaluate on dev set for comparison
    pipeline.fit(train_df['text'], train_labels)
    dev_preds = pipeline.predict(dev_df['text'])
    metrics = calculate_all_metrics(dev_labels, dev_preds)
    
    # Save trial results
    svm_dir = config.SAVE_DIR / "svm"
    svm_dir.mkdir(parents=True, exist_ok=True)
    
    # Add CV scores to metrics
    metrics["cv_f1_scores"] = cv_scores.tolist()
    metrics["mean_cv_f1"] = mean_cv_score
    
    with (svm_dir / f'svm_{trial_number}.json').open('w') as f:
        combined_results = {**metrics, **trial.params}
        json.dump(combined_results, f)
    
    # Use cross-validation score for optimization
    return mean_cv_score

def main():
    print("\nHYPERPARAMETER TUNING")
    print("=====================")
    print(f"Running {NUM_TRIALS} trials...")
    
    # Check if GPU is available for NumPy/SciPy operations
    device = get_device()
    logging.info(f"Using device: {device} (Note: scikit-learn SVM implementation will utilize CPU)")
    
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
        study.optimize(objective, n_trials=NUM_TRIALS, n_jobs=3)
    except KeyboardInterrupt:
        print("Hyperparameter tuning interrupted.")
    
    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value (Mean CV F1): {trial.value}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    # Train final model with best parameters and save
    best_params = trial.params.copy()
    
    # Create final model pipeline with best parameters
    final_pipeline = create_pipeline_from_params(best_params)
    
    # Train on combined training + validation for final model
    combined_df = pd.concat([train_df, dev_df])
    combined_labels = np.concatenate([train_labels, dev_labels])
    
    print("\nTraining final model with best parameters on all data...")
    final_pipeline.fit(combined_df['text'], combined_labels)
    
    # Save the final model
    final_model_path = config.SAVE_DIR / "svm" / "final_model.pkl"
    with open(final_model_path, 'wb') as f:
        pickle.dump(final_pipeline, f)
    
    print(f"Final model saved to {final_model_path}")
    
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
    
    # Create the pipeline
    return Pipeline(pipeline_steps)