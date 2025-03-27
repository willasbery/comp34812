import logging
import pandas as pd
from pathlib import Path
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import os
import torch

from src.utils.utils import get_device, prepare_data, calculate_all_metrics
from src.config import config

# Set up logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

# Optuna config
N_TRIALS = 250  # Number of Optuna trials

def objective(trial):
    """Optuna objective function for hyperparameter optimization."""
    # Load data
    logging.info("Loading datasets...")
    train_df = pd.read_csv(config.TRAIN_FILE)
    aug_train_df = pd.read_csv(config.AUG_TRAIN_FILE)
    dev_df = pd.read_csv(config.DEV_FILE)
    
    logging.info(f"Training data shape: {train_df.shape}")
    logging.info(f"Development data shape: {dev_df.shape}")
    
    # Prepare data
    train_df, dev_df, train_labels, dev_labels = prepare_data(train_df, aug_train_df, dev_df)
    
    # TF-IDF vectorizer parameters
    max_features = trial.suggest_categorical("max_features", [5000, 10000, 15000, 20000])
    min_df = trial.suggest_categorical("min_df", [1, 2, 3])
    ngram_range_str = trial.suggest_categorical("ngram_range", ["1,1", "1,2", "1,3"])
    ngram_range = tuple(map(int, ngram_range_str.split(",")))
    
    # XGBoost hyperparameters
    xgb_params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'hist',
        'random_state': 42
    }
    
    # Create pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            ngram_range=ngram_range,
            stop_words='english'
        )),
        ('scaler', StandardScaler(with_mean=False)),  # TF-IDF matrices are sparse
        ('xgb', xgb.XGBClassifier(
            **xgb_params,
            enable_categorical=True  # Enable categorical feature support
        ))
    ])
    
    # Train model
    logging.info(f"Training XGBoost with hyperparameters: {xgb_params}")
    
    X_train = pipeline.named_steps['tfidf'].fit_transform(train_df['text'])
    X_train = pipeline.named_steps['scaler'].fit_transform(X_train)
    
    X_dev = pipeline.named_steps['tfidf'].transform(dev_df['text'])
    X_dev = pipeline.named_steps['scaler'].transform(X_dev)
    
    # Create DMatrix objects
    dtrain = xgb.DMatrix(X_train, label=train_labels)
    ddev = xgb.DMatrix(X_dev, label=dev_labels)
    
    # Train model
    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=xgb_params['n_estimators'],
        evals=[(ddev, 'eval')],
        early_stopping_rounds=50,
        verbose_eval=False
    )
    
    # Make predictions
    dev_preds = (model.predict(ddev) >= 0.5).astype(int)    
    metrics = calculate_all_metrics(dev_labels, dev_preds)
    
    # Report intermediate values for pruning
    trial.report(metrics['W Macro-F1'], step=model.best_iteration)
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()
    
    return metrics['W Macro-F1']

def main():
    print("\nHYPERPARAMETER TUNING")
    print("=====================")
    print(f"Running {N_TRIALS} trials...")
    
    # Check if GPU is available
    device = get_device()
    if torch.cuda.is_available():
        # Enable GPU acceleration for XGBoost
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    logging.info(f"Using device: {device}")
    
    # Create a study with TPE sampler and MedianPruner
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=2)
    
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
        study_name='xgboost_evidence_detection'
    )
    
    try:
        study.optimize(objective, n_trials=N_TRIALS, n_jobs=-1)
    except KeyboardInterrupt:
        print("Hyperparameter tuning interrupted.")
    
    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value (Accuracy): {trial.value}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

if __name__ == "__main__":
    main()