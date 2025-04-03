import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Hyperparameter tuning
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from src.utils.utils import get_device, prepare_data, calculate_all_metrics
from src.config import config
from src.utils.GloveVectorizer import GloveVectorizer

import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

NUM_TRIALS = 100
    
train_df = pd.read_csv(config.TRAIN_FILE)
aug_train_df = pd.read_csv(config.AUG_TRAIN_HIGH_REPLACEMENT_FILE)
dev_df = pd.read_csv(config.DEV_FILE)
   
train_df, dev_df, train_labels, dev_labels = prepare_data(train_df, aug_train_df, dev_df)

logging.info(f"Training data shape: {train_df.shape}")
logging.info(f"Dev data shape: {dev_df.shape}")

def objective(trial):    
    # XGBoost hyperparameters
    xgb_params = {
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'use_label_encoder': False,
        'tree_method': 'hist',
        'max_bin': 256,
        'random_state': 42
    }
    
    logging.info(f"Training XGBoost with hyperparameters: {xgb_params}")
    
    CHUNK_SIZE = 1000
    
    # Get embeddings for training data in chunks
    X_train_chunks = []
    for i in range(0, len(train_df), CHUNK_SIZE):
        
        chunk = train_df['text'].iloc[i:i + CHUNK_SIZE]
        X_chunk = GloveVectorizer().fit_transform(chunk)
        X_chunk = StandardScaler(with_mean=False).fit_transform(X_chunk)
        
        X_train_chunks.append(X_chunk)
    
    X_train = np.vstack(X_train_chunks)
    del X_train_chunks
    
    # Process dev data
    X_dev_chunks = []
    for i in range(0, len(dev_df), CHUNK_SIZE):
        chunk = dev_df['text'].iloc[i:i + CHUNK_SIZE]
        
        X_chunk = GloveVectorizer().transform(chunk)
        X_chunk = StandardScaler(with_mean=False).fit_transform(X_chunk)
        
        X_dev_chunks.append(X_chunk)
        
    X_dev = np.vstack(X_dev_chunks)
    del X_dev_chunks
    
    dtrain = xgb.DMatrix(X_train, label=train_labels, 
                        enable_categorical=True,
                        nthread=-1)
    ddev = xgb.DMatrix(X_dev, label=dev_labels,
                       enable_categorical=True,
                       nthread=-1)
    del X_train, X_dev
    
    # Train model
    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=xgb_params['n_estimators'],
        evals=[(ddev, 'eval')],
        early_stopping_rounds=25,
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
    logging.info("\nHYPERPARAMETER TUNING")
    logging.info("=====================")
    logging.info(f"Running {NUM_TRIALS} trials...")
    
    # Create a study with TPE sampler and MedianPruner
    sampler = TPESampler(seed=42, 
                         n_startup_trials=int(NUM_TRIALS / 10), # First 10% of trials are random, then TPE
                         multivariate=True, 
                         constant_liar=True) # constant_liar = True as we are doing distributed optimisation

    pruner = MedianPruner(n_startup_trials=5, 
                          n_warmup_steps=5, 
                          interval_steps=2)
    
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
        study_name='xgboost_evidence_detection'
    )
    
    try:
        study.optimize(objective, n_trials=NUM_TRIALS, n_jobs=-1)
    except KeyboardInterrupt:
        logging.info("Hyperparameter tuning interrupted.")
    
    logging.info("\nBest trial:")
    trial = study.best_trial
    logging.info(f"  Value (Accuracy): {trial.value}")
    logging.info("  Params:")
    for key, value in trial.params.items():
        logging.info(f"    {key}: {value}")
