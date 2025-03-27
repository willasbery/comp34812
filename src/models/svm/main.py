import logging
import pandas as pd
import json

# Hyperparameter tuning
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# SVM
from sklearn.svm import SVC

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler

from src.utils.utils import get_device, prepare_data, calculate_all_metrics
from src.utils.TextPreprocessor import TextPreprocessor
from src.utils.FeatureExtractor import FeatureExtractor
from src.utils.GloveVectorizer import GloveVectorizer
from src.config import config

# Set up logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)


NUM_TRIALS = 50

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
    
    # Create pipeline
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_features', Pipeline([
                ('glove', GloveVectorizer())
            ])),
            ('custom_features', FeatureExtractor())
        ])),
        ('scaler', StandardScaler()), 
        ('svm', SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            degree=degree if kernel == "poly" else 3,
            probability=True
        ))
    ],
    verbose=True)
    
    # Train model
    logging.info(f"Training SVM with hyperparameters: C={C}, kernel={kernel}, gamma={gamma}")
    print(train_df['text'][:10].values)
    pipeline.fit(train_df['text'].values, train_labels)  # Convert to numpy array
    
    # Evaluate on dev set
    dev_preds = pipeline.predict(dev_df['text'].values)  # Convert to numpy array
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
    sampler = TPESampler(seed=42, 
                         n_startup_trials=int(NUM_TRIALS / 10), # First 10% of trials are random, then TPE
                         multivariate=True, 
                         constant_liar=True)  # TPE sampler as requested
    pruner = MedianPruner(n_startup_trials=5, 
                          n_warmup_steps=5, 
                          interval_steps=2)
    
    study = optuna.create_study(
        direction='maximize',  # Maximize macro-F1
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