"""
Basic DeBERTa model for evidence detection with peft training to allow for larger model sizes.
"""
import logging
import os
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import optuna
from optuna.samplers import TPESampler
import json
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    matthews_corrcoef,
    confusion_matrix,
)
from datasets import Dataset as HFDataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

# Disable wandb
os.environ['WANDB_DISABLED'] = 'true'


# Path configuration
DATA_DIR = Path(__file__).parent.parent / "data"
TRAIN_FILE = DATA_DIR / "train.csv"
DEV_FILE = DATA_DIR / "dev.csv"
AUG_TRAIN_FILE = DATA_DIR / "train_augmented.csv"
ANOTHER_AUG_FILE = DATA_DIR / "positive_examples.csv"
AUG_TRAIN_HIGH_REPLACEMENT_FILE = DATA_DIR / "train_augmented_high_replacement_fraction.csv"
SAVE_DIR = DATA_DIR / "results" / "transformer"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Training parameters
BATCH_SIZE = 8
NUM_EPOCHS = 3
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
DROPOUT_RATE = 0.11
MAX_SEQ_LENGTH = 384
# BASE_MODEL = 'cross-encoder/nli-deberta-v3-large'
BASE_MODEL = 'microsoft/deberta-v2-xlarge-mnli'

# Optuna parameters
N_TRIALS = 6


# Hyperparameter search space
BATCH_SIZES = [4, 8, 16]
LEARNING_RATES = [5e-5, 1e-5, 5e-6]
WEIGHT_DECAYS = [0.1, 0.01, 0.001]
WARMUP_RATIOS = [0.05, 0.1, 0.15]
DROPOUT_RATES = [0, 0.05, 0.1, 0.15]
MAX_SEQ_LENGTHS = [256, 384, 512]

def get_device() -> torch.device:
    """Determine the device to use for computations."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def preprocess_function(examples, tokenizer, max_seq_length):
    """Process examples for BERT/DeBERTa classification."""
    # Combine claim and evidence
    claims = []
    evidences = []

    # Create inputs and targets
    for claim, evidence in zip(examples['Claim'], examples['Evidence']):
        formatted_claim = f"Claim: {claim}"
        formatted_evidence = f"Evidence: {evidence}"
        claims.append(formatted_claim)
        evidences.append(formatted_evidence)
    
    # Tokenize inputs
    model_inputs = tokenizer(
        claims,
        evidences,
        max_length=max_seq_length,
        padding=False,
        truncation=True,
    )
    
    # Add labels (binary classification)
    model_inputs["labels"] = examples['label']
    return model_inputs

def convert_to_hf_dataset(dataframe):
    """Convert pandas dataframe to HuggingFace dataset format."""
    return HFDataset.from_pandas(dataframe)

def load_data(tokenizer, max_seq_length):
    """Load and prepare the training and development datasets."""
    logging.info("Loading datasets...")
    
    # Load CSV files into pandas dataframes
    train_df = pd.read_csv(TRAIN_FILE)
    dev_df = pd.read_csv(DEV_FILE)

    try:
        train_augmented_df = pd.read_csv(AUG_TRAIN_FILE)
        another_aug_df = pd.read_csv(ANOTHER_AUG_FILE)
        train_df = pd.concat([train_df, train_augmented_df, another_aug_df])
    except Exception as e:
        logging.error(f"Error loading or concatenating augmented training data: {e}")
        raise
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Development data shape: {dev_df.shape}")
    
    # Check and report class distribution
    train_positive = (train_df['label'] == 1).sum()
    train_negative = (train_df['label'] == 0).sum()
    dev_positive = (dev_df['label'] == 1).sum()
    dev_negative = (dev_df['label'] == 0).sum()
    
    print(f"Training data distribution: Positive: {train_positive} ({train_positive/len(train_df)*100:.1f}%), "
                 f"Negative: {train_negative} ({train_negative/len(train_df)*100:.1f}%)")
    print(f"Dev data distribution: Positive: {dev_positive} ({dev_positive/len(dev_df)*100:.1f}%), "
                 f"Negative: {dev_negative} ({dev_negative/len(dev_df)*100:.1f}%)")
    
    # Add a sequential index to keep track of original order (if not already present)
    if 'original_index' not in dev_df.columns:
        dev_df['original_index'] = list(range(len(dev_df)))
    
    # Convert to HuggingFace datasets
    train_dataset = convert_to_hf_dataset(train_df)
    dev_dataset = convert_to_hf_dataset(dev_df)
    
    # Apply preprocessing (tokenization)
    train_dataset = train_dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, max_seq_length),
        batched=True,
        batch_size=1000,
        remove_columns=['Claim', 'Evidence', 'label']
    )
    
    # For dev dataset, keep track of original indices but remove other columns
    columns_to_remove = [col for col in dev_df.columns if col not in ['original_index']]
    dev_dataset = dev_dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, max_seq_length),
        batched=True,
        batch_size=1000,
        remove_columns=columns_to_remove
    )
    
    # Set format for pytorch
    train_dataset.set_format(type='torch')
    dev_dataset.set_format(type='torch')
    
    return train_dataset, dev_dataset, dev_df

def compute_metrics(eval_pred):
    """Calculate evaluation metrics for classification."""
    predictions, labels = eval_pred
    
    # For binary classification, get the predicted class (0 or 1)
    predictions = predictions.argmax(axis=1)
    
    # Calculate metrics with more focus on positive class
    accuracy = accuracy_score(labels, predictions)
    
    # Get more detailed metrics for both classes
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )
    
    # Weighted metrics
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    
    # Matthews Correlation Coefficient
    mcc = matthews_corrcoef(labels, predictions)
    
    # Return both class-specific and overall metrics
    metrics = {
        'Accuracy': accuracy,
        'Positive_Precision': precision[1] if len(precision) > 1 else 0,
        'Positive_Recall': recall[1] if len(recall) > 1 else 0,
        'Positive_F1': f1[1] if len(f1) > 1 else 0,
        'W Macro-P': weighted_precision,
        'W Macro-R': weighted_recall,
        'W Macro-F1': weighted_f1,
        'MCC': mcc
    }
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    classes = ['Negative', 'Positive']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, f'{cm[i, j]}\n({cm_norm[i, j]:.2f})',
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_path)
    plt.close()

def train_model(
    model,
    train_dataset,
    eval_dataset,
    output_dir,
    tokenizer,
    **kwargs
):
    """Train the classification model."""
    logging.info("Starting training...")
    
    # Free up CUDA memory before training
    torch.cuda.empty_cache()
    
    # Create data collator for dynamic padding
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding='longest'
    )
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        greater_is_better=True,
        **kwargs
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )
    
    trainer.train()
    
    # Evaluate on dev set and plot confusion matrix
    eval_results = trainer.evaluate()
    dev_preds = trainer.predict(eval_dataset)
    y_true = dev_preds.label_ids
    y_pred = dev_preds.predictions.argmax(axis=1)

    # Save predictions to a CSV file with original dev data for alignment
    # First, load the original dev CSV to maintain alignment
    dev_df = pd.read_csv(DEV_FILE)
    
    # Create a dataframe with predictions
    predictions_df = pd.DataFrame({'prediction': y_pred})

    
    
    # Check if the evaluation dataset has original indices
    if hasattr(eval_dataset, 'original_index') or 'original_index' in eval_dataset.features:
        # Get original indices if present
        try:
            original_indices = [item['original_index'] for item in eval_dataset]
            # Sort predictions by original index
            predictions_df['original_index'] = original_indices
            predictions_df = predictions_df.sort_values('original_index')
            del predictions_df['original_index']  # Remove after sorting
        except Exception as e:
            logging.warning(f"Couldn't use original indices: {e}")
    
    # Ensure the predictions align with the original data
    if len(dev_df) == len(predictions_df):
        # Add predictions to the original dev dataframe
        dev_df['prediction'] = predictions_df['prediction'].values
        predictions_csv_path = os.path.join(output_dir, "predictions_with_data.csv")
        dev_df.to_csv(predictions_csv_path, index=False)
        print(f"Predictions with original data saved to {predictions_csv_path}")
        
        # Also save just the predictions for convenience
        predictions_only_path = os.path.join(output_dir, "predictions.csv")
        predictions_df.to_csv(predictions_only_path, index=False)
    else:
        print(f"Prediction count ({len(predictions_df)}) doesn't match dev data count ({len(dev_df)})")
        # Save just the predictions
        predictions_csv_path = os.path.join(output_dir, "predictions.csv")
        predictions_df.to_csv(predictions_csv_path, index=False)
        print(f"Predictions saved to {predictions_csv_path}")
    
    # Plot and save confusion matrix
    cm_save_path = os.path.join(output_dir, "confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, cm_save_path)
    
    trainer.save_model()
    logging.info(f"Model saved to {output_dir}")
    
    return eval_results

def objective(trial):
    """Optuna objective function for hyperparameter optimization."""
    # Get hyperparameters from trial
    batch_size = trial.suggest_categorical("batch_size", BATCH_SIZES)
    learning_rate = trial.suggest_categorical("learning_rate", LEARNING_RATES)
    weight_decay = trial.suggest_categorical("weight_decay", WEIGHT_DECAYS)
    warmup_ratio = trial.suggest_categorical("warmup_ratio", WARMUP_RATIOS)
    dropout = trial.suggest_categorical("dropout", DROPOUT_RATES)
    max_seq_length = trial.suggest_categorical("max_seq_length", MAX_SEQ_LENGTHS)
    
    device = get_device()
    logging.info(f"Trial {trial.number}: Using device: {device}")
    
    # Free GPU memory
    torch.cuda.empty_cache()
    
    # LoRA configuration (fixed for all trials)
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
    )
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, 
        num_labels=2,
        hidden_dropout_prob=dropout,
        attention_probs_dropout_prob=dropout,
        ignore_mismatched_sizes=True,
    )
    
    model = get_peft_model(model, peft_config)
    model.to(device)
    
    # Load data with current max_seq_length
    train_dataset, dev_dataset, dev_df = load_data(tokenizer, max_seq_length)
    
    # Training parameters
    training_params = {
        'per_device_train_batch_size': batch_size,
        'per_device_eval_batch_size': batch_size,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'num_train_epochs': NUM_EPOCHS,
        'warmup_ratio': warmup_ratio,
        'lr_scheduler_type': 'cosine_with_restarts',
        'eval_strategy': 'steps',
        'eval_steps': 500,
        'save_strategy': 'steps',
        'save_steps': 500,
        'save_total_limit': 2,
        'load_best_model_at_end': True,
        'metric_for_best_model': 'MCC',
        'fp16': device.type == 'cuda', 
        'optim': 'adamw_torch',
        'logging_steps': 100,
        'logging_first_step': True,
        'group_by_length': True,
        'seed': 42,
        'dataloader_num_workers': 4,
        'label_smoothing_factor': 0.05,
        'max_grad_norm': 1.0,
        'gradient_checkpointing': True,
    }
    
    # Set trial output directory
    trial_dir = SAVE_DIR / f"trial_{trial.number}"
    
    try:
        # Train with current hyperparameters
        eval_results = train_model(
            model,
            train_dataset,
            dev_dataset,
            trial_dir,
            tokenizer,
            **training_params
        )
        
        # Log the hyperparameters and results
        params = {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio,
            "dropout": dropout,
            "max_seq_length": max_seq_length,
        }
        
        with open(trial_dir / "hyperparameters.json", "w") as f:
            json.dump({**params, **eval_results}, f, indent=2)
        
        # Return Matthews Correlation Coefficient as the objective value
        return eval_results["eval_MCC"]
    
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        # Return very bad score for failed trials
        return -1.0

def run_optuna_experiment():
    """Run Optuna hyperparameter optimization experiment."""
    logging.info("Starting hyperparameter optimization with Optuna...")
    
    # Create output directory for study
    study_dir = SAVE_DIR / "optuna_study"
    study_dir.mkdir(exist_ok=True)
    
    # Create a pruner to terminate unpromising trials
    pruner = optuna.pruners.MedianPruner()
    
    # Create a storage for the study
    storage_name = f"sqlite:///{study_dir}/optuna_study.db"
    
    # Create TPE sampler for Bayesian optimization
    sampler = TPESampler(seed=42)
    
    # Create the study
    study = optuna.create_study(
        direction="maximize",
        pruner=pruner,
        storage=storage_name,
        study_name="deberta_claim_evidence",
        load_if_exists=True,
        sampler=sampler
    )
    
    # Run optimization
    study.optimize(objective, n_trials=N_TRIALS)
    
    # Get best trial
    best_trial = study.best_trial
    
    # Log additional information about the Bayesian optimization
    logging.info(f"Using Bayesian optimization with TPE sampler")
    logging.info(f"Best trial: {best_trial.number}")
    logging.info(f"Best value: {best_trial.value}")
    logging.info("Best hyperparameters:")
    
    for param, value in best_trial.params.items():
        logging.info(f"\t{param}: {value}")
    
    # Save best parameters
    best_params = {
        "batch_size": best_trial.params["batch_size"],
        "learning_rate": best_trial.params["learning_rate"],
        "weight_decay": best_trial.params["weight_decay"],
        "warmup_ratio": best_trial.params["warmup_ratio"],
        "dropout": best_trial.params["dropout"],
        "max_seq_length": best_trial.params["max_seq_length"],
        "mcc_score": best_trial.value
    }
    
    with open(study_dir / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)
    
    # Plot optimization history
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_html(str(study_dir / "optimization_history.html"))
    
    # Plot parameter importance
    fig = optuna.visualization.plot_param_importances(study)
    fig.write_html(str(study_dir / "param_importances.html"))
    
    # Plot parameter relationships
    fig = optuna.visualization.plot_parallel_coordinate(study)
    fig.write_html(str(study_dir / "parallel_coordinate.html"))
    
    # Plot high-dimensional parameter relationships
    fig = optuna.visualization.plot_contour(study)
    fig.write_html(str(study_dir / "contour.html"))
    
    return best_params

def main():
    """Main execution function."""
    device = get_device()
    logging.info(f"Using device: {device}")

    # Run Optuna hyperparameter optimization
    best_params = run_optuna_experiment()
    
    # Optional: Train final model with best parameters
    logging.info("Training final model with best parameters...")
    
    # Free GPU memory
    torch.cuda.empty_cache()

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
    )

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, 
        num_labels=2,
        hidden_dropout_prob=best_params["dropout"],
        attention_probs_dropout_prob=best_params["dropout"],
        ignore_mismatched_sizes=True,
    )

    model = get_peft_model(model, peft_config)
    model.to(device)

    # Load data with best max_seq_length
    train_dataset, dev_dataset, dev_df = load_data(tokenizer, best_params["max_seq_length"])
    
    # Training parameters with best hyperparameters
    training_params = {
        'per_device_train_batch_size': best_params["batch_size"],
        'per_device_eval_batch_size': best_params["batch_size"],
        'learning_rate': best_params["learning_rate"],
        'weight_decay': best_params["weight_decay"],
        'num_train_epochs': NUM_EPOCHS,
        'warmup_ratio': best_params["warmup_ratio"],
        'lr_scheduler_type': 'cosine_with_restarts',
        'eval_strategy': 'steps',
        'eval_steps': 500,
        'save_strategy': 'steps',
        'save_steps': 500,
        'save_total_limit': 5,
        'load_best_model_at_end': True,
        'metric_for_best_model': 'MCC',
        'fp16': device.type == 'cuda', 
        'optim': 'adamw_torch',
        'logging_steps': 100,
        'logging_first_step': True,
        'group_by_length': True,
        'seed': 42,
        'dataloader_num_workers': 4,
        'label_smoothing_factor': 0.05,
        'max_grad_norm': 1.0,
        'gradient_checkpointing': True,
    }
    
    # Train with best parameters
    model_save_path = SAVE_DIR / f"{BASE_MODEL.split('/')[-1]}_best"
    eval_results = train_model(
        model,
        train_dataset,
        dev_dataset,
        model_save_path,
        tokenizer,
        **training_params
    )
    
    print(f"Final evaluation results: {eval_results}")

if __name__ == "__main__":
    main()