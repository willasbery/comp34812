"""
DeBERTa model for claim-evidence classification with hyperparameter optimization.

This module implements a fine-tuned DeBERTa model for binary classification of claim-evidence pairs,
using Optuna for hyperparameter optimization and PEFT for efficient fine-tuning.
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
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    matthews_corrcoef,
    confusion_matrix,
)
from datasets import Dataset as HFDataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch import nn
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
DATA_DIR = Path("/kaggle/working/")
TRAIN_FILE = "/kaggle/input/ed-uom/train.csv"
DEV_FILE = "/kaggle/input/ed-uom/dev.csv"
AUG_TRAIN_FILE = "/kaggle/input/ed-uom/train_augmented.csv"
NEW_AUG = "/kaggle/input/ed-uom/train_augmented_new.csv"
AUG_TRAIN_HIGH_REPLACEMENT_FILE = DATA_DIR / "train_augmented_high_replacement_fraction.csv"
SAVE_DIR = DATA_DIR / "results" / "transformer"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Training parameters
BATCH_SIZE = 8
NUM_EPOCHS = 3
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.05
WARMUP_RATIO = 0.07
DROPOUT_RATE = 0.05
FF_DROPOUT_RATE = 0.05
MAX_SEQ_LENGTH = 512
BASE_MODEL = 'microsoft/deberta-v3-large'


# Optuna parameters
N_TRIALS = 10

WEIGHT_DECAYS = [0.001, 0.1]
WARMUP_RATIOS = [0.05, 0.15]
DROPOUT_RATES = [0.05]
FF_DROPOUT_RATES = [0.05]

def get_device() -> torch.device:
    """
    Determine the best available device for computations.
    
    Returns:
        torch.device: The device to use (cuda, mps, or cpu)
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def preprocess_function(examples, tokenizer, max_seq_length):
    """
    Process examples for DeBERTa classification by formatting and tokenizing claim-evidence pairs.
    
    Args:
        examples: Dataset examples containing Claim and Evidence text
        tokenizer: Tokenizer for the model
        max_seq_length: Maximum sequence length for tokenization
        
    Returns:
        dict: Tokenized inputs with labels
    """
    claims = []
    evidences = []

    for claim, evidence in zip(examples['Claim'], examples['Evidence']):
        formatted_claim = f"Claim: {claim}"
        formatted_evidence = f"Evidence: {evidence}"
        claims.append(formatted_claim)
        evidences.append(formatted_evidence)
    
    model_inputs = tokenizer(
        claims,
        evidences,
        max_length=max_seq_length,
        padding=False,
        truncation=True,
    )
    
    model_inputs["labels"] = examples['label']
    return model_inputs

def convert_to_hf_dataset(dataframe):
    """
    Convert pandas dataframe to HuggingFace dataset format.
    
    Args:
        dataframe: Pandas dataframe containing the data
        
    Returns:
        HFDataset: HuggingFace dataset
    """
    return HFDataset.from_pandas(dataframe)

def load_data(tokenizer, max_seq_length):
    """
    Load and prepare the training and development datasets.
    
    Args:
        tokenizer: Tokenizer for the model
        max_seq_length: Maximum sequence length for tokenization
        
    Returns:
        tuple: Processed training dataset, evaluation dataset, and the original dev dataframe
    """
    logging.info("Loading datasets...")
    
    train_df = pd.read_csv(AUG_TRAIN_FILE)
    dev_df = pd.read_csv(DEV_FILE)


    
    print(f"Training data shape: {train_df.shape}")
    print(f"Development data shape: {dev_df.shape}")
    
    train_positive = (train_df['label'] == 1).sum()
    train_negative = (train_df['label'] == 0).sum()
    dev_positive = (dev_df['label'] == 1).sum()
    dev_negative = (dev_df['label'] == 0).sum()
    
    print(f"Training data distribution: Positive: {train_positive} ({train_positive/len(train_df)*100:.1f}%), "
                 f"Negative: {train_negative} ({train_negative/len(train_df)*100:.1f}%)")
    print(f"Dev data distribution: Positive: {dev_positive} ({dev_positive/len(dev_df)*100:.1f}%), "
                 f"Negative: {dev_negative} ({dev_negative/len(dev_df)*100:.1f}%)")
    
    if 'original_index' not in dev_df.columns:
        dev_df['original_index'] = list(range(len(dev_df)))
    
    train_dataset = convert_to_hf_dataset(train_df)
    dev_dataset = convert_to_hf_dataset(dev_df)
    
    train_dataset = train_dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, max_seq_length),
        batched=True,
        batch_size=1000,
        remove_columns=['Claim', 'Evidence', 'label']
    )
    
    columns_to_remove = [col for col in dev_df.columns if col not in ['original_index']]
    dev_dataset = dev_dataset.map(
        lambda examples: {**preprocess_function(examples, tokenizer, max_seq_length),
                           'original_index': examples['original_index']},
        batched=True,
        batch_size=1000,
        remove_columns=columns_to_remove
    )
    
    train_dataset.set_format(type='torch')
    dev_dataset.set_format(type='torch')
    
    return train_dataset, dev_dataset, dev_df

def compute_metrics(eval_pred):
    """
    Calculate evaluation metrics for classification.
    
    Args:
        eval_pred: Tuple of predictions and labels
        
    Returns:
        dict: Dictionary of metrics including accuracy, precision, recall, F1, and MCC
    """
    predictions, labels = eval_pred
    
    predictions = predictions.argmax(axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )
    
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    
    mcc = matthews_corrcoef(labels, predictions)
    
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
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save the confusion matrix plot
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    classes = ['Negative', 'Positive']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
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
    """
    Train the classification model with the given parameters.
    
    Args:
        model: Model to train
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        output_dir: Directory to save model and results
        tokenizer: Tokenizer for the model
        **kwargs: Additional training arguments
        
    Returns:
        dict: Evaluation results
    """
    logging.info("Starting training...")
    
    torch.cuda.empty_cache()
    
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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=8)]
    )
    
    trainer.train()
    
    # Get the final model from the trainer
    model = trainer.model

    # If using PEFT, merge adapters before saving
    if isinstance(model, PeftModel):
        logging.info("Merging PEFT adapters into the base model...")
        model = model.merge_and_unload()
        logging.info("Adapters merged.")

    # Save the potentially merged model and tokenizer
    logging.info(f"Saving final model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logging.info(f"Model and tokenizer saved to {output_dir}")

    eval_results = trainer.evaluate(eval_dataset)
    dev_preds = trainer.predict(eval_dataset)
    y_true = dev_preds.label_ids
    y_pred = dev_preds.predictions.argmax(axis=1)

    dev_df = pd.read_csv(DEV_FILE)
    predictions_df = pd.DataFrame({'prediction': y_pred})
    
    if hasattr(eval_dataset, 'original_index') or 'original_index' in eval_dataset.features:
        try:
            original_indices = [item['original_index'] for item in eval_dataset]
            predictions_df['original_index'] = original_indices
            predictions_df = predictions_df.sort_values('original_index')
            del predictions_df['original_index']
        except Exception as e:
            logging.warning(f"Couldn't use original indices: {e}")
    
    if len(dev_df) == len(predictions_df):
        dev_df['prediction'] = predictions_df['prediction'].values
        predictions_csv_path = os.path.join(output_dir, "predictions_with_data.csv")
        dev_df.to_csv(predictions_csv_path, index=False)
        print(f"Predictions with original data saved to {predictions_csv_path}")
        
        predictions_only_path = os.path.join(output_dir, "predictions.csv")
        predictions_df.to_csv(predictions_only_path, index=False)
    else:
        print(f"Prediction count ({len(predictions_df)}) doesn't match dev data count ({len(dev_df)})")
        predictions_csv_path = os.path.join(output_dir, "predictions.csv")
        predictions_df.to_csv(predictions_csv_path, index=False)
        print(f"Predictions saved to {predictions_csv_path}")
    
    cm_save_path = os.path.join(output_dir, "confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, cm_save_path)
    
    return eval_results

def objective(trial):
    """
    Optuna objective function for hyperparameter optimization.
    
    Args:
        trial: Optuna trial object
        
    Returns:
        float: Matthews Correlation Coefficient (objective value to maximize)
    """
    weight_decay = trial.suggest_float("weight_decay", WEIGHT_DECAYS[0], WEIGHT_DECAYS[1], log=True)
    warmup_ratio = trial.suggest_float("warmup_ratio", WARMUP_RATIOS[0], WARMUP_RATIOS[1])
    
    device = get_device()
    logging.info(f"Trial {trial.number}: Using device: {device}")
    
    torch.cuda.empty_cache()

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["query_proj", "key_proj", "value_proj", "dense"],
        init_lora_weights='pissa',
        layers_to_transform=[i for i in range(6, 24)]
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, 
        num_labels=2,
        hidden_dropout_prob=DROPOUT_RATE,
        attention_probs_dropout_prob=DROPOUT_RATE,
    )

    hidden_size = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(hidden_size, hidden_size),
        nn.GELU(),
        nn.LayerNorm(hidden_size),
        nn.Dropout(FF_DROPOUT_RATE),
        nn.Linear(hidden_size, 2)
    )
    model.config.num_labels = 2

    model = get_peft_model(model, peft_config)
    model.to(device)
    
    train_dataset, dev_dataset, dev_df = load_data(tokenizer, MAX_SEQ_LENGTH)
    
    training_params = {
        'per_device_train_batch_size': BATCH_SIZE,
        'per_device_eval_batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'weight_decay': weight_decay,
        'num_train_epochs': NUM_EPOCHS,
        'warmup_ratio': warmup_ratio,
        'lr_scheduler_type': 'cosine',
        'evaluation_strategy': 'steps',
        'eval_steps': 1000,
        'save_strategy': 'steps',
        'save_steps': 1000,
        'save_total_limit': 1,
        'load_best_model_at_end': True,
        'metric_for_best_model': 'MCC',
        'fp16': torch.cuda.is_available(),
        'optim': 'adamw_torch',
        'logging_steps': 100,
        'logging_first_step': True,
        'group_by_length': True,
        'seed': 42,
        'label_smoothing_factor': 0.1,
    }
    
    trial_dir = SAVE_DIR / f"trial_{trial.number}"
    
    try:
        eval_results = train_model(
            model,
            train_dataset,
            dev_dataset,
            trial_dir,
            tokenizer,
            **training_params
        )
        
        params = {
            "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio,
        }
        
        with open(trial_dir / "hyperparameters.json", "w") as f:
            json.dump({**params, **eval_results}, f, indent=2)
        
        return eval_results["eval_MCC"]
    
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        return -1.0

def run_optuna_experiment():
    """
    Run Optuna hyperparameter optimization experiment.
    
    Returns:
        dict: Best hyperparameters found by Optuna
    """
    logging.info("Starting hyperparameter optimization with Optuna...")
    
    study_dir = SAVE_DIR / "optuna_study"
    study_dir.mkdir(exist_ok=True)
    
    pruner = optuna.pruners.MedianPruner()
    storage_name = f"sqlite:///{study_dir}/optuna_study.db"
    sampler = TPESampler(seed=42)
    
    study = optuna.create_study(
        direction="maximize",
        pruner=pruner,
        storage=storage_name,
        study_name="deberta_claim_evidence",
        load_if_exists=True,
        sampler=sampler
    )
    
    study.optimize(objective, n_trials=N_TRIALS)
    
    best_trial = study.best_trial
    
    logging.info(f"Using Bayesian optimization with TPE sampler")
    logging.info(f"Best trial: {best_trial.number}")
    logging.info(f"Best value: {best_trial.value}")
    logging.info("Best hyperparameters:")
    
    for param, value in best_trial.params.items():
        logging.info(f"\t{param}: {value}")
    
    best_params = {
        "weight_decay": best_trial.params["weight_decay"],
        "warmup_ratio": best_trial.params["warmup_ratio"],
    }
    
    with open(study_dir / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)
    
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_html(str(study_dir / "optimization_history.html"))
    
    fig = optuna.visualization.plot_param_importances(study)
    fig.write_html(str(study_dir / "param_importances.html"))
    
    fig = optuna.visualization.plot_parallel_coordinate(study)
    fig.write_html(str(study_dir / "parallel_coordinate.html"))
    
    fig = optuna.visualization.plot_contour(study)
    fig.write_html(str(study_dir / "contour.html"))
    
    return best_params

def main():
    """
    Main execution function for the DeBERTa model training pipeline.
    
    1. Runs hyperparameter optimization with Optuna
    2. Trains the final model with the best hyperparameters
    """
    device = get_device()
    logging.info(f"Using device: {device}")

    best_params = run_optuna_experiment()
    
    torch.cuda.empty_cache()

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["query_proj", "key_proj", "value_proj", "dense"],
        init_lora_weights='pissa',
        layers_to_transform=[i for i in range(6, 24)]
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, 
        num_labels=2,
        hidden_dropout_prob=DROPOUT_RATE,
        attention_probs_dropout_prob=DROPOUT_RATE,
    )

    hidden_size = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(hidden_size, hidden_size),
        nn.GELU(),
        nn.LayerNorm(hidden_size),
        nn.Dropout(FF_DROPOUT_RATE),
        nn.Linear(hidden_size, 2)
    )
    model.config.num_labels = 2

    model = get_peft_model(model, peft_config)
    model.to(device)

    train_dataset, dev_dataset, dev_df = load_data(tokenizer, MAX_SEQ_LENGTH)
    
    training_params = {
        'per_device_train_batch_size': BATCH_SIZE,
        'per_device_eval_batch_size': BATCH_SIZE,
        'learning_rate': 5e-6,
        'weight_decay': best_params["weight_decay"],
        'num_train_epochs': 10,
        'warmup_ratio': best_params["warmup_ratio"],
        'lr_scheduler_type': 'cosine',
        'evaluation_strategy': 'steps',
        'eval_steps': 1000,
        'save_strategy': 'steps',
        'save_steps': 1000,
        'save_total_limit': 5,
        'load_best_model_at_end': True,
        'metric_for_best_model': 'MCC',
        'fp16': torch.cuda.is_available(),
        'optim': 'adamw_torch',
        'logging_steps': 100,
        'logging_first_step': True,
        'group_by_length': True,
        'seed': 42,
        'label_smoothing_factor': 0.1,
    }
    
    model_save_path = SAVE_DIR / BASE_MODEL.split('/')[-1]
    train_model(
        model,
        train_dataset,
        dev_dataset,
        model_save_path,
        tokenizer,
        **training_params
    )

if __name__ == "__main__":
    main()

