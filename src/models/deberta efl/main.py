"""
Basic DeBERTa model for evidence detection with peft training to allow for larger model sizes.
Uses EFL to augment the training data.
"""

import logging
import os
import pandas as pd
import torch
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
    matthews_corrcoef
)
from datasets import Dataset as HFDataset

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
AUG_TRAIN_HIGH_REPLACEMENT_FILE = DATA_DIR / "train_augmented_high_replacement_fraction.csv"
SAVE_DIR = DATA_DIR / "results" / "transformer"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Training parameters
BATCH_SIZE = 8
NUM_EPOCHS = 10
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
DROPOUT_RATE = 0.11
MAX_SEQ_LENGTH = 384
BASE_MODEL = 'microsoft/deberta-v2-xlarge-mnli'


# Optuna parameters
N_TRIALS = 10

def get_device() -> torch.device:
    """Determine the device to use for computations."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def preprocess_dataframe(df):
    """Preprocess dataframe by creating augmented text examples with labels."""
    """
        THIS DOES NOT WORK:
        the reason it doesn't work is because the two similar inputs lead to two different labels
        this means that the model will get confused and not know what to do
    """
    texts = []
    labels = []
    
    # Create inputs and targets
    for _, row in df.iterrows():
        claim = row['Claim']
        evidence = row['Evidence']
        label = row['label']
        label_mapping = {
            1: ("Supporting", 1, "Non-Supporting", 0),
            0: ("Non-Supporting", 1, "Supporting", 0)
        }
        
        claim_label, label_1, claim_label_2, label_2 = label_mapping[label]
        
        formatted_input = f"Claim: {claim}\n\nEvidence: {evidence}\n\nLabel: {claim_label}"
        texts.append(formatted_input)
        labels.append(label_1)
        
        formatted_input_2 = f"Claim: {claim}\n\nEvidence: {evidence}\n\nLabel: {claim_label_2}"
        texts.append(formatted_input_2)
        labels.append(label_2)
        
    # Create new dataframe with processed text and labels
    processed_df = pd.DataFrame({'text': texts, 'labels': labels})
    return processed_df

def tokenize_function(examples, tokenizer):
    """Tokenize preprocessed text examples."""
    # Tokenize the text
    tokenized = tokenizer(
        examples['text'],
        max_length=MAX_SEQ_LENGTH,
        padding=False,
        truncation=True,
    )
    
    # Add labels to the tokenized output
    tokenized['labels'] = examples['labels']
    return tokenized

def convert_to_hf_dataset(dataframe):
    """Convert pandas dataframe to HuggingFace dataset format."""
    return HFDataset.from_pandas(dataframe)

def load_data(tokenizer):
    """Load and prepare the training and development datasets."""
    logging.info("Loading datasets...")
    
    # Load CSV files into pandas dataframes
    train_df = pd.read_csv(TRAIN_FILE)
    dev_df = pd.read_csv(DEV_FILE)

    try:
        train_augmented_df = pd.read_csv(AUG_TRAIN_FILE)
        train_df = pd.concat([train_df, train_augmented_df])
    except Exception as e:
        logging.error(f"Error loading or concatenating augmented training data: {e}")
        raise
    
    # Preprocess dataframes
    train_df = preprocess_dataframe(train_df)
    dev_df = preprocess_dataframe(dev_df)
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Development data shape: {dev_df.shape}")
    
    # Check and report class distribution
    train_positive = (train_df['labels'] == 1).sum()
    train_negative = (train_df['labels'] == 0).sum()
    dev_positive = (dev_df['labels'] == 1).sum()
    dev_negative = (dev_df['labels'] == 0).sum()
    
    print(f"Training data distribution: Positive: {train_positive} ({train_positive/len(train_df)*100:.1f}%), "
          f"Negative: {train_negative} ({train_negative/len(train_df)*100:.1f}%)")
    print(f"Dev data distribution: Positive: {dev_positive} ({dev_positive/len(dev_df)*100:.1f}%), "
                 f"Negative: {dev_negative} ({dev_negative/len(dev_df)*100:.1f}%)")
    
    # Convert to HuggingFace datasets
    train_dataset = convert_to_hf_dataset(train_df)
    dev_dataset = convert_to_hf_dataset(dev_df)
    
    # Apply preprocessing (tokenization)
    train_dataset = train_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        batch_size=1000,
        remove_columns=['text']  # Only remove the text column, keep labels
    )   
    
    dev_dataset = dev_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        batch_size=1000,
        remove_columns=['text']  # Only remove the text column, keep labels
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
    trainer.save_model()
    logging.info(f"Model saved to {output_dir}")

def main():
    """Main execution function."""
    device = get_device()
    logging.info(f"Using device: {device}")

    # Free GPU memory
    torch.cuda.empty_cache()

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1
    )

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, 
        num_labels=2,
        hidden_dropout_prob=DROPOUT_RATE,
        attention_probs_dropout_prob=DROPOUT_RATE,
        ignore_mismatched_sizes=True,
    )
    model = get_peft_model(model, peft_config)
    model.to(device)

    # Load data
    train_dataset, dev_dataset, dev_df = load_data(tokenizer)
    
    # Training parameters with focus on preventing overfitting
    training_params = {
        'per_device_train_batch_size': BATCH_SIZE,
        'per_device_eval_batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'num_train_epochs': NUM_EPOCHS,
        'warmup_ratio': WARMUP_RATIO,
        'lr_scheduler_type': 'cosine',
        'evaluation_strategy': 'steps',
        'eval_steps': 500,
        'save_strategy': 'steps',
        'save_steps': 500,
        'save_total_limit': 5,
        'load_best_model_at_end': True,
        'metric_for_best_model': 'MCC',
        'gradient_accumulation_steps': 1,
        'fp16': torch.cuda.is_available(),
        'optim': 'adamw_torch',
        'logging_steps': 100,
        'logging_first_step': True,
        'group_by_length': True,
        'seed': 42,
        'dataloader_num_workers': 4,
        'label_smoothing_factor': 0.05,
        'max_grad_norm': 1.0,
    }
    
    # Train with default parameters
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