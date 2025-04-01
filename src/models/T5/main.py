import logging
import os
import pandas as pd
import torch
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorForSeq2Seq
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    matthews_corrcoef
)
from datasets import Dataset as HFDataset
from typing import Dict, List, Tuple
import numpy as np

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

# Disable wandb
os.environ['WANDB_DISABLED'] = 'true'

# Path configuration
DATA_DIR = Path(__file__).parent.parent.parent / "data"
TRAIN_FILE = DATA_DIR / "train.csv"
DEV_FILE = DATA_DIR / "dev.csv"
AUG_TRAIN_FILE = DATA_DIR / "train_augmented.csv"
AUG_TRAIN_HIGH_REPLACEMENT_FILE = DATA_DIR / "train_augmented_high_replacement_fraction.csv"
SAVE_DIR = DATA_DIR / "results" / "T5"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Training parameters
BATCH_SIZE = 8
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.05
WARMUP_RATIO = 0.15
MAX_SEQ_LENGTH = 256
MAX_TARGET_LENGTH = 8
BASE_MODEL = 'google/flan-t5-small'

def get_device() -> torch.device:
    """Determine the device to use for computations."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def preprocess_function(examples, tokenizer):
    """Process examples for T5 with proper tokenization."""
    inputs = []
    targets = []
    
    # Create inputs and targets
    for claim, evidence, label in zip(examples['Claim'], examples['Evidence'], examples['label']):
        # Simplify the instruction and make it more direct
        instruction = "Does the evidence support the claim? Answer yes or no."
        formatted_input = f"{instruction}\n\nClaim: {claim}\n\nEvidence: {evidence}"
        inputs.append(formatted_input)
        
        # Simple target format
        target = "yes" if label == 1 else "no"
        targets.append(target)
    
    # Tokenize inputs
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_SEQ_LENGTH,
        padding=False,
        truncation=True,
    )
    
    # Separately tokenize targets - crucial to do this correctly
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=MAX_TARGET_LENGTH,
            padding=False,
            truncation=True,
        )
    
    # Set the labels
    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs

def convert_to_hf_dataset(dataframe):
    """Convert pandas dataframe to HuggingFace dataset format."""
    return HFDataset.from_pandas(dataframe)

def load_data(tokenizer):
    """Load and prepare the training and development datasets."""
    logging.info("Loading datasets...")
    
    # Load CSV files into pandas dataframes
    train_df = pd.read_csv(TRAIN_FILE)
    train_augmented_df = pd.read_csv(AUG_TRAIN_FILE)
    dev_df = pd.read_csv(DEV_FILE)

    train_df = pd.concat([train_df, train_augmented_df])
    
    logging.info(f"Training data shape: {train_df.shape}")
    logging.info(f"Development data shape: {dev_df.shape}")
    
    # Check and report class distribution
    train_positive = (train_df['label'] == 1).sum()
    train_negative = (train_df['label'] == 0).sum()
    dev_positive = (dev_df['label'] == 1).sum()
    dev_negative = (dev_df['label'] == 0).sum()
    
    logging.info(f"Training data distribution: Positive: {train_positive} ({train_positive/len(train_df)*100:.1f}%), "
                 f"Negative: {train_negative} ({train_negative/len(train_df)*100:.1f}%)")
    logging.info(f"Dev data distribution: Positive: {dev_positive} ({dev_positive/len(dev_df)*100:.1f}%), "
                 f"Negative: {dev_negative} ({dev_negative/len(dev_df)*100:.1f}%)")
    
    # Convert to HuggingFace datasets
    train_dataset = convert_to_hf_dataset(train_df)
    dev_dataset = convert_to_hf_dataset(dev_df)
    
    # Apply preprocessing (tokenization)
    train_dataset = train_dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        batch_size=1000,
        remove_columns=['Claim', 'Evidence', 'label']
    )
    
    dev_dataset = dev_dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        batch_size=1000,
        remove_columns=['Claim', 'Evidence', 'label']
    )
    
    # Set format for pytorch
    train_dataset.set_format(type='torch')
    dev_dataset.set_format(type='torch')
    
    return train_dataset, dev_dataset, dev_df

def postprocess_text(preds, labels):
    """Convert model outputs to binary labels based on support/doesn't support."""
    # Normalize predictions and labels
    preds = [pred.strip().lower() for pred in preds]
    labels = [label.strip().lower() for label in labels]
    
    # Debug - check unique values
    unique_preds = set(preds)
    print(f"Unique prediction values: {unique_preds}")
    
    # Map variations of support/doesn't support to binary
    support_variations = [
        "support", "yes", "true", "correct", "valid", "1", "positive"
    ]
    doesnt_support_variations = [
        "doesn't support", "does not support", "doesnt support", "no", "false", 
        "incorrect", "invalid", "0", "negative"
    ]
    
    # Convert to binary
    pred_binary = []
    for pred in preds:
        # Check for explicit support indicators
        if any(variant in pred for variant in support_variations):
            pred_binary.append(1)
        # Check for explicit doesn't support indicators
        elif any(variant in pred for variant in doesnt_support_variations):
            pred_binary.append(0)
        # Default to 0 for anything unclear
        else:
            pred_binary.append(0)
            
    # For labels, we have consistent format from preprocessing
    label_binary = [1 if "yes" in label and "no" not in label else 0 for label in labels]
    
    return pred_binary, label_binary

# Custom Seq2SeqTrainer that uses generate() instead of forward() during evaluation
class T5ClassificationTrainer(Trainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Override to use generate() instead of forward() for evaluation.
        This is needed for text classification with T5.
        """
        # If only computing loss, use parent method which correctly calculates loss
        if prediction_loss_only:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        
        # Extract inputs
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]
        
        # First, calculate loss using forward pass
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
        
        # Then, generate predictions for evaluation metrics
        with torch.no_grad():
            # Generate outputs with improved parameters for diversity
            generated_tokens = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=8,          # Match training_args.generation_max_length
                min_length=1, 
                do_sample=True,        # Enable sampling for diversity
                top_p=0.9,             # Nucleus sampling - consider top 90% probability tokens
                temperature=0.7,       # Lower temperature for more focused but still diverse outputs
                num_return_sequences=1,
                early_stopping=True,
            )
        
        # Return loss, generated tokens, and labels
        return (loss, generated_tokens, labels)

def compute_metrics(eval_pred):
    """Calculate evaluation metrics for text classification."""
    predictions, labels = eval_pred
    
    # Get a tokenizer for decoding
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    
    # Decode predictions to text
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Filter out -100 padding tokens from labels
    filtered_labels = [[l for l in label if l != -100] for label in labels]
    
    # Decode labels to text
    decoded_labels = tokenizer.batch_decode(filtered_labels, skip_special_tokens=True)
    
    # Convert text to binary labels
    pred_binary, label_binary = postprocess_text(decoded_preds, decoded_labels)
    
    # Calculate metrics with more focus on positive class
    accuracy = accuracy_score(label_binary, pred_binary)
    
    # Get more detailed metrics for both classes
    precision, recall, f1, _ = precision_recall_fscore_support(
        label_binary, pred_binary, average=None, zero_division=0
    )
    
    # Weighted metrics
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        label_binary, pred_binary, average='weighted', zero_division=0
    )
    
    # Matthews Correlation Coefficient
    mcc = matthews_corrcoef(label_binary, pred_binary)
    
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
    """Train the T5 model for text classification."""
    logging.info("Starting training...")
    
    # Free up CUDA memory before training
    torch.cuda.empty_cache()
    
    # Create data collator for T5
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding='longest'
    )
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        predict_with_generate=True,
        generation_max_length=3,
        greater_is_better=True,
        **kwargs
    )
    
    # Use custom trainer for better text generation during evaluation
    trainer = T5ClassificationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # Reduced patience
    )
    
    trainer.train()
    trainer.save_model()
    logging.info(f"Model saved to {output_dir}")

# Add a specific debug function to validate tokenization
def debug_tokenization(tokenizer):
    """Test tokenization of yes/no to ensure it works properly."""
    print("\n=== TOKENIZATION DEBUGGING ===")
    
    # Test regular tokenization
    yes_tokens = tokenizer.encode("yes", add_special_tokens=False)
    no_tokens = tokenizer.encode("no", add_special_tokens=False)
    print(f"Regular tokenization:")
    print(f"  'yes' -> {yes_tokens} -> '{tokenizer.decode(yes_tokens)}'")
    print(f"  'no' -> {no_tokens} -> '{tokenizer.decode(no_tokens)}'")
    
    # Test target tokenization 
    with tokenizer.as_target_tokenizer():
        yes_tokens = tokenizer.encode("yes", add_special_tokens=True)
        no_tokens = tokenizer.encode("no", add_special_tokens=True)
        print(f"Target tokenization (with special tokens):")
        print(f"  'yes' -> {yes_tokens} -> '{tokenizer.decode(yes_tokens)}'")
        print(f"  'no' -> {no_tokens} -> '{tokenizer.decode(no_tokens)}'")

def main():
    """Main execution function."""
    device = get_device()
    logging.info(f"Using device: {device}")

    # Free GPU memory
    torch.cuda.empty_cache()

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    
    # Debug tokenization specifically
    debug_tokenization(tokenizer)
    
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
    model.to(device)

    # Load data
    train_dataset, dev_dataset, dev_df = load_data(tokenizer)
    
    # Training parameters with focus on preventing overfitting
    training_params = {
        'per_device_train_batch_size': BATCH_SIZE,
        'per_device_eval_batch_size': BATCH_SIZE * 2,
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'num_train_epochs': NUM_EPOCHS,
        'warmup_ratio': WARMUP_RATIO,
        'lr_scheduler_type': 'cosine',
        'evaluation_strategy': 'steps',
        'eval_steps': 300,
        'save_strategy': 'steps',
        'save_steps': 300,
        'load_best_model_at_end': True,
        'metric_for_best_model': 'MCC',
        'gradient_accumulation_steps': 2,  # Reduced from 4
        'fp16': torch.cuda.is_available(),
        'optim': 'adamw_torch',
        'logging_steps': 100,
        'logging_first_step': True,
        'group_by_length': True,
        'seed': 42,
        'dataloader_num_workers': 4,
        'label_smoothing_factor': 0.1,  # Increased from 0.05
        'max_grad_norm': 1.0,  # Added gradient clipping
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