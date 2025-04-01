"""
Basic DeBERTa model for evidence detection with peft training to allow for larger model sizes.
"""

import logging
import os
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
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
# BASE_MODEL = 'microsoft/deberta-v2-xlarge-mnli'

BASE_MODEL = 'microsoft/deberta-v3-large'  # Upgraded model
GRADIENT_ACCUMULATION_STEPS = 4  # Higher gradient accumulation

# Optuna parameters
N_TRIALS = 10

def get_device() -> torch.device:
    """Determine the device to use for computations."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def preprocess_function(examples, tokenizer):
    """Process examples for BERT/DeBERTa classification."""
    # Combine claim and evidence
    inputs = []

    # Create inputs and targets
    for claim, evidence in zip(examples['Claim'], examples['Evidence']):
        formatted_input = f"Claim: {claim}\n\nEvidence: {evidence}"
        inputs.append(formatted_input)        
    
    # Tokenize inputs
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_SEQ_LENGTH,
        padding=False,
        truncation=True,
    )
    
    # Add labels (binary classification)
    model_inputs["labels"] = examples['label']
    return model_inputs

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

    # Save predictions to a CSV file
    predictions_df = pd.DataFrame({'prediction': y_pred})
    predictions_csv_path = os.path.join(output_dir, "predictions.csv")
    predictions_df.to_csv(predictions_csv_path, index=False)
    logging.info(f"Predictions saved to {predictions_csv_path}")
    
    # Plot and save confusion matrix
    cm_save_path = os.path.join(output_dir, "confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, cm_save_path)
    
    trainer.save_model()
    logging.info(f"Model saved to {output_dir}")
    
    return eval_results

def find_learning_rate(model, train_dataset, tokenizer, device, batch_size=8, num_iter=100, 
                      start_lr=1e-7, end_lr=1):
    """
    Runs a learning rate finder to determine the optimal learning rate.
    
    Args:
        model: The model to train
        train_dataset: HuggingFace dataset for training
        tokenizer: Tokenizer for creating batches
        device: Device to run on (cuda/mps/cpu)
        batch_size: Batch size for training
        num_iter: Number of iterations to run
        start_lr: Starting learning rate
        end_lr: Ending learning rate
    
    Returns:
        Suggested learning rate
    """
    logging.info("Running learning rate finder...")
    
    # Create dataloader
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='longest')
    dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        collate_fn=data_collator,
        shuffle=True
    )
    
    # Setup optimizer with very low learning rate
    optimizer = AdamW(model.parameters(), lr=start_lr)
    
    # Calculate learning rate multiplier
    lr_multiplier = (end_lr / start_lr) ** (1 / num_iter)
    
    # Storage for learning rates and losses
    learning_rates = []
    losses = []
    
    # Get initial loss
    model.train()
    
    # Main loop
    smoothed_loss = None
    best_loss = float('inf')
    best_lr = start_lr
    
    # Get an iterator of the dataloader that cycles
    dataloader_iter = iter(dataloader)
    
    pbar = tqdm(range(num_iter))
    for iteration in pbar:
        # Get batch (with cycling)
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)
        
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Record learning rate and loss
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        # Update smoothed loss
        if smoothed_loss is None:
            smoothed_loss = loss.item()
        else:
            smoothed_loss = 0.9 * smoothed_loss + 0.1 * loss.item()
        
        losses.append(smoothed_loss)
        
        # Update progress bar
        pbar.set_description(f"LR: {current_lr:.2e}, Loss: {smoothed_loss:.4f}")
        
        # Check if loss is getting better
        if smoothed_loss < best_loss and iteration > 10:  # Skip the first few iterations
            best_loss = smoothed_loss
            best_lr = current_lr
        
        # Check for divergence (stop if loss is exploding)
        if smoothed_loss > 4 * best_loss or torch.isnan(loss).item():
            logging.info(f"Loss diverging, stopping early at lr={current_lr:.2e}")
            break
        
        # Step optimizer
        optimizer.step()
        
        # Increase learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_multiplier
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(learning_rates, losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate Finder')
    
    # Find the suggested learning rate (where loss is lowest)
    suggested_lr = learning_rates[np.argmin(losses[10:])+10] if len(losses) > 10 else best_lr
    
    # Mark the suggested learning rate
    plt.axvline(x=suggested_lr, color='r', linestyle='--', 
                label=f'Suggested LR: {suggested_lr:.2e}')
    plt.legend()
    
    # Save the plot
    os.makedirs(str(SAVE_DIR), exist_ok=True)
    plt.savefig(os.path.join(SAVE_DIR, 'lr_finder.png'))
    plt.close()
    
    logging.info(f"Learning rate finder suggests: {suggested_lr:.2e}")
    
    # Reset the model and optimizer
    for param in model.parameters():
        param.grad = None
    
    return suggested_lr

def main():
    """Main execution function."""
    device = get_device()
    logging.info(f"Using device: {device}")

    # Free GPU memory
    torch.cuda.empty_cache()

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=16,  # Increased rank
        lora_alpha=32,  # Higher scale
        lora_dropout=0.1,
        target_modules=["query_proj", "key_proj", "value_proj", "dense"],  # Target both attention and FFN
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
    
    # Run learning rate finder to find optimal learning rate
    FIND_LR = True  # Set to False to skip LR finder
    if FIND_LR:
        suggested_lr = find_learning_rate(
            model=model,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            device=device,
            batch_size=BATCH_SIZE,
            num_iter=100
        )
    else:
        suggested_lr = LEARNING_RATE
    
    # Training parameters with focus on preventing overfitting
    training_params = {
        'per_device_train_batch_size': BATCH_SIZE,
        'per_device_eval_batch_size': BATCH_SIZE,
        'learning_rate': suggested_lr,  # Use the suggested learning rate from LR finder
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
        'gradient_accumulation_steps': GRADIENT_ACCUMULATION_STEPS,
        'fp16': device.type == 'cuda',  # Enable fp16 only for CUDA
        'bf16': device.type == 'cuda' and torch.cuda.get_device_capability()[0] >= 8,
        'optim': 'adamw_torch',
        'logging_steps': 100,
        'logging_first_step': True,
        'group_by_length': True,
        'seed': 42,
        'dataloader_num_workers': 4,
        'label_smoothing_factor': 0.1,  # Increased for better regularization
        'max_grad_norm': 1.0,
        'gradient_checkpointing': True,  # Enable gradient checkpointing
    }
    
    # Train with optimized parameters
    model_save_path = SAVE_DIR / BASE_MODEL.split('/')[-1]
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