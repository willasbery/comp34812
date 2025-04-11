import logging
import pandas as pd
import json
import numpy as np
import pickle
import time

# Hyperparameter tuning
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# RoBERTa
from transformers import AutoModelForSequenceClassification, RobertaTokenizer, TrainingArguments, Trainer

from src.utils.utils import get_device, prepare_data, calculate_all_metrics, timer
from src.config import config

logger = logging.getLogger(__name__)

train_df = pd.read_csv(config.TRAIN_FILE)
dev_df = pd.read_csv(config.DEV_FILE)
train_aug_df = pd.read_csv(config.AUG_TRAIN_FILE)

with timer("Data preparation", logger):
    train_df, dev_df, train_labels, dev_labels = prepare_data(train_df, train_aug_df, dev_df)
    logger.info(f"Prepared data: {len(train_df)} training samples, {len(dev_df)} validation samples")
    
    
def tokenize_data(data: pd.DataFrame, tokenizer: RobertaTokenizer):
    return tokenizer(
        data['text'].tolist(),
        padding=True,
        truncation=True,
        max_length=384,
        return_tensors='pt'
    )
    
def main():
    logger.info("\n" + "="*70)
    logger.info("EVIDENCE DETECTION ROBERTA MODEL TRAINING")
    logger.info("="*70)
    logger.info("Training on RoBERTa base model")
    
    # Check if GPU is available for NumPy/SciPy operations
    device = get_device()
    logger.info(f"Using device: {device}")
    
    id2label = {0: 'contradiction', 1: 'entailment'}
    label2id = {'contradiction': 0, 'entailment': 1}
    
    tokenizer = RobertaTokenizer.from_pretrained(config.ROBERTA_MODEL)
    
    train_encodings = tokenize_data(train_df, tokenizer)
    dev_encodings = tokenize_data(dev_df, tokenizer)
    
    training_args = TrainingArguments(
        output_dir='roberta-base_trial',
        per_device_train_batch_size=16,
        num_train_epochs=10,
        learning_rate=1.9e-5,
        weight_decay=1.018e-05,
        warmup_ratio=0.0987
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(config.ROBERTA_MODEL, 
                                                               num_labels=2, 
                                                               problem_type='single_label_classification',
                                                               id2label=id2label,
                                                               label2id=label2id)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_encodings,
        eval_dataset=dev_encodings
    )

    trainer.train()
