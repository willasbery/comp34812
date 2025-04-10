---
{}
---
language: en
license: cc-by-4.0
tags:
- text-classification
repo: https://github.com/willasbery/comp34812

---

# Model Card for m17832wa-j08328hd-ED

<!-- Provide a quick summary of what the model is/does. -->

This is a classification model that was trained to
      detect whether the evidence provided supports the claim.


## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

This model is based upon the `microsoft/deberta-v3-large` model that was fine-tuned
      on 29k pairs of texts.

- **Developed by:** Harvey Dennis and William Asbery
- **Language(s):** English
- **Model type:** Supervised
- **Model architecture:** Transformers
- **Finetuned from model [optional]:** deberta-v3-large

### Model Resources

<!-- Provide links where applicable. -->

- **Repository:** https://huggingface.co/microsoft/deberta-v3-base
- **Paper or documentation:** https://arxiv.org/pdf/2111.09543

## Training Details

### Training Data

<!-- This is a short stub of information on the training data that was used, and documentation related to data pre-processing or additional filtering (if applicable). -->

29K pairs of texts drawn from emails, news articles and blog posts (21.5K are original and 6.5K are augmented from the original texts).

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Training Hyperparameters

<!-- This is a summary of the values of hyperparameters used in training the model. -->


      - learning_rate: 5e-05
      - weight_decay: 0.03
      - warmup_ratio: 0.11
      - dropout_rate: 0.05
      - max_seq_length: 512
      - batch_size: 8
      - seed: 42
      - num_epochs: 8 (early stopping enabled)

#### Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->


      - overall training time (early stopping occurred): 1.5 hours
      - duration per training epoch: 30 minutes
      - model size: 10MB

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data & Metrics

#### Testing Data

<!-- This should describe any evaluation data used (e.g., the development/validation set provided). -->

The development dataset provided, amounting to 6K pairs.

#### Metrics

<!-- These are the evaluation metrics being used. -->


      - Weighted Precision
      - Weighted Recall
      - Weighted F1-score
      - Accuracy
      - MCC

### Results


    The model obtained:
      - Weighted Precision of 89.6%
      - Weighted Recall of 89.3%
      - Weighted F1-score of 89.4%
      - Accuracy of 89.3%
      - MCC of 0.74
    

## Technical Specifications

### Hardware


      - RAM: at least 16 GB
      - Storage: at least 2GB,
      - GPU: P100

### Software


      - Transformers 4.47.0
      - Pytorch 2.5.1+cu121
      - PEFT 0.14.0
      - Optuna 4.2.1
      - Scikit-learn 1.2.2
    

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

Any inputs (concatenation of two sequences) longer than
      512 tokens will be truncated by the model.

## Additional Information

<!-- Any other information that would be useful for other people to know. -->

The hyperparameters for both the LoRA optimiser and model were determined by experimentation
      with different values using Optuna and a TPE sampler.
