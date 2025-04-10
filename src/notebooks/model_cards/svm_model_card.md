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

This model is an SVM classifier that was trained on 29k pairs of texts.

- **Developed by:** Harvey Dennis and William Asbery
- **Language(s):** English
- **Model type:** Supervised
- **Model architecture:** SVM
- **Finetuned from model [optional]:** [More Information Needed]

### Model Resources

<!-- Provide links where applicable. -->

- **Repository:** [More Information Needed]
- **Paper or documentation:** [More Information Needed]

## Training Details

### Training Data

<!-- This is a short stub of information on the training data that was used, and documentation related to data pre-processing or additional filtering (if applicable). -->

29K pairs of texts drawn from emails, news articles and blog posts (21.5K are original and 6.5K are augmented from the original texts).

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Training Hyperparameters

<!-- This is a summary of the values of hyperparameters used in training the model. -->


      - embedding_dim: 300
      - ngram: 2
      - pca_components (SVD): 540
      - vocab_size: 12,000
      - C: 1.96
      - kernel: rbf
      - use_tf_idf_weightings: True
      - gamma: scale
      - seed: 42

#### Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->


      - overall training time: 3 hours
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
      - Weighted Precision of 82.7%
      - Weighted Recall of 82.5%
      - Weighted F1-score of 82.6%
      - Accuracy of 82.5%
      - MCC of 0.57
    

## Technical Specifications

### Hardware


      - RAM: at least 16 GB recommended
      - Storage: minimal (model size is relatively small),
      - CPU: standard CPU sufficient (no GPU required)

### Software


      - Scikit-learn 1.3.2
      - Optuna 4.2.1
      - NLTK 3.9.1
      - Scipy 1.7.0
      - Gensim 4.3.3
      - XGBoost 3.0.0
      - Googletrans 4.0.2
    

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

Any inputs (concatenation of two sequences) longer than
      512 tokens will be truncated by the model.

## Additional Information

<!-- Any other information that would be useful for other people to know. -->

The hyperparameters for both the LoRA optimiser and model were determined by experimentation
      with different values using Optuna and a TPE sampler.
