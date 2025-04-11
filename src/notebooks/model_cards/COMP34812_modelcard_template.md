{{ card_data }}

---

# Model Card for {{ model_id | default("My Model", true) }}

<!-- Provide a quick summary of what the model is/does. -->

{{ model_summary | default("", true) }}


## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

{{ model_description | default("", true) }}

- **Developed by:** {{ developers | default("[More Information Needed]", true)}}
- **Language(s):** {{ language | default("[More Information Needed]", true)}}
- **Model type:** {{ model_type | default("[More Information Needed]", true)}}
- **Model architecture:** {{ model_architecture | default("[More Information Needed]", true)}}
- **Finetuned from model [optional]:** {{ base_model | default("[More Information Needed]", true)}}

### Model Resources

<!-- Provide links where applicable. -->

- **Repository:** {{ base_model_repo | default("[More Information Needed]", true)}}
- **Paper or documentation:** {{ base_model_paper | default("[More Information Needed]", true)}}

## Training Details

### Training Data

<!-- This is a short stub of information on the training data that was used, and documentation related to data pre-processing or additional filtering (if applicable). -->

{{ training_data | default("[More Information Needed]", true)}}

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Training Hyperparameters

<!-- This is a summary of the values of hyperparameters used in training the model. -->

{{ hyperparameters | default("[More Information Needed]", true)}}

#### Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->

{{ speeds_sizes_times | default("[More Information Needed]", true)}}

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data & Metrics

#### Testing Data

<!-- This should describe any evaluation data used (e.g., the development/validation set provided). -->

{{ testing_data | default("[More Information Needed]", true)}}

#### Metrics

<!-- These are the evaluation metrics being used. -->

{{ testing_metrics | default("[More Information Needed]", true)}}

### Results

{{ results | default("[More Information Needed]", true)}}

## Technical Specifications

### Hardware

{{ hardware_requirements | default("[More Information Needed]", true)}}

### Software

{{ software | default("[More Information Needed]", true)}}

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

{{ bias_risks_limitations | default("[More Information Needed]", true)}}

## Additional Information

<!-- Any other information that would be useful for other people to know. -->

{{ additional_information | default("[More Information Needed]", true)}}

