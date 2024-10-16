---
license: mit
datasets:
- scfengv/TVL-game-layer-dataset
language:
- zh
metrics:
- accuracy
base_model:
- google-bert/bert-base-chinese
pipeline_tag: text-classification
tags:
- multi-label

model-index:
  - name: scfengv/TVL_GameLayerClassifier
    results:
      - task:
          type: multi-label text-classification
        dataset:
          name: scfengv/TVL-game-layer-dataset
          type: scfengv/TVL-game-layer-dataset
        metrics:
          - name: Accuracy
            type: Accuracy
            value: 0.985764
            
          - name: F1 score (Micro)
            type: F1 score (Micro)
            value: 0.993132

          - name: F1 score (Macro)
            type: F1 score (Macro)
            value: 0.993694
---
# Model Card for Model ID

<!-- Provide a quick summary of what the model is/does. -->

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->


- **Developed by:** [scfengv](https://huggingface.co/scfengv)
- **Model type:** BERT Multi-label Text Classification
- **Language:** Chinese (Zh)
- **Finetuned from model:** [google-bert/bert-base-chinese](https://huggingface.co/google-bert/bert-base-chinese)

### Model Sources

<!-- Provide the basic links for the model. -->

- **Repository:** [scfengv/NLP-Topic-Modeling-for-TVL-livestream-comments](https://github.com/scfengv/NLP-Topic-Modeling-for-TVL-livestream-comments)

## How to Get Started with the Model

Use the code below to get started with the model.

```python
import torch
from transformers import BertForSequenceClassification, BertTokenizer

model = BertForSequenceClassification.from_pretrained("scfengv/TVL_GameLayerClassifier")
tokenizer = BertTokenizer.from_pretrained("scfengv/TVL_GameLayerClassifier")

# Prepare your text
text = "Your text here" ## Please refer to Dataset
inputs = tokenizer(text, return_tensors = "pt", padding = True, truncation = True, max_length = 512)

# Make prediction
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.sigmoid(outputs.logits)

# Print predictions
print(predictions)
```

## Training Details

- **Hardware Type:** NVIDIA Quadro RTX8000
- **Library:** PyTorch
- **Hours used:** 2hr 13mins

### Training Data

- [scfengv/TVL-game-layer-dataset](https://huggingface.co/datasets/scfengv/TVL-game-layer-dataset)
  - train


### Training Hyperparameters

The model was trained using the following hyperparameters:

```
Learning rate: 1e-05
Batch size: 32
Number of epochs: 10
Optimizer: Adam
Loss function: torch.nn.BCEWithLogitsLoss()
```

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

- [scfengv/TVL-game-layer-dataset](https://huggingface.co/datasets/scfengv/TVL-game-layer-dataset)
  - validation
  - Remove Emoji
  - Emoji2Desc
  - Remove Punctuation

### Results (validation)

- Accuracy score: 0.985764
- F1 score (Micro): 0.993132
- F1 Score (Macro): 0.993694