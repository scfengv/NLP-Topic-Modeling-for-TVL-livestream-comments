# Topic-Modeling-for-TVL-livestream-comments

## Abstract
This study is a collaborative research project with the [Top Volleyball League](https://tvl.ctvba.org.tw) (TVL), aiming to develop a topic model for analyzing comments on TVL’s YouTube live broadcasts. By quantitatively analyzing the discussion topics among the audience, we seek to understand the temporal distribution of topic popularity and capture audience attention, thereby facilitating the subsequent development of more commercial applications. All data used in this study are sourced from the comment data of all TVL live broadcasts in 2022. The data were preprocessed using five different methods to evaluate the model’s performance. The classification model consists of three classifiers designed to categorize primary topics—Chat, Game, Cheer, and Broadcast (hereinafter referred to as the General Layer); secondary topics—further subdividing the "Game" category into Player, Team, Judge, Coach, and Tactics (hereinafter referred to as the Game Layer); and to perform sentiment analysis (hereinafter referred to as the SA Layer).

## Model Description

### Model
- General layer: [scfengv/TVL_GeneralLayerClassifier](https://huggingface.co/scfengv/TVL_GeneralLayerClassifier)
  - Accuracy score: 0.952902
  - F1 score (Micro): 0.968717
  - F1 Score (Macro): 0.970818

- Game layer: [scfengv/TVL_GameLayerClassifier](https://huggingface.co/scfengv/TVL_GameLayerClassifier)
  - Accuracy score: 0.985764
  - F1 score (Micro): 0.993132
  - F1 Score (Macro): 0.993694

### Dataset
- General layer: [scfengv/TVL-general-layer-dataset](https://huggingface.co/datasets/scfengv/TVL-general-layer-dataset)
- Game layer: [scfengv/TVL-game-layer-dataset](https://huggingface.co/datasets/scfengv/TVL-game-layer-dataset)

### Training Details

- **Hardware Type:** NVIDIA Quadro RTX8000
- **Library:** PyTorch
- **Hours used:** 
  - General Layer: 2hr 56mins
  - Game Layer: 2hr 13mins

### Training Hyperparameters

```python
Learning rate: 1e-05
Batch size: 32
Number of epochs: 10
Optimizer: Adam
Loss function: torch.nn.BCEWithLogitsLoss()
```

### Usage
```python
import torch
from transformers import BertForSequenceClassification, BertTokenizer

model = BertForSequenceClassification.from_pretrained("scfengv/TVL_GeneralLayerClassifier")
tokenizer = BertTokenizer.from_pretrained("scfengv/TVL_GeneralLayerClassifier")

text = "Your text here"  ## Please refer to Dataset
inputs = tokenizer(text, return_tensors = "pt", padding = True, truncation = True, max_length = 512)

with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.sigmoid(outputs.logits)

print(predictions)
```