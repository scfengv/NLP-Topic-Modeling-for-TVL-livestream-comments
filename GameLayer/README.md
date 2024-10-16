# Model Details of TVL_GameLayerClassifier

## Base Model
This model is fine-tuned from [google-bert/bert-base-chinese](https://huggingface.co/google-bert/bert-base-chinese).

## Model Architecture
- **Type**: BERT-based text classification model
- **Hidden Size**: 768
- **Number of Layers**: 12
- **Number of Attention Heads**: 12
- **Intermediate Size**: 3072
- **Max Sequence Length**: 512
- **Vocabulary Size**: 21,128

## Key Components
1. **Embeddings**
   - Word Embeddings
   - Position Embeddings
   - Token Type Embeddings
   - Layer Normalization

2. **Encoder**
   - 12 layers of:
     - Self-Attention Mechanism
     - Intermediate Dense Layer
     - Output Dense Layer
     - Layer Normalization

3. **Pooler**
   - Dense layer for sentence representation

4. **Classifier**
   - Output layer with 5 classes

## Training Hyperparameters

The model was trained using the following hyperparameters:

```
Learning rate: 1e-05
Batch size: 32
Number of epochs: 10
Optimizer: Adam
Loss function: torch.nn.BCEWithLogitsLoss()
```

## Training Infrastructure

- **Hardware Type:** NVIDIA Quadro RTX8000
- **Library:** PyTorch
- **Hours used:** 2hr 13mins

## Model Parameters
- Total parameters: ~102M (estimated)
- All parameters are in 32-bit floating point (F32) format

## Input Processing
- Uses BERT tokenization
- Supports sequences up to 512 tokens

## Output
- 5-class classification

## Performance Metrics
- Accuracy score: 0.985764
- F1 score (Micro): 0.993132
- F1 score (Macro): 0.993694

## Training Dataset
This model was trained on the [scfengv/TVL-game-layer-dataset](https://huggingface.co/datasets/scfengv/TVL-game-layer-dataset).

## Testing Dataset

- [scfengv/TVL-game-layer-dataset](https://huggingface.co/datasets/scfengv/TVL-game-layer-dataset)
  - validation
  - Remove Emoji
  - Emoji2Desc
  - Remove Punctuation

## Usage

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

## Additional Notes
- This model is specifically designed for TVL Game layer classification tasks.
- It's based on the Chinese BERT model, indicating it's optimized for Chinese text.

For more detailed information about the model architecture or usage, please refer to the BERT documentation and the specific fine-tuning process used for this classifier.
