# Model Details of TVL_GeneralLayerClassifier

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
   - Output layer with 4 classes

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
- **Hours used:** 2hr 56mins

## Model Parameters
- Total parameters: ~102M (estimated)
- All parameters are in 32-bit floating point (F32) format

## Input Processing
- Uses BERT tokenization
- Supports sequences up to 512 tokens

## Output
- 4-class multi-label classification

## Performance Metrics
- Accuracy score: 0.952902
- F1 score (Micro): 0.968717
- F1 score (Macro): 0.970818

## Training Dataset
This model was trained on the [scfengv/TVL-general-layer-dataset](https://huggingface.co/datasets/scfengv/TVL-general-layer-dataset).

## Testing Dataset

- [scfengv/TVL-general-layer-dataset](https://huggingface.co/datasets/scfengv/TVL-general-layer-dataset)
  - validation
  - Remove Emoji
  - Emoji2Desc
  - Remove Punctuation

## Usage

```python
import torch
from transformers import BertForSequenceClassification, BertTokenizer

model = BertForSequenceClassification.from_pretrained("scfengv/TVL_GeneralLayerClassifier")
tokenizer = BertTokenizer.from_pretrained("scfengv/TVL_GeneralLayerClassifier")

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
- This model is specifically designed for TVL general layer classification tasks.
- It's based on the Chinese BERT model, indicating it's optimized for Chinese text.

For more detailed information about the model architecture or usage, please refer to the BERT documentation and the specific fine-tuning process used for this classifier.
