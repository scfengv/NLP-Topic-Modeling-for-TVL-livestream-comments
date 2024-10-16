# -*- coding: utf-8 -*-

import torch
import pandas as pd
import numpy as np
import pandas as pd
from utils import *
from sklearn import metrics
from datasets import load_dataset
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification

parser = ArgumentParser()
parser.add_argument("--param_path", dest = "path", help = "path to parameter json file", default = None)
parser.add_argument("--layer", dest = "layer")
args = parser.parse_args()
params = load_json(args.path)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.layer == "general":
        ds = load_dataset("scfengv/TVL-general-layer-dataset")
        ds = ds.map(lambda datasets: {"target": [datasets[col] for col in params["General_col"]]},
                    remove_columns = [col for col in ds["train"].column_names if col not in ["text", "target"]])

    elif args.layer == "game":
        ds = load_dataset("scfengv/TVL-game-layer-dataset")
        ds = ds.map(lambda datasets: {"target": [datasets[col] for col in params["Game_col"]]},
                    remove_columns = [col for col in ds["train"].column_names if col not in ["text", "target"]])
        
    tokenizer = BertTokenizer.from_pretrained(params["Tokenizer"])

    train_dataset = ds["train"]
    valid_dataset = ds["validation"]
    training_set = CustomDataset(train_dataset, tokenizer, params["MAX_LEN"])
    validation_set = CustomDataset(valid_dataset, tokenizer, params["MAX_LEN"])

    train_params = {"batch_size": params["TRAIN_BATCH_SIZE"],
                    "shuffle": True,
                    "num_workers": 0
                    }

    test_params = {"batch_size": params["VALID_BATCH_SIZE"],
                    "shuffle": False,
                    "num_workers": 0
                    }

    training_loader = DataLoader(training_set, **train_params)
    validation_loader = DataLoader(validation_set, **test_params)

    if args.layer == "general":
        label2id = params["label2id_general"]
    elif args.layer == "game":
        label2id = params["label2id_game"]

    model = BertForSequenceClassification.from_pretrained(
        params["Model"], num_labels = len(label2id),
        problem_type = "multi_label_classification",
        label2id = label2id
    )
    model.to(device)

    optimizer = torch.optim.Adam(params =  model.parameters(), lr = params["LEARNING_RATE"])

    val_targets = []
    val_outputs = []

    checkpoint_path = f"{args.layer}/current_checkpoint.pt"
    best_model_path = f"{args.layer}/best_model"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok = True)
    os.makedirs(best_model_path, exist_ok = True)

    trained_model = train_model(1, params["EPOCHS"], np.inf, 
                                training_loader, validation_loader, 
                                model, optimizer, 
                                checkpoint_path, best_model_path, 
                                val_targets, val_outputs, tokenizer)
    
    val_preds = (np.array(val_outputs) > 0.5).astype(int)
    accuracy = metrics.accuracy_score(val_targets, val_preds)
    f1_score_micro = metrics.f1_score(val_targets, val_preds, average = "micro")
    f1_score_macro = metrics.f1_score(val_targets, val_preds, average = "macro")
    print(f"Accuracy Score = {accuracy}")
    print(f"F1 Score (Micro) = {f1_score_micro}")
    print(f"F1 Score (Macro) = {f1_score_macro}")

if __name__ == "__main__":
    main()
