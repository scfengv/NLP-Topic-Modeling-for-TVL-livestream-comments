import gc
import json
import torch
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import pipeline

class PseudoLabelGenerator():
    def __init__(self, models, data):
        self.models = models
        self.data = data  ## List
        self.labels = {}
        self.scores = {}
        
    def classifier(self):
        for model in tqdm(self.models):
            print(f"=== Processing with model: {model}... ===")
            if model != "IDEA-CCNL/Erlangshen-MegatronBert-1.3B-Sentiment":
                sentiment = ["positive", "negative"]
                id2label = {0: "positive", 1: "negative"}
                label2id = {"positive": 0, "negative": 1}
                classifier = pipeline("zero-shot-classification", model, device = 0, id2label = id2label, label2id = label2id)
            else:
                classifier = pipeline("sentiment-analysis", model, device = 0)

            model_labels = []
            model_scores = []

            print("=== Labeling... ===")
            for sentence in tqdm(self.data):
                if model != "IDEA-CCNL/Erlangshen-MegatronBert-1.3B-Sentiment":
                    output = classifier(sentence, candidate_labels = sentiment, return_all_scores = True)
                    index = output["scores"].index(max(output["scores"]))
                    label = output["labels"][index]
                    score = output["scores"][index]

                else:
                    output = classifier(sentence)
                    score = output[0]["score"]
                    label = output[0]["label"]

                model_labels.append(label.lower())
                model_scores.append(score)

            self.labels[model] = model_labels
            self.scores[model] = model_scores

            del classifier
            torch.cuda.empty_cache()
            gc.collect()
    
    def Generate_PLTS(self):
        sentence, sentiment, score = [], [], []
        print("=== Generating PLTS... ===")
        for i in tqdm(range(len(self.data))):
            labels = []
            scores = []

            for model in self.models:
                labels.append(self.labels[model][i])
                scores.append(self.scores[model][i])

            if all(label == labels[0] for label in labels):
                sentence.append(self.data[i])
                sentiment.append(labels[0])
                score.append(np.mean(scores))
        
        return sentence, sentiment, score
    
def save_json(sentences, sentiments, scores, filename):
    data = []
    for sentence, sentiment, score in zip(sentences, sentiments, scores):
        data.append({
            "text": sentence, "label": sentiment, "score": score
        })
    with open(f"data/{filename}.json", "w", encoding = "utf-8") as f:
        json.dump(data, f, indent = 4)
    
    
def main():
    warnings.simplefilter("ignore")
    model0 = "joeddav/xlm-roberta-large-xnli"
    model1 = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    model2 = "IDEA-CCNL/Erlangshen-MegatronBert-1.3B-Sentiment"

    models = [model0, model1, model2]
    df = pd.read_csv("data/RawData.csv", index_col = False)
    data = list(df["text"])
    PseudoLabelGenerator_ = PseudoLabelGenerator(models, data)
    PseudoLabelGenerator_.classifier()
    sentence, sentiment, score = PseudoLabelGenerator_.Generate_PLTS()
    save_json(sentence, sentiment, score, "PseudoLabelTrainingSet")

if __name__ == "__main__":
    main()