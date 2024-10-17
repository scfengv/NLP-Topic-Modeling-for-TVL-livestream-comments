import pandas as pd

df = pd.read_json("data/PseudoLabelTrainingSet.json", encoding = "utf-8")
print(df.head())