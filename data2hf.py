import pandas as pd

from utils import *
from argparse import ArgumentParser
from huggingface_hub import HfApi, login
from datasets import Dataset, DatasetDict

parser = ArgumentParser()
parser.add_argument("--layer", dest = "layer")
args = parser.parse_args()

token = "HF TOKEN"
login(token)
api = HfApi()

print("Reading file")
if args.layer == "general":
    df = pd.read_csv("data/GeneralLayer.csv", index_col = False, encoding = "utf-8")
elif args.layer == "game":
    df = pd.read_csv("data/GameLayer.csv", index_col = False, encoding = "utf-8")
elif args.layer == "sentiment":
    df = pd.read_json("data/PseudoLabelTrainingSet.json", encoding = "utf-8")

print("Preprocessing")

train_size = 0.8
if args.layer != "sentiment":
    train_df = df.sample(frac = train_size, random_state = 200)

else:
    n = int(train_size * len(df[df["label"] == "negative"]))
    train_df = PLTSNormalDistribution(df, n)

valid_df = df.drop(train_df.index)

RMEmoji_df = Remove_Emoji(df)
Emoji2Desc_df = Emoji2Desc(df)
RMPunc_df = Remove_Punc(df)

print("Generating Dataset")
train_dataset = Dataset.from_pandas(train_df, preserve_index = False)
valid_dataset = Dataset.from_pandas(valid_df, preserve_index = False)
RMEmoji_dataset = Dataset.from_pandas(RMEmoji_df, preserve_index = False)
Emoji2Desc_dataset = Dataset.from_pandas(Emoji2Desc_df, preserve_index = False)
RMPunc_dataset = Dataset.from_pandas(RMPunc_df, preserve_index = False)

dataset_dict = DatasetDict({
    "train": train_dataset,
    "validation": valid_dataset,
    "RemoveEmoji": RMEmoji_dataset,
    "Emoji2Desc": Emoji2Desc_dataset,
    "RemovePunc": RMPunc_dataset
})

print("Uploading")
hf_username = "scfengv"
dataset_name = f"TVL-{args.layer}-layer-dataset"

repo_id = f"{hf_username}/{dataset_name}"
api.create_repo(repo_id = repo_id, repo_type = "dataset", exist_ok = True)

dataset_dict.push_to_hub(repo_id, token = token)
print("Uploaded")