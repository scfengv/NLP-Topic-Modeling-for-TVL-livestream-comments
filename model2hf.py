from argparse import ArgumentParser
from huggingface_hub import login, HfApi, Repository

parser = ArgumentParser()
parser.add_argument("--layer", dest = "layer")
args = parser.parse_args()

token = "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"
login(token)
api = HfApi()

hf_username = "scfengv"

if args.layer == "general":
    layer = "GeneralLayer"
elif args.layer == "game":
    layer = "GameLayer"
elif args.layer == "sentiment":
    layer = "SentimentLayer"

model_name = f"TVL_{layer}Classifier"

repo_url = api.create_repo(repo_id = f"{hf_username}/{model_name}", private = False, exist_ok = True)
repo = Repository(local_dir = f"{layer}/best_model", clone_from = f"{hf_username}/{model_name}")
repo.push_to_hub()