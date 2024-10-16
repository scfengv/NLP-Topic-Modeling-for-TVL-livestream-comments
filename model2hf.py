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
    dataset_name = "TVL_GeneralLayerClassifier"
elif args.layer == "game":
    dataset_name = "TVL_GameLayerClassifier"

repo_url = api.create_repo(repo_id = f"{hf_username}/{dataset_name}", private = False, exist_ok = True)
repo = Repository(local_dir = "GeneralLayer/best_model", clone_from = f"{hf_username}/{dataset_name}")
repo.push_to_hub()