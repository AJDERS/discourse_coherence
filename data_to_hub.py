import os
import pandas as pd
from datasets import Dataset, DatasetDict
from huggingface_hub import login, create_repo, HfApi

def main():
    # Login to HuggingFace Hub
    api = HfApi(token=os.environ.get("HUGGINGFACE_HUB_TOKEN", True))

    # Create a repository on HuggingFace Hub
    api.create_repo(
        "ajders/ddisco",
        repo_type="dataset",
        exist_ok=True,
        token=os.environ.get("HUGGINGFACE_HUB_TOKEN", True)
    )

    # Upload data readme to HuggingFace Hub
    api.upload_file(
        path_or_fileobj="HFREADME.md",
        path_in_repo="README.md",
        repo_id="ajders/ddisco",
        repo_type="dataset",
        token=os.environ.get("HUGGINGFACE_HUB_TOKEN", True)
    )

    # Load data and push to hub

    # Read the data
    train_df = pd.read_csv("ddisco/ddisco.train.tsv", sep="\t")
    test_df = pd.read_csv("ddisco/ddisco.test.tsv", sep="\t")

    # Create a DatasetDict
    ds_dict = {
        'train' : Dataset.from_pandas(train_df),
        'test' : Dataset.from_pandas(test_df)
    }

    ds = DatasetDict(ds_dict)
    ds.push_to_hub("ajders/ddisco", token=os.environ.get("HUGGINGFACE_HUB_TOKEN", True))


if __name__ == "__main__":
    main()