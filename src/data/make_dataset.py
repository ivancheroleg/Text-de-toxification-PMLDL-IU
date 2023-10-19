"""
Script to prepare the dataset with usage of the raw data and Hugging Face's library.
"""

import pandas as pd
from tqdm import tqdm
from datasets import DatasetDict, Dataset, Value

"""
Here you can specify the path to the data (should be .tsv), name of the dataset and the train/validation/test ratios.
"""
dataset_name = "dataset"
train_ratio = 0.9
val_ratio = 0.05
test_ratio = 0.05
filepath = "./data/raw/filtered.tsv"

df = pd.read_csv(filepath, sep='\t')

# delete unnamed column and columns for similarity and length
df = df.drop(df.columns[0], axis=1)
df = df.drop(df.columns[2], axis=1)
df = df.drop(df.columns[2], axis=1)

# The data is not consistent. Reference is not always less toxic, so we need to change positions pairwise for values
# and sentences.
temp_df = df.copy()

df.loc[temp_df.ref_tox > temp_df.trn_tox, 'reference'] = temp_df.loc[temp_df.ref_tox > temp_df.trn_tox, 'translation']
df.loc[temp_df.ref_tox > temp_df.trn_tox, 'translation'] = temp_df.loc[temp_df.ref_tox > temp_df.trn_tox, 'reference']
df.loc[temp_df.ref_tox > temp_df.trn_tox, 'trn_tox'] = temp_df.loc[temp_df.ref_tox > temp_df.trn_tox, 'ref_tox']
df.loc[temp_df.ref_tox > temp_df.trn_tox, 'ref_tox'] = temp_df.loc[temp_df.ref_tox > temp_df.trn_tox, 'trn_tox']

sentences = df.iloc[:, 0:2]
sentences.columns = ["non-toxic", "toxic"]

# The schema of the dataset
schema = {
    "train": {
        "translation": {
            "non-toxic": Value("string"),
            "toxic": Value("string"),
        },
    },
    "validation": {
        "translation": {
            "non-toxic": Value("string"),
            "toxic": Value("string"),
        },
    },
    "test": {
        "translation": {
            "non-toxic": Value("string"),
            "toxic": Value("string"),
        },
    },
}

train_len = int(len(sentences) * train_ratio)
val_len = int(len(sentences) * val_ratio)
test_len = int(len(sentences) * test_ratio)

# To get same structure as in wmt16 dataset, I will use pairwise split.
train_pairs = []
val_pairs = []
test_pairs = []

# Create dataset dict
dataset = DatasetDict(schema)

# Add pairs to lists
print("Creating dataset...")

for i in tqdm(range(train_len)):
    train_pairs.append({"non-toxic": sentences.iloc[i, 0], "toxic": sentences.iloc[i, 1]})

for i in tqdm(range(train_len, train_len + val_len)):
    val_pairs.append({"non-toxic": sentences.iloc[i, 0], "toxic": sentences.iloc[i, 1]})

for i in tqdm(range(train_len + val_len, train_len + val_len + test_len)):
    test_pairs.append({"non-toxic": sentences.iloc[i, 0], "toxic": sentences.iloc[i, 1]})

# Add pairs through dataset dict
dataset["train"] = Dataset.from_dict({"translation": train_pairs})
dataset["validation"] = Dataset.from_dict({"translation": val_pairs})
dataset["test"] = Dataset.from_dict({"translation": test_pairs})

path = "./data/interim/" + dataset_name

dataset.save_to_disk(path)

print("Done!")