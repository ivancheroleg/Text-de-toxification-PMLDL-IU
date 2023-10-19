"""
Script to download the data from the github repository and save it in the data/raw folder.
.zip file is removed after unzipping.
"""

import os
import zipfile
import wget

print("Downloading data from github repository...")

# download data
url = 'https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip'
wget.download(url, out="./data/raw/")

# unzip data
with zipfile.ZipFile("./data/raw/filtered_paranmt.zip", 'r') as zip_ref:
    zip_ref.extractall("./data/raw/")

# remove zip file
os.remove("./data/raw/filtered_paranmt.zip")

print("\nDone!")
