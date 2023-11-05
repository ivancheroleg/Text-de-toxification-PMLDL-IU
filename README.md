# Practical Machine Learning and Deep Learning Assignment 1.
This repository includes models, descriptions, references, notebooks on PMLDL Assignment 1 about Text De-toxification.

## Author
- Ivan Chernakov
- BS21-DS-02
- i.chernakov@innopolis.university

## Requirements
Run the following command to install all the required packages:
```pip install -r requirements.txt``` if needed.

## Getting Started
To download data run the following command:
```python .\src\data\download_data.py```

To make dataset run the following command:
```python .\src\data\make_dataset.py```
## Models
Unfortunately, I am limited in LFS storage, so I can't upload models to the repository. But you can download them from the following links:
- [T5-small](https://disk.yandex.ru/d/IKHRiOuM7Jwa1A)

## Structure
```
text-detoxification
├── README.md # The top-level README
│
├── data 
│   ├── interim  # Intermediate data that has been transformed.
│   └── raw      # The original, immutable data.
│
├── notebooks    #  Jupyter notebooks.         
│ 
├── references
│
├── reports
│   └── figures  # Generated graphics and figures to be used in reporting
│
├── requirements.txt # The requirements file for reproducing the analysis environment
└── src                 # Source code for use in this 
    │                 
    ├── data    
    │   ├── download_data.py        
    │   └── make_dataset.py
    │
    └── models          # Scripts to train models and then use trained models to make predictions
        ├── predict_model.py
        └── train_model.py
```