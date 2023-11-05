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

To train the model you can apply script ```train_model.py```. Requires manually setting the parameters in the script.

## Dataset
Dataset is located in ```data\interim``` folder. The dataset was made with [Hugging Face tutorial](https://huggingface.co/docs/datasets/create_dataset). It transforms the original ```.tsv``` file into this structure with toxic and non-toxic sentences as a translation pair:
```
DatasetDict({
    train: Dataset({
        features: ['translation'],
        num_rows: 502214
    })
    validation: Dataset({
        features: ['translation'],
        num_rows: 27900
    })
    test: Dataset({
        features: ['translation'],
        num_rows: 27900
    })
})
```

## Models
Unfortunately, I am limited in LFS storage, so I can't upload models to the repository. But you can download them from the following links:
- [T5-small](https://disk.yandex.ru/d/IKHRiOuM7Jwa1A)

## Structure
```
text-detoxification
├── README.md # The top-level README
│
├── requirements.txt # The requirements file for reproducing the analysis environment
│
├── data 
│   ├── interim  # Intermediate data that has been transformed.
│   └── raw      # The original, immutable data.
│
├── models       # Trained and serialized models, model predictions, or model summaries (LFS limited) 
│
├── notebooks    #  Jupyter notebooks.  
│   ├── 1.0-initial-data-exploration.ipynb # Initial data exploration
│   ├── 2.0-preprocessing-making-dataset.ipynb # Preprocessing and making dataset
│   ├── 3.0-t5-small-model.ipynb # T5-small model training and evaluation
│   └── 4.0-evaluation.ipynb # Evaluation of the model     
│ 
├── references # Data dictionaries, manuals, and all other explanatory materials.
│
├── reports
│   ├── figures  # Generated graphics and figures to be used in reporting
│   ├── solution-building.md # The solution building process 
│   └── final-solution.md # The final solution report
│
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