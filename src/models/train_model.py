from datasets import load_from_disk

from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer

from datasets import load_metric

import numpy as np

DATASET_PATH = "../../data/interim/dataset"
MODEL_SAVE_PATH = "../../models"
MODEL_NAME = "t5-small-fine-tuned"

model_checkpoint = "t5-small"

# AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

prefix = "Translate toxic to non-toxic:"

max_input_length = 128
max_target_length = 128
source_sentence = "toxic"
target_sentence = "non-toxic"


def preprocess_function(examples):
    # Inputs
    inputs = [prefix + example[source_sentence] for example in examples["translation"]]
    targets = [example[target_sentence] for example in examples["translation"]]

    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


metric = load_metric("sacrebleu")

dataset = load_from_disk(DATASET_PATH)

cropped_datasets = dataset
cropped_datasets['train'] = dataset['train'].select(range(25000))
cropped_datasets['validation'] = dataset['validation'].select(range(2500))
cropped_datasets['test'] = dataset['test'].select(range(2500))
tokenized_datasets = cropped_datasets.map(preprocess_function, batched=True)

# Create a model for the pretrained model
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

# Defining the parameters for training
batch_size = 32
model_name = model_checkpoint.split("/")[-1]

args = Seq2SeqTrainingArguments(
    f"{model_name}-finetuned-{source_sentence}-to-{target_sentence}",
    evaluation_strategy="epoch",
    learning_rate=0.01,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=10,
    predict_with_generate=True,
    fp16=True,
    report_to="wandb",
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


# simple postprocessing for text
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


# compute metrics function to pass to trainer
def compute_metrics(eval_preds):
    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)

    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Save the model
trainer.save_model(MODEL_SAVE_PATH + "/" + MODEL_NAME)
