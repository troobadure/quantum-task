import numpy as np
import pandas as pd
import json

from transformers import AutoModelForTokenClassification, BertTokenizerFast, DataCollatorForTokenClassification
from transformers import pipeline, TrainingArguments, Trainer

import datasets
from datasets import Dataset, DatasetDict

# Install required packages
# !pip install transformers datasets tokenizers seqeval -q
# !pip install accelerate -U

# Define the paths to the CSV files
train_csv = "task1/concat_train.csv"
val_csv = "task1/concat_val.csv"

# Load the CSV files into DataFrames
train_df = pd.read_csv(train_csv)
val_df = pd.read_csv(val_csv)

# Group the DataFrames by 'id' and aggregate 'tokens' and 'ner_tags' into lists
train_grouped = train_df.groupby('id').agg({'tokens': list, 'ner_tags': list}).reset_index()
val_grouped = val_df.groupby('id').agg({'tokens': list, 'ner_tags': list}).reset_index()

# Rename columns for clarity
train_grouped.columns = ['id', 'tokens', 'ner_tags']
val_grouped.columns = ['id', 'tokens', 'ner_tags']

# Convert the DataFrames to a DatasetDict
dataset_dict = DatasetDict({"train": Dataset.from_pandas(train_grouped), 'val': Dataset.from_pandas(val_grouped)})

# Define the tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")

def tokenize_and_align_labels(examples, label_all_tokens=True):
    """
    Function to tokenize and align labels with respect to the tokens. This function is specifically designed for
    this Named Entity Recognition (NER) task since alignment of the labels is necessary after tokenization.

    Parameters:
    - examples (dict): A dictionary containing the tokens and the corresponding NER tags.
                       - "tokens": list of words in a sentence.
                       - "ner_tags": list of corresponding entity tags for each word.
    - label_all_tokens (bool): A flag to indicate whether all tokens should have labels.
                               If False, only the first token of a word will have a label,
                               the other tokens (subwords) corresponding to the same word will be assigned -100.

    Returns:
    - tokenized_inputs (dict): A dictionary containing the tokenized inputs and the corresponding labels aligned with the tokens.
    """
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Tokenize and align labels for the datasets
tokenized_datasets = dataset_dict.map(tokenize_and_align_labels, batched=True)

# Define the model with 3 labels for I-MNT, B-MNT and O
bert_model = AutoModelForTokenClassification.from_pretrained("bert-base-cased", num_labels=3)

# Define the label list
label_list = ['O', 'B-MNT', 'I-MNT']

# Define the data collator
data_collator = DataCollatorForTokenClassification(tokenizer)

# Define the evaluation metric
seqeval_metric = datasets.load_metric("seqeval")

def compute_metrics(eval_preds):
    """
    Function to compute the evaluation metrics for Named Entity Recognition (NER) tasks.
    The function computes precision, recall, F1 score and accuracy.

    Parameters:
    - eval_preds (tuple): A tuple containing the predicted logits and the true labels.

    Returns:
    - A dictionary containing the precision, recall, F1 score and accuracy.
    """
    pred_logits, labels = eval_preds
    pred_logits = np.argmax(pred_logits, axis=2)
    predictions = [
        [label_list[eval_preds] for (eval_preds, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(pred_logits, labels)
    ]
    true_labels = [
        [label_list[l] for (eval_preds, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(pred_logits, labels)
    ]
    results = seqeval_metric.compute(predictions=predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# Define the training arguments
args = TrainingArguments(
    "task1/models/bert-ner",
    evaluation_strategy="epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    use_cpu=True
)

# Define the trainer
trainer = Trainer(
    bert_model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["val"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save the fine-tuned model
bert_model.save_pretrained("task1/models/bert_ner_finetuned")
tokenizer.save_pretrained("task1/models/tokenizer")

# Define the id2label and label2id mappings
id2label = {str(i): label for i, label in enumerate(label_list)}
label2id = {label: str(i) for i, label in enumerate(label_list)}

# Update the config file with the id2label and label2id mappings
config = json.load(open("task1/models/bert_ner_finetuned/config.json"))
config["id2label"] = id2label
config["label2id"] = label2id
json.dump(config, open("task1/models/bert_ner_finetuned/config.json", "w"))

# Load the fine-tuned model
model_fine_tuned = AutoModelForTokenClassification.from_pretrained("task1/models/bert_ner_finetuned")

# Create a pipeline for named entity recognition
pipe = pipeline("ner", model=model_fine_tuned, tokenizer=tokenizer)

# Sample usage of the pipeline
example = 'Mount Everest is the highest peak in the world.'

ner_results = pipe(example)
print(ner_results)
