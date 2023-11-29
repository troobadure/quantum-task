import torch
from transformers import pipeline, BertTokenizerFast, BertForTokenClassification
import sys

def classify_entities(text):
    # Load the fine-tuned BERT model
    model = BertForTokenClassification.from_pretrained("C:/Users/vladi/Projects/Python Projects/Interviews/Quantum Task/task1/models/bert_ner_finetuned", local_files_only=True)
    tokenizer = BertTokenizerFast.from_pretrained('C:/Users/vladi/Projects/Python Projects/Interviews/Quantum Task/task1/models/tokenizer', local_files_only=True)
    pipe = pipeline("ner", model=model, tokenizer=tokenizer)

    # Run the model inference
    with torch.no_grad():
        outputs = pipe(text)

    return outputs

if __name__ == "__main__":
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        entities = classify_entities(text)
        print("Entities:", entities)
