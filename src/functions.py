import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def sentiment_classifier(model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer, input: list):
    inputs = tokenizer(input, return_tensors="pt", padding=True, Truncation=True).to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    return outputs


