from functions import sentiment_classifier
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

local_distilbert = "./distilbert_model"

tokenizer = AutoTokenizer.from_pretrained(
    local_distilbert,
    local_files_only=True
)
model = AutoModelForSequenceClassification.from_pretrained(
    local_distilbert,
    local_files_only=True,
    dtype=torch.float16,
    attn_implementation="sdpa"
)

text = ["YURRR", "Hello how are you?", "Wow this is so delish"]

outputs = sentiment_classifier(model=model, tokenizer=tokenizer, input=text)

print(outputs)
# print(len(outputs))
for output in outputs.logits:
    print(output)
    predicted_class_id = torch.argmax(output, dim=-1).item()
    predicted_label = model.config.id2label[predicted_class_id]
    print(f"Predicted label: {predicted_label} confidences: {torch.argmax(torch.softmax(output, dim=-1))}")