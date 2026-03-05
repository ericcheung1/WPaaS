import torch
import pandas as pd

def sentiment_classifier(model, tokenizer, input: list):
    inputs = tokenizer(input, return_tensors="pt", padding=True, Truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)
    
    return outputs

def parse_payload(input_payload: dict) -> pd.DataFrame:
    """
    """
    user_id = input_payload["user_id"]
    comments = input_payload["comments"]
    
    reference_table = {
        "user_id": user_id,
        "comments": comments
    }

    reference_dataframe = pd.DataFrame(reference_table)
    return reference_dataframe


def format_output(outputs, model, reference_dataframe):
    for output in outputs.logits:
        print(output)
        predicted_class_id = torch.argmax(output, dim=-1).item()
        predicted_label = model.config.id2label[predicted_class_id]
        print(f"Predicted label: {predicted_label} confidences: {torch.argmax(torch.softmax(output, dim=-1))}")