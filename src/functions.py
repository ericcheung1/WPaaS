import torch
import pandas as pd

def sentiment_classifier(model, tokenizer, reference_dataframe: pd.DataFrame):
    input = reference_dataframe["comments"].tolist()
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


def format_output(outputs, reference_dataframe):
    """
    """
    results = []
    for output in outputs.logits:
        prediction_indice = torch.argmax(output, dim=-1).item()
        prediction_confidences = torch.softmax(output, dim=-1).detach().cpu().numpy()
        pred_dict = {0: "NEGATIVE", 1: "POSITIVE"}
        predicted_label = pred_dict[int(prediction_indice)]
        results.append({
            "sentiment_classification": predicted_label,
            "sentiment_confidence": prediction_confidences
        })
    
    results_dataframe = pd.DataFrame(results)
    reference_dataframe["sentiment_prediction"] = results_dataframe["sentiment_classification"]
    reference_dataframe["sentiment_confidence"] = results_dataframe["sentiment_confidence"]
    return reference_dataframe
