import torch
from common.schemas import comments_all

text_index = 0
id_index = 1
pred_result_index = 0

def process_inputs(in_payload: comments_all):
    """
    """
    comments = []

    for item in in_payload.comments:
        comment_pair = (item.comment.lower().strip(), item.comment_id)
        comments.append(comment_pair)

    if in_payload.contains_id == False:
        inputs = [x[text_index] for x in comments]
        return inputs, []
    
    else:
        ids = [x[id_index] for x in comments]
        inputs = [x[text_index] for x in comments]
        return inputs, ids


def sentiment_classifier(model, tokenizer, input, ids=[]):
    """
    """
    
    inputs = tokenizer(input, return_tensors="pt", padding=True, Truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)

    if not ids:
        return outputs, []
    elif ids:
        return outputs, ids


def format_output(outputs):
    """
    """
    results = []
    result_dict = {0: "NEGATIVE", 1: "POSITIVE"}

    if not outputs[id_index]:
        for logit in outputs[pred_result_index].logits: 
            result_index = torch.argmax(logit, dim=-1).item()
            result_conf = torch.softmax(logit, dim=-1).detach().cpu().numpy()
            pred_label = result_dict[int(result_index)]
            results.append({
            "sentiment_classification": pred_label,
            "sentiment_confidence": result_conf.astype("float64").tolist()
            })
    
    else:
        for logit, c_id in zip(outputs[pred_result_index].logits, outputs[id_index]):
            result_index = torch.argmax(logit, dim=-1).item()
            result_conf = torch.softmax(logit, dim=-1).detach().cpu().numpy()
            pred_label = result_dict[int(result_index)]
            results.append({
            "sentiment_classification": pred_label,
            "sentiment_confidence": result_conf.astype("float64").tolist(),
            "comment_id": c_id
            })
    
    sentiment_result = {"sentiment": results}
    
    return sentiment_result
