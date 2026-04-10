import torch
from common.schemas import comment_object, comments_all

text_index = 0
id_index = 1
pred_result_index = 0

def process_inputs(in_payload: comments_all):
    """
    """
    comments = []

    for item in my_comment.comments:
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
    # NOTE: fix format_output figure out what to do with comment_id
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
    
    return results

def format_json():
    return 0

if __name__ == "__main__":
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from pathlib import Path

    a_comment_object = comment_object(comment="THIS IS TESTING")
    b_comment_object = comment_object(comment=":))")
    try:
        my_comment = comments_all(comments=[a_comment_object, b_comment_object])
    except ValueError as e:
        print(f"{e}")

    print(f"{my_comment}")
    # print(my_comment.comments.comment_id)
    input = [(item.comment.lower().strip(), item.comment_id) for item in my_comment.comments]
    print(input)
    text_index = 0
    print([x[text_index] for x in input])

    local_distilbert = Path("models/sentiment/distilbert_model")
    print(f"local distilber path: {local_distilbert}")

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
    inputs, ids = process_inputs(my_comment)
    outputs = sentiment_classifier(model, tokenizer, inputs, ids)

    result = format_output(outputs)
    print(result)
