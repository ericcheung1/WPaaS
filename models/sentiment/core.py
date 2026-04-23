from common.schemas import payload
import numpy as np

text_idx, pred_result_idx = 0, 0
id_idx = 1

def process_inputs(in_payload: payload) -> tuple[list, list]:
    """
    Takes payload object and returns the input texts and 
    if applicable, the text ids
    """
    comments = []

    for item in in_payload.texts:
        comment_pair = (item.text.lower().strip(), item.text_id)
        comments.append(comment_pair)

    if in_payload.contains_id == False:
        inputs = [comment[text_idx] for comment in comments]
        
        return inputs, []
    
    else:
        ids = []
        inputs = []
        for comment in comments:
            ids.append(comment[id_idx])
            inputs.append(comment[text_idx])

        return inputs, ids


def sentiment_classifier(model_session, tokenizer, input, ids=[]) -> tuple[list, list]:
    """
    Takes a onnx model session, tokenizer from tokenizers library,
    tokenized text input, and text ids. Runs distilbert model on 
    text inputs and returns output logits as a list and the associated
    ids
    """
    tokenized_inputs = tokenizer.encode_batch(input)
    token_ids = np.array([item.ids for item in tokenized_inputs])
    attention_masks = np.array([item.attention_mask for item in tokenized_inputs])
    
    inputs = {
        "input_ids": token_ids,
        "attention_mask": attention_masks
    }

    # runs onnx distilbert on tokenized inputs
    # outputs is a n-dim numpy array
    outputs = model_session.run(None, inputs)
    # outputs[0] grabs model logits and converts to a list
    output_list = outputs[0].tolist()
    
    if ids:
        return output_list, ids
    else:
        return output_list, []


def softmax(input, axis=None):
    x = np.array(input)
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def format_output(outputs):
    """
    Takes the output object from the onnx distilbert model and calculates 
    confidences with softmax, prediction result index, and final prediction
    result. Returns the prediction, confidences, and ids if applicable
    """
    senti_results = []
    result_dict = {0: "NEGATIVE", 1: "POSITIVE"}

    if not outputs[id_idx]:
        for result in outputs[pred_result_idx]:

            result_conf = softmax(result)
            result_index = np.argmax(result)
            pred_label = result_dict[int(result_index)]
            
            senti_results.append({
            "sentiment_classification": pred_label,
            "sentiment_confidence": result_conf.tolist()
            })
    
    else:
        for result, id in zip(outputs[pred_result_idx], outputs[id_idx]):

            result_conf = softmax(result)
            result_index = np.argmax(result)
            pred_label = result_dict[int(result_index)]
            
            senti_results.append({
            "sentiment_classification": pred_label,
            "sentiment_confidence": result_conf.tolist(),
            "text_id": id
            })
    
    sentiment_result = {"sentiment": senti_results}
    
    return sentiment_result
