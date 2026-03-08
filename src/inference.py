import torch
import uvicorn
from fastapi import FastAPI, status
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from functions import sentiment_classifier, parse_payload, format_output

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

app = FastAPI()

class sentiment_payload(BaseModel):
    user_id: list[str]
    comment_id: list[str]
    comments: list[str]


@app.get("/ping", status_code=status.HTTP_201_CREATED)
def ping():
    return "server online...!"


@app.post("/invocation")
def get_payload(input_payload: sentiment_payload):
    payload = input_payload.model_dump()
    reference_dataframe = parse_payload(payload)
    outputs = sentiment_classifier(model, tokenizer, reference_dataframe)
    results_json = format_output(outputs, reference_dataframe)

    return results_json

if __name__ == "__main__":
    uvicorn.run("inference:app", reload=True)