import torch
import uvicorn
from fastapi import FastAPI, status
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from models.sentiment.core import process_inputs, sentiment_classifier, format_output
from common.schemas import payload
from pathlib import Path

local_distilbert = Path("models/sentiment/distilbert_model")

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


@app.get("/healthcheck", status_code=status.HTTP_201_CREATED)
def ping():
    return "\n"


@app.post("/sentiment")
def process_sentiment(input_payload: payload):

    inputs, ids = process_inputs(input_payload)
    outputs = sentiment_classifier(model, tokenizer, inputs, ids)
    results = format_output(outputs)

    return results

if __name__ == "__main__":
    uvicorn.run("models.sentiment.main:app", reload=True)