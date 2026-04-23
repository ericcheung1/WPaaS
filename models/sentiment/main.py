import uvicorn
from tokenizers import Tokenizer
import onnxruntime as ort
from fastapi import FastAPI, status
from models.sentiment.core import process_inputs, sentiment_classifier, format_output
from common.schemas import payload
from pathlib import Path

distilbert_onnx = Path("models/sentiment/distilbert_fp16_onnx/distilbert_fp16.onnx")
tokenizer_json = Path("models/sentiment/distilbert_fp16_onnx/tokenizer.json")

tokenizer = Tokenizer.from_file(str(tokenizer_json))
tokenizer.enable_padding(
    pad_id=0,
    pad_token="[PAD]",
    direction="right"
    )

model_session = ort.InferenceSession(distilbert_onnx, providers=["CPUExecutionProvider"])

app = FastAPI()


@app.get("/healthcheck", status_code=status.HTTP_201_CREATED)
def ping():
    return {"message": "service healthy"}


@app.post("/sentiment")
def process_sentiment(input_payload: payload):

    inputs, ids = process_inputs(input_payload)
    outputs = sentiment_classifier(model_session, tokenizer, inputs, ids)
    results = format_output(outputs)

    return results

if __name__ == "__main__":
    uvicorn.run("models.sentiment.main:app", reload=True)