# Word processing-as-a-service NLP Inference API

A lightweight REST API built using FastAPI for serving machine learning models such as sentiment analysis. The API exposes simple HTTP endpoint(s) that allow external applications to submit data and receive predictions.

This project is a generalization of my earlier Canucks sentiment analysis project [canucks-sentiment](https://github.com/ericcheung1/canucks-sentiment), where sentiment models were used to analyze fan discussions from Reddit and results were hardcoded in a Flask web app.

### Features

- REST API for ML inference
- Sentiment analysis endpoint
- JSON request/response format
- Containerized with Docker
- Model weights included and stored in Git LFS

### Example Request

POST /invocation

```
Request:

{
    "user_id":["ec125"],
    "comment_id":["012ecc"],
    "comments":["this is pretty cool!"]
}


Response: 

[
    {
        "sentiment_classification":"POSITIVE",
        "sentiment_confidence":[ 0.011138916015625,0.98876953125],
        "comment_id":"012ecc"
    }
]
```
 Note: first index of the `sentiment_confidence` array is confidence of the text having negative sentiment and second index is confidence of text having positive sentiment

### Run Locally

This repository uses Git LFS to store model weights.
Make sure Git LFS is installed with `git lfs install` before cloning.

#### With Source
Clone the repo, change into the `src` directory, install dependencies with `pip install -r requirements.txt`, then start FastAPI server with `uvicorn inference:app --host 0.0.0.0 --port 8000`.

#### With Docker
Pull the latest build with `docker pull ghcr.io/ericcheung1/wpaas:main`, then start container with `docker run -p 8000:8000 ghcr.io/ericcheung1/wpaas:main`.

### Model

This API serves a fine-tuned DistilBERT model for sentiment classification. Model weights are included in the repository using Git LFS.