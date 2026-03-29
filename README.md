# ML Inference API

A lightweight REST API built using FastAPI for serving machine learning models such as sentiment analysis. The API exposes simple HTTP endpoints that allow external applications to submit data and receive predictions.

This project is a generalization of my earlier Canucks sentiment analysis project `https://github.com/ericcheung1/canucks-sentiment`, where sentiment models were used to analyze fan discussions from Reddit.

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
# Note: first index of confidence array is the confidence of the text being negative and second index is confidence of text being positive

[
    {
        "sentiment_classification":"POSITIVE",
        "sentiment_confidence":[ 0.011138916015625,0.98876953125],
        "comment_id":"012ecc"
    }
]
```

### Run Locally

This repository uses Git LFS to store model weights.
Make sure Git LFS is installed with `git lfs install` before cloning.

#### With Source
Clone the repo, install dependencies with `pip install -r requirements.txt`, change into the `src` directory, then start FastAPI server with `uvicorn src/inference:app --host 0.0.0.0 --port 8000`.

#### With Docker
Pull the latest build with `docker pull ghcr.io/ericcheung1/text-processing:main`, then start container with `docker run -p 8000:8000 ghcr.io/ericcheung1/text-processing:main`.

## Model

This API serves a fine-tuned DistilBERT model for sentiment classification. Model weights are included in the repository using Git LFS.