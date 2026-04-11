FROM python:3.12-slim

WORKDIR /

COPY /models/sentiment/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY common/ ./common/ 
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "models.sentiment.main:app", "--host", "0.0.0.0", "--port", "8000"]