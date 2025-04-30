FROM python:3.13-slim AS base_image

WORKDIR /app

RUN apt-get update
RUN pip install --upgrade pip setuptools

COPY requirements.txt .
RUN pip3 install -r requirements.txt

FROM base_image AS download_model

WORKDIR /app

ARG MODEL_NAME
COPY download.py .
RUN ./download.py

FROM base_image AS t2v_transformers

# Add label for image metadata
LABEL name="model2vec"

WORKDIR /app
COPY --from=download_model /app/models /app/models
COPY . .

ENTRYPOINT ["/bin/sh", "-c"]
CMD ["uvicorn app:app --host 0.0.0.0 --port 8080"]
