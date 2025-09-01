# Model2Vec Inference module

🎯 Overview
-----------

This is the the inference container modified from https://github.com/weaviate/t2v-model2vec-models, so that I could input and output embedding in an openai-compatible way.

It is built to support only static (`model2vec`) models.

[Model2Vec](https://github.com/MinishLab/model2vec) models significantly
surpass other static embeddings models like GLoVe and BPEmb in performance,
and the pre-built Docker images containing these models are notably compact in size.

The reason of forking this repo
-------------------------------

Because I prefer openai-compatible endpoint for this embedding model: `minishlab/potion-base-8M`.

Worth to note that, this `minishlab/potion-base-8M` is a distilled version of the `baai/bge-base-en-v1.5` Sentence Transformer. It uses static embeddings, allowing text embeddings to be computed orders of magnitude faster on both GPU and CPU. It is designed for applications where computational resources are limited or where real-time performance is critical.

Below is the format of curl request to get the embedding output:

```bash
curl -X 'POST' \
  'http://localhost:8080/v1/embeddings' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "input": "hi, how are you",
  "model": "minishlab/potion-base-8M"
}'
```

OR

```bash
curl -X 'POST' \
  'http://localhost:8080/v1/embeddings' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "input": ["hi, how are you", "thank you"],
  "model": "minishlab/potion-base-8M"
}'
```

🐳 Build your own image
--------------------------

- Build the docker container: `docker build -t potion-base-8m:latest . --build-arg MODEL_NAME=minishlab/potion-base-8M`

- Start the docker container: `docker run -p 8080:8080 potion-base-8m`

- Stop the running docker container: `docker stop <CONTAINER_NAME>`


You could change the embeddings model from `minishlab/potion-base-8M` to other models as follows:

|Model Name|Language|
|---|---|---|
|[`potion-multilingual-128M`](https://huggingface.co/minishlab/potion-multilingual-128M)|Multilingual|
|[`potion-retrieval-32M`](https://huggingface.co/minishlab/potion-retrieval-32M)|English|
|[`potion-base-4M`](https://huggingface.co/minishlab/potion-base-4M)|English|
|[`potion-base-8M`](https://huggingface.co/minishlab/potion-base-8M)|English|
|[`potion-base-32M`](https://huggingface.co/minishlab/potion-base-32M)|English|

You just need to copy `.env.example` to `.env`, and then change the `MODEL_NAME` value there.

🔗 Useful Resources
--------------------

- [Locally Hosted Transformers Text Embeddings + Weaviate](https://weaviate.io/developers/weaviate/model-providers/transformers/embeddings)
- [Model2Vec](https://github.com/MinishLab/model2vec)
- [The Minish Lab](https://huggingface.co/minishlab)
