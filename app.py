# Reference: https://github.com/BerriAI/litellm/issues/1647

import os
from typing import Optional, List
from logging import getLogger
from fastapi import FastAPI, Depends, Response, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from vectorizer import Vectorizer, VectorInput
from meta import Meta

logger = getLogger("uvicorn")

vec: Vectorizer
meta_config: Meta

get_bearer_token = HTTPBearer(auto_error=False)
allowed_tokens: List[str] = None


def get_allowed_tokens() -> List[str] | None:
    if (
        tokens := os.getenv("AUTHENTICATION_ALLOWED_TOKENS", "").strip()
    ) and tokens != "":
        return tokens.strip().split(",")


def is_authorized(auth: Optional[HTTPAuthorizationCredentials]) -> bool:
    if allowed_tokens is not None and (
        auth is None or auth.credentials not in allowed_tokens
    ):
        return False
    return True


async def lifespan(app: FastAPI):
    global vec
    global meta_config
    global allowed_tokens

    allowed_tokens = get_allowed_tokens()
    model_path = "./models"

    meta_config = Meta(model_path)
    vec = Vectorizer(model_path)

    yield


app = FastAPI(lifespan=lifespan)


@app.get("/.well-known/live", response_class=Response)
@app.get("/.well-known/ready", response_class=Response)
async def live_and_ready(response: Response):
    response.status_code = status.HTTP_204_NO_CONTENT


@app.get("/meta")
def meta(
    response: Response,
    auth: Optional[HTTPAuthorizationCredentials] = Depends(get_bearer_token),
):
    if is_authorized(auth):
        return meta_config.get()
    else:
        response.status_code = status.HTTP_401_UNAUTHORIZED
        return {"error": "Unauthorized"}


@app.get("/v1/models")
async def list_models(
    response: Response,
    auth: Optional[HTTPAuthorizationCredentials] = Depends(get_bearer_token)
):
    if is_authorized(auth):
        try:
            config = meta_config.get()
            model_name = config.get("model_path", "minishlab/potion-base-8M")
            
            # Extract just the model name if it's a path
            if "/" in model_name and not model_name.startswith("minishlab/"):
                model_display_name = "minishlab/potion-base-8M"
            else:
                model_display_name = model_name
            
            return {
                "object": "list",
                "data": [
                    {
                        "id": model_display_name,
                        "object": "model",
                        "created": 1700000000,  # Static timestamp
                        "owned_by": "minishlab",
                        "permission": [],
                        "root": model_display_name,
                        "parent": None
                    }
                ]
            }
        except Exception as e:
            logger.exception("Something went wrong while listing models.")
            response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
            return {"error": str(e)}
    else:
        response.status_code = status.HTTP_401_UNAUTHORIZED
        return {"error": "Unauthorized"}


@app.post("/v1/embeddings")
@app.post("/embeddings")
async def embed(item: VectorInput,
                response: Response,
                auth: Optional[HTTPAuthorizationCredentials] = Depends(get_bearer_token)):
    if is_authorized(auth):
        try:
            if isinstance(item.input, list):
                item.input = tuple(item.input)  # Convert list to tuple for hashability
            
            vector = await vec.vectorize(item.input, item.config)
            return {
                "object": "list",
                "data": [{
                    "object": "embedding",
                    "index": index,
                    "embedding": embedding_item
                } for index, embedding_item in enumerate(vector.tolist())],
                "model": item.model
            }
        except Exception as e:
            logger.exception("Something went wrong while vectorizing data.")
            response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
            return {"error": str(e)}
    else:
        response.status_code = status.HTTP_401_UNAUTHORIZED
        return {"error": "Unauthorized"}
