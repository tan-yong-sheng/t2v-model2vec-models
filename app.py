# Reference: https://github.com/BerriAI/litellm/issues/1647

import os
from typing import Optional, List
from logging import getLogger
from fastapi import FastAPI, Depends, Response, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from dotenv import load_dotenv
from vectorizer import Vectorizer, VectorInput
from meta import Meta

# Load environment variables from .env file
load_dotenv()

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


def get_available_model():
    """Get the available model name from environment or config"""
    model_name = os.getenv("MODEL_NAME")
    if not model_name:
        config = meta_config.get()
        model_name = config.get("model_path", "minishlab/potion-base-8M")
    
    # Extract just the model name if it's a path
    if "/" in model_name and not model_name.startswith("minishlab/"):
        return "minishlab/potion-base-8M"
    else:
        return model_name


@app.get("/v1/models")
@app.get("/models")
async def list_models(
    response: Response,
    auth: Optional[HTTPAuthorizationCredentials] = Depends(get_bearer_token)
):
    if is_authorized(auth):
        try:
            model_display_name = get_available_model()
            
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
            # Validate model parameter
            available_model = get_available_model()
            if item.model != available_model:
                response.status_code = status.HTTP_400_BAD_REQUEST
                return {"error": f"Model '{item.model}' not found. Available model: '{available_model}'"}
            
            # Validate encoding format
            if item.encoding_format not in ["float", "base64"]:
                response.status_code = status.HTTP_400_BAD_REQUEST
                return {"error": "encoding_format must be 'float' or 'base64'"}
            
            if isinstance(item.input, list):
                item.input = tuple(item.input)  # Convert list to tuple for hashability
            
            vector = await vec.vectorize(item.input, item.config)
            
            # Handle dimensions parameter (truncate if specified)
            if item.dimensions and item.dimensions > 0:
                vector = vector[:, :item.dimensions]
            
            # Convert to list for JSON serialization
            vector_list = vector.tolist()
            
            # Handle encoding format
            if item.encoding_format == "base64":
                import base64
                import numpy as np
                # Convert to base64 encoded string
                data = []
                for i, embedding in enumerate(vector_list):
                    arr = np.array(embedding, dtype=np.float32)
                    encoded = base64.b64encode(arr.tobytes()).decode('utf-8')
                    data.append({
                        "object": "embedding",
                        "index": i,
                        "embedding": encoded
                    })
            else:
                # Default float format
                data = []
                for i, embedding in enumerate(vector_list):
                    data.append({
                        "object": "embedding", 
                        "index": i,
                        "embedding": embedding
                    })
            
            # Calculate token usage (approximate)
            if isinstance(item.input, tuple):
                input_texts = list(item.input)
            else:
                input_texts = [item.input]
            
            total_tokens = sum(len(text.split()) for text in input_texts)
            
            return {
                "object": "list",
                "data": data,
                "model": item.model,
                "usage": {
                    "prompt_tokens": total_tokens,
                    "total_tokens": total_tokens
                }
            }
        except Exception as e:
            logger.exception("Something went wrong while vectorizing data.")
            response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
            return {"error": str(e)}
    else:
        response.status_code = status.HTTP_401_UNAUTHORIZED
        return {"error": "Unauthorized"}
