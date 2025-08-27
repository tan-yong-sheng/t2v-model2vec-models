import asyncio
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel
from typing import Optional
from cachetools import cached, TTLCache

from model2vec import StaticModel


class VectorInputConfig(BaseModel):
    pooling_strategy: Optional[str] = None
    task_type: Optional[str] = None

    def __hash__(self):
        return hash((self.pooling_strategy, self.task_type))

    def __eq__(self, other):
        if isinstance(other, VectorInputConfig):
            return (
                self.pooling_strategy == other.pooling_strategy
                and self.task_type == other.task_type
            )
        return False


class VectorInput(BaseModel):
    input: str | list
    model: str
    config: Optional[VectorInputConfig] = None
    encoding_format: Optional[str] = "float"  # OpenAI compatibility: "float" or "base64"
    dimensions: Optional[int] = None  # OpenAI compatibility: for truncating embeddings

    def __hash__(self):
        if isinstance(self.input, list):
            input_hashable = tuple(self.input)  # Convert list to tuple for hashability
        else:
            input_hashable = self.input
        return hash((input_hashable, self.config))

    def __eq__(self, other):
        if isinstance(other, VectorInput):
            return self.input == other.input and self.config == other.config
        return False


class Model2VecVectorizer:
    model: StaticModel

    def __init__(self, model_path: str):
        self.model = StaticModel.load_local(model_path)

    @cached(cache=TTLCache(maxsize=1024, ttl=600))
    def vectorize(self, text: str | list[str], config: VectorInputConfig):
        # Convert list to tuple for caching purposes
        if isinstance(text, list):
            text = tuple(text)
        
        if isinstance(text, str):
            input_list = [text]
        else:
            input_list = list(text)  # Convert tuple back to list for processing
        
        embeddings = self.model.encode(input_list, use_multiprocessing=True)
        return embeddings


class Vectorizer:
    executor: ThreadPoolExecutor

    def __init__(
        self,
        model_path: str,
    ):
        self.executor = ThreadPoolExecutor()
        self.vectorizer = Model2VecVectorizer(model_path)

    async def vectorize(self, input: str, config: VectorInputConfig):
        return await asyncio.wrap_future(
            self.executor.submit(self.vectorizer.vectorize, input, config)
        )
