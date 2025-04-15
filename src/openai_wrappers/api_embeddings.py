import asyncio

from typing import List

from openai import OpenAI
from openai.types import CreateEmbeddingResponse

from core.globals import EMBEDDING_MODEL


# max batch size -- 2048
# https://community.openai.com/t/embeddings-api-max-batch-size/655329/2


def create_embeddings(client: OpenAI, texts: List[str]) -> CreateEmbeddingResponse:
    return client.embeddings.create(
        input=texts,
        model=EMBEDDING_MODEL,
        encoding_format="float",
    )

async def async_create_embeddings(client: OpenAI, texts: List[str]) -> CreateEmbeddingResponse:
    return await asyncio.to_thread(create_embeddings, client, texts)
