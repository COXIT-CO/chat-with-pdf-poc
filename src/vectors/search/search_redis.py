import asyncio

from typing import List

import openai

from core.logger import info
from core.repositories.repo_files import FileItem
from openai_wrappers.types import ChatMessageContentItemDocSearch
from vectors.repositories.repo_redis import RedisRepository
from vectors.search.search_utils import create_query_embedding


async def search_redis(
        loop: asyncio.AbstractEventLoop,
        openai_client: openai.OpenAI,
        redis_repo: RedisRepository,
        document: FileItem,
        query: str,
) -> List[ChatMessageContentItemDocSearch]:
    """
    Performs a vector search in Redis for a given document and query.

    This function creates an embedding for the query using OpenAI, then searches
    for similar vectors in Redis that belong to the specified document.

    Args:
        loop: The asyncio event loop for timing operations
        openai_client: OpenAI client instance for creating embeddings
        redis_repo: Redis repository for vector search operations
        document: File item representing the document to search within
        query: The search query string

    Returns:
        List of ChatMessageContentItemDocSearch objects containing the search results

    Raises:
        Exception: If embedding creation fails
    """
    success, embedding, err = await create_query_embedding(openai_client, query)
    if not success:
        raise Exception(err)

    t0 = loop.time()
    search_results = redis_repo.search_vectors(document.file_name, embedding, 10)
    info(f"vector search resolved in {loop.time() - t0:2f}s")

    content = []
    for s in search_results:
        content.append(ChatMessageContentItemDocSearch(
            paragraph_id=s.metadata["id"],
            text=s.metadata["text"],
            type="doc_search",
            highlight_box=s.metadata["paragraph_box"],
            page_n=int(s.metadata["page_n"]),
        ))

    return content
