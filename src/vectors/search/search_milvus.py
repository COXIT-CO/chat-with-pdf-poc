import asyncio

from typing import List

import numpy as np
import openai

from core.logger import info
from core.repositories.repo_files import FileItem
from openai_wrappers.types import ChatMessageContentItemDocSearch
from vectors.repositories.repo_milvus import MilvusRepository, collection_from_file_name
from vectors.search.search_utils import create_query_embedding


async def search_milvus(
        loop: asyncio.AbstractEventLoop,
        openai_client: openai.OpenAI,
        milvus_repo: MilvusRepository,
        document: FileItem,
        query: str,
) -> List[ChatMessageContentItemDocSearch]:
    """
    Performs a semantic search in Milvus vector database for the given query against a document.

    This function creates an embedding for the query using OpenAI, then searches the Milvus
    collection corresponding to the document for similar content. The search results are
    returned as a list of document search items with relevance scores.

    Args:
        loop: The asyncio event loop for timing operations
        openai_client: OpenAI client instance for creating embeddings
        milvus_repo: Repository for interacting with Milvus vector database
        document: The document to search within
        query: The search query text

    Returns:
        List of ChatMessageContentItemDocSearch objects containing the search results
        with paragraph IDs, text content, relevance scores, and position information

    Raises:
        Exception: If embedding creation fails
    """
    success, embedding, err = await create_query_embedding(openai_client, query)
    if not success:
        raise Exception(err)

    t0 = loop.time()
    search_results = list(milvus_repo.search(
        collection_from_file_name(document.file_name),
        np.array(embedding)
    ))
    info(f"vector search resolved in {loop.time() - t0:2f}s")

    return [
        ChatMessageContentItemDocSearch(
            paragraph_id=s.par_id,
            text=f"SCORE: {s.distance}\n{s.text}",
            type="doc_search",
            highlight_box=list(s.paragraph_box),
            page_n=s.page_n
        )
        for s in search_results
    ]
