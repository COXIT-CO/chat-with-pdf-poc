import asyncio
from typing import List, Optional, Dict, Union, Any, Literal

from openai import OpenAI
from pydantic import BaseModel

import aiohttp

from core.logger import error


class VectorStoreFileCreate(BaseModel):
    vector_store_id: str
    file_id: str
    attributes: Optional[Dict[str, Union[str, bool, int, float]]] = None
    chunking_strategy: Optional[Dict] = None


class VectorStoreCreate(BaseModel):
    name: str
    file_ids: Optional[List[str]] = None


class VectorStoreFilesList(BaseModel):
    vector_store_id: str
    limit: int = 100


class VectorStoreSearchFilterComparison(BaseModel):
    type: Literal["eq", "ne", "lt", "lte", "gt", "gte"]
    property: str
    value: str


class VectorStoreSearchFilterCompound(BaseModel):
    type: Literal["and", "or"]
    filters: List[Union["VectorStoreSearchFilterCompound", VectorStoreSearchFilterComparison]]


class VectorStoreSearch(BaseModel):
    vector_store_id: str
    query: str
    filters: Optional[List[Union[VectorStoreSearchFilterCompound, VectorStoreSearchFilterComparison]]] = None
    max_num_results: Optional[int] = 10


class VectorStoreSearchRespItemContentItem(BaseModel):
    type: str
    text: str


class VectorStoreSearchRespItem(BaseModel):
    file_id: str
    filename: str
    score: float
    content: List[VectorStoreSearchRespItemContentItem]
    attributes: Optional[Dict[str, Any]] = None


def vector_store_create(client: OpenAI, data: VectorStoreCreate):
    vector_store = client.vector_stores.create(**data.model_dump(exclude_none=True))
    return vector_store


def vector_stores_list(client: OpenAI):
    vector_stores = client.vector_stores.list()
    return vector_stores


def vector_store_retrieve(client: OpenAI, vector_store_id: str):
    vector_store = client.vector_stores.retrieve(vector_store_id=vector_store_id)
    return vector_store


def vector_store_file_create(client: OpenAI, data: VectorStoreFileCreate):
    vector_store_id = data.vector_store_id
    params = data.model_dump(exclude={"vector_store_id"}, exclude_none=True)

    vector_store_file = client.vector_stores.files.create(
        vector_store_id=vector_store_id,
        **params
    )
    return vector_store_file


def vector_store_files_list(client: OpenAI, data: VectorStoreFilesList):
    files = list(client.vector_stores.files.list(
        vector_store_id=data.vector_store_id,
        limit=data.limit
    ))

    if len(files) == data.limit:
        while True:
            try:
                batch = list(client.vector_stores.files.list(
                    vector_store_id=data.vector_store_id,
                    limit=data.limit,
                    after=(files[-1] or {}).get("id")
                ))
                files.extend(batch)
                if len(batch) != data.limit:
                    break
            except Exception as e:
                error(f"Error fetching additional vector store files: {e}")
                break

    return files


def vector_store_file_delete(client: OpenAI, vector_store_id: str, file_id: str):
    """
    Delete a file from a vector store.

    Args:
        client: The OpenAI client instance
        vector_store_id: The ID of the vector store
        file_id: The ID of the file to delete

    Returns:
        The deleted vector store file object
    """
    deleted_file = client.vector_stores.files.delete(
        vector_store_id=vector_store_id,
        file_id=file_id
    )
    return deleted_file


async def vector_store_search(
        session: aiohttp.ClientSession,
        data: VectorStoreSearch
) -> List[VectorStoreSearchRespItem]:
    api_key = OpenAI().api_key

    url = f"https://api.openai.com/v1/vector_stores/{data.vector_store_id}/search"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = data.model_dump(exclude={"vector_store_id"}, exclude_none=True)

    async with session.post(url, headers=headers, json=payload) as resp:
        if resp.status == 200:
            result = await resp.json()

            try:
                # Parse JSON directly into Pydantic models
                items = [VectorStoreSearchRespItem.model_validate(item) for item in result["data"]]
                return items
            except Exception as e:
                error_msg = f"Failed to convert search results: {str(e)}"
                error(error_msg)
                raise ValueError(error_msg)
        else:
            error_text = await resp.text()
            text = f"Error searching vector store: {error_text}"
            error(text)
            raise Exception(text)


# Async wrappers for synchronous functions

async def async_vector_store_create(client: OpenAI, data: VectorStoreCreate):
    """Async wrapper for vector_store_create"""
    return await asyncio.to_thread(vector_store_create, client, data)


async def async_vector_stores_list(client: OpenAI):
    """Async wrapper for vector_stores_list"""
    return await asyncio.to_thread(vector_stores_list, client)


async def async_vector_store_retrieve(client: OpenAI, vector_store_id: str):
    """Async wrapper for vector_store_retrieve"""
    return await asyncio.to_thread(vector_store_retrieve, client, vector_store_id)


async def async_vector_store_file_create(client: OpenAI, data: VectorStoreFileCreate):
    """Async wrapper for vector_store_file_create"""
    return await asyncio.to_thread(vector_store_file_create, client, data)


async def async_vector_store_files_list(client: OpenAI, data: VectorStoreFilesList):
    """Async wrapper for vector_store_files_list"""
    return await asyncio.to_thread(vector_store_files_list, client, data)


async def async_vector_store_file_delete(client: OpenAI, vector_store_id: str, file_id: str):
    """
    Async wrapper for vector_store_file_delete.

    Args:
        client: The OpenAI client instance
        vector_store_id: The ID of the vector store
        file_id: The ID of the file to delete

    Returns:
        The deleted vector store file object
    """
    return await asyncio.to_thread(vector_store_file_delete, client, vector_store_id, file_id)
