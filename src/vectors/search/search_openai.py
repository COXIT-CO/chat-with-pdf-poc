import json
import asyncio

from typing import List

from aiohttp import ClientSession

from core.logger import info
from processing.p_utils import generate_paragraph_id
from core.repositories.repo_files import FileItem
from openai_wrappers.api_vector_store import VectorStoreSearch, vector_store_search
from openai_wrappers.types import ChatMessageContentItemDocSearch


async def search_openai(
        loop: asyncio.AbstractEventLoop,
        http_session: ClientSession,
        tool_name: str,
        document: FileItem,
        query: str,
) -> List[ChatMessageContentItemDocSearch]:
    post = VectorStoreSearch(
        vector_store_id=document.vector_store_id,
        query=query
    )

    try:
        t0 = loop.time()
        resp = await vector_store_search(http_session, post)
        info(f"Vector store search for '{query}' took {loop.time() - t0:.3f} seconds")
    except Exception as e:
        err = f"Error while executing tool {tool_name}: vector store search failed: {str(e)}"
        raise Exception(err)

    content = []
    for obj in resp:
        highlight_box = None
        try:
            highlight_box = json.loads(obj.attributes.get("paragraph_box", json.dumps(None)))
        except Exception:
            pass
        page_n = None
        try:
            page_n = int(obj.attributes.get("page_n", None))
        except Exception:
            pass

        section_name = obj.attributes.get("section_number")

        for content_i in obj.content:
            paragraph_id = obj.attributes.get("paragraph_id", generate_paragraph_id(content_i.text))

            content.append(ChatMessageContentItemDocSearch(
                paragraph_id=paragraph_id,
                text=content_i.text,
                type="doc_search",
                highlight_box=highlight_box,
                page_n=page_n,
                section_name=section_name,
            ))

    return content
