import asyncio
from typing import Tuple, List

import openai

from core.globals import PROCESSING_STRATEGY, SAVE_STRATEGY
from core.repositories.repo_files import FileItem
from core.tools.tool_abstract import build_tool_call
from core.tools.tool_context import ToolContext
from openai_wrappers.types import ToolCall, ChatMessageTool

from vectors.repositories.repo_milvus import MilvusRepository
from vectors.repositories.repo_redis import RedisRepository

from vectors.search.search_milvus import search_milvus
from vectors.search.search_openai import search_openai
from vectors.search.search_redis import search_redis


async def vector_search_chat_messages(
        ctx: ToolContext,
        tool_name: str,
        document: FileItem,
        query: str,
        tool_call: ToolCall,
) -> Tuple[bool, List[ChatMessageTool]]:

    loop = asyncio.get_running_loop()

    if PROCESSING_STRATEGY == "openai_fs":
        future = search_openai(
            loop, ctx.http_session, tool_name, document, query
        )

    elif PROCESSING_STRATEGY == "local_fs":
        if SAVE_STRATEGY == "redis":
            assert ctx.redis_repository is not None, "Redis repository is not initialized"
            redis_repo: RedisRepository = ctx.redis_repository

            assert ctx.openai is not None, "OpenAI client is not initialized"
            openai_client: openai.OpenAI = ctx.openai

            future = search_redis(
                loop, openai_client, redis_repo, document, query
            )

        elif SAVE_STRATEGY == "milvus":
            assert ctx.milvus_repository is not None, "Milvus repository is not initialized"
            milvus_repo: MilvusRepository = ctx.milvus_repository

            assert ctx.openai is not None, "OpenAI client is not initialized"
            openai_client: openai.OpenAI = ctx.openai

            future = search_milvus(
                loop, openai_client, milvus_repo, document, query
            )

        else:
            raise ValueError(f"Unknown save strategy: {SAVE_STRATEGY}")
    else:
        raise ValueError(f"Unknown processing strategy: {PROCESSING_STRATEGY}")

    try:
        content_items = await future
    except Exception as e:
        return False, [
            build_tool_call(str(e), tool_call)
        ]

    if not content_items:
        return True, [
            build_tool_call(
                f"Executed tool {tool_name}: no results found for query '{query}'",
                tool_call
            )
        ]

    return True, [
        build_tool_call(
            content_items, tool_call
        )
    ]
