from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from aiohttp import ClientSession

from core.configs import EvalConfig
from core.logger import error
from openai_wrappers.types import ChatMessage


@dataclass
class ChatCompletionsUsage:
    completion_tokens: Optional[int] = 0
    prompt_tokens: Optional[int] = 0


async def call_chat_completions_non_streaming(
        http_session: ClientSession,
        messages: List[ChatMessage],
        model: str,
        eval_config: EvalConfig,
        tools: List[Dict[str, str]] = None,
):
    # Request payload with stream=False
    payload = {
        "model": model,
        "messages": [m.model_dump() for m in messages],
        "stream": False,
        "temperature": 0.7,
    }

    if tools:
        payload["tools"] = tools

    # Headers including authorization
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {eval_config.chat_endpoint_api_key}"
    }

    async with http_session.post(
            f"{eval_config.chat_endpoint}/chat/completions",
            json=payload,
            headers=headers
    ) as response:
        if response.status == 200:
            # Parse the JSON response
            result = await response.json()
            return result
        else:
            error_text = await response.text()
            return {
                "error": True,
                "status_code": response.status,
                "message": error_text
            }


def try_get_usage(resp: Dict[str, Any]) -> ChatCompletionsUsage:
    usage = ChatCompletionsUsage()
    try:
        usage_dict = resp["usage"]
        usage.completion_tokens += usage_dict["completion_tokens"]
        usage.prompt_tokens += usage_dict["prompt_tokens"]
    except Exception as e:
        error(f"failed to parse usage: {e}")
    return usage