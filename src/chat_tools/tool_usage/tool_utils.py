from typing import List, Iterator

from ..chat_models import ChatMessage, ChatMessageUser, ToolCall, ChatMessageTool, ChatMessageAssistant


def messages_since_last_user_message(messages: List[ChatMessage]):
    messages_since_last_user_msg = []
    for message in reversed(messages):
        if isinstance(message, ChatMessageUser):
            break
        messages_since_last_user_msg.insert(0, message)
    return messages_since_last_user_msg


def get_unanswered_tool_calls(messages: List[ChatMessage]) -> Iterator[ToolCall]:
    # Get all tool call IDs from responses
    tool_messages: List[ChatMessageTool] = [
        m for m in messages if isinstance(m, ChatMessageTool)
    ]
    answered_tool_call_ids = {
        m.tool_call_id for m in tool_messages if m.tool_call_id
    }

    # yield all unanswered tool calls
    for m in messages:
        if isinstance(m, ChatMessageAssistant) and m.tool_calls:
            for tool_call in m.tool_calls:
                if tool_call.id not in answered_tool_call_ids:
                    yield tool_call
