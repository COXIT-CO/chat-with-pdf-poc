"""
Chat Tools - A package for building and managing chat tools.

This package provides utilities and abstractions for creating and managing
tools that can be used in chat-based applications.
"""

__version__ = "0.1.0"

from .chat_models import (
    ChatMessage, ChatMessageSystem, ChatMessageUser, 
    ChatMessageAssistant, ChatMessageTool, ChatTool,
    ToolCall, model_validate_chat_message
)

# Import tool_usage submodule
from . import tool_usage

__all__ = [
    'ChatMessage',
    'ChatMessageSystem',
    'ChatMessageUser',
    'ChatMessageAssistant',
    'ChatMessageTool',
    'ChatTool',
    'ToolCall',
    'model_validate_chat_message',
    'tool_usage',
]