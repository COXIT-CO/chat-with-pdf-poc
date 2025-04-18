from typing import Union, List, Optional, Any, Literal

from pydantic import BaseModel


class ChatMessageContentItemDocSearch(BaseModel):
    paragraph_id: str
    text: str
    type: str
    highlight_box: Optional[List[float]] = None
    page_n: Optional[int] = None
    section_name: Optional[str] = None


class ChatMessageContentItemText(BaseModel):
    text: str
    type: str


class ChatMessageContentItemImage(BaseModel):
    image_url: str
    type: str


class ChatMessageContentItemAudio(BaseModel):
    input_audio: str
    type: Literal["input_audio"]


class ChatMessageContentItemFile(BaseModel):
    file: str
    type: Literal["file"]


class ChatMessageBase(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Union[str, List[Union[
        ChatMessageContentItemText,
        ChatMessageContentItemImage,
        ChatMessageContentItemAudio,
        ChatMessageContentItemFile,

        ChatMessageContentItemDocSearch
    ]]]


class ChatMessageSystem(ChatMessageBase):
    role: Literal["system", "developer"]
    name: Optional[str] = None


class ChatMessageUser(ChatMessageBase):
    role: Literal["user"]
    name: Optional[str] = None


class ToolCallFunction(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: Literal["function"]
    function: ToolCallFunction


class ChatMessageAssistant(ChatMessageBase):
    role: Literal["assistant"]
    refusal: Optional[str] = None
    name: Optional[str] = None
    audio: Optional[Any] = None
    tool_calls: Optional[List[ToolCall]] = None


class ChatMessageTool(ChatMessageBase):
    role: Literal["tool"]
    tool_call_id: Optional[str] = None


ChatMessage = Union[ChatMessageSystem, ChatMessageUser, ChatMessageAssistant, ChatMessageTool]
