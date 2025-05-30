from typing import Union, List, Optional, Any, Literal, Dict

from pydantic import BaseModel


type ChatMessage = Union[ChatMessageSystem, ChatMessageUser, ChatMessageAssistant, ChatMessageTool]


def model_validate_chat_message(obj: Union[Dict[str, Any], BaseModel]) -> ChatMessage:
    """Validate and convert a dictionary or model to a ChatMessage."""
    if isinstance(obj, (ChatMessageSystem, ChatMessageUser, ChatMessageAssistant, ChatMessageTool)):
        return obj

    if not isinstance(obj, dict):
        obj = obj.model_dump()

    role = obj.get("role")

    if role in ["system", "developer"]:
        return ChatMessageSystem.model_validate(obj)
    elif role == "user":
        return ChatMessageUser.model_validate(obj)
    elif role == "assistant":
        return ChatMessageAssistant.model_validate(obj)
    elif role == "tool":
        return ChatMessageTool.model_validate(obj)
    else:
        raise ValueError(f"Unknown role: {role}")


class ChatToolParameterProperty(BaseModel):
    type: str
    description: str
    enum: List[str] = None


class ChatToolParameters(BaseModel):
    type: str
    properties: Dict[str, ChatToolParameterProperty]
    required: List[str]
    additionalProperties: bool = False


class ChatToolFunction(BaseModel):
    name: str
    description: str
    parameters: ChatToolParameters
    strict: bool = True


class ChatTool(BaseModel):
    type: str
    function: ChatToolFunction


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
