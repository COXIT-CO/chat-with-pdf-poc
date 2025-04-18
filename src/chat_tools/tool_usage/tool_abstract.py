from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from ..chat_models import ChatMessageTool, ToolCall, ChatMessage, ChatTool


def build_tool_call(content: Any, tool_call: ToolCall) -> ChatMessageTool:
    return ChatMessageTool(
        role="tool",
        content=content,
        tool_call_id=tool_call.id
    )


class ToolProps(BaseModel):
    tool_name: str
    system_prompt: Optional[str] = None
    depends_on: Optional[List[str]] = None
    # confirmation loop prompt ?


class Tool:
    @property
    def name(self) -> str:
        """
        Get the unique identifier name of the tool.

        Returns:
            str: The name of the tool that uniquely identifies it in the system.

        Raises:
            NotImplementedError: This is an abstract property that must be implemented by concrete tool classes.
        """
        raise NotImplementedError("property 'name' is not implemented for tool")

    def validate_tool_call_args(self, ctx: Any, tool_call: ToolCall, args: Dict[str, Any]) -> (bool, List[ChatMessage]):
        """
        Validate that the provided tool call and arguments are compatible with this tool.

        This method should verify that the provided ToolCall instance and arguments match the expected
        interface of the tool, including checking parameter types, required fields,
        and any other tool-specific validation rules.

        Args:
            ctx (Any): The context in which the tool is being called
            tool_call (ToolCall): The tool call to validate, containing function
                                  details and parameters.
            args (Dict[str, Any]): The arguments to validate, containing function
                                   details and parameters.

        Returns:
            tuple: A pair containing:
                - bool: Indicates whether the validation was successful (True) or failed (False)
                - List[ChatMessage]: List of chat messages containing validation results or error details.
                                     They will help the model correct itself and execute the tool correctly.
                  Messages can be of type ChatMessage, ChatMessageSystem, ChatMessageUser,
                  ChatMessageAssistant, or ChatMessageTool.

        Raises:
            NotImplementedError: This is an abstract method that must be implemented by concrete tool classes.
        """
        raise NotImplementedError(f"method 'validate_tool_call_args' is not implemented for tool {self.name}")

    async def execute(self, ctx: Optional[Any], tool_call: ToolCall, args: Dict[str, Any], ) -> (bool, List[ChatMessage]):
        """
        Execute the tool with the provided tool call and return execution results.

        Args:
            ctx (Any): global context e.g. configuration settings, user session data
            tool_call (ToolCall): The tool call containing function details and parameters
                                 to be executed. Includes function name and parameter
                                 values for this specific execution
            args (Dict[str, Any]): The arguments to be used in the execution


        Returns:
            tuple: A pair containing:
                - bool: Success status of the execution
                - List[ChatMessage]: List of chat messages generated during tool execution.
                  Messages can be of type ChatMessage, ChatMessageSystem, ChatMessageUser,
                  ChatMessageAssistant, or ChatMessageTool.

        Raises:
            NotImplementedError: This is an abstract method that must be implemented by concrete tool classes.
        """
        raise NotImplementedError(f"method 'execute' is not implemented for tool {self.name}")

    def as_chat_tool(self) -> ChatTool:
        """
        Convert the tool instance into a ChatTool configuration object.

        This method should create a ChatTool instance that describes the tool's interface,
        including its function name, description, and parameter specifications. This configuration
        is used by the chat system to understand how to interact with the tool.

        Returns:
            ChatTool: A configuration object containing the tool's type, function details,
                     and parameter specifications.

        Raises:
            NotImplementedError: This is an abstract method that must be implemented by concrete tool classes.
        """
        raise NotImplementedError(f"method 'as_chat_tool' is not implemented for tool {self.name}")

    def props(self) -> ToolProps:
        raise NotImplementedError(f"method 'props' is not implemented for tool {self.name}")
