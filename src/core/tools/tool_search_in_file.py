import json

from typing import Dict, Any, List

from core.logger import warn, error
from core.tools.tool_context import ToolContext
from openai_wrappers.api_vector_store import VectorStoreSearch, vector_store_search
from chat_tools.tool_usage.tool_abstract import Tool, build_tool_call, ToolProps
from chat_tools.chat_models import (
    ToolCall, ChatMessage, ChatTool,
    ChatToolFunction, ChatToolParameters,
    ChatToolParameterProperty, ChatMessageContentItemText
)


SYSTEM = """TOOL: search_in_doc
        Use this tool to retrieve relevant information from a document
        
        When search_in_doc is called, vector search is performed: search(query, chunks_of_doc), and most relevant chunks are retrieved
        
        Call when any condition bellow is met:
        * User asks a question about a document or related to a document
        * User directly asks to search in the document
        
        Prerequisites:
        * call list_documents to ensure the document exists and is ready
        
        Arguments:
        * query: descriptive, brief query used in search
        * document_name: document name retrieved from the list_documents tool
        * filters: optional dictionary to refine search results e.g. {"key": "value"}, retrieved from the list_documents tool
          filters should only be used when user directly asks to use them.
          
        search_in_doc produces:
        * pieces of text containing relevant information that will help you to answer user's question
"""


class SearchInFile(Tool):
    @property
    def name(self) -> str:
        return "search_in_doc"

    def validate_tool_call_args(self, ctx: ToolContext, tool_call: ToolCall, args: Dict[str, Any]) -> (bool, List[ChatMessage]):
        document_name = args.get("document_name")
        query = args.get("query")
        filters = args.get("filters")

        if not document_name or not query:
            return False, [
                build_tool_call(
                    f"Error validating tool {self.name}: Required args are either empty or missing. Non-empty 'document_name' and 'query' are required.",
                    tool_call
                )
            ]

        if filters:
            if not isinstance(filters, dict):
                return False, [
                    build_tool_call(
                        f"Error validating tool {self.name}: 'filters' should be a dictionary if provided.",
                        tool_call
                    )
                ]
            keys = list(filters.keys())
            if keys != ["section_name"]:
                return False, [
                    build_tool_call(
                        f"Error validating tool {self.name}: 'filters' keys could only contain 'section_name'.",
                        tool_call
                    )
                ]

        return True, []

    async def execute(self, ctx: ToolContext, tool_call: ToolCall, args: Dict[str, Any]) -> (bool, List[ChatMessage]):
        document_name = args.get("document_name")
        query = args.get("query")

        try:
            files = await ctx.files_repository.get_files_by_filter(
                "user_id=?",
                (ctx.user_id,)
            )
        except Exception as e:
            err = f"Error while executing tool {self.name}: couldn't get user's documents: {str(e)}"
            warn(err)
            return False, [
                build_tool_call(
                    err,tool_call
                )
            ]

        document = next((f for f in files if f.file_name_orig == document_name), None)
        if not document:
            return False, [
                build_tool_call(
                    f"Error while executing tool {self.name}: document {document_name} not found."
                    f"All documents:\n{json.dumps([f.model_dump() for f in files])}",
                    tool_call
                )
            ]

        post = VectorStoreSearch(
            vector_store_id=document.vector_store_id,
            query=query
        )

        try:
            resp = await vector_store_search(post)
        except Exception as e:
            err = f"Error while executing tool {self.name}: vector store search failed: {str(e)}"
            error(err)
            return False, [
                build_tool_call(
                    err,tool_call
                )
            ]

        content = []
        for obj in resp:
            for content_i in obj.content:
                content.append(ChatMessageContentItemText(
                    text=content_i.text,
                    type="text"
                ))

        if not content:
            return True, [
                build_tool_call(
                    f"No results found for the query: {query}.",
                    tool_call
                )
            ]
        return True, [
            build_tool_call(
                content, tool_call
            )
        ]

    def as_chat_tool(self) -> ChatTool:
        return ChatTool(
            type="function",
            function=ChatToolFunction(
                name="search_in_doc",
                description="Searches for relevant information within a specified document.",
                parameters=ChatToolParameters(
                    type="object",
                    properties={
                        "document_name": ChatToolParameterProperty(
                            type="string",
                            description="The name of the document to search within.",
                            enum=[]
                        ),
                        "query": ChatToolParameterProperty(
                            type="string",
                            description="The search query to find relevant information.",
                            enum=[]
                        ),
                    },
                    required=["document_name", "query"]
                )
            )
        )

    def props(self):
        return ToolProps(
            tool_name=self.name,
            system_prompt=SYSTEM,
            depends_on=["list_documents"]
        )
