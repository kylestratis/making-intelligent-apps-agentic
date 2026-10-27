"""MCP Client module for connecting to and interacting with MCP servers.

Step 03 adds resource discovery and reading (``get_available_resources``,
``get_resource``) and makes ``use_tool`` follow ``resource_link`` results.
"""

import logging
from contextlib import AsyncExitStack
from typing import Any, TypedDict

from mcp import Client
from mcp.client.stdio import StdioServerParameters
from mcp.types import (
    BlobResourceContents,
    Resource,
    ResourceTemplate,
    TextResourceContents,
)

logger = logging.getLogger(__name__)


class ToolResult(TypedDict):
    """Result of a tool call, shaped for an Anthropic ``tool_result`` block.

    Attributes:
        content: Text (or base64 data) produced by the tool, one entry per
            content block.
        is_error: True when the server reported that the tool call failed.
    """

    content: list[str]
    is_error: bool


def contents_to_text(
    contents: list[TextResourceContents | BlobResourceContents],
) -> list[str]:
    """Flatten resource contents into strings (text, or base64 for blobs).

    Args:
        contents: Contents returned by ``read_resource``.

    Returns:
        One string per content item.
    """
    return [
        item.text if isinstance(item, TextResourceContents) else item.blob
        for item in contents
    ]


class MCPClient:
    """MCP Client for connecting to and interacting with MCP servers.

    This client owns the connect/disconnect lifecycle. An ``AsyncExitStack``
    holds the SDK ``Client`` so a single ``disconnect()`` call tears down the
    session and the server subprocess.

    Attributes:
        name: Friendly name for this client connection.
        command: The command to execute to start the MCP server.
        server_args: Arguments to pass to the server command.
        env_vars: Optional environment variables for the server process.
    """

    def __init__(
        self,
        name: str,
        command: str,
        server_args: list[str],
        env_vars: dict[str, str] | None = None,
    ) -> None:
        """Initialize the MCPClient with server connection parameters.

        Args:
            name: Friendly name for this client connection.
            command: The command to execute (e.g., "uv", "python", "node").
            server_args: List of arguments to pass to the command.
            env_vars: Optional environment variables for the server process.
        """
        self.name = name
        self.command = command
        self.server_args = server_args
        self.env_vars = env_vars
        self._client: Client | None = None
        self._exit_stack: AsyncExitStack = AsyncExitStack()
        self._connected: bool = False

    async def connect(self) -> None:
        """Connect to the MCP server via stdio.

        Raises:
            RuntimeError: If the client is already connected.
        """
        if self._connected:
            raise RuntimeError("Client is already connected")

        server_parameters = StdioServerParameters(
            command=self.command,
            args=self.server_args,
            env=self.env_vars if self.env_vars else None,
        )

        self._client = await self._exit_stack.enter_async_context(
            Client(server_parameters)
        )
        self._connected = True

    async def get_available_tools(self) -> list[dict[str, Any]]:
        """Retrieve tools that the server has made available.

        Returns:
            List of tool definitions in Claude's tool-use format, each with
            ``name``, ``description``, and ``input_schema``.

        Raises:
            RuntimeError: If the client is not connected to a server.
        """
        if not self._connected:
            raise RuntimeError("Client not connected to a server")

        tools_result = await self._client.list_tools()

        if not tools_result.tools:
            logger.warning("No tools found on server")

        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            for tool in tools_result.tools
        ]

    async def get_available_resources(self) -> list[Resource]:
        """Retrieve the list of available resources from the server.

        Returns:
            List of Resource objects containing metadata (name, URI, description).

        Raises:
            RuntimeError: If the client is not connected to a server.
        """
        if not self._connected:
            raise RuntimeError("Client not connected to a server")

        resources_result = await self._client.list_resources()

        if not resources_result.resources:
            logger.warning("No resources found on server")

        return resources_result.resources

    async def get_available_resource_templates(self) -> list[ResourceTemplate]:
        """Retrieve resource templates (parameterized resource patterns).

        Returns:
            List of ResourceTemplate objects.

        Raises:
            RuntimeError: If the client is not connected to a server.
        """
        if not self._connected:
            raise RuntimeError("Client not connected to a server")

        templates_result = await self._client.list_resource_templates()

        if not templates_result.resource_templates:
            logger.warning("No resource templates found on server")

        return templates_result.resource_templates

    async def get_resource(
        self, uri: str
    ) -> list[TextResourceContents | BlobResourceContents]:
        """Fetch the actual content of a resource by its URI.

        Exercise:
            Call ``self._client.read_resource(uri)`` and return the ``contents``
            list of the result (``TextResourceContents`` or
            ``BlobResourceContents`` items).

        Args:
            uri: The unique resource identifier (e.g., "resource://math-constants").

        Returns:
            List of resource contents (text or binary blobs).

        Raises:
            RuntimeError: If the client is not connected to a server.
            NotImplementedError: Until the exercise is completed.
        """
        if not self._connected:
            raise RuntimeError("Client not connected to a server")

        # TODO: read the resource from the server and return its contents.
        raise NotImplementedError("TODO: implement get_resource")

    async def use_tool(
        self, tool_name: str, arguments: dict[str, Any] | None = None
    ) -> ToolResult:
        """Execute a tool on the MCP server.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Dictionary of arguments for the tool.

        Returns:
            The tool's content as strings plus whether the call failed.
            Exercise: follow resource links via ``get_resource``.

        Raises:
            RuntimeError: If the client is not connected to a server.
        """
        if not self._connected:
            raise RuntimeError("Client not connected to a server")

        logger.debug(f"Calling tool {tool_name} with arguments {arguments}")

        tool_call_result = await self._client.call_tool(tool_name, arguments)

        results: list[str] = []
        for content in tool_call_result.content:
            match content.type:
                case "text":
                    results.append(content.text)
                case "image" | "audio":
                    # Binary data (base64 encoded)
                    results.append(content.data)
                case "resource":
                    # Resource embedded directly in the result
                    results.extend(contents_to_text([content.resource]))
                case "resource_link":
                    # TODO: follow the link. Load the linked resource with
                    # self.get_resource(content.uri) and add its text to `results`
                    # (contents_to_text helps). For now we only report the link.
                    results.append(f"{content.name}: {content.uri}")

        if not results:
            logger.warning(f"No content in tool call result for tool {tool_name}")

        return {"content": results, "is_error": tool_call_result.is_error}

    async def disconnect(self) -> None:
        """Clean up resources and disconnect from the server.

        Closing the exit stack exits the SDK ``Client``, which closes the
        session and terminates the server subprocess.
        """
        await self._exit_stack.aclose()
        self._connected = False
        self._client = None
