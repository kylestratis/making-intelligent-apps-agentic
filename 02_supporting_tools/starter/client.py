"""MCP Client module for connecting to and interacting with MCP servers.

Step 02 adds tool discovery (``get_available_tools``) and tool execution
(``use_tool``) on top of the connection lifecycle from step 01.
"""

import logging
from contextlib import AsyncExitStack
from typing import Any, TypedDict

from mcp import Client
from mcp.client.stdio import StdioServerParameters
from mcp.types import TextResourceContents

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

    async def use_tool(
        self, tool_name: str, arguments: dict[str, Any] | None = None
    ) -> ToolResult:
        """Execute a tool on the MCP server.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Dictionary of arguments for the tool.

        Returns:
            The tool's content as strings plus whether the call failed.

        Exercise:
            (a) Propagate ``is_error`` from the server's ``CallToolResult``.
            (b) Handle the ``resource_link`` content type.

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
                    if isinstance(content.resource, TextResourceContents):
                        results.append(content.resource.text)
                    else:
                        results.append(content.resource.blob)
                case "resource_link":
                    # TODO (b): a resource_link points at a resource instead of
                    # embedding it. Append "<name>: <uri>" to `results` so Claude
                    # can see what the link refers to.
                    raise NotImplementedError("TODO (b): handle resource_link")

        if not results:
            logger.warning(f"No content in tool call result for tool {tool_name}")

        # TODO (a): report whether the call failed. The CallToolResult has an
        # `is_error` field; return it instead of the hardcoded False below.
        return {"content": results, "is_error": False}

    async def disconnect(self) -> None:
        """Clean up resources and disconnect from the server.

        Closing the exit stack exits the SDK ``Client``, which closes the
        session and terminates the server subprocess.
        """
        await self._exit_stack.aclose()
        self._connected = False
        self._client = None
