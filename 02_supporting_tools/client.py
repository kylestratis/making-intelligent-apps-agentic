"""MCP Client module for connecting to and interacting with MCP servers.

This module provides the MCPClient class which manages connections to MCP servers
via stdio (standard input/output) and provides methods to interact with server
capabilities like tools, resources, and prompts.
"""

import logging
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import TextResourceContents

logger = logging.getLogger(__name__)


class MCPClient:
    """MCP Client for connecting to and interacting with MCP servers.

    This client handles the lifecycle of connecting to an MCP server via stdio,
    maintaining a session, and providing methods to interact with server features.

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
        self._session: ClientSession | None = None
        self._exit_stack: AsyncExitStack = AsyncExitStack()
        self._connected: bool = False

    async def connect(self) -> None:
        """Connect to the MCP server via stdio.

        This method:
        1. Creates server parameters for stdio connection
        2. Establishes a stdio connection to the server subprocess
        3. Creates a ClientSession for MCP protocol communication
        4. Initializes the session with the server

        Raises:
            RuntimeError: If the client is already connected.
        """
        if self._connected:
            raise RuntimeError("Client is already connected")

        # Configure server connection parameters
        server_parameters = StdioServerParameters(
            command=self.command,
            args=self.server_args,
            env=self.env_vars if self.env_vars else None,
        )

        # Connect to stdio server, starting the subprocess
        stdio_connection = await self._exit_stack.enter_async_context(
            stdio_client(server_parameters)
        )
        self.read, self.write = stdio_connection

        # Start MCP client session with read/write streams
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream=self.read, write_stream=self.write)
        )

        # Initialize the session (handshake with server)
        await self._session.initialize()
        self._connected = True

    async def get_available_tools(self) -> list[dict[str, Any]]:
        """Retrieve tools that the server has made available.

        This method queries the MCP server for its list of available tools
        and formats them in a way compatible with Claude's tool use API.

        Returns:
            List of tool definitions, each containing:
                - name: Tool identifier
                - description: What the tool does
                - input_schema: JSON schema for tool parameters

        Raises:
            RuntimeError: If the client is not connected to a server.
        """
        if not self._connected:
            raise RuntimeError("Client not connected to a server")

        # Request tools list from the MCP server
        tools_result = await self._session.list_tools()

        if not tools_result.tools:
            logger.warning("No tools found on server")

        # Format tools for Claude API compatibility
        available_tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema,
            }
            for tool in tools_result.tools
        ]

        return available_tools

    async def use_tool(
        self, tool_name: str, arguments: dict[str, Any] | None = None
    ) -> list[str]:
        """Execute a tool on the MCP server.

        This method calls a tool on the MCP server and processes the results.
        It handles different types of content that tools can return (text,
        images, resources, etc.).

        Args:
            tool_name: Name of the tool to execute.
            arguments: Dictionary of arguments for the tool.

        Returns:
            List of string results from the tool execution.

        Raises:
            RuntimeError: If the client is not connected to a server.
        """
        if not self._connected:
            raise RuntimeError("Client not connected to a server")

        logger.debug(f"Calling tool {tool_name} with arguments {arguments}")

        # Execute the tool on the MCP server
        tool_call_result = await self._session.call_tool(
            name=tool_name, arguments=arguments
        )

        # Process tool results - handle different content types
        results = []
        if tool_call_result.content:
            for content in tool_call_result.content:
                match content.type:
                    case "text":
                        # Most common case - text results
                        results.append(content.text)
                    case "image" | "audio":
                        # Binary data (base64 encoded)
                        results.append(content.data)
                    case "resource":
                        # Resource references
                        if isinstance(content.resource, TextResourceContents):
                            results.append(content.resource.text)
                        else:
                            results.append(content.resource.blob)
        else:
            logger.warning(f"No content in tool call result for tool {tool_name}")

        return results

    async def disconnect(self) -> None:
        """Clean up resources and disconnect from the server.

        This method closes the exit stack, which automatically:
        - Closes the ClientSession
        - Terminates the stdio connection
        - Cleans up the server subprocess
        """
        if self._exit_stack:
            await self._exit_stack.aclose()
            self._connected = False
            self._session = None
