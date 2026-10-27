"""MCP Client module for connecting to and interacting with MCP servers.

This module provides the MCPClient class, which manages the lifecycle of a
connection to an MCP server over stdio using the MCP Python SDK v2 ``Client``.
Later steps add methods for tools, resources, and prompts.
"""

from contextlib import AsyncExitStack
from typing import Any

from mcp import Client
from mcp.client.stdio import StdioServerParameters


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

        The SDK ``Client`` starts the server subprocess and performs the
        protocol handshake when it is entered, so there is no separate
        ``initialize()`` call and no read/write streams to manage.

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

    async def get_available_tools(self) -> list[Any]:
        """Retrieve tools that the server has made available.

        Returns:
            List of available tools from the server.

        Note:
            This method will be implemented in a future step.
        """

    async def use_tool(self, tool_name: str, arguments: dict[str, Any] | None = None):
        """Execute a tool on the MCP server.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Optional dictionary of arguments for the tool.

        Note:
            This method will be implemented in a future step.
        """

    async def disconnect(self) -> None:
        """Clean up resources and disconnect from the server.

        Closing the exit stack exits the SDK ``Client``, which closes the
        session and terminates the server subprocess.
        """
        await self._exit_stack.aclose()
        self._connected = False
        self._client = None
