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
from mcp.types import (
    BlobResourceContents,
    Prompt,
    PromptMessage,
    Resource,
    ResourceTemplate,
    TextResourceContents,
)

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

    async def get_available_resources(self) -> list[Resource]:
        """Retrieve the list of available resources from the server.

        Resources are static or dynamic data that the server can provide,
        such as files, database contents, API data, etc.

        Returns:
            List of Resource objects containing metadata (name, URI, description).

        Raises:
            RuntimeError: If the client is not connected to a server.
        """
        if not self._connected:
            raise RuntimeError("Client not connected to a server")

        resources_result = await self._session.list_resources()

        if not resources_result.resources:
            logger.warning("No resources found on server")

        return resources_result.resources

    async def get_available_resource_templates(self) -> list[ResourceTemplate]:
        """Retrieve resource templates from the server.

        Resource templates are patterns for dynamic resources that can be
        instantiated with parameters (e.g., "file://{path}" template).

        Returns:
            List of ResourceTemplate objects.

        Raises:
            RuntimeError: If the client is not connected to a server.
        """
        if not self._connected:
            raise RuntimeError("Client not connected to a server")

        templates_result = await self._session.list_resource_templates()

        if not templates_result.resourceTemplates:
            logger.warning("No resource templates found on server")

        return templates_result.resourceTemplates

    async def get_resource(
        self, uri: str
    ) -> list[TextResourceContents | BlobResourceContents]:
        """Fetch the actual content of a resource by its URI.

        Args:
            uri: The unique resource identifier (e.g., "file:///path/to/file").

        Returns:
            List of resource contents (text or binary blobs).

        Raises:
            RuntimeError: If the client is not connected to a server.
        """
        if not self._connected:
            raise RuntimeError("Client not connected to a server")

        logger.debug(f"Reading resource: {uri}")

        # Request the resource content from the server
        resource_result = await self._session.read_resource(uri=uri)

        if not resource_result.contents:
            logger.warning(f"No content returned for resource URI: {uri}")

        return resource_result.contents

    async def get_available_prompts(self) -> list[Prompt]:
        """Retrieve the list of available prompt templates from the server.

        Prompts are reusable templates that provide instructions or guidance
        for the LLM. They can include parameters that get filled in at runtime.

        Returns:
            List of Prompt objects containing metadata (name, description, arguments).

        Raises:
            RuntimeError: If the client is not connected to a server.
        """
        if not self._connected:
            raise RuntimeError("Client not connected to a server")

        prompts_result = await self._session.list_prompts()

        if not prompts_result.prompts:
            logger.warning("No prompts found on server")

        return prompts_result.prompts

    async def load_prompt(
        self, name: str, arguments: dict[str, str] | None = None
    ) -> list[PromptMessage]:
        """Load a specific prompt template with the given arguments.

        This fetches a prompt from the server and fills in any parameters.
        The result can be used as system instructions or conversation context.

        Args:
            name: The name of the prompt to load.
            arguments: Dictionary of argument values to fill in the prompt template.

        Returns:
            List of PromptMessage objects containing the rendered prompt.

        Raises:
            RuntimeError: If the client is not connected to a server.
        """
        if not self._connected:
            raise RuntimeError("Client not connected to a server")

        logger.debug(f"Loading prompt '{name}' with arguments: {arguments}")

        # Request the prompt from the server
        prompt_result = await self._session.get_prompt(
            name=name, arguments=arguments or {}
        )

        if not prompt_result.messages:
            logger.warning(f"No messages returned for prompt: {name}")

        return prompt_result.messages

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
