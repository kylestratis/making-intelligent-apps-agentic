"""MCP Client module for connecting to and interacting with MCP servers.

Step 06 lets ``MCPClient`` connect to a remote server by URL (streamable HTTP)
as well as to a local server over stdio; the rest matches step 05. The client
also provides capabilities the server can call back into:
elicitation (``handle_elicitation`` asks the user for input) and roots
(``list_roots`` tells the server which directory it may use). With spec
2026-07-28 the server asks for these by returning ``input_required``; the SDK
``Client`` runs the callbacks and retries the tool call automatically.
"""

import logging
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, TypedDict

from mcp import Client
from mcp.client import ClientRequestContext
from mcp.client.stdio import StdioServerParameters
from mcp.types import (
    BlobResourceContents,
    ContentBlock,
    ElicitRequestParams,
    ElicitResult,
    ListRootsResult,
    Prompt,
    Resource,
    ResourceTemplate,
    Root,
    TextResourceContents,
)

logger = logging.getLogger(__name__)

# The directory shared with the server as a root (the repository root)
REPO_ROOT = Path(__file__).parents[1].resolve()


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


def to_anthropic_block(content: ContentBlock) -> dict[str, Any]:
    """Translate one MCP prompt content block into an Anthropic content block.

    Args:
        content: Content of a ``PromptMessage`` (text, image, or embedded resource).

    Returns:
        An Anthropic content block dict. Unsupported types become text.
    """
    match content.type:
        case "text":
            return {"type": "text", "text": content.text}
        case "image":
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": content.mime_type,
                    "data": content.data,
                },
            }
        case "resource":
            return {"type": "text", "text": contents_to_text([content.resource])[0]}
        case _:
            return {"type": "text", "text": str(content)}


async def handle_elicitation(
    context: ClientRequestContext, params: ElicitRequestParams
) -> ElicitResult:
    """Answer a server's elicitation request by prompting the user in the terminal.

    Args:
        context: SDK request context (unused).
        params: The server's request. For form requests, ``message`` explains
            why input is needed and ``requested_schema`` is a flat JSON schema
            whose ``properties`` are the fields to collect.

    Returns:
        ``accept`` with the typed values, or ``decline`` when the request is
        not a form or the user leaves a required field empty.
    """
    if params.mode != "form":
        return ElicitResult(action="decline")

    print(f"\n  [Server asks] {params.message}")
    properties = params.requested_schema.get("properties", {})
    required = params.requested_schema.get("required", [])
    convert = {"number": float, "integer": int}

    content = {}
    for field, spec in properties.items():
        label = spec.get("title", field)
        suffix = " (required)" if field in required else ""
        value = input(f"  {label}{suffix}: ").strip()
        if not value:
            if field in required:
                return ElicitResult(action="decline")
            continue
        content[field] = convert.get(spec.get("type"), str)(value)

    return ElicitResult(action="accept", content=content)


async def list_roots(context: ClientRequestContext) -> ListRootsResult:
    """Tell the server which directory it may work in, without asking the user.

    Args:
        context: SDK request context (unused).

    Returns:
        A single root pointing at the repository directory.
    """
    return ListRootsResult(roots=[Root(uri=REPO_ROOT.as_uri(), name="workshop repo")])


class MCPClient:
    """MCP Client for connecting to and interacting with MCP servers.

    This client owns the connect/disconnect lifecycle. An ``AsyncExitStack``
    holds the SDK ``Client`` so a single ``disconnect()`` call tears down the
    session (and the server subprocess, for stdio). Give it either a ``command``
    to launch a local server, or a ``url`` to reach a remote one.

    Attributes:
        name: Friendly name for this client connection.
        command: The command to execute to start a local MCP server.
        server_args: Arguments to pass to the server command.
        url: URL of a remote streamable HTTP server, such as
            ``http://localhost:8000/mcp``.
        env_vars: Optional environment variables for the server process.
    """

    def __init__(
        self,
        name: str,
        command: str | None = None,
        server_args: list[str] | None = None,
        url: str | None = None,
        env_vars: dict[str, str] | None = None,
    ) -> None:
        """Initialize the MCPClient with server connection parameters.

        Args:
            name: Friendly name for this client connection.
            command: The command to execute (e.g., "uv", "python", "node").
            server_args: List of arguments to pass to the command.
            url: URL of a remote server. Use instead of ``command``.
            env_vars: Optional environment variables for the server process.

        Raises:
            ValueError: If neither or both of ``command`` and ``url`` are given.
        """
        if (command is None) == (url is None):
            raise ValueError("Provide exactly one of command or url")
        self.name = name
        self.command = command
        self.server_args = server_args or []
        self.url = url
        self.env_vars = env_vars
        self._client: Client | None = None
        self._exit_stack: AsyncExitStack = AsyncExitStack()
        self._connected: bool = False

    async def connect(self) -> None:
        """Connect to the MCP server over stdio, or over HTTP when a URL is set.

        Raises:
            RuntimeError: If the client is already connected.
        """
        if self._connected:
            raise RuntimeError("Client is already connected")

        if self.url:
            server = self.url
        else:
            server = StdioServerParameters(
                command=self.command,
                args=self.server_args,
                env=self.env_vars if self.env_vars else None,
            )

        self._client = await self._exit_stack.enter_async_context(
            Client(
                server,
                elicitation_callback=handle_elicitation,
                list_roots_callback=list_roots,
            )
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

        Args:
            uri: The unique resource identifier (e.g., "resource://math-constants").

        Returns:
            List of resource contents (text or binary blobs).

        Raises:
            RuntimeError: If the client is not connected to a server.
        """
        if not self._connected:
            raise RuntimeError("Client not connected to a server")

        logger.debug(f"Reading resource: {uri}")
        resource_result = await self._client.read_resource(uri)

        if not resource_result.contents:
            logger.warning(f"No content returned for resource URI: {uri}")

        return resource_result.contents

    async def use_tool(
        self, tool_name: str, arguments: dict[str, Any] | None = None
    ) -> ToolResult:
        """Execute a tool on the MCP server.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Dictionary of arguments for the tool.

        Returns:
            The tool's content as strings plus whether the call failed.
            Resource links are followed and replaced by the linked content.

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
                    # A pointer to a resource: follow it and return its contents
                    results.extend(
                        contents_to_text(await self.get_resource(content.uri))
                    )

        if not results:
            logger.warning(f"No content in tool call result for tool {tool_name}")

        return {"content": results, "is_error": tool_call_result.is_error}

    async def get_available_prompts(self) -> list[Prompt]:
        """Retrieve the list of available prompt templates from the server.

        Returns:
            List of Prompt objects containing metadata (name, description, arguments).

        Raises:
            RuntimeError: If the client is not connected to a server.
        """
        if not self._connected:
            raise RuntimeError("Client not connected to a server")

        prompts_result = await self._client.list_prompts()

        if not prompts_result.prompts:
            logger.warning("No prompts found on server")

        return prompts_result.prompts

    async def load_prompt(
        self, name: str, arguments: dict[str, str] | None = None
    ) -> list[dict[str, Any]]:
        """Render a prompt template and translate it into Anthropic messages.

        Args:
            name: The name of the prompt to load.
            arguments: Values for the prompt's arguments.

        Returns:
            Messages like ``{"role": "user", "content": [{"type": "text", ...}]}``
            that can be appended to a conversation.

        Raises:
            RuntimeError: If the client is not connected to a server.
        """
        if not self._connected:
            raise RuntimeError("Client not connected to a server")

        logger.debug(f"Loading prompt '{name}' with arguments: {arguments}")
        prompt_result = await self._client.get_prompt(name, arguments)

        if not prompt_result.messages:
            logger.warning(f"No messages returned for prompt: {name}")

        return [
            {"role": message.role, "content": [to_anthropic_block(message.content)]}
            for message in prompt_result.messages
        ]

    async def disconnect(self) -> None:
        """Clean up resources and disconnect from the server.

        Closing the exit stack exits the SDK ``Client``, which closes the
        session and terminates the server subprocess.
        """
        await self._exit_stack.aclose()
        self._connected = False
        self._client = None
