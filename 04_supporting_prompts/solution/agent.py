"""CLI chat agent with full MCP support: tools, resources, and prompts.

This demonstrates an intelligent application that leverages all three MCP
primitive types:
- Tools: Claude decides when to execute during conversation
- Resources: Pre-loaded context injected with each user message
- Prompts: Templates the user picks with ``/prompt``; their rendered messages
  are added to the conversation with their own roles

Chat commands: ``/prompts`` lists prompt templates, ``/prompt <name>`` runs one.
"""

import asyncio
import os
from pathlib import Path

from anthropic import Anthropic
from client import MCPClient
from dotenv import load_dotenv
from mcp import MCPError
from mcp.types import Prompt, TextResourceContents

# Load environment variables from .env file
load_dotenv()

SYSTEM_PROMPT = "You are a helpful assistant."


async def load_resource_context(
    mcp_client: MCPClient, resource_uris: list[str]
) -> list[dict]:
    """Load resource contents and format them for Claude's API.

    Args:
        mcp_client: The MCP client instance.
        resource_uris: List of resource URIs to load.

    Returns:
        List of formatted content blocks (text or images) ready for Claude.
    """
    context_blocks = []

    for uri in resource_uris:
        try:
            print(f"  [Loading resource: {uri}]")
            resource_contents = await mcp_client.get_resource(uri)

            for content in resource_contents:
                if isinstance(content, TextResourceContents):
                    context_blocks.append(
                        {"type": "text", "text": f"[Resource: {uri}]\n{content.text}"}
                    )
                elif content.mime_type and content.mime_type.startswith("image/"):
                    context_blocks.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": content.mime_type,
                                "data": content.blob,
                            },
                        }
                    )
                else:
                    print(f"  [Warning: Unsupported content type {content.mime_type}]")

        except MCPError as e:
            print(f"  [Error loading resource {uri}: {e}]")

    return context_blocks


async def run_agent_loop(
    client: Anthropic,
    mcp_client: MCPClient,
    tools: list[dict],
    conversation: list[dict],
) -> None:
    """Call Claude repeatedly, executing tool calls, until it answers in text.

    Args:
        client: The Anthropic client.
        mcp_client: The MCP client used to execute tools.
        tools: Tool definitions in Claude's tool-use format.
        conversation: The running message history. Assistant turns and tool
            results are appended to it in place.
    """
    while True:
        response = client.messages.create(
            model="claude-sonnet-5-5",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=conversation,
            tools=tools,
            tool_choice={"type": "auto"},
        )
        conversation.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            text_blocks = [
                block.text
                for block in response.content
                if block.type == "text" and block.text.strip()
            ]
            print(
                f"\nAssistant: {text_blocks[0] if text_blocks else '[No text response]'}\n"
            )
            return

        tool_results = []
        for tool_use in (b for b in response.content if b.type == "tool_use"):
            print(f"  [Using tool: {tool_use.name}]")
            tool_result = await mcp_client.use_tool(
                tool_name=tool_use.name, arguments=tool_use.input
            )
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": "\n".join(tool_result["content"]),
                    "is_error": tool_result["is_error"],
                }
            )
        conversation.append({"role": "user", "content": tool_results})


def print_prompts(prompts: list[Prompt]) -> None:
    """Print the server's prompt templates and their arguments.

    Args:
        prompts: Prompts returned by ``MCPClient.get_available_prompts``.
    """
    for prompt in prompts:
        arguments = ", ".join(arg.name for arg in prompt.arguments or [])
        print(f"  /prompt {prompt.name}  ({arguments or 'no arguments'})")
        if prompt.description:
            print(f"      {prompt.description}")


def ask_prompt_arguments(prompt: Prompt) -> dict[str, str]:
    """Ask the user for a value for each argument of a prompt.

    Args:
        prompt: The prompt template to collect arguments for.

    Returns:
        Mapping of argument name to the value the user typed. Empty optional
        arguments are left out.
    """
    values = {}
    for arg in prompt.arguments or []:
        label = f"{arg.name} (required)" if arg.required else arg.name
        value = input(f"  {label}: ").strip()
        if value:
            values[arg.name] = value
    return values


async def main():
    """Run the interactive chat agent with full MCP support."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable not set. "
            "Please create a .env file with your API key."
        )

    client = Anthropic(api_key=api_key)

    mcp_client = MCPClient(
        name="calculator_server_connection",
        command="uv",
        server_args=[
            "--directory",
            str(Path(__file__).parents[2].resolve()),
            "run",
            "calculator_server.py",
        ],
    )

    # Connect to the MCP server and discover all capabilities
    await mcp_client.connect()
    available_tools = await mcp_client.get_available_tools()
    available_resources = await mcp_client.get_available_resources()
    available_prompts = await mcp_client.get_available_prompts()

    print("Connected to MCP server")
    print(f"  Tools: {', '.join(tool['name'] for tool in available_tools)}")
    print(f"  Resources: {', '.join(r.name for r in available_resources) or 'none'}")
    print(f"  Prompts: {', '.join(p.name for p in available_prompts) or 'none'}\n")

    print("Welcome to your AI Assistant with complete MCP support!")
    print("Type your message and press Enter to chat.")
    print("Type /prompts to list prompt templates, /prompt <name> to use one.")
    print("Type 'quit' or 'exit' to end the conversation.\n")

    conversation: list[dict] = []

    try:
        while True:
            user_input = input("You: ").strip()

            if user_input.lower() in ["quit", "exit", "goodbye"]:
                print("\nAssistant: Goodbye! Have a great day!")
                break

            if not user_input:
                continue

            if user_input == "/prompts":
                print_prompts(available_prompts)
                continue

            if user_input.startswith("/prompt "):
                name = user_input.removeprefix("/prompt ").strip()
                prompt = next((p for p in available_prompts if p.name == name), None)
                if prompt is None:
                    print(f"  [Unknown prompt: {name}. Try /prompts]\n")
                    continue
                print(f"  [Loading prompt: {name}]")
                arguments = ask_prompt_arguments(prompt)
                # Rendered messages join the conversation with their own roles
                conversation.extend(await mcp_client.load_prompt(name, arguments))
                if conversation[-1]["role"] != "user":
                    continue  # nothing for Claude to answer yet
            else:
                # Resources are loaded as extra context in the user message.
                # For simplicity we load all of them; a smarter client would select.
                user_content = [{"type": "text", "text": user_input}]
                user_content.extend(
                    await load_resource_context(
                        mcp_client, [r.uri for r in available_resources]
                    )
                )
                conversation.append({"role": "user", "content": user_content})

            await run_agent_loop(client, mcp_client, available_tools, conversation)

    finally:
        # Always disconnect from MCP server when done
        await mcp_client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
