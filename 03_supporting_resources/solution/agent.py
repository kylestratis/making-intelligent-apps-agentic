"""CLI chat agent with MCP tool and resource support.

This demonstrates an intelligent application that can both use tools and
inject resource context into conversations. Resources provide background
information that helps Claude answer questions more effectively.
"""

import asyncio
import os
from pathlib import Path

from anthropic import Anthropic
from client import MCPClient
from dotenv import load_dotenv
from mcp import MCPError
from mcp.types import TextResourceContents

# Load environment variables from .env file
load_dotenv()


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


async def main():
    """Run the interactive chat agent with MCP tool and resource support."""
    # Initialize the Anthropic client
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable not set. "
            "Please create a .env file with your API key."
        )

    client = Anthropic(api_key=api_key)

    # Initialize MCP client for calculator server
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

    # Connect to the MCP server and discover capabilities
    await mcp_client.connect()
    available_tools = await mcp_client.get_available_tools()
    available_resources = await mcp_client.get_available_resources()

    # Display server capabilities to the user
    tool_names = [tool["name"] for tool in available_tools]
    print("Connected to MCP server")
    print(f"  Tools: {', '.join(tool_names)}")
    print(f"  Resources: {', '.join(r.name for r in available_resources) or 'none'}\n")

    print("Welcome to your AI Assistant with tools and resources!")
    print("Type your message and press Enter to chat.")
    print("Type 'quit' or 'exit' to end the conversation.\n")

    try:
        while True:
            # Get user input
            user_input = input("You: ").strip()

            # Check for exit commands
            if user_input.lower() in ["quit", "exit", "goodbye"]:
                print("\nAssistant: Goodbye! Have a great day!")
                break

            # Skip empty inputs
            if not user_input:
                continue

            # Each user message starts a new conversation (no history is kept).
            # Resources are loaded as extra context in the user message.
            # For simplicity we load all of them; a smarter client would select.
            user_content = [{"type": "text", "text": user_input}]
            if available_resources:
                user_content.extend(
                    await load_resource_context(
                        mcp_client, [r.uri for r in available_resources]
                    )
                )
            conversation_messages = [{"role": "user", "content": user_content}]

            # Agentic loop - continue until we get a final text response
            # This allows Claude to use multiple tools and reason about results
            while True:
                # Call Claude API with available tools
                response = client.messages.create(
                    model="claude-sonnet-5-5",
                    max_tokens=4096,
                    messages=conversation_messages,
                    tools=available_tools,
                    tool_choice={"type": "auto"},  # Let Claude decide when to use tools
                )

                # Add assistant's response to conversation
                conversation_messages.append(
                    {"role": "assistant", "content": response.content}
                )

                # Check if Claude wants to use tools
                if response.stop_reason == "tool_use":
                    # Extract all tool use blocks from the response
                    tool_use_blocks = [
                        block for block in response.content if block.type == "tool_use"
                    ]

                    # Execute each tool and collect results
                    tool_results = []
                    for tool_use in tool_use_blocks:
                        print(f"  [Using tool: {tool_use.name}]")

                        # Call the tool via MCP client
                        tool_result = await mcp_client.use_tool(
                            tool_name=tool_use.name, arguments=tool_use.input
                        )

                        # Format result for Claude; is_error tells the model
                        # the call failed so it can recover or explain
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use.id,
                                "content": "\n".join(tool_result["content"]),
                                "is_error": tool_result["is_error"],
                            }
                        )

                    # Add tool results to conversation as a user message
                    # This allows Claude to see the results and continue reasoning
                    conversation_messages.append(
                        {"role": "user", "content": tool_results}
                    )

                    # Continue the loop - Claude will process results
                    continue

                else:
                    # No more tool use - extract final text response
                    text_blocks = [
                        block.text
                        for block in response.content
                        if block.type == "text" and block.text.strip()
                    ]

                    if text_blocks:
                        print(f"\nAssistant: {text_blocks[0]}\n")
                    else:
                        print("\nAssistant: [No text response available]\n")

                    # Break out of the agentic loop
                    break

    finally:
        # Always disconnect from MCP server when done
        await mcp_client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
