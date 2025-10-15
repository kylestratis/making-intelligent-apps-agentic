"""CLI chat agent with full MCP support: tools, resources, and prompts.

This demonstrates a complete intelligent application that leverages all three
MCP primitive types:
- Tools: Claude decides when to execute during conversation
- Resources: Pre-loaded context injected at message start
- Prompts: Template instructions that guide Claude's behavior
"""

import asyncio
import os
from pathlib import Path

from anthropic import Anthropic
from client import MCPClient
from dotenv import load_dotenv
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

            # Process each piece of content in the resource
            for content in resource_contents:
                if isinstance(content, TextResourceContents):
                    # Add text content with a label
                    context_blocks.append(
                        {
                            "type": "text",
                            "text": f"[Resource: {uri}]\n{content.text}",
                        }
                    )
                elif content.mimeType and content.mimeType.startswith("image/"):
                    # Add image content (base64 encoded)
                    context_blocks.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": content.mimeType,
                                "data": content.blob,
                            },
                        }
                    )
                else:
                    print(f"  [Warning: Unsupported content type {content.mimeType}]")

        except Exception as e:
            print(f"  [Error loading resource {uri}: {e}]")

    return context_blocks


async def load_prompt_as_system(
    mcp_client: MCPClient, prompt_name: str, arguments: dict[str, str] | None = None
) -> str:
    """Load a prompt template and convert it to a system message.

    Prompts from MCP servers are typically used to guide Claude's behavior,
    similar to system prompts.

    Args:
        mcp_client: The MCP client instance.
        prompt_name: Name of the prompt to load.
        arguments: Optional parameters to fill into the prompt template.

    Returns:
        Formatted system instruction string.
    """
    try:
        print(f"  [Loading prompt: {prompt_name}]")
        prompt_messages = await mcp_client.load_prompt(prompt_name, arguments or {})

        # Extract text from prompt messages
        # Prompts can contain multiple messages with different roles
        system_parts = []
        for message in prompt_messages:
            # Handle both string content and structured content
            if isinstance(message.content, str):
                system_parts.append(message.content)
            elif hasattr(message.content, "text"):
                system_parts.append(message.content.text)

        return "\n\n".join(system_parts)

    except Exception as e:
        print(f"  [Error loading prompt {prompt_name}: {e}]")
        return ""


async def main():
    """Run the interactive chat agent with full MCP support."""
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
            str(Path(__file__).parent.parent.resolve()),
            "run",
            "calculator_server.py",
        ],
    )

    # Connect to the MCP server and discover all capabilities
    await mcp_client.connect()
    available_tools = await mcp_client.get_available_tools()
    available_resources = await mcp_client.get_available_resources()
    available_prompts = await mcp_client.get_available_prompts()

    # Display server capabilities to the user
    tool_names = [tool["name"] for tool in available_tools]
    print("Connected to MCP server")
    print(f"  Tools: {', '.join(tool_names) if tool_names else 'none'}")
    print(f"  Resources: {len(available_resources)} available")
    print(f"  Prompts: {len(available_prompts)} available")
    if available_prompts:
        print(f"    Available prompts: {', '.join(p.name for p in available_prompts)}")
    print()

    print("Welcome to your AI Assistant with complete MCP support!")
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

            # Build user message with optional resource context
            user_content = [{"type": "text", "text": user_input}]

            # Load resource context if available
            # For this example, we load all resources as context
            # In production, you might intelligently select relevant resources
            if available_resources:
                resource_uris = [r.uri for r in available_resources]
                resource_context = await load_resource_context(
                    mcp_client, resource_uris
                )
                user_content.extend(resource_context)

            # Load prompt as system instruction if available
            # For this example, we check if the user's query matches a prompt pattern
            # and load it with the appropriate arguments
            system_instruction = "You are a helpful assistant."
            if available_prompts:
                # Check if we have a calculate_operation prompt and if the user
                # is asking for a calculation
                for prompt in available_prompts:
                    if prompt.name == "calculate_operation":
                        # Pass the user's input as the operation argument
                        prompt_text = await load_prompt_as_system(
                            mcp_client, prompt.name, {"operation": user_input}
                        )
                        if prompt_text:
                            system_instruction = prompt_text
                        break

            # Start conversation for this turn
            conversation_messages = [{"role": "user", "content": user_content}]

            # Agentic loop - continue until we get a final text response
            while True:
                # Call Claude API with tools and system instruction
                response = client.messages.create(
                    model="claude-sonnet-4-5-20250929",
                    max_tokens=4096,
                    system=system_instruction,  # Prompts guide behavior here
                    messages=conversation_messages,
                    tools=available_tools if available_tools else None,
                    tool_choice={"type": "auto"} if available_tools else None,
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

                        # Format result for Claude
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use.id,
                                "content": "\n".join(tool_result),
                            }
                        )

                    # Add tool results to conversation
                    conversation_messages.append(
                        {"role": "user", "content": tool_results}
                    )

                    # Continue the loop - Claude will process results
                    continue

                else:
                    # No more tool use - extract final text response
                    text_blocks = [
                        content.text
                        for content in response.content
                        if hasattr(content, "text") and content.text.strip()
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
