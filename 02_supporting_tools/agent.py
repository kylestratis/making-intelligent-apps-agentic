"""CLI chat agent with MCP tool use capability.

This demonstrates an intelligent application that can discover and use tools
from an MCP server. The agent implements an agentic loop where Claude can
use tools multiple times and reason about the results before responding.
"""

import asyncio
import os
from pathlib import Path

from anthropic import Anthropic
from client import MCPClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


async def main():
    """Run the interactive chat agent with MCP tool support."""
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

    # Connect to the MCP server and get available tools
    await mcp_client.connect()
    available_tools = await mcp_client.get_available_tools()

    # Display available tools to the user
    tool_names = [tool["name"] for tool in available_tools]
    print(f"Connected to MCP server with tools: {', '.join(tool_names)}\n")

    print("Welcome to your AI Assistant with tool support!")
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

            # Start a new conversation for this user message
            # Note: For simplicity, we're not maintaining history between messages
            # A more advanced version would maintain full conversation history
            conversation_messages = [{"role": "user", "content": user_input}]

            # Agentic loop - continue until we get a final text response
            # This allows Claude to use multiple tools and reason about results
            while True:
                # Call Claude API with available tools
                response = client.messages.create(
                    model="claude-sonnet-4-5-20250929",
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

                        # Format result for Claude
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use.id,
                                "content": "\n".join(tool_result),
                            }
                        )

                    # Add tool results to conversation as a user message
                    # This allows Claude to see the results and continue reasoning
                    conversation_messages.append({"role": "user", "content": tool_results})

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
