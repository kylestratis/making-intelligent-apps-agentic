"""CLI chat agent with MCP client integration.

This demonstrates an intelligent application that connects to an MCP server
and maintains conversation history. The MCP client is instantiated but not
yet used for tool calls - that will come in a later step.
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
    """Run the interactive chat agent with MCP client."""
    # Initialize the Anthropic client
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable not set. "
            "Please create a .env file with your API key."
        )

    client = Anthropic(api_key=api_key)

    # Initialize MCP client for calculator server
    # The client connects to a server that provides calculator tools
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

    # Connect to the MCP server
    await mcp_client.connect()

    # System prompt defines the assistant's behavior
    system_prompt = "You are a helpful assistant."

    # Conversation history maintains context across turns
    conversation_history = []

    print("Welcome to your AI Assistant!")
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

            # Add user message to conversation history
            conversation_history.append({
                "role": "user",
                "content": user_input,
            })

            # Call Claude API with conversation history
            # Note: We're not using MCP tools yet - that comes in the next step
            response = client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=1024,
                system=system_prompt,
                messages=conversation_history,
            )

            # Extract assistant's response
            assistant_message = response.content[0].text

            # Add assistant's response to conversation history
            conversation_history.append({
                "role": "assistant",
                "content": assistant_message,
            })

            # Display the response
            print(f"\nAssistant: {assistant_message}\n")

    finally:
        # Always disconnect from MCP server when done
        await mcp_client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
