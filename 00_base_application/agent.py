"""Simple CLI chat agent using Claude API.

This demonstrates a basic intelligent application that maintains conversation
history and provides a clean command-line interface, and will the base of our
project.
"""

import os

from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def main():
    """Run the interactive chat agent."""
    # Initialize the Anthropic client
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable not set. "
            "Please create a .env file with your API key."
        )

    client = Anthropic(api_key=api_key)

    # System prompt defines the assistant's behavior
    system_prompt = "You are a helpful assistant."

    # Conversation history maintains context across turns
    conversation_history = []

    print("Welcome to your AI Assistant!")
    print("Type your message and press Enter to chat.")
    print("Type 'quit' or 'exit' to end the conversation.\n")

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


if __name__ == "__main__":
    main()
