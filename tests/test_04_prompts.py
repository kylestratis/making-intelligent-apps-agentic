"""Tests for step 04: listing prompts and rendering them as chat messages."""

from conftest import connected_client

STEP = "04_supporting_prompts"


async def test_prompt_is_listed():
    """The server's calculate_operation prompt is discoverable."""
    async with connected_client(STEP) as client:
        prompts = await client.get_available_prompts()
    assert "calculate_operation" in [prompt.name for prompt in prompts]


async def test_load_prompt_renders_argument_as_message():
    """Exercise: load_prompt returns Anthropic-style message dicts."""
    async with connected_client(STEP) as client:
        messages = await client.load_prompt(
            "calculate_operation", {"operation": "12 * 7"}
        )
    assert messages[0]["role"] == "user"
    assert "12 * 7" in messages[0]["content"][0]["text"]
