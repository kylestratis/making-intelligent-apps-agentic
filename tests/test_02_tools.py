"""Tests for step 02: tool discovery, tool results, errors, and resource links."""

from conftest import connected_client

STEP = "02_supporting_tools"


async def test_add_returns_text():
    """A normal tool call returns its text and is not an error."""
    async with connected_client(STEP) as client:
        result = await client.use_tool("add", {"a": 2, "b": 3})
    assert result["is_error"] is False
    assert "5" in result["content"][0]


async def test_divide_by_zero_is_error():
    """Exercise (a): a failing tool call must set is_error."""
    async with connected_client(STEP) as client:
        result = await client.use_tool("divide", {"a": 1, "b": 0})
    assert result["is_error"] is True
    assert "zero" in result["content"][0].lower()


async def test_resource_link_returns_name_and_uri():
    """Exercise (b): a resource_link result reports its name and URI as text."""
    async with connected_client(STEP) as client:
        result = await client.use_tool("get_constants_reference", {})
    text = "\n".join(result["content"])
    assert "resource://math-constants" in text
    assert "math-constants" in text
