"""Tests for step 03: reading resources and following resource links."""

from conftest import connected_client

STEP = "03_supporting_resources"


async def test_get_resource_reads_text():
    """Exercise: get_resource returns the text contents of a resource."""
    async with connected_client(STEP) as client:
        contents = await client.get_resource("resource://math-constants")
    assert "Mathematical Constants" in contents[0].text


async def test_link_resolves_to_same_content():
    """Exercise: use_tool follows a resource_link and returns the content."""
    async with connected_client(STEP) as client:
        contents = await client.get_resource("resource://math-constants")
        result = await client.use_tool("get_constants_reference", {})
    assert result["is_error"] is False
    assert contents[0].text in "\n".join(result["content"])
