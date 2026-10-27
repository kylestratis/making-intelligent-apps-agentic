"""Tests for step 05: elicitation callback and roots, end to end over stdio."""

from conftest import ROOT, connected_client, load_client_module
from mcp.types import ElicitRequestFormParams

STEP = "05_elicitation"
FORM = ElicitRequestFormParams(
    message="Tell us about you",
    requested_schema={
        "type": "object",
        "properties": {
            "name": {"type": "string", "title": "Name"},
            "email": {"type": "string", "title": "Email"},
        },
        "required": ["name", "email"],
    },
)


def script_input(monkeypatch, answers: list[str]) -> None:
    """Replace ``input()`` so each call returns the next scripted answer."""
    replies = iter(answers)
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(replies))


async def test_handler_accepts_filled_form(monkeypatch):
    """Exercise: filled-in fields produce an accept result with the values."""
    script_input(monkeypatch, ["Ada", "ada@example.com"])
    module = load_client_module(STEP)
    result = await module.handle_elicitation(None, FORM)
    assert result.action == "accept"
    assert result.content == {"name": "Ada", "email": "ada@example.com"}


async def test_handler_declines_empty_input(monkeypatch):
    """Exercise: an empty required field produces a decline result."""
    script_input(monkeypatch, [""])
    module = load_client_module(STEP)
    result = await module.handle_elicitation(None, FORM)
    assert result.action == "decline"


async def test_signup_tool_accepted_end_to_end(monkeypatch):
    """The server's signup tool completes after the client answers the form."""
    script_input(monkeypatch, ["Ada", "ada@example.com"])
    async with connected_client(STEP) as client:
        result = await client.use_tool("signup_math_facts", {})
    assert result["is_error"] is False
    assert "Welcome Ada" in result["content"][0]


async def test_signup_tool_declined_end_to_end(monkeypatch):
    """Declining the form is reported by the server, not treated as an error."""
    script_input(monkeypatch, [""])
    async with connected_client(STEP) as client:
        result = await client.use_tool("signup_math_facts", {})
    assert result["is_error"] is False
    assert "chose not to sign up" in result["content"][0]


async def test_count_files_uses_client_roots():
    """count_files works inside the shared root and is refused outside it."""
    async with connected_client(STEP) as client:
        inside = await client.use_tool("count_files", {"directory": str(ROOT)})
        outside = await client.use_tool("count_files", {"directory": "/"})
    assert inside["is_error"] is False
    assert outside["is_error"] is True
