"""Tests for step 06: connecting to the server over streamable HTTP."""

import socket
import subprocess
import sys
import time

import pytest
from conftest import SERVER, load_client_module

URL = "http://localhost:8000/mcp"


def port_open() -> bool:
    """Return True when something is already listening on localhost:8000."""
    with socket.socket() as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", 8000)) == 0


@pytest.fixture
def http_server():
    """Start ``calculator_server.py --http`` and stop it after the test."""
    if port_open():
        pytest.skip("port 8000 is already in use")
    process = subprocess.Popen(
        [sys.executable, str(SERVER), "--http"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(100):
            if port_open():
                break
            time.sleep(0.1)
        else:
            pytest.fail("HTTP server did not start")
        yield
    finally:
        process.terminate()
        process.wait(timeout=10)


async def test_connects_over_http_and_calls_tool(http_server):
    """The step 06 client lists tools and runs one over streamable HTTP."""
    module = load_client_module("06_remote_servers")
    client = module.MCPClient(name="test", url=URL)
    await client.connect()
    try:
        names = [tool["name"] for tool in await client.get_available_tools()]
        result = await client.use_tool("add", {"a": 2, "b": 3})
    finally:
        await client.disconnect()
    assert "add" in names
    assert "5" in result["content"][0]


async def test_elicitation_works_over_http(http_server, monkeypatch):
    """The input_required round trip also completes over HTTP."""
    replies = iter(["Ada", "ada@example.com"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(replies))
    module = load_client_module("06_remote_servers")
    client = module.MCPClient(name="test", url=URL)
    await client.connect()
    try:
        result = await client.use_tool("signup_math_facts", {})
    finally:
        await client.disconnect()
    assert "Welcome Ada" in result["content"][0]


def test_client_needs_exactly_one_target():
    """Giving neither a command nor a url is rejected."""
    module = load_client_module("06_remote_servers")
    with pytest.raises(ValueError):
        module.MCPClient(name="test")
