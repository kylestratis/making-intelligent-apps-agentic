"""Check that your machine is ready for the workshop.

Run with ``uv run python check_setup.py``. Each check prints a PASS or FAIL
line; the script exits with status 1 if any check fails. The API key itself is
never printed.
"""

import asyncio
import importlib
import os
import shutil
import sys
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).parent
REQUIRED_PYTHON = (3, 12)


def report(ok: bool, label: str, hint: str = "") -> bool:
    """Print one PASS/FAIL line.

    Args:
        ok: Whether the check passed.
        label: Short description of the check.
        hint: What to do when the check fails.

    Returns:
        The value of ``ok``, so callers can collect results.
    """
    print(f"[{'PASS' if ok else 'FAIL'}] {label}")
    if not ok and hint:
        print(f"       -> {hint}")
    return ok


def check_python() -> bool:
    """Check the interpreter is Python 3.12 or newer."""
    version = ".".join(map(str, sys.version_info[:3]))
    return report(
        sys.version_info >= REQUIRED_PYTHON,
        f"Python {version} (need 3.12+)",
        "Install Python 3.12+ (uv can do it: `uv python install 3.12`).",
    )


def check_uv() -> bool:
    """Check that the ``uv`` executable is on the PATH."""
    return report(
        shutil.which("uv") is not None,
        "uv is installed",
        "Install uv: https://docs.astral.sh/uv/getting-started/installation/",
    )


def check_dependencies() -> bool:
    """Check that the workshop's Python packages can be imported."""
    results = []
    for module in ("anthropic", "mcp", "dotenv", "pytest"):
        try:
            importlib.import_module(module)
            results.append(report(True, f"import {module}"))
        except ImportError:
            results.append(report(False, f"import {module}", "Run `uv sync`."))
    return all(results)


def check_api_key() -> bool:
    """Check that ``ANTHROPIC_API_KEY`` is set in ``.env`` (value is never shown)."""
    key = dotenv_values(ROOT / ".env").get("ANTHROPIC_API_KEY") or os.environ.get(
        "ANTHROPIC_API_KEY"
    )
    has_key = bool(key) and key != "your_api_key_here"
    return report(
        has_key,
        "ANTHROPIC_API_KEY is set",
        "Run `cp .env.example .env`, then put your key in .env.",
    )


async def list_server_tools() -> list[str]:
    """Launch ``calculator_server.py`` over stdio and list its tools.

    Returns:
        The names of the tools the server exposes.
    """
    from mcp import Client
    from mcp.client.stdio import StdioServerParameters

    server = StdioServerParameters(
        command=sys.executable, args=[str(ROOT / "calculator_server.py")]
    )
    async with Client(server) as client:
        return [tool.name for tool in (await client.list_tools()).tools]


def check_server() -> bool:
    """Check that the example server starts and lists tools."""
    try:
        tools = asyncio.run(asyncio.wait_for(list_server_tools(), timeout=30))
    except Exception as error:  # noqa: BLE001 - report any startup failure
        return report(
            False, f"calculator server launches ({error!r})", "Run `uv sync`."
        )
    return report(
        bool(tools), f"calculator server launches and lists {len(tools)} tools"
    )


def main() -> None:
    """Run every check and exit non-zero if any failed."""
    results = [
        check_python(),
        check_uv(),
        check_dependencies(),
        check_api_key(),
        check_server(),
    ]
    print(
        "\nAll checks passed. You are ready!"
        if all(results)
        else "\nSome checks failed."
    )
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
