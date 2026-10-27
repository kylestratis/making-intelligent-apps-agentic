"""Shared pytest helpers for the workshop exercise tests.

The tests import each step's ``client.py`` by file path so one test run can
target either the ``starter/`` or the ``solution/`` folder. Pick the folder
with the ``WORKSHOP_VARIANT`` environment variable (default: ``solution``)::

    uv run pytest                              # check the solutions
    WORKSHOP_VARIANT=starter uv run pytest     # check your own work
"""

import importlib.util
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).parent.parent
SERVER = ROOT / "calculator_server.py"
VARIANT = os.environ.get("WORKSHOP_VARIANT", "solution")


def load_client_module(step: str) -> ModuleType:
    """Import a step's ``client.py`` under a unique module name.

    Args:
        step: Step directory name, such as ``"02_supporting_tools"``.

    Returns:
        The loaded module. Exercise steps load from the folder named by
        ``WORKSHOP_VARIANT``; flat steps load from the step directory.
    """
    path = ROOT / step / VARIANT / "client.py"
    if not path.exists():
        path = ROOT / step / "client.py"
    spec = importlib.util.spec_from_file_location(f"{step}_client", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@asynccontextmanager
async def connected_client(step: str) -> AsyncIterator:
    """Connect a step's ``MCPClient`` to ``calculator_server.py`` over stdio.

    Args:
        step: Step directory name, such as ``"02_supporting_tools"``.

    Yields:
        A connected ``MCPClient``, disconnected on exit.
    """
    module = load_client_module(step)
    client = module.MCPClient(
        name="test", command=sys.executable, server_args=[str(SERVER)]
    )
    await client.connect()
    try:
        yield client
    finally:
        await client.disconnect()
