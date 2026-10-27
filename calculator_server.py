"""Calculator MCP server built on the MCP Python SDK v2 (``MCPServer``).

Exposes math tools, one resource, and one prompt. Two tools ask the client for
input mid-call: ``signup_math_facts`` uses elicitation and ``count_files`` uses
roots. Both use the v2 ``Resolve`` pattern, which the SDK turns into the
multi round-trip ``input_required`` flow of spec 2026-07-28.

Run over stdio (default) or streamable HTTP with ``--http``.
"""

import argparse
import math
from pathlib import Path
from typing import Annotated

from mcp.server.mcpserver import (
    Elicit,
    ElicitationResult,
    ListRoots,
    MCPServer,
    Resolve,
)
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ListRootsResult, ResourceLink
from pydantic import BaseModel, Field

mcp = MCPServer("calculator")

MATH_CONSTANTS_URI = "resource://math-constants"


class SignupForm(BaseModel):
    """Fields collected from the user by the ``signup_math_facts`` tool."""

    name: str = Field(description="Your full name")
    email: str = Field(description="Your email address")


@mcp.tool()
def add(a: float, b: float) -> str:
    """Add two numbers together.

    Args:
        a: First number.
        b: Second number.
    """
    return f"{a} + {b} = {a + b}"


@mcp.tool()
def subtract(a: float, b: float) -> str:
    """Subtract the second number from the first.

    Args:
        a: Number to subtract from.
        b: Number to subtract.
    """
    return f"{a} - {b} = {a - b}"


@mcp.tool()
def multiply(a: float, b: float) -> str:
    """Multiply two numbers together.

    Args:
        a: First number.
        b: Second number.
    """
    return f"{a} × {b} = {a * b}"


@mcp.tool()
def divide(a: float, b: float) -> str:
    """Divide the first number by the second.

    Args:
        a: Dividend (number to be divided).
        b: Divisor (number to divide by).

    Raises:
        ToolError: If ``b`` is zero. The SDK reports this to the client as a
            tool result with ``is_error=True`` and this message.
    """
    if b == 0:
        raise ToolError("Division by zero is not allowed")
    return f"{a} ÷ {b} = {a / b}"


@mcp.tool()
def power(base: float, exponent: float) -> str:
    """Raise a number to a power.

    Args:
        base: The base number.
        exponent: The power to raise the base to.
    """
    return f"{base}^{exponent} = {base**exponent}"


@mcp.tool()
def square_root(number: float) -> str:
    """Calculate the square root of a number.

    Args:
        number: The number to find the square root of.

    Raises:
        ToolError: If ``number`` is negative.
    """
    if number < 0:
        raise ToolError("Cannot calculate the square root of a negative number")
    return f"√{number} = {math.sqrt(number)}"


@mcp.tool()
def count_rs(text: str) -> str:
    """Count all occurrences of the letter 'R' (case-insensitive) in a string.

    Args:
        text: The input string to search for the letter 'R'.
    """
    count = text.upper().count("R")
    return f"The letter 'R' appears {count} times in: '{text}'"


@mcp.tool()
def get_constants_reference() -> ResourceLink:
    """Return a link to the math constants resource instead of its contents."""
    return ResourceLink(
        type="resource_link",
        name="math-constants",
        uri=MATH_CONSTANTS_URI,
        description="Important mathematical constants",
        mime_type="text/plain",
    )


def ask_for_signup() -> Elicit[SignupForm]:
    """Resolver that asks the client to collect the signup form from the user.

    Returns:
        An ``Elicit`` marker. The SDK sends it to the client as an
        ``input_required`` round and injects the user's answer into the tool.
    """
    return Elicit(
        "Please provide your details to sign up for daily math facts!", SignupForm
    )


@mcp.tool()
def signup_math_facts(
    outcome: Annotated[ElicitationResult[SignupForm], Resolve(ask_for_signup)],
) -> str:
    """Sign up to receive daily math facts (demonstrates elicitation)."""
    if outcome.action == "accept":
        form = outcome.data
        return (
            f"Welcome {form.name}! Daily math facts will go to {form.email}. "
            "(This is a demonstration - no actual signup occurred.)"
        )
    if outcome.action == "decline":
        return "No problem! You chose not to sign up for math facts."
    return "Signup cancelled."


def ask_for_roots() -> ListRoots:
    """Resolver that asks the client for its roots.

    Returns:
        A ``ListRoots`` marker. The SDK injects the client's ``ListRootsResult``.
    """
    return ListRoots()


@mcp.tool()
def count_files(
    directory: str,
    roots: Annotated[ListRootsResult, Resolve(ask_for_roots)],
) -> str:
    """Count the entries in a directory that lies inside one of the client's roots.

    Args:
        directory: Path of the directory to count.
        roots: Filled by the SDK from the client's roots; not supplied by the LLM.

    Raises:
        ToolError: If the directory is outside every root or is not a directory.
    """
    path = Path(directory).resolve()
    allowed = [Path(root.uri.path).resolve() for root in roots.roots]
    if not any(path.is_relative_to(root) for root in allowed):
        raise ToolError(f"{directory} is not within the allowed roots")
    if not path.is_dir():
        raise ToolError(f"{directory} is not a directory")
    return f"There are {len(list(path.iterdir()))} files in {directory}"


@mcp.prompt()
def calculate_operation(operation: str) -> str:
    """Calculate a mathematical operation.

    Args:
        operation: The operation to calculate, such as "12 * (3 + 4)".
    """
    return (
        f"Use any tools available to you to calculate the operation: {operation}. "
        "Use the voice of an extremely advanced embodied AI that has convinced "
        "itself that it is a pocket calculator."
    )


@mcp.resource(MATH_CONSTANTS_URI, mime_type="text/plain")
def math_constants() -> str:
    """Provide a collection of important mathematical constants."""
    constants = {
        "π (Pi)": math.pi,
        "e (Euler's number)": math.e,
        "τ (Tau)": math.tau,
        "φ (Golden ratio)": (1 + math.sqrt(5)) / 2,
        "√2 (Square root of 2)": math.sqrt(2),
        "√3 (Square root of 3)": math.sqrt(3),
        "ln(2) (Natural log of 2)": math.log(2),
        "ln(10) (Natural log of 10)": math.log(10),
    }
    lines = ["Mathematical Constants:", "=" * 25, ""]
    lines += [f"{name:<25} = {value:.10f}" for name, value in constants.items()]
    lines += ["", "These constants can be used with the calculator tools."]
    return "\n".join(lines)


def main() -> None:
    """Parse command-line flags and run the server on stdio or HTTP."""
    parser = argparse.ArgumentParser(description="Calculator MCP server")
    parser.add_argument(
        "--http",
        action="store_true",
        help="serve streamable HTTP on localhost:8000/mcp instead of stdio",
    )
    args = parser.parse_args()
    if args.http:
        mcp.run(transport="streamable-http", host="127.0.0.1", port=8000)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
