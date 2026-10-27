# Making Apps Intelligent: Building MCP Clients

A hands-on workshop that teaches you how to build Model Context Protocol (MCP) clients by progressively enhancing a simple chat application. It uses the **MCP Python SDK v2** (`mcp` 2.3.0, protocol spec 2026-07-28) and Claude (`claude-sonnet-5-5`).

## What You'll Learn

You will turn a basic Claude chat agent into an MCP client that can:
- Connect to MCP servers over stdio and over HTTP
- Discover and call server-provided tools, and handle tool errors
- Load resources and follow resource links
- Run server prompt templates as part of the conversation
- Answer server requests for user input (elicitation) and share roots

## Setup

You need Python 3.12+, an Anthropic API key ([get one here](https://console.anthropic.com/settings/keys)), and basic familiarity with Python and async/await.

```bash
# 1. Install uv (https://docs.astral.sh/uv/)
curl -LsSf https://astral.sh/uv/install.sh | sh      # macOS/Linux
# powershell -c "irm https://astral.sh/uv/install.ps1 | iex"   # Windows

# 2. Clone and install dependencies
git clone <repository-url>
cd making-intelligent-apps-agentic
uv sync

# 3. Add your API key
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY=...

# 4. Verify everything works
uv run python check_setup.py
```

`check_setup.py` prints a PASS or FAIL line for the Python version, uv, the dependencies, your API key (presence only, never the value), and whether the example server starts.

**No local setup?** Open the repo in GitHub Codespaces or a VS Code dev container. `.devcontainer/devcontainer.json` installs Python 3.12 and uv and runs `uv sync`. You still need to put your key in `.env`.

## Repository Structure

```
making-intelligent-apps-agentic/
├── 00_base_application/       # Simple chat agent (baseline)
├── 01_create_client_module/   # MCP client connect/disconnect (live-coded)
├── 02_supporting_tools/       # EXERCISE: tools, errors, resource links
│   ├── starter/               #   code with TODOs
│   └── solution/              #   finished code
├── 03_supporting_resources/   # EXERCISE: resources and following links
├── 04_supporting_prompts/     # EXERCISE: prompts as conversation messages
├── 05_elicitation/            # EXERCISE: elicitation callback (+ roots)
├── 06_remote_servers/         # Connect to a server over HTTP
├── tests/                     # One test file per exercise (no API key needed)
├── calculator_server.py       # Example MCP server
├── check_setup.py             # Setup checker
├── .devcontainer/             # Codespaces / dev container fallback
└── pyproject.toml             # Pinned dependencies
```

## The Example Server

`calculator_server.py` is built with `MCPServer` from the v2 SDK.

**Tools**
| Tool | What it does |
| --- | --- |
| `add`, `subtract`, `multiply` | Basic arithmetic |
| `divide` | Division. Dividing by zero returns an error result (`is_error=True`) |
| `power`, `square_root` | Exponent and root (negative roots are errors) |
| `count_rs` | Count the letter "R" in a string |
| `get_constants_reference` | Returns a **resource link** to `resource://math-constants` |
| `signup_math_facts` | Asks you for your name and email (**elicitation**) |
| `count_files` | Counts entries in a directory inside the client's **roots** |

**Resource:** `resource://math-constants` (pi, e, the golden ratio, and more).
**Prompt:** `calculate_operation` (takes an `operation` argument).

By default the client starts the server for you over stdio. To run it as an HTTP server instead:

```bash
uv run python calculator_server.py --http     # serves http://localhost:8000/mcp
```

## How the Exercises Work

Steps 02 to 05 are exercises. Each has two folders:

- `starter/`: working code with `TODO` stubs. The docstring of each stub says what to implement. The starter already contains the finished code of the previous step, so every exercise starts from a known-good base.
- `solution/`: the finished code for that step.

**Workflow**
1. Open `NN_step/starter/client.py` and find the `TODO` markers.
2. Implement them. Run your agent: `uv run python NN_step/starter/agent.py`.
3. Check your work with the tests (below).
4. Stuck, or behind the group? Copy the solution over your file, for example `cp 02_supporting_tools/solution/client.py 02_supporting_tools/starter/client.py`. To catch up to step N, start step N from its own `starter/` (it includes the previous step's solution), or copy the previous step's `solution/` files into the current step's `starter/`.

### Tests

The tests need no API key. They call `MCPClient` methods directly against `calculator_server.py`.

```bash
uv run pytest                                  # run against the solutions (all pass)
WORKSHOP_VARIANT=starter uv run pytest         # run against your starter code
WORKSHOP_VARIANT=starter uv run pytest tests/test_02_tools.py   # one step
```

Unfinished TODOs fail with `NotImplementedError` or a clear assertion message.

---

## Workshop Steps

### Step 0: Base Application
A plain chat loop with conversation history and no tools.
```bash
uv run python 00_base_application/agent.py
```

### Step 1: Create Client Module
The presenter live-codes `MCPClient`. It keeps a `connect()`/`disconnect()` design with an `AsyncExitStack`. `connect()` builds an SDK `Client` from `StdioServerParameters` and enters it on the exit stack. In v2 there is no `initialize()` call and no read/write streams.
```bash
uv run python 01_create_client_module/agent.py
```

### Step 2: Supporting Tools (exercise)
Add `get_available_tools()` and `use_tool()` and the agentic loop. `use_tool` returns `{"content": [...], "is_error": bool}`.
- **TODO (a):** pass the server's `is_error` through, so Claude knows a call failed.
- **TODO (b):** handle `resource_link` results by returning the link's name and URI as text.

```bash
uv run python 02_supporting_tools/starter/agent.py
```
Try: `What is 25 times 17?`, `What is 1 divided by 0?`, `Give me the math constants reference.`

### Step 3: Supporting Resources (exercise)
- **TODO:** `get_resource(uri)` wraps `read_resource` and returns text/blob contents.
- **TODO:** make `use_tool` follow `resource_link` results by loading them through `get_resource`.

The agent still loads all resources as context on every turn.
```bash
uv run python 03_supporting_resources/starter/agent.py
```
Try: `What is the golden ratio?`, `Give me the math constants reference.`

### Step 4: Supporting Prompts (exercise)
- **TODO:** `load_prompt(name, arguments)` wraps `get_prompt` and turns each `PromptMessage` into an Anthropic message dict.

The agent gets two commands: `/prompts` lists templates, and `/prompt <name>` asks for each argument and adds the rendered messages to the conversation (with their roles, not as a system prompt).
```bash
uv run python 04_supporting_prompts/starter/agent.py
```
Try: `/prompts`, then `/prompt calculate_operation` and enter `(3 + 4) * 12`.

### Step 5: Elicitation (exercise)
Servers can ask the user for input in the middle of a tool call. With protocol 2026-07-28 the server replies `input_required`, and the SDK `Client` calls your callback and retries the call for you.
- **TODO:** `handle_elicitation(context, params)` prints the server's message, prompts for each requested field, and returns `ElicitResult(action="accept", content=...)`, or `"decline"` on empty input.

It is passed to `Client(..., elicitation_callback=...)`. A silent `list_roots_callback` is already registered, so `count_files` works.
```bash
uv run python 05_elicitation/starter/agent.py
```
Try: `Sign me up for math facts.` and `How many files are in <path to this repo>?`

### Step 6: Remote Servers
`MCPClient` accepts either `command`/`server_args` or a `url`. `connect()` branches to `Client(url)`. Nothing else changes.
```bash
# terminal 1
uv run python calculator_server.py --http
# terminal 2
uv run python 06_remote_servers/agent.py --url http://localhost:8000/mcp
```
Without `--url` it starts the server over stdio, like step 05.

---

## Understanding MCP Primitives

- **Tools**: actions Claude chooses to run. Failures come back with `is_error=True`.
- **Resources**: data the client chooses to load as context. Tools can also return links to them.
- **Prompts**: templates the user picks, rendered into conversation messages.
- **Elicitation and roots**: things the *client* provides to the server (user input and file access).

## Troubleshooting

- **"ANTHROPIC_API_KEY not set"**: create `.env` with `cp .env.example .env` and add your key. `uv run python check_setup.py` tells you what is missing.
- **Import errors**: run `uv sync --reinstall`.
- **Server won't connect**: run `uv run python check_setup.py`; it starts the server and lists its tools.
- **Port 8000 busy** (step 6 and `test_06_remote.py`): stop whatever is using it.

## Resources

- [MCP Python SDK docs (v2)](https://py.sdk.modelcontextprotocol.io)
- [Build an MCP client tutorial](https://modelcontextprotocol.io/docs/develop/build-client)
- [MCP specification](https://modelcontextprotocol.io/specification)
- [Anthropic API documentation](https://docs.anthropic.com)
