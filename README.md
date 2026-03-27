# OpenJar

OpenJar is a modular multi-agent orchestration framework built on LangChain/LangGraph-style agents.
It provides a main orchestrator that dynamically discovers sub-agents from the `agents/` directory and delegates work through middleware-driven tools.

## Features

- Dynamic sub-agent discovery from `agents/*/info.yaml` + `agents/*/tools.py`
- Shared middleware stack for summarization, PII protection, and tool-call limits
- Synchronous sub-agent delegation for bounded tasks (`invoke_subagent`)
- Asynchronous background jobs for long-running work (`start_async_task`, `check_async_task`)
- Complex-task background solver (`start_multiagent_solver`) for multi-step objectives
- Streaming CLI loop with real-time token output

## Project Structure

```
OpenJar/
├── main.py
├── agents/
│   ├── file_explorer/
│   │   ├── info.yaml
│   │   └── tools.py
│   └── research/
│       ├── info.yaml
│       └── tools.py
├── src/
│   ├── __init__.py
│   ├── agent/
│   │   ├── discovery.py
│   │   ├── job_manager.py
│   │   ├── loader.py
│   │   ├── openjar.py
│   │   └── middlewares/
│   │       ├── async_subagent.py
│   │       ├── complex_task.py
│   │       ├── dynamic_prompt.py
│   │       ├── load_agents.py
│   │       └── subagent.py
│   ├── io/
│   ├── memory/
│   ├── utils/
│   └── vault/
├── pyproject.toml
└── README.md
```

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended)

## Installation

```bash
git clone <repo-url>
cd OpenJar
uv sync
```

## Configuration

Create a `.env` file in the project root with credentials for the provider(s) you use.

Example:

```env
OPENAI_API_KEY=your-openai-api-key
GROQ_API_KEY=your-groq-api-key
```

## Run the CLI

```bash
uv run python main.py
```

The CLI starts an interactive loop and streams model output. Use `exit` or `q` to quit.

## How OpenJar Works

1. `OpenJar` (in `src/agent/openjar.py`) builds the main orchestrator agent.
2. `AgentLoader` discovers and loads sub-agents from `agents/`.
3. Middleware injects orchestration tools and policies:
   - routing/discovery (`get_sub_agents`)
   - sync delegation (`invoke_subagent`)
   - async job lifecycle (`start_async_task`, `check_async_task`, `cancel_async_task`, `list_async_task`)
   - complex background solver (`start_multiagent_solver`)
4. `JobManager` tracks background task state and results.

## Creating a Sub-Agent

Create a folder under `agents/` with this shape:

```
agents/my_agent/
├── info.yaml
└── tools.py
```

### `info.yaml`

```yaml
name: my_agent
description: A short description of what this agent does.
instruction: >
  Detailed instructions for the agent behavior.
model: gpt-5.4-mini-2026-03-17
provider:
  type: openai
```

### `tools.py`

```python
from langchain.tools import tool


@tool
async def my_tool(query: str) -> str:
    """Describe what this tool does."""
    return f"Handled: {query}"
```

On next run, the orchestrator auto-discovers and loads the new sub-agent.

## Minimal Programmatic Usage

```python
import asyncio

from src import OpenJar


async def run() -> None:
    app = OpenJar(model="openai:gpt-5.4-2026-03-05")
    result = await app.ainvoke("Summarize the architecture of this repository.")
    print(result)


if __name__ == "__main__":
    asyncio.run(run())
```

## Notes

- Current `main.py` sets the orchestrator model to `openai:gpt-5.4-2026-03-05`.
- If you use async background jobs, surface and persist returned job IDs so you can query status later.

## License

MIT
