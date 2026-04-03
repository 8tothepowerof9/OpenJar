# OpenJar

OpenJar is a modular multi-agent orchestrator built on LangChain and LangGraph.
It discovers sub-agents from the `agents/` directory, exposes orchestration tools through middleware, and streams responses through Redis-backed channels.

## What It Does

- Loads sub-agents dynamically from `agents/*/info.yaml` and `agents/*/tools.py`
- Routes work to specialized agents with sync and async delegation
- Supports background task execution and completion notifications
- Streams model output incrementally to a CLI listener
- Keeps orchestration logic centralized in a lightweight main agent

## Requirements

- Python 3.11+
- `uv` (recommended dependency manager)
- Redis server running locally (default: `redis://localhost:6379`)
- Provider API key(s), for example OpenAI

## Installation

```bash
git clone <repo-url>
cd OpenJar
uv sync
```

## Environment Setup

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your-openai-api-key
GROQ_API_KEY=your-groq-api-key
```

Only define keys for providers you actually use.

## Run

Start Redis first, then run the app:

```bash
uv run .\main.py
```

Notes:

- On non-Windows shells, use `uv run ./main.py` or `uv run python main.py`.
- Type `exit` or `q` in the CLI to quit.

## Architecture Overview

1. `OpenJar` in `src/agent/openjar.py` creates the main orchestrator agent.
2. `AgentLoader` in `src/agent/loader.py` discovers and loads sub-agents.
3. Middleware in `src/agent/middlewares/` injects orchestration capabilities.
4. `JobManager` in `src/agent/job_manager.py` handles background task lifecycle.
5. Redis pub/sub is used to stream chunks and async notifications per thread.

## Core Files

```text
main.py
src/agent/openjar.py
src/agent/loader.py
src/agent/job_manager.py
src/agent/middlewares/
agents/file_explorer/
agents/research/
```

## Add a New Sub-Agent

Create a new folder:

```text
agents/my_agent/
  info.yaml
  tools.py
```

Example `info.yaml`:

```yaml
name: my_agent
description: A short description of what this agent does.
instruction: >
  Detailed instructions for the agent behavior.
model: gpt-5.4-mini-2026-03-17
provider:
  type: openai
```

Example `tools.py`:

```python
from langchain.tools import tool


@tool
async def my_tool(query: str) -> str:
    """Describe what this tool does."""
    return f"Handled: {query}"
```

The orchestrator will discover the new sub-agent on next run.

## Programmatic Usage

`OpenJar` currently exposes streaming methods (`astream`, `listen_to_thread`).
Minimal example:

```python
import asyncio

from src import OpenJar


async def run() -> None:
    app = OpenJar(model="openai:gpt-5.4-2026-03-05")
    thread_id = "demo-thread"
    config = {"configurable": {"thread_id": thread_id}}

    listener_task = asyncio.create_task(_listen(app, thread_id))
    await app.astream("Summarize this repository architecture.", config=config)

    listener_task.cancel()
    await app.redis_client.aclose()


async def _listen(app: OpenJar, thread_id: str) -> None:
    async for event in app.listen_to_thread(thread_id):
        print(event)


if __name__ == "__main__":
    asyncio.run(run())
```

## Troubleshooting

- No output in CLI:
  - confirm Redis is running
  - confirm API keys are present in `.env`
- Sub-agent not found:
  - verify `agents/<name>/info.yaml` and `agents/<name>/tools.py` both exist
  - verify YAML fields are valid

## Roadmap / Planned Features

- Long-term and mid-term memory layers
- Prompt-injection defense and sensitive-data redaction
- Centralized environment and secret management
- Trigger and schedule orchestration
- Speech-to-text (STT) and text-to-speech (TTS)
- Tool runtime sandboxing
- Least-privilege data access for browser and online sources
- Tool code safety validation
- Human-in-the-loop approvals and checkpoints
- Agent steering and policy controls
- Persistent memory update workflows
- Browser automation and control
- Multi-channel chat interfaces
- Enhanced CLI application experience
- Multi-modal input support
- Cost tracking, tracing, and observability
- Multi-session chat threading
- Reusable skill system
- Infinite-loop prevention safeguards
- Document and file understanding pipeline
- Sandboxed code execution
- Agent teams / Swarms / Different patterns of multiagents
- Restrict passing full context between agents
- Customize main agent's soul/personality
- Add workflow to start everytime agent is woken up, or start of new day (or detect user start to engage)
