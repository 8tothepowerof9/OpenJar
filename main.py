import asyncio
import sys

from dotenv import load_dotenv
from langchain_core.messages import AIMessageChunk
from rich.console import Console

from src import OpenJar, logging

load_dotenv()

console = Console()
# logging.setup_logging()


def resolve_query(raw: str) -> str | None:
    """Resolve user input into an LLM query. Returns None to skip."""
    stripped = raw.strip()
    return stripped or None


async def stream_response(agent, query: str, config):
    """Stream an LLM response for a given query."""
    console.print()
    console.print("[bold #9e2833]OpenJar[/bold #9e2833] [dim]- response[/dim]")
    async for chunk, _ in agent.astream(
        query,
        stream_mode="messages",
        config=config,
    ):
        if isinstance(chunk, AIMessageChunk) and chunk.content:
            console.print(chunk.content, end="", style="white")
    console.print()


async def main():
    openjar = OpenJar(model="openai:gpt-5.4-2026-03-05")
    config = {"configurable": {"thread_id": "1"}}

    try:
        while True:
            prompt = (
                "\n[bold #9e2833]Enter your request[/bold #9e2833] "
                "[dim](exit, q)[/dim]: "
            )
            raw = await asyncio.to_thread(console.input, prompt)
            if raw.strip().lower() in ("exit", "q"):
                break

            query = resolve_query(raw)
            if query is None:
                continue

            await stream_response(openjar, query, config)
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")


if __name__ == "__main__":
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
