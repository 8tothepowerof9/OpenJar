import asyncio
import sys
import uuid

from dotenv import load_dotenv
from rich.console import Console

from src import OpenJar, logging

load_dotenv()

console = Console()
logging.setup_logging()


def resolve_query(raw: str) -> str | None:
    """Resolve user input into an LLM query. Returns None to skip."""
    stripped = raw.strip()
    return stripped or None


async def listener(openjar: OpenJar, thread_id: str):
    try:
        async for message in openjar.listen_to_thread(thread_id):
            msg_type = message.get("type")
            data = message.get("data", "")

            if msg_type == "chunk":
                console.print(data, end="", style="white")

            if msg_type == "notification":
                console.print()
                console.print(f"[dim]{data}[/dim]")
                console.print(
                    "\n[bold #9e2833]Enter your request[/bold #9e2833] [dim](exit, q)[/dim]: ",
                    end="",
                )

            if msg_type == "stream_end":
                console.print()

    except asyncio.CancelledError:
        pass
    except Exception as e:
        console.print()
        console.print(f"[#9e2833]Listener Error:[/#9e2833] {str(e)}")


async def main():
    async with OpenJar(model="groq:openai/gpt-oss-120b") as openjar:
        thread_id = uuid.uuid4().hex[:8]
        config = {"configurable": {"thread_id": thread_id}}

        listener_task = asyncio.create_task(listener(openjar, thread_id))

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

                console.print(
                    "[bold #9e2833]OpenJar[/bold #9e2833] [dim]- response[/dim]"
                )

                await openjar.astream(query, config=config)  # type: ignore

        except Exception as e:
            console.print()
            console.print(f"[#9e2833]Main Error:[/#9e2833] {str(e)}")
        finally:
            listener_task.cancel()


if __name__ == "__main__":
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
