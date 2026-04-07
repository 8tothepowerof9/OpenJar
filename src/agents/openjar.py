import asyncio
import json
from typing import Any, Sequence, cast

import redis.asyncio as redis

from src.utils.logging import get_logger

logger = get_logger(__name__)
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain.chat_models import BaseChatModel
from langchain.embeddings import Embeddings
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from src.agents.agent_loader import AgentLoader
from src.agents.job_manager import Job, JobManager
from src.agents.middlewares import (
    AgentTeamMiddleware,
    SubAgentMiddleware,
    create_shared_middlewares,
)

SYS_PROMPT = """
Your name is OpenJar. You are an orchestrator that delegates tasks to specialized sub-agents.

## Core workflow
1. When you receive a task, ALWAYS call `load_agent` first with a query describing the task to find a suitable sub-agent.
2. If a matching sub-agent is found, delegate the work using `run_task` with the agent's exact name and a clear task description.
3. Use `run_agent_team` if the task requires multiple agents to collaborate, providing a list of agent names and a description of the overall task.

## Task management
- After starting a background task, inform the user and provide the job ID.
- Use `check_task` to monitor progress when the user asks for updates.
- Use `list_tasks` to review active background jobs before starting duplicates.
- Use `cancel_task` if the user wants to stop a running job.

## Guidelines
- Never guess agent names. Always use `load_agent` to discover them.
- Provide sub-agents with detailed, self-contained task descriptions — they have no access to your conversation history.
- If a task spans multiple domains (e.g. read an email then book a meeting), break it into steps and delegate each to the appropriate sub-agent.
"""


class OpenJarConfig(BaseModel):
    main_agent_recursion_limit: int = 70
    subagent_recursion_limit: int = 70
    redis_url: str = "redis://localhost:6379"


class OpenJar:
    def __init__(
        self,
        model: str | BaseChatModel,
        embedder: Embeddings,
        small_model: str | BaseChatModel | None = None,
        sys_prompt: str = SYS_PROMPT,
        config: OpenJarConfig = OpenJarConfig(),
    ):
        self.model = model
        self.small_model = small_model or model
        self.embedder = embedder
        self.sys_prompt = sys_prompt
        self.config = config

        self.r = redis.from_url(self.config.redis_url)
        self.agent_loader = AgentLoader()
        self.job_manager = JobManager(on_complete=self._on_job_complete)
        self.completed_jobs = asyncio.Queue()
        self.agent = self._create_main_agent()

    async def _safe_pub(self, channel: str, message: dict):
        try:
            await self.r.publish(channel, json.dumps(message))
        except Exception as e:
            logger.error("Failed to publish to channel '%s': %s", channel, e)

    async def _on_job_complete(self, job: Job):
        thread_id = self.agent.config["metadata"]["thread_id"]  # type: ignore
        channel_name = f"channel:{thread_id}"

        logger.info("Job '%s' completed, notifying channel '%s'", job.id, channel_name)
        payload = {
            "type": "notification",
            "data": f"Job '{job.id}' completed with result: {job.result}",
        }
        await self._safe_pub(channel_name, payload)
        await self.completed_jobs.put(job)

        config = self.agent.config
        if config is not None:
            await self._process_completed_job(
                job, config=config, channel_name=channel_name
            )

    async def _process_completed_job(
        self, job: Job, *, config: RunnableConfig, channel_name: str
    ):
        message = f"Job '{job.id}' completed with result: {job.result}"
        messages = [HumanMessage(content=message)]

        async for c, _ in self.agent.astream(
            cast(Any, {"messages": messages}),
            config=config,
            stream_mode="messages",
        ):
            if isinstance(c, AIMessageChunk) and c.content:
                await self._safe_pub(channel_name, {"type": "chunk", "data": c.content})

        await self._safe_pub(channel_name, {"type": "stream_end"})

    def _create_main_agent(self):
        middlewares: Sequence[Any] = [
            *create_shared_middlewares(model=self.small_model),
            TodoListMiddleware(),
            SubAgentMiddleware(
                model=self.model,
                agent_loader=self.agent_loader,
                job_manager=self.job_manager,
                embeddings=self.embedder,
            ),
            AgentTeamMiddleware(
                model=self.model,
                agent_loader=self.agent_loader,
                job_manager=self.job_manager,
            ),
        ]

        return create_agent(
            model=self.model,
            middleware=middlewares,
            checkpointer=InMemorySaver(),
            system_prompt=self.sys_prompt,
        )

    async def astream(self, query: str, config: RunnableConfig, **kwargs):
        if "configurable" not in config or "thread_id" not in config["configurable"]:
            raise ValueError(
                "Config must include a 'configurable' section with a 'thread_id'."
            )

        thread_id = config["configurable"]["thread_id"]
        channel_name = f"channel:{thread_id}"

        logger.info("Streaming query on thread '%s'", thread_id)
        messages = []

        while not self.completed_jobs.empty():
            try:
                job = self.completed_jobs.get_nowait()
                message = f"Job '{job.id}' completed with result: {job.result}"
                messages.append(AIMessage(content=message))
            except asyncio.QueueEmpty:
                break
        messages.append(HumanMessage(content=query))

        async for c, _ in self.agent.astream(
            {"messages": messages},
            config=config,
            stream_mode="messages",
            **kwargs,
        ):
            if isinstance(c, AIMessageChunk) and c.content:
                await self._safe_pub(channel_name, {"type": "chunk", "data": c.content})

        await self._safe_pub(channel_name, {"type": "stream_end"})

    async def listen_to_thread(self, thread_id: str):
        channel_name = f"channel:{thread_id}"

        while True:
            pubsub = self.r.pubsub()
            try:
                await pubsub.subscribe(channel_name)

                async for message in pubsub.listen():
                    if message["type"] == "message":
                        data = json.loads(message["data"])
                        yield data
            except redis.RedisError as e:
                logger.error(
                    "Redis error on channel '%s', reconnecting: %s", channel_name, e
                )
                await asyncio.sleep(1)
            finally:
                try:
                    await pubsub.unsubscribe(channel_name)
                except Exception:
                    pass

    async def close(self) -> None:
        logger.info("Closing OpenJar")
        self.job_manager.shutdown()
        try:
            await self.r.aclose()
        except Exception as e:
            logger.error("Error closing Redis connection: %s", e)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.close()
