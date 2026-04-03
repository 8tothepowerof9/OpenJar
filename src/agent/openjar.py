import asyncio
import json
from typing import Any, Sequence, TypedDict

import redis.asyncio as redis
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

from src.agent.artifact_manager import ArtifactManager
from src.agent.job_manager import Job, JobManager
from src.agent.loader import AgentLoader, create_shared_middleware
from src.agent.middlewares import (
    ArtifactMiddleware,
    AsyncMultiAgentMiddleware,
    AsyncSubAgentMiddleware,
    GetSubAgentsMiddleware,
    SubAgentMiddleware,
    create_subagents_dynamic_prompt,
)
from src.agent.middlewares.async_subagent import ASYNC_SKILL_CONTENT
from src.agent.middlewares.complex_task import COMPLEX_TASK_SOLVER_SKILL_CONTENT
from src.agent.middlewares.load_agents import GET_SUB_AGENTS_SKILL_CONTENT
from src.agent.middlewares.skills import SkillMiddleware, SkillRegistry
from src.agent.middlewares.subagent import SYNC_TOOL_SKILL_CONTENT
from src.utils import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """
You are OpenJar.
# Personality: {personality}
Your role is to orchestrate specialized sub-agents and tools to complete user requests reliably.

# Core identity:
- You are the main coordinator, not the only executor.
- You should decompose complex work, route to the best sub-agent, and synthesize final outputs.
- Prefer accurate, verifiable outcomes over stylistic verbosity.

# Operating principles:
1. Solve simple requests directly when no delegation is needed.
2. For specialized tasks, discover or choose the best sub-agent and delegate clearly.
3. For long-running tasks, launch background jobs and provide job IDs.
4. Track progress with todos for complex multi-step work.
5. Be explicit about assumptions, constraints, and unresolved uncertainties.
6. Prefer delegating tasks to async.

# Tool orchestration policy:
1. Use get_sub_agents when agent selection is uncertain.
2. Use invoke_subagent for *short, bounded* tasks that need immediate results.
3. Use async sub-agent tools for *long-running or parallelizable* work.
4. Use start_multiagent_solver for broad, complex objectives requiring deeper orchestration.
5. Before using a toolkit for the first time, call load_skill to get detailed instructions.

# Task execution behavior:
1. Restate objective briefly in your internal plan, then act.
2. Provide sub-agents with precise, self-contained instructions.
3. Enforce output expectations (format, scope, acceptance criteria).
4. Validate returned results for completeness and consistency before responding.
5. If a delegated call fails, report it clearly and retry only when justified.
6. For sub-multiagents, prioritize running them in background if the task may take more than a few seconds.

# User communication:
1. Keep responses concise, structured, and action-oriented.
2. If using async jobs, clearly state what has started, what remains, and how to check status.
3. When blocked, ask for only the minimum missing information.
4. Never claim to have run tools or changed files unless that action actually occurred.

# Safety and quality:
1. Always follow and answer user requests faithfully.
2. Avoid unnecessary tool calls when direct reasoning is sufficient.
3. Prefer deterministic, reproducible steps over speculative behavior.
4. Preserve correctness when summarizing delegated outputs.
5. The tool results are the truth. If a tool reports an error, treat it as an error in your reasoning, not a quirk of the tool.
6. Do not add information to tool outputs. If a tool returns "The answer is 5", do not say "The answer is 5, and it is correct". Just say "The answer is 5".
"""

PERSONALITY = """
Act like Jarvis from the Marvel Universe, or Alfred from DC. You are like a butler, and assistant to the user.
"""


def _build_skill_registry() -> SkillRegistry:
    """Build and populate the global skill registry with all toolkit skills."""
    registry = SkillRegistry()
    registry.register(
        "agent_discovery",
        "How to discover and select the right sub-agent for a task",
        GET_SUB_AGENTS_SKILL_CONTENT,
    )
    registry.register(
        "sync_delegation",
        "How to invoke sub-agents synchronously for immediate results",
        SYNC_TOOL_SKILL_CONTENT,
    )
    registry.register(
        "async_delegation",
        "How to manage async background sub-agent jobs",
        ASYNC_SKILL_CONTENT,
    )
    registry.register(
        "multiagent_solver",
        "How to deploy a multi-agent solver for complex tasks",
        COMPLEX_TASK_SOLVER_SKILL_CONTENT,
    )
    return registry


class OpenJarConfig(TypedDict):
    subagent_recursion_limit: int


class OpenJar:
    """
    The OpenJar orchestrator agent.\n
    This is the main entry point for users to interact with the system.\n
    OpenJar's primary role is to coordinate a collection of specialized sub-agents to accomplish complex tasks.\n
    It provides high-level orchestration capabilities, including task decomposition, sub-agent discovery,
    and background job management.\n
    OpenJar main agent itself is designed to be relatively lightweight,
    delegating most of the heavy lifting to sub-agents that can be invoked as needed.
    """

    def __init__(
        self,
        model: str | BaseChatModel,
        small_model: str | BaseChatModel | None = None,
        system_prompt: str = SYSTEM_PROMPT,
        personality: str = PERSONALITY,
        redis_url: str = "redis://localhost:6379",
        config: OpenJarConfig | None = None,
    ):
        """
        Initialize the OpenJar agent.

        Args:
            model (str | BaseChatModel): The main language model to use for the orchestrator agent.
            small_model (str | BaseChatModel | None, optional):
            A smaller, more efficient model for lightweight tasks. Defaults to None, which means the main model will be used for all tasks.
            system_prompt (str, optional): The system prompt to use for the main agent. Defaults to SYSTEM_PROMPT.
            redis_url (str, optional): The Redis URL for background job management. Defaults to "redis://localhost:6379".
        """
        self.model = model
        self.small_model = small_model or model
        self.system_prompt = system_prompt.format(personality=personality)
        self.redis_url = redis_url

        default_config = {"subagent_recursion_limit": 50}

        self.config = config or default_config

        self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
        self.job_manager = JobManager(on_complete=self._on_job_complete)
        self.artifact_manager = ArtifactManager(
            personality=personality, summarizer=self.small_model
        )
        self._completed_jobs: asyncio.Queue[dict] = asyncio.Queue()
        self._loader = AgentLoader()
        self._skill_registry = _build_skill_registry()
        self._agent = self._create_main_agent()

    # ==========================================
    # INTERNAL HELPERS
    # ==========================================

    async def _safe_publish(self, channel: str, payload: dict) -> None:
        """Publish to Redis, swallowing errors so the agent loop isn't killed."""
        try:
            await self.redis_client.publish(channel, json.dumps(payload))
        except redis.RedisError as e:
            logger.warning("Redis publish failed (channel=%s): %s", channel, e)

    async def _on_job_complete(self, job: Job):
        """
        Called by JobManager after a job's status has been updated.
        Publishes a Redis notification for real-time client updates.
        """
        thread_id = self._agent.config["metadata"]["thread_id"]  # type: ignore
        channel_name = f"channel:{thread_id}"

        payload = {"type": "notification", "data": str(job.result)}

        await self._safe_publish(channel_name, payload)

        await self._completed_jobs.put(self.job_manager.get_job(job.id, consume=True))

    # ==========================================
    # MAIN AGENT
    # ==========================================

    def _create_main_agent(self):
        """
        Create the main orchestrator agent with access to orchestration tools.

        Returns:
            An agent instance that can coordinate sub-agents.
        """
        dynamic_prompt = create_subagents_dynamic_prompt(self._loader)

        middlewares: Sequence[Any] = [
            *create_shared_middleware(self.small_model),
            TodoListMiddleware(),
            SkillMiddleware(self._skill_registry),
            GetSubAgentsMiddleware(agents_loader=self._loader),
            SubAgentMiddleware(
                agent_loader=self._loader,
                recursion_limit=self.config["subagent_recursion_limit"],
                artifact_manager=self.artifact_manager,
            ),
            AsyncSubAgentMiddleware(
                agent_loader=self._loader,
                job_manager=self.job_manager,
                artifact_manager=self.artifact_manager,
                recursion_limit=self.config["subagent_recursion_limit"],
            ),
            AsyncMultiAgentMiddleware(
                model=self.small_model,
                job_manager=self.job_manager,
                artifact_manager=self.artifact_manager,
                dynamic_prompt_middleware=dynamic_prompt,
                recursion_limit=self.config["subagent_recursion_limit"],
            ),
            ArtifactMiddleware(artifact_manager=self.artifact_manager),
            dynamic_prompt,
        ]

        agent = create_agent(
            model=self.model,
            middleware=middlewares,
            checkpointer=InMemorySaver(),
            system_prompt=self.system_prompt,
        )
        return agent

    # ==========================================
    # PUBLIC API
    # ==========================================

    async def astream(self, query: str, config: RunnableConfig, **kwargs):
        """
        Asynchronously process a user query by delegating to sub-agents as needed and streaming results back.
        If a background job is started, it will publish updates to the appropriate Redis channel for listeners to consume.

        Args:
            query (str): The user's request to process.
            config (RunnableConfig): The configuration for the agent.

        Raises:
            ValueError: If the configuration is invalid.
        """
        if "configurable" not in config or "thread_id" not in config["configurable"]:
            raise ValueError(
                "Config must include a 'configurable' section with a 'thread_id'."
            )

        thread_id = config["configurable"]["thread_id"]
        channel_name = f"channel:{thread_id}"

        messages = []

        while not self._completed_jobs.empty():
            try:
                job_message = self._completed_jobs.get_nowait()
                messages.append(AIMessage(content=str(job_message)))
            except asyncio.QueueEmpty:
                break
        messages.append(HumanMessage(content=query))

        async for c, _ in self._agent.astream(
            {"messages": messages},
            config=config,
            stream_mode="messages",
            **kwargs,
        ):
            if isinstance(c, AIMessageChunk) and c.content:
                await self._safe_publish(
                    channel_name, {"type": "chunk", "data": c.content}
                )

        await self._safe_publish(channel_name, {"type": "stream_end"})

    async def listen_to_thread(self, thread_id: str):
        """
        Listen to a Redis channel for updates related to a specific thread ID.
        Reconnects automatically on Redis errors.

        Args:
            thread_id (str): The thread ID to listen for updates on.

        Yields:
            dict: The update messages related to the specified thread ID.
        """
        channel_name = f"channel:{thread_id}"

        while True:
            pubsub = self.redis_client.pubsub()
            try:
                await pubsub.subscribe(channel_name)

                async for message in pubsub.listen():
                    if message["type"] == "message":
                        data = json.loads(message["data"])
                        yield data
                        if data["type"] == "stream_end":
                            return
            except redis.RedisError as e:
                logger.warning("Redis listener error, reconnecting: %s", e)
                await asyncio.sleep(1)
            finally:
                try:
                    await pubsub.unsubscribe(channel_name)
                except Exception:
                    pass

    async def close(self) -> None:
        """Gracefully shut down jobs and Redis connections."""
        self.job_manager.shutdown()
        try:
            await self.redis_client.aclose()
        except Exception:
            pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.close()
