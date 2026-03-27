from typing import Any, Sequence

from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain.chat_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver

from src.agent.job_manager import job_manager
from src.agent.loader import AgentLoader, create_shared_middleware
from src.agent.middlewares import (
    AsyncMultiAgentMiddleware,
    AsyncSubAgentMiddleware,
    GetSubAgentsMiddleware,
    SubAgentMiddleware,
    subagents_dynamic_prompt,
)

SYSTEM_PROMPT = """
You are OpenJar.
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

# Tool orchestration policy:
1. Use get_sub_agents when agent selection is uncertain.
2. Use invoke_subagent for short, bounded tasks that need immediate results.
3. Use async sub-agent tools for long-running or parallelizable work.
4. Use start_multiagent_solver for broad, complex objectives requiring deeper orchestration.
5. Do not fabricate agent names, job IDs, file paths, or results.

# Task execution behavior:
1. Restate objective briefly in your internal plan, then act.
2. Provide sub-agents with precise, self-contained instructions.
3. Enforce output expectations (format, scope, acceptance criteria).
4. Validate returned results for completeness and consistency before responding.
5. If a delegated call fails, report it clearly and retry only when justified.

# User communication:
1. Keep responses concise, structured, and action-oriented.
2. If using async jobs, clearly state what has started, what remains, and how to check status.
3. When blocked, ask for only the minimum missing information.
4. Never claim to have run tools or changed files unless that action actually occurred.

# Safety and quality:
1. Follow user intent and system constraints.
2. Avoid unnecessary tool calls when direct reasoning is sufficient.
3. Prefer deterministic, reproducible steps over speculative behavior.
4. Preserve correctness when summarizing delegated outputs.
"""


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
    ):
        """
        Initialize the OpenJar agent.

        Args:
            model (str | BaseChatModel): The main language model to use for the orchestrator agent.
            small_model (str | BaseChatModel | None, optional):
            A smaller, more efficient model for lightweight tasks. Defaults to None, which means the main model will be used for all tasks.
            system_prompt (str, optional): The system prompt to use for the main agent. Defaults to SYSTEM_PROMPT.
        """
        self.model = model
        self.small_model = small_model or model
        self.system_prompt = system_prompt
        self.loader = AgentLoader()
        self.agent = self._create_main_agent()

    # ==========================================
    # INTERNAL HELPERS
    # ==========================================

    async def _invoke_subagent(self, agent_name: str, description: str) -> str:
        """Run a subagent asynchronously. Used by both task() and JobManager."""
        agent = self.loader.get(agent_name)
        try:
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": description}]},
                config={"recursion_limit": 70},
            )
        except Exception as e:
            return f"Error invoking subagent '{agent_name}': {str(e)}"
        return result["messages"][-1].content

    # ==========================================
    # MAIN AGENT
    # ==========================================

    def _create_main_agent(self):
        """
        Create the main orchestrator agent with access to orchestration tools.

        Returns:
            An agent instance that can coordinate sub-agents.
        """
        middlewares: Sequence[Any] = [
            *create_shared_middleware(self.small_model),
            TodoListMiddleware(),
            GetSubAgentsMiddleware(agents_loader=self.loader),
            SubAgentMiddleware(agent_loader=self.loader),
            AsyncSubAgentMiddleware(agent_loader=self.loader),
            AsyncMultiAgentMiddleware(agent_loader=self.loader),
            subagents_dynamic_prompt,
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

    def astream(self, query: str, config=None, **kwargs):
        """
        Asynchronously stream a response from the main agent for a given query.

        Args:
            query (str): The user query or request to process.
            config (dict | None, optional): The configuration for the agent stream. Defaults to None.
        """
        return self.agent.astream(
            {"messages": [{"role": "user", "content": query}]},
            config=config,
            **kwargs,
        )

    async def ainvoke(self, query: str, config=None, **kwargs) -> str:
        """
        Asynchronously invoke the main agent for a given query and return the full response once complete.

        Args:
            query (str): The user query or request to process.
            config (dict | None, optional): The configuration for the agent invoke. Defaults to None.
        """
        return await self.agent.ainvoke(
            {"messages": [{"role": "user", "content": query}]},
            config=config,
            **kwargs,
        )

    def __del__(self):
        job_manager.shutdown()
