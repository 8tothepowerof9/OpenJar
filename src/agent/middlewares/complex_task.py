from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import (
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
)
from langchain.chat_models import BaseChatModel
from langchain.tools import tool
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool

from src.agent.artifact_manager import LENGTH_THRESHOLD, Artifact, ArtifactManager
from src.agent.job_manager import JobManager
from src.agent.middlewares.async_subagent import ContextSchema

COMPLEX_TASK_SOLVER_TOOL_DESCRIPTION = """
Use this tool to launch a dedicated multi-agent solver for complex objectives.

This tool starts background execution and returns a job ID. Use it when a task is too
large or multi-step for a single direct response and benefits from coordinated tool use.

Input:
1. description: A complete task brief including objective, constraints, required output,
   and any important context.

When to use:
1. The request requires decomposition into multiple subtasks.
2. The task likely needs iterative reasoning, retries, or broad tool usage.
3. The user asks for thorough end-to-end execution on a complex problem.

When not to use:
1. The task is simple and can be answered directly.
2. A single specialized sub-agent call is sufficient.

Best practices for description:
1. State the final deliverable first.
2. Include explicit acceptance criteria.
3. Mention relevant files, boundaries, and non-goals.
4. Specify desired output format.
"""

COMPLEX_TASK_SOLVER_SKILL_CONTENT = """\
## start_multiagent_solver

You can delegate long-running, difficult, multi-step work to a background multi-agent solver.
Use start_multiagent_solver for complex tasks that need planning, orchestration, and
tool-assisted execution beyond a single short turn.
Your goal is to trigger the solver with a single,
comprehensive instruction set to minimize round-trips.

Operational guidance:
1. Prefer direct answering for straightforward requests.
2. Use start_multiagent_solver when complexity, breadth, or uncertainty is high.
3. Provide a self-contained description with clear success criteria.
4. Return or track the job ID so progress and results can be retrieved later.

Task brief checklist:
1. Objective and expected final result.
2. Key constraints and assumptions.
3. Scope boundaries and what to avoid.
4. Output structure (summary, patch, report, etc.).

Quality guidance:
- Be explicit about correctness requirements.
- Ask for concise, verifiable outputs.
- Prefer deterministic instructions over open-ended phrasing.

Execution note:
- This tool runs asynchronously in the background.
- It should be used for substantial objectives, not routine single-step actions.
"""

SOLVER_SYSTEM_PROMPT = """
Your role is to orchestrate synchronized specialized sub-agents and tools to complete user requests reliably.
Your goal is to complete the task in the fewest possible turns

# Operating principles:
1. Solve simple requests directly when no delegation is needed.
2. For specialized tasks, discover or choose the best sub-agent and delegate clearly.
3. Track progress with todos for complex multi-step work.

# Task execution behavior:
1. Restate objective briefly in your internal plan, then act.
2. Provide sub-agents with precise, self-contained instructions.
3. Validate returned results for completeness and consistency before responding.

# Safety and quality:
1. Always follow and answer user requests faithfully.
2. Do not attempt to delegate this task again; you are the final executor.
"""

BLACKLISTED_TOOLS = frozenset(
    [
        "start_async_task",
        "check_async_task",
        "cancel_async_task",
        "list_async_task",
        "update_async_task",
        "start_multiagent_solver",
    ]
)


class AsyncMultiAgentMiddleware(AgentMiddleware):
    """
    Middleware to enable agents to deploy a multiagent solver for complex tasks
    via a `start_multiagent_solver` tool.

    Unlike other middlewares, this one overrides wrap_model_call to dynamically
    inject the solver tool (built from the current request's tools/model).
    It does NOT inject a system prompt — the agent learns how to use the tool
    via the skill system.
    """

    def __init__(
        self,
        model: str | BaseChatModel,
        job_manager: JobManager,
        artifact_manager: ArtifactManager,
        dynamic_prompt_middleware: Any = None,
        solve_system_prompt: str = SOLVER_SYSTEM_PROMPT,
        tool_description: str = COMPLEX_TASK_SOLVER_TOOL_DESCRIPTION,
        recursion_limit: int = 70,
    ) -> None:
        super().__init__()
        self.model = model
        self.job_manager = job_manager
        self.artifact_manager = artifact_manager
        self.dynamic_prompt_middleware = dynamic_prompt_middleware
        self.solver_system_prompt = solve_system_prompt
        self.tool_description = tool_description
        self.recursion_limit = recursion_limit

    @staticmethod
    def _get_tool_name(t: BaseTool | dict[str, Any]) -> str:
        if isinstance(t, BaseTool):
            return t.name
        elif isinstance(t, dict):
            return t.get("name", "unknown")
        else:
            raise ValueError(f"Invalid tool type: {type(t)}")

    async def _invoke_multiagent_solver(
        self,
        description: str,
        thread_id: str,
        job_id: str,
        available_tools: list[BaseTool | dict[str, Any]],
        active_model: str | BaseChatModel | None,
    ) -> str | Artifact:
        model = active_model or self.model
        if model is None:
            return "Error invoking subagent: no model configured for multiagent solver."

        middlewares = []
        if self.dynamic_prompt_middleware is not None:
            middlewares.append(self.dynamic_prompt_middleware)

        agent = create_agent(
            model=model,
            tools=available_tools,
            system_prompt=self.solver_system_prompt,
            middleware=middlewares,
        )

        try:
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": description}]},
                config={
                    "recursion_limit": self.recursion_limit,
                    "configurable": {"thread_id": thread_id},
                },
                context=ContextSchema(
                    call_origin="complex_task_solver",
                    job_id=job_id,
                ),  # type: ignore
            )
        except Exception as e:
            return f"Error invoking multiagent solver: {str(e)}"

        content = result["messages"][-1].content

        if len(content) > LENGTH_THRESHOLD:
            artifact = await self.artifact_manager.aadd_artifact(
                content, "complex_task_solver", description
            )
            return artifact
        else:
            return content

    def _build_start_multiagent_solver_tool(
        self,
        available_tools: list[BaseTool | dict[str, Any]],
        active_model: str | BaseChatModel | None,
    ) -> BaseTool:
        @tool(description=self.tool_description)
        async def start_multiagent_solver(description: str) -> str:
            """
            Deploy a multiagent system to solve a complex task.
            The multiagent has access to all sub-agents available to the agent, and can use them as needed.

            Args:
                description (str): A detailed description of the complex task to be solved.

            Returns:
                str: The result of the multiagent solving process.
            """
            agent_name = "complex_task_solver"
            job_id = self.job_manager.submit(
                agent_name,
                description,
                lambda an, desc, tid, jid: self._invoke_multiagent_solver(
                    desc, tid, jid, available_tools, active_model
                ),
            )
            return f"Background job started for multiagent solver. Job ID: {job_id}"

        return start_multiagent_solver

    def _override_with_solver_tool(
        self, request: ModelRequest[None]
    ) -> ModelRequest[None]:
        """Build the solver tool from the current request's tools/model and inject it."""
        tools = request.tools
        available_tools = [
            t for t in tools if self._get_tool_name(t) not in BLACKLISTED_TOOLS
        ]
        solver_tool = self._build_start_multiagent_solver_tool(
            available_tools, request.model
        )
        new_tools = [*request.tools, solver_tool]
        return request.override(tools=new_tools)

    def wrap_model_call(
        self,
        request: ModelRequest[None],
        handler: Callable[[ModelRequest[None]], ModelResponse[Any]],
    ) -> ModelResponse[Any] | AIMessage | ExtendedModelResponse[Any]:
        return handler(self._override_with_solver_tool(request))

    async def awrap_model_call(
        self,
        request: ModelRequest[None],
        handler: Callable[[ModelRequest[None]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any] | AIMessage | ExtendedModelResponse[Any]:
        return await handler(self._override_with_solver_tool(request))
