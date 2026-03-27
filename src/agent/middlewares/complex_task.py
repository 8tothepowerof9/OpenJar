from collections.abc import Awaitable, Callable
from typing import Any, cast

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import (
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
)
from langchain.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from src.agent.job_manager import job_manager
from src.agent.loader import AgentLoader
from src.agent.middlewares.dynamic_prompt import subagents_dynamic_prompt

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

COMPLEX_TASK_SOLVER_SYSTEM_PROMPT = """
## start_multiagent_solver

You job is to delegate long-running, difficult, multi-step work to a background multi-agent solver.
Use start_multiagent_solver for complex tasks that need planning, orchestration, and
tool-assisted execution beyond a single short turn.

Operational guidance:
1. Prefer direct answering for straightforward requests.
2. Use start_multiagent_solver when complexity, breadth, or uncertainty is high.
3. Provide a self-contained description with clear success criteria.
4. Return or track the job ID so progress and results can be retrieved later.

Task brief checklist:
1. Objective and expected final artifact.
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


class AsyncMultiAgentMiddleware(AgentMiddleware):
    """
    Middleware to enable agents to deploy a multiagent solver for complex tasks
    via a `start_multiagent_solver` tool.
    """

    def __init__(
        self,
        agent_loader: AgentLoader,
        model: str | BaseChatModel | None = None,
        system_prompt: str = COMPLEX_TASK_SOLVER_SYSTEM_PROMPT,
        tool_description: str = COMPLEX_TASK_SOLVER_TOOL_DESCRIPTION,
        recursion_limit: int = 70,
    ) -> None:
        """
        Initialize the AsyncMultiAgentMiddleware.

        Args:
            agent_loader (AgentLoader): The loader for initializing sub-agents.
            model (str | BaseChatModel | None, optional): The chat model to use. Defaults to None.
            system_prompt (str, optional): The system prompt for the middleware. Defaults to COMPLEX_TASK_SOLVER_SYSTEM_PROMPT.
            tool_description (str, optional): The description for the start_multiagent_solver tool. Defaults to COMPLEX_TASK_SOLVER_TOOL_DESCRIPTION.
            recursion_limit (int, optional): The recursion limit for the middleware. Defaults to 70.
        """
        self.agent_loader = agent_loader
        self.model = model
        self.system_prompt = system_prompt
        self.tool_description = tool_description
        self.recursion_limit = recursion_limit

        self._available_tools: list[BaseTool | dict[str, Any]] = []
        self._active_model: str | BaseChatModel | None = None
        self.blacklisted_tools = [
            "start_async_task",
            "check_async_task",
            "cancel_async_task",
            "list_async_task",
            "update_async_task",
        ]

    @staticmethod
    def _get_tool_name(tool: BaseTool | dict[str, Any]) -> str:
        if isinstance(tool, BaseTool):
            return tool.name
        elif isinstance(tool, dict):
            return tool.get("name", "unknown")
        else:
            raise ValueError(f"Invalid tool type: {type(tool)}")

    async def _invoke_multiagent_solver(self, agent_name: str, description: str) -> str:
        """
        Invoke the multiagent solver with the given agent name and description.

        Args:
            agent_name (str): The name of the agent to invoke.
            description (str): The description of the task to be solved.

        Returns:
            str: The result of the multiagent solving process.
        """
        model = self._active_model or self.model
        if model is None:
            return "Error invoking subagent: no model configured for multiagent solver."

        agent = create_agent(
            model=model,
            tools=self._available_tools,
            system_prompt=self.system_prompt,
            middleware=[subagents_dynamic_prompt],
        )

        try:
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": description}]},
                config={"recursion_limit": self.recursion_limit},
            )
        except Exception as e:
            return f"Error invoking subagent '{agent_name}': {str(e)}"

        return result["messages"][-1].content

    def _build_start_multiagent_solver_tool(self) -> BaseTool:
        @tool(description=self.tool_description)
        def start_multiagent_solver(description: str) -> str:
            """
            Deploy a multiagent system to solve a complex task.
            The multigent has access to all sub-agents available to the agent, and can use them as needed.

            Args:
                description (str): A detailed description of the complex task to be solved.

            Returns:
                str: The result of the multiagent solving process.
            """
            agent_name = "complex_task_solver"
            job_id = job_manager.submit(
                agent_name, description, self._invoke_multiagent_solver
            )
            return f"Background job started for multiagent solver. Job ID: {job_id}"

        return start_multiagent_solver

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """
        Wrap tool calls to intercept calls to `start_multiagent_solver` and route them to the appropriate handler.

        Args:
            request (ToolCallRequest): The tool call request containing the tool name and arguments.
            handler (Callable[[ToolCallRequest], ToolMessage  |  Command[Any]]): The handler to process the tool call.

        Returns:
            ToolMessage | Command[Any]: The result of the tool call.
        """
        if request.tool_call["name"] == "start_multiagent_solver":
            return handler(
                request.override(tool=self._build_start_multiagent_solver_tool())
            )

        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """
        Asynchronous version of `wrap_tool_call`.

        Args:
            request (ToolCallRequest): The tool call request containing the tool name and arguments.
            handler (Callable[[ToolCallRequest], Awaitable[ToolMessage  |  Command[Any]]]): The handler to process the tool call.

        Returns:
            ToolMessage | Command[Any]: The result of the tool call.
        """
        if request.tool_call["name"] == "start_multiagent_solver":
            return await handler(
                request.override(tool=self._build_start_multiagent_solver_tool())
            )

        return await handler(request)

    def wrap_model_call(
        self,
        request: ModelRequest[None],
        handler: Callable[[ModelRequest[None]], ModelResponse[Any]],
    ) -> ModelResponse[Any] | AIMessage | ExtendedModelResponse[Any]:
        """
        Wraps the model call to include the `start_multiagent_solver` tool and system prompt,
        which allows the agent to deploy a multiagent system to solve complex tasks.

        Args:
            request (ModelRequest[None]): Model request to execute (includes state and runtime).
            handler (Callable[[ModelRequest[None]], Awaitable[ModelResponse[Any]]]):
            Asynchronous callback that executes the model request and returns `ModelResponse`.

        Returns:
            ModelResponse[Any] | AIMessage | ExtendedModelResponse[Any]: The model call result.
        """
        tools = request.tools

        available_tools = [
            t for t in tools if self._get_tool_name(t) not in self.blacklisted_tools
        ]
        self._available_tools = available_tools
        self._active_model = request.model

        new_tools = [*request.tools, self._build_start_multiagent_solver_tool()]

        if request.system_message is not None:
            new_system_content = [
                *request.system_message.content_blocks,
                {"type": "text", "text": f"\n\n{self.system_prompt}"},
            ]
        else:
            new_system_content = [{"type": "text", "text": self.system_prompt}]

        new_system_message = SystemMessage(
            content=cast("list[str | dict[str, str]]", new_system_content)
        )

        return handler(
            request.override(tools=new_tools, system_message=new_system_message)
        )

    async def awrap_model_call(
        self,
        request: ModelRequest[None],
        handler: Callable[[ModelRequest[None]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any] | AIMessage | ExtendedModelResponse[Any]:
        """
        Asynchronous version of `wrap_model_call`.

        Args:
            request (ModelRequest[None]): Model request to execute (includes state and runtime).
            handler (Callable[[ModelRequest[None]], Awaitable[ModelResponse[Any]]]):
            Asynchronous callback that executes the model request and returns `ModelResponse`.

        Returns:
            ModelResponse[Any] | AIMessage | ExtendedModelResponse[Any]: The model call result.
        """
        tools = request.tools

        available_tools = [
            t for t in tools if self._get_tool_name(t) not in self.blacklisted_tools
        ]
        self._available_tools = available_tools
        self._active_model = request.model

        new_tools = [*request.tools, self._build_start_multiagent_solver_tool()]

        if request.system_message is not None:
            new_system_content = [
                *request.system_message.content_blocks,
                {"type": "text", "text": f"\n\n{self.system_prompt}"},
            ]
        else:
            new_system_content = [{"type": "text", "text": self.system_prompt}]

        new_system_message = SystemMessage(
            content=cast("list[str | dict[str, str]]", new_system_content)
        )

        return await handler(
            request.override(tools=new_tools, system_message=new_system_message)
        )
