from collections.abc import Awaitable, Callable
from typing import Any, cast

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import (
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import tool

from src.agent.loader import AgentLoader

SYNC_RUN_SUBAGENT_TOOL_DESCRIPTION = """
Use this tool to run a sub-agent synchronously and return its final result in the current turn.

Use synchronous invocation for focused tasks that should complete quickly and whose output
is required before you can continue responding.

Inputs:
1. agent_name: Exact sub-agent name to invoke.
2. description: Detailed task instructions, including scope, constraints, and expected output.

Best practices:
1. Pass a complete, self-contained description so the sub-agent can execute without follow-up.
2. Include success criteria and output format when relevant.
3. Prefer specialized agents that match the task domain.

When not to use:
1. Long-running work that should continue in the background.
2. Tasks where you can answer directly without sub-agent delegation.
"""

SYNC_TOOL_SYSTEM_PROMPT = """
## invoke_subagent

You can delegate work to sub-agents using invoke_subagent.
This call is synchronous: it blocks until the sub-agent finishes and returns a result.

Use this tool when:
1. You need specialized capability from another agent.
2. The task is bounded enough to finish in one sub-agent run.
3. You need the result immediately to continue.

Invocation checklist:
1. Choose the correct agent name.
2. Write a clear, outcome-focused description.
3. Specify constraints, assumptions, and required output shape.
4. Invoke once per distinct task objective.

Description quality guidance:
- State the objective first.
- Include relevant context and files.
- Define what a complete answer must contain.
- Request concise but sufficient output.

If the task may be slow or involves waiting on external steps, use async sub-agent tools instead.
"""


class SubAgentMiddleware(AgentMiddleware):
    """
    Middleware that adds a tool for invoking sub-agents synchronously.
    """

    def __init__(
        self,
        *,
        agent_loader: AgentLoader,
        system_prompt: str = SYNC_TOOL_SYSTEM_PROMPT,
        tool_description: str = SYNC_RUN_SUBAGENT_TOOL_DESCRIPTION,
        recursion_limit: int = 70,
    ) -> None:
        """
        Initialize the SubAgentMiddleware.

        Args:
            agent_loader (AgentLoader): The loader for initializing sub-agents.
            system_prompt (str, optional): The system prompt for the middleware. Defaults to SYNC_TOOL_SYSTEM_PROMPT.
            tool_description (str, optional): The description for the invoke_subagent tool. Defaults to SYNC_RUN_SUBAGENT_TOOL_DESCRIPTION.
            recursion_limit (int, optional): The maximum number of recursive calls allowed. Defaults to 70.
        """
        super().__init__()
        self.system_prompt = system_prompt
        self.tool_description = tool_description
        self.loader = agent_loader
        self.recursion_limit = recursion_limit

        @tool(description=self.tool_description)
        async def invoke_subagent(agent_name: str, description: str) -> str:
            """
            Invoke a subagent synchronously

            Args:
                agent_name (str): The name of the subagent to invoke
                description (str): A detailed description of the task to be performed

            Returns:
                str: The result of the subagent invocation
            """
            agent = self.loader.get(agent_name)
            try:
                result = await agent.ainvoke(
                    {"messages": [{"role": "user", "content": description}]},
                    config={"recursion_limit": self.recursion_limit},
                )
            except Exception as e:
                return f"Error invoking subagent '{agent_name}': {str(e)}"
            return result["messages"][-1].content

        self.tools = [invoke_subagent]

    def wrap_model_call(
        self,
        request: ModelRequest[None],
        handler: Callable[[ModelRequest[None]], ModelResponse[Any]],
    ) -> ModelResponse[Any] | AIMessage | ExtendedModelResponse[Any]:
        """
        Wraps the model call to include the subagent tool system prompt.

        Args:
            request (ModelRequest[None]): Model request to execute (includes state and runtime).
            handler (Callable[[ModelRequest[None]], Awaitable[ModelResponse[Any]]]): Asynchronous callback that executes the model request and returns
                `ModelResponse`.

        Returns:
            ModelResponse[Any] | AIMessage | ExtendedModelResponse[Any]: The model call result.
        """
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

        return handler(request.override(system_message=new_system_message))

    async def awrap_model_call(
        self,
        request: ModelRequest[None],
        handler: Callable[[ModelRequest[None]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any] | AIMessage | ExtendedModelResponse[Any]:
        """
        Asynchronous version of `wrap_model_call`. Update the system message to include the get sub-agents prompt

        Args:
            request (ModelRequest[None]): Model request to execute (includes state and runtime).
            handler (Callable[[ModelRequest[None]], Awaitable[ModelResponse[Any]]]): Asynchronous callback that executes the model request and returns
                `ModelResponse`.

        Returns:
            ModelResponse[Any] | AIMessage | ExtendedModelResponse[Any]: The model call result.
        """
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

        return await handler(request.override(system_message=new_system_message))
