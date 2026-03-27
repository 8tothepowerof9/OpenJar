from collections.abc import Awaitable, Callable
from difflib import SequenceMatcher
from typing import Any, cast

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import (
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from src.agent.loader import AgentLoader

GET_SUB_AGENTS_TOOL_DESCRIPTION = """

Use this tool to discover available sub-agents and pick the best one for a task.

Call this tool before invoking a sub-agent when the best agent is unclear.
Provide a short natural-language query describing the work you want done.

The tool returns ranked agent candidates with names and descriptions.
Use those results to choose an agent for synchronous or asynchronous invocation.

When to use:
1. You need to route work to the most suitable sub-agent.
2. You are unsure which specialized agent matches the request.
3. You want to verify available options before invoking a sub-agent.

When not to use:
1. You already know the exact sub-agent name with high confidence.
2. No sub-agent execution is needed for the current response.
"""

GET_SUB_AGENTS_SYSTEM_PROMPT = """

## get_sub_agents

You can call get_sub_agents to discover which sub-agent should handle a task.
Use it whenever agent routing is ambiguous.

Routing workflow:
1. Summarize the user objective as a short query.
2. Call get_sub_agents(query).
3. Select the best returned agent for the task.
4. Invoke that agent with a precise, outcome-focused description.

Good query examples:
- "search repository for authentication bug and suggest fix"
- "summarize research papers about retrieval augmented generation"
- "extract structured data from files and return csv"

Selection guidance:
- Prefer the agent whose description most directly matches required skills.
- If several agents are plausible, pick the narrowest specialist.
- If no strong match appears, ask for clarification or proceed without sub-agents.

Do not fabricate agent names. Use only names returned by available tools.
"""


class GetSubAgentsMiddleware(AgentMiddleware):
    """
    Middleware to provide a tool for retrieving relevant sub-agents based on a query,
    and to inject a system prompt guiding the agent to use this tool effectively.
    """

    def __init__(
        self,
        *,
        agents_loader: AgentLoader,
        system_prompt: str = GET_SUB_AGENTS_SYSTEM_PROMPT,
        tool_description: str = GET_SUB_AGENTS_TOOL_DESCRIPTION,
        top_k: int = 5,
    ) -> None:
        """
        Initialize the GetSubAgentsMiddleware.

        Args:
            agents_loader (AgentLoader): The loader for initializing sub-agents.
            system_prompt (str, optional): The system prompt for the middleware. Defaults to GET_SUB_AGENTS_SYSTEM_PROMPT.
            tool_description (str, optional): The description for the get_sub_agents tool. Defaults to GET_SUB_AGENTS_TOOL_DESCRIPTION.
            top_k (int, optional): The number of top agents to return. Defaults to 5.
        """
        super().__init__()
        self.agent_loader = agents_loader
        self.system_prompt = system_prompt
        self.tool_description = tool_description
        self.top_k = top_k

        @tool(description=self.tool_description)
        def get_sub_agents(query: str) -> str:
            """
            Retrieve suitable sub-agents name and description based on query.

            Args:
                query (str): query to find

            Returns:
                str: formatted string of agent name and description
            """
            agents = self._search_agent(query)
            return self._format_agents(agents)

        self.tools = [get_sub_agents]

    def _similarity(self, query: str, text: str) -> float:
        return SequenceMatcher(None, query.lower(), text.lower()).ratio()

    def _format_agents(self, agents: list[tuple[str, str]]) -> str:
        if not agents:
            return "No agents found."

        lines = []
        width = max(len(name) for name, _ in agents)
        for i, (name, description) in enumerate(agents, 1):
            lines.append(f"  {i}. {name:<{width}}  -  {description}")
        return "\n".join(lines)

    # TODO: Add semantic retrieval instead
    def _search_agent(self, query: str | None) -> list[tuple[str, str]]:
        all_agents = [
            (agent.name, agent.description)
            for agent in self.agent_loader.agents.values()
        ]

        if not query:
            return all_agents[: self.top_k]

        scored = []
        for name, description in all_agents:
            name_score = self._similarity(query, name)
            desc_score = self._similarity(query, description)
            score = 0.4 * name_score + 0.6 * desc_score
            scored.append((score, name, description))

        scored.sort(key=lambda x: x[0], reverse=True)

        if scored and scored[0][0] < 0.1:
            return all_agents

        return [(name, description) for _, name, description in scored[: self.top_k]]

    def wrap_model_call(
        self,
        request: ModelRequest[None],
        handler: Callable[[ModelRequest[None]], ModelResponse[Any]],
    ) -> ModelResponse[Any] | AIMessage | ExtendedModelResponse[Any]:
        """
        Update the system message to include the get sub-agents prompt

        Args:
            request (ModelRequest[None]): Model request to execute (includes state and runtime).
            handler (Callable[[ModelRequest[None]], ModelResponse[Any]]): Async callback that executes the model request and returns
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
