from __future__ import annotations

from langchain.agents.middleware import AgentMiddleware
from langchain_core.tools import tool

from src.agent.artifact_manager import LENGTH_THRESHOLD, ArtifactManager
from src.agent.loader import AgentLoader

SYNC_RUN_SUBAGENT_TOOL_DESCRIPTION = """
Use this tool to run a sub-agent synchronously and return its final result in the current turn.
If the result is too long, the return value will be an Artifact reference containing a summary.

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

SYNC_TOOL_SKILL_CONTENT = """\
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
        artifact_manager: ArtifactManager,
        tool_description: str = SYNC_RUN_SUBAGENT_TOOL_DESCRIPTION,
        recursion_limit: int = 70,
    ) -> None:
        super().__init__()
        self.tool_description = tool_description
        self.loader = agent_loader
        self.artifact_manager = artifact_manager
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

            content = result["messages"][-1].content

            if len(content) > LENGTH_THRESHOLD:
                artifact = await self.artifact_manager.aadd_artifact(
                    content, agent_name, description
                )
                return f"Result too long, saved as {artifact}"
            else:
                return content

        self.tools = [invoke_subagent]
