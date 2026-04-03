import numpy as np
from langchain.agents.middleware import AgentMiddleware
from langchain_core.embeddings import Embeddings
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings

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

GET_SUB_AGENTS_SKILL_CONTENT = """\
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
    Middleware to provide a tool for retrieving relevant sub-agents based on a query.
    """

    def __init__(
        self,
        *,
        agents_loader: AgentLoader,
        embeddings: Embeddings | None = None,
        tool_description: str = GET_SUB_AGENTS_TOOL_DESCRIPTION,
        top_k: int = 5,
    ) -> None:
        super().__init__()
        self.agent_loader = agents_loader
        self.embeddings = embeddings or OpenAIEmbeddings()
        self.tool_description = tool_description
        self.top_k = top_k
        self._agent_embeddings: np.ndarray | None = None
        self._agent_texts: list[str] = []

        @tool(description=self.tool_description)
        async def get_sub_agents(query: str) -> str:
            """
            Retrieve suitable sub-agents name and description based on query.

            Args:
                query (str): query to find

            Returns:
                str: formatted string of agent name and description
            """
            agents = await self._search_agent(query)
            return self._format_agents(agents)

        self.tools = [get_sub_agents]

    async def _build_agent_embeddings(self) -> None:
        """Compute and cache embeddings for all agent descriptions."""
        all_agents = list(self.agent_loader.agents.values())
        self._agent_texts = [
            f"{agent.name}: {agent.description}" for agent in all_agents
        ]
        vectors = await self.embeddings.aembed_documents(self._agent_texts)
        self._agent_embeddings = np.array(vectors)

    def _cosine_similarity(
        self, query_vec: np.ndarray, doc_vecs: np.ndarray
    ) -> np.ndarray:
        """Compute cosine similarity between a query vector and document vectors."""
        query_norm = query_vec / np.linalg.norm(query_vec)
        doc_norms = doc_vecs / np.linalg.norm(doc_vecs, axis=1, keepdims=True)
        return doc_norms @ query_norm

    def _format_agents(self, agents: list[tuple[str, str, bool]]) -> str:
        if not agents:
            return "No agents found."

        lines = []
        width = max(len(name) for name, _, _ in agents)
        for i, (name, description, is_multi) in enumerate(agents, 1):
            lines.append(
                f"  {i}. {name:<{width}}  -  {description} {'[Multi-agent]' if is_multi else ''}"
            )
        return "\n".join(lines)

    async def _search_agent(self, query: str | None) -> list[tuple[str, str, bool]]:
        all_agents = [
            (agent.name, agent.description, agent.is_multi)
            for agent in self.agent_loader.agents.values()
        ]

        if not query:
            return all_agents[: self.top_k]

        if self._agent_embeddings is None or len(self._agent_texts) != len(all_agents):
            await self._build_agent_embeddings()

        assert self._agent_embeddings is not None

        query_vec = np.array(await self.embeddings.aembed_query(query))
        scores = self._cosine_similarity(query_vec, self._agent_embeddings)
        top_indices = np.argsort(scores)[::-1][: self.top_k]

        return [
            (all_agents[i][0], all_agents[i][1], all_agents[i][2]) for i in top_indices
        ]
