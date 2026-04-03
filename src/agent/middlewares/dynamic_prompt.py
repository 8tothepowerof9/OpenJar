from langchain.agents.middleware import ModelRequest, dynamic_prompt

from src.agent.loader import AgentLoader

THRESHOLD = 10


def create_subagents_dynamic_prompt(loader: AgentLoader):
    """Factory that returns a dynamic_prompt using the already-loaded AgentLoader.

    Avoids rescanning the filesystem on every model call.
    """

    @dynamic_prompt
    def subagents_dynamic_prompt(request: ModelRequest) -> str:
        original_prompt = request.system_prompt or ""
        agents = loader.agents  # already loaded, no filesystem I/O

        if len(agents) < THRESHOLD:
            lines = ["\n## Available sub-agents:"]
            for agent in agents.values():
                tag = " [Multi-Agent]" if agent.is_multi else ""
                lines.append(f"- {agent.name} - {agent.description}{tag}")
            original_prompt += "\n".join(lines)

        return original_prompt

    return subagents_dynamic_prompt
