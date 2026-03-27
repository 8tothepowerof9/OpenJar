from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt

from src.agent.discovery import DiscoveredAgent, discover_agents

THRESHOLD = 8


def format_agents_list(agents: list[DiscoveredAgent]) -> str:
    """Format a list of discovered agents into a string for the system prompt."""
    if not agents:
        return "No sub-agents available."

    lines = ["\n## Available sub-agents:"]
    for agent in agents:
        lines.append(f"- {agent.name}: {agent.description}")
    return "\n".join(lines)


@dynamic_prompt
def subagents_dynamic_prompt(request: ModelRequest) -> str:
    """
    A dynamic prompt function that append a SUB-AGENT section to the
    system prompt if the number of sub-agents is smaller than 8.
    This allows the agent to quickly get an overview of the available
    sub-agents and their capabilities,
    without having to query for them explicitly.

    Args:
        request (ModelRequest): The model request object containing the current system prompt and other context.

    Returns:
        str: The modified system prompt with the SUB-AGENT section appended if applicable.
    """
    original_prompt = request.system_prompt or ""
    agents = discover_agents()

    if len(agents) < THRESHOLD:
        original_prompt += format_agents_list(agents)

    return original_prompt
