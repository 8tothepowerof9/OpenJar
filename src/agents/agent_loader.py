import importlib.util
from hashlib import sha1
from pathlib import Path

from langchain.agents import create_agent
from langchain.tools import BaseTool

from src.agents.agent_discovery import AGENTS_DIR, discover_agents
from src.utils.logging import get_logger

logger = get_logger(__name__)


class AgentLoader:
    def __init__(self, agent_dir: Path = AGENTS_DIR):
        self.agent_dir = agent_dir
        agents = discover_agents(agent_dir)
        self.agents = {agent.name: agent for agent in agents}

    def create_agent_by_name(self, name: str, **kwargs):
        agent = self.agents.get(name)

        if not agent:
            logger.warning("Agent '%s' not found", name)
            return None

        model = f"{agent.provider}:{agent.model}"
        tools = self._build_tools(
            agent.tools_path,
        )

        return create_agent(model, tools=tools, system_prompt=agent.prompt, **kwargs)

    def get_agent_blueprint_by_name(self, name: str):
        agent = self.agents.get(name)

        if not agent:
            logger.warning("Agent blueprint '%s' not found", name)
            return None

        model = f"{agent.provider}:{agent.model}"
        tools = self._build_tools(
            agent.tools_path,
        )

        return model, tools, agent.prompt, agent.name, agent.description

    def _build_tools(self, tools_path: Path):
        module_name = f"tools_{sha1(str(tools_path).encode('utf-8')).hexdigest()[:12]}"

        try:
            spec = importlib.util.spec_from_file_location(module_name, tools_path)
            if spec is None or spec.loader is None:
                logger.warning("Could not load tools spec from %s", tools_path)
                return []
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as e:
            logger.error("Failed to load tools module from %s: %s", tools_path, e)
            return []

        tools = []
        for attr_name in dir(module):
            try:
                attr = getattr(module, attr_name)
                if isinstance(attr, BaseTool):
                    tools.append(attr)
            except Exception as e:
                logger.debug("Skipping attribute '%s': %s", attr_name, e)
                continue

        logger.debug("Loaded %d tool(s) from %s", len(tools), tools_path)
        return tools
