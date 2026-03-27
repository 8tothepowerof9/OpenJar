import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.tools import BaseTool

from src.agent.discovery import AGENTS_DIR, DiscoveredAgent, discover_agents
from src.agent.middlewares import CallbackMiddleware, create_shared_middleware
from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class SubAgent:
    """A loaded sub-agent with its metadata and runnable."""

    name: str
    description: str
    instruction: str
    model: str
    tools: list = field()
    _runnable: Any = field(repr=False)

    def invoke(self, messages, **kwargs) -> Any:
        return self._runnable.invoke(messages, **kwargs)

    async def ainvoke(self, messages, **kwargs) -> Any:
        return await self._runnable.ainvoke(messages, **kwargs)


class AgentLoader:
    """Dynamically discovers and loads sub-agents from a directory.

    Each sub-folder must contain:
      - info.yaml  (name, description, instruction, optional model)
      - tools.py   (LangChain @tool-decorated functions)
    """

    def __init__(self, agents_dir: Path = AGENTS_DIR):
        self.agents_dir = agents_dir
        self._agents: dict[str, SubAgent] = {}

    @property
    def agents(self) -> dict[str, SubAgent]:
        if not self._agents:
            self.load()
        return self._agents

    def load(self) -> None:
        """Scan the agents directory and load all valid sub-agents."""
        self._agents.clear()

        for discovered in discover_agents(self.agents_dir):
            agent = self._build_agent(discovered)
            self._agents[agent.name] = agent

        logger.info(
            f"Loaded {len(self._agents)} agents. Available: {', '.join(self._agents)}"
        )

    def get(self, name: str) -> SubAgent:
        """Retrieve a loaded sub-agent by name."""
        return self.agents[name]

    def _build_agent(self, discovered: DiscoveredAgent) -> SubAgent:
        tools = self._load_tools(discovered.tools_path)

        name = discovered.name
        description = discovered.description
        instruction = discovered.instruction
        model = discovered.model
        provider = discovered.provider

        middlewares = [
            *create_shared_middleware(model=f"{provider['type']}:{model}"),
            CallbackMiddleware(),  # TODO: Empty for now
        ]

        runnable = create_agent(
            model=f"{provider['type']}:{model}",
            tools=tools,
            system_prompt=instruction.strip(),
            middleware=middlewares,
        )

        return SubAgent(
            name=name,
            description=description,
            instruction=instruction,
            model=model,
            tools=tools,
            _runnable=runnable,
        )

    @staticmethod
    def _load_tools(tools_path: Path) -> list:
        """Import tools.py and return all @tool-decorated callables."""
        spec = importlib.util.spec_from_file_location("tools", tools_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from {tools_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        tools = []
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, BaseTool):
                tools.append(attr)
        return tools
