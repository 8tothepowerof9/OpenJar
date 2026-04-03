import importlib.util
import re
from dataclasses import dataclass, field
from hashlib import sha1
from pathlib import Path
from typing import Any, TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool, Tool
from langgraph.checkpoint.memory import InMemorySaver

from src.agent.discovery import (
    AGENTS_DIR,
    DiscoveredAgent,
    DiscoveredMultiAgent,
    discover_agents,
)
from src.agent.middlewares import create_shared_middleware
from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class SubAgent:
    """A loaded sub-agent/sub-multiagent with its metadata and runnable."""

    name: str
    description: str
    instruction: str
    model: str
    tools: list = field()
    _runnable: Any = field(repr=False)
    is_multi: bool = False

    def invoke(self, messages, **kwargs) -> Any:
        return self._runnable.invoke(messages, **kwargs)

    async def ainvoke(self, messages, **kwargs) -> Any:
        return await self._runnable.ainvoke(messages, **kwargs)


class AgentLoaderConfig(TypedDict, total=False):
    """
    Configuration for the AgentLoader, including callback functions.
    """


class AgentLoader:
    """Dynamically discovers and loads sub-agents from a directory.

    Each sub-folder must contain:
      - info.yaml  (name, description, instruction, optional model)
      - tools.py   (LangChain @tool-decorated functions)
    """

    def __init__(
        self, agents_dir: Path = AGENTS_DIR, config: AgentLoaderConfig | None = None
    ):
        """
        Initialize the AgentLoader with the directory to scan for agents and optional configuration.

        Args:
            agents_dir (Path, optional): The directory to scan for agents. Defaults to AGENTS_DIR.
            config (AgentLoaderConfig | None, optional): The configuration for the agent loader. Defaults to None.
        """
        self.agents_dir = agents_dir
        self._agents: dict[str, SubAgent] = {}
        self.config = config or {}

    @property
    def agents(self) -> dict[str, SubAgent]:
        if not self._agents:
            self.load()
        return self._agents

    def load(self) -> None:
        """Scan the agents directory and load all valid sub-agents."""
        self._agents.clear()

        for discovered in discover_agents(self.agents_dir):
            try:
                agent = self._build_agent(discovered)
            except Exception as e:
                logger.error(
                    "Skipping agent '%s': %s", discovered.name, e
                )
                continue
            if agent.name in self._agents:
                raise ValueError(
                    f"Duplicate agent name '{agent.name}' encountered while loading. "
                    "Each agent must have a unique name."
                )
            self._agents[agent.name] = agent

        logger.info(
            f"Loaded {len(self._agents)} agents. Available: {', '.join(self._agents)}"
        )

    def get(self, name: str) -> SubAgent:
        """Retrieve a loaded sub-agent by name."""
        return self.agents[name]

    def _build_agent(
        self, discovered: DiscoveredAgent | DiscoveredMultiAgent
    ) -> SubAgent:
        tools = self._load_tools(discovered.tools_path)

        name = discovered.name
        description = discovered.description
        instruction = discovered.instruction
        model = discovered.model
        model_identifier = self._build_model_identifier(discovered.provider, model)

        middlewares = [
            *create_shared_middleware(model=model_identifier),
        ]

        if isinstance(discovered, DiscoveredMultiAgent):
            for sub in discovered.sub_agents:
                sub_model_identifier = self._build_model_identifier(
                    sub.provider, sub.model
                )
                sub_agent = create_agent(
                    model=sub_model_identifier,
                    tools=self._load_tools(sub.tools_path),
                    system_prompt=sub.instruction.strip(),
                    middleware=middlewares,
                    checkpointer=InMemorySaver(),
                )
                tool = Tool(
                    name=re.sub(r"[^a-zA-Z0-9]+", "_", sub.name).lower(),
                    description=sub.description,
                    func=lambda query, agent=sub_agent: self._invoke_sub_agent_tool(
                        query, agent
                    ),
                )
                tools.append(tool)

            middlewares.append(TodoListMiddleware())

        runnable = create_agent(
            model=model_identifier,
            tools=tools,
            system_prompt=instruction.strip(),
            middleware=middlewares,
            checkpointer=InMemorySaver(),
        )

        return SubAgent(
            name=name,
            description=description,
            instruction=instruction,
            model=model,
            tools=tools,
            _runnable=runnable,
            is_multi=isinstance(discovered, DiscoveredMultiAgent),
        )

    @staticmethod
    def _load_tools(tools_path: Path) -> list:
        """Import tools.py and return all @tool-decorated callables.

        Returns an empty list (instead of crashing) when the module
        cannot be loaded or contains broken tool definitions.
        """
        module_name = f"tools_{sha1(str(tools_path).encode('utf-8')).hexdigest()[:12]}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, tools_path)
            if spec is None or spec.loader is None:
                logger.error("Cannot create module spec from %s", tools_path)
                return []
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as e:
            logger.error("Failed to load tools from %s: %s", tools_path, e)
            return []

        tools = []
        for attr_name in dir(module):
            try:
                attr = getattr(module, attr_name)
                if isinstance(attr, BaseTool):
                    tools.append(attr)
            except Exception as e:
                logger.warning(
                    "Error inspecting %s in %s: %s", attr_name, tools_path, e
                )
        return tools

    @staticmethod
    def _build_model_identifier(provider: object, model: str) -> str:
        """Build '<provider>:<model>' with a safe fallback when provider config is invalid."""
        provider_type = "openai"
        if isinstance(provider, dict):
            raw_provider_type = provider.get("type")
            if isinstance(raw_provider_type, str) and raw_provider_type.strip():
                provider_type = raw_provider_type.strip()
            else:
                logger.warning(
                    "Invalid provider type '%s'; defaulting to openai",
                    raw_provider_type,
                )
        else:
            logger.warning(
                "Invalid provider config '%s'; defaulting to openai", provider
            )
        return f"{provider_type}:{model}"

    @staticmethod
    def _invoke_sub_agent_tool(query: str, agent: Any) -> Any:
        result = agent.invoke({"messages": [HumanMessage(content=query)]})
        return result["messages"][-1].content
