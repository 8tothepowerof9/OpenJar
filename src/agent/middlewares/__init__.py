from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.agent.middlewares.artifact import ArtifactMiddleware
    from src.agent.middlewares.async_subagent import AsyncSubAgentMiddleware
    from src.agent.middlewares.base import SystemPromptMiddleware
    from src.agent.middlewares.complex_task import AsyncMultiAgentMiddleware
    from src.agent.middlewares.dynamic_prompt import create_subagents_dynamic_prompt
    from src.agent.middlewares.load_agents import GetSubAgentsMiddleware
    from src.agent.middlewares.shared import create_shared_middleware
    from src.agent.middlewares.skills import SkillMiddleware, SkillRegistry
    from src.agent.middlewares.subagent import SubAgentMiddleware

_MOVED_PATHS = {
    "AsyncSubAgentMiddleware": "src.agent.middlewares.async_subagent",
    "AsyncMultiAgentMiddleware": "src.agent.middlewares.complex_task",
    "create_subagents_dynamic_prompt": "src.agent.middlewares.dynamic_prompt",
    "GetSubAgentsMiddleware": "src.agent.middlewares.load_agents",
    "SubAgentMiddleware": "src.agent.middlewares.subagent",
    "SystemPromptMiddleware": "src.agent.middlewares.base",
    "SkillMiddleware": "src.agent.middlewares.skills",
    "SkillRegistry": "src.agent.middlewares.skills",
    "create_shared_middleware": "src.agent.middlewares.shared",
    "ArtifactMiddleware": "src.agent.middlewares.artifact",
}


# Lazy import to prevent circular
def __getattr__(name: str) -> Any:
    if name in _MOVED_PATHS:
        import importlib

        module = importlib.import_module(_MOVED_PATHS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")
