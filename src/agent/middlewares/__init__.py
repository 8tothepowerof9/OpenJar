from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.agent.middlewares.async_subagent import AsyncSubAgentMiddleware
    from src.agent.middlewares.callback import CallbackMiddleware
    from src.agent.middlewares.complex_task import AsyncMultiAgentMiddleware
    from src.agent.middlewares.dynamic_prompt import subagents_dynamic_prompt
    from src.agent.middlewares.load_agents import GetSubAgentsMiddleware
    from src.agent.middlewares.shared import create_shared_middleware
    from src.agent.middlewares.subagent import SubAgentMiddleware

_MOVED_PATHS = {
    "AsyncSubAgentMiddleware": "src.agent.middlewares.async_subagent",
    "CallbackMiddleware": "src.agent.middlewares.callback",
    "AsyncMultiAgentMiddleware": "src.agent.middlewares.complex_task",
    "subagents_dynamic_prompt": "src.agent.middlewares.dynamic_prompt",
    "GetSubAgentsMiddleware": "src.agent.middlewares.load_agents",
    "SubAgentMiddleware": "src.agent.middlewares.subagent",
    "create_shared_middleware": "src.agent.middlewares.shared",
}


# Lazy import to prevent circular
def __getattr__(name: str) -> Any:
    if name in _MOVED_PATHS:
        import importlib

        module = importlib.import_module(_MOVED_PATHS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")
