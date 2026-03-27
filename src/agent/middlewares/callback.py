from collections.abc import Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import AgentState
from langgraph.runtime import Runtime


class CallbackMiddleware(AgentMiddleware):
    """
    Middleware that allows for callbacks to be executed before and after agent and model calls.
    Useful for calling an external function when an event happens in the agent execution.
    """

    def __init__(
        self,
        before_agent_cb: Callable[[AgentState[Any], Runtime[None]], None] | None = None,
        after_agent_cb: Callable[[AgentState[Any], Runtime[None]], None] | None = None,
        before_model_cb: Callable[[AgentState[Any], Runtime[None]], None] | None = None,
        after_model_cb: Callable[[AgentState[Any], Runtime[None]], None] | None = None,
    ):
        """
        Initialize the CallbackMiddleware with the provided callbacks for agent and model calls.

        Args:
            before_agent_cb (Callable[[AgentState[Any], Runtime[None]], None] | None): A callback function to be called before an agent call is executed.
            after_agent_cb (Callable[[AgentState[Any], Runtime[None]], None] | None): A callback function to be called after an agent call is executed.
            before_model_cb (Callable[[AgentState[Any], Runtime[None]], None] | None): A callback function to be called before a model call is executed.
            after_model_cb (Callable[[AgentState[Any], Runtime[None]], None] | None): A callback function to be called after a model call is executed.
        """

        self.before_agent_cb = before_agent_cb
        self.after_agent_cb = after_agent_cb
        self.before_model_cb = before_model_cb
        self.after_model_cb = after_model_cb

    def after_agent(
        self, state: AgentState[Any], runtime: Runtime[None]
    ) -> dict[str, Any] | None:
        if self.after_agent_cb:
            self.after_agent_cb(state, runtime)

        return None

    def before_agent(
        self, state: AgentState[Any], runtime: Runtime[None]
    ) -> dict[str, Any] | None:
        if self.before_agent_cb:
            self.before_agent_cb(state, runtime)

        return None

    def before_model(
        self, state: AgentState[Any], runtime: Runtime[None]
    ) -> dict[str, Any] | None:
        if self.before_model_cb:
            self.before_model_cb(state, runtime)

        return None

    def after_model(
        self, state: AgentState[Any], runtime: Runtime[None]
    ) -> dict[str, Any] | None:
        if self.after_model_cb:
            self.after_model_cb(state, runtime)

        return None
