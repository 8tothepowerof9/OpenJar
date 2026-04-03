from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, cast

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import (
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage, SystemMessage


class SystemPromptMiddleware(AgentMiddleware):
    """Base for middlewares that inject a section into the system prompt.

    Subclasses set ``system_prompt`` and get ``wrap_model_call`` /
    ``awrap_model_call`` for free — no copy-paste needed.
    """

    system_prompt: str = ""

    def _inject_system_prompt(
        self, request: ModelRequest[None]
    ) -> ModelRequest[None]:
        if not self.system_prompt:
            return request

        if request.system_message is not None:
            new_content = [
                *request.system_message.content_blocks,
                {"type": "text", "text": f"\n\n{self.system_prompt}"},
            ]
        else:
            new_content = [{"type": "text", "text": self.system_prompt}]

        new_msg = SystemMessage(
            content=cast("list[str | dict[str, str]]", new_content)
        )
        return request.override(system_message=new_msg)

    def wrap_model_call(
        self,
        request: ModelRequest[None],
        handler: Callable[[ModelRequest[None]], ModelResponse[Any]],
    ) -> ModelResponse[Any] | AIMessage | ExtendedModelResponse[Any]:
        return handler(self._inject_system_prompt(request))

    async def awrap_model_call(
        self,
        request: ModelRequest[None],
        handler: Callable[[ModelRequest[None]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any] | AIMessage | ExtendedModelResponse[Any]:
        return await handler(self._inject_system_prompt(request))
