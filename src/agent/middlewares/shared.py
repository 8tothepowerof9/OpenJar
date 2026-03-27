from typing import Any, Sequence

from langchain.agents.middleware import (
    ClearToolUsesEdit,
    ContextEditingMiddleware,
    PIIMiddleware,
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
)
from langchain.chat_models import BaseChatModel


def create_shared_middleware(
    model: str | BaseChatModel,
) -> Sequence[Any]:
    """Return the middleware stack shared by the orchestrator and all sub-agents.

    This includes PII protection, summarization, tool-call limits, and
    context editing.
    """
    return [
        SummarizationMiddleware(
            model=model,
            trigger=("tokens", 50000),
            keep=("messages", 30),
        ),
        ToolCallLimitMiddleware(thread_limit=30, run_limit=30),
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),
        PIIMiddleware("ip", strategy="mask", apply_to_input=True),
        PIIMiddleware(
            "api_key",
            detector=r"sk-[a-zA-Z0-9]{32}",
            strategy="redact",
            apply_to_input=True,
        ),
        ContextEditingMiddleware(
            edits=[
                ClearToolUsesEdit(
                    trigger=20000,
                    keep=3,
                ),
            ],
        ),
    ]
