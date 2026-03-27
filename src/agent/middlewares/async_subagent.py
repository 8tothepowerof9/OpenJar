from collections.abc import Awaitable, Callable
from typing import Any, cast

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import (
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import tool

from src.agent.job_manager import job_manager
from src.agent.loader import AgentLoader

ASYNC_RUN_SUBAGENT_TOOL_DESCRIPTION = """
Use this tool to start a sub-agent task asynchronously in the background.

Use it for longer-running or non-blocking work where you do not need the final
result immediately in the same model turn.

Inputs:
1. agent_name: Exact sub-agent name to run.
2. description: Detailed task instructions with expected output.

Returns:
- A job ID that you can use with check_async_task, cancel_async_task, and list_async_task.

Best practices:
1. Include clear objective, constraints, and completion criteria in description.
2. Start one job per distinct objective.
3. Store or reference job IDs so you can retrieve results later.
"""

ASYNC_CHECK_SUBAGENT_TOOL_DESCRIPTION = """
Use this tool to check status and retrieve results for a background sub-agent job.

Input:
1. job_id: The ID returned by start_async_task.

Behavior:
1. If running or queued, it returns current status.
2. If completed, it returns the final result.
3. If failed, it returns error details.

Use this tool whenever you need progress or output from an async job.
"""

ASYNC_CANCEL_SUBAGENT_TOOL_DESCRIPTION = """
Use this tool to cancel a background sub-agent job that is still pending or running.

Input:
1. job_id: The ID returned by start_async_task.

Use when:
1. The task is no longer needed.
2. Requirements changed.
3. A duplicate or incorrect job was launched.

After cancellation, verify final job state with check_async_task or list_async_task.
"""

ASYNC_LIST_SUBAGENT_TOOL_DESCRIPTION = """
Use this tool to list all known async sub-agent jobs and their statuses.

Use when:
1. You need to find a job ID.
2. You want an overview of running, completed, or failed jobs.
3. You are deciding whether to poll, cancel, or start additional jobs.

This is useful for job lifecycle management across multiple background tasks.
"""

ASYNC_SYSTEM_PROMPT = """
## async sub-agent tools

You can delegate long-running work to background sub-agent jobs.
Use async tools to avoid blocking the current response when immediate results are not required.

Available tools:
1. start_async_task(agent_name, description): Launch background work and get a job ID.
2. check_async_task(job_id): Get status and retrieve result when complete.
3. cancel_async_task(job_id): Cancel pending or running work.
4. list_async_task(): List all jobs and statuses.

Recommended workflow:
1. Start a job with a precise, self-contained description.
2. Return the job ID to the user when relevant.
3. Poll with check_async_task until completion.
4. Summarize final result once available.
5. Cancel stale or incorrect jobs when appropriate.

Decision guidance:
- Use synchronous sub-agent calls for short tasks requiring immediate output.
- Use async jobs for slow tasks, parallel tasks, or tasks that involve waiting.

Reliability guidance:
- Do not invent job IDs.
- Keep track of the job ID from start_async_task output.
- If a job cannot be found, communicate that clearly and suggest listing jobs.
"""


class AsyncSubAgentMiddleware(AgentMiddleware):
    """
    Middleware that provides tools for running, checking, canceling and
    listing asynchronous sub-agent tasks as background jobs.
    This allows agents to delegate long-running work without blocking the main response generation.
    """

    def __init__(
        self,
        *,
        agent_loader: AgentLoader,
        run_subagent_prompt: str = ASYNC_RUN_SUBAGENT_TOOL_DESCRIPTION,
        check_subagent_prompt: str = ASYNC_CHECK_SUBAGENT_TOOL_DESCRIPTION,
        cancel_subagent_prompt: str = ASYNC_CANCEL_SUBAGENT_TOOL_DESCRIPTION,
        list_subagent_prompt: str = ASYNC_LIST_SUBAGENT_TOOL_DESCRIPTION,
        system_prompt: str = ASYNC_SYSTEM_PROMPT,
        recursion_limit: int = 70,
    ) -> None:
        """
        Initializes the AsyncSubAgentMiddleware with the provided parameters.

        Args:
            agent_loader (AgentLoader): An instance of AgentLoader to load sub-agents by name.
            run_subagent_prompt (str, optional): The prompt for the start_async_task tool. Defaults to ASYNC_RUN_SUBAGENT_TOOL_DESCRIPTION.
            check_subagent_prompt (str, optional): The prompt for the check_async_task tool. Defaults to ASYNC_CHECK_SUBAGENT_TOOL_DESCRIPTION.
            cancel_subagent_prompt (str, optional): The prompt for the cancel_async_task tool. Defaults to ASYNC_CANCEL_SUBAGENT_TOOL_DESCRIPTION.
            list_subagent_prompt (str, optional): The prompt for the list_async_task tool. Defaults to ASYNC_LIST_SUBAGENT_TOOL_DESCRIPTION.
            system_prompt (str, optional): The system prompt for the middleware. Defaults to ASYNC_SYSTEM_PROMPT.
            recursion_limit (int, optional): The recursion limit for the middleware. Defaults to 70.
        """
        super().__init__()
        self.agent_loader = agent_loader
        self.run_subagent_prompt = run_subagent_prompt
        self.check_subagent_prompt = check_subagent_prompt
        self.cancel_subagent_prompt = cancel_subagent_prompt
        self.list_subagent_prompt = list_subagent_prompt
        self.system_prompt = system_prompt
        self.recursion_limit = recursion_limit

        @tool(description=self.run_subagent_prompt)
        def start_async_task(agent_name: str, description: str) -> str:
            """
            Launch a subagent task in the background for long-running tasks.
            Poll the status or retrieve the result with the job(...) tool.

            Args:
                agent_name (str): The name of the subagent to launch.
                description (str): A description of the task to perform.

            Returns:
                str: A message containing the background job ID.
            """
            job_id = job_manager.submit(agent_name, description, self._invoke_subagent)
            return f"Background job started. Job ID: {job_id}"

        @tool(description=self.check_subagent_prompt)
        def check_async_task(job_id: str) -> str:
            """
            Check current status and retrieve result for a background job.

            Args:
                job_id (str): The job ID returned by background_task(...).

            Returns:
                str: Status information or the completed result.
            """
            result = job_manager.get_job(job_id)

            return self._format_check(result, job_id)

        @tool(description=self.cancel_subagent_prompt)
        def cancel_async_task(job_id: str) -> str:
            """
            Cancel a background job if it's still running.

            Args:
                job_id (str): The job ID returned by background_task(...).

            Returns:
                str: A message indicating the result of the cancellation attempt.
            """
            result = job_manager.cancel_job(job_id)
            return f"Job status: {result.get('status')}. Message: {result.get('message', '')}"

        @tool(description=self.list_subagent_prompt)
        def list_async_task() -> str:
            """
            List all background jobs with their current status.

            Returns:
                str: A list of all background jobs with their current status.
            """
            jobs = job_manager.get_all()
            return self._format_list(jobs)

        self.tools = [
            start_async_task,
            check_async_task,
            cancel_async_task,
            list_async_task,
        ]

    async def _invoke_subagent(self, agent_name: str, description: str) -> str:
        agent = self.agent_loader.get(agent_name)

        try:
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": description}]},
                config={"recursion_limit": self.recursion_limit},
            )
        except Exception as e:
            return f"Error invoking subagent '{agent_name}': {str(e)}"
        return result["messages"][-1].content

    def _format_check(self, result: dict[str, Any], job_id: str) -> str:
        if not result:
            return f"status: not_found\njob_id: {job_id}\nmessage: No job found"

        job_id = result.get("job_id", "")
        status = result.get("status", "unknown")
        agent = result.get("agent_name", "")
        description = result.get("description", "")
        output = result.get("result", "")
        error = result.get("error", "")

        lines = [
            f"status: {status}",
            f"job_id: {job_id}",
            f"agent: {agent}",
            f"description: {description}",
        ]

        if output:
            lines.append(f"result: {output}")
        if error:
            lines.append(f"error: {error}")

        return "\n".join(lines)

    def _format_list(self, jobs: list) -> str:
        if not jobs:
            return "No background jobs found."

        lines = []
        for job in jobs:
            job_id = job.get("job_id", "")
            status = job.get("status", "unknown")
            agent = job.get("agent_name", "")
            description = job.get("description", "")
            lines.append(
                f"job_id: {job_id} | status: {status} | agent: {agent} | description: {description}"
            )

        return "\n".join(lines)

    def wrap_model_call(
        self,
        request: ModelRequest[None],
        handler: Callable[[ModelRequest[None]], ModelResponse[Any]],
    ) -> ModelResponse[Any] | AIMessage | ExtendedModelResponse[Any]:
        """
        Wraps the model call to include the async subagent tool system prompt.

        Args:
            request (ModelRequest[None]): Model request to execute (includes state and runtime).
            handler (Callable[[ModelRequest[None]], Awaitable[ModelResponse[Any]]]): Asynchronous callback that executes the model request and returns
                `ModelResponse`.

        Returns:
            ModelResponse[Any] | AIMessage | ExtendedModelResponse[Any]: The model call result.
        """
        if request.system_message is not None:
            new_system_content = [
                *request.system_message.content_blocks,
                {"type": "text", "text": f"\n\n{self.system_prompt}"},
            ]
        else:
            new_system_content = [{"type": "text", "text": self.system_prompt}]

        new_system_message = SystemMessage(
            content=cast("list[str | dict[str, str]]", new_system_content)
        )

        return handler(request.override(system_message=new_system_message))

    async def awrap_model_call(
        self,
        request: ModelRequest[None],
        handler: Callable[[ModelRequest[None]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any] | AIMessage | ExtendedModelResponse[Any]:
        """
        Asynchronous version of `wrap_model_call`. Update the system message to include the get async sub-agents prompt

        Args:
            request (ModelRequest[None]): Model request to execute (includes state and runtime).
            handler (Callable[[ModelRequest[None]], Awaitable[ModelResponse[Any]]]): Asynchronous callback that executes the model request and returns
                `ModelResponse`.

        Returns:
            ModelResponse[Any] | AIMessage | ExtendedModelResponse[Any]: The model call result.
        """
        if request.system_message is not None:
            new_system_content = [
                *request.system_message.content_blocks,
                {"type": "text", "text": f"\n\n{self.system_prompt}"},
            ]
        else:
            new_system_content = [{"type": "text", "text": self.system_prompt}]
        new_system_message = SystemMessage(
            content=cast("list[str | dict[str, str]]", new_system_content)
        )

        return await handler(request.override(system_message=new_system_message))
