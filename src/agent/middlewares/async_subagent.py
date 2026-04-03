from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.tools import tool

from src.agent.artifact_manager import LENGTH_THRESHOLD, Artifact, ArtifactManager
from src.agent.job_manager import Job, JobManager
from src.agent.loader import AgentLoader
from src.utils import get_logger

logger = get_logger(__name__)

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

ASYNC_SKILL_CONTENT = """\
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


@dataclass
class ContextSchema:
    """Context schema for async sub-agent tool calls."""

    call_origin: str
    job_id: str


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
        job_manager: JobManager,
        artifact_manager: ArtifactManager,
        run_subagent_prompt: str = ASYNC_RUN_SUBAGENT_TOOL_DESCRIPTION,
        check_subagent_prompt: str = ASYNC_CHECK_SUBAGENT_TOOL_DESCRIPTION,
        cancel_subagent_prompt: str = ASYNC_CANCEL_SUBAGENT_TOOL_DESCRIPTION,
        list_subagent_prompt: str = ASYNC_LIST_SUBAGENT_TOOL_DESCRIPTION,
        recursion_limit: int = 70,
    ) -> None:
        super().__init__()
        self.agent_loader = agent_loader
        self.job_manager = job_manager
        self.artifact_manager = artifact_manager
        self.recursion_limit = recursion_limit

        @tool(description=run_subagent_prompt)
        async def start_async_task(agent_name: str, description: str) -> str:
            """
            Launch a subagent task in the background for long-running tasks.
            Poll the status or retrieve the result with the job(...) tool.

            Args:
                agent_name (str): The name of the subagent to launch.
                description (str): A description of the task to perform.

            Returns:
                str: A message containing the background job ID.
            """
            job_id = self.job_manager.submit(
                agent_name,
                description,
                self._invoke_subagent,
            )
            return f"Background job started. Job ID: {job_id}"

        @tool(description=check_subagent_prompt)
        async def check_async_task(job_id: str) -> str:
            """
            Check current status and retrieve result for a background job.

            Args:
                job_id (str): The job ID returned by background_task(...).

            Returns:
                str: Status information or the completed result.
            """
            result = self.job_manager.get_job(job_id)

            return self._format_check(result, job_id)

        @tool(description=cancel_subagent_prompt)
        async def cancel_async_task(job_id: str) -> str:
            """
            Cancel a background job if it's still running.

            Args:
                job_id (str): The job ID returned by background_task(...).

            Returns:
                str: A message indicating the result of the cancellation attempt.
            """
            result = self.job_manager.cancel_job(job_id)
            return f"Job status: {result.get('status')}. Message: {result.get('message', '')}"

        @tool(description=list_subagent_prompt)
        def list_async_task() -> str:
            """
            List all background jobs with their current status.

            Returns:
                str: A list of all background jobs with their current status.
            """
            jobs = self.job_manager.get_all()
            return self._format_list(jobs)

        self.tools = [
            start_async_task,
            check_async_task,
            cancel_async_task,
            list_async_task,
        ]

    async def _invoke_subagent(
        self,
        agent_name: str,
        description: str,
        thread_id: str,
        job_id: str = "",
    ) -> str | Artifact:
        agent = self.agent_loader.get(agent_name)

        try:
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": description}]},
                config={
                    "recursion_limit": self.recursion_limit,
                    "configurable": {"thread_id": thread_id},
                },
                context=ContextSchema(
                    call_origin="async_subagent",
                    job_id=job_id,
                ),
            )
        except Exception as e:
            return f"Error invoking subagent '{agent_name}': {str(e)}"

        content = result["messages"][-1].content

        if len(content) > LENGTH_THRESHOLD:
            artifact = await self.artifact_manager.aadd_artifact(content, description)
            return artifact
        else:
            return content

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

    def _format_list(self, jobs: list[Job]) -> str:
        if not jobs:
            return "No background jobs found."

        lines = []
        for job in jobs:
            job_id = job.id
            status = job.status
            agent = job.agent_name
            description = job.description
            lines.append(
                f"job_id: {job_id} | status: {status} | agent: {agent} | description: {description}"
            )

        return "\n".join(lines)
