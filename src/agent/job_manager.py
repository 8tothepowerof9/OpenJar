import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine

from src.agent.artifact_manager import Artifact
from src.utils import get_logger

logger = get_logger(__name__)


class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    id: str
    agent_name: str
    description: str
    thread_id: str
    status: JobStatus = JobStatus.PENDING
    result: Artifact | str | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None


class JobManager:
    """Manages background subagent jobs using asyncio Tasks."""

    def __init__(
        self,
        on_complete: Callable[[Job], Coroutine[Any, Any, None]] | None = None,
    ):
        self._jobs: dict[str, Job] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._on_complete = on_complete

    def submit(
        self,
        agent_name: str,
        description: str,
        invoke_fn: Callable[[str, str, str, str], Coroutine[Any, Any, Artifact | str]],
    ) -> str:
        """Submit a background job. Returns the job ID."""
        job_id = uuid.uuid4().hex[:8]
        thread_id = uuid.uuid4().hex[:8]

        job = Job(
            id=job_id,
            agent_name=agent_name,
            description=description,
            thread_id=thread_id,
            status=JobStatus.RUNNING,
        )
        self._jobs[job_id] = job

        task = asyncio.create_task(
            self._run_job(job_id, invoke_fn, agent_name, description, thread_id)
        )
        self._tasks[job_id] = task

        return job_id

    async def _run_job(
        self,
        job_id: str,
        invoke_fn: Callable[[str, str, str, str], Coroutine[Any, Any, Artifact | str]],
        agent_name: str,
        description: str,
        thread_id: str,
    ) -> None:
        try:
            result = await invoke_fn(agent_name, description, thread_id, job_id)

            self._jobs[job_id].status = JobStatus.COMPLETED
            self._jobs[job_id].result = result
            self._jobs[job_id].completed_at = time.time()

        except asyncio.CancelledError:
            self._jobs[job_id].status = JobStatus.FAILED
            self._jobs[job_id].error = "Job was cancelled by user."
            self._jobs[job_id].completed_at = time.time()
            raise

        except Exception as e:
            self._jobs[job_id].status = JobStatus.FAILED
            self._jobs[job_id].error = str(e)
            self._jobs[job_id].completed_at = time.time()
        finally:
            self._tasks.pop(job_id, None)
            if self._on_complete:
                try:
                    await self._on_complete(self._jobs[job_id])
                except Exception as e:
                    logger.warning(
                        f"Error in on_complete callback for job {job_id}: {e}"
                    )
                    pass

    def cancel_job(self, job_id: str) -> dict:
        """Attempt to cancel a running task."""
        task = self._tasks.get(job_id)
        job = self._jobs.get(job_id)

        if job is None:
            return {"error": f"No job found with ID '{job_id}'"}

        if task is None or task.done():
            return {
                "status": job.status.value,
                "message": "Job is not running or already completed.",
            }

        task.cancel()

        return {"status": "cancelled", "job_id": job_id}

    def get_job(self, job_id: str, consume: bool = False) -> dict[str, Any]:
        job = self._jobs.get(job_id)
        if job is None:
            return {"error": f"No job found with ID '{job_id}'"}

        info = {
            "job_id": job.id,
            "agent_name": job.agent_name,
            "result": job.result,
            "description": job.description,
            "status": job.status.value,
            "error": job.error,
            "thread_id": job.thread_id,
        }

        if job.status == JobStatus.RUNNING:
            info["message"] = "Job is still running."
        elif job.status == JobStatus.FAILED:
            info["error"] = job.error
            if consume:
                self._delete_job(job_id)
        elif job.status == JobStatus.COMPLETED:
            info["result"] = job.result
            if consume:
                self._delete_job(job_id)

        return info

    def get_all(self) -> list[Job]:
        return list(self._jobs.values())

    def delete_job(self, job_id: str) -> dict:
        job = self._jobs.get(job_id)
        if job is None:
            return {"error": f"No job found with ID '{job_id}'"}
        if job.status == JobStatus.RUNNING:
            return {"status": "running", "message": "Cannot delete a running job."}

        self._delete_job(job_id)
        return {"status": "deleted", "job_id": job_id}

    def _delete_job(self, job_id: str) -> None:
        self._jobs.pop(job_id, None)
        self._tasks.pop(job_id, None)

    def shutdown(self):
        for task in self._tasks.values():
            task.cancel()
        self._jobs.clear()
        self._tasks.clear()
