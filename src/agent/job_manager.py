import asyncio
import threading
import time
import uuid
from collections.abc import Coroutine
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


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
    status: JobStatus = JobStatus.PENDING
    result: str | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None


class JobManager:
    """Manages background subagent jobs using a thread pool."""

    def __init__(self, max_workers: int = 4):
        self._jobs: dict[str, Job] = {}
        self._futures: dict[str, Future] = {}
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.Lock()
        self._completed_unnotified: list[str] = []

    def submit(
        self,
        agent_name: str,
        description: str,
        invoke_fn: Callable[[str, str], Coroutine[Any, Any, str]],
    ) -> str:
        """Submit a background job. Returns the job ID."""
        job_id = uuid.uuid4().hex[:8]
        job = Job(
            id=job_id,
            agent_name=agent_name,
            description=description,
            status=JobStatus.RUNNING,
        )
        with self._lock:
            self._jobs[job_id] = job

        future = self._executor.submit(
            self._run_job, job_id, invoke_fn, agent_name, description
        )
        with self._lock:
            self._futures[job_id] = future

        return job_id

    def get_all(self) -> list[Job]:
        """Return a list of all jobs and its ids"""
        with self._lock:
            return list(self._jobs.values())

    def _run_job(
        self,
        job_id: str,
        invoke_fn: Callable[[str, str], Coroutine[Any, Any, str]],
        agent_name: str,
        description: str,
    ) -> None:
        """Executed in a worker thread."""
        try:
            result = asyncio.run(invoke_fn(agent_name, description))
            with self._lock:
                self._jobs[job_id].status = JobStatus.COMPLETED
                self._jobs[job_id].result = result
                self._jobs[job_id].completed_at = time.time()
                self._completed_unnotified.append(job_id)
        except Exception as e:
            with self._lock:
                self._jobs[job_id].status = JobStatus.FAILED
                self._jobs[job_id].error = str(e)
                self._jobs[job_id].completed_at = time.time()
                self._completed_unnotified.append(job_id)
        finally:
            with self._lock:
                self._futures.pop(job_id, None)

    def get_job(self, job_id: str, consume: bool = False) -> dict:
        """
        Retrieve status and result for a job.
        If consume is True and the job is finished, it will be deleted.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return {"error": f"No job found with ID '{job_id}'"}

            info = {
                "job_id": job.id,
                "agent_name": job.agent_name,
                "description": job.description,
                "status": job.status.value,
                "error": job.error,
            }

            if job.status == JobStatus.RUNNING:
                info["message"] = "Job is still running."

            elif job.status == JobStatus.FAILED:
                info["error"] = job.error
                if consume:
                    self._delete_job_locked(job_id)

            elif job.status == JobStatus.COMPLETED:
                info["result"] = job.result
                if consume:
                    self._delete_job_locked(job_id)

            return info

    def delete_job(self, job_id: str) -> dict:
        """Delete a finished job from memory."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return {"error": f"No job found with ID '{job_id}'"}
            if job.status == JobStatus.RUNNING:
                return {
                    "status": "running",
                    "message": "Cannot delete a running job.",
                }
            self._delete_job_locked(job_id)
            return {"status": "deleted", "job_id": job_id}

    def _delete_job_locked(self, job_id: str) -> None:
        """Delete a job and all tracking references. Caller must hold lock."""
        self._jobs.pop(job_id, None)
        self._futures.pop(job_id, None)
        self._completed_unnotified = [
            pending_id
            for pending_id in self._completed_unnotified
            if pending_id != job_id
        ]

    def cancel_job(self, job_id: str) -> dict:
        """Attempt to cancel a running job."""
        with self._lock:
            future = self._futures.get(job_id)
            job = self._jobs.get(job_id)
            if job is None:
                return {"error": f"No job found with ID '{job_id}'"}
            if future is None or future.done():
                return {
                    "status": job.status.value,
                    "message": "Job is not running or already completed.",
                }
            cancelled = future.cancel()
            if cancelled:
                job.status = JobStatus.FAILED
                job.error = "Job was cancelled by user."
                job.completed_at = time.time()
                self._completed_unnotified.append(job_id)
                return {"status": "cancelled", "job_id": job_id}
            else:
                return {
                    "status": "running",
                    "message": "Failed to cancel the job.",
                }

    def shutdown(self):
        """Shut down the thread pool."""
        self._executor.shutdown(wait=False)

        with self._lock:
            self._jobs.clear()
            self._futures.clear()
            self._completed_unnotified.clear()


job_manager = JobManager()
