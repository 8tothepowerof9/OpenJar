import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict

from src.utils.logging import get_logger

logger = get_logger(__name__)


class JobStatus(Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


@dataclass
class Job:
    id: str
    agent: str
    task_description: str
    status: JobStatus = JobStatus.RUNNING
    result: Dict | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None


class JobManager:
    def __init__(self, on_complete):
        self._jobs = {}
        self._tasks = {}
        self._on_complete = on_complete

    def submit(self, agent: str, task_description: str, invoke_fn):
        job_id = str(uuid.uuid4().hex[:8])

        job = Job(
            id=job_id,
            agent=agent,
            task_description=task_description,
            status=JobStatus.RUNNING,
        )

        self._jobs[job_id] = job

        task = asyncio.create_task(self._run_job(job, invoke_fn))
        self._tasks[job_id] = task

        logger.info("Job %s submitted: agent='%s', task='%s'", job_id, agent, task_description)
        return job_id

    async def _run_job(self, job: Job, invoke_fn):
        try:
            result = await invoke_fn(job.agent, job.task_description)

            self._jobs[job.id].status = JobStatus.COMPLETED
            self._jobs[job.id].result = result
            self._jobs[job.id].completed_at = time.time()
            logger.info("Job %s completed successfully", job.id)
        except asyncio.CancelledError:
            self._jobs[job.id].status = JobStatus.CANCELED
            self._jobs[job.id].error = "Job was canceled."
            self._jobs[job.id].completed_at = time.time()
            logger.info("Job %s was canceled", job.id)
        except Exception as e:
            self._jobs[job.id].status = JobStatus.FAILED
            self._jobs[job.id].error = str(e)
            self._jobs[job.id].completed_at = time.time()
            logger.error("Job %s failed: %s", job.id, e)
        finally:
            self._tasks.pop(job.id, None)
            if self._on_complete:
                try:
                    await self._on_complete(job)
                except Exception as e:
                    logger.error("on_complete callback failed for job %s: %s", job.id, e)

    def cancel(self, job_id: str):
        task = self._tasks.get(job_id)
        job = self._jobs.get(job_id)

        if job is None:
            return {
                "status": "error",
                "message": f"No job found with id {job_id}",
            }

        if task is None or task.done():
            return {
                "status": job.status.value,
                "message": "Job is not running or already completed.",
            }

        task.cancel()

        return {"status": "cancelled", "message": "Cancellation requested."}

    def get_job(self, job_id: str):
        return self._jobs.get(job_id)

    def get_all(self):
        return list(self._jobs.values())

    def shutdown(self):
        logger.info("Shutting down JobManager, canceling %d running task(s)", len(self._tasks))
        for task in self._tasks.values():
            task.cancel()
        self._tasks.clear()
        self._jobs.clear()
