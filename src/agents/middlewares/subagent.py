from typing import Any

import numpy as np
from langchain.agents.middleware import AgentMiddleware
from langchain.chat_models import BaseChatModel
from langchain.tools import tool
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage

from src.agents.agent_loader import AgentLoader
from src.agents.job_manager import JobManager, JobStatus
from src.agents.middlewares.shared import create_shared_middlewares
from src.utils.logging import get_logger

logger = get_logger(__name__)

RUN_TASK_TOOL_DESCRIPTION = """
Run a specific sub-agent to perform a long-running task in the background. 
Requires the 'agent_name' and a clear 'task_description'. 
Returns a job_id to track progress.
"""

CHECK_TASK_TOOL_DESCRIPTION = """
Check the status and result of a background task using its job_id. 
Use this to see if a sub-agent has finished its work.
"""

CANCEL_TASK_TOOL_DESCRIPTION = """
Terminate a running background task using its job_id if the task is no longer needed.
"""

LIST_TASKS_TOOL_DESCRIPTION = """
Retrieve a list of the 5 most recent running background tasks. 
Useful for monitoring active sub-agent operations.
"""

LOAD_AGENT_TOOL_DESCRIPTION = """
Search for available sub-agents by providing a query related to the task you need help with. 
Returns agent names and descriptions that match your needs.
If no query is provided, returns a list of all available agents,  use this sparingly as it can return a long list.
"""


class SubAgentMiddleware(AgentMiddleware):
    def __init__(
        self,
        *,
        model: str | BaseChatModel,
        agent_loader: AgentLoader,
        job_manager: JobManager,
        embeddings: Embeddings,
        run_prompt: str = RUN_TASK_TOOL_DESCRIPTION,
        check_prompt: str = CHECK_TASK_TOOL_DESCRIPTION,
        cancel_prompt: str = CANCEL_TASK_TOOL_DESCRIPTION,
        list_prompt: str = LIST_TASKS_TOOL_DESCRIPTION,
        load_prompt: str = LOAD_AGENT_TOOL_DESCRIPTION,
        call_config: dict = {},
    ):
        super().__init__()
        self.model = model
        self.agent_loader = agent_loader
        self.job_manager = job_manager
        self.embeddings = embeddings
        self.run_prompt = run_prompt
        self.check_prompt = check_prompt
        self.cancel_prompt = cancel_prompt
        self.list_prompt = list_prompt
        self.load_prompt = load_prompt
        self.call_config = call_config or {"recursion_limit": 70}

        all_agents = list(self.agent_loader.agents.values())
        agent_texts = [f"{agent.name}: {agent.description}" for agent in all_agents]

        self.agent_embeddings = self.embeddings.embed_documents(agent_texts)

        @tool(description=self.run_prompt)
        async def run_task(agent_name: str, task_description: str) -> str:
            job_id = self.job_manager.submit(
                agent_name,
                task_description,
                invoke_fn=self._invoke_agent,
            )
            return f"Background job {job_id} started."

        @tool(description=self.check_prompt)
        async def check_task(job_id: str) -> str:
            result = self.job_manager.get_job(job_id)

            if result:
                msg = f"Job {job_id} - Status: {result.status}"
                if result.status == JobStatus.FAILED or result.status == JobStatus.CANCELED:
                    msg += f" - Error: {result.error}"
                return msg
            return f"No job found with id {job_id}"

        @tool(description=self.cancel_prompt)
        async def cancel_task(job_id: str) -> str:
            result = self.job_manager.cancel(job_id)
            return f"Status: {result['status']} - Message: {result['message']}"

        @tool(description=self.list_prompt)
        async def list_tasks() -> str:
            jobs = self.job_manager.get_all()

            running_jobs = [job for job in jobs if job.status == JobStatus.RUNNING]
            running_jobs.sort(key=lambda x: x.created_at, reverse=True)
            running_jobs = running_jobs[:5]

            output = "Recent Running Jobs:\n"
            for job in running_jobs:
                output += f"- Job ID: {job.id}, Agent: {job.agent}, Task: {job.task_description}\n"
            return output

        @tool(description=self.load_prompt)
        async def load_agent(query: str | None) -> str:
            if query:
                agents = self._load_agent_by_query(query)
                if not agents:
                    return f"No agents found matching query: '{query}'"
                output = "Matching Agents:\n"
                for agent in agents:
                    output += (
                        f"- Name: {agent.name}, Description: {agent.description}\n"
                    )
                return output
            else:
                output = "Available Agents:\n"
                for agent in all_agents[:15]:
                    output += (
                        f"- Name: {agent.name}, Description: {agent.description}\n"
                    )
                return output

        self.tools = [run_task, check_task, cancel_task, list_tasks, load_agent]

    async def _invoke_agent(self, agent_name: str, task_description: str) -> Any:
        agent = self.agent_loader.create_agent_by_name(
            agent_name,
            middleware=create_shared_middlewares(model=self.model),
        )

        if agent is None:
            logger.warning("Sub-agent '%s' not found", agent_name)
            return f"Error: Agent '{agent_name}' not found."

        logger.info("Invoking sub-agent '%s'", agent_name)
        try:
            result = await agent.ainvoke(
                {"messages": [HumanMessage(content=task_description)]},
                config={
                    "recursion_limit": self.call_config.get("recursion_limit", 70),
                },
            )
        except Exception as e:
            logger.error("Error invoking sub-agent '%s': %s", agent_name, e)
            return f"Error invoking agent: '{agent_name}': {str(e)}"

        logger.info("Sub-agent '%s' finished", agent_name)
        return result

    def _load_agent_by_query(self, query: str) -> list:
        all_agents = list(self.agent_loader.agents.values())
        query_embedding = self.embeddings.embed_query(query)

        scored_agents = []
        q_vec = np.array(query_embedding)

        for i, agent_vec in enumerate(self.agent_embeddings):
            a_vec = np.array(agent_vec)
            score = np.dot(q_vec, a_vec)
            scored_agents.append((score, all_agents[i]))

        scored_agents.sort(key=lambda x: x[0], reverse=True)

        results = [agent for score, agent in scored_agents if score > 0.3][:3]
        logger.debug("Query '%s' matched %d agent(s)", query, len(results))
        return results
