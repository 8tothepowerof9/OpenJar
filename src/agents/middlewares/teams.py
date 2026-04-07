import operator
from typing import Annotated, Optional, TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.chat_models import BaseChatModel
from langchain.messages import HumanMessage
from langchain.tools import tool
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from src.agents.agent_loader import AgentLoader
from src.agents.job_manager import JobManager
from src.agents.middlewares.shared import create_shared_middlewares
from src.utils.logging import get_logger

logger = get_logger(__name__)

RUN_TEAM_TOOL_DESCRIPTION = """
Run a team of specialized agents to collaboratively complete a task.
Agents contribute to a shared blackboard. Returns a job ID for the background task.
"""


class BlackBoardSchema(TypedDict):
    user_request: str
    shared_memory: dict[str, str]
    available_agents: list[str]
    called_agents: Annotated[list[str], operator.add]
    next_agent: Optional[str]
    next_prompt: Optional[str]
    round: int
    final_result: Optional[str]


class ControllerDecision(BaseModel):
    next_agent: str = Field(
        description="The name of the next agent to call or 'finish' if complete."
    )
    next_prompt: str = Field(
        description="Specific instructions for the next agent based on what has been done so far."
    )


class AgentTeamMiddleware(AgentMiddleware):
    def __init__(
        self,
        *,
        model: str | BaseChatModel,
        agent_loader: AgentLoader,
        job_manager: JobManager,
        run_prompt: str = RUN_TEAM_TOOL_DESCRIPTION,
        call_config: dict = {},
    ):
        super().__init__()
        self.model = model
        self.agent_loader = agent_loader
        self.job_manager = job_manager
        self.run_prompt = run_prompt
        self.call_config = call_config or {"recursion_limit": 30}
        self.max_rounds = 10

        @tool(description=self.run_prompt)
        async def run_agent_team(
            member_names: list[str], task_description: str, max_rounds: int = 10
        ) -> str:
            logger.info("Starting agent team with members: %s", member_names)
            self.max_rounds = max_rounds
            workflow = StateGraph(BlackBoardSchema)
            workflow.add_node("controller", self._controller_node)
            workflow.add_node("synthesizer", self._synthesizer_node)

            for member in member_names:
                workflow.add_node(member, self._create_specialist_node(member))
                workflow.add_edge(member, "controller")

            def router(state: BlackBoardSchema):
                if state.get("next_agent", "").lower() == "finish":
                    return "synthesizer"
                return (
                    state["next_agent"] if state["next_agent"] in member_names else END
                )

            workflow.add_conditional_edges("controller", router)
            workflow.add_edge("synthesizer", END)
            workflow.set_entry_point("controller")

            graph = workflow.compile()

            initial_state: BlackBoardSchema = {
                "user_request": task_description,
                "shared_memory": {},
                "available_agents": member_names,
                "called_agents": [],
                "next_agent": None,
                "next_prompt": None,
                "round": 0,
                "final_result": None,
            }

            async def invoke(_name: str, _desc: str):
                return await graph.ainvoke(initial_state)

            job_id = self.job_manager.submit(
                agent="agent_team",
                task_description=task_description,
                invoke_fn=invoke,
            )
            return f"Background job {job_id} started."

        self.tools = [run_agent_team]

    async def _synthesizer_node(self, state: BlackBoardSchema):
        logger.info("Synthesizing final result...")
        findings = "\n\n".join(
            [f"## {k}\n{v}" for k, v in state["shared_memory"].items()]
        )

        sys_prompt = (
            "You are a synthesizer. Review the user request and agent findings to "
            "create a comprehensive, well-formatted final response."
        )
        synthesizer = create_agent(self.model, system_prompt=sys_prompt, tools=[])
        msg = HumanMessage(
            content=f"Request: {state['user_request']}\n\nFindings:\n{findings}"
        )

        result = await synthesizer.ainvoke({"messages": [msg]})
        return {"final_result": result["messages"][-1].content}

    def _create_specialist_node(self, agent_name: str):
        async def specialist_node(state: BlackBoardSchema):
            blueprint = self.agent_loader.get_agent_blueprint_by_name(agent_name)
            if not blueprint:
                raise ValueError(f"Agent '{agent_name}' not found")

            model, tools, prompt, name, _ = blueprint
            sys_prompt = (
                f"{prompt}\n\nYou are part of a team. Perform your role based on the prompt "
                "and shared context. Provide a detailed markdown report of your findings."
            )

            agent = create_agent(
                model,
                tools=tools,
                system_prompt=sys_prompt,
                middleware=create_shared_middlewares(self.model),
            )

            context = "\n\n".join(
                [f"[{k}]: {v[:1500]}" for k, v in state["shared_memory"].items()]
            )
            msg = HumanMessage(
                content=f"Task: {state['next_prompt']}\n\nContext:\n{context}"
            )

            result = await agent.ainvoke(
                {"messages": [msg]},
                config={"recursion_limit": self.call_config.get("recursion_limit", 30)},
            )

            report_content = result["messages"][-1].content
            updated_memory = state["shared_memory"].copy()
            updated_memory[name] = report_content

            logger.info("Specialist '%s' completed their task.", name)
            return {"shared_memory": updated_memory}

        return specialist_node

    async def _controller_node(self, state: BlackBoardSchema):
        current_round = state["round"] + 1
        if current_round >= self.max_rounds:
            return {"next_agent": "finish", "round": current_round}

        sys_prompt = (
            "You are the central controller. Analyze which agents have already contributed "
            "and decide which specialist should run next to fulfill the user request. "
            "If the task is complete, return 'finish'."
        )

        controller = create_agent(
            self.model,
            system_prompt=sys_prompt,
            tools=[],
            response_format=ControllerDecision,
        )

        agents_finished = ", ".join(state["shared_memory"].keys()) or "None"

        messages = HumanMessage(
            content=(
                f"User Request: {state['user_request']}\n"
                f"Available Agents: {', '.join(state['available_agents'])}\n"
                f"Agents who have completed their work: {agents_finished}\n"
                f"Round: {current_round}/{self.max_rounds}"
            )
        )

        result = await controller.ainvoke({"messages": [messages]})
        decision: ControllerDecision = result["structured_response"]

        return {
            "next_agent": decision.next_agent,
            "called_agents": (
                [decision.next_agent] if decision.next_agent.lower() != "finish" else []
            ),
            "round": current_round,
            "next_prompt": decision.next_prompt,
        }
