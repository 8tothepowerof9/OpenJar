import uuid
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

LENGTH_THRESHOLD = 1500

SUMMARIZE_PROMPT = """\
You are a summarization agent. You will receive a task description and its result content.
Personality: {personality}

Your job:
1. Condense the content into a very concise and dense summary containing a few sentences.
2. Preserve all notable findings, key data points, specific names, numbers, and conclusions.
3. Retain any warnings, errors, or actionable recommendations verbatim.
4. Use bullet points for multiple distinct findings.
5. Start with a one-sentence overview, then list the key details.

Do NOT add interpretation or commentary beyond what the content states.
Respond like your are notifying a human collaborator that the task has completed and here are the key results.
Example: "Sir, task X has completed. Here are the key results: [summary]. Would you like me to provide the full content of the result?"
Respond according to your given personality and style, but ensure the summary is clear and informative.
\
"""


@dataclass
class Artifact:
    id: str
    summary: str
    full_content: str

    def __str__(self) -> str:
        return f"Artifact {self.id}: {self.summary}"


class ArtifactManager:
    def __init__(
        self,
        personality: str,
        summarizer: str | BaseChatModel,
        system_prompt: str = SUMMARIZE_PROMPT,
    ):
        """
        Initializes the ArtifactManager with a summarization agent and an empty artifact store.

        Args:
            personality (str): The personality to inject into the summarization agent's system prompt.
            summarizer (str | BaseChatModel): The model or model name to use for summarization.
            system_prompt (str, optional): The system prompt for the summarization agent. Defaults to SUMMARIZE_PROMPT.
        """
        self.summarizer = create_agent(
            model=summarizer,
            tools=[],
            system_prompt=system_prompt.format(personality=personality),
        )
        self.artifacts: dict[str, Artifact] = {}

    def add_artifact(self, content: str, task_description: str) -> Artifact:
        """
        Add a new artifact to the manager.

        Args:
            content (str): The full content of the artifact to be added.
            task_description (str): A description of the task associated with the artifact.


        Returns:
            Artifact: The added artifact.
        """
        id = uuid.uuid4().hex[:8]
        summary = self._summarize(content, task_description)
        artifact = Artifact(id=id, summary=summary, full_content=content)
        self.artifacts[id] = artifact
        return artifact

    async def aadd_artifact(self, content: str, task_description: str) -> Artifact:
        """
        Asynchronously add a new artifact to the manager.

        Args:
            content (str): The full content of the artifact to be added.
            task_description (str): A description of the task associated with the artifact.

        Returns:
            Artifact: The added artifact.
        """
        id = uuid.uuid4().hex[:8]
        summary = await self._asummarize(content, task_description)
        artifact = Artifact(id=id, summary=summary, full_content=content)
        self.artifacts[id] = artifact
        return artifact

    def get_artifact(self, id: str) -> Artifact | None:
        return self.artifacts.get(id)

    def _summarize(self, content: str, task_description: str) -> str:
        content = f"""
        # Task: {task_description}
        # Result content to summarize:
        {content}
        """

        result = self.summarizer.invoke({"messages": [HumanMessage(content=content)]})
        return result["messages"][-1].content

    async def _asummarize(self, content: str, task_description: str) -> str:
        content = f"""
        # Task: {task_description}
        # Result content to summarize:
        {content}
        """

        result = await self.summarizer.ainvoke(
            {"messages": [HumanMessage(content=content)]}
        )
        return result["messages"][-1].content
