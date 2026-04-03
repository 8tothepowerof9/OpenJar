from langchain.agents.middleware import AgentMiddleware
from langchain_core.tools import tool

from src.agent.artifact_manager import ArtifactManager

GET_ARTIFACT_TOOL_DESCRIPTION = """
Use this tool to get an artifact created by an agent. 
The input should be the artifact ID. The output will be the it's full content.
"""


class ArtifactMiddleware(AgentMiddleware):
    """
    Middleware to provide a tool for getting/retrieving artifacts created by agents.
    """

    def __init__(
        self,
        *,
        artifact_manager: ArtifactManager,
        tool_description: str = GET_ARTIFACT_TOOL_DESCRIPTION,
    ) -> None:
        super().__init__()
        self.artifact_manager = artifact_manager
        self.tool_description = tool_description

        @tool(description=self.tool_description)
        async def get_artifact(artifact_id: str) -> str:
            artifact = self.artifact_manager.get_artifact(artifact_id)
            if artifact is None:
                return f"No artifact found with ID: {artifact_id}"
            return artifact.full_content

        self.tools = [get_artifact]
