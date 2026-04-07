import json
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

from src.utils.logging import get_logger

AGENTS_DIR = Path(__file__).parent.parent.parent / "agents"

logger = get_logger(__name__)


@dataclass(frozen=True)
class Agent:
    id: str
    folder: Path
    name: str
    description: str
    prompt: str
    model: str
    provider: str
    requirements_path: Path
    tools_path: Path


def safe_load_json(file_path: Path) -> dict:
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.error("Error loading JSON from %s: %s", file_path, e)
        return {}


def create_agent(
    info: dict, folder: Path, tools_path: Path, requirements_path: Path
) -> Agent:
    return Agent(
        id=str(uuid.uuid4()),
        folder=folder,
        name=str(info.get("name", folder.name)).strip(),
        description=str(info.get("description", "")).strip(),
        prompt=str(info.get("prompt", "")).strip(),
        model=str(info.get("model", "gpt-4o")).strip(),
        provider=str(info.get("provider", "openai")).strip(),
        requirements_path=requirements_path,
        tools_path=tools_path,
    )


def install_requirements(requirements_path: Path) -> bool:
    if not requirements_path.exists() or requirements_path.stat().st_size == 0:
        return False

    try:
        subprocess.check_call(
            [sys.executable, "-m", "uv", "add", "-r", str(requirements_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        return True
    except subprocess.CalledProcessError as e:
        logger.error("Failed to install requirements from %s: %s", requirements_path, e)
        return False


def discover_agents(agents_dir: Path = AGENTS_DIR) -> list[Agent]:
    agents = []
    for folder in agents_dir.iterdir():
        if folder.is_dir():
            info = safe_load_json(folder / "info.json")
            tools_path = folder / "tools.py"
            requirements_path = folder / "requirements.txt"

            agent_name = info.get("name", folder.name)
            if any(agent.name == agent_name for agent in agents):
                logger.warning(
                    "Duplicate agent name '%s', skipping %s", agent_name, folder
                )
                continue

            if agent_name in ["controller", "synthesizer", "finish"]:
                logger.warning(
                    "Agent in %s has invalid name '%s', skipping", folder, agent_name
                )
                continue

            install_requirements(requirements_path)
            agent = create_agent(info, folder, tools_path, requirements_path)
            agents.append(agent)
            logger.info("Discovered agent '%s' from %s", agent.name, folder)

    logger.info("Total agents discovered: %d", len(agents))
    return agents
