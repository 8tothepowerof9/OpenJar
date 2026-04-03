import uuid
from dataclasses import dataclass
from pathlib import Path

import yaml

AGENTS_DIR = Path(__file__).parent.parent.parent / "agents"


def _normalize_provider(provider: object) -> dict:
    """Return a provider dict with a guaranteed string 'type' field."""
    if isinstance(provider, dict):
        provider_type = provider.get("type")
        if isinstance(provider_type, str) and provider_type.strip():
            normalized = dict(provider)
            normalized["type"] = provider_type.strip()
            return normalized

    return {"type": "openai"}


@dataclass(frozen=True)
class DiscoveredAgent:
    """Metadata discovered from an agent folder."""

    id: str
    folder: Path
    name: str
    description: str
    instruction: str
    model: str
    provider: dict
    tools_path: Path


@dataclass(frozen=True)
class DiscoveredMultiAgent:
    """Metadata discovered from a multiagent folder"""

    id: str
    folder: Path
    name: str
    description: str
    instruction: str
    model: str
    provider: dict
    tools_path: Path
    sub_agents: list[DiscoveredAgent]


def _safe_load_yaml_dict(path: Path) -> dict | None:
    """Load a YAML file and return a dict, or None if loading fails."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return None

    return data if isinstance(data, dict) else {}


def _build_agent(info: dict, folder: Path, tools_path: Path) -> DiscoveredAgent:
    """Build a DiscoveredAgent from parsed metadata."""
    return DiscoveredAgent(
        id=uuid.uuid4().hex,
        folder=folder,
        name=str(info.get("name", folder.name)).strip(),
        description=str(info.get("description", "")).strip(),
        instruction=str(info.get("instruction", "")).strip(),
        model=str(info.get("model", "gpt-4o")).strip() or "gpt-4o",
        provider=_normalize_provider(info.get("provider")),
        tools_path=tools_path,
    )


def _discover_sub_agents(folder: Path) -> list[DiscoveredAgent]:
    """Discover immediate child subagents within an agent folder."""
    sub_agents: list[DiscoveredAgent] = []

    for sub_item in folder.iterdir():
        if not sub_item.is_dir():
            continue

        sub_info = sub_item / "info.yaml"
        sub_tools = sub_item / "tools.py"
        if not sub_info.exists() or not sub_tools.exists():
            continue

        s_info = _safe_load_yaml_dict(sub_info)
        if s_info is None:
            continue

        sub_agents.append(_build_agent(s_info, sub_item, sub_tools))

    return sub_agents


def discover_agents(
    agents_dir: Path = AGENTS_DIR,
) -> list[DiscoveredAgent | DiscoveredMultiAgent]:
    """Discover agent metadata by scanning agents/* for info.yaml and tools.py."""
    discovered: list[DiscoveredAgent | DiscoveredMultiAgent] = []

    if not agents_dir.is_dir():
        return discovered

    for folder in sorted(agents_dir.iterdir()):
        if not folder.is_dir():
            continue

        info_path = folder / "info.yaml"
        tools_path = folder / "tools.py"

        if not info_path.exists() or not tools_path.exists():
            continue

        info = _safe_load_yaml_dict(info_path)
        if info is None:
            continue

        name = str(info.get("name", folder.name)).strip()
        if not name:
            continue

        sub_agents = _discover_sub_agents(folder)

        if sub_agents:
            discovered.append(
                DiscoveredMultiAgent(
                    id=uuid.uuid4().hex,
                    folder=folder,
                    name=name,
                    description=str(info.get("description", "")).strip(),
                    instruction=str(info.get("instruction", "")).strip(),
                    model=str(info.get("model", "gpt-4o")).strip() or "gpt-4o",
                    provider=_normalize_provider(info.get("provider")),
                    tools_path=tools_path,
                    sub_agents=sub_agents,
                )
            )
        else:
            discovered.append(_build_agent(info, folder, tools_path))

    seen: set[str] = set()
    duplicates: list[str] = []
    for agent in discovered:
        if agent.name in seen:
            duplicates.append(agent.name)
        else:
            seen.add(agent.name)

    if duplicates:
        raise ValueError(
            f"Duplicate agent names discovered: {', '.join(sorted(set(duplicates)))}. "
            "Each agent folder must resolve to a unique name field in its info.yaml."
        )

    return discovered
