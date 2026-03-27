from dataclasses import dataclass
from pathlib import Path

import yaml

AGENTS_DIR = Path(__file__).parent.parent.parent / "agents"


@dataclass(frozen=True)
class DiscoveredAgent:
    """Metadata discovered from an agent folder."""

    folder: Path
    name: str
    description: str
    instruction: str
    model: str
    provider: dict
    tools_path: Path


def discover_agents(agents_dir: Path = AGENTS_DIR) -> list[DiscoveredAgent]:
    """Discover agent metadata by scanning agents/* for info.yaml and tools.py."""
    discovered: list[DiscoveredAgent] = []

    if not agents_dir.is_dir():
        return discovered

    for folder in sorted(agents_dir.iterdir()):
        if not folder.is_dir():
            continue

        info_path = folder / "info.yaml"
        tools_path = folder / "tools.py"

        if not info_path.exists() or not tools_path.exists():
            continue

        try:
            with open(info_path, "r", encoding="utf-8") as f:
                info = yaml.safe_load(f) or {}
        except Exception:
            continue

        name = str(info.get("name", folder.name)).strip()
        if not name:
            continue

        discovered.append(
            DiscoveredAgent(
                folder=folder,
                name=name,
                description=str(info.get("description", "")).strip(),
                instruction=str(info.get("instruction", "")).strip(),
                model=str(info.get("model", "gpt-4o")).strip() or "gpt-4o",
                provider=info.get("provider") or {"type": "openai"},
                tools_path=tools_path,
            )
        )

    return discovered
