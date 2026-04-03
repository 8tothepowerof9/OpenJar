from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.tools import tool

from src.agent.middlewares.base import SystemPromptMiddleware


@dataclass
class Skill:
    """A registered skill with a name, summary, and full instruction content."""

    name: str
    summary: str
    content: str


class SkillRegistry:
    """Central registry of skills that can be loaded on-demand by the agent."""

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def register(self, name: str, summary: str, content: str) -> None:
        """Register a skill. Overwrites if already registered."""
        self._skills[name] = Skill(name=name, summary=summary, content=content)

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    @property
    def skill_names(self) -> list[str]:
        return list(self._skills.keys())

    def format_skill_list(self) -> str:
        if not self._skills:
            return "No skills available."
        lines = []
        for skill in self._skills.values():
            lines.append(f"- {skill.name}: {skill.summary}")
        return "\n".join(lines)


SKILL_SYSTEM_PROMPT = """\
## load_skill

You have access to specialized toolkits. Before using a toolkit for the first time, \
call load_skill to get detailed usage instructions. You only need to load each skill once per conversation.

Available skills:
{skill_list}

Workflow:
1. Identify which toolkit you need for the task.
2. Call load_skill(skill_name) to learn how to use it.
3. Follow the returned instructions when using the toolkit's tools.
"""


class SkillMiddleware(SystemPromptMiddleware):
    """Middleware that provides a load_skill tool for on-demand prompt loading.

    Instead of injecting all toolkit guidance into the system prompt upfront,
    this middleware lets the agent pull in detailed instructions only when needed.
    """

    def __init__(self, registry: SkillRegistry) -> None:
        super().__init__()
        self.registry = registry
        self.system_prompt = SKILL_SYSTEM_PROMPT.format(
            skill_list=registry.format_skill_list()
        )

        @tool(
            description=(
                "Load detailed usage instructions for a toolkit before using it. "
                "Call this once per skill before using the toolkit's tools."
            )
        )
        def load_skill(skill_name: str) -> str:
            """Load usage instructions for a toolkit.

            Args:
                skill_name: Name of the skill to load.

            Returns:
                Detailed instructions for using the toolkit.
            """
            skill = self.registry.get(skill_name)
            if skill is None:
                available = ", ".join(self.registry.skill_names)
                return f"Unknown skill '{skill_name}'. Available skills: {available}"
            return skill.content

        self.tools = [load_skill]
