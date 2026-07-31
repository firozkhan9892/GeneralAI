"""Tests for SkillSelector."""

from __future__ import annotations

import pytest

from app.kernel.skills.builtins import register_builtin_skills
from app.kernel.skills.executor import SkillSelector
from app.kernel.skills.models import Skill, SkillDescriptor


class TestSkillSelectorRegistration:
    """Tests for SkillSelector registration."""

    def test_register_skill(self) -> None:
        selector = SkillSelector()
        desc = SkillDescriptor(name="test_skill", description="A test skill")
        selector.register_skill(desc)
        assert selector.has_skill("test_skill") is True

    def test_register_skill_overwrite(self) -> None:
        selector = SkillSelector()
        desc1 = SkillDescriptor(name="test_skill", description="First")
        desc2 = SkillDescriptor(name="test_skill", description="Second")
        selector.register_skill(desc1)
        selector.register_skill(desc2)
        assert selector.has_skill("test_skill") is True

    def test_has_skill_not_registered(self) -> None:
        selector = SkillSelector()
        assert selector.has_skill("nonexistent") is False

    def test_list_skills_empty(self) -> None:
        selector = SkillSelector()
        assert selector.list_skills() == []

    def test_list_skills_after_registration(self) -> None:
        selector = SkillSelector()
        selector.register_skill(SkillDescriptor(name="skill_a"))
        selector.register_skill(SkillDescriptor(name="skill_b"))
        skills = selector.list_skills()
        assert "skill_a" in skills
        assert "skill_b" in skills
        assert len(skills) == 2


class TestSkillSelectorSelect:
    """Tests for SkillSelector.select."""

    @pytest.mark.asyncio
    async def test_select_existing_skill(self) -> None:
        selector = SkillSelector()
        selector.register_skill(
            SkillDescriptor(name="my_skill", description="My skill")
        )
        skill = await selector.select("my_skill", {"param": "value"})
        assert isinstance(skill, Skill)
        assert skill.name == "my_skill"
        assert skill.parameters == {"param": "value"}
        assert skill.descriptor.name == "my_skill"

    @pytest.mark.asyncio
    async def test_select_with_no_parameters(self) -> None:
        selector = SkillSelector()
        selector.register_skill(SkillDescriptor(name="no_params"))
        skill = await selector.select("no_params")
        assert isinstance(skill, Skill)
        assert skill.parameters == {}

    @pytest.mark.asyncio
    async def test_select_unregistered_skill(self) -> None:
        selector = SkillSelector()
        with pytest.raises(KeyError):
            await selector.select("nonexistent")

    @pytest.mark.asyncio
    async def test_select_resolves_required_tools(self) -> None:
        from app.kernel.skills.models import ToolRequirement

        selector = SkillSelector()
        selector.register_skill(
            SkillDescriptor(
                name="tool_dependent",
                required_tools=(
                    ToolRequirement(tool_name="calculator"),
                    ToolRequirement(tool_name="uuid"),
                ),
            )
        )
        skill = await selector.select("tool_dependent", {})
        assert "calculator" in skill.resolved_tools
        assert "uuid" in skill.resolved_tools


class TestSkillSelectorBuiltinRegistration:
    """Tests for registering built-in skills."""

    @pytest.mark.asyncio
    async def test_register_builtin_skills(self) -> None:
        selector = SkillSelector()
        register_builtin_skills(selector)
        assert selector.has_skill("echo")
        assert selector.has_skill("calculator")
        assert selector.has_skill("summarize")
        assert selector.has_skill("search_memory")
        assert selector.has_skill("respond")

    @pytest.mark.asyncio
    async def test_resolve_builtin_skill(self) -> None:
        selector = SkillSelector()
        register_builtin_skills(selector)
        skill = await selector.select("echo", {"message": "hello"})
        assert skill.name == "echo"
        assert skill.parameters == {"message": "hello"}
