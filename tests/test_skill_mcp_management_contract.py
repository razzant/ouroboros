"""The skill MCP facts exposed by management reads match their published shape."""

from pathlib import Path
from types import SimpleNamespace
import typing

from ouroboros.gateway.management_contracts import InstalledSkill, SkillMcpServerFact
from ouroboros.gateway.schema import validate_ingress
from ouroboros.skill_mcp import card_facts
from tests.test_gateway_parity import _js_typedef_fields


def test_installed_skill_mcp_facts_match_the_producer():
    skill = SimpleNamespace(
        name="browser",
        manifest=SimpleNamespace(mcp_servers=[{"id": "chrome", "title": "Chrome"}]),
    )
    settings = {"MCP_ENABLED": True, "MCP_SERVERS": [
        {"id": "chrome", "skill": "browser", "enabled": True},
    ]}
    status = {"chrome": {"tool_count": 2, "listed_at": "2026-11-06T00:00:00Z"}}

    row, = card_facts(skill, settings, status)
    assert row == {"id": "chrome", "title": "Chrome", "conflict": False,
                   "bound": True, "configured": True, "tool_count": 2, "listed": True, "error": "", "notes": ""}
    assert set(row) == set(typing.get_type_hints(SkillMcpServerFact))
    twin = Path(__file__).resolve().parents[1] / "web/modules/management_contract_types.js"
    assert _js_typedef_fields(twin.read_text(), "SkillMcpServerFact") == set(row)
    assert validate_ingress(row, SkillMcpServerFact) == []
    assert validate_ingress({**row, "listed": "yes"}, SkillMcpServerFact)
    assert validate_ingress({**row, "bound": "yes"}, SkillMcpServerFact)
    assert validate_ingress({**row, "notes": []}, SkillMcpServerFact)
    assert "mcp_servers" in InstalledSkill.__optional_keys__
    assert typing.get_args(typing.get_type_hints(InstalledSkill)["mcp_servers"]) == (SkillMcpServerFact,)
