"""Architecture preserves the skill boundary's contract-bearing facts."""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _book(chapter):
    return (ROOT / "docs/architecture" / chapter).read_text(encoding="utf-8")


@pytest.mark.parametrize("fact,obsolete", [
    ("Local delete unloads and removes payload and the entire state directory under the lifecycle lock",
     "Local delete removes payload and records the uninstall tombstone"),
    ("without the configured GitHub token (this computer's git credentials still apply)", "HTTPS unauthenticated"),
    ("authentication refusals once with it alone, without a credential helper", "retrying only auth refusals once with the configured token"),
    ("`pending_update` names the candidate release; payload/lifecycle snapshots are retained separately",
     "retain payload/lifecycle as `pending_update`"),
    ("Preflight failure persists as PENDING, never advisory-overridable BLOCKERS", ""),
])
def test_skill_lifecycle_facts_match_the_current_contract(fact, obsolete):
    text = _book("13-external-skills-layer.md")
    assert fact in text
    assert not obsolete or obsolete not in text
    with pytest.raises(AssertionError):
        assert fact in text.replace(fact, "")


def test_frozen_plugin_row_names_the_surface_and_host_mediated_permissions():
    text = _book("11-frozen-contracts-v1.md")
    row = next(line for line in text.splitlines() if line.startswith("| `PluginAPI`"))
    module = ast.parse((ROOT / "ouroboros/contracts/plugin_api.py").read_text(encoding="utf-8"))
    api = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "PluginAPI")
    count = sum(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in api.body)
    assert count == 18 and f"{count}-method extension surface" in row
    for name in ("companion_process", "supervised_task", "subscribe_event", "inject_chat", "presence",
                 "notify_owner", "context_snapshot", "page_module", "realtime_provider", "live_conversation", "call_peer"):
        assert f"`{name}`" in row
    assert "every registry entry satisfies `ToolEntryProtocol`" in text


def test_host_relay_and_author_protocol_pointers_are_concrete():
    text = _book("12-host-service-companions-and-chat-ids.md")
    relay = next(line for line in text.splitlines() if "host_service_ws_relay_dropped" in line)
    assert "durable" in relay and "logs/events.jsonl" in relay
    assert "author protocol" not in text
    assert "CREATING_SKILLS.md` `inject_chat`" in text
    author = (ROOT / "docs/CREATING_SKILLS.md").read_text(encoding="utf-8")
    for heading in ("Presence continuation", "Reporting actual Presence delivery"):
        assert f'CREATING_SKILLS.md` “{heading}”' in text
        assert f"#### {heading}" in author
