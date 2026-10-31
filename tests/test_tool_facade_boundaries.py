"""New tool forms preserve dispatch authority and exact result decisions."""
from copy import deepcopy

import pytest

from ouroboros.tools.registry import ToolContext, ToolRegistry


@pytest.fixture
def registry(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_a, **_k: (True, ""))
    monkeypatch.setattr("ouroboros.tools.github.github_cli_configured", lambda: True)
    result = ToolRegistry(repo_dir=repo, drive_root=data)
    result.set_context(ToolContext(repo_dir=repo, drive_root=data))
    return result






@pytest.mark.parametrize("withheld,requested,args", [
    ("commit_reviewed", "vcs_commit_reviewed", {"commit_message": "no"}),
    ("vcs_commit_reviewed", "commit_reviewed", {"commit_message": "no"}),
])
def test_disabled_old_or_new_operation_cannot_be_bypassed(registry, withheld, requested, args):
    registry._ctx.task_contract = {"disabled_tools": [withheld]}
    result = registry.execute_result(requested, args)
    assert result.status == "blocked" and result.code == "RESOURCE_CONSTRAINT_BLOCKED"




def test_duplicate_commit_hidden_but_still_callable_without_new_grants(registry, monkeypatch):
    from dataclasses import replace

    from ouroboros import body_candidate

    entries = registry._entries
    old, new = entries["vcs_commit_reviewed"], entries["commit_reviewed"]
    assert old.handler is new.handler
    assert old.schema["parameters"] == new.schema["parameters"]
    assert old.is_code_tool == new.is_code_tool and old.safety_policy == new.safety_policy
    assert "vcs_commit_reviewed" not in registry.available_tools()
    assert registry.get_schema_by_name("vcs_commit_reviewed") is None
    assert "commit_reviewed" in registry.available_tools()
    # Test the actual entry dispatcher, without performing a commit.
    seen = []
    entry = replace(old, handler=lambda ctx, **args: seen.append(args) or "OK: dispatched old spelling")
    monkeypatch.setattr(registry, "_entries", {**entries, old.name: entry})
    monkeypatch.setattr(body_candidate, "authoring_seam", lambda *_a, **_k: None)
    assert "dispatched old spelling" in registry.execute(old.name, {"commit_message": "fixture"})
    assert seen[0]["commit_message"] == "fixture"


@pytest.mark.parametrize("spelling", ["vcs_commit_reviewed", "commit_reviewed"])
def test_sealed_presence_keeps_exact_selected_commit_name_and_binding(registry, spelling):
    from ouroboros.presence_authority import PresenceCapabilityCeiling, PresenceToolGrant, presence_ceiling_payload
    from ouroboros.presence_capabilities import PresenceArgumentBinding
    from ouroboros.tools.registry_core import _presence_bound_args

    ceiling = PresenceCapabilityCeiling(
        skill_name="behavior", skill_content_hash="a" * 64, profile_fingerprint="b" * 64,
        state_fingerprint="c" * 64, selection_fingerprint="d" * 64, model_slot="main", inline_max_rounds=3,
        tool_grants=(PresenceToolGrant(spelling, (PresenceArgumentBinding(("scope",), "static", static_value="host-scope"),)),),
        resource_grants=(), digest="0" * 64)
    payload = presence_ceiling_payload(ceiling)
    registry._ctx.task_contract = {"capability_ceiling": payload}
    original = deepcopy(payload)
    other = "commit_reviewed" if spelling == "vcs_commit_reviewed" else "vcs_commit_reviewed"
    # This fixture has a sealed ceiling but no Presence-origin task metadata.
    automatic = {"finish_task", "presence_cancel_work"}
    assert set(registry.available_tools()) == {spelling} | automatic
    assert registry.get_schema_by_name(spelling) is not None
    assert registry.get_schema_by_name(other) is None
    visible = {s["function"]["name"] for s in registry.schemas(core_only=True)}
    assert spelling in visible and other not in visible
    assert visible <= {spelling} | automatic
    args, error = _presence_bound_args(registry._ctx, spelling, {"scope": "model-scope"})
    assert not error and args["scope"] == "host-scope"
    assert _presence_bound_args(registry._ctx, other, {})[1]
    assert registry._ctx.task_contract["capability_ceiling"] == original


def test_registered_short_result_reads_do_not_dispose_or_consume_children(registry):
    import json

    from ouroboros.task_status import load_effective_task_result
    from ouroboros.task_tree_ledger import tree_ledger_rows
    from tests._delivery_candidate_shared import write_child

    ctx = registry._ctx
    ctx.task_id = "parent1"
    ctx.task_metadata = {"root_task_id": "parent1", "budget_drive_root": str(ctx.drive_root)}
    write_child(ctx.drive_root)
    before = load_effective_task_result(ctx.drive_root, "child1")
    for view in ("answer", "summary", "evidence"):
        response = registry.execute_result("get_task_result", {"task_id": "child1", "view": view})
        assert response.status == "ok", response.text
        observed = json.loads(response.text)
        assert observed["view"] == view and observed["child_result_sha256"]
        assert tree_ledger_rows("parent1", data_root=ctx.drive_root) == []
        assert load_effective_task_result(ctx.drive_root, "child1") == before




@pytest.mark.parametrize("spelling", ["ocr_pdf", "extract_pdf_text"])
def test_pdf_exact_alias_preserves_disabled_and_sealed_grants(registry, spelling):
    from ouroboros.presence_authority import PresenceCapabilityCeiling, PresenceToolGrant, presence_ceiling_payload
    from ouroboros.presence_capabilities import PresenceArgumentBinding
    from ouroboros.tools.registry_core import _presence_bound_args

    old, new = registry._entries["ocr_pdf"], registry._entries["extract_pdf_text"]
    assert old.handler is new.handler and old.schema["parameters"] == new.schema["parameters"]
    assert "ocr_pdf" not in registry.available_tools() and "extract_pdf_text" in registry.available_tools()
    other = "extract_pdf_text" if spelling == "ocr_pdf" else "ocr_pdf"
    registry._ctx.task_contract = {"disabled_tools": [spelling]}
    denied = registry.execute_result(other, {"path": "not-read.pdf"})
    assert denied.status == "blocked" and denied.code == "RESOURCE_CONSTRAINT_BLOCKED"
    ceiling = PresenceCapabilityCeiling(
        skill_name="pdf-reader", skill_content_hash="a" * 64, profile_fingerprint="b" * 64,
        state_fingerprint="c" * 64, selection_fingerprint="d" * 64, model_slot="main", inline_max_rounds=3,
        tool_grants=(PresenceToolGrant(spelling, (PresenceArgumentBinding(("path",), "static", static_value="one.pdf"),)),),
        resource_grants=(), digest="0" * 64)
    registry._ctx.task_contract = {"capability_ceiling": presence_ceiling_payload(ceiling)}
    original = deepcopy(registry._ctx.task_contract)
    assert registry.get_schema_by_name(spelling) is not None
    assert registry.get_schema_by_name(other) is None
    args, error = _presence_bound_args(registry._ctx, spelling, {"path": "different.pdf"})
    assert not error and args["path"] == "one.pdf"
    assert _presence_bound_args(registry._ctx, other, {})[1]
    assert registry._ctx.task_contract == original
