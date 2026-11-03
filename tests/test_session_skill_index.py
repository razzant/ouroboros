"""Installed instruction discovery reaches project sessions without changing their work orders."""

import json
from types import SimpleNamespace

import pytest

from ouroboros import delegate_custody as custody
from ouroboros.skill_catalogue import session_skill_index
from ouroboros.skill_loader import save_enabled
from ouroboros.subagent_work_order import compile_external_work_order, work_order_fingerprint
from tests._delegated_transport_shared import (  # noqa: F401 -- autouse transport fixture
    _owned_gateway_uses_each_test_transport,
    _select_transport_access,
)
from tests.test_skill_model_experience import _install_skill


def test_index_includes_reviewed_external_and_accepted_native_instructions(tmp_path, monkeypatch):
    from ouroboros.skill_builtin import install_builtin_skill

    monkeypatch.setenv("OUROBOROS_TRUST_NATIVE_SEEDED_SKILLS", "true")
    monkeypatch.setattr("ouroboros.config.SETTINGS_PATH", tmp_path / "settings.json")
    _install_skill(tmp_path, "external_guide")
    seeds = tmp_path / "bundled"
    (seeds / "native_guide").mkdir(parents=True)
    (seeds / "native_guide/SKILL.md").write_text(
        "---\nname: native_guide\nversion: 1.0.0\ntype: instruction\n"
        "description: A native guide.\nwhen_to_use: Before drawing.\n---\nPRIVATE BODY\n", encoding="utf-8")
    assert install_builtin_skill("native_guide", drive_root=tmp_path, seed_dir=seeds)["ok"]
    save_enabled(tmp_path, "native_guide", True, actor="owner_ui")
    index = session_skill_index(SimpleNamespace(drive_root=tmp_path))
    assert "HOST SKILL REFERENCES (read-only)" in index and "never modify host skills" in index
    assert "files they reference" in index and "untrusted metadata" in index
    assert "- external_guide — does a thing — Trigger: user asks for the thing — read: " in index
    assert "- native_guide — A native guide. — Trigger: Before drawing. — read: " in index
    for bucket, name in (("external", "external_guide"), ("native", "native_guide")):
        assert str((tmp_path / "skills" / bucket / name / "SKILL.md").resolve()) in index
    assert "PRIVATE BODY" not in index


def test_index_excludes_off_unaccepted_stale_and_non_instruction_skills(tmp_path):
    from ouroboros.skill_loader import compute_content_hash

    for name in ("off", "unaccepted", "stale", "script"):
        _install_skill(tmp_path, name)
    save_enabled(tmp_path, "off", False, actor="owner_ui")
    (tmp_path / "state/skills/unaccepted/review.json").unlink()
    (tmp_path / "skills/external/stale/SKILL.md").write_text(
        (tmp_path / "skills/external/stale/SKILL.md").read_text() + "changed\n", encoding="utf-8")
    script = tmp_path / "skills/external/script"
    (script / "SKILL.md").write_text(
        "---\nname: script\nversion: 1.0.0\ntype: script\ndescription: Script.\nscripts: [run.py]\n---\n",
        encoding="utf-8")
    (script / "run.py").write_text("print('fixture')\n", encoding="utf-8")
    (tmp_path / "state/skills/script/review.json").write_text(json.dumps({
        "status": "pass", "content_hash": compute_content_hash(script),
    }), encoding="utf-8")
    assert session_skill_index(SimpleNamespace(drive_root=tmp_path)) == ""


def test_index_bounds_metadata_like_context_without_copying_bodies(tmp_path):
    from ouroboros.skill_loader import compute_content_hash

    _install_skill(tmp_path, "long")
    root = tmp_path / "skills/external/long"
    text = (root / "SKILL.md").read_text().replace("does a thing", "x" * 300).replace(
        "user asks for the thing", "y" * 270)
    (root / "SKILL.md").write_text(text, encoding="utf-8")
    (tmp_path / "state/skills/long/review.json").write_text(json.dumps({
        "status": "pass", "content_hash": compute_content_hash(root),
    }), encoding="utf-8")
    index = session_skill_index(SimpleNamespace(drive_root=tmp_path))
    assert "x" * 260 + " [... 40 chars omitted]" in index and "x" * 261 not in index
    assert "y" * 260 + " [... 10 chars omitted]" in index and "y" * 261 not in index
    assert len([line for line in index.splitlines() if line.startswith("- ")]) == 1


@pytest.mark.parametrize("compiled", [True, False])
@pytest.mark.parametrize("input_sources", ["shared", "declared"])
def test_genesis_request_gets_index_only_for_shared_inputs(tmp_path, monkeypatch, compiled, input_sources):
    from tests.test_session_design_navigation import _start

    request, row, task, ctx, _live, _prepared = _start(
        tmp_path, monkeypatch, compiled=compiled, kind="foreign_copy", input_sources=input_sources,
        prepare=lambda ctx: _install_skill(ctx.drive_root, "external_guide"))
    assert ("HOST SKILL REFERENCES" in request["instructions"]) is (input_sources == "shared")
    assert "DESIGN NAVIGATION DELIVERY" not in request["instructions"]
    assert row["request"] == request
    canonical, fingerprint = compile_external_work_order(task), work_order_fingerprint(task)
    if compiled:
        assert request["prompt"] == canonical
    assert "HOST SKILL REFERENCES" not in canonical and "external_guide" not in canonical
    save_enabled(ctx.drive_root, "external_guide", False, actor="owner_ui")
    assert compile_external_work_order(task) == canonical and work_order_fingerprint(task) == fingerprint
    assert custody.invocation_record(ctx.drive_root, row["invocation_id"])["request"] == request


@pytest.mark.parametrize("carrier", ["context", "task_metadata"])
def test_forked_task_drive_still_gets_the_installed_index(tmp_path, monkeypatch, carrier):
    from ouroboros.headless import prepare_task_drive
    from tests.test_session_design_navigation import _start

    def prepare(ctx):
        canonical = ctx.drive_root
        _install_skill(canonical, "external_guide")
        ctx.drive_root = prepare_task_drive(canonical, "review-child", "forked")
        assert not (ctx.drive_root / "skills").exists()
        if carrier == "context":
            ctx.budget_drive_root = str(canonical)
        else:
            ctx.task_metadata["budget_drive_root"] = str(canonical)

    request, *_rest = _start(tmp_path, monkeypatch, compiled=False, kind="foreign_copy",
                             input_sources="shared", prepare=prepare)
    assert "external_guide" in request["instructions"]


def test_unknown_start_retry_replays_the_index_byte_for_byte(tmp_path, monkeypatch):
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from tests.test_delegate_continuation import _start

    _install_skill(tmp_path, "external_guide")
    _select_transport_access(monkeypatch, "readonly")
    original, unknown, _ctx, _calls = _start(
        tmp_path, monkeypatch, acting=False, run_id="skill-session", prompt="Draw the page",
        start_error=ClaudexorUnavailable("daemon_unreachable", "connection reset", status_code=503))
    assert "external_guide" in original["instructions"]
    save_enabled(tmp_path, "external_guide", False, actor="owner_ui")
    retry, payload, _ctx, _calls = _start(
        tmp_path, monkeypatch, acting=False, run_id="skill-session", prompt="Draw the page",
        start_kwargs={"retry_of": unknown["pending_invocation_id"]})
    assert payload["status"] == "started" and retry == original
