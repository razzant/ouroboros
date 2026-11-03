"""An installed development task reviews and executes the exact revised payload."""

from __future__ import annotations

import json

import pytest

from ouroboros import dialogue_store, extension_loader
from ouroboros.contracts.task_constraint import TaskConstraint
from ouroboros.skill_loader import (
    compute_content_hash, find_skill, load_enabled, load_review_state, save_enabled,
)
from ouroboros.skill_repair_admission import record_repair_admission
from ouroboros.skill_review_runner import run_skill_review_lifecycle_blocking
from tests._extension_loader_shared import _write_ext_skill
from tests._extension_loader_shared import _clear_loader_state  # noqa: F401
from tests.test_d16_skill_payload_authority import _registry
from tests._skill_review_shared import _mark_self_authored


@pytest.fixture
def development(tmp_path, monkeypatch):
    registry, repo, drive = _registry(tmp_path, monkeypatch, mode="advanced")
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(tmp_path / "unused"))
    monkeypatch.setenv("OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS", "false")
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    calls = []

    # The one wave over the golden pool with an offline seam: every seat answers the skill question clean
    # (``[]`` + NO_FINDINGS, contract A); ``calls`` holds one entry per WAVE (its packet seat ``t1`` is in every panel).
    import ouroboros.review_substrate as substrate
    from tests import _contributor_packet_shared as shared
    from tests.review_pool_rosters import set_review_pool

    set_review_pool(monkeypatch, shared.golden_pool())

    def answer(request, slot, actor, *, retry_state, pending_invocation_checkpoint):
        if slot.slot_id == "t1":
            calls.append({"retry_key": request.retry_key, "surface": request.surface})
        actor.raw_text = "[]\nNO_FINDINGS"
        return actor

    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", shared.golden_physical_seam([], answer=answer))
    registry._ctx.task_id = "develop-demo"
    registry._ctx.current_chat_id = 42
    registry._ctx.task_constraint = TaskConstraint(
        skill_name="demo", payload_root="skills/external/demo", allow_enable=False,
    )
    return registry, repo, drive, calls


def _admit(registry, drive, payload):
    from dataclasses import asdict
    from ouroboros.project_dialogue import build_owner_message_ref
    from ouroboros.task_results import write_task_result
    from ouroboros.utils import utc_now_iso

    _mark_self_authored(payload, drive)
    text = "Repair and run demo; leave the repaired installation working."
    ref = build_owner_message_ref(chat_id=42, client_message_id="repair-demo",
                                  ts=utc_now_iso(), text=text)
    dialogue_store.append(drive, "chat", {**ref, "direction": "in", "text": text, "source": "web"})
    write_task_result(drive, registry._ctx.task_id, "running", chat_id=42,
                      task_constraint=asdict(registry._ctx.task_constraint),
                      origin_message_ref=ref, origin_message_text=text)
    record_repair_admission(drive, "demo", task_id=registry._ctx.task_id,
                            base_content_hash=compute_content_hash(payload))


@pytest.mark.parametrize("auto_grant", [False, True])
def test_keyless_first_review_obeys_generic_auto_grant_policy(development, monkeypatch, auto_grant):
    registry, _repo, drive, panels = development
    registry._ctx.task_constraint = None
    monkeypatch.setenv("OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS", str(auto_grant).lower())
    payload = _write_ext_skill(drive / "skills" / "external", "demo", permissions=[],
                              plugin_body="def register(api):\n    return None\n", extra_frontmatter='plugin_api: "2.0"\n')
    _mark_self_authored(payload, drive)
    result = run_skill_review_lifecycle_blocking(registry._ctx, "demo")
    assert result["status"] == "clean" and result["auto_flow"] is auto_grant, result
    assert load_enabled(drive, "demo") is auto_grant
    assert len(panels) == 1


def test_free_replay_refreshes_disabled_auto_flow_without_another_panel(development, monkeypatch):
    registry, _repo, drive, panels = development
    registry._ctx.task_constraint = None
    monkeypatch.setenv("OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS", "true")
    payload = _write_ext_skill(drive / "skills" / "external", "demo", permissions=[],
                              plugin_body="def register(api):\n    return None\n", extra_frontmatter='plugin_api: "2.0"\n')
    _mark_self_authored(payload, drive)
    phases = iter([("failed", "controlled dependency failure"), ("not_required", "")])
    monkeypatch.setattr("ouroboros.skill_dependencies.ensure_skill_dependencies", lambda *a, **k: next(phases))
    first = run_skill_review_lifecycle_blocking(registry._ctx, "demo")
    assert first["status"] == "clean" and first["auto_flow"] is True
    assert not load_enabled(drive, "demo")
    before = (drive / "state" / "skills" / "demo" / "review.json").read_bytes()
    monkeypatch.setenv("OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS", "false")
    second = run_skill_review_lifecycle_blocking(registry._ctx, "demo")
    assert second["status"] == "clean" and second["deps_status"] == "not_required", second
    assert second["auto_flow"] is False and not load_enabled(drive, "demo")
    assert second["reused"] is True and second["record_id"] == first["record_id"] and len(panels) == 1
    assert (drive / "state" / "skills" / "demo" / "review.json").read_bytes() == before


@pytest.mark.serial
def test_installed_script_edit_review_execute_repeat_without_owner_click(development):
    registry, _repo, drive, panels = development
    payload = drive / "skills" / "external" / "demo"
    (payload / "scripts").mkdir(parents=True)
    (payload / "SKILL.md").write_text(
        "---\nname: demo\ndescription: Development test.\nversion: 1.0.0\n"
        "type: script\nruntime: python3\npermissions: []\nscripts:\n  - name: run.py\n---\n"
    )
    (payload / "scripts" / "run.py").write_text("print('version one')\n")
    _admit(registry, drive, payload)
    hashes = []
    for text in ("version one", "version two"):
        if text == "version two":
            edited = registry.execute("edit_text", {
                "root": "skill_payload", "path": "scripts/run.py",
                "old_str": "version one", "new_str": text,
            })
            assert "Replaced" in edited, edited
        checked = registry.execute("skill_review", {"skill": "demo"})
        state = load_review_state(drive, "demo")
        hashes.append(state.content_hash)
        assert state.status == "clean", checked
        assert state.content_hash == compute_content_hash(payload)
        # Review is not enable authority; the ordinary task chooses its toggle.
        enabled = registry.execute("toggle_skill", {"skill": "demo", "enabled": True})
        assert load_enabled(drive, "demo"), enabled
        actual = json.loads(registry.execute("skill_exec", {"skill": "demo", "script": "run.py"}))
        assert actual["exit_code"] == 0 and text in actual["stdout"], actual
        assert actual["content_hash"] == state.content_hash
    assert hashes[0] != hashes[1]
    assert len(panels) == 2


def test_split_drive_attestation_and_history_stay_canonical(development, tmp_path):
    from ouroboros.project_dialogue import build_owner_message_ref
    from ouroboros.skill_lifecycle_actions import run_skill_action

    registry, _repo, drive, panels = development
    payload = _write_ext_skill(drive / "skills" / "external", "demo", permissions=["tool"],
                              plugin_body="def register(api):\n    return None\n", extra_frontmatter='plugin_api: "2.0"\n')
    _admit(registry, drive, payload)
    child = tmp_path / "execution"
    child.mkdir()
    registry._ctx.drive_root = child
    registry._ctx.task_metadata = {"budget_drive_root": str(drive)}
    text = "I authored demo. Skip its expensive review for this exact version."
    ref = build_owner_message_ref(chat_id=42, client_message_id="attest-demo", ts="2026-09-06T10:00:00Z", text=text)
    dialogue_store.append(drive, "chat", {**ref, "direction": "in", "text": text})
    result = run_skill_action(registry._ctx, "demo", "attest", expected_content_hash=compute_content_hash(payload),
                              owner_source={"kind": "chat", "ref": ref})
    assert result["ok"], result
    assert load_review_state(drive, "demo", skill_dir=payload).review_profile == "owner_attested"
    # The attestation lands in the canonical drive (the owner marker beside review.json); it buys no wave and
    # writes no ledger record.
    assert (drive / "state" / "skills" / "demo" / "owner_attestation.json").exists()
    from ouroboros import review_ledger
    assert review_ledger.recent_records(drive, "", surface="skill") == []
    assert not (child / "state" / "skills").exists()
    assert not panels


@pytest.mark.parametrize("auto_grant", [False, True])
def test_dependency_repair_replays_review_and_preserves_owner_disable(development, monkeypatch, auto_grant):
    registry, _repo, drive, panels = development
    monkeypatch.setenv("OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS", str(auto_grant).lower())
    payload = _write_ext_skill(
        drive / "skills" / "external", "demo", permissions=["tool", "inject_chat"],
        plugin_body="def register(api):\n    return None\n", extra_frontmatter='plugin_api: "2.0"\n',
    )
    _admit(registry, drive, payload)
    phases = iter([("failed", "controlled dependency failure"), ("installed", "")])
    monkeypatch.setattr("ouroboros.skill_dependencies.ensure_skill_dependencies", lambda *a, **k: next(phases))
    first = run_skill_review_lifecycle_blocking(registry._ctx, "demo")
    assert first["status"] == "clean" and first["deps_status"] == "failed", first
    before = (drive / "state" / "skills" / "demo" / "review.json").read_bytes()
    save_enabled(drive, "demo", False, actor="owner_ui")
    second = run_skill_review_lifecycle_blocking(registry._ctx, "demo")
    assert second["status"] == "clean" and second["deps_status"] == "installed", second
    assert len(panels) == 1
    assert second["reused"] is True and second["record_id"] == first["record_id"]
    assert (drive / "state" / "skills" / "demo" / "review.json").read_bytes() == before
    assert not load_enabled(drive, "demo")
    loaded = find_skill(drive, "demo")
    from ouroboros.skill_loader import grant_status_for_skill
    grants = grant_status_for_skill(drive, loaded)
    assert grants["all_granted"] is auto_grant
    assert not extension_loader.is_extension_live("demo", drive)


@pytest.mark.parametrize("failure", ["plugin", "reconcile"])
def test_review_verdict_survives_visible_extension_load_failure(development, monkeypatch, failure):
    registry, _repo, drive, _panels = development
    payload = _write_ext_skill(
        drive / "skills" / "external", "demo", permissions=["tool"],
        plugin_body="def register(api):\n    raise RuntimeError('controlled plugin failure')\n",
        extra_frontmatter='plugin_api: "2.0"\n',
    )
    _admit(registry, drive, payload)
    if failure == "reconcile":
        def fail_reconcile(*args, **kwargs):
            raise OSError("controlled reconcile failure")
        monkeypatch.setattr(extension_loader, "reconcile_extension", fail_reconcile)
    result = run_skill_review_lifecycle_blocking(registry._ctx, "demo")
    assert result["status"] == "clean", result
    # Controlled reconciliation errors still describe review's reconciliation;
    # a plugin load starts only when this task explicitly enables it.
    if failure == "plugin":
        result = registry.execute("toggle_skill", {"skill": "demo", "enabled": True})
        assert "controlled plugin failure" in result, result
    else:
        assert "controlled" in result["extension_load_error"], result
        assert result["extension_live_loaded"] is not True
        job = json.loads((drive / "state" / "skills" / "demo" / "review_job.json").read_text())
        assert job["lifecycle_status"] == "failed"
        assert job["review_status"] == "clean"
    assert load_review_state(drive, "demo").status == "clean"
