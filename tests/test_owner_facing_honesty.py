"""v6.70.0 honesty & reliability tests: full owner-facing review rationale,
truncation floor, acceptance format repair, self-locating tool errors,
scout skill_payload access, secret-redacting logs, stray-process invariant
(startup + TTL-cached live), project-thread pointers, tool-trace arg width,
ground-truth probe guidance, and the decision-turn outcome contract."""

from __future__ import annotations

import logging
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


# ---------------------------------------------------------------------------
# 2.1 Full reviewer rationale + response_ref in the projection
# ---------------------------------------------------------------------------

def test_actor_projection_publishes_full_reason_and_response_ref():
    from ouroboros.review_substrate import _review_actor_projection

    long_reason = "the reviewer explained at length: " + ("x" * 5000)
    actor = {
        "slot_id": "s1", "model": "m", "status": "ok",
        "parsed": {"verdict": "PASS", "summary": long_reason},
        "reason": long_reason,
        "response_ref": {"call_id": "c1", "manifest_ref": "calls/t/c1.json"},
        "quorum_contribution": True,
    }
    row = _review_actor_projection(actor, "task_acceptance")
    assert row["reason"] == long_reason  # no 500-char cap, no OMISSION NOTE
    assert "OMISSION NOTE" not in row["reason"]
    assert row["response_ref"]["call_id"] == "c1"  # durable full copy is reachable


def test_panel_projection_reason_uncapped():
    from ouroboros.review_substrate import compact_review_projection

    long_reason = "panel cause: " + ("y" * 3000)
    runs = [{
        "request": {"surface": "task_acceptance", "policy": {}},
        "actors": [], "aggregate_signal": "DEGRADED",
        "degraded_reasons": [long_reason],
    }]
    panel = compact_review_projection(runs)["panels"][0]
    assert long_reason in panel["reason"]
    assert "OMISSION NOTE" not in panel["reason"]


def test_projection_still_redacts_secrets():
    from ouroboros.review_substrate import _review_actor_projection

    leaky = "reason with a token sk-or-v1-" + "a" * 60
    actor = {"slot_id": "s1", "model": "m", "status": "ok",
             "parsed": {"verdict": "PASS"}, "reason": leaky}
    row = _review_actor_projection(actor, "task_acceptance")
    assert "sk-or-v1-" + "a" * 60 not in row["reason"]


# ---------------------------------------------------------------------------
# 2.2 Truncation floor — a cut cheaper than its marker is forbidden
# ---------------------------------------------------------------------------

def test_truncation_floor_skips_marker_longer_than_savings():
    from ouroboros.utils import truncate_review_artifact

    text = "z" * 562
    out = truncate_review_artifact(text, limit=500)  # the historical absurd case
    assert out == text  # 62 chars saved < marker length -> no cut

    long_text = "z" * 5000
    cut = truncate_review_artifact(long_text, limit=500)
    assert cut.startswith("z" * 500) and "OMISSION NOTE" in cut


def test_reflection_marker_uses_canonical_format():
    from ouroboros.reflection import _truncate_with_notice as reflect_cut

    out = reflect_cut("q" * 9000, 100)
    assert "OMISSION NOTE" in out and "[+" not in out


def test_task_contract_marker_uses_canonical_format():
    import inspect

    from ouroboros.contracts import task_contract

    source = inspect.getsource(task_contract)
    assert "chars omitted)" not in source  # legacy marker retired
    # Both bounded fields delegate to the shared primitive (marker + floor SSOT).
    assert source.count("truncate_review_artifact") >= 2
    assert "OMISSION NOTE" not in source  # no hand-rolled marker copies remain


# ---------------------------------------------------------------------------
# 2.6 Acceptance format repair uses the second permitted physical send
# ---------------------------------------------------------------------------

def test_acceptance_malformed_response_gets_one_repair_resend(tmp_path):
    from ouroboros.review_substrate import ReviewRequest, ReviewSlot, run_review_request

    calls = {"n": 0}

    class _FlakyJson:
        def chat(self, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return {"content": "sorry, here are my thoughts in prose (no JSON)"}, {}
            return {"content": '{"verdict": "PASS", "findings": [], "summary": "ok"}'}, {}

    result = run_review_request(
        ReviewRequest(surface="task_acceptance", goal="g", task_id="t",
                      policy={"min_successful_slots": 1}),
        slots=[ReviewSlot(slot_id="s1", model="m")],
        drive_root=tmp_path, llm=_FlakyJson(),
    )
    assert calls["n"] == 2  # extraction/format repair used the second send
    assert result.aggregate_signal == "PASS"


def test_acceptance_repair_does_not_third_send(tmp_path):
    from ouroboros.review_substrate import ReviewRequest, ReviewSlot, run_review_request

    calls = {"n": 0}

    class _AlwaysProse:
        def chat(self, **kwargs):
            calls["n"] += 1
            return {"content": "still prose"}, {}

    result = run_review_request(
        ReviewRequest(surface="task_acceptance", goal="g", task_id="t",
                      policy={"min_successful_slots": 1}),
        slots=[ReviewSlot(slot_id="s1", model="m")],
        drive_root=tmp_path, llm=_AlwaysProse(),
    )
    assert calls["n"] == 2  # hard two-physical-send rail
    assert result.aggregate_signal == "DEGRADED"


# ---------------------------------------------------------------------------
# 2.7 Self-locating tool errors
# ---------------------------------------------------------------------------

def _ctx(tmp_path):
    return SimpleNamespace(
        repo_dir=str(tmp_path), drive_root=str(tmp_path / "data"),
        system_repo_dir=str(tmp_path), workspace_root=None, workspace_mode="",
        task_metadata={}, task_contract={}, task_constraint=None, task_id="t",
    )


def test_access_error_names_profile_visible_roots(tmp_path):
    from ouroboros.tools.core_file_tools import _access_or_block

    normalized, error = _access_or_block(_ctx(tmp_path), "definitely_not_a_root", "read")
    assert "TOOL_ARG_ERROR" in error
    assert "Roots your profile can read:" in error


def test_affordance_map_names_invisible_roots(tmp_path):
    from ouroboros.tool_access import filesystem_affordance_map

    result = filesystem_affordance_map(_ctx(tmp_path))
    assert "invisible_roots" in result
    assert isinstance(result["invisible_roots"], list)


# ---------------------------------------------------------------------------
# 2.8 Readonly scouts can read skill payloads (owner-approved policy change)
# ---------------------------------------------------------------------------

def test_local_readonly_subagent_reads_skill_payload():
    from ouroboros.tool_access import decide_tool_access

    for operation in ("read", "list", "search"):
        decision = decide_tool_access(
            profile="local_readonly_subagent", root="skill_payload", operation=operation,
        )
        assert decision.allow, operation
    for operation in ("write", "edit", "shell"):
        decision = decide_tool_access(
            profile="local_readonly_subagent", root="skill_payload", operation=operation,
        )
        assert not decision.allow, operation


# ---------------------------------------------------------------------------
# 2.10 Secret-redacting root log filter
# ---------------------------------------------------------------------------

def test_log_filter_masks_bot_token():
    import server as server_mod

    record = logging.LogRecord(
        name="httpx", level=logging.INFO, pathname="", lineno=0,
        msg='HTTP Request: POST https://api.telegram.org/bot123456789:AAHsecretsecretsecretsecr/getUpdates "200 OK"',
        args=(), exc_info=None,
    )
    keep = server_mod._SecretRedactingLogFilter().filter(record)
    assert keep is True  # never drops the line
    assert "AAHsecretsecretsecretsecr" not in record.getMessage()


def test_httpx_logger_quieted():
    import server  # noqa: F401 — importing applies the logging setup

    assert logging.getLogger("httpx").level >= logging.WARNING


# ---------------------------------------------------------------------------
# 2.11 Stray server process invariant (report-only)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(sys.platform == "win32", reason="POSIX pgrep/getuid semantics")
def test_stray_server_check_reports_foreign_pid(tmp_path, monkeypatch):
    from ouroboros import agent_startup_checks as checks

    env = SimpleNamespace(drive_root=str(tmp_path))
    (tmp_path / "state").mkdir(parents=True, exist_ok=True)

    real_run = subprocess.run

    def fake_run(cmd, **kwargs):
        if cmd and cmd[0] == "pgrep":
            return SimpleNamespace(stdout="424242\n", returncode=0)
        return real_run(cmd, **kwargs)

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr(
        "ouroboros.platform_layer.process_command",
        lambda pid: "/usr/bin/python3 -m ouroboros.cli server --no-ui" if pid == 424242 else "",
    )
    monkeypatch.setattr("ouroboros.platform_layer.process_group_id", lambda pid: pid)
    result, issues = checks.check_stray_server_processes(env)
    assert issues == 1
    assert result["status"] == "stray_processes"
    assert result["processes"][0]["pid"] == 424242


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX pgrep/getuid semantics")
def test_stray_server_check_matches_packaged_install(tmp_path, monkeypatch):
    """A packaged desktop sibling runs "EMBEDDED_PYTHON .../Ouroboros/repo/server.py"
    (capital O, no 'ouroboros server' argv) — the shape must still match."""
    from ouroboros import agent_startup_checks as checks

    env = SimpleNamespace(drive_root=str(tmp_path))
    (tmp_path / "state").mkdir(parents=True, exist_ok=True)
    def fake_run(cmd, **kwargs):
        if cmd and cmd[0] == "pgrep":
            return SimpleNamespace(stdout="515151\n", returncode=0)
        return SimpleNamespace(stdout="", returncode=1)

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr(
        "ouroboros.platform_layer.process_command",
        lambda pid: "/Users/o/Ouroboros/python/bin/python3 /Users/o/Ouroboros/repo/server.py" if pid == 515151 else "",
    )
    monkeypatch.setattr("ouroboros.platform_layer.process_group_id", lambda pid: pid)
    result, issues = checks.check_stray_server_processes(env)
    assert issues == 1 and result["status"] == "stray_processes"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX pgrep/getuid semantics")
def test_stray_server_check_annotates_same_install_scope(tmp_path, monkeypatch):
    """The launcher reaps PROVEN same-install strays per generation, so the
    report must say which class a survivor is in: a same_install WARN points at
    a direct run or a spared/kill-surviving pid, a foreign one never does."""
    import pathlib

    from ouroboros import agent_startup_checks as checks

    monkeypatch.setattr("ouroboros.config.REPO_DIR", pathlib.PurePosixPath("/opt/Ouroboros/repo"))
    env = SimpleNamespace(drive_root=str(tmp_path))
    (tmp_path / "state").mkdir(parents=True, exist_ok=True)
    ours = "/opt/Ouroboros/python/bin/python3 /opt/Ouroboros/repo/server.py"
    theirs = "/Users/o/Ouroboros/python/bin/python3 /Users/o/Ouroboros/repo/server.py"

    def fake_run(cmd, **kwargs):
        if cmd and cmd[0] == "pgrep":
            return SimpleNamespace(stdout="616161\n626262\n", returncode=0)
        return SimpleNamespace(stdout="", returncode=1)

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr(
        "ouroboros.platform_layer.process_command",
        lambda pid: {616161: ours, 626262: theirs}.get(pid, ""),
    )
    monkeypatch.setattr("ouroboros.platform_layer.process_group_id", lambda pid: pid)
    result, issues = checks.check_stray_server_processes(env)
    assert issues == 1
    scopes = {row["pid"]: row["scope"] for row in result["processes"]}
    assert scopes == {616161: "same_install", 626262: "foreign"}


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX pgrep/getuid semantics")
def test_stray_server_check_ok_when_clean(tmp_path, monkeypatch):
    from ouroboros import agent_startup_checks as checks

    env = SimpleNamespace(drive_root=str(tmp_path))
    monkeypatch.setattr(
        "subprocess.run",
        lambda cmd, **kwargs: SimpleNamespace(stdout="", returncode=1),
    )
    result, issues = checks.check_stray_server_processes(env)
    assert issues == 0 and result["status"] == "ok"


# ---------------------------------------------------------------------------
# 2.12 Project-thread pointer in the initiating-chat summary
# ---------------------------------------------------------------------------

def test_project_thread_note_names_project(monkeypatch):
    from ouroboros import projects_registry

    monkeypatch.setattr(
        projects_registry, "project_chat_for_task_tree",
        lambda root, tid, parent="", rroot="": 77,
    )
    monkeypatch.setattr(
        projects_registry, "list_projects",
        lambda root: [{"chat_id": 77, "name": "site relaunch"}],
    )
    note = projects_registry.project_thread_note_for_task({"id": "t1", "chat_id": 1})
    assert "site relaunch" in note
    # A task already IN the project chat gets no pointer.
    assert projects_registry.project_thread_note_for_task({"id": "t1", "chat_id": 77}) == ""


# ---------------------------------------------------------------------------
# 2.5 Decision-turn outcome contract
# ---------------------------------------------------------------------------

def test_ephemeral_turn_gets_decision_rule(tmp_path):
    from ouroboros.context import build_runtime_section

    env = SimpleNamespace(repo_dir=str(tmp_path), drive_root=str(tmp_path / "data"))
    section = build_runtime_section(env, {"_ephemeral_turn": True, "id": "t"})
    assert "decision_turn_rule" in section
    assert "promise" in section
    assert "final no-tool response MUST be self-contained" in section
    assert "not durable conversation history" in section
    plain = build_runtime_section(env, {"id": "t"})
    assert "decision_turn_rule" not in plain


def test_system_prompt_separates_routing_annotation_from_final_reply():
    system_prompt = (
        Path(__file__).resolve().parents[1] / "prompts" / "SYSTEM.md"
    ).read_text(encoding="utf-8")
    normalized = " ".join(system_prompt.split())
    assert "exactly ONE routing decision" in normalized
    assert "A typed routing annotation is metadata" in normalized
    assert "one self-contained final response" in normalized
    assert "exactly ONE owner-visible outcome" not in normalized


# ---------------------------------------------------------------------------
# Round-1 adversarial-review fixes (v6.70.0)
# ---------------------------------------------------------------------------

def test_reflection_tiny_identifier_fields_get_hard_slice():
    """A multi-line omission marker inside a one-line backlog value (kind=40,
    priority=10) would be worse damage than the cut it discloses."""
    from ouroboros.reflection import _truncate_with_notice as reflect_cut

    out = reflect_cut("agenda_item_" + "k" * 200, 40)
    assert out == ("agenda_item_" + "k" * 200)[:40]
    assert "OMISSION NOTE" not in out


def test_response_ref_projection_never_leaks_host_paths(tmp_path):
    """Feed the projection the REAL persist_call() return shape — nested refs
    with absolute host paths — and require flat hash anchors, no paths."""
    from ouroboros.observability import persist_call
    from ouroboros.review_substrate import _review_actor_projection

    ref = persist_call(
        tmp_path, task_id="t", call_id="c9", call_type="review_response",
        payload={"message": {"content": "hello"}},
        manifest={"surface": "task_acceptance"},
    )
    actor = {"slot_id": "s1", "model": "m", "status": "ok",
             "parsed": {"verdict": "PASS"}, "reason": "ok", "response_ref": ref}
    row = _review_actor_projection(actor, "task_acceptance")
    projected = row["response_ref"]
    assert projected["call_id"] == "c9"
    assert projected.get("sha256")  # content-hash anchor survives the projection
    assert all(str(tmp_path) not in str(v) for v in projected.values())
    assert all("/" not in str(v) for v in projected.values())


def test_repair_resend_blocked_by_rail_keeps_first_answer(tmp_path):
    """When the physical-attempt rail blocks the format-repair resend, the
    malformed FIRST answer must survive as forensics (not a bare error actor)."""
    from ouroboros.review_substrate import ReviewRequest, ReviewSlot, run_review_request
    from ouroboros.usage_accounting import PhysicalAttemptLimitExceeded

    calls = {"n": 0}

    class _RailBlocked:
        def chat(self, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return {"content": "prose answer, no JSON — but substantive"}, {}
            raise PhysicalAttemptLimitExceeded("physical attempt limit reached")

    result = run_review_request(
        ReviewRequest(surface="task_acceptance", goal="g", task_id="t",
                      policy={"min_successful_slots": 1}),
        slots=[ReviewSlot(slot_id="s1", model="m")],
        drive_root=tmp_path, llm=_RailBlocked(),
    )
    actor = result.actors[0]
    status = actor["status"] if isinstance(actor, dict) else actor.status
    raw = (actor.get("raw_text") if isinstance(actor, dict) else actor.raw_text) or ""
    assert status != "error"
    assert "prose answer" in raw


def test_repair_resend_empty_keeps_first_answer(tmp_path):
    """An EMPTY repair resend keeps the substantive malformed first answer
    instead of landing an empty actor (forensics symmetry with the
    exception paths)."""
    from ouroboros.review_substrate import ReviewRequest, ReviewSlot, run_review_request

    calls = {"n": 0}

    class _ResendEmpty:
        def chat(self, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return {"content": "prose answer, no JSON — but substantive"}, {}
            return {"content": ""}, {}

    result = run_review_request(
        ReviewRequest(surface="task_acceptance", goal="g", task_id="t",
                      policy={"min_successful_slots": 1}),
        slots=[ReviewSlot(slot_id="s1", model="m")],
        drive_root=tmp_path, llm=_ResendEmpty(),
    )
    actor = result.actors[0]
    status = actor["status"] if isinstance(actor, dict) else actor.status
    raw = (actor.get("raw_text") if isinstance(actor, dict) else actor.raw_text) or ""
    assert calls["n"] == 2
    assert status != "empty"
    assert "prose answer" in raw


def test_repair_first_attempt_is_persisted(tmp_path):
    """P1 forensics: a SUCCESSFUL repair must not make the malformed first
    answer unreconstructible — attempt 1 gets its own persisted call record."""
    from ouroboros.review_substrate import ReviewRequest, ReviewSlot, run_review_request

    calls = {"n": 0}

    class _FlakyJson:
        def chat(self, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return {"content": "sorry, prose thoughts (no JSON)"}, {}
            return {"content": '{"verdict": "PASS", "findings": [], "summary": "ok"}'}, {}

    run_review_request(
        ReviewRequest(surface="task_acceptance", goal="g", task_id="t",
                      policy={"min_successful_slots": 1}),
        slots=[ReviewSlot(slot_id="s1", model="m")],
        drive_root=tmp_path, llm=_FlakyJson(),
    )
    attempt1 = list(tmp_path.rglob("*attempt1_response*"))
    assert attempt1, "first malformed attempt must be persisted as its own call record"


def test_repair_resend_transport_failure_keeps_first_answer(tmp_path):
    """Symmetric forensics: a TRANSPORT failure on the repair resend also falls
    back to the preserved first malformed answer, not an error actor."""
    from ouroboros.review_substrate import ReviewRequest, ReviewSlot, run_review_request

    calls = {"n": 0}

    class _ResendDies:
        def chat(self, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return {"content": "prose answer, no JSON — but substantive"}, {}
            raise RuntimeError("upstream 500")

    result = run_review_request(
        ReviewRequest(surface="task_acceptance", goal="g", task_id="t",
                      policy={"min_successful_slots": 1}),
        slots=[ReviewSlot(slot_id="s1", model="m")],
        drive_root=tmp_path, llm=_ResendDies(),
    )
    actor = result.actors[0]
    status = actor["status"] if isinstance(actor, dict) else actor.status
    raw = (actor.get("raw_text") if isinstance(actor, dict) else actor.raw_text) or ""
    assert calls["n"] == 2
    assert status != "error"
    assert "prose answer" in raw


def test_tool_trace_arg_width_is_200():
    from ouroboros.agent_task_pipeline import build_trace_summary

    long_arg = "x" * 1000
    summary = build_trace_summary({
        "tool_calls": [{"tool": "run_command", "args": {"command": long_arg}, "status": "ok"}],
    })
    assert "x" * 200 in summary  # 200-char per-arg window (was 60)
    assert "x" * 260 not in summary


def test_not_found_error_self_locates_without_ancestor_hint(tmp_path):
    from ouroboros.tools import core_file_tools as tools_core

    ctx = _ctx(tmp_path)
    out = tools_core._read_file(ctx, path="definitely/missing/file.py", root="system_repo")
    assert "NOT_FOUND" in out
    # Owner decision 22a: resolved-path hint yes, nearest-ancestor walk no.
    src = __import__("inspect").getsource(tools_core)
    assert "nearest existing ancestor" not in src


def test_access_blocked_message_has_single_period(tmp_path):
    from ouroboros.tools.core_file_tools import _access_or_block

    _, error = _access_or_block(_ctx(tmp_path), "deliverables", "write")
    assert "TOOL_ACCESS_BLOCKED" in error
    assert ".." not in error.replace("…", "")


def test_promote_chat_description_carries_ground_truth_probe():
    from ouroboros.tools.control import _PROMOTE_CHAT_DESCRIPTION

    assert "ground-truth its existence with one cheap probe" in _PROMOTE_CHAT_DESCRIPTION
    assert "memory of past work is not evidence" in _PROMOTE_CHAT_DESCRIPTION


def test_degraded_owner_line_bounds_each_reason():
    import inspect

    # v7 L-B split: the degraded-owner-line writer lives with the host
    # acceptance review owner; loop.py re-exports it.
    from ouroboros import loop_acceptance_review as loop_mod

    src = inspect.getsource(loop_mod)
    assert "more in the task result" in src  # overflow disclosure, not silence
    # Bounded preview per cause via the shared primitive; full causes live in
    # the structured decision record.
    assert "truncate_review_artifact(str(r), limit=300)" in src


def test_redaction_filter_ssot_is_observability():
    import server as server_mod
    from ouroboros.observability import SecretRedactingLogFilter

    assert server_mod._SecretRedactingLogFilter is SecretRedactingLogFilter


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX pgrep/getuid semantics")
def test_stray_scan_restricted_to_current_user(tmp_path, monkeypatch):
    from ouroboros import agent_startup_checks as checks

    env = SimpleNamespace(drive_root=str(tmp_path))
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = list(cmd)
        return SimpleNamespace(stdout="", returncode=1)

    monkeypatch.setattr("subprocess.run", fake_run)
    checks.check_stray_server_processes(env)
    import os

    assert seen["cmd"][:3] == ["pgrep", "-U", str(os.getuid())]


def test_health_invariants_stray_probe_is_ttl_cached(tmp_path, monkeypatch):
    from ouroboros import context as context_mod

    calls = {"n": 0}

    def fake_check(env):
        calls["n"] += 1
        return {"status": "stray_processes", "processes": [{"pid": 4242, "command": "ouroboros server"}]}, 1

    monkeypatch.setattr(
        "ouroboros.agent_startup_checks.check_stray_server_processes", fake_check,
    )
    monkeypatch.setitem(context_mod._STRAY_PROBE_CACHE, "ts", 0.0)
    monkeypatch.setitem(context_mod._STRAY_PROBE_CACHE, "note", "")
    env = SimpleNamespace(drive_root=str(tmp_path))
    first = context_mod._stray_server_note(env)
    second = context_mod._stray_server_note(env)
    assert "STRAY SERVER PROCESS" in first and "4242" in first
    assert second == first
    assert calls["n"] == 1  # TTL cache: one live probe, not one per turn


def test_ephemeral_turn_producer_sets_flag():
    """context keys decision_turn_rule on task['_ephemeral_turn'] — pin that the
    chat-turn producer actually sets it (integration seam, run-2 gate finding)."""
    import inspect

    from supervisor import worker_chat_lane

    src = inspect.getsource(worker_chat_lane)
    assert 'task["_ephemeral_turn"] = True' in src


def test_actor_records_carry_response_ref_end_to_end(tmp_path):
    """The substrate's own persistence path populates response_ref on actor
    records, and the projection surfaces its flat hash anchors."""
    from ouroboros.review_substrate import (
        ReviewRequest, ReviewSlot, _review_actor_projection, run_review_request,
    )

    class _Ok:
        def chat(self, **kwargs):
            return {"content": '{"verdict": "PASS", "findings": [], "summary": "ok"}'}, {}

    result = run_review_request(
        ReviewRequest(surface="task_acceptance", goal="g", task_id="t",
                      policy={"min_successful_slots": 1}),
        slots=[ReviewSlot(slot_id="s1", model="m")],
        drive_root=tmp_path, llm=_Ok(),
    )
    actor = result.actors[0]
    ref = actor["response_ref"] if isinstance(actor, dict) else actor.response_ref
    assert ref.get("call_id")
    row = _review_actor_projection(
        actor if isinstance(actor, dict) else actor.__dict__, "task_acceptance",
    )
    assert row["response_ref"].get("call_id")
    assert row["response_ref"].get("sha256")


def test_affordance_map_carries_label_path_pairs(tmp_path):
    """root_paths gives label=resolved-path for every visible root (v6.54.3
    lesson on the context digest — bare labels left the model guessing)."""
    from ouroboros.tool_access import filesystem_affordance_map

    result = filesystem_affordance_map(_ctx(tmp_path))
    paths = result.get("root_paths")
    assert isinstance(paths, dict) and paths
    assert "skill_payload" not in paths  # needs bucket/skill args
    assert result["default_root"] == "active_workspace"
    assert result["skill_payload_selector"] == (
        "root=skill_payload requires bucket + skill_name"
    )
    for label, path in paths.items():
        assert path and str(path).startswith("/") or ":" in str(path), (label, path)
    # The read-only orchestrator root is resolvable when visible to the profile.
    if "subagent_projects" in result.get("readonly_roots", []):
        assert paths.get("subagent_projects")
