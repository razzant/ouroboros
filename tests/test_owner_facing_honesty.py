"""v6.70.0 honesty & reliability tests: full owner-facing review rationale,
truncation floor, self-locating tool errors, scout skill_payload access,
secret-redacting logs, stray-process invariant (startup + TTL-cached live),
project-thread pointers, tool-trace arg width, ground-truth probe guidance,
and the decision-turn outcome contract. (The acceptance panel's format-repair
resend and owner cause line left with the acceptance machine.)"""

from __future__ import annotations

import logging
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


def test_task_contract_preserves_complete_authority_fields():
    from ouroboros.contracts import task_contract

    intent = "intent-" + ("i" * 900) + "-tail"
    claim = "claim-" + ("c" * 700) + "-tail"
    support = "support-" + ("s" * 800) + "-tail"
    contract = task_contract.build_task_contract({
        "delegation_budget": {"intent_note": intent},
        "acceptance_claims": [{"claim": claim, "support": support}],
    })
    assert contract["delegation_budget"]["intent_note"] == intent
    assert contract["acceptance_claims"][0]["claim"] == claim
    assert contract["acceptance_claims"][0]["support"] == support


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
    from ouroboros.tools.core import _access_or_block

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
    from ouroboros.observability import SecretRedactingLogFilter

    record = logging.LogRecord(
        name="httpx", level=logging.INFO, pathname="", lineno=0,
        msg='HTTP Request: POST https://api.telegram.org/bot123456789:AAHsecretsecretsecretsecr/getUpdates "200 OK"',
        args=(), exc_info=None,
    )
    keep = SecretRedactingLogFilter().filter(record)
    assert keep is True  # never drops the line
    assert "AAHsecretsecretsecretsecr" not in record.getMessage()


# ---------------------------------------------------------------------------
# 2.11 Stray server process invariant (report-only)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(sys.platform == "win32", reason="POSIX pgrep/getuid semantics")
def test_stray_server_check_reports_foreign_pid(tmp_path, monkeypatch):
    from ouroboros import agent_startup_checks as checks

    env = SimpleNamespace(drive_root=str(tmp_path))
    (tmp_path / "state").mkdir(parents=True, exist_ok=True)

    def fake_run(cmd, **kwargs):
        if cmd and cmd[0] == "pgrep":
            return SimpleNamespace(stdout="424242\n", returncode=0)
        return SimpleNamespace(stdout="", returncode=1)  # the batch reads nothing: a live 424242 elsewhere is not this pid

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
def test_stray_scan_reads_every_command_line_with_one_ps(tmp_path, monkeypatch):
    """A machine full of test workers and editors whose argv names ouroboros: one ps for all of them,
    and only a pid that exited between pgrep and ps is read on its own."""
    from ouroboros import agent_startup_checks as checks

    env = SimpleNamespace(drive_root=str(tmp_path))
    (tmp_path / "state").mkdir(parents=True, exist_ok=True)
    rows = {101: "/usr/bin/python3 -m pytest /w/ouroboros/tests/test_x.py",
            102: "/opt/o/python/bin/python3 /opt/o/Ouroboros/repo/server.py", 103: "vim /w/ouroboros/notes.md"}
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd[0])
        if cmd[0] == "pgrep":
            return SimpleNamespace(stdout="101\n102\n103\n104\n", returncode=0)
        assert cmd[:4] == ["ps", "-ww", "-o", "pid=,command="]
        asked = [int(pid) for pid in cmd[-1].split(",")]
        return SimpleNamespace(stdout="".join(f"{pid:>6} {rows[pid]}\n" for pid in asked if pid in rows), returncode=1)

    alone = []
    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr("ouroboros.platform_layer.process_command", lambda pid: alone.append(pid) or "")
    monkeypatch.setattr("ouroboros.platform_layer.process_group_id", lambda pid: pid)
    result, issues = checks.check_stray_server_processes(env)
    assert issues == 1 and [row["pid"] for row in result["processes"]] == [102]
    assert calls == ["pgrep", "ps"] and alone == [104]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX pgrep/getuid semantics")
def test_stray_scan_reads_the_known_groups_once(tmp_path, monkeypatch):
    import json

    from ouroboros import agent_startup_checks as checks

    env = SimpleNamespace(drive_root=str(tmp_path))
    (tmp_path / "state").mkdir(parents=True, exist_ok=True)
    (tmp_path / "state" / "process_ledger.jsonl").write_text(
        "".join(json.dumps({"pid": pid}) + "\n" for pid in (7001, 7002, 7003)), encoding="utf-8")
    server = "/opt/o/python/bin/python3 /opt/o/Ouroboros/repo/server.py"

    def fake_run(cmd, **kwargs):
        if cmd[0] == "pgrep":
            return SimpleNamespace(stdout="201\n202\n203\n", returncode=0)
        return SimpleNamespace(stdout="".join(f"{pid} {server}\n" for pid in (201, 202, 203)), returncode=0)

    looked = []
    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr("ouroboros.platform_layer.process_group_id",
                        lambda pid: looked.append(pid) or (7002 if pid == 202 else pid))
    result, issues = checks.check_stray_server_processes(env)
    # 202 shares a known pid's process group and stays out; the other two servers are reported.
    assert issues == 1 and [row["pid"] for row in result["processes"]] == [201, 203]
    known_reads = [pid for pid in looked if pid not in (201, 202, 203)]
    assert known_reads and len(known_reads) == len(set(known_reads))


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
# 2.5 Ordinary turn context
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("obsolete_marker", [False, True])
def test_runtime_has_no_disappeared_routing_actor_rules(tmp_path, obsolete_marker):
    from ouroboros.context import build_runtime_section

    env = SimpleNamespace(repo_dir=str(tmp_path), drive_root=str(tmp_path / "data"))
    section = build_runtime_section(env, {"_ephemeral_turn": obsolete_marker, "id": "t"})
    assert "decision_turn_rule" not in section
    assert "promoted_task_toolset" not in section


# ---------------------------------------------------------------------------
# Round-1 adversarial-review fixes (v6.70.0)
# ---------------------------------------------------------------------------

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


def test_tool_trace_arg_width_is_200():
    from ouroboros.agent_task_pipeline import build_trace_summary

    long_arg = "x" * 1000
    summary = build_trace_summary({
        "tool_calls": [{"tool": "run_command", "args": {"command": long_arg}, "status": "ok"}],
    })
    assert "x" * 200 in summary  # 200-char per-arg window (was 60)
    assert "x" * 260 not in summary


def test_not_found_error_self_locates_without_ancestor_hint(tmp_path):
    from ouroboros.tools import core as tools_core

    ctx = _ctx(tmp_path)
    out = tools_core._read_file(ctx, path="definitely/missing/file.py", root="system_repo")
    assert "NOT_FOUND" in out
    # Owner decision 22a: the error names the resolved path and never walks to an existing ancestor.
    assert "definitely/missing/file.py" in out.replace("\\", "/")
    assert "ancestor" not in out.lower()


def test_access_blocked_message_has_single_period(tmp_path):
    from ouroboros.tools.core import _access_or_block

    _, error = _access_or_block(_ctx(tmp_path), "deliverables", "write")
    assert "TOOL_ACCESS_BLOCKED" in error
    assert ".." not in error.replace("…", "")


def test_start_task_description_carries_ground_truth_probe():
    from ouroboros.tools.control import _START_TASK_DESCRIPTION

    assert "ground-truth its existence with one cheap probe" in _START_TASK_DESCRIPTION
    assert "memory of past work is not evidence" in _START_TASK_DESCRIPTION


def test_redaction_filter_ssot_is_observability():
    """process_logging defines no filter of its own; the bootstrap attaches observability's
    (tests/test_process_logging.py checks the attached class in a fresh interpreter)."""
    from ouroboros import process_logging

    assert not [value for value in vars(process_logging).values()
                if isinstance(value, type) and issubclass(value, logging.Filter)]


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


def test_native_chat_lane_has_no_ephemeral_entry_point():
    """Swarm admission must not regain a separate model-owning actor."""
    import inspect
    from supervisor import worker_chat_lane

    assert not hasattr(worker_chat_lane, "handle_chat_ephemeral")
    assert "ephemeral" not in inspect.signature(worker_chat_lane._run_chat_task).parameters


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
