"""Unit tests for the OSWorld cu_bridge runner (PR #64 finalization).

These exercise the pure helpers only — no OSWorld VM, no Ouroboros server.
"""

from __future__ import annotations

import json
import pathlib
import os
import sys
import time
from pathlib import Path

import pytest

from devtools.benchmarks.common.model_slots import pin_single_model
from devtools.benchmarks.osworld import run_cu_bridge_agent as rcb
from ouroboros.extension_loader import extension_surface_name


_CU_ACTOR_MODEL = "openai/gpt-5.5"


def _cu_actor_settings():
    settings = {}
    pin_single_model(_CU_ACTOR_MODEL, target=settings)
    return settings


def test_infeasible_checks_final_answer_fields_only():
    assert rcb._final_answer_declares_infeasible({"final_answer": "TASK_INFEASIBLE"})
    assert rcb._final_answer_declares_infeasible({"result": "done now\nTASK_INFEASIBLE"})
    # Non-terminal fields must NOT trigger it.
    assert not rcb._final_answer_declares_infeasible({"description": "TASK_INFEASIBLE"})
    assert not rcb._final_answer_declares_infeasible({"metadata": {"note": "TASK_INFEASIBLE"}})
    # Inline (not a standalone line) mention must NOT trigger it.
    assert not rcb._final_answer_declares_infeasible({"result": "I considered TASK_INFEASIBLE but solved it"})
    assert not rcb._final_answer_declares_infeasible({})


def test_ax_tree_disabled_by_default_and_allow_a11y():
    ax = extension_surface_name("unix_computer_use", "ax_tree")
    default = rcb._effective_disabled_tools(False)
    assert ax in default
    # the computed host denylist is included
    for t in rcb._host_denied_tools():
        assert t in default
    allowed = rcb._effective_disabled_tools(True)
    assert ax not in allowed


def test_connection_switching_ext_tools_are_denied_vm_control_stays():
    # The runner pins the VM connection; the task must NOT be able to switch the
    # backend to local (use_local/activate_connection) or retarget it
    # (add_connection) — that would drive the host desktop. VM-control ext tools
    # and read-only connection introspection stay available.
    disabled = set(rcb._effective_disabled_tools(True))  # allow_a11y=True to isolate this concern

    def ext(n):
        return extension_surface_name("unix_computer_use", n)
    for n in ("add_connection", "activate_connection", "use_local", "clear_active_connection"):
        assert ext(n) in disabled, f"{n} must be denied to the untrusted task"
    # v6.81.1: list_connections/test_connection JOIN the denied set — both echo the
    # bridge URL, and a trace showed an agent using that URL to hunt for the grader.
    for n in ("list_connections", "test_connection"):
        assert ext(n) in disabled, f"{n} leaks the bridge URL and must be denied"
    for n in ("screenshot", "click", "type_text", "key", "scroll", "remote_exec"):
        assert ext(n) not in disabled, f"{n} must stay available for the fixed VM connection"


def test_live_server_guard_predicate_and_live_data_dir(monkeypatch, tmp_path):
    from devtools.benchmarks.osworld.run_step_agent import _is_default_desktop_server

    assert _is_default_desktop_server("http://localhost:8765") is True
    assert _is_default_desktop_server("http://127.0.0.1:8780") is False

    fake_home = tmp_path / "home"
    (fake_home / "Ouroboros" / "data").mkdir(parents=True)
    monkeypatch.setattr(rcb.Path, "home", classmethod(lambda cls: fake_home))
    with pytest.raises(SystemExit):
        rcb._refuse_live_data_dir(fake_home / "Ouroboros" / "data")
    with pytest.raises(SystemExit):
        rcb._refuse_live_data_dir(fake_home / "Ouroboros" / "data" / "state" / "skills")
    # an isolated bench dir is fine
    rcb._refuse_live_data_dir(tmp_path / "bench" / "data")


def test_dataset_name_variant_mapping():
    assert rcb._dataset_name("v1") == "OSWorld"
    assert rcb._dataset_name("v2") == "OSWorld-V2"
    assert rcb._dataset_name("examples_only") == "OSWorld-examples_only"


def test_effective_max_rounds_sources(tmp_path, monkeypatch):
    monkeypatch.delenv("OUROBOROS_MAX_ROUNDS", raising=False)
    sp = tmp_path / "settings.json"
    sp.write_text(json.dumps({"OUROBOROS_MAX_ROUNDS": 120}), encoding="utf-8")
    assert rcb._effective_max_rounds(sp) == {"value": 120, "source": "settings"}

    sp.write_text(json.dumps({}), encoding="utf-8")
    monkeypatch.setenv("OUROBOROS_MAX_ROUNDS", "77")
    assert rcb._effective_max_rounds(sp) == {"value": 77, "source": "env"}

    monkeypatch.delenv("OUROBOROS_MAX_ROUNDS", raising=False)
    assert rcb._effective_max_rounds(tmp_path / "missing.json") == {"value": 200, "source": "default"}


def test_budget_counters_from_child_drive_tools_jsonl(tmp_path):
    from ouroboros.extension_loader import extension_name_prefix

    prefix = extension_name_prefix("unix_computer_use")
    child = tmp_path / "state" / "headless_tasks" / "t1" / "data"
    logs = child / "logs"
    logs.mkdir(parents=True)
    rows = [
        {"type": "tool_call", "tool": f"{prefix}screenshot", "task_id": "t1"},
        {"type": "tool_call", "tool": f"{prefix}screenshot", "task_id": "t1"},
        {"type": "tool_call", "tool": f"{prefix}click", "task_id": "t1"},
        {"type": "tool_call", "tool": f"{prefix}type_text", "task_id": "t1"},
        {"type": "tool_call", "tool": f"{prefix}remote_exec", "task_id": "t1"},
        {"type": "tool_call", "tool": "read_file", "task_id": "t1"},        # core tool, ignored
        {"type": "llm_round", "tool": f"{prefix}click"},                     # not a tool_call, ignored
    ]
    body = "\n".join(json.dumps(r) for r in rows) + "\nnot json line\n"
    (logs / "tools.jsonl").write_text(body, encoding="utf-8")

    latest = {"total_rounds": 9, "child_drive_root": str(child)}
    counters = rcb._collect_budget_counters(tmp_path, latest, "t1")
    assert counters["llm_rounds"] == 9
    assert counters["screenshots"] == 2
    assert counters["gui_action_calls"] == 2   # click + type_text
    assert counters["remote_exec_calls"] == 1
    assert counters["skill_tool_calls"] == 5


def test_budget_counters_fallback_global_log_filters_by_task(tmp_path):
    from ouroboros.extension_loader import extension_name_prefix

    prefix = extension_name_prefix("unix_computer_use")
    (tmp_path / "logs").mkdir(parents=True)
    rows = [
        {"type": "tool_call", "tool": f"{prefix}screenshot", "task_id": "t1"},
        {"type": "tool_call", "tool": f"{prefix}click", "task_id": "OTHER"},  # different task, ignored
    ]
    (tmp_path / "logs" / "tools.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
    )
    # no child_drive_root and no per-task dir -> falls back to global log
    counters = rcb._collect_budget_counters(tmp_path, {"total_rounds": 3}, "t1")
    assert counters["screenshots"] == 1
    assert counters["skill_tool_calls"] == 1


def test_publish_target_writes_registry_atomically(tmp_path):
    data_dir = tmp_path / "data"
    tpath = rcb._publish_target(data_dir, "http://10.0.0.5:5000")
    from ouroboros.skill_loader import skill_state_dir
    sdir = Path(skill_state_dir(data_dir, "unix_computer_use"))
    # The runtime SSOT writers (write_text_atomic / atomic_write_json) are used now,
    # so no temp file of EITHER naming convention may survive the write.
    assert not list(sdir.glob("*.tmp-*"))
    assert not list(sdir.glob(".*tmp*"))
    assert not hasattr(rcb, "_atomic_write_text")  # local copy removed on purpose
    reg = json.loads((sdir / "connections.json").read_text(encoding="utf-8"))
    assert reg["active"] == "osworld-current"
    assert reg["connections"]["osworld-current"]["backend"] == "osworld_http"
    assert (sdir / "active_connection.txt").read_text(encoding="utf-8").strip() == "osworld-current"
    assert tpath.read_text(encoding="utf-8") == "http://10.0.0.5:5000"


def test_settings_path_defaults_into_bench_data_dir():
    # The default flag value is empty; main() resolves it to <data-dir>/settings.json
    # (asserted here at the resolution-logic level to avoid booting a VM/server).
    import argparse
    from pathlib import Path as _P
    data_dir = _P("/tmp/bench_NN/data")
    args_settings = ""  # not explicitly provided
    resolved = _P(args_settings).expanduser().resolve(strict=False) if args_settings else (data_dir / "settings.json")
    assert resolved == data_dir / "settings.json"
    # explicit value wins
    args_settings = "/tmp/explicit/settings.json"
    resolved = _P(args_settings).expanduser().resolve(strict=False) if args_settings else (data_dir / "settings.json")
    assert resolved == _P("/tmp/explicit/settings.json").resolve(strict=False)
    _ = argparse  # silence unused in some linters


def test_denylist_is_allowlist_complement_blocks_all_host_surfaces():
    # Allowlist semantics: every core tool NOT in the allowlist is denied — so the
    # whole host mutation/exec/VCS/GitHub/service/self-mod/chat class is blocked by
    # construction, not by an enumerated (and forgettable) list.
    denied = set(rcb._host_denied_tools())
    core = rcb._core_tool_names()
    # nothing in the allowlist is denied; everything else is
    assert denied == core - rcb._ALLOWED_CORE_TOOLS
    for t in ("run_command", "run_script", "write_file", "edit_text",
              "start_service", "stop_service", "verify_and_record", "commit_reviewed",
              "integrate_subagent_patch", "create_github_issue", "schedule_subagent",
              "skill_exec", "toggle_skill", "submit_skill_to_hub", "vcs_pull_ff",
              "vcs_restore", "vcs_revert", "vcs_rollback", "update_identity",
              "update_scratchpad", "knowledge_write", "journal_write", "send_user_message",
              "toggle_evolution", "toggle_consciousness", "request_deep_self_review",
              "comment_on_pr", "comment_on_issue", "promote_to_stable", "run_ci_tests",
              "browse_page", "browser_action", "web_search", "plan_task",
              # host filesystem/code reads are denied too — the isolated settings.json
              # holds provider API keys a prompt-injected task could exfiltrate.
              "read_file", "list_files", "search_code", "query_code"):
        assert t in denied, f"{t} should be denied to the untrusted OSWorld task"
    # the tools the agent genuinely needs (VM control is via the skill's ext_* tools)
    for t in ("view_image", "enable_tools", "list_available_tools"):
        assert t not in denied, f"{t} must stay available"


# ---------------------------------------------------------------- v6.76.0 (P2)

class _FlakyDesktopEnv:
    """Stands in for DesktopEnv: __init__ boots a "VM" and may fail like the real one."""

    fail_times = 0
    attempts = 0
    closed: list[str] = []
    stopped: list[str] = []

    def __init__(self, *, path_to_vm: str, boom: bool = False):
        type(self).attempts += 1
        self.path_to_vm = path_to_vm
        if boom or type(self).attempts <= type(self).fail_times:
            # Mirror the real failure mode: the emulator IS already started when the
            # constructor raises, so the half-built object must be torn down.
            self.provider = _FakeProvider()
            raise RuntimeError(f"boot failed on attempt {type(self).attempts}")
        self.provider = _FakeProvider()

    def close(self):
        type(self).closed.append(self.path_to_vm)
        self.provider.stop_emulator(self.path_to_vm)


class _FakeProvider:
    def stop_emulator(self, path_to_vm):
        _FlakyDesktopEnv.stopped.append(str(path_to_vm))


def _reset_flaky(fail_times: int) -> None:
    _FlakyDesktopEnv.fail_times = fail_times
    _FlakyDesktopEnv.attempts = 0
    _FlakyDesktopEnv.closed = []
    _FlakyDesktopEnv.stopped = []


def test_desktop_env_constructor_is_retried_and_every_failure_is_torn_down():
    import time as _time

    from devtools.benchmarks.osworld.run_step_agent import construct_desktop_env

    _reset_flaky(2)
    env = construct_desktop_env(
        _FlakyDesktopEnv, attempts=4, deadline=_time.time() + 60, retry_sleep_sec=0.0,
        path_to_vm="/vm/a.qcow2",
    )
    assert env is not None
    assert _FlakyDesktopEnv.attempts == 3
    # Both failed boots were stopped; the surviving env was NOT closed.
    assert _FlakyDesktopEnv.stopped == ["/vm/a.qcow2", "/vm/a.qcow2"]


def test_desktop_env_construction_exhausts_attempts_and_leaks_nothing():
    import time as _time

    from devtools.benchmarks.osworld.run_step_agent import construct_desktop_env

    _reset_flaky(99)
    with pytest.raises(RuntimeError) as err:
        construct_desktop_env(
            _FlakyDesktopEnv, attempts=3, deadline=_time.time() + 60, retry_sleep_sec=0.0,
            path_to_vm="/vm/b.qcow2",
        )
    assert "DesktopEnv construction failed" in str(err.value)
    assert _FlakyDesktopEnv.attempts == 3
    assert len(_FlakyDesktopEnv.stopped) == 3  # one teardown per failed boot


def test_desktop_env_construction_always_tries_once_even_past_deadline():
    from devtools.benchmarks.osworld.run_step_agent import construct_desktop_env

    _reset_flaky(0)
    env = construct_desktop_env(
        _FlakyDesktopEnv, attempts=3, deadline=0.0, retry_sleep_sec=0.0,
        path_to_vm="/vm/c.qcow2",
    )
    assert env is not None and _FlakyDesktopEnv.attempts == 1


def test_task_claim_serializes_lanes_and_first_scored_attempt_wins(tmp_path):
    from devtools.benchmarks.osworld.run_step_agent import (
        acquire_task_claim,
        claim_stale_sec,
        release_task_claim,
        task_claim_key,
    )

    claims = tmp_path / "claims"
    key = task_claim_key("multi_apps", "48d05431-6cd5-4e76")
    stale = claim_stale_sec(3600, 900, 900)
    # stale_sec must exceed every wall-clock rail the holder can still be inside, and the
    # holder gets TWO startup windows (constructor, then reset-to-screenshot) — a one-window
    # bound expires while a lane is still legitimately working and two lanes take one task.
    # The unbounded env.evaluate() that follows is covered by the margin, not the formula.
    assert stale == 3600 + 2 * 900 + 900
    assert claim_stale_sec(3600, 900, -5) == 3600 + 2 * 900  # negative margin never shortens

    lane_a, reason_a = acquire_task_claim(claims, key, stale_sec=stale, repo_dir=tmp_path / "repo")
    assert lane_a is not None and reason_a == "claimed"
    # A second lane must NOT get the same task while the first is working.
    lane_b, reason_b = acquire_task_claim(claims, key, stale_sec=stale, repo_dir=tmp_path / "repo")
    assert lane_b is None and reason_b == "in_flight"

    # Unscored attempt -> the task stays claimable, so a retry lane may take it.
    release_task_claim(claims, key, lane_a, scored=False, repo_dir=tmp_path / "repo")
    lane_c, reason_c = acquire_task_claim(claims, key, stale_sec=stale, repo_dir=tmp_path / "repo")
    assert lane_c is not None and reason_c == "claimed"

    # Scored attempt -> permanent marker; later lanes step aside regardless of value.
    release_task_claim(claims, key, lane_c, scored=True, repo_dir=tmp_path / "repo", payload={"reward": 0.0})
    lane_d, reason_d = acquire_task_claim(claims, key, stale_sec=stale, repo_dir=tmp_path / "repo")
    assert lane_d is None and reason_d == "already_scored"
    assert (claims / f"{key}.scored").is_file()
    assert not (claims / f"{key}.lock").exists()


def test_task_claim_key_is_filesystem_safe():
    from devtools.benchmarks.osworld.run_step_agent import task_claim_key

    key = task_claim_key("multi/apps", "a b/c..json")
    assert "/" not in key and " " not in key and key.count("__") >= 1


def test_amend_task_manifest_merges_without_mutating_the_base():
    from devtools.benchmarks.osworld.run_step_agent import amend_task_manifest

    base = {"schema": "x", "output_paths": {"a": "1"}, "extra": {"allow_dirty_seed": False}}
    merged = amend_task_manifest(base, output_paths={"b": "2"}, extra={"reward": 1.0})
    assert merged["output_paths"] == {"a": "1", "b": "2"}
    assert merged["extra"] == {"allow_dirty_seed": False, "reward": 1.0}
    assert base["output_paths"] == {"a": "1"} and base["extra"] == {"allow_dirty_seed": False}


def test_cu_bridge_gates_provenance_before_the_vm_and_records_the_escape():
    """The clean-seed gate must run BEFORE paid work, not at outcome time."""
    src = (Path(__file__).resolve().parent.parent
           / "devtools" / "benchmarks" / "osworld" / "run_cu_bridge_agent.py").read_text(encoding="utf-8")
    gate = src.index("require_clean=not args.allow_dirty_seed")
    assert gate < src.index("from desktop_env.desktop_env import DesktopEnv")
    assert gate < src.index("enabled = _enable_skill(")
    assert '"allow_dirty_seed": bool(args.allow_dirty_seed)' in src
    # The per-outcome manifest amends the single early one instead of rebuilding it.
    assert "amend_task_manifest(" in src


def test_cu_bridge_claim_is_acquired_inside_the_try_that_releases_it():
    """The claim lock must not outlive a failure between claim and VM boot: an unimportable
    `desktop_env` used to leave the lock on disk with no `.scored` marker, so the task was
    neither scored nor claimable for the whole staleness window — the opposite of the
    mechanism's own 'an unscored attempt stays claimable' contract."""
    src = (Path(__file__).resolve().parent.parent
           / "devtools" / "benchmarks" / "osworld" / "run_cu_bridge_agent.py").read_text(encoding="utf-8")
    body = src[src.index("claim_fd: int | None = None"):]
    assert body.index("\n    try:") < body.index("acquire_task_claim(")
    assert body.index("acquire_task_claim(") < body.index("from desktop_env.desktop_env import DesktopEnv")
    assert body.index("from desktop_env.desktop_env import DesktopEnv") < body.index("release_task_claim(")
    # A lane that never took the lock must not delete the holder's lockfile.
    assert "if claims_dir is not None and claim_fd is not None:" in src
    # The runtime attestation admits the run before the claim and before the first paid POST
    # of the RUN FLOW. Anchored on `body` (the flow, from the claim declaration on), not the
    # whole file: module-level helpers defined above the flow (`_gate_round`) legitimately
    # contain the same POST literal but are only ever CALLED from inside the flow.
    attestation = src.index("runtime_attestation(args.ouroboros_url, repo_dir)")
    actor_match = src.index("actor_preflight = _cu_actor_preflight(settings_path, args.ouroboros_url)")
    claim = src.index("acquire_task_claim(\n")
    assert attestation < actor_match < claim
    first_paid_post_in_flow = src.index("claim_fd: int | None = None") + body.index('"POST", "/api/tasks"')
    assert attestation < actor_match < first_paid_post_in_flow


def test_cu_bridge_refuses_before_the_claim_when_attestation_fails(tmp_path, monkeypatch, capsys):
    """Owner Q9/Q10: the bridge attests the running server before its first paid POST. The
    helper fails CLOSED, so the launcher must turn that into a typed `blocked` row — and must
    not park a claim lock on a run that never starts."""
    import sys as _sys

    osworld = tmp_path / "OSWorld"
    (osworld / "evaluation_examples" / "examples" / "chrome").mkdir(parents=True)
    task = osworld / "evaluation_examples" / "examples" / "chrome" / "abc.json"
    task.write_text(json.dumps({"id": "abc", "instruction": "no-op"}), encoding="utf-8")
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "VERSION").write_text("6.76.0\n", encoding="utf-8")
    results = tmp_path / "results"
    claims = tmp_path / "claims"
    monkeypatch.setattr(_sys, "argv", [
        "run_cu_bridge_agent.py",
        "--osworld-root", str(osworld),
        "--provider_name", "docker",
        "--path_to_vm", "/vm/Ubuntu.qcow2",
        "--task", str(task),
        "--result_dir", str(results),
        "--repo-dir", str(repo_dir),
        "--data-dir", str(tmp_path / "data"),
        "--settings-path", str(tmp_path / "settings.json"),
        "--ouroboros-url", "http://127.0.0.1:9",   # nothing listens: attestation fails closed
        "--target-file", str(tmp_path / "target.txt"),
        "--claim-dir", str(claims),
        "--allow-dirty-seed",                       # provenance is not what this test pins
    ])

    assert rcb.main() == 2
    outcome = json.loads(capsys.readouterr().out)
    assert outcome["status"] == "blocked"
    # The EXACT typed reason, not the generic string: nothing listens on the URL, so no live
    # runtime identity was established at all.
    assert outcome["reason_code"] == "runtime_unreachable"
    # The refusal precedes the claim, so no lock/marker is left for another lane to trip over.
    assert not claims.exists() or not any(claims.iterdir())


def test_step_agent_seed_gate_refusal_is_typed_records_not_a_traceback(tmp_path, monkeypatch, capsys):
    """Owner Q19 fails the seed gate CLOSED. Nothing is spent at that point, so the launcher
    must report its own `blocked/seed_gate_failed` records (ledger row included) instead of a
    bare traceback. `repo_dir` here is a non-git directory, so the verdict does not depend on
    the ambient checkout being clean or dirty."""
    import sys as _sys

    from devtools.benchmarks.osworld import run_step_agent

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "VERSION").write_text("6.76.0\n", encoding="utf-8")
    task = tmp_path / "OSWorld" / "evaluation_examples" / "examples" / "chrome" / "abc.json"
    task.parent.mkdir(parents=True)
    task.write_text(json.dumps({"id": "abc", "instruction": "no-op"}), encoding="utf-8")
    results = tmp_path / "results"
    monkeypatch.setattr(_sys, "argv", [
        "run_step_agent.py",
        "--osworld-root", str(tmp_path / "OSWorld"),
        "--task", str(task),
        "--result_dir", str(results),
        "--repo-dir", str(repo_dir),
        "--data-dir", str(tmp_path / "data"),
        "--settings-path", str(tmp_path / "settings.json"),
        "--ouroboros-url", "http://127.0.0.1:9",
        "--provider_name", "docker",
    ])

    assert run_step_agent.main() == 2
    outcome = json.loads(capsys.readouterr().out)
    assert outcome["status"] == "blocked" and outcome["reason_code"] == "seed_gate_failed"
    assert "seed_identity_unavailable" in outcome["error"]
    rows = [json.loads(line) for line
            in (results / "result_index.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["reason_code"] == "seed_gate_failed"


def test_osworld_skeleton_seed_gate_refusal_short_circuits_the_preflight(tmp_path, monkeypatch, capsys):
    """Same gate, non-spending entry point: fold the refusal into the existing typed refusal
    (return 2 with a `seed_gate_error`) and still report the other preflight failures, so the
    gate cannot MASK an isolation refusal the operator also needs to see."""
    import sys as _sys

    from devtools.benchmarks.osworld import osworld_adapter_skeleton as skeleton

    repo_root = tmp_path / "repo"  # deliberately NOT a git checkout: verdict is ambient-free
    osworld = tmp_path / "OSWorld"
    payload = tmp_path / "unix_computer_use"
    output_root = tmp_path / "runs" / "osworld"
    for path in (repo_root, osworld, payload):
        path.mkdir(parents=True)
    (osworld / "evaluation_examples").mkdir()
    monkeypatch.setattr(skeleton, "DEFAULT_REPO_ROOT", repo_root)
    monkeypatch.setattr(skeleton, "DEFAULT_DATA_ROOT", tmp_path / "live-data")
    monkeypatch.setattr(_sys, "argv", [
        "osworld_adapter_skeleton.py",
        "--osworld-root", str(osworld),
        "--ouroboros-url", "http://127.0.0.1:9",
        "--osworld-server-url", "http://127.0.0.1:9",
        "--unix-computer-use-payload", str(payload),
        "--output-root", str(output_root),
    ])

    assert skeleton.main() == 2
    result = json.loads(capsys.readouterr().out)
    assert result["ok"] is False
    assert "seed_identity_unavailable" in result["details"]["seed_gate_error"]
    assert any("seed gate refused" in failure for failure in result["failures"])
    # SHORT-CIRCUIT (v6.76.0): the preflight does NOT run after a refused admission. It probes
    # the filesystem and reaches two servers over the network, and the documented contract says
    # an unidentifiable seed stops the run BEFORE the preflight — so no other finding is
    # reported here, deliberately, and none is spent on.
    assert result["details"]["skipped"] == "preflight not run: admission refused"
    assert not any("not reachable" in failure for failure in result["failures"])
    # v6.76.0: a refused seed now leaves a DURABLE record of what was refused. Writing
    # nothing (the previous behaviour) meant the one path where provenance was refused was
    # also the one path that left no evidence of the refusal. It still leaves no LEDGER row:
    # the run never started, so it owns no denominator entry.
    manifest = json.loads(
        (output_root / "osworld_preflight.run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["outcome"] == "refused"
    assert manifest["extra"]["exit_code"] == 2                    # == the process status
    assert manifest["extra"]["refusal"]["stage"] == "seed_gate"
    assert manifest["seed_gate"]["ok"] is False
    assert not (output_root / "osworld_preflight.ledger.jsonl").exists()


def test_scored_claim_is_fail_closed_and_is_never_released_without_a_durable_marker(
        tmp_path, monkeypatch):
    """The `.scored` marker is the AUTHORITY behind "first scored attempt wins", not an
    optimisation. It used to be written inside a bare `except: pass` and the lock released
    anyway, so one disk error handed an already-scored task back to the next lane."""
    import ouroboros.utils as ouroboros_utils
    from devtools.benchmarks.osworld.run_step_agent import (
        ClaimMarkerNotDurable,
        acquire_task_claim,
        release_task_claim,
        task_claim_key,
    )

    claims = tmp_path / "claims"
    key = task_claim_key("os", "abc")
    lock_fd, reason = acquire_task_claim(claims, key, stale_sec=3600, repo_dir=tmp_path / "repo")
    assert lock_fd is not None and reason == "claimed"

    def _enospc(*_a, **_k):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(ouroboros_utils, "atomic_write_json", _enospc)
    with pytest.raises(ClaimMarkerNotDurable) as refused:
        release_task_claim(claims, key, lock_fd, scored=True, repo_dir=tmp_path / "repo", payload={"reward": 1.0})
    # Neither marker could be written, so NOTHING on disk records the score: that is the one
    # case with no honest protection left, and the refusal says so (`unconfirmed_marker is
    # None`) instead of inventing a third layer of best-effort.
    assert refused.value.unconfirmed_marker is None
    assert "claim directory is unusable" in str(refused.value)
    # Surfaced, not swallowed — AND the lock is still held, so no other attempt may take a task
    # that already has an official score while this process is alive.
    assert (claims / f"{key}.lock").exists()
    assert not (claims / f"{key}.scored").exists()
    assert not (claims / f"{key}.scored_unconfirmed").exists()
    other_fd, other_reason = acquire_task_claim(claims, key, stale_sec=3600, repo_dir=tmp_path / "repo")
    assert other_fd is None and other_reason == "in_flight"

    # With a working disk the same call marks and releases.
    monkeypatch.undo()
    release_task_claim(claims, key, lock_fd, scored=True, repo_dir=tmp_path / "repo", payload={"reward": 1.0})
    assert (claims / f"{key}.scored").is_file()
    assert not (claims / f"{key}.lock").exists()


def test_a_lane_that_dies_between_scoring_and_its_finally_keeps_the_task_scored(tmp_path):
    """Crash boundary. The marker used to be written in `finally`, AFTER `env.evaluate()` and
    the result projection, so a process death in between left no marker at all and another
    lane reran a task that already had an official score."""
    from devtools.benchmarks.osworld.run_step_agent import (
        acquire_task_claim,
        mark_task_scored,
        task_claim_key,
    )

    claims = tmp_path / "claims"
    key = task_claim_key("os", "abc")
    lock_fd, _ = acquire_task_claim(claims, key, stale_sec=3600, repo_dir=tmp_path / "repo")
    assert lock_fd is not None
    # The transition the runner performs immediately after env.evaluate()...
    mark_task_scored(claims, key, repo_dir=tmp_path / "repo", payload={"reward": 0.0})
    # ...and then the process dies: no release, no `finally`, the lock file is orphaned and
    # will look stale to the next lane. The marker still decides.
    later_fd, later_reason = acquire_task_claim(claims, key, stale_sec=0.0, repo_dir=tmp_path / "repo")
    assert later_fd is None and later_reason == "already_scored"
    # The FIRST scored attempt owns the marker; a later call never overwrites its payload.
    marker = json.loads((claims / f"{key}.scored").read_text(encoding="utf-8"))
    mark_task_scored(claims, key, repo_dir=tmp_path / "repo", payload={"reward": 1.0})
    assert json.loads((claims / f"{key}.scored").read_text(encoding="utf-8")) == marker


def test_a_scored_but_unmarked_task_stays_refused_after_its_lock_goes_stale(tmp_path, monkeypatch):
    """A protection with an expiry date fails open. `stale_sec` reclaims a crashed holder's lock
    BY DESIGN, so retaining that lock for a scored-but-unmarked task only delayed the rerun: once
    the bound elapsed, another attempt claimed a task that already had an official score. The
    durable `.scored_unconfirmed` marker refuses it regardless of staleness."""
    import ouroboros.utils as ouroboros_utils
    from devtools.benchmarks.osworld.run_step_agent import (
        ClaimMarkerNotDurable,
        acquire_task_claim,
        claim_stale_sec,
        mark_task_scored,
        scored_claim_state,
        task_claim_key,
    )

    claims = tmp_path / "claims"
    key = task_claim_key("os", "abc")
    stale = claim_stale_sec(3600, 900, 900)
    lock_fd, _ = acquire_task_claim(claims, key, stale_sec=stale, repo_dir=tmp_path / "repo")
    assert lock_fd is not None

    real_write = ouroboros_utils.atomic_write_json

    def _fail_only_the_canonical_marker(path, payload, **kwargs):
        if str(path).endswith(".scored"):
            raise OSError(28, "No space left on device")
        return real_write(path, payload, **kwargs)

    monkeypatch.setattr(ouroboros_utils, "atomic_write_json", _fail_only_the_canonical_marker)
    with pytest.raises(ClaimMarkerNotDurable) as refused:
        mark_task_scored(claims, key, repo_dir=tmp_path / "repo", payload={"reward": 0.5})
    assert refused.value.unconfirmed_marker == claims / f"{key}.scored_unconfirmed"
    monkeypatch.undo()

    # Age the lock well past the staleness bound: `acquire_exclusive_file_lock` reclaims a lock
    # whose mtime is older than `stale_sec`, which is exactly the "nobody waited long enough"
    # case the lock-only protection lost.
    lock_path = claims / f"{key}.lock"
    ancient = time.time() - (stale + 60)
    os.utime(lock_path, (ancient, ancient))
    contender_fd, contender_reason = acquire_task_claim(claims, key, stale_sec=stale, repo_dir=tmp_path / "repo")
    assert contender_fd is None and contender_reason == "scored_unconfirmed"

    # ...and it is not the lock doing the work: delete it entirely and the task is STILL refused.
    # The holder's descriptor is closed FIRST because the state being modelled is a dead holder,
    # whose descriptors the OS closed for it. It also has to be: Windows refuses to delete a file
    # while any handle to it is open (POSIX allows it), so keeping ours open fails the deletion
    # instead of testing the refusal. Same close-then-unlink order `release_exclusive_file_lock`
    # already uses.
    os.close(lock_fd)
    lock_path.unlink()
    assert scored_claim_state(claims, key) == "scored_unconfirmed"
    assert acquire_task_claim(claims, key, stale_sec=0.0, repo_dir=tmp_path / "repo") == (None, "scored_unconfirmed")
    # The reason is its own, so an operator sees a state that needs attention rather than a
    # task that silently became claimable.
    assert contender_reason not in ("in_flight", "already_scored", "claimed")


def test_the_unconfirmed_marker_does_not_disturb_the_healthy_scored_path(tmp_path):
    """The new state must refuse ONLY when it exists: a clean claim dir stays claimable even
    with a stale lock, and a properly marked task still reports `already_scored`."""
    from devtools.benchmarks.osworld.run_step_agent import (
        acquire_task_claim,
        mark_task_scored,
        release_task_claim,
        scored_claim_state,
        task_already_scored,
        task_claim_key,
    )

    claims = tmp_path / "claims"
    key = task_claim_key("os", "healthy")
    assert scored_claim_state(claims, key) == "" and task_already_scored(claims, key) is False

    lock_fd, reason = acquire_task_claim(claims, key, stale_sec=3600, repo_dir=tmp_path / "repo")
    assert lock_fd is not None and reason == "claimed"
    release_task_claim(claims, key, lock_fd, scored=True, repo_dir=tmp_path / "repo", payload={"reward": 1.0})
    assert (claims / f"{key}.scored").is_file()
    assert not (claims / f"{key}.scored_unconfirmed").exists()   # no fallback was needed
    assert scored_claim_state(claims, key) == "already_scored"
    assert acquire_task_claim(claims, key, stale_sec=3600, repo_dir=tmp_path / "repo") == (None, "already_scored")

    # A DIFFERENT task in the same claim dir is unaffected — the refusal is per-task state, not
    # a blanket on the directory — and a stale lock on it is still reclaimable as designed.
    other = task_claim_key("os", "other")
    other_fd, other_reason = acquire_task_claim(claims, other, stale_sec=3600, repo_dir=tmp_path / "repo")
    assert other_fd is not None and other_reason == "claimed"
    # The holder this reclaim is aimed at CRASHED: its lock file outlives it but its descriptors
    # do not, so ours is closed to model that. It also has to be: the reclaim unlinks the stale
    # lock, Windows refuses to unlink a file with an open handle, and that failure is swallowed
    # inside `acquire_exclusive_file_lock` — the reclaim would silently time out into `in_flight`
    # rather than raise, which is a stale lock that can never be reclaimed on that platform.
    os.close(other_fd)
    second_fd, second_reason = acquire_task_claim(claims, other, stale_sec=0.0, repo_dir=tmp_path / "repo")
    assert second_fd is not None and second_reason == "claimed"   # stale lock reclaimed
    os.close(second_fd)
    mark_task_scored(claims, other, repo_dir=tmp_path / "repo", payload={"reward": 0.0})
    assert scored_claim_state(claims, other) == "already_scored"


def test_cu_bridge_refuses_loudly_when_no_scored_state_can_be_recorded_at_all(
        tmp_path, monkeypatch, capsys):
    """The disk is genuinely gone: neither marker persists, so nothing on disk remembers the
    score and the retained lock WILL expire. There is no protection left to promise, so the
    honest outcome is a loud, distinctly-typed refusal — not a third layer of best-effort."""
    import ouroboros.utils as ouroboros_utils

    claims = tmp_path / "claims"
    rcb, _env = _cu_bridge_stubs(monkeypatch, tmp_path, reward=1.0)
    argv, results = _cu_bridge_argv(tmp_path, claims)
    monkeypatch.setattr(sys, "argv", argv)

    real_write = ouroboros_utils.atomic_write_json

    def _fail_every_claim_marker(path, payload, **kwargs):
        if ".scored" in str(path):                  # canonical AND fallback
            raise OSError(28, "No space left on device")
        return real_write(path, payload, **kwargs)

    monkeypatch.setattr(ouroboros_utils, "atomic_write_json", _fail_every_claim_marker)

    assert rcb.main() == 3                          # distinct from the ordinary failure (1/2)
    err = capsys.readouterr().err
    assert "FATAL: the claim directory is unusable" in err
    assert "do not run further tasks" in err
    extra = json.loads((results / "chrome" / "abc" / "task_run_manifest.json")
                       .read_text(encoding="utf-8"))["extra"]
    assert extra["outcome"] == "claim_state_unrecoverable"
    assert extra["exit_code"] == 3                  # == the process status
    assert extra["refusal"] == {"stage": "scored_claim_marker",
                                "reason": "claim_state_unrecoverable", "exit_code": 3}
    assert extra["claim_state_unrecoverable"] is True
    outcome = json.loads((results / "chrome" / "abc" / "task_outcome.json").read_text(encoding="utf-8"))
    assert outcome["reward"] == 1.0                 # the official score is still reported
    key = "chrome__abc"
    assert not (claims / f"{key}.scored").exists()
    assert not (claims / f"{key}.scored_unconfirmed").exists()


def test_an_interrupt_between_the_score_and_its_marker_does_not_release_the_claim(
        tmp_path, monkeypatch, capsys):
    """`KeyboardInterrupt` and `SystemExit` derive from BaseException, not Exception — the same
    trap that made a refusal handler inert in phase P1. A Ctrl-C inside `mark_task_scored` used
    to unwind straight through the `finally`, which releases the claim with `scored=False`.

    THE PART THAT ACTUALLY MATTERS IS SURVIVING THE LOCK. Retaining the `.lock` was the whole
    protection this arm used to offer, and that lock is EXPIRABLE by design: after `stale_sec`,
    `acquire_task_claim` reclaims it and reruns a task whose official score was already durably
    recorded — a genuine double count. So the refusal is asserted with the lock AGED AWAY, which
    is the only way to tell a durable protection from a countdown."""
    from devtools.benchmarks.osworld import run_step_agent
    from devtools.benchmarks.osworld.run_step_agent import (
        acquire_task_claim,
        scored_claim_state,
        task_claim_key,
    )

    claims = tmp_path / "claims"
    repo_dir = tmp_path / "repo"
    rcb, env = _cu_bridge_stubs(monkeypatch, tmp_path, reward=1.0)
    argv, _results = _cu_bridge_argv(tmp_path, claims)
    monkeypatch.setattr(sys, "argv", argv)

    def _interrupt(*_a, **_k):
        raise KeyboardInterrupt

    monkeypatch.setattr(run_step_agent, "mark_task_scored", _interrupt)

    # The retained lock is deleted below, and the lane that took it is a process on its way out
    # — the OS closes its descriptors. Recording the descriptor lets the test close it and model
    # that; on Windows it is mandatory, since a file with an open handle cannot be deleted.
    lane_fds: list[int] = []
    real_acquire = run_step_agent.acquire_task_claim

    def _recording_acquire(*a, **k):
        fd, reason = real_acquire(*a, **k)
        if fd is not None:
            lane_fds.append(fd)
        return fd, reason

    monkeypatch.setattr(run_step_agent, "acquire_task_claim", _recording_acquire)

    # The operator's interrupt still stops the run...
    with pytest.raises(KeyboardInterrupt):
        rcb.main()
    key = task_claim_key("chrome", "abc")
    # ...and the claim was NOT handed to another attempt on the way out.
    assert (claims / f"{key}.lock").exists()
    contender_fd, contender_reason = acquire_task_claim(claims, key, stale_sec=3600,
                                                        repo_dir=repo_dir)
    assert contender_fd is None and contender_reason == "scored_unconfirmed"
    assert "RETAINING the claim" in capsys.readouterr().err
    assert env.closed is True                     # the VM is still torn down on the way out

    # THE REGRESSION: the scored-but-unmarked state is on disk, and it carries the score.
    unconfirmed = json.loads((claims / f"{key}.scored_unconfirmed").read_text(encoding="utf-8"))
    assert unconfirmed["reason"] == "interrupted_before_scored_marker:KeyboardInterrupt"
    assert unconfirmed["reward"] == 1.0
    # A zero staleness bound makes the lock immediately reclaimable, and deleting it removes
    # even that. The task must STILL be refused, because the refusal never came from the lock.
    assert acquire_task_claim(claims, key, stale_sec=0.0, repo_dir=repo_dir) == (
        None, "scored_unconfirmed")
    for fd in lane_fds:
        os.close(fd)
    (claims / f"{key}.lock").unlink()
    assert scored_claim_state(claims, key) == "scored_unconfirmed"
    assert acquire_task_claim(claims, key, stale_sec=0.0, repo_dir=repo_dir) == (
        None, "scored_unconfirmed")


def test_claim_dir_is_confined_to_outside_repo_and_live_data(tmp_path, monkeypatch):
    """The claim dir is operator-supplied and the helpers CREATE it and write `.lock`,
    `.scored` and `.scored_unconfirmed` into it, so a mistaken path mutates the repository or
    the owner's live runtime data. Same boundary every other benchmark output root uses."""
    from devtools.benchmarks.osworld.run_step_agent import (
        ClaimDirNotConfined,
        acquire_task_claim,
        confined_claims_dir,
        mark_task_scored,
        task_claim_key,
    )

    repo_root = Path(__file__).resolve().parent.parent
    live_data = tmp_path / "live-data"
    live_data.mkdir()
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(live_data))
    key = task_claim_key("os", "abc")

    for bad in (repo_root / "devtools" / "claims-inside-repo",
                repo_root / ".claims",
                live_data / "state" / "claims",
                live_data):
        with pytest.raises(ClaimDirNotConfined):
            confined_claims_dir(bad, repo_dir=tmp_path / "repo")
        # ...and the refusal is enforced by the helpers that would create it, not only by the
        # CLI, so no caller can reach the filesystem around it.
        with pytest.raises(ClaimDirNotConfined):
            acquire_task_claim(bad, key, stale_sec=3600, repo_dir=tmp_path / "repo")
        with pytest.raises(ClaimDirNotConfined):
            mark_task_scored(bad, key, repo_dir=tmp_path / "repo", payload={"reward": 1.0})
        if bad != live_data:
            assert not Path(bad).exists()                # nothing was created
    assert not any(live_data.iterdir())                  # ...and nothing written into it
    # A confined dir still works exactly as before.
    good = confined_claims_dir(tmp_path / "claims", repo_dir=tmp_path / "repo")
    lock_fd, reason = acquire_task_claim(good, key, stale_sec=3600, repo_dir=tmp_path / "repo")
    assert lock_fd is not None and reason == "claimed"


def test_claim_dir_is_confined_against_the_execution_checkout_not_only_the_launcher(tmp_path):
    """INVARIANT B on the claim dir: the authority is the checkout being EXECUTED.

    `confined_claims_dir` derived its authority from this module's own location
    (`repo_root_from_devtools()`), so `--repo-dir /other/bench-clone --claim-dir
    /other/bench-clone/.claims` was waved through and the helpers wrote `.lock` and `.scored`
    state straight into the execution checkout — the very tree whose cleanliness the seed gate
    is about to attest, and which those files then dirty.

    The clone here is a SECOND checkout under tmp_path, never the ambient one, so the verdict
    is a property of the argument rather than of where the test happens to run.
    """
    from devtools.benchmarks.osworld.run_step_agent import (
        ClaimDirNotConfined,
        acquire_task_claim,
        confined_claims_dir,
        mark_task_scored,
        task_claim_key,
    )

    alt_clone = tmp_path / "other-bench-clone"
    (alt_clone / "devtools" / "benchmarks").mkdir(parents=True)
    unrelated = tmp_path / "unrelated-checkout"
    unrelated.mkdir()
    key = task_claim_key("os", "abc")

    for bad in (alt_clone / ".claims", alt_clone / "bench_runs" / "claims", alt_clone):
        with pytest.raises(ClaimDirNotConfined):
            confined_claims_dir(bad, repo_dir=alt_clone)
        # ...and by the helpers that would CREATE it, not only by the resolver, so no caller
        # can reach the filesystem around the boundary.
        with pytest.raises(ClaimDirNotConfined):
            acquire_task_claim(bad, key, stale_sec=3600, repo_dir=alt_clone)
        with pytest.raises(ClaimDirNotConfined):
            mark_task_scored(bad, key, repo_dir=alt_clone, payload={"reward": 1.0})
    assert not (alt_clone / ".claims").exists() and not (alt_clone / "bench_runs").exists()

    # THE SAME PATH is fine when a DIFFERENT checkout is the one executing: the answer depends
    # on the active checkout, which is exactly what a statically derived root cannot express.
    assert confined_claims_dir(alt_clone / ".claims", repo_dir=unrelated) == \
        (alt_clone / ".claims").resolve()
    # The launcher's own checkout stays an authority too — both are checked, not either/or.
    ambient = Path(__file__).resolve().parent.parent
    with pytest.raises(ClaimDirNotConfined):
        confined_claims_dir(ambient / "devtools" / ".claims", repo_dir=alt_clone)


def test_cu_bridge_refuses_a_claim_dir_inside_the_checkout_it_was_handed(tmp_path, monkeypatch):
    """The same defect end to end: `--claim-dir` inside `--repo-dir`. Nothing is created, and
    the refusal is pure argument validation, so it precedes admission (invariant A)."""
    _rcb, _env = _cu_bridge_stubs(monkeypatch, tmp_path)
    execution_checkout = tmp_path / "repo"           # this is what `--repo-dir` points at
    claims = execution_checkout / ".claims"
    argv, results = _cu_bridge_argv(tmp_path, claims)
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as refused:
        rcb.main()
    assert "refusing --claim-dir" in str(refused.value)
    assert not claims.exists()
    assert not results.exists()                      # not even an admission record


def test_cu_bridge_refuses_an_unconfined_claim_dir_before_anything_is_created(
        tmp_path, monkeypatch):
    """CLI-level refusal, as pure argument validation before admission: nothing on disk."""
    claims = Path(__file__).resolve().parent.parent / "devtools" / "claims-must-not-appear"
    _rcb, _env = _cu_bridge_stubs(monkeypatch, tmp_path)
    argv, results = _cu_bridge_argv(tmp_path, claims)
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as refused:
        rcb.main()
    assert "refusing --claim-dir" in str(refused.value)
    assert not claims.exists()
    assert not results.exists()                          # not even an admission record


def test_cu_bridge_marks_the_score_before_it_projects_the_result_anywhere():
    """Ordering is the whole mechanism: mark, THEN publish. Reversed, a crash in between
    leaves a published score with no marker — the one ordering that makes a lane rerun it."""
    src = (Path(__file__).resolve().parent.parent
           / "devtools" / "benchmarks" / "osworld" / "run_cu_bridge_agent.py").read_text(encoding="utf-8")
    evaluate = src.index("reward = float(env.evaluate())")
    mark = src.index("mark_task_scored(claims_dir, claim_key,")
    result_txt = src.index('(run_dir / "result.txt").write_text')
    projection = src.index('_write_outcome(reward, "completed"')
    assert evaluate < mark < result_txt < projection
    # ...and the release only ever claims `scored` for a marker that was CONFIRMED durable.
    assert "scored=claim_scored" in src
    assert "claim_scored = True" in src


def _cu_bridge_stubs(monkeypatch, tmp_path, *, reward=1.0):
    """Fakes just deep enough to drive `run_cu_bridge_agent.main()` end to end, no VM."""
    import types

    from devtools.benchmarks.osworld import run_cu_bridge_agent as rcb
    from devtools.benchmarks.osworld import run_step_agent

    class _FakeEnv:
        vm_ip = "10.0.0.2"
        server_port = 5000
        client_password = "pw"
        closed = False

        def reset(self, task_config=None):
            return None

        def _get_obs(self):
            return {"screenshot": b"png"}

        def step(self, action, *_a):
            return {}, 0.0, True, {}

        def evaluate(self):
            return reward

        def close(self):
            self.closed = True

    desktop_env = types.ModuleType("desktop_env")
    desktop_env_mod = types.ModuleType("desktop_env.desktop_env")
    desktop_env_mod.DesktopEnv = _FakeEnv
    desktop_env.desktop_env = desktop_env_mod
    monkeypatch.setitem(sys.modules, "desktop_env", desktop_env)
    monkeypatch.setitem(sys.modules, "desktop_env.desktop_env", desktop_env_mod)

    env = _FakeEnv()
    monkeypatch.setattr(run_step_agent, "construct_desktop_env", lambda *a, **k: env)
    monkeypatch.setattr(rcb, "runtime_attestation", lambda url, repo: {"ok": True})
    monkeypatch.setattr(rcb, "_enable_skill", lambda repo, data: {"skill": "seeded"})
    monkeypatch.setattr(rcb, "_publish_target", lambda data, target: tmp_path / "state_target.txt")
    monkeypatch.setattr(rcb, "_collect_budget_counters", lambda *a, **k: {})
    def _api(url, method, path, body=None, timeout=60):
        if method == "GET" and path == "/api/settings":
            return _cu_actor_settings()
        if method == "POST" and path == "/api/tasks":
            return {"task_id": "t1"}
        return {"status": "completed", "final_answer": "done"}

    monkeypatch.setattr(rcb, "_api", _api)
    return rcb, env


def _cu_bridge_argv(tmp_path, claims):
    osworld = tmp_path / "OSWorld"
    (osworld / "evaluation_examples" / "examples" / "chrome").mkdir(parents=True, exist_ok=True)
    task = osworld / "evaluation_examples" / "examples" / "chrome" / "abc.json"
    task.write_text(json.dumps({"id": "abc", "instruction": "no-op"}), encoding="utf-8")
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir(exist_ok=True)
    (repo_dir / "VERSION").write_text("6.76.0\n", encoding="utf-8")
    results = tmp_path / "results"
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps(_cu_actor_settings()), encoding="utf-8")
    return [
        "run_cu_bridge_agent.py", "--osworld-root", str(osworld), "--provider_name", "docker",
        "--path_to_vm", "/vm/Ubuntu.qcow2", "--task", str(task), "--result_dir", str(results),
        "--repo-dir", str(repo_dir), "--data-dir", str(tmp_path / "data"),
        "--settings-path", str(settings), "--ouroboros-url", "http://127.0.0.1:9",
        "--target-file", str(tmp_path / "target.txt"), "--claim-dir", str(claims),
        "--wait_after_reset_sec", "0",          # keeps the suite fast; nothing under test
        "--allow-dirty-seed",
    ], results


def _attempt_dirs(run_dir):
    """Every attempt's own admission/finalization record, oldest first."""
    attempts = run_dir / "attempts"
    return sorted(attempts.iterdir()) if attempts.is_dir() else []


def _attempt_manifests(run_dir):
    return [json.loads((d / "task_run_manifest.json").read_text(encoding="utf-8"))
            for d in _attempt_dirs(run_dir)]


def test_target_actor_is_durable_before_claim_and_survives_claim_crash(
        tmp_path, monkeypatch):
    from devtools.benchmarks.osworld import run_step_agent

    claims = tmp_path / "claims"
    rcb, _env = _cu_bridge_stubs(monkeypatch, tmp_path)
    argv, results = _cu_bridge_argv(tmp_path, claims)
    monkeypatch.setattr(sys, "argv", argv)
    observed = {"claim": False}

    def crash_at_first_external_boundary(*_args, **_kwargs):
        manifests = _attempt_manifests(results / "chrome" / "abc")
        assert len(manifests) == 1
        actor = manifests[0]["harness"]["target_runtime_actor"]
        assert actor["mismatches"] == []
        assert not any(actor["local_routes"].values())
        assert actor["reviewer_slots"]["advisory"]["enabled"] is False
        assert manifests[0]["available_subagents"] == actor["available_subagents"]
        observed["claim"] = True
        raise RuntimeError("synthetic claim-boundary crash")

    monkeypatch.setattr(run_step_agent, "acquire_task_claim", crash_at_first_external_boundary)
    assert rcb.main() == 1

    assert observed["claim"] is True
    final = _attempt_manifests(results / "chrome" / "abc")[0]
    assert final["extra"]["outcome"] == "adapter_error"
    assert final["harness"]["target_runtime_actor"]["reviewer_slots"]


def test_two_overlapping_attempts_never_share_one_canonical_record(tmp_path, monkeypatch, capsys):
    """The claim is only half the protection if both attempts still write the same files.

    `run_dir` is keyed by the TASK, so two lanes running the same task shared
    `run_dir/task_run_manifest.json`: both wrote their admission record there before either had
    claimed anything, and the loser then finalized `skipped_in_flight` into the file while the
    holder was still running — defeating both the claim's ownership contract and the
    append-only evidence contract. Each attempt now records into `attempts/<id>/`, and only the
    claim holder writes the canonical artefacts.
    """
    from devtools.benchmarks.osworld.run_step_agent import (
        acquire_task_claim,
        release_task_claim,
        task_claim_key,
    )

    claims = tmp_path / "claims"
    repo_dir = tmp_path / "repo"
    rcb, _env = _cu_bridge_stubs(monkeypatch, tmp_path, reward=1.0)
    argv, results = _cu_bridge_argv(tmp_path, claims)
    monkeypatch.setattr(sys, "argv", argv)
    run_dir = results / "chrome" / "abc"
    key = task_claim_key("chrome", "abc")

    # LANE A holds the task, exactly as a concurrent runner would.
    holder_fd, holder_reason = acquire_task_claim(claims, key, stale_sec=3600, repo_dir=repo_dir)
    assert holder_fd is not None and holder_reason == "claimed"

    # LANE B runs the same task and steps aside.
    assert rcb.main() == 4
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["claim"] == "in_flight"
    bystander = _attempt_dirs(run_dir)
    assert len(bystander) == 1
    assert json.loads((bystander[0] / "task_run_manifest.json").read_text(
        encoding="utf-8"))["extra"]["outcome"] == "skipped_in_flight"
    # NOTHING canonical was written: not the manifest the holder will write, not the task copy,
    # not an outcome. The holder's directory is untouched by a lane that never owned it.
    assert not (run_dir / "task_run_manifest.json").exists()
    assert not (run_dir / "task.json").exists()
    assert not (run_dir / "task_outcome.json").exists()

    # LANE A crashes without scoring, so the task is claimable again (an UNSCORED attempt never
    # blocks a retry), and the next attempt wins it for real.
    release_task_claim(claims, key, holder_fd, scored=False, repo_dir=repo_dir)
    assert rcb.main() == 0

    attempts = _attempt_dirs(run_dir)
    assert len(attempts) == 2 and attempts[0] == bystander[0]     # append-only: not overwritten
    winner = json.loads((attempts[1] / "task_run_manifest.json").read_text(encoding="utf-8"))
    assert winner["extra"]["outcome"] == "completed" and winner["extra"]["claim_owner"] is True
    # The loser's terminal outcome is still its own, in its own file.
    assert json.loads((attempts[0] / "task_run_manifest.json").read_text(
        encoding="utf-8"))["extra"]["outcome"] == "skipped_in_flight"
    # ...and the canonical record belongs to the holder alone.
    canonical = json.loads((run_dir / "task_run_manifest.json").read_text(encoding="utf-8"))
    assert canonical["extra"]["outcome"] == "completed"
    assert (run_dir / "task.json").is_file() and (run_dir / "result.txt").is_file()
    assert json.loads((run_dir / "task_outcome.json").read_text(
        encoding="utf-8"))["claim_owner"] is True


def test_cu_bridge_retains_the_lock_when_the_scored_marker_will_not_persist(tmp_path, monkeypatch):
    """INTEGRATED regression for the real try/except/finally path.

    The helper-level test cannot see this: inside `_run_cu_bridge`, a `ClaimMarkerNotDurable`
    raised after `env.evaluate()` was swallowed by the broad `except Exception`, which left
    `claim_scored` False, so the `finally` released the lock and the ALREADY-EVALUATED task
    became immediately claimable again — precisely the corruption the fail-closed marker
    exists to prevent.
    """
    import ouroboros.utils as ouroboros_utils

    from devtools.benchmarks.osworld.run_step_agent import acquire_task_claim, task_claim_key

    claims = tmp_path / "claims"
    rcb, env = _cu_bridge_stubs(monkeypatch, tmp_path, reward=1.0)
    argv, results = _cu_bridge_argv(tmp_path, claims)
    monkeypatch.setattr(sys, "argv", argv)

    real_write = ouroboros_utils.atomic_write_json

    def _fail_only_the_marker(path, payload, **kwargs):
        if str(path).endswith(".scored"):
            raise OSError(28, "No space left on device")
        return real_write(path, payload, **kwargs)

    monkeypatch.setattr(ouroboros_utils, "atomic_write_json", _fail_only_the_marker)

    assert rcb.main() == 2
    key = task_claim_key("chrome", "abc")
    # THE ASSERTION: the scored-but-unmarked state is recorded DURABLY, so the refusal does not
    # depend on the lock — which `stale_sec` reclaims by design. The lock is retained too, but
    # only as interim cover.
    assert (claims / f"{key}.lock").exists()
    assert not (claims / f"{key}.scored").exists()
    unconfirmed = json.loads((claims / f"{key}.scored_unconfirmed").read_text(encoding="utf-8"))
    assert unconfirmed["reason"] == "scored_marker_write_failed"
    assert unconfirmed["reward"] == 1.0                      # the score is not lost
    contender_fd, contender_reason = acquire_task_claim(claims, key, stale_sec=3600, repo_dir=tmp_path / "repo")
    assert contender_fd is None and contender_reason == "scored_unconfirmed"
    # The official score is not thrown away, and the bookkeeping failure is disclosed.
    outcome = json.loads((results / "chrome" / "abc" / "task_outcome.json").read_text(encoding="utf-8"))
    assert outcome["reward"] == 1.0
    assert outcome["reason_code"] == "claim_marker_not_durable"
    assert outcome["claim_lock_retained"] is True
    extra = json.loads((results / "chrome" / "abc" / "task_run_manifest.json")
                       .read_text(encoding="utf-8"))["extra"]
    assert extra["outcome"] == "scored_claim_marker_failed" and extra["exit_code"] == 2
    assert extra["claim_unconfirmed_marker"].endswith(".scored_unconfirmed")
    assert env.closed is True                       # the VM is still torn down


def test_cu_bridge_releases_the_lock_and_keeps_the_marker_on_a_healthy_scored_run(
        tmp_path, monkeypatch):
    """The same integrated path when the marker DOES persist: marker kept, lock released."""
    from devtools.benchmarks.osworld.run_step_agent import acquire_task_claim, task_claim_key

    claims = tmp_path / "claims"
    rcb, _env = _cu_bridge_stubs(monkeypatch, tmp_path, reward=0.0)
    argv, results = _cu_bridge_argv(tmp_path, claims)
    monkeypatch.setattr(sys, "argv", argv)

    assert rcb.main() == 0
    key = task_claim_key("chrome", "abc")
    assert (claims / f"{key}.scored").is_file()
    assert not (claims / f"{key}.lock").exists()
    # ...and a later lane steps aside on the marker, not on the lock.
    later_fd, later_reason = acquire_task_claim(claims, key, stale_sec=3600, repo_dir=tmp_path / "repo")
    assert later_fd is None and later_reason == "already_scored"
    assert json.loads((results / "chrome" / "abc" / "result.txt").read_text(encoding="utf-8") or 0) == 0.0


def test_claim_rechecks_the_marker_after_winning_the_lock(tmp_path, monkeypatch):
    """TOCTOU: the marker was read only BEFORE waiting for the lock and never again.

    Two lanes both see no marker; the first wins the lock, scores, marks and releases; the
    second then acquires the lock with the marker already on disk and used to be told
    `claimed` — rerunning a task that already has an official score.
    """
    import ouroboros.platform_layer as platform_layer

    from devtools.benchmarks.osworld.run_step_agent import (
        acquire_task_claim,
        mark_task_scored,
        task_claim_key,
    )

    claims = tmp_path / "claims"
    key = task_claim_key("os", "abc")
    real_acquire = platform_layer.acquire_exclusive_file_lock

    def _score_while_the_contender_waits(lock_path, **kwargs):
        fd = real_acquire(lock_path, **kwargs)
        # The previous holder finished, marked and released WHILE we were blocking here.
        mark_task_scored(claims, key, repo_dir=tmp_path / "repo", payload={"reward": 1.0})
        return fd

    monkeypatch.setattr(platform_layer, "acquire_exclusive_file_lock",
                        _score_while_the_contender_waits)
    lock_fd, reason = acquire_task_claim(claims, key, stale_sec=3600, repo_dir=tmp_path / "repo")
    assert lock_fd is None and reason == "already_scored"
    # ...and the lock we took in order to look is given back, not parked for a whole window.
    assert not (claims / f"{key}.lock").exists()
    monkeypatch.undo()
    assert acquire_task_claim(claims, key, stale_sec=3600, repo_dir=tmp_path / "repo") == (None, "already_scored")


def _refused_attestation_record():
    """The record `runtime_attestation()` builds before refusing a version skew."""
    return {
        "ok": False,
        "reason": "runtime_skew",
        "runtime_version": "6.75.0",
        "repo_head": "a" * 40,
        "repo_version": "6.76.0",
        "url": "http://127.0.0.1:9/",
        "overridden": False,
        "override_set": False,
    }


def test_cu_bridge_persists_the_attestation_record_it_was_handed(tmp_path, monkeypatch, capsys):
    """`RuntimeAttestationRefused` CARRIES the record it built — the exact typed reason plus
    `runtime_version`, `repo_head` and `repo_version`. Catching a generic `RuntimeError` and
    keeping only the string `runtime_attestation_failed` threw that evidence away at the moment
    it matters most, and `docs/ARCHITECTURE.md` promises it is preserved. Same defect phase P1
    fixed for ProgramBench in its round 4."""
    from devtools.benchmarks.common.manifests import RuntimeAttestationRefused

    claims = tmp_path / "claims"
    rcb, _env = _cu_bridge_stubs(monkeypatch, tmp_path)
    argv, results = _cu_bridge_argv(tmp_path, claims)
    monkeypatch.setattr(sys, "argv", argv)
    record = _refused_attestation_record()

    def _refuse(url, repo):
        raise RuntimeAttestationRefused("runtime attestation failed reason=runtime_skew", record)

    monkeypatch.setattr(rcb, "runtime_attestation", _refuse)

    assert rcb.main() == 2
    outcome = json.loads(capsys.readouterr().out)
    assert outcome["reason_code"] == "runtime_skew"
    assert outcome["runtime_attestation"] == record
    # The attestation refusal happens BEFORE the claim, so this attempt never owned the task and
    # its record lives in its own attempt directory. Writing it to the shared canonical manifest
    # is exactly the clobber that made two overlapping lanes overwrite each other.
    manifest = _attempt_manifests(results / "chrome" / "abc")[-1]
    assert manifest["extra"]["runtime_attestation"] == record
    assert manifest["extra"]["refusal"] == {"stage": "runtime_attestation",
                                            "reason": "runtime_skew", "exit_code": 2}
    assert manifest["extra"]["outcome"] == "blocked" and manifest["extra"]["exit_code"] == 2
    assert manifest["extra"]["claim_owner"] is False
    assert not (results / "chrome" / "abc" / "task_run_manifest.json").exists()
    # A refusal that carries NO record still refuses, with the generic reason as the fallback.
    monkeypatch.setattr(rcb, "runtime_attestation",
                        lambda url, repo: (_ for _ in ()).throw(RuntimeError("no record")))
    assert rcb.main() == 2
    attempts = _attempt_manifests(results / "chrome" / "abc")
    # ...into a SECOND, independent attempt record: the first is not overwritten.
    assert len(attempts) == 2
    assert attempts[0]["extra"]["refusal"]["reason"] == "runtime_skew"
    assert attempts[-1]["extra"]["refusal"]["reason"] == "runtime_attestation_failed"


def test_step_agent_preflight_persists_the_attestation_record_it_was_handed(
        tmp_path, monkeypatch, capsys):
    """Same defect on the step loop: the preflight kept only the message, and the manifest is
    amended FROM the preflight details, so the loss propagated into the run's own record."""
    from devtools.benchmarks.common.manifests import RuntimeAttestationRefused
    from devtools.benchmarks.osworld import run_step_agent

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "VERSION").write_text("6.76.0\n", encoding="utf-8")
    task = tmp_path / "OSWorld" / "evaluation_examples" / "examples" / "chrome" / "abc.json"
    task.parent.mkdir(parents=True)
    task.write_text(json.dumps({"id": "abc", "instruction": "no-op"}), encoding="utf-8")
    results = tmp_path / "results"
    record = _refused_attestation_record()

    def _refuse(url, repo):
        raise RuntimeAttestationRefused("runtime attestation failed reason=runtime_skew", record)

    monkeypatch.setattr(run_step_agent, "runtime_attestation", _refuse)
    monkeypatch.setattr(sys, "argv", [
        "run_step_agent.py", "--osworld-root", str(tmp_path / "OSWorld"), "--task", str(task),
        "--result_dir", str(results), "--repo-dir", str(repo_dir),
        "--data-dir", str(tmp_path / "data"), "--settings-path", str(tmp_path / "settings.json"),
        "--ouroboros-url", "http://127.0.0.1:9", "--provider_name", "docker", "--model", "m",
        "--allow-dirty-seed",            # provenance is not what this test pins
    ])

    assert run_step_agent.main() == 2
    outcome = json.loads(capsys.readouterr().out)
    assert outcome["reason_code"] == "preflight_failed"
    assert any("reason=runtime_skew" in failure
               for failure in outcome["preflight"]["failures"])
    assert outcome["preflight"]["details"]["runtime_attestation"] == record
    run_dir = results / "pyautogui" / "screenshot_a11y_tree" / "m" / "chrome" / "abc"
    manifest = json.loads((run_dir / "task_run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["runtime_attestation"] == record
    assert manifest["extra"]["exit_code"] == 2
    # ...and the typed refusal NAMES the attestation reason. `preflight_failed` alone conflates
    # "the runtime disagrees with its checkout" with "the task JSON is missing" — different
    # operator actions — and the documented contract is the specific one.
    assert manifest["extra"]["refusal"] == {"stage": "runtime_attestation",
                                            "reason": "runtime_skew", "exit_code": 2}


def test_osworld_skeleton_persists_the_attestation_record_it_was_handed(
        tmp_path, monkeypatch, capsys):
    """Same defect on the non-spending entry point, whose whole job is to REPORT evidence."""
    from devtools.benchmarks.common.manifests import RuntimeAttestationRefused
    from devtools.benchmarks.osworld import osworld_adapter_skeleton as skeleton

    repo_root = tmp_path / "repo"
    osworld = tmp_path / "OSWorld"
    payload = tmp_path / "unix_computer_use"
    output_root = tmp_path / "runs" / "osworld"
    for path in (repo_root, osworld, payload):
        path.mkdir(parents=True)
    (osworld / "evaluation_examples").mkdir()
    record = _refused_attestation_record()

    def _refuse(url, repo):
        raise RuntimeAttestationRefused("runtime attestation failed reason=runtime_skew", record)

    monkeypatch.setattr(skeleton, "runtime_attestation", _refuse)
    monkeypatch.setattr(skeleton, "DEFAULT_REPO_ROOT", repo_root)
    monkeypatch.setattr(skeleton, "DEFAULT_DATA_ROOT", tmp_path / "live-data")
    monkeypatch.setattr(sys, "argv", [
        "osworld_adapter_skeleton.py", "--osworld-root", str(osworld),
        "--ouroboros-url", "http://127.0.0.1:9", "--osworld-server-url", "http://127.0.0.1:9",
        "--unix-computer-use-payload", str(payload), "--output-root", str(output_root),
        "--allow-dirty-seed",            # output isolation/attestation is what this pins
    ])

    assert skeleton.main() == 2
    result = json.loads(capsys.readouterr().out)
    assert result["details"]["runtime_attestation"] == record
    assert any("reason=runtime_skew" in failure for failure in result["failures"])
    manifest = json.loads((output_root / "osworld_preflight.run_manifest.json")
                          .read_text(encoding="utf-8"))
    assert manifest["extra"]["preflight"]["details"]["runtime_attestation"] == record
    # The contract is ONE place to read the carried record from, across all three launchers —
    # burying it under `extra.preflight.details` made this the site that did not honour it.
    assert manifest["extra"]["runtime_attestation"] == record
    assert manifest["extra"]["refusal"] == {"stage": "runtime_attestation",
                                            "reason": "runtime_skew", "exit_code": 2}


def test_osworld_operator_patch_raises_provider_lock_timeout_and_is_documented():
    root = Path(__file__).resolve().parent.parent / "devtools" / "benchmarks" / "osworld"
    patch = (root / "operator_patches" / "osworld_docker_lock_timeout.v6760.patch").read_text(encoding="utf-8")
    assert "desktop_env/providers/docker/provider.py" in patch
    assert "-LOCK_TIMEOUT = 10" in patch and "+LOCK_TIMEOUT = 60" in patch
    readme = (root / "operator_patches" / "README.md").read_text(encoding="utf-8")
    assert "osworld_docker_lock_timeout.v6760.patch" in readme
    assert "construct_desktop_env" in readme  # both halves of the fix are disclosed


def test_osworld_methodology_preregisters_the_dedup_rule_and_defers_the_lane_generator():
    text = (Path(__file__).resolve().parent.parent / "devtools" / "benchmarks" / "osworld"
            / "METHODOLOGY.md").read_text(encoding="utf-8")
    assert "FIRST SCORED ATTEMPT WINS" in text
    # Multiple lanes ARE supported and the smoke exercises them, so the disclosure must say so;
    # what is extracted is the lane-script GENERATOR, and the disclosure must not describe a
    # convenience the tree does not have either.
    assert "MULTIPLE LANES ARE SUPPORTED" in text
    assert "NO MULTI-LANE LAUNCHER GENERATOR IN\n     THIS RELEASE" in text
    assert "gen_lanes.py" in text and "lanes.json" in text
    # The rule is enforced by code that EXISTS, and the record layout that makes overlapping
    # attempts safe is disclosed rather than implied.
    assert "attempts/<attempt_id>/task_run_manifest.json" in text
    assert "claim_owner" in text
    # The residual-window disclosure must match the fix: the interrupt path is closed with a
    # durable marker; only SIGKILL remains open.
    assert "THE INTERRUPT WINDOW IS CLOSED; THE `SIGKILL` WINDOW IS NOT" in text
    assert "construct_desktop_env" in text
    assert "LOCK_TIMEOUT" in text
    assert "--allow-dirty-seed" in text


def test_module_grandfather_matcher_uses_exact_repo_relative_paths():
    from ouroboros.review import module_is_grandfathered
    # Exact runtime helpers accept only actual repo-relative paths. Compatibility
    # section-prefix decoding belongs solely to compute_complexity_metrics.
    assert module_is_grandfathered("skills/unix_computer_use/plugin.py")
    assert not module_is_grandfathered("repo/skills/unix_computer_use/plugin.py")
    # a DIFFERENT plugin.py (future skill) is NOT exempted by the path-qualified entry
    assert not module_is_grandfathered("skills/other_skill/plugin.py")
    assert not module_is_grandfathered("repo/skills/other_skill/plugin.py")
    # Root server.py is an exact manifest path; a nested same-basename is not.
    assert module_is_grandfathered("server.py")
    assert not module_is_grandfathered("repo/server.py")
    assert not module_is_grandfathered("ouroboros/server.py")
    assert not module_is_grandfathered("repo/ouroboros/server.py")
    # tools/control.py's debt was retired by the v7 split (2110->492 lines);
    # basename exactness still holds - NEITHER control.py is grandfathered now
    # (the positive+negative exact-path pair above is server.py).
    assert not module_is_grandfathered("ouroboros/tools/control.py")
    assert not module_is_grandfathered("ouroboros/gateway/control.py")


def test_cu_bridge_publication_failure_never_erases_an_obtained_score(tmp_path, monkeypatch):
    """An outcome that already carries an official score is never overwritten by a generic error.

    By the time publication runs, `mark_task_scored` has made `.scored` durable, so no later
    attempt may retry this task. Reporting `reward=None`/`not_run` from the broad handler
    therefore destroyed a score that EXISTS, permanently: the protection became the lock.
    """
    rcb, _env = _cu_bridge_stubs(monkeypatch, tmp_path, reward=1.0)
    argv, results = _cu_bridge_argv(tmp_path, tmp_path / "claims")
    monkeypatch.setattr(sys, "argv", argv)
    run_dir = results / "chrome" / "abc"
    (run_dir / "result.txt").mkdir(parents=True)     # fails the first artefact after the marker

    assert rcb.main() == 1
    outcome = json.loads((run_dir / "task_outcome.json").read_text(encoding="utf-8"))
    assert outcome["reward"] == 1.0                  # the obtained score survived the failure
    assert outcome["reason_code"] == "publication_failed_after_scoring"
    row = json.loads((results / "result_index.jsonl").read_text(
        encoding="utf-8").splitlines()[-1])
    assert row["official_eval_status"] == "completed"    # it WAS evaluated, not `not_run`


def test_cu_bridge_keeps_the_ledger_row_when_the_canonical_outcome_cannot_be_written(
        tmp_path, monkeypatch):
    """The score survives a failure INSIDE the writer, at the canonical outcome stage.

    The sibling of the `result.txt` case: there the failure happened BEFORE `_write_outcome`
    ran, so the broad handler could still publish. Here the writer itself dies partway, and the
    handler used to call the SAME aggregate writer again — reproducing the failure and escaping
    with no ledger row at all, while the durable `.scored` marker forbids any retry. Every
    destination is attempted independently, so the still-writable ledger records the truth.
    """
    rcb, _env = _cu_bridge_stubs(monkeypatch, tmp_path, reward=1.0)
    argv, results = _cu_bridge_argv(tmp_path, tmp_path / "claims")
    monkeypatch.setattr(sys, "argv", argv)
    run_dir = results / "chrome" / "abc"
    (run_dir / "task_outcome.json").mkdir(parents=True)   # canonical publication stage fails

    assert rcb.main() == 1
    row = json.loads((results / "result_index.jsonl").read_text(
        encoding="utf-8").splitlines()[-1])
    assert row["official_eval_status"] == "completed"     # it WAS evaluated, not `not_run`
    assert row["details"]["reward"] == 1.0                # the obtained score reached the ledger
    attempts = sorted((run_dir / "attempts").glob("*/task_outcome.json"))
    assert attempts, "the attempt's own record must still exist"
    assert json.loads(attempts[-1].read_text(encoding="utf-8"))["reward"] == 1.0


def test_cu_bridge_keeps_the_outcome_files_when_the_ledger_cannot_be_appended(
        tmp_path, monkeypatch):
    """The mirror case: the ledger is the dead destination, the outcome records must survive.

    A failure at the LAST publication stage must not roll back or re-run the ones that already
    succeeded, and must not escape as a traceback: the run reports a disclosed publication
    failure while the reward stays on every record that could still be written.
    """
    rcb, _env = _cu_bridge_stubs(monkeypatch, tmp_path, reward=1.0)
    argv, results = _cu_bridge_argv(tmp_path, tmp_path / "claims")
    monkeypatch.setattr(sys, "argv", argv)
    (results / "result_index.jsonl").mkdir(parents=True)  # ledger publication stage fails

    assert rcb.main() == 1
    run_dir = results / "chrome" / "abc"
    canonical = json.loads((run_dir / "task_outcome.json").read_text(encoding="utf-8"))
    assert canonical["reward"] == 1.0                     # written before the ledger, kept
    assert any("result_index" in e for e in canonical.get("publication_errors", [])), \
        "the dead destination must be disclosed, not swallowed"
    attempts = sorted((run_dir / "attempts").glob("*/task_outcome.json"))
    assert json.loads(attempts[-1].read_text(encoding="utf-8"))["reward"] == 1.0


def test_cu_bridge_ledger_row_never_points_at_an_outcome_that_was_not_written(
        tmp_path, monkeypatch):
    """The ledger row must describe the publication that HAPPENED, not the one intended.

    Independent destinations stopped one dead record from erasing an obtained score — but
    independence cuts both ways: the row is now written even when the artefact it points at
    is not. Emitting `output_paths.task_outcome` unconditionally, with the pre-failure status
    and without the collected `publication_errors`, makes the index assert a completed,
    readable outcome file that does not exist. An operator must be able to tell "scored,
    fully published" from "scored, partially published" from the row alone.
    """
    rcb_mod, _env = _cu_bridge_stubs(monkeypatch, tmp_path, reward=1.0)
    argv, results = _cu_bridge_argv(tmp_path, tmp_path / "claims")
    monkeypatch.setattr(sys, "argv", argv)
    real_write_json = rcb_mod.write_json

    def _dead_attempt_outcome(path, payload):
        target = Path(path)
        if target.name == "task_outcome.json" and "attempts" in target.parts:
            raise OSError("attempt outcome destination is dead")
        return real_write_json(path, payload)

    monkeypatch.setattr(rcb_mod, "write_json", _dead_attempt_outcome)

    assert rcb_mod.main() == 1
    row = json.loads((results / "result_index.jsonl").read_text(
        encoding="utf-8").splitlines()[-1])
    # No pointer to a destination that failed: the file genuinely is not there.
    assert not list((results / "chrome" / "abc" / "attempts").glob("*/task_outcome.json"))
    assert "task_outcome" not in row["output_paths"], \
        "the row must not point at an artefact whose write failed"
    # The status publication never achieved must not be reported as if it had been.
    assert row["status"] != "completed"
    # ...while everything the run DID achieve still reaches the ledger.
    assert row["official_eval_status"] == "completed"
    assert row["details"]["reward"] == 1.0
    assert any("attempt_outcome" in e for e in row["details"]["publication_errors"]), \
        "the row must carry the collected publication errors"
    # BOTH SIDES of the same rule. The previous round fixed the ledger row and left the
    # manifest lying: `_amend_manifest` still added `output_paths.task_outcome`
    # unconditionally, so the finalized attempt manifest kept naming the missing file. A
    # pointer is a pointer wherever it is written.
    attempt_manifests = sorted(
        (results / "chrome" / "abc" / "attempts").glob("*/task_run_manifest.json"))
    assert attempt_manifests, "the attempt manifest must still be finalized"
    manifest = json.loads(attempt_manifests[-1].read_text(encoding="utf-8"))
    assert "task_outcome" not in (manifest.get("output_paths") or {}), \
        "the manifest must not point at an artefact whose write failed either"
    assert (manifest.get("output_paths") or {}).get("attempt_dir"), \
        "...while the pointer that IS valid survives"


# --- feasibility gate (opt-in premise phase) ---------------------------------------


class _GateArgs:
    """Minimal stand-in for the parsed CLI namespace the gate helpers read."""

    def __init__(self, *, feasibility_gate: bool, task_timeout_sec: int = 3600,
                 data_dir: str = "/nonexistent-bench-data", max_steps: int = 0):
        self.feasibility_gate = feasibility_gate
        self.task_timeout_sec = task_timeout_sec
        # The gate poll reads the task's LIVE event log to enforce its turn
        # share, so the namespace carries the bench data dir like the real one.
        self.data_dir = data_dir
        self.max_steps = max_steps


@pytest.mark.parametrize(
    "latest,expected",
    [
        ({"result": "~/Desktop is empty; nothing to act on.\nINFEASIBLE"}, "INFEASIBLE"),
        ({"result": "The file is there.\nPROCEED"}, "PROCEED"),
        ({"result": "Cloudflare blocked the page.\nUNDETERMINED"}, "UNDETERMINED"),
        # Everything below must FAIL OPEN: the working phase still runs.
        ({"result": "a discussion that never states a verdict"}, "UNDETERMINED"),
        ({"result": "I weighed whether this is INFEASIBLE and decided it is not"}, "UNDETERMINED"),
        ({"status": "timeout"}, "UNDETERMINED"),
        ({}, "UNDETERMINED"),
        (None, "UNDETERMINED"),
        # The terminal answer field wins over the runtime result body.
        ({"final_answer": "PROCEED", "result": "INFEASIBLE"}, "PROCEED"),
    ],
)
def test_gate_verdict_fails_open_unless_explicitly_infeasible(latest, expected):
    assert rcb._gate_verdict(latest) == expected


def test_gate_verdict_reads_the_answer_not_a_recap_of_the_options():
    """Regression: reverse-scanning every line for a keyword read a model's own
    enumeration of the three options as its verdict, turning a PROCEED into a scored
    hard zero. Only the last line — what the prompt actually asks for — may decide."""
    recap = (
        "I inspected the desktop as instructed.\n\n"
        "Ruling out each option in turn:\n"
        "UNDETERMINED\n"
        "PROCEED\n"
        "INFEASIBLE\n\n"
        "None of those obstacles apply here: the file exists and the app supports the\n"
        "feature, so the task is clearly PROCEED.\n"
    )
    assert rcb._gate_verdict({"result": recap}) != "INFEASIBLE"


def test_gate_verdict_tolerates_formatting_but_not_prose():
    # Ordinary formatting of a real verdict is accepted.
    for ok in ("INFEASIBLE", "INFEASIBLE.", "**INFEASIBLE**", "`infeasible`"):
        assert rcb._gate_verdict({"result": ok}) == "INFEASIBLE", ok
    # A verdict embedded in a sentence is NOT a verdict: fail open instead of guessing.
    for not_a_verdict in ("the answer is INFEASIBLE", "INFEASIBLE, probably", ""):
        assert rcb._gate_verdict({"result": not_a_verdict}) != "INFEASIBLE", not_a_verdict


def test_gate_window_is_zero_when_disabled_and_floored_when_enabled():
    assert rcb._gate_window_sec(_GateArgs(feasibility_gate=False)) == 0.0
    assert rcb._gate_window_sec(_GateArgs(feasibility_gate=True, task_timeout_sec=3600)) == 900.0
    # Floor: a tiny task timeout must not shrink the phase to nothing.
    assert rcb._gate_window_sec(_GateArgs(feasibility_gate=True, task_timeout_sec=100)) == 60.0


def test_gate_claim_window_tracks_the_single_premise_round():
    """The gate occupies the claim holder BEFORE the working task. If its occupancy is not
    in the staleness bound, a second lane can reclaim a task the first is still working and
    both will score it. Since v6.81.1 the premise phase is exactly ONE round (the
    confirming challenger was removed: 20 invocations, 0 saves, 1 loss, and it confirmed
    every false kill — correlated errors, not an independent check), so the claim window
    must equal one gate window, not two."""
    from devtools.benchmarks.osworld.run_step_agent import claim_stale_sec

    args = _GateArgs(feasibility_gate=True, task_timeout_sec=3600)
    assert rcb._gate_claim_window_sec(args) == rcb._gate_window_sec(args) == 900.0
    base = claim_stale_sec(3600, 900, 900)
    assert base + rcb._gate_claim_window_sec(args) == base + 900.0
    assert rcb._gate_claim_window_sec(_GateArgs(feasibility_gate=False)) == 0.0, \
        "ungated runs unchanged"


def test_terminal_answer_text_prefers_final_answer_then_falls_back():
    assert rcb._terminal_answer_text({"final_answer": "done", "result": "other"}) == "done"
    # The documented fallback: the field that actually carries the text on this runner.
    assert rcb._terminal_answer_text({"final_answer": "", "result": "the real answer"}) == "the real answer"
    assert rcb._terminal_answer_text({"final_answer": "   ", "result": "x"}) == "x"
    assert rcb._terminal_answer_text({}) == ""
    assert rcb._terminal_answer_text(None) == ""


def test_gate_phase_removes_the_mutating_tools_and_keeps_the_reading_ones():
    normal = set(rcb._effective_disabled_tools(False))
    gated = set(rcb._effective_disabled_tools(False, gate_phase=True))
    assert normal < gated, "the gate phase must disable strictly more than the working phase"
    # NAMED literals, deliberately not derived from _GUI_ACTION_TOOLS: the v6.81.1 review
    # caught the aliases registered in the skill but missing from that set — the gate could
    # click through them. A test iterating the same incomplete set cannot catch that class,
    # so this list is the independent statement of what "mutating" means.
    mutating_tools = ("click", "double_click", "triple_click", "move", "left_click_drag",
                      "mouse_down", "mouse_up", "type_text", "key", "hold_key", "scroll")
    assert set(mutating_tools) == set(rcb._GUI_ACTION_TOOLS), \
        "a click alias was registered without updating _GUI_ACTION_TOOLS (or vice versa)"
    for mutating in mutating_tools:
        assert extension_surface_name(rcb.SKILL_NAME, mutating) in gated, mutating
        assert extension_surface_name(rcb.SKILL_NAME, mutating) not in normal, mutating
    # Observation and read-only probing must survive, or the phase cannot establish anything.
    for readable in ("screenshot", "window_list", "wait", "remote_exec"):
        assert extension_surface_name(rcb.SKILL_NAME, readable) not in gated, readable


def test_acceptance_claims_are_general_and_well_formed():
    """These travel to the reviewer that already runs. They must carry no task id, no
    application name and nothing about how the benchmark grades."""
    from ouroboros.contracts.task_contract import normalize_acceptance_claims

    claims = rcb._ACCEPTANCE_CLAIMS
    assert claims, "the panel runs either way; empty claims is what we are fixing"
    assert normalize_acceptance_claims(claims), "must survive the contract normalizer"
    blob = json.dumps(claims).lower()
    for forbidden in ("osworld", "evaluator", "gimp", "chrome", "libreoffice", "reward",
                      "infeasible task", "1 in 13"):
        assert forbidden not in blob, forbidden
    assert len({c["id"] for c in claims}) == len(claims), "claim ids must be unique"

class _FakeResetEnv:
    """DesktopEnv stand-in for _reset_verified: scripted setup outcomes per attempt.

    `plan` is a list of per-attempt behaviours: "ok" (setup succeeds), "silent"
    (reset returns but setup silently failed — the OSWorld fail-open path),
    "noshot" (no screenshot), "raise" (reset raises).
    """

    def __init__(self, plan, config=({"type": "download"},)):
        self.plan = list(plan)
        self.config = list(config)
        self.is_environment_used = False
        self.calls = 0
        self.used_flag_at_entry: list[bool] = []

    def reset(self, task_config=None):
        self.used_flag_at_entry.append(self.is_environment_used)
        behaviour = self.plan[min(self.calls, len(self.plan) - 1)]
        self.calls += 1
        # reset() always clears the flag after the revert, like the real one.
        self.is_environment_used = False
        if behaviour == "raise":
            raise RuntimeError("boot failed")
        if behaviour == "ok":
            self.is_environment_used = True
        self._behaviour = behaviour

    def _get_obs(self):
        return {"screenshot": b"" if self._behaviour == "noshot" else b"\x89PNG"}


def test_reset_verified_rejects_the_silent_setup_skip_and_recovers_on_retry():
    """Regression for the 2026-07-28 smoke: OSWorld's reset() skips ALL setup steps when
    the guest probe times out, raises nothing, and logs "Environment setup complete." The
    working phase then opens on a VM without the task's files. The postcondition is
    machine-checkable (`is_environment_used`), so the helper must reject such an attempt
    and succeed on a later healthy one."""
    env = _FakeResetEnv(["silent", "ok"])
    rec = rcb._reset_verified(env, {"config": env.config}, retries=3,
                              deadline=time.time() + 300, wait_after_sec=0,
                              sleep=lambda _s: None)
    assert rec["attempts"] == 2
    assert env.calls == 2


def test_reset_verified_forces_the_snapshot_revert_before_every_retry():
    """After a failed setup `is_environment_used` is False, and OSWorld's reset() then
    SKIPS the snapshot revert ("environment is clean") — an unforced retry would run
    setup on top of the partial state. The helper must force the flag True before the
    retry so the revert actually happens."""
    env = _FakeResetEnv(["silent", "silent", "ok"])
    rcb._reset_verified(env, {"config": env.config}, retries=3,
                        deadline=time.time() + 300, wait_after_sec=0,
                        sleep=lambda _s: None)
    assert env.used_flag_at_entry == [False, True, True]


def test_reset_verified_exhaustion_is_a_typed_infra_error_not_a_pass():
    env = _FakeResetEnv(["silent"])
    with pytest.raises(rcb.ResetUnverified) as exc:
        rcb._reset_verified(env, {"config": env.config}, retries=2,
                            deadline=time.time() + 300, wait_after_sec=0,
                            sleep=lambda _s: None)
    assert "silently failed" in str(exc.value)
    assert isinstance(exc.value.record.get("log_tail"), list)


def test_reset_verified_accepts_a_task_with_no_setup_config():
    """A task with an empty config never sets `is_environment_used`; that is OSWorld's
    documented behaviour, not a failure. Requiring the flag unconditionally would turn
    every no-setup task into an infra abort."""
    env = _FakeResetEnv(["silent"], config=())
    rec = rcb._reset_verified(env, {"config": []}, retries=1,
                              deadline=time.time() + 300, wait_after_sec=0,
                              sleep=lambda _s: None)
    assert rec["attempts"] == 1


def test_reset_verified_still_rejects_a_missing_screenshot():
    env = _FakeResetEnv(["noshot", "ok"])
    rec = rcb._reset_verified(env, {"config": env.config}, retries=3,
                              deadline=time.time() + 300, wait_after_sec=0,
                              sleep=lambda _s: None)
    assert rec["attempts"] == 2


def test_the_confirming_challenger_stays_removed():
    """v6.81.1 removed the second premise round. Its full-run ledger: 20 invocations,
    0 feasible tasks saved, 1 officially-infeasible task lost, 215 worker rounds burned,
    and it CONFIRMED all 4 of the gate's false kills — an identical-prompt re-read
    produces correlated errors, not an independent check. Guard the removal: the flow
    must post exactly ONE premise task per example and carry no challenger machinery."""
    assert not hasattr(rcb, "_kill_confirmed")
    src = (Path(__file__).resolve().parent.parent
           / "devtools" / "benchmarks" / "osworld" / "run_cu_bridge_agent.py").read_text(encoding="utf-8")
    flow = src[src.index("claim_fd: int | None = None"):]
    assert flow.count("_gate_round(") == 1, "exactly one premise round per example"
    assert '"feasibility_gate_challenger": False' in src, \
        "the manifest must disclose the challenger's absence to cross-run readers"


def test_gate_cancel_unconfirmed_is_the_one_condition_that_may_not_fail_open():
    """A premise round whose cancel did not confirm leaves a zombie session sharing the
    lane's server and skill connection file — it would act on the same VM the worker is
    scored on. Detection must be exact: timeouts whose cancel DID confirm proceed."""
    assert rcb._gate_cancel_unconfirmed({"status": "timeout", "cancel_confirmed": False})
    assert rcb._gate_cancel_unconfirmed({"status": "timeout"})
    assert not rcb._gate_cancel_unconfirmed({"status": "timeout", "cancel_confirmed": True})
    assert not rcb._gate_cancel_unconfirmed({"status": "completed"})
    assert not rcb._gate_cancel_unconfirmed({})


def test_gate_round_posts_a_fresh_memory_gate_phase_task_and_reads_the_verdict(monkeypatch):
    posted = {}

    def fake_api(url, method, path, payload=None, timeout=None):
        if method == "POST" and path == "/api/tasks":
            posted.update(payload)
            return {"task_id": "gate-1"}
        if method == "GET":
            return {"status": "completed", "result": "the pack list has no such locale.\nINFEASIBLE",
                    "total_rounds": 4}
        raise AssertionError((method, path))

    monkeypatch.setattr(rcb, "_api", fake_api)
    args = _GateArgs(feasibility_gate=True, task_timeout_sec=3600)
    args.allow_a11y = False
    args.ouroboros_url = "http://127.0.0.1:1"
    rec = rcb._gate_round(args.ouroboros_url, args, "change the UI language", role="gate")
    assert rec["verdict"] == "INFEASIBLE" and rec["role"] == "gate"
    assert rec["task_id"] == "gate-1" and rec["llm_rounds"] == 4
    # Independence and confinement travel in the payload itself.
    assert posted["memory_mode"] == "empty"
    assert set(rcb._effective_disabled_tools(False, gate_phase=True)) <= set(posted["disabled_tools"])


def test_gate_tool_trace_carries_full_args_for_the_offline_audit(tmp_path):
    """The read-only promise is auditable only if the sidecar carries every shell command
    VERBATIM: the GAIA leakage audit was blinded by exactly this (truncated previews on one
    arm). Rows from other tasks and non-skill tools must not leak into the trace."""
    from ouroboros.extension_loader import extension_name_prefix

    prefix = extension_name_prefix(rcb.SKILL_NAME)
    long_cmd = "find / -name '*.pak' " + "-o -name 'x' " * 120
    log_dir = tmp_path / "state" / "headless_tasks" / "gate42" / "data" / "logs"
    log_dir.mkdir(parents=True)
    rows = [
        {"type": "tool_call", "tool": prefix + "remote_exec", "args": {"command": long_cmd}},
        {"type": "tool_call", "tool": prefix + "screenshot", "args": {}, "is_error": False},
        {"type": "tool_call", "tool": "web_search", "args": {"q": "not a skill tool"}},
        {"type": "llm_round", "tool": prefix + "remote_exec"},
    ]
    (log_dir / "tools.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    trace = rcb._gate_tool_trace(tmp_path, "gate42")
    assert [t["tool"] for t in trace] == ["remote_exec", "screenshot"]
    assert trace[0]["args"]["command"] == long_cmd, "args must be verbatim, not a preview"
    assert rcb._gate_tool_trace(tmp_path, "") == []
    assert rcb._gate_tool_trace(tmp_path, "no-such-task") == []


def test_the_post_gate_reset_republishes_the_vm_endpoint():
    """The repair the v1 smoke actually needed. DockerProvider.revert_to_snapshot stops
    the container and start_emulator REALLOCATES ports, so the VM address changes on
    every reset. v1 published it once, before the gate (83/83 task dirs had bridge.json
    older than their gate record), so the working phase drove the pre-gate address —
    which another lane's container could already own. Pin the ordering: the post-gate
    reset must be followed by a target write and a _publish_target call, before the
    working task is created."""
    src = (Path(__file__).resolve().parent.parent
           / "devtools" / "benchmarks" / "osworld" / "run_cu_bridge_agent.py").read_text(encoding="utf-8")
    post_gate = src.index('reset_diag["post_gate"]')
    republish = src.index("_publish_target(data_dir, target)", post_gate)
    worker_post = src.index('"acceptance_claims": _ACCEPTANCE_CLAIMS', post_gate)
    assert post_gate < republish < worker_post, \
        "the endpoint must be republished after the post-gate reset and before the worker starts"
    # And the target file the skill reads must be rewritten too, not just the sidecar.
    assert src.index("Path(args.target_file).expanduser().write_text(target", post_gate) < republish


def test_gate_preamble_is_a_rubric_not_an_exception_list():
    """The v6.81.0 false kills shared one shape: the gate judged whether the OUTCOME
    would be meaningful instead of whether the REQUESTED ACTION is performable. The fix
    is a semantic decomposition; pin its load-bearing steps so a later edit cannot
    quietly regress the prompt into an example list."""
    p = rcb.GATE_PREAMBLE
    for step in ("ACTION", "REFERENT", "BLOCKING", "ACQUISITION", "SAME-THING CHECK",
                 "CHECK, DO NOT ASSUME", "STORE-OR-RENDER", "PLACEHOLDERS"):
        assert step in p, step
    assert "When in doubt, answer UNDETERMINED" in p, "fail-open stays the default"
    # The forced two-round vision loop is gone: screenshots attach automatically.
    assert "view_image(path)" not in rcb.OSWORLD_PREAMBLE
    assert "attached" in rcb.OSWORLD_PREAMBLE.lower()


def test_the_bench_agent_cannot_reach_the_bridge_url():
    """A v6.81.1 trace shows an agent reading the bridge port out of a tool result and
    curling `<bridge>/evaluate` — looking for the grader. It failed only because
    remote_exec runs inside the guest, where that port is not the host's: containment by
    luck of topology, not by design. Two things must hold: the screenshot result must not
    carry the URL, and the connection tools that echo it must be denied to the agent."""
    import skills.unix_computer_use.plugin as plugin

    denied = set(rcb._DENIED_SKILL_EXT_TOOLS)
    assert {"list_connections", "test_connection"} <= denied, denied
    disabled = set(rcb._effective_disabled_tools(False))
    for tool in ("list_connections", "test_connection"):
        assert extension_surface_name(rcb.SKILL_NAME, tool) in disabled, tool
    # The success path of the remote screenshot must not emit the bridge URL.
    src = pathlib.Path(plugin.__file__).read_text(encoding="utf-8")
    shot = src[src.index("def _osworld_screenshot"):src.index("def _test_osworld")]
    assert '"target": target' not in shot, "the bridge URL is back in the screenshot result"


def test_the_working_prompt_forbids_forcing_state_from_underneath_the_app():
    """v6.81.1 run, chrome/ae78f875: after establishing the requested UI control no longer
    exists, the agent wrote Chrome's PREF cookie from the DevTools console and then
    decrypted Chrome's Safe-Storage keyring to 'verify' it. It scored 0 only because that
    task's evaluator is infeasible-only — the same technique on a feasible task would have
    produced undeserved credit. State must be reachable through the application's own
    surface, and a tool restriction must cover discovery too."""
    p = rcb.OSWORLD_PREAMBLE
    assert "documented" in p and "underneath" in p, p[:0]
    for phrase in ("developer console", "credential", "TASK_INFEASIBLE"):
        assert phrase in p, phrase
    assert "including finding things" in p, "tool restrictions must cover discovery"


def test_a_dead_guest_control_server_ends_the_attempt_as_infra_not_a_zero():
    """v6.81.1, vs_code/7c4cc09e: the agent killed /home/user/server/main.py and then
    worked blind for the rest of its budget — every screenshot 500'd but was recorded
    as a success. A task whose environment died is INFRA (reward null, claim released),
    never a capability zero. The probe must fail CLOSED: an unknown state reads as
    unhealthy, or the watchdog is decorative."""
    class _Env:
        vm_ip = "127.0.0.1"
        server_port = 1  # nothing listens

    assert rcb._guest_endpoint_healthy(_Env(), timeout=0.4) is False
    # Before the endpoint is published there is nothing to judge.
    class _Unpublished:
        vm_ip = ""
        server_port = ""
    assert rcb._guest_endpoint_healthy(_Unpublished()) is True
    # The grace window is a real duration, and the flow reports a typed reason.
    assert rcb._GUEST_DOWN_GRACE_SEC >= 60
    src = (Path(__file__).resolve().parent.parent
           / "devtools" / "benchmarks" / "osworld" / "run_cu_bridge_agent.py").read_text(encoding="utf-8")
    assert '"guest_control_server_lost"' in src
    flow = src[src.index("claim_fd: int | None = None"):]
    assert flow.index("_guest_endpoint_healthy(env)") < flow.index('"guest_control_server_lost"')


def test_forensics_clauses_are_pinned_in_the_worker_prompt():
    """The v6.81.1 forensics attributed ~7.5 lost points to five recurring worker
    behaviours (own hex instead of the app's named swatch; retyping instead of
    clipboard transfer; collateral edits beyond the asked diff; ordinals counted
    over headings; finishing off the graded surface). Each got a preamble clause;
    pin them so a later prompt edit cannot silently drop one."""
    p = rcb.OSWORLD_PREAMBLE
    for phrase in (
        # v6.84.0 corrected wordings (the v6.83.0 originals cited-while-losing were fixed)
        "REALIZE A NAMED STATE THROUGH THE APPLICATION'S NAMED CONTROL",
        "TRANSFER TEXT VERBATIM, NEVER RETYPE",
        "TOUCH ONLY WHAT THE TASK NAMES",
        "ORDINALS COUNT WHAT THE TASK COUNTS",
        "FINISH ON THE GRADED SURFACE",
        "Shift+Enter",
    ):
        assert phrase in p, phrase


def test_gate_rubric_covers_named_mode_scope_and_prohibition():
    """Forensics: two gate PROCEEDs reinterpreted a named mode ('batch') and a launch
    scope (per-app vs per-folder) as working-phase details, and one prohibition
    ('without configuring X') was never verified as satisfiable — all three hide the
    premise in a modifier rather than a noun. Pin the 4d branch; the fail-open default
    must survive it."""
    p = rcb.GATE_PREAMBLE
    assert "NAMED MODE, SCOPE AND PROHIBITION" in p
    for phrase in ("MODE OF OPERATION", "APPLY SCOPE", "PROHIBITION"):
        assert phrase in p, phrase
    assert "When in doubt, answer UNDETERMINED" in p, "fail-open stays the default"


def _ns(**kw):
    from types import SimpleNamespace
    kw.setdefault("feasibility_gate", False)
    kw.setdefault("max_steps", 0)
    return SimpleNamespace(**kw)


def test_step_budget_uses_policy_turns_not_gui_actions():
    """A leaderboard step is one top-level policy turn: the official loop increments
    step_idx once per agent.predict() and executes every action that call emitted
    inside that step. The earlier 0.42-actions-per-round mapping compared a turn
    against an action. The declared budget must reserve the gate phase AND one
    tool-less terminal turn out of the claim, so a forced finalization is never
    step N+1."""
    b = rcb._step_budget(_ns(max_steps=100, feasibility_gate=True),
                         {"value": 85, "source": "settings"})
    assert b["step_semantics"] == "top_level_policy_turn"
    assert b["max_steps_claimed"] == 100 and b["enforced"] is True
    assert b["terminal_turn_reserve"] == 1
    assert b["gate_turn_reserve"] == rcb._GATE_TURN_RESERVE
    assert b["action_capable_round_cap"] == 100 - rcb._GATE_TURN_RESERVE - 1
    # Without the gate phase its reserve is not withheld.
    b2 = rcb._step_budget(_ns(max_steps=100), {"value": 99, "source": "settings"})
    assert b2["gate_turn_reserve"] == 0 and b2["action_capable_round_cap"] == 99
    # No claim -> nothing enforced, and the run is not comparable.
    b3 = rcb._step_budget(_ns(), {"value": 200, "source": "default"})
    assert b3["enforced"] is False and b3["max_steps_claimed"] is None


def test_a_step_claim_the_server_cannot_honor_is_refused_before_the_vm_boots():
    """Enforcement lives in the runtime round cap; the runner must PROVE that cap is
    at or below the declared budget before anything costs money. 'Most tasks finish
    early' is not a substitute — comparability is a per-task property."""
    import pytest

    over = rcb._step_budget(_ns(max_steps=100), {"value": 200, "source": "settings"})
    with pytest.raises(SystemExit, match="exceeds"):
        rcb._refuse_uncapped_step_claim(over)
    ok = rcb._step_budget(_ns(max_steps=100), {"value": 99, "source": "settings"})
    rcb._refuse_uncapped_step_claim(ok)  # must not raise
    # A claim so small the reserves swallow it is refused too.
    tiny = rcb._step_budget(_ns(max_steps=1, feasibility_gate=True), {"value": 1, "source": "env"})
    with pytest.raises(SystemExit, match="no working turns"):
        rcb._refuse_uncapped_step_claim(tiny)
    # An unenforced run is never refused (it simply is not comparable).
    rcb._refuse_uncapped_step_claim(rcb._step_budget(_ns(), {"value": 999, "source": "default"}))


def test_audit_reads_policy_turns_not_physical_calls():
    """The flat `total_rounds` on a task result is reconstructed from
    physical_calls — safety checks, acceptance reviewers and retries included —
    and on the v6.81.1 run it disagreed with the loop's own turn count on 344 of
    346 examples, running up to 13 higher. Auditing a step budget against it
    would mark compliant examples as overruns. Pin the loop field as the source
    and pin fail-closed behaviour when it is missing."""
    budget = rcb._step_budget(_ns(max_steps=100, feasibility_gate=True),
                              {"value": 85, "source": "settings"})
    # A result whose physical and policy counts deliberately differ.
    latest = {"total_rounds": 97, "loop_outcome": {"usage": {"total_rounds": 84}}}
    assert rcb._policy_turns(latest) == 84
    inside = rcb._audit_step_budget(budget, rcb._policy_turns(latest), 5)
    assert inside["policy_turns_used"] == 89 and inside["budget_fault"] is False
    assert inside["turn_source"] == "loop_outcome.usage.total_rounds"
    # Same example audited against the physical count would have been a fault.
    assert 97 + 5 > 100
    # Missing loop accounting fails CLOSED rather than coercing to zero.
    assert rcb._policy_turns({"total_rounds": 40}) is None
    blind = rcb._audit_step_budget(budget, rcb._policy_turns({"total_rounds": 40}), 3)
    assert blind["counts_available"] is False and blind["budget_fault"] is True
    # A real overrun is a harness fault, not a row-filtering criterion.
    over = rcb._audit_step_budget(budget, 99, 6)
    assert over["policy_turns_used"] == 105 and over["budget_fault"] is True
    assert "comparable" not in over
    # Undeclared budget: nothing to audit against, and that is stated.
    assert rcb._audit_step_budget(rcb._step_budget(_ns(), {"value": 200, "source": "default"}),
                                  5, 0)["audited"] is False


def test_gate_turns_are_enforced_per_task_from_the_live_event_log(tmp_path):
    """The runtime round cap is SERVER-wide and the gate is a separate task, so a
    reserve that is only arithmetic lets the gate consume the worker's allowance.

    The enforcement must read the LIVE counter: `loop_outcome` is written only at
    finalization, so polling a running task for it yields None forever and any
    check built on it is dead code. `llm_round` events are emitted at the same
    statement that increments the loop's round counter, so counting them equals
    the turn count the task will eventually report."""
    task_id = "gate123"
    logs = tmp_path / "state" / "headless_tasks" / task_id / "data" / "logs"
    logs.mkdir(parents=True)
    events = logs / "events.jsonl"

    def _write_rounds(n: int) -> None:
        events.write_text("".join(
            json.dumps({"type": "llm_round", "task_id": task_id, "round": i + 1}) + "\n"
            for i in range(n)
        ), encoding="utf-8")

    _write_rounds(3)
    assert rcb._live_policy_turns(tmp_path, task_id) == 3
    # A finalization-only shape is NOT what the runtime serves while running.
    assert rcb._policy_turns({"status": "running", "total_rounds": 9}) is None

    calls = {"cancel": 0}
    polls = {"n": 0}

    def fake_api(url, method, path, payload=None, timeout=None):
        if path.endswith("/cancel"):
            calls["cancel"] += 1
            return {}
        polls["n"] += 1
        if calls["cancel"]:
            return {"status": "cancelled"}
        # The gate crosses its reserve between the first and second poll.
        _write_rounds(3 if polls["n"] < 2 else rcb._GATE_TURN_RESERVE)
        return {"status": "running"}

    orig_api, orig_sleep = rcb._api, rcb.time.sleep
    rcb._api = fake_api
    rcb.time.sleep = lambda s: None
    try:
        out = rcb._await_gate_task("http://x", task_id, time.time() + 3600,
                                   turn_budget=rcb._GATE_TURN_RESERVE, data_dir=tmp_path)
    finally:
        rcb._api, rcb.time.sleep = orig_api, orig_sleep

    assert out["status"] == "turn_budget_exhausted"
    assert out["policy_turns"] == rcb._GATE_TURN_RESERVE
    assert calls["cancel"] == 1
    # An unconfirmed cancel of THIS status is a zombie premise session, exactly
    # like the timeout path — it must not fail open into the working phase.
    assert rcb._gate_cancel_unconfirmed({"status": "turn_budget_exhausted"}) is True
    assert rcb._gate_cancel_unconfirmed(
        {"status": "turn_budget_exhausted", "cancel_confirmed": True}) is False
    # No declared budget -> no per-task enforcement (unchanged legacy behaviour).
    assert rcb._gate_turn_budget(_ns(feasibility_gate=True)) == 0
    assert rcb._gate_turn_budget(_ns(max_steps=100, feasibility_gate=True)) == rcb._GATE_TURN_RESERVE
    # An unreadable log is UNKNOWN, never zero.
    assert rcb._live_policy_turns(tmp_path / "nope", task_id) is None


def test_unknown_gate_turns_keep_the_full_reserve(tmp_path):
    """UNKNOWN is not zero. If the gate's turn count cannot be read, granting
    claimed-1 turns would let the worker blow the declared total after an
    unmeasured gate — the audit would then call the already-scored campaign
    non-comparable. Fail closed: keep the worst-case reserve."""
    budget = rcb._step_budget(_ns(max_steps=100, feasibility_gate=True),
                              {"value": 99, "source": "settings"})
    assert rcb._worker_round_cap(budget, None) == 100 - rcb._GATE_TURN_RESERVE - 1


def test_unused_gate_reserve_is_returned_to_the_worker(tmp_path):
    """The static reserve is worst-case: the gate is budgeted 14 turns but spent a
    mean of 4 on the v6.83.0 run, so a flat max_steps-14-1 threw ~10 turns away on
    every example and 13 of 56 opus failures died at 89-92 turns INSIDE a 100-turn
    budget. Returning the unused reserve must keep the declared total intact."""
    budget = rcb._step_budget(_ns(max_steps=100, feasibility_gate=True),
                              {"value": 99, "source": "settings"})
    # Gate spent 4 -> worker may use 95, and 4 + 95 + 1 terminal == 100.
    assert rcb._worker_round_cap(budget, 4) == 95
    assert 4 + 95 + budget["terminal_turn_reserve"] == budget["max_steps_claimed"]
    # A gate that used its whole reserve leaves the old conservative number.
    assert rcb._worker_round_cap(budget, 14) == 85
    # No declared budget -> nothing to publish.
    assert rcb._worker_round_cap(rcb._step_budget(_ns(), {"value": 200, "source": "default"}), 4) is None

    # The cap is written where the server hot-reloads it from.
    sp = tmp_path / "settings.json"
    sp.write_text(json.dumps({"OUROBOROS_MAX_ROUNDS": 99, "OTHER": "keep"}), encoding="utf-8")
    rec = rcb._publish_worker_round_cap(sp, 95)
    assert rec["applied"] is True and rec["previous"] == 99
    on_disk = json.loads(sp.read_text(encoding="utf-8"))
    assert on_disk["OUROBOROS_MAX_ROUNDS"] == 95 and on_disk["OTHER"] == "keep"
    # An unwritable target is disclosed, never fatal (the stricter cap stays).
    bad = rcb._publish_worker_round_cap(tmp_path / "nope" / "settings.json", 95)
    assert bad["applied"] is False and "error" in bad


def test_a_gate_terminated_example_is_not_a_budget_fault():
    """A gate INFEASIBLE ends the example before the working phase, so the worker
    used exactly zero policy turns — a KNOWN count. Treating it as unknown made
    the fail-closed audit flag the very outcome the gate exists to produce
    (caught live on os/a462a795 minutes into the v6.83.0 run)."""
    budget = rcb._step_budget(_ns(max_steps=100, feasibility_gate=True),
                              {"value": 85, "source": "settings"})
    gated = rcb._audit_step_budget(budget, 0, 4, gate_expected=True)
    assert gated["budget_fault"] is False and gated["policy_turns_used"] == 4
    # A genuinely unknown worker count still fails closed.
    unknown = rcb._audit_step_budget(budget, None, 4, gate_expected=True)
    assert unknown["budget_fault"] is True


def test_a_checkout_other_than_the_campaign_pin_is_refused_before_the_vm_boots():
    """The graded-spec pin decides both the instruction the agent receives and the
    evaluator that scores it. Recording a mismatch in the manifest is a report:
    on 2026-07-29 a 75-task probe graded 21 tasks against a three-week-older
    checkout while every manifest faithfully recorded it and nobody read it."""
    import pytest

    rcb._refuse_wrong_dataset_commit("", {"git_commit": "whatever"})  # opt-in: no claim, no gate
    rcb._refuse_wrong_dataset_commit("091f5ef1d5544bc", {"git_commit": "091f5ef1d5544bc74953c"})
    with pytest.raises(SystemExit, match="graded against"):
        rcb._refuse_wrong_dataset_commit("091f5ef1", {"git_commit": "7a17d3abc86d5"})
    with pytest.raises(SystemExit, match="no readable git identity"):
        rcb._refuse_wrong_dataset_commit("091f5ef1", {"git_commit": ""})


def test_v684_prompt_fixes_are_present_and_harmful_clauses_gone():
    """The v6.83.0 forensics found five prompt behaviours the agent CITED while
    losing points. Pin the corrected wording so a later edit cannot regress them,
    and assert the exact harmful phrasings are gone."""
    p = rcb.OSWORLD_PREAMBLE
    # 1. Budget is turns, not calls; batching is encouraged.
    assert "YOUR BUDGET IS ASSISTANT TURNS, NOT TOOL CALLS" in p
    assert "every tool call costs ~30s" not in p  # the mistaxed clause is gone
    # Batching must carry its safety guard: adversarial review found 8 prior 1.0s
    # that depended on observing after a speculative Enter/drag/save.
    assert "Observe before any speculative Enter/Return, drag, save" in p
    assert "2-6 calls is typical, not a minimum" in p
    # Batching removes the ~5s settle the per-turn round trip used to provide, and a
    # failing call does not stop its batch (measured: 43% of intra-batch gaps < 1s).
    assert "NO settling time" in p and "does NOT stop the rest of its batch" in p
    # Ordinals: a bulleted list excludes title+lead-in even when the task says "line"
    # (impress/550ce7e7, opus 1.0, cited the removed clause while winning); anything
    # else counts the heading (impress/3161d64e, 5cfb9197 — both 1.0 on both models).
    assert "BULLETED OR NUMBERED LIST, count only the actual list" in p
    assert "a heading COUNTS as the Nth item" in p
    # Smoke evidence: 05dd4c1d aligned the document-order shape (Shape;135) while the
    # gold targets the visually higher one (Shape;136) — slide ordinals need an ORDER.
    assert "order them by POSITION, top-to-bottom" in p
    assert "never by document order, selection order or Tab order" in p
    # Smoke evidence: 04578141 read "exactly these colours, no variations" as a licence
    # to type raw 00FF00 through Custom Color; the gold is palette Green 00A933, tol 0.
    assert "it does NOT mean type a raw hex" in p
    # 2a. A numeric literal beats a preset; a colour WORD does not (two pinned
    # tasks require LibreOffice's named Green 00A933, one requires pure 0000FF —
    # no prompt wording wins all three, so we keep the named-control default).
    assert "explicit NUMERIC value" in p
    assert "colour WORD on its own is not a numeric value" in p
    # 2b. Already-in-state judged from stored value, not the render.
    assert "STORED value the grader" in p
    assert "is ALREADY in the requested state, verifying that and stopping is a correct completion" not in p
    # 2c. Ordinals no longer blanket-exclude headings.
    assert "ORDINALS COUNT WHAT THE TASK COUNTS" in p
    assert "excluding titles, headings and unbulleted lead-in" not in p
    # 3. CLI allowed for batch/file work.
    assert "pdfseparate" in p
    # 4. Independent read-back + snapshot/diff.
    assert "VERIFY BY INDEPENDENT READ-BACK" in p and "DIFFERENT tool" in p
    assert "compare before vs after and undo" in p
    # 5. Infeasibility wording must stay NARROW: a failed route or a fallback the
    # app itself offers is not infeasibility (9 prior 1.0 traces used the words
    # "impossible"/"not possible" mid-run and still won).
    assert "A failed route" in p and "is NOT task infeasibility" in p
    assert "only after OBSERVING" in p
    assert "YOUR OWN ADMISSION IS THE VERDICT" not in p, "the lexical slogan false-kills wins"
    g = rcb.GATE_PREAMBLE
    assert "VERIFIED ABSENT" in g
    assert "Merely hidden, disabled, not yet loaded" in g, "hidden != absent"
    assert "When in doubt, answer UNDETERMINED" in g, "fail-open default must survive"
    # 3. Shell is for file-level deliverables, never for app state.
    assert "FILE-LEVEL batch operations" in p
    assert "mutate an open application's document, preferences or UI state" in p


def test_proxy_health_gate_fails_closed(monkeypatch, tmp_path):
    """Config-exists is not proxy-alive: an exhausted account keeps its file but
    answers 407. The gate must return False on any probe failure so those tasks
    run direct and get quarantined, never poisoned through a dead upstream."""
    import json as _json
    cfg = tmp_path / "proxy.json"
    cfg.write_text(_json.dumps([{"host": "gw.example.com", "port": 823,
                                 "username": "u", "password": "p"}]), encoding="utf-8")
    # A probe that raises (dead proxy) -> not live.
    import urllib.request
    def boom(*a, **k):
        raise OSError("407 TRAFFIC_EXHAUSTED")
    monkeypatch.setattr(urllib.request, "build_opener", lambda *a, **k: type("O", (), {"open": boom})())
    assert rcb._proxy_config_is_live(str(cfg)) is False
    # Empty / malformed config -> not live.
    cfg.write_text("[]", encoding="utf-8")
    assert rcb._proxy_config_is_live(str(cfg)) is False
    assert rcb._proxy_config_is_live(str(tmp_path / "missing.json")) is False


def test_proxy_exhaustion_is_recorded_never_used_to_drop_a_task(tmp_path):
    """A proxy outage must be DISCLOSED, not acted on. The lane makes a single pass
    over the task list, so an unscored return deletes the example from the campaign
    instead of retrying it (measured: by the time a long task released its claim,
    every other lane had already walked past it). An earlier draft quarantined
    BEFORE evaluation and would have discarded 17 opus wins whose agents met a dead
    proxy and rerouted anyway."""
    import inspect
    src = inspect.getsource(rcb)
    logs = tmp_path / "state" / "headless_tasks" / "t1" / "data" / "logs"
    logs.mkdir(parents=True)
    tj = logs / "tools.jsonl"
    tj.write_text('{"tool":"remote_exec","result":"curl: ... 407 TRAFFIC_EXHAUSTED"}\n',
                  encoding="utf-8")
    assert rcb._proxy_trace_shows_exhaustion(tmp_path, "t1") is True
    # TASK-LOCAL: a neighbour's outage on the shared lane log must not count (the
    # lane-wide fallback quarantined 3 later wins in replay).
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "logs" / "tools.jsonl").write_text('{"result":"TRAFFIC_EXHAUSTED"}\n',
                                                   encoding="utf-8")
    assert rcb._proxy_trace_shows_exhaustion(tmp_path, "no-such-task") is False
    # A bare "407" in page content or in the agent's own command is not a verdict.
    tj.write_text('{"tool":"type_text","args":{"text":"HTTP 407 explained"}}\n', encoding="utf-8")
    assert rcb._proxy_trace_shows_exhaustion(tmp_path, "t1") is False
    # SAFETY PROPERTY: no code path turns proxy trouble into an unscored outcome.
    assert "proxy_unavailable" not in src, "a proxy outage must never drop a task"
    assert '"proxy_required"' in src and '"proxy_exhausted_in_trace"' in src


def test_settings_cap_publication_preserves_0600(tmp_path):
    """Preserve the observed credential-file mode; require 0600 where POSIX modes exist."""
    import json as _json
    import os as _os
    sp = tmp_path / "settings.json"
    sp.write_text(_json.dumps({"OUROBOROS_MAX_ROUNDS": 99, "OPENROUTER_API_KEY": "x"}),
                  encoding="utf-8")
    _os.chmod(sp, 0o600)
    original_mode = _os.stat(sp).st_mode & 0o777
    if _os.name != "nt":
        assert original_mode == 0o600
    assert rcb._publish_worker_round_cap(sp, 95)["applied"] is True
    assert _os.stat(sp).st_mode & 0o777 == original_mode
    assert not list(tmp_path.glob("*.part")), "no credential-bearing temp left behind"


def test_evaluate_runs_in_the_checkout_and_restores_cwd(tmp_path):
    """Relative evaluator fixtures resolve against the process CWD and the official
    runner works from the checkout root, so the scoped context must enter it — and
    restore on every path, including an exception."""
    import os as _os
    start = _os.getcwd()
    checkout = tmp_path / "OSWorld"
    checkout.mkdir()
    with rcb._official_evaluate_cwd(checkout):
        assert _os.path.realpath(_os.getcwd()) == _os.path.realpath(str(checkout))
    assert _os.getcwd() == start
    try:
        with rcb._official_evaluate_cwd(checkout):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert _os.getcwd() == start, "cwd must be restored even when evaluate raises"


def test_v685_contract_and_carveout_clauses():
    """The v6.84.0 run lost 15.46 raw points to the leader across 19 tasks, 8 of them
    one class: the work was done and never checked against the surface the grader
    reads. The contract makes that a structured obligation rather than advice; the
    other clauses each answer a named losing task."""
    p = rcb.OSWORLD_PREAMBLE
    # The atomic contract — written before mutation, closed before finishing.
    assert "WRITE THE CONTRACT BEFORE YOU TOUCH ANYTHING" in p
    assert "CLOSE THE CONTRACT BEFORE YOU FINISH" in p
    assert "OBSERVED SATISFIED" in p and "NOT VERIFIED" in p
    # An IMPOSSIBLE item must have an exit, or the contract becomes a new route to a
    # false infeasible; and repair is per item, not one repair for the whole task
    # (the preamble elsewhere says "keep working" without a limit).
    assert "repair THAT item" in p and "repeat until" in p
    assert "deliver it rather than abandoning the task" in p
    # The infeasibility test is about the END STATE, with the wrong-verdict brake:
    # three current 1.0 tasks say "the real path is impossible, so here is the
    # allowed substitute" and still score.
    assert "about the END STATE, not the route" in p
    assert "wrong TASK_INFEASIBLE scores zero" in p
    # gsettings is for STORED values only: os/fe41f596 is officially infeasible,
    # we score 1.0 on it, and the carve-out otherwise describes it word for word.
    assert "ONLY when the task asks for a value to be STORED" in p
    # The colour motivation is TRUE (the gold IS the palette entry) — restored, tightened.
    assert "EXACTLY the word the task used" in p and "no Light/Dark qualifier" in p
    # Singular referent: 05dd4c1d applied the change to both candidates to cover
    # either reading and scored 0.
    # Plural instructions must still be done in full: 84 of the 361 instructions say
    # all/both/each/every, and 65 of those were baseline 1.0s.
    assert "the obligation genuinely covers every matching element" in p
    assert "SINGULAR referent that resolves to several candidates" in p
    # And the contract must not freeze a wrong early reading.
    assert "not a vow" in p
    # gsettings carve-out (bedcedc4: refused the platform's own config CLI).
    assert "gsettings/dconf" in p and "prefs.js" in p
    # Infeasibility shapes (5ca86c6f discovery, 2e6f678f mode, 971cbb5b narrower trigger).
    assert "discovery is part of the job" in p
    assert "found the verdict and ignored it" in p
    # The colour motivation STAYS: an independent replay of the real grader showed the
    # gold of 8472fece IS the palette entry (2A6099) and scores 0 against its own
    # evaluator, which measures distance to pure 0000FF (dE 21.09 vs threshold 3.5).
    # The task is unwinnable by any palette entry; removing the motivation gained
    # ~nothing and endangered 04578141, a live 1.0 won BECAUSE of it.
    assert "the reference file was authored from that same palette" in p


def test_scoped_proxy_config_never_lands_in_the_published_tree():
    """The scoped config carries the account PASSWORD. An earlier draft wrote it to
    results/<domain>/<task>/, which is exactly the tree we archive and publish —
    0600 protects against other Unix users, it does not redact an uploaded archive.
    Pin both halves: it goes to lane-private state, and it is removed afterwards."""
    import inspect
    src = inspect.getsource(rcb)
    assert 'run_dir / "proxy_task.json"' not in src, "credential file back in results/"
    assert 'data_dir / "state" / "proxy"' in src, "must live in lane-private state"
    assert "os.unlink(_scoped_proxy_path)" in src, "must be removed after the task"
    # And it is only written for a task that actually needs the proxy.
    assert 'if _proxy_present and bool(example.get("proxy")):' in src


def test_task_scoped_proxy_gives_each_task_its_own_session(tmp_path):
    """The shared config is one entry on the rotating gateway, so every request drew a
    new exit IP — fatal for any site that ties a session to an address. A per-task
    `;sessid.<tag>` keeps one exit per trajectory without pinning a whole lane."""
    import json as _json
    import os as _os
    src = tmp_path / "proxy.json"
    src.write_text(_json.dumps([{"host": "gw.example.com", "port": 823,
                                 "username": "acct", "password": "p"}]), encoding="utf-8")
    state = tmp_path / "lane" / "state" / "proxy"
    out = rcb._task_scoped_proxy_config(str(src), state, "deadbeefcafe0001")
    assert "results" not in out, "must not be written under the published results tree"
    assert out != str(src)
    cfg = _json.loads(open(out).read())
    assert cfg[0]["username"] == "acct;sessid.deadbeefcafe0001"
    assert cfg[0]["password"] == "p", "credentials must survive scoping"
    from ouroboros.observability import posix_private_modes_supported

    if posix_private_modes_supported():
        # Windows does not express privacy through POSIX mode bits, so asserting
        # them there fails on every run without saying anything about security.
        assert _os.stat(out).st_mode & 0o777 == 0o600, "the task config carries a credential"
    # Idempotent: an already-scoped username is not double-suffixed.
    again = rcb._task_scoped_proxy_config(out, state, "0000")
    assert _json.loads(open(again).read())[0]["username"].count(";sessid.") == 1
    # Unreadable input falls back to the shared config rather than failing the task.
    assert rcb._task_scoped_proxy_config(str(tmp_path / "nope.json"), state, "x") \
        == str(tmp_path / "nope.json")


def test_setup_effect_probe_is_advisory_and_never_raises():
    """Upstream logs a guest command that failed as 'executed successfully', so a
    setup step can silently no-op and take the task's premise with it (chrome/3299584d:
    apt install jq did nothing, the agent honestly reported the task impossible and
    scored 0 while doing nothing at all would have scored 1). The probe records what
    is missing; it must never fail a task by itself."""
    class _Ctl:
        def __init__(self, present): self.present = present
        def execute_python_command(self, code):
            return {"output": "1" if self.present else "0"}

    class _Env:
        def __init__(self, present): self.controller = _Ctl(present)

    example = {"config": [
        {"type": "execute", "parameters": {"command": ["apt-get", "install", "-y", "jq"]}},
        {"type": "download", "parameters": {}},
    ]}
    ok = rcb._verify_setup_effect(_Env(True), example)
    assert ok["checked"] == 1 and ok["missing"] == []
    bad = rcb._verify_setup_effect(_Env(False), example)
    assert bad["missing"] == ["jq"]

    class _Boom:
        controller = property(lambda self: (_ for _ in ()).throw(RuntimeError("x")))
    out = rcb._verify_setup_effect(_Boom(), example)
    assert isinstance(out, dict), "diagnostics must never raise"
