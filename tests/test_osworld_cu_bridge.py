"""The OSWorld cu_bridge helpers and the environment the runner publishes.

This module owns the pure predicate helpers (infeasibility, a11y, connection switching,
the live-server guard, dataset naming, round and budget counters), the publication of the
target registry and the settings cap, the DesktopEnv construction retry, the guest-health
probe, and the per-task proxy sessions the runner hands out.

The claim custody, the provenance refusals, the feasibility gate and the worker prompt
clauses were split verbatim into ``tests/test_osworld_cu_bridge_claims.py``,
``tests/test_osworld_cu_bridge_provenance.py``, ``tests/test_osworld_cu_bridge_gate.py``
and ``tests/test_osworld_cu_bridge_prompts.py``; the stubs they share live in
``tests/_osworld_cu_bridge_shared.py``.

These exercise the pure helpers only — no OSWorld VM, no Ouroboros server.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from devtools.benchmarks.osworld import run_cu_bridge_agent as rcb
from ouroboros.extension_loader import extension_surface_name


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
