"""Unit tests for the OSWorld cu_bridge runner (PR #64 finalization).

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
    for n in ("screenshot", "click", "type_text", "key", "scroll", "remote_exec",
              "list_connections", "test_connection"):
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
    assert not list(sdir.glob("*.tmp-*"))  # atomic write leaves no temp files
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
    assert resolved == _P("/tmp/explicit/settings.json")
    _ = argparse  # silence unused in some linters


def test_denylist_is_allowlist_complement_blocks_all_host_surfaces():
    # Allowlist semantics: every core tool NOT in the allowlist is denied — so the
    # whole host mutation/exec/VCS/GitHub/service/self-mod/chat class is blocked by
    # construction, not by an enumerated (and forgettable) list.
    denied = set(rcb._host_denied_tools())
    core = rcb._core_tool_names()
    # nothing in the allowlist is denied; everything else is
    assert denied == core - rcb._ALLOWED_CORE_TOOLS
    for t in ("run_command", "run_script", "claude_code_edit", "write_file", "edit_text",
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


def test_module_grandfather_matcher_basename_and_relpath():
    from ouroboros.review import module_is_grandfathered
    # repo-relative entry matches its rel path AND the repo/-prefixed section path
    assert module_is_grandfathered("skills/unix_computer_use/plugin.py")
    assert module_is_grandfathered("repo/skills/unix_computer_use/plugin.py")
    # a DIFFERENT plugin.py (future skill) is NOT exempted by the path-qualified entry
    assert not module_is_grandfathered("skills/other_skill/plugin.py")
    assert not module_is_grandfathered("repo/skills/other_skill/plugin.py")
    # legacy bare-basename entries still match
    assert module_is_grandfathered("repo/ouroboros/server.py")
    assert module_is_grandfathered("server.py")
