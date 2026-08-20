"""Structural contracts for the semantic-no-op OSWorld cu_bridge extraction.

`run_cu_bridge_agent.py` is a benchmark launcher, so the extraction has to leave
three things exactly where they were: the `main()` entry point, the admission /
finalization seams the launcher gate walks, and the module surface tests and
operators reach through `run_cu_bridge_agent.<name>`.
"""

from __future__ import annotations

import ast
import pathlib

from devtools.benchmarks.osworld import (
    cu_bridge_budget,
    cu_bridge_gate,
    cu_bridge_prompts,
    cu_bridge_runtime,
    cu_bridge_tool_policy,
    run_cu_bridge_agent as rcb,
)


REPO = pathlib.Path(__file__).parents[1]
OSWORLD = REPO / "devtools" / "benchmarks" / "osworld"
_LEAVES = (
    cu_bridge_runtime,
    cu_bridge_prompts,
    cu_bridge_tool_policy,
    cu_bridge_gate,
    cu_bridge_budget,
)

_MOVED_OWNERS = {
    "SKILL_NAME": cu_bridge_runtime,
    "_api": cu_bridge_runtime,
    "_text_declares_infeasible": cu_bridge_runtime,
    "_terminal_answer_text": cu_bridge_runtime,
    "_final_answer_declares_infeasible": cu_bridge_runtime,
    "GATE_PREAMBLE": cu_bridge_prompts,
    "GATE_SUFFIX": cu_bridge_prompts,
    "OSWORLD_PREAMBLE": cu_bridge_prompts,
    "_ACCEPTANCE_CLAIMS": cu_bridge_prompts,
    "_ALLOWED_CORE_TOOLS": cu_bridge_tool_policy,
    "_core_tool_names": cu_bridge_tool_policy,
    "_host_denied_tools": cu_bridge_tool_policy,
    "_GUI_ACTION_TOOLS": cu_bridge_tool_policy,
    "_DENIED_SKILL_EXT_TOOLS": cu_bridge_tool_policy,
    "_effective_disabled_tools": cu_bridge_tool_policy,
    "_COMPUTER_USE_SHORT_TOOLS": cu_bridge_tool_policy,
    "_gate_window_sec": cu_bridge_gate,
    "_gate_claim_window_sec": cu_bridge_gate,
    "_gate_verdict": cu_bridge_gate,
    "_DesktopEnvLogCapture": cu_bridge_gate,
    "ResetUnverified": cu_bridge_gate,
    "_reset_verified": cu_bridge_gate,
    "_live_policy_turns": cu_bridge_gate,
    "_policy_turns": cu_bridge_gate,
    "_await_gate_task": cu_bridge_gate,
    "_gate_round": cu_bridge_gate,
    "_GATE_TURN_RESERVE": cu_bridge_gate,
    "_GUEST_DOWN_GRACE_SEC": cu_bridge_gate,
    "_guest_endpoint_healthy": cu_bridge_gate,
    "_gate_cancel_unconfirmed": cu_bridge_gate,
    "_gate_tool_trace": cu_bridge_gate,
    "_gate_turn_budget": cu_bridge_gate,
    "_effective_max_rounds": cu_bridge_budget,
    "_step_budget": cu_bridge_budget,
    "_official_evaluate_cwd": cu_bridge_budget,
    "_worker_round_cap": cu_bridge_budget,
    "_publish_worker_round_cap": cu_bridge_budget,
    "_proxy_trace_shows_exhaustion": cu_bridge_budget,
    "_verify_setup_effect": cu_bridge_budget,
    "_task_scoped_proxy_config": cu_bridge_budget,
    "_proxy_config_is_live": cu_bridge_budget,
    "_refuse_wrong_dataset_commit": cu_bridge_budget,
    "_refuse_uncapped_step_claim": cu_bridge_budget,
    "_audit_step_budget": cu_bridge_budget,
    "_collect_budget_counters": cu_bridge_budget,
}


def test_cu_bridge_leaves_never_import_the_launcher_and_own_no_entry_point():
    for module in _LEAVES:
        source = pathlib.Path(module.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        assert not any(
            isinstance(node, ast.FunctionDef) and node.name == "main"
            for node in tree.body
        ), module.__name__
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert "run_cu_bridge_agent" not in (node.module or ""), module.__name__
            if isinstance(node, ast.Import):
                assert not any("run_cu_bridge_agent" in alias.name for alias in node.names)
        # The admission/finalization seams belong to the launcher alone: a leaf
        # writing a manifest would publish outside the gate the launcher walks.
        assert "admit_benchmark_run(" not in source, module.__name__
        assert "benchmark_run_manifest(" not in source, module.__name__
        assert "finalize_run_manifest(" not in source, module.__name__


def test_cu_bridge_launcher_keeps_main_and_both_manifest_seams():
    source = (OSWORLD / "run_cu_bridge_agent.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert any(
        isinstance(node, ast.FunctionDef) and node.name == "main" for node in tree.body
    )
    assert "admit_benchmark_run(" in source
    assert "finalize_run_manifest(" in source
    assert "benchmark_run_manifest(" not in source
    assert "runtime_attestation(args.ouroboros_url, repo_dir)" in source


def test_cu_bridge_launcher_reexports_every_moved_identity():
    for name, owner in _MOVED_OWNERS.items():
        assert hasattr(rcb, name), name
        assert getattr(rcb, name) is getattr(owner, name), name


def test_cu_bridge_extraction_size_bounds_have_meaningful_headroom():
    counts = {
        path.name: len(path.read_text(encoding="utf-8").splitlines())
        for path in (
            OSWORLD / "run_cu_bridge_agent.py",
            *(pathlib.Path(module.__file__) for module in _LEAVES),
        )
    }
    assert counts["run_cu_bridge_agent.py"] < 1500
    assert all(
        count <= 1000
        for name, count in counts.items()
        if name != "run_cu_bridge_agent.py"
    ), counts
