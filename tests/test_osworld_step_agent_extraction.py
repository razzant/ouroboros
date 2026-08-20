"""Structural contracts for the semantic-no-op OSWorld step-loop extraction.

`run_step_agent.py` is a benchmark launcher AND the module `run_cu_bridge_agent`
imports its shared helpers from, so the extraction has to leave `main()`, the
admission/finalization seams and the whole importable surface exactly where they
were.
"""

from __future__ import annotations

import ast
import pathlib

from devtools.benchmarks.osworld import (
    run_step_agent as rsa,
    step_agent_actions,
    step_agent_claims,
    step_agent_common,
    step_agent_env,
    step_agent_policy,
)


REPO = pathlib.Path(__file__).parents[1]
OSWORLD = REPO / "devtools" / "benchmarks" / "osworld"
_LEAVES = (
    step_agent_common,
    step_agent_env,
    step_agent_claims,
    step_agent_actions,
    step_agent_policy,
)

_MOVED_OWNERS = {
    "StepAgentConfig": step_agent_common,
    "TaskRecordConfig": step_agent_common,
    "PreflightConfig": step_agent_common,
    "_safe_slug": step_agent_common,
    "_http_json": step_agent_common,
    "VMWARE_FUSION_PATHS": step_agent_env,
    "ALIGNED_UPSTREAM": step_agent_env,
    "SUPPORTED_PROVIDERS": step_agent_env,
    "osworld_checkout_info": step_agent_env,
    "provider_preflight_failures": step_agent_env,
    "_install_optional_dependency_stubs": step_agent_env,
    "_ensure_vmrun_on_path": step_agent_env,
    "_DEFAULT_DESKTOP_PORT": step_agent_env,
    "_LOOPBACK_HOSTS": step_agent_env,
    "_is_default_desktop_server": step_agent_env,
    "_teardown_partial_desktop_env": step_agent_env,
    "construct_desktop_env": step_agent_env,
    "ClaimDirNotConfined": step_agent_claims,
    "confined_claims_dir": step_agent_claims,
    "task_claim_key": step_agent_claims,
    "claim_stale_sec": step_agent_claims,
    "acquire_task_claim": step_agent_claims,
    "UNCONFIRMED_SCORE_SUFFIX": step_agent_claims,
    "ClaimMarkerNotDurable": step_agent_claims,
    "record_unconfirmed_score": step_agent_claims,
    "mark_task_scored": step_agent_claims,
    "scored_claim_state": step_agent_claims,
    "task_already_scored": step_agent_claims,
    "release_task_claim": step_agent_claims,
    "SPECIAL_ACTIONS": step_agent_actions,
    "_json_from_text": step_agent_actions,
    "_shell_action": step_agent_actions,
    "_click_action": step_agent_actions,
    "_type_action": step_agent_actions,
    "_hotkey_action": step_agent_actions,
    "_wait_action": step_agent_actions,
    "_normalize_structured_action": step_agent_actions,
    "_initial_observation_with_retries": step_agent_policy,
    "OuroborosStepAgent": step_agent_policy,
}

# The exact names run_cu_bridge_agent.py imports FROM run_step_agent.py. The
# split must not force that importer to learn the new owners.
_CU_BRIDGE_IMPORTS = (
    "_is_default_desktop_server", "confined_claims_dir", "scored_claim_state",
    "task_claim_key", "amend_task_manifest", "ClaimMarkerNotDurable",
    "acquire_task_claim", "claim_stale_sec", "construct_desktop_env",
    "mark_task_scored", "osworld_checkout_info", "record_unconfirmed_score",
    "release_task_claim",
)


def test_step_agent_leaves_never_import_the_launcher_and_own_no_entry_point():
    for module in _LEAVES:
        source = pathlib.Path(module.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        assert not any(
            isinstance(node, ast.FunctionDef) and node.name == "main"
            for node in tree.body
        ), module.__name__
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert "run_step_agent" not in (node.module or ""), module.__name__
            if isinstance(node, ast.Import):
                assert not any("run_step_agent" in alias.name for alias in node.names)
        assert "admit_benchmark_run(" not in source, module.__name__
        assert "benchmark_run_manifest(" not in source, module.__name__
        assert "finalize_run_manifest(" not in source, module.__name__


def test_step_agent_launcher_keeps_main_the_seams_and_the_attestation_call():
    source = (OSWORLD / "run_step_agent.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {
        node.name for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    assert {"main", "admit_step_loop_run", "_preflight", "_run_step_loop"} <= names
    assert "admit_benchmark_run(" in source
    assert "finalize_run_manifest(" in source
    assert "benchmark_run_manifest(" not in source
    # `_preflight` stays in the launcher precisely because this attestation call
    # is pinned to this file by tests/test_devtools_benchmarks.py.
    assert "runtime_attestation(config.ouroboros_url, config.repo_dir)" in source


def test_step_agent_launcher_reexports_every_moved_identity():
    for name, owner in _MOVED_OWNERS.items():
        assert hasattr(rsa, name), name
        assert getattr(rsa, name) is getattr(owner, name), name


def test_cu_bridge_still_imports_its_shared_helpers_from_the_launcher():
    for name in _CU_BRIDGE_IMPORTS:
        assert hasattr(rsa, name), name


def test_no_step_loop_module_fabricates_bash_history():
    """The family-wide half of tests/test_devtools_benchmarks.py's NW-6 check.

    That test names ONE file, which is exactly what the split can make vacuous:
    `_shell_action` moved to the actions leaf, and an absence assertion aimed at
    the file the code left would pass over an empty haystack. Here the whole
    family is read and the documented omission must still be present somewhere,
    so the haystack is provably non-empty.
    """
    sources = {
        path.name: path.read_text(encoding="utf-8")
        for path in [
            OSWORLD / "run_step_agent.py",
            *sorted(OSWORLD.glob("step_agent_*.py")),
        ]
    }
    assert any(".bash_history" in text for text in sources.values()), sorted(sources)
    for name, text in sources.items():
        assert "hist.open(" not in text, name
        assert "record_history" not in text, name
        assert ".bash_history'" not in text, name


def test_step_agent_extraction_size_bounds_have_meaningful_headroom():
    counts = {
        path.name: len(path.read_text(encoding="utf-8").splitlines())
        for path in (
            OSWORLD / "run_step_agent.py",
            *(pathlib.Path(module.__file__) for module in _LEAVES),
        )
    }
    assert all(count <= 1000 for count in counts.values()), counts
