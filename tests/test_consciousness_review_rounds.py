"""Review-round pins for the consciousness redesign: the origin survives the globalized project
view, a consciousness-started deep review stays inside the allowance, and a started root reads
the runtime mode it actually runs in (split out of test_consciousness_authority.py for size)."""
from __future__ import annotations

import types

from tests.test_consciousness_authority import _wake_task

def test_globalized_project_promotion_keeps_the_origin(monkeypatch):
    """A Full consciousness project task's post-task promotion must not shed the origin: the
    campaign it may produce stays inside the consciousness limits."""
    from ouroboros import agent_task_pipeline as pipeline

    captured: list = []
    monkeypatch.setattr(pipeline, "_update_improvement_backlog", lambda env, entry: None)
    monkeypatch.setattr("ouroboros.post_task_evolution.maybe_promote",
                        lambda env, task, entry, llm: captured.append(task))
    task = {"id": "p-1", "type": "task", "metadata": dict(_wake_task("full")["metadata"]),
            "task_contract": {"disabled_tools": []}}
    entry = {"backlog_candidates": [{"summary": "make it better"}]}
    pipeline._run_global_backlog_promotion_only(types.SimpleNamespace(), task, entry, None)
    assert captured and captured[0]["metadata"] == {
        "globalized_from_project_task": True, "initiator": "consciousness",
        "usage_category": "consciousness_task", "consciousness_autonomy": "full"}
    captured.clear()
    pipeline._run_global_backlog_promotion_only(types.SimpleNamespace(), {"id": "p-2", "type": "task"}, entry, None)
    assert captured[0]["metadata"] == {"globalized_from_project_task": True}


# --- round 3: a consciousness-started deep review stays inside the allowance; a started root
# --- reads the mode it runs in ---------------------------------------------------------


def test_a_consciousness_started_review_keeps_the_tree_category():
    """The allowance discovers its roots by the consciousness categories: a review root whose only
    priced rows said `deep_self_review` was invisible to it (astra round 3)."""
    from ouroboros.deep_self_review import _review_usage_scope
    from ouroboros.review_records import ReviewRequest
    from ouroboros.usage_accounting import UsageScope

    def scope(current):
        return _review_usage_scope(current, ReviewRequest(surface="deep_self_review", goal="review"))

    kept = scope(UsageScope(category="consciousness_task", source="agent.task"))
    assert kept.category == "consciousness_task" and kept.source == "deep_self_review"
    assert kept.review_wave_id.startswith("wave-")  # the review names its round (#1544)
    wake = scope(UsageScope(category="consciousness", source="agent.task"))
    assert wake.category == "consciousness"
    own = scope(UsageScope(category="task", source="agent.task"))
    assert own.category == "deep_self_review" and own.source == "deep_self_review"
    assert scope(UsageScope()).category == "deep_self_review"


def test_a_started_root_reads_the_mode_it_runs_in_but_the_wake_keeps_mains(tmp_path, monkeypatch):
    """The dispatcher caps a started root to light (Act/Observe); its Runtime block says so. The
    wake itself is a direct turn and keeps Main's block byte-identical (В31=B)."""
    import json

    from ouroboros.context import build_runtime_section
    from tests.test_context_runtime_section import _make_health_env

    env = _make_health_env(tmp_path)
    monkeypatch.setattr("ouroboros.config.get_runtime_mode", lambda: "advanced")
    started = {"id": "root-1", "type": "task", "metadata": dict(_wake_task("act")["metadata"])}
    payload = json.loads(build_runtime_section(env, started).split("\n\n", 1)[1])
    assert payload["runtime_mode"] == "light" and "forbids Ouroboros repo mutation" in payload["runtime_mode_rule"]
    wake = {"id": "wake-1", "type": "task", "_is_direct_chat": True, "metadata": dict(_wake_task("act")["metadata"])}
    payload = json.loads(build_runtime_section(env, wake).split("\n\n", 1)[1])
    assert payload["runtime_mode"] == "advanced" and "runtime_mode_rule" not in payload
    full = {"id": "root-2", "type": "task", "metadata": dict(_wake_task("full")["metadata"])}
    assert json.loads(build_runtime_section(env, full).split("\n\n", 1)[1])["runtime_mode"] == "advanced"


def test_an_integration_reads_the_mode_the_task_runs_in(monkeypatch):
    """A capped tree (Act/Observe: light) cannot land a system-repo patch the install mode alone
    would allow (astra scope round 4)."""
    from ouroboros.tools.subagent_integration import _integration_runtime_mode

    monkeypatch.setattr("ouroboros.tools.subagent_integration.get_runtime_mode", lambda: "advanced")
    assert _integration_runtime_mode(types.SimpleNamespace(task_metadata=dict(_wake_task("act")["metadata"]))) == "light"
    assert _integration_runtime_mode(types.SimpleNamespace(task_metadata=dict(_wake_task("full")["metadata"]))) == "advanced"
    assert _integration_runtime_mode(types.SimpleNamespace(task_metadata={})) == "advanced"


def test_a_capped_tree_cannot_schedule_a_self_worktree_child(monkeypatch):
    """Act may write, but never into its own repository — its children included (В21=A,
    PLAN §5.4): on an advanced install the per-task light cap keeps a self_worktree child
    off while an external-workspace child stays available (astra scope round 4)."""
    from ouroboros.tools.control_scheduling import _build_acting_constraint

    monkeypatch.delenv("OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS", raising=False)
    monkeypatch.delenv("OUROBOROS_BOOT_RUNTIME_MODE", raising=False)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    capped = types.SimpleNamespace(task_metadata=dict(_wake_task("act")["metadata"]))
    refused = _build_acting_constraint(write_surface="self_worktree", write_root="", protected_paths_grant=False,
                                       external_tool_grants=None, parent_workspace_root="", ctx=capped)
    assert "MUTATIVE_SUBAGENTS_DISABLED" in str(getattr(refused, "text", refused))
    allowed = _build_acting_constraint(write_surface="external_workspace", write_root="/tmp/x", protected_paths_grant=False,
                                       external_tool_grants=None, parent_workspace_root="", ctx=capped)
    assert isinstance(allowed, dict), allowed
    full = types.SimpleNamespace(task_metadata=dict(_wake_task("full")["metadata"]))
    assert isinstance(_build_acting_constraint(write_surface="self_worktree", write_root="", protected_paths_grant=False,
                                               external_tool_grants=None, parent_workspace_root="", ctx=full), dict)
    # A Light install whose owner explicitly enabled mutative subagents admits self_worktree children —
    # but never for a capped tree: the cap is the level's, not the install's (astra scope round 6).
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    monkeypatch.setenv("OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS", "true")
    refused = _build_acting_constraint(write_surface="self_worktree", write_root="", protected_paths_grant=False,
                                       external_tool_grants=None, parent_workspace_root="", ctx=capped)
    assert "light cap" in str(getattr(refused, "text", refused))
    assert isinstance(_build_acting_constraint(write_surface="self_worktree", write_root="", protected_paths_grant=False,
                                               external_tool_grants=None, parent_workspace_root="", ctx=full), dict)


def test_a_capped_tree_may_not_land_a_system_repo_patch_in_any_mode():
    from ouroboros.tools.subagent_integration import _capped_self_repo_refusal

    capped = types.SimpleNamespace(task_metadata=dict(_wake_task("act")["metadata"]))
    assert "INTEGRATE_CAPPED_TREE" in _capped_self_repo_refusal(capped, "child-1")
    assert _capped_self_repo_refusal(types.SimpleNamespace(task_metadata=dict(_wake_task("full")["metadata"])), "c") == ""
    assert _capped_self_repo_refusal(types.SimpleNamespace(task_metadata={}), "c") == ""
