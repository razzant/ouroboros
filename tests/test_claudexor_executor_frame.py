"""The executor fact a delegated run leaves on the chat frame.

Split verbatim out of ``tests/test_claudexor_owned_daemon.py`` by theme. This module
owns the resolved harness route reaching the canonical frame assembler, the silence
that a native, blocked or undecided run must keep instead, and the survival of the
fact through history replay and the frozen gateway contract.

Everything here is offline: no daemon is spawned, no network is touched.
"""

import pathlib




# ---------------------------------------------------------------------------
# Phase 6, owner directive #1: the executor fact reaches the chat frame.
# «бейдж точно нужен, но не рекламный … что ТУТ бабл \ субагент на codex»
# ---------------------------------------------------------------------------


def _agent_with_metadata(task, task_id="child-1"):
    import types

    from ouroboros.agent import OuroborosAgent

    agent = object.__new__(OuroborosAgent)
    agent._current_task_metadata = {
        "delegation_role": "subagent", "role": "impl", "root_task_id": "r",
        "parent_task_id": "p", "model": "m", "task_group_id": "g",
    }
    agent._current_task_id = task_id
    # Since synthesis the fact is read from the ONE record the dispatch
    # resolution stamped onto the task (`resolve_subagent_dispatch` ->
    # record_fields) — the same principle this file always asserted ("a
    # projection of the decision, never a second derivation"), one level
    # stronger: the projection reads the durable record, not a live object.
    agent._record_executor_facts(task if isinstance(task, dict) else {})
    return agent, types


def test_resolved_harness_route_reaches_the_frame_assembler():
    """The chip's fact comes from the ONE place the executor was decided: the
    dispatch resolution is stamped onto the live metadata that the canonical
    frame assembler already projects — never re-derived per surface."""
    agent, _ = _agent_with_metadata(
        {"effective_executor": "harness", "executor_route": "codex"})
    frame = agent._subagent_progress_meta("running")
    assert frame["executor_route"] == "codex"
    # The frame keeps carrying the execution facts it always did.
    assert frame["subagent_event"] == "running"
    assert frame["delegation_role"] == "subagent"


def test_no_executor_fact_when_the_run_is_native_blocked_or_undecided():
    """Absent fact -> empty/absent, so the renderer draws NO chip: the native
    API path is the ordinary case and must not print 'api' on every bubble."""
    native, _ = _agent_with_metadata(
        {"effective_executor": "native", "executor_route": ""}, "child-2")
    assert native._subagent_progress_meta("running")["executor_route"] == ""
    # A blocked or unresolved dispatch records nothing at all.
    blocked, _ = _agent_with_metadata(
        {"effective_executor": "blocked", "executor_route": "codex"}, "child-3")
    assert "executor_route" not in blocked._current_task_metadata
    undecided, _ = _agent_with_metadata({}, "child-4")
    assert "executor_route" not in undecided._current_task_metadata


def test_the_executor_fact_survives_history_replay_and_the_frozen_contract():
    """End-to-end plumbing: the field is in the progress-meta allowlist (so a
    reloaded bubble keeps its chip) and in BOTH contract mirrors."""
    from ouroboros.gateway.contracts import ChatOutbound
    from ouroboros.gateway.history import _PROGRESS_META_FIELDS

    assert "executor_route" in _PROGRESS_META_FIELDS
    assert "executor_route" in ChatOutbound.__annotations__
    js = (pathlib.Path(__file__).resolve().parents[1] / "web" / "modules" / "api_types.js")
    assert "executor_route" in js.read_text(encoding="utf-8")
