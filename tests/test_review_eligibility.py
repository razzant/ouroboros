"""Observed effects, cognitive recovery and the persisted review decision.

Reviewable effects are observed production, never requests, declarations or queue
membership. Cognitive memory keeps its separate meaning, a direct cognitive turn gets
no owner-source selector, and the review decision stays observable in the outcome.
"""
from __future__ import annotations

import pathlib
from types import SimpleNamespace

import pytest

from ouroboros.outcomes import (
    derive_loop_outcome,
    turn_has_reviewable_effects,
    _unresolved_tool_errors,
)


def _call(tool, *, is_error=False, status="ok", root=None):
    args = {}
    if root is not None:
        args["root"] = root
    return {"tool": tool, "is_error": is_error, "status": status, "args": args}


# ----- turn_has_reviewable_effects -----

def test_commit_is_a_reviewable_effect():
    assert turn_has_reviewable_effects({"tool_calls": [_call("commit_reviewed")]})


def test_blocked_or_errored_commit_is_not_an_effect():
    # A commit_reviewed that did NOT commit (REVIEW_BLOCKED -> status "blocked",
    # GIT_ERROR -> status "error") is excluded by the _OK_TOOL_STATUSES gate, so
    # only a successful commit (status "ok") counts.
    assert not turn_has_reviewable_effects({"tool_calls": [_call("commit_reviewed", status="blocked")]})
    assert not turn_has_reviewable_effects({"tool_calls": [_call("commit_reviewed", is_error=True, status="error")]})


def test_review_blocked_commit_end_to_end_not_an_effect():
    # End-to-end: the real REVIEW_BLOCKED / GIT_ERROR commit_reviewed text maps to a
    # non-OK status via _extract_result_metadata, so the effect gate excludes it and
    # a blocked/failed commit never triggers a false required review.
    from ouroboros.loop_tool_execution import _extract_result_metadata, _is_tool_execution_failure

    for txt in (
        "⚠️ REVIEW_BLOCKED (attempt 1): Critical issues found by reviewers.\nCommit has NOT been created.",
        "⚠️ GIT_ERROR (commit): hook rejected the commit.",
    ):
        is_err = _is_tool_execution_failure(True, txt)
        meta = _extract_result_metadata("commit_reviewed", txt, is_err)
        assert meta.get("status") != "ok"
        call = {"tool": "commit_reviewed", "is_error": is_err, **meta}
        assert not turn_has_reviewable_effects({"tool_calls": [call]})


def test_user_files_write_is_a_reviewable_effect():
    assert turn_has_reviewable_effects({"tool_calls": [_call("write_file", root="user_files")]})


def test_artifact_and_repo_writes_are_effects():
    assert turn_has_reviewable_effects({"tool_calls": [_call("write_file", root="artifact_store")]})
    assert turn_has_reviewable_effects({"tool_calls": [_call("edit_text", root="system_repo")]})


def test_scratch_write_is_not_an_effect():
    assert not turn_has_reviewable_effects({"tool_calls": [_call("write_file", root="task_drive")]})


def test_skill_payload_and_runtime_data_writes_are_effects():
    # Exclusion model: any non-scratch successful write is real work. runtime_data
    # in light mode is a skill-payload write (documented skill authoring).
    assert turn_has_reviewable_effects({"tool_calls": [_call("write_file", root="skill_payload")]})
    assert turn_has_reviewable_effects({"tool_calls": [_call("write_file", root="runtime_data")]})


def test_cognitive_updates_are_not_effects():
    assert not turn_has_reviewable_effects({"tool_calls": [
        _call("update_identity"), _call("update_scratchpad"), _call("knowledge_write"),
    ]})


def test_direct_cognitive_turn_does_not_receive_acceptance_selector(monkeypatch, tmp_path):
    """A direct identity update must continue as natural conversation.

    Owner-source observations are internal protocol text. Injecting one after a
    cognitive-only tool call makes Main answer the selector instead of the
    owner's introduction; no delivery control asks for a selection on this turn.
    """
    import queue

    from ouroboros import loop
    from ouroboros.loop_messages import prepare_acceptance_observation

    monkeypatch.setattr(loop, "get_task_review_mode", lambda: "required")
    ctx = SimpleNamespace(
        task_id="direct-cognitive", task_attempt=1, drive_root=tmp_path,
        task_metadata={}, task_contract={}, is_direct_chat=True,
        _loop_mailbox_seen_ids=set(), _owner_directives=[],
    )
    messages = [{"role": "user", "content": "Please remember my name."}]
    trace = {"tool_calls": [_call("update_identity")], "reasoning_notes": []}

    prepare_acceptance_observation(ctx, trace, queue.Queue(), messages)

    assert not any(row.get("acceptance_observation") for row in messages)
    assert "ACCEPTANCE_SUBJECT_OBSERVATION" not in str(messages)


def test_errored_write_is_not_an_effect():
    assert not turn_has_reviewable_effects({"tool_calls": [
        _call("write_file", is_error=True, status="light_mode_blocked", root="user_files"),
    ]})


def test_empty_trace_has_no_effects():
    assert not turn_has_reviewable_effects({"tool_calls": []})
    assert not turn_has_reviewable_effects({})


@pytest.mark.parametrize("tool", ["run_command", "run_script"])
@pytest.mark.parametrize("registered", [False, True])
def test_process_outputs_require_observed_production(tool, registered):
    # Declared paths are intentions; successful registration is the produced fact.
    assert turn_has_reviewable_effects({"tool_calls": [
        {"tool": tool, "is_error": False, "status": "ok",
         "args": {"outputs": ["report.html"]}, "artifact_registered": registered},
    ]}) is registered


def test_registered_artifact_flag_is_an_effect():
    # e.g. stop_service / user_files write that registered a canonical artifact.
    # Detection uses the structured flag captured from the full result, so it
    # holds even when the ARTIFACT_OUTPUTS marker is past the 700-char trace preview.
    assert turn_has_reviewable_effects({"tool_calls": [
        {"tool": "stop_service", "is_error": False, "status": "ok",
         "args": {}, "artifact_registered": True, "result": "x" * 2000},
    ]})


def test_process_without_outputs_is_not_an_effect():
    assert not turn_has_reviewable_effects({"tool_calls": [
        {"tool": "run_command", "is_error": False, "status": "ok",
         "args": {"cmd": ["ls"]}},
    ]})


def test_cwd_coding_effect_class_is_empty_since_d10():
    # D10 retired the one cwd-based coding tool; the effect CLASS stays wired
    # (an empty set), so a successor lands as one entry, not a new mechanism.
    from ouroboros.outcomes import _EFFECT_CODING_TOOLS

    assert _EFFECT_CODING_TOOLS == frozenset()
    assert not turn_has_reviewable_effects({"tool_calls": [
        {"tool": "claude_code_edit", "is_error": False, "status": "ok", "args": {"cwd": "Desktop"}},
    ]})


def test_start_service_declared_outputs_are_not_produced_artifacts():
    assert not turn_has_reviewable_effects({"tool_calls": [
        {"tool": "start_service", "is_error": False, "status": "ok",
         "args": {"cmd": ["serve"], "outputs": ["report.html"]}},
    ]})


# ----- cognitive / root redirect recovery -----

def test_cognitive_block_is_advisory_not_a_failure():
    # A self-initiated cognitive write via the wrong tool is an advisory redirect:
    # it must never fail the task, even with no follow-up correction.
    trace = {"tool_calls": [
        {"tool": "write_file", "is_error": True, "status": "cognitive_tool_required",
         "args": {"root": "runtime_data", "path": "memory/identity.md"}},
    ]}
    assert _unresolved_tool_errors(trace) == []


def test_cognitive_block_advisory_even_with_unrelated_followup():
    trace = {"tool_calls": [
        {"tool": "write_file", "is_error": True, "status": "cognitive_tool_required",
         "args": {"root": "runtime_data", "path": "memory/knowledge/topicA.md"}},
        {"tool": "update_scratchpad", "is_error": False, "status": "ok", "args": {}},
    ]}
    assert _unresolved_tool_errors(trace) == []


def test_root_required_recovers_via_user_files_write():
    trace = {"tool_calls": [
        {"tool": "write_file", "is_error": True, "status": "root_required_user_files",
         "args": {"path": "/home/x/Desktop/a.html"}},
        {"tool": "write_file", "is_error": False, "status": "ok",
         "args": {"root": "user_files", "path": "Desktop/a.html"}},
    ]}
    assert _unresolved_tool_errors(trace) == []


def test_root_required_not_recovered_by_non_user_files_write():
    # A blocked home write must NOT be masked by a write to a different root.
    trace = {"tool_calls": [
        {"tool": "write_file", "is_error": True, "status": "root_required_user_files",
         "args": {"path": "/home/x/Desktop/a.html"}},
        {"tool": "write_file", "is_error": False, "status": "ok",
         "args": {"root": "active_workspace", "path": "notes.txt"}},
    ]}
    unresolved = _unresolved_tool_errors(trace)
    assert len(unresolved) == 1
    assert unresolved[0]["status"] == "root_required_user_files"


def test_root_required_not_recovered_by_different_user_files_filename():
    # Same root, different file is NOT recovery of the originally blocked deliverable.
    trace = {"tool_calls": [
        {"tool": "write_file", "is_error": True, "status": "root_required_user_files",
         "args": {"path": "/home/x/Desktop/a.html"}},
        {"tool": "write_file", "is_error": False, "status": "ok",
         "args": {"root": "user_files", "path": "Desktop/b.html"}},
    ]}
    unresolved = _unresolved_tool_errors(trace)
    assert len(unresolved) == 1
    assert unresolved[0]["status"] == "root_required_user_files"


def test_root_required_recovers_via_user_files_batch_files():
    # Recovery via a batched files[] user_files write counts (path+files[] both sides).
    trace = {"tool_calls": [
        {"tool": "write_file", "is_error": True, "status": "root_required_user_files",
         "args": {"path": "/home/x/Desktop/a.html"}},
        {"tool": "write_file", "is_error": False, "status": "ok",
         "args": {"root": "user_files", "files": [{"path": "Desktop/a.html", "content": "x"}]}},
    ]}
    assert _unresolved_tool_errors(trace) == []


def test_root_required_active_workspace_recovers_via_active_workspace_write():
    # Parity with the user_files contract (cumulative review r2): the
    # ACTIVE_WORKSPACE redirect is recovered only by a later write of the SAME
    # file via root=active_workspace.
    trace = {"tool_calls": [
        {"tool": "write_file", "is_error": True, "status": "root_required_active_workspace",
         "args": {"path": "/app/src/main.py"}},
        {"tool": "write_file", "is_error": False, "status": "ok",
         "args": {"root": "active_workspace", "path": "src/main.py"}},
    ]}
    assert _unresolved_tool_errors(trace) == []


def test_root_required_active_workspace_recovers_via_default_root_write():
    # active_workspace is the DEFAULT root for write_file/edit_text (scope r2
    # finding): a retry that simply omits root writes to the demanded place and
    # must earn the recovery credit — no false execution-axis degradation.
    # user_files stays explicit-only (it is never a default root).
    trace = {"tool_calls": [
        {"tool": "write_file", "is_error": True, "status": "root_required_active_workspace",
         "args": {"path": "/app/src/main.py"}},
        {"tool": "write_file", "is_error": False, "status": "ok",
         "args": {"path": "src/main.py"}},
    ]}
    assert _unresolved_tool_errors(trace) == []
    trace_uf = {"tool_calls": [
        {"tool": "write_file", "is_error": True, "status": "root_required_user_files",
         "args": {"path": "/home/x/Desktop/a.html"}},
        {"tool": "write_file", "is_error": False, "status": "ok",
         "args": {"path": "Desktop/a.html"}},  # rootless != user_files
    ]}
    assert len(_unresolved_tool_errors(trace_uf)) == 1


def test_root_required_active_workspace_not_recovered_by_user_files_write():
    trace = {"tool_calls": [
        {"tool": "write_file", "is_error": True, "status": "root_required_active_workspace",
         "args": {"path": "/app/src/main.py"}},
        {"tool": "write_file", "is_error": False, "status": "ok",
         "args": {"root": "user_files", "path": "main.py"}},
    ]}
    unresolved = _unresolved_tool_errors(trace)
    assert len(unresolved) == 1
    assert unresolved[0]["status"] == "root_required_active_workspace"


def test_root_required_terminal_not_recovered_by_artifact_output():
    # Terminal branch: a run_command with ARTIFACT_OUTPUTS must NOT clear a
    # ROOT_REQUIRED block via the generic recovery path — only a real user_files
    # write of the blocked file recovers it.
    trace = {"tool_calls": [
        {"tool": "write_file", "is_error": True, "status": "root_required_user_files",
         "args": {"path": "/home/x/Desktop/a.html"}},
        {"tool": "run_command", "is_error": False, "status": "ok",
         "args": {"cmd": ["build"], "outputs": ["a.html"]}, "artifact_registered": True},
    ]}
    unresolved = _unresolved_tool_errors(trace)
    assert len(unresolved) == 1
    assert unresolved[0]["status"] == "root_required_user_files"


# ----- persistence into loop_outcome -----

def test_loop_outcome_surfaces_review_decision():
    outcome = derive_loop_outcome(
        "final answer",
        {},
        {"tool_calls": [], "review_decision": {"eligibility": "eligible", "trigger": "required_effect"}},
    )
    assert outcome["schema_version"] == 3
    assert outcome["review_eligibility"] == "eligible"
    assert outcome["review_trigger"] == "required_effect"


def test_loop_outcome_defaults_when_no_decision():
    outcome = derive_loop_outcome("final answer", {}, {"tool_calls": []})
    assert outcome["review_eligibility"] == "not_eligible"
    assert outcome["review_trigger"] == "not_evaluated"


# ----- label-only invariant (v6.35.0): host-forced `required` review must NOT
# inject a draft assistant turn or a review payload that replaces the deliverable —
# the old re-loop that rewrote the deliverable into a meta-essay is removed. A
# result review's findings reach the mind as one host note before delivery
# (tools/review_result.py); the mind's own answer stays the deliverable. -----

def test_acceptance_review_is_label_only_no_transcript_injection():
    # v7 L-B split: the negatives sweep the whole loop family so no leaf can
    # revive the injection; the verdict recording lives with the result review,
    # which binds a settled review of the delivered result into review_runs.
    loop_dir = pathlib.Path(__file__).resolve().parents[1] / "ouroboros"
    src = "".join(
        path.read_text(encoding="utf-8")
        for path in [loop_dir / "loop.py", *sorted(loop_dir.glob("loop_*.py"))]
    )
    # The old injection strings are gone (no draft re-commit, no review payload).
    assert "Do NOT replace your user-facing answer with a status report" not in src
    assert "[TASK ACCEPTANCE REVIEW]" not in src
    # The verdict is still recorded on the objective axis (immune signal kept).
    assert "review_runs" in (loop_dir / "tools" / "review_result.py").read_text(encoding="utf-8")
