"""A retained answer over an evidence read the host cannot complete.

The answer-invalidating evidence fingerprint is the one fallible read every
retention, control, selection, publication and forced-exit path shares. These
scenarios break it on purpose and pin that no path fails after an answer was
retained: the read is typed UNKNOWN, the answer is kept and published as
unavailable (never current, never approved), the last KNOWN state never moves,
and a forced exit preserves the text as unverified.
"""

from __future__ import annotations

import json
from types import SimpleNamespace as NS

import pytest


def _raise_fingerprint(monkeypatch):
    import ouroboros.loop_delivery as delivery_mod

    def _explode(*_a, **_k):
        raise RuntimeError("the evidence fingerprint is what is broken")

    monkeypatch.setattr(delivery_mod, "delivery_evidence_fingerprint", _explode)


def _delivery_ctx(tmp_path, trace):
    """A real registry and round context for the delivery leaf (as test_delivery_candidate builds them)."""
    import ouroboros.loop as loop
    from ouroboros.tools.registry import ToolRegistry

    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_id = "parent1"
    ctx = loop._RoundLimitContext(
        [{"role": "user", "content": "task"}], NS(), "test-model", "medium", 0, tmp_path / "logs",
        "parent1", 1, None, {}, "", False, 10, drive_root=tmp_path, status_drive_root=tmp_path, root_task_id="parent1",
    )
    loop._finalize_limit_ctx(ctx, registry, trace)
    return registry, ctx


def test_the_first_answer_is_retained_over_unknown_evidence_when_the_fingerprint_is_broken(tmp_path, monkeypatch):
    """Retention never waits on the fallible evidence read: the FIRST complete answer
    is kept and published as unavailable (never as approved); an unchanged repeat under
    the same broken read stays that one candidate; a repaired read moves the answer
    onto known evidence through the ordinary replacement."""
    import ouroboros.loop as loop
    import ouroboros.loop_delivery as delivery_mod

    _raise_fingerprint(monkeypatch)
    trace = {"tool_calls": [], "reasoning_notes": []}
    registry, ctx = _delivery_ctx(tmp_path, trace)
    candidate = loop._replace_delivery_candidate(registry, ctx, trace, "first complete answer", control="candidate")
    assert registry._ctx._delivery_candidate is candidate and candidate.revision == 1
    assert candidate.evidence_fingerprint == "" and candidate.evidence_revision == 0
    assert candidate.acceptance_binding["authoritative"] is False
    assert candidate.acceptance_binding["acceptance_status"] == "unaccepted"
    published = trace["delivery_candidate"]
    assert published["content_sha256"] == candidate.content_sha256 and published["revision"] == 1
    assert published["evidence_current"] is False and published["subject_sha256"] == ""
    assert published["evidence_status"] == "unavailable"
    again = loop._replace_delivery_candidate(registry, ctx, trace, "first complete answer", control="candidate")
    assert again is candidate and candidate.revision == 1 and trace["delivery_candidate"]["revision"] == 1
    replaced = loop._replace_delivery_candidate(registry, ctx, trace, "a different complete answer", control="candidate")
    assert replaced is not candidate and replaced.revision == 2 and replaced.evidence_fingerprint == ""
    assert trace["delivery_candidate"]["evidence_status"] == "unavailable"
    monkeypatch.setattr(delivery_mod, "delivery_evidence_fingerprint", lambda *_a, **_k: "repaired-fingerprint")
    repaired = loop._replace_delivery_candidate(registry, ctx, trace, "a different complete answer", control="candidate")
    assert repaired is not replaced and repaired.revision == 3
    assert repaired.evidence_fingerprint == "repaired-fingerprint" and repaired.evidence_revision == 1
    assert trace["delivery_candidate"]["evidence_current"] is True and trace["delivery_candidate"]["subject_sha256"]
    assert "evidence_status" not in trace["delivery_candidate"]


def test_a_failed_subject_publication_keeps_the_retained_answer_unapproved(tmp_path, monkeypatch):
    """The subject hash is computed at publication: its failure must publish the
    retained answer as unavailable, not crash after the answer was already retained
    and not claim currency."""
    import ouroboros.loop as loop
    import ouroboros.loop_delivery as delivery_mod

    monkeypatch.setattr(delivery_mod, "delivery_subject_hash",
                        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("subject hash is fallible")))
    trace = {"tool_calls": [], "reasoning_notes": []}
    registry, ctx = _delivery_ctx(tmp_path, trace)
    candidate = loop._replace_delivery_candidate(registry, ctx, trace, "first complete answer", control="candidate")
    assert registry._ctx._delivery_candidate is candidate and candidate.evidence_fingerprint  # the fingerprint itself worked
    published = trace["delivery_candidate"]
    assert published["content_sha256"] == candidate.content_sha256
    assert published["subject_sha256"] == "" and published["evidence_status"] == "unavailable"
    assert published["evidence_current"] is False and published["acceptance_binding"]["authoritative"] is False
    monkeypatch.undo()
    loop._publish_delivery_candidate(registry, candidate, trace)
    assert trace["delivery_candidate"]["subject_sha256"] and trace["delivery_candidate"]["evidence_current"] is True
    assert "evidence_status" not in trace["delivery_candidate"]


# ── the ONE typed evidence read: no path fails after an answer was retained ──


def test_the_evidence_state_types_a_broken_read_as_unknown_without_moving_known_state(tmp_path, monkeypatch):
    """`_delivery_evidence_state` owns the fallible read: a broken fingerprint is
    UNKNOWN (revision unchanged, last KNOWN fingerprint kept, nothing superseded),
    never an exception — and a repaired read continues from the KNOWN state, so the
    outage itself is never counted as a change."""
    import ouroboros.loop as loop
    import ouroboros.loop_delivery as delivery_mod

    trace = {"tool_calls": [], "reasoning_notes": []}
    registry, ctx = _delivery_ctx(tmp_path, trace)
    monkeypatch.setattr(delivery_mod, "delivery_evidence_fingerprint", lambda *_a, **_k: "known-1")
    assert loop._delivery_evidence_state(registry, ctx, trace) == (1, "known-1")
    _raise_fingerprint(monkeypatch)
    assert loop._delivery_evidence_state(registry, ctx, trace) == (1, "")
    assert registry._ctx._delivery_evidence_fingerprint == "known-1"
    assert registry._ctx._delivery_evidence_revision == 1
    monkeypatch.setattr(delivery_mod, "delivery_evidence_fingerprint", lambda *_a, **_k: "known-1")
    assert loop._delivery_evidence_state(registry, ctx, trace) == (1, "known-1")   # the outage was not a change
    monkeypatch.setattr(delivery_mod, "delivery_evidence_fingerprint", lambda *_a, **_k: "known-2")
    assert loop._delivery_evidence_state(registry, ctx, trace) == (2, "known-2")


@pytest.mark.parametrize("path", ["arm", "post_tool", "selection"])
def test_every_retained_answer_path_survives_a_broken_read_and_claims_nothing(tmp_path, monkeypatch, path):
    """Retention was guarded, but arming a control round, the post-tool budget
    context and an answer selection still reached the broken read
    unguarded AFTER the answer was retained. Each now types it unknown: the answer
    stays, is published as unavailable (never current, never bound), and explicit
    selection preserves the same bytes over unknown evidence without borrowing
    the old candidate's verified subject."""
    import ouroboros.loop as loop
    import ouroboros.loop_delivery as delivery_mod
    from ouroboros.tools.control_runtime import _finish_task

    trace = {"tool_calls": [], "reasoning_notes": []}
    registry, ctx = _delivery_ctx(tmp_path, trace)
    monkeypatch.setattr(delivery_mod, "delivery_evidence_fingerprint", lambda *_a, **_k: "known-1")
    candidate = loop._replace_delivery_candidate(registry, ctx, trace, "complete answer", control="candidate")
    assert candidate.evidence_fingerprint == "known-1" and trace["delivery_candidate"]["evidence_current"] is True
    _raise_fingerprint(monkeypatch)
    transcript = lambda: "\n".join(str(row.get("content") or "") for row in ctx.messages)  # noqa: E731
    if path == "arm":
        loop._arm_delivery_control(registry, ctx, trace)
        assert candidate.content_sha256 in transcript() and "select complete answer bytes or this answer_sha256" in transcript()
        assert candidate.finalization_control == "awaiting_control"
    elif path == "post_tool":
        loop._prepare_post_tool_budget_context(registry, ctx, trace, "test-model", False, "medium")
        assert candidate.finalization_control == "effect_revision_required"
        assert "select complete answer bytes or this answer_sha256" in transcript()
    else:
        registry._ctx._delivery_control_required = True
        candidate.finalization_control = "awaiting_control"
        registry._ctx._completion_observation = delivery_mod.completion_observation(registry._ctx, trace)
        payload = json.loads(_finish_task(registry._ctx, action="finish", answer_sha256=candidate.content_sha256))
        assert payload["status"] == "completion_requested"
        assert delivery_mod.consume_completion_request(registry, ctx, trace)
        assert registry._ctx._delivery_candidate.full_text == candidate.full_text
        assert registry._ctx._completion_selected["observation"]["evidence_fingerprint"] == ""
    published = trace["delivery_candidate"]
    assert published["evidence_current"] is False and published["subject_sha256"] == ""
    assert published["evidence_status"] == "unavailable"
    assert published["acceptance_binding"]["authoritative"] is False
    assert candidate.evidence_fingerprint == "known-1" and registry._ctx._delivery_evidence_revision == 1
    restated = loop._replace_delivery_candidate(registry, ctx, trace, "complete answer", control="replace")
    assert restated is not candidate and restated.revision == 2 and restated.evidence_fingerprint == ""
    assert restated.acceptance_binding["authoritative"] is False
    assert loop._replace_delivery_candidate(registry, ctx, trace, "complete answer", control="candidate") is restated
    loop._arm_delivery_control(registry, ctx, trace)
    assert restated.content_sha256 in transcript() and "action=stop with a rationale" in transcript()
    loop._prepare_post_tool_budget_context(registry, ctx, trace, "test-model", False, "medium")
    assert restated.finalization_control == "awaiting_control"   # unchanged unknown evidence arms nothing new
    assert trace["delivery_candidate"]["evidence_status"] == "unavailable"


def test_a_forced_exit_over_an_unreadable_fingerprint_preserves_the_answer_as_unverified(tmp_path, monkeypatch):
    """A KNOWN candidate under a read the host cannot complete is never current
    (its approval cannot be verified), so the forced rail preserves the text
    unaccepted — and its notice says the evidence could not be read, not that
    newer evidence arrived."""
    import ouroboros.loop as loop
    import ouroboros.loop_delivery as delivery_mod

    trace = {"tool_calls": [], "reasoning_notes": []}
    registry, ctx = _delivery_ctx(tmp_path, trace)
    monkeypatch.setattr(delivery_mod, "delivery_evidence_fingerprint", lambda *_a, **_k: "known-1")
    candidate = loop._replace_delivery_candidate(registry, ctx, trace, "complete answer", control="candidate")
    _raise_fingerprint(monkeypatch)
    assert loop._current_delivery_candidate(ctx, trace) is None
    preserved = loop._publish_stale_forced_candidate(ctx, trace, candidate, "budget_exhausted", "")
    notice = str(ctx.accumulated_usage["terminal_host_notice"])
    assert "STALE-EVIDENCE NOTICE" in notice and "could no longer read" in notice
    assert "newer task evidence" not in notice
    assert preserved.acceptance_binding["authoritative"] is False
    assert preserved.acceptance_binding["stale_evidence"] is True
    assert trace["delivery_candidate"]["evidence_current"] is False


@pytest.mark.parametrize("criteria", ["original requirements", "changed requirements"])
def test_forced_selection_keeps_the_answer_over_an_unreadable_fingerprint_without_currency(tmp_path, monkeypatch, criteria):
    """An acknowledged subject on forced selection keeps the same answer bytes when
    the fingerprint is unreadable, whether or not the criteria changed: the answer is
    re-bound to the UNKNOWN evidence, never published as current, never approved."""
    from ouroboros import loop, loop_delivery as delivery
    from ouroboros.loop_forced_finalization import _resolve_forced_delivery_control
    import ouroboros.loop_messages as loop_messages

    trace = {"tool_calls": [], "reasoning_notes": []}
    registry, ctx = _delivery_ctx(tmp_path, trace)
    registry._ctx._delivery_effective_criteria = "original requirements"
    monkeypatch.setattr(delivery, "delivery_evidence_fingerprint", lambda *_a, **_k: "known")
    candidate = loop._replace_delivery_candidate(registry, ctx, trace, "complete answer", control="candidate")
    registry._ctx._delivery_control_required = True
    monkeypatch.setattr(loop_messages, "acknowledge_acceptance_observation", lambda *_a, **_k: True)
    _raise_fingerprint(monkeypatch)
    control = json.dumps({"action": "finish", "answer_sha256": candidate.content_sha256, "acceptance_subject": {
        "owner_source_sha256": "observed-owner", "effective_criteria": criteria}})
    text, reason, retained, replaced = _resolve_forced_delivery_control(registry._ctx, control, ctx=ctx, llm_trace=trace)
    assert text == "complete answer" and retained and not replaced and not reason
    assert candidate.evidence_fingerprint == ""
    assert candidate.acceptance_binding["authoritative"] is False
    assert trace["delivery_candidate"]["evidence_current"] is False
