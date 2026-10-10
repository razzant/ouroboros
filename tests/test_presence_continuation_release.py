"""An early-release decision belongs to its selection and its review wait."""
from __future__ import annotations

import json

from ouroboros import presence_continuation as pc
from ouroboros.acceptance_settlement import awaited_panel_has_settled
from ouroboros.task_results import load_task_result
from tests.test_presence_continuation_bootstrap import (
    ANSWER, event, finish, install_bootstrap_harness, nominate, wait_for,
)

CORRECTION = "The corrected status report: the backup check still needs attention."


def test_ready_first_review_cannot_release_its_old_answer_at_the_second_park(tmp_path, monkeypatch):
    h = install_bootstrap_harness(monkeypatch, tmp_path / "data", enforcement="advisory")
    h.verdict = "FAIL"
    keep = pc.keep_author_for_criticism
    raced = []

    def settle_before_wait(review):
        result = keep(review)
        assert result
        if not raced:
            # Real reviewer completion lands after the release choice but before
            # wait_for_acceptance_feedback's readiness check. No fake verdict or
            # readiness result: collect the actual local review operation.
            raced.append(dict(review.tools._ctx._presence_release))
            h.release.set()
            wait_for(lambda: awaited_panel_has_settled(review.tools._ctx, review.llm_trace),
                     what="first reviewer wins the release/park race")
            assert h.settled.wait(15)
            h.release.clear()
            h.settled.clear()
            h.verdict = "PASS"
        return result

    monkeypatch.setattr(pc, "keep_author_for_criticism", settle_before_wait)

    def select(_messages, **kwargs):
        return finish("select", answer_sha256=h.agents[0].tools._ctx._delivery_candidate.content_sha256,
                      **kwargs)

    h.script = [nominate(), lambda messages: select(messages, pending_review="finish"),
                nominate(CORRECTION), lambda messages: select(messages, pending_review="wait"), select]
    execution = h.start(event())
    try:
        initial = execution.initial.result(timeout=40)
        assert len(raced) == 1 and raced[0]["text"] == ANSWER
        assert len(h.reviews) == 2 and h.calls == 4
        row = load_task_result(h.data, initial.task_id)
        (tmp_path / "release-race-facts.json").write_text(json.dumps({
            "initial": initial.__dict__, "raced_choice": raced[0],
            "continuation": row.get("presence_continuation"), "model_calls_at_park": h.calls,
            "reviews_at_park": len(h.reviews)}, indent=2))
        assert (initial.status, initial.outcome, initial.text, initial.output_ref) == (
            "continuing", "deferred", "", "")
        assert row["presence_continuation"]["outputs"] == []
        assert not getattr(h.agents[0].tools._ctx, "_presence_released", [])
        h.release.set()
        final = execution.result.result(timeout=30)
        assert (final.outcome, final.text) == ("message", CORRECTION)
        assert h.calls == 5 and len(h.reviews) == 2
        assert h.agents[0].tools._ctx.inline_max_rounds == 10
        stored = load_task_result(h.data, initial.task_id)
        assert stored["presence_continuation"]["initial"] == row["presence_continuation"]["initial"]
        assert stored["presence_continuation"]["outputs"] == []
    finally:
        h.release.set()
        execution.result.result(timeout=30)
        assert h.settled.wait(15)
        assert not h.executions.live()
