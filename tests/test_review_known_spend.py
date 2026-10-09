"""Reviewer known-spend admission preserves every prior physical attempt."""
from types import SimpleNamespace

from ouroboros.review_substrate import ReviewRequest, ReviewSlot, run_review_request


class UnknownThenRetryLLM:
    """A $1-bound send fails AFTER dispatch against a $1 global cap, then retries.

    ``landed_usd`` is another actor's actual price that settles in between, in its
    own context: it is neither this reviewer's attempt nor one of its sends.
    """

    def __init__(self, drive_root, landed_usd=None):
        self.drive_root, self.landed_usd = drive_root, landed_usd
        self.sends, self.attempts, self.refusals = 0, [], []

    def _send(self, *, fail=False):
        self.sends += 1
        if fail:
            raise RuntimeError("dispatched")
        return {"content": "[]"}, {"prompt_tokens": 4}

    def chat(self, **_kwargs):
        import contextvars
        from ouroboros.usage_accounting import (
            AttemptRequest, BudgetExceeded, capture_attempt_ids, execute_physical_attempt,
        )

        request = AttemptRequest(
            model="same/model", provider="openrouter",
            reservation_usd=1.0, global_limit_usd=1.0,
        )
        with capture_attempt_ids() as self.attempts:
            try:
                execute_physical_attempt(request, lambda: self._send(fail=True))
            except RuntimeError:
                pass
            if self.landed_usd is not None:
                contextvars.Context().run(
                    execute_physical_attempt,
                    AttemptRequest(model="same/model", provider="openrouter", drive_root=self.drive_root,
                                   task_id="sibling", reservation_usd=self.landed_usd, global_limit_usd=1.0),
                    lambda: "sibling answer", extractor=lambda _response: ({}, self.landed_usd, True),
                )
            try:
                message, usage = execute_physical_attempt(
                    request, self._send, extractor=lambda response: (response[1], 0.0, True))
            except BudgetExceeded as exc:
                self.refusals.append(exc)
                raise
            return message, {**usage, "ledger_attempt_ids": list(self.attempts)}


def _unknown_then_retry_rows(tmp_path, llm):
    """The reviewer's usage rows, and the global (known, open-bound) money after it."""
    from ouroboros.usage_accounting import usage_projection

    ctx = SimpleNamespace(task_id="budget-refusal-usage", event_queue=None, pending_events=[])
    run_review_request(
        ReviewRequest(surface="task_acceptance", goal="review", task_id=ctx.task_id),
        slots=[ReviewSlot(slot_id="slot_a", model="same/model")],
        drive_root=tmp_path, llm=llm, usage_ctx=ctx,
    )
    money = usage_projection(tmp_path, global_limit_usd=1.0)
    return ([event["ledger_attempt_ids"] for event in ctx.pending_events if event.get("type") == "llm_usage"],
            (money["settled_usd"], money["unresolved_upper_bound_usd"]))


def test_terminal_budget_refusal_keeps_prior_dispatched_retry_usage(tmp_path):
    # Another actual price settles AT the cap: known spend reached it, so the
    # retry is refused before any send, and the prior unknown send keeps its row.
    llm = UnknownThenRetryLLM(tmp_path, landed_usd=1.0)
    rows, (known, unknown_bound) = _unknown_then_retry_rows(tmp_path, llm)

    assert (known, unknown_bound) == (1.0, 1.0)  # the landed price is known; the prior send stays unknown
    assert llm.sends == 1
    assert [exc.limit_scope for exc in llm.refusals] == ["global"]
    assert rows == [llm.attempts[:1]] and len(llm.attempts) == 1


def test_unresolved_prior_send_bound_is_not_a_budget_refusal(tmp_path):
    # The prior send's open $1 bound equals the $1 cap, but nothing KNOWN was
    # spent: the retry is sent, and each physical attempt keeps its own row.
    llm = UnknownThenRetryLLM(tmp_path)
    rows, (known, unknown_bound) = _unknown_then_retry_rows(tmp_path, llm)

    assert (known, unknown_bound) == (0.0, 1.0)  # unknown is never a zero, nor counted as spending
    assert llm.sends == 2 and not llm.refusals
    assert rows == [[attempt] for attempt in llm.attempts] and len(llm.attempts) == 2
