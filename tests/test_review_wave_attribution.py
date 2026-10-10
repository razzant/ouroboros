"""Every reviewer send is billed to a review round (#1544).

A caller may name its wave (skill and plan review). Otherwise a review's
paid-cycle identity (``retry_key``: one per commit, scope or acceptance cycle)
is its wave, and a review without one gets a fresh wave. Either way every slot
of the round carries the same wave, on its usage row and on the returned run.
Only a caller-named wave splits the prompt-cache estimate (owner choice A).
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from ouroboros.review_substrate import ReviewRequest, ReviewSlot, run_review_request
from tests._review_substrate_shared import FakeLLM

_SLOTS = [ReviewSlot(slot_id="slot_a", model="same/model"), ReviewSlot(slot_id="slot_b", model="other/model")]


def _review(tmp_path, request):
    ctx = SimpleNamespace(task_id=request.task_id, event_queue=None, pending_events=[])
    result = run_review_request(request, slots=_SLOTS, drive_root=tmp_path, llm=FakeLLM(), usage_ctx=ctx)
    rows = [event for event in ctx.pending_events if event.get("type") == "llm_usage"]
    assert sorted(row["review_slot_id"] for row in rows) == ["slot_a", "slot_b"]
    return rows, result


def test_every_slot_of_a_paid_cycle_is_billed_to_that_cycle(tmp_path):
    rows, result = _review(tmp_path, ReviewRequest(
        surface="task_acceptance", goal="review", task_id="t-cycle", retry_key="task_acceptance:rev-1"))
    assert {row["review_wave_id"] for row in rows} == {"task_acceptance:rev-1"}
    assert result.request["resolved_wave_id"] == "task_acceptance:rev-1"
    assert result.request["usage_attribution"] == {}  # the caller's attribution stays as given


def test_a_review_without_a_cycle_key_gets_one_fresh_wave_for_all_its_slots(tmp_path):
    first, result = _review(tmp_path, ReviewRequest(surface="deep_self_review", goal="review", task_id="t-a"))
    second, _ = _review(tmp_path, ReviewRequest(surface="deep_self_review", goal="review", task_id="t-b"))
    [wave] = {row["review_wave_id"] for row in first}
    assert wave.startswith("wave-") and result.request["resolved_wave_id"] == wave
    assert {row["review_wave_id"] for row in second} != {wave}  # the next review is its own round


def test_the_callers_wave_wins_over_the_cycle_key(tmp_path):
    rows, _ = _review(tmp_path, ReviewRequest(
        surface="skill_review", goal="review", task_id="t-skill", retry_key="skill:digest",
        usage_attribution={"review_skill": "weather", "review_wave_id": "skill-wave-7"}))
    assert {(row["review_skill"], row["review_wave_id"]) for row in rows} == {("weather", "skill-wave-7")}


def test_a_multi_model_fan_out_shares_one_wave_across_its_rows(monkeypatch):
    """Each fan-out row sends its own request, so a round without a cycle key
    names its wave once for all of them; a cycle key stays the wave."""
    import ouroboros.review_substrate as substrate
    from ouroboros.tools import review
    from ouroboros.tools.review_multi_model import _multi_model_review_async

    seen = []

    def fake_run_review_request(request, slots, **_kwargs):
        seen.append({"wave": request.resolved_wave_id, "attribution": dict(request.usage_attribution)})
        return SimpleNamespace(actors=[{"status": "ok", "raw_text": "[]", "usage": {}, "slot_id": slots[0].slot_id,
                                        "prompt_ref": {}, "response_ref": {}}])

    monkeypatch.setattr(substrate, "run_review_request", fake_run_review_request)
    monkeypatch.setattr(review, "LLMClient", lambda: object())
    models = ["model/a", "model/b", "model/c"]
    asyncio.run(_multi_model_review_async("diff", "review this", models, None))
    assert len(seen) == 3 and len({row["wave"] for row in seen}) == 1 and seen[0]["wave"].startswith("wave-")
    assert not any(row["attribution"] for row in seen)  # a derived round, never a caller-named wave
    seen.clear()
    asyncio.run(_multi_model_review_async("diff", "review this", models, None, retry_key="commit:cycle-2"))
    assert len(seen) == 3 and not any(row["wave"] for row in seen)  # the substrate uses the key


def test_attribution_is_not_reconciliation_or_attempt_identity():
    """Recording the wave on a request changes no identity a replay or a late result is matched by."""
    from ouroboros.review_custody import _attempt_key
    from ouroboros.review_dispatch import review_reconciliation_identity
    from ouroboros.review_substrate import resolve_review_wave

    slot = ReviewSlot(slot_id="one", model="test/model")
    for retry_key in ("", "cycle-one"):
        request = ReviewRequest(surface="task_acceptance", goal="same", retry_key=retry_key)
        before = (_attempt_key(request, slot), review_reconciliation_identity(request, [slot], root_task_id="root"))
        resolve_review_wave(request, {}, "")
        after = (_attempt_key(request, slot), review_reconciliation_identity(request, [slot], root_task_id="root"))
        assert before == after


@pytest.mark.parametrize("route", ["api_chat", "agent_session"])
def test_reviews_that_run_their_executor_directly_name_their_round(tmp_path, monkeypatch, route):
    """Deep self-review builds its own usage scope instead of the substrate's; its real entry
    point still sends under the round's wave."""
    from dataclasses import asdict

    from ouroboros import deep_self_review as deep
    from ouroboros import observability
    from ouroboros import usage_accounting as ua
    from ouroboros.review_execution import AgentSessionReviewExecutor, ReviewAttemptResult
    from ouroboros.review_native_episode import NativeToolRoundReviewExecutor
    from ouroboros.reviewer_slot_config import ConfiguredReviewerSlot

    repo = tmp_path / "repo"
    repo.mkdir()
    row = ConfiguredReviewerSlot(slot_id="deep_review", kind=route, target_id="test/model", effort="low")
    observed = []

    def capture(self):
        observed.append({"scope": asdict(ua.current_usage_scope()), "request": asdict(self.assignment.request)})
        body = "# Independent deep review\nNo findings."
        return ReviewAttemptResult(message={"content": body}, usage={}, raw_text=body)

    monkeypatch.setattr(NativeToolRoundReviewExecutor, "execute", capture)
    monkeypatch.setattr(AgentSessionReviewExecutor, "execute", capture)
    monkeypatch.setattr(observability, "persist_call", lambda *args, **kwargs: {})
    monkeypatch.setattr(deep, "deep_review_route", lambda row: ("", row.target_id))
    monkeypatch.setattr(deep, "_retrieving_task", lambda *args, **kwargs: ("review task", {
        "required_sources": [], "memory": {"inlined": 0, "total": 0, "dispositions": {}},
        "bible_chars": 0, "governance_manifest": {}}))
    monkeypatch.setattr(deep, "_memory_line", lambda memory: "fixture memory")
    monkeypatch.setattr(deep, "_record_execution", lambda *args, **kwargs: None)
    with ua.usage_scope(ua.UsageScope(drive_root=tmp_path, task_id="review-owner", non_task_operation=True)):
        _text, usage = deep.run_deep_self_review(repo, tmp_path, object(), lambda text: None,
                                                 task_id="review-owner", slot=row)
        assert "execution_status" not in usage
    [seen] = observed
    assert seen["scope"]["review_wave_id"].startswith("wave-") and not seen["scope"]["review_slot_id"]
    assert seen["request"]["resolved_wave_id"] == seen["scope"]["review_wave_id"]
    assert not seen["scope"]["cache_wave"]  # the prompt-cache split stays per task (owner choice A)


@pytest.mark.parametrize("attribution, retry_key, cache_wave", [
    ({}, "commit:cycle-1", ""),                                    # a derived round: split per task
    ({"review_wave_id": "plan:fingerprint:2"}, "plan:fingerprint:2", "plan:fingerprint:2"),  # caller-named
])
def test_only_a_caller_named_wave_splits_the_prompt_cache_estimate(tmp_path, attribution, retry_key, cache_wave):
    """The owner chose to record every round without changing the cache estimate behind
    reservations and commit-gate admission: a derived round leaves the split per task, a wave
    the caller names (plan and skill review) splits it, as before (#1544)."""
    from ouroboros._usage_cache_splits import _surface
    from ouroboros.usage_accounting import current_usage_scope

    observed = []

    class _Watching(FakeLLM):
        def chat(self, *args, **kwargs):
            scope = current_usage_scope()
            observed.append((scope.review_wave_id, scope.cache_wave, _surface()))
            return super().chat(*args, **kwargs)

    ctx = SimpleNamespace(task_id="t-cache", event_queue=None, pending_events=[])
    run_review_request(
        ReviewRequest(surface="task_acceptance", goal="review", task_id="t-cache", retry_key=retry_key,
                      usage_attribution=dict(attribution)),
        slots=[ReviewSlot(slot_id="slot_a", model="same/model")], drive_root=tmp_path, llm=_Watching(), usage_ctx=ctx)
    [(wave, cache, surface)] = observed
    assert wave == retry_key and cache == cache_wave
    assert surface == "|".join(part for part in ("task_acceptance_review", cache_wave, "slot_a") if part)
