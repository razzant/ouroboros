"""Late results, paid dispatch and transport windows of the review substrate.

The post-cutoff upstream custody theme (the adaptive-timeout/custody train),
split by theme out of ``tests/test_review_substrate_v2.py`` by the v7next D06
lane: the write-ahead paid stamp, logical-deadline and transport-window
separation, late-result replay without a second paid dispatch, and the
pending-invocation restore paths. Bodies are tip bytes.
"""

import json
import time
from types import SimpleNamespace

from ouroboros.review_substrate import ReviewRequest, ReviewSlot, run_review_request

from tests._review_substrate_shared import FakeLLM

def test_actor_projection_carries_bounded_disclosed_finding_rows():
    from ouroboros.review_substrate import (
        MAX_PROJECTED_ACTOR_FINDINGS, compact_review_projection,
    )

    secret = "sk-or-" + ("FindingSecret456" * 4)
    long_recommendation = "Re-run the verifier with the fixed seed. " * 120
    findings = [
        {
            "severity": "critical",
            "item": f"finding {index}",
            "evidence": f"evidence {index}",
            "recommendation": f"fix {index}",
        }
        for index in range(MAX_PROJECTED_ACTOR_FINDINGS + 2)
    ]
    findings[0]["evidence"] = "credential=" + secret
    findings[1]["recommendation"] = long_recommendation
    run = {
        "request": {"surface": "task_acceptance", "policy": {"min_successful_slots": 1}},
        "aggregate_signal": "FAIL",
        "actors": [
            {
                "slot_id": "with-findings",
                "model": "model-a",
                "status": "ok",
                "signal": "FAIL",
                "parsed": {"verdict": "FAIL", "summary": "s", "findings": findings},
                "quorum_contribution": True,
            },
            {
                "slot_id": "clean",
                "model": "model-b",
                "status": "ok",
                "signal": "PASS",
                "parsed": {"verdict": "PASS", "summary": "ok", "findings": []},
                "quorum_contribution": True,
            },
            {
                "slot_id": "transport-hole",
                "model": "model-c",
                "status": "error",
                "error": "timed out",
                "parsed": None,
            },
            {
                "slot_id": "odd-shape",
                "model": "model-d",
                "status": "ok",
                "signal": "FAIL",
                "parsed": {
                    "verdict": "FAIL",
                    "summary": "s",
                    "findings": [{
                        "weird_key": "the only copy of this evidence",
                        "password": "hunter2-odd-shape",
                    }],
                },
            },
            {
                # A non-string value under a KNOWN key keeps structural
                # key-based masking: str() first would flatten the nested
                # secret past the key-name redactor.
                "slot_id": "nested-evidence",
                "model": "model-f",
                "status": "ok",
                "signal": "FAIL",
                "parsed": {
                    "verdict": "FAIL",
                    "summary": "s",
                    "findings": [{
                        "severity": "high",
                        "item": "nested shape",
                        "evidence": {"password": "hunter2-nested-shape"},
                    }],
                },
            },
            {
                # The array-ladder reviewer contract shapes findings as
                # {item, verdict, severity, reason}: the substantive `reason`
                # text must survive projection.
                "slot_id": "triad-shape",
                "model": "model-e",
                "status": "ok",
                "signal": "FAIL",
                "parsed": [{
                    "item": "missing rollback test",
                    "verdict": "FAIL",
                    "severity": "high",
                    "reason": "the new path has no failure-injection coverage",
                }],
            },
        ],
    }

    panel = compact_review_projection([run])["panels"][0]
    actors = {actor["slot_id"]: actor for actor in panel["actors"]}
    rendered = json.dumps(panel, ensure_ascii=False)

    rows = actors["with-findings"]["findings"]
    assert len(rows) == MAX_PROJECTED_ACTOR_FINDINGS
    assert actors["with-findings"]["findings_omitted"] == 2
    assert rows[2] == {
        "severity": "critical", "item": "finding 2",
        "evidence": "evidence 2", "recommendation": "fix 2",
    }
    # The count stays beside the rows: coverage keeps the full total.
    assert actors["with-findings"]["coverage"]["findings"] == len(findings)
    # Redaction covers finding bodies exactly like reasons.
    assert secret not in rendered
    assert "***REDACTED***" in rows[0]["evidence"]
    # A clipped string discloses its own cut instead of clipping silently.
    assert "OMISSION NOTE" in rows[1]["recommendation"]
    assert len(rows[1]["recommendation"]) < len(long_recommendation)

    # A reviewer that reported no findings states that as an empty disclosed
    # list; a reviewer with no parsed response leaves a hole, not a zero.
    assert actors["clean"]["findings"] == []
    assert actors["clean"]["findings_omitted"] == 0
    assert "findings" not in actors["transport-hole"]
    assert "findings_omitted" not in actors["transport-hole"]

    # An unknown finding shape still ships its evidence as a bounded row, and
    # structural key-based secret masking applies BEFORE serialization.
    odd_rows = actors["odd-shape"]["findings"]
    assert odd_rows and "the only copy of this evidence" in odd_rows[0]["item"]
    assert "hunter2-odd-shape" not in rendered
    # #447 G11: key-name redaction leaves a typed fingerprint, not bare deletion.
    assert "***REDACTED[" in odd_rows[0]["item"]

    nested_rows = actors["nested-evidence"]["findings"]
    assert "hunter2-nested-shape" not in rendered
    assert "***REDACTED[" in nested_rows[0]["evidence"]
    assert nested_rows[0]["item"] == "nested shape"

    # A list-shaped parsed response (array reviewer contract) projects its
    # rows too, and the substantive `reason`/`verdict` fields survive.
    triad_rows = actors["triad-shape"]["findings"]
    assert triad_rows == [{
        "severity": "high", "verdict": "FAIL", "item": "missing rollback test",
        "reason": "the new path has no failure-injection coverage",
    }]
    assert actors["triad-shape"]["findings_omitted"] == 0

def test_spent_owner_deadline_does_not_dispatch_a_review_worker(tmp_path):
    calls = []
    paid = []

    class NeverCalledLLM:
        def chat(self, **_kwargs):
            calls.append(1)
            raise AssertionError("review transport dispatched after owner deadline")

    ctx = SimpleNamespace(
        task_id="spent-review", task_attempt=1, task_metadata={},
        event_queue=None, pending_events=[],
        _review_paid_stamp=lambda: paid.append(1),
    )
    result = run_review_request(
        ReviewRequest(
            surface="scope", goal="review", task_id="spent-review",
            deadline_at="2000-01-01T00:00:00Z",
        ),
        slots=[ReviewSlot(slot_id="slot_a", model="same/model", timeout_sec=300)],
        drive_root=tmp_path, llm=NeverCalledLLM(), usage_ctx=ctx,
    )

    actor = result.actors[0]
    assert calls == [] and paid == []
    assert actor["status"] == "not_dispatched"
    assert actor["operation_state"] == "not_dispatched"
    assert actor["operation_id"] == ""
    assert actor["late_result_pending"] is False

def test_review_worker_does_not_retry_after_its_logical_deadline(tmp_path):
    import threading
    from ouroboros.review_custody import _ACTIVE, _ACTIVE_LOCK, _attempt_key

    calls = []
    release = threading.Event()
    first_finished = threading.Event()

    class LateTransportFailure:
        """Holds the paid call open until the test releases it: the caller must
        observe 'in_flight' deterministically (a fixed 0.05 s sleep let a slow
        macOS runner settle the worker before the assertion)."""

        def chat(self, **_kwargs):
            calls.append(1)
            assert release.wait(10), "test never released the gated review call"
            first_finished.set()
            raise TimeoutError("late transport failure")

    ctx = SimpleNamespace(
        task_id="late-failure", task_attempt=1, task_metadata={},
        event_queue=None, pending_events=[],
    )
    request = ReviewRequest(
        surface="multi_model_review", goal="review", task_id="late-failure",
        task_attempt=1,
    )
    slot = ReviewSlot(slot_id="slot_a", model="same/model", timeout_sec=0.01)
    result = run_review_request(
        request, slots=[slot], drive_root=tmp_path,
        llm=LateTransportFailure(), usage_ctx=ctx,
    )
    assert result.actors[0]["operation_state"] == "in_flight"
    release.set()
    assert first_finished.wait(10.0)

    key = _attempt_key(request, slot)
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        with _ACTIVE_LOCK:
            active = key in _ACTIVE
        if not active:
            break
        time.sleep(0.01)
    assert not active
    assert calls == [1]

def test_late_review_result_is_replayed_without_a_second_paid_dispatch(tmp_path):
    import threading
    from types import SimpleNamespace
    from ouroboros.review_custody import _ACTIVE, _ACTIVE_LOCK, _attempt_key

    calls = []
    release = threading.Event()

    class GatedLLM:
        """Holds the paid call open until the test releases it, so the first
        poll window is GUARANTEED to expire while the call is still in flight.
        The previous 0.08s-sleep-vs-0.05s-window race flaked on slow CI hosts:
        an oversleeping poll wait could observe the settled result and return
        'settled' instead of 'in_flight'."""

        def chat(self, **_kwargs):
            calls.append(dict(_kwargs))
            assert release.wait(10), "test never released the gated review call"
            return {"content": '{"verdict":"PASS","findings":[],"summary":"late"}'}, {}

    ctx = SimpleNamespace(
        task_id="late-review",
        task_attempt=1,
        event_queue=None,
        pending_events=[],
    )
    request = ReviewRequest(
        surface="scope",
        goal="review diff",
        task_id="late-review",
        task_attempt=1,
    )
    slot = ReviewSlot(
        slot_id="slot_a", model="same/model", timeout_sec=0.05,
        transport_timeout_sec=10,
    )
    first = run_review_request(request, slots=[slot], drive_root=tmp_path, llm=GatedLLM(), usage_ctx=ctx)
    assert first.actors[0]["operation_state"] == "in_flight"
    assert first.actors[0]["late_result_pending"] is True

    release.set()
    # The settled-attempt cache is written in the same critical section that
    # retires the active attempt, so waiting for the key to leave _ACTIVE is
    # the event that makes the replay below deterministic (same drain wait as
    # test_review_worker_does_not_retry_after_its_logical_deadline).
    key = _attempt_key(request, slot)
    deadline = time.monotonic() + 5.0
    active = True
    while time.monotonic() < deadline:
        with _ACTIVE_LOCK:
            active = key in _ACTIVE
        if not active:
            break
        time.sleep(0.01)
    assert not active
    second_slot = ReviewSlot(
        slot_id="slot_a", model="same/model", timeout_sec=0.02,
        transport_timeout_sec=10,
    )
    second = run_review_request(request, slots=[second_slot], drive_root=tmp_path, llm=GatedLLM(), usage_ctx=ctx)
    assert len(calls) == 1
    assert second.actors[0]["status"] == "ok"
    assert second.actors[0]["operation_state"] == "late_settled"
    assert second.actors[0]["late_result_pending"] is False

def test_explicit_retry_key_joins_worker_after_prompt_history_changes(tmp_path):
    import threading
    from ouroboros.review_custody import _ACTIVE, _ACTIVE_LOCK, _attempt_key

    calls = []
    release = threading.Event()

    class GatedLLM:
        """Blocks the single paid call until the test releases it, so BOTH poll
        windows below expire while the worker is provably still in flight (the
        previous 0.08s-sleep-vs-0.02s-window race flaked on slow CI hosts)."""

        def chat(self, **kwargs):
            calls.append(kwargs)
            assert release.wait(10), "test never released the gated review call"
            return {"content": '{"verdict":"PASS","findings":[],"summary":"late"}'}, {}

    ctx = SimpleNamespace(task_id="history-retry", event_queue=None, pending_events=[])
    slot = ReviewSlot(slot_id="slot_a", model="same/model", timeout_sec=0.02)
    first_request = ReviewRequest(
        surface="scope_review", goal="review", task_id="history-retry",
        retry_key="snapshot-1/cycle-1", messages=[{"role": "user", "content": "first"}],
    )
    first = run_review_request(
        first_request,
        slots=[slot], drive_root=tmp_path, llm=GatedLLM(), usage_ctx=ctx,
    )
    second = run_review_request(
        ReviewRequest(
            surface="scope_review", goal="review", task_id="history-retry",
            retry_key="snapshot-1/cycle-1",
            messages=[{"role": "user", "content": "first\nprior round: pending"}],
        ),
        slots=[slot], drive_root=tmp_path, llm=GatedLLM(), usage_ctx=ctx,
    )

    assert len(calls) == 1
    assert first.actors[0]["operation_id"] == second.actors[0]["operation_id"]
    assert first.actors[0]["operation_state"] == second.actors[0]["operation_state"] == "in_flight"
    release.set()
    # Drain the released worker before teardown (same wait as
    # test_review_worker_does_not_retry_after_its_logical_deadline).
    key = _attempt_key(first_request, slot)
    deadline = time.monotonic() + 5.0
    active = True
    while time.monotonic() < deadline:
        with _ACTIVE_LOCK:
            active = key in _ACTIVE
        if not active:
            break
        time.sleep(0.01)
    assert not active

def test_explicit_retry_key_replays_normally_settled_actor_without_second_dispatch(tmp_path):
    calls = []

    class FastLLM:
        def chat(self, **kwargs):
            calls.append(kwargs)
            return {"content": '{"verdict":"PASS","findings":[],"summary":"done"}'}, {}

    ctx = SimpleNamespace(task_id="settled-retry", event_queue=None, pending_events=[])
    slot = ReviewSlot(slot_id="slot_a", model="same/model", timeout_sec=1)
    first = run_review_request(
        ReviewRequest(
            surface="plan_review", goal="review", task_id="settled-retry",
            retry_key="plan-1/cycle-1", messages=[{"role": "user", "content": "first"}],
        ),
        slots=[slot], drive_root=tmp_path, llm=FastLLM(), usage_ctx=ctx,
    )
    second = run_review_request(
        ReviewRequest(
            surface="plan_review", goal="review", task_id="settled-retry",
            retry_key="plan-1/cycle-1",
            messages=[{"role": "user", "content": "first plus rendered history"}],
        ),
        slots=[slot], drive_root=tmp_path, llm=FastLLM(), usage_ctx=ctx,
    )

    assert len(calls) == 1
    assert first.actors[0]["operation_id"] == second.actors[0]["operation_id"]
    assert second.actors[0]["operation_state"] == "settled"

def test_late_plan_api_error_replays_same_terminal_attempt(tmp_path):
    import threading
    from ouroboros.review_custody import _ACTIVE, _ACTIVE_LOCK, _attempt_key

    calls = []
    release = threading.Event()

    class GatedAPIError:
        """Holds the paid call open until the test releases it, so the poll
        window is GUARANTEED to expire while the call is still in flight.  The
        previous 0.05s-sleep-vs-0.01s-window race flaked the in_flight
        assertion on slow CI hosts (same event gate as
        test_replayed_late_review_does_not_charge_same_context_twice)."""

        def chat(self, **kwargs):
            calls.append(kwargs)
            assert release.wait(10), "test never released the gated review call"
            raise RuntimeError("provider ended the paid request")

    ctx = SimpleNamespace(task_id="late-plan-error", event_queue=None, pending_events=[])
    request = ReviewRequest(
        surface="plan_review", goal="review", task_id="late-plan-error",
        retry_key="plan-envelope/cycle-1",
    )
    slot = ReviewSlot(
        slot_id="slot_a", model="same/model", timeout_sec=0.01,
        transport_timeout_sec=10,
    )
    first = run_review_request(
        request, slots=[slot], drive_root=tmp_path, llm=GatedAPIError(), usage_ctx=ctx,
    )
    assert first.actors[0]["operation_state"] == "in_flight"
    operation_id = first.actors[0]["operation_id"]
    release.set()
    # The settled-attempt cache is written in the same critical section that
    # retires the active attempt, so draining BOTH facts under _ACTIVE_LOCK is
    # the event that makes the replay below deterministic (same wait as
    # test_replayed_late_review_does_not_charge_same_context_twice).
    key = _attempt_key(request, slot)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        with _ACTIVE_LOCK:
            settled = key in getattr(ctx, "_review_settled_attempts", {})
            active = key in _ACTIVE
        if settled and not active:
            break
        time.sleep(0.01)
    else:
        raise AssertionError("late plan API error was not retained for reconciliation")

    second = run_review_request(
        request, slots=[slot], drive_root=tmp_path, llm=GatedAPIError(), usage_ctx=ctx,
    )
    assert len(calls) == 1
    assert second.actors[0]["status"] == "error"
    assert second.actors[0]["operation_id"] == operation_id
    assert second.actors[0]["operation_state"] == "late_settled"
    assert "provider ended the paid request" in second.actors[0]["error"]

def test_review_paid_stamp_is_write_ahead_of_a_slow_worker(tmp_path):
    import threading
    from types import SimpleNamespace
    from ouroboros.review_custody import _ACTIVE, _ACTIVE_LOCK, _attempt_key

    order = []
    release = threading.Event()

    class GatedLLM:
        """Holds the transport call open until the test releases it, so the
        poll window is GUARANTEED to expire while the worker is in flight.
        The previous 0.08s-sleep-vs-0.02s-window race flaked the in_flight
        assertion on slow CI hosts (same event gate as
        test_replayed_late_review_does_not_charge_same_context_twice)."""

        def chat(self, **_kwargs):
            order.append("transport")
            assert release.wait(10), "test never released the gated review call"
            return {"content": '{"verdict":"PASS","findings":[],"summary":"ok"}'}, {}

    ctx = SimpleNamespace(
        task_id="paid-before-worker", event_queue=None, pending_events=[],
        _review_paid_stamp=lambda: order.append("paid"),
    )
    request = ReviewRequest(surface="scope", goal="review", task_id="paid-before-worker")
    slot = ReviewSlot(slot_id="slot_a", model="same/model", timeout_sec=0.02)
    result = run_review_request(
        request, slots=[slot], drive_root=tmp_path, llm=GatedLLM(), usage_ctx=ctx,
    )
    assert result.actors[0]["operation_state"] == "in_flight"
    release.set()
    # Drain the released worker before judging the order: the caller can
    # return in_flight before the worker thread has entered chat at all, so
    # order[1] raced an IndexError; the drain proves the transport entry
    # exists.  ``order`` is append-only, so the paid-before-transport contract
    # is judged unchanged (same drain as
    # test_explicit_retry_key_joins_worker_after_prompt_history_changes).
    key = _attempt_key(request, slot)
    deadline = time.monotonic() + 5.0
    active = True
    while time.monotonic() < deadline:
        with _ACTIVE_LOCK:
            active = key in _ACTIVE
        if not active:
            break
        time.sleep(0.01)
    assert not active
    assert order[0] == "paid"
    assert order[1] == "transport"

def test_replayed_late_review_does_not_charge_same_context_twice(tmp_path):
    import threading
    from types import SimpleNamespace
    from ouroboros.review_custody import _ACTIVE, _ACTIVE_LOCK, _attempt_key

    calls = []
    release = threading.Event()

    class GatedLLM:
        """Holds the paid call open until the test releases it, so the first
        poll window is GUARANTEED to expire while the call is still in flight.
        The previous 0.06s-sleep-vs-0.02s-window race flaked on slow CI hosts:
        an oversleeping poll wait could observe the settled result and return
        'settled' instead of 'in_flight' (same event gate as
        test_late_review_result_is_replayed_without_a_second_paid_dispatch)."""

        def chat(self, **_kwargs):
            calls.append(1)
            assert release.wait(10), "test never released the gated review call"
            return {"content": '{"verdict":"PASS","findings":[],"summary":"late"}'}, {}

    ctx = SimpleNamespace(
        task_id="paid-replay", event_queue=None, pending_events=[],
        _review_paid_stamp=lambda: calls.append("paid"),
    )
    request = ReviewRequest(surface="scope", goal="review", task_id="paid-replay")
    first = run_review_request(
        request,
        slots=[ReviewSlot(slot_id="slot_a", model="same/model", timeout_sec=0.02)],
        drive_root=tmp_path, llm=GatedLLM(), usage_ctx=ctx,
    )
    assert first.actors[0]["operation_state"] == "in_flight"
    release.set()
    # The late result is cached only after the worker has actually settled.  The
    # settled-attempt cache is written in the same critical section that retires
    # the active attempt, so this drain is the event that makes the replay below
    # deterministic (same wait as
    # test_review_worker_does_not_retry_after_its_logical_deadline).
    key = _attempt_key(request, ReviewSlot(slot_id="slot_a", model="same/model"))
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        with _ACTIVE_LOCK:
            settled = key in getattr(ctx, "_review_settled_attempts", {})
            active = key in _ACTIVE
        if settled and not active:
            break
        time.sleep(0.01)
    else:
        raise AssertionError("late review worker did not settle into same-context custody")
    second = run_review_request(
        request,
        slots=[ReviewSlot(slot_id="slot_a", model="same/model", timeout_sec=0.01)],
        drive_root=tmp_path, llm=GatedLLM(), usage_ctx=ctx,
    )
    assert calls.count("paid") == 1
    assert sum(1 for item in calls if item == 1) == 1
    assert second.actors[0]["operation_state"] == "late_settled"

def test_review_slot_timeout_is_not_used_as_transport_timeout(tmp_path):
    captured = []

    class CapturingLLM:
        def chat(self, **kwargs):
            captured.append(kwargs)
            return {"content": '{"verdict":"PASS","findings":[],"summary":"ok"}'}, {}

    result = run_review_request(
        ReviewRequest(surface="scope", goal="review", task_id="transport-separation"),
        slots=[ReviewSlot(
            slot_id="slot_a", model="same/model", timeout_sec=0.5,
            transport_timeout_sec=17,
        )],
        drive_root=tmp_path,
        llm=CapturingLLM(),
    )
    assert result.aggregate_signal == "PASS"
    assert captured and captured[0]["timeout"] == 17

def test_review_transport_timeout_is_narrowed_by_request_deadline(tmp_path, monkeypatch):
    from datetime import datetime, timedelta, timezone

    captured = []

    class CapturingLLM:
        def chat(self, **kwargs):
            captured.append(kwargs)
            return {"content": '{"verdict":"PASS","findings":[],"summary":"ok"}'}, {}

    monkeypatch.setenv("OUROBOROS_FINALIZATION_GRACE_SEC", "0")
    deadline = (datetime.now(timezone.utc) + timedelta(seconds=5)).isoformat()
    run_review_request(
        ReviewRequest(
            surface="scope", goal="review", task_id="deadline-transport",
            deadline_at=deadline,
        ),
        slots=[ReviewSlot(slot_id="slot_a", model="same/model", timeout_sec=30)],
        drive_root=tmp_path,
        llm=CapturingLLM(),
    )
    assert captured and 0 < captured[0]["timeout"] <= 5

def test_api_chat_retry_recomputes_transport_window(tmp_path, monkeypatch):
    from ouroboros import review_execution

    captured = []
    transport_windows = iter((479.96, 1.0))

    class RetryingLLM:
        def chat(self, **kwargs):
            captured.append(kwargs["timeout"])
            if len(captured) == 1:
                return {"content": ""}, {}
            return {"content": '{"verdict":"PASS","findings":[],"summary":"ok"}'}, {}

    monkeypatch.setattr(
        review_execution,
        "review_transport_timeout",
        lambda *_args: next(transport_windows),
    )
    result = run_review_request(
        ReviewRequest(surface="scope_review", goal="retry", task_id="retry-timeout"),
        slots=[ReviewSlot(slot_id="slot_a", model="same/model")],
        drive_root=tmp_path,
        llm=RetryingLLM(),
    )

    assert result.aggregate_signal == "PASS"
    assert captured == [479.96, 1.0]

def test_direct_anthropic_route_keeps_provider_default_transport(tmp_path):
    captured = []

    class CapturingLLM:
        def chat(self, **kwargs):
            captured.append(kwargs)
            return {"content": '{"verdict":"PASS","findings":[],"summary":"ok"}'}, {}

    run_review_request(
        ReviewRequest(surface="scope", goal="review", task_id="anthropic-timeout"),
        slots=[ReviewSlot(slot_id="slot_a", model="anthropic::claude-test", timeout_sec=0.5)],
        drive_root=tmp_path,
        llm=CapturingLLM(),
    )
    assert captured and captured[0]["timeout"] is None

def test_late_error_is_not_cached_as_a_permanent_review_verdict(tmp_path):
    import threading
    from types import SimpleNamespace
    from ouroboros.review_custody import _ACTIVE, _ACTIVE_LOCK, _attempt_key

    calls = []
    release = threading.Event()

    class ErrorThenSuccess:
        def chat(self, **_kwargs):
            calls.append(1)
            if len(calls) == 1:
                # Gated: the first poll window (0.005s) is GUARANTEED to
                # expire while the call is still in flight.  The previous
                # 0.03s-sleep-vs-0.005s-window race flaked the in_flight
                # assertion on slow CI hosts (same event gate as
                # test_late_agent_session_preflight_failure_can_retry).
                assert release.wait(10), "test never released the gated review call"
                raise RuntimeError("transient provider failure")
            return {"content": '{"verdict":"PASS","findings":[],"summary":"ok"}'}, {}

    ctx = SimpleNamespace(task_id="late-error", event_queue=None, pending_events=[])
    request = ReviewRequest(surface="plan_review", goal="review", task_id="late-error")
    slot = ReviewSlot(slot_id="slot_a", model="same/model", timeout_sec=0.005)
    first = run_review_request(
        request,
        slots=[slot],
        drive_root=tmp_path,
        llm=ErrorThenSuccess(),
        usage_ctx=ctx,
    )
    assert first.actors[0]["operation_state"] == "in_flight"
    release.set()
    # The first physical operation must settle before the retry is admitted.
    # A transient error is NOT retained for replay, so the deterministic
    # signal that the retry may dispatch is the attempt leaving _ACTIVE.  The
    # previous fixed 0.05s sleep raced the settle on slow CI hosts: the second
    # run then JOINED the still-active attempt instead of retrying (same drain
    # as test_late_agent_session_preflight_failure_can_retry).
    key = _attempt_key(request, slot)
    deadline = time.monotonic() + 5.0
    active = True
    while time.monotonic() < deadline:
        with _ACTIVE_LOCK:
            active = key in _ACTIVE
        if not active:
            break
        time.sleep(0.01)
    assert not active
    second = run_review_request(
        request,
        slots=[ReviewSlot(slot_id="slot_a", model="same/model", timeout_sec=0.5)],
        drive_root=tmp_path,
        llm=ErrorThenSuccess(),
        usage_ctx=ctx,
    )
    assert len(calls) >= 2
    assert second.actors[0]["status"] == "ok"

def test_late_agent_session_failure_is_replayed_without_a_second_run(tmp_path, monkeypatch):
    import threading
    from ouroboros.review_custody import _ACTIVE, _ACTIVE_LOCK, _attempt_key
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_substrate import ReviewCoordinator

    calls = []
    release = threading.Event()

    def gated_terminal_failure(
        self, request, slot, *, operation_id="", retry_state=None,
        logical_deadline_monotonic=None,
    ):
        # Blocks until the test releases it, so the first poll window is
        # GUARANTEED to expire while the delegated run is still in flight.
        # The previous fixed 0.03s sleep raced the 0.005s poll window on slow
        # CI hosts: an oversleeping poll wait could observe the settled result
        # and surface 'settled' at the in_flight assertion below (same event
        # gate as test_replayed_late_review_does_not_charge_same_context_twice).
        calls.append(operation_id)
        assert release.wait(10), "test never released the gated session run"
        actor = self._error_actor(
            request, slot, "delegated run settled failed",
            operation_id=operation_id,
        )
        actor.usage = {"delegated_run_started": True, "delegated_run_id": "run-1"}
        return actor

    monkeypatch.setattr(ReviewCoordinator, "_run_slot", gated_terminal_failure)
    ctx = SimpleNamespace(task_id="late-session-error", event_queue=None, pending_events=[])
    request = ReviewRequest(
        surface="plan_review", goal="review", task_id="late-session-error",
        session_root=str(tmp_path), session_task="review this tree",
    )
    slot = ReviewSlot(
        slot_id="slot_a", model="same/model", timeout_sec=0.005,
        route=ReviewRouteKind.AGENT_SESSION,
    )
    first = run_review_request(
        request, slots=[slot], drive_root=tmp_path, llm=FakeLLM(), usage_ctx=ctx,
    )
    assert first.actors[0]["operation_state"] == "in_flight"
    release.set()
    # The settled-attempt cache is written in the same critical section that
    # retires the active attempt, so draining BOTH facts under _ACTIVE_LOCK is
    # the event that makes the replay below deterministic (same wait as
    # test_replayed_late_review_does_not_charge_same_context_twice).
    key = _attempt_key(request, slot)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        with _ACTIVE_LOCK:
            settled = key in getattr(ctx, "_review_settled_attempts", {})
            active = key in _ACTIVE
        if settled and not active:
            break
        time.sleep(0.01)
    else:
        raise AssertionError("late session failure did not settle into custody")
    second = run_review_request(
        request, slots=[slot], drive_root=tmp_path, llm=FakeLLM(), usage_ctx=ctx,
    )
    assert len(calls) == 1
    assert second.actors[0]["status"] == "error"
    assert second.actors[0]["operation_state"] == "late_settled"

def test_late_agent_session_preflight_failure_can_retry(tmp_path, monkeypatch):
    import threading
    from ouroboros.review_custody import _ACTIVE, _ACTIVE_LOCK, _attempt_key
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_substrate import ReviewCoordinator

    calls = []
    release = threading.Event()

    def preflight_then_success(
        self, request, slot, *, operation_id="", retry_state=None,
        logical_deadline_monotonic=None,
    ):
        calls.append(operation_id)
        if len(calls) == 1:
            # Gated: the first poll window (0.005s) is GUARANTEED to expire
            # while the preflight attempt is still in flight.  The previous
            # fixed 0.02s sleep raced that window on slow CI hosts and could
            # surface 'settled' at the in_flight assertion below (same event
            # gate as
            # test_explicit_retry_key_joins_worker_after_prompt_history_changes).
            assert release.wait(10), "test never released the gated preflight"
            return self._error_actor(
                request, slot, "route unavailable before dispatch",
                operation_id=operation_id,
            )
        actor = self._error_actor(request, slot, "unused", operation_id=operation_id)
        actor.status, actor.error, actor.raw_text = "ok", "", '{"verdict":"PASS","findings":[]}'
        return actor

    monkeypatch.setattr(ReviewCoordinator, "_run_slot", preflight_then_success)
    ctx = SimpleNamespace(task_id="late-session-preflight", event_queue=None, pending_events=[])
    request = ReviewRequest(
        surface="plan_review", goal="review", task_id="late-session-preflight",
        session_root=str(tmp_path), session_task="review this tree",
    )
    slot = ReviewSlot(
        slot_id="slot_a", model="same/model", timeout_sec=0.005,
        route=ReviewRouteKind.AGENT_SESSION,
    )
    first = run_review_request(
        request, slots=[slot], drive_root=tmp_path, llm=FakeLLM(), usage_ctx=ctx,
    )
    assert first.actors[0]["operation_state"] == "in_flight"
    release.set()
    # A plain preflight error is retryable, so it leaves NOTHING in the settled
    # cache; the deterministic signal that the retry may dispatch is the
    # attempt leaving _ACTIVE (retired under _ACTIVE_LOCK in
    # _settle_review_attempt).  The previous fixed 0.04s sleep raced that
    # settle on slow CI hosts: the second run then JOINED the still-active
    # attempt instead of dispatching the retry (same drain as
    # test_late_unknown_session_start_restores_exact_pending_invocation).
    key = _attempt_key(request, slot)
    deadline = time.monotonic() + 5.0
    active = True
    while time.monotonic() < deadline:
        with _ACTIVE_LOCK:
            active = key in _ACTIVE
        if not active:
            break
        time.sleep(0.01)
    assert not active
    second = run_review_request(
        request, slots=[ReviewSlot(
            slot_id="slot_a", model="same/model", timeout_sec=0.5,
            route=ReviewRouteKind.AGENT_SESSION,
        )], drive_root=tmp_path, llm=FakeLLM(), usage_ctx=ctx,
    )
    assert len(calls) == 2
    assert second.actors[0]["status"] == "ok"

def test_late_unknown_session_start_restores_exact_pending_invocation(tmp_path, monkeypatch):
    from ouroboros.review_custody import _ACTIVE, _ACTIVE_LOCK, _attempt_key
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_substrate import ReviewCoordinator

    calls, paid = [], []

    def pending_then_success(
        self, request, slot, *, operation_id="", retry_state=None,
        logical_deadline_monotonic=None,
    ):
        calls.append(dict(retry_state or {}))
        if len(calls) == 1:
            time.sleep(0.02)
            actor = self._error_actor(request, slot, "start outcome unknown", operation_id=operation_id)
            actor.usage = {"pending_invocation_id": "invocation-1"}
            return actor
        assert retry_state == {"pending_invocation_id": "invocation-1"}
        actor = self._error_actor(request, slot, "unused", operation_id=operation_id)
        actor.status, actor.error, actor.raw_text = "ok", "", '{"verdict":"PASS","findings":[]}'
        return actor

    monkeypatch.setattr(ReviewCoordinator, "_run_slot", pending_then_success)
    ctx = SimpleNamespace(
        task_id="late-session-pending", event_queue=None, pending_events=[],
        _review_paid_stamp=lambda: paid.append("paid"),
    )
    request = ReviewRequest(
        surface="plan_review", goal="review", task_id="late-session-pending",
        session_root=str(tmp_path), session_task="review this tree",
    )
    first_slot = ReviewSlot(
        slot_id="slot_a", model="same/model", timeout_sec=0.005,
        route=ReviewRouteKind.AGENT_SESSION,
    )
    first = run_review_request(
        request, slots=[first_slot], drive_root=tmp_path, llm=FakeLLM(), usage_ctx=ctx,
    )
    assert first.actors[0]["operation_state"] == "in_flight"
    # The pending-invocation row is persisted inside the same _ACTIVE_LOCK
    # critical section that retires the active attempt, so draining the key out
    # of _ACTIVE is the deterministic signal that the restored invocation is
    # durably recorded.  The previous fixed 0.04s sleep raced the worker's
    # settle on slow CI hosts: the second run then JOINED the still-active
    # attempt instead of dispatching the restored pending retry (same drain as
    # test_review_worker_does_not_retry_after_its_logical_deadline).
    key = _attempt_key(request, first_slot)
    deadline = time.monotonic() + 5.0
    active = True
    while time.monotonic() < deadline:
        with _ACTIVE_LOCK:
            active = key in _ACTIVE
        if not active:
            break
        time.sleep(0.01)
    assert not active
    second = run_review_request(
        request, slots=[ReviewSlot(
            slot_id="slot_a", model="same/model", timeout_sec=0.5,
            route=ReviewRouteKind.AGENT_SESSION,
        )], drive_root=tmp_path, llm=FakeLLM(), usage_ctx=ctx,
    )
    assert calls == [{}, {"pending_invocation_id": "invocation-1"}]
    assert paid == ["paid"]
    assert second.actors[0]["status"] == "ok"

def test_review_slots_keep_independent_logical_windows(tmp_path):
    import threading
    from ouroboros.review_custody import _ACTIVE, _ACTIVE_LOCK, _attempt_key

    short_release = threading.Event()

    class GatedShortLLM:
        """The short slot's call stays open until the test releases it, so its
        0.05s window is GUARANTEED to expire (in_flight) while the long slot
        answers inside its own independent window.  The previous 0.25s-sleep-
        vs-0.05s-window margin was NOT discriminating on a loaded CI host: the
        heal wave measured a 0.207s poll oversleep there, which eats the whole
        0.2s absolute margin (same event gate as
        test_replayed_late_review_does_not_charge_same_context_twice)."""

        def chat(self, **kwargs):
            if kwargs.get("model") == "short/model":
                assert short_release.wait(10), "test never released the short slot"
            return {"content": '{"verdict":"PASS","findings":[],"summary":"ok"}'}, {}

    request = ReviewRequest(surface="scope", goal="review", task_id="independent-windows")
    short_slot = ReviewSlot(slot_id="short", model="short/model", timeout_sec=0.05)
    result = run_review_request(
        request,
        slots=[
            short_slot,
            ReviewSlot(slot_id="long", model="long/model", timeout_sec=0.5),
        ],
        drive_root=tmp_path,
        llm=GatedShortLLM(),
    )
    rows = {actor["slot_id"]: actor for actor in result.actors}
    assert rows["short"]["operation_state"] == "in_flight"
    assert rows["long"]["status"] == "ok"
    short_release.set()
    # Drain the released short worker before teardown (same drain as
    # test_explicit_retry_key_joins_worker_after_prompt_history_changes).
    key = _attempt_key(request, short_slot)
    deadline = time.monotonic() + 5.0
    active = True
    while time.monotonic() < deadline:
        with _ACTIVE_LOCK:
            active = key in _ACTIVE
        if not active:
            break
        time.sleep(0.01)
    assert not active
