"""ONE delivery-class predicate (`review_execution.delivery_retrieves`).

A reviewer row RETRIEVES the subject with its own tools when it is a hosted
session or an api row whose saved delivery is NATIVE (tool rounds); a PACKET
api row receives the assembled packet. The predicate is route × delivery —
F8: a pool row's ``subagent_id`` is identity only and says nothing about
delivery. Before this predicate existed, four callers carried their own
inline copy of that rule — a class that can drift.
"""

from ouroboros.review_execution import ReviewRouteKind, delivery_retrieves
from ouroboros.review_substrate import ReviewSlot
from ouroboros.reviewer_slot_config import ConfiguredReviewerSlot
from ouroboros.tools.plan_review_runtime import slot_retrieves


def test_predicate_accepts_route_kind_or_wire_string():
    assert delivery_retrieves(ReviewRouteKind.AGENT_SESSION, False) is True
    assert delivery_retrieves("agent_session", None) is True
    assert delivery_retrieves(ReviewRouteKind.API_CHAT, True) is True
    assert delivery_retrieves("api_chat", False) is False
    assert delivery_retrieves(ReviewRouteKind.API_CHAT, None) is False
    assert delivery_retrieves(None, False) is False


def test_owner_deadline_reaches_every_seat_of_the_one_wave(monkeypatch, tmp_path):
    """R23: the owner deadline is a bound of every retrieving episode, so every
    seat of the one commit-gate wave — the packet seat and the retrieving
    two-part seat alike — carries the task's deadline_at exactly as the advisory
    does; a context without one yields ''. Behavioural: the request each seat
    hands the substrate carries the context's deadline — the stub captures it
    and refuses before any send."""
    import asyncio
    from types import SimpleNamespace

    import ouroboros.review_substrate as substrate
    from ouroboros.tools import review as review_mod
    from ouroboros.tools.review import _owner_deadline_at

    ctx = SimpleNamespace(task_metadata={"deadline_at": "2030-01-01T00:00:00Z"})
    assert _owner_deadline_at(ctx) == "2030-01-01T00:00:00Z"
    assert _owner_deadline_at(SimpleNamespace(task_metadata={})) == ""
    assert _owner_deadline_at(SimpleNamespace()) == "" and _owner_deadline_at(None) == ""

    seen = []

    def _capture(request, **_kwargs):
        seen.append((request.surface, request.deadline_at, bool(request.session_task)))
        raise RuntimeError("captured before any send")

    monkeypatch.setattr(substrate, "run_review_request", _capture)
    ctx = SimpleNamespace(task_metadata={"deadline_at": "2030-01-01T00:00:00Z"}, task_id="t-deadline",
                          drive_root=str(tmp_path), pending_events=[], event_queue=None)
    _, payload, _ = asyncio.run(review_mod._query_model(object(), "openai/fake-reviewer", [], asyncio.Semaphore(1), ctx=ctx))
    assert "captured before any send" in payload["error"]
    _, payload, _ = asyncio.run(review_mod._query_model(
        object(), "openai/fake-coupling", [], asyncio.Semaphore(1), ctx=ctx, slot_id="slot_2",
        native_retrieval=True, session_task="two-part brief", session_root=str(tmp_path)))
    assert "captured before any send" in payload["error"]
    assert seen == [("multi_model_review", "2030-01-01T00:00:00Z", False),
                    ("multi_model_review", "2030-01-01T00:00:00Z", True)]


def test_slot_properties_and_plan_review_facade_share_the_predicate():
    api = ReviewSlot(slot_id="t1", model="m", effort="low", subagent_id="packet-critic", native_retrieval_override=False)
    native = ReviewSlot(slot_id="t2", model="m", effort="low", subagent_id="api-critic", native_retrieval_override=True)
    session = ReviewSlot(slot_id="t3", model="m", effort="low", route=ReviewRouteKind.AGENT_SESSION, subagent_id="builder")
    assert [s.retrieves for s in (api, native, session)] == [False, True, True]
    assert [slot_retrieves(s) for s in (api, native, session)] == [False, True, True]
    assert native.native_retrieval is True and session.native_retrieval is False
    # The executor-selecting property and the delivery predicate can never
    # disagree: wire-string routes normalize the same way, and an id alone
    # never makes an api slot retrieve (F8).
    stringy = ReviewSlot(slot_id="t4", model="m", effort="low", route="api_chat", native_retrieval_override=True)
    id_only = ReviewSlot(slot_id="t5", model="m", effort="low", subagent_id="api-critic")
    assert stringy.retrieves is True and stringy.native_retrieval is True
    assert id_only.retrieves is False and id_only.native_retrieval is False

    rows = [
        ConfiguredReviewerSlot(slot_id="a", kind="api_chat", target_id="m", effort="low", delivery="packet"),
        ConfiguredReviewerSlot(slot_id="b", kind="api_chat", target_id="m", effort="low", subagent_id="b", delivery="native"),
        ConfiguredReviewerSlot(slot_id="c", kind="agent_session", target_id="codex=m", effort="low"),
    ]
    assert [r.retrieves for r in rows] == [False, True, True]
