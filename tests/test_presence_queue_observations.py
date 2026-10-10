"""Transport facts through authenticated Host and the same live review-wait author.

Real ASGI routes/admission/task sources/loop/gate; model and reviewer are scripted
by the shared harness. This is not a separate Host/Slack process qualification.
"""

import copy
import json
import re
from types import SimpleNamespace

import pytest

from ouroboros.presence_continuation import ReviewWaitBinding, reentry_note
from ouroboros.presence_observations import transport_queue_observation
from ouroboros.task_results import load_task_result
from tests.test_presence_continuation import ANSWER, NEW_WORDS, finish, harness as harness, wait_for
from tests.test_presence_continuation_host import _HEADERS, host as host, poll, turn
from tests.test_presence_reentry import _read_all


KEY = "telegram:bot-1:room-1:topic-1"
pytestmark = pytest.mark.serial


@pytest.fixture(autouse=True)
def settle_parked_authors(host):
    host.parked_for_test = []
    yield
    host.release.set()
    for ref in host.parked_for_test:
        wait_for(lambda: poll(host, ref).status_code == 200, timeout=60, what="test author cleanup")


def snapshot(texts=("queued after the original",), *, second=0):
    return {"schema_version": 1, "source": "fixture bridge inbox", "conversation_key": KEY,
            "observed_at": f"2026-10-06T10:00:{second:02d}+00:00", "after_source_event_id": "telegram:bot-1:42",
            "complete": True, "pending_count": len(texts), "omitted_count": 0,
            "events": [{"source_event_id": f"queue-{i}", "text": text, "text_chars": len(text),
                        "text_truncated": False, "actor": {"platform_actor_id": "alex"}}
                       for i, text in enumerate(texts)]}


def refresh(h, ref, value, **extra):
    return h.client.post(f"/presence/work/{ref}", headers=_HEADERS,
                         json={"binding_id": h.binding, "transport_queue": value, **extra})


@pytest.mark.parametrize("early", [False, True])
@pytest.mark.parametrize("reporting", [0, 1])
def test_refreshed_queue_reaches_same_author_without_holding_room(host, monkeypatch, early, reporting):
    h = host
    if early:
        monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
    seen = []

    def scripted(messages):
        notes = [str(row.get("content")) for row in messages if "[PRESENCE CONVERSATION RESUMED]" in str(row.get("content"))]
        if notes:
            seen.append(notes[-1])
            assert "fresh queued words" in notes[-1] and NEW_WORDS in notes[-1]
            assert '"observed_at": "2026-10-06T10:00:01+00:00"' in notes[-1]
            assert "old queued words" not in notes[-1]
            assert "not canonical chat or owner directives" in notes[-1]
            return finish("final", answer_sha256=h.ctxs[0]._delivery_candidate.content_sha256)
        return finish("nominate", message=ANSWER, **({"pending_review": "finish"} if early else {}))

    h.script = scripted
    first = turn(h, 42, reporting=reporting, transport_queue=snapshot(("old queued words",))).json()
    ref = first["continuation_ref"]
    assert first["status"] == "continuing" and first["text"] == (ANSWER if early else "")
    updated = snapshot(("fresh queued words",), second=1)
    assert refresh(h, ref, updated).json()["status"] == "recorded"
    h.quick.add("telegram:bot-1:43")
    assert turn(h, 43, NEW_WORDS, reporting=reporting).json()["text"] == "Noted."
    # Replay does not substitute a newer queue observation or change the first envelope.
    assert turn(h, 42, reporting=reporting, transport_queue=snapshot(("old queued words",))).json() == first
    assert refresh(h, ref, snapshot(("old queued words",))).json()["status"] == "stale"
    h.release.set()
    wait_for(lambda: poll(h, ref).status_code == 200, what="same author terminal")
    assert seen and h.calls == 2 and len(h.reviews) == 1
    assert poll(h, ref).json()["text"] == ("" if early else ANSWER)
    assert load_task_result(h.data, ref)["presence_transport_queue"]["observed_at"] == updated["observed_at"]


def _park(h):
    h.script = lambda messages: (finish("final", message=ANSWER)
                                 if any("[PRESENCE CONVERSATION RESUMED]" in str(row.get("content")) for row in messages)
                                 else finish("nominate", message=ANSWER))
    ref = turn(h, 42).json()["continuation_ref"]
    h.parked_for_test.append(ref)
    return ref


def test_complete_empty_snapshot_replaces_old_queue_and_get_is_passive(host):
    h = host
    ref = _park(h)
    first, empty = snapshot(), snapshot((), second=1)
    assert refresh(h, ref, first).json()["status"] == "recorded"
    assert refresh(h, ref, first).json()["status"] == "duplicate"
    assert refresh(h, ref, empty).json()["status"] == "recorded"
    before = load_task_result(h.data, ref)
    assert poll(h, ref).status_code == 202
    assert load_task_result(h.data, ref) == before
    event = SimpleNamespace(**before["metadata"]["presence"]["event"])
    observation = transport_queue_observation(h.data, ref, event)
    assert observation["status"] == "available" and observation["snapshot"]["events"] == []
    assert refresh(h, ref, first).json()["status"] == "stale"
    conflicting = snapshot(("different",), second=1)
    response = refresh(h, ref, conflicting)
    assert response.status_code == 400 and response.json()["code"] == "presence_observation_invalid"
    assert load_task_result(h.data, ref) == before


@pytest.mark.parametrize("change", [
    {"conversation_key": "telegram:bot-1:another-room:topic-1"},
    {"after_source_event_id": "telegram:bot-1:other"}, {"complete": False}, {"pending_count": 2},
    {"omitted_count": 1}, {"observed_at": "2026-10-06T10:00:00"},
    {"events": [{"source_event_id": "queue-0", "text": "cut", "text_truncated": True}]},
])
def test_partial_or_wrong_source_queue_is_rejected_before_author_start(host, change):
    response = turn(host, 42, transport_queue={**snapshot(), **change})
    assert response.status_code == 400 and host.calls == 0


def test_long_queue_is_readable_by_actor_through_existing_scoped_tool(host):
    from ouroboros.tools.registry import ToolRegistry

    h = host
    ref = _park(h)
    texts = [f"event {i}: " + "q" * 3000 for i in range(12)] + ["z" * 30_000 + " last queued words"]
    value = snapshot(tuple(texts))
    assert refresh(h, ref, value).status_code == 200
    stored = load_task_result(h.data, ref)
    event = SimpleNamespace(**stored["metadata"]["presence"]["event"])
    binding = ReviewWaitBinding(None, h.data, ref, "", event, cursor={"offset": 0})
    note = reentry_note(binding, h.ctxs[0])
    assert "queued event(s) omitted" in note and "last queued words" not in note
    args = json.loads(re.search(r'get_task_result\((\{[^\n]+?\})\)', note).group(1))
    registry = ToolRegistry(repo_dir=h.repo, drive_root=h.data)
    registry.set_context(h.ctxs[0])
    full = _read_all(registry, args)
    assert full["transport_queue"]["snapshot"] == value
    assert all(event["text"] == text for event, text in zip(full["transport_queue"]["snapshot"]["events"], texts))


def test_failed_retention_and_foreign_binding_do_not_replace_queue(host, monkeypatch):
    h = host
    ref = _park(h)
    assert refresh(h, ref, snapshot()).status_code == 200
    before = copy.deepcopy(load_task_result(h.data, ref))
    response = refresh(h, ref, snapshot(second=1), binding_id="b" * 32)
    assert response.status_code == 404
    missing = refresh(h, "not-a-task", snapshot(second=1))
    assert missing.status_code == 404 and missing.json()["code"] == "presence_work_not_found"

    def fail(*args, **kwargs):
        raise OSError("fixture source write failed")

    monkeypatch.setattr("ouroboros.artifacts.store_actor_source_bytes", fail)
    response = refresh(h, ref, snapshot(second=1))
    assert response.status_code == 500 and response.json()["disposition"] == "retry"
    assert load_task_result(h.data, ref) == before
