"""Request history deduplication preserves the blob serialization and readers."""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros import loop_llm_call, observability


@pytest.mark.parametrize("original,sent,omitted", [
    ({"role": "user", "content": "Hello Ω"}, {"content": "Hello Ω", "role": "user"}, True),
    ({"content": ("one", "two")}, {"content": ["one", "two"]}, True),
    ({"content": Path("fixture.txt")}, {"content": "fixture.txt"}, True),
    ({"content": "original"}, {"content": "adapted"}, False),
    ({"content": 1}, {"content": True}, False),
    ({"content": 1}, {"content": 1.0}, False),
])
def test_loop_persists_only_distinct_serialized_send_history(tmp_path, monkeypatch, original, sent, omitted):
    messages, send_messages = [original], [sent]

    def send(_llm, kwargs, **_options):
        assert kwargs["messages"] is send_messages
        return {"content": "done"}, {"cost": 0.0}

    monkeypatch.setattr(loop_llm_call, "_send_main_candidate", send)
    response, _ = loop_llm_call.call_llm_with_retry(
        object(), messages, "test-model", None, "low", 1,
        tmp_path / "logs", "task", 0, None, {}, initial_messages=send_messages,
    )
    assert response == {"content": "done"}
    call = next((tmp_path / "observability" / "calls" / "task").glob("*_request.json"))
    manifest, restored, _ = observability.read_call_payload(tmp_path, task_id="task", call_id=call.stem)
    stored = observability.read_blob_ref(tmp_path, manifest["full_payload_ref"])
    assert ("send_messages" not in stored) is omitted
    expected = json.loads(json.dumps(send_messages, ensure_ascii=False, sort_keys=True, default=str))
    assert json.dumps(restored["send_messages"]) == json.dumps(expected)
    assert restored == {**stored, "send_messages": expected}
    if omitted:
        expanded = json.dumps(restored, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        assert manifest["full_payload_ref"]["size"] < len(expanded)


@pytest.mark.parametrize("sent", [[], None, [{"content": "different"}]])
def test_request_reader_preserves_explicit_send_history(tmp_path, sent):
    payload = {"messages": [{"content": "canonical"}], "send_messages": sent}
    observability.persist_call(tmp_path, task_id="task", call_id="old_request", call_type="llm_request",
                               payload=payload)
    _, restored, _ = observability.read_call_payload(tmp_path, task_id="task", call_id="old_request")
    assert restored == payload


def test_other_call_payload_is_not_expanded(tmp_path):
    payload = {"messages": [{"content": "canonical"}]}
    observability.persist_call(tmp_path, task_id="task", call_id="other", call_type="other", payload=payload)
    _, restored, _ = observability.read_call_payload(tmp_path, task_id="task", call_id="other")
    assert restored == payload


@pytest.mark.parametrize("duplicate", [False, True])
def test_review_closure_retains_request_sources_without_duplicate_history(tmp_path, duplicate):
    from ouroboros import artifacts, context_compaction as cc, headless
    from tests.test_context_reclaim_materializer import _request, _unit

    parent, task = tmp_path / "canonical", "request-source"
    child = headless.prepare_task_drive(parent, task, "empty")
    original = _unit("read", "original evidence")
    request = _request(original, 10)
    ref = cc._persist_reclaim_checkpoint(original, request,
        SimpleNamespace(fingerprint="original", units=[]), drive_root=child, task_id=task)
    raw = artifacts.read_actor_source_bytes(child, task, ref)
    unit = cc._atomic_units(original)[-1]
    capsule, _ = cc._capsule_message(unit, "summary",
                                    [cc._part(unit.unit_id, unit.source_text)], ref, request)
    payload = {"messages": [capsule]}
    if duplicate:
        payload["send_messages"] = payload["messages"]
    observability.persist_call(child, task_id=task, call_id="request", call_type="llm_request", payload=payload)
    _, state = observability.promote_child_task_refs(parent, child, task, {})
    assert state["call_inventory_preserved"] and state["status"] == "complete"
    shutil.rmtree(child)
    assert artifacts.read_actor_source_bytes(parent, task, ref) == raw
    _, restored, _ = observability.read_call_payload(parent, task_id=task, call_id="request")
    assert restored == {"messages": [capsule], "send_messages": [capsule]}
