from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace


def test_query_model_timeout_becomes_error_actor(monkeypatch, request):
    from ouroboros.tools.review import _query_model
    from ouroboros.observability import read_blob_ref
    from ouroboros.tools.review_helpers import review_drive_root

    captured = {}
    release = threading.Event()
    entered = threading.Event()

    def finish_physical_operation():
        release.set()
        if worker := captured.get("worker"):
            worker.join(timeout=5)
            assert not worker.is_alive(), "the test owns its late review operation"

    request.addfinalizer(finish_physical_operation)

    class HangingClient:
        async def chat_async(self, **kwargs):
            captured.update(kwargs, worker=threading.current_thread())
            entered.set()
            assert release.wait(5), "the caller must release the controlled review"
            return {"content": "late"}, {}

    monkeypatch.setenv("OUROBOROS_REVIEW_MODEL_TIMEOUT_SEC", "0.01")
    monkeypatch.setattr(
        "ouroboros.review_execution.review_transport_timeout",
        lambda *_args, **_kwargs: 2700.0,
    )

    model, result, headers = asyncio.run(
        _query_model(
            HangingClient(), "fake/reviewer", [], asyncio.Semaphore(1),
            slot_id="slot_1",
        )
    )

    assert entered.wait(5), "the physical reviewer must enter before its facts are inspected"
    assert model == "fake/reviewer"
    assert headers is None
    assert result["error"].startswith("Error: Timeout after 0.01s")
    assert "physical review operation remains in flight" in result["error"]
    assert result["operation_state"] == "in_flight"
    assert result["late_result_pending"] is True
    assert captured["timeout"] == 2700.0
    assert result["prompt_ref"]["manifest_ref"]["path"]
    assert result["response_ref"]["manifest_ref"]["path"]
    prompt = read_blob_ref(
        review_drive_root(None), result["prompt_ref"]["redacted_projection_ref"]
    )
    assert prompt["slot"]["slot_id"] == "slot_1"
    assert prompt["slot"]["model"] == "fake/reviewer"
    assert prompt["slot"]["route"] == "api_chat"

    from ouroboros.tools.review import _parse_model_response
    from ouroboros.triad_review import parse_model_review_results

    flattened = _parse_model_response(model, result, headers)
    assert flattened["operation_id"] == result["operation_id"]
    assert flattened["operation_state"] == "in_flight"
    assert flattened["late_result_pending"] is True
    [record] = parse_model_review_results({"results": [flattened]}).actor_records
    assert record.to_dict()["operation_id"] == result["operation_id"]
    assert record.to_dict()["operation_state"] == "in_flight"
    assert record.to_dict()["late_result_pending"] is True


def test_blocking_triad_does_not_finalize_quorum_with_a_pending_reviewer(monkeypatch, tmp_path):
    from ouroboros.tools import review

    clean = lambda model: {
        "model": model, "verdict": "PASS", "text": "[]", "slot_id": model,
        "operation_id": f"op-{model}", "operation_state": "settled",
        "late_result_pending": False,
    }
    pending = {
        "model": "m3", "verdict": "ERROR", "text": "Timeout",
        "slot_id": "m3", "operation_id": "op-m3",
        "operation_state": "in_flight", "late_result_pending": True,
    }
    monkeypatch.setattr(
        review, "_handle_multi_model_review",
        lambda *_args, **_kwargs: json.dumps({"results": [clean("m1"), clean("m2"), pending]}),
    )
    ctx = SimpleNamespace(
        _last_triad_raw_results=[], _triad_withheld_seat_records=[],
        _review_degraded_reasons=[], _review_advisory=[],
        drive_logs=lambda: tmp_path, task_id="pending-triad",
    )
    prepared = {
        "prompt": "p", "stable_prefix_len": 0,
        "models": ["m1", "m2", "m3"], "routes": [], "row_plan": {},
        "session_task": "", "target_repo": tmp_path, "blocking_review": True,
    }

    blocked = review._dispatch_unified_review(ctx, "message", prepared)

    assert blocked and "REVIEW_PENDING" in blocked
    assert ctx._last_review_block_reason == "review_late_result_pending"
    [raw] = [row for row in ctx._last_triad_raw_results if row["model_id"] == "m3"]
    assert raw["operation_id"] == "op-m3"
    assert raw["operation_state"] == "in_flight"
    assert raw["late_result_pending"] is True


def test_custody_lost_slot_is_named_in_pending_report(monkeypatch, tmp_path):
    from ouroboros.tools import review

    monkeypatch.setattr(
        review,
        "_handle_multi_model_review",
        lambda *_args, **_kwargs: json.dumps({"results": [{"model": "placeholder"}]}),
    )

    def _lost_row(run_ctx, _model_results, _row_plan=None):
        run_ctx._last_triad_raw_results = [{
            "slot_id": "custody_lost_slot_1",
            "status": "error",
            "operation_state": "custody_lost",
            "late_result_pending": False,
        }]
        return [], [], [], []

    monkeypatch.setattr(review, "_collect_review_findings", _lost_row)
    ctx = SimpleNamespace(
        _last_triad_raw_results=[], _triad_withheld_seat_records=[],
        _review_degraded_reasons=[], _review_advisory=[],
        drive_logs=lambda: tmp_path, task_id="custody-lost-triad",
    )
    prepared = {
        "prompt": "p", "stable_prefix_len": 0,
        "models": ["placeholder"], "routes": [], "row_plan": {},
        "session_task": "", "target_repo": tmp_path, "blocking_review": True,
    }

    blocked = review._dispatch_unified_review(ctx, "message", prepared)

    assert blocked and "REVIEW_PENDING" in blocked
    assert "custody_lost_slot_1" in blocked
    assert ctx._last_review_block_reason == "review_late_result_pending"


def test_query_model_uses_configured_review_effort(monkeypatch):
    from ouroboros.tools.review import _query_model
    import ouroboros.review_substrate as substrate

    captured_efforts = []

    def fake_run_review_request(_request, slots, **_kwargs):
        captured_efforts.append(slots[0].effort)
        return SimpleNamespace(
            actors=[{
                "status": "ok",
                "raw_text": "[]",
                "usage": {},
                "prompt_ref": {},
                "response_ref": {},
            }]
        )

    monkeypatch.setenv("OUROBOROS_EFFORT_REVIEW", "high")
    monkeypatch.setattr(substrate, "run_review_request", fake_run_review_request)

    model, result, headers = asyncio.run(
        _query_model(object(), "fake/reviewer", [], asyncio.Semaphore(1))
    )

    assert model == "fake/reviewer"
    assert headers is None
    assert result["choices"][0]["message"]["content"] == "[]"
    assert captured_efforts == ["high"]
