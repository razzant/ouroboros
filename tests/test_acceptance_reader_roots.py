"""Only the named-source closure owner may relocate a native episode's readers."""
from __future__ import annotations

import json
import shutil
from dataclasses import replace

import pytest

from ouroboros.acceptance_retrieving import acceptance_retrieving_work_order, retain_review_source
from ouroboros.artifacts import read_actor_source_bytes, store_actor_source_bytes, store_task_artifact_bytes
from ouroboros.outcome_receipt_store import append_verification_receipt
from ouroboros.review_substrate import run_review_request
from ouroboros.task_results import write_task_result
from ouroboros.utils import append_jsonl
from tests.test_acceptance_delivery import _CLEAN_VERDICT, _EpisodeLLM, _tool_call, _fake_session
from tests.test_acceptance_source_first import _SOURCE_PATH, _prepared_request


@pytest.fixture(autouse=True)
def _provider_catalog_stays_off_the_wire(provider_catalog_offline):
    """Every work order here measures the native row's window; see `provider_catalog_offline`."""


@pytest.mark.parametrize("paged", [False, True])
def test_native_episode_reads_retained_external_sources_after_original_cleanup(tmp_path, paged):
    request, slot, author, canonical, workspace = _prepared_request(tmp_path, session=False)
    if not paged:
        request.evidence.pop("__immutable_core_overflow__")
        acceptance_retrieving_work_order(request, [slot], session_root=str(workspace), data_root=author)
    tid = request.task_id
    markers = ["live-original-result-7f", "live-original-receipt-3c", "live-original-trajectory-9b",
               "live-original-tool-bytes-2d", "live-original-artifact-8a"]
    write_task_result(author, tid, "completed", result=markers[0])
    assert append_verification_receipt(author, tid, {"tool": "verify_and_record", "status": "pass", "check": markers[1]})
    full = store_actor_source_bytes(author, tid, category="tool_results", source_id="external-output",
                                    data=markers[3].encode(), extension="txt")
    append_jsonl(author / "logs" / "tools.jsonl", {"task_id": tid, "result": markers[2], "result_source_ref": full})
    store_task_artifact_bytes(author, tid, "external.txt", markers[4].encode())

    class Reader(_EpisodeLLM):
        def _reply(self, kwargs):
            self.calls.append(dict(kwargs))
            if len(self.calls) == 1:
                prompt = kwargs["messages"][-1]["content"]
                assert not any(marker in prompt for marker in markers), "markers must come from actual external reads"
                closure = request.policy["review_source_closure"]
                retained = {row["name"]: row for row in closure["sources"] if row["status"] == "retained"}
                for row in retained.values():
                    assert row["retained_path"] in prompt
                shutil.rmtree(author)
                calls = [
                    _tool_call("read_file", retained["task-result"]["source_ref"]["read"]["arguments"], "result"),
                    _tool_call("read_file", retained["verification-receipts"]["source_ref"]["read"]["arguments"], "receipt"),
                    _tool_call("read_file", retained["tool-trajectory"]["source_ref"]["read"]["arguments"], "trajectory"),
                    _tool_call("read_file", {"root": "artifact_store", "path": full["path"]}, "full-output"),
                    _tool_call("read_file", retained["artifact:external.txt"]["source_ref"]["read"]["arguments"], "artifact"),
                ]
                if paged:
                    calls.append(_tool_call("read_file", {"root": "artifact_store", "path": _SOURCE_PATH.search(prompt).group(1)}, "packet"))
                reply = {"tool_calls": calls}
            else:
                outputs = {m["tool_call_id"]: m["content"] for m in kwargs["messages"] if m.get("role") == "tool"}
                for name, marker in zip(("result", "receipt", "trajectory", "full-output", "artifact"), markers):
                    assert marker in outputs[name], outputs[name]
                if paged:
                    assert "TRAJECTORY-RESULT-3-passed" in outputs["packet"]
                reply = {"content": json.dumps(_CLEAN_VERDICT)}
            return reply, {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.0}

    llm = Reader(canonical, [])
    # Real coordinator, operation custody, executor, registry and file readers.
    # Only the model transport is scripted.
    result = run_review_request(request, slots=[slot], drive_root=canonical, llm=llm)
    assert result.aggregate_signal == "PASS", result.actors
    assert len(llm.calls) == 2
    delivery = request.slot_source_delivery[slot.slot_id]
    assert delivery["status"] == ("paged" if paged else "inline")
    assert request.policy["native_data_root"] == delivery["reader_root"] != str(author)
    assert delivery["custody_root"] == str(canonical)
    assert delivery["external_ref_closure"] == "retained" and not author.exists()
    assert read_actor_source_bytes(delivery["reader_root"], tid, delivery["source"]) == read_actor_source_bytes(canonical, tid, delivery["custody_source"])
    assert not (canonical / "task_results" / (tid + ".json")).exists()
    assert not (canonical / "logs" / "tools.jsonl").exists()
    assert len(result.actors[0]["usage"]["native_tool_receipts"]) >= len(markers)


@pytest.mark.parametrize("session", [False, True])
@pytest.mark.parametrize("paged", [False, True])
def test_retained_packet_cannot_substitute_for_a_gone_original_reader_root(monkeypatch, tmp_path, session, paged):
    fake = _fake_session(monkeypatch)
    request, slot, author, canonical, workspace = _prepared_request(tmp_path, session=session)
    if not paged:
        request.evidence.pop("__immutable_core_overflow__")
        acceptance_retrieving_work_order(request, [slot], session_root=str(workspace), data_root=author)
    retain_review_source(request, slot.slot_id, canonical)
    shutil.rmtree(author)
    llm = _EpisodeLLM(canonical, [])
    result = run_review_request(request, slots=[slot], drive_root=canonical, llm=llm)
    assert result.aggregate_signal != "PASS"
    assert "original_reader_root_unavailable" in result.actors[0]["error"]
    assert not llm.calls and all(not instance.start_requests for instance in fake.instances)
    delivery = request.slot_source_delivery[slot.slot_id]
    # Packet-only retention remains deliberately insufficient; the new closure
    # refuses earlier, and the packet helper independently gives the same gap.
    from ouroboros.review_execution import ReviewRouteUnavailable
    with pytest.raises(ReviewRouteUnavailable, match="original_reader_root_unavailable"):
        retain_review_source(request, slot.slot_id, canonical)
    assert delivery["reader_source_gap"] == "original_reader_root_unavailable"
    assert request.policy["native_data_root"] == str(author) and not author.exists()
    assert read_actor_source_bytes(canonical, request.task_id, delivery["custody_source"])


@pytest.mark.parametrize("target,code", [("", "session_route_unconfigured"), ("=model", "session_target_unparsable")])
def test_session_preparation_keeps_real_route_failure_per_row_without_fallback(monkeypatch, tmp_path, target, code):
    from ouroboros.review_execution import REVIEW_SESSION_ROUTE_ENV
    from ouroboros.review_dispatch import task_acceptance_row_refusal
    fake = _fake_session(monkeypatch)
    monkeypatch.setenv(REVIEW_SESSION_ROUTE_ENV, "off")
    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "off")
    request, good, author, canonical, workspace = _prepared_request(tmp_path)
    bad = replace(good, slot_id="bad-route", session_target=target)
    acceptance_retrieving_work_order(request, [bad, good], session_root=str(workspace), data_root=author)
    delivery = request.slot_source_delivery[bad.slot_id]
    assert delivery["status"] == "unavailable"
    assert delivery["preparation_error"]["code"] == code
    assert "compact first send exceeds" not in delivery["reason"]
    assert "first_send_ceiling" not in delivery  # no measured size claim
    assert task_acceptance_row_refusal(request, bad)["status"] == code
    assert request.slot_source_delivery[good.slot_id]["status"] == "paged"
    assert request.slot_source_delivery[good.slot_id]["preparation_measurement_basis"] == "synthetic_session_invocation_dispatch_rechecks"
    llm = _EpisodeLLM(canonical, [])
    result = run_review_request(request, slots=[bad], drive_root=canonical, llm=llm)
    assert result.aggregate_signal != "PASS" and code in result.actors[0]["error"]
    assert not llm.calls and all(not instance.start_requests for instance in fake.instances)


def test_native_preparation_exception_is_not_an_overflow_result(monkeypatch, tmp_path):
    """Unit fault injection at the measurer; live reader integration is above."""
    from ouroboros import review_native_episode
    from ouroboros.review_dispatch import task_acceptance_row_refusal
    request, slot, author, _canonical, workspace = _prepared_request(tmp_path, session=False)
    measure = review_native_episode.native_first_send_chars
    def broken(root, **kwargs):
        if kwargs["slot_id"] == "bad-prepare":
            raise ValueError("invalid inspection preparation")
        return measure(root, **kwargs)
    monkeypatch.setattr(review_native_episode, "native_first_send_chars", broken)
    bad = replace(slot, slot_id="bad-prepare")
    acceptance_retrieving_work_order(request, [bad, slot], session_root=str(workspace), data_root=author)
    error = request.slot_source_delivery[bad.slot_id]["preparation_error"]
    assert error == {"code": "review_source_preparation_failed", "type": "ValueError", "message": "invalid inspection preparation"}
    assert task_acceptance_row_refusal(request, bad)["status"] == "review_source_preparation_failed"
    assert request.slot_source_delivery[slot.slot_id]["status"] == "paged"
