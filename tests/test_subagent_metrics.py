"""Observed row windows, correction identity, and archive-cursor recovery."""
import json
from unittest.mock import Mock

import pytest

from ouroboros import subagent_metrics as metrics
from ouroboros.utils import append_jsonl


def event(producer="one", **fields):
    return {"row_id": "worker", "row_identity": {"kind": "api_model", "target_id": "test/model"},
            "producer_id": producer, "purpose": "review", "ts": "2026-10-22T00:00:00Z",
            "dispatched": True, "terminal": True, "answer_received": True, "usable_verdict": True,
            "interrupted": False, "duration_sec": 2, "source_refs": [{"call_id": producer}], **fields}


def row(root):
    return metrics.read(root, "worker")["rows"][0]


def test_pending_received_malformed_usable_silence_and_interruption_are_separate(tmp_path):
    values = [event("pending", terminal=False, answer_received=None, usable_verdict=False, duration_sec=None),
              event("malformed", usable_verdict=False), event("valid"),
              event("silent", answer_received=False, usable_verdict=False),
              event("interrupted", answer_received=False, usable_verdict=False, interrupted=True),
              event("unsent", dispatched=False, answer_received=False, usable_verdict=False, duration_sec=None)]
    for value in values:
        metrics.publish(tmp_path, value)
    facts = row(tmp_path)
    assert {key: facts[key] for key in metrics.COUNTS} == {
        "dispatched": 5, "terminal": 5, "answer_received": 2, "usable_verdict": 1,
        "no_answer": 1, "pending": 1, "interrupted": 1}
    assert facts["denominators"]["non_interrupted_terminal_dispatches"] == 3
    assert facts["duration_samples"] == {"count": 4, "sum_sec": 8}
    assert facts["quota"]["state"] == "unknown"


def test_replay_and_late_answer_replace_the_same_producer(tmp_path):
    first = event(terminal=False, answer_received=None, usable_verdict=False, duration_sec=None)
    metrics.publish(tmp_path, first)
    assert row(tmp_path)["pending"] == 1
    metrics.publish(tmp_path, first)
    assert row(tmp_path)["observations"] == 1
    metrics.publish(tmp_path, event(duration_sec=4))
    metrics.publish(tmp_path, event(duration_sec=4))
    metrics.publish(tmp_path, first)  # an old pending replay cannot erase settlement
    facts = row(tmp_path)
    assert (facts["pending"], facts["usable_verdict"], facts["observations"]) == (0, 1, 1)
    assert facts["duration_samples"] == {"count": 1, "sum_sec": 4}


def test_late_answer_corrects_silence_without_a_second_dispatch(tmp_path):
    metrics.publish(tmp_path, event(answer_received=False, usable_verdict=False))
    assert row(tmp_path)["no_answer"] == 1
    metrics.publish(tmp_path, event())
    facts = row(tmp_path)
    assert (facts["no_answer"], facts["dispatched"], facts["usable_verdict"]) == (0, 1, 1)


def test_failed_append_keeps_known_gap_until_that_producer_is_replayed(tmp_path, monkeypatch):
    metrics.publish(tmp_path, event())
    with monkeypatch.context() as broken:
        broken.setattr(metrics, "append_jsonl", lambda *a, **kw: False)
        metrics.publish(tmp_path, event("missing"))
    metrics.publish(tmp_path, event("another"))
    assert row(tmp_path)["coverage"]["state"] == "partial"
    assert row(tmp_path)["coverage"]["missing_publications"] == 1
    metrics.publish(tmp_path, event("missing"))
    assert row(tmp_path)["coverage"]["state"] == "observed_window"
    assert row(tmp_path)["dispatched"] == 3


def test_purpose_and_execution_identity_partition_same_saved_row(tmp_path):
    metrics.publish(tmp_path, event("review"))
    metrics.publish(tmp_path, event("plan", purpose="plan"))
    metrics.publish(tmp_path, event("exec", purpose="execution"))
    metrics.publish(tmp_path, event("changed", row_identity={"kind": "api_model", "target_id": "other/model"}))
    facts = metrics.read(tmp_path, "worker")
    assert len(facts["rows"]) == 4
    assert len(metrics.read(tmp_path, "worker", row_identity=event()["row_identity"], purpose="review")["rows"]) == 1
    assert all(value["observations"] == 1 for value in facts["rows"])


def test_same_producer_cannot_change_its_execution_identity(tmp_path):
    metrics.publish(tmp_path, event())
    metrics.publish(tmp_path, event(row_identity={"kind": "api_model", "target_id": "different"}))
    facts = metrics.read(tmp_path, "worker")
    assert facts["coverage"]["state"] == "unknown"
    assert len(facts["rows"]) == 1
    assert facts["rows"][0]["row_identity"] == event()["row_identity"]


def test_cursor_recovers_missed_publication_rotation_and_passive_reads(tmp_path, monkeypatch):
    from ouroboros import swarm_rollup
    metrics.publish(tmp_path, event())
    archive = tmp_path / "archive/events_20261022T000000.jsonl"
    archive.parent.mkdir()
    (tmp_path / "logs/events.jsonl").rename(archive)
    append_jsonl(tmp_path / "logs/events.jsonl", {"type": metrics.EVENT, **event("two")})
    decode = Mock(wraps=swarm_rollup.json.loads)
    monkeypatch.setattr(swarm_rollup, "json", type("Decoder", (), {"loads": staticmethod(decode)}))
    metrics.catch_up(tmp_path)
    assert decode.call_count == 1 and row(tmp_path)["dispatched"] == 2
    decode.reset_mock()
    metrics.catch_up(tmp_path)
    assert decode.call_count == 0
    monkeypatch.setattr(swarm_rollup, "fold_event_suffix", lambda *args: pytest.fail("inspection scanned history"))
    assert row(tmp_path)["dispatched"] == 2


def test_torn_shortened_and_malformed_sources_do_not_fabricate_coverage(tmp_path):
    metrics.publish(tmp_path, event())
    path = tmp_path / "logs/events.jsonl"
    with path.open("a") as stream:
        stream.write(json.dumps({"type": metrics.EVENT, **event("two")}))
    metrics.catch_up(tmp_path)
    assert row(tmp_path)["dispatched"] == 1
    assert row(tmp_path)["coverage"]["stream"] == "incomplete"
    with path.open("a") as stream:
        stream.write("\n")
    metrics.catch_up(tmp_path)
    assert row(tmp_path)["coverage"]["stream"] == "complete"
    assert row(tmp_path)["dispatched"] == 2
    with path.open("a") as stream:
        stream.write(json.dumps({"type": metrics.EVENT, **event("three")}) + "\nbroken\n")
    with pytest.raises(ValueError):
        metrics.catch_up(tmp_path)
    assert row(tmp_path)["dispatched"] == 2  # whole suffix rolled back
    assert row(tmp_path)["coverage"]["state"] == "unknown"
    path.write_text("{}\n")
    with pytest.raises(ValueError, match="shortened"):
        metrics.catch_up(tmp_path)
    assert row(tmp_path)["dispatched"] == 2
    assert "rate" not in json.dumps(row(tmp_path))
