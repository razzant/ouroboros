"""Recover an unpublished usable observation without rewriting published history."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros.artifacts import read_actor_source_bytes
from ouroboros.context_input_selection import historical_inputs_exhibit
from tests.test_historical_inputs import _input
from tests import test_main_authored_context as context_fixtures

main_loop = context_fixtures.main_loop


@pytest.mark.parametrize("failed_root", ["first", "second"])
def test_never_published_first_input_retries_exact_observation(main_loop, monkeypatch, tmp_path, failed_root):
    from ouroboros import artifacts

    f = main_loop
    f.ctx.budget_drive_root = tmp_path / "canonical"
    roots = [f.ctx.budget_drive_root, f.ctx.drive_root]
    failed = roots[failed_root == "second"]
    original_store = artifacts.store_actor_source_bytes
    attempted = []
    unavailable = True

    def store(root, task_id, **kwargs):
        if kwargs.get("source_id") == "historical-author-input":
            attempted.append((root, kwargs["data"]))
            if unavailable and root == failed:
                raise OSError("transient historical store failure")
        return original_store(root, task_id, **kwargs)

    monkeypatch.setattr(artifacts, "store_actor_source_bytes", store)
    _input(f, "ORIGINAL_USABLE_HISTORY", "Revise the feature list.")
    f.run([{"content": "Revised."}])
    before = deepcopy(f.ctx._historical_author_inputs["anchors"][0])
    assert before["status"] == "unavailable" and not before.get("source_ref")
    original_raw = attempted[0][1]
    observed = deepcopy(f.ctx._last_context_observation)

    unavailable = False
    # The retry must use the retained usable observation, not today's messages.
    f.ctx.messages = [{"role": "system", "content": "TODAYS_UNOBSERVED_INPUT"}]
    exhibit = historical_inputs_exhibit(f.ctx)
    assert exhibit["status"] == "captured"
    assert len(exhibit["anchors"]) == 1 and exhibit["latest_matches_first"]
    first = exhibit["anchors"][0]
    assert first["position"] == "first"
    assert first["view_sha256"] == before["view_sha256"]
    assert first["observed_view_revision"] == before["observed_view_revision"]
    assert first["physical_attempt_id"] == before["physical_attempt_id"]
    assert f.ctx._last_context_observation == observed
    for root in roots:
        assert read_actor_source_bytes(root, f.ctx.task_id, first["source_ref"]) == original_raw
    assert "ORIGINAL_USABLE_HISTORY" in original_raw.decode()
    assert "TODAYS_UNOBSERVED_INPUT" not in original_raw.decode()
    assert historical_inputs_exhibit(f.ctx) == exhibit


def test_never_published_first_input_is_not_replaced_by_a_later_observation(main_loop, monkeypatch):
    from ouroboros import artifacts
    from tests.test_main_authored_context import call

    f = main_loop
    original_store = artifacts.store_actor_source_bytes
    unavailable = True

    def store(root, task_id, **kwargs):
        if unavailable and kwargs.get("source_id") == "historical-author-input":
            raise OSError("transient historical store failure")
        return original_store(root, task_id, **kwargs)

    def next_response(_kwargs):
        nonlocal unavailable
        unavailable = False
        return {"content": "Revised after reading."}

    monkeypatch.setattr(artifacts, "store_actor_source_bytes", store)
    f.run([call("read_file", {"path": "evidence.txt"}, "read"), next_response])
    exhibit = historical_inputs_exhibit(f.ctx)
    first, latest = exhibit["anchors"]
    assert exhibit["status"] == "unavailable"
    assert first["position"] == "first" and not first.get("source_ref")
    assert first["status"] == "unavailable"
    assert latest["position"] == "latest" and latest["status"] == "captured"
    assert first["physical_attempt_id"] != latest["physical_attempt_id"]
    assert f.source in json.dumps(json.loads(read_actor_source_bytes(
        f.ctx.drive_root, f.ctx.task_id, latest["source_ref"]))).replace("\\n", "\n")


def test_never_published_input_does_not_retry_after_its_physical_source_is_lost(main_loop, monkeypatch):
    from ouroboros import artifacts

    f = main_loop
    original_store = artifacts.store_actor_source_bytes
    attempted = []

    def unavailable(root, task_id, **kwargs):
        if kwargs.get("source_id") == "historical-author-input":
            attempted.append(kwargs["data"])
            raise OSError("transient historical store failure")
        return original_store(root, task_id, **kwargs)

    monkeypatch.setattr(artifacts, "store_actor_source_bytes", unavailable)
    f.run([{"content": "Revised."}])
    first = deepcopy(f.ctx._historical_author_inputs["anchors"][0])
    captured = json.loads(attempted[0])
    assert captured["physical_source_status"] == "observed_projection"
    Path(captured["physical_source_identity"]["path"]).unlink()

    monkeypatch.setattr(artifacts, "store_actor_source_bytes", original_store)
    exhibit = historical_inputs_exhibit(f.ctx)
    assert exhibit["status"] == "unavailable"
    assert exhibit["anchors"] == [first]
    assert not first.get("source_ref")


def test_unreadable_historical_record_preserves_specific_first_gap(main_loop):
    from ouroboros.context_input_selection import capture_historical_inputs
    from ouroboros.task_results import task_result_path

    f = main_loop
    _input(f, "CURRENT_INPUT_AFTER_UNREADABLE_RECORD", "Continue the correction.")
    f.run([{"content": "Continued."}])
    # Reuse the real usable observation at the historical-capture boundary. A
    # corrupt result before model dispatch is refused by owner-pause admission.
    cold = SimpleNamespace(task_id=f.ctx.task_id, drive_root=f.ctx.drive_root,
                           task_attempt=1, _last_context_observation=deepcopy(f.ctx._last_context_observation))
    path = task_result_path(f.ctx.drive_root, f.ctx.task_id)
    path.write_text("{unfinished historical record", encoding="utf-8")
    capture_historical_inputs(cold)

    exhibit = historical_inputs_exhibit(cold)
    first, latest = exhibit["anchors"]
    assert exhibit["status"] == "unavailable"
    assert first == {"position": "first", "status": "unavailable", "reason": "record_unavailable"}
    assert latest["position"] == "latest" and latest["status"] == "captured"
    captured = json.loads(read_actor_source_bytes(f.ctx.drive_root, f.ctx.task_id, latest["source_ref"]))
    assert "CURRENT_INPUT_AFTER_UNREADABLE_RECORD" in json.dumps(captured["selected_messages"])
    assert path.read_text(encoding="utf-8") == "{unfinished historical record"
