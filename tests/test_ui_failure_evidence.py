"""Offline checks of failure custody and the census fixture's observation order."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_ui_smoke_inflight_indicator import _observe_census
from tests.ui_failure_evidence import FailureEvidence


def _fake_browser(monkeypatch, *, capture_error=False, close_error=None):
    actions = []
    monkeypatch.setattr("tests.ui_failure_evidence.importlib.metadata.version", lambda name: "fixture")

    def screenshot(**kwargs):
        actions.append("screenshot")
        if capture_error:
            raise OSError("synthetic capture error")
        Path(kwargs["path"]).write_bytes(b"fixture png")

    def stop(**kwargs):
        actions.append("trace_save" if kwargs else "trace_discard")
        if kwargs:
            Path(kwargs["path"]).write_bytes(b"fixture trace")

    def close():
        actions.append("close")
        if close_error:
            raise close_error

    page = SimpleNamespace(
        context=SimpleNamespace(tracing=SimpleNamespace(
            start=lambda **kwargs: actions.append("trace_start"), stop=stop)),
        add_init_script=lambda script: actions.append("observe"), on=lambda *args: None,
        evaluate=lambda *args: {"feed": "fixture"}, screenshot=screenshot,
        content=lambda: "<html>fixture</html>",
    )
    browser = SimpleNamespace(close=close, version="fixture")
    return page, browser, actions


@pytest.mark.parametrize("enabled", [False, True])
def test_success_retains_no_rich_bundle(tmp_path, monkeypatch, enabled):
    page, browser, actions = _fake_browser(monkeypatch)
    with FailureEvidence(page, browser, tmp_path if enabled else None, "case", "engine"):
        actions.append("scenario")
    assert actions == (["trace_start", "observe", "scenario", "trace_discard", "close"]
                       if enabled else ["scenario", "close"])
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(("close_error", "capture_error"), [
    (None, False),
    pytest.param(RuntimeError("synthetic close error"), True, id='close_error1-True'),
])
def test_first_failure_survives_capture_and_close(tmp_path, monkeypatch, capture_error, close_error):
    page, browser, actions = _fake_browser(monkeypatch, capture_error=capture_error,
                                            close_error=close_error)
    sentinel = AssertionError("original assertion")
    evidence = FailureEvidence(page, browser, tmp_path, "case", "engine")
    original_close = browser.close

    def close():
        evidence.checkpoint("callback_during_close", request_id=3)
        original_close()

    browser.close = close
    with pytest.raises(AssertionError) as caught:
        with evidence:
            evidence.checkpoint("read_to_latest:wait_for_live_edge")
            raise sentinel
    assert caught.value is sentinel
    assert actions.index("screenshot") < actions.index("trace_save") < actions.index("close")
    data = json.loads((evidence.bundle / "evidence.json").read_text(encoding="utf-8"))
    assert data["primary_exception"] == {"type": "AssertionError", "message": "original assertion"}
    assert data["failure_stage"] == "read_to_latest:wait_for_live_edge"
    assert [row["stage"] for row in data["timeline"]][-3:] == [
        "teardown_start", "callback_during_close", "teardown_end"]
    assert {row["operation"] for row in data["capture_errors"]} == (
        ({"screenshot"} if capture_error else set()) | ({"browser_close"} if close_error else set()))


def test_failed_writer_does_not_replace_assertion(tmp_path, monkeypatch, capsys):
    page, browser, _ = _fake_browser(monkeypatch)
    root = tmp_path / "not-a-directory"
    root.write_text("blocked", encoding="utf-8")
    with pytest.raises(AssertionError, match="original assertion"):
        with FailureEvidence(page, browser, root, "case", "engine"):
            raise AssertionError("original assertion")
    assert "diagnostics_incomplete" in capsys.readouterr().err


def test_cleanup_failure_is_failure_when_scenario_passed(tmp_path, monkeypatch):
    sentinel = RuntimeError("synthetic close error")
    page, browser, _ = _fake_browser(monkeypatch, close_error=sentinel)
    with pytest.raises(RuntimeError) as caught:
        with FailureEvidence(page, browser, tmp_path, "case", "engine"):
            pass
    assert caught.value is sentinel
    data = json.loads(next(tmp_path.rglob("evidence.json")).read_text(encoding="utf-8"))
    assert data["failure_stage"] == "browser_close"


@pytest.mark.parametrize("cross_generation", [False, True])
def test_census_records_before_fetch_and_actual_response_rows(tmp_path, cross_generation):
    state = {"reads": 0, "rows": [], "fixture_generation": 0}
    evidence = FailureEvidence(None, None, tmp_path, "fixture", "engine")
    delivered = []

    def fetch():
        if cross_generation:
            state.update(rows=[{"phase": "thinking", "chat_id": 1}], fixture_generation=1)
        return SimpleNamespace(status=200, json=lambda: {"supervisor_ready": True})

    route = SimpleNamespace(request=SimpleNamespace(url="http://fixture/api/state?chat_id=1"),
                            fetch=fetch, fulfill=lambda **kwargs: delivered.append(json.loads(kwargs["body"])))
    _observe_census(route, state, evidence)
    # Mutating the fixture later must not rewrite earlier journal rows.
    state["rows"].append({"phase": "later"})
    start = evidence.timeline[0]
    built = next(row for row in evidence.timeline if row["stage"] == "response_built")
    assert start["fixture_generation"] == 0 and start["rows"] == []
    assert built["fixture_generation"] == int(cross_generation)
    assert built["rows"] == delivered[0]["active_chat_activities"]
    assert bool(built["rows"]) == cross_generation
    assert {row["request_id"] for row in evidence.timeline} == {1}
    assert [row["stage"] for row in evidence.timeline] == [
        "intercepted", "fetch_returned", "response_built", "fulfill_started", "fulfilled"]
    assert evidence.details["pending_request_ids"] == []
    evidence._json("census.json", evidence.timeline)
    assert json.loads((evidence.bundle / "census.json").read_text(encoding="utf-8")) == evidence.timeline


@pytest.mark.parametrize("operation", ["fetch", "fulfill"])
def test_census_abort_preserves_original_exception(tmp_path, operation):
    evidence = FailureEvidence(None, None, tmp_path, "fixture", "engine")
    state = {"reads": 0, "rows": [], "fixture_generation": 0}
    sentinel = RuntimeError("synthetic route failure")

    def fail(**kwargs):
        raise sentinel

    route = SimpleNamespace(request=SimpleNamespace(url="http://fixture/api/state"),
                            fetch=lambda: SimpleNamespace(status=200, json=lambda: {}), fulfill=fail)
    if operation == "fetch":
        route.fetch = fail
    with pytest.raises(RuntimeError) as caught:
        _observe_census(route, state, evidence)
    assert caught.value is sentinel
    assert evidence.timeline[-1]["stage"] == "aborted"
    assert evidence.timeline[-1]["operation"] == operation
    assert evidence.details["requests"] == {1: "aborted"}
    assert evidence.details["pending_request_ids"] == []
