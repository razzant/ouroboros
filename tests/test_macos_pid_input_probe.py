"""Public PID experiment, separate from shipping routes. Only the build and analysis tests run by default.

Opt-in desktop cases collect measurements, including negative results; passing collection is NOT a
qualification of background input. Run through scripts/safe_test.py with the intended product
interpreter, OUROBOROS_MACOS_DESKTOP_QA=1 and OUROBOROS_TEST_EVIDENCE_ROOT set outside the checkout.
This launches owned windows and foreground decoy typing: coordinate a quiet interval first.
Keep the actual interpreter/permission evidence; a helper run is not installed-tool qualification.
Inspect measurement.json and receiver logs, not just pytest's exit. Each arm uses a fresh target;
no activation, primer, cursor restoration or input fallback is performed on that target.
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
import uuid
from collections import Counter

import pytest

from tests.test_macos_computer_use_native import REPO, Owned, _build, desktop_only

PROBE = REPO / "tests/fixtures/macos_pid_input_probe.swift"
NATIVE = REPO / "skills/unix_computer_use/macos_native.py"
DECOY = "ordinaryforegroundletters" * 8  # deliberately no 'k'


def build_probe(directory):
    swiftc = subprocess.run(["xcrun", "--find", "swiftc"], capture_output=True, text=True, check=False)
    if swiftc.returncode != 0:
        pytest.skip("xcrun swiftc is not installed: " + swiftc.stderr.strip())
    binary = directory / "PIDInputProbe"
    result = subprocess.run(["xcrun", "swiftc", "-swift-version", "5", "-Onone", "-o", str(binary), str(PROBE)],
                            cwd=directory, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    return binary


@pytest.mark.skipif(sys.platform != "darwin" or not shutil.which("xcrun"), reason="requires macOS SDK")
def test_pid_probe_compiles_without_running(tmp_path):
    assert build_probe(tmp_path).stat().st_size > 0


def measurements(target_rows, front_rows, before, after, intended_window):
    """Receiver logs decide effects; a sender's post count is never treated as receipt or action count."""
    t0, f0 = before
    t1, f1 = after
    start, end = min(t0["t"], f0["t"]), max(t1["t"], f1["t"])
    target = [r for r in target_rows if start < r["t"] <= end]
    front = [r for r in front_rows if start < r["t"] <= end]
    raw = [r for r in target if r["event"] == "raw_input"]
    tagged = [r for r in raw if r.get("tag") == 42001]
    return {
        "interpretation": "measurement_only",
        "target_raw_event_count": len(raw),
        "target_key_downs": [r for r in raw if r["type"] == 10],
        "front_raw_event_count": sum(r["event"] == "raw_input" for r in front),
        "target_probe_event_count": len(tagged),
        "target_probe_events_by_type": dict(Counter(str(r["type"]) for r in tagged)),
        "target_probe_events_by_window": dict(Counter(str(r["window_number"]) for r in tagged)),
        "probe_events_in_intended_window": sum(r["window_number"] == intended_window for r in tagged),
        "primary_commit_delta": t1["count"] - t0["count"],
        "secondary_commit_delta": t1["secondary"]["count"] - t0["secondary"]["count"],
        "primary_field_before_after": [t0["field"], t1["field"]],
        "secondary_field_before_after": [t0["secondary"]["field"], t1["secondary"]["field"]],
        "target_received_decoy": any(r.get("tag") == 42002 for r in raw),
        "front_received_probe": any(r["event"] == "raw_input" and r.get("tag") == 42001 for r in front),
        "decoy_text_exact": f1["text"] == DECOY,
        "front_pid_before_after": [f0["front_pid"], f1["front_pid"]],
        "key_before_after": [[t0["key_window"], t0["secondary"]["key_window"], f0["key_window"]],
                             [t1["key_window"], t1["secondary"]["key_window"], f1["key_window"]]],
        "responders_before_after": [[t0["first_responder"], t0["secondary"]["first_responder"], f0["first_responder"]],
                                    [t1["first_responder"], t1["secondary"]["first_responder"], f1["first_responder"]]],
        "focus_events": [r for r in target + front if r["event"] in (
            "app_became_active", "app_resigned_active", "window_became_key", "window_resigned_key", "workspace_activated")],
        "cursor_before_after": [f0["mouse"], f1["mouse"]],
        "pointer_events": [r for r in front if r["event"] == "pointer_event"],
        "pasteboard_before_after": [f0["pasteboard_change_count"], f1["pasteboard_change_count"]],
    }


def test_receipt_counts_effects_and_keyboard_theft_are_independent():
    def state(t):
        return {"t": t, "count": 0, "field": "", "key_window": False, "first_responder": "none",
                "secondary": {"count": 0, "field": "", "key_window": False, "first_responder": "none"},
                "text": DECOY, "front_pid": 100, "mouse": [1, 2], "pasteboard_change_count": 9}
    before, after = state(1), state(4)
    rows = [{"event": "raw_input", "t": 2, "tag": 42001, "type": 1, "window_number": 42},
            {"event": "raw_input", "t": 2.1, "tag": 42001, "type": 2, "window_number": 42},
            {"event": "raw_input", "t": 3, "tag": 42002, "type": 10, "window_number": 42}]
    out = measurements(rows, [], (before, before), (after, after), 43)
    assert out["target_probe_event_count"] == 2 and out["probe_events_in_intended_window"] == 0
    assert out["secondary_commit_delta"] == out["primary_commit_delta"] == 0
    assert out["target_received_decoy"] and out["front_pid_before_after"] == [100, 100]
    after["secondary"]["count"] = 1
    out = measurements([], [], (before, before), (after, after), 43)
    assert out["secondary_commit_delta"] == 1 and out["target_raw_event_count"] == 0  # AX effects need no input


@desktop_only
@pytest.mark.parametrize("route", ["ax", "pid-click", "pid-key"])
def test_measure_second_window(route, tmp_path):
    """Fresh target/decoy per route. No warm-up click, target activation, focus setter, or replay."""
    binary, probe = _build(tmp_path), build_probe(tmp_path)
    owned = []
    try:
        target = Owned(binary, "target", tmp_path)
        owned.append(target)
        assert target.wait_for("ready")["application_class"] == "QAApplication"
        target.send("second")
        target.wait_for("second_ready")
        front = Owned(binary, "front", tmp_path)
        owned.append(front)
        assert front.wait_for("ready")["application_class"] == "QAApplication"
        front.wait_for("app_became_active")
        assert front.wait_for("pointer_monitor")["installed"], "pointer observation unavailable"
        t0, f0 = target.mark("before"), front.mark("before")
        assert f0["app_active"] and f0["key_window"] and f0["front_pid"] == front.pid, "decoy did not acquire the front at setup"
        assert not t0["app_active"] and not t0["key_window"] and not t0["secondary"]["key_window"], "target is not cold/background"
        front.send(f"type 40 {DECOY}")
        started = front.wait_for("typing_started", "typing_unsupported", after=f0["t"])
        if started["event"] == "typing_unsupported":
            pytest.skip("post-event grant missing; no grant requested and no measurement made")
        front.wait_for("front_text", after=started["t"])
        assert any(r["event"] == "raw_input" for r in front.rows()), "raw receiver instrumentation unavailable"
        selected = t0["secondary"]
        if route == "ax":
            options = {"app": f"pid:{target.pid}", "do": "press", "observe": True,
                       "select": {"window": str(selected["window_id"]), "identifier": "qa.second.commit"}}
            command = [sys.executable, "-I", "-B", str(NATIVE), "element", json.dumps(options)]
        else:
            command = [str(probe), str(target.pid), str(selected["window_id"])]
            if route == "pid-click":
                x, y, w, h = selected["commit_frame"]
                command += ["click", str(x + w / 2), str(y + h / 2)]
            else:
                command += ["key"]
        dispatch_before = front.mark("dispatch-before")
        began = time.monotonic()
        sent = subprocess.run(command, text=True, capture_output=True, timeout=25)
        ended = time.monotonic()
        dispatch_after = front.mark("dispatch-after")
        receipt = {"route": route, "argv": command, "returncode": sent.returncode, "stdout": sent.stdout,
                   "stderr": sent.stderr, "began": began, "ended": ended, "helper_ms": (ended - began) * 1000}
        (tmp_path / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        front.wait_for("typing_finished", "typing_stopped", after=started["t"], timeout=20)
        time.sleep(0.3)  # allow late delivery to appear in the independent logs
        t1, f1 = target.mark("after"), front.mark("after")
        result = measurements(target.rows(), front.rows(), (t0, f0), (t1, f1), selected["window_id"])
        landed = [r["t"] for r in front.rows() if r["event"] == "front_text" and r["t"] > started["t"]]
        result.update({"route": route, "dispatch_receiver_clock": [dispatch_before["t"], dispatch_after["t"]],
                       "decoy_spanned_dispatch": bool(landed) and landed[0] < dispatch_before["t"] and dispatch_after["t"] < landed[-1],
                       "helper_returncode": sent.returncode, "helper_ms": receipt["helper_ms"]})
        (tmp_path / "measurement.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, sort_keys=True))
    finally:
        for process in reversed(owned):
            process.stop()
        evidence = os.environ.get("OUROBOROS_TEST_EVIDENCE_ROOT")
        if evidence:
            destination = pathlib.Path(evidence) / "macos-pid-probe" / f"{route}-{uuid.uuid4().hex}"
            destination.mkdir(parents=True)
            for path in tmp_path.iterdir():
                if path.suffix in (".json", ".jsonl", ".stderr"):
                    shutil.copy2(path, destination / path.name)
