"""Required result review sees what a desktop task did (#1919).

The chain is the production one: the handler ``unix_computer_use`` registers runs behind the real
extension dispatcher (``_extension_completion``); the loop records the call with its typed status,
code and retained sources (``_execute_single_tool``, ``process_tool_results``); the result review
reads that trace: ``produced_material`` -> the frozen subject -> ``result_review_at_delivery``, whose
paid seats are the offline golden seam. Faked are only the skill's OS boundary (``subprocess.run``
and ``urlopen``, while its handler runs) and the paid safety check; nothing here reaches a desktop,
a Mac over SSH, a guest VM or a model.
"""
from __future__ import annotations

import itertools
import json
import subprocess
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

import ouroboros.review_substrate as substrate
from ouroboros import artifacts, review_ledger
from ouroboros import loop_tool_execution as execution
from ouroboros.extension_surface_names import extension_surface_name
from ouroboros.tools import extension_dispatch
from ouroboros.tools import review_result as rr
from skills.unix_computer_use import macos_native as native
from tests._review_session_route_shared import _owned_gateway_uses_each_test_transport  # noqa: F401
from tests._tool_result_delivery_shared import measured_fit
from tests.test_result_review import CRITICAL, _floor_ctx, _seam
from tests.test_unix_computer_use_effects import API, TIMEOUT, Guest, Transport, fail_at, load_plugin
from tests.test_unix_computer_use_element_act import LONG_BEFORE, LONG_LABEL, LONG_TEXT, NORMALIZED, FakeAX, long_field
from tests.test_unix_computer_use_targeting import KEYNOTE, app_dict

WARNING = ("⚠️ SAFETY_WARNING: The Safety Supervisor flagged this action as suspicious.\nReason: fixture\n"
           "The command was allowed, but consider whether this is the right approach.")
CALL_IDS = itertools.count(1)


@pytest.fixture(autouse=True)
def _live_extension(monkeypatch):
    """The host's liveness registry admits the fixture skill; everything after it is the real dispatcher."""
    monkeypatch.setattr("ouroboros.extension_loader.is_extension_live", lambda *_a, **_kw: True)


def trace_of(ctx) -> dict:
    if not isinstance(getattr(ctx, "_execution_trace", None), dict):
        ctx._execution_trace = {"tool_calls": []}
    return ctx._execution_trace


def invoke(ctx, skill, short, handler, args, *, warning="", allowed=True, fit=None, boundary=()):
    """One model tool call as the loop runs it; returns the trace row the result review reads.
    ``allowed``/``warning`` is the paid safety check's verdict on the call."""
    name = extension_surface_name(skill, short)

    def bound(**kwargs):
        with mock.patch("time.sleep", lambda _seconds: None):
            for patch in boundary:  # the skill's OS/network boundary, faked only while its handler runs
                patch.start()
            try:
                return handler(**kwargs)
            finally:
                for patch in reversed(boundary):
                    patch.stop()

    descriptor = {"name": name, "skill": skill, "handler": bound, "wants_ctx": False, "timeout_sec": 20,
                  "extension_generation": "fixture-generation"}
    registry = SimpleNamespace(_ctx=ctx, CODE_TOOLS=frozenset(), execute_result=lambda fn, call_args: (
        extension_dispatch._dispatch_extension_tool_result(ctx, fn, descriptor, call_args)))
    call = {"id": f"call-{next(CALL_IDS)}", "function": {"name": name, "arguments": json.dumps(args)}}
    with mock.patch("ouroboros.safety.check_safety", lambda *_a, **_kw: (allowed, warning)):
        row = execution._execute_single_tool(registry, call, Path(ctx.drive_root) / "logs", ctx.task_id)
    trace = trace_of(ctx)
    execution.process_tool_results([row], [], trace, lambda *_a, **_kw: None, registry, fit_candidate=fit)
    return trace["tool_calls"][-1]


class Desktop:
    """The registered handlers of ``unix_computer_use`` on one faked machine."""

    def __init__(self, ctx, tmp_path, monkeypatch, label, platform, session="native", tools=(), answer=None,
                 guest=None):
        self.module, self.ctx = load_plugin(), ctx
        self.transport, self.guest = Transport(answer), guest
        monkeypatch.setattr(self.module, "_platform", lambda: platform)
        monkeypatch.setattr(self.module, "_session_type", lambda: session)
        monkeypatch.setattr(self.module, "_which", lambda name: f"/usr/bin/{name}" if name in tools else "")
        monkeypatch.setattr(self.module._ComputerUse, "_x11_keysym", staticmethod(lambda _name: None))
        api = API(tmp_path / f"skill-state-{label}")
        self.module.register(api)
        self.handlers = api.tools

    def call(self, short, *, warning="", allowed=True, fit=None, **args):
        boundary = [mock.patch.object(subprocess, "run", self.transport)]
        if self.guest is not None:
            boundary.append(mock.patch.object(urllib.request, "urlopen", self.guest))
        return invoke(self.ctx, "unix_computer_use", short, self.handlers[short], args, warning=warning,
                      allowed=allowed, fit=fit, boundary=boundary)


MAC_TOOLS = ("cliclick", "osascript", "screencapture")


def mac(ctx, tmp_path, monkeypatch, answer=None):
    return Desktop(ctx, tmp_path, monkeypatch, "mac", "macos", tools=MAC_TOOLS, answer=answer)


def ssh(ctx, tmp_path, monkeypatch, answer=None):
    desktop = Desktop(ctx, tmp_path, monkeypatch, "ssh", "linux", "x11", answer=answer)
    configured = desktop.call("add_connection", name="mac", backend="ssh_macos", host="mac.invalid", user="owner",
                              activate=True)
    assert configured["is_error"] is False and '"observed_effect"' not in configured["result"]
    return desktop


def subject_text(ctx) -> str:
    """The exact subject the bound result review's seats read."""
    record_id = trace_of(ctx)["review_runs"][-1]["record_id"]
    digest = review_ledger.load_record(review_ledger.ledger_root(ctx), record_id)["subject"]["diff_sha"]
    text = rr._retained_subject_text(ctx, digest)
    assert text, "the reviewed subject is retained"
    return text


def effect_gaps(produced) -> list:
    return [issue for issue in produced["issues"] if "effect receipt" in issue]


# --- What a desktop input produces ----------------------------------------------------------------

@pytest.mark.parametrize("backend", ["mac", "x11", "wayland", "ssh", "osworld"])
def test_required_reviews_a_root_result_whose_only_material_is_a_desktop_input(tmp_path, monkeypatch, backend):
    ctx = _floor_ctx(tmp_path, monkeypatch)
    sends: list[dict] = []
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam(sends, {"t1": CRITICAL}))
    if backend == "mac":
        row = mac(ctx, tmp_path, monkeypatch).call("click", x=10, y=20, raw=True)
    elif backend == "x11":
        row = Desktop(ctx, tmp_path, monkeypatch, "x11", "linux", "x11", tools=("xdotool",)).call("key", keys="ctrl+s")
    elif backend == "wayland":
        desktop = Desktop(ctx, tmp_path, monkeypatch, "wayland", "linux", "wayland", tools=("ydotool",))
        row = desktop.call("left_click_drag", start_x=1, start_y=2, end_x=30, end_y=40, raw=True)
    elif backend == "ssh":
        row = ssh(ctx, tmp_path, monkeypatch).call("type_text", text="hello")
    else:
        desktop = Desktop(ctx, tmp_path, monkeypatch, "vm", "linux", "x11", guest=Guest([0]))
        desktop.call("add_connection", name="vm", backend="osworld_http", target="http://guest.invalid:5000",
                     activate=True)
        row = desktop.call("key", keys="Left Left")  # a sequence: before #1919 it dropped its steps' marks
    assert (row["is_error"], row["tool_result_code"]) == (False, "OK"), row

    produced = rr.produced_material(ctx)
    assert [effect["tool"] for effect in produced["effects"]] == [row["tool"]] and not effect_gaps(produced)
    assert json.loads(produced["effects"][0]["receipt"])["observed_effect"] is True

    shown: list[str] = []
    assert rr.result_review_at_delivery(ctx, "Done on the desktop.", trace_of(ctx), shown.append) is True
    assert len(sends) == 3 and "FAIL" in shown[0], "Required: the host reviewed it and shows the findings first"
    subject = subject_text(ctx)
    assert f"- effect declared by {row['tool']}: " in subject and '"observed_effect": true' in subject
    assert "tool producer's reports of execution, not proof of a UI change or goal success" in subject


@pytest.mark.parametrize("case", ["wayland_drag", "x11_hold", "mac_keys", "ssh_keys"])
def test_a_reported_failure_keeps_the_receipt_of_what_ran(tmp_path, monkeypatch, case):
    ctx = _floor_ctx(tmp_path, monkeypatch)
    sends: list[dict] = []
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam(sends))
    if case == "wayland_drag":
        desktop = Desktop(ctx, tmp_path, monkeypatch, "wayland", "linux", "wayland", tools=("ydotool",),
                          answer=fail_at(3, outcome=TIMEOUT))
        row, done = desktop.call("left_click_drag", start_x=1, start_y=2, end_x=3, end_y=4, raw=True), 3
    elif case == "x11_hold":
        desktop = Desktop(ctx, tmp_path, monkeypatch, "x11", "linux", "x11", tools=("xdotool",), answer=fail_at(1))
        row, done = desktop.call("hold_key", keys="shift", duration_ms=300), 1
    elif case == "mac_keys":
        row, done = mac(ctx, tmp_path, monkeypatch, answer=fail_at(1)).call("key", keys="cmd+a cmd+c"), 1
    else:
        desktop = ssh(ctx, tmp_path, monkeypatch)
        desktop.transport.answer = lambda argv, index: 255 if index == 2 else 0  # the third remote step fails
        row, done = desktop.call("key", keys="down down return"), 2
    assert (row["is_error"], row["tool_result_code"]) == (True, "TOOL_REPORTED_FAILURE"), row

    [effect] = rr.produced_material(ctx)["effects"]
    receipt = json.loads(effect["receipt"])
    assert receipt["ok"] is False and receipt["observed_effect"] is True and receipt["steps_done"] == done, receipt
    assert receipt["error"], "the failure's own diagnostics stay in the receipt"

    assert rr.result_review_at_delivery(ctx, "Started the shortcut.", trace_of(ctx), lambda _text: None) is False
    assert len(sends) == 3, "the host reviewed the partial run under Required"
    subject = subject_text(ctx)
    assert effect["receipt"] in subject and '"ok": false' in subject and '"steps_done"' in subject


def test_reads_refusals_failures_and_timeouts_produce_nothing_and_force_no_review(tmp_path, monkeypatch):
    ctx = _floor_ctx(tmp_path, monkeypatch)
    sends: list[dict] = []
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam(sends))

    def answer(argv, _index):
        if argv[0] == "osascript":
            return 0, ("0, 0, 1440, 900" if "bounds" in argv[-1] else "Finder, Safari")
        if argv[:2] == ["cliclick", "p"]:
            return 0, "12,34"
        return {"t:boom": 1, "kp:return": TIMEOUT}.get(argv[-1], 0)
    desktop = mac(ctx, tmp_path, monkeypatch, answer=answer)
    rows = [desktop.call("screenshot"), desktop.call("cursor_position"), desktop.call("window_list"),
            desktop.call("ax_tree"), desktop.call("capabilities"), desktop.call("wait", ms=5),
            desktop.call("scroll", clicks=3), desktop.call("click", x=1, y=1, button="middle"),
            desktop.call("type_text", text="boom"), desktop.call("key", keys="return"),
            desktop.call("key", keys="return return")]
    remote = ssh(ctx, tmp_path, monkeypatch)
    rows.append(remote.call("remote_exec", command="touch /tmp/written-by-a-shell"))
    assert json.loads(rows[-1]["result"])["ok"] is True, "the remote shell ran; it still declares no effect"

    calls_before = len(desktop.transport.calls)
    blocked = desktop.call("click", allowed=False, warning="⚠️ SAFETY_VIOLATION: blocked.", x=1, y=1, raw=True)
    assert blocked["tool_result_code"] == "SAFETY_VIOLATION" and len(desktop.transport.calls) == calls_before

    def raises(**_kwargs):
        raise RuntimeError("the handler crashed after an unknown amount of work")
    crashed = invoke(ctx, "fixture_skill", "act", raises, {})
    assert crashed["tool_result_code"] == "EXTENSION_ERROR"
    # A host-owned outcome is never a producer's receipt, whatever text it carries.
    trace_of(ctx)["tool_calls"].append({
        "tool": extension_surface_name("unix_computer_use", "click"), "tool_call_id": "host-timeout",
        "result": json.dumps({"ok": True, "observed_effect": True}), "is_error": True, "status": "timeout",
        "tool_result_status": "timeout", "tool_result_code": "EXTENSION_TIMEOUT"})

    produced = rr.produced_material(ctx)
    assert produced["effects"] == [] and not effect_gaps(produced), produced
    assert [row["is_error"] for row in rows[6:11]] == [True] * 5, "refusals, a failed and a timed-out process"
    assert rr.result_review_at_delivery(ctx, "I looked at the screen.", trace_of(ctx), lambda _text: None) is False
    assert sends == [] and not trace_of(ctx).get("review_runs")
    assert "Effect receipts" not in rr.answer_text(reply="Observed.", produced=produced, verification="")


@pytest.mark.parametrize("value", [False, "true", 1, None, {"executed": True}])
@pytest.mark.parametrize("ok", [True, False])
def test_only_a_literal_true_is_an_effect(tmp_path, monkeypatch, value, ok):
    ctx = _floor_ctx(tmp_path, monkeypatch)
    payload = json.dumps({"ok": ok, "observed_effect": value, "error": "" if ok else "partly done"})
    row = invoke(ctx, "fixture_skill", "act", lambda: payload, {})
    assert row["tool_result_code"] == ("OK" if ok else "TOOL_REPORTED_FAILURE")
    produced = rr.produced_material(ctx)
    assert produced["effects"] == [] and not effect_gaps(produced)


def test_a_host_note_never_hides_a_receipt_and_stays_beside_it(tmp_path, monkeypatch):
    ctx = _floor_ctx(tmp_path, monkeypatch)
    sends: list[dict] = []
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam(sends))
    clicked = mac(ctx, tmp_path, monkeypatch).call("click", warning=WARNING, x=1, y=2, raw=True)
    partial = ssh(ctx, tmp_path, monkeypatch, answer=fail_at(1)).call("key", warning=WARNING, keys="down return")
    assert (clicked["tool_result_code"], partial["tool_result_code"]) == ("SAFETY_WARNING", "TOOL_REPORTED_FAILURE")
    for row in (clicked, partial):
        assert row["host_annotations"] == [WARNING] and row["producer_source_ref"]
        with pytest.raises(ValueError):
            json.loads(row["result"])  # what the model saw: the payload, the note and the source locator

    effects = rr.produced_material(ctx)["effects"]
    assert [effect["tool"] for effect in effects] == [clicked["tool"], partial["tool"]]
    for effect in effects:
        assert WARNING not in effect["receipt"] and json.loads(effect["receipt"])["observed_effect"] is True
        assert effect["host_notes"] == [WARNING]
    assert rr.result_review_at_delivery(ctx, "Saved it.", trace_of(ctx), lambda _text: None) is False
    subject = subject_text(ctx)
    assert subject.count(json.dumps([WARNING], ensure_ascii=False)) == 2, "each receipt keeps its host note"


def test_a_partial_view_still_reads_the_complete_receipt(tmp_path, monkeypatch):
    ctx = _floor_ctx(tmp_path, monkeypatch)
    desktop = mac(ctx, tmp_path, monkeypatch)
    text = "A long paragraph typed into the editor. " * 900
    plain = desktop.call("type_text", fit=measured_fit(window=4_000, reserve=1_000), text=text)
    noted = desktop.call("type_text", warning=WARNING, fit=measured_fit(window=4_000, reserve=1_000), text=text + "!")
    for row in (plain, noted):
        assert row["result_partial"] is True and row["result_source_ref"], "the model saw a head+tail view"
    assert noted["producer_source_ref"]["path"] != noted["result_source_ref"]["path"]

    effects = rr.produced_material(ctx)["effects"]
    assert [json.loads(effect["receipt"])["cmd"] for effect in effects] == [
        ["cliclick", f"t:{text}"], ["cliclick", f"t:{text}!"]]
    assert "host_notes" not in effects[0] and effects[1]["host_notes"] == [WARNING]


def test_an_element_act_receipt_keeps_its_complete_label_and_values_in_the_retained_source(tmp_path, monkeypatch):
    """The helper's own ASCII line (``macos_native.main``) is the producer's payload; the host's retained producer source
    is what the result review reads, whole, while the model's view of the same call is only a head and a tail."""
    ctx = _floor_ctx(tmp_path, monkeypatch)
    desktop = mac(ctx, tmp_path, monkeypatch)
    nodes = long_field()
    ax = FakeAX(nodes, windows=["win"], focused="win", on_set=lambda el, v: nodes[el].__setitem__("value", v + NORMALIZED))
    real, lines = native.op_act, []

    def helper(_self, op, *args, timeout=20):
        if op == "resolve":
            return {"ok": True, "candidates": [app_dict(KEYNOTE)]}
        assert op == "act"
        monkeypatch.setattr(native, "op_act", lambda pid, do, sel, n, ms: real(
            pid, do, sel, n, ms, ax=ax, front=lambda: app_dict(KEYNOTE), settle=lambda _s: None))
        try:
            with mock.patch("builtins.print", lambda text: lines.append(text)):
                native.main(["act", *[str(a) for a in args]])
        finally:
            monkeypatch.setattr(native, "op_act", real)
        return json.loads(lines[-1])

    monkeypatch.setattr(desktop.module._ComputerUse, "_native", helper)
    row = desktop.call("element_act", fit=measured_fit(window=2_000, reserve=500), app="Keynote", do="set_value",
                       select={"role": "TextField", "label": LONG_LABEL}, text=LONG_TEXT)
    assert row["result_partial"] is True and row["result_source_ref"], "the model saw a head+tail view"
    assert lines[-1].isascii() and ax.sets == [("field", LONG_TEXT)] and ax.balanced()

    effects = rr.produced_material(ctx)["effects"]
    assert [effect["tool"] for effect in effects] == [extension_surface_name("unix_computer_use", "element_act")]
    receipt = json.loads(effects[0]["receipt"])
    assert receipt["observed_effect"] is True and receipt["action_outcome"] == "completed" and receipt["verified"] is False
    assert {"label": receipt["element"]["label"], "label_after": receipt["element_after"]["label"],
            "before": receipt["element"]["value"], "after": receipt["element_after"]["value"]} == {
        "label": LONG_LABEL, "label_after": LONG_LABEL, "before": LONG_BEFORE, "after": LONG_TEXT + NORMALIZED}
    assert "value_len" not in receipt["element"] and "value_len" not in receipt["element_after"]
    assert effects[0]["receipt"] in rr.answer_text(reply="Typed.", produced=rr.produced_material(ctx), verification="")


@pytest.mark.parametrize("damage", ["not_retained", "corrupt", "deleted"])
def test_an_unavailable_producer_source_is_a_named_gap_never_a_guess(tmp_path, monkeypatch, damage):
    ctx = _floor_ctx(tmp_path, monkeypatch)
    sends: list[dict] = []
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam(sends))
    desktop = mac(ctx, tmp_path, monkeypatch)
    if damage == "not_retained":
        with mock.patch.object(artifacts, "store_actor_source_bytes", side_effect=OSError("disk full")):
            row = desktop.call("click", warning=WARNING, x=1, y=2, raw=True)
        assert row["producer_source_ref"] == {}
    else:
        row = desktop.call("click", warning=WARNING, x=1, y=2, raw=True)
        path = artifacts.task_artifact_dir_path(ctx.drive_root, ctx.task_id) / row["producer_source_ref"]["path"]
        if damage == "corrupt":
            path.write_bytes(path.read_bytes().replace(b'"ok": true', b'"ok": TRUE'))
        else:
            path.unlink()
    assert '"observed_effect": true' in row["result"], "the annotated preview still reads like a receipt"

    produced = rr.produced_material(ctx)
    assert produced["effects"] == [], "an unverifiable receipt is never read from the preview"
    [gap] = effect_gaps(produced)
    assert row["tool"] in gap
    assert rr.result_review_at_delivery(ctx, "Clicked.", trace_of(ctx), lambda _text: None) is False and sends == []


def test_the_same_receipt_is_reviewed_once_and_a_new_receipt_is_a_new_subject(tmp_path, monkeypatch):
    ctx = _floor_ctx(tmp_path, monkeypatch)
    sends: list[dict] = []
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam(sends))
    desktop = mac(ctx, tmp_path, monkeypatch)
    desktop.call("click", x=1, y=2, raw=True)
    answer = "Clicked Save."
    assert rr.result_review_at_delivery(ctx, answer, trace_of(ctx), lambda _text: None) is False
    first = subject_text(ctx)
    assert len(sends) == 3
    assert rr.result_review_at_delivery(ctx, answer, trace_of(ctx), lambda _text: None) is False
    assert len(sends) == 3 and subject_text(ctx) == first, "the same receipt is the same subject: reused at $0"

    desktop.call("key", keys="cmd+s")
    assert rr.result_review_at_delivery(ctx, answer, trace_of(ctx), lambda _text: None) is False
    assert len(sends) == 6, "a new receipt is new material: a new review"
    clicked, saved = (effect["receipt"] for effect in rr.produced_material(ctx)["effects"])
    second = subject_text(ctx)
    assert clicked in first and saved not in first and clicked in second and saved in second
    current = [run for run in trace_of(ctx)["review_runs"] if not run.get("superseded_by_revision")]
    assert [run["aggregate_signal"] for run in current] == ["PASS"]


# --- Modes and roles are unchanged ----------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [{"mode": "auto"}, {"mode": "off"}, {"root": False}])
def test_auto_off_and_a_child_are_never_forced_by_a_desktop_input(tmp_path, monkeypatch, kwargs):
    ctx = _floor_ctx(tmp_path, monkeypatch, **kwargs)
    sends: list[dict] = []
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam(sends))
    mac(ctx, tmp_path, monkeypatch).call("click", x=1, y=2, raw=True)
    assert rr.produced_material(ctx)["effects"], "the receipt is produced material in every mode"
    assert rr.result_review_at_delivery(ctx, "Clicked.", trace_of(ctx), lambda _text: None) is False
    assert sends == [] and not trace_of(ctx).get("review_runs")


def test_blocking_holds_a_failed_desktop_result_and_presence_is_never_held(tmp_path, monkeypatch):
    ctx = _floor_ctx(tmp_path / "blocking", monkeypatch, enforcement="blocking")
    sends: list[dict] = []
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam(sends, {"t1": CRITICAL}))
    mac(ctx, tmp_path / "blocking", monkeypatch).call("click", x=1, y=2, raw=True)
    shown: list[str] = []
    assert rr.result_review_at_delivery(ctx, "Clicked.", trace_of(ctx), shown.append) is True
    assert rr.result_review_at_delivery(ctx, "Clicked.", trace_of(ctx), shown.append) is True
    assert "not delivered until a review passes it" in shown[-1] and len(sends) == 3

    ctx = _floor_ctx(tmp_path / "presence", monkeypatch)
    ctx._presence_completion = {"outcome": "spoken"}
    sends.clear()
    mac(ctx, tmp_path / "presence", monkeypatch).call("click", x=1, y=2, raw=True)
    shown.clear()
    assert rr.result_review_at_delivery(ctx, "Clicked.", trace_of(ctx), shown.append) is False
    assert shown == [] and len(sends) == 3, "its owed review runs with nobody waiting for it"
