"""The desktop skill's results reach every host reader as compact JSON of the same object (#2478).

Every ``unix_computer_use`` result, on the local, place and remote paths, leaves through one serializer
(``plugin._json``). The previous form indented it; the compact form drops only that layout, so every key,
its order, value and string survive, and the host's readers take the same facts from either text: the
dispatcher's typed self-reported failure, the retained producer source and the effect receipts the
result review reads, the same-round image attachment with its capture identity, and the observation an
image's coordinates are mapped through. The producers are the registered handlers on faked desktops (the
harness of ``test_result_review_desktop_effects``: ``subprocess.run`` and ``urlopen`` are the boundary);
nothing reaches a desktop, a Mac over SSH, a guest VM or a model.
"""
from __future__ import annotations

import io
import json
from types import SimpleNamespace

import pytest

from ouroboros import artifacts
from ouroboros import loop_tool_execution as execution
from ouroboros.tools import extension_dispatch
from ouroboros.tools import review_result as rr
from tests.test_result_review import _floor_ctx
from tests.test_result_review_desktop_effects import (  # noqa: F401 - the liveness fixture is autouse here too
    WARNING,
    Desktop,
    _live_extension,
    mac,
    ssh,
)
from tests.test_unix_computer_use_effects import Guest, Response

# A string a result must carry exactly: Unicode, quotes, JSON-looking text, escapes and line breaks.
TEXT = 'Отчёт: "Q4", final\n\tline 2 🚀 \\ {"ok": true, "x": null}\u2028end'
DISPLAY = {"index": 1, "main": True, "x": 0, "y": 0, "width": 1440, "height": 900}


def previous(payload) -> str:
    """The form every result had before."""
    return json.dumps(payload, ensure_ascii=False, indent=2)


def compact(payload) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def serialized(desktop, monkeypatch) -> list:
    """Each text the skill's serializer wrote, beside both forms of its payload taken at that moment."""
    seen, real = [], desktop.module._json

    def spy(payload):
        text = real(payload)
        seen.append({"text": text, "compact": compact(payload), "previous": previous(payload)})
        return text
    monkeypatch.setattr(desktop.module, "_json", spy)
    return seen


def native_facts(desktop, monkeypatch) -> None:
    """The macOS helper's display and application lists (a window title is ``TEXT``); other facts stay unread."""
    real = desktop.module._ComputerUse._native
    apps = {"ok": True, "frontmost": None, "titles_available": True, "truncated": False, "apps": [
        {"app": "Keynote", "pid": 4821, "app_id": "com.apple.iWork.Keynote", "scriptable": True, "windows": [
            {"title": TEXT, "minimized": False, "main": True, "bounds": [0, 25, 1440, 875], "window_id": 21},
            {"title": "", "minimized": True, "main": False, "bounds": None, "window_id": 22}]},
        {"app": "Helper", "pid": 612, "app_id": "com.vendor.helper", "windows": []}]}

    def native(self, op, *args, timeout=20):
        if op == "displays":
            return {"ok": True, "displays": [dict(DISPLAY)]}
        return json.loads(json.dumps(apps)) if op == "apps" else real(self, op, *args, timeout=timeout)
    monkeypatch.setattr(desktop.module._ComputerUse, "_native", native)


class ScreenGuest(Guest):
    """An OSWorld guest with a screen: ``GET /screenshot`` answers a real PNG, ``/execute`` as ``Guest``."""

    def __call__(self, request, timeout=None):
        if isinstance(request, str):
            from PIL import Image

            buffer = io.BytesIO()
            Image.new("RGB", (64, 40), (7, 11, 13)).save(buffer, format="PNG")
            return Response(buffer.getvalue())
        return super().__call__(request, timeout)


def run(desktop, seen, *requests) -> list:
    """Each request as the loop runs it: its trace row and what the serializer wrote for it."""
    rows = []
    for short, args in requests:
        start = len(seen)
        rows.append((desktop.call(short, **args), seen[start:]))
    return rows


def decoded(rows, short: str) -> dict:
    return next(json.loads(row["result"]) for row, _written in rows if row["tool"].endswith("_" + short))


@pytest.mark.parametrize("backend", ["mac", "x11", "osworld"])
def test_every_transport_writes_the_compact_text_of_the_object_it_wrote_before(tmp_path, monkeypatch, backend):
    ctx = _floor_ctx(tmp_path, monkeypatch)
    if backend == "mac":
        # cmd+c fails: the Command chord program given "c" (key() of Command and one character) exits 1.
        desktop = mac(ctx, tmp_path, monkeypatch, answer=lambda argv, _index: int(argv[0] == "osascript" and argv[-1] == "c"))
        native_facts(desktop, monkeypatch)
        seen = serialized(desktop, monkeypatch)
        rows = run(desktop, seen, ("screenshot", {}), ("window_list", {}), ("type_text", {"text": TEXT}),
                   ("key", {"keys": "cmd+a cmd+c"}), ("key", {"keys": "return", "app": "Nowhere"}))
        assert decoded(rows, "type_text")["cmd"] == ["cliclick", "t:" + TEXT]
        listed = decoded(rows, "window_list")
        assert listed["frontmost"] is None and listed["windows"][0]["windows"][0]["title"] == TEXT
        assert listed["windows"][0]["windows"][1] == {"title": "", "minimized": True, "main": False, "bounds": None,
                                                      "window_id": 22}
    elif backend == "x11":
        desktop = Desktop(ctx, tmp_path, monkeypatch, "x11", "linux", "x11", tools=("xdotool",),
                          answer=lambda argv, _index: 1 if "ctrl+s" in argv else 0)
        seen = serialized(desktop, monkeypatch)
        rows = run(desktop, seen, ("type_text", {"text": TEXT}), ("key", {"keys": "ctrl+s"}), ("scroll", {"clicks": 3}))
        assert decoded(rows, "type_text")["cmd"] == ["xdotool", "type", "--delay", "0", "--", TEXT]
    elif backend == "wayland":
        desktop = Desktop(ctx, tmp_path, monkeypatch, "wayland", "linux", "wayland", tools=("ydotool",))
        seen = serialized(desktop, monkeypatch)
        rows = run(desktop, seen, ("left_click_drag", {"start_x": 1, "start_y": 2, "end_x": 30, "end_y": 40, "raw": True}),
                   ("key", {"keys": "ctrl+s"}))
    elif backend == "ssh":
        desktop = ssh(ctx, tmp_path, monkeypatch)
        desktop.transport.answer = lambda argv, _index: (
            (0, TEXT) if argv[-1] == "uname -a" else 255 if argv[0] == "ssh" and "kp:return" in argv[-1] else 0)
        seen = serialized(desktop, monkeypatch)
        rows = run(desktop, seen, ("remote_exec", {"command": "uname -a"}), ("type_text", {"text": "hello"}),
                   ("key", {"keys": "down down return"}))
        assert decoded(rows, "remote_exec")["output"] == TEXT
    else:
        desktop = Desktop(ctx, tmp_path, monkeypatch, "vm", "linux", "x11", guest=ScreenGuest([0]))
        desktop.call("add_connection", name="vm", backend="osworld_http", target="http://guest.invalid:5000",
                     activate=True)
        seen = serialized(desktop, monkeypatch)
        rows = run(desktop, seen, ("screenshot", {"app": "Firefox"}))
        sent = len(desktop.guest.commands)
        rows += run(desktop, seen, ("type_text", {"text": TEXT}))
        # The skill parses the clipboard program's answer itself: a paste that ran gets no typed fallback.
        assert decoded(rows, "type_text")["method"] == "clipboard" and len(desktop.guest.commands) == sent + 1
        desktop.guest.codes = [0, 1]
        rows += run(desktop, seen, ("key", {"keys": "ctrl+a ctrl+c"}))
        # The remote capture is written, read back and written again with the display it did not select.
        assert len(rows[0][1]) == 2 and decoded(rows, "screenshot")["display"] == {
            "selected_by": "default", "app_display_unread": "app= is not applied to osworld_http captures"}

    codes = []
    for row, written in rows:
        assert written, f"{row['tool']} answered without the skill's serializer"
        for item in written:
            assert item["text"] == item["compact"] and "\n" not in item["text"]
            assert previous(json.loads(item["text"])) == item["previous"], "every key, order, value and string survives"
        text = written[-1]["text"]
        assert row["result"] == text, "the model reads the producer's own text"
        for form in (text, previous(json.loads(text))):
            typed = extension_dispatch._extension_completion(form, "")
            assert (typed.status == "error", typed.code) == (row["is_error"], row["tool_result_code"]), form
        codes.append(row["tool_result_code"])
    assert {"OK", "TOOL_REPORTED_FAILURE"} <= set(codes), codes


def test_the_host_readers_take_the_same_facts_from_the_compact_text(tmp_path, monkeypatch):
    ctx = _floor_ctx(tmp_path, monkeypatch)
    attached = []

    def attach(_ctx, path, *, stamp=None, observation=None, capture_source=""):
        attached.append((path, capture_source))
        return True, "attached"
    monkeypatch.setattr("ouroboros.tools.vision.attach_local_image_to_context", attach)
    desktop = mac(ctx, tmp_path, monkeypatch, answer=lambda argv, _index: int(argv[0] == "osascript" and argv[-1] == "c"))
    native_facts(desktop, monkeypatch)
    seen = serialized(desktop, monkeypatch)

    shot = desktop.call("screenshot")
    receipt = json.loads(shot["result"])
    # The image: the same file and capture identity (the producer's display index) from either text.
    assert receipt["mapped_input_available"] is True and attached == [
        (receipt["auto_attach_image"], json.dumps([shot["tool"], DISPLAY["index"]], sort_keys=True))]
    execution._maybe_auto_attach_image({"fn_name": shot["tool"], "result": previous(receipt), "is_error": False},
                                       SimpleNamespace(_ctx=ctx), staged_messages=[])
    assert attached[1] == attached[0]

    # The observation: coordinates read off this image map through the transform this text names.
    transform = receipt["coord_transform"]
    clicked = desktop.call("click", x=10, y=20, observation_id=receipt["observation_id"])
    point = f"c:{round(10 * transform['sx'] + transform['ox'])},{round(20 * transform['sy'] + transform['oy'])}"
    assert desktop.transport.calls[-1] == ["cliclick", point] and clicked["tool_result_code"] == "OK"

    failed = desktop.call("key", keys="cmd+a cmd+c")
    noted = desktop.call("type_text", warning=WARNING, text=TEXT)
    # What the model saw carries the host note; the retained producer source is the compact text, byte for byte.
    source = artifacts.read_actor_source_bytes(ctx.drive_root, ctx.task_id, noted["producer_source_ref"])
    assert source == seen[-1]["text"].encode("utf-8") and noted["host_annotations"] == [WARNING]
    assert json.loads(source)["cmd"] == ["cliclick", "t:" + TEXT]

    # The result review's effect receipts are exactly these compact texts (the failed step keeps its mark).
    effects = rr.produced_material(ctx)["effects"]
    assert [effect["tool"] for effect in effects] == [clicked["tool"], failed["tool"], noted["tool"]]
    texts = [clicked["result"], failed["result"], source.decode("utf-8")]
    assert [effect["receipt"] for effect in effects] == texts and all(text == compact(json.loads(text)) for text in texts)
    assert [json.loads(text)["ok"] for text in texts] == [True, False, True]
