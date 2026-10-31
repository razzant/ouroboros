"""Focused contract tests for the small desktop attention seam."""

from __future__ import annotations


def test_macos_attention_is_visual_only(monkeypatch):
    from ouroboros import platform_layer

    calls = []
    monkeypatch.setattr(platform_layer, "IS_MACOS", True)
    monkeypatch.setattr(platform_layer.subprocess, "run", lambda *args, **kwargs: (calls.append((args, kwargs)) or type("Result", (), {"returncode": 0})()))
    shown = []
    result = platform_layer.request_native_attention(lambda: shown.append(True))

    assert result == {"ok": True, "status": "window_only", "sound_played": False, "window_attention": "launcher"}
    assert shown == [True]
    assert calls == [], "macOS attention must not bypass notification permission with a sound"


def test_windows_attention_uses_message_beep(monkeypatch):
    from ouroboros import platform_layer

    calls = []
    class Winsound:
        MB_ICONASTERISK = 1
        @staticmethod
        def MessageBeep(value):
            calls.append(value)
    monkeypatch.setattr(platform_layer, "IS_MACOS", False)
    monkeypatch.setattr(platform_layer, "IS_WINDOWS", True)
    monkeypatch.setattr(platform_layer, "IS_LINUX", False)
    monkeypatch.setitem(__import__("sys").modules, "winsound", Winsound)

    assert platform_layer.request_native_attention() == {"ok": True, "status": "native_sound", "sound_played": True}
    assert calls == [1]


def test_linux_attention_uses_canberra(monkeypatch):
    from ouroboros import platform_layer

    calls = []
    monkeypatch.setattr(platform_layer, "IS_MACOS", False)
    monkeypatch.setattr(platform_layer, "IS_WINDOWS", False)
    monkeypatch.setattr(platform_layer, "IS_LINUX", True)
    monkeypatch.setattr(platform_layer.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(platform_layer.subprocess, "run", lambda *args, **kwargs: (calls.append((args, kwargs)) or type("Result", (), {"returncode": 0})()))

    result = platform_layer.request_native_attention()
    assert result == {"ok": True, "status": "native_sound", "sound_played": True}
    assert calls[0][0][0] == ["/usr/bin/canberra-gtk-play", "-i", "message-new-instant"]


def test_failed_sound_backend_is_reported_and_can_fallback(monkeypatch):
    from ouroboros import platform_layer

    monkeypatch.setattr(platform_layer, "IS_MACOS", False)
    monkeypatch.setattr(platform_layer, "IS_WINDOWS", False)
    monkeypatch.setattr(platform_layer, "IS_LINUX", True)
    monkeypatch.setattr(platform_layer.shutil, "which", lambda name: "/usr/bin/" + name)
    calls = []
    monkeypatch.setattr(platform_layer.subprocess, "run", lambda *args, **kwargs: (calls.append(args[0]) or type("Result", (), {"returncode": 1})()))

    result = platform_layer.request_native_attention(lambda: None)
    assert result["ok"] is True
    assert result["status"] == "window_only"
    assert result["sound_played"] is False
    assert calls == [["/usr/bin/canberra-gtk-play", "-i", "message-new-instant"]]


def test_sound_off_can_raise_window_without_playing_sound(monkeypatch):
    from ouroboros import platform_layer

    shown = []
    monkeypatch.setattr(platform_layer, "IS_MACOS", False)
    monkeypatch.setattr(platform_layer, "IS_WINDOWS", False)
    monkeypatch.setattr(platform_layer, "IS_LINUX", False)

    assert platform_layer.request_native_attention(lambda: shown.append(True), sound=False) == {
        "ok": True, "status": "window_only", "sound_played": False, "window_attention": "launcher",
    }
    assert shown == [True]


def test_unsupported_attention_is_explicit(monkeypatch):
    from ouroboros import platform_layer

    monkeypatch.setattr(platform_layer, "IS_MACOS", False)
    monkeypatch.setattr(platform_layer, "IS_WINDOWS", False)
    monkeypatch.setattr(platform_layer, "IS_LINUX", False)

    assert platform_layer.request_native_attention() == {
        "ok": False,
        "status": "unsupported",
        "reason": "no_system_attention_backend",
    }
