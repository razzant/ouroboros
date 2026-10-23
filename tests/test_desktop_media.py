"""Native microphone declarations and the GTK boundary without opening a desktop window."""

import ast
from pathlib import Path
import plistlib
from types import SimpleNamespace

import pytest

from ouroboros import desktop_media as media

ROOT = Path(__file__).resolve().parents[1]
ORIGIN = "http://127.0.0.1:8765"


class Request:
    def __init__(self, audio=True, video=False, display=False):
        self.flags = {"is-for-audio-device": audio, "is-for-video-device": video,
                      "is-for-display-device": display}
        self.outcomes = []

    def get_property(self, key):
        return self.flags[key]

    def find_property(self, key):
        return key in self.flags

    def allow(self):
        self.outcomes.append("allow")

    def deny(self):
        self.outcomes.append("deny")


class View:
    def __init__(self, url=ORIGIN):
        self.url = url
        self.handlers = {}

    def get_uri(self):
        return self.url

    def connect(self, event, callback):
        assert event == "permission-request"
        self.handlers[1] = callback
        return 1

    def handler_is_connected(self, key):
        return key in self.handlers

    def disconnect(self, key):
        del self.handlers[key]


@pytest.mark.parametrize("url", [ORIGIN, ORIGIN + "/?room=one", ORIGIN + "/static/index.html"])
def test_audio_only_on_own_origin_is_granted(url):
    request = Request()
    assert media.handle_permission_request(View(url), request, origin=ORIGIN, media_type=Request)
    assert request.outcomes == ["allow"]


@pytest.mark.parametrize("url", ["https://127.0.0.1:8765", "http://127.0.0.1:8766",
    "http://localhost:8765", "http://127.0.0.1.evil.test:8765", "http://evil.test/",
    "http://127.0.0.1:8765@evil.test/", "file:///tmp/page.html", "about:blank", "", None])
def test_other_origins_and_unreadable_navigation_are_denied(url):
    request = Request()
    assert media.handle_permission_request(View(url), request, origin=ORIGIN, media_type=Request)
    assert request.outcomes == ["deny"]


@pytest.mark.parametrize("audio,video,display", [(False, False, False), (True, True, False),
    (False, True, False), (True, False, True)])
def test_video_display_and_non_audio_requests_are_denied(audio, video, display):
    request = Request(audio, video, display)
    media.handle_permission_request(View(), request, origin=ORIGIN, media_type=Request)
    assert request.outcomes == ["deny"]


def test_other_permission_classes_and_unknown_flags_fail_closed():
    request = Request()
    media.handle_permission_request(View(), request, origin=ORIGIN, media_type=str)
    assert request.outcomes == ["deny"]
    request = Request()
    request.get_property = lambda _: (_ for _ in ()).throw(ValueError("unreadable flag"))
    media.handle_permission_request(View(), request, origin=ORIGIN, media_type=Request)
    assert request.outcomes == ["deny"]
    request = Request()
    request.find_property = lambda _: (_ for _ in ()).throw(ValueError("unreadable display flag"))
    media.handle_permission_request(View(), request, origin=ORIGIN, media_type=Request)
    assert request.outcomes == ["deny"]


def test_origin_comparison_preserves_port_and_loopback_identity():
    assert media.loopback_origin("https://[::1]/path") == ("https", "::1", 443)
    assert media.loopback_origin("http://localhost") == ("http", "localhost", 80)
    assert media.loopback_origin("http://127.0.0.1:0") == ("http", "127.0.0.1", 0)
    for url in ["https://example.test", "http://192.168.1.2", "http://127.0.0.1:bad", "http://user@localhost"]:
        assert media.loopback_origin(url) is None


class Hook:
    def __init__(self):
        self.handlers = []

    def __iadd__(self, handler):
        self.handlers.append(handler)
        return self

    def __isub__(self, handler):
        self.handlers.remove(handler)
        return self

    def fire(self):
        for handler in list(self.handlers):
            handler()


def test_native_view_binds_before_load_and_releases_on_close(monkeypatch):
    monkeypatch.setattr(media.sys, "platform", "linux")
    view = View()
    window = SimpleNamespace(gui=SimpleNamespace(__name__="webview.platforms.gtk",
        webkit=SimpleNamespace(WebView=View, UserMediaPermissionRequest=Request)),
        native=SimpleNamespace(get_children=lambda: [view]),
        events=SimpleNamespace(before_show=Hook(), closed=Hook()))
    assert media.attach_media_permissions(window, ORIGIN) is window
    assert view.handlers == {}
    window.events.before_show.fire()
    request = Request()
    assert view.handlers[1](view, request)
    assert request.outcomes == ["allow"]
    view.url = "https://elsewhere.test"
    request = Request()
    view.handlers[1](view, request)
    assert request.outcomes == ["deny"]
    window.events.closed.fire()
    assert view.handlers == {} and not window.events.before_show.handlers and not window.events.closed.handlers


def test_non_linux_shells_keep_native_prompt(monkeypatch):
    for platform in ["darwin", "win32"]:
        monkeypatch.setattr(media.sys, "platform", platform)
        window = object()
        assert media.attach_media_permissions(window, ORIGIN) is window


def test_macos_bundle_declares_microphone_camera_and_signed_entitlements():
    spec = ast.parse((ROOT / "Ouroboros.spec").read_text())
    bundle = next(n for n in ast.walk(spec) if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Name) and n.func.id == "BUNDLE")
    info = next(k.value for k in bundle.keywords if k.arg == "info_plist")
    fields = {ast.literal_eval(k): v for k, v in zip(info.keys, info.values)}
    assert "microphone" in ast.literal_eval(fields["NSMicrophoneUsageDescription"]).lower()
    assert "speech recognition" in ast.literal_eval(fields["NSSpeechRecognitionUsageDescription"]).lower()
    assert "camera" in ast.literal_eval(fields["NSCameraUsageDescription"]).lower()
    entitlements = plistlib.loads((ROOT / "entitlements.plist").read_bytes())
    assert entitlements["com.apple.security.device.audio-input"] is True
    assert entitlements["com.apple.security.device.camera"] is True


def test_launcher_keeps_display_probe_alias_and_origin_bound_hook():
    import launcher
    assert launcher._display_ready is media.display_ready
    source = (ROOT / "launcher.py").read_text()
    assert "attach_media_permissions(background.attach(window), url)" in source
    assert "def _on_closing()" in source  # Background owns the existing shutdown callback.


def test_linux_bundle_restores_resource_paths_after_frozen_hooks(tmp_path, monkeypatch):
    import os

    # XDG uses ':' separators: keep the fixture free of Windows drive letters.
    monkeypatch.chdir(tmp_path)
    bundle = Path('.')
    native = bundle / 'linux-gtk'
    native.mkdir()
    (native / 'manifest.json').write_text('{}')
    monkeypatch.setattr(os, 'environ', {'GI_TYPELIB_PATH': '/frozen/gi_typelibs',
                                      'PYWEBVIEW_GUI': 'qt', 'XDG_DATA_DIRS': '/session/share'})
    media.prepare_linux_bundle(bundle)
    assert Path.cwd() == tmp_path / native
    assert os.environ['PYWEBVIEW_GUI'] == 'gtk'
    assert os.environ['GI_TYPELIB_PATH'] == str(native / 'lib/x86_64-linux-gnu/girepository-1.0')
    assert os.environ['GIO_MODULE_DIR'].startswith(str(native))
    monkeypatch.chdir(tmp_path)
    media.prepare_linux_bundle(bundle)
    assert os.environ['XDG_DATA_DIRS'] == f'{native / "share"}:/session/share'


def test_source_install_without_bundle_keeps_backend_environment(tmp_path, monkeypatch):
    import os

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(os, 'environ', {'PYWEBVIEW_GUI': 'qt'})
    media.prepare_linux_bundle(tmp_path)
    assert Path.cwd() == tmp_path
    assert dict(os.environ) == {'PYWEBVIEW_GUI': 'qt'}
