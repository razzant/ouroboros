"""Iteration-2 harness/runtime fixes (TB2.1 trace-deepdive follow-up):

#1 run_tb.apply_all_model — single-model run defaults to 1 reviewer at low effort (configurable).
#2 shell._resolve_effective_timeout — SSOT hygiene: a configured value equal to the default (600) is
   honored, not silently dropped to the in-code 360; override/ceiling/deadline-clamp preserved.
#3 vision.view_image — bring a LOCAL image natively into the active model's context (local-only, not
   web-gated, reuses vlm_query's exact trust checks; blind-model + fail-closed paths covered).
"""
import os
import pathlib
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _restore_env():
    """apply_all_model / the timeout tests mutate os.environ directly; snapshot and
    restore around every test so they cannot leak (e.g. a non-vision OUROBOROS_MODEL)
    into other tests in the same pytest session."""
    snapshot = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(snapshot)


@pytest.fixture(autouse=True)
def _stub_tool_timeout_settings(monkeypatch):
    """The #2 timeout resolver consults load_settings(); stub it to {} so these tests are
    deterministic regardless of the developer's real data/settings.json (CI is already
    isolated via OUROBOROS_DATA_DIR, but local runs should not depend on it)."""
    try:
        monkeypatch.setattr("ouroboros.tools.shell.load_settings", lambda: {})
    except Exception:
        pass
    yield

# Minimal valid 1x1 PNG.
_PNG_1x1 = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde"
    b"\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


# ----------------------------- #1 review knob -----------------------------
def test_apply_all_model_one_low_reviewer_by_default(monkeypatch):
    from devtools.benchmarks.terminal_bench.run_tb import apply_all_model

    for k in ("OUROBOROS_REVIEW_MODELS", "OUROBOROS_EFFORT_REVIEW", "OUROBOROS_EFFORT_SCOPE_REVIEW"):
        monkeypatch.delenv(k, raising=False)
    apply_all_model("google/gemini-3.5-flash")
    assert os.environ["OUROBOROS_REVIEW_MODELS"] == "google/gemini-3.5-flash"  # one reviewer, no commas
    assert os.environ["OUROBOROS_EFFORT_REVIEW"] == "low"
    assert os.environ["OUROBOROS_EFFORT_SCOPE_REVIEW"] == "low"


def test_apply_all_model_configurable_slots_and_effort(monkeypatch):
    from devtools.benchmarks.terminal_bench.run_tb import apply_all_model

    for k in ("OUROBOROS_REVIEW_MODELS", "OUROBOROS_EFFORT_REVIEW", "OUROBOROS_EFFORT_SCOPE_REVIEW"):
        monkeypatch.delenv(k, raising=False)
    apply_all_model("m", review_slots=3, review_effort="medium")
    assert os.environ["OUROBOROS_REVIEW_MODELS"] == "m,m,m"
    assert os.environ["OUROBOROS_EFFORT_REVIEW"] == "medium"


# ------------------------- #2 timeout resolver SSOT -------------------------
def test_timeout_resolver_unset_honors_config_default_not_incode_360(monkeypatch):
    from ouroboros.tools.shell import _resolve_effective_timeout as resolve

    monkeypatch.delenv("OUROBOROS_TOOL_TIMEOUT_SEC", raising=False)
    assert resolve(360) == 600  # the bug returned 360


def test_timeout_resolver_env_equal_to_default_is_honored(monkeypatch):
    from ouroboros.tools.shell import _resolve_effective_timeout as resolve

    monkeypatch.setenv("OUROBOROS_TOOL_TIMEOUT_SEC", "600")
    assert resolve(360) == 600  # was silently dropped to 360 (== default skipped)


def test_timeout_resolver_ceiling_and_override(monkeypatch):
    from ouroboros.config import get_per_call_timeout_ceiling_sec
    from ouroboros.tools.shell import _resolve_effective_timeout as resolve

    ceiling = get_per_call_timeout_ceiling_sec()
    monkeypatch.setenv("OUROBOROS_TOOL_TIMEOUT_SEC", "999999")
    assert resolve(360) == ceiling  # env clamped to ceiling
    assert resolve(360, None, 300) == 300  # per-call override honored
    assert resolve(360, None, 999999) == ceiling  # override clamped to ceiling


# ------------------------------- #3 view_image -------------------------------
def _ctx(drive_root, active_model="google/gemini-3.5-flash"):
    from ouroboros.tools.registry import ToolContext

    ctx = MagicMock(spec=ToolContext)
    ctx.event_queue = None
    ctx.task_id = "t"
    ctx.current_task_type = "task"
    ctx.drive_root = str(drive_root)
    ctx.active_model = active_model
    ctx.task_model_override = ""
    ctx.messages = []
    return ctx


def _img_under_uploads():
    tmp = tempfile.mkdtemp()
    uploads = pathlib.Path(tmp) / "uploads"
    uploads.mkdir()
    img = uploads / "chart.png"
    img.write_bytes(_PNG_1x1)
    return tmp, uploads, img


def test_view_image_injects_native_block_for_vision_model():
    from ouroboros.tools.vision import _view_image

    tmp, uploads, img = _img_under_uploads()
    try:
        ctx = _ctx(tmp)
        with patch("ouroboros.tools.vision._allowed_file_roots", return_value=[uploads]):
            res = _view_image(ctx, str(img))
        assert "local image block" in res.lower()
        blocks = [
            b
            for m in ctx.messages
            for b in (m.get("content") or [])
            if isinstance(b, dict)
        ]
        assert any(
            b.get("type") == "image_url" and str(b.get("image_url", {}).get("url", "")).startswith("data:image/")
            for b in blocks
        ), "expected a native data:image/ block appended to the conversation"
    finally:
        shutil.rmtree(tmp)


def test_view_image_blind_model_still_attaches_for_send_time_routing():
    from ouroboros.tools.vision import _view_image

    tmp, uploads, img = _img_under_uploads()
    try:
        ctx = _ctx(tmp, active_model="some-blind-model-without-vision")
        with patch("ouroboros.tools.vision._allowed_file_roots", return_value=[uploads]):
            res = _view_image(ctx, str(img))
        assert "local image block" in res.lower()
        assert ctx.messages  # send-time routing will caption or omit for blind routes
    finally:
        shutil.rmtree(tmp)


def test_view_image_non_image_fail_closed():
    from ouroboros.tools.vision import _view_image

    tmp = tempfile.mkdtemp()
    uploads = pathlib.Path(tmp) / "uploads"
    uploads.mkdir()
    bad = uploads / "notes.txt"
    bad.write_bytes(b"this is plain text, not an image")
    try:
        ctx = _ctx(tmp)
        with patch("ouroboros.tools.vision._allowed_file_roots", return_value=[uploads]):
            res = _view_image(ctx, str(bad))
        assert "⚠️" in res and "supported image" in res.lower()
    finally:
        shutil.rmtree(tmp)


def test_view_image_path_outside_roots_rejected():
    from ouroboros.tools.vision import _view_image

    tmp = tempfile.mkdtemp()
    uploads = pathlib.Path(tmp) / "uploads"
    uploads.mkdir()
    outside = pathlib.Path(tmp) / "secret.png"  # NOT under uploads
    outside.write_bytes(_PNG_1x1)
    try:
        ctx = _ctx(tmp)
        with patch("ouroboros.tools.vision._allowed_file_roots", return_value=[uploads]):
            res = _view_image(ctx, str(outside))
        assert "⚠️" in res and "workspace" in res.lower()
    finally:
        shutil.rmtree(tmp)


def test_view_image_is_not_web_gated():
    # The reward-hacking gate keys on membership in _WEB_TOOLS; view_image must NOT be in it,
    # so it is available even under allowed_resources.web=false.
    from ouroboros.tools.registry import _WEB_TOOLS

    assert "view_image" not in _WEB_TOOLS
