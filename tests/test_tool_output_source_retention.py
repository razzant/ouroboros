"""A shell or browser result is retained whole before the view the mind is shown is bounded.

The producers used to cut their own text (shell 50 000 characters, browser text/markdown 30 000, HTML 50 000,
evaluate 20 000) before the loop's exact-source delivery ever saw it, so the omitted middle and tail were
unreadable anywhere. These tests run the REAL handlers (a real ``run_command`` subprocess through the registry
and the loop's ``handle_tool_calls``; the real ``browse_page`` / ``browser_action`` handlers over a page that
answers with large text) into the REAL delivery (``tool_result_delivery`` under the measured frame) and read the
retained source back with the REAL ``read_file``. Negative controls: a small result is delivered whole and
unchanged, and a source that cannot be written is disclosed as unavailable, never as complete.
"""
from __future__ import annotations

import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from ouroboros import artifacts, config
from ouroboros.loop_tool_execution import StatefulToolExecutor, handle_tool_calls, process_tool_results
from ouroboros.tool_result_delivery import FULL_SOURCE_MARKER, FULL_SOURCE_UNAVAILABLE, RESULT_VIEW_MARKER
from ouroboros.tools import browser
from ouroboros.tools.registry import ToolContext, ToolRegistry
from tests._tool_result_delivery_shared import measured_fit

HEAD, MIDDLE, TAIL = "HEAD_MARKER_41d2", "MIDDLE_MARKER_7f3a", "TAIL_MARKER_9c1d"
SECRET = "synthetic-retained-secret-7b1e9d"
WINDOW, RESERVE = 40_000, 4_000  # a frame that cannot hold a 100 000-character result whole


def _marker_json(content: str, marker: str):
    line = next(line for line in content.split("\n") if line.startswith(marker))
    return json.loads(line[len(marker):])


def _large(total: int, *, filler: str = "x") -> str:
    """``total`` characters with a marker at the head, one at the exact middle and one at the tail."""
    half = (total - len(HEAD) - len(MIDDLE) - len(TAIL)) // 2
    return HEAD + filler * half + MIDDLE + filler * (total - len(HEAD) - len(MIDDLE) - len(TAIL) - half) + TAIL


@pytest.fixture
def registry(tmp_path, monkeypatch):
    repo, data = tmp_path / "repo", tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    for key, value in (("OUROBOROS_RUNTIME_MODE", "advanced"), ("OUROBOROS_SAFETY_MODE", "off"),
                       ("OUROBOROS_CONSCIOUSNESS_SAFETY_MODE", "off")):
        monkeypatch.setenv(key, value)
    # One synthetic custom secret the command may select, so the selected-secret masking is the real one.
    (data / "settings.json").write_text(json.dumps({"SYNTHETIC_RETAINED_SECRET": SECRET}), encoding="utf-8")
    monkeypatch.setattr(config, "SETTINGS_PATH", data / "settings.json")
    ctx = ToolContext(repo_dir=repo, drive_root=data, task_id="retention-task")
    tools = ToolRegistry(repo_dir=repo, drive_root=data)
    tools.set_context(ctx)
    return tools


def _run(registry, tmp_path, name, args, *, window=WINDOW):
    """One call through the loop's real executor and measured-frame delivery."""
    logs = tmp_path / "logs"
    logs.mkdir(exist_ok=True)
    messages, trace = [{"role": "user", "content": "go"}], {"tool_calls": []}
    errors = handle_tool_calls(
        [{"id": "call-1", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}],
        registry, logs, registry._ctx.task_id, StatefulToolExecutor(), messages, trace, lambda _t: None,
        fit_candidate=measured_fit(window=window, reserve=RESERVE), tool_schemas=[])
    return errors, messages[-1]["content"], trace["tool_calls"][0]


def _source_bytes(registry, row) -> str:
    return artifacts.read_actor_source_bytes(registry._ctx.drive_root, registry._ctx.task_id, row["result_source_ref"]).decode("utf-8")


def _read_omitted(registry, content: str) -> str:
    """What the view tells the mind to read for the omitted middle, read through the real ``read_file``."""
    read = _marker_json(content, RESULT_VIEW_MARKER)["read_omitted"]
    return registry.execute("read_file", {key: value for key, value in read.items() if key != "tool"})


SCRIPT = ("import os, sys\n"
          "secret = os.environ.get('CHILD_SECRET', '')\n"
          "half = {half}\n"
          "print({head!r}); print('a' * half); print({middle!r}, secret); print('b' * half); print({tail!r}, secret)\n"
          "sys.stderr.write('STDERR_LINE ' + secret + '\\n')\n")


def _shell_args(total: int) -> dict:
    half = (total - 400) // 2
    return {"cmd": [sys.executable, "-c", SCRIPT.format(half=half, head=HEAD, middle=MIDDLE, tail=TAIL)],
            "env_from_settings": {"CHILD_SECRET": "SYNTHETIC_RETAINED_SECRET"}}


@pytest.mark.parametrize("total", [60_000, 400_000])  # just above the old 50 000 cut, and far above it
def test_a_large_shell_output_keeps_its_middle_and_tail_in_the_exact_source(registry, tmp_path, total):
    errors, content, row = _run(registry, tmp_path, "run_command", _shell_args(total))
    assert errors == 0
    # The view the mind sees is bounded (the frame cannot hold the result) and names its exact source.
    assert row["result_partial"] is True and row["result_source_status"] == "ready"
    view = _marker_json(content, RESULT_VIEW_MARKER)
    assert _marker_json(content, FULL_SOURCE_MARKER) == row["result_source_ref"]
    assert MIDDLE not in content, "the omitted middle is not in the view"
    # The retained source is the producer's COMPLETE text: nothing cut before retention.
    source = _source_bytes(registry, row)
    assert len(source) == view["complete_chars"] > 50_000 and hashlib.sha256(source.encode()).hexdigest() == view["complete_sha256"]
    for marker in (HEAD, MIDDLE, TAIL, "STDERR_LINE"):
        assert marker in source, marker
    assert "...(truncated)..." not in source
    # Labels, selected-secret masking (head, middle, tail and stderr alike) and process facts are as they were.
    assert source.index("STDOUT:\n") < source.index(MIDDLE) < source.index(TAIL) < source.index("\n\nSTDERR:\n")
    assert SECRET not in source and source.count("***") == 3, source.count("***")
    assert view["facts"]["exit_code"] == 0 and view["facts"]["status"] == "ok" and view["facts"]["code"] == "OK"
    # The reader the view names returns the omitted middle, and the source's last page returns the tail.
    omitted = _read_omitted(registry, content)
    assert MIDDLE in omitted and HEAD not in omitted.split(MIDDLE)[1]
    tail_page = registry.execute("read_file", {"root": "artifact_store", "path": row["result_source_ref"]["path"],
                                               "start_char": max(0, len(source) - 400), "max_chars": 400})
    assert TAIL in tail_page


def test_a_failing_command_keeps_its_complete_output_and_exit_fact(registry, tmp_path):
    args = _shell_args(120_000)
    args["cmd"][-1] += "sys.exit(3)\n"
    _errors, content, row = _run(registry, tmp_path, "run_command", args)
    source = _source_bytes(registry, row)
    view = _marker_json(content, RESULT_VIEW_MARKER)
    assert MIDDLE in source and TAIL in source and SECRET not in source
    assert source.startswith("⚠️ SHELL_EXIT_ERROR") and "STDOUT:\n" in source
    assert view["facts"]["exit_code"] == 3 and row["is_error"] is True


def test_a_small_shell_output_is_delivered_whole_and_unchanged(registry, tmp_path):
    _errors, content, row = _run(registry, tmp_path, "run_command", {"cmd": [sys.executable, "-c", "print('small output')"]})
    assert RESULT_VIEW_MARKER not in content and FULL_SOURCE_MARKER not in content
    assert "result_partial" not in row and "result_source_ref" not in row
    assert "STDOUT:\nsmall output\n" in content and content.startswith("exit_code=0")


def test_an_unwritable_source_is_disclosed_as_unavailable_never_as_complete(registry, tmp_path, monkeypatch):
    def refuse(*_args, **_kwargs):
        raise OSError("disk refused the source")

    monkeypatch.setattr(artifacts, "store_actor_source_bytes", refuse)
    _errors, content, row = _run(registry, tmp_path, "run_command", _shell_args(120_000))
    assert row["result_partial"] is True and row["result_source_status"] == "source_unavailable"
    assert row["result_source_ref"] == {}
    assert FULL_SOURCE_UNAVAILABLE in content and "Do not treat this partial result as complete" in content
    view = _marker_json(content, RESULT_VIEW_MARKER)
    assert view["delivery"]["source_status"] == "source_unavailable" and "read_omitted" not in view
    assert HEAD in content and MIDDLE not in content, "still an honest head and tail, with the gap named"
    assert FULL_SOURCE_MARKER not in content


# --- browser: the real handlers over a page that answers with large content --------------------------------

class _LargePage:
    """A page answering each extraction with its configured large text; the handlers' own calls are real."""

    url = "http://example.test/"

    def __init__(self, *, text="", html="", evaluated=""):
        self.text, self.html, self.evaluated = text, html, evaluated
        self.default_ms = None

    def set_default_timeout(self, ms):
        self.default_ms = ms

    def goto(self, *_args, **_kwargs):
        return None

    def inner_text(self, _selector):
        return self.text

    def content(self):
        return self.html

    def title(self):
        return "large page"

    def evaluate(self, _expression):
        return self.evaluated  # the markdown extractor and a caller's expression both come through here


def _browser_ctx(tmp_path):
    state = SimpleNamespace(last_screenshot_b64="", _browser_engine="chromium", _browser_device="")
    return SimpleNamespace(browser_state=state, repo_dir=tmp_path, drive_root=tmp_path, task_id="browser-retention")


def _bind_page(monkeypatch, page):
    monkeypatch.setattr(browser, "_ensure_browser", lambda ctx, *a, **k: (page, ctx.browser_state))
    monkeypatch.setattr(browser, "_readonly_subagent", lambda ctx: False)


def _deliver_text(tmp_path, ctx, text, *, window=WINDOW):
    """The loop's real measured-frame delivery of one browser result, then the reader's registry."""
    row = {"fn_name": "browse_page", "tool_call_id": "b1", "result": text, "is_error": False,
           "tool_args": {}, "args_for_log": {}, "result_meta": {"status": "ok"}}
    messages, trace = [{"role": "user", "content": "go"}], {"tool_calls": []}
    process_tool_results([row], messages, trace, lambda *_a, **_k: None, SimpleNamespace(_ctx=ctx),
                         fit_candidate=measured_fit(window=window, reserve=RESERVE), tool_schemas=[])
    reader = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    reader.set_context(ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id=ctx.task_id))
    return messages[-1]["content"], trace["tool_calls"][0], reader


@pytest.mark.parametrize(("output", "total"), [("text", 120_000), ("markdown", 120_000), ("html", 60_000)])
def test_browse_page_output_is_whole_and_its_middle_and_tail_are_readable(tmp_path, monkeypatch, output, total):
    ctx = _browser_ctx(tmp_path)
    text = _large(total)
    _bind_page(monkeypatch, _LargePage(text=text, html=text, evaluated=text))
    produced = browser._browse_page(ctx, "http://example.test/", output=output)
    assert text in produced, "the handler returns the page's whole text, byte for byte"
    assert "... [truncated]" not in produced
    view, row, reader = _deliver_text(tmp_path, ctx, produced)
    assert row["result_partial"] is True and row["result_source_status"] == "ready" and MIDDLE not in view
    source = artifacts.read_actor_source_bytes(tmp_path, ctx.task_id, row["result_source_ref"]).decode("utf-8")
    assert source == produced and MIDDLE in source and TAIL in source
    assert MIDDLE in _read_omitted(reader, view)


def test_browser_evaluate_output_is_whole_and_retained(tmp_path, monkeypatch):
    ctx = _browser_ctx(tmp_path)
    text = _large(25_000)
    _bind_page(monkeypatch, _LargePage(evaluated=text))
    produced = browser._browser_action(ctx, "evaluate", value="document.body.innerText")
    assert text in produced and "... [truncated]" not in produced
    view, row, reader = _deliver_text(tmp_path, ctx, produced, window=8_000)
    assert row["result_partial"] is True and MIDDLE not in view
    source = artifacts.read_actor_source_bytes(tmp_path, ctx.task_id, row["result_source_ref"]).decode("utf-8")
    assert source == produced and MIDDLE in source and TAIL in source
    assert MIDDLE in _read_omitted(reader, view)


def test_small_browser_output_is_delivered_whole_and_unchanged(tmp_path, monkeypatch):
    ctx = _browser_ctx(tmp_path)
    _bind_page(monkeypatch, _LargePage(text="small page text", html="<p>small</p>", evaluated="# small"))
    for output, expected in (("text", "small page text"), ("html", "<p>small</p>"), ("markdown", "# small")):
        produced = browser._browse_page(ctx, "http://example.test/", output=output)
        assert expected in produced and "[truncated]" not in produced
        view, row, _reader = _deliver_text(tmp_path, ctx, produced)
        assert view == produced and "result_partial" not in row


def test_an_unwritable_browser_source_is_disclosed_as_unavailable(tmp_path, monkeypatch):
    def refuse(*_args, **_kwargs):
        raise OSError("disk refused the source")

    ctx = _browser_ctx(tmp_path)
    _bind_page(monkeypatch, _LargePage(text=_large(120_000)))
    produced = browser._browse_page(ctx, "http://example.test/", output="text")
    monkeypatch.setattr(artifacts, "store_actor_source_bytes", refuse)
    view, row, _reader = _deliver_text(tmp_path, ctx, produced)
    assert row["result_partial"] is True and row["result_source_status"] == "source_unavailable"
    assert FULL_SOURCE_UNAVAILABLE in view and "Do not treat this partial result as complete" in view
    assert MIDDLE not in view and FULL_SOURCE_MARKER not in view
