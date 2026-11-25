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
import ast
import json
import re
import sys
from types import SimpleNamespace

import pytest

from ouroboros import artifacts, config
from ouroboros.loop_tool_execution import StatefulToolExecutor, handle_tool_calls, process_tool_results
from ouroboros.tool_result_delivery import FULL_SOURCE_MARKER, FULL_SOURCE_UNAVAILABLE, RESULT_VIEW_MARKER
from ouroboros.tools import browser
from ouroboros.tools.registry import ToolContext, ToolRegistry
from tests._browser_fakes import close_generations
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


def _registry(tmp_path, monkeypatch, *, forked=False):
    """One task's registry; ``forked``: its execution drive is a child of the canonical (budget) one, as a fork's is."""
    repo, data = tmp_path / "repo", tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    for key, value in (("OUROBOROS_RUNTIME_MODE", "advanced"), ("OUROBOROS_SAFETY_MODE", "off"),
                       ("OUROBOROS_CONSCIOUSNESS_SAFETY_MODE", "off")):
        monkeypatch.setenv(key, value)
    # One synthetic custom secret the command may select, so the selected-secret masking is the real one.
    (data / "settings.json").write_text(json.dumps({"SYNTHETIC_RETAINED_SECRET": SECRET}), encoding="utf-8")
    monkeypatch.setattr(config, "SETTINGS_PATH", data / "settings.json")
    drive = data / "state" / "headless_tasks" / "retention-task" / "data" if forked else data
    drive.mkdir(parents=True, exist_ok=True)
    ctx = ToolContext(repo_dir=repo, drive_root=drive, budget_drive_root=str(data) if forked else "", task_id="retention-task")
    tools = ToolRegistry(repo_dir=repo, drive_root=drive)
    tools.set_context(ctx)
    return tools


@pytest.fixture
def registry(tmp_path, monkeypatch):
    return _registry(tmp_path, monkeypatch)


@pytest.fixture
def forked_registry(tmp_path, monkeypatch):
    return _registry(tmp_path, monkeypatch, forked=True)


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

    async def goto(self, *_args, **_kwargs):
        return None

    async def inner_text(self, _selector):
        return self.text

    async def content(self):
        return self.html

    async def title(self):
        return "large page"

    async def evaluate(self, _expression):
        return self.evaluated  # the markdown extractor and a caller's expression both come through here


def _browser_ctx(tmp_path):
    state = SimpleNamespace(last_screenshot_b64="", _browser_engine="chromium", _browser_device="")
    return SimpleNamespace(browser_state=state, repo_dir=tmp_path, drive_root=tmp_path, task_id="browser-retention")


def _bind_page(monkeypatch, page):
    monkeypatch.setattr(browser, "_ensure_browser", lambda ctx, *a, **k: (page, ctx.browser_state))
    monkeypatch.setattr(browser, "_readonly_subagent", lambda ctx: False)


@pytest.fixture
def _browser_ctx_closing(tmp_path):
    """The browser context of one test; the session loop its page ran on closes with the test."""
    ctx = _browser_ctx(tmp_path)
    yield ctx
    close_generations(ctx.browser_state)


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
def test_browse_page_output_is_whole_and_its_middle_and_tail_are_readable(tmp_path, monkeypatch, _browser_ctx_closing, output, total):
    ctx = _browser_ctx_closing
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


def test_browser_evaluate_output_is_whole_and_retained(tmp_path, monkeypatch, _browser_ctx_closing):
    ctx = _browser_ctx_closing
    text = _large(25_000)
    _bind_page(monkeypatch, _LargePage(evaluated=text))
    produced = browser._browser_action(ctx, "evaluate", value="document.body.innerText")
    assert text in produced and "... [truncated]" not in produced
    view, row, reader = _deliver_text(tmp_path, ctx, produced, window=8_000)
    assert row["result_partial"] is True and MIDDLE not in view
    source = artifacts.read_actor_source_bytes(tmp_path, ctx.task_id, row["result_source_ref"]).decode("utf-8")
    assert source == produced and MIDDLE in source and TAIL in source
    assert MIDDLE in _read_omitted(reader, view)


def test_small_browser_output_is_delivered_whole_and_unchanged(tmp_path, monkeypatch, _browser_ctx_closing):
    ctx = _browser_ctx_closing
    _bind_page(monkeypatch, _LargePage(text="small page text", html="<p>small</p>", evaluated="# small"))
    for output, expected in (("text", "small page text"), ("html", "<p>small</p>"), ("markdown", "# small")):
        produced = browser._browse_page(ctx, "http://example.test/", output=output)
        assert expected in produced and "[truncated]" not in produced
        view, row, _reader = _deliver_text(tmp_path, ctx, produced)
        assert view == produced and "result_partial" not in row


def test_an_unwritable_browser_source_is_disclosed_as_unavailable(tmp_path, monkeypatch, _browser_ctx_closing):
    def refuse(*_args, **_kwargs):
        raise OSError("disk refused the source")

    ctx = _browser_ctx_closing
    _bind_page(monkeypatch, _LargePage(text=_large(120_000)))
    produced = browser._browse_page(ctx, "http://example.test/", output="text")
    monkeypatch.setattr(artifacts, "store_actor_source_bytes", refuse)
    view, row, _reader = _deliver_text(tmp_path, ctx, produced)
    assert row["result_partial"] is True and row["result_source_status"] == "source_unavailable"
    assert FULL_SOURCE_UNAVAILABLE in view and "Do not treat this partial result as complete" in view
    assert MIDDLE not in view and FULL_SOURCE_MARKER not in view


# --- verify_and_record: a cut failing check keeps its whole masked output, readable without a rerun ---------------

TAIL_LINE = "LAST_STDERR_LINE_5e83"
TOKEN = "ghp_SYNTHETICfixtureToken0123456789abcd"  # a recognized conventional token shape; synthetic, never a real credential
CHECK_SCRIPT = ("import os, sys\n"
                "secret, token = os.environ['CHILD_SECRET'], 'ghp_' + {token!r}\n"  # built here: the command text never holds it whole
                "open(sys.argv[1], 'a').write('ran\\n')\n"  # one line per execution: a rerun would add a second
                "sys.stdout.write('a' * {lead} + secret + 'b' * {half} + ' ' + token + ' ' + 'b' * {half})\n"  # the secret crosses the 4 000-character cut
                "sys.stderr.write('E ' + secret + ' ' + token + ' ' + {tail!r})\n"
                "sys.exit({code})\n")


def _check_args(tmp_path, *, lead=3_990, half=10_350, code=3, expected=TAIL_LINE):
    counter = tmp_path / "runs.txt"
    script = CHECK_SCRIPT.format(lead=lead, half=half, token=TOKEN[4:], tail=TAIL_LINE, code=code)
    return {"contract_kind": "explicit_command", "check": [sys.executable, "-c", script, str(counter)],
            "expected": expected, "env_from_settings": {"CHILD_SECRET": "SYNTHETIC_RETAINED_SECRET"}}, counter


def _receipts(registry):
    from ouroboros.outcomes import read_verification_receipts

    return read_verification_receipts(registry._ctx.drive_root, registry._ctx.task_id)


def _reader_args(content: str, path: str) -> dict:
    """The arguments of the `read_file(...)` call the reply names, as the mind would send them."""
    call = re.search(r"read_file\(.*?max_chars=\d+\)", content).group(0)
    args = {kw.arg: ast.literal_eval(kw.value) for kw in ast.parse(call).body[0].value.keywords}
    assert args["path"] == path and args["root"] == "artifact_store"
    return args


@pytest.mark.parametrize(("code", "expected", "matched"), [(3, TAIL_LINE, True), (0, "ABSENT_SENTINEL", False)])
def test_a_cut_check_output_is_readable_whole_from_the_same_run(registry, tmp_path, code, expected, matched):
    args, counter = _check_args(tmp_path, code=code, expected=expected)  # over 24 600 masked characters, last stderr line past the cut
    errors, content, row = _run(registry, tmp_path, "verify_and_record", args)
    receipt = _receipts(registry)[-1]
    # Verdict, exit and matcher are the check's own, whichever of the two failed it: nonzero with a matching
    # matcher, or zero with a failing one. The handled FAIL is a completed call, not a tool error.
    assert errors == 0 and "FAIL: exit=%d" % code in content
    assert (receipt["status"], receipt["returncode"], receipt["matched"]) == ("fail", code, matched)
    view = content.split("Host-attested receipt recorded.\n\n", 1)[1]  # the header repeats `expected`, the output does not
    assert receipt["summary"].count("…[truncated") == 1 and TAIL_LINE not in receipt["summary"] and TAIL_LINE not in view.split("\nFull masked output")[0]
    # Nothing of the Settings secret survives in any cut view, though it straddles the inline cutoff.
    for shown in (content, receipt["summary"], json.dumps(receipt)):
        assert SECRET not in shown and SECRET[:8] not in shown and TOKEN not in shown
    source = receipt["output_source"]
    assert set(source) == {"kind", "root", "path", "size", "sha256"} and source["root"] == "artifact_store"
    # The reply names the real read_file call; reading its last page returns the sentinel the view cut off.
    page = registry.execute("read_file", _reader_args(content, source["path"]))
    assert TAIL_LINE in page and SECRET not in page and TOKEN not in page
    # The retained body is the whole masked capture, stdout then stderr, untrimmed; the check ran exactly once.
    body = artifacts.read_actor_source_bytes(registry._ctx.drive_root, registry._ctx.task_id, {**source, "kind": "task_source"}).decode()
    assert len(body) == source["size"] > 24_600 and body.startswith("a" * 3_990 + "***")
    # The Settings value and the recognized conventional token are masked in the retained copy as in the views.
    assert SECRET not in body and TOKEN not in body and body.count("***REDACTED***") == 2
    assert body.replace("***REDACTED***", "").count("***") == 2 and "\nE *** ***REDACTED*** " + TAIL_LINE in body
    assert counter.read_text().splitlines() == ["ran"]


def test_a_short_check_output_stores_no_source(registry, tmp_path):
    args, _counter = _check_args(tmp_path, lead=100, half=100)
    _errors, content, _row = _run(registry, tmp_path, "verify_and_record", args)
    receipt = _receipts(registry)[-1]
    assert TAIL_LINE in content and "retained" not in content and "truncated" not in content
    assert not {"output_source", "output_source_unavailable"} & set(receipt)
    assert not (registry._ctx.drive_root / "task_results" / "artifacts" / registry._ctx.task_id / "source_handles").exists()


def test_a_check_output_between_the_two_bounds_is_cut_in_the_reply_only(registry, tmp_path):
    args, _counter = _check_args(tmp_path, half=2_000)  # about 8 000 masked characters: over the reply's 4 000, under the receipt's 20 000
    _errors, content, _row = _run(registry, tmp_path, "verify_and_record", args)
    receipt = _receipts(registry)[-1]
    # The reply is cut and links the whole text; the receipt summary is not cut, so it holds the last stderr line itself.
    view = content.split("Host-attested receipt recorded.\n\n", 1)[1].split("\nFull masked output")[0]
    assert view.count("…[truncated") == 1 and TAIL_LINE not in view
    assert "…[truncated" not in receipt["summary"] and receipt["summary"].endswith(TAIL_LINE)
    assert receipt["output_source"]["size"] == len(receipt["summary"])  # uncut: the retained copy and the summary are one text
    assert TAIL_LINE in registry.execute("read_file", _reader_args(content, receipt["output_source"]["path"]))


def _finished(monkeypatch, *, stdout="", stderr="", code=3):
    """Replace only the child process: the command already returned, so masking, receipt and reply stay the real handlers."""
    from ouroboros.tools import shell

    monkeypatch.setattr(shell, "_tracked_subprocess_run",
                        lambda cmd, **_kwargs: SimpleNamespace(stdout=stdout, stderr=stderr, returncode=code))


def _fake_capture(*, lead=3_990, half=10_350):
    """What ``CHECK_SCRIPT`` writes, built in the test: the Settings secret crosses the 4 000-character cut."""
    return ("a" * lead + SECRET + "b" * half + " " + TOKEN + " " + "b" * half, "E " + SECRET + " " + TOKEN + " " + TAIL_LINE)


@pytest.mark.parametrize("failing", ["store_actor_source_bytes", "read_actor_source_bytes"])  # the write, and its read-back
def test_an_unverified_check_source_never_names_an_address(registry, tmp_path, monkeypatch, failing):
    def refuse(*_args, **_kwargs):
        raise OSError("disk refused the source")

    monkeypatch.setattr(artifacts, failing, refuse)
    _finished(monkeypatch, **dict(zip(("stdout", "stderr"), _fake_capture())))
    args, _counter = _check_args(tmp_path)
    errors, content, _row = _run(registry, tmp_path, "verify_and_record", args)
    receipt = _receipts(registry)[-1]
    assert errors == 0 and "FAIL: exit=3" in content and (receipt["status"], receipt["returncode"]) == ("fail", 3)
    # Only what the host verified is promised: no address, and no claim about bytes or a rerun; the receipt keeps its 20 000.
    assert "No verified address for the full output" in content and "read_file(" not in content
    assert "source_handles" not in content and "not retained" not in content and "again" not in content
    assert "output_source" not in receipt and receipt["output_source_unavailable"] == "source_unavailable"
    assert receipt["summary"].count("…[truncated") == 1 and len(receipt["summary"]) > 20_000


def _selected_path_value(registry, tmp_path, monkeypatch, selected):
    (registry._ctx.drive_root / "settings.json").write_text(json.dumps(
        {"SYNTHETIC_RETAINED_SECRET": SECRET, "SYNTHETIC_PATH_SECRET": selected}), encoding="utf-8")
    _finished(monkeypatch, **dict(zip(("stdout", "stderr"), _fake_capture())))
    args, _counter = _check_args(tmp_path)
    args["env_from_settings"]["PATH_SECRET"] = "SYNTHETIC_PATH_SECRET"
    _errors, content, _row = _run(registry, tmp_path, "verify_and_record", args)
    return content, _receipts(registry)[-1]


def test_an_address_a_selected_value_would_mask_is_not_offered(registry, tmp_path, monkeypatch):
    content, receipt = _selected_path_value(registry, tmp_path, monkeypatch, "source_handles")  # a value inside every address
    # The registry wrapper masks the reply and the receipt again; an address it would alter is never shown or stored.
    assert "read_file(" not in content and "source_handles" not in content.replace("***", "") and "output_source" not in receipt
    assert receipt["output_source_unavailable"] == "address_masked" and receipt["returncode"] == 3
    assert not (registry._ctx.drive_root / "task_results" / "artifacts" / registry._ctx.task_id / "source_handles").exists()


def test_a_selected_value_only_in_the_prose_does_not_discard_the_source(registry, tmp_path, monkeypatch):
    # "run" is in "same run" and "rerunning" of the note, not in the read_file call: the prose is masked for display
    # by the wrapper, the address is intact, and the source is retained and named.
    content, receipt = _selected_path_value(registry, tmp_path, monkeypatch, "run")
    assert "output_source_unavailable" not in receipt and receipt["output_source"]["root"] == "artifact_store"
    assert "***" in content and TAIL_LINE in registry.execute("read_file", _reader_args(content, receipt["output_source"]["path"]))


@pytest.mark.parametrize("selected", ["Bearer", "ghp_"])  # an auth scheme and a token prefix chosen as Settings values
def test_a_selected_value_equal_to_a_token_prefix_does_not_hide_the_token(registry, tmp_path, monkeypatch, selected):
    # Conventional shapes are recognized on the ORIGINAL text; masking the prefix first would leave the token bare.
    bearer = "bearerTOKEN0123456789abcdefghijklmnop"
    (registry._ctx.drive_root / "settings.json").write_text(json.dumps({"SYNTHETIC_PREFIX": selected}), encoding="utf-8")
    _finished(monkeypatch, stdout=("x" * 5_000 + f" Authorization: Bearer {bearer} {TOKEN} " + "y" * 100), stderr=TAIL_LINE)
    _errors, content, _row = _run(registry, tmp_path, "verify_and_record", {
        "contract_kind": "explicit_command", "check": ["true"], "env_from_settings": {"PREFIX": "SYNTHETIC_PREFIX"}})
    source = _receipts(registry)[-1]["output_source"]
    body = artifacts.read_actor_source_bytes(registry._ctx.drive_root, registry._ctx.task_id, {**source, "kind": "task_source"}).decode()
    assert bearer not in body and TOKEN not in body and TOKEN[4:] not in body and selected not in body and TAIL_LINE in body


def _views_of(registry, raw, *, captured=False):
    """``check_output_views`` over ``raw``, then the page the named ``read_file`` call returns."""
    from ouroboros.tools.process_facts import check_output_views

    _summary, reply, retention = check_output_views(registry._ctx, raw, captured=captured)
    reader = _reader_args(reply, retention["output_source"]["path"])
    return reply, reader, registry.execute("read_file", reader)


def test_a_whitespace_only_long_capture_is_retained_and_labelled_captured(registry):
    reply, reader, page = _views_of(registry, "\n" * 5_000, captured=True)
    assert reply.startswith("Captured masked output:") and "No captured output" not in reply and "Captured masked output of this same run (5000 chars" in reply
    assert reader["start_char"] >= 0 and reader["start_line"] >= 1 and "NOT_FOUND" not in page


def test_the_tail_locator_points_at_the_real_end_past_trailing_whitespace(registry):
    reply, reader, page = _views_of(registry, "q" * 5_000 + " END_OF_OUTPUT" + "\n" * 6_000)
    assert "END_OF_OUTPUT" in page and reader["start_char"] >= 0


def test_the_tail_locator_of_crlf_output_is_never_negative(registry):
    # 1 001 lines: 5 008 characters as written, 4 008 once read_file folds CRLF; the locator uses the folded text.
    reply, reader, page = _views_of(registry, "abc\r\n" * 1_000 + "END_ROW\r\n")
    assert reader["start_char"] >= 0 and "END_ROW" in page and "(4008 chars" in reply


def test_crlf_output_that_fits_once_folded_is_shown_whole_without_a_source(registry):
    from ouroboros.tools.process_facts import check_output_views

    _summary, reply, retention = check_output_views(registry._ctx, "ab\r\n" * 1_100 + "END_ROW")  # 4 407 as written, 3 307 folded
    assert retention == {} and reply.endswith("END_ROW") and "truncated" not in reply


def test_the_reply_is_whole_and_trimmed_at_the_threshold_and_cut_with_its_whitespace_one_character_past_it(registry):
    from ouroboros.tools.process_facts import check_output_views

    whole = check_output_views(registry._ctx, "\n  " + "k" * 3_997)  # 4 000 characters: the reply's whole capacity
    assert whole == ("k" * 3_997, "k" * 3_997, {})
    summary, reply, retention = check_output_views(registry._ctx, "\n  " + "k" * 3_998)  # one more
    assert reply.startswith("\n  kk") and reply.count("…[truncated 1 of 4001 chars]") == 1 and "output_source" in retention
    assert summary == "\n  " + "k" * 3_998  # the receipt's bound is a different one: not cut, nothing of it trimmed
    assert "(4001 chars" in reply


def test_separate_secrets_keep_the_innocent_text_between_them(registry, tmp_path, monkeypatch):
    # A selected value and a recognized token that do not overlap are masked each by itself: the words between survive.
    _finished(monkeypatch, stdout="x" * 5_000 + f" {SECRET} between these two, quite ordinary words {TOKEN} after", stderr=TAIL_LINE)
    _errors, content, _row = _run(registry, tmp_path, "verify_and_record", {
        "contract_kind": "explicit_command", "check": ["true"], "env_from_settings": {"CHILD_SECRET": "SYNTHETIC_RETAINED_SECRET"}})
    source = _receipts(registry)[-1]["output_source"]
    body = artifacts.read_actor_source_bytes(registry._ctx.drive_root, registry._ctx.task_id, {**source, "kind": "task_source"}).decode()
    assert SECRET not in body and TOKEN not in body and " between these two, quite ordinary words " in body and body.endswith(TAIL_LINE)


@pytest.mark.parametrize(("ended", "known"), [
    ("stop", True),  # Stop now handed back the text the check had written when it ended
    ("stop", False),  # Stop now with nothing captured
    ("timeout", True),  # a timeout carries what the child had flushed (a command can write stderr before it hangs)
    ("timeout", False),  # a timeout with no bytes
])
def test_an_interrupted_check_shows_only_what_was_captured_and_claims_nothing_else(registry, tmp_path, monkeypatch, ended, known):
    import subprocess

    from ouroboros.tools import shell
    from ouroboros.tools.shell_process import CommandStopped

    def interrupted(cmd, **_kwargs):
        if ended == "timeout":
            raise subprocess.TimeoutExpired(cmd, 5, output=_large(30_000).encode() if known else None,
                                            stderr=TAIL_LINE.encode() if known else b"")
        raise CommandStopped(cmd, _large(30_000) if known else "", TAIL_LINE if known else "", exit_observed=True)

    monkeypatch.setattr(shell, "_tracked_subprocess_run", interrupted)
    _errors, content, _row = _run(registry, tmp_path, "verify_and_record", {"contract_kind": "explicit_command", "check": ["true"]})
    receipt = _receipts(registry)[-1]
    assert receipt["status"] == "fail" and receipt["returncode"] is None and receipt["matched"] is None
    if known:
        assert "\n\nCaptured masked output:\n" in content and "Captured masked output of this same run" in content
        assert receipt["summary"].split("\n", 1)[1].startswith("Captured masked output:\n")
        page = registry.execute("read_file", _reader_args(content, receipt["output_source"]["path"]))
        assert TAIL_LINE in page and receipt["output_source"]["size"] > 30_000
    else:  # no bytes: no source, and the reply says so rather than showing an empty or complete-looking view
        assert content.endswith("No captured output is available.") and receipt["summary"].endswith("No captured output is available.")
        assert "read_file(" not in content
        assert not {"output_source", "output_source_unavailable"} & set(receipt)


def test_a_stopped_check_whose_receipt_cannot_be_written_still_names_its_address(registry, tmp_path, monkeypatch):
    from ouroboros.tool_access import canonical_data_root
    from ouroboros.tools import shell, verify
    from ouroboros.tools.shell_process import CommandStopped

    def stopped(cmd, **_kwargs):
        raise CommandStopped(cmd, _large(30_000), TAIL_LINE, exit_observed=True)

    monkeypatch.setattr(shell, "_tracked_subprocess_run", stopped)
    monkeypatch.setattr(verify, "append_verification_receipt", lambda *_a, **_k: False)
    _errors, content, _row = _run(registry, tmp_path, "verify_and_record", {"contract_kind": "explicit_command", "check": ["true"]})
    # No receipt was recorded and the call says so; the source the host already verified is still named, not a rerun offered.
    assert "TOOL_ERROR (verify_and_record): receipt_custody_failed" in content and "Do not treat this verification as attested" in content
    assert _receipts(registry) == []
    path = re.search(r"path='([^']+)'", content).group(1)
    assert TAIL_LINE in registry.execute("read_file", _reader_args(content, path)) and "again" not in content.split("Observed result:")[0]
    stored = artifacts.task_artifact_dir_path(canonical_data_root(registry._ctx), registry._ctx.task_id) / path
    assert hashlib.sha256(stored.read_bytes()).hexdigest() == re.search(r"sha256 ([0-9a-f]{64})", content).group(1)


@pytest.mark.parametrize("size", [3_000, 4_000, 4_001, 21_000])  # whole in the reply; at its limit; one past it; past the receipt's bound too
def test_bytes_equal_note_is_shown_beside_the_process_output_never_in_it(registry, tmp_path, monkeypatch, size):
    (tmp_path / "repo" / "a.bin").write_bytes(b"same")
    (tmp_path / "repo" / "b.bin").write_bytes(b"other")
    out = "\n  " + "a" * (size - 3 - len(TAIL_LINE)) + TAIL_LINE  # leading whitespace is part of what the process wrote
    _finished(monkeypatch, stdout=out, code=0)
    result = registry.execute_result("verify_and_record", {
        "contract_kind": "explicit_command", "check": ["true"], "expected_match": "bytes_equal", "artifact_paths": ["a.bin", "b.bin"]})
    receipt = _receipts(registry)[-1]
    view = result.text.split("Host-attested receipt recorded.\n\n", 1)[1]
    # The note is shown, whatever the output's length; it takes nothing from it and is in no durable copy of it.
    assert "[bytes_equal] bytes differ at offset 0 (sizes 4 vs 5).\na.bin @0: 73 61 6d 65\nb.bin @0: 6f 74 68 65 72" in view
    assert "[bytes_equal]" not in receipt["summary"] and (size > 20_000 or receipt["summary"].endswith(TAIL_LINE))
    # The durable receipt keeps the diagnostic too, as its own key (#3844 r3): offset and hexdump, never in the summary.
    assert receipt["comparison_detail"].startswith("bytes differ at offset 0 (sizes 4 vs 5).\na.bin @0: 73 61 6d 65")
    assert result.meta["verification"]["detail"].startswith("bytes differ at offset 0") and receipt["matched"] is False
    if size <= 4_000:
        assert out.strip() in view and "truncated" not in view and "output_source" not in receipt
        return
    source = receipt["output_source"]
    body = artifacts.read_actor_source_bytes(registry._ctx.drive_root, registry._ctx.task_id, {**source, "kind": "task_source"}).decode()
    assert body == out and source["size"] == len(out)
    assert receipt["summary"] == out if size <= 20_000 else receipt["summary"] == out[:20_000] + f"\n…[truncated {size - 20_000} of {size} chars]"
    assert view.index("read_file(") < view.index("[bytes_equal]") and TAIL_LINE in registry.execute(
        "read_file", _reader_args(result.text, source["path"]))


def test_the_bytes_equal_detail_is_masked_where_it_is_shown_and_recorded(registry, tmp_path, monkeypatch):
    names = (f"{TOKEN}-a.bin", f"{TOKEN}-b.bin")  # a conventional token shape in the operands' own names
    for name, data in zip(names, (b"same", b"other")):
        (tmp_path / "repo" / name).write_bytes(data)
    _finished(monkeypatch, stdout="ok", code=0)
    result = registry.execute_result("verify_and_record", {
        "contract_kind": "explicit_command", "check": ["true"], "expected_match": "bytes_equal", "artifact_paths": list(names)})
    assert "[bytes_equal] bytes differ at offset 0" in result.text
    receipt = _receipts(registry)[-1]
    assert receipt["comparison_detail"].startswith("bytes differ at offset 0")
    for shown in (result.text, result.meta["verification"]["detail"], receipt["summary"], receipt["comparison_detail"]):
        assert TOKEN not in shown and TOKEN[4:] not in shown


def test_the_tail_locator_of_a_many_line_output_is_addressed_by_line(registry, tmp_path):
    # read_file counts start_char inside its max_lines window (2 000 by default): a whole-text offset would miss the end.
    script = ("import os, sys\n"
              "for n in range(6000): print('row', n, os.environ['CHILD_SECRET'] if n % 2999 == 0 else '')\n"
              f"sys.stderr.write('E {TAIL_LINE}')\nsys.exit(1)\n")
    args = {"contract_kind": "explicit_command", "check": [sys.executable, "-c", script],
            "env_from_settings": {"CHILD_SECRET": "SYNTHETIC_RETAINED_SECRET"}}
    _errors, content, _row = _run(registry, tmp_path, "verify_and_record", args)
    source = _receipts(registry)[-1]["output_source"]
    reader = _reader_args(content, source["path"])
    assert reader["start_line"] > 2_000 and reader["max_chars"] == 4_000
    page = registry.execute("read_file", reader)
    assert TAIL_LINE in page and "row 5999" in page and "row 1 " not in page and SECRET not in page
    assert "row 0" in registry.execute("read_file", {"root": "artifact_store", "path": source["path"], "max_chars": 200})


def test_a_forks_cut_output_is_retained_where_its_own_tools_read_it(forked_registry, tmp_path):
    """A fork's receipt stays on its execution drive, where custody promotes no ``source_handles``; the retained
    output is written once beside the budget drive's task store, which ``read_file(root='artifact_store')`` reaches
    by ``retained_actor_source_read_target`` from any execution drive of the same task, and from no other task's."""
    import shutil

    from ouroboros.outcomes import verification_receipts_path
    from ouroboros.tool_access import canonical_data_root, resource_root_path

    ctx = forked_registry._ctx
    canonical, task, old_drive = canonical_data_root(ctx), ctx.task_id, ctx.drive_root
    assert canonical != old_drive.resolve() and canonical == (tmp_path / "data").resolve()
    own_store, canonical_store = resource_root_path(ctx, "artifact_store"), artifacts.task_artifact_dir_path(canonical, task)
    assert own_store != canonical_store.resolve()
    args, counter = _check_args(tmp_path)
    errors, content, _row = _run(forked_registry, tmp_path, "verify_and_record", args)
    receipt = _receipts(forked_registry)[-1]
    source = receipt["output_source"]
    assert errors == 0 and receipt["returncode"] == 3
    # The receipt is on the execution drive, the source on the canonical one: nothing of it sits in the fork's own store.
    assert verification_receipts_path(old_drive, task).is_file() and not verification_receipts_path(canonical, task).exists()
    assert (canonical_store / source["path"]).is_file() and not (own_store / "source_handles").exists()
    reader = _reader_args(content, source["path"])
    assert TAIL_LINE in forked_registry.execute("read_file", reader)
    # The execution drive retires. A new registry on a separate, empty drive of the same task and canonical root runs
    # the exact recipe the reply returned: it reads the verified bytes, from the one canonical copy; the check ran once.
    shutil.rmtree(old_drive)
    fresh_drive = canonical / "state" / "headless_tasks" / "retention-task-retry" / "data"
    fresh_drive.mkdir(parents=True)
    fresh = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=fresh_drive)
    fresh.set_context(ToolContext(repo_dir=ctx.repo_dir, drive_root=fresh_drive, budget_drive_root=str(canonical), task_id=task))
    assert not any(fresh_drive.iterdir()) and not old_drive.exists()
    again = fresh.execute("read_file", reader)
    assert TAIL_LINE in again and TOKEN not in again and SECRET not in again
    # The address is the task's own: the same recipe from another task of this canonical root, with no lineage to this
    # one, resolves in that task's artifact_store and finds nothing; the owner's reader is unchanged by that miss.
    other_drive = canonical / "state" / "headless_tasks" / "unrelated-task" / "data"
    other_drive.mkdir(parents=True)
    other = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=other_drive)
    other.set_context(ToolContext(repo_dir=ctx.repo_dir, drive_root=other_drive, budget_drive_root=str(canonical), task_id="unrelated-task"))
    missed = other.execute("read_file", reader)
    assert "NOT_FOUND" in missed and TAIL_LINE not in missed and TOKEN not in missed and SECRET not in missed
    assert TAIL_LINE in fresh.execute("read_file", reader)
    assert not (fresh_drive / "task_results").exists() and hashlib.sha256((canonical_store / source["path"]).read_bytes()).hexdigest() == source["sha256"]
    assert counter.read_text().splitlines() == ["ran"]


def test_an_executor_cmp_note_is_masked_whole_before_its_cut(registry, tmp_path, monkeypatch):
    from ouroboros import workspace_executor
    from ouroboros.tools import verify
    from ouroboros.tools.process_facts import masked_process_text

    cmp_out = "x" * 390 + TOKEN + " differ: byte 1, line 1"  # the token crosses the note's 400-character cut
    monkeypatch.setattr(verify, "_bytes_equal_confinement_block", lambda *_a, **_k: "")
    monkeypatch.setattr(workspace_executor, "execute", lambda *_a, **_k: SimpleNamespace(returncode=1, stdout=cmp_out, stderr=""))
    matched, detail = verify._compare_files_bytes_equal(registry._ctx, ["a.bin", "b.bin"], tmp_path, use_executor=True)
    shown = masked_process_text(detail)  # the caller's one masked text for the note, the metadata and the receipt
    # Masked whole first: a prefix the cut leaves is never shorter than the token shape, so no fragment survives it.
    assert matched is False and "x" * 390 in shown and TOKEN[:8] not in shown and TOKEN[:8] not in detail


def test_a_withheld_reader_keeps_the_source_and_says_why_it_is_not_offered(registry, monkeypatch):
    from ouroboros.tools.process_facts import check_output_views

    monkeypatch.setattr(registry._ctx, "task_metadata", {"disabled_tools": ["read_file"]})
    _summary, reply, retention = check_output_views(registry._ctx, _large(30_000))
    source = retention["output_source"]  # retained and verified all the same: another reader may open it
    assert "read_file(" not in reply and "withholds 'read_file'" in reply and source["path"] in reply
    stored = artifacts.read_actor_source_bytes(registry._ctx.drive_root, registry._ctx.task_id, {**source, "kind": "task_source"})
    assert hashlib.sha256(stored).hexdigest() == source["sha256"]


def test_a_raw_json_line_break_keeps_the_receipt_one_readable_row(registry, tmp_path, monkeypatch):
    out = "a" * 4_100 + "\u2028" + TAIL_LINE + "\u2029\x85"  # long, so unstripped; JSON writes these raw, splitlines splits
    _finished(monkeypatch, stdout=out, code=0)
    _run(registry, tmp_path, "verify_and_record", {"contract_kind": "explicit_command", "check": ["true"]})
    receipts = _receipts(registry)  # read_receipts drops a row a raw separator splits
    assert len(receipts) == 1 and receipts[0]["status"] == "pass" and f"\n{TAIL_LINE}\n" in receipts[0]["summary"]
    assert not set("\u2028\u2029\x85") & set(receipts[0]["summary"])
    source = {**receipts[0]["output_source"], "kind": "task_source"}
    assert artifacts.read_actor_source_bytes(registry._ctx.drive_root, registry._ctx.task_id, source).decode() == out


def test_a_timeout_whose_wait_returned_no_stream_says_its_output_is_unknown(registry, tmp_path, monkeypatch):
    import subprocess

    from ouroboros.tools import shell

    def timed_out(cmd, **_kwargs):
        raise subprocess.TimeoutExpired(cmd, 5)  # as the Stop-aware wait and Windows raise it: neither stream attached

    monkeypatch.setattr(shell, "_tracked_subprocess_run", timed_out)
    _errors, content, _row = _run(registry, tmp_path, "verify_and_record", {"contract_kind": "explicit_command", "check": ["true"]})
    receipt = _receipts(registry)[-1]
    assert receipt["status"] == "fail" and receipt["returncode"] is None and receipt["matched"] is None
    for text in (content, receipt["summary"]):  # unknown, never "nothing captured"
        assert "whether it wrote any is unknown" in text and "No captured output" not in text
    assert "read_file(" not in content and not {"output_source", "output_source_unavailable"} & set(receipt)


# --- the offered call is this task's own dispatch: Presence bindings and grants, consulted once, never executed -----

def _ceiling(read_file=True, bindings=(), roots=("artifact_store",)):
    """A synthetic Presence ceiling: ``read_file`` granted (with ``bindings``) or not, and read grants over ``roots``."""
    from ouroboros.presence_authority import (PresenceArgumentBinding, PresenceCapabilityCeiling, PresenceResourceGrant,
                                              PresenceToolGrant, presence_ceiling_payload)

    zero = "0" * 64
    provisional = PresenceCapabilityCeiling(
        skill_name="synthetic-presence", skill_content_hash=zero, profile_fingerprint=zero, state_fingerprint=zero,
        selection_fingerprint=zero, model_slot="main", inline_max_rounds=1,
        tool_grants=(PresenceToolGrant("read_file", tuple(PresenceArgumentBinding((key,), "static", static_value=value)
                                                          for key, value in bindings)),) if read_file else (),
        resource_grants=tuple(PresenceResourceGrant(root, ("read",), ".") for root in roots), digest=zero)
    return presence_ceiling_payload(provisional)


# (stdout, stderr) of a stubbed check; the final marker is the check's own stderr, joined after "\n" as verify does.
THREE_LINES = ("a" * 12_000 + "\n" + "b" * 12_000, TAIL_LINE)  # the natural reader starts at line 2
STARTS_AT_ONE = ("a" * 3_500 + "\n" + "b" * 1_400, TAIL_LINE)  # the natural reader starts at line 1


def _check_under(registry, tmp_path, monkeypatch, ceiling, streams=THREE_LINES):
    """A cut failing check through the real handler, receipt and reply; only the command's result is stubbed. The
    ceiling reaches the context when the check returns, so dispatch admits verify_and_record and what is judged is the
    view's own consultation of the read_file call it would offer. Returns the executions list too."""
    from ouroboros.tools import shell

    calls = []
    stdout, stderr = streams

    def finished(cmd, **_kwargs):
        calls.append(cmd)
        registry._ctx.task_contract = {"capability_ceiling": ceiling} if ceiling is not None else {}
        return SimpleNamespace(returncode=3, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(shell, "_tracked_subprocess_run", finished)
    errors, content, _row = _run(registry, tmp_path, "verify_and_record",
                                 {"contract_kind": "explicit_command", "check": ["synthetic-check"], "expected": TAIL_LINE})
    receipt = _receipts(registry)[-1]
    assert errors == 0 and len(calls) == 1 and (receipt["status"], receipt["returncode"]) == ("fail", 3)
    source = receipt["output_source"]  # kept whatever the reader's fate: verified bytes of this run
    stored = artifacts.read_actor_source_bytes(registry._ctx.drive_root, registry._ctx.task_id, {**source, "kind": "task_source"})
    assert stored.decode() == stdout + "\n" + stderr and hashlib.sha256(stored).hexdigest() == source["sha256"]
    assert source["path"] in content
    return content, receipt, calls


@pytest.mark.parametrize(("ceiling", "code", "said"), [
    # A static binding retargets the reader to another file: the call would read FROM_STATUS_FILE, not this tail.
    (_ceiling(bindings=(("root", "active_workspace"), ("path", "status.txt")), roots=("active_workspace", "artifact_store")),
     "target_changed", "read_file resolves active_workspace:status.txt to another file."),
    # The range changes and that window misses the end: an earlier start, one line, or an empty cursor past it all.
    (_ceiling(bindings=(("start_line", 1),)), "presence_window", "set start_line for read_file; that window (lines 1-2"),
    (_ceiling(bindings=(("max_lines", 1),)), "presence_window", "set max_lines for read_file; that window (lines 2-2"),
    (_ceiling(bindings=(("start_char", 10**6),)), "presence_window", "set start_char for read_file; that window"),
    (_ceiling(read_file=False), "presence_refused", "PRESENCE_CAPABILITY_BLOCKED"),
    (_ceiling(roots=("active_workspace",)), "read_refused", "PRESENCE_RESOURCE_BLOCKED"),
    # A root label the admission itself rejects: the refusal is the handler's own, the retained source still stands.
    (_ceiling(bindings=(("root", "not_a_root"),)), "read_refused", "unknown root 'not_a_root'"),
])
def test_a_reader_this_task_would_rebind_or_refuse_is_not_offered(registry, tmp_path, monkeypatch, ceiling, code, said):
    content, receipt, calls = _check_under(registry, tmp_path, monkeypatch, ceiling)
    assert "read_file(" not in content and receipt["output_source_call_unavailable"] == code
    assert "cannot read its end through the named call: " in content and said in content
    assert "output_source_unavailable" not in receipt and TAIL_LINE not in receipt["summary"] and len(calls) == 1


# The cursor the view proposes for THREE_LINES: 4 000 characters before the end, counted inside line 2's window.
NATURAL_START_CHAR = len(THREE_LINES[0] + "\n" + THREE_LINES[1]) - 4_000 - len("a" * 12_000 + "\n")


def _guidance(content):
    """The one paging sentence after the offered call: its effective start_line, qualified by this task's bindings."""
    return re.search(r"max_chars=\d+\); ([^\n]*)", content).group(1).strip()


def test_a_presence_that_grants_the_exact_reader_is_offered_the_unchanged_call(registry, tmp_path, monkeypatch):
    content, receipt, _calls = _check_under(registry, tmp_path, monkeypatch, _ceiling())
    assert "output_source_call_unavailable" not in receipt
    assert _reader_args(content, receipt["output_source"]["path"])["max_chars"] == 4000
    guidance = _guidance(content)
    assert "start_line=2" in guidance and "only where this task's bindings permit" in guidance


@pytest.mark.parametrize(("bindings", "streams", "shown"), [
    # A wider line window still holds the end: offered with the bound max_lines.
    ((("max_lines", 2000),), THREE_LINES, {"start_line": 2, "max_lines": 2000}),
    # Natural start 1, start_line overwritten to 2: the same cursor still reaches the end, which starts at line 2.
    ((("start_line", 2),), STARTS_AT_ONE, {"start_line": 2, "max_lines": 3}),
    # The natural start is already 2: a start pinned there changes nothing and never advertises line 1.
    ((("start_line", 2),), THREE_LINES, {"start_line": 2, "max_lines": 2}),
    # A cursor pinned to exactly the proposed nonzero value: the call is unchanged, yet no smaller cursor is free.
    ((("start_char", NATURAL_START_CHAR),), THREE_LINES, {"start_line": 2, "start_char": NATURAL_START_CHAR}),
    # A pinned zero cursor with a budget that holds the window.
    ((("start_char", 0), ("max_chars", 30_000)), THREE_LINES, {"start_line": 2, "start_char": 0, "max_chars": 30_000}),
], ids=["wider_max_lines", "start_line_moved_to_2", "start_line_pinned_at_natural_2", "start_char_pinned_at_natural",
        "pinned_start_char"])
def test_a_binding_that_keeps_the_end_in_its_window_is_offered_its_effective_call(
        registry, tmp_path, monkeypatch, bindings, streams, shown):
    content, receipt, calls = _check_under(registry, tmp_path, monkeypatch, _ceiling(bindings=bindings), streams=streams)
    assert "output_source_call_unavailable" not in receipt
    reader = _reader_args(content, receipt["output_source"]["path"])
    assert {key: reader[key] for key in shown} == shown
    # The guidance names the window's effective start and promises no other cursor or line: any change is the bindings'.
    guidance = _guidance(content)
    assert f"start_line={reader['start_line']}" in guidance and "only where this task's bindings permit" in guidance
    assert "smaller" not in guidance and "start_line=1" not in content and "line 1" not in content
    page = registry.execute("read_file", reader)  # the real advertised read, through this task's own bound dispatch
    assert page.rstrip().endswith(TAIL_LINE) and "a" * 100 not in page
    assert len(calls) == 1  # still one execution after the read: no rerun


# --- R5: fail-closed mask, controlled omission codes, root aliases, real entry preparation -----------------------


def test_an_injected_masking_fault_keeps_the_true_receipt_and_withholds_text_and_ref(registry, tmp_path, monkeypatch):
    from ouroboros import secret_masking
    from ouroboros.tools import shell

    monkeypatch.setattr(shell, "_tracked_subprocess_run",
                        lambda cmd, **_k: SimpleNamespace(stdout="a" * 5_000 + SECRET, stderr="E " + TAIL_LINE, returncode=3))
    # An ORDINARY failure at the finish of the selected walk (masked_process_text's own ``walk.finish`` call — its
    # only production caller), injected in this unit fixture only: the composition seam must turn ANY fault into
    # the whole-text refusal, not merely pass a pre-wrapped MaskingFailed along the path that already raised it.
    monkeypatch.setattr(secret_masking.MaskWalk, "finish",
                        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("injected")))
    args, _counter = _check_args(tmp_path)
    errors, content, _row = _run(registry, tmp_path, "verify_and_record", args)
    receipt = _receipts(registry)[-1]
    # The true FAIL receipt survives: the check's own exit, verdict and matcher; only the text and the ref are withheld.
    assert errors == 0 and (receipt["status"], receipt["returncode"], receipt["matched"]) == ("fail", 3, True)
    assert receipt["output_source_unavailable"] == "masking_failed" and "output_source" not in receipt  # the verbatim code
    for shown in (content, receipt["summary"]):  # no old or partial text, and no address, in any durable or shown copy
        assert "a" * 100 not in shown and SECRET not in shown and "read_file(" not in shown and "source_handles" not in shown
    assert "no output source was retained" in receipt["summary"] and "no output source was retained" in content


def test_a_selected_value_equal_to_an_omission_code_does_not_rewrite_the_code(registry, tmp_path, monkeypatch):
    from ouroboros import secret_masking
    from ouroboros.tools import shell

    (registry._ctx.drive_root / "settings.json").write_text(json.dumps(
        {"SYNTHETIC_RETAINED_SECRET": SECRET, "SYNTHETIC_CODE_SECRET": "masking_failed"}), encoding="utf-8")
    monkeypatch.setattr(shell, "_tracked_subprocess_run",
                        lambda cmd, **_k: SimpleNamespace(stdout="a" * 5_000 + SECRET, stderr="E " + TAIL_LINE, returncode=3))
    monkeypatch.setattr(secret_masking.MaskWalk, "finish",
                        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("injected")))
    args, _counter = _check_args(tmp_path)
    args["env_from_settings"]["CODE_SECRET"] = "SYNTHETIC_CODE_SECRET"
    _errors, content, _row = _run(registry, tmp_path, "verify_and_record", args)
    receipt = _receipts(registry)[-1]
    # The controlled code round-trips unchanged in the durable row (and the typed result); the same value
    # in the shown prose is masked by the wrapper.
    assert receipt["output_source_unavailable"] == "masking_failed" and "output_source" not in receipt
    assert (receipt["status"], receipt["returncode"], receipt["matched"]) == ("fail", 3, True)
    assert "masking_failed" not in content


def test_a_root_alias_admitting_the_same_physical_file_is_still_offered(registry):
    """The same file under another admitted label is still this run's file: the task's own retained output also
    lives beneath its ``runtime_data`` root, and this Presence ceiling granted both spellings and pinned the
    aliased one (its label padded, so the normalized selector the admission resolved must be what is advertised).
    The REAL ``read_admission`` resolves alias and artifact address to one physical file, the offer stands, and
    the REAL registry executing the printed call returns this run's tail."""
    from ouroboros.tools.process_facts import check_output_views, masked_process_text

    raw = "a" * 30_000 + "\n" + TAIL_LINE
    planned = artifacts.actor_source_reference(category="tool_results", source_id="verify_and_record-output",
                                               data=masked_process_text(raw).encode("utf-8"), extension="txt")
    alias = f"task_results/artifacts/{registry._ctx.task_id}/{planned['path']}"
    registry._ctx.task_contract = {"capability_ceiling": _ceiling(
        bindings=(("root", " runtime_data "), ("path", alias)), roots=("artifact_store", "runtime_data"))}
    _summary, reply, retention = check_output_views(registry._ctx, raw)
    source = retention["output_source"]
    assert source["path"] == planned["path"] and source["root"] == "artifact_store"
    assert "output_source_call_unavailable" not in retention  # an alias of the same file was admitted, not retargeted
    assert (registry._ctx.drive_root / alias).is_file()       # the aliased spelling physically reaches the retained bytes
    call = re.search(r"read_file\(.*?max_chars=\d+\)", reply).group(0)
    reader = {kw.arg: ast.literal_eval(kw.value) for kw in ast.parse(call).body[0].value.keywords}
    assert reader["root"] == "runtime_data" and reader["path"] == alias  # the selector the admission resolved, aliased address
    page = registry.execute("read_file", reader)  # the REAL dispatch, its own handler admission, over the printed call
    assert page.rstrip().endswith(TAIL_LINE)


def test_an_argument_refusal_from_the_real_entry_is_a_stable_code_and_keeps_the_genuine_source(registry):
    """An argument the REAL public ``read_file`` entry of this version does not accept, pinned through the ceiling:
    the actual preparation on the actual entry refuses the bound call (the validator is never monkeypatched), the
    offer names the stable typed code with the refusal's own reason, keeps the genuine source readable, and prints
    no fabricated call; the same real registry refuses the same argument the same way."""
    from ouroboros.tools.process_facts import check_output_views

    # ``tail_lines`` is no spelling ``read_file`` knows (``end_line`` was the example until the entry began to read it).
    registry._ctx.task_contract = {"capability_ceiling": _ceiling(bindings=(("tail_lines", 5),))}
    _summary, reply, retention = check_output_views(registry._ctx, "a" * 30_000 + "\n" + TAIL_LINE)
    assert retention["output_source_call_unavailable"] == "arg_refused"
    assert "unsupported argument" in reply and "tail_lines" in reply  # the real entry's own refusal, not a stub's text
    assert "read_file(" not in reply and retention["output_source"]["path"] in reply  # no fabricated call; the source stands
    stored = artifacts.read_actor_source_bytes(registry._ctx.drive_root, registry._ctx.task_id,
                                               {**retention["output_source"], "kind": "task_source"})
    assert hashlib.sha256(stored).hexdigest() == retention["output_source"]["sha256"]  # and it is genuinely readable
    refused = registry.execute("read_file", {"root": "artifact_store", "path": retention["output_source"]["path"],
                                             "start_line": 1, "tail_lines": 5})
    assert "TOOL_ARG_ERROR" in refused and "tail_lines" in refused  # what the offer declined is exactly what dispatch declines


def test_a_missing_tool_record_stops_the_consultation_rather_than_bypassing_it(registry, monkeypatch):
    """No real ``read_file`` record: there is no entry to prepare the call against, so nothing establishes this
    task's dispatch. The consultation refuses with the stable unestablished code and keeps the genuine source;
    it never skips the checks onward into an advertised call nothing verified could execute."""
    from ouroboros.tools import core
    from ouroboros.tools.process_facts import check_output_views

    registered = tuple(core.get_tools())
    monkeypatch.setattr(core, "get_tools", lambda: tuple(item for item in registered if item.name != "read_file"))
    _summary, reply, retention = check_output_views(registry._ctx, "a" * 30_000 + "\n" + TAIL_LINE)
    assert retention["output_source_call_unavailable"] == "admission_unknown"
    assert "read_file(" not in reply and retention["output_source"]["path"] in reply  # the genuine ref, no call
    stored = artifacts.read_actor_source_bytes(registry._ctx.drive_root, registry._ctx.task_id,
                                               {**retention["output_source"], "kind": "task_source"})
    assert hashlib.sha256(stored).hexdigest() == retention["output_source"]["sha256"]
