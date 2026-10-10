"""Source-addressed first show of tool results (owner Q4, Lane A).

The Main loop no longer cuts a tool result at a per-tool character cap. A batch
is delivered as one candidate-aware projection: every result is measured whole
inside the round's real transcript, and when the frame cannot hold the batch
each result keeps its complete text as an exact readable source while the model
sees a head+tail view with exact ranges, the typed facts and the source
address. Requested forms spend the frame's free room; unsolicited bodies share
the reclaim-low-water eighth. Nothing here is a tool-name or path exemption and
nothing invents a cap when the frame is unknown.
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import math
import pathlib
from types import SimpleNamespace

import pytest

from ouroboros import artifacts
from ouroboros.context_budget import RECLAIM_LOW_WATER_DIVISOR
from ouroboros.loop_tool_execution import (
    _truncate_tool_result,
    handle_tool_calls,
    process_tool_results,
)
from ouroboros.tool_capabilities import (
    DEFAULT_TOOL_RESULT_LIMIT,
    TOOL_RESULT_LIMITS,
    requested_result_view,
    tool_result_limit,
)
from ouroboros.tool_result_delivery import (
    FULL_SOURCE_MARKER,
    FULL_SOURCE_UNAVAILABLE,
    RESULT_VIEW_BASIS,
    RESULT_VIEW_MARKER,
    project_tool_result_batch,
    render_result_view,
    split_allowance,
)
from ouroboros.tools.core import _read_file
from ouroboros.tools.core_file_tools import _render_line_slice
from ouroboros.tools.registry import ToolContext
from ouroboros.tools.tool_result import ToolResult
from tests._tool_result_delivery_shared import measured_fit


# --- fixtures and helpers -------------------------------------------------

def _ctx(tmp_path: pathlib.Path, task_id: str = "delivery-task") -> ToolContext:
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    return ToolContext(repo_dir=repo, drive_root=tmp_path, task_id=task_id)


def _row(call_id, tool, result, *, args=None, is_error=False, meta=None, typed=None):
    row = {
        "fn_name": tool,
        "tool_call_id": call_id,
        "result": result,
        "is_error": is_error,
        "tool_args": dict(args or {}),
        "args_for_log": dict(args or {}),
        "result_meta": dict(meta if meta is not None else {"status": "ok"}),
    }
    if typed is not None:
        row["tool_result"] = typed
    return row


def _deliver(ctx, rows, fit, messages=None):
    messages = list(messages or [])
    trace = {"tool_calls": []}
    process_tool_results(rows, messages, trace, lambda *_a, **_k: None,
                         SimpleNamespace(_ctx=ctx), fit_candidate=fit, tool_schemas=[])
    return messages, trace


def _marker_json(content: str, marker: str):
    for line in content.split("\n"):
        if line.startswith(marker):
            return json.loads(line[len(marker):])
    raise AssertionError(f"{marker} missing from delivered content")


def _view_json(content):
    return _marker_json(content, RESULT_VIEW_MARKER)


def _source_ref(content):
    return _marker_json(content, FULL_SOURCE_MARKER)


def _shown_chars(view) -> int:
    return sum(hi - lo for lo, hi in view["shown_ranges"])


def _source_dir(tmp_path, task_id):
    return tmp_path / "task_results" / "artifacts" / task_id / "source_handles"


# --- the shared allowance -------------------------------------------------

def test_parallel_giants_share_one_measured_allowance_and_no_result_is_starved(tmp_path):
    """Three parallel giants and a small fourth result in one round: the batch is
    projected ONCE under the measured frame, the giants share the unsolicited
    eighth, every one of them keeps its minimum (envelope, status, source) plus a
    head and a tail, and the last result is delivered whole and byte-identical."""
    ctx = _ctx(tmp_path)
    giants = [(f"g{i}", f"G{i} head\n" + str(i) * 300_000 + f"\nG{i} TAIL") for i in range(3)]
    rows = [_row(cid, "run_command", text, args={"cmd": ["x"]}, meta={"status": "ok", "exit_code": 0})
            for cid, text in giants]
    rows.append(_row("last", "run_command", "final small result",
                     args={"cmd": ["y"]}, meta={"status": "ok", "exit_code": 0}))
    calls = []
    window, reserve = 40_000, 4_000
    messages, trace = _deliver(ctx, rows, measured_fit(window=window, reserve=reserve, calls=calls))

    receipt = trace["tool_result_delivery"][0]
    assert receipt["status"] == "projected" and receipt["boundary_tokens"] == window
    eighth = math.ceil(window / RECLAIM_LOW_WATER_DIVISOR)
    assert receipt["unsolicited_allowance_tokens"] == min(receipt["free_tokens"], eighth) == eighth
    # The final candidate (the whole transcript plus the delivered batch) is under the bound.
    assert calls[-1]["estimated_input_tokens"] + reserve <= window
    assert receipt["measurements"] >= 3  # whole, minimum, and at least one projected frame

    assert messages[3]["content"] == "final small result"
    assert "result_partial" not in trace["tool_calls"][3]
    total_shown = 0
    for i, (cid, text) in enumerate(giants):
        content = messages[i]["content"]
        row = trace["tool_calls"][i]
        view = _view_json(content)
        assert row["result_partial"] is True and row["result_source_status"] == "ready"
        assert f"G{i} head" in content and f"G{i} TAIL" in content
        (h0, h1), (t0, t1) = view["shown_ranges"]
        assert h0 == 0 and 0 < h1 < t0 < t1 == view["complete_chars"] == len(text)
        assert view["omitted_ranges"] == [[h1, t0]] and view["range_basis"] == RESULT_VIEW_BASIS
        assert view["complete_sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert view["facts"] == {"is_error": False, "status": "ok", "exit_code": 0}
        ref = _source_ref(content)
        assert row["result_source_ref"] == ref and row["result_source_view"]["source_status"] == "ready"
        assert artifacts.read_actor_source_bytes(tmp_path, ctx.task_id, ref).decode("utf-8") == text
        assert "Do not rerun this tool to recover omitted output" in content
        total_shown += _shown_chars(view)
    # The unsolicited bodies together stay inside the eighth (conservative: the
    # per-char rate below is at most the projection's own rate).
    rate = (receipt["whole_tokens"] - receipt["minimum_tokens"]) / sum(len(t) for _, t in giants)
    assert total_shown * rate <= eighth
    # Each giant kept a comparable share: no first-come-first-served starvation.
    shares = [_shown_chars(_view_json(messages[i]["content"])) for i in range(3)]
    assert max(shares) - min(shares) <= 2
    assert len(receipt["partial"]) == 3 and {p["tool_call_id"] for p in receipt["partial"]} == {"g0", "g1", "g2"}


def test_requested_head_tail_spends_free_room_while_unsolicited_shares_the_eighth(tmp_path):
    ctx = _ctx(tmp_path)
    text = "H" * 100_000 + "\nMIDDLE\n" + "T" * 100_000
    rows = [
        _row("asked", "run_command", text,
             args={"cmd": ["a"], "view_head_chars": 30_000, "view_tail_chars": 10_000}),
        _row("plain", "run_command", text, args={"cmd": ["b"]}),
    ]
    messages, trace = _deliver(ctx, rows, measured_fit(window=60_000, reserve=2_000))
    receipt = trace["tool_result_delivery"][0]
    assert receipt["status"] == "projected"
    asked, plain = (_view_json(m["content"]) for m in messages)
    assert asked["shown_ranges"] == [[0, 30_000], [len(text) - 10_000, len(text)]]
    assert trace["tool_calls"][0]["result_source_view"]["requested"] is True
    assert trace["tool_calls"][1]["result_source_view"]["requested"] is False
    # The requested 40,000 chars exceed the unsolicited eighth, which bounds the
    # plain result; the plain head and tail split the allowance evenly.
    assert _shown_chars(plain) < 40_000
    (p0, p1), (q0, q1) = plain["shown_ranges"]
    assert p0 == 0 and q1 == len(text) and abs((p1 - p0) - (q1 - q0)) <= 1
    assert "MIDDLE" not in messages[1]["content"]


def test_the_boundary_is_the_smallest_known_positive_bound_never_a_borrowed_one(tmp_path):
    ctx = _ctx(tmp_path)
    rows = [_row("r", "run_command", "x" * 400_000, args={"cmd": ["x"]})]
    # Only a target is known: it bounds. Only a window: it bounds. Both: the smaller.
    for kwargs, expected in (
        ({"target": 30_000}, 30_000),
        ({"window": 50_000}, 50_000),
        ({"window": 50_000, "target": 30_000}, 30_000),
        ({"window": 20_000, "target": 30_000}, 20_000),
    ):
        _messages, trace = _deliver(ctx, rows, measured_fit(reserve=1_000, **kwargs))
        receipt = trace["tool_result_delivery"][0]
        assert receipt["boundary_tokens"] == expected, kwargs
        assert receipt["reserve_tokens"] == 1_000
        assert receipt["unsolicited_allowance_tokens"] == min(
            receipt["free_tokens"], math.ceil(expected / RECLAIM_LOW_WATER_DIVISOR))


def test_unknown_frame_delivers_whole_discloses_and_writes_no_source(tmp_path):
    ctx = _ctx(tmp_path)
    full = "x" * 500_000
    rows = [_row("u", "run_command", full, args={"cmd": ["x"]})]
    messages, trace = _deliver(ctx, rows, measured_fit())  # neither window nor target known
    assert messages[0]["content"] == full
    assert "result_partial" not in trace["tool_calls"][0]
    receipt = trace["tool_result_delivery"][0]
    assert receipt["status"] == "capacity_unknown" and receipt["boundary_tokens"] is None
    assert not _source_dir(tmp_path, ctx.task_id).exists()
    # No callback at all (the unwired caller): whole, best effort, no receipt and no cap.
    messages2, trace2 = _deliver(ctx, rows, None)
    assert messages2[0]["content"] == full and "tool_result_delivery" not in trace2


def test_accepted_true_is_not_read_as_fit(tmp_path):
    """The measurement always says accepted=True (that flag is the measurement
    itself); the projection decides on the token arithmetic alone."""
    ctx = _ctx(tmp_path)
    calls = []
    rows = [_row("a", "run_command", "x" * 200_000, args={"cmd": ["x"]})]
    messages, trace = _deliver(ctx, rows, measured_fit(window=10_000, reserve=1_000, calls=calls))
    assert all(call["accepted"] is True for call in calls)
    assert trace["tool_calls"][0]["result_partial"] is True
    assert trace["tool_result_delivery"][0]["status"] == "projected"


def test_a_batch_that_fits_is_delivered_byte_identical_without_sources_or_receipt(tmp_path):
    ctx = _ctx(tmp_path)
    texts = ["short\r\nwindows\rlines", "exact " * 200, "{\"json\": true}"]
    rows = [_row(f"s{i}", "run_command", t, args={"cmd": ["x"]}) for i, t in enumerate(texts)]
    messages, trace = _deliver(ctx, rows, measured_fit(window=50_000, reserve=2_000))
    assert [m["content"] for m in messages] == texts  # raw bytes, newline forms included
    assert all("result_partial" not in row for row in trace["tool_calls"])
    assert "tool_result_delivery" not in trace
    assert not _source_dir(tmp_path, ctx.task_id).exists()


# --- the envelope -----------------------------------------------------------

def test_source_write_failure_is_disclosed_without_any_readability_claim(tmp_path, monkeypatch):
    def _boom(*_a, **_k):
        raise OSError("disk full")
    monkeypatch.setattr(artifacts, "store_actor_source_bytes", _boom)
    ctx = _ctx(tmp_path)
    rows = [_row("f", "run_command", "x" * 200_000 + "\nTAIL", args={"cmd": ["x"]})]
    messages, trace = _deliver(ctx, rows, measured_fit(window=20_000, reserve=2_000))
    content = messages[0]["content"]
    assert FULL_SOURCE_UNAVAILABLE in content and FULL_SOURCE_MARKER not in content
    assert "exact source persistence failed" in content and "TAIL" in content
    view = _view_json(content)
    assert "read_omitted" not in view and view["omitted_ranges"]
    row = trace["tool_calls"][0]
    assert row["result_partial"] is True and row["result_source_ref"] == {}
    assert row["result_source_status"] == "source_unavailable"
    assert row["result_source_view"]["source_status"] == "source_unavailable"


def test_failed_command_keeps_status_exit_code_and_tail_outside_the_omitted_body(tmp_path):
    ctx = _ctx(tmp_path)
    text = "build started\n" + "." * 300_000 + "\nerror: linker failed\nexit 1"
    typed = ToolResult(status="error", code="TOOL_REPORTED_FAILURE", text=text,
                       meta={"exit_code": 1}, producer_text=text,
                       host_annotations=("[host] build exceeded the soft budget",))
    rows = [_row("fail", "run_command", text, args={"cmd": ["make"]}, is_error=True, typed=typed,
                 meta={"status": "error", "tool_result_status": "error",
                       "tool_result_code": "TOOL_REPORTED_FAILURE", "exit_code": 1, "signal": "SIGKILL"})]
    messages, trace = _deliver(ctx, rows, measured_fit(window=20_000, reserve=2_000))
    content = messages[0]["content"]
    view = _view_json(content)
    assert view["facts"] == {"is_error": True, "status": "error", "code": "TOOL_REPORTED_FAILURE",
                             "exit_code": 1, "signal": "SIGKILL"}
    assert content.startswith("build started\n") and "error: linker failed\nexit 1" in content
    row = trace["tool_calls"][0]
    assert row["is_error"] is True and row["result_partial"] is True
    # Host annotations and the producer text ride outside the omitted body.
    assert "[host] build exceeded the soft budget" in content
    assert row["host_annotations"] == ["[host] build exceeded the soft budget"]
    assert row["producer_source_ref"]["path"].startswith("source_handles/")
    assert artifacts.read_actor_source_bytes(tmp_path, ctx.task_id, row["producer_source_ref"]).decode("utf-8") == text


def test_small_whole_text_is_byte_identical_and_a_view_never_grows():
    text = "short result with \r\n windows newline"
    assert render_result_view(text, head_chars=0, tail_chars=0) == (text, None)
    assert render_result_view(text, head_chars=5, tail_chars=5) == (text, None)  # the envelope would grow it
    big = "x" * 5_000
    delivered, view = render_result_view(big, head_chars=100, tail_chars=100,
                                         source_ref={"root": "artifact_store", "path": "p.txt"})
    assert len(delivered) < len(big) and view["shown_ranges"] == [[0, 100], [4_900, 5_000]]
    assert view["omitted_ranges"] == [[100, 4_900]] and view["read_omitted"]["start_char"] == 100
    assert view["read_omitted"]["max_chars"] == 4_800 and view["read_omitted"]["path"] == "p.txt"
    assert render_result_view(big, head_chars=2_500, tail_chars=2_500) == (big, None)
    assert split_allowance(0, None, 10) == (0, 0) and split_allowance(10, None, 10) == (10, 0)
    assert split_allowance(7, None, 100) == (4, 3)
    assert split_allowance(100, (30, 10), 1_000) == (30, 10)  # a request that fits is honored exactly
    assert split_allowance(20, (30, 10), 1_000) == (15, 5)    # one that must shrink keeps its proportions


def test_even_the_minimum_request_not_fitting_is_named_and_sources_are_kept(tmp_path):
    ctx = _ctx(tmp_path)
    rows = [_row(f"r{i}", "run_command", "z" * 50_000 + f"\nEND{i}", args={"cmd": ["x"]}) for i in range(3)]
    messages, trace = _deliver(ctx, rows, measured_fit(window=600, reserve=200))
    receipt = trace["tool_result_delivery"][0]
    assert receipt["status"] == "minimum_view_unfit"
    for i, message in enumerate(messages):
        view = _view_json(message["content"])
        assert view["shown_ranges"] == [] and view["omitted_ranges"] == [[0, 50_005]]
        assert trace["tool_calls"][i]["result_source_status"] == "ready"
        ref = _source_ref(message["content"])
        assert artifacts.read_actor_source_bytes(tmp_path, ctx.task_id, ref).decode("utf-8") == rows[i]["result"]


def test_native_image_message_in_the_transcript_is_measured_not_altered(tmp_path):
    ctx = _ctx(tmp_path)
    image = {"role": "user", "content": [
        {"type": "text", "text": "look at this"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 60_000}},
    ]}
    before = json.dumps(image, sort_keys=True)
    rows = [_row("i", "run_command", "x" * 100_000, args={"cmd": ["x"]})]
    messages, trace = _deliver(ctx, rows, measured_fit(window=30_000, reserve=2_000), messages=[image])
    assert json.dumps(messages[0], sort_keys=True) == before
    assert trace["tool_calls"][0]["result_partial"] is True
    assert trace["tool_result_delivery"][0]["status"] == "projected"


# --- the projection seam ----------------------------------------------------

def test_project_tool_result_batch_keeps_its_default_contract_and_rejects_unknown_policy(tmp_path):
    """The default policy is the consolidator's moved contract: the callback's own
    ``accepted`` verdict drives a prefix search (the measured policy never reads it)."""
    rows = [_row("p", "run_command", "x" * 100_000, args={"cmd": ["x"]})]
    measured = measured_fit(window=10_000, reserve=1_000)

    def fit(messages, schemas):
        facts = measured(messages, schemas)
        return {**facts, "accepted": facts["estimated_input_tokens"] + 1_000 <= 10_000}

    out, receipt = project_tool_result_batch(rows, [], [], drive_root=tmp_path, task_id="t", fit_candidate=fit)
    assert rows[0]["result"] == "x" * 100_000 and "result_partial" not in rows[0]  # inputs untouched
    assert receipt["status"] == "projected"
    assert out[0]["result_partial"] is True and "[Tool result source view]" in out[0]["result"]
    assert fit([*[], {"role": "tool", "tool_call_id": "p", "content": out[0]["result"]}], [])["accepted"] is True
    with pytest.raises(ValueError):
        project_tool_result_batch(rows, [], [], drive_root=tmp_path, task_id="t",
                                  fit_candidate=fit, policy="not_a_policy")


def test_handle_tool_calls_delivers_through_the_measured_frame(tmp_path):
    from ouroboros.tools.tool_result import LegacyTextResultAdapter
    from ouroboros.loop_tool_execution import StatefulToolExecutor

    class _Tools:
        CODE_TOOLS = set()

        def __init__(self):
            self._ctx = SimpleNamespace(task_metadata={}, drive_root=tmp_path, task_id="h")

        def get_timeout(self, _name):
            return 10

        def execute_result(self, name, args):
            return LegacyTextResultAdapter.from_text(name, "HEAD\n" + "k" * 300_000 + "\nTAIL")

    assert inspect.signature(handle_tool_calls).parameters["fit_candidate"].kind is inspect.Parameter.KEYWORD_ONLY
    tools = _Tools()
    logs = tmp_path / "logs"
    logs.mkdir()
    messages = [{"role": "user", "content": "run it"}]
    trace = {"tool_calls": []}
    calls = []
    errors = handle_tool_calls(
        [{"id": "call-1", "type": "function",
          "function": {"name": "run_command", "arguments": json.dumps({"cmd": ["x"]})}}],
        tools, logs, "h", StatefulToolExecutor(), messages, trace, lambda _t: None,
        fit_candidate=measured_fit(window=20_000, reserve=2_000, calls=calls), tool_schemas=[{"type": "function"}],
    )
    assert errors == 0 and calls and calls[-1]["estimated_input_tokens"] + 2_000 <= 20_000
    assert messages[-1]["role"] == "tool" and messages[-1]["tool_call_id"] == "call-1"
    assert "HEAD\n" in messages[-1]["content"] and "\nTAIL" in messages[-1]["content"]
    assert trace["tool_calls"][0]["result_partial"] is True
    assert trace["tool_result_delivery"][0]["round_id"] is None or isinstance(trace["tool_result_delivery"][0]["round_id"], str)


def test_truncate_tool_result_compat_invents_no_cap_without_an_allowance():
    big = "q" * 400_000
    for tool in ("run_command", "read_file", "get_task_result", "web_fetch", "unknown_tool"):
        assert _truncate_tool_result(big, tool) == big, tool
        assert _truncate_tool_result(big, tool, {"path": "docs/architecture/06-agent-core.md"}) == big, tool
    view = _truncate_tool_result(big, "run_command", {"cmd": ["x"]}, allowance_chars=1_000)
    assert len(view) < 3_000 and FULL_SOURCE_UNAVAILABLE in view and "omitted 500\u2013399500" in view


# --- the readers --------------------------------------------------------------

def test_render_line_slice_max_chars_names_complete_lines_and_the_next_cursor():
    content = "aaaa\nbbbb\ncccc\ndddd\n"
    extent = {}
    out = _render_line_slice("f", content, max_lines=4, start_line=1, start_char=2, max_chars=9, extent=extent)
    assert out == "# f \u2014 lines 1\u20134 of 4 (from char 2 of this window, 9 chars; next start_char=11)\naa\nbbbb\nc"
    assert extent["partial_head"] is True and extent["partial_tail"] is True
    assert extent["first_line"] == 2 and extent["end_line"] == 2 and extent["line_ends"] == (8,)
    assert extent["source_start_char"] == 2 and extent["source_end_char"] == 11
    # Without a budget the facts are the ones every existing consumer already reads.
    extent2 = {}
    out2 = _render_line_slice("f", content, max_lines=4, start_line=1, start_char=2, extent=extent2)
    assert out2 == "# f \u2014 lines 1\u20134 of 4 (from char 2 of this window)\naa\nbbbb\ncccc\ndddd\n"
    assert extent2["end_line"] == 4 and extent2["partial_tail"] is False and extent2["line_ends"] == (8, 13, 18)
    # A budget larger than the window changes nothing; zero or junk means no budget.
    assert _render_line_slice("f", content, max_lines=4, max_chars=10_000) == "# f \u2014 lines 1\u20134 of 4\n" + content
    assert _render_line_slice("f", content, max_lines=4, max_chars=0) == "# f \u2014 lines 1\u20134 of 4\n" + content


def test_one_giant_unicode_line_is_reconstructed_exactly_through_char_pages(tmp_path):
    ctx = _ctx(tmp_path)
    line = "".join(f"{i:05d}\u03a9\U0001f989" for i in range(30_000))  # 210,000 code points, no newline
    rows = [_row("line", "run_command", line, args={"cmd": ["x"]})]
    messages, trace = _deliver(ctx, rows, measured_fit(window=20_000, reserve=2_000))
    content = messages[0]["content"]
    view, ref = _view_json(content), _source_ref(content)
    assert view["complete_chars"] == len(line) and view["range_basis"] == RESULT_VIEW_BASIS
    (o0, o1), = view["omitted_ranges"]
    # The view's own address reads exactly the omitted range, nothing else.
    args = dict(view["read_omitted"])
    assert args.pop("tool") == "read_file"
    header, body = _read_file(ctx, **args).split("\n", 1)
    assert body == line[o0:o1] and f"next start_char={o1}" in header
    assert ctx.last_read_view["source_start_char"] == o0 and ctx.last_read_view["source_end_char"] == o1
    # Pages of 7,000 code points reconstruct the whole line exactly; each page
    # names the cursor the next page starts from.
    pieces, cursor = [], 0
    while cursor < len(line):
        rendered = _read_file(ctx, ref["path"], root="artifact_store", start_line=1, max_lines=1,
                              start_char=cursor, max_chars=7_000)
        header, body = rendered.split("\n", 1)
        assert body and body == line[cursor:cursor + len(body)]
        if cursor + len(body) < len(line):
            assert f"next start_char={cursor + len(body)}" in header
        pieces.append(body)
        cursor += len(body)
    assert "".join(pieces) == line and len(pieces) == 30
    # Reading a saved result never repeats the action: the source is a plain file read.
    assert artifacts.read_actor_source_bytes(tmp_path, ctx.task_id, ref).decode("utf-8") == line


def test_requested_result_view_classifies_declared_range_parameters():
    assert requested_result_view({"cmd": ["x"]}) is None
    assert requested_result_view(None) is None
    assert requested_result_view({"view_head_chars": 100, "view_tail_chars": 0}) == (100, 0)
    assert requested_result_view({"view_tail_chars": "500"}) == (0, 500)
    assert requested_result_view({"view_head_chars": -3}) == (0, 0)
    assert requested_result_view({"path": "a", "max_lines": 10}) == (None, None)
    assert requested_result_view({"task_id": "t", "source_start_char": 0, "source_end_char": 9}) == (None, None)
    assert requested_result_view({"task_id": "t", "offset": 2, "limit": 5}) == (None, None)


def test_producer_page_sizes_stay_positive_ints_for_every_paging_consumer():
    for name in list(TOOL_RESULT_LIMITS) + ["unknown_tool", "", "get_task_result"]:
        limit = tool_result_limit(name)
        assert type(limit) is int and limit > 0, name
    assert tool_result_limit("unknown_tool") == DEFAULT_TOOL_RESULT_LIMIT
    for module in ("ouroboros.tools.github_checks", "ouroboros.tools.chronicle",
                   "ouroboros.tools.delegate_terminal_evidence", "ouroboros.skill_publish_result",
                   "ouroboros.tools.delegate", "ouroboros.tools.followup",
                   "ouroboros.skill_catalogue", "ouroboros.delegate_supervision",
                   "ouroboros.delegate_interactions"):
        source = inspect.getsource(importlib.import_module(module))
        assert "tool_result_limit(" in source, module
