"""Long matched lines show their match and an exact read_file range on both backends.

search_code rows for long lines carry a literal window around the first match
plus the read_file(start_line, start_char, max_lines, max_chars) of that window
in read_file's own text model (universal newlines, ``str.splitlines`` ends,
Unicode code points). Short lines keep their old rows. Real ripgrep and the
Python scanner are both exercised, and every advertised range is read back; an
empty read_file line has no positive max_chars and is addressed whole.
"""
from __future__ import annotations

import json
import re

import pytest

from ouroboros import code_search_rg
from ouroboros.tools.core import _code_search
from ouroboros.tools.core_file_tools import _read_file, delivered_source_prefix
from ouroboros.tools.registry import ToolContext

ROW = re.compile(r"^(?P<root>[a-z_]+):(?P<path>[^:]+):(?P<line>\d+): (?P<shown>.*) \[(?P<chars>\d+)-char line; "
                 r"(?P<which>match|first of (?P<count>\d+) matches) at char (?P<at>\d+) \((?P<width>\d+) chars\); "
                 r"(?P<address>[^\]]*)\]$")
ADDRESS = re.compile(r"^read_file\(start_line=(\d+), start_char=(\d+), max_lines=1, max_chars=(\d+)\)$")


@pytest.fixture(params=["python", "rg"])
def backend(request, monkeypatch):
    if request.param == "rg" and not code_search_rg._rg_binary():
        pytest.skip("ripgrep is not installed")
    if request.param == "python":
        monkeypatch.setattr(code_search_rg, "_rg_binary", lambda: "")
    return request.param


def _search(tmp_path, name, raw, query, **kwargs):
    (tmp_path / name).write_bytes(raw)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path / "data")
    return ctx, _code_search(ctx, query=query, path=".", include=name, **kwargs)


def _rows(result):
    return [line for line in result.split("\n\n", 1)[1].splitlines()] if "\n\n" in result else []


def _read_back(ctx, row):
    """Read the advertised range and return (row facts, delivered source characters)."""
    found = ROW.match(row)
    assert found, row
    address = ADDRESS.match(found["address"])
    assert address, row
    start_line, start_char, max_chars = map(int, address.groups())
    rendered = _read_file(ctx, found["path"], root=found["root"], start_line=start_line, max_lines=1,
                          start_char=start_char, max_chars=max_chars)
    assert not rendered.startswith("⚠️"), rendered
    body = delivered_source_prefix(ctx.last_read_view, rendered, len(rendered))
    assert ctx.last_read_view["start_line"] == start_line and len(body) == max_chars, rendered
    return found, start_line, start_char, body


def _read_file_lines(raw):
    return raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n").splitlines()


def test_long_line_shows_match_and_reads_back_in_native_review_delivery(tmp_path, backend):
    from tests.test_native_read_source_extent import _executor

    raw = ("x" * 160000 + "LATE_SENTINEL" + "tail" * 200).encode()
    ctx, result = _search(tmp_path, "evidence.json", raw, "LATE_SENTINEL")
    assert len(result) < 2000, result[:400]
    (row,) = _rows(result)
    found, start_line, start_char, body = _read_back(ctx, row)
    assert (found["line"], start_line, int(found["at"]), start_char) == ("1", 1, 160000, 159940)
    assert body == "x" * 60 + "LATE_SENTINEL" + ("tail" * 200)[:167]
    assert found["shown"] == "…" + body + "…" and found["width"] == "13"
    executor = _executor(tmp_path)
    registry, _ = executor._inspection_registry(str(tmp_path), tmp_path / "inspection")
    args = {"query": "LATE_SENTINEL", "path": ".", "include": "evidence.json"}
    message = executor._execute_inspection_call(
        registry, {"id": "search-1", "function": {"name": "search_code", "arguments": json.dumps(args)}},
        {}, round_idx=1)
    assert "LATE_SENTINEL" in message["content"] and "RESULT TRUNCATED" not in message["content"]
    assert "read_file(start_line=1, start_char=159940, max_lines=1, max_chars=240)" in message["content"]


def test_short_lines_keep_their_rows(tmp_path, backend):
    ctx, result = _search(tmp_path, "short.txt", b"short LATE_SENTINEL line  \nnext\n", "LATE_SENTINEL")
    assert _rows(result) == ["active_workspace:short.txt:1: short LATE_SENTINEL line"]


def test_a_line_at_the_window_width_stays_whole(tmp_path, backend):
    edge = ("y" * (code_search_rg.MATCH_WINDOW_CHARS - 6) + "NEEDLE").encode()
    _, result = _search(tmp_path, "edge.txt", edge, "NEEDLE")
    assert _rows(result) == ["active_workspace:edge.txt:1: " + edge.decode()]


@pytest.mark.parametrize("prefix", [
    "αβγ\n", "a b c\n", "a\x1cb\x1dc\x1ed\n",
], ids=["multibyte", "unicode-separators", "info-separators"])
def test_earlier_separators_and_multibyte_text_keep_exact_read_file_ranges(tmp_path, backend, prefix):
    tail = "λ" * 900 + "NEEDLE" + "ω" * 900
    raw = (prefix + "head " + tail + "\n").encode()
    ctx, result = _search(tmp_path, "mixed.txt", raw, "NEEDLE")
    (row,) = _rows(result)
    found, start_line, start_char, body = _read_back(ctx, row)
    lines = _read_file_lines(raw)
    expected_line = next(i for i, line in enumerate(lines, 1) if "NEEDLE" in line)
    column = lines[expected_line - 1].index("NEEDLE")
    assert (start_line, int(found["at"])) == (expected_line, column), row
    assert body == lines[expected_line - 1][start_char:start_char + 240]
    assert body[column - start_char:column - start_char + 6] == "NEEDLE"


def test_repeated_zero_width_and_oversized_matches_are_disclosed(tmp_path, backend):
    raw = ("a" * 500 + "NEEDLE" + "b" * 500 + "NEEDLE" + "c" * 500 + "NEEDLE\n").encode()
    ctx, result = _search(tmp_path, "repeat.txt", raw, "NEEDLE")
    (row,) = _rows(result)
    found, _, start_char, body = _read_back(ctx, row)
    assert found["count"] == "3" and found["at"] == "500" and body[60:66] == "NEEDLE"
    long = ("z" * 300 + "x" * 5000 + "\n").encode()
    _, result = _search(tmp_path, "run.txt", long, "x+", regex=True)
    (row,) = _rows(result)
    found, _, start_char, body = _read_back(ctx, row)
    assert (found["at"], found["width"], start_char) == ("300", "5000", 240)
    assert body == "z" * 60 + "x" * 180
    _, result = _search(tmp_path, "run.txt", long, "$", regex=True)
    (row,) = _rows(result)
    found, _, start_char, body = _read_back(ctx, row)
    assert (found["at"], found["width"], start_char) == ("5300", "0", 5240) and body == "x" * 60
    assert found["shown"] == "…" + "x" * 60


@pytest.mark.parametrize(("raw", "query", "line", "ending"), [
    (b"\x0c" + b"x" * 400 + b"NEEDLE\n", "^", 1, "\x0c"),
    (b"x" * 400 + b"\x0c\x0cNEEDLE\n", r"\x0cNEEDLE", 2, "\x0c"),
    (b"\r" + b"z" * 400 + b"\n", "^", 1, "\n"),
], ids=["leading-form-feed", "between-form-feeds", "leading-lone-cr"])
def test_an_empty_read_file_line_is_addressed_by_a_valid_whole_line_call(tmp_path, raw, query, line, ending):
    """rg's LF line can hold a read_file line end right at the match: the match then sits on an
    EMPTY read_file line. The advertised call must satisfy read_file's own schema and, run through
    the registry, deliver that whole line: nothing but its line end."""
    import jsonschema

    from ouroboros.tools.registry import ToolRegistry

    if not code_search_rg._rg_binary():
        pytest.skip("ripgrep is not installed")
    ctx, result = _search(tmp_path, "empty.txt", raw, query, regex=True)
    (row,) = _rows(result)
    found = ROW.match(row)
    assert found and (found["shown"], found["chars"], found["at"]) == ("", "0", "0"), row
    name, params = re.fullmatch(r"(\w+)\((.*)\)", found["address"]).groups()
    call = {"path": found["path"], "root": found["root"],
            **{key: int(value) for key, value in (item.split("=") for item in params.split(", "))}}
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path / "data")
    registry.set_context(ctx)
    jsonschema.validate(call, registry.get_schema_by_name(name)["function"]["parameters"])
    assert (name, call["start_line"], call["max_lines"], set(call) - {"path", "root"}) == (
        "read_file", line, 1, {"start_line", "max_lines"}), row  # the whole line, no invented char budget
    rendered = registry.execute_result(name, call)
    view = ctx.last_read_view
    assert rendered.status == "ok" and (view["start_line"], view["end_line"]) == (line, line), rendered.text
    assert delivered_source_prefix(view, rendered.text, len(rendered.text)) == ending, rendered.text


def test_byte_order_mark_keeps_the_column_read_file_counts(tmp_path, backend):
    raw = b"\xef\xbb\xbf" + ("q" * 400 + "NEEDLE" + "r" * 400 + "\nsecond " + "s" * 400 + "NEEDLE\n").encode()
    ctx, result = _search(tmp_path, "bom.txt", raw, "NEEDLE")
    first, second = _rows(result)
    found, line, start_char, body = _read_back(ctx, first)
    assert (line, found["at"], body[60:66]) == (1, "401", "NEEDLE")  # read_file keeps U+FEFF
    found, line, start_char, body = _read_back(ctx, second)
    assert (line, found["at"], body[60:66]) == (2, "407", "NEEDLE")


def test_invalid_utf8_discloses_that_no_read_file_range_exists(tmp_path, backend):
    raw = b"bad \xff byte\n" + ("w" * 400 + "NEEDLE" + "w" * 400 + "\n").encode()
    ctx, result = _search(tmp_path, "bad.txt", raw, "NEEDLE")
    (row,) = _rows(result)
    found = ROW.match(row)
    assert found and found["address"] == "no read_file range: " + code_search_rg.NOT_UTF8_GAP, row
    assert found["shown"][61:67] == "NEEDLE"
    from ouroboros.tools.registry import ToolRegistry  # the reader really cannot decode it

    refused = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path / "data").execute(
        "read_file", {"path": "bad.txt", "start_line": 2, "max_lines": 1})
    assert refused.startswith("⚠️") and "can't decode" in refused and "NEEDLE" not in refused, refused
    raw = ("v" * 300 + "NEEDLE").encode() + b"\xfe" + b"v" * 300 + b"\n"
    _, result = _search(tmp_path, "badline.txt", raw, "NEEDLE")
    (row,) = _rows(result)
    found = ROW.match(row)
    assert found and found["address"].startswith("no read_file range: ") and "NEEDLE" in found["shown"], row


def test_result_and_scan_limits_still_count_matched_lines(tmp_path, backend):
    raw = "".join("p" * 400 + f"NEEDLE{i}" + "\n" for i in range(5)).encode()
    _, result = _search(tmp_path, "many.txt", raw, "NEEDLE", max_results=3)
    assert len(_rows(result)) == 3 and "truncated at 3 results" in result
    assert all(ROW.match(row) for row in _rows(result))


@pytest.mark.parametrize(("long_line", "separator"), [(False, "\u0085"), (False, "\u2029"), (True, "\u2028")])
def test_rg_json_records_split_only_on_lf_not_unicode_inside_strings(tmp_path, separator, long_line):
    if not code_search_rg._rg_binary():
        pytest.skip("ripgrep is not installed")
    suffix = "x" * 500 + "NEEDLE" + "y" * 500 if long_line else "NEEDLE end"
    raw = ("prefix" + separator + suffix + "\n").encode()
    ctx, result = _search(tmp_path, "unicode.txt", raw, "NEEDLE")
    assert "Found 1 match" in result and "NEEDLE" in result, result
    if long_line:
        (row,) = _rows(result)
        found, line, start, body = _read_back(ctx, row)
        assert line == 2 and found["at"] == "500" and body[60:66] == "NEEDLE"


@pytest.mark.parametrize("separator", ["\x0c", "\u2028", "\r"])
def test_every_matched_read_file_line_survives_with_its_own_count(tmp_path, backend, separator):
    raw = ("x" * 300 + "NEEDLE" + separator + "y" * 300 + "NEEDLE\n").encode()
    ctx, result = _search(tmp_path, "separated.txt", raw, "NEEDLE")
    rows = _rows(result)
    assert len(rows) == 2 and "Found 2 matches" in result, result
    for expected_line, row in enumerate(rows, 1):
        found, line, _, body = _read_back(ctx, row)
        assert line == expected_line and body.endswith("NEEDLE") and found["which"] == "match", row


def test_split_rg_event_still_obeys_the_result_limit(tmp_path, backend):
    raw = ("x" * 300 + "NEEDLE\r").encode() * 3
    _, result = _search(tmp_path, "limited.txt", raw, "NEEDLE", max_results=2)
    assert len(_rows(result)) == 2 and "truncated at 2 results" in result, result


def test_rg_zero_width_crlf_match_uses_the_same_line_for_count_and_address(tmp_path):
    if not code_search_rg._rg_binary():
        pytest.skip("ripgrep is not installed")
    ctx, result = _search(tmp_path, "crlf.txt", b"x" * 400 + b"\r\n", "$", regex=True)
    found, line, _, body = _read_back(ctx, _rows(result)[0])
    assert line == 1 and found["which"] == "match" and found["at"] == "400" and body == "x" * 60


def test_rg_json_transport_declares_utf8_independent_of_locale(tmp_path, monkeypatch):
    if not code_search_rg._rg_binary():
        pytest.skip("ripgrep is not installed")
    original = code_search_rg.subprocess.run
    observed = []

    def run(argv, *args, **kwargs):
        if "--json" in argv:
            observed.append(kwargs.get("encoding"))
            assert kwargs.get("encoding") == "utf-8"
        return original(argv, *args, **kwargs)

    monkeypatch.setattr(code_search_rg.subprocess, "run", run)
    ctx, result = _search(tmp_path, "unicode-wire.txt", ("λ" * 300 + "NEEDLE\n").encode(), "NEEDLE")
    found, line, _, body = _read_back(ctx, _rows(result)[0])
    assert observed == ["utf-8"] and line == 1 and found["at"] == "300" and body.endswith("NEEDLE")


def test_ripgrep_without_a_reported_position_keeps_its_line(tmp_path, monkeypatch):
    from tests.test_code_search_rg import _install_fake_rg

    _install_fake_rg(tmp_path, monkeypatch)
    line = "k" * 500 + "needle" + "k" * 10
    ctx, result = _search(tmp_path, "fake.txt", (line + "\n").encode(), "needle")
    assert _rows(result) == ["active_workspace:fake.txt:1: " + line]
