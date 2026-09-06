from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
import sys
import types
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest


_PACKAGE = "telegram_format_parity_test"


def _load_skill():
    root = Path(__file__).resolve().parents[1] / "skills" / "telegram"
    for name in [key for key in sys.modules if key == _PACKAGE or key.startswith(f"{_PACKAGE}.")]:
        sys.modules.pop(name, None)
    package = types.ModuleType(_PACKAGE)
    package.__path__ = [str(root)]
    sys.modules[_PACKAGE] = package
    spec = importlib.util.spec_from_file_location(f"{_PACKAGE}.plugin", root / "plugin.py")
    plugin = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = plugin
    assert spec.loader is not None
    spec.loader.exec_module(plugin)
    return plugin, sys.modules[f"{_PACKAGE}.lib.telegram_api"]


def _assert_balanced(value: str) -> None:
    ET.fromstring(f"<root>{value}</root>")


def test_gfm_table_becomes_aligned_pre_grid():
    _plugin, telegram_api = _load_skill()
    rendered = telegram_api.markdown_to_telegram_html(
        "| Name | Score |\n| --- | ---: |\n| Ada | 9 |\n| Grace | 10 |"
    )
    assert rendered == (
        "<pre>Name  | Score\n"
        "------+------\n"
        "Ada   | 9\n"
        "Grace | 10</pre>"
    )
    _assert_balanced(rendered)


@pytest.mark.parametrize(
    "source",
    [
        "price | quality\n---\nnext",
        "`price | quality`\n---\nnext",
    ],
)
def test_pipe_less_delimiter_does_not_create_gfm_table(source):
    _plugin, telegram_api = _load_skill()
    rendered = telegram_api.markdown_to_telegram_html(source)
    assert "<pre>" not in rendered
    assert "---" in rendered


def test_gfm_table_requires_matching_header_and_delimiter_cell_counts():
    _plugin, telegram_api = _load_skill()
    source = "A | B\n--- | --- | ---\none | two"
    assert "<pre>" not in telegram_api.markdown_to_telegram_html(source)


@pytest.mark.parametrize(
    "source",
    [
        "    a | b\n    --- | ---\n    c | d",
        "\ta | b\n\t--- | ---\n\tc | d",
    ],
)
def test_indented_code_that_resembles_a_table_stays_literal(source):
    _plugin, telegram_api = _load_skill()
    assert telegram_api.markdown_to_telegram_html(source) == source


@pytest.mark.parametrize("indent", ["", " ", "  ", "   "])
def test_real_and_lazy_indented_tables_convert(indent):
    _plugin, telegram_api = _load_skill()
    source = f"{indent}A | B\n{indent}--- | ---\n{indent}1 | 2"
    assert telegram_api.markdown_to_telegram_html(source).startswith("<pre>")


def test_single_dash_gfm_table_delimiters_convert():
    _plugin, telegram_api = _load_skill()
    rendered = telegram_api.markdown_to_telegram_html("A|B\n-|-\n1|2")
    assert rendered == "<pre>A   | B\n----+----\n1   | 2</pre>"


def test_single_dash_without_pipes_does_not_create_gfm_table():
    _plugin, telegram_api = _load_skill()
    rendered = telegram_api.markdown_to_telegram_html("A | B\n-\n1 | 2")
    assert "<pre>" not in rendered


def test_gfm_table_caps_rows_columns_and_cell_length():
    _plugin, telegram_api = _load_skill()
    header = "| " + " | ".join(f"column-{index}" for index in range(7)) + " |"
    delimiter = "| " + " | ".join("---" for _ in range(7)) + " |"
    rows = [
        "| " + " | ".join([f"row-{index}", "x" * 30, "c", "d", "e", "f", "hidden"]) + " |"
        for index in range(35)
    ]
    rendered = telegram_api.markdown_to_telegram_html("\n".join([header, delimiter, *rows]))
    content = rendered.removeprefix("<pre>").removesuffix("</pre>")
    assert "xxxxxxxxxxxxxxxxxxxxxxx…" in content
    assert "hidden" not in content
    assert "row-29" not in content
    assert content.splitlines()[-1] == "…table truncated"
    assert len(content.splitlines()) == 32


def test_task_lists_use_inert_checkbox_glyphs():
    _plugin, telegram_api = _load_skill()
    rendered = telegram_api.markdown_to_telegram_html(
        "- [ ] todo\n- [x] done\n* [X] also done\n+ [x] plus done\n1. [ ] ordered todo\n1. plain"
    )
    assert rendered == "☐ todo\n☑ done\n☑ also done\n☑ plus done\n☐ ordered todo\n1. plain"


def test_latex_delimiters_survive_as_literal_source():
    _plugin, telegram_api = _load_skill()
    source = r"$$x_*[y](https://example.com)* < z$$ and \(*a_b*\) and \[[link](https://example.com)\]"
    rendered = telegram_api.markdown_to_telegram_html(source)
    assert "$$x_*[y](https://example.com)* &lt; z$$" in rendered
    assert r"\(*a_b*\)" in rendered
    assert r"\[[link](https://example.com)\]" in rendered
    assert "<i>" not in rendered
    assert "<a " not in rendered


@pytest.mark.parametrize(
    ("source", "expected_content"),
    [
        ("a `x ```f``` y` b", "a <code>x <pre>f</pre> y</code> b"),
        ("before $$ ```f``` $$ after", "before $$ <pre>f</pre> $$ after"),
    ],
)
def test_nested_fence_placeholders_are_fully_reconstructed(source, expected_content):
    _plugin, telegram_api = _load_skill()
    rendered = telegram_api.markdown_to_telegram_html(source)
    assert rendered == expected_content
    assert "\x00" not in rendered
    assert "PRE0" not in rendered


def test_plain_text_nul_is_removed_from_rendered_output():
    _plugin, telegram_api = _load_skill()
    rendered = telegram_api.markdown_to_telegram_html("before\x00after")
    assert rendered == "beforeafter"
    assert "\x00" not in rendered


def test_triple_asterisk_emphasis_is_properly_nested():
    _plugin, telegram_api = _load_skill()
    rendered = telegram_api.markdown_to_telegram_html("***x***")
    assert rendered == "<b><i>x</i></b>"
    _assert_balanced(rendered)


def test_bold_italic_and_triple_emphasis_convert_together():
    _plugin, telegram_api = _load_skill()
    rendered = telegram_api.markdown_to_telegram_html(
        "**bold** and *ital* and ***both***"
    )
    assert rendered == "<b>bold</b> and <i>ital</i> and <b><i>both</i></b>"
    _assert_balanced(rendered)


def test_block_aware_chunking_keeps_old_boundary_fence_whole_and_balances_chunks():
    _plugin, telegram_api = _load_skill()
    client = telegram_api.TelegramClient("token")
    calls = []

    async def call(_method, *, data, **_kwargs):
        calls.append(dict(data))
        return {"result": {"message_id": len(calls)}}

    client.call = call
    fence = "```text\n" + ("x" * 3600) + "\n```"
    asyncio.run(client.send_message(42, fence))
    assert len(calls) == 1
    assert calls[0]["text"].startswith("<pre>")
    assert calls[0]["text"].endswith("</pre>")
    assert len(calls[0]["text"]) <= 4096
    _assert_balanced(calls[0]["text"])

    calls.clear()
    oversized = "**" + ("word " * 2200) + "**"
    asyncio.run(client.send_message(42, oversized))
    assert len(calls) >= 2
    assert all(len(item["text"]) <= 4096 for item in calls)
    for item in calls:
        _assert_balanced(item["text"])


def test_block_aware_chunking_keeps_quote_and_table_blocks_whole():
    _plugin, telegram_api = _load_skill()
    quote = "> first line\n> second line\n"
    table = "| A | B |\n| --- | --- |\n| one | two |"
    source = ("p" * 4085) + "\n\n" + quote + "\n" + table
    chunks = telegram_api.markdown_to_telegram_chunks(source)
    rendered_quote = telegram_api.markdown_to_telegram_html(quote)
    rendered_table = telegram_api.markdown_to_telegram_html(table)

    assert any(rendered_quote in chunk for chunk in chunks)
    assert any(rendered_table in chunk for chunk in chunks)
    assert all(len(chunk) <= 4096 for chunk in chunks)
    for chunk in chunks:
        _assert_balanced(chunk)


# Termination guards: pytest-timeout, not ``signal.alarm`` — Windows has no
# SIGALRM, and an unhandled alarm would kill the whole pytest worker. The
# bound is a HANG guard, not a perf budget: the 100 KB single-block case takes
# ~4 s on a fast Linux host and exceeded 10 s on windows-latest under xdist
# (a thread-method timeout kills the worker), while the CI ceiling is 300 s.
@pytest.mark.timeout(120)
def test_chunker_terminates_for_oversized_link_tag_and_pre_block():
    _plugin, telegram_api = _load_skill()
    link_source = (
        ("filler " * 700)
        + " [link](https://example.com/"
        + ("q" * 4200)
        + ") "
        + ("tail " * 200)
    )
    pre_source = "```text\n" + ("x" * 12_000) + "\n```"

    link_chunks = telegram_api.markdown_to_telegram_chunks(link_source)
    pre_chunks = telegram_api.markdown_to_telegram_chunks(pre_source)

    assert link_chunks
    assert pre_chunks
    assert all(telegram_api._u16len(chunk) <= 4096 for chunk in link_chunks + pre_chunks)
    for chunk in pre_chunks:
        _assert_balanced(chunk)


@pytest.mark.timeout(120)
def test_chunker_balances_100kb_single_block_paragraph_within_timeout():
    _plugin, telegram_api = _load_skill()
    source = "**" + ("word " * 20_000) + "**"

    chunks = telegram_api.markdown_to_telegram_chunks(source)

    assert len(chunks) > 1
    assert all(telegram_api._u16len(chunk) <= 4096 for chunk in chunks)
    for chunk in chunks:
        _assert_balanced(chunk)


def test_chunker_counts_astral_characters_as_two_utf16_code_units():
    _plugin, telegram_api = _load_skill()
    source = " ".join(f"word{index} 😀" for index in range(900))
    chunks = telegram_api.markdown_to_telegram_chunks(source)
    assert len(chunks) > 1
    assert all(telegram_api._u16len(chunk) <= 4096 for chunk in chunks)


def test_plain_send_chunks_astral_text_by_utf16_units():
    _plugin, telegram_api = _load_skill()
    client = telegram_api.TelegramClient("token")
    calls = []

    async def call(_method, *, data, **_kwargs):
        calls.append(dict(data))
        return {"result": {"message_id": len(calls)}}

    client.call = call
    asyncio.run(client.send_message(42, "😀" * 3000, parse_mode=""))

    assert len(calls) == 2
    assert all(telegram_api._u16len(item["text"]) <= 4096 for item in calls)
    assert "".join(item["text"] for item in calls) == "😀" * 3000


def test_chunker_drops_whitespace_only_chunks_without_losing_content():
    _plugin, telegram_api = _load_skill()
    source = "```\n" + ("a" * 9000) + "\n```\n\n```\n" + ("b" * 9000) + "\n```"
    chunks = telegram_api.markdown_to_telegram_chunks(source)
    plain = "".join(telegram_api._telegram_html_to_plain(chunk) for chunk in chunks)

    assert "a" * 9000 in plain
    assert "b" * 9000 in plain
    assert all(telegram_api._telegram_html_to_plain(chunk).strip() for chunk in chunks)

    boundary_chunks = telegram_api.markdown_to_telegram_chunks("```\n" + ("x" * 4085) + "\n```")
    assert all(telegram_api._telegram_html_to_plain(chunk).strip() for chunk in boundary_chunks)
    assert all(telegram_api._telegram_html_to_plain(chunk).strip("\n") for chunk in boundary_chunks)

    short_chunks = telegram_api.markdown_to_telegram_chunks("normal short message")
    assert short_chunks == ["normal short message"]
    assert telegram_api._telegram_html_to_plain(short_chunks[0]) == "normal short message"


def test_send_message_defensively_skips_whitespace_only_chunks(monkeypatch):
    _plugin, telegram_api = _load_skill()
    client = telegram_api.TelegramClient("token")
    calls = []

    monkeypatch.setattr(
        telegram_api,
        "markdown_to_telegram_chunks",
        lambda _source: ["first", "<pre>\n</pre>", "second"],
    )

    async def call(_method, *, data, **_kwargs):
        calls.append(dict(data))
        return {"result": {"message_id": len(calls)}}

    client.call = call
    asyncio.run(client.send_message(42, "source"))
    assert [item["text"] for item in calls] == ["first", "second"]


def test_chunk_boundary_sweep_includes_closing_tag_cost():
    _plugin, telegram_api = _load_skill()
    for size in range(4070, 4121):
        chunks = telegram_api.markdown_to_telegram_chunks("**" + ("x" * size) + "**")
        assert all(telegram_api._u16len(chunk) <= 4096 for chunk in chunks), size
        for chunk in chunks:
            _assert_balanced(chunk)


def test_oversized_crossed_emphasis_is_balanced_after_splitting():
    _plugin, telegram_api = _load_skill()
    source = "**a *" + ("b" * 9000) + "** c*"
    chunks = telegram_api.markdown_to_telegram_chunks(source)
    assert len(chunks) > 1
    assert all(telegram_api._u16len(chunk) <= 4096 for chunk in chunks)
    for chunk in chunks:
        _assert_balanced(chunk)


def test_literal_legacy_placeholder_names_survive_reconstruction():
    _plugin, telegram_api = _load_skill()
    source = (
        "PREPLACEHOLDER0 CODEPLACEHOLDER0 LITERALPLACEHOLDER0\n"
        "```text\nreal pre\n``` `real code` $$real literal$$"
    )
    rendered = telegram_api.markdown_to_telegram_html(source)
    assert "PREPLACEHOLDER0 CODEPLACEHOLDER0 LITERALPLACEHOLDER0" in rendered
    assert "<pre>real pre\n</pre>" in rendered
    assert "<code>real code</code>" in rendered
    assert "$$real literal$$" in rendered


class _Api:
    def __init__(self, state_dir: Path):
        self.state_dir = state_dir
        self.logs = []

    def get_state_dir(self):
        return str(self.state_dir)

    def get_settings(self, _keys):
        return {"TELEGRAM_BOT_TOKEN": "token"}

    def log(self, level, message, **_fields):
        self.logs.append((level, message))


class _Client:
    def __init__(self, *, audio_error=None, keyboard_error=None):
        self.audio_error = audio_error
        self.keyboard_error = keyboard_error
        self.audio = []
        self.documents = []
        self.keyboards = []
        self.messages = []

    async def send_audio(self, chat_id, file_bytes, filename, **kwargs):
        self.audio.append((chat_id, file_bytes, filename, kwargs))
        if self.audio_error:
            raise self.audio_error

    async def send_document(self, chat_id, file_bytes, filename, **kwargs):
        self.documents.append((chat_id, file_bytes, filename, kwargs))

    async def send_message_with_inline_keyboard(self, chat_id, text, keyboard):
        self.keyboards.append((chat_id, text, keyboard))
        if self.keyboard_error:
            raise self.keyboard_error

    async def send_message(self, chat_id, text, parse_mode="HTML"):
        self.messages.append((chat_id, text, parse_mode))
        return 1


def _configured_api(tmp_path: Path) -> _Api:
    (tmp_path / "settings.json").write_text(
        json.dumps({"TELEGRAM_CHAT_ID": "42", "TELEGRAM_MIRROR_MODE": "all"}),
        encoding="utf-8",
    )
    return _Api(tmp_path)


@pytest.mark.parametrize(
    ("filename", "mime", "expected"),
    [
        ("track.mp3", "application/octet-stream", "audio"),
        ("track.m4a", "application/octet-stream", "audio"),
        ("track.bin", "audio/mpeg", "audio"),
        ("track.wav", "audio/wav", "document"),
        ("track.ogg", "audio/ogg", "document"),
    ],
)
def test_document_audio_routing(tmp_path, monkeypatch, filename, mime, expected):
    plugin, _telegram_api = _load_skill()
    client = _Client()
    monkeypatch.setattr(plugin, "TelegramClient", lambda _token, **_kwargs: client)
    event = {
        "chat_id": 42,
        "transport": {},
        "file_base64": base64.b64encode(b"audio").decode("ascii"),
        "filename": filename,
        "mime": mime,
    }
    asyncio.run(plugin._make_document(_configured_api(tmp_path))(event))
    assert (len(client.audio), len(client.documents)) == (
        (1, 0) if expected == "audio" else (0, 1)
    )


def test_send_audio_rejection_falls_back_to_document_once(tmp_path, monkeypatch):
    plugin, _telegram_api = _load_skill()
    client = _Client(audio_error=plugin.TelegramRequestRejected("rejected", status_code=400))
    monkeypatch.setattr(plugin, "TelegramClient", lambda _token, **_kwargs: client)
    event = {
        "chat_id": 42,
        "file_base64": base64.b64encode(b"audio").decode("ascii"),
        "filename": "track.mp3",
        "mime": "audio/mpeg",
    }
    asyncio.run(plugin._make_document(_configured_api(tmp_path))(event))
    assert len(client.audio) == 1
    assert len(client.documents) == 1


@pytest.mark.parametrize("status_code", [401, 429, 500])
def test_send_audio_non_format_rejection_never_double_sends(tmp_path, monkeypatch, status_code):
    """Auth (401) and transient (429/5xx) rejections re-raise without the
    document fallback: the upload may have been throttled mid-flight, and a
    second sendDocument would risk a duplicate delivery."""
    plugin, _telegram_api = _load_skill()
    client = _Client(
        audio_error=plugin.TelegramRequestRejected("rejected", status_code=status_code)
    )
    monkeypatch.setattr(plugin, "TelegramClient", lambda _token, **_kwargs: client)
    api = _configured_api(tmp_path)
    event = {
        "chat_id": 42,
        "file_base64": base64.b64encode(b"audio").decode("ascii"),
        "filename": "track.mp3",
        "mime": "audio/mpeg",
    }
    asyncio.run(plugin._make_document(api)(event))
    assert len(client.audio) == 1
    assert client.documents == []
    assert any(level == "error" for level, _message in api.logs)


def test_links_event_renders_at_most_twelve_url_buttons(tmp_path, monkeypatch):
    plugin, _telegram_api = _load_skill()
    client = _Client()
    monkeypatch.setattr(plugin, "TelegramClient", lambda _token, **_kwargs: client)
    actions = [
        {"label": f"Link {index}", "url": f"https://example.com/{index}"}
        for index in range(14)
    ]
    event = {"chat_id": 42, "transport": {}, "title": "References", "actions": actions}
    asyncio.run(plugin._make_links(_configured_api(tmp_path))(event))
    assert client.keyboards == [
        (
            42,
            "References",
            [[{"text": f"Link {index}", "url": f"https://example.com/{index}"}] for index in range(12)],
        )
    ]
    assert client.messages == []


def test_links_event_filters_actions_before_twelve_button_cap(tmp_path, monkeypatch):
    plugin, _telegram_api = _load_skill()
    client = _Client()
    monkeypatch.setattr(plugin, "TelegramClient", lambda _token, **_kwargs: client)
    actions = [
        {"label": f"Invalid {index}"}
        for index in range(12)
    ] + [{"label": "Valid", "url": "https://example.com/valid"}]
    event = {"chat_id": 42, "transport": {}, "title": "References", "actions": actions}
    asyncio.run(plugin._make_links(_configured_api(tmp_path))(event))
    assert client.keyboards == [
        (42, "References", [[{"text": "Valid", "url": "https://example.com/valid"}]])
    ]
    assert client.messages == []


def test_links_keyboard_failure_falls_back_to_plain_text(tmp_path, monkeypatch):
    plugin, _telegram_api = _load_skill()
    client = _Client(
        keyboard_error=plugin.TelegramRequestRejected(
            "keyboard rejected",
            status_code=400,
            plain_retry_safe=True,
        )
    )
    monkeypatch.setattr(plugin, "TelegramClient", lambda _token, **_kwargs: client)
    event = {
        "chat_id": 42,
        "transport": {},
        "title": "",
        "actions": [{"label": "Docs", "url": "https://example.com/docs"}],
    }
    asyncio.run(plugin._make_links(_configured_api(tmp_path))(event))
    assert len(client.keyboards) == 1
    assert client.messages == [(42, "Links\nDocs — https://example.com/docs", "")]


def test_links_keyboard_transport_failure_has_no_plain_fallback(tmp_path, monkeypatch):
    plugin, _telegram_api = _load_skill()
    client = _Client(keyboard_error=plugin.TelegramTransportError("ambiguous delivery"))
    monkeypatch.setattr(plugin, "TelegramClient", lambda _token, **_kwargs: client)
    api = _configured_api(tmp_path)
    event = {
        "chat_id": 42,
        "transport": {},
        "title": "Links",
        "actions": [{"label": "Docs", "url": "https://example.com/docs"}],
    }
    asyncio.run(plugin._make_links(api)(event))
    assert len(client.keyboards) == 1
    assert client.messages == []
    assert api.logs == [("error", "Telegram links error: ambiguous delivery")]


def test_links_event_honors_telegram_only_transport_filter(tmp_path, monkeypatch):
    plugin, _telegram_api = _load_skill()
    client = _Client()
    monkeypatch.setattr(plugin, "TelegramClient", lambda _token, **_kwargs: client)
    (tmp_path / "settings.json").write_text(
        json.dumps({"TELEGRAM_CHAT_ID": "42", "TELEGRAM_MIRROR_MODE": "telegram_only"}),
        encoding="utf-8",
    )
    event = {
        "chat_id": 42,
        "transport": {"kind": "web"},
        "actions": [{"label": "Docs", "url": "https://example.com/docs"}],
    }
    asyncio.run(plugin._make_links(_Api(tmp_path))(event))
    assert client.keyboards == []
    assert client.messages == []
