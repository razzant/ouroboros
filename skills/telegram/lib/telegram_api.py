from __future__ import annotations

import base64
import html as html_lib
import mimetypes
import re
from typing import Any, Dict, Optional

import httpx

# Telegram hard-caps a single sendMessage at 4096 UTF-16 code units.
_TELEGRAM_TEXT_LIMIT = 4096
_TABLE_MAX_ROWS = 30
_TABLE_MAX_COLUMNS = 6
_TABLE_MAX_CELL_CHARS = 24
# Match Ouroboros's existing per-photo transfer ceiling for every Telegram download.
_MAX_TELEGRAM_DOWNLOAD_BYTES = 10 * 1024 * 1024
_NOT_MODIFIED_PREFIX = "bad request: message is not modified"


def _u16len(value: str) -> int:
    """Return Telegram's text length: UTF-16 code units, not code points."""
    return sum(2 if ord(char) > 0xFFFF else 1 for char in value)


def _take_u16_prefix(value: str, budget: int) -> tuple[str, str]:
    """Split value at a UTF-16-unit boundary without bisecting a code point."""
    used = 0
    index = 0
    while index < len(value):
        width = 2 if ord(value[index]) > 0xFFFF else 1
        if used + width > budget:
            break
        used += width
        index += 1
    return value[:index], value[index:]


class TelegramRequestRejected(RuntimeError):
    """Telegram returned an explicit negative API response."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 0,
        plain_retry_safe: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = int(status_code or 0)
        self.plain_retry_safe = bool(plain_retry_safe)

    @property
    def transient(self) -> bool:
        return self.status_code == 429 or self.status_code >= 500


class TelegramTransportError(RuntimeError):
    """Telegram could not return a trustworthy API response yet."""


# Shared degraded-transport pacing contract for the poller and notifier loops:
# the first retry waits TELEGRAM_RETRY_INITIAL_SEC and each further consecutive
# failure doubles the wait monotonically up to TELEGRAM_RETRY_MAX_SEC; any
# successful API round resets the wait to the initial value.
TELEGRAM_RETRY_INITIAL_SEC = 5
TELEGRAM_RETRY_MAX_SEC = 60


def next_telegram_retry_delay(current: float) -> float:
    """Next monotone backoff step after one more consecutive transient failure."""
    return min(max(float(current), TELEGRAM_RETRY_INITIAL_SEC) * 2, TELEGRAM_RETRY_MAX_SEC)


def is_transient_telegram_error(exc: BaseException) -> bool:
    """Whether *exc* is a typed transient Telegram failure worth retrying."""
    if isinstance(exc, TelegramTransportError):
        return True
    return isinstance(exc, TelegramRequestRejected) and exc.transient


def _chunk_raw_text(text: str, limit: int = _TELEGRAM_TEXT_LIMIT) -> list[str]:
    """Split raw text into <=limit UTF-16-unit pieces on line/space boundaries."""
    if _u16len(text) <= limit:
        return [text]
    chunks: list[str] = []
    buf = ""
    for line in text.split("\n"):
        while _u16len(line) > limit:
            # A single very long line: break on the last space within the window,
            # else hard-cut at the limit.
            prefix, remainder = _take_u16_prefix(line, limit)
            cut = prefix.rfind(" ")
            if cut > 0:
                piece = line[:cut]
                line = line[cut:].lstrip(" ")
            else:
                piece = prefix
                line = remainder
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.append(piece)
        candidate = f"{buf}\n{line}" if buf else line
        if _u16len(candidate) > limit:
            if buf:
                chunks.append(buf)
            buf = line
        else:
            buf = candidate
    if buf:
        chunks.append(buf)
    return chunks or [""]


def _escape_html(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _split_table_row(line: str) -> list[str]:
    row = line.strip()
    if row.startswith("|"):
        row = row[1:]
    if row.endswith("|") and not row.endswith(r"\|"):
        row = row[:-1]
    cells: list[str] = []
    current: list[str] = []
    escaped = False
    for char in row:
        if char == "|" and not escaped:
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(char)
        if char == "\\" and not escaped:
            escaped = True
        else:
            escaped = False
    cells.append("".join(current).strip())
    return [cell.replace(r"\|", "|") for cell in cells]


def _is_table_delimiter(line: str, header: str) -> bool:
    cells = _split_table_row(line)
    return (
        "|" in line
        and len(cells) == len(_split_table_row(header))
        and all(re.fullmatch(r":?-+:?", cell) for cell in cells)
    )


def _is_non_code_indent(line: str) -> bool:
    """Return whether CommonMark treats the line as non-code indentation."""
    columns = 0
    for char in line:
        if char == " ":
            columns += 1
        elif char == "\t":
            columns += 4 - (columns % 4)
        else:
            break
        if columns >= 4:
            return False
    return True


def _is_table_start(header_line: str, delimiter_line: str) -> bool:
    return (
        _is_non_code_indent(header_line)
        and "|" in header_line
        and _is_table_delimiter(delimiter_line, header_line)
    )


def _truncate_table_cell(cell: str) -> str:
    if len(cell) <= _TABLE_MAX_CELL_CHARS:
        return cell
    return cell[: _TABLE_MAX_CELL_CHARS - 1] + "…"


def _table_html(lines: list[str]) -> str:
    rows = [_split_table_row(lines[0])]
    rows.extend(_split_table_row(line) for line in lines[2:])
    column_count = min(max((len(row) for row in rows), default=0), _TABLE_MAX_COLUMNS)
    truncated = len(rows) > _TABLE_MAX_ROWS or any(len(row) > _TABLE_MAX_COLUMNS for row in rows)
    visible_rows = rows[:_TABLE_MAX_ROWS]
    normalized: list[list[str]] = []
    for row in visible_rows:
        normalized.append(
            [_truncate_table_cell(row[index] if index < len(row) else "") for index in range(column_count)]
        )
    widths = [
        max(3, max((len(row[index]) for row in normalized), default=0))
        for index in range(column_count)
    ]

    def render(row: list[str]) -> str:
        return " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)).rstrip()

    grid: list[str] = []
    if normalized:
        grid.append(render(normalized[0]))
        grid.append("-+-".join("-" * width for width in widths))
        grid.extend(render(row) for row in normalized[1:])
    if truncated:
        grid.append("…table truncated")
    return f"<pre>{_escape_html(chr(10).join(grid))}</pre>"


def _replace_gfm_tables(text: str, pre_placeholder_map: dict[str, str]) -> str:
    lines = text.splitlines(keepends=True)
    output: list[str] = []
    index = 0
    while index < len(lines):
        body = lines[index].rstrip("\r\n")
        if (
            index + 1 < len(lines)
            and _is_table_start(body, lines[index + 1].rstrip("\r\n"))
        ):
            end = index + 2
            while end < len(lines):
                candidate = lines[end].rstrip("\r\n")
                if not candidate.strip() or "|" not in candidate:
                    break
                end += 1
            placeholder = f"\x00PRE{len(pre_placeholder_map)}\x00"
            pre_placeholder_map[placeholder] = _table_html(
                [line.rstrip("\r\n") for line in lines[index:end]]
            )
            newline = "\n" if lines[end - 1].endswith(("\n", "\r")) else ""
            output.append(placeholder + newline)
            index = end
            continue
        output.append(lines[index])
        index += 1
    return "".join(output)


def markdown_to_telegram_html(text: str) -> str:
    """Convert standard rich Markdown text into Telegram-compliant HTML syntax."""
    if not text:
        return text

    # Placeholder dictionaries
    pre_placeholder_map: dict[str, str] = {}
    code_placeholder_map: dict[str, str] = {}
    literal_placeholder_map: dict[str, str] = {}

    # 1. Protect fenced blocks before table detection and inline formatting.
    def replace_pre(match: re.Match) -> str:
        code_content = match.group(1)
        placeholder = f"\x00PRE{len(pre_placeholder_map)}\x00"
        pre_placeholder_map[placeholder] = f"<pre>{_escape_html(code_content)}</pre>"
        return placeholder

    text = re.sub(
        r"```(?:[A-Za-z0-9_-]*[ \t]*\r?\n)?(.*?)```",
        replace_pre,
        text,
        flags=re.DOTALL,
    )

    # 2. Preserve supported LaTeX delimiters as literal text. Placeholders keep
    # emphasis and link regexes from interpreting math source.
    def replace_literal(match: re.Match) -> str:
        placeholder = f"\x00LITERAL{len(literal_placeholder_map)}\x00"
        literal_placeholder_map[placeholder] = _escape_html(match.group(0))
        return placeholder

    text = re.sub(r"\$\$(.+?)\$\$", replace_literal, text, flags=re.DOTALL)
    text = re.sub(r"\\\((.+?)\\\)", replace_literal, text, flags=re.DOTALL)
    text = re.sub(r"\\\[(.+?)\\\]", replace_literal, text, flags=re.DOTALL)

    # 3. Convert GFM pipe tables to bounded monospace grids, then escape all
    # remaining literal HTML from the source.
    text = _replace_gfm_tables(text, pre_placeholder_map)
    text = _escape_html(text)

    # 4. Extract inline code blocks.
    def replace_code(match: re.Match) -> str:
        inner = match.group(1)
        placeholder = f"\x00CODE{len(code_placeholder_map)}\x00"
        code_placeholder_map[placeholder] = f"<code>{inner}</code>"
        return placeholder

    text = re.sub(r"`([^`\n]+)`", replace_code, text)

    # 5. Headers, task lists, and ordinary list formatting line-by-line.
    lines = []
    for line in text.split("\n"):
        header_match = re.match(r"^(\s*)#{1,6}\s+(.+)$", line)
        if header_match:
            indent = header_match.group(1) or ""
            content = header_match.group(2)
            lines.append(f"{indent}<b>{content}</b>")
        else:
            task_match = re.match(r"^(\s*)(?:[-*+]|\d+\.)\s+\[([ xX])\]\s+(.+)$", line)
            if task_match:
                glyph = "☑" if task_match.group(2).lower() == "x" else "☐"
                lines.append(f"{task_match.group(1)}{glyph} {task_match.group(3)}")
                continue
            # Replace starting list bullet * or - with •
            bullet_match = re.match(r"^(\s*)[*-]\s+(.+)$", line)
            if bullet_match:
                lines.append(f"{bullet_match.group(1)}• {bullet_match.group(2)}")
            else:
                lines.append(line)
    text = "\n".join(lines)

    # 6. Bold and Italic replacing outside of protected blocks.
    # Asterisk patterns match anywhere — `**bold**` and `*italic*` are unambiguous.
    # Underscore patterns require non-word context on both sides so identifiers
    # like `chat_id`, `state_dir`, `OUROBOROS_MODEL` inside bold spans do NOT
    # trigger spurious italic wraps that cross outer tag boundaries (which
    # would produce malformed nested HTML and a Telegram 400 Bad Request).
    text = re.sub(r"\*\*\*(.+?)\*\*\*", r"<b><i>\1</i></b>", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\w)__(?=\S)([^_\n]+?)(?<=\S)__(?!\w)", r"<b>\1</b>", text)
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    text = re.sub(r"(?<!\w)_(?=\S)([^_\n]+?)(?<=\S)_(?!\w)", r"<i>\1</i>", text)

    # 7. Links [text](url) -> <a href="url">text</a>
    text = re.sub(r"\[(.*?)\]\((.*?)\)", r'<a href="\2">\1</a>', text)

    # 8. Reconstruct protected blocks (longest placeholder first prevents
    # substring prefix collisions such as PLACEHOLDER1/PLACEHOLDER10).
    placeholders = sorted(
        [
            *pre_placeholder_map.items(),
            *code_placeholder_map.items(),
            *literal_placeholder_map.items(),
        ],
        key=lambda item: len(item[0]),
        reverse=True,
    )
    for _ in range(len(placeholders) + 1):
        reconstructed = text
        for placeholder, replacement in placeholders:
            reconstructed = reconstructed.replace(placeholder, replacement)
        if reconstructed == text:
            break
        text = reconstructed

    # Telegram rejects NUL bytes even without parse mode. Remove any source NUL
    # or unresolved sentinel defensively before this text reaches a send path.
    text = text.replace("\x00", "")

    return text


def _markdown_blocks(text: str) -> list[str]:
    """Split markdown without bisecting fenced, table, or quote blocks."""
    lines = text.splitlines(keepends=True)
    blocks: list[str] = []
    index = 0

    def body(at: int) -> str:
        return lines[at].rstrip("\r\n")

    def starts_special(at: int) -> bool:
        if re.match(r"^\s*```", body(at)) or re.match(r"^\s*>", body(at)):
            return True
        return (
            at + 1 < len(lines)
            and _is_table_start(body(at), body(at + 1))
        )

    while index < len(lines):
        start = index
        if re.match(r"^\s*```", body(index)):
            index += 1
            while index < len(lines):
                closing = bool(re.match(r"^\s*```\s*$", body(index)))
                index += 1
                if closing:
                    break
        elif (
            index + 1 < len(lines)
            and _is_table_start(body(index), body(index + 1))
        ):
            index += 2
            while index < len(lines) and body(index).strip() and "|" in body(index):
                index += 1
        elif re.match(r"^\s*>", body(index)):
            index += 1
            while index < len(lines) and re.match(r"^\s*>", body(index)):
                index += 1
        elif not body(index).strip():
            index += 1
            while index < len(lines) and not body(index).strip():
                index += 1
        else:
            index += 1
            while index < len(lines) and body(index).strip() and not starts_special(index):
                index += 1
        blocks.append("".join(lines[start:index]))
    return blocks or [""]


_HTML_TOKEN_RE = re.compile(r"<[^>]+>|&(?:#[0-9]+|#x[0-9A-Fa-f]+|[A-Za-z]+);|\s|[^\s<&]+|[<&]")
_HTML_TAG_RE = re.compile(r"</?([A-Za-z0-9]+)")


def _split_html_balanced(value: str, limit: int = _TELEGRAM_TEXT_LIMIT) -> list[str]:
    """Split one oversized HTML block while closing and reopening active tags."""
    chunks: list[str] = []
    current = ""
    stack: list[tuple[str, str]] = []

    def suffix(active_stack: list[tuple[str, str]]) -> str:
        return "".join(f"</{name}>" for name, _opening in reversed(active_stack))

    def closing_suffix() -> str:
        return suffix(stack)

    def reopen_prefix() -> str:
        return "".join(opening for _name, opening in stack)

    def flush() -> None:
        nonlocal current
        if current:
            chunks.append(current + closing_suffix())
        current = reopen_prefix()

    def append_slices(raw: str) -> None:
        while raw:
            piece, raw = _take_u16_prefix(raw, limit)
            if not piece:
                piece, raw = raw[0], raw[1:]
            chunks.append(piece)

    def hard_reset() -> None:
        nonlocal current
        payload = current + closing_suffix()
        if payload:
            if _u16len(payload) <= limit:
                chunks.append(payload)
            else:
                append_slices(payload)
        current = ""
        stack.clear()

    tokens = _HTML_TOKEN_RE.findall(value)
    iterations = 0
    for token_index, token in enumerate(tokens):
        iterations += 1
        if iterations > 200_000:
            hard_reset()
            append_slices("".join(tokens[token_index:]))
            return chunks or [""]

        tag_match = _HTML_TAG_RE.match(token) if token.startswith("<") else None
        if tag_match:
            name = tag_match.group(1).lower()
            is_close = token.startswith("</")
            output = token
            if is_close:
                matching_index = next(
                    (index for index in range(len(stack) - 1, -1, -1) if stack[index][0] == name),
                    None,
                )
                if matching_index is None:
                    output = _escape_html(token)
                    projected_stack = list(stack)
                else:
                    intervening = stack[matching_index + 1 :]
                    output = suffix(intervening) + token + "".join(
                        opening for _tag, opening in intervening
                    )
                    projected_stack = stack[:matching_index] + intervening
            elif token.endswith("/>"):
                projected_stack = list(stack)
            else:
                projected_stack = stack + [(name, token)]

            projected_suffix = suffix(projected_stack)
            projected_length = _u16len(current) + _u16len(output) + _u16len(projected_suffix)
            if projected_length > limit and current != reopen_prefix():
                flush()
                projected_length = _u16len(current) + _u16len(output) + _u16len(projected_suffix)
            if projected_length > limit:
                hard_reset()
                append_slices(_escape_html(token))
                continue
            current += output
            stack[:] = projected_stack
            continue

        remaining = token
        while remaining:
            iterations += 1
            if iterations > 200_000:
                hard_reset()
                append_slices(remaining + "".join(tokens[token_index + 1 :]))
                return chunks or [""]

            available = limit - _u16len(current) - _u16len(closing_suffix())
            if available <= 0:
                flush()
                available = limit - _u16len(current) - _u16len(closing_suffix())
                if available <= 0:
                    hard_reset()
                    append_slices(remaining)
                    remaining = ""
                    break
            if _u16len(remaining) <= available:
                current += remaining
                break
            if remaining.startswith("&") and remaining.endswith(";"):
                flush()
                if _u16len(remaining) > limit - _u16len(current) - _u16len(closing_suffix()):
                    hard_reset()
                    append_slices(_escape_html(remaining))
                    remaining = ""
                continue
            piece, remaining = _take_u16_prefix(remaining, available)
            if not piece:
                hard_reset()
                append_slices(remaining)
                remaining = ""
                break
            current += piece
            flush()
    if current:
        chunks.append(current + closing_suffix())
    return chunks or [""]


def markdown_to_telegram_chunks(text: str, limit: int = _TELEGRAM_TEXT_LIMIT) -> list[str]:
    """Convert markdown block-by-block, then pack balanced Telegram HTML."""
    chunks: list[str] = []
    buffer = ""

    def append_visible(chunk: str) -> None:
        if _telegram_html_to_plain(chunk).strip():
            chunks.append(chunk)

    for block in _markdown_blocks(text):
        converted = markdown_to_telegram_html(block)
        if _u16len(converted) > limit:
            if buffer:
                append_visible(buffer)
                buffer = ""
            for chunk in _split_html_balanced(converted, limit):
                append_visible(chunk)
        elif _u16len(buffer) + _u16len(converted) <= limit:
            buffer += converted
        else:
            if buffer:
                append_visible(buffer)
            buffer = converted
    if buffer:
        append_visible(buffer)
    return chunks


def _telegram_html_to_plain(value: str) -> str:
    return html_lib.unescape(re.sub(r"<[^>]+>", "", value))


class TelegramClient:
    def __init__(self, token: str, *, trust_env: bool = False):
        self.token = str(token or "").strip()
        self.trust_env = bool(trust_env)
        if not self.token:
            raise ValueError("TELEGRAM_BOT_TOKEN is missing")
        self.api_base = f"https://api.telegram.org/bot{self.token}"
        self.file_base = f"https://api.telegram.org/file/bot{self.token}"

    async def call(self, method: str, *, data: Optional[dict] = None, files: Optional[dict] = None, timeout: int = 30) -> Dict[str, Any]:
        method_text = str(method or "")
        safe_method = method_text if re.fullmatch(r"[A-Za-z][A-Za-z0-9]{0,63}", method_text) else "request"
        try:
            # trust_env trades ambient-proxy/SSL_CERT isolation for a proxy-routed install's
            # only egress; decided once by the caller via net_transport.env_proxies_configured.
            async with httpx.AsyncClient(timeout=timeout, trust_env=self.trust_env) as client:
                response = await client.post(f"{self.api_base}/{method_text}", data=data, files=files)
        except httpx.TimeoutException:
            raise TelegramTransportError(f"Telegram API timed out during {safe_method}.") from None
        except httpx.HTTPError as exc:
            raise TelegramTransportError(
                f"Telegram API transport failed during {safe_method} ({type(exc).__name__})."
            ) from None
        try:
            payload = response.json()
        except ValueError:
            raise TelegramTransportError(
                f"Telegram API returned invalid JSON during {safe_method}."
            ) from None
        if not isinstance(payload, dict):
            raise TelegramTransportError(
                f"Telegram API returned an invalid response during {safe_method}."
            )
        try:
            api_status = int(payload.get("error_code") or response.status_code)
        except (TypeError, ValueError):
            api_status = response.status_code
        description = str(payload.get("description") or "").strip()
        if safe_method == "editMessageText" and description.casefold().startswith(_NOT_MODIFIED_PREFIX):
            # Preserve the one benign signal callers rely on without forwarding
            # Telegram's free-form suffix (which may echo the token-bearing URL).
            raise TelegramRequestRejected(
                "Telegram API editMessageText: message is not modified.",
                status_code=response.status_code,
            )
        if response.status_code >= 400:
            raise TelegramRequestRejected(
                f"Telegram API {safe_method} returned HTTP {response.status_code}.",
                status_code=response.status_code,
                plain_retry_safe=(safe_method == "sendMessage" and response.status_code == 400),
            )
        if not payload.get("ok"):
            raise TelegramRequestRejected(
                f"Telegram API rejected {safe_method}.",
                status_code=api_status,
                plain_retry_safe=(safe_method == "sendMessage" and api_status == 400),
            )
        return payload

    async def _download_bytes(self, file_path: str) -> bytes:
        try:
            async with httpx.AsyncClient(timeout=30, trust_env=self.trust_env) as client:
                async with client.stream("GET", f"{self.file_base}/{file_path}") as response:
                    if response.status_code >= 400:
                        raise RuntimeError(f"Telegram file download returned HTTP {response.status_code}")
                    try:
                        announced = int(response.headers.get("content-length", ""))
                    except (TypeError, ValueError):
                        announced = 0
                    if announced > _MAX_TELEGRAM_DOWNLOAD_BYTES:
                        raise RuntimeError("Telegram file exceeds the 10 MiB download limit.")
                    content = bytearray()
                    async for chunk in response.aiter_bytes():
                        if len(content) + len(chunk) > _MAX_TELEGRAM_DOWNLOAD_BYTES:
                            raise RuntimeError("Telegram file exceeds the 10 MiB download limit.")
                        content.extend(chunk)
        except httpx.TimeoutException:
            raise RuntimeError("Telegram file download timed out.") from None
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"Telegram file download transport failed ({type(exc).__name__})."
            ) from None
        return bytes(content)

    async def get_updates(self, offset: int) -> list[dict]:
        payload = await self.call("getUpdates", data={"timeout": 20, "offset": offset}, timeout=25)
        return list(payload.get("result") or [])

    async def send_message(self, chat_id: int, text: str, parse_mode: str = "HTML") -> int:
        """Send a text message, chunking past Telegram's 4096-unit limit.

        Markdown is converted at block granularity before chunks are packed.
        Protected blocks stay whole unless one block exceeds Telegram's limit;
        oversized blocks are split with active HTML tags re-balanced.
        Returns the LAST chunk's message_id (0 on parse failure / empty input).
        """
        source = str(text or "")
        chunks = (
            markdown_to_telegram_chunks(source)
            if parse_mode == "HTML"
            else _chunk_raw_text(source)
        )
        last_message_id = 0
        for formatted in chunks:
            visible_text = (
                _telegram_html_to_plain(formatted) if parse_mode == "HTML" else formatted
            )
            if not visible_text.strip():
                continue
            data = {"chat_id": str(chat_id), "text": formatted}
            if parse_mode:
                data["parse_mode"] = parse_mode
            try:
                payload = await self.call("sendMessage", data=data, timeout=20)
            except TelegramRequestRejected as exc:
                if parse_mode != "HTML" or not exc.plain_retry_safe:
                    raise
                payload = await self.call(
                    "sendMessage",
                    data={"chat_id": str(chat_id), "text": _telegram_html_to_plain(formatted)},
                    timeout=20,
                )
            try:
                last_message_id = int((payload.get("result") or {}).get("message_id") or 0)
            except (TypeError, ValueError):
                last_message_id = 0
        return last_message_id

    async def edit_message_text(self, chat_id: int, message_id: int, text: str, parse_mode: str = "HTML") -> bool:
        """Replace the text of an existing message in-place (silent mode). Returns True on success.

        Failures (message too old, deleted, identical content, parse error) are
        suppressed so the caller can fall back to send_message + reset tracking.
        """
        formatted = markdown_to_telegram_html(text) if parse_mode == "HTML" else text
        data = {
            "chat_id": str(chat_id),
            "message_id": str(message_id),
            "text": formatted,
        }
        if parse_mode:
            data["parse_mode"] = parse_mode
        try:
            await self.call("editMessageText", data=data, timeout=20)
            return True
        except Exception as exc:
            # "message is not modified" means the bubble already shows this exact
            # text — treat it as success so the caller does NOT post a duplicate.
            if "not modified" in str(exc).lower():
                return True
            return False

    async def send_chat_action(self, chat_id: int, action: str = "typing") -> None:
        await self.call("sendChatAction", data={"chat_id": str(chat_id), "action": action}, timeout=10)

    async def send_photo(self, chat_id: int, image_base64: str, *, caption: str = "", mime: str = "image/png", parse_mode: str = "HTML") -> None:
        filename = "image.png" if mime == "image/png" else "image.jpg"
        files = {"photo": (filename, base64.b64decode(image_base64), mime)}
        formatted = markdown_to_telegram_html(caption) if parse_mode == "HTML" else caption
        data = {"chat_id": str(chat_id), "caption": formatted}
        if parse_mode:
            data["parse_mode"] = parse_mode
        await self.call("sendPhoto", data=data, files=files, timeout=30)

    async def send_document(
        self, chat_id: int, file_bytes: bytes, filename: str = "file", *, caption: str = "", parse_mode: str = "HTML"
    ) -> None:
        """Send an arbitrary document/file to a chat via sendDocument."""
        safe_name = (str(filename or "file").replace("\r", " ").replace("\n", " ").strip() or "file")
        files = {"document": (safe_name, file_bytes, "application/octet-stream")}
        formatted = markdown_to_telegram_html(caption) if (caption and parse_mode == "HTML") else caption
        data = {"chat_id": str(chat_id)}
        if formatted:
            data["caption"] = formatted
            if parse_mode:
                data["parse_mode"] = parse_mode
        await self.call("sendDocument", data=data, files=files, timeout=60)

    async def send_audio(
        self,
        chat_id: int,
        file_bytes: bytes,
        filename: str,
        *,
        caption: str = "",
        mime: str = "audio/mpeg",
        parse_mode: str = "HTML",
    ) -> None:
        """Send MP3/M4A bytes through Telegram's native audio player."""
        safe_name = (str(filename or "audio").replace("\r", " ").replace("\n", " ").strip() or "audio")
        files = {"audio": (safe_name, file_bytes, str(mime or "audio/mpeg"))}
        formatted = markdown_to_telegram_html(caption) if (caption and parse_mode == "HTML") else caption
        data = {"chat_id": str(chat_id)}
        if formatted:
            data["caption"] = formatted
            if parse_mode:
                data["parse_mode"] = parse_mode
        await self.call("sendAudio", data=data, files=files, timeout=60)

    async def send_message_with_inline_keyboard(
        self, chat_id: int, text: str, keyboard: list[list[dict]], parse_mode: str = "HTML"
    ) -> None:
        """Send a message with an inline keyboard (list of button rows)."""
        import json as _json
        reply_markup = _json.dumps({"inline_keyboard": keyboard})
        formatted = markdown_to_telegram_html(text) if parse_mode == "HTML" else text
        data = {"chat_id": str(chat_id), "text": formatted, "reply_markup": reply_markup}
        if parse_mode:
            data["parse_mode"] = parse_mode
        await self.call(
            "sendMessage",
            data=data,
            timeout=20,
        )

    async def answer_callback_query(self, callback_query_id: str, *, text: str = "") -> None:
        """Acknowledge a callback query from an inline button press."""
        data: dict = {"callback_query_id": callback_query_id}
        if text:
            data["text"] = text
        await self.call("answerCallbackQuery", data=data, timeout=10)

    async def download_photo(self, file_id: str) -> tuple[str, str]:
        payload = await self.call("getFile", data={"file_id": file_id}, timeout=20)
        file_path = str((payload.get("result") or {}).get("file_path") or "").strip()
        if not file_path:
            raise RuntimeError("Telegram file path is missing")
        content = await self._download_bytes(file_path)
        mime = mimetypes.guess_type(file_path)[0] or "image/jpeg"
        return base64.b64encode(content).decode("ascii"), mime

    async def download_file(self, file_id: str) -> bytes:
        """Download an arbitrary file from Telegram servers and return its raw bytes."""
        payload = await self.call("getFile", data={"file_id": file_id}, timeout=20)
        file_path = str((payload.get("result") or {}).get("file_path") or "").strip()
        if not file_path:
            raise RuntimeError("Telegram file path is missing")
        return await self._download_bytes(file_path)

    async def edit_message_text_with_inline_keyboard(
        self, chat_id: int, message_id: int, text: str, keyboard: list[list[dict]], parse_mode: str = "HTML"
    ) -> bool:
        """Edit a message and keyboard in-place; return whether it is current."""
        import json as _json
        reply_markup = _json.dumps({"inline_keyboard": keyboard})
        formatted = markdown_to_telegram_html(text) if parse_mode == "HTML" else text
        data = {
            "chat_id": str(chat_id),
            "message_id": str(message_id),
            "text": formatted,
            "reply_markup": reply_markup,
        }
        if parse_mode:
            data["parse_mode"] = parse_mode
        try:
            await self.call("editMessageText", data=data, timeout=20)
            return True
        except Exception as exc:
            return str(exc) == "Telegram API editMessageText: message is not modified."


_LOCALIZED_TEXTS = {
    "en": {
        "menu_title_strict": "🤖 **Ouroboros Control Panel**\nStrict mode is active. Commands are blocked.\n\nSelect action:",
        "menu_title": "🤖 **Ouroboros Control Centre**\nCommand Mode: `{command_mode}`\nLanguage: `{lang}`\n\nExplore and monitor the core using the buttons below:",
        "btn_metrics": "📉 Status & Metrics",
        "btn_mind": "🧠 Mind & BG",
        "btn_language": "🌐 Select language",
        "btn_refresh": "🔄 Update parameter card",
        "btn_back": "⬅️ Back to main panel",
        "btn_stop_bg": "🔴 Pause background thoughts",
        "btn_start_bg": "🟢 Resume background thoughts",
        "btn_thoughts": "💭 What are you thinking about?",
        "metrics_title": "📊 **Ouroboros live metrics**\n\n{info_text}\n---",
        "mind_title": "🧠 **Background Consciousness**\n\nCurrent state: {state_str}\n\nBackground thinking processes information between your chat queries.",
        "mind_thoughts": "\n\n**Recent thoughts catalog:**\n{thoughts_text}",
        "mind_state_active": "🟢 **Thinking** (running)",
        "mind_state_sleeping": "🔴 **Sleeping** (paused)",
        "lang_title": "🌐 Select chatbot bridge interface language:\nCurrently active: **English**",
        "lang_en": "🇬🇧 English",
        "lang_ru": "🇷🇺 Русский",
        "help_text": (
            "🤖 **Ouroboros Telegram Help**\n\n"
            "Available commands:\n"
            "• `/menu` — Show interactive control panel with active tabs\n"
            "• `/language` — Change bridge interface language\n"
            "• `/status` — Request live system status (if allowed)\n"
            "• `/help` — Show this friendly usage guide\n\n"
            "Modes description (changed in Web UI → Settings → Telegram):\n"
            "• **strict** — block all command injections (only `/menu`, `/help`, `/language`)\n"
            "• **safe_commands** — allow status monitoring\n"
            "• **full_access** — allow status + background loop start/stop"
        ),
        "slash_blocked_strict": "⛔ Slash commands are not allowed in strict mode. Use `/menu` to see available options, or change mode in Settings → Telegram.",
        "slash_blocked_mode": "⛔ This command is not allowed in the current mode. Use `/menu` to see available options.",
        "not_authorized": "Not authorized",
        "updating_status": "Updating status metrics...",
        "extracting_thoughts": "Extracting thoughts...",
        "injecting_consciousness": "Injecting consciousness signal...",
        "restricted_safe": "⛔ Restricted in safe mode",
        "lang_changed": "✅ Language changed to English",
        "unknown_command": "Unknown command",
        "metrics_budget_status": "• **Budget Status:**\n  Spent: `${spent_usd:.4f}`\n  Limit: `${total_budget:.2f}`\n  Remaining: `${rem:.4f}`\n\n• **System Environment:**\n  Branch: `{branch}`\n  BG Thoughts: `{bg_status}`",
        "bg_active_label": "ACTIVE",
        "bg_sleeping_label": "SLEEPING",
        "btn_settings": "⚙️ Settings",
        "settings_title": "⚙️ **Telegram Settings**\nConfigure the Telegram control panel and message display:",
        "btn_silent_on": "🔕 Silent Mode: ON",
        "btn_silent_off": "🔔 Silent Mode: OFF",
        "silent_toggled_on": "🔕 Silent mode enabled — new thoughts will replace the last message",
        "silent_toggled_off": "🔔 Silent mode disabled — each thought becomes a new message",
    },
    "ru": {
        "menu_title_strict": "🤖 **Панель управления Ouroboros**\nРежим Strict активен. Команды заблокированы.\n\nВыберите действие:",
        "menu_title": "🤖 **Центр управления Ouroboros**\nРежим команд: `{command_mode}`\nЯзык: `{lang}`\n\nУправляйте и следите за ядром с помощью кнопок:",
        "btn_metrics": "📉 Статус и метрики",
        "btn_mind": "🧠 Фоновое сознание",
        "btn_language": "🌐 Выбор языка",
        "btn_refresh": "🔄 Обновить показатели",
        "btn_back": "⬅️ Назад в меню",
        "btn_stop_bg": "🔴 Приостановить размышления",
        "btn_start_bg": "🟢 Продолжить размышления",
        "btn_thoughts": "💭 О чём ты думаешь сейчас?",
        "metrics_title": "📊 **Живые показатели Ouroboros**\n\n{info_text}\n---",
        "mind_title": "🧠 **Фоновое Сознание**\n\nТекущее состояние: {state_str}\n\nФоновое мышление анализирует информацию между вашими запросами.",
        "mind_thoughts": "\n\n**Последние мысли из лога:**\n{thoughts_text}",
        "mind_state_active": "🟢 **Думает** (активно)",
        "mind_state_sleeping": "🔴 **Спит** (на паузе)",
        "lang_title": "🌐 Выберите язык интерфейса бота-моста:\nАктивный язык: **Русский**",
        "lang_en": "🇬🇧 English",
        "lang_ru": "🇷🇺 Русский",
        "help_text": (
            "🤖 **Справка по Telegram в Ouroboros**\n\n"
            "Доступные команды:\n"
            "• `/menu` — Открыть интерактивную панель управления\n"
            "• `/language` — Изменить и настроить язык интерфейса\n"
            "• `/status` — Запросить текущий статус системы (если разрешено)\n"
            "• `/help` — Показать это руководство\n\n"
            "Описание режимов (меняется в Web UI → Settings → Telegram):\n"
            "• **strict** — блокировать ввод команд (доступны только `/menu`, `/help`, `/language`)\n"
            "• **safe_commands** — разрешить просмотр метрик и статуса\n"
            "• **full_access** — доступ ко всем кнопкам, включая запуск/паузу фонового сознания"
        ),
        "slash_blocked_strict": "⛔ Слэш-команды запрещены в режиме strict. Используйте `/menu` или измените режим в Settings → Telegram.",
        "slash_blocked_mode": "⛔ Эта команда не разрешена в текущем режиме. Используйте `/menu` для вызова управления.",
        "not_authorized": "Доступ ограничен",
        "updating_status": "Обновление метрик...",
        "extracting_thoughts": "Извлечение мыслей...",
        "injecting_consciousness": "Отправка сигнала сознания...",
        "restricted_safe": "⛔ Ограничено в режиме Safe",
        "lang_changed": "✅ Язык интерфейса изменен на Русский",
        "unknown_command": "Неизвестная команда",
        "metrics_budget_status": "• **Бюджетный статус:**\n  Потрачено: `${spent_usd:.4f}`\n  Лимит: `${total_budget:.2f}`\n  Осталось: `${rem:.4f}`\n\n• **Окружение системы:**\n  Ветка Git: `{branch}`\n  Фоновые мысли: `{bg_status}`",
        "bg_active_label": "АКТИВНЫ",
        "bg_sleeping_label": "СПЯТ",
        "btn_settings": "⚙️ Настройки",
        "settings_title": "⚙️ **Настройки Telegram**\nНастройте панель Telegram и отображение сообщений:",
        "btn_silent_on": "🔕 Тихий режим: ВКЛ",
        "btn_silent_off": "🔔 Тихий режим: ВЫКЛ",
        "silent_toggled_on": "🔕 Тихий режим включён — новые мысли будут заменять предыдущее сообщение",
        "silent_toggled_off": "🔔 Тихий режим выключен — каждая мысль становится отдельным сообщением",
    }
}
