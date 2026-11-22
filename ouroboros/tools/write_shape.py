"""Literal write targets inside interpreter bodies (a leaf of shell_guards.py).

``python_body_ast`` parses an inline Python body (re-reading Windows-path
literals verbatim when the plain parse fails); ``script_literal_write_targets_and_unknown``
reads the literal write targets of a Node or Ruby body and reports when the
body writes through something it cannot resolve. ``shell_guards`` consumes both
when it lists the write targets of a command for the after-command
undeclared-output note. Neither answer blocks a command.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize


_NODE_FS_PREFIX = r'''(?:fs(?:\.promises)?\.|require\(['"]fs['"]\)(?:\.promises)?\.)'''
_NODE_LITERAL_WRITE_RE = re.compile(
    rf'''(?is){_NODE_FS_PREFIX}'''
    r'''(?:writeFile(?:Sync)?|appendFile(?:Sync)?|createWriteStream|mkdir(?:Sync)?|rm(?:Sync)?|rmdir(?:Sync)?|unlink(?:Sync)?)\s*\(\s*(['"])(.*?)\1'''
)
_NODE_LITERAL_DESTINATION_RE = re.compile(
    rf'''(?is){_NODE_FS_PREFIX}(?:rename(?:Sync)?|copyFile(?:Sync)?)'''
    r'''\s*\(\s*[^,()]*,\s*(['"])(.*?)\1'''
)
_NODE_WRITE_CALL_RE = re.compile(
    rf'''(?is){_NODE_FS_PREFIX}(?:writeFile(?:Sync)?|appendFile(?:Sync)?|createWriteStream|'''
    r'''mkdir(?:Sync)?|rm(?:Sync)?|rmdir(?:Sync)?|unlink(?:Sync)?|rename(?:Sync)?|copyFile(?:Sync)?)\s*\('''
)
_NODE_OPAQUE_EXEC_RE = re.compile(
    r'''(?is)(?:child_process\.|require\(['"]child_process['"]\)\.)'''
    r'''(?:exec|spawn|execSync|spawnSync)\s*\('''
)
_RUBY_LITERAL_WRITE_RE = re.compile(
    r'''(?is)(?:File\.(?:write|delete)|FileUtils\.(?:touch|mkdir|mkdir_p|makedirs|'''
    r'''rm|rm_r|rm_rf|remove|remove_dir|rmdir|remove_entry|remove_entry_secure)|'''
    r'''File\.(?:open|new)(?=\s*\([^)]*,\s*['"][^'"]*[wax+])'''
    r''')\s*\(\s*(['"])(.*?)\1'''
)
_RUBY_FILEUTILS_COPY_RE = re.compile(
    r'''(?is)(?:File\.rename|FileUtils\.(?:copy|cp|cp_r|mv|move|install))'''
    r'''\s*\(\s*[^,()]*,\s*(['"])(.*?)\1'''
)
_RUBY_WRITE_CALL_RE = re.compile(
    r'''(?is)(?:File\.(?:write|delete|rename)|FileUtils\.[A-Za-z_]+)\s*\('''
)
_RUBY_MULTI_TARGET_TAIL_RE = re.compile(
    r'''(?is)(?:File\.delete|FileUtils\.(?:touch|mkdir|mkdir_p|makedirs|rm|rm_r|rm_rf|'''
    r'''remove|remove_dir|rmdir|remove_entry|remove_entry_secure))\s*\(\s*(['"])(.*?)\1\s*,'''
)


def script_literal_write_targets_and_unknown(family: str, body: str) -> tuple[list[str], bool]:
    """Literal non-Python script targets plus execution/argument uncertainty."""
    if family == "node":
        targets = [match.group(2) for match in _NODE_LITERAL_WRITE_RE.finditer(body)]
        destinations = list(_NODE_LITERAL_DESTINATION_RE.finditer(body))
        targets.extend(match.group(2) for match in destinations)
        resolved = len(_NODE_LITERAL_WRITE_RE.findall(body)) + len(destinations)
        unknown = bool(_NODE_OPAQUE_EXEC_RE.search(body)) or len(_NODE_WRITE_CALL_RE.findall(body)) != resolved
        return list(dict.fromkeys(targets)), unknown
    if family == "ruby":
        targets = [match.group(2) for match in _RUBY_LITERAL_WRITE_RE.finditer(body)]
        copies = list(_RUBY_FILEUTILS_COPY_RE.finditer(body))
        targets.extend(match.group(2) for match in copies)
        resolved = len(_RUBY_LITERAL_WRITE_RE.findall(body)) + len(copies)
        ambiguous = (
            len(_RUBY_WRITE_CALL_RE.findall(body)) != resolved
            or bool(_RUBY_MULTI_TARGET_TAIL_RE.search(body))
        )
        return list(dict.fromkeys(targets)), ambiguous
    return [], False


_STRING_TOKEN_PREFIX_RE = re.compile(r"^[A-Za-z]*")


def _verbatim_string_source(code: str) -> str | None:
    """``code`` with every non-raw string literal re-spelled so that its VALUE is
    the source text between its quotes: a backslash that escapes neither a quote
    nor another backslash is doubled. ``None`` when the source does not tokenize.
    f-strings tokenize as their own token kind on 3.12+ and are left alone there.
    """
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except (tokenize.TokenError, SyntaxError):
        return None
    offsets = [0]
    for line in io.StringIO(code).readlines():
        offsets.append(offsets[-1] + len(line))
    out: list[str] = []
    cursor = 0
    for token in tokens:
        if token.type != tokenize.STRING:
            continue
        text = token.string
        prefix = _STRING_TOKEN_PREFIX_RE.match(text).group(0)
        if "r" in prefix.lower():
            continue
        quote = text[len(prefix)]
        width = 3 if text[len(prefix):len(prefix) + 3] == quote * 3 else 1
        inner = text[len(prefix) + width:len(text) - width]
        respelled: list[str] = []
        index = 0
        while index < len(inner):
            char = inner[index]
            if char == "\\" and index + 1 < len(inner) and inner[index + 1] in ("\\", quote):
                respelled.append(inner[index:index + 2])
                index += 2
                continue
            respelled.append("\\\\" if char == "\\" else char)
            index += 1
        start = offsets[token.start[0] - 1] + token.start[1]
        end = offsets[token.end[0] - 1] + token.end[1]
        out.append(code[cursor:start])
        out.append(prefix + quote * width + "".join(respelled) + quote * width)
        cursor = end
    out.append(code[cursor:])
    return "".join(out)


def python_body_ast(code: str) -> ast.AST | None:
    """AST of an inline Python body, or ``None`` when it cannot be parsed.

    A body that fails to parse is re-read with its string literals VERBATIM
    (``_verbatim_string_source``): a Windows path typed into a plain literal
    (``open("C:\\Users\\x")``) is not a valid Python string — ``\\U`` opens a
    unicode escape — and reading it as UNKNOWN turned every pure read naming a
    Windows path into a fail-closed outside-root WRITE on the windows-latest
    serial pass (the first Windows execution of that suite). The retry runs only
    after the normal parse failed and changes nothing but the literals' values,
    which become the source characters the model typed — the spelling the guards
    compare. A body unparseable for any other reason (foreign syntax, a broken
    f-string on 3.12+) stays ``None`` and keeps the fail-closed UNKNOWN verdict.
    """
    try:
        return ast.parse(code)
    except Exception:
        pass
    source = _verbatim_string_source(code)
    if source is None:
        return None
    try:
        return ast.parse(source)
    except Exception:
        return None
