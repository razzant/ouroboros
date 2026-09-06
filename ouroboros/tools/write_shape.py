"""Write-shape classification SSOT for shell commands (extracted from
shell_guards.py at the module-size gate; shell_guards re-exports every name).

ONE seam decides whether a command line is WRITE-SHAPED before any deterministic
write/owner-control guard acts on it: interpreter argv takes the mode-aware
``interpreter_write_shape``, everything else takes ``non_interpreter_write_shape``
(membership floor for unconditional writers, real-channel evidence for the
pure-filter utilities, prose words yield to the caller's read-carve). No guard
may consume a coarser write fact than this composition's — that is the
family-wide application of the v6.80.0 scope-floor read-carve contract.
"""

from __future__ import annotations

import ast
import io
import pathlib
import re
import tokenize
from typing import Any, Callable, List, Optional

from ouroboros.shell_parse import shell_argv, shell_argv_with_inline, shell_argv_with_path_tokens

SHELL_WRITE_INDICATORS = (
    "rm ", "rm\t", ">", "sed -i", "tee ", "truncate",
    "mv ", "cp ", "chmod ", "chown ", "unlink ", "delete", "trash",
    "rsync ", "write_text", ".write(", ".writelines(",
    "os.remove(", "os.unlink(", "os.mkdir(", "os.makedirs(", "sort -o",
    "writefilesync", "appendfilesync", "createwritestream",
)
# Preserve the longstanding coarse ``open(`` signal without matching it as the
# suffix of another callable such as ``urlopen(``.
_OPEN_CALL_WRITE_INDICATOR_RE = re.compile(r"(?<![A-Za-z0-9_])open\(")
_SAFE_STDIO_REDIRECT_TOKENS = frozenset({
    ">/dev/null",
    "1>/dev/null",
    "2>/dev/null",
    "2>&1",
    "1>&2",
    "2>&-",
})

INTERPRETER_WRITE_RE = re.compile(
    # The open() clause anchors on the MODE argument (python short quoted [a-z+]
    # write flag, or perl '>' / '>>' / '+>' / '+<'), not any quoted arg containing
    # w/a/x/+ — a read `open(my $fh, '<', '/tmp/f.txt')` must not match on the 'x'
    # in the filename.
    # `remove(`/`rename(` are anchored to the os/fs namespaces so the two write
    # vocabularies AGREE with the AST walker on the same line (#447 A4:
    # `xs.remove(1)` is list.remove — AST False, unanchored regex True; pandas
    # `df.rename(` writes nothing). Residual: bare `Path(p).rename(q)` no longer
    # matches here — the AST lane still models it for the light/runtime_data gates.
    r"""(?is)(?:\.write\(|write_text\(|write_bytes\(|fs\.write|fs\.append|"""
    r"""createwritestream|unlink\(|(?:os|fs)\s*\.\s*rename\(|mkdir\(|rmtree\(|(?:os|fs)\s*\.\s*remove\(|"""
    r"""open\s*\([^)]*,\s*(?:mode\s*=\s*)?['"](?=[a-z+]{1,3}['"])[a-z+]*[wax+][a-z+]*['"]|"""
    r"""open\s*\([^)]*,\s*(?:mode\s*=\s*)?['"]\s*(?:\+\s*[<>]|>{1,2}))"""
)
# Wider write-indicator net for the read-vs-write runtime_data scan (v6.54.3):
# includes filesystem-mutating calls the base write regex misses (shutil.copy*/move,
# touch, symlink/link, chmod/chown, makedirs/removedirs, truncate) — a hit here
# without AST-resolved targets stays on the conservative full mention scan instead
# of being treated as a pure read. This list and the AST walker are NOT one
# vocabulary and no invariant ties them: `<mod>.open(p,"w")` (io/codecs/gzip/bz2/lzma)
# matches here, while `_python_path_open_target` reads arg 0 as the MODE — right for
# `Path(p).open("w")`, wrong here — so the walker answers "no targets" and the fence
# reads that as a proven read. Measured, not hypothetical: it truncates a repo source
# file. DISCLOSED, not detected (owner direction: weaken, never strengthen; a false
# invariant is removed rather than made true). NB: the leading
# (?is) of INTERPRETER_WRITE_RE.pattern already applies globally to the whole
# concatenated expression — a second mid-pattern global flag is a hard
# re.error on Python 3.11+ (review round 2).
_INTERPRETER_ANY_WRITE_RE = re.compile(
    INTERPRETER_WRITE_RE.pattern
    + r"""|(?:makedirs\(|removedirs\(|rmdir\(|copyfile\(|copy2\(|copytree\(|os\.replace\(|"""
    + r"""shutil\.(?:copy|move)\(|\.touch\(|symlink\(|os\.link\(|\.link_to\(|hardlink_to\(|"""
    + r"""chmod\(|chown\(|truncate\(|"""
    # OPAQUE / unmodeled write-capable calls (adversarial review r2 #1): an
    # external process (subprocess/os.system/popen) can `rm`/`mv`/`dd` anything,
    # and archive-extract / db-open write to a directory the AST never resolves.
    # A hit here has no AST-resolvable target, so it falls through to the
    # conservative full mention scan (blocks drive paths OUTSIDE the task roots)
    # instead of being mis-classified as a pure read. Pure reads (open()/read_text
    # with no write token) still match nothing and stay allowed.
    + r"""subprocess\.|os\.system\(|os\.popen\(|Popen\(|check_call\(|check_output\(|"""
    # sqlite3.connect: `:memory:` and an explicit `file:...?mode=ro` URI open
    # nothing writable — the AST lane carves both, so the regex must agree
    # (#447 A4: a read-only DB open is not a write shape).
    + r"""\.extractall\(|unpack_archive\(|make_archive\(|"""
    + r"""sqlite3\.connect\((?!\s*['"](?:file:)?:memory:|\s*['"]file:[^'"]*?[?&]mode=ro\b)|"""
    # LIBRARY save-APIs (fable-5 cumulative review F1): to_csv/savefig/.save &co
    # write files while carrying no base write-token, so an interpreter command
    # using them was classified as a PURE READ and skipped the runtime_data
    # mention scan entirely. A false positive here only re-applies the
    # conservative pre-v6.54.3 always-scan behavior (fail-closed direction).
    # The mode-shaped single-arg .open("w"/"ab"/"x+") is the pathlib positional
    # form the comma-anchored open() token above cannot see; the tight 1-3 char
    # mode lookahead keeps .open("<path>") reads out.
    + r"""\.save\(|\.to_csv\(|\.to_excel\(|\.to_parquet\(|\.to_json\(|\.to_pickle\(|"""
    + r"""savefig\(|np\.save|imwrite\(|pickle\.dump\(|json\.dump\(|"""
    # RUBY native write idioms (fable-5 review): with the membership floor gone the
    # vocabulary must see ruby's spellings — File.delete/Dir.delete, IO.binwrite/
    # syswrite, FileUtils.* (variable args the literal-target regex cannot see; rare
    # read helpers like compare_file just fall back to the conservative scan).
    + r"""file\.delete\(|dir\.delete\(|binwrite\(|syswrite\(|fileutils\.[a-z_]+|"""
    + r"""file\.new\s*\([^)]*,\s*['"][^'"]*[wax+]|"""
    + r"""\.open\(\s*(?:mode\s*=\s*)?['"](?=[a-z+]{1,3}['"])[a-z+]*[wax+][a-z+]*['"])"""
)

# Interpreter-lane / prose refinements (see interpreter_write_shape): bare
# ENGLISH WORDS excluded so "count deleted rows" is not a write (structural
# deletion — os.remove/unlink/rmtree/File.delete/fileutils/boundary-worded rm —
# stays covered); command words take a left word boundary so 'cp ' misses 'scp ';
# 'truncate' takes a right boundary ("results truncated" reads); '>' counts only as
# a real redirect (token-initial or glued into an operand), never inside a located
# code body (`<$fh>`, `a > b`, '=>', '>='). Parenless perl builtins (`rename $a, $b`)
# are a disclosed residual.
_INTERPRETER_LANE_EXCLUDED_INDICATORS = frozenset({"delete", "trash"})
# For NON-interpreter argv the same three bare words are PROSE, not channels: the
# real channels are head membership (`truncate -s 0 f` blocks on its head), the
# per-segment writer targets, and the option indicators ('sed -i', 'sort -o').
_PROSE_WORD_INDICATORS = frozenset({"delete", "trash", "truncate"})
_COMMAND_WORD_INDICATORS = frozenset({
    "rm ", "rm\t", "tee ", "mv ", "cp ", "chmod ", "chown ", "unlink ", "rsync ",
})
_COMMAND_WORD_BOUNDARY_RES = {
    indicator: re.compile(r"(?<![A-Za-z0-9_.-])" + re.escape(indicator))
    for indicator in _COMMAND_WORD_INDICATORS
}
_TRUNCATE_BOUNDARY_RE = re.compile(r"truncate(?![a-z])")
_REDIRECT_SHAPE_TOKEN_RE = re.compile(r"^(?:(?:&|\d?)>>?(?=$|[^&|-])|>&.)")
_MIDTOKEN_REDIRECT_RE = re.compile(r"(?<![<>=&|'\"-])>{1,2}(?![>=&])")

# LIGHT_SHELL_WRITER_COMMANDS members that are PURE FILTERS in their default
# invocation: they write only through an explicit channel (sed -i, sort -o, a
# second uniq operand, tar create/extract, gzip without -l/-t/-c, a redirect) —
# every one of which the target parser or an option indicator reports on its own.
# Bare membership made `sort /etc/hosts` "write-like" (EXT-3). Unconditional
# writers (cp/mv/rm/mkdir/touch/chmod/...) keep the membership floor, and the
# interpreter members (ruby/perl) take interpreter_write_shape instead.
PURE_FILTER_WRITER_COMMANDS = frozenset({"gunzip", "gzip", "sed", "sort", "tar", "uniq"})

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


def segment_write_shape(argv: List[str]) -> bool:
    """Write shape for one already-tokenized command row."""
    from ouroboros.tools.registry_guard_process import _is_pure_read_inspection
    from ouroboros.tools.shell_guards import interpreter_family

    if not argv:
        return False
    executable = str(argv[0]).replace("\\", "/").rsplit("/", 1)[-1].lower().removesuffix(".exe")
    if interpreter_family(executable):
        return bool(interpreter_write_shape(argv))
    return bool(non_interpreter_write_shape(
        argv, argv, executable, is_pure_read=_is_pure_read_inspection,
    ))


def _shell_write_indicator_scan(
    raw_cmd: Any,
    *,
    include_bare_open: bool,
    interpreter_lane: bool = False,
    exclude_prose_words: bool = False,
) -> bool:
    if isinstance(raw_cmd, list):
        text = " ".join(str(x) for x in raw_cmd).lower()
    else:
        text = str(raw_cmd).lower()
    tokens = [str(token).lower() for token in shell_argv_with_inline(raw_cmd)]
    filtered_tokens: List[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in _SAFE_STDIO_REDIRECT_TOKENS:
            i += 1
            continue
        if token in {">", "1>", "2>"} and i + 1 < len(tokens) and tokens[i + 1] == "/dev/null":
            i += 2
            continue
        filtered_tokens.append(token)
        i += 1
    filtered_text = " ".join(filtered_tokens)
    for token in _SAFE_STDIO_REDIRECT_TOKENS:
        text = text.replace(token, " ")

    if interpreter_lane:
        from ouroboros.shell_parse import shell_command_string
        from ouroboros.tools.shell_guards import interpreter_inline_code

        argv = shell_argv(raw_cmd)
        bodies = list(interpreter_inline_code(argv))
        # An sh -c wrap hides the interpreter one level down; locate the inner
        # bodies too so a '>' comparison inside them is not read as a redirect.
        if argv and str(argv[0]).rsplit("/", 1)[-1].lower() in {"sh", "bash", "zsh"}:
            inner = shell_command_string(argv)
            if inner:
                bodies.extend(interpreter_inline_code(shell_argv(inner)))
        inline_bodies = frozenset(body.lower() for body in bodies)
    else:
        inline_bodies = frozenset()

    def _in_located_body(tok: str) -> bool:
        # Joined flags carry the body INSIDE the token (`-cBODY`, `--eval=BODY`).
        return any(body and body in tok for body in inline_bodies)

    def _indicator_hits(scan_text: str, *, allow_bare_redirect: bool) -> bool:
        for indicator in SHELL_WRITE_INDICATORS:
            if indicator == ">":
                if not allow_bare_redirect:
                    continue
                if interpreter_lane:
                    # Token-level only: a real redirect is its own shell token or
                    # glued into an operand; a '>' inside a located inline-code
                    # body is not a write channel.
                    if any(
                        _REDIRECT_SHAPE_TOKEN_RE.match(tok)
                        or (not _in_located_body(tok) and _MIDTOKEN_REDIRECT_RE.search(tok))
                        for tok in filtered_tokens
                    ):
                        return True
                    continue
            if exclude_prose_words and indicator in _PROSE_WORD_INDICATORS:
                continue
            if interpreter_lane:
                if indicator in _INTERPRETER_LANE_EXCLUDED_INDICATORS:
                    continue
                if indicator == "truncate":
                    if _TRUNCATE_BOUNDARY_RE.search(scan_text):
                        return True
                    continue
                boundary = _COMMAND_WORD_BOUNDARY_RES.get(indicator)
                if boundary is not None:
                    if boundary.search(scan_text):
                        return True
                    continue
            if indicator in scan_text:
                return True
        return False

    if _indicator_hits(filtered_text, allow_bare_redirect=True) or _indicator_hits(
        text, allow_bare_redirect=False
    ):
        return True
    if include_bare_open and (
        _OPEN_CALL_WRITE_INDICATOR_RE.search(filtered_text) or _OPEN_CALL_WRITE_INDICATOR_RE.search(text)
    ):
        return True
    return False


def shell_has_write_indicator(raw_cmd: Any) -> bool:
    return _shell_write_indicator_scan(raw_cmd, include_bare_open=True)


def interpreter_write_shape(raw_cmd: Any) -> bool:
    """Mode-aware write-shape classification for an INTERPRETER command line.

    The coarse ``open(`` token marks a read-only ``open(p, 'rb')`` as writeish —
    the class the light-mode runtime_data lane already re-judges ("the original
    GAIA class", review round 8). For interpreter argv the bare token is replaced
    by ``_INTERPRETER_ANY_WRITE_RE`` (write-mode opens incl. perl '>' spellings,
    pathlib ``.open('w')``, save-APIs, ruby File.delete/FileUtils, opaque
    subprocess/exec escapes); every shell-level indicator still counts. Disclosed
    residual: ``open(p, m)`` with the mode in a variable is not write-shaped — the
    external-workspace runtime/secret read guard and the LLM safety supervisor
    stay the covering controls.
    """
    if _shell_write_indicator_scan(raw_cmd, include_bare_open=False, interpreter_lane=True):
        return True
    if isinstance(raw_cmd, list):
        text = " ".join(str(x) for x in raw_cmd)
    else:
        text = str(raw_cmd)
    return bool(_INTERPRETER_ANY_WRITE_RE.search(text))


def non_interpreter_write_shape(
    raw_cmd: Any,
    argv: List[str],
    executable: str,
    *,
    is_pure_read: Optional[Callable[[str], bool]] = None,
) -> bool:
    """Mode-aware write shape for NON-interpreter argv (the composition's other half).

    Unconditional writers keep the membership floor. Pure-filter utilities
    (``PURE_FILTER_WRITER_COMMANDS``) are write-shaped only through a real channel
    — an option indicator, a redirect, or a writer target the parser reports (the
    caller ORs ``explicit_write_targets`` in). The bare prose words yield to the
    caller's read-carve: on a provably read-only inspection line (`grep -n delete
    ouroboros/safety.py`, ``rg truncate …``) a word is not a write channel; any
    head the carve cannot prove stays fail-closed on the full legacy scan.
    """
    from ouroboros.tools.shell_guards import LIGHT_SHELL_WRITER_COMMANDS, interpreter_family

    member = bool(argv) and (
        (interpreter_family(executable) or executable) in LIGHT_SHELL_WRITER_COMMANDS
    )
    if member and executable not in PURE_FILTER_WRITER_COMMANDS:
        return True
    if _shell_write_indicator_scan(raw_cmd, include_bare_open=True, exclude_prose_words=True):
        return True
    if _shell_write_indicator_scan(raw_cmd, include_bare_open=True):
        # Only the prose words fired. A SINGLE-segment pure-filter head's real
        # channels are all target/option evidence already judged above, so prose
        # inside its script/pattern argument (`sed -n '/delete/p' f`) is never a
        # channel — but the head speaks only for ITS OWN segment: a compound line
        # (`sort f && find … -delete`) goes to the carve, which judges every
        # segment and fails closed (sol-max round-2). Otherwise a provably
        # read-only line reads; anything unproven keeps the legacy fail-closed
        # classification.
        compound = any(
            str(t) in ("&&", "||", ";", "|", "&") for t in shell_argv(raw_cmd)
        )
        if executable in PURE_FILTER_WRITER_COMMANDS and not compound:
            return False
        if is_pure_read is None:
            return True
        if isinstance(raw_cmd, list):
            text_lower = " ".join(str(x) for x in raw_cmd).lower()
        else:
            text_lower = str(raw_cmd).lower()
        return not is_pure_read(text_lower)
    return False


# --- Workspace write candidates: the per-segment write/mention walk over writer-target rows ---

def _no_deliverables_decision(_path: Any) -> None:
    """Deliverables policy is a TARGET policy: a mention takes no decision."""
    return None


def _directory_change_argv(argv: list) -> bool:
    return bool(argv) and pathlib.PurePath(
        str(argv[0])
    ).name.lower().removesuffix(".exe") in {"cd", "pushd"}


def _workspace_write_candidates(
    target_rows: list, explicit_write_targets: list[str], raw_cmd: Any,
) -> list[tuple[str, bool, int]]:
    """Write/mention candidates for the workspace write guard, per segment.

    A segment's parsed TARGETS are write candidates; every other token it carries
    stays a MENTION-only candidate. Protected-runtime-root refusals keep running
    for every candidate, so a path a writer merely READS (`cp ../data/settings.json
    ./x`) still refuses, while the Deliverables decision and the outside-root
    refusal apply to real write targets only.
    """
    candidates: list[tuple[str, bool, int]] = []
    index_by_token: dict[tuple[str, int], int] = {}

    def _add(token: Any, is_write: bool, row_index: int) -> None:
        token_text = str(token)
        if not token_text.strip():
            return
        position = index_by_token.get((token_text, row_index))
        if position is not None:
            if is_write and not candidates[position][1]:
                candidates[position] = (token_text, True, row_index)
            return
        index_by_token[(token_text, row_index)] = len(candidates)
        candidates.append((token_text, is_write, row_index))

    for row_index, (segment_argv, targets, inline_code, unprovable) in enumerate(target_rows):
        if _directory_change_argv(segment_argv):
            # A directory change is not itself a write. Its operand becomes a
            # write candidate only when a later segment has a parsed or
            # fail-closed write channel, because that later relative write is
            # evaluated from the changed directory.
            later_write = any(
                later_unprovable
                or (later_targets and not _directory_change_argv(later_argv))
                for later_argv, later_targets, _later_inline, later_unprovable
                in target_rows[row_index + 1:]
            )
            targets = targets if later_write else []
        # Inline-code targets are already extracted paths; an argv-shaped
        # segment's targets still need the embedded-path pass (sed's in-script
        # `w FILE` hides the path inside the script operand).
        if targets and not inline_code:
            write_tokens = [str(token) for token in shell_argv_with_path_tokens(list(targets))]
        else:
            write_tokens = [str(token) for token in targets]
        if unprovable:
            # Uncertainty widens only this row and its attached program bodies.
            write_tokens.extend(str(token) for token in segment_argv[1:])
            write_tokens.extend(
                str(token) for token in shell_argv_with_path_tokens(list(segment_argv[1:]))
            )
            for body in inline_code:
                write_tokens.extend(
                    str(token) for token in shell_argv_with_path_tokens(str(body))
                )
        write_set = set(write_tokens)
        for token in segment_argv:
            _add(token, str(token) in write_set, row_index)
        for token in write_tokens:
            _add(token, True, row_index)
    associated_writes = {text for text, is_write, _row in candidates if is_write}
    for token in explicit_write_targets:
        if str(token) not in associated_writes:
            _add(token, True, -1)
    # The MENTION lane keeps the full harvest of the raw command text: an embedded
    # Windows drive/UNC spelling does not survive POSIX tokenization, so the
    # per-segment argv alone would stop the protected-root and outside-root scans
    # from ever seeing it. Such a harvested token is the SAME target in its
    # unmangled spelling when removing the separators the tokenizer swallowed
    # makes the two texts identical, so it keeps the write policy.
    collapsed_writes = {
        text.replace("\\", "")
        for text, is_write, _row_index in candidates
        if is_write and text.replace("\\", "")
    }
    associated_tokens = {text for text, _is_write, row in candidates if row >= 0}
    for token in shell_argv_with_path_tokens(raw_cmd):
        if str(token) in associated_tokens:
            continue
        _add(token, str(token).replace("\\", "") in collapsed_writes, -1)
    return candidates
