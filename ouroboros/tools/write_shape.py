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

import re
from typing import Any, Callable, List, Optional

from ouroboros.shell_parse import shell_argv, shell_argv_with_inline

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
