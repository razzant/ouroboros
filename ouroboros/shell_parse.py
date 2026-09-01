"""Small shell argv parsing helpers shared by tool guardrails."""

from __future__ import annotations

import ast
import json
import pathlib
import re
import shlex
from typing import Any, List


EMBEDDED_ABSOLUTE_PATH_RE = re.compile(r"(?<![A-Za-z0-9_.\-])/[^\s'\"\\),;\]]+")
_HTML_CLOSING_TAG_PATH_RE = re.compile(r"/[A-Za-z][A-Za-z0-9:-]*>")
# A path absolute in WINDOWS grammar: a drive letter plus a separator, or a UNC
# share. The UNC alternative REQUIRES both segments (`\\server\share`) because a
# bare `\\`-plus-text is not a path at all — it is the commonest escape idiom in
# every other language (`s.replace(/\\/g,'/')`, `"a\\b"`, `'\\b'`), and matching it
# made the light-mode inline fence refuse ordinary `node -e` payloads that named no
# repo path (v6.89.x). A share segment cannot be spelled by accident.
# A BACKTICK is deliberately NOT a delimiter, in either grammar. Treating it as one
# (added and then removed 2026-08-05) closed the template-literal exact-root hole
# (``rmSync(String.raw`<root>`)`` harvests root+backtick = a sibling, so deleting the
# exact root is ALLOWED — only the root itself; anything under it still resolves
# inside) but also newly refused writes to real sibling paths with a literal backtick
# in the name, which the base allowed. Owner policy: protection may be weakened, never
# strengthened, and a review-found adjacent hole is disclosed, not fenced. The hole is
# pinned as disclosed in test_interpreter_family_write_fence.py.
EMBEDDED_WINDOWS_ABSOLUTE_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])(?:"
    r"[A-Za-z]:[\\/][^\s'\"),;\]]+"
    r"|\\\\[^\s'\"),;\]\\/]+[\\/][^\s'\"),;\]]+"
    r")"
)
_SHELLS = {"sh", "bash", "zsh"}


def recover_stringified_argv(text: Any) -> List[str] | None:
    """Recover a stringified argv list — ``'["go","test"]'`` (JSON) or ``"['go','test']"``
    (Python literal) — into a real ``["go", "test"]`` argv, or ``None`` when ``text`` is not
    a parseable list-of-strings (a plain command string is NOT shell-split here — the caller
    owns that fallback). SSOT shared by ``run_command`` (``shell._run_shell``) and
    ``verify_and_record`` (``verify._normalize_check``) so any command-taking tool recovers
    the same stringified-argv mistake identically (Bible P7 DRY / P2 class-fix). ``json``'s
    ``JSONDecodeError`` is a ``ValueError`` subclass, so the two parsers share one guard."""
    if not isinstance(text, str):
        return None
    for parse in (json.loads, ast.literal_eval):
        try:
            parsed = parse(text)
        except (ValueError, SyntaxError):
            continue
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
            return list(parsed)
    return None


def normalize_check_argv(check: Any) -> List[str] | None:
    """Normalize a verify_and_record ``check`` into the argv that is BOTH executed and
    shell-guard-inspected — ONE SSOT so the guard sees exactly what runs (they previously
    each hardcoded ``sh -lc`` and could drift). A string is first recovered as a stringified
    argv (``'["go","test"]'`` → ``["go","test"]``) via ``recover_stringified_argv``; a genuine
    command string runs as a NON-login ``sh -c`` one-liner — non-login so it inherits the
    bootstrapped PATH instead of a profile-reset PATH, matching run_command's toolchain
    resolution. A list/tuple becomes a trimmed argv. Empty / other type → ``None``."""
    if isinstance(check, str):
        text = check.strip()
        if not text:
            return None
        recovered = recover_stringified_argv(text)
        return recovered if recovered is not None else ["sh", "-c", text]
    if isinstance(check, (list, tuple)):
        argv = [str(part) for part in check if str(part or "").strip()]
        return argv or None
    return None


def shell_argv(raw_cmd: Any) -> List[str]:
    if isinstance(raw_cmd, list):
        return [str(x) for x in raw_cmd if str(x).strip()]
    try:
        return [str(x) for x in shlex.split(str(raw_cmd or "")) if str(x).strip()]
    except ValueError:
        return [str(x) for x in str(raw_cmd or "").split() if str(x).strip()]


def unwrap_env_argv(argv: List[str]) -> List[str]:
    if not argv or pathlib.PurePath(argv[0]).name.lower() != "env":
        return argv
    idx = 1
    options_with_arg = {"-u", "--unset", "-C", "--chdir", "--argv0"}
    while idx < len(argv):
        token = argv[idx]
        if token == "--":
            idx += 1
            break
        if token == "-S" and idx + 1 < len(argv):
            return shell_argv(argv[idx + 1])
        if token.startswith("--split-string="):
            return shell_argv(token.split("=", 1)[1])
        if token in options_with_arg:
            idx += 2
            continue
        if (
            any(token.startswith(prefix + "=") for prefix in ("--unset", "--chdir", "--argv0"))
            or token.startswith("-")
            or ("=" in token and not token.startswith("="))
        ):
            idx += 1
            continue
        break
    return argv[idx:] if idx < len(argv) else []


def strip_leading_env_assignments(argv: List[str]) -> List[str]:
    idx = 0
    while idx < len(argv) and "=" in argv[idx] and not argv[idx].startswith("="):
        idx += 1
    return argv[idx:]


_SEGMENT_SEPARATORS = frozenset({";", ";;", "&&", "||", "|", "|&", "&", "(", ")", "\n"})
# The characters the punctuation lexer below treats as SYNTAX. Every one of them is also
# a legal byte inside an argument, so which role a given occurrence plays is decided by
# QUOTING — information ``shlex`` discards. ``_LITERAL_MARK`` carries it across the lexer.
_PUNCTUATION_CHARS = ";&|()<>"
# NUL is the one byte that can never reach a real command line (``execve`` rejects it),
# which makes it the safe marker for "this character came from inside quotes / an escape".
# A NUL already present in the raw text is doubled on entry, so the marking round-trips.
_LITERAL_MARK = "\x00"


def _normalize_shell_source(text: str) -> str:
    """Rewrite a command into the form the punctuation lexer can tokenize WITHOUT
    losing either of the two things it would otherwise destroy.

    Unquoted newlines and backtick command substitutions become ``;``. The shell treats
    an unquoted newline like ``;``; ``shlex.split`` instead folds it into surrounding
    whitespace, which let ``cmd1\\ncmd2`` masquerade as a single command and slip a glued
    ``git`` invocation past per-segment inspection. Backslash-newline line-continuations
    collapse to a space (also matching the shell); quoted newlines are preserved verbatim.
    Unquoted backticks (legacy command substitution `` `git -C <runtime> reset` ``) become
    ``;`` so the substituted command is its own segment and is inspected — ``$()`` is
    already split by the punctuation lexer, backticks are not. Single-quoted backticks
    stay literal.

    And in the OTHER direction: a punctuation character that is QUOTED (or backslash
    escaped) is a literal argument byte, not syntax, so it is emitted preceded by
    ``_LITERAL_MARK``. ``shlex`` strips quotes and hands back bare text, after which
    ``echo '&&' x`` and ``echo && x`` are the same token list — two DIFFERENT commands
    with one identity, which is a false-green path in ``_outcome_receipts``. The mark
    survives lexing inside its token, so ``shell_tokens_typed`` can still tell the two
    apart; it is stripped again before any token is returned.
    """
    out: List[str] = []
    quote: str | None = None
    i = 0
    text = text.replace(_LITERAL_MARK, _LITERAL_MARK * 2)
    n = len(text)

    def emit_literal(ch: str) -> None:
        if ch in _PUNCTUATION_CHARS:
            out.append(_LITERAL_MARK)
        out.append(ch)

    while i < n:
        c = text[i]
        if quote:
            emit_literal(c)
            if c == "\\" and quote == '"' and i + 1 < n:
                emit_literal(text[i + 1])
                i += 2
                continue
            if c == quote:
                quote = None
            i += 1
            continue
        if c in ("'", '"'):
            quote = c
            out.append(c)
        elif c == "\\" and i + 1 < n and text[i + 1] == "\n":
            out.append(" ")
            i += 2
            continue
        elif c == "\\" and i + 1 < n and text[i + 1] in _PUNCTUATION_CHARS:
            # ``\&`` is a literal ampersand, not the operator — the shell's own rule.
            # OUTSIDE quotes the mark alone does not suffice: the lexer would still see
            # a bare punctuation character and split the token there, so the character is
            # also re-quoted (it can never be ``'`` itself) and the adjacent quoting is
            # concatenated back into one token exactly as the shell would.
            out.append(_LITERAL_MARK + "'" + text[i + 1] + "'")
            i += 2
            continue
        elif c == "\n":
            out.append(";")
        elif c == "`":
            out.append(";")
        else:
            out.append(c)
        i += 1
    return "".join(out)


def _unmark(token: str) -> str:
    """Drop the ``_LITERAL_MARK`` bytes ``_normalize_shell_source`` inserted, restoring
    the token exactly as the shell would pass it."""
    if _LITERAL_MARK not in token:
        return token
    out: List[str] = []
    i = 0
    while i < len(token):
        if token[i] == _LITERAL_MARK and i + 1 < len(token):
            out.append(token[i + 1])
            i += 2
            continue
        out.append(token[i])
        i += 1
    return "".join(out)


def shell_tokens_typed(raw_cmd: Any) -> List[tuple[str, bool]] | None:
    """Tokens of a shell command paired with whether each one is a CONTROL OPERATOR —
    real syntax — rather than a literal argument that merely spells like one.

    THE tokenizer of this module; ``shell_tokens`` and ``canonical_command_text`` are
    its two views. The flag is the piece ``shlex`` cannot give back: it strips quotes
    before yielding tokens, so ``echo '&&' x`` and ``echo && x`` arrive identical.
    ``_normalize_shell_source`` marks quoted/escaped punctuation on the way IN, this
    reads the mark, and the mark never leaves (tokens come back unmarked).

    A list is already tokenized, so every element is a literal argument: a caller
    passing argv cannot have glued an operator, and ``["a", "&&", "b"]`` really does
    run ``a`` with two arguments. Returns ``None`` when the text cannot be lexed
    (unbalanced quotes) so each caller picks its own fallback rather than inheriting a
    silent one.
    """
    if isinstance(raw_cmd, (list, tuple)):
        return [(str(x), False) for x in raw_cmd]
    text = _normalize_shell_source(str(raw_cmd or ""))
    try:
        lexer = shlex.shlex(text, posix=True, punctuation_chars=_PUNCTUATION_CHARS)
        lexer.whitespace_split = True
        raw = [t for t in lexer if t]
    except ValueError:
        return None
    return [
        (_unmark(t), _LITERAL_MARK not in t and all(ch in _PUNCTUATION_CHARS for ch in t))
        for t in raw
    ]


def shell_tokens(raw_cmd: Any) -> List[str] | None:
    """Operator-aware tokens of a shell command, control operators KEPT as their own
    tokens — the TEXT view of ``shell_tokens_typed``, shared by ``shell_segments``
    (which then drops the separators) and, indirectly, by every guard built on it.

    Robust against operators glued to adjacent words (``a;b``, ``a&&b``, ``$(cmd)``)
    and unquoted newlines — the cases plain ``shlex.split`` fuses into a single token.
    Quotes are respected, so whitespace INSIDE a quoted argument stays inside its token.
    A token that merely SPELLS like an operator is indistinguishable here by design:
    the guards that split on this view treat a quoted ``&&`` as a separator and so
    inspect MORE segments, which is the fail-safe direction. Callers that need the
    distinction (identity, not guarding) read ``shell_tokens_typed``.
    """
    typed = shell_tokens_typed(raw_cmd)
    return None if typed is None else [token for token, _ in typed]


def canonical_command_text(raw_cmd: Any) -> str:
    """The COMPARISON-STABLE form of a shell command: the same command written two
    cosmetically different ways yields the same text, and two DIFFERENT commands never
    do.

    Structural, not textual: the command is tokenized by ``shell_tokens_typed`` and
    rebuilt with exactly one space between tokens, so only the whitespace BETWEEN tokens
    is collapsed. Whitespace INSIDE a token is part of the argument and survives verbatim
    (re-quoted through ``shlex.quote``) — a flat ``" ".join(text.split())`` instead
    rewrites ``python -c "assert v == 'a  b'"`` into a command that asserts something
    else, and that mattered: this text is a verification's IDENTITY in
    ``_outcome_receipts``, so collapsing it let a green close an unrelated red.

    Nothing is dropped and nothing is re-classified. A token is rendered bare only when
    the lexer saw it as SYNTAX, so ``a && b`` and ``a '&&' b`` (which runs ``a`` with two
    arguments) canonicalize apart — round-5: the old form stripped leading/trailing
    separator-looking tokens AFTER ``shlex`` had discarded quoting, so
    ``shlex.join([..., "&&"])`` canonicalized to the same text as the argv WITHOUT that
    final argument, and a passing run of one could clear a failing run of the other. A
    trailing ``;`` or newline is likewise kept: it is a no-op to the shell, but proving
    that requires knowing it was syntax, and failing to equate two spellings of one
    command only ever leaves a red standing.

    Falls back to the merely STRIPPED raw text when the command cannot be lexed
    (unbalanced quotes): that can only fail to equate two spellings of one command,
    never equate two different ones.
    """
    typed = shell_tokens_typed(raw_cmd)
    if typed is None:
        return str(raw_cmd or "").strip()
    return " ".join(
        token if is_operator else shlex.quote(token) for token, is_operator in typed
    )


def shell_segments(raw_cmd: Any) -> List[List[str]]:
    """Split a shell command into per-command argv segments on control operators.

    Robust against operators glued to adjacent words (``a;b``, ``a&&b``,
    ``$(cmd)``) and unquoted newlines — the cases plain ``shlex.split`` fuses
    into a single token, which previously let ``cd ws;git -C <runtime> reset``
    masquerade as one ``cd`` segment with the ``-C`` selector never inspected.

    Lists are assumed already tokenized (a caller passing an argv list cannot
    glue operators) and are split on standalone separator tokens only.
    """
    tokens = shell_tokens(raw_cmd)
    if tokens is None:
        tokens = [t for t in str(raw_cmd or "").split() if t]
    segments: List[List[str]] = []
    current: List[str] = []
    for token in tokens:
        if token in _SEGMENT_SEPARATORS:
            if current:
                segments.append(current)
                current = []
            continue
        current.append(token)
    if current:
        segments.append(current)
    return segments


def collect_leading_env(argv: List[str]) -> tuple[dict, List[str]]:
    """Peel leading environment assignments off a command segment.

    Handles both the bare ``VAR=val cmd`` form and the ``env VAR=val cmd``
    wrapper, returning ``(assignments, remaining_argv)``. git honours
    ``GIT_DIR`` / ``GIT_WORK_TREE`` from the environment over cwd/``-C``, so
    guards must inspect these rather than discard them.
    """
    assignments: dict = {}
    rest = unwrap_env_argv(list(argv))
    # unwrap_env_argv drops the ``env`` wrapper but also its inline VAR=val
    # tokens; recover those from the original argv when an env wrapper was used.
    if argv and pathlib.PurePath(argv[0]).name.lower() == "env":
        for token in argv[1:]:
            if token == "--":
                break
            if token.startswith("-"):
                continue
            if "=" in token and not token.startswith("="):
                key, _, value = token.partition("=")
                assignments[key] = value
            else:
                break
    idx = 0
    while idx < len(rest) and "=" in rest[idx] and not rest[idx].startswith("="):
        key, _, value = rest[idx].partition("=")
        assignments[key] = value
        idx += 1
    return assignments, rest[idx:]


# Heads that only forward to the command that follows them: sudo behind one of
# these is still an invocation. A wrapper flag that consumes a VALUE token
# (``nice -n 10 sudo ...``) hides the wrapped head — a disclosed residual that
# can only miss a hang, never refuse work (this is a hang guard, not a
# privilege boundary).
_SUDO_FORWARDING_WRAPPERS = frozenset({
    "command", "builtin", "exec", "nohup", "time", "nice", "ionice", "stdbuf",
    "setsid", "timeout",
})


def sudo_noninteractive_violation(raw_cmd: Any) -> bool:
    """True when the command actually INVOKES interactive sudo/sudoedit.

    Judged by COMMAND POSITION per segment (the head after env/wrapper peeling),
    never by token mention: ``rg sudo README.md`` or ``ls /usr/bin/sudo`` name
    sudo as DATA and invoke nothing (issue #447 A3). Interactive sudo hangs
    forever on a password prompt in the headless runtime; ``sudo -n`` is fine,
    ``-S`` (password on stdin) is refused outright as before.
    """
    for segment in shell_segments(raw_cmd):
        _env, command = collect_leading_env(segment)
        while command:
            head = pathlib.PurePath(str(command[0])).name.lower()
            if head in _SHELLS:
                inline = shell_command_string(command)
                if inline and sudo_noninteractive_violation(inline):
                    return True
                break
            if head == "sudoedit":
                return True
            if head == "sudo":
                has_noninteractive = False
                for option in _sudo_option_tokens(command[1:]):
                    if option == "-S" or (option.startswith("-") and not option.startswith("--") and "S" in option[1:]):
                        return True
                    if option == "-n" or (option.startswith("-") and not option.startswith("--") and "n" in option[1:]):
                        has_noninteractive = True
                    if option.startswith("--non-interactive"):
                        has_noninteractive = True
                if not has_noninteractive:
                    return True
                # `sudo -n sh -c "sudo ..."` — keep walking the wrapped command.
                command = _sudo_wrapped_command(command[1:])
                continue
            if head in _SUDO_FORWARDING_WRAPPERS:
                command = command[1:]
                while command and str(command[0]).startswith("-"):
                    command = command[1:]
                continue
            break
    return False


def shell_command_string(argv: List[str]) -> str:
    for idx, arg in enumerate(argv[1:], start=1):
        if arg == "-c" or (arg.startswith("-") and not arg.startswith("--") and "c" in arg[1:]):
            return argv[idx + 1] if idx + 1 < len(argv) else ""
    return ""


def shell_argv_with_inline(raw_cmd: Any) -> List[str]:
    argv = shell_argv(raw_cmd)
    if argv and pathlib.PurePath(argv[0]).name.lower() in _SHELLS:
        inline = shell_command_string(argv)
        if inline:
            return argv + shell_argv(inline)
    return argv


def slash_normalize_path_text(text: Any) -> str:
    value = str(text or "").replace("\\", "/")
    while "//" in value:
        value = value.replace("//", "/")
    return value


def is_absolute_path_text(text: Any) -> bool:
    value = str(text or "")
    return (
        value.startswith("/")
        or bool(re.match(r"^[A-Za-z]:[\\/]", value))
        or value.startswith("\\\\")
    )


def path_text_is_inside(candidate: Any, root: Any) -> bool:
    candidate_text = slash_normalize_path_text(candidate).rstrip("/")
    root_text = slash_normalize_path_text(root).rstrip("/")
    if not candidate_text or not root_text:
        return False
    candidate_key = candidate_text.casefold()
    root_key = root_text.casefold()
    return candidate_key == root_key or candidate_key.startswith(root_key + "/")


def shell_argv_with_path_tokens(raw_cmd: Any) -> List[str]:
    tokens = list(shell_argv_with_inline(raw_cmd))
    raw_texts = [" ".join(str(x) for x in raw_cmd)] if isinstance(raw_cmd, list) else [str(raw_cmd or "")]
    seen = {str(token) for token in tokens}

    def add_token(value: str) -> None:
        if value and value not in seen:
            tokens.append(value)
            seen.add(value)

    for text in [*raw_texts, *[str(token) for token in tokens]]:
        for match in embedded_absolute_path_tokens(text):
            add_token(match)
        for match in EMBEDDED_WINDOWS_ABSOLUTE_PATH_RE.findall(text):
            add_token(match)
    return tokens


def embedded_absolute_path_tokens(text: Any) -> List[str]:
    """Extract POSIX absolute paths while ignoring HTML closing-tag fragments."""

    raw = str(text or "")
    tokens: List[str] = []
    for match in EMBEDDED_ABSOLUTE_PATH_RE.finditer(raw):
        value = match.group(0)
        if match.start() > 0 and raw[match.start() - 1] == "<" and _HTML_CLOSING_TAG_PATH_RE.fullmatch(value):
            continue
        tokens.append(value)
    return tokens


# -A/--askpass and -b/--background are FLAGS (no argument): listing them here
# made the walker eat the wrapped command's head as an "option value".
_SUDO_OPTIONS_WITH_ARG = frozenset({
    "-a", "-C", "-c", "-D", "-g", "-h", "-p", "-R", "-r", "-T", "-t", "-U", "-u",
    "--auth-type", "--chdir", "--close-from", "--command-timeout",
    "--context", "--group", "--host", "--login-class", "--prompt", "--role", "--type", "--user",
    "--other-user",
})


def _sudo_option_tokens(rest: List[str]) -> List[str]:
    options: List[str] = []
    idx = 0
    while idx < len(rest):
        token = rest[idx]
        if token == "--":
            break
        if not token.startswith("-") or token == "-":
            break
        options.append(token)
        idx += 2 if token in _SUDO_OPTIONS_WITH_ARG else 1
    return options


def _sudo_wrapped_command(rest: List[str]) -> List[str]:
    """The command sudo runs: its own options and leading VAR=val tokens peeled."""
    idx = 0
    while idx < len(rest):
        token = rest[idx]
        if token == "--":
            idx += 1
            break
        if not token.startswith("-") or token == "-":
            break
        idx += 2 if token in _SUDO_OPTIONS_WITH_ARG else 1
    return strip_leading_env_assignments(rest[idx:])
