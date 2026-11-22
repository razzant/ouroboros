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
POSIX_SHELL_HEADS = frozenset({"sh", "bash", "zsh", "dash", "ash"})


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
    resolution. A list/tuple keeps every argument, including an empty one (``pytest -m ""``),
    and only drops ``None``; a blank ``argv[0]`` is refused instead of shifting the
    arguments. Empty / other type → ``None``."""
    if isinstance(check, str):
        text = check.strip()
        if not text:
            return None
        recovered = recover_stringified_argv(text)
        if recovered is not None:
            if not recovered or not str(recovered[0] or "").strip():
                return None
            return recovered
        return ["sh", "-c", text]
    if isinstance(check, (list, tuple)):
        if not check or not str(check[0] or "").strip():
            return None
        return [str(part) for part in check if part is not None]
    return None


def shell_argv(raw_cmd: Any) -> List[str]:
    """The argv a guard inspects: a list/tuple keeps every element and a command string
    keeps every token ``shlex`` yields, including an empty quoted one (``''``). An empty
    argument still occupies a position the program sees (``git -C ''`` keeps the cwd and
    ``sh -c '' x`` runs an empty body), so dropping it would show the guard a different
    command than the one that executes. Only an unparseable string falls back to plain
    whitespace words, which never contain an empty one."""
    if isinstance(raw_cmd, (list, tuple)):
        return [str(x) for x in raw_cmd]
    try:
        return shlex.split(str(raw_cmd or ""))
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


def sequential_effective_cwds(target_rows: list, initial_cwd: pathlib.Path) -> list[pathlib.Path]:
    """Symlink-resolved cwd in force at the start of each parsed command row."""
    current = pathlib.Path(initial_cwd).resolve(strict=False)
    result: list[pathlib.Path] = []
    for argv, _targets, _inline, _unknown in target_rows:
        result.append(current)
        head = pathlib.PurePath(str(argv[0])).name.lower() if argv else ""
        if head not in {"cd", "pushd"}:
            continue
        operand = next((str(item) for item in argv[1:] if str(item) and not str(item).startswith("-")), "")
        if operand:
            path = pathlib.Path(operand).expanduser()
            current = (path if path.is_absolute() else current / path).resolve(strict=False)
    return result


def directory_destination_child_name(command: str, argv: List[str], source: str) -> str:
    """Return the child path a directory destination receives from ``source``."""
    source_text = str(source or "").replace("\\", "/").rstrip("/")
    if command == "cp" and "--parents" in {str(item) for item in argv[1:]}:
        parts = [part for part in pathlib.PurePosixPath(source_text).parts if part not in {"", ".", "/"}]
        if not parts or ".." in parts:
            return ""
        return "/".join(parts)
    return pathlib.PurePath(source_text).name


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
        raw = list(lexer)
    except ValueError:
        return None
    return [
        (_unmark(t), bool(t) and _LITERAL_MARK not in t and all(ch in _PUNCTUATION_CHARS for ch in t))
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


# The ONE redirection grammar of this module. A redirection operator is only an
# operator at the START of a token (after an optional file-descriptor prefix), so
# `s/>/x/` and `--pretty=<x>` never match, while both the glued (`2>/dev/null`)
# and the split (`2` `>` `/dev/null`) spellings the two lexers emit are covered.
_REDIRECT_OPERATOR_RE = re.compile(
    r"^(?P<fd>[0-9]+)?(?P<op><<<|<<|<&|<|&>>|&>|>>|>\||>&|>)(?P<operand>.*)$"
)


def split_redirections(tokens: List[str]) -> tuple[List[str], List[str]]:
    """Split redirections off a command segment.

    Returns ``(argv_without_redirections, output_targets)``. Input redirections
    (``<``, ``<<``, ``<<<``, ``<&``) and descriptor duplication/closure
    (``2>&1``, ``>&2``, ``>&-``) yield no target; ``/dev/null`` is exempted on
    the clean operand. Without this split the writer-target lane counted the
    redirect OPERAND as the command's own operand — `cp x y >> log.txt` reported
    `log.txt` and LOST the destination `y`, and a glued `2>/dev/null;` was forged
    into the path `/dev/null;`.
    """
    items = [str(token) for token in (tokens or [])]
    argv: List[str] = []
    targets: List[str] = []
    index = 0
    while index < len(items):
        token = items[index]
        match = _REDIRECT_OPERATOR_RE.match(token)
        if match is None and token.isdigit() and index + 1 < len(items):
            # A bare descriptor token belongs to the operator that follows it.
            following = _REDIRECT_OPERATOR_RE.match(items[index + 1])
            if following is not None and not following.group("fd"):
                match = following
                index += 1
        if match is None:
            argv.append(token)
            index += 1
            continue
        operator = match.group("op")
        operand = match.group("operand")
        if not operand and index + 1 < len(items):
            index += 1
            operand = items[index]
        index += 1
        if operator.startswith("<"):
            continue
        if operator == ">&" and (operand == "-" or operand.isdigit()):
            continue
        if not operand or operand == "/dev/null":
            continue
        targets.append(operand)
    return argv, targets


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
            if head in POSIX_SHELL_HEADS:
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


# An explicit Windows interpreter takes everything after its inline switch as its own
# command text (cmd /c|/k, PowerShell -c|-Command). After PowerShell's -File script or
# -CommandWithArgs command, the remaining argv are passed on as that one's arguments.
_CMD_INLINE_SWITCHES = frozenset({"/c", "/k"})
_POWERSHELL_INLINE_SWITCHES = frozenset({"-c", "-command", "/c"})
_POWERSHELL_ARGUMENT_SWITCHES = frozenset({"-file", "-f", "-commandwithargs", "-cwa"})


def inline_operand_start(argv: List[str], shell_name: str, *, strict: bool = False) -> int:
    """Index where an explicit ``cmd``/``powershell``/``pwsh`` inline command operand begins
    (``len(argv)`` after a trailing switch), 0 when there is none. ``strict`` stops at a
    PowerShell switch that passes the remaining argv to a script as arguments."""
    switches = _CMD_INLINE_SWITCHES if shell_name == "cmd" else _POWERSHELL_INLINE_SWITCHES
    for idx, arg in enumerate(argv[1:], start=1):
        word = str(arg or "").strip().lower()
        if word in switches:
            return idx + 1
        if strict and shell_name != "cmd" and word in _POWERSHELL_ARGUMENT_SWITCHES:
            return 0
    return 0


def inline_shell_command(argv: List[str], shell_name: str) -> str:
    """The inline command text an explicit shell receives, joined; "" when there is none."""
    if shell_name in {"bash", "sh", "zsh"}:
        return shell_command_string(argv)
    start = inline_operand_start(argv, shell_name)
    return " ".join(str(part) for part in argv[start:]) if start else ""


def shell_argv_with_inline(raw_cmd: Any) -> List[str]:
    argv = shell_argv(raw_cmd)
    if argv and pathlib.PurePath(argv[0]).name.lower() in POSIX_SHELL_HEADS:
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
