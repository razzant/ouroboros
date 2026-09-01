"""Pure-read-inspection carve for shell guard predicates (#447 A2/A7).

Extracted verbatim from ``tools/registry.py`` at the module byte gate: the
owner-control mention detectors' shared read-carve — a HEAD allowlist over
per-command segments with per-command option denial, wrapper peeling, and a
fail-closed refusal of nested execution. ``registry`` re-exports
``_is_pure_read_inspection`` for its guard family and tests.
"""

from __future__ import annotations


# Commands that can only READ. This is an ALLOWLIST on purpose: an unrecognised
# command head is treated as executable access, so the enumeration fails CLOSED.
# (A denylist of "write markers" fails OPEN — every new spelling of a POST walks
# around it, which is exactly the keyword-gate antipattern BIBLE P5 forbids.)
_READ_ONLY_INSPECTION_COMMANDS = frozenset({
    "grep", "egrep", "fgrep", "zgrep", "rg", "ag", "ack", "ripgrep",
    "cat", "bat", "head", "tail", "less", "more", "nl", "strings",
    "ls", "find", "fd", "stat", "file", "wc", "sort", "uniq", "cut", "tr", "column",
    "basename", "dirname", "realpath", "readlink", "diff", "cmp", "jq", "yq",
    "echo", "printf", "true", "pwd", "date", "tree",
})
# Wrappers that do not themselves act: the real command head follows them.
_COMMAND_HEAD_WRAPPERS = frozenset({
    "sudo", "env", "command", "builtin", "exec", "nohup", "time", "nice", "ionice",
    "stdbuf", "\\",
})
# ``git`` read-only classification: the git_shell_policy SSOT
# (``_git_subcommand_is_readonly``), shared with the shell git guards and the
# affordance map — two divergent allowlists disagreed on the same line (#447 A7).
# Allowlist MEMBERSHIP IS NOT ENOUGH: several read heads execute or write through their
# own options. Per command, because short flags are not portable — ``grep -o`` prints
# matches, ``sort -o`` writes a file. Text reaching here is lowercased, so an upper-case
# spelling (``git grep -O``, ``fd -X``) collapses onto the same entry.
_SEARCH_TOOL_EXEC_OPTIONS = frozenset({"--pre", "--pre-glob", "--hostname-bin", "--pager"})
_DENIED_READ_OPTIONS: dict = {
    # find/fd run and delete: -exec/-execdir/-ok/-okdir/-x, -delete, and the -f* writers.
    "find": frozenset({
        "-exec", "-execdir", "-ok", "-okdir", "-delete",
        "-fls", "-fprint", "-fprint0", "-fprintf",
    }),
    "fd": frozenset({"-x", "--exec", "--exec-batch"}),
    "rg": _SEARCH_TOOL_EXEC_OPTIONS,
    "ripgrep": _SEARCH_TOOL_EXEC_OPTIONS,
    "ag": _SEARCH_TOOL_EXEC_OPTIONS,
    "ack": _SEARCH_TOOL_EXEC_OPTIONS,
    # yq edits the named file in place with -i/--inplace; without this the family
    # read-carve exempted `yq -i '.OUROBOROS_SAFETY_MODE="off"' settings.json` as
    # "pure inspection" (jq has no in-place edit and stays a stdout-only read).
    "yq": frozenset({"-i", "--inplace"}),
    "sort": frozenset({"-o", "--output", "--compress-program"}),
    "less": frozenset({"-o", "--log-file", "-k", "--lesskey-file"}),
    "more": frozenset({"-o"}),
    "file": frozenset({"-c", "--compile"}),
    # git: external diff/textconv helpers execute a configured program, -o/--output and
    # git grep -O write or spawn a pager, --exec-path relocates the git binaries.
    "git": frozenset({
        "-c", "--config-env", "--exec-path", "--ext-diff", "--textconv",
        "-o", "--output", "--open-files-in-pager",
    }),
}
# The executable itself must be a bare name or live in a system bin: ``/tmp/evil/grep``
# and ``./grep`` are shadowing, not inspection.
_TRUSTED_EXECUTABLE_DIRS = frozenset({
    "/bin", "/usr/bin", "/usr/local/bin", "/sbin", "/usr/sbin", "/opt/homebrew/bin",
})


def _trusted_read_head(token: str) -> str:
    """The allowlist-comparable command name, or "" when the executable is untrusted."""
    if "\\" in token:
        return ""  # a windows/escaped path is not a form we can resolve — fail closed
    directory, sep, name = token.rpartition("/")
    if sep and directory not in _TRUSTED_EXECUTABLE_DIRS:
        return ""
    return name.removesuffix(".exe")


def _denied_read_option(token: str, denied: frozenset) -> bool:
    """True when an argument spells an execution/mutation option of its command."""
    if not token.startswith("-") or token in {"-", "--"}:
        return False
    name = token.split("=", 1)[0]
    if name in denied:
        return True
    if name.startswith("--"):
        return False
    return any(f"-{letter}" in denied for letter in name[1:])  # bundled short cluster


# Spellings that make a shell run a command NESTED inside another one. The read exemption
# fails closed on all of them: the head-allowlist can only vouch for heads it actually sees,
# and a nested command's head is not one of them ("echo" vouching for the "curl -X POST" it
# interpolates). Refusing the CONSTRUCT rather than enumerating the payloads inside it is the
# point — no list of "what a write looks like" is ever complete (BIBLE P5).
_NESTED_EXECUTION_MARKERS = ("$(", "`", "<(", ">(")
# Bare tokens the lexer emits for the same constructs (and for a plain subshell). These used to
# be STRIPPED from the token list before the head was taken, which is precisely how the nested
# command escaped validation; they are refused instead.
_NESTED_EXECUTION_TOKENS = frozenset({"$", "(", ")", "<(", ">(", "$("})


def _is_pure_read_inspection(text_lower: str) -> bool:
    """True when EVERY command in a shell line is a read-only source inspection.

    Structural, not keyword-based: the line is split into per-command segments with
    the shared lexer (``shell_parse.shell_segments``) and each segment's HEAD is
    matched against an allowlist. An unknown head — any interpreter, HTTP client,
    or shell — is not an inspection, whatever flags or payload spelling it carries.

    Head membership is NECESSARY, NOT SUFFICIENT (review round 2): an allowed head can
    still execute through its own options (``find -exec``, ``rg --pre``, git's external
    diff/textconv) or through what precedes it. So the options are validated per command
    (``_DENIED_READ_OPTIONS``), a leading environment assignment is REFUSED rather than
    dropped (``PATH=``/``LD_PRELOAD=``/``GIT_EXTERNAL_DIFF=`` change what actually runs),
    wrappers may not carry their own flags (``env -i``, ``sudo -e``), and the executable
    must resolve to a bare name or a system bin. Anything unrecognised stays fail-closed.

    NESTED EXECUTION IS REFUSED BEFORE ANY OF THAT (review round 3). Only the heads the lexer
    actually surfaces get validated, so a command substitution hid its command from every check
    above: ``echo "$(curl -X POST .../api/owner/scope-review-floor)"`` presented the allowlisted
    ``echo``, and the write-shape detector does not recognise an HTTP POST, so the exemption was
    granted to a line that existed to reach the owner-only endpoint. A quoted substitution is
    one opaque argument token to the lexer, which is why this is a check on the TEXT and on the
    tokens, not something the per-segment head walk could have caught.
    """
    from ouroboros.shell_parse import shell_segments

    if any(marker in text_lower for marker in _NESTED_EXECUTION_MARKERS):
        return False
    segments = shell_segments(text_lower)
    if not segments:
        return False
    for segment in segments:
        if any(token in _NESTED_EXECUTION_TOKENS for token in segment):
            return False
        tokens = [token for token in segment if token]
        while tokens and tokens[0] in _COMMAND_HEAD_WRAPPERS:
            tokens = tokens[1:]
            if tokens and tokens[0].startswith("-"):
                return False  # a wrapper's own options can rebuild the environment
        if not tokens:
            continue  # a bare wrapper executes nothing
        if "=" in tokens[0] and not tokens[0].startswith(("-", "=")):
            return False  # leading env assignment: never silently discarded
        head = _trusted_read_head(tokens[0])
        if head == "git":
            from ouroboros.git_shell_policy import (
                _git_output_file_args,
                _git_subcommand_and_args,
                _git_subcommand_is_readonly,
            )

            subcmd, sub_args = _git_subcommand_and_args(tokens)
            if not subcmd or not _git_subcommand_is_readonly(subcmd, sub_args):
                return False
            if _git_output_file_args(sub_args):
                return False  # `git log --output=<file>` truncates and writes <file>
        elif not head or head not in _READ_ONLY_INSPECTION_COMMANDS:
            return False
        denied = _DENIED_READ_OPTIONS.get(head)
        if denied and any(_denied_read_option(token, denied) for token in tokens[1:]):
            return False
        if head == "uniq" and sum(1 for t in tokens[1:] if t == "-" or not t.startswith("-")) >= 2:
            # uniq's SECOND positional operand is its output file ('-' is the
            # stdin operand, not a flag): `... | uniq - settings.json` writes.
            return False
    return True

