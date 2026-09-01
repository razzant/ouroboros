"""Git command classifiers for shell-tool safety guards."""

from __future__ import annotations

import pathlib
import os
from typing import Any

from ouroboros.shell_parse import (
    collect_leading_env,
    shell_argv,
    shell_command_string,
    shell_segments,
    strip_leading_env_assignments,
    unwrap_env_argv,
)
from ouroboros.utils import safe_relpath

GIT_READONLY_SUBCOMMANDS = frozenset([
    "status", "diff", "log", "show", "ls-files", "describe", "rev-parse",
    "cat-file", "shortlog", "version", "help", "blame", "grep",
    "for-each-ref", "rev-list", "show-ref", "ls-tree", "merge-base",
    "check-ignore", "count-objects", "var", "name-rev", "verify-commit",
])
# NOT here: multi-mode subcommands (`branch`, `tag`, `reflog`, `remote`) are
# mode-parsed in _git_subcommand_is_readonly — an unconditional entry would
# classify `git reflog expire`/`git remote add` as read-only and skip the
# target checks (the SC-10 hole: reflog sat in this set unconditionally).
# Subcommands that reach the network (gated by allowed_resources.network in
# external workspaces, where local git is otherwise unrestricted).
GIT_NETWORK_SUBCOMMANDS = frozenset([
    "clone", "fetch", "pull", "push", "ls-remote", "submodule",
    "remote", "archive", "lfs",
])
_SHELL_SEPARATORS = frozenset({";", "&&", "||", "|", "&", "(", ")"})
_BRANCH_MUTATING_FLAGS = frozenset({
    "-d", "-D", "-m", "-M", "-c", "-C", "-f", "-u",
    "--delete", "--move", "--copy", "--force", "--set-upstream-to",
    "--unset-upstream", "--edit-description", "--track", "--no-track",
})
_BRANCH_READONLY_FLAGS = frozenset({
    "-l", "--list", "-a", "--all", "-r", "--remotes", "-v", "-vv",
    "--verbose", "--show-current", "--contains", "--merged", "--no-merged",
    "--points-at", "--format", "--sort", "--color", "--no-color",
    "--column", "--no-column", "--abbrev", "--no-abbrev", "--ignore-case",
})
_TAG_MUTATING_FLAGS = frozenset({
    "-a", "-s", "-u", "-d", "-f", "-m", "-F",
    "--annotate", "--sign", "--local-user", "--delete",
    "--force", "--message", "--file", "--cleanup", "--create-reflog",
})
# `-v`/`--verify` checks a tag's GPG signature and writes nothing — it is
# read-only inspection, not mutation (it sat in the mutating set, refusing
# `git tag -v <tag>` at a runtime target against the SYSTEM.md contract).
_TAG_READONLY_FLAGS = frozenset({
    "-l", "--list", "-n", "-v", "--verify", "--sort", "--format", "--points-at",
    "--contains", "--merged", "--no-merged", "--column", "--no-column",
    "--ignore-case", "--color", "--no-color",
})
# `git remote` and `git reflog` are VERB-dispatched: the first non-flag token
# selects the mode. remote: bare listing (`git remote`, `git remote -v`) and
# `show`/`get-url` only read; every other verb (add/remove/rename/set-url/
# set-head/set-branches/prune/update) mutates. reflog: the default is `show`
# (any ref positional falls through to it) and `exists`/`list` also only read;
# `expire`/`delete`/`drop` rewrite or remove reflog entries.
_REMOTE_READONLY_MODES = frozenset({"show", "get-url"})
_REFLOG_MUTATING_MODES = frozenset({"expire", "delete", "drop", "write"})
# Verb-dispatched read modes for the remaining multi-mode subcommands (#447 A7,
# same shape as branch/tag/remote/reflog). Bare `git stash` pushes and bare
# `git worktree`/`git bisect` print usage — neither is classified read-only
# (conservative); bare `git notes` defaults to `list` and only reads.
_STASH_READONLY_MODES = frozenset({"list", "show"})
_WORKTREE_READONLY_MODES = frozenset({"list"})
_NOTES_READONLY_MODES = frozenset({"", "list", "show", "get-ref"})
_BISECT_READONLY_MODES = frozenset({"log", "visualize", "view", "terms"})
# git config: legacy flag-dispatched modes. `-f`/`--file`/`--blob` make config a
# general file reader OUTSIDE the repo, so they disqualify the read-only
# classification outright (the `--no-index` precedent: reads must not ride the
# repo-inspection exemption at arbitrary targets).
_CONFIG_READONLY_ACTION_FLAGS = frozenset({
    "--get", "--get-all", "--get-regexp", "--get-urlmatch", "-l", "--list",
    "--get-color", "--get-colorbool",
})
_CONFIG_MUTATING_FLAGS = frozenset({
    "--add", "--replace-all", "--unset", "--unset-all", "--rename-section",
    "--remove-section", "-e", "--edit", "--fixed-value",
})
_CONFIG_FILE_FLAGS = frozenset({"-f", "--file", "--blob"})
_CONFIG_VALUE_FLAGS = frozenset({"--default", "--type", "-t"})
_CONFIG_READONLY_VERBS = frozenset({"get", "list"})
_CONFIG_MUTATING_VERBS = frozenset({"set", "unset", "edit", "rename-section", "remove-section"})


def _git_config_readonly(args: list[str]) -> bool:
    """`git config` reads through --get*/--list (or the new `get`/`list` verbs)
    and through the one-positional legacy `git config <key>` form; a value
    positional, a mutating flag/verb, or an external-file selector is not
    read-only inspection."""
    read_hint = False
    positionals: list[str] = []
    skip_value = False
    for arg in args:
        text = str(arg)
        if skip_value:
            skip_value = False
            continue
        flag = text.split("=", 1)[0] if text.startswith("-") else text
        if flag in _CONFIG_FILE_FLAGS or flag in _CONFIG_MUTATING_FLAGS:
            return False
        if text.startswith("-"):
            if flag in _CONFIG_READONLY_ACTION_FLAGS:
                read_hint = True
            skip_value = flag in _CONFIG_VALUE_FLAGS and "=" not in text
            continue
        positionals.append(text)
    if positionals and positionals[0].lower() in _CONFIG_MUTATING_VERBS:
        return False
    if positionals and positionals[0].lower() in _CONFIG_READONLY_VERBS:
        return True
    if read_hint:
        return len(positionals) <= 1
    return len(positionals) == 1


# `gh auth` verbs that MUTATE the host GitHub identity. `status`/`token` (and an
# unknown future verb) only read — fail-open on unknown verbs is deliberate: the
# hazard is a closed, named set of identity mutations, and the LLM safety layer
# still reviews intent (#447 A7: the old substring scan refused `rg "gh auth"`).
_GH_AUTH_MUTATING_VERBS = frozenset({"login", "logout", "refresh", "switch", "setup-git"})


def gh_shell_block_reason(raw_cmd: Any) -> str:
    """Positional gh policy: judged only where `gh` is a segment's command head."""
    for segment in shell_segments(raw_cmd):
        _env, command = collect_leading_env(segment)
        if not command:
            continue
        head = pathlib.PurePath(str(command[0])).name.lower()
        if head in {"bash", "sh", "zsh"}:
            inline = shell_command_string(command)
            if inline and (nested := gh_shell_block_reason(inline)):
                return nested
            continue
        if head != "gh":
            continue
        words = [str(t).lower() for t in command[1:] if not str(t).startswith("-")]
        if len(words) >= 2 and words[0] == "repo" and words[1] in {"create", "delete"}:
            return "⚠️ SAFETY_VIOLATION: Creating/deleting GitHub repositories requires admin approval."
        if len(words) >= 2 and words[0] == "auth" and words[1] in _GH_AUTH_MUTATING_VERBS:
            return "⚠️ SAFETY_VIOLATION: Modifying GitHub authentication is not permitted. Read-only `gh auth status` / `gh auth token` are allowed."
    return ""


def _git_subcommand_and_args(cmd_parts: list[str]) -> tuple[str, list[str]]:
    parts = strip_leading_env_assignments([str(p) for p in cmd_parts])
    if not parts or pathlib.PurePath(parts[0]).name.lower() != "git":
        return "", []
    i = 1
    while i < len(parts):
        part = parts[i]
        if part in _SHELL_SEPARATORS:
            # `git --version; echo done` — the git invocation ends at the
            # separator; what follows is a different command, not a subcommand.
            return "", []
        if part.startswith("-"):
            i += 2 if part in ("-C", "-c", "--git-dir", "--work-tree") else 1
            continue
        return part.lower(), parts[i + 1:]
    return "", []


def _git_option_value_flags(args: list[str]) -> set[int]:
    value_taking_flags = {
        "--contains", "--merged", "--no-merged", "--points-at", "--format",
        "--sort", "--color", "--column", "--abbrev", "-n", "-m", "-F", "-u",
        "--message", "--file", "--local-user", "--set-upstream-to",
    }
    return {idx + 1 for idx, arg in enumerate(args[:-1]) if arg in value_taking_flags}


def _short_flag_chars(arg: str) -> set[str]:
    text = str(arg or "")
    return set(text[1:]) if text.startswith("-") and not text.startswith("--") else set()


def _git_branch_readonly(args: list[str]) -> bool:
    value_indexes = _git_option_value_flags(args)
    read_hint = not args
    explicit_list = False
    positionals = []
    for idx, arg in enumerate(args):
        if idx in value_indexes:
            continue
        if arg in _BRANCH_MUTATING_FLAGS or _short_flag_chars(arg) & set("dDmMcCfFu"):
            return False
        if arg.startswith("--") and "=" in arg:
            flag = arg.split("=", 1)[0]
            if flag in _BRANCH_MUTATING_FLAGS:
                return False
            explicit_list = explicit_list or flag == "--list"
            read_hint = read_hint or flag in _BRANCH_READONLY_FLAGS
            continue
        if arg.startswith("-"):
            chars = _short_flag_chars(arg)
            if arg == "--list" or "l" in chars:
                explicit_list = True
            if arg in _BRANCH_READONLY_FLAGS or chars <= set("alrv"):
                read_hint = True
                continue
            return False
        positionals.append(arg)
    return bool(read_hint and (not positionals or explicit_list))


def _git_tag_readonly(args: list[str]) -> bool:
    value_indexes = _git_option_value_flags(args)
    read_hint = not args
    positionals = []
    for idx, arg in enumerate(args):
        if idx in value_indexes:
            continue
        if arg in _TAG_MUTATING_FLAGS or _short_flag_chars(arg) & set("asudfmF"):
            return False
        if arg.startswith("--") and "=" in arg:
            flag = arg.split("=", 1)[0]
            if flag in _TAG_MUTATING_FLAGS:
                return False
            read_hint = read_hint or flag in _TAG_READONLY_FLAGS
            continue
        if arg.startswith("-"):
            chars = _short_flag_chars(arg)
            if arg in _TAG_READONLY_FLAGS or chars <= set("ln"):
                read_hint = True
                continue
            return False
        positionals.append(arg)
    return read_hint or not positionals


def _git_verb_mode(args: list[str]) -> str:
    """The dispatch verb of a verb-dispatched subcommand: the first non-flag
    token, NOT ``args[0]`` — flags before the verb still dispatch (measured:
    ``git remote -v add origin <url>`` ADDS the remote), so judging ``args[0]``
    alone would read that spelling as a bare listing."""
    for arg in args:
        text = str(arg)
        if not text.startswith("-"):
            return text.lower()
    return ""


def _git_remote_readonly(args: list[str]) -> bool:
    """Bare ``git remote`` / ``git remote -v`` lists and ``show``/``get-url``
    inspect — none of them writes, so all are read-only inspection. Every other
    verb mutates config/refs and stays target-checked; the network dimension
    (``remote show`` without ``-n`` contacts the remote) is enforced separately
    by the per-subcommand network fence, never by this mutation classifier."""
    verb = _git_verb_mode(args)
    return not verb or verb in _REMOTE_READONLY_MODES


def _git_reflog_readonly(args: list[str]) -> bool:
    """``git reflog`` defaults to ``show`` and any non-verb token falls through
    to it as a revision (measured: ``git reflog --all expire`` errors as
    "ambiguous argument", it does NOT expire), so only the explicit mutating
    verbs — ``expire``/``delete``/``drop`` — are mutations. Everything else
    (bare, ``show``, ``exists``, ``list``, a ref positional) only reads."""
    return _git_verb_mode(args) not in _REFLOG_MUTATING_MODES


def _git_subcommand_is_readonly(subcmd: str, args: list[str]) -> bool:
    """The subcommand alone reads and never writes (before flags are considered)."""
    if subcmd in GIT_READONLY_SUBCOMMANDS:
        return True
    if subcmd == "branch":
        return _git_branch_readonly(args)
    if subcmd == "tag":
        return _git_tag_readonly(args)
    if subcmd == "reflog":
        return _git_reflog_readonly(args)
    if subcmd == "remote":
        return _git_remote_readonly(args)
    if subcmd == "config":
        return _git_config_readonly(args)
    if subcmd == "stash":
        return _git_verb_mode(args) in _STASH_READONLY_MODES
    if subcmd == "worktree":
        return _git_verb_mode(args) in _WORKTREE_READONLY_MODES
    if subcmd == "notes":
        return _git_verb_mode(args) in _NOTES_READONLY_MODES
    if subcmd == "bisect":
        return _git_verb_mode(args) in _BISECT_READONLY_MODES
    return False


def _git_output_file_args(args: list[str]) -> list[str]:
    """Values of git's ``--output=<file>`` diff option, which TRUNCATES and writes
    <file>. ``log``/``show``/``diff`` all accept it (measured against real git:
    ``git log -1 --output=<file>`` replaced the file's contents), so a subcommand
    from the read-only set carrying it is NOT a read-only invocation.

    ``-o`` is deliberately NOT matched: in this family it means something else
    entirely (``git ls-files -o`` == ``--others``), and matching it would refuse an
    ordinary untracked-file listing. ``--output-indicator-new=<char>`` and its
    siblings are not files, which is why the glued form is matched on ``--output=``
    rather than on the ``--output`` prefix."""
    values: list[str] = []
    idx = 0
    while idx < len(args):
        text = str(args[idx])
        if text == "--output" and idx + 1 < len(args):
            values.append(str(args[idx + 1]))
            idx += 2
            continue
        if text.startswith("--output="):
            values.append(text.split("=", 1)[1])
        idx += 1
    return values


def _git_reads_arbitrary_files(args: list[str]) -> bool:
    """``--no-index`` turns ``git diff``/``git grep`` into a general FILE reader:
    ``git diff --no-index /dev/null <data>/settings.json`` prints the file verbatim
    (measured). It writes nothing, so it is not a mutation — but it is not repo
    inspection either, and must not ride the runtime/secret read-guard exemption.
    Disclosed residual: git IMPLIES ``--no-index`` when it runs outside a work tree,
    and that spelling is not classified here (no shell-parser arms race)."""
    return any(str(arg) == "--no-index" for arg in args)


def _git_invocation_block_reason(parts: list[str], *, allow_network: bool = True) -> str:
    subcmd, args = _git_subcommand_and_args(parts)
    if not subcmd:
        return ""
    if _git_subcommand_is_readonly(subcmd, args):
        # A read-only SUBCOMMAND carrying a file-WRITING flag is not a read-only
        # invocation: `git log|show|diff --output=<file>` truncates <file>. Judged
        # here, in the one classifier every consumer asks, so the exemption key and
        # both target resolvers get the same honest answer.
        if _git_output_file_args(args):
            return f"git {subcmd} --output"
        if subcmd in GIT_NETWORK_SUBCOMMANDS and not allow_network:
            # Read-only inspection can still reach the network (`git remote show`
            # without `-n` contacts the remote): the contract fence is enforced
            # per-subcommand in every lane, read-only or not — the ls-remote
            # precedent, and the same coarse rule the external resolver's tail
            # applies. Only `remote` currently sits in both sets.
            return f"task_contract.allowed_resources.network=false blocks git {subcmd}"
        return ""
    if subcmd == "ls-remote":
        return "" if allow_network else "task_contract.allowed_resources.network=false blocks git ls-remote"
    return f"git {subcmd}"


# Subcommands whose mutation lands at an EXPLICIT DESTINATION path, leaving the
# working directory untouched: `git init <dir>` and `git clone <url> <dir>` create
# a NEW repository at <dir>. Checking such an invocation against its cwd is a pure
# false block — and a load-bearing one, because the DEFAULT (non-workspace) lane's
# default cwd IS the system repo, so `git init ~/projects/foo` from direct chat or
# light mode was refused even though nothing in the runtime was touched. With no
# explicit destination (`git init`, `git clone <url>`) the cwd IS the target and
# the working-directory check applies as before.
_GIT_DESTINATION_SUBCOMMANDS = frozenset({"init", "clone"})
# SPLIT-form (`-b main`) options of init/clone that consume the next token. The
# glued form (`--initial-branch=main`) needs no table. This is a CAPABILITY aid,
# never a safety boundary: an option missing from it merely leaves its value among
# the positionals, where the path-SHAPE test below decides — and a value that does
# not look like a path leaves the cwd as the judged target (the conservative
# answer), while one that does is resolved and containment-checked like any path.
_GIT_DESTINATION_VALUE_FLAGS = frozenset({
    "-b", "--initial-branch", "--branch", "-o", "--origin", "-u", "--upload-pack",
    "-c", "--config", "--depth", "--reference", "--reference-if-able", "--template",
    "--separate-git-dir", "--object-format", "--ref-format", "--filter", "-j",
    "--jobs", "--shallow-since", "--shallow-exclude", "--server-option",
    "--bundle-uri", "--revision",
})
# The subset whose value git interprets as something OTHER than a filesystem
# path: a branch/remote/ref name, a number, a filter spec, a config pair. A
# hierarchical ref (`-b feature/x`) is indistinguishable from a relative path by
# SHAPE, so without this the destination-branch argument scan resolved it under
# the effective base — a false block for the Q4=A headline spelling `git clone -b
# feature/x <url> ~/projects/x` from the default lane's system-repo cwd. Values
# of the PATH-taking flags (`--template`, `--reference`, `--separate-git-dir`,
# `--bundle-uri`, `--upload-pack`) are deliberately NOT here: those genuinely
# name filesystem paths and must keep meeting the containment scan.
_GIT_DESTINATION_NONPATH_VALUE_FLAGS = frozenset({
    "-b", "--initial-branch", "--branch", "-o", "--origin", "-c", "--config",
    "--depth", "-j", "--jobs", "--filter", "--object-format", "--ref-format",
    "--shallow-since", "--shallow-exclude", "--server-option", "--revision",
})


def _git_remote_url(token: str) -> bool:
    """The token names a REMOTE (``scheme://host/path``, ``user@host:path``) rather
    than a local path. Structural, not a scheme list: a ``://`` separator, or a
    colon appearing before any path separator. A Windows drive spelling (``C:\\x``)
    is a local path, not a remote."""
    text = str(token or "")
    if not text or text.startswith("-"):
        return False
    if "://" in text:
        return True
    if len(text) > 2 and text[1] == ":" and text[0].isalpha() and text[2] in "\\/":
        return False  # C:\proj / C:/proj
    head = text.split("/", 1)[0]
    return ":" in head and not text.startswith(("/", ".", "~"))


def _git_path_shaped(token: str) -> bool:
    """The token names a filesystem path rather than a flag, a branch/remote name,
    a URL or an option spec (``blob:none``). Used where the effective base may
    ITSELF be a runtime root: joining a non-path token onto it would resolve
    "inside the runtime" and refuse the whole invocation."""
    text = str(token or "")
    if not text or text.startswith("-") or _git_remote_url(text):
        return False
    if text in (".", "..") or text.startswith("~"):
        return True
    return "/" in text or os.sep in text or _shell_path(text).is_absolute()


def _git_destination_positionals(subcmd: str, args: list[str]) -> list[str]:
    """Operands of init/clone with known split-form option VALUES consumed."""
    positionals: list[str] = []
    skip = False
    for arg in args:
        text = str(arg)
        if skip:
            skip = False
            continue
        if text.startswith("-"):
            skip = text in _GIT_DESTINATION_VALUE_FLAGS
            continue
        positionals.append(text)
    return positionals


def _git_explicit_destination(subcmd: str, args: list[str]) -> str:
    """The destination path of `init`/`clone`, or "" when the cwd is the target.

    Judged STRUCTURALLY, not by positional index arithmetic over a flag table that
    knows nothing about these two subcommands: the destination is the LAST operand,
    and only when it looks like a path. A leftover option value (`-b main`), a bare
    directory name (`git init proj`, which lands INSIDE the cwd) and `git clone
    <src>` with no destination all answer "" — the cwd is then the target and the
    working-directory containment check runs, which is the conservative answer."""
    if subcmd not in _GIT_DESTINATION_SUBCOMMANDS:
        return ""
    positionals = _git_destination_positionals(subcmd, args)
    if not positionals or not _git_path_shaped(positionals[-1]):
        return ""
    if subcmd == "init":
        return positionals[-1]
    # clone: the first URL-or-path operand is the SOURCE. A destination exists only
    # when an operand FOLLOWS it — `git clone /tmp/src` clones into <cwd>/src.
    for idx, token in enumerate(positionals):
        if _git_remote_url(token) or _git_path_shaped(token):
            return positionals[-1] if idx < len(positionals) - 1 else ""
    return ""


def is_readonly_git_command(raw_cmd: Any) -> bool:
    """True when EVERY segment of ``raw_cmd`` is a READ-ONLY git invocation.

    The exemption key for the external-workspace runtime-read guard: read-only git
    (status/log/diff/show/rev-parse/branch- and tag-listing) is allowed in every
    lane INCLUDING at a runtime target, which is the owner contract and the
    recorded false-block class (f14baf8f). Deliberately ALL-or-nothing per segment,
    so a compound command that merely starts with git — `git status && cat
    <data>/settings.json` — is NOT exempt and still meets the runtime-read guard.
    Network-reaching read-only git (``ls-remote``) is unaffected here: the network
    fence is enforced by ``external_workspace_git_violation``, which runs first.

    Two flag families are NOT read-only however read-only their subcommand looks,
    and both were measured against real git: ``--output=<file>`` (log/show/diff)
    TRUNCATES and writes <file>, and ``--no-index`` (diff/grep) prints ANY file on
    the host — `git diff --no-index /dev/null <data>/settings.json` dumps the
    credentials. Neither may ride this exemption; both then meet the ordinary
    runtime/secret guard, which refuses them only when they name a protected path.

    ``cd``/``pushd``/``popd`` count as neutral: they read and write nothing, so `cd
    <repo> && git log` is the same read-only inspection as `git -C <repo> log` and
    must not be refused for spelling it differently. Disclosed residual: git reached
    through a transparent wrapper (``nice``/``xargs``) or from inside interpreter
    code is not recognised as a git segment, so such a command still meets the full
    guard and is refused when it names a runtime path — deliberately not chased with
    a shell parser arms race (BIBLE P5)."""
    segments = shell_segments(raw_cmd)
    if not segments:
        return False
    saw_git = False
    for segment in segments:
        if not segment:
            continue
        _env, command = collect_leading_env(segment)
        if not command:
            return False
        name = pathlib.PurePath(str(command[0]).strip("`'\"")).name.lower()
        if name in {"cd", "pushd", "popd"}:
            continue
        if name in {"bash", "sh", "zsh"}:
            inline = shell_command_string(command)
            if not inline or not is_readonly_git_command(inline):
                return False
            saw_git = True
            continue
        if name != "git":
            return False
        # allow_network=True so this answers ONLY "does this invocation mutate?".
        if _git_invocation_block_reason(command, allow_network=True):
            return False
        if _git_reads_arbitrary_files(_git_subcommand_and_args(command)[1]):
            return False
        saw_git = True
    return saw_git


def run_shell_git_block_reason(raw_cmd: Any, *, allow_network: bool = True) -> str:
    argv = strip_leading_env_assignments(unwrap_env_argv(shell_argv(raw_cmd)))
    if not argv:
        return ""
    first = pathlib.PurePath(argv[0]).name.lower()
    if first in {"bash", "sh", "zsh"}:
        inline = shell_command_string(argv)
        return run_shell_git_block_reason(inline, allow_network=allow_network) if inline else ""
    for idx, token in enumerate(argv):
        if pathlib.PurePath(str(token)).name.lower() == "git":
            reason = _git_invocation_block_reason(argv[idx:], allow_network=allow_network)
            if reason:
                return reason
    return ""


def _resolve_workspace_shell_cwd(active_root: pathlib.Path, cwd: str = "") -> pathlib.Path:
    root = pathlib.Path(active_root).resolve(strict=False)
    if cwd and str(cwd).strip() not in ("", ".", "./"):
        raw = pathlib.Path(str(cwd)).expanduser()
        return raw.resolve(strict=False) if _rooted(raw) else (root / safe_relpath(str(cwd))).resolve(strict=False)
    return root


def _casefold_relative_to(target: pathlib.Path, root: pathlib.Path) -> bool:
    """Symlink-resolved, case-insensitive "target is under root" (prefix compare
    on resolved parts). Casefold is unconditional — the same conservative trade
    the admission/user_files guards make (``tool_access.paths_overlap_casefold``):
    APFS/NTFS are case-insensitive, so ``/users/anton/ouroboros/REPO`` and the
    real repo are ONE directory there, and a case-sensitive compare is a bypass."""
    try:
        target_parts = pathlib.Path(target).resolve(strict=False).parts
        root_parts = pathlib.Path(root).resolve(strict=False).parts
    except (OSError, ValueError):
        return False
    if len(target_parts) < len(root_parts):
        return False
    return tuple(part.casefold() for part in target_parts[: len(root_parts)]) == tuple(
        part.casefold() for part in root_parts
    )


def _overlaps_protected(target: pathlib.Path, root: pathlib.Path) -> bool:
    """BIDIRECTIONAL containment: the target inside the protected root, OR the
    protected root inside the target. One direction alone lets an ANCESTOR
    target through — ``git -C ~/Ouroboros init`` puts repo/ and data/ inside a
    task-created work tree even though the target contains (rather than is
    contained by) every protected root."""
    return _casefold_relative_to(target, root) or _casefold_relative_to(root, target)


def _shell_path(text: str) -> pathlib.Path:
    return pathlib.Path(os.path.expandvars(str(text or ""))).expanduser()


def _rooted(path: pathlib.Path) -> bool:
    """True for any path the shell/git resolves from a filesystem ROOT, not the cwd.

    On Windows ``Path('/Users/x').is_absolute()`` is False (root, no drive) while
    git and cmd resolve it from the CURRENT DRIVE's root — joining it under the
    workspace base instead silently un-protects a runtime target spelled
    POSIX-style (the windows temp-under-$HOME confinement class). Anchor presence
    is the honest cross-OS shape test; on POSIX it equals ``is_absolute()``."""
    return path.is_absolute() or bool(path.anchor)


def _resolve_shell_arg(value: str, base_dir: pathlib.Path) -> pathlib.Path:
    target = _shell_path(value)
    if not _rooted(target):
        target = base_dir / target
    return target.resolve(strict=False)


def _git_effective_base(invocation: list[str], base: pathlib.Path) -> pathlib.Path:
    """The directory git actually works in: global `-C <path>` flags (before the
    subcommand) CHDIR sequentially BEFORE git does anything, so the EFFECTIVE
    working directory — not the shell cwd — is what an invocation targets. Chained
    so `git -C /tmp/proj commit` from a runtime cwd (the default lane's default cwd
    IS the system repo) stays allowed, while `git -C <runtime> commit` from anywhere
    hits the same containment check. `-C` AFTER the subcommand is subcommand syntax
    (`git commit -C <commit>` reuses a message) and is not a path."""
    effective_base = base
    idx = 1
    while idx < len(invocation):
        part = str(invocation[idx])
        if part == "-C" and idx + 1 < len(invocation):
            effective_base = _resolve_shell_arg(str(invocation[idx + 1]), effective_base)
            idx += 2
            continue
        if part.startswith("-C") and len(part) > 2:
            # git accepts the GLUED spelling `-C<path>` too; parsing only the
            # split form judged the wrong base for it (#447 A7).
            effective_base = _resolve_shell_arg(part[2:], effective_base)
            idx += 1
            continue
        if part.startswith("-"):
            idx += 2 if part in ("-c", "--git-dir", "--work-tree") else 1
            continue
        break  # the subcommand
    return effective_base


def external_workspace_git_violation(
    raw_cmd: Any,
    *,
    active_root: pathlib.Path,
    cwd: str = "",
    protected_roots: list[pathlib.Path] | None = None,
    allow_network: bool = True,
    inherited_env: "dict[str, str] | None" = None,
) -> str:
    """Target-aware git policy: full git is legitimate task work.

    Tasks routinely need `git clone`, `git checkout`, `git commit`, even a real
    `git push` to a task-local remote. Since the Q4=A unwind (2026-08-08) this is
    the ONE resolver for external workspaces AND the default (non-workspace)
    shell lane — direct chat, light mode, self_modification-profile tasks — so
    the deterministic guard only protects what actually needs protecting:

    - no MUTATING git invocation may target the Ouroboros runtime (system repo
      or any data drive) via cwd, `-C`, `--git-dir`, `--work-tree`, `GIT_DIR`/
      `GIT_WORK_TREE` env, or a path argument. Containment is bidirectional
      (an ancestor target such as ``git -C ~/Ouroboros init`` is a violation
      too), casefold, and symlink-resolved;
    - READ-ONLY git (status/log/diff/show/rev-parse/branch- and tag-listing)
      stays allowed EVERYWHERE, including at a runtime target — blocking it is
      the recorded false-block class (v4.5.1, f14baf8f);
    - network-reaching subcommands respect ``allowed_resources.network`` in
      every lane, read-only or not.

    Everything else stays allowed here; the LLM safety layer still reviews the
    command for genuinely dangerous intent. Disclosed residuals (deliberately
    NOT chased — no shell-parser arms race): git launched through a transparent
    wrapper (``nice``/``xargs``/``time``) or from inside interpreter code is not
    a per-segment ``git`` command and is not classified here (the pre-flip text
    classifier never saw the interpreter form either); `-c alias.*=...`/
    ``include.path`` config indirection is not parsed — only the explicit
    cwd/flag/env/argument target vectors above are resolved.
    """
    roots = [pathlib.Path(p) for p in (protected_roots or [])]
    base = _resolve_workspace_shell_cwd(pathlib.Path(active_root), cwd)
    segments = shell_segments(raw_cmd)
    if not segments:
        return ""

    def _protected_label(target: pathlib.Path) -> str:
        for root in roots:
            if _overlaps_protected(target, root):
                return str(root)
        return ""

    _resolve = _resolve_shell_arg
    current_base = base
    dir_stack: list[pathlib.Path] = []
    # GIT_DIR/GIT_WORK_TREE exported in EARLIER segments (or inherited from an
    # enclosing shell that carried them into this `sh -c ...`).
    session_env: dict[str, str] = {
        k: v for k, v in (inherited_env or {}).items() if k in ("GIT_DIR", "GIT_WORK_TREE")
    }
    for segment in segments:
        if not segment:
            continue
        # Peel leading env assignments (VAR=val / env VAR=val) FIRST so a prefix
        # like `GIT_DIR=x bash -c '...'` is captured before the shell recursion.
        env_assigns, command = collect_leading_env(segment)
        if not command:
            # A pure-assignment segment (`GIT_DIR=... ` alone) exports into the
            # shell session and applies to LATER git segments. Carry it forward.
            for var in ("GIT_DIR", "GIT_WORK_TREE"):
                if var in env_assigns:
                    session_env[var] = env_assigns[var]
            continue
        cmd_name = pathlib.PurePath(str(command[0]).strip("`'\"")).name.lower()
        # Recurse into nested shells (sh -c "..."), carrying any GIT_DIR/
        # GIT_WORK_TREE env (segment-local + session) INTO the nested inspection
        # so `GIT_DIR=<runtime> bash -c 'git reset'` cannot retarget the repo.
        if cmd_name in {"bash", "sh", "zsh"}:
            inline = shell_command_string(command)
            if inline:
                nested = external_workspace_git_violation(
                    inline,
                    active_root=active_root,
                    cwd=str(current_base),
                    protected_roots=roots,
                    allow_network=allow_network,
                    inherited_env={**session_env, **env_assigns},
                )
                if nested:
                    return nested
            continue
        if cmd_name in {"export", "declare", "typeset"}:
            # `export GIT_DIR=...`, and bash `declare -x` / zsh-ksh `typeset -x`,
            # all export into the environment git honours. Capture GIT_* either
            # way (flags like -x are ignored; only VAR=val tokens are read).
            for token in command[1:]:
                text = str(token)
                if "=" in text and not text.startswith(("-", "=")):
                    key, _, value = text.partition("=")
                    if key in ("GIT_DIR", "GIT_WORK_TREE"):
                        session_env[key] = value
            continue
        # `pushd` chdirs exactly like `cd` (and remembers where it came from), so a
        # walker that tracks only `cd` judges a later git segment against the
        # ORIGINAL base while the shell has really moved into the runtime. `pushd`
        # with a rotation/flag operand (`pushd +1`, `pushd -n`) leaves the base
        # alone — the conservative answer, and the disclosed residual.
        if cmd_name in {"cd", "pushd"} and len(command) >= 2 and not str(command[1]).startswith(("-", "+")):
            if cmd_name == "pushd":
                dir_stack.append(current_base)
            target = _shell_path(str(command[1]))
            current_base = target if _rooted(target) else (current_base / target)
            current_base = current_base.resolve(strict=False)
            continue
        if cmd_name == "popd" and dir_stack:
            current_base = dir_stack.pop()
            continue
        if cmd_name != "git":
            continue
        invocation = command
        _subcmd, _sub_args = _git_subcommand_and_args(invocation)
        _output_files = _git_output_file_args(_sub_args)
        # READ-ONLY git is never target-checked: `git status`/`log`/`diff` at the
        # system repo is the vcs_status-equivalent inspection lane, and blocking
        # it is the recorded false-block class (v4.5.1; f14baf8f). Classification
        # deliberately probes with allow_network=True so it answers ONLY "does
        # this invocation mutate?" — the contract's network fence is enforced
        # separately at the tail, for read-only and mutating git alike.
        if _output_files and _git_subcommand_is_readonly(_subcmd, _sub_args):
            # `git log|show|diff --output=<file>` READS the repository (allowed in
            # every lane, at a runtime target too) and WRITES <file>. The mutation
            # lands at the FILE, exactly like init/clone's destination — so judge
            # the file and never the cwd, or `git log --output=/tmp/x` from the
            # default lane's system-repo cwd becomes a false block.
            for value in _output_files:
                if _protected_label(_resolve(value, _git_effective_base(invocation, current_base))):
                    return "git invocation writes into the Ouroboros runtime"
        elif _git_invocation_block_reason(invocation, allow_network=True):
            effective_base = _git_effective_base(invocation, current_base)
            # `git init <dir>` / `git clone <url> <dir>` mutate the DESTINATION, not
            # the working directory, so for those the destination — never the cwd —
            # is what containment must judge. Without this, the default lane (whose
            # default cwd IS the system repo) refused `git init ~/projects/foo`,
            # contradicting the owner contract that mutating git is free in every
            # lane and mode OUTSIDE the runtime roots.
            destination = _git_explicit_destination(_subcmd, _sub_args)
            if destination:
                if _protected_label(_resolve(destination, effective_base)):
                    return "git invocation targets the Ouroboros runtime"
            else:
                for root in roots:
                    if _overlaps_protected(effective_base, root):
                        return "git working directory targets the Ouroboros runtime"
            # Git is legitimate task work in host scratch (a repo under /tmp, a
            # /build tree, a sibling checkout), so the cwd is NOT confined to the
            # declared active workspace — only the Ouroboros runtime roots above
            # are protected (per this function's contract).
            # GIT_DIR / GIT_WORK_TREE environment retargeting (this segment runs
            # git). Merge env exported in earlier segments; segment-local wins.
            # Relative values resolve against the post-`-C` effective base — the
            # cwd git actually sees when it reads the environment.
            effective_env = {**session_env, **env_assigns}
            for var in ("GIT_DIR", "GIT_WORK_TREE"):
                val = effective_env.get(var)
                if not val:
                    continue
                target = _resolve(val, effective_base)
                if _protected_label(target):
                    return f"git invocation targets the Ouroboros runtime via {var}"
            j = 1
            while j < len(invocation):
                part = str(invocation[j])
                value = ""
                if part in {"--git-dir", "--work-tree"} and j + 1 < len(invocation):
                    value = str(invocation[j + 1])
                    j += 2
                elif part.startswith("--git-dir=") or part.startswith("--work-tree="):
                    value = part.split("=", 1)[1]
                    j += 1
                elif part in {"-c", "-C"}:
                    j += 2
                else:
                    j += 1
                if not value:
                    continue
                target = _resolve(value, effective_base)
                if _protected_label(target):
                    return "git invocation targets the Ouroboros runtime"
            # When the cwd check was skipped (init/clone with a destination) the base
            # may itself BE a runtime root, so resolving every bare token under it
            # would refuse the whole invocation — the subcommand word, a branch name
            # and the remote URL all resolve "inside the runtime" there. Skip the
            # tokens that do not NAME a path (flags, URLs, `blob:none`, a bare branch
            # name) instead of skipping every relative one: dropping relatives left
            # the mirror hole, where a relative destination pointing back INTO the
            # runtime (`git clone --depth 1 <url> repo/newtree` from ~/Ouroboros) was
            # never resolved. Every other invocation keeps the full argument scan.
            skip_value = False
            pending_path_value = False
            for arg in invocation[1:]:
                text = str(arg)
                if skip_value:
                    skip_value = False
                    continue
                is_path_value = pending_path_value
                pending_path_value = False
                if not is_path_value and destination and text.startswith("-"):
                    # Option VALUES are classified by the FLAG's documented type,
                    # not by the token's shape: `-b feature/x` is a ref name to
                    # git (resolving it under a runtime base is the slash-branch
                    # false block), while `--separate-git-dir repo` is a PATH to
                    # git even as a bare name (the shape test would let it slip
                    # the scan from the runtime's parent directory).
                    flag = text.split("=", 1)[0]
                    if flag in _GIT_DESTINATION_NONPATH_VALUE_FLAGS:
                        skip_value = "=" not in text
                        continue
                    if flag in _GIT_DESTINATION_VALUE_FLAGS:
                        if "=" in text:
                            text = text.split("=", 1)[1]
                            is_path_value = True
                        else:
                            pending_path_value = True
                            continue
                if "=" in text and text.startswith("--"):
                    text = text.split("=", 1)[1]
                if destination and not is_path_value and not _git_path_shaped(text):
                    continue
                candidate = _shell_path(text)
                if not _rooted(candidate):
                    # Relative args resolve under the effective base and are ALWAYS
                    # canonicalized. The former ".."-only shortcut assumed a plain
                    # descend cannot reach a protected root — untrue through a
                    # SYMLINK (`ln -s <runtime> ./p && git -C /tmp/x init ./p`),
                    # which resolve() follows. This branch only runs once the base
                    # itself passed containment, so a plain descend still resolves
                    # outside the runtime and nothing new is refused.
                    candidate = (effective_base / candidate).resolve(strict=False)
                if _protected_label(candidate):
                    return "git invocation targets the Ouroboros runtime"
        if not allow_network:
            subcmd, _ = _git_subcommand_and_args(invocation)
            if subcmd in GIT_NETWORK_SUBCOMMANDS:
                return f"task_contract.allowed_resources.network=false blocks git {subcmd}"
    return ""


def workspace_git_safety_violation(
    raw_cmd: Any,
    *,
    active_root: pathlib.Path,
    cwd: str = "",
    allow_network: bool = True,
) -> str:
    root = pathlib.Path(active_root).resolve(strict=False)
    base = _resolve_workspace_shell_cwd(root, cwd)
    try:
        base.relative_to(root)
        base_inside_root = True
    except Exception:
        base_inside_root = False
    argv = strip_leading_env_assignments(unwrap_env_argv(shell_argv(raw_cmd)))
    if not argv:
        return ""
    first = pathlib.PurePath(argv[0]).name.lower()
    if first in {"bash", "sh", "zsh"}:
        inline = shell_command_string(argv)
        return workspace_git_safety_violation(
            inline,
            active_root=root,
            cwd=str(base) if inline else "",
            allow_network=allow_network,
        ) if inline else ""
    for idx, token in enumerate(argv):
        if pathlib.PurePath(str(token)).name.lower() != "git":
            continue
        parts = argv[idx:]
        saw_root_selector = False
        j = 1
        while j < len(parts):
            part = parts[j]
            if part in {"-C", "--git-dir", "--work-tree"} and j + 1 < len(parts):
                saw_root_selector = True
                try:
                    target = pathlib.Path(parts[j + 1])
                    if not _rooted(target):
                        target = base / target
                    target.resolve(strict=False).relative_to(root)
                except Exception:
                    return f"git {part} escapes the active workspace"
                j += 2
                continue
            if (
                part.startswith("--git-dir=")
                or part.startswith("--work-tree=")
                or (part.startswith("-C") and len(part) > 2 and not part.startswith("--"))
            ):
                saw_root_selector = True
                value = part[2:] if part.startswith("-C") else part.split("=", 1)[1]
                try:
                    target = pathlib.Path(value)
                    if not _rooted(target):
                        target = base / target
                    target.resolve(strict=False).relative_to(root)
                except Exception:
                    return "git root selector escapes the active workspace"
                j += 1
                continue
            if part == "-c":
                j += 2
                continue
            if part.startswith("-"):
                j += 1
                continue
            break
        if not base_inside_root and not saw_root_selector:
            return "git cwd escapes the active workspace"
        reason = _git_invocation_block_reason(parts, allow_network=allow_network)
        if reason:
            return reason
    return ""
