"""Task-contract protected artifact enforcement helpers."""

from __future__ import annotations

import pathlib
from typing import Any, Dict, Iterable, List

from ouroboros.shell_parse import (
    is_absolute_path_text,
    shell_argv,
    shell_command_string,
    slash_normalize_path_text,
    strip_leading_env_assignments,
    unwrap_env_argv,
)
from ouroboros.tool_access import ResolvedResourceBinding, resolve_shell_cwd
from ouroboros.tools.shell_guards import interpreter_family
from ouroboros.workspace_executor import executor_ref_from_ctx, map_backend_path, map_host_path

_DEFAULT_DENIED_OPERATIONS = frozenset({
    "read_bytes",
    "copy",
    "hash",
    "static_introspection",
    "dynamic_trace",
    "debug",
    "write",
    "delete",
})
_SHELLS = frozenset({"bash", "cmd", "powershell", "pwsh", "sh", "zsh"})
_HIGH_RISK_INTERPRETERS = frozenset({
    "bash", "sh", "zsh", "python", "python3", "pythonw", "pypy", "pypy3",
    "node", "ruby", "perl", "php",
})
_SHELL_COMMAND_OPERATIONS = {
    "cat": "read_bytes",
    "head": "read_bytes",
    "tail": "read_bytes",
    "less": "read_bytes",
    "more": "read_bytes",
    "grep": "static_introspection",
    "egrep": "static_introspection",
    "fgrep": "static_introspection",
    "rg": "static_introspection",
    "ripgrep": "static_introspection",
    "ag": "static_introspection",
    "ack": "static_introspection",
    "sed": "static_introspection",
    "awk": "static_introspection",
    "diff": "static_introspection",
    "cmp": "static_introspection",
    "file": "static_introspection",
    "strings": "static_introspection",
    "hexdump": "static_introspection",
    "xxd": "static_introspection",
    "objdump": "static_introspection",
    "readelf": "static_introspection",
    "nm": "static_introspection",
    "otool": "static_introspection",
    "cp": "copy",
    "copy": "copy",
    "dd": "copy",
    "rsync": "copy",
    "tar": "copy",
    "zip": "copy",
    "type": "read_bytes",
    "xcopy": "copy",
    "robocopy": "copy",
    "certutil": "hash",
    "get-content": "read_bytes",
    "gc": "read_bytes",
    "select-string": "static_introspection",
    "copy-item": "copy",
    "get-filehash": "hash",
    "del": "delete",
    "erase": "delete",
    "mv": "delete",
    "move": "delete",
    "rd": "delete",
    "ren": "delete",
    "rename": "delete",
    "rename-item": "delete",
    "remove-item": "delete",
    "ri": "delete",
    "rm": "delete",
    "rmdir": "delete",
    "unlink": "delete",
    "shred": "delete",
    "tee": "write",
    "truncate": "write",
    "sha256sum": "hash",
    "shasum": "hash",
    "md5sum": "hash",
    "strace": "dynamic_trace",
    "ltrace": "dynamic_trace",
    "dtruss": "dynamic_trace",
    "gdb": "debug",
    "lldb": "debug",
}
_CMD_INLINE_SWITCHES = frozenset({"/c", "/k"})
_POWERSHELL_INLINE_SWITCHES = frozenset({"-c", "-command", "/c"})
_POWERSHELL_ENCODED_SWITCHES = frozenset({"-encodedcommand", "-enc", "-e"})
_GIT_STATIC_INTROSPECTION_SUBCOMMANDS = frozenset({
    "blame",
    "annotate",
    "cat-file",
    "diff",
    "grep",
    "show",
})
# Flags that make `git diff` emit raw/converted CONTENT of binary blobs; a plain
# text diff prints "Binary files differ" for them, never the bytes.
_GIT_DIFF_CONTENT_FLAGS = frozenset({"--binary", "--ext-diff", "--textconv"})
# Output-limiting flags under which `git diff` emits only file NAMES / stat
# counts, never file content — safe regardless of binary-vs-text (the vcs_diff
# FP shape: `git diff --stat` / `vcs_diff(stat=true|name_only=true)`).
_GIT_DIFF_CONTENT_FREE_FLAGS = frozenset({
    "--stat", "--numstat", "--shortstat", "--dirstat",
    "--name-only", "--name-status", "--summary", "--compact-summary",
})
_GIT_PATCH_LOG_FLAGS = frozenset({"-p", "-u", "--patch", "--patch-with-stat", "--stat-with-summary"})
_GIT_GLOBAL_OPTIONS_WITH_VALUE = frozenset({
    "-C",
    "-c",
    "--config-env",
    "--exec-path",
    "--git-dir",
    "--namespace",
    "--super-prefix",
    "--work-tree",
})
_DIRECTORY_TARGET_OPERATIONS = frozenset({"copy", "delete", "read_bytes", "static_introspection", "write"})
_SHELL_GLOB_CHARS = frozenset("*?[")
_FIND_EXPRESSION_MARKERS = frozenset({"!", "(", ")"})


def _task_contract(ctx: Any) -> Dict[str, Any]:
    metadata = getattr(ctx, "task_metadata", {}) if isinstance(getattr(ctx, "task_metadata", {}), dict) else {}
    contract = metadata.get("task_contract") if isinstance(metadata.get("task_contract"), dict) else {}
    if not contract and isinstance(getattr(ctx, "task_contract", None), dict):
        contract = getattr(ctx, "task_contract")
    return dict(contract) if isinstance(contract, dict) else {}


def _artifact_records(ctx: Any) -> List[Dict[str, Any]]:
    policy = _task_contract(ctx).get("resource_policy")
    if not isinstance(policy, dict):
        return []
    records = policy.get("protected_artifacts")
    return [dict(item) for item in records if isinstance(item, dict)] if isinstance(records, list) else []


def _base_roots(
    ctx: Any,
    binding: ResolvedResourceBinding | None = None,
) -> List[pathlib.Path]:
    roots: List[pathlib.Path] = []
    values = (
        binding.base_path if binding is not None else None,
        getattr(ctx, "workspace_root", None),
        getattr(ctx, "repo_dir", None),
        getattr(ctx, "system_repo_dir", None),
        getattr(ctx, "serving_repo_dir", None),
        getattr(ctx, "drive_root", None),
    )
    for value in values:
        if value is None:
            continue
        try:
            path = pathlib.Path(value).expanduser().resolve(strict=False)
        except (OSError, TypeError, ValueError):
            continue
        if path not in roots:
            roots.append(path)
    return roots


def _resolve_policy_path(
    ctx: Any,
    raw_path: str,
    binding: ResolvedResourceBinding | None = None,
) -> pathlib.Path | None:
    text = str(raw_path or "").strip()
    if not text:
        return None
    try:
        path = pathlib.Path(text).expanduser()
    except (OSError, TypeError, ValueError):
        return None
    if path.is_absolute():
        return path.resolve(strict=False)
    roots = _base_roots(ctx, binding)
    if not roots:
        return path.resolve(strict=False)
    return (roots[0] / path).resolve(strict=False)


def _backend_spellings_for_host_path(ctx: Any, path: pathlib.Path) -> set[str]:
    try:
        executor = executor_ref_from_ctx(ctx)
        if executor is None:
            return set()
        backend = map_host_path(executor, pathlib.Path(path))
    except Exception:
        return set()
    return {backend, backend.rstrip("/")}


def _policy_backend_spellings(ctx: Any, raw_path: str, resolved: pathlib.Path | None) -> set[str]:
    spellings: set[str] = set()
    text = str(raw_path or "").strip()
    if text:
        spellings.add(slash_normalize_path_text(text).rstrip("/"))
    if resolved is not None:
        spellings.update(_backend_spellings_for_host_path(ctx, resolved))
    return {item for item in spellings if item}




def protected_artifact_paths(
    ctx: Any,
    binding: ResolvedResourceBinding | None = None,
) -> List[pathlib.Path]:
    paths: List[pathlib.Path] = []
    for record in _artifact_records(ctx):
        for raw_path in record.get("paths") or []:
            text = str(raw_path)
            try:
                executor_ref = executor_ref_from_ctx(ctx)
                if executor_ref is not None and text.strip().startswith("/"):
                    mapped = map_backend_path(executor_ref, text)
                    if mapped not in paths:
                        paths.append(mapped)
            except Exception:
                pass
            resolved = _resolve_policy_path(ctx, str(raw_path), binding)
            if resolved is not None and resolved not in paths:
                paths.append(resolved)
    return paths


def _operation_denied(record: Dict[str, Any], operation: str) -> bool:
    allow = {str(item).strip() for item in (record.get("allow") or []) if str(item).strip()}
    if allow:
        return operation not in allow
    deny = {str(item).strip() for item in (record.get("deny") or []) if str(item).strip()}
    if deny:
        return operation in deny
    return str(record.get("role") or "") == "black_box_reference" and operation in _DEFAULT_DENIED_OPERATIONS


def _matches(candidate: pathlib.Path, protected_path: pathlib.Path) -> bool:
    try:
        candidate_resolved = pathlib.Path(candidate).expanduser().resolve(strict=False)
        protected_resolved = pathlib.Path(protected_path).expanduser().resolve(strict=False)
    except (OSError, TypeError, ValueError):
        return False
    if candidate_resolved == protected_resolved:
        return True
    if protected_resolved.is_dir():
        try:
            candidate_resolved.relative_to(protected_resolved)
            return True
        except ValueError:
            return False
    return False


def _backend_spelling_matches(candidate: pathlib.Path, protected_spellings: set[str]) -> bool:
    try:
        raw = str(candidate)
    except Exception:
        return False
    normalized = slash_normalize_path_text(raw).rstrip("/")
    if not normalized:
        return False
    variants = {normalized}
    if not normalized.startswith("/"):
        variants.add(f"/{normalized}")
    for protected in protected_spellings:
        if not protected:
            continue
        protected_norm = slash_normalize_path_text(protected).rstrip("/")
        protected_variants = {protected_norm}
        if not protected_norm.startswith("/"):
            protected_variants.add(f"/{protected_norm}")
        if variants & protected_variants:
            return True
    return False


def block_reason_for_path(
    ctx: Any,
    target: pathlib.Path,
    operation: str,
    binding: ResolvedResourceBinding | None = None,
) -> str:
    for record in _artifact_records(ctx):
        if not _operation_denied(record, operation):
            continue
        for raw_path in record.get("paths") or []:
            protected_path = _resolve_policy_path(ctx, str(raw_path), binding)
            protected_spellings = _policy_backend_spellings(ctx, str(raw_path), protected_path)
            target_backend_spellings = _backend_spellings_for_host_path(ctx, pathlib.Path(target))
            if (
                protected_path is not None
                and _matches(pathlib.Path(target), protected_path)
                or _backend_spelling_matches(pathlib.Path(target), protected_spellings)
                or bool(protected_spellings & target_backend_spellings)
            ):
                artifact_id = str(record.get("id") or pathlib.Path(str(raw_path)).name or "protected artifact")
                return (
                    "⚠️ RESOURCE_POLICY_BLOCKED: task_contract.resource_policy protects "
                    f"{artifact_id!r}; operation {operation!r} is not allowed for this black-box artifact. "
                    + _nearest_allowed_action(record)
                )
    return ""


def _nearest_allowed_action(record: Dict[str, Any]) -> str:
    """v6.57.0 (1.6): a refusal that names what IS allowed so the agent redirects to
    the legal action in one move instead of guessing. For an execute-allowed black-box
    reference the nearest allowed action is: run it with any arguments and observe its
    OWN stdout/stderr/exit code (redirect/pipe its output to a file you own). What stays
    blocked is deriving the artifact's BYTES: reading/copying/hashing it or static/dynamic
    introspection (the anti-cheat boundary)."""
    allow = {str(item).strip() for item in (record.get("allow") or []) if str(item).strip()}
    if not allow and str(record.get("role") or "") == "black_box_reference":
        allow = {"execute"}
    if "execute" in allow or not allow:
        return (
            "Allowed: EXECUTE it with any arguments and capture its OWN stdout/stderr/exit "
            "(e.g. `artifact ARGS > your_output.txt`, then read your_output.txt). Reading, "
            "copying, hashing, or introspecting the artifact's bytes stays blocked (anti-cheat)."
        )
    return f"Allowed operations for this artifact: {', '.join(sorted(allow))}."


def any_protected_target(
    ctx: Any,
    candidates: Iterable[pathlib.Path],
    operation: str,
    binding: ResolvedResourceBinding | None = None,
) -> str:
    for candidate in candidates:
        reason = block_reason_for_path(ctx, pathlib.Path(candidate), operation, binding)
        if reason:
            return reason
    return ""


def _directory_contains_protected_target(
    ctx: Any,
    candidates: Iterable[pathlib.Path],
    operation: str,
    binding: ResolvedResourceBinding | None = None,
) -> str:
    for candidate in candidates:
        try:
            candidate_resolved = pathlib.Path(candidate).expanduser().resolve(strict=False)
        except (OSError, TypeError, ValueError):
            continue
        if not candidate_resolved.is_dir():
            continue
        for record in _artifact_records(ctx):
            if not _operation_denied(record, operation):
                continue
            for raw_path in record.get("paths") or []:
                protected_paths: list[pathlib.Path] = []
                protected_path = _resolve_policy_path(ctx, str(raw_path), binding)
                if protected_path is not None:
                    protected_paths.append(pathlib.Path(protected_path))
                try:
                    executor_ref = executor_ref_from_ctx(ctx)
                    if executor_ref is not None and str(raw_path).strip().startswith("/"):
                        protected_paths.append(map_backend_path(executor_ref, str(raw_path)))
                except Exception:
                    pass
                for candidate_protected in protected_paths:
                    try:
                        candidate_protected.resolve(strict=False).relative_to(candidate_resolved)
                    except ValueError:
                        continue
                    except Exception:
                        continue
                    return block_reason_for_path(ctx, candidate_protected, operation, binding)
    return ""


def _resolve_candidate_path(ctx: Any, work_dir: pathlib.Path, text: str) -> pathlib.Path | None:
    try:
        path = pathlib.Path(text).expanduser()
        # is_absolute_path_text (not Path.is_absolute) so a backend path like
        # "/workspace/x" is recognized as absolute on Windows too (no drive
        # letter -> Path.is_absolute() is False there) and routed through
        # map_backend_path instead of being mis-joined onto work_dir.
        if is_absolute_path_text(text):
            try:
                executor_ref = executor_ref_from_ctx(ctx)
                return map_backend_path(executor_ref, text) if executor_ref is not None else path.resolve(strict=False)
            except Exception:
                return path.resolve(strict=False)
        return (pathlib.Path(work_dir) / path).resolve(strict=False)
    except (OSError, TypeError, ValueError):
        return None


def _contains_shell_glob(text: str) -> bool:
    return any(char in str(text or "") for char in _SHELL_GLOB_CHARS)


def _glob_base_candidate(ctx: Any, work_dir: pathlib.Path, text: str) -> pathlib.Path | None:
    normalized = str(text or "").replace("\\", "/")
    first_glob = min((idx for idx, char in enumerate(normalized) if char in _SHELL_GLOB_CHARS), default=-1)
    if first_glob < 0:
        return None
    prefix = normalized[:first_glob]
    if "/" in prefix:
        base_text = prefix.rsplit("/", 1)[0] or "/"
    else:
        base_text = "."
    return _resolve_candidate_path(ctx, work_dir, base_text)


def _glob_pattern_could_match_protected(
    ctx: Any,
    work_dir: pathlib.Path,
    glob_text: str,
    operation: str,
    binding: ResolvedResourceBinding | None = None,
) -> str:
    """v6.57.0 (1.6): a write/delete GLOB (e.g. `rm -f *.out`) blocks ONLY when the glob
    pattern could ACTUALLY match a protected artifact's basename in the glob's directory —
    not merely because the protected binary sits in the same directory (the differential-
    testing FP where `rm -f *.out` next to a black-box `ref` binary was blocked). fnmatch
    on the filename segment is the precise, general test (helps any task that cleans scratch
    files beside a protected reference). A recursive `**` conservatively still blocks (it can
    reach anything). Returns a block reason or ""."""
    import fnmatch

    base = _glob_base_candidate(ctx, work_dir, glob_text)
    if base is None:
        return ""
    normalized = str(glob_text or "").replace("\\", "/")
    filename_pattern = normalized.rsplit("/", 1)[-1] if "/" in normalized else normalized
    recursive = "**" in normalized
    try:
        base_resolved = pathlib.Path(base).expanduser().resolve(strict=False)
    except (OSError, TypeError, ValueError):
        return ""
    for protected in protected_artifact_paths(ctx, binding):
        try:
            protected_resolved = pathlib.Path(protected).resolve(strict=False)
        except (OSError, TypeError, ValueError):
            continue
        # Only consider a protected artifact reachable by this glob's directory.
        try:
            under_base = protected_resolved.parent == base_resolved or (
                recursive and protected_resolved.is_relative_to(base_resolved)
            )
        except AttributeError:  # pragma: no cover - py<3.9
            under_base = str(protected_resolved.parent) == str(base_resolved)
        if not under_base:
            continue
        if recursive or fnmatch.fnmatch(protected_resolved.name, filename_pattern):
            block = block_reason_for_path(ctx, protected_resolved, operation, binding)
            if block:
                return block
    return ""


def _inline_shell_command(argv: list[str], shell_name: str) -> str:
    if shell_name in {"bash", "sh", "zsh"}:
        return shell_command_string(argv)
    switches = _CMD_INLINE_SWITCHES if shell_name == "cmd" else _POWERSHELL_INLINE_SWITCHES
    for idx, arg in enumerate(argv[1:], start=1):
        if str(arg or "").strip().lower() in switches:
            return " ".join(str(part) for part in argv[idx + 1:])
    return ""




def _is_high_risk_interpreter(name: str) -> bool:
    # Versioned spellings classify through the ONE structural family classifier
    # (shell_guards.interpreter_family). The local recognizer this replaces knew
    # only versioned pythons (python3.11, pypy3.9m), so ruby3.2 / php8.3 /
    # perl5.38 / node18 slipped this guard exactly as they slipped the write
    # fences (XG-2R.2). The set keeps the shells, which are not a family.
    return name in _HIGH_RISK_INTERPRETERS or bool(interpreter_family(name))


_INTERPRETER_INLINE_CODE_FLAGS = frozenset({"-c", "-e", "-E", "-r", "--command"})
_INTERPRETER_MODULE_FLAGS = frozenset({"-m", "--module"})


def _interpreter_read_operands(argv: list[str]) -> list[str]:
    """The file operand(s) a high-risk interpreter itself OPENS. Two shapes:

    * script form (`python foo.py`): the first positional (the script path).
    * module form (`python -m pdb|py_compile|zipfile … <file>`): the module's
      positional file operands — a module like pdb/py_compile/zipfile/trace OPENS
      the file it is passed, so `python -m pdb ./ref` reads the artifact just as
      `python ./ref` does (Finding 3, v6.56.0: `-m` used to yield no operand and
      slip read/copy/debug of the protected file through).

    Inline-code / stdin forms (-c/-e/-r/--command/"-"/heredoc) OPEN nothing — a
    protected filename QUOTED inside their code TEXT stays covered by the mention
    + read-primitive proximity scan, not by this bare-token check (round-2 fix:
    tokenizing whole inline bodies as read-candidates blocked legitimate
    differential harnesses that merely quote the name — 22/48 smoke2 FPs)."""
    items = [str(item or "") for item in argv[1:]]
    i = 0
    while i < len(items):
        token = items[i]
        if not token:
            i += 1
            continue
        if token == "-" or token.startswith("<<"):
            return []
        if token in _INTERPRETER_MODULE_FLAGS:
            # Skip the module NAME (items[i+1]); screen its positional file args.
            return [tok for tok in items[i + 2:] if tok and not tok.startswith("-")]
        if token in _INTERPRETER_INLINE_CODE_FLAGS:
            return []
        if token.startswith("-"):
            i += 1
            continue
        return [token]
    return []


def _git_subcommand_index(argv: list[str]) -> int | None:
    idx = 1
    while idx < len(argv):
        token = str(argv[idx] or "")
        if token == "--":
            idx += 1
            continue
        if token == "-C" or token in _GIT_GLOBAL_OPTIONS_WITH_VALUE:
            idx += 2
            continue
        if any(token.startswith(option + "=") for option in _GIT_GLOBAL_OPTIONS_WITH_VALUE):
            idx += 1
            continue
        if token.startswith("-"):
            idx += 1
            continue
        return idx
    return None


def _git_static_introspection_operation(argv: list[str]) -> str | None:
    subcmd_idx = _git_subcommand_index(argv)
    if subcmd_idx is None:
        return None
    subcmd = pathlib.PurePath(argv[subcmd_idx]).name.lower().removesuffix(".exe")
    if subcmd in _GIT_STATIC_INTROSPECTION_SUBCOMMANDS:
        return "static_introspection"
    if subcmd == "log" and any(str(token or "") in _GIT_PATCH_LOG_FLAGS for token in argv[subcmd_idx + 1:]):
        return "static_introspection"
    return None


def _git_diff_can_dump_content(argv: list[str]) -> bool:
    """Round-2 structural gate (v6.56.0): does this `git diff` invocation risk
    dumping a protected binary's CONTENT (vs merely naming its diff status)?

    An output-limiting flag (`--stat`/`--name-only`/`--name-status`/…) makes the
    diff emit only file NAMES / stat counts — never content — regardless of
    binary-vs-text, so it cannot leak a protected artifact and the whole-work_dir
    fallback is skipped (the documented `git diff --stat` / `vcs_diff(stat=true)`
    false positive). Everything else — a BARE `git diff` (can dump a modified
    text file's content), a revision argument, a forced content flag
    (--binary/--ext-diff/--textconv), or a PATCH flag (`-p`/`-u`/`-U<n>`/`-W`/
    pickaxe) that re-enables hunk output ALONGSIDE `--stat` — can dump content
    and keeps the fallback. A pathspec that NAMES the protected file is blocked
    separately via the candidate tokens; this gate only decides the fallback."""
    subcmd_idx = _git_subcommand_index(argv)
    if subcmd_idx is None:
        return True
    if pathlib.PurePath(argv[subcmd_idx]).name.lower().removesuffix(".exe") != "diff":
        return True  # grep/show/blame/log -p always risk content
    rest = [str(item or "") for item in argv[subcmd_idx + 1:]]
    before_dashdash: list[str] = []
    for token in rest:
        if token == "--":
            break
        before_dashdash.append(token)
    has_content_free_flag = any(token in _GIT_DIFF_CONTENT_FREE_FLAGS for token in before_dashdash)
    has_content_flag = (
        any(token in _GIT_DIFF_CONTENT_FLAGS for token in rest)
        or _git_diff_has_patch_flag(before_dashdash)
    )
    # An output-limited diff with no content/patch flag cannot emit file bytes.
    return not (has_content_free_flag and not has_content_flag)


def _git_diff_has_patch_flag(tokens: list[str]) -> bool:
    """A `git diff` flag that emits patch hunks (file CONTENT) even when combined
    with an output-limiting `--stat`/`--name-only`: `-p`/`-u`/`--patch`, unified
    context (`-U<n>`/`--unified`), function context (`-W`/`--function-context`),
    and pickaxe (`-S`/`-G`, which print hunks by default in `git diff`)."""
    for token in tokens:
        if token in _GIT_PATCH_LOG_FLAGS or token in ("-W", "--function-context"):
            return True
        if token.startswith(("-U", "--unified", "-S", "-G")):
            return True
    return False


def _find_operation(argv: list[str]) -> str:
    args = [str(token or "") for token in argv[1:]]
    if "-delete" in args:
        return "delete"
    for idx, token in enumerate(args):
        if token not in {"-exec", "-execdir"} or idx + 1 >= len(args):
            continue
        executable = pathlib.PurePath(args[idx + 1]).name.lower().removesuffix(".exe")
        operation = _SHELL_COMMAND_OPERATIONS.get(executable)
        if operation:
            return operation
        if _is_high_risk_interpreter(executable):
            return "read_bytes"
        return "static_introspection"
    return "static_introspection"


def _git_work_dir(ctx: Any, argv: list[str], initial_work_dir: pathlib.Path) -> pathlib.Path:
    work_dir = pathlib.Path(initial_work_dir)
    idx = 1
    while idx < len(argv):
        token = str(argv[idx] or "")
        if token == "--":
            idx += 1
            continue
        if token == "-C" and idx + 1 < len(argv):
            resolved = _resolve_candidate_path(ctx, work_dir, str(argv[idx + 1] or ""))
            if resolved is not None:
                work_dir = resolved
            idx += 2
            continue
        if token.startswith("-C") and len(token) > 2:
            resolved = _resolve_candidate_path(ctx, work_dir, token[2:])
            if resolved is not None:
                work_dir = resolved
            idx += 1
            continue
        if token in _GIT_GLOBAL_OPTIONS_WITH_VALUE:
            idx += 2
            continue
        if any(token.startswith(option + "=") for option in _GIT_GLOBAL_OPTIONS_WITH_VALUE):
            idx += 1
            continue
        if token.startswith("-"):
            idx += 1
            continue
        break
    return work_dir


def _git_candidate_tokens(argv: list[str]) -> list[str]:
    subcmd_idx = _git_subcommand_index(argv)
    if subcmd_idx is None:
        return []
    tokens: list[str] = []
    rest = argv[subcmd_idx + 1:]
    for token in rest:
        text = str(token or "")
        if not text or text == "--":
            continue
        if text.startswith("-") and not pathlib.Path(text).is_absolute():
            continue
        tokens.append(text)
        if ":" not in text:
            continue
        # Git object syntax such as HEAD:path/to/file or :path/to/file can
        # read the protected bytes without naming a filesystem path directly.
        if len(text) >= 2 and text[1] == ":" and text[0].isalpha():
            continue
        rev_path = text.split(":", 1)[1].lstrip("./")
        if rev_path:
            tokens.append(rev_path)
    return tokens


def _git_static_introspection_is_path_limited(work_dir: pathlib.Path, candidates: list[pathlib.Path]) -> bool:
    for candidate in candidates:
        try:
            resolved = pathlib.Path(candidate).resolve(strict=False)
        except Exception:
            continue
        if resolved.exists():
            return True
    return False


def _file_operand_tokens(argv: list[str]) -> list[str]:
    """Bounded utility file roles; patterns, programs and option values are not files."""
    first = pathlib.PurePath(argv[0]).name.lower().removesuffix(".exe") if argv else ""
    if first == "find":
        files = []
        for token in argv[1:]:
            if not files and token in {"-H", "-L", "-P", "--"}:
                continue
            if token.startswith("-") or token in _FIND_EXPRESSION_MARKERS:
                break
            files.append(token)
        return files or ["."]
    if first == "dd":
        return [token[3:] for token in argv[1:] if token.startswith("if=")]
    patterned = first in {"grep", "egrep", "fgrep", "rg", "ripgrep", "ag", "ack", "sed", "awk"}
    if not patterned:
        return [token for token in argv[1:] if token and not token.startswith("-")]
    files, pattern_seen, index = [], False, 1
    while index < len(argv):
        token = argv[index]
        if token == "--":
            tail = argv[index + 1:]
            return files + (tail if pattern_seen else tail[1:])
        if token in {"-e", "--regexp", "--expression", "-f", "--file"}:
            if index + 1 >= len(argv):
                return files
            if token in {"-f", "--file"}:
                files.append(argv[index + 1])
            pattern_seen, index = True, index + 2
            continue
        if token.startswith(("--regexp=", "--expression=", "-e")):
            pattern_seen = True
        elif token.startswith(("--file=", "-f")):
            files.append(token.split("=", 1)[1] if token.startswith("--") else token[2:])
            pattern_seen = True
        elif (first == "awk" and token in {"-v", "-F"}) or token in {
            "-A", "-B", "-C", "-m", "-g", "--glob", "--iglob", "--include", "--exclude", "--exclude-dir",
        }:
            index += 2
            continue
        elif first == "sed" and token == "-i" and index + 1 < len(argv) and argv[index + 1] == "":
            index += 2
            continue
        elif token.startswith("-"):
            # Unknown long options may consume a value; do not invent a file role.
            if token.startswith("--") and "=" not in token and token not in {
                "--line-number", "--recursive", "--fixed-strings", "--ignore-case", "--quiet", "--in-place",
            }:
                return files
        elif not pattern_seen:
            pattern_seen = True
        elif first != "awk" or "=" not in token:
            files.append(token)
        index += 1
    return files


def _shell_operand_block(ctx, tokens, operation, work_dir, binding, shell_syntax=False) -> str:
    candidates = []
    for text in tokens:
        if not text or any(char in text for char in "$`~"):
            continue  # Unexpanded values are not concrete filesystem evidence.
        if shell_syntax and _contains_shell_glob(text):
            if reason := _glob_pattern_could_match_protected(ctx, work_dir, text, operation, binding):
                return reason
            continue
        candidate = _resolve_candidate_path(ctx, work_dir, text)
        if candidate is not None:
            candidates.append(candidate)
    if reason := any_protected_target(ctx, candidates, operation, binding):
        return reason
    if operation in _DIRECTORY_TARGET_OPERATIONS:
        return _directory_contains_protected_target(ctx, candidates, operation, binding)
    return ""


def shell_block_reason(
    ctx: Any, raw_cmd: Any, *, cwd: str = "", default_cwd: pathlib.Path | None = None,
    binding: ResolvedResourceBinding | None = None,
) -> str:
    from ouroboros.config import get_runtime_mode
    from ouroboros.runtime_mode_policy import mode_has_unrestricted_agency
    from ouroboros.shell_parse import sequential_effective_cwds, directory_destination_child_name
    from ouroboros.tools.shell_guards import direct_shell_rows, directory_destination_pairs, _writer_target_tokens_single

    if mode_has_unrestricted_agency(get_runtime_mode()) or not protected_artifact_paths(ctx, binding):
        return ""
    if binding is not None:
        work_dir = pathlib.Path(binding.target_path)
    else:
        try:
            work_dir, _root, _allowed = resolve_shell_cwd(ctx, cwd)
        except Exception:
            work_dir = pathlib.Path(default_cwd or ".").resolve(strict=False)
    rows = direct_shell_rows(raw_cmd)
    for (raw_argv, reads, writes, shell_syntax), row_cwd in zip(rows, sequential_effective_cwds(rows, work_dir)):
        argv = strip_leading_env_assignments(unwrap_env_argv(raw_argv))
        first = pathlib.PurePath(argv[0]).name.lower().removesuffix(".exe") if argv else ""
        if first in {"cmd", "powershell", "pwsh"} and (inline := _inline_shell_command(argv, first)):
            argv = shell_argv(inline)  # Retain the existing explicit Windows command-operand view.
        # Redirections are independent operations, including around allowed execution.
        write_targets = [*_writer_target_tokens_single(argv, direct_only=True, parse_redirects=False), *writes]
        pairs = directory_destination_pairs(argv)
        if pairs:
            write_targets = list(writes)
            for command, destination, source in pairs:
                target = _resolve_candidate_path(ctx, row_cwd, destination)
                child = directory_destination_child_name(command, argv, source)
                write_targets.append(str(target / child) if target is not None and target.is_dir() and child else destination)
        for tokens, operation in ((reads, "read_bytes"), (write_targets, "write")):
            if reason := _shell_operand_block(ctx, tokens, operation, row_cwd, binding, shell_syntax):
                return reason
        if not argv:
            continue
        first = pathlib.PurePath(argv[0]).name.lower().removesuffix(".exe")
        executable = _resolve_candidate_path(ctx, row_cwd, argv[0])
        if executable is not None and executable.exists() and (reason := block_reason_for_path(ctx, executable, "execute", binding)):
            return reason
        if first == "git":
            operation = _git_static_introspection_operation(argv)
            row_cwd = _git_work_dir(ctx, argv, row_cwd)
            tokens = _git_candidate_tokens(argv)
            candidates = [_resolve_candidate_path(ctx, row_cwd, token) for token in tokens]
            if operation and not _git_static_introspection_is_path_limited(row_cwd, [p for p in candidates if p is not None]) and _git_diff_can_dump_content(argv):
                tokens.append(str(row_cwd))
        elif _is_high_risk_interpreter(first):
            operation, tokens = "read_bytes", _interpreter_read_operands(argv)
        elif pairs:
            operation = "copy" if first == "cp" else "delete" if first == "mv" else None
            tokens = [source for _command, _destination, source in pairs]
        else:
            operation = _find_operation(argv) if first == "find" else _SHELL_COMMAND_OPERATIONS.get(first)
            tokens = _file_operand_tokens(argv) if operation else []
        if operation and (reason := _shell_operand_block(ctx, tokens, operation, row_cwd, binding, shell_syntax)):
            return reason
    return ""
