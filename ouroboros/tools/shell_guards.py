"""Shared shell guard helpers for process tools."""

from __future__ import annotations

import ast
import pathlib
import re
from typing import Any, Dict, List

from ouroboros.runtime_mode_policy import FROZEN_CONTRACT_PATH_PREFIXES, PROTECTED_RUNTIME_PATHS
from ouroboros.shell_parse import (
    EMBEDDED_WINDOWS_ABSOLUTE_PATH_RE,
    embedded_absolute_path_tokens,
    normalize_check_argv,
    shell_argv,
    shell_argv_with_inline,
    shell_command_string,
    strip_leading_env_assignments,
    unwrap_env_argv,
)

PROTECTED_RUNTIME_PATHS_LOWER = frozenset(
    p.lower() for p in PROTECTED_RUNTIME_PATHS
) | frozenset(prefix.lower() for prefix in FROZEN_CONTRACT_PATH_PREFIXES)

SHELL_WRITE_INDICATORS = (
    "rm ", "rm\t", ">", "sed -i", "tee ", "truncate",
    "mv ", "cp ", "chmod ", "chown ", "unlink ", "delete", "trash",
    "rsync ", "write_text", "open(", ".write(", ".writelines(",
    "os.remove(", "os.unlink(", "os.mkdir(", "os.makedirs(", "sort -o",
    "writefilesync", "appendfilesync", "createwritestream",
)
_SAFE_STDIO_REDIRECT_TOKENS = frozenset({
    ">/dev/null",
    "1>/dev/null",
    "2>/dev/null",
    "2>&1",
    "1>&2",
    "2>&-",
})

LIGHT_SHELL_WRITER_COMMANDS = frozenset({
    "chmod", "chown", "cp", "gunzip", "gzip", "ln", "mkdir", "mv",
    "perl", "rm", "ruby", "sed", "sort", "tar", "touch", "truncate", "uniq", "unzip",
})

INTERPRETER_WRITE_RE = re.compile(
    r"""(?is)(?:\.write\(|write_text\(|write_bytes\(|fs\.write|fs\.append|"""
    r"""createwritestream|unlink\(|rename\(|mkdir\(|rmtree\(|remove\(|"""
    r"""open\s*\([^)]*,\s*['"][^'"]*[wax+])"""
)
# Wider write-indicator net for the read-vs-write runtime_data scan (v6.54.3):
# includes filesystem-mutating calls the python AST analysis does NOT model
# (shutil.copy*/move, touch, symlink/link, chmod/chown, makedirs/removedirs,
# truncate) — a hit here without AST-resolved targets stays on the conservative
# full mention scan instead of being treated as a pure read. NB: the leading
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
    + r"""\.extractall\(|unpack_archive\(|make_archive\(|sqlite3\.connect\(|"""
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
    + r"""\.open\(\s*(?:mode\s*=\s*)?['"](?=[a-z+]{1,3}['"])[a-z+]*[wax+][a-z+]*['"])"""
)
EMBEDDED_RELATIVE_PATH_RE = re.compile(r"(?<![A-Za-z0-9_.-])(?:\.\.?/)+[^\s'\"\\),;\]]+")
_REDIRECT_TARGET_TOKENS = frozenset({">", ">>", "1>", "1>>", "2>", "2>>", "&>", "&>>"})
_SCRIPT_INTERPRETERS = frozenset({"python", "python3", "node", "ruby", "perl", "php"})
_SCRIPT_LITERAL_WRITE_RE = {
    "node": re.compile(
        r"""(?is)(?:fs\.|require\(['"]fs['"]\)\.)"""
        r"""(?:writeFileSync|appendFileSync|createWriteStream|mkdirSync|rmSync|rmdirSync|unlinkSync)\s*\(\s*(['"])(.*?)\1"""
    ),
    "ruby": re.compile(
        r"""(?is)(?:File\.write|File\.open|FileUtils\.(?:touch|mkdir_p|rm|rm_rf|remove|copy|cp|mv))\s*\(\s*(['"])(.*?)\1"""
    ),
}


def _pure_path_flavor(text: str):
    """Pure-path flavor matching the LITERAL's own shape, host-independent.

    A Windows-shaped literal (drive letter, UNC, or backslash-only separators)
    must derive parent/join with WINDOWS semantics on every host:
    PurePosixPath('C:\\\\x\\\\y').parent is '.', which turned a real write target
    into a cwd-shaped false-allow on the windows CI full-test (v6.55.0). POSIX
    shapes keep POSIX semantics everywhere, so POSIX behavior is unchanged."""
    if re.match(r"^[A-Za-z]:[\\/]", text) or text.startswith("\\\\") or ("\\" in text and "/" not in text):
        return pathlib.PureWindowsPath
    return pathlib.PurePosixPath


def _python_literal_path(node: ast.AST, names: dict[str, str]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return names.get(node.id)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Path" and node.args:
        return _python_literal_path(node.args[0], names)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "Path"
        and node.args
    ):
        return _python_literal_path(node.args[0], names)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "cwd":
        base = node.func.value
        if isinstance(base, ast.Name) and base.id in {"Path", "pathlib"}:
            return "."
        if isinstance(base, ast.Attribute) and base.attr == "Path":
            return "."
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "getcwd"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "os"
    ):
        return "."
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left = _python_literal_path(node.left, names)
        right = _python_literal_path(node.right, names)
        if left is not None and right is not None:
            return str(_pure_path_flavor(left)(left) / right)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _python_literal_path(node.left, names)
        right = _python_literal_path(node.right, names)
        if left is not None and right is not None:
            return left + right
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            else:
                return None
        return "".join(parts)
    if isinstance(node, ast.Attribute) and node.attr == "parent":
        base = _python_literal_path(node.value, names)
        if base is not None:
            return str(_pure_path_flavor(base)(base).parent)
    return None


def _python_write_mode_from_open_call(node: ast.Call) -> str:
    mode = ""
    if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
        mode = str(node.args[1].value or "")
    for keyword in node.keywords:
        if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
            mode = str(keyword.value.value or "")
    return mode


def _python_path_open_target(node: ast.AST, names: dict[str, str]) -> tuple[str | None, bool]:
    if not isinstance(node, ast.Call):
        return None, False
    func = node.func
    if not (isinstance(func, ast.Attribute) and func.attr == "open"):
        return None, False
    mode = ""
    if node.args and isinstance(node.args[0], ast.Constant):
        mode = str(node.args[0].value or "")
    for keyword in node.keywords:
        if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
            mode = str(keyword.value.value or "")
    if not any(flag in mode for flag in ("w", "a", "x", "+")):
        return None, False
    return _python_literal_path(func.value, names), True


def _python_write_targets_and_unknown(inline_code: str) -> tuple[list[str], bool]:
    try:
        tree = ast.parse(inline_code)
    except Exception:
        return [], True
    names: dict[str, str] = {}
    write_handles: dict[str, str] = {}
    targets: list[str] = []
    unknown = False
    for node in ast.walk(tree):
        if isinstance(node, ast.With):
            for item in node.items:
                if isinstance(item.optional_vars, ast.Name):
                    target = None
                    if isinstance(item.context_expr, ast.Call):
                        if isinstance(item.context_expr.func, ast.Name) and item.context_expr.func.id == "open":
                            mode = _python_write_mode_from_open_call(item.context_expr)
                            if any(flag in mode for flag in ("w", "a", "x", "+")) and item.context_expr.args:
                                target = _python_literal_path(item.context_expr.args[0], names)
                        else:
                            maybe_target, is_write_open = _python_path_open_target(item.context_expr, names)
                            if is_write_open:
                                target = maybe_target
                    if target is not None:
                        write_handles[item.optional_vars.id] = target
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            literal = _python_literal_path(node.value, names)
            if literal is not None:
                names[node.targets[0].id] = literal
            if isinstance(node.value, ast.Call):
                handle_target: str | None = None
                if isinstance(node.value.func, ast.Name) and node.value.func.id == "open":
                    mode = _python_write_mode_from_open_call(node.value)
                    if any(flag in mode for flag in ("w", "a", "x", "+")) and node.value.args:
                        handle_target = _python_literal_path(node.value.args[0], names)
                else:
                    target, is_write_open = _python_path_open_target(node.value, names)
                    if is_write_open:
                        handle_target = target
                if handle_target is not None:
                    write_handles[node.targets[0].id] = handle_target
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Attribute):
            if (
                isinstance(func.value.value, ast.Name)
                and func.value.value.id == "sys"
                and func.value.attr in {"stdout", "stderr"}
                and func.attr in {"write", "writelines"}
            ):
                continue
        if isinstance(func, ast.Name) and func.id == "open":
            mode = ""
            if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                mode = str(node.args[1].value or "")
            for keyword in node.keywords:
                if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
                    mode = str(keyword.value.value or "")
            if any(flag in mode for flag in ("w", "a", "x", "+")):
                target = _python_literal_path(node.args[0], names) if node.args else None
                if target is None:
                    unknown = True
                else:
                    targets.append(target)
        elif isinstance(func, ast.Attribute) and func.attr in {
            "write_text", "write_bytes", "unlink", "rename", "replace", "mkdir", "rmdir",
        }:
            target = _python_literal_path(func.value, names)
            if target is None:
                unknown = True
            else:
                targets.append(target)
        elif isinstance(func, ast.Attribute) and func.attr in {"write", "writelines"}:
            if isinstance(func.value, ast.Name) and func.value.id in write_handles:
                targets.append(write_handles[func.value.id])
                continue
            target, is_write_open = _python_path_open_target(func.value, names)
            if is_write_open and target is not None:
                targets.append(target)
        elif isinstance(func, ast.Attribute) and func.attr == "open":
            target, is_write_open = _python_path_open_target(node, names)
            if is_write_open and target is None:
                unknown = True
            elif is_write_open:
                targets.append(target)
        elif isinstance(func, ast.Attribute) and func.attr in {
            "remove", "unlink", "makedirs", "mkdir", "rmdir", "removedirs", "rmtree",
        }:
            target = _python_literal_path(node.args[0], names) if node.args else None
            if target is None:
                unknown = True
            else:
                targets.append(target)
    resolved = list(dict.fromkeys(targets))
    # A derivation that collapsed to a degenerate cwd-shape ('.'/'') was NOT
    # really grounded (e.g. .parent of a literal whose separators the resolver
    # could not read) — trusting it false-allowed a real runtime_data write on
    # the windows CI full-test (v6.55.0). Degenerate ⇒ UNKNOWN: the caller falls
    # back to the conservative full mention scan; benign relative writes mention
    # no drive paths and still pass that scan untouched.
    concrete = [t for t in resolved if str(t or "").strip() not in ("", ".", "./")]
    if len(concrete) != len(resolved):
        unknown = True
    return concrete, unknown


# Same resolve(strict=False) containment semantics on all platforms (SSOT).
from ouroboros.tool_access import path_is_relative_to as _path_inside


def runtime_data_write_targets(
    raw_cmd: Any,
    *,
    drive_root: pathlib.Path,
    work_dir: pathlib.Path,
    allowed_roots: List[pathlib.Path],
) -> List[str]:
    """Find write-like path mentions under runtime data but outside task artifact roots."""

    try:
        drive = pathlib.Path(drive_root).resolve(strict=False)
        cwd = pathlib.Path(work_dir).resolve(strict=False)
    except Exception:
        return []
    allowed = [pathlib.Path(root).resolve(strict=False) for root in allowed_roots]
    try:
        home = pathlib.Path.home().resolve(strict=False)
    except Exception:
        home = pathlib.Path("~").expanduser()
    blocked: List[str] = []
    scan_texts = [str(token or "") for token in shell_argv_with_inline(raw_cmd)]
    if isinstance(raw_cmd, str):
        # POSIX-mode shlex EATS backslashes in UNQUOTED tokens, so a bare Windows
        # path argv (cp C:\Users\...\data\x D:\y) reaches the token loop mangled
        # (C:Users...) and matches nothing — the windows CI full-test caught the
        # resulting false-allow (v6.55.0). The raw command string preserves the
        # separators; harvesting candidates from it too is a superset on POSIX
        # shapes (no backslashes to eat) and dedups via the blocked list.
        scan_texts.append(raw_cmd)
    for text in scan_texts:
        expanded_texts = {
            text,
            text.replace("$OUROBOROS_DATA_DIR", str(drive))
            .replace("${OUROBOROS_DATA_DIR}", str(drive))
            .replace("%OUROBOROS_DATA_DIR%", str(drive)),
            text.replace("$HOME", str(home)).replace("${HOME}", str(home)).replace("%USERPROFILE%", str(home)),
            text.replace("~/", f"{home}/"),
        }
        candidates: List[str] = []
        for expanded in expanded_texts:
            if expanded.startswith(("/", "~")) or re.match(r"^[A-Za-z]:[\\/]", expanded):
                candidates.append(expanded)
            candidates.extend(embedded_absolute_path_tokens(expanded))
            candidates.extend(EMBEDDED_WINDOWS_ABSOLUTE_PATH_RE.findall(expanded))
            candidates.extend(EMBEDDED_RELATIVE_PATH_RE.findall(expanded))
        for candidate in candidates:
            candidate_variants = {candidate}
            if "\\\\" in candidate:
                candidate_variants.add(candidate.replace("\\\\", "\\"))
            for candidate_text in candidate_variants:
                try:
                    raw_path = pathlib.Path(candidate_text).expanduser()
                    path = raw_path.resolve(strict=False) if raw_path.is_absolute() else (cwd / raw_path).resolve(strict=False)
                except Exception:
                    continue
                if not _path_inside(path, drive) or any(_path_inside(path, root) for root in allowed):
                    continue
                rendered = str(path)
                if rendered not in blocked:
                    blocked.append(rendered)
    return blocked


# SHELL-level write signals only (redirects, pipeline writers, file utilities):
# the interpreter-CODE tokens of SHELL_WRITE_INDICATORS (open(/.write(/os.remove(
# etc.) are deliberately absent — for interpreter commands those are re-judged by
# the regex+AST refinement, and the coarse `open(` token otherwise classified a
# read-only open() as writeish, keeping the old full mention scan alive for the
# exact GAIA read class this refinement exists to fix (review round 8).
_SHELL_LEVEL_WRITE_INDICATORS = (
    "rm ", "rm\t", ">", "sed -i", "tee ", "truncate",
    "mv ", "cp ", "chmod ", "chown ", "unlink ", "trash",
    "rsync ", "sort -o",
)


def _shell_level_write_signal(raw_cmd: Any) -> bool:
    """shell_has_write_indicator, restricted to SHELL-level signals (see above).

    Also treats any LIGHT_SHELL_WRITER_COMMANDS token (mkdir/touch/rm/...) in the
    flattened argv as a write signal — a ``sh -c "mkdir … && touch …"`` one-liner
    carries no indicator substring yet plainly writes."""
    if isinstance(raw_cmd, list):
        text = " ".join(str(x) for x in raw_cmd).lower()
    else:
        text = str(raw_cmd).lower()
    tokens = [str(token).lower() for token in shell_argv_with_inline(raw_cmd)]
    for token in tokens:
        if pathlib.PurePath(token).name.removesuffix(".exe") in LIGHT_SHELL_WRITER_COMMANDS:
            return True
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
    return any(ind in filtered_text for ind in _SHELL_LEVEL_WRITE_INDICATORS) or any(
        ind in text for ind in _SHELL_LEVEL_WRITE_INDICATORS if ind != ">"
    )


def _secret_runtime_data_mentions(
    raw_cmd: Any,
    *,
    drive_root: pathlib.Path,
    work_dir: pathlib.Path,
    allowed_roots: List[pathlib.Path] | None = None,
) -> List[str]:
    """Mentioned drive paths whose NAME marks secret/control state (v6.54.3).

    Reuses the subagent secret-name SSOT from tools.core (lazy import — core does
    not import this module) over every path the mention scanner can extract. The
    owner's real secret/control state (settings.json, tokens, memory/, .env) lives
    at the DRIVE ROOT, outside any task's own roots, and stays blocked. The task's
    OWN task_drive/artifact_store are exempt (adversarial review r2 #2): a staged
    attachment or own scratch file that merely NAME-matches the secret regex —
    e.g. ``secret_santa.docx``, ``token_usage.json`` — is the task's own content,
    not an owner credential, and reading it must not be blocked."""
    try:
        from ouroboros.tools.core import _is_subagent_secret_data_path
    except Exception:
        return []
    mentions = runtime_data_write_targets(
        raw_cmd, drive_root=drive_root, work_dir=work_dir,
        allowed_roots=list(allowed_roots or []),
    )
    try:
        drive = pathlib.Path(drive_root).resolve(strict=False)
    except Exception:
        return []
    hits: List[str] = []
    for text in mentions:
        try:
            rel = str(pathlib.Path(text).resolve(strict=False).relative_to(drive)).replace("\\", "/")
        except (OSError, ValueError):
            continue
        if _is_subagent_secret_data_path(rel):
            hits.append(text)
    return hits


def _project_store_runtime_data_mentions(
    raw_cmd: Any,
    *,
    drive_root: pathlib.Path,
    work_dir: pathlib.Path,
) -> List[str]:
    """Mentioned drive paths that target the per-project facts store (``projects/<id>/``).

    Parity with ``read_file(root=runtime_data)`` / ``_data_read``, which deny the
    per-project store to generic data tools unconditionally (the store is reachable
    ONLY via the project-scoped knowledge tools — no cross-project peeking). Without
    this, the light-mode read relaxation let an interpreter read another project's
    facts through a plain ``open()`` while the file API blocked it (v6.55.0)."""
    try:
        from ouroboros.project_facts import project_store_access_block
    except Exception:
        return []
    mentions = runtime_data_write_targets(
        raw_cmd, drive_root=drive_root, work_dir=work_dir, allowed_roots=[],
    )
    try:
        drive = pathlib.Path(drive_root).resolve(strict=False)
    except Exception:
        return []
    hits: List[str] = []
    for text in mentions:
        try:
            rel = str(pathlib.Path(text).resolve(strict=False).relative_to(drive)).replace("\\", "/")
        except (OSError, ValueError):
            continue
        if project_store_access_block(rel):
            hits.append(text)
    return hits


def runtime_data_guard_targets(
    raw_cmd: Any,
    *,
    writeish: bool,
    drive_root: pathlib.Path,
    work_dir: pathlib.Path,
    allowed_roots: List[pathlib.Path],
) -> List[str]:
    """Structural read-vs-write refinement of the runtime_data mention scan (v6.54.3).

    A WRITEISH command keeps the conservative behavior: every mentioned path under
    the drive outside the task's own roots blocks. A non-writeish interpreter
    command is a READ shape — blocking it on a mere path mention recast a would-be
    file-not-found into a security block (GAIA: scripts opening the task's own
    staged attachment through a mis-guessed absolute path), teaching the model to
    distrust the file API. Reads mirror the generic ``read_file(root=runtime_data)``
    policy, which light mode already permits for the task's own agent, so:

    - no write indicators at all → no block (pure read);
    - python with LITERAL write targets the AST fully resolved → block only those
      write targets that land under the drive outside the task roots;
    - anything else with write indicators (dynamic python write paths, write calls
      the AST does not model such as shutil.copy*/touch/chmod, non-python
      interpreters) → fail closed on the full mention scan.
    """
    argv = [str(t) for t in shell_argv_with_inline(raw_cmd)]
    executable = pathlib.PurePath(argv[0]).name.lower().removesuffix(".exe") if argv else ""
    interpreterish = executable.startswith("python") or executable in {
        "node", "ruby", "perl", "php", "sh", "bash", "zsh",
    }
    # Non-interpreter commands: the caller's coarse writeish decides, as before.
    # Interpreter commands: the coarse token list contains interpreter-CODE
    # markers (`open(` matches a read-only open), so the refinement below
    # re-judges those — only genuine SHELL-level write signals (redirects, tee,
    # mv/cp pipelines) keep the conservative full scan (review round 8).
    if writeish and not interpreterish:
        return runtime_data_write_targets(
            raw_cmd, drive_root=drive_root, work_dir=work_dir, allowed_roots=allowed_roots,
        )
    if interpreterish and _shell_level_write_signal(raw_cmd):
        return runtime_data_write_targets(
            raw_cmd, drive_root=drive_root, work_dir=work_dir, allowed_roots=allowed_roots,
        )
    # Even a PURE READ never touches secret/control runtime files (settings.json,
    # tokens, .env, key material): the read-vs-write relaxation mirrors the
    # generic runtime_data read policy for ordinary files only — secret-named
    # paths stay blocked on mere mention (review round 2 hardening).
    secret_hits = _secret_runtime_data_mentions(
        raw_cmd, drive_root=drive_root, work_dir=work_dir, allowed_roots=allowed_roots,
    )
    if secret_hits:
        return secret_hits
    # A pure read also never reaches the per-project facts store: parity with the
    # generic read_file(root=runtime_data) policy (no cross-project peeking), which
    # the secret-name-only check above did not cover (v6.55.0).
    project_store_hits = _project_store_runtime_data_mentions(
        raw_cmd, drive_root=drive_root, work_dir=work_dir,
    )
    if project_store_hits:
        return project_store_hits
    inline = shell_command_string(shell_argv(raw_cmd)) or " ".join(argv[1:])
    if not _INTERPRETER_ANY_WRITE_RE.search(inline):
        return []
    if executable.startswith("python"):
        targets, unknown = _python_write_targets_and_unknown(inline)
        # Trust the AST only when it POSITIVELY resolved every write target
        # (targets found, nothing unknown). A write indicator with zero AST
        # targets means a call the AST does not model — stay conservative.
        if targets and not unknown:
            try:
                drive = pathlib.Path(drive_root).resolve(strict=False)
                cwd = pathlib.Path(work_dir).resolve(strict=False)
            except Exception:
                return []
            allowed = [pathlib.Path(root).resolve(strict=False) for root in allowed_roots]
            blocked: List[str] = []
            for candidate in targets:
                try:
                    raw_path = pathlib.Path(str(candidate)).expanduser()
                    path = raw_path.resolve(strict=False) if raw_path.is_absolute() else (cwd / raw_path).resolve(strict=False)
                except Exception:
                    continue
                if _path_inside(path, drive) and not any(_path_inside(path, root) for root in allowed):
                    rendered = str(path)
                    if rendered not in blocked:
                        blocked.append(rendered)
            return blocked
    return runtime_data_write_targets(
        raw_cmd, drive_root=drive_root, work_dir=work_dir, allowed_roots=allowed_roots,
    )


def shell_has_write_indicator(raw_cmd: Any) -> bool:
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
    return any(indicator in filtered_text for indicator in SHELL_WRITE_INDICATORS) or any(
        indicator in text for indicator in SHELL_WRITE_INDICATORS if indicator != ">"
    )


def process_shell_guard_args(name: str, args: Dict[str, Any], *, ctx: Any = None, runtime_mode: str = "") -> Dict[str, Any]:
    """Normalize process-tool arguments into the command shape inspected by shell guards."""

    if name == "verify_and_record":
        # The verification `check` is run like run_command, so its resolved argv must pass
        # the SAME shell guards (subagent-secret read, protected-artifact, sudo). Use the
        # SSOT normalizer so the guard inspects EXACTLY the argv that executes (no `-lc`/`-c`
        # or recovery drift between guard and execution).
        cmd = normalize_check_argv(args.get("check")) or []
        return {"cmd": cmd, "cwd": args.get("cwd", ""), "__tool_name": name}
    if name == "run_script":
        interpreter = str(args.get("interpreter") or "python3").strip() or "python3"
        script = str(args.get("script") or "")
        cwd = args.get("cwd", "")
        if (
            not str(cwd or "").strip()
            and ctx is not None
            and str(runtime_mode or "").strip() == "light"
            and not bool(getattr(ctx, "is_workspace_mode", lambda: False)())
        ):
            try:
                cwd = str(ctx.task_drive_root())
            except Exception:
                cwd = ""
        return {
            "cmd": [interpreter, "-c", script],
            "cwd": cwd,
            "__tool_name": name,
        }
    return {**args, "__tool_name": name}


def parse_porcelain_paths(output: str) -> list[str]:
    paths: list[str] = []
    for raw_line in str(output or "").splitlines():
        line = raw_line.rstrip()
        if len(line) < 4:
            continue
        path_text = line[3:].strip()
        if " -> " in path_text:
            old_path, new_path = path_text.rsplit(" -> ", 1)
            paths.extend([old_path.strip(), new_path.strip()])
        else:
            paths.append(path_text)
    return sorted({p for p in paths if p})


def _candidate_path_inside(root: pathlib.Path, work_dir: pathlib.Path, path_text: str) -> bool:
    text = str(path_text or "").strip()
    if not text or text in {"-", "--"}:
        return False
    if text.startswith(("-", "$")) or text in {"|", "&&", "||", ";", ">", ">>"}:
        return False
    try:
        root_resolved = pathlib.Path(root).resolve()
        base = pathlib.Path(text)
        if not base.is_absolute():
            base = work_dir / base
        candidate = base.expanduser().resolve(strict=False)
        candidate.relative_to(root_resolved)
        return True
    except (OSError, ValueError):
        return False


def repo_target_mentioned(argv: List[str], *, repo_dir: pathlib.Path, cwd: str = "") -> bool:
    work_dir = pathlib.Path(repo_dir)
    if cwd and str(cwd).strip() not in ("", ".", "./"):
        try:
            work_dir = (pathlib.Path(repo_dir) / str(cwd)).resolve(strict=False)
        except OSError:
            pass
    return any(_candidate_path_inside(pathlib.Path(repo_dir), work_dir, token) for token in argv[1:])


_COMMAND_SEPARATOR_TOKENS = frozenset({"&&", "||", ";", "|", "&"})


def writer_target_tokens(argv: List[str]) -> List[str]:
    """Write TARGETS of a (possibly compound) command line.

    Compound lines are split at shell separators and each segment contributes
    only its OWN command's targets (v6.56.0): without segmentation, `touch a &&
    ./program b` credited every token after `&&` to `touch`, so a mere MENTION
    of a protected/readonly path in a later command read as a write to it."""
    segments: List[List[str]] = [[]]
    for token in argv or []:
        if str(token) in _COMMAND_SEPARATOR_TOKENS:
            segments.append([])
            continue
        segments[-1].append(token)
    if len(segments) == 1:
        return _writer_target_tokens_single(segments[0])
    targets: List[str] = []
    for segment in segments:
        if segment:
            targets.extend(_writer_target_tokens_single(segment))
    return list(dict.fromkeys(target for target in targets if str(target or "").strip()))


def _writer_target_tokens_single(argv: List[str]) -> List[str]:
    if not argv:
        return []
    cmd = pathlib.PurePath(argv[0]).name.lower().removesuffix(".exe")
    operands = [arg for arg in argv[1:] if arg and not arg.startswith("-")]
    targets: List[str] = []
    if cmd == "cp":
        targets.extend(operands[-1:] if len(operands) >= 2 else [])
    elif cmd == "ln":
        # The LINK NAME is the write target; the SOURCE is only pointed at, and
        # symlink-following reads are containment-checked at resolve time anyway.
        targets.extend(operands[-1:] if len(operands) >= 2 else [])
    elif cmd in {"chmod", "chown"}:
        targets.extend(operands[1:] if len(operands) >= 2 else [])
    elif cmd == "sed":
        targets.extend(operands[1:] if len(operands) >= 2 else operands)
    elif cmd == "sort":
        for idx, arg in enumerate(argv[1:], start=1):
            if arg == "-o" and idx + 1 < len(argv):
                targets.append(argv[idx + 1])
            if arg.startswith("--output="):
                targets.append(arg.split("=", 1)[1])
    elif cmd == "uniq":
        targets.extend(operands[1:2] if len(operands) >= 2 else [])
    elif cmd in LIGHT_SHELL_WRITER_COMMANDS:
        targets.extend(operands)

    if (cmd in _SCRIPT_INTERPRETERS or cmd.startswith("python")) and "-c" in argv:
        try:
            inline_code = str(argv[argv.index("-c") + 1])
        except Exception:
            inline_code = ""
        if cmd.startswith("python"):
            try:
                tree = ast.parse(inline_code)
            except Exception:
                tree = None
            if tree is not None:
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Call):
                        continue
                    if (
                        isinstance(node.func, ast.Name)
                        and node.func.id == "open"
                        and node.args
                        and isinstance(node.args[0], ast.Constant)
                        and isinstance(node.args[0].value, str)
                    ):
                        mode = ""
                        if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                            mode = str(node.args[1].value or "")
                        for keyword in node.keywords:
                            if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
                                mode = str(keyword.value.value or "")
                        if any(flag in mode for flag in ("w", "a", "x", "+")):
                            targets.append(node.args[0].value)
                    if (
                        isinstance(node.func, ast.Attribute)
                        and node.func.attr in {"write_text", "write_bytes"}
                        and isinstance(node.func.value, ast.Call)
                        and node.func.value.args
                        and isinstance(node.func.value.args[0], ast.Constant)
                        and isinstance(node.func.value.args[0].value, str)
                    ):
                        targets.append(node.func.value.args[0].value)
        else:
            pattern = _SCRIPT_LITERAL_WRITE_RE.get(cmd)
            if pattern:
                targets.extend(match.group(2) for match in pattern.finditer(inline_code) if match.group(2))

    for index, token in enumerate(argv):
        token_text = str(token)
        token_name = pathlib.PurePath(token_text).name.lower().removesuffix(".exe")
        if token_text in _SAFE_STDIO_REDIRECT_TOKENS:
            continue
        if token_text in _REDIRECT_TARGET_TOKENS and index + 1 < len(argv):
            if str(argv[index + 1]) == "/dev/null":
                continue
            targets.append(str(argv[index + 1]))
            continue
        redirect_match = re.match(r"^(?:[12]|&)?(?:>|>>)(.+)$", token_text)
        if redirect_match and redirect_match.group(1) not in {"/dev/null", "&1", "&2", "&-"}:
            targets.append(redirect_match.group(1))
        if token_name == "tee":
            for tee_target in argv[index + 1 :]:
                tee_target_text = str(tee_target)
                if tee_target_text in {"|", "&&", "||", ";"}:
                    break
                if tee_target_text.startswith("-"):
                    continue
                targets.append(tee_target_text)

    return list(dict.fromkeys(target for target in targets if str(target or "").strip()))


def shell_writer_targets_protected(raw_cmd: Any) -> bool:
    argv = strip_leading_env_assignments(unwrap_env_argv(shell_argv(raw_cmd)))
    if not argv:
        return False
    executable = pathlib.PurePath(argv[0]).name.lower().removesuffix(".exe")
    if executable in {"bash", "sh", "zsh"}:
        inline = shell_command_string(argv)
        return bool(inline and shell_writer_targets_protected(inline))
    if executable not in LIGHT_SHELL_WRITER_COMMANDS:
        return False
    target_text = " ".join(writer_target_tokens(argv)).replace("\\", "/").lower()
    return bool(target_text and any(cf in target_text for cf in PROTECTED_RUNTIME_PATHS_LOWER))


def _workspace_executor_state_target(path: pathlib.Path, drive_root: pathlib.Path) -> bool:
    try:
        rel_parts = pathlib.Path(path).resolve(strict=False).relative_to(
            pathlib.Path(drive_root).resolve(strict=False)
        ).parts
    except (OSError, ValueError):
        return False
    lowered = [str(part).casefold() for part in rel_parts]
    return "state" in lowered and "workspace_executor_processes" in lowered


def workspace_executor_state_write_block(
    raw_cmd: Any,
    *,
    drive_root: pathlib.Path,
    cwd: str = "",
    default_cwd: pathlib.Path | None = None,
) -> str:
    try:
        drive = pathlib.Path(drive_root).resolve(strict=False)
        work_dir = pathlib.Path(cwd).expanduser() if str(cwd or "").strip() else pathlib.Path(default_cwd or ".")
        if not work_dir.is_absolute():
            work_dir = pathlib.Path(default_cwd or ".") / work_dir
        work_dir = work_dir.resolve(strict=False)
    except Exception:
        return ""
    targets = [
        target for target in runtime_data_write_targets(raw_cmd, drive_root=drive, work_dir=work_dir, allowed_roots=[])
        if _workspace_executor_state_target(pathlib.Path(target), drive)
    ]
    if not targets:
        return ""
    return (
        "⚠️ WORKSPACE_EXECUTOR_STATE_WRITE_BLOCKED: workspace executor process records "
        "are owner/runtime control-plane state. Use process/service lifecycle tools "
        "instead of shell-writing state/workspace_executor_processes. Paths: "
        + ", ".join(targets[:5])
    )


def light_shell_repo_mutation(
    raw_cmd: Any,
    *,
    repo_dir: pathlib.Path,
    cwd: str = "",
    detect_interpreter_inline: bool = False,
) -> bool:
    """Detect simple shell writer commands that target the repo in light mode."""
    argv = shell_argv(raw_cmd)
    if not argv:
        return False
    cmd_lower = " ".join(argv).lower()

    unwrapped = unwrap_env_argv(argv)
    if unwrapped != argv:
        return light_shell_repo_mutation(
            unwrapped,
            repo_dir=repo_dir,
            cwd=cwd,
            detect_interpreter_inline=detect_interpreter_inline,
        )
    argv = strip_leading_env_assignments(argv)
    if not argv:
        return False
    executable = pathlib.PurePath(argv[0]).name.lower().removesuffix(".exe")

    if executable in {"bash", "sh", "zsh"}:
        inline = shell_command_string(argv)
        if inline:
            return light_shell_repo_mutation(
                inline,
                repo_dir=repo_dir,
                cwd=cwd,
                detect_interpreter_inline=detect_interpreter_inline,
            )

    if executable in LIGHT_SHELL_WRITER_COMMANDS and repo_target_mentioned([argv[0], *writer_target_tokens(argv)], repo_dir=repo_dir, cwd=cwd):
        return True

    if detect_interpreter_inline and executable in {"python", "python3", "node", "ruby", "perl", "php"}:
        inline = shell_command_string(argv) or " ".join(argv[1:])
        if INTERPRETER_WRITE_RE.search(inline):
            if executable in {"python", "python3"} or executable.startswith("python"):
                targets, unknown = _python_write_targets_and_unknown(inline)
                if targets and repo_target_mentioned([argv[0], *targets], repo_dir=repo_dir, cwd=cwd):
                    return True
                if unknown:
                    return True
                return False
            targets = writer_target_tokens(argv)
            if targets:
                return repo_target_mentioned([argv[0], *targets], repo_dir=repo_dir, cwd=cwd)
            # Non-Python interpreters with write indicators but no literal target
            # stay fail-closed: a dynamic path may still target the repo.
            return True
        return False

    if any(ind in cmd_lower for ind in (" > ", " >> ", " | tee ")):
        return repo_target_mentioned(argv, repo_dir=repo_dir, cwd=cwd)
    return False
