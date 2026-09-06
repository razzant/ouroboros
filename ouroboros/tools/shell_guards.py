"""Shared shell guard helpers for process tools."""

from __future__ import annotations

import ast
import ntpath
import pathlib
import re
from typing import Any, Dict, List

from ouroboros.runtime_mode_policy import FROZEN_CONTRACT_PATH_PREFIXES, PROTECTED_RUNTIME_PATHS
from ouroboros.shell_parse import (
    EMBEDDED_WINDOWS_ABSOLUTE_PATH_RE,
    collect_leading_env,
    embedded_absolute_path_tokens,
    env_chdir_operand,
    interpreter_reads_program_from_stdin,
    normalize_check_argv,
    replacement_target_uncertain,
    shell_argv,
    shell_argv_with_inline,
    shell_command_string,
    shell_segment_rows,
    shell_segments,
    split_redirections,
    strip_leading_env_assignments,
    unwrap_env_argv,
)

PROTECTED_RUNTIME_PATHS_LOWER = frozenset(
    p.lower() for p in PROTECTED_RUNTIME_PATHS
) | frozenset(prefix.lower() for prefix in FROZEN_CONTRACT_PATH_PREFIXES)

# Write-shape classification lives in its own leaf (extracted at the module-size
# gate); every historical name stays importable from here.
from ouroboros.tools.write_shape import (  # noqa: E402,F401
    INTERPRETER_WRITE_RE,
    PURE_FILTER_WRITER_COMMANDS,
    SHELL_WRITE_INDICATORS,
    _INTERPRETER_ANY_WRITE_RE,
    _INTERPRETER_LANE_EXCLUDED_INDICATORS,
    _OPEN_CALL_WRITE_INDICATOR_RE,
    _SAFE_STDIO_REDIRECT_TOKENS,
    _shell_write_indicator_scan,
    interpreter_write_shape,
    non_interpreter_write_shape,
    python_body_ast,
    script_literal_write_targets_and_unknown,
    segment_write_shape as _segment_write_shape,
    shell_has_write_indicator,
)

LIGHT_SHELL_WRITER_COMMANDS = frozenset({
    "chmod", "chown", "cp", "gunzip", "gzip", "ln", "mkdir", "mv",
    "perl", "rm", "ruby", "sed", "sort", "tar", "touch", "truncate", "uniq", "unzip",
})

EMBEDDED_RELATIVE_PATH_RE = re.compile(r"(?<![A-Za-z0-9_.-])(?:\.\.?/)+[^\s'\"\\),;\]]+")
# ONE structural owner of "is this executable a script interpreter, and of which
# family?" (XG-2R.2). The write fences used to match interpreter basenames by exact
# set plus an ad-hoc `startswith("python")`, so the versioned basenames every other
# family ships under — ruby3.2, php8.3, perl5.38, node18, the same class INFRA-1
# fixed for python only — bypassed both guards. A versioned basename IS the
# interpreter: recognition is by family stem + optional dotted version, not by
# enumerating spellings. Both fence guards (light_shell_repo_mutation and the
# registry runtime_data scan) and the protected-artifact interpreter check consume
# THIS function; do not grow a second classifier next to it.
_INTERPRETER_FAMILY_STEMS: tuple[tuple[str, str], ...] = (
    # (basename stem, family) — longer stems before their prefixes ("nodejs" before
    # "node"). pypy executes python code, so it classifies as the python family.
    ("python", "python"),
    ("pypy", "python"),
    ("nodejs", "node"),
    ("node", "node"),
    ("ruby", "ruby"),
    ("perl", "perl"),
    ("php", "php"),
)
# What may follow the stem: nothing, or a dotted version (ruby3.2, perl5.38.2,
# node18). Anything else (perldoc, php-fpm, python-config) is NOT the interpreter.
_INTERPRETER_VERSION_SUFFIX_RE = re.compile(r"^(?:[0-9]+(?:\.[0-9]+)*)?$")


def interpreter_family(executable: str) -> str:
    """The script-interpreter FAMILY of an executable spelling, or "".

    Accepts a bare basename or a full path, any case, with or without ``.exe`` —
    the same normalization every guard already applies — so a resolver-injected
    absolute path classifies identically to the basename the model typed.
    Windowed/ABI python spellings (pythonw, python3.7m) stay in the family:
    they execute the same code the fence exists to inspect.
    """
    name = pathlib.PurePath(str(executable or "")).name.lower().removesuffix(".exe")
    for stem, family in _INTERPRETER_FAMILY_STEMS:
        if not name.startswith(stem):
            continue
        suffix = name[len(stem):]
        if family == "python":
            suffix = suffix.removeprefix("w").removesuffix("m")
        if _INTERPRETER_VERSION_SUFFIX_RE.match(suffix):
            return family
    return ""


def _light_writer_command(executable: str) -> bool:
    """LIGHT_SHELL_WRITER_COMMANDS membership with versioned interpreter spellings
    canonicalized to their family name (ruby3.2 is `ruby`, perl5.38 is `perl`)."""
    return (interpreter_family(executable) or executable) in LIGHT_SHELL_WRITER_COMMANDS


# Where each family's inline code SITS — a locator, not a safety oracle. Containment
# does not depend on this table being complete (see light_shell_repo_mutation: an
# unknown flag makes the invocation unprovable, hence blocked); a missing entry costs
# only the precision that lets a provable write outside the repo through.
# Entries verified by execution against the real interpreters (php's from the upstream
# CLI manual); the spellings that do NOT execute their argument are deliberately
# absent — `ruby -c`/`perl -c` are compile checks over a FILENAME, `ruby -E` is an
# encoding selector, `python -e` and `ruby --eval` do not exist, `php -F` takes a file.
_INTERPRETER_INLINE_FLAGS: Dict[str, tuple[str, ...]] = {
    "python": ("-c",),
    "node": ("-e", "--eval", "-p", "--print"),
    "ruby": ("-e",),
    "perl": ("-e", "-E"),
    "php": ("-r", "--run", "-B", "--process-begin", "-R", "--process-code",
            "-E", "--process-end"),
}


def interpreter_inline_code(argv: List[str]) -> List[str]:
    """Every inline-code BODY an interpreter argv carries, for its own family.

    Three spellings, all verified to execute: a separate token (``-e CODE``), a
    joined short flag (``-eCODE``, ``python -cCODE``) and a long flag with ``=``
    (``node --eval=CODE``). php can carry several bodies at once (-B/-R/-E), so
    this returns a list. Empty for a non-interpreter, and for an interpreter that
    was handed a SCRIPT FILE rather than code.
    """
    family = interpreter_family(argv[0]) if argv else ""
    if not family:
        return []
    # `-c` is accepted for EVERY family, not just python: `process_shell_guard_args`
    # normalizes a `run_script` call into the synthetic argv `[interpreter, "-c",
    # script]` whatever the interpreter is, and that synthetic shape is what the
    # workspace/protected guards inspect. Reading `-c` per-family only (node wants
    # `-e`) silently stopped locating those bodies, and a `run_script` writing through
    # a symlink out of the workspace stopped being seen — the regression this line
    # exists to prevent (caught by tests/test_headless_cli.py's symlink-escape test).
    flags = (*_INTERPRETER_INLINE_FLAGS.get(family, ()), "-c")
    bodies: List[str] = []
    index = 1
    while index < len(argv):
        token = str(argv[index] or "")
        for flag in flags:
            if token == flag:
                if index + 1 < len(argv):
                    bodies.append(str(argv[index + 1] or ""))
                    index += 1
                break
            if flag.startswith("--") and token.startswith(f"{flag}="):
                bodies.append(token[len(flag) + 1:])
                break
            if not flag.startswith("--") and len(token) > len(flag) and token.startswith(flag):
                bodies.append(token[len(flag):])
                break
        index += 1
    return [body for body in bodies if body]


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
    if isinstance(func.value, ast.Name) and func.value.id == "os":
        flags = node.args[1] if len(node.args) > 1 else next((word.value for word in node.keywords if word.arg == "flags"), None)
        flag_names = {part.attr for part in ast.walk(flags) if isinstance(part, ast.Attribute)} if flags else set()
        if not flag_names & {"O_WRONLY", "O_RDWR", "O_CREAT", "O_TRUNC", "O_APPEND"} and ("O_RDONLY" in flag_names or isinstance(flags, ast.Constant) and flags.value == 0):
            return None, False
        return _python_literal_path(node.args[0], names) if node.args else None, True
    mode = ""
    if node.args and isinstance(node.args[0], ast.Constant):
        mode = str(node.args[0].value or "")
    for keyword in node.keywords:
        if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
            mode = str(keyword.value.value or "")
    if not any(flag in mode for flag in ("w", "a", "x", "+")):
        return None, False
    return _python_literal_path(func.value, names), True


# Constructs that move execution OUTSIDE what this AST models, making any
# "no write targets" conclusion worthless: the payload is computed at runtime
# (exec/eval/compile/__import__) or handed to another process (os.system, popen,
# subprocess, exec*/spawn*). This is a LANGUAGE-level list — what defeats the
# parser — not another file-API vocabulary, and it replaces the same idea that was
# already scattered through `_INTERPRETER_ANY_WRITE_RE`'s "opaque" branch. Seeing
# any of these means UNKNOWN, which the callers treat as write-capable.
_PYTHON_OPAQUE_EXEC_NAMES = frozenset({"exec", "eval", "compile", "__import__"})
_PYTHON_OPAQUE_EXEC_ATTRS = frozenset({
    "system", "popen", "Popen", "check_call", "check_output", "getoutput", "getstatusoutput",
    "execv", "execve", "execl", "execle", "execlp", "execvp", "execvpe",
    "spawnl", "spawnle", "spawnlp", "spawnv", "spawnve", "spawnvp",
    "import_module", "load_module", "dlopen", "spawn",
})
_PYTHON_OPAQUE_SUBPROCESS_ATTRS = frozenset({"run", "call"})

# The rest of the module's ONE write vocabulary, in the form the AST walker needs.
# `_INTERPRETER_ANY_WRITE_RE` above already enumerates these spellings as writes;
# the walker used to model only `open`/`Path`/`os.remove`-family calls, so a
# `shutil.copy` or `Path.touch` payload parsed cleanly, produced ZERO targets, and
# both callers read that as PROOF the payload cannot write. Two write vocabularies
# in one module meant the weaker one signed the proof (v6.89.0 panel A4). Anything
# the regex calls a write is modelled here or is UNKNOWN — never silently absent.
#
# The value is the index of the positional argument naming what gets WRITTEN, or
# None where no argument does (a cwd-relative extract, a hardlink whose direction
# depends on the pathlib spelling). None — and a missing/unresolvable argument —
# means UNKNOWN, the same answer the opaque-exec branch gives.
_PYTHON_WRITE_ARG_INDEX: dict = {
    "copyfile": 1, "copy2": 1, "copytree": 1, "symlink": 1,
    "chmod": 0, "chown": 0, "truncate": 0,
    "extractall": 0, "make_archive": 0, "unpack_archive": 1,
    "save": 0, "savefig": 0, "imwrite": 0,
    "to_csv": 0, "to_excel": 0, "to_parquet": 0, "to_json": 0, "to_pickle": 0,
    "link_to": None, "hardlink_to": None,
}
# Spellings the regex only treats as writes behind their module (`shutil.copy` is
# a write; `some_dict.copy()` is not). Mirroring that discrimination is what keeps
# the walker from calling every ordinary `.copy()`/`.move()` write-capable.
_PYTHON_MODULE_WRITE_ARG_INDEX: dict = {
    ("shutil", "copy"): 1, ("shutil", "move"): 1, ("os", "link"): 1,
    ("sqlite3", "connect"): 0, ("json", "dump"): 1, ("pickle", "dump"): 1,
}


def _python_vocabulary_write_target(
    node: ast.Call, callee: str, names: dict, write_handles: dict,
) -> "str | None":
    """Where a vocabulary write call writes, or None when that is not derivable."""
    index = _PYTHON_WRITE_ARG_INDEX.get(callee)
    if index is None and isinstance(node.func, ast.Attribute):
        receiver = node.func.value
        if isinstance(receiver, ast.Name):
            index = _PYTHON_MODULE_WRITE_ARG_INDEX.get((receiver.id, callee))
    if index is None or index >= len(node.args):
        return None
    argument = node.args[index]
    # `json.dump(obj, fh)` / `pickle.dump(obj, fh)` write through an already-open
    # handle, so the handle's own target is the answer the walker already knows.
    if isinstance(argument, ast.Name) and argument.id in write_handles:
        return write_handles[argument.id]
    return _python_literal_path(argument, names)


def _python_call_is_vocabulary_write(node: ast.Call, callee: str) -> bool:
    """Whether this call is a write by the module's shared vocabulary."""
    if callee in _PYTHON_WRITE_ARG_INDEX:
        return True
    receiver = node.func.value if isinstance(node.func, ast.Attribute) else None
    if not (
        isinstance(receiver, ast.Name)
        and (receiver.id, callee) in _PYTHON_MODULE_WRITE_ARG_INDEX
    ):
        return False
    # `sqlite3.connect(":memory:")` opens no file — the sentinel is not a path, and
    # resolving it relative to the cwd refused an in-memory scratch DB as a repo
    # write. An explicit `file:...?mode=ro` URI is a demonstrated READ-ONLY open
    # (#447 A4) — same carve.
    first = node.args[0] if node.args else None
    if not isinstance(first, ast.Constant):
        return True
    value = str(first.value or "")
    if value.removeprefix("file:").startswith(":memory:"):
        return False
    return not (
        callee == "connect"
        and value.startswith("file:")
        and "mode=ro" in value.partition("?")[2].split("&")
    )


def _python_call_is_opaque(node: ast.Call) -> bool:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id in _PYTHON_OPAQUE_EXEC_NAMES
    if isinstance(func, ast.Attribute):
        if func.attr in _PYTHON_OPAQUE_EXEC_ATTRS:
            return True
        # `run`/`call` are ordinary method names on anything, so they only count
        # when the receiver really is subprocess.
        if func.attr in _PYTHON_OPAQUE_SUBPROCESS_ATTRS:
            receiver = func.value
            return isinstance(receiver, ast.Name) and receiver.id == "subprocess"
    return False


def _python_write_targets_and_unknown(inline_code: str) -> tuple[list[str], bool]:
    tree = python_body_ast(inline_code)
    if tree is None:
        return [], True
    names: dict[str, str] = {}
    write_handles: dict[str, str] = {}
    targets: list[str] = []
    unknown = False
    # Receiver proof (#447 A4): a receiver that is provably NOT a path object —
    # a literal, a Name bound to a str/collection literal, or an instance of a
    # class DEFINED IN THIS PAYLOAD (whose method bodies this same walk already
    # inspects) — cannot perform a filesystem write through a colliding method
    # name (`s.replace`, `xs.remove`, `A().save`). Everything unproven stays on
    # the fail-closed path exactly as before.
    str_names: set[str] = set()
    non_path_names: set[str] = set()
    local_classes = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}

    def _receiver_not_path(receiver: ast.AST) -> bool:
        if isinstance(receiver, ast.Constant):
            return True
        if isinstance(receiver, ast.Name):
            return receiver.id in str_names or receiver.id in non_path_names
        return (
            isinstance(receiver, ast.Call)
            and isinstance(receiver.func, ast.Name)
            and receiver.func.id in local_classes
        )

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
            bound = node.targets[0].id
            # Track the binding KIND, not just the resolved text: `s = 'a,b'` is a
            # str object (its methods are str methods, never Path ops), while
            # `p = Path('a,b')` resolves to the same text but IS a path receiver.
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                str_names.add(bound)
                non_path_names.discard(bound)
            elif isinstance(node.value, (ast.List, ast.Tuple, ast.Set, ast.Dict)) or (
                isinstance(node.value, ast.Constant) and not isinstance(node.value.value, str)
            ):
                non_path_names.add(bound)
                str_names.discard(bound)
            else:
                str_names.discard(bound)
                non_path_names.discard(bound)
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
        if _python_call_is_opaque(node):
            # Execution leaves the parse here, so "no write targets" proves nothing.
            unknown = True
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
            "touch",
        }:
            if _receiver_not_path(func.value):
                # `'a,b'.replace(',', ';')` — and `s = 'a,b'; s.replace(...)` — is
                # str.replace, not Path.replace: a receiver provably bound to a
                # non-path literal has no filesystem method at all, so there is no
                # such thing as a repo write through it. Crediting one refused
                # ordinary text one-liners (`.replace` is the most common string
                # method in Python) with a security message whose advice ("move
                # your cwd") the agent could not connect to the cause. The module
                # already draws exactly this receiver line for the other ambiguous
                # spellings (`shutil.copy` writes, `some_dict.copy()` does not).
                continue
            target = _python_literal_path(func.value, names)
            if target is None:
                unknown = True
            else:
                targets.append(target)
        elif isinstance(func, ast.Attribute) and func.attr == "write" and isinstance(func.value, ast.Name) and func.value.id == "os":
            descriptor = node.args[0] if node.args else None
            if descriptor is not None and (_python_literal_path(descriptor, names) is not None or _python_path_open_target(descriptor, names)[1]): unknown = True
        elif isinstance(func, ast.Attribute) and func.attr in {"write", "writelines"}:
            if isinstance(func.value, ast.Name) and func.value.id in write_handles:
                targets.append(write_handles[func.value.id])
                continue
            target, is_write_open = _python_path_open_target(func.value, names)
            if not is_write_open and isinstance(func.value, ast.Call):
                # The CHAINED builtin form `open(p, "w").write(x)`: the receiver is the
                # open call itself, which `_python_path_open_target` only reads in its
                # `Path(p).open("w")` spelling.
                inner = func.value
                if isinstance(inner.func, ast.Name) and inner.func.id == "open" and any(
                    flag in _python_write_mode_from_open_call(inner) for flag in ("w", "a", "x", "+")
                ):
                    target = _python_literal_path(inner.args[0], names) if inner.args else None
                    is_write_open = True
            if is_write_open and target is not None:
                targets.append(target)
            else:
                # A `.write` whose receiver this walker cannot trace to a path (a
                # handle returned by a helper, `tempfile.NamedTemporaryFile()`, an
                # object held in a container) is a write it cannot LOCATE — which is
                # UNKNOWN, not proof of a read. `sys.stdout/stderr` are excluded above.
                unknown = True
        elif isinstance(func, ast.Attribute) and func.attr == "open":
            target, is_write_open = _python_path_open_target(node, names)
            if is_write_open and target is None:
                unknown = True
            elif is_write_open:
                targets.append(target)
        elif isinstance(func, ast.Attribute) and func.attr in {
            "remove", "unlink", "makedirs", "mkdir", "rmdir", "removedirs", "rmtree",
        }:
            if _receiver_not_path(func.value):
                # `xs = [1, 2]; xs.remove(item)` is list.remove — a collection-
                # literal receiver never touches the filesystem (#447 A4).
                continue
            first = node.args[0] if node.args else None
            if isinstance(first, ast.Constant) and not isinstance(first.value, (str, bytes)):
                # `d.remove(1)` is list.remove. A non-string literal is not an
                # UNRESOLVABLE path (which would rightly be UNKNOWN) — it is not a
                # path at all, so the conservative fallback never should have fired.
                continue
            target = _python_literal_path(node.args[0], names) if node.args else None
            if target is None:
                unknown = True
            else:
                targets.append(target)
        else:
            # The rest of the module's shared write vocabulary. Reached last so the
            # branches above keep owning the spellings they already model.
            callee = (
                func.attr if isinstance(func, ast.Attribute)
                else func.id if isinstance(func, ast.Name) else ""
            )
            if isinstance(func, ast.Attribute) and _receiver_not_path(func.value):
                # `A().save()` on a class DEFINED in this payload: its method body
                # is walked by this same pass, so any real write inside it gets its
                # own verdict — the call site itself proves nothing (#447 A4).
                continue
            if callee and _python_call_is_vocabulary_write(node, callee):
                target = _python_vocabulary_write_target(node, callee, names, write_handles)
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


def _expand_known_runtime_roots(text: str, drive: pathlib.Path, home: pathlib.Path) -> str:
    """Expand the existing runtime/home spellings for inspection, never execution."""
    return (text.replace("$OUROBOROS_DATA_DIR", str(drive))
            .replace("${OUROBOROS_DATA_DIR}", str(drive))
            .replace("%OUROBOROS_DATA_DIR%", str(drive))
            .replace("$HOME", str(home)).replace("${HOME}", str(home))
            .replace("%USERPROFILE%", str(home)).replace("~/", f"{home}/"))


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
        expanded_texts = {text, _expand_known_runtime_roots(text, drive, home)}
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
    target_rows: List[tuple] | None = None,
) -> List[str]:
    """Apply runtime-data policy to the same per-row effects as the workspace guard.

    Secret/control and project-store reads retain their separate boundaries.
    Known writes are checked by their targets, so reading a log while writing
    task scratch is legitimate. Unknown effects keep the existing conservative
    mention fallback for their own row; prose is never reclassified by a second
    substring vocabulary. The historical ``writeish`` hint cannot override the
    shared target facts in either direction.
    """
    from ouroboros.shell_parse import sequential_effective_cwds
    from ouroboros.tools.write_shape import _workspace_write_candidates

    secret_hits = _secret_runtime_data_mentions(
        raw_cmd, drive_root=drive_root, work_dir=work_dir, allowed_roots=allowed_roots,
    )
    project_hits = _project_store_runtime_data_mentions(
        raw_cmd, drive_root=drive_root, work_dir=work_dir,
    )
    blocked = list(dict.fromkeys([*secret_hits, *project_hits]))
    rows = target_rows if target_rows is not None else writer_target_rows(raw_cmd)
    cwds = sequential_effective_cwds(rows, pathlib.Path(work_dir))
    drive = pathlib.Path(drive_root).resolve(strict=False)
    allowed = [pathlib.Path(root).resolve(strict=False) for root in allowed_roots]
    home = pathlib.Path.home().resolve(strict=False)
    for candidate, is_write, row_index in _workspace_write_candidates(rows, [], raw_cmd):
        if not is_write or candidate == "/dev/null":
            continue
        cwd = cwds[row_index] if 0 <= row_index < len(cwds) else pathlib.Path(work_dir)
        try:
            path = pathlib.Path(_expand_known_runtime_roots(candidate, drive, home)).expanduser()
            path = (cwd / path).resolve(strict=False)
        except (OSError, ValueError, RuntimeError):
            continue
        if _path_inside(path, drive) and not any(_path_inside(path, root) for root in allowed):
            rendered = str(path)
            if rendered not in blocked:
                blocked.append(rendered)
    return blocked


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


# Absolute in WINDOWS grammar: `C:\…`/`C:/…`, or a UNC share with both segments.
# Deliberately NOT `shell_parse.is_absolute_path_text`, which also admits a leading
# `/`: on Windows a drive-less rooted path means "this drive", which is exactly what
# joining it onto the work dir already answers correctly.
_WINDOWS_ABSOLUTE_TOKEN_RE = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\[^\\/\s]+[\\/])")


def _windows_shaped_path_inside(root: pathlib.Path, path_text: str) -> bool:
    """Containment for a Windows-spelled token, judged with WINDOWS path grammar.

    POSIX ``pathlib`` does not read ``C:\\repo\\x`` as absolute, so the generic branch
    below JOINED it onto the work dir — and since the default shell cwd IS the system
    repository, every Windows-shaped path merely MENTIONED in an inline payload read as
    repo-internal and the light-mode fence refused it. A drive/UNC token is absolute
    where it comes from, so it is compared as one against the root read the same way:
    on POSIX a foreign drive is simply not under a POSIX root, and on Windows the token
    IS absolute natively, so the resolving branch below answers it instead of this one.

    DISCLOSED residual: a backslash is a legal character in a POSIX filename, so a
    literal `C:\\x` created relative to the repo cwd is no longer counted as a repo
    mention here. Nothing in the repository is named that, and this fence is a
    pre-execution mention scan, not the last control — the cost of the alternative was
    refusing every payload that merely SPELLS a Windows path.
    """
    try:
        return pathlib.PureWindowsPath(ntpath.normpath(path_text)).is_relative_to(
            pathlib.PureWindowsPath(ntpath.normpath(str(root)))
        )
    except (OSError, ValueError):
        return False


def _candidate_path_inside(root: pathlib.Path, work_dir: pathlib.Path, path_text: str) -> bool:
    text = str(path_text or "").strip()
    if not text or text in {"-", "--"}:
        return False
    if text.startswith(("-", "$")) or text in {"|", "&&", "||", ";", ">", ">>"}:
        return False
    if _WINDOWS_ABSOLUTE_TOKEN_RE.match(text) and not pathlib.Path(text).is_absolute():
        # The host's own grammar cannot read this token, so reading it with the host's
        # grammar anyway (joining a foreign absolute path onto the work dir) is the one
        # answer guaranteed to be wrong. Judge it where it comes from.
        return _windows_shaped_path_inside(root, text)
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


def repo_target_mentioned(
    argv: List[str],
    *,
    repo_dir: pathlib.Path,
    cwd: str = "",
    work_dir: pathlib.Path | None = None,
) -> bool:
    """Whether any argv path operand lands inside ``repo_dir``.

    v6.74.0 (D1): callers pass the RESOLVED ``work_dir`` from the shared
    ``resolve_shell_cwd`` resolver. The legacy ``cwd`` fallback joins the raw
    string onto ``repo_dir`` — which turned a resource-root LABEL such as
    ``cwd="task_drive"`` into a repo-internal path and false-blocked writes to
    the task's own drive — and is kept only for callers with a genuinely
    repo-relative cwd."""
    if work_dir is not None:
        resolved_work_dir = pathlib.Path(work_dir)
    else:
        resolved_work_dir = pathlib.Path(repo_dir)
        if cwd and str(cwd).strip() not in ("", ".", "./"):
            try:
                resolved_work_dir = (pathlib.Path(repo_dir) / str(cwd)).resolve(strict=False)
            except OSError:
                pass
    return any(
        _candidate_path_inside(pathlib.Path(repo_dir), resolved_work_dir, token)
        for token in argv[1:]
    )


# A sed SCRIPT that can write or execute: the `w FILE`/`W FILE` command shape
# (addressed `1w FILE` included — digits stay out of the lookbehind), the GNU
# `e`/`e cmd` execute command, or a substitute's trailing flag run carrying w/e
# after the closing `/` (`s/a/b/gw f`, `s/x/y/e` — the flag class [gpimM0-9]
# keeps replacement words like `raw ` out). Word-embedded letters (`/delete/p`,
# `s/e/x/`) stay reads; exotic non-`/` delimiters are a disclosed residual.
_SED_SCRIPT_WRITE_RE = re.compile(
    r"(?<![A-Za-z_])[wW]\s+\S|(?<![A-Za-z_])e(?:\s*(?:$|;)|\s+\S)|/[gpimM0-9]*[we](?=\s|$|;)"
)
# A wrapper body is a command line; `cd` can move later relative writes.
_SHELL_WRAPPER_HEADS = frozenset({"sh", "bash", "zsh", "dash", "ash"})
_DIRECTORY_CHANGE_COMMANDS = frozenset({"cd", "pushd"})
_MAX_INLINE_RECURSION = 3
def writer_target_rows(raw_cmd: Any, _depth: int = 0) -> List[tuple]:
    """Per-SEGMENT write facts: ``(segment_argv, targets, inline_code, unprovable)``.
    Shell bodies recurse only to ``_MAX_INLINE_RECURSION``. Unknown body effects,
    replacement templates, and write shapes without targets are unprovable."""
    rows: List[tuple] = []
    structured_rows = shell_segment_rows(raw_cmd)
    for row_index, (segment, leading_operator, heredoc_bodies) in enumerate(structured_rows):
        wrapper_cwd = env_chdir_operand(segment)
        _assignments, argv = collect_leading_env(segment)
        if not argv:
            continue
        executable = pathlib.PurePath(str(argv[0])).name.lower().removesuffix(".exe")
        program_argv, _stdin_redirects = split_redirections(argv)
        if _depth < _MAX_INLINE_RECURSION and executable in _SHELL_WRAPPER_HEADS:
            shell_body = shell_command_string(argv)
            stdin_bodies = heredoc_bodies if not shell_body and interpreter_reads_program_from_stdin(program_argv) else ()
            nested = writer_target_rows(shell_body, _depth + 1)
            for body in stdin_bodies:
                nested.extend(writer_target_rows(body, _depth + 1))
            if nested:
                if wrapper_cwd and any(row[1] or row[3] for row in nested):
                    rows.append((["cd", wrapper_cwd], [wrapper_cwd], (), False))
                rows.extend(nested)
                continue
            if stdin_bodies:
                rows.append((argv, [], tuple(stdin_bodies), True))
                continue
            if heredoc_bodies and not shell_body:
                rows.append((program_argv, [], tuple(heredoc_bodies), True))
                continue
        family = interpreter_family(executable)
        inline_code = tuple(interpreter_inline_code([str(token) for token in argv]))
        stdin_program = bool(family) and not inline_code and interpreter_reads_program_from_stdin(program_argv)
        unattached_heredoc = bool(family and heredoc_bodies and not inline_code and not stdin_program)
        if stdin_program or unattached_heredoc:
            inline_code = (*inline_code, *heredoc_bodies)
        # A code body is program text, not a target. Python targets + UNKNOWN
        # come from `_python_write_targets_and_unknown` only.
        targets = [
            t for t in _writer_target_tokens_single(argv, include_inline=False)
            if t not in inline_code
        ]
        body_unprovable = unattached_heredoc
        for body in inline_code:
            if family == "python":
                body_targets, body_unknown = _python_write_targets_and_unknown(body)
                targets.extend(body_targets)
                body_unprovable = body_unprovable or body_unknown
            else:
                body_targets, body_unknown = script_literal_write_targets_and_unknown(family, body)
                targets.extend(body_targets)
                body_unprovable = body_unprovable or body_unknown
        targets, placeholder_unprovable = replacement_target_uncertain(
            argv, targets, write_shaped=_segment_write_shape(argv),
        )
        targets = list(dict.fromkeys(t for t in targets if str(t or "").strip()))
        if executable in _DIRECTORY_CHANGE_COMMANDS:
            targets.extend(str(t) for t in argv[1:] if not str(t).startswith("-"))
        segment_argv, _redirect_targets = split_redirections(argv)
        # `-` means the unobserved program arrives on stdin.
        missing_stdin_program = bool(family) and not inline_code and any(
            str(token) == "-" for token in argv[1:])
        # A write-shaped Perl body remains uncertain even beside file operands.
        opaque_perl_body = (
            family == "perl" and bool(inline_code) and interpreter_write_shape(argv)
        )
        unprovable = (
            missing_stdin_program or body_unprovable or opaque_perl_body or placeholder_unprovable
            or (not targets and _segment_write_shape(segment_argv))
        )
        row_argv = segment_argv
        if placeholder_unprovable and leading_operator in {"|", "|&"} and row_index:
            row_argv = [*segment_argv, *structured_rows[row_index - 1][0][1:]]
        if wrapper_cwd and (targets or unprovable):
            rows.append((["cd", wrapper_cwd], [wrapper_cwd], (), False))
        if not row_argv and not targets and not inline_code and not unprovable:
            continue
        rows.append((row_argv, targets, inline_code, unprovable))
    return rows


def writer_target_tokens(argv: List[str]) -> List[str]:
    """Write TARGETS of a (possibly compound) command line, flattened.

    Unlike the row view, the light/protected lanes keep unfiltered code-body
    signals (XG-7B3.1); only the workspace path lane subtracts them."""
    targets: List[str] = []
    for segment in shell_segments(argv):
        _assignments, segment_argv = collect_leading_env(segment)
        if segment_argv:
            targets.extend(_writer_target_tokens_single(segment_argv))
    return list(dict.fromkeys(target for target in targets if str(target or "").strip()))


_DIRECTORY_DESTINATION_COMMANDS = frozenset({"cp", "ln", "mv"})
_DIRECTORY_DESTINATION_OPTIONS_WITH_ARGS = frozenset({
    "-S",
    "-t",
    "--suffix",
    "--target-directory",
})
_DIRECTORY_DESTINATION_LONG_FLAGS = frozenset({
    "--attributes-only",
    "--backup",
    "--context",
    "--dereference",
    "--force",
    "--interactive",
    "--no-clobber",
    "--no-dereference",
    "--no-target-directory",
    "--parents",
    "--preserve",
    "--recursive",
    "--reflink",
    "--relative",
    "--sparse",
    "--symbolic",
    "--symbolic-link",
    "--target-directory",
    "--verbose",
})


def directory_destination_pairs(argv: List[str]) -> List[tuple[str, str, str]]:
    """Return ``(command, destination-directory, source)`` for simple dir copies.

    ``writer_target_tokens`` intentionally sees only the destination operand.  For
    a command such as ``cp source Deliverables/``, however, the file created is
    ``Deliverables/source``.  This narrow companion covers the ordinary
    ``cp``/``mv``/``ln`` directory form without pretending to parse arbitrary
    shell or archive syntax.  Unknown long options are left to the existing
    best-effort parser rather than guessed as operands.
    """
    if not argv:
        return []
    command = pathlib.PurePath(str(argv[0])).name.lower().removesuffix(".exe")
    if command not in _DIRECTORY_DESTINATION_COMMANDS:
        return []
    operands: list[str] = []
    target_directory = ""
    i = 1
    while i < len(argv):
        token = str(argv[i] or "")
        if not token:
            i += 1
            continue
        if token == "--":
            operands.extend(str(item) for item in argv[i + 1:] if str(item or ""))
            break
        if token.startswith("-") and token != "-":
            # GNU cp/mv/ln accept attached short option arguments (``-tDIR``
            # and ``-Ssuffix``). Keep the target parser active for those
            # forms instead of failing open to the generic root check.
            if not token.startswith("--"):
                short = token[1:]
                positions = [(short.find(flag), flag) for flag in ("t", "S") if flag in short]
                if positions:
                    position, flag = min(positions)
                    attached = short[position + 1:]
                    if flag == "t":
                        if attached:
                            target_directory = attached
                            i += 1
                        elif i + 1 < len(argv):
                            target_directory = str(argv[i + 1] or "")
                            i += 2
                        else:
                            return []
                    elif attached:
                        i += 1
                    elif i + 1 < len(argv):
                        i += 2
                    else:
                        return []
                    continue
            option = token.split("=", 1)[0]
            if option in _DIRECTORY_DESTINATION_OPTIONS_WITH_ARGS:
                if "=" in token:
                    if option in {"-t", "--target-directory"}:
                        target_directory = token.split("=", 1)[1]
                    i += 1
                    continue
                if i + 1 >= len(argv):
                    return []
                if option in {"-t", "--target-directory"}:
                    target_directory = str(argv[i + 1] or "")
                i += 2
                continue
            if token.startswith("--"):
                if option not in _DIRECTORY_DESTINATION_LONG_FLAGS:
                    return []
                i += 1
                continue
            # Short clusters are flags for these commands.  ``-t`` and ``-S``
            # were handled above because they consume a following operand.
            i += 1
            continue
        operands.append(token)
        i += 1
    if target_directory:
        sources = operands
        destination = target_directory
    else:
        if len(operands) < 2:
            return []
        sources = operands[:-1]
        destination = operands[-1]
    if not destination or not sources:
        return []
    result: list[tuple[str, str, str]] = []
    for source in sources:
        source = str(source or "")
        if source:
            result.append((command, destination, source))
    return result


def _writer_target_tokens_single(argv: List[str], *, include_inline: bool = True) -> List[str]:
    if not argv:
        return []
    argv, redirect_targets = split_redirections(argv)
    if not argv:
        return list(dict.fromkeys(t for t in redirect_targets if str(t or "").strip()))
    cmd = pathlib.PurePath(argv[0]).name.lower().removesuffix(".exe")
    # A literal '-' is the STDIN OPERAND, not a flag: dropping it hid uniq's
    # output operand (`uniq - OUT` writes OUT) from every consumer (sol-max r2).
    operands = [arg for arg in argv[1:] if arg and (arg == "-" or not arg.startswith("-"))]
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
        # sed's write channels are -i (any spelling, incl. GNU attached `-ibak`)
        # AND the in-script `w`/`W` file commands and GNU `e` execute (fable-5
        # round-2: POSIX `sed 'w f' in` writes f with no -i at all). A pure
        # filter is only a script PROVABLY free of those; a -f script file or a
        # single-letter w/W/e command shape fails closed to the operand fallback.
        sed_args = [str(a) for a in argv[1:]]
        inplace = any(
            # -i in ANY short spelling, clustered included (`-ni.bak`, `-nibak`):
            # 'i' anywhere in the leading cluster letters means in-place.
            (
                t.startswith("-")
                and not t.startswith("--")
                and "i" in t.split(".", 1)[0][1:]
            )
            or t == "--in-place"
            or t.startswith("--in-place=")
            for t in sed_args
        )
        scripts: list = []
        script_unprovable = False
        expect_expr = False
        for t in sed_args:
            if expect_expr:
                scripts.append(t)
                expect_expr = False
            elif t in ("-e", "--expression"):
                expect_expr = True
            elif t.startswith("--expression="):
                scripts.append(t.split("=", 1)[1])
            elif t in ("-f", "--file") or t.startswith("--file="):
                script_unprovable = True
        if not scripts and operands:
            scripts.append(operands[0])
        writing_scripts = [s for s in scripts if _SED_SCRIPT_WRITE_RE.search(s)]
        if inplace or script_unprovable or writing_scripts:
            # The `w FILE` filename lives INSIDE the script operand; reporting the
            # script text as a target lets the cwd-joining consumers (light fence,
            # protected lane) see where it lands, exactly like the old operand
            # fallback did.
            targets.extend(writing_scripts)
            targets.extend(operands[1:] if len(operands) >= 2 else operands)
    elif cmd == "tar":
        # Mode letters are the LEADING cluster letters only (`-cf/o.tar` is
        # create+file with an attached path — the 't' inside the path is not
        # list mode; sol-max r2). Old-style `tar tf a.tar` carries the letters
        # in the first operand. Write modes (c/x/r/u/A/d, --extract/--create/…)
        # keep the operand fallback plus the attached/long file and -C/--directory
        # values; pure list (`t` with no write letter) reads.
        tar_args = [str(a) for a in argv[1:]]
        mode_letters = ""
        attached_value = ""
        for t in tar_args:
            m = re.match(r"^-([A-Za-z]+)(.*)$", t)
            if m:
                mode_letters += m.group(1)
                if attached_value == "" and m.group(2):
                    attached_value = m.group(2)
        old_style = ""
        if not mode_letters and operands and re.fullmatch(r"[A-Za-z]+", operands[0] or ""):
            old_style = operands[0]
            mode_letters = old_style
        long_write = any(
            t in ("--create", "--extract", "--get", "--append", "--update", "--delete", "--concatenate", "--catenate")
            for t in tar_args
        )
        write_mode = long_write or any(ch in mode_letters for ch in "cxruAd")
        listing = ("t" in mode_letters or "--list" in tar_args) and not write_mode
        if not listing:
            targets.extend(op for op in operands if op != old_style)
            if attached_value:
                targets.append(attached_value)
            for t in tar_args:
                if t.startswith(("--file=", "--directory=")):
                    targets.append(t.split("=", 1)[1])
    elif cmd in {"gzip", "gunzip"}:
        # Read modes by LEADING cluster letters only (`-S.tgz` is a suffix value,
        # not test mode): -l/--list, -t/--test read; -c/--stdout writes stdout
        # only. The default invocation replaces its operand (file <-> file.gz).
        readonly_mode = False
        for t in (str(a) for a in argv[1:]):
            if t in ("--list", "--test", "--stdout", "--to-stdout"):
                readonly_mode = True
                break
            m = re.match(r"^-([A-Za-z]+)", t) if not t.startswith("--") else None
            if m and any(ch in m.group(1) for ch in "ltc"):
                readonly_mode = True
                break
        if not readonly_mode:
            targets.extend(operands)
    elif cmd == "sort":
        for idx, arg in enumerate(argv[1:], start=1):
            if arg in ("-o", "--output") and idx + 1 < len(argv):
                targets.append(argv[idx + 1])
            elif arg.startswith("--output="):
                targets.append(arg.split("=", 1)[1])
            elif arg.startswith("-o") and len(arg) > 2 and not arg.startswith("--"):
                # Attached GNU spelling: `sort -oFILE`.
                targets.append(arg[2:])
    elif cmd == "uniq":
        targets.extend(operands[1:2] if len(operands) >= 2 else [])
    elif _light_writer_command(cmd):
        targets.extend(operands)

    # Inline code, through the ONE per-family flag table: `-c` alone found python
    # bodies and left `node -e` / `ruby -e` / `php -r` / `perl -e` unparsed, so
    # their literal write targets were invisible here (XG-7B3.1).
    for inline_code in interpreter_inline_code(argv) if include_inline else ():
        if interpreter_family(cmd) == "python":
            # ONE python body scanner: `_python_write_targets_and_unknown` already
            # models shutil/os/pathlib writers and reports an UNPROVABLE body. The
            # narrower duplicate that used to live here saw only literal
            # `open(...,'w')` and `write_text`, so `shutil.copy('a','<outside>/b')`
            # carried no target at all.
            body_targets, _body_unknown = _python_write_targets_and_unknown(inline_code)
            targets.extend(body_targets)
        else:
            body_targets, _body_unknown = script_literal_write_targets_and_unknown(
                interpreter_family(cmd), inline_code,
            )
            targets.extend(body_targets)

    for index, token in enumerate(argv):
        token_name = pathlib.PurePath(str(token)).name.lower().removesuffix(".exe")
        if token_name == "tee":
            for tee_target in argv[index + 1 :]:
                tee_target_text = str(tee_target)
                if tee_target_text in {"|", "&&", "||", ";"}:
                    break
                if tee_target_text.startswith("-"):
                    continue
                targets.append(tee_target_text)
    targets.extend(redirect_targets)

    return list(dict.fromkeys(target for target in targets if str(target or "").strip()))


def shell_writer_targets_protected(raw_cmd: Any) -> bool:
    argv = strip_leading_env_assignments(unwrap_env_argv(shell_argv(raw_cmd)))
    if not argv:
        return False
    executable = pathlib.PurePath(argv[0]).name.lower().removesuffix(".exe")
    if executable in {"bash", "sh", "zsh"}:
        inline = shell_command_string(argv)
        return bool(inline and shell_writer_targets_protected(inline))
    if not _light_writer_command(executable):
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


# Characters that make an argv token CODE rather than a path, a module name or a
# flag value. Structural and family-independent: it is what lets the fence tell
# `python -m pytest -q` / `node build.js` (no inline code — the content lives in a
# file, and this fence is not a script-content scanner) from an inline payload,
# WITHOUT depending on the inline-flag table being complete. An unknown flag
# carrying code is still recognized here, which is the case the inversion exists for.
_INLINE_CODE_SHAPE_RE = re.compile(r"""[(){};$`'"]|\[\s*['"]""")


def _carries_inline_code(argv: List[str], bodies: List[str]) -> bool:
    """Whether this interpreter invocation carries code IN ITS ARGV.

    The SHAPE decides, for located bodies and for every other token alike — which is
    what separates a real payload from a FILENAME sitting behind the same flag
    (`ruby -c script.rb` compile-checks a file; the synthetic `[interp, "-c", body]`
    that `process_shell_guard_args` builds for run_script carries code). Membership in
    the flag table is not enough on its own, and not required either: an unknown flag
    carrying code is still recognized here, which is the case the inversion exists for.
    """
    return any(
        _INLINE_CODE_SHAPE_RE.search(str(token or ""))
        for token in (*bodies, *argv[1:])
    )


def _dynamic_write_could_hit_repo(
    inline: str,
    *,
    repo_dir: pathlib.Path,
    cwd: str = "",
    work_dir: pathlib.Path | None = None,
) -> bool:
    """Whether a write whose TARGET the scan could not resolve might land in the repo.

    Two ways it can: the code names a repo path somewhere in its text (the classic
    ``repo = Path('<repo>'); (repo / name).write_text(...)`` shape), or the resolved
    cwd IS inside the repo, where any unresolved RELATIVE path lands.

    Without this, extending inline inspection to `run_command` (XG-7B3.1) refused an
    ordinary user_files deliverable whose filename the code computes — a real
    owner workflow, blocked over a write that provably cannot reach the repo.
    Fail-closed is right where the danger exists, not everywhere.
    """
    resolved_work_dir = pathlib.Path(work_dir) if work_dir is not None else None
    if resolved_work_dir is not None:
        try:
            resolved_work_dir.resolve(strict=False).relative_to(pathlib.Path(repo_dir).resolve(strict=False))
            return True
        except (OSError, ValueError):
            pass
    text = str(inline or "")
    mentioned = [
        *embedded_absolute_path_tokens(text),
        *EMBEDDED_WINDOWS_ABSOLUTE_PATH_RE.findall(text),
        *EMBEDDED_RELATIVE_PATH_RE.findall(text),
    ]
    return bool(mentioned) and repo_target_mentioned(
        ["", *mentioned], repo_dir=repo_dir, cwd=cwd, work_dir=work_dir,
    )


def light_shell_repo_mutation(
    raw_cmd: Any,
    *,
    repo_dir: pathlib.Path,
    cwd: str = "",
    work_dir: pathlib.Path | None = None,
    detect_interpreter_inline: bool = True,
) -> bool:
    """Detect simple shell writer commands that target the repo in light mode.

    ``work_dir`` is the RESOLVED shell cwd (v6.74.0 D1) — pass it from
    ``resolve_shell_cwd`` so a resource-root label cwd is never misread as a
    repo-relative path.

    ``detect_interpreter_inline`` DEFAULTS ON (XG-7B3.1). It was opt-in and only
    `run_script` opted in, so an interpreter's inline code reached the repo
    unexamined through `run_command`: `node18 -e "...writeFileSync('ordinary.py')"`
    in light mode mutated the file and only the POST-execution tripwire noticed,
    which reports and does not roll back. A guard whose reason to fire is what the
    interpreter executes cannot depend on which tool name invoked it, so the
    default is now the safe one and a caller has to ask for less."""
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
            work_dir=work_dir,
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
                work_dir=work_dir,
                detect_interpreter_inline=detect_interpreter_inline,
            )

    if _light_writer_command(executable) and repo_target_mentioned([argv[0], *writer_target_tokens(argv)], repo_dir=repo_dir, cwd=cwd, work_dir=work_dir):
        return True

    # Versioned interpreter basenames classify through the ONE structural family
    # classifier (XG-2R.2), and inline code is judged by PROOF, not by vocabulary
    # (XG-7B3.1 r2): the fence asks "can I prove this cannot write into the repo?"
    # and blocks when it cannot. Chasing write spellings was whack-a-mole by
    # construction — no token list can prove the ABSENCE of a write in arbitrary
    # code, and two holes outlived that enumeration. The inversion closes ONE of them:
    # `python -c "exec(open(...).read())"` is unprovable and therefore refused. Only
    # python can answer the proof, by parsing; that is what keeps the v6.54.3 read
    # allowance (whose FP evidence was python scripts opening their own staged
    # attachment). Everything else — other families, unknown flags, unparseable or
    # computed payloads — is WRITE-CAPABLE.
    # Scan side only: advanced/pro is untouched. A SCRIPT or module invocation
    # (`python -m pytest`) hands nothing inline and is not judged here.
    #
    # The COST, stated at full size rather than at its most flattering:
    #  * A non-python inline invocation that names the repo by an ABSOLUTE path, or by
    #    a `./`- or `../`-prefixed relative one, is refused even for reading. It is not
    #    refused otherwise — the resolved-cwd half of `_dynamic_write_could_hit_repo`
    #    stays python-only, because the default shell cwd IS the system repository and
    #    applying it to every family refused ordinary `node -e`/`ruby -e` work that
    #    provably writes elsewhere or only reads.
    #  * A PLAIN relative spelling is INVISIBLE to this branch. The mention scan is the
    #    same three-source harvest `runtime_data_write_targets` performs — POSIX absolute,
    #    Windows absolute, `./`/`../` relative — and none of those regexes anchors on a
    #    bare separator, so `node -e "…('ouroboros/safety.py')"` names the repo
    #    to a reader but not to the scan and RUNS — for a write as much as for a read.
    #    Disclosed, not closed: widening the regex would be a strengthening.
    #  * The OTHER hole named above, `node -e "eval(process.env.C)"`, stays OPEN — a
    #    non-python payload the parser cannot read, naming no repo path, runs even
    #    with the cwd in the repo. `test_unparseable_interpreter_code_is_treated_as_
    #    write_capable` pins it as `is False` so it cannot be re-tightened by accident.
    if detect_interpreter_inline and interpreter_family(executable):
        inline = shell_command_string(argv) or " ".join(argv[1:])
        bodies = interpreter_inline_code(argv)
        if _carries_inline_code(argv, bodies):
            if interpreter_family(executable) == "python":
                targets, unknown = _python_write_targets_and_unknown(
                    "\n".join(bodies) if bodies else inline
                )
                if not unknown:
                    # Fully understood: decide on the targets it really has (none = a
                    # proven read, which stays allowed).
                    return bool(targets) and repo_target_mentioned(
                        [argv[0], *targets], repo_dir=repo_dir, cwd=cwd, work_dir=work_dir,
                    )
                return _dynamic_write_could_hit_repo(
                    inline, repo_dir=repo_dir, cwd=cwd, work_dir=work_dir,
                )
            mentioned = [
                *embedded_absolute_path_tokens(inline),
                *EMBEDDED_WINDOWS_ABSOLUTE_PATH_RE.findall(inline),
                *EMBEDDED_RELATIVE_PATH_RE.findall(inline),
            ]
            return bool(mentioned) and repo_target_mentioned(
                ["", *mentioned], repo_dir=repo_dir, cwd=cwd, work_dir=work_dir,
            )

    if any(ind in cmd_lower for ind in (" > ", " >> ", " | tee ")):
        return repo_target_mentioned(argv, repo_dir=repo_dir, cwd=cwd, work_dir=work_dir)
    return False
