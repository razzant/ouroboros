"""Shared shell guard helpers for process tools."""

from __future__ import annotations

import ast
import pathlib
import re
from typing import Any, Dict, List

from ouroboros.shell_parse import (
    POSIX_SHELL_HEADS,
    collect_leading_env,
    normalize_check_argv,
    recover_stringified_argv,
    shell_argv,
    shell_command_string,
    shell_segments,
    shell_tokens_typed,
    split_redirections,
)
from ouroboros.tools.write_shape import python_body_ast, script_literal_write_targets_and_unknown


LIGHT_SHELL_WRITER_COMMANDS = frozenset({
    "chmod", "chown", "cp", "gunzip", "gzip", "ln", "mkdir", "mv",
    "perl", "rm", "ruby", "sed", "sort", "tar", "touch", "truncate", "uniq", "unzip",
})

# ONE structural owner of "is this executable a script interpreter, and of which
# family?" (XG-2R.2). The write fences used to match interpreter basenames by exact
# set plus an ad-hoc `startswith("python")`, so the versioned basenames every other
# family ships under — ruby3.2, php8.3, perl5.38, node18, the same class INFRA-1
# fixed for python only — bypassed both guards. A versioned basename IS the
# interpreter: recognition is by family stem + optional dotted version, not by
# enumerating spellings. The write-target parsers below and the protected-artifact
# interpreter check consume THIS function; do not grow a second classifier next to it.
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


# Where each family's inline code SITS — a locator, not a safety oracle. A missing
# entry costs only precision: that family's inline body is not read for write targets.
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
# parser — not another file-API vocabulary. Seeing any of these means UNKNOWN;
# the after-command output audit drops that flag and keeps only resolved targets.
_PYTHON_OPAQUE_EXEC_NAMES = frozenset({"exec", "eval", "compile", "__import__"})
_PYTHON_OPAQUE_EXEC_ATTRS = frozenset({
    "system", "popen", "Popen", "check_call", "check_output", "getoutput", "getstatusoutput",
    "execv", "execve", "execl", "execle", "execlp", "execvp", "execvpe",
    "spawnl", "spawnle", "spawnlp", "spawnv", "spawnve", "spawnvp",
    "import_module", "load_module", "dlopen", "spawn",
})
_PYTHON_OPAQUE_SUBPROCESS_ATTRS = frozenset({"run", "call"})

# The module's ONE write vocabulary, in the form the AST walker needs. The walker
# used to model only `open`/`Path`/`os.remove`-family calls, so a `shutil.copy` or
# `Path.touch` payload parsed cleanly, produced ZERO targets, and its callers read
# that as PROOF the payload cannot write (v6.89.0 panel A4). A known write call is
# modelled here or is UNKNOWN — never silently absent.
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
            and (
                (isinstance(receiver.func, ast.Name) and receiver.func.id in local_classes)
                or (
                    isinstance(receiver.func, ast.Name)
                    and receiver.func.id in {"list", "tuple", "set", "dict"}
                )
            )
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
            elif (
                isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id in {"list", "tuple", "set", "dict"}
            ):
                # Built-in collection constructors produce collection receivers;
                # their ``remove``/``replace`` methods cannot mutate the file
                # system.  Treat them like literal collections so a string item
                # named ``BIBLE.md`` is not promoted to a filesystem target.
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
    # the windows CI full-test (v6.55.0). Degenerate ⇒ UNKNOWN, never a claimed target.
    concrete = [t for t in resolved if str(t or "").strip() not in ("", ".", "./")]
    if len(concrete) != len(resolved):
        unknown = True
    return concrete, unknown


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
_DIRECTORY_CHANGE_COMMANDS = frozenset({"cd", "pushd"})


def direct_shell_rows(raw_cmd: Any) -> List[tuple]:
    """Actual argv, input/output redirects and whether explicit shell syntax applies.

    Interpreter/program bodies, substitutions, heredocs and unknown utility forms
    make no target claim. This view never changes the command that executes.
    """
    argv = list(raw_cmd) if isinstance(raw_cmd, (list, tuple)) else recover_stringified_argv(raw_cmd) or shell_argv(raw_cmd)
    if not argv:
        return []
    head = pathlib.PurePath(argv[0]).name.lower().removesuffix(".exe")
    body = shell_command_string(argv) if head in POSIX_SHELL_HEADS else ""
    if not body:
        return [(argv, [], [], False)]
    typed = shell_tokens_typed(body)
    if typed is None or any(syntax and token in {"(", ")", "<<", "<<-", "<<<"} for token, syntax in typed):
        return []
    rows, segment, reads, writes = [], [], [], []
    index = 0
    while index <= len(typed):
        token, syntax = typed[index] if index < len(typed) else (";", True)
        if syntax and token in {";", "&&", "||", "|", "|&", "&"}:
            if segment and segment[0] in _DIRECTORY_CHANGE_COMMANDS and any(c in " ".join(segment[1:]) for c in "$`*?{}~"):
                return rows  # Following relative cwd is unknown; retain only earlier target facts.
            rows.append((segment, reads, writes, True))
            segment, reads, writes = [], [], []
        elif syntax and token in {">", ">>", ">|", "&>", "&>>", ">&", "<", "<&"}:
            if segment and segment[-1].isdigit():
                segment.pop()  # Descriptor syntax is not a utility file operand.
            index += 1
            if index < len(typed):
                operand = typed[index][0]
                if token == "<":
                    reads.append(operand)
                elif token != "<&" and not (token == ">&" and (operand.isdigit() or operand == "-")):
                    writes.append(operand)
        else:
            segment.append(token)
        index += 1
    return rows


def direct_utility_target_rows(raw_cmd: Any, *, certain_only: bool = False) -> List[tuple]:
    """Certain utility writes and real output redirects (reads stay independent); ``certain_only`` drops word-guessed targets."""
    return [(argv, [*_writer_target_tokens_single(argv, direct_only=True, parse_redirects=False, certain_only=certain_only), *writes], (), False)
            for argv, _reads, writes, _shell in direct_shell_rows(raw_cmd)]


def writer_target_tokens(argv: List[str]) -> List[str]:
    """Write TARGETS of a (possibly compound) command line, flattened; an inline
    code body contributes the targets its parser finds, and a ``perl`` or ``ruby``
    body is also returned whole. Its one runtime caller is the undeclared-output
    audit (``shell_audit``)."""
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


def _writer_target_tokens_single(
    argv: List[str], *,
    direct_only: bool = False, parse_redirects: bool = True, certain_only: bool = False,
) -> List[str]:
    if not argv:
        return []
    argv, redirect_targets = split_redirections(argv) if parse_redirects else (argv, [])
    if not argv:
        return list(dict.fromkeys(t for t in redirect_targets if str(t or "").strip()))
    cmd = pathlib.PurePath(argv[0]).name.lower().removesuffix(".exe")
    if direct_only:
        if cmd == "dd":
            return [token[3:] for token in argv[1:] if token.startswith("of=")] + redirect_targets
        if cmd in _DIRECTORY_DESTINATION_COMMANDS:
            pairs = directory_destination_pairs(argv)
            return list(dict.fromkeys([*[dest for _, dest, _ in pairs],
                                       *([src for _, _, src in pairs] if cmd == "mv" else []), *redirect_targets]))
        if cmd == "sed":
            # Only explicit -i [backup] expression FILE...; the expression stays opaque.
            tail = argv[2:] if len(argv) > 2 and (argv[1] == "-i" or argv[1].startswith("-i.")) else []
            tail = tail[1:] if tail and tail[0] == "" else tail
            return [*tail[1:], *redirect_targets] if tail and all(not p.startswith("-") for p in tail[1:]) else redirect_targets
        if cmd == "tar":
            # The archive is read; -C is the extraction destination in either order.
            if len(argv) == 5 and argv[1] == "-xf" and argv[3] == "-C":
                return [argv[4], *redirect_targets]
            if len(argv) == 5 and argv[1] == "-C" and argv[3] == "-xf":
                return [argv[2], *redirect_targets]
            return redirect_targets
        if cmd == "rsync":
            return [argv[-1], *redirect_targets] if len(argv) == 3 and all(not a.startswith("-") for a in argv[1:]) and ":" not in argv[-1] else redirect_targets
        # Below here a WRITE is guessed from the command word alone; a top-level
        # principal is not fenced by that guess (owner 5A), subordinate children are.
        if certain_only or cmd not in {"touch", "rm", "mkdir", "tee", "sort", "uniq", "gzip"}:
            return redirect_targets
        if cmd in {"touch", "mkdir", "uniq", "gzip"} and any(arg.startswith("-") and arg != "--" for arg in argv[1:]):
            return redirect_targets  # Option operands need their own concrete role; do not guess.
    # A literal '-' is the STDIN OPERAND, not a flag: dropping it hid uniq's
    # output operand (`uniq - OUT` writes OUT) from every consumer (sol-max r2).
    operands = [arg for arg in argv[1:] if arg and (arg == "-" or not arg.startswith("-"))]
    targets: List[str] = []
    if cmd in {"cp", "ln"}:
        # The last operand is written (for ln the LINK NAME; the SOURCE is only
        # pointed at, and symlink-following reads are containment-checked at resolve time).
        targets.extend(operands[-1:] if len(operands) >= 2 else [])
    elif cmd in {"chmod", "chown"}:
        targets.extend(operands[1:] if len(operands) >= 2 else [])
    elif cmd == "sed":
        # Legacy observation includes -i and in-script w/W/e; direct_only above does not.
        sed_args = [str(a) for a in argv[1:]]
        inplace = any(
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
            # Preserve the historical observation shape for non-direct consumers.
            targets.extend(writing_scripts)
            targets.extend(operands[1:] if len(operands) >= 2 else operands)
    elif cmd == "tar":
        # Legacy tar observation: leading mode letters, never letters inside an attached path.
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
    for inline_code in interpreter_inline_code(argv) if not direct_only else ():
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
        if direct_only and index:
            break
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
