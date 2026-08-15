r"""
AST-based guard: platform-specific APIs must live in platform_layer.py only.

Scans all .py files under ouroboros/, supervisor/ and scripts/ (plus server.py)
for direct use of the following forbidden patterns:
  - Imports (top-level AND function-local — postmortem evasion class) of
    platform-specific modules: fcntl, msvcrt, winreg, resource, select,
    selectors, termios, pty, posix, nt
  - Direct attribute access: os.kill, os.killpg, os.setsid, os.getpgid,
    os.set_blocking — including through an ALIASED module import
    (`import os as o; o.kill(...)`)
  - Direct attribute access: signal.SIGKILL, signal.SIGTERM (aliases likewise)
  - Indirect reach for the same attribute by a LITERAL name:
    `getattr(os, "set_blocking")`, `os.__dict__["killpg"]`
  - Dynamic import of a forbidden module by a LITERAL name, through every spelling
    of the importer that NAMES it in the source: `importlib.import_module("fcntl")`,
    a bare `import_module("fcntl")` after `from importlib import import_module`, the
    same under an alias (`from importlib import import_module as imp; imp("termios")`),
    `__import__("pty")` and `importlib.__import__("pty")`
  - Platform-conditional subprocess kwargs, whether passed as a literal call
    keyword (`start_new_session=...`) or splatted from a LITERAL dict
    (`**{"start_new_session": True}`) — callers must splat
    platform_layer.subprocess_new_group_kwargs() / subprocess_hidden_kwargs()
    instead (**kwargs of a helper CALL carries no literal key and stays legal).
  - Platform-exclusive FILESYSTEM PATHS handed to a filesystem entry point:
    `/proc`, `/sys`, `/dev` and the Windows `\\.\` / `\\?\` namespaces, plus the
    DOS device names, in literal or f-string form — `pathlib.Path("/proc/self/
    stat").read_text()`, `os.open("/dev/tty", ...)`, `os.readlink(f"/proc/{pid}/
    ns/pid")`
  - Platform-exclusive CLOCKS: `time.clock_gettime`, `time.CLOCK_BOOTTIME` (Linux),
    `time.CLOCK_UPTIME` (BSD/macOS) and their neighbours, aliases included

WHY THE PATH RULE EXISTS, since the gate looked strict for a year without it and was
structurally unable to fire: every other rule here is a membership test against a
hand-written set of IDENTIFIERS, and each past hardening added a new SHAPE for
reaching those same identifiers (an alias, a dynamic import, a splatted dict) rather
than a new CATEGORY of platform dependency. A kernel pseudo-filesystem is an API you
call by PATH. `pathlib.Path("/proc/sys/kernel/random/boot_id").read_text()` names no
forbidden module, no forbidden attribute and no forbidden kwarg — so every rule
returned clean on it while `platform_layer` was already reading that exact file, and
the postmortem class "platform-specific code outside platform_layer" was declared
closed by a gate that could not see the most common way to write it. The categories
below are the axes of platform dependency, not a longer list of names.

BOUNDARY, stated so nobody mistakes this for total coverage: a static scan cannot
see a value it has to execute. `**kw` where `kw` is a variable, `getattr(os,
"set_" + name)`, `import_module("fc" + "ntl")`, a module fetched through
`sys.modules[...]`, a callable passed in as a parameter, or any name assembled at
runtime are NOT detected and never will be by this test — that residue is covered
by review and by the runtime behaviour of platform_layer itself, not here. What is
closed is every form that spells the forbidden name literally in the source, which
is every form that actually occurs in this repo. The importer SIDE is treated the
same way: the scan knows the names `import_module` and `__import__` bind, including
under an alias it saw bound, but a dynamic importer reached through a name it never
saw bound is part of the same residue. The PATH rule has the matching residue and it
is the same shape: `root = "/proc"; open(root + "/self/stat")` routes the literal
through a variable and is not detected, which is why the prescribed fix is a NAMED
constant in platform_layer (`POSIX_CONTROLLING_TERMINAL`) rather than a local — the
platform fact then has one home, and the gate's claim is precisely "no
platform-exclusive path is spelled outside platform_layer".

tests/ is deliberately NOT scanned: probing platform behaviour is legitimate
there, and web/ contains no Python. launcher.py is the immutable outer shell and
is intentionally excluded.

Known legitimate exceptions live in ALLOWLISTED_VIOLATIONS below, each with a
justification comment. Never silence a finding by weakening a rule — add an
explicit allowlist entry or (better) route the code through platform_layer.

Runs on every `make test` on all platforms.
"""

import ast
import pathlib
import sys
from typing import List, Set

import pytest

IS_WINDOWS_PLATFORM = sys.platform == "win32"

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# The files that ARE the platform layer — the only ones allowed to contain
# platform-specific code. Naming two rather than one does not loosen the rule: the
# rule is CONFINEMENT ("no platform-exclusive API outside the layer"), and an
# explicit short list is as closed as a single path. What the split buys is that
# `/proc` — a filesystem that is absent rather than different on macOS and Windows —
# and the boot-anchored clock read out of it sit together in a module that says so,
# instead of being 130 lines in the middle of everything else.
#
# Add to this list only for a genuine platform surface, and only with a module
# docstring that says which platform fact it owns. It is not a size valve.
ALLOWED_FILES = (
    REPO_ROOT / "ouroboros" / "platform_layer.py",
    REPO_ROOT / "ouroboros" / "platform_layer_proc.py",
)
ALLOWED_FILE = ALLOWED_FILES[0]

# Directories to scan. `scripts/` is included because the execd staging/bundle
# builders live there and run on the same three platforms as the app; `tests/`
# stays out because probing platform behaviour is legitimate in a test, and
# `web/` holds no Python.
SCAN_DIRS = [
    REPO_ROOT / "ouroboros",
    REPO_ROOT / "supervisor",
    REPO_ROOT / "scripts",
]

# Also scan server.py at the root
SCAN_FILES = [
    REPO_ROOT / "server.py",
]

# ── Forbidden patterns ──────────────────────────────────────────────────

# Platform-specific modules that must not be imported outside platform_layer.py
# (top-level OR function-local: local imports were the postmortem evasion class).
FORBIDDEN_IMPORTS: Set[str] = {
    "fcntl",
    "msvcrt",
    "winreg",
    "resource",
    "select",
    # `selectors` is a thin wrapper OVER select and inherits its platform matrix
    # (no kqueue on Linux, no epoll on macOS, select-only on Windows), so
    # forbidding `select` while allowing `selectors` closed one door and left the
    # one beside it open.
    "selectors",
    "termios",
    "pty",
    # `posix` and `nt` are the raw per-OS syscall modules `os` is a portable facade
    # OVER: `import posix; posix.kill(...)` reaches the same syscall as `os.kill` while
    # naming neither `os` nor anything the attribute checks watch, and it exists on
    # exactly one of the three platforms. `msvcrt` was already forbidden while its
    # POSIX counterpart was not, which is the asymmetry this closes.
    "posix",
    "nt",
}

# os.* calls that are platform-specific
FORBIDDEN_OS_ATTRS: Set[str] = {
    "kill",
    "killpg",
    "setsid",
    "getpgid",
    "set_blocking",
}

# subprocess kwargs that encode a platform choice; passing them as a literal
# call keyword outside platform_layer duplicates the platform branch. Callers
# splat the platform_layer helpers (subprocess_new_group_kwargs /
# subprocess_hidden_kwargs) instead.
FORBIDDEN_SUBPROCESS_KWARGS: Set[str] = {
    "start_new_session",
    "creationflags",
}

# Explicit, justified exceptions: (relative path, category, name).
# Each entry MUST carry a comment explaining why it is legitimate. Adding an
# entry is a reviewed decision — never a way to mute a new violation quietly.
ALLOWLISTED_VIOLATIONS: Set[tuple] = {
    # extension_process_runner runs INSIDE the extension child process on macOS
    # only (guarded by sys.platform == "darwin" with a try/except fallback) to
    # disable core dumps before plugin import. It is a self-contained child
    # bootstrap that must not import the full platform_layer surface; the
    # function-local `import resource` is deliberate and darwin-gated.
    ("ouroboros/extension_process_runner.py", "import", "resource"),
}

# signal.* constants/calls that are platform-specific
FORBIDDEN_SIGNAL_ATTRS: Set[str] = {
    "SIGKILL",
    "SIGTERM",
}

# time.* clocks that exist on some platforms and not others. `clock_gettime` itself is
# absent on Windows; `CLOCK_BOOTTIME` is Linux-only, `CLOCK_UPTIME*` BSD/macOS-only. This
# is the CALL half of the same platform facility `/proc/sys/kernel/random/boot_id` was
# the PATH half of — platform_layer pairs them in `boot_anchored_monotonic_ms`, and a
# second caller picking one of the two apart is how the pair stops being checkable.
FORBIDDEN_TIME_ATTRS: Set[str] = {
    "clock_gettime",
    "clock_gettime_ns",
    "clock_settime",
    "clock_settime_ns",
    "pthread_getcpuclockid",
    "CLOCK_BOOTTIME",
    "CLOCK_MONOTONIC_RAW",
    "CLOCK_PROCESS_CPUTIME_ID",
    "CLOCK_TAI",
    "CLOCK_THREAD_CPUTIME_ID",
    "CLOCK_UPTIME",
    "CLOCK_UPTIME_RAW",
}

# Filesystem roots that exist on ONE platform family and are an API there. Reaching one
# by path is a platform call that spells no platform identifier — the category the gate
# was blind to; see the module docstring.
FORBIDDEN_PATH_ROOTS: tuple = (
    "/proc",  # Linux process/kernel pseudo-filesystem; absent on macOS and Windows
    "/sys",  # Linux sysfs
    "/dev",  # POSIX device tree: no /dev/null, /dev/tty or /dev/urandom on Windows
)
# Windows-exclusive path namespaces, so the gate's imagination is not itself POSIX-only.
# Written with doubled escapes: the values are `\\.\` and `\\?\`.
FORBIDDEN_PATH_PREFIXES: tuple = (
    "\\\\.\\",  # device namespace (\\.\PIPE\…, \\.\COM1, \\.\PhysicalDrive0)
    "\\\\?\\",  # extended-length path namespace
)
# DOS reserved device names, which are paths on Windows and ordinary filenames elsewhere.
FORBIDDEN_PATH_NAMES: Set[str] = {"NUL", "CON", "CONIN$", "CONOUT$", "AUX", "PRN"}

# Calls that turn a string into a HOST filesystem path. The trigger of the rule is the
# platform-exclusive literal; this set exists only to separate "used as a path" from
# "text". A POSIX shell command built for a remote Linux TARGET legitimately contains
# `2>/dev/null` (ouroboros/remote_ssh_bootstrap.py, ouroboros/workspace_executor.py) and
# a shell-guard token table legitimately compares against `"/dev/null"` — neither is this
# host reaching a device, and a rule that could not tell the difference would be muted
# by allowlist entries within a week.
FILESYSTEM_ENTRY_POINTS: Set[str] = {
    "open", "fdopen",
    "Path", "PurePath", "PosixPath", "WindowsPath", "PurePosixPath", "PureWindowsPath",
    "stat", "lstat", "statvfs", "pathconf", "access", "chmod", "chown", "chflags",
    "listdir", "scandir", "walk", "chdir", "readlink", "symlink", "link",
    "mkdir", "makedirs", "rmdir", "removedirs", "remove", "unlink",
    "rename", "renames", "truncate", "utime",
    "exists", "lexists", "isfile", "isdir", "islink", "ismount",
    "abspath", "realpath", "getsize", "getmtime", "getctime", "join",
    "glob", "iglob", "copy", "copy2", "copyfile", "copytree", "move", "rmtree",
}


def _collect_python_files() -> List[pathlib.Path]:
    """Collect all .py files to scan."""
    files = []
    for scan_dir in SCAN_DIRS:
        if scan_dir.exists():
            for py_file in scan_dir.rglob("*.py"):
                if py_file.resolve() not in {p.resolve() for p in ALLOWED_FILES}:
                    files.append(py_file)
    for f in SCAN_FILES:
        if f.exists():
            files.append(f)
    return sorted(set(files))


def _scan_file(filepath: pathlib.Path) -> List[str]:
    """Scan a single file for platform-specific API violations.

    Returns a list of violation descriptions.
    """
    violations = []
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filepath))
    except (SyntaxError, UnicodeDecodeError):
        return violations

    rel_path = str(filepath.relative_to(REPO_ROOT))

    def report(category: str, name: str, lineno: int, message: str) -> None:
        if (rel_path, category, name) in ALLOWLISTED_VIOLATIONS:
            return
        violations.append(f"{rel_path}:{lineno}: {message}")

    # Check imports in EVERY scope (module, function, method, class body):
    # a function-local `import fcntl` is the documented evasion of the old
    # top-level-only scan. The same pass records ALIASES for os/signal, because
    # `import os as o` then `o.kill(...)` reaches the same syscall under a name
    # the attribute checks below would otherwise not recognise.
    #
    # It also records every LOCAL NAME bound to a dynamic importer. The old scan
    # matched the importer by shape — `<anything>.import_module(...)` or the bare
    # name `__import__` — so `from importlib import import_module` followed by
    # `import_module('fcntl')` named a forbidden module in plain source and passed:
    # the call's `func` is a Name, and the only Name the scan knew was `__import__`.
    dynamic_importers: set[str] = {"__import__", "import_module"}
    aliases: dict[str, str] = {"os": "os", "signal": "signal", "time": "time"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in ("importlib", "builtins"):
            for alias in node.names:
                if alias.name in ("import_module", "__import__"):
                    dynamic_importers.add(alias.asname or alias.name)
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name.split(".")[0]
                if alias.name in ("os", "signal", "time") and alias.asname:
                    aliases[alias.asname] = alias.name
                if mod in FORBIDDEN_IMPORTS:
                    report(
                        "import", mod, node.lineno,
                        f"Import of platform-specific module '{mod}'",
                    )
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_mod = node.module.split(".")[0]
            if top_mod in FORBIDDEN_IMPORTS:
                report(
                    "import", top_mod, node.lineno,
                    f"Import from platform-specific module '{node.module}'",
                )

    def forbidden_attr(module: str, attr: str) -> str:
        """The category a `<module>.<attr>` reach violates, or ""."""
        if module == "os" and attr in FORBIDDEN_OS_ATTRS:
            return "os_attr"
        if module == "signal" and attr in FORBIDDEN_SIGNAL_ATTRS:
            return "signal_attr"
        if module == "time" and attr in FORBIDDEN_TIME_ATTRS:
            return "time_attr"
        return ""

    def platform_path(node: ast.expr) -> str:
        """The platform-exclusive filesystem root a literal names, or "".

        A composite counts by its LEADING literal, because that is where the root lives:
        `f"/proc/{pid}/stat"` is a `/proc` reach whatever the interpolation holds — the
        spelling the process fingerprint used — and so is `"/proc" + rest`. What the tail
        contains is the caller's business; the platform fact is the root.
        """

        while isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            node = node.left
        if isinstance(node, ast.JoinedStr):
            head = node.values[0] if node.values else None
        else:
            head = node
        if not (isinstance(head, ast.Constant) and isinstance(head.value, str)):
            return ""
        text = head.value
        if text in FORBIDDEN_PATH_NAMES:
            return text
        for root in FORBIDDEN_PATH_ROOTS:
            if text == root or text.startswith(root + "/"):
                return root
        for prefix in FORBIDDEN_PATH_PREFIXES:
            if text.startswith(prefix):
                return prefix
        return ""

    # Check ALL attribute access for os.kill, signal.SIGKILL, etc. — under the
    # canonical name or any alias bound above.
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            module = aliases.get(node.value.id, "")
            category = forbidden_attr(module, node.attr)
            if category:
                spelling = (
                    f"{module}.{node.attr}"
                    if node.value.id == module
                    else f"{node.value.id}.{node.attr} (alias of {module})"
                )
                report(
                    category, node.attr, node.lineno,
                    f"Direct use of platform-specific {spelling}",
                )

    # Indirect reach for the SAME attribute by a literal name. Only literals are
    # judged: a computed name is outside a static scan's reach (see the module
    # docstring's BOUNDARY note) and pretending otherwise would be worse than
    # saying so.
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # getattr(os, "set_blocking")
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Name)
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            module = aliases.get(node.args[0].id, "")
            category = forbidden_attr(module, node.args[1].value)
            if category:
                report(
                    category, node.args[1].value, node.lineno,
                    f"Indirect getattr({node.args[0].id}, "
                    f"'{node.args[1].value}') reaches platform-specific "
                    f"{module}.{node.args[1].value}",
                )
        # Every importer spelling that names the module literally:
        #   importlib.import_module("fcntl") / importlib.__import__("pty")  — attribute
        #   __import__("pty")                                              — builtin name
        #   import_module("fcntl") / imp("termios")                        — bound name
        # The last two are the class this scan used to miss: `from importlib import
        # import_module [as imp]` makes the importer a plain Name, and only
        # `__import__` was recognised there.
        dynamic_import = (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in ("import_module", "__import__")
        ) or (isinstance(node.func, ast.Name) and node.func.id in dynamic_importers)
        if (
            dynamic_import
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            mod = node.args[0].value.split(".")[0]
            if mod in FORBIDDEN_IMPORTS:
                report(
                    "import", mod, node.lineno,
                    f"Dynamic import of platform-specific module '{mod}'",
                )

    # os.__dict__["killpg"] — the mapping form of the same attribute reach.
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "__dict__"
            and isinstance(node.value.value, ast.Name)
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            module = aliases.get(node.value.value.id, "")
            category = forbidden_attr(module, node.slice.value)
            if category:
                report(
                    category, node.slice.value, node.lineno,
                    f"Indirect {node.value.value.id}.__dict__['{node.slice.value}'] "
                    f"reaches platform-specific {module}.{node.slice.value}",
                )

    # Check ImportFrom for direct imports like `from os import kill`
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == "os":
                for alias in node.names:
                    if alias.name in FORBIDDEN_OS_ATTRS:
                        report(
                            "os_attr", alias.name, node.lineno,
                            f"Direct import of platform-specific os.{alias.name}",
                        )
            elif node.module == "signal":
                for alias in node.names:
                    if alias.name in FORBIDDEN_SIGNAL_ATTRS:
                        report(
                            "signal_attr", alias.name, node.lineno,
                            f"Direct import of platform-specific signal.{alias.name}",
                        )
            elif node.module == "time":
                for alias in node.names:
                    if alias.name in FORBIDDEN_TIME_ATTRS:
                        report(
                            "time_attr", alias.name, node.lineno,
                            f"Direct import of platform-specific time.{alias.name}",
                        )

    # Platform-exclusive PATHS handed to the filesystem. The category the name-list
    # rules above are structurally unable to see: `/proc` is Linux's process API and
    # calling it spells no module, attribute or kwarg.
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute):
            callee = node.func.attr
        elif isinstance(node.func, ast.Name):
            callee = node.func.id
        else:
            continue
        if callee not in FILESYSTEM_ENTRY_POINTS:
            continue
        for argument in [*node.args, *(keyword.value for keyword in node.keywords)]:
            root = platform_path(argument)
            if root:
                report(
                    "platform_path", root, node.lineno,
                    f"Platform-exclusive path '{root}' reached through '{callee}(...)'; "
                    f"a path that exists on one platform and not another is a platform "
                    f"API — name it in platform_layer and import the name",
                )

    # Check every call for platform-conditional subprocess kwargs, in BOTH
    # spellings that name the key literally:
    #   f(cmd, start_new_session=True)         — a literal call keyword
    #   f(cmd, **{"start_new_session": True})  — a literal dict, splatted
    # The second reads as the legal `**helper()` shape but is the platform branch
    # written out by hand, and `**{...}` is already idiomatic in this repo
    # (ouroboros/local_model.py), so leaving it unchecked left the door open next
    # to the one that was closed. `**subprocess_new_group_kwargs()` splats a CALL
    # result — no literal key exists in the source — and stays legal.
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg in FORBIDDEN_SUBPROCESS_KWARGS:
                report(
                    "subprocess_kwarg", keyword.arg, node.lineno,
                    f"Literal platform-conditional subprocess kwarg "
                    f"'{keyword.arg}='; splat the platform_layer helper "
                    f"(e.g. **subprocess_new_group_kwargs()) instead",
                )
            if keyword.arg is not None or not isinstance(keyword.value, ast.Dict):
                continue
            for key in keyword.value.keys:
                if (
                    isinstance(key, ast.Constant)
                    and key.value in FORBIDDEN_SUBPROCESS_KWARGS
                ):
                    report(
                        "subprocess_kwarg", key.value, node.lineno,
                        f"Platform-conditional subprocess kwarg '{key.value}' "
                        f"splatted from a LITERAL dict (**{{'{key.value}': ...}}); "
                        f"splat the platform_layer helper "
                        f"(e.g. **subprocess_new_group_kwargs()) instead",
                    )

    # A literal dict that CARRIES a forbidden key is the same platform branch even
    # when it is not splatted at the call site (built on one line, splatted on the
    # next). Only literal keys are judged; a dict assembled at runtime is outside
    # the boundary this scan claims.
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key in node.keys:
            if (
                isinstance(key, ast.Constant)
                and key.value in FORBIDDEN_SUBPROCESS_KWARGS
            ):
                report(
                    "subprocess_kwarg", key.value, node.lineno,
                    f"Literal dict declares platform-conditional subprocess "
                    f"kwarg '{key.value}'; build it in platform_layer and splat "
                    f"the helper instead",
                )

    return violations


def test_no_platform_specific_apis_outside_platform_layer():
    """All platform-specific API usage must be in platform_layer.py.

    This test scans all .py files under ouroboros/, supervisor/ and scripts/
    (plus server.py) for direct use of platform-specific APIs. Any usage outside
    of ouroboros/platform_layer.py is a violation.
    """
    all_violations = []
    files = _collect_python_files()

    for filepath in files:
        all_violations.extend(_scan_file(filepath))

    if all_violations:
        report = "\n".join(f"  {v}" for v in all_violations)
        pytest.fail(
            f"Found {len(all_violations)} platform-specific API violation(s) "
            f"outside ouroboros/platform_layer.py:\n{report}\n\n"
            f"All platform-specific code must go through platform_layer.py. "
            f"See docs/DEVELOPMENT.md 'Platform Abstraction Rule'."
        )


def _scan_source(tmp_path: pathlib.Path, source: str, monkeypatch) -> List[str]:
    """Run the real scanner over one synthetic module inside a fake repo root."""

    package = tmp_path / "ouroboros"
    package.mkdir(exist_ok=True)
    module = package / "probe.py"
    module.write_text(source, encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "REPO_ROOT", tmp_path)
    return _scan_file(module)


# Every form the scan MUST catch. Each was verified undetected before the hardening
# that closed it landed — the gate looked strict and was structurally blind to each.
INDIRECT_EVASIONS = [
    # dict-splat: reads like the legal `**helper()` shape, but the platform branch
    # is written out by hand. `**{...}` is already idiomatic in this repo.
    ("import subprocess\ndef f(c):\n    subprocess.Popen(c, **{'start_new_session': True})\n",
     "start_new_session"),
    ("import subprocess\ndef f(c):\n    subprocess.Popen(c, **{'creationflags': 8})\n",
     "creationflags"),
    # aliased module import, then the same attribute
    ("import os as o\ndef f(p):\n    o.kill(p, 9)\n", "kill"),
    ("import signal as sg\ndef f():\n    return sg.SIGKILL\n", "SIGKILL"),
    # literal-name indirection onto the same attribute
    ("import os\ndef f():\n    return getattr(os, 'set_blocking')\n", "set_blocking"),
    ("import os\ndef f():\n    return os.__dict__['killpg']\n", "killpg"),
    # dynamic import of a forbidden module by literal name
    ("import importlib\ndef f():\n    return importlib.import_module('fcntl')\n", "fcntl"),
    ("def f():\n    return __import__('pty')\n", "pty"),
    # …through the importer bound as a BARE NAME, which the attribute/`__import__`
    # shapes above do not cover: the call's func is a Name the scan had never bound.
    ("from importlib import import_module\ndef f():\n    return import_module('fcntl')\n",
     "fcntl"),
    ("from importlib import import_module as imp\ndef f():\n    return imp('termios')\n",
     "termios"),
    ("import importlib\ndef f():\n    return importlib.__import__('pty')\n", "pty"),
    # selectors: a wrapper over the already-forbidden select
    ("import selectors\ndef f():\n    return selectors.DefaultSelector()\n", "selectors"),
    ("from selectors import DefaultSelector\n", "selectors"),
    # posix / nt: the raw syscall modules `os` is a facade over. `posix.kill` is
    # `os.kill` under a name none of the attribute checks watch, and `msvcrt` was
    # forbidden while its POSIX counterpart was not.
    ("import posix\ndef f(p):\n    return posix.kill(p, 9)\n", "posix"),
    ("from posix import kill\n", "posix"),
    ("import nt\ndef f():\n    return nt.getpid()\n", "nt"),
    ("from importlib import import_module\ndef f():\n    return import_module('posix')\n",
     "posix"),
]

# The real Windows device prefix, spelled once so the escaping is readable: four
# characters, `\`, `\`, `.`, `\`, followed by a pipe name.
_WINDOWS_PIPE = "\\\\.\\PIPE\\ouroboros"
_WINDOWS_LONG = "\\\\?\\C:\\ouroboros"

# Platform-exclusive PATHS and CLOCKS. These are not more spellings of the names above —
# they are the CATEGORY the scan had no rule for, which is why the first entry, the exact
# defect a paid review found in `execd_state._process_fingerprint`, passed this gate while
# `platform_layer` was already reading the same file four lines from its own boot clock.
PLATFORM_FACILITY_EVASIONS = [
    ("import pathlib\ndef f():\n"
     "    return pathlib.Path('/proc/sys/kernel/random/boot_id').read_text()\n", "/proc"),
    ("import os\ndef f(pid):\n    return os.readlink(f'/proc/{pid}/ns/pid')\n", "/proc"),
    ("import pathlib\ndef f(pgid):\n    return pathlib.Path(f'/proc/{pgid}').exists()\n",
     "/proc"),
    ("from pathlib import Path\ndef f():\n    return Path('/proc').iterdir()\n", "/proc"),
    ("import os.path\ndef f(p):\n    return os.path.join('/proc', p)\n", "/proc"),
    ("def f():\n    return open('/sys/class/net/eth0/address').read()\n", "/sys"),
    # `/dev/tty`, which `cli_connections` opened directly for the owner password.
    ("import os\ndef f():\n    return os.open('/dev/tty', os.O_RDWR)\n", "/dev"),
    ("def f():\n    return open('/dev/urandom', 'rb').read(16)\n", "/dev"),
    # Windows-exclusive namespaces, so the gate's imagination is not itself POSIX-only.
    (f"def f():\n    return open({_WINDOWS_PIPE!r}, 'rb')\n", "\\\\.\\"),
    # …including as the leading literal of a concatenation, which is where the ROOT
    # lives; what the tail holds is the caller's business.
    (f"import pathlib\ndef f(p):\n    return pathlib.Path({_WINDOWS_LONG!r} + p)\n",
     "\\\\?\\"),
    ("import os\ndef f(pid):\n    return os.open('/proc/' + str(pid), os.O_RDONLY)\n",
     "/proc"),
    ("def f():\n    return open('NUL', 'w')\n", "NUL"),
    # The CALL half of the facility whose PATH half is `/proc/sys/kernel/random/boot_id`.
    ("import time\ndef f():\n    return time.clock_gettime(time.CLOCK_BOOTTIME)\n",
     "clock_gettime"),
    ("import time as t\ndef f():\n    return t.CLOCK_UPTIME\n", "CLOCK_UPTIME"),
    ("from time import clock_gettime\n", "clock_gettime"),
    ("import time\ndef f():\n    return getattr(time, 'CLOCK_BOOTTIME')\n",
     "CLOCK_BOOTTIME"),
]
ALL_INDIRECT_EVASIONS = INDIRECT_EVASIONS + PLATFORM_FACILITY_EVASIONS


@pytest.mark.parametrize("source,expected", ALL_INDIRECT_EVASIONS)
def test_indirect_platform_reaches_are_violations(tmp_path, monkeypatch, source, expected):
    violations = _scan_source(tmp_path, source, monkeypatch)
    assert violations, f"undetected evasion: {source!r}"
    assert any(expected in v for v in violations), violations


# Shapes that must stay legal, so the hardening does not turn into noise that
# gets muted by allowlist entries.
LEGAL_SHAPES = [
    # THE prescribed pattern: splat a helper CALL. No literal key in the source.
    "import subprocess\nfrom ouroboros.platform_layer import subprocess_new_group_kwargs\n"
    "def f(c):\n    subprocess.Popen(c, **subprocess_new_group_kwargs())\n",
    # A cross-platform os call is not a platform branch.
    "import os\ndef f():\n    return os.getpid()\n",
    # A genuinely dynamic import names no forbidden module in the source — under
    # either importer spelling, so the bare-name hardening does not turn the
    # ordinary plugin-loading idiom into a violation.
    "import importlib\ndef f(n):\n    return importlib.import_module(n)\n",
    "from importlib import import_module\ndef f(n):\n    return import_module(n)\n",
    "from importlib import import_module\ndef f():\n    return import_module('json')\n",
    "import os\ndef f():\n    return getattr(os, 'getcwd')\n",
    # Consuming the helper's OWN output: the platform decision stayed inside
    # platform_layer, which is the point (ouroboros/local_model.py does this).
    "from ouroboros.platform_layer import subprocess_hidden_kwargs\n"
    "def f(kw):\n    hidden = subprocess_hidden_kwargs()\n"
    "    return {**kw, **hidden}\n",
    # Shell TEXT built for a POSIX TARGET is not this host reaching a device, and all
    # three spellings below are in the repo today (remote_ssh_bootstrap builds the
    # bootstrap script, workspace_executor builds the kill script, tools/shell_guards
    # keeps a redirect token table). A path rule that could not tell them from a real
    # `open("/dev/tty")` would be muted by allowlist entries within a week.
    "def f(cmd):\n    return cmd + ' 2>/dev/null'\n",
    "REDIRECTS = ('>/dev/null', '1>/dev/null', '2>/dev/null')\n",
    "def f(token):\n    return token == '/dev/null'\n",
    "import subprocess\ndef f(c):\n    return subprocess.run(c, stderr=subprocess.DEVNULL)\n",
    # A portable path at a filesystem entry point is not a platform reach.
    "import pathlib\ndef f(root):\n    return pathlib.Path(root) / 'state.json'\n",
    "def f():\n    return open('config.json', 'rb')\n",
    # THE prescribed fix for a platform path: platform_layer NAMES it, callers import
    # the name, and the portable `os.open` stays where it is.
    "import os\nfrom ouroboros.platform_layer import POSIX_CONTROLLING_TERMINAL\n"
    "def f():\n    return os.open(POSIX_CONTROLLING_TERMINAL, os.O_RDWR)\n",
    # A portable clock is not a platform clock.
    "import time\ndef f():\n    return time.monotonic(), time.time()\n",
]


@pytest.mark.parametrize("source", LEGAL_SHAPES)
def test_the_prescribed_patterns_stay_legal(tmp_path, monkeypatch, source):
    assert _scan_source(tmp_path, source, monkeypatch) == []


def test_the_scan_states_its_own_boundary():
    """The residue a static scan cannot see must be NAMED, not silently implied.

    `**kw` from a variable and `getattr(os, "set_" + name)` require executing the
    program to resolve. They stay undetected on purpose; the docstring says so,
    and this test fails if that admission is ever quietly deleted while the blind
    spots remain.
    """

    assert "BOUNDARY" in (__doc__ or "")
    assert "getattr(os," in (__doc__ or "")
    assert "**kw" in (__doc__ or "")
    # The PATH rule's residue is the same shape and must be admitted in the same place,
    # together with the fix that removes it (a NAMED constant in platform_layer).
    assert 'root = "/proc"' in (__doc__ or "")
    assert "POSIX_CONTROLLING_TERMINAL" in (__doc__ or "")


@pytest.mark.parametrize(
    "source",
    [
        "import subprocess\ndef f(c, kw):\n    subprocess.Popen(c, **kw)\n",
        "import os\ndef f(n):\n    return getattr(os, 'set_' + n)\n",
        # A platform path routed through a variable, or concatenated: the same class of
        # residue as the two above, and the reason the prescribed fix is a shared NAME.
        "import os\ndef f():\n    root = '/proc'\n"
        "    return os.open(root + '/self/stat', os.O_RDONLY)\n",
        "import pathlib\ndef f(pid):\n    return pathlib.Path('/pr' + 'oc') / str(pid)\n",
    ],
)
def test_the_named_blind_spots_are_really_blind(tmp_path, monkeypatch, source):
    """Pins the admission to reality: if one of these starts being caught, the
    docstring's BOUNDARY note is stale and must be narrowed."""

    assert _scan_source(tmp_path, source, monkeypatch) == []


def test_the_allowlist_holds_no_dead_exemptions(monkeypatch):
    """A waiver for a file that no longer violates anything pardons the next one.

    The Guard Proof Rule's clause, applied here because this gate just grew a category.
    It has already retired one waiver: the `/dev/tty` literal in `cli_connections` was
    pardoned only because a second gate forbade the fix, and once
    `tests/test_cli_entrypoint.py` admitted `ouroboros.platform_layer` the call sites moved
    to `POSIX_CONTROLLING_TERMINAL` and the entry went with them. Subtract the LIVE
    findings from the waiver set and fail on the remainder.
    """

    waivers = set(ALLOWLISTED_VIOLATIONS)
    assert waivers, "nothing to check, and this test would then prove nothing"
    # Rescan with the waivers OFF, so what comes back is what they are pardoning.
    monkeypatch.setattr(sys.modules[__name__], "ALLOWLISTED_VIOLATIONS", set())
    unwaived: dict = {}
    for filepath in _collect_python_files():
        findings = _scan_file(filepath)
        if findings:
            unwaived[str(filepath.relative_to(REPO_ROOT))] = findings
    dead = {
        (path, category, name)
        for path, category, name in waivers
        if not any(name in text for text in unwaived.get(path, ()))
    }
    assert dead == set(), (
        f"these waivers pardon nothing that still happens; delete them: {sorted(dead)}"
    )


def test_the_kernel_boot_id_is_read_in_exactly_ONE_place():
    """The defect the paid review found, pinned as a property instead of a fix.

    `execd_state._process_fingerprint` read `/proc/sys/kernel/random/boot_id` itself
    while `platform_layer.boot_anchored_monotonic_ms` read the same file from a module
    constant three lines above. Two readers of one kernel file is two places to decide
    what an UNREADABLE boot id means, and the copy decided wrong: it kept building a
    fingerprint whose remaining fields (`start_ticks` counts from boot, a pid-namespace
    inode can repeat across one) are only meaningful with the anchor it had just failed
    to get. The gate this test lives beside could not see it, because a pseudo-file is
    reached by path and every rule it held was a list of names.
    """

    literal = "/proc/sys/kernel/random/boot_id"
    holders = sorted(
        str(path.relative_to(REPO_ROOT))
        for path in [*_collect_python_files(), *ALLOWED_FILES]
        if literal in path.read_text(encoding="utf-8", errors="ignore")
    )
    assert holders == ["ouroboros/platform_layer_proc.py"], (
        "the boot id must have exactly ONE reader; it now lives with the rest of "
        "the /proc surface in platform_layer_proc.py"
    )


def test_the_fingerprint_and_the_boot_CLOCK_agree_on_the_boot_identity():
    """One reader means one answer; this is what "one answer" buys.

    Runs meaningfully on every platform: off Linux both report the shared
    `BOOT_IDENTITY_UNKNOWN` sentinel, and on Linux both report the SAME kernel boot id —
    which is the property custody depends on when it compares a stored fingerprint
    against a lease deadline written on the boot-anchored clock.
    """

    import os as _os

    from ouroboros.platform_layer import (
        BOOT_IDENTITY_UNKNOWN,
        boot_anchored_monotonic_ms,
        process_fingerprint,
    )

    identity, _elapsed_ms = boot_anchored_monotonic_ms()
    fingerprint = process_fingerprint(_os.getpid())
    if identity == BOOT_IDENTITY_UNKNOWN:
        assert fingerprint is None or fingerprint["boot_id"] == BOOT_IDENTITY_UNKNOWN
    else:
        assert fingerprint is not None, "a host that can name its boot can fingerprint"
        assert fingerprint["boot_id"] == identity


def test_custody_uses_the_LAYERS_fingerprint_and_not_a_copy_of_it():
    from ouroboros import execd_state, platform_layer

    assert execd_state._process_fingerprint is platform_layer.process_fingerprint


# A rule set nothing reads is a rule set that forbids nothing, and this gate had been
# hardened four times along one axis (more SPELLINGS of the same names) and never along
# the other (more KINDS of platform dependency). The cheapest way for the NEXT category to
# arrive dead is a set declared at the top that no pass consults — so every category is
# proven to refuse something, the Guard Proof Rule applied to the guard itself.
@pytest.mark.parametrize(
    "source,category",
    [
        ("import fcntl\n", "FORBIDDEN_IMPORTS"),
        ("import os\ndef f(p):\n    os.kill(p, 9)\n", "FORBIDDEN_OS_ATTRS"),
        ("import signal\ndef f():\n    return signal.SIGKILL\n", "FORBIDDEN_SIGNAL_ATTRS"),
        ("import subprocess\ndef f(c):\n    subprocess.Popen(c, start_new_session=True)\n",
         "FORBIDDEN_SUBPROCESS_KWARGS"),
        ("import time\ndef f():\n    return time.clock_gettime(0)\n", "FORBIDDEN_TIME_ATTRS"),
        ("def f():\n    return open('/proc/self/stat')\n", "FORBIDDEN_PATH_ROOTS"),
        (f"def f():\n    return open({_WINDOWS_PIPE!r})\n", "FORBIDDEN_PATH_PREFIXES"),
        ("def f():\n    return open('NUL', 'w')\n", "FORBIDDEN_PATH_NAMES"),
    ],
)
def test_each_forbidden_category_actually_produces_a_finding(
    tmp_path, monkeypatch, source, category
):
    assert _scan_source(tmp_path, source, monkeypatch), f"{category} never fires"


def test_platform_layer_exports_core_symbols():
    """platform_layer.py must export the core cross-platform symbols."""
    from ouroboros.platform_layer import (
        IS_WINDOWS,
        IS_MACOS,
        IS_LINUX,
    )
    # Smoke check: flags are booleans
    assert isinstance(IS_WINDOWS, bool)
    assert isinstance(IS_MACOS, bool)
    assert isinstance(IS_LINUX, bool)
    # Exactly one should be True (or none on exotic platforms)
    assert sum([IS_WINDOWS, IS_MACOS, IS_LINUX]) <= 1


def test_normalize_repo_path_handles_windows_style_paths():
    """normalize_repo_path must handle Windows-style backslash paths on any OS.

    Regression test for: on Linux/macOS PurePath does NOT convert backslashes,
    so 'ouroboros\\\\tools\\\\registry.py' would bypass SAFETY_CRITICAL_PATHS
    matching if we used PurePath without explicit backslash replacement.
    (git.py protected-path matching routes through this SSOT.)
    """
    from ouroboros.runtime_mode_policy import normalize_repo_path

    # Windows-style safety-critical path must normalise to POSIX form
    assert normalize_repo_path("ouroboros\\tools\\registry.py") == "ouroboros/tools/registry.py"
    assert normalize_repo_path("ouroboros\\safety.py") == "ouroboros/safety.py"
    assert normalize_repo_path("BIBLE.md") == "BIBLE.md"

    # Mixed separators
    assert normalize_repo_path("ouroboros/tools\\git.py") == "ouroboros/tools/git.py"

    # Leading ./ stripped
    assert normalize_repo_path("./ouroboros/safety.py") == "ouroboros/safety.py"

    # Regular POSIX paths unaffected
    assert normalize_repo_path("ouroboros/tools/registry.py") == "ouroboros/tools/registry.py"


@pytest.mark.skipif(not IS_WINDOWS_PLATFORM, reason="Windows-only")
def test_win32_overlapped_class_cached():
    """_win32_overlapped_class must return the same class object on every call.

    ctypes rejects pointer arguments when the underlying Structure class differs
    even if the layout is identical.  If lock creates one OVERLAPPED class and
    unlock creates another, UnlockFileEx will raise ctypes.ArgumentError.
    """
    from ouroboros.platform_layer import _win32_overlapped_class
    cls1 = _win32_overlapped_class()
    cls2 = _win32_overlapped_class()
    assert cls1 is cls2, (
        "_win32_overlapped_class() returned different class objects — "
        "this will cause ctypes.ArgumentError in unlock path"
    )
