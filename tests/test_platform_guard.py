"""
AST-based guard: platform operations live in platform_layer.py; its private
platform_win32_abi.py owns guarded native declarations.

Scans all .py files under ouroboros/ and supervisor/ (plus server.py)
for direct use of the following forbidden patterns:
  - Top-level imports of platform-specific modules: fcntl, msvcrt, winreg, resource
  - Direct attribute access: os.kill, os.killpg, os.setsid, os.getpgid
  - Direct attribute access: signal.SIGKILL, signal.SIGTERM

Console windows on Windows have their own two scans below: no process the
launcher-process modules (launcher.py, ouroboros/launcher_*.py) start opens a
console window, and no product code detaches a process from its console.
Other subprocess flags (creationflags, start_new_session) are checked by code
review: platform_layer.py provides subprocess_new_group_kwargs() and
subprocess_hidden_kwargs(). launcher.py is excluded from the API scan above.

Runs on every `make test` on all platforms.
"""

import ast
import pathlib
import sys
from typing import List, Set

import pytest

IS_WINDOWS_PLATFORM = sys.platform == "win32"

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# The only file exempt from operational API scanning. The declaration-only
# Win32 substrate remains scanned and must not acquire process/lock operations.
ALLOWED_FILE = REPO_ROOT / "ouroboros" / "platform_layer.py"

# Directories to scan
SCAN_DIRS = [
    REPO_ROOT / "ouroboros",
    REPO_ROOT / "supervisor",
]

# Also scan server.py at the root
SCAN_FILES = [
    REPO_ROOT / "server.py",
]

# ── Forbidden patterns ──────────────────────────────────────────────────

# Platform-specific modules that must not be imported outside platform_layer.py
FORBIDDEN_IMPORTS: Set[str] = {
    "fcntl",
    "msvcrt",
    "winreg",
    "resource",
}

# os.* calls that are platform-specific
FORBIDDEN_OS_ATTRS: Set[str] = {
    "kill",
    "killpg",
    "setsid",
    "getpgid",
}

# signal.* constants/calls that are platform-specific
FORBIDDEN_SIGNAL_ATTRS: Set[str] = {
    "SIGKILL",
    "SIGTERM",
}


def _collect_python_files() -> List[pathlib.Path]:
    """Collect all .py files to scan."""
    files = []
    for scan_dir in SCAN_DIRS:
        if scan_dir.exists():
            for py_file in scan_dir.rglob("*.py"):
                if py_file.resolve() != ALLOWED_FILE.resolve():
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

    # Check top-level imports (not inside if/try/def/class)
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name.split(".")[0]
                if mod in FORBIDDEN_IMPORTS:
                    violations.append(
                        f"{rel_path}:{node.lineno}: "
                        f"Top-level import of platform-specific module '{mod}'"
                    )
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_mod = node.module.split(".")[0]
            if top_mod in FORBIDDEN_IMPORTS:
                violations.append(
                    f"{rel_path}:{node.lineno}: "
                    f"Top-level import from platform-specific module '{node.module}'"
                )

    # Check ALL attribute access for os.kill, signal.SIGKILL, etc.
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "os" and node.attr in FORBIDDEN_OS_ATTRS:
                violations.append(
                    f"{rel_path}:{node.lineno}: "
                    f"Direct use of platform-specific os.{node.attr}"
                )
            if node.value.id == "signal" and node.attr in FORBIDDEN_SIGNAL_ATTRS:
                violations.append(
                    f"{rel_path}:{node.lineno}: "
                    f"Direct use of platform-specific signal.{node.attr}"
                )

    # Check ImportFrom for direct imports like `from os import kill`
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == "os":
                for alias in node.names:
                    if alias.name in FORBIDDEN_OS_ATTRS:
                        violations.append(
                            f"{rel_path}:{node.lineno}: "
                            f"Direct import of platform-specific os.{alias.name}"
                        )
            elif node.module == "signal":
                for alias in node.names:
                    if alias.name in FORBIDDEN_SIGNAL_ATTRS:
                        violations.append(
                            f"{rel_path}:{node.lineno}: "
                            f"Direct import of platform-specific signal.{alias.name}"
                        )

    return violations


def test_no_platform_specific_apis_outside_platform_layer():
    """All platform-specific API usage must be in platform_layer.py.

    This test scans all .py files under ouroboros/ and supervisor/ (plus server.py)
    for direct use of platform-specific APIs. Any usage outside of
    ouroboros/platform_layer.py is a violation.
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


# ── Console windows (Windows) ───────────────────────────────────────────

# The packaged desktop launcher is the one Ouroboros process without a console. Windows gives each console
# program it starts a window of its own unless the start asks for none; the server itself starts without one.
# So a launcher-process module never lets a spawn open a console window: hidden flags, or the handoff's own
# console choice (a terminal run keeps its terminal); only the handoff's POSIX branch replaces the process image.
LAUNCHER_PROCESS_FILES = sorted([REPO_ROOT / "launcher.py", *(REPO_ROOT / "ouroboros").glob("launcher_*.py")])
_SPAWNS = {"subprocess": {"run", "Popen", "call", "check_call", "check_output", "getoutput", "getstatusoutput"},
           "asyncio": {"create_subprocess_exec", "create_subprocess_shell"}}
_HIDDEN_HELPERS = {"merge_hidden_kwargs", "subprocess_hidden_kwargs", "inherited_console_kwargs"}
_IMAGE_REPLACEMENT = {("ouroboros/launcher_bootstrap.py", "exec_source_launcher")}
# A process started without a console to pass on (or given its own) gives every console grandchild a window;
# no product spawn needs that. A deliberate exception names its file and reason here.
CONSOLE_DETACHING_ALLOWLIST: dict = {}
_CONSOLE_DETACHING = {"DETACHED_PROCESS", "CREATE_NEW_CONSOLE", "AllocConsole", "FreeConsole"}


def _asks_for_hidden_window(call: ast.Call) -> bool:
    for keyword in call.keywords:
        value = keyword.value
        if keyword.arg is None and isinstance(value, ast.Call) \
                and (getattr(value.func, "id", None) or getattr(value.func, "attr", None)) in _HIDDEN_HELPERS:
            return True
        if keyword.arg == "creationflags" and "CREATE_NO_WINDOW" in ast.unparse(value):
            return True
    return False


def test_launcher_process_spawns_open_no_console_window():
    offenders = []
    for path in LAUNCHER_PROCESS_FILES:
        rel = path.relative_to(REPO_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        replacing = {id(node) for function in ast.walk(tree) if isinstance(function, ast.FunctionDef)
                     and (rel, function.name) in _IMAGE_REPLACEMENT for node in ast.walk(function)}
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)):
                continue
            module, name = node.func.value.id, node.func.attr
            if name in _SPAWNS.get(module, ()) and not _asks_for_hidden_window(node):
                offenders.append(f"{rel}:{node.lineno}: {module}.{name} without hidden-window flags")
            if module == "os" and id(node) not in replacing and (
                    name.startswith(("exec", "spawn", "posix_spawn")) or name in {"system", "popen", "startfile"}):
                offenders.append(f"{rel}:{node.lineno}: os.{name}; the launcher replaces itself only through "
                                 "launcher_bootstrap.exec_source_launcher")
    assert not offenders, "\n".join(offenders)


def test_no_product_code_detaches_a_process_from_its_console():
    offenders = []
    files = [*SCAN_FILES, REPO_ROOT / "launcher.py", *(path for base in SCAN_DIRS for path in base.rglob("*.py"))]
    for path in sorted(set(files)):
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel in CONSOLE_DETACHING_ALLOWLIST:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            name = (node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute)
                    else node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None)
            if name in _CONSOLE_DETACHING:
                offenders.append(f"{rel}:{node.lineno}: {name}")
    assert not offenders, "\n".join(offenders)


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


def test_win32_declarations_are_private_to_the_platform_layer():
    for path in _collect_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                       else [node.module or "", *(alias.name for alias in node.names)]
                       if isinstance(node, ast.ImportFrom) else [])
            assert all(name.split(".")[-1] != "platform_win32_abi" for name in modules), str(path)


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
def test_win32_lock_and_unlock_share_the_same_overlapped_abi():
    """Both APIs must accept the identical structure class, not just equal layouts.

    ctypes rejects pointer arguments for structurally equal but distinct classes.
    The platform's one Windows ABI declaration owns both native signatures.
    """
    from ouroboros.platform_layer import _OVERLAPPED, _kernel32
    from ouroboros import platform_win32_abi

    assert _OVERLAPPED is platform_win32_abi._OVERLAPPED
    assert _kernel32 is platform_win32_abi._kernel32
    assert _kernel32.LockFileEx.argtypes[-1]._type_ is _OVERLAPPED
    assert _kernel32.UnlockFileEx.argtypes[-1]._type_ is _OVERLAPPED
