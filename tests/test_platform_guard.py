"""
AST-based guard: platform-specific APIs must live in platform_layer.py only.

Scans all .py files under ouroboros/ and supervisor/ (plus server.py)
for direct use of the following forbidden patterns:
  - Top-level imports of platform-specific modules: fcntl, msvcrt, winreg, resource
  - Direct attribute access: os.kill, os.killpg, os.setsid, os.getpgid
  - Direct attribute access: signal.SIGKILL, signal.SIGTERM

Note: subprocess flags (creationflags, start_new_session) and launcher.py
are NOT covered by this guard — platform_layer.py provides
subprocess_new_group_kwargs() and subprocess_hidden_kwargs() helpers,
and caller code uses them (checked by code review, not by this AST guard).
launcher.py is the immutable outer shell and is intentionally excluded.

Runs on every `make test` on all platforms.
"""

import ast
import pathlib
import sys
from typing import List, Set

import pytest

IS_WINDOWS_PLATFORM = sys.platform == "win32"

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# The ONE file allowed to contain platform-specific code
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
