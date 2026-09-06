"""Shared text projection of the typed pre-execution shell guard (v7 D02).

The characterization suites that predate the typed cutover assert on the exact
denial TEXT; the typed codes are pinned in tests/test_registry_guard_process.py
and the tool_result suites. Keeping the projection here keeps those files at
their pre-cutover line counts (size-ratchet) and in one spelling.
"""

from __future__ import annotations

from ouroboros.tools.registry_guard_process import _run_shell_safety_check


def _shell_guard_text(reg, args, runtime_mode, binding=None):
    result = _run_shell_safety_check(reg, args, runtime_mode, binding)
    return None if result is None else result.text
