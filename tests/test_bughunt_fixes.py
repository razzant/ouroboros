"""Tests for the multi-agent bughunt fixes (2026-05-01).

Eight verified bugs were found by three parallel audit agents
covering concurrency/resource leaks, silent error swallowing, and
path-safety / sandbox holes. Each test below pins one of the fixes
so a regression would be caught immediately.
"""

from __future__ import annotations


import pytest


# test_chat_id_zero_preserved_not_nulled removed in v5.15.x — was a pure
# source-string pin reading ouroboros/agent.py for the literal patch
# (`int(task.get("chat_id") or 0) or None` must not appear). Per the user's
# `delete_trust_behavioral` choice, no behavioral replacement was added.
# Note: a directed behavioral guard (a task with chat_id=0 routed through
# the agent loop without collapse to None) is NOT currently in the suite;
# adding one would be a separate follow-up if this regression class
# resurfaces.


# ---------------------------------------------------------------------------
# P0 — case-insensitive safety-critical path check
# ---------------------------------------------------------------------------
# P1 — append_jsonl returns bool
# ---------------------------------------------------------------------------
# P1 — safe_relpath rejects NUL bytes and control chars
# ---------------------------------------------------------------------------
# P2 — safe_read distinguishes missing vs unreadable
# ---------------------------------------------------------------------------

def test_safe_read_warns_when_file_unreadable_but_exists(tmp_path, caplog):
    """Bug 6: previously DEBUG-only when a file existed-but-unreadable.
    For BIBLE.md or identity.md that's a real infrastructure error the
    operator must see."""
    from ouroboros.context import safe_read

    f = tmp_path / "secret.md"
    f.write_text("constitutional content", encoding="utf-8")
    f.chmod(0o000)  # unreadable

    try:
        with caplog.at_level("WARNING"):
            result = safe_read(f, fallback="(fallback)")
        # On systems where root can still read (containers without proper
        # cap drops), this might succeed. Skip if the file is still readable.
        if result == "constitutional content":
            pytest.skip("OS allows reading despite chmod 000")
        assert result == "(fallback)"
        assert any("safe_read" in r.message for r in caplog.records), (
            "safe_read must log at WARNING when an existing file is unreadable"
        )
    finally:
        f.chmod(0o644)


# ---------------------------------------------------------------------------
# P2 — PID lock FD leak
# ---------------------------------------------------------------------------


# ``test_imports_after_bughunt_dont_raise`` was retired in v5.8.3-rc.5.
# pytest itself imports every test module during collection, and
# ``tests/test_smoke.py`` already pins the broader import surface; the
# bughunt-specific list above duplicated coverage that pytest collection +
# smoke already provide.
