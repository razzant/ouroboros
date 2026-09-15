"""Tests for BackgroundConsciousness graceful context degradation.

Background consciousness used to skip its ENTIRE wakeup cycle with a bare
``OverflowError`` any time the assembled context exceeded ``BG_CONTEXT_MAX_
CHARS`` — most commonly because ``state.json`` embeds an unbounded ``usage_
accounting.by_root`` map whenever the owner has a total budget limit set,
and consciousness dumped ``state.json`` raw. A wakeup cycle skip is not a
degraded cycle, it is NO cycle: this could (and did, on the fork this was
first fixed on) leave background consciousness dead for hours until a human
intervened.

Two independent fixes, covered here:

1. The "Drive state" section now reuses ``ouroboros.context._drive_state_
   section`` — the SAME typed, disclosed projection already used on the
   foreground chat path — instead of a raw ``state.json`` dump. It excludes
   ``usage_accounting`` (the unbounded map) entirely and discloses every
   omitted key with an on-demand ``read_file`` pointer (BIBLE Principle 1:
   "No silent truncation ... relocation to on-demand reads with a visible
   pointer").
2. ``_build_context`` tracks each section's drop priority and, if the
   assembled context still overflows, drops the LARGEST non-P1 (tier-0)
   section repeatedly instead of failing the whole cycle. P1 sections
   (BIBLE, identity, scratchpad, knowledge, drive state, bg_prompt, ...)
   are never dropped; if the P1 core alone overflows, that is a genuine
   emergency and ``OverflowError`` is still raised exactly as before. A
   cycle that dropped sections says so IN the context text itself (BIBLE
   Principle 1, "Provenance matters" — a cycle must not silently reason
   with fewer inputs than it had).
"""

import json
import os
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def _make_consciousness_fixture(extra_state_fields=None):
    """Minimal on-disk layout _build_context needs, mirroring the existing
    TestBackgroundContext fixture in test_consciousness.py."""
    from ouroboros.consciousness import BackgroundConsciousness

    tmpdir = pathlib.Path(tempfile.mkdtemp())
    drive_root = tmpdir / "drive"
    repo_dir = tmpdir / "repo"
    (repo_dir / "prompts").mkdir(parents=True, exist_ok=True)
    (repo_dir / "docs").mkdir(parents=True, exist_ok=True)
    (drive_root / "memory" / "knowledge").mkdir(parents=True, exist_ok=True)
    (drive_root / "logs").mkdir(parents=True, exist_ok=True)
    (drive_root / "state").mkdir(parents=True, exist_ok=True)

    (repo_dir / "prompts" / "CONSCIOUSNESS.md").write_text("Consciousness prompt", encoding="utf-8")
    (repo_dir / "BIBLE.md").write_text("Bible", encoding="utf-8")
    (repo_dir / "VERSION").write_text("1.2.3", encoding="utf-8")
    (repo_dir / "pyproject.toml").write_text('version = "1.2.3"', encoding="utf-8")
    (repo_dir / "README.md").write_text("README", encoding="utf-8")
    (repo_dir / "docs" / "ARCHITECTURE.md").write_text('# Ouroboros v1.2.3', encoding="utf-8")
    (repo_dir / "docs" / "DEVELOPMENT.md").write_text('# Dev', encoding="utf-8")
    (drive_root / "memory" / "identity.md").write_text("I am Ouroboros", encoding="utf-8")
    (drive_root / "memory" / "scratchpad.md").write_text("scratchpad", encoding="utf-8")
    for name in ("chat.jsonl", "progress.jsonl", "tools.jsonl", "events.jsonl", "supervisor.jsonl", "task_reflections.jsonl"):
        (drive_root / "logs" / name).write_text("", encoding="utf-8")

    state = {"spent_usd": 0, "current_sha": "deadbeef"}
    state.update(extra_state_fields or {})
    (drive_root / "state" / "state.json").write_text(json.dumps(state), encoding="utf-8")

    with patch.object(BackgroundConsciousness, '_build_registry', return_value=MagicMock()):
        bc = BackgroundConsciousness(
            drive_root=drive_root,
            repo_dir=repo_dir,
            event_queue=None,
            owner_chat_id_fn=lambda: None,
        )
    return bc


class TestDriveStateTypedProjection(unittest.TestCase):
    def test_unbounded_usage_accounting_by_root_is_excluded(self):
        """The exact failure mode this closes: state.json's usage_accounting.
        by_root map (written whenever the owner sets a total budget limit)
        must never be inlined raw into the consciousness context."""
        huge_by_root = {f"root-{i}": {"settled_usd": 0, "reserved_usd": 0} for i in range(2000)}
        bc = _make_consciousness_fixture({
            "usage_accounting": {"by_root": huge_by_root, "settled_usd": 12.5},
        })
        text = bc._build_context()
        self.assertNotIn("root-1999", text)
        self.assertNotIn("settled_usd", text)
        # The typed projection's disclosure note NAMES the omission (P1: no
        # silent truncation) — "usage_accounting" appears there, but never as
        # an inlined JSON blob with per-root entries.
        self.assertIn("Drive state", text)
        self.assertIn("Omitted keys:", text)
        self.assertIn("usage_accounting", text)
        self.assertIn("read_file", text)

    def test_named_typed_keys_still_present(self):
        bc = _make_consciousness_fixture({"evolution_cycle": 7})
        text = bc._build_context()
        self.assertIn('"current_sha": "deadbeef"', text)
        self.assertIn('"evolution_cycle": 7', text)


class TestGracefulAssemblePrimitive(unittest.TestCase):
    """Direct tests of the drop-priority assembly helper, independent of the
    full _build_context fixture."""

    def test_no_drop_when_already_under_budget(self):
        from ouroboros.consciousness import BackgroundConsciousness

        sections = [("a", "x" * 100, 0), ("b", "y" * 100, 10)]
        text, dropped = BackgroundConsciousness._graceful_assemble(sections, max_chars=1000)
        self.assertEqual(dropped, [])
        self.assertIn("x" * 100, text)
        self.assertIn("y" * 100, text)

    def test_drops_largest_non_p1_section_first(self):
        from ouroboros.consciousness import BackgroundConsciousness

        sections = [
            ("p1_core", "a" * 50, 0),
            ("small_droppable", "b" * 50, 10),
            ("huge_droppable", "c" * 500, 10),
        ]
        text, dropped = BackgroundConsciousness._graceful_assemble(sections, max_chars=150)
        self.assertEqual(dropped, ["huge_droppable"])
        self.assertIn("a" * 50, text)
        self.assertIn("b" * 50, text)
        self.assertNotIn("c" * 500, text)

    def test_drops_multiple_sections_in_size_order_until_it_fits(self):
        from ouroboros.consciousness import BackgroundConsciousness

        sections = [
            ("p1_core", "a" * 10, 0),
            ("mid", "b" * 200, 20),
            ("big", "c" * 400, 10),
            ("small", "d" * 50, 10),
        ]
        text, dropped = BackgroundConsciousness._graceful_assemble(sections, max_chars=15)
        # Largest first regardless of priority tier, repeatedly, until it fits.
        self.assertEqual(dropped, ["big", "mid", "small"])
        self.assertEqual(text, "a" * 10)

    def test_never_drops_p1_even_if_still_over_budget(self):
        from ouroboros.consciousness import BackgroundConsciousness

        sections = [("p1_core", "a" * 500, 0), ("droppable", "b" * 50, 10)]
        text, dropped = BackgroundConsciousness._graceful_assemble(sections, max_chars=10)
        self.assertEqual(dropped, ["droppable"])
        self.assertIn("a" * 500, text)
        self.assertNotIn("b" * 50, text)


class TestBuildContextOverflowIntegration(unittest.TestCase):
    def test_overflow_from_droppable_content_degrades_instead_of_raising(self):
        """A huge but droppable section (recent chat) must not kill the
        whole cycle — it is dropped and the drop is disclosed in-context."""
        from ouroboros import consciousness as consciousness_mod

        bc = _make_consciousness_fixture()

        with (
            patch.object(consciousness_mod, "build_recent_sections", return_value=["## Recent chat\n\n" + "x" * 80_000]),
            patch.object(consciousness_mod, "BG_CONTEXT_MAX_CHARS", 50_000),
        ):
            text = bc._build_context()

        self.assertIn("## Context degradation", text)
        self.assertIn("section(s) dropped this cycle due to context overflow", text)
        self.assertIn("recent[0]", text)
        self.assertNotIn("x" * 80_000, text)
        # P1 content survives the degradation.
        self.assertIn("I am Ouroboros", text)

    def test_p1_only_overflow_still_raises(self):
        """If even the never-dropped core alone exceeds the budget, that is
        a genuine emergency — the cycle must still skip exactly as before."""
        from ouroboros import consciousness as consciousness_mod

        bc = _make_consciousness_fixture()
        (bc._repo_dir / "BIBLE.md").write_text("z" * 200_000, encoding="utf-8")

        with patch.object(consciousness_mod, "BG_CONTEXT_MAX_CHARS", 1_000):
            with self.assertRaises(OverflowError):
                bc._build_context()


if __name__ == "__main__":
    unittest.main()
