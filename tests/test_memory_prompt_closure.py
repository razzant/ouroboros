"""The memory practice leaves timing and closure to the mind and keeps only OFF Light drafts."""
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PHRASE = "I see what waits, choose its closure and timing, and settle it when I fold."


def assert_practice(system, wake):
    memory = system.split("## Memory\n", 1)[1].split("\n## ", 1)[0]
    assert PHRASE in memory and "With my wakes off," in memory
    assert "what awaits my decision" in wake and "other open rooms keep a reader" in wake
    assert "pending drafts" not in wake


def test_memory_practice_and_wake_facts_name_the_new_decision_surface():
    system = (REPO / "prompts/SYSTEM.md").read_text(encoding="utf-8")
    wake = (REPO / "prompts/CONSCIOUSNESS.md").read_text(encoding="utf-8")
    assert_practice(system, wake)
    with pytest.raises(AssertionError):
        assert_practice(system.replace(PHRASE, "The host chooses when."), wake)
    with pytest.raises(AssertionError):
        assert_practice(system, wake + "\npending drafts")
