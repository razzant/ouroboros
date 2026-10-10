"""Test reasoning effort resolution via config.resolve_effort()."""

import os
from unittest.mock import patch
from ouroboros.config import resolve_effort


# ---------------------------------------------------------------------------
# Task / Chat
# ---------------------------------------------------------------------------

def test_task_effort_default_is_medium():
    """Default task effort is 'medium' when no env var is set."""
    with patch.dict(os.environ, {}, clear=True):
        assert resolve_effort("task") == "medium"
        assert resolve_effort("chat") == "medium"
        assert resolve_effort("") == "medium"


def test_task_effort_via_new_env():
    """OUROBOROS_EFFORT_TASK controls task/chat effort."""
    for effort in ("none", "low", "medium", "high"):
        with patch.dict(os.environ, {"OUROBOROS_EFFORT_TASK": effort}, clear=True):
            assert resolve_effort("task") == effort


def test_task_effort_legacy_alias_no_longer_honoured():
    """v5.15.0 retired OUROBOROS_INITIAL_REASONING_EFFORT — only the new key is read."""
    with patch.dict(os.environ, {"OUROBOROS_INITIAL_REASONING_EFFORT": "high"}, clear=True):
        # Legacy alias is ignored; default applies.
        assert resolve_effort("task") == "medium"


def test_task_effort_invalid_falls_back_to_medium():
    """Invalid effort values fall back to 'medium'."""
    with patch.dict(os.environ, {"OUROBOROS_EFFORT_TASK": "extreme"}, clear=True):
        assert resolve_effort("task") == "medium"


# ---------------------------------------------------------------------------
# Evolution and consciousness: the top of the owner's effort range
# ---------------------------------------------------------------------------

def test_evolution_effort_default_is_the_range_top():
    """An evolution task starts at the top of the range: the shipped `high`."""
    with patch.dict(os.environ, {}, clear=True):
        assert resolve_effort("evolution") == "high"


def test_evolution_effort_follows_the_range_top_even_above_high():
    """The top of the range decides, even above High; the retired role key is inert."""
    with patch.dict(os.environ, {"OUROBOROS_EFFORT_MAX": "ultra", "OUROBOROS_EFFORT_EVOLUTION": "medium"}, clear=True):
        assert resolve_effort("evolution") == "ultra"
    with patch.dict(os.environ, {"OUROBOROS_EFFORT_MAX": "medium"}, clear=True):
        assert resolve_effort("evolution") == "medium"


def test_consciousness_effort_is_the_range_top_never_below_recommended():
    """A wake-up starts at the top of the range; the tolerant read keeps that top at or
    above the recommended level, and the retired role key changes nothing."""
    with patch.dict(os.environ, {}, clear=True):
        assert resolve_effort("consciousness") == "high"
    with patch.dict(os.environ, {"OUROBOROS_EFFORT_TASK": "xhigh"}, clear=True):
        assert resolve_effort("consciousness") == "xhigh"
    with patch.dict(os.environ, {"OUROBOROS_EFFORT_MAX": "max", "OUROBOROS_EFFORT_CONSCIOUSNESS": "none"}, clear=True):
        assert resolve_effort("consciousness") == "max"


# ---------------------------------------------------------------------------
# Case-insensitivity
# ---------------------------------------------------------------------------

def test_task_type_is_case_insensitive():
    """Task type matching is case-insensitive."""
    with patch.dict(os.environ, {}, clear=True):
        assert resolve_effort("EVOLUTION") == "high"
        assert resolve_effort("CONSCIOUSNESS") == "high"
