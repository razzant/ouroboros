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
# Evolution
# ---------------------------------------------------------------------------

def test_evolution_effort_default_is_high():
    """Default evolution effort is 'high'."""
    with patch.dict(os.environ, {}, clear=True):
        assert resolve_effort("evolution") == "high"


def test_evolution_effort_configurable():
    """Evolution effort can be overridden via OUROBOROS_EFFORT_EVOLUTION."""
    with patch.dict(os.environ, {"OUROBOROS_EFFORT_EVOLUTION": "medium"}, clear=True):
        assert resolve_effort("evolution") == "medium"


# ---------------------------------------------------------------------------
# Consciousness
# ---------------------------------------------------------------------------

def test_consciousness_effort_default_inherits_task_effort():
    """An empty consciousness slot is Main's effort: a wake-up is an ordinary Main turn."""
    with patch.dict(os.environ, {}, clear=True):
        assert resolve_effort("consciousness") == resolve_effort("task")
    with patch.dict(os.environ, {"OUROBOROS_EFFORT_TASK": "xhigh"}, clear=True):
        assert resolve_effort("consciousness") == "xhigh"
    with patch.dict(os.environ, {"OUROBOROS_EFFORT_TASK": "xhigh", "OUROBOROS_EFFORT_CONSCIOUSNESS": "bogus"}, clear=True):
        assert resolve_effort("consciousness") == "xhigh"  # an invalid value is treated as empty


def test_consciousness_effort_configurable():
    """Consciousness effort can be overridden via OUROBOROS_EFFORT_CONSCIOUSNESS."""
    with patch.dict(os.environ, {"OUROBOROS_EFFORT_CONSCIOUSNESS": "none"}, clear=True):
        assert resolve_effort("consciousness") == "none"


# ---------------------------------------------------------------------------
# Case-insensitivity
# ---------------------------------------------------------------------------

def test_task_type_is_case_insensitive():
    """Task type matching is case-insensitive."""
    with patch.dict(os.environ, {}, clear=True):
        assert resolve_effort("EVOLUTION") == "high"
        assert resolve_effort("CONSCIOUSNESS") == resolve_effort("task")
