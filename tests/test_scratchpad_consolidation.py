"""Scratchpad consolidation is the integrating mind's fold, never an automatic pass (#918).

Verifies:
- the automatic writer is gone: no 30 000-character threshold, no ``should_consolidate_scratchpad``
  predicate, no ``consolidate_scratchpad`` Light pass and no post-task stage that runs one
  (the root post-task census in ``test_authored_memory`` and ``test_root_post_task_synthesis``
  shows the late phase buying no scratchpad call and leaving the blocks untouched)
- a pre-block flat scratchpad is never rewritten, by a fold or by an append
- the Light route the remaining Light operations use keeps an explicit local light model or
  inherits a local main one
"""
import importlib
import os

import pytest

from ouroboros.memory import Memory

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _get_consolidator():
    import sys
    sys.path.insert(0, REPO)
    return importlib.import_module("ouroboros.consolidator")


def test_no_threshold_predicate_or_automatic_writer_remains():
    """The former 30 000 threshold, its predicate and the Light pass it admitted."""
    from ouroboros import context_budget, post_task_synthesis

    mod = _get_consolidator()
    for name in ("SCRATCHPAD_CONSOLIDATION_THRESHOLD", "should_consolidate_scratchpad", "consolidate_scratchpad"):
        assert not hasattr(mod, name), name
    assert not hasattr(context_budget, "SCRATCHPAD_CONSOLIDATION_THRESHOLD_CHARS")
    assert not hasattr(post_task_synthesis, "_run_scratchpad_consolidation")


def test_flat_scratchpad_consolidation_is_retired():
    """Pre-block flat scratchpads require manual upgrade; they are not rewritten."""
    mod = _get_consolidator()
    assert not hasattr(mod, "_consolidate_scratchpad_flat")


@pytest.mark.parametrize("operation", ["fold", "append"])
def test_a_flat_scratchpad_is_never_rewritten(tmp_path, operation):
    drive = tmp_path / "data"
    (drive / "memory").mkdir(parents=True)
    (drive / "logs").mkdir(parents=True)
    mem = Memory(drive_root=drive)
    legacy = "legacy flat scratchpad\n" + ("x" * 35000)
    mem.scratchpad_path().write_text(legacy, encoding="utf-8")

    with pytest.raises(RuntimeError, match="LEGACY_SCRATCHPAD_REQUIRES_MANUAL_UPGRADE"):
        if operation == "fold":
            mem.fold_scratchpad_blocks("A fold over a flat file.", ["0" * 64])
        else:
            mem.append_scratchpad_block("An append over a flat file.")

    assert mem.scratchpad_path().read_text(encoding="utf-8") == legacy
    assert not mem.scratchpad_blocks_path().exists()


def test_consolidation_route_preserves_explicit_local_light(monkeypatch):
    mod = _get_consolidator()
    monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", "local-light")
    monkeypatch.setenv("USE_LOCAL_LIGHT", "true")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    assert mod._consolidation_route() == ("local-light", True)


def test_consolidation_route_empty_light_inherits_local_main(monkeypatch):
    mod = _get_consolidator()
    monkeypatch.setenv("OUROBOROS_MODEL", "local-main")
    monkeypatch.delenv("OUROBOROS_MODEL_LIGHT", raising=False)
    monkeypatch.setenv("USE_LOCAL_MAIN", "true")
    monkeypatch.delenv("USE_LOCAL_LIGHT", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    assert mod._consolidation_route() == ("local-main", True)
