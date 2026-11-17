"""Self-pin of the root conftest's clock guard: a product module that binds ``deadline_utils.utc_now`` by name keeps
the clock that stood there at its first import, so a module first imported while a test had replaced that clock would
carry the test's clock into every later test of the process. The guard lives in tests/conftest.py; this file
exercises it directly, in both directions, and checks that every test runs under it."""
from __future__ import annotations

import pathlib
import sys
import types
from datetime import datetime, timezone

from ouroboros import deadline_utils, utils

conftest = next(
    m for m in list(sys.modules.values())
    if getattr(m, "__file__", "")
    and pathlib.Path(m.__file__).as_posix().endswith("tests/conftest.py")
    and hasattr(m, "_give_late_imports_the_real_clock")
)


def _imported(monkeypatch, name, clock):
    """A module as if first imported just now, with ``clock`` bound to its ``utc_now``."""
    module = types.ModuleType(name)
    module.utc_now = clock
    monkeypatch.setitem(sys.modules, name, module)
    conftest._CLOCK_CHECKED.discard(name)
    return module


def test_every_test_runs_under_the_clock_guard(request):
    assert "_late_imports_keep_the_real_clock" in request.fixturenames


def test_a_module_first_imported_under_a_tests_clock_gets_the_real_one_back(monkeypatch):
    def fake():  # what ``monkeypatch.setattr(deadline_utils, "utc_now", ...)`` leaves in a late import
        return datetime.fromtimestamp(100.0, timezone.utc)

    poisoned = _imported(monkeypatch, "ouroboros._clock_guard_probe", fake)
    supervised = _imported(monkeypatch, "supervisor._clock_guard_probe", lambda: datetime.fromtimestamp(0, timezone.utc))
    conftest._give_late_imports_the_real_clock()
    assert poisoned.utc_now is deadline_utils.utc_now and supervised.utc_now is deadline_utils.utc_now
    assert poisoned.utc_now().year >= 2026


def test_a_clock_a_product_module_defines_and_a_module_outside_the_product_are_left_alone(monkeypatch):
    def fake():
        return datetime.fromtimestamp(100.0, timezone.utc)

    own = _imported(monkeypatch, "ouroboros._clock_guard_own", utils.utc_now_iso)  # a product function under that name
    outside = _imported(monkeypatch, "tests._clock_guard_outside", fake)  # a test helper may keep its own clock
    conftest._give_late_imports_the_real_clock()
    assert own.utc_now is utils.utc_now_iso and outside.utc_now is fake


def test_nothing_is_handed_out_while_a_replacement_still_stands(monkeypatch):
    def fake():
        return datetime.fromtimestamp(100.0, timezone.utc)

    monkeypatch.setattr(deadline_utils, "utc_now", fake)
    late = _imported(monkeypatch, "ouroboros._clock_guard_late", fake)
    conftest._give_late_imports_the_real_clock()
    assert late.utc_now is fake  # the test's own clock is not undone under it
    assert "ouroboros._clock_guard_late" not in conftest._CLOCK_CHECKED  # so the next test's start still repairs it
