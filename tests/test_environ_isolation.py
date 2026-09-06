"""The autouse os.environ snapshot in conftest closes the env-leak class.

Any test may mutate the environment (apply_settings_to_env, direct writes);
the fixture must hand the next test the exact pre-test environment back, on
the REAL os._Environ mapping (a plain-dict swap would sever the putenv sync
subprocesses inherit from). The contract is pinned by driving the fixture's
own generator directly, so the proof is self-contained and order-free.
"""

from __future__ import annotations

import os

from tests.conftest import restored_os_environ


def test_restored_os_environ_reverts_mutations_on_the_real_mapping():
    canary = "OURO_TEST_LEAK_CANARY"
    assert canary not in os.environ
    gen = restored_os_environ()
    next(gen)
    os.environ[canary] = "leaked"
    try:
        next(gen)
    except StopIteration:
        pass
    assert canary not in os.environ, "the snapshot must revert any mutation"
    assert type(os.environ).__name__ == "_Environ", (
        "restore must mutate the real mapping, never swap in a plain dict"
    )


def test_restored_os_environ_restores_deleted_and_changed_values():
    key = "OURO_TEST_LEAK_BASELINE"
    os.environ[key] = "original"
    try:
        gen = restored_os_environ()
        next(gen)
        os.environ[key] = "mutated"
        del os.environ[key]
        try:
            next(gen)
        except StopIteration:
            pass
        assert os.environ.get(key) == "original"
    finally:
        os.environ.pop(key, None)


def test_the_snapshot_is_registered_autouse_for_every_test(request):
    """Pins the WIRING (autouse=True on the conftest fixture), not the helper
    body: without it the generator above is correct and never runs."""
    assert "_os_environ_isolation" in request.fixturenames
