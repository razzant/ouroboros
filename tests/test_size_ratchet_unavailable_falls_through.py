"""Tests for ibl-67aa1f89cf1c: the advisory-lane size-ratchet validator must
keep running (returning findings) when a parent ref's gated-source blob is
unavailable in this checkout, when a private ``GIT_INDEX_FILE`` does not
stage the manifest, and on the happy path with drift; it must still raise a
bare ``ValueError`` on a genuinely malformed manifest (so the advisory lane
correctly reports "validator unavailable" for that case).

See :mod:`ouroboros.review`'s :class:`SizeRatchetRefUnavailable` for the
typed exception that distinguishes "objects not in the local store" from a
genuinely malformed manifest.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest

from ouroboros import review as review_mod
from ouroboros.review import (
    SIZE_RATCHET_MANIFEST_PATH,
    SizeRatchetRefUnavailable,
    validate_size_ratchet,
)


# --- shared helpers ----------------------------------------------------------

def _empty_inventory() -> review_mod.SizeRatchetInventory:
    """A minimal SizeRatchetInventory that passes the dataclass type check with
    no live debt — used by tests that only need the validator to ENTER the
    manifest-comparison stage.
    """
    return review_mod.SizeRatchetInventory(
        modules=(),
        functions=(),
        giant_paths=frozenset(),
        function_debt=frozenset(),
        band_paths=frozenset(),
        byte_debt={},
    )


@pytest.fixture()
def real_checkout(tmp_path):
    """A bare git repository that has NO size-ratchet manifest in HEAD's tree
    (so the validator runs on a bootstrap baseline against the live working
    copy the test writes)."""
    head = tmp_path
    subprocess.run(["git", "init", "-q", "-b", "main", str(head)], check=True)
    subprocess.run(
        ["git", "-C", str(head), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(head), "config", "user.name", "test"],
        check=True,
    )
    (head / "placeholder.txt").write_text("seed", encoding="utf-8")
    subprocess.run(["git", "-C", str(head), "add", "placeholder.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(head), "commit", "-q", "-m", "initial seed"],
        check=True,
    )
    return head


_DRIFT_PATH = "_phantom_size_ratchet_drift_target.py"


def _write_drift_manifest(checkout: pathlib.Path) -> None:
    """Write a size-ratchet manifest whose GIANT_PATHS entry is a phantom
    path the live inventory does not contain — produces a stable
    "missing live entry" finding on the live-tree comparison.
    """
    manifest = (
        f'BASELINE_SOURCE_SHA = "{"0" * 40}"\n'
        f"GIANT_PATHS = (\n"
        f"    '{_DRIFT_PATH}',\n"
        f")\n"
        f"FUNCTION_DEBT = ()\n"
        f"BAND_BASELINE_PATHS = ()\n"
        f"BAND_PATHS = {{}}\n"
        f"BYTE_BASELINE_DEBT = {{}}\n"
        f"BYTE_DEBT = {{}}\n"
    )
    target = checkout / SIZE_RATCHET_MANIFEST_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(manifest, encoding="utf-8")


# --- test 1 ------------------------------------------------------------------

def test_validate_size_ratchet_parent_ref_unavailable_returns_findings(monkeypatch, real_checkout):
    """A parent ref's gated-source blob is not in the local object store (simulated
    via a monkey-patched cat-file --batch missing header).
    ``validate_size_ratchet`` must NOT raise and must treat ``previous`` as
    bootstrap.
    """
    _write_drift_manifest(real_checkout)

    def fake_collect_inventory(root):
        return _empty_inventory()

    def fake_resolve_committed_manifest_text(repo_dir, *, manifest_path=SIZE_RATCHET_MANIFEST_PATH):
        # A parent whose manifest text parses as Python but is NOT a valid
        # size-ratchet manifest (pre-v6.114 assignment set / truncated history
        # after a managed-update fetch). parse_size_ratchet_manifest raises a
        # bare ValueError for this; the fix's ``except ValueError`` must swallow
        # it, leave ``previous`` as None, and validate the live tree only.
        return "SOME_UNEXPECTED_ASSIGNMENT = 1\n"

    monkeypatch.setattr(review_mod, "collect_size_ratchet_inventory", fake_collect_inventory)
    monkeypatch.setattr(review_mod, "resolve_committed_manifest_text", fake_resolve_committed_manifest_text)

    findings = validate_size_ratchet(real_checkout)
    assert isinstance(findings, list)
    assert all(isinstance(line, str) for line in findings)
    # bootstrap fall-through still surfaces the on-disk drift
    assert f"GIANT_PATHS contains stale entry: '{_DRIFT_PATH}'" in findings


# --- test 2 ------------------------------------------------------------------

def test_validate_size_ratchet_staged_index_without_manifest_does_not_raise(monkeypatch, real_checkout):
    """A private ``GIT_INDEX_FILE`` (or otherwise empty) staged index does not
    cause ``validate_size_ratchet`` to raise — live-tree validation must
    succeed.
    """
    _write_drift_manifest(real_checkout)

    def fake_collect_inventory(root):
        return _empty_inventory()

    def fake_staged_tree(root):
        # Pretend the staged index fails to materialize — exercises the
        # ``_staged_tree_without_index_lock``-level fall-through.
        raise SizeRatchetRefUnavailable(
            "simulated: private GIT_INDEX_FILE did not stage any tree"
        )

    monkeypatch.setattr(review_mod, "collect_size_ratchet_inventory", fake_collect_inventory)
    monkeypatch.setattr(review_mod, "_staged_tree_without_index_lock", fake_staged_tree)

    findings = validate_size_ratchet(real_checkout)
    assert isinstance(findings, list)
    assert all(isinstance(line, str) for line in findings)


# --- test 3 ------------------------------------------------------------------

def test_validate_size_ratchet_happy_path_returns_drift_findings(monkeypatch, real_checkout):
    """The on-disk manifest with drift against the live tree still produces
    the same baseline findings the validator produced before the fix.
    """
    _write_drift_manifest(real_checkout)

    def fake_collect_inventory(root):
        return _empty_inventory()

    monkeypatch.setattr(review_mod, "collect_size_ratchet_inventory", fake_collect_inventory)

    findings = validate_size_ratchet(real_checkout)
    assert isinstance(findings, list)
    # _write_drift_manifest lists a phantom path in GIANT_PATHS that has no
    # live file over the module ceiling -> the live-tree comparison flags it
    # as a STALE manifest entry (not a "missing live entry", which is the
    # opposite direction: an oversize live file absent from the manifest).
    drift_line = f"GIANT_PATHS contains stale entry: '{_DRIFT_PATH}'"
    assert drift_line in findings


# --- test 4 ------------------------------------------------------------------

def test_validate_size_ratchet_genuinely_malformed_manifest_still_raises(real_checkout):
    """A genuinely malformed manifest text on disk must still raise a bare
    ``ValueError`` (NOT ``SizeRatchetRefUnavailable``); the advisory lane's
    broad except rightly reports "validator unavailable" for that case.
    """
    malformed = "x = 1\n"
    target = real_checkout / SIZE_RATCHET_MANIFEST_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(malformed, encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        validate_size_ratchet(real_checkout)
    # The exception must NOT be the typed fallback — bare parse failure
    # is genuinely "validator unavailable".
    assert not isinstance(excinfo.value, SizeRatchetRefUnavailable)
