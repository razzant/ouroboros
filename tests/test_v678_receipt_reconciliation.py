"""v6.78.0 (phase P4) — verification-receipt IDENTITY, reconciliation, and the
disclosures that describe them.

The pure half of phase P4, and the file the review rounds kept landing in. A receipt is
reconciled by ONE typed identity key; the outstanding sets count identities rather than
rows; every reviewer-facing projection of a receipt discloses what it bounded and names
the authority that actually decided. Each section below is the round that forced it,
kept because the defect is easier to re-introduce than to re-find. The red-receipt nudge
and the other consumers of these identities live in `test_v678_acceptance_state.py`.

Offline and layout-free: plain dicts through the real helpers, no ambient checkout state.
"""

from __future__ import annotations


import pytest


# --------------------------------------------------------------------------------------
# P4 review round — the three defects in the receipt-identity projections.
# --------------------------------------------------------------------------------------
# P4 review round 2 — "latest" was the wrong shape for "is anything still outstanding",
# and the disclosed-bounding fix had not reached every changed review surface.
# --------------------------------------------------------------------------------------
# P4 review round 3 — both findings are SECOND-ORDER effects of the rounds 1-2 fixes:
# the shared projection's `bound` callback had one field routed around it, and the
# disclosed URL bound was applied BEFORE the dedup it invalidated.


# --------------------------------------------------------------------------------------
# P4 review round 4 — ONE canonical identity. The phase carried two unreconciled notions
# of it: a normalized/sorted/set-shaped COMPARISON identity (`_reconciles`,
# `paths_identity_sha256`) and a raw/ordered/list-shaped PROJECTION identity (the carried
# items, the omitted counts, the outstanding rows). Every round-4 finding was a place
# where the two disagreed. `receipt_canonical_identity` is now the only derivation, and
# comparison, hashing, counting and projection all read it.
# --------------------------------------------------------------------------------------


def test_outstanding_counts_identities_not_rows():
    """Round-4: `_outstanding` returned outstanding ROWS where it documented outstanding
    IDENTITIES. Two consecutive failures of the same check with no green between them
    are ONE red verification; reporting two sent the reviewer chasing a second problem
    that did not exist. The kept row is the FRESHEST evidence for that identity."""
    from ouroboros import outcomes as O
    from ouroboros._outcome_receipts import unreconciled_failed
    from ouroboros.review_evidence import _accept_verification_summary

    first = {"status": "fail", "check": "pytest tests/a.py", "summary": "3 failed"}
    again = {"status": "fail", "check": "pytest  tests/a.py", "summary": "1 failed"}
    assert unreconciled_failed([first, again]) == [again]
    assert _accept_verification_summary([first, again])["unreconciled_red_count"] == 1
    assert _accept_verification_summary([first, again])["unreconciled_red"] is True
    # The freshest receipt represents it, so the reviewer reads the latest summary.
    assert O.latest_unreconciled_failed_receipt([first, again])["summary"] == "1 failed"
    # Same for the criterion-id and path-set identities.
    assert len(unreconciled_failed([
        {"status": "fail", "criterion_id": "c1", "check": "make a"},
        {"status": "fail", "criterion_id": "c1", "check": "make a -v"},
    ])) == 1
    assert len(unreconciled_failed([
        {"status": "fail", "paths": ["b.md", "report.md"]},
        {"status": "fail", "paths": ["report.md", "b.md"]},
    ])) == 1
    # Two DISTINCT reds are still two, and a repeat of one of them does not hide the other.
    red_b = {"status": "fail", "check": "pytest tests/b.py"}
    assert len(unreconciled_failed([first, red_b, again])) == 2
    assert _accept_verification_summary([first, red_b, again])["unreconciled_red_count"] == 2
    # Identity-LESS reds are indistinguishable, not known-equal: never merged (that would
    # UNDERCOUNT, and undercounting open reds is the direction that hides a problem).
    assert len(unreconciled_failed([{"status": "fail"}, {"status": "fail"}])) == 2


def test_receipt_check_text_round_trips_so_two_argvs_are_two_verifications(tmp_path, monkeypatch):
    """Round-4 follow-up (owner-authorised, `tools/verify.py`): the receipt stored its
    check as `" ".join(argv)`, which is not injective — argv `["echo","a b"]` and
    `["echo","a","b"]` rendered to the SAME text, so once the identity canonicalized
    that text a green on one could clear a red on the other. `shlex.join` is the exact
    inverse of the lexer the identity re-tokenizes with, so the stored text round-trips.

    Local, tmp_path-rooted harness — no ambient repo/data layout, no `__file__` paths.
    """
    import pathlib
    import shlex
    import shutil
    import sys

    from ouroboros import _outcome_receipts as R
    from ouroboros.outcomes import read_verification_receipts
    from ouroboros.shell_parse import shell_tokens
    from ouroboros.tools.registry import ToolRegistry

    # The property, independent of any filesystem: join is the lexer's inverse.
    for argv in (["echo", "a b"], ["echo", "a", "b"], ["sh", "-c", "pytest -q | tail"],
                 ["python", "-c", "assert v == 'a  b'"], ["pytest", "-m", ""]):
        assert shell_tokens(shlex.join(argv)) == argv

    # ...so the two argv spellings are two DIFFERENT verifications.
    assert R.receipt_identity({"check": shlex.join(["echo", "a b"])}) != \
        R.receipt_identity({"check": shlex.join(["echo", "a", "b"])})
    assert R.receipt_identity({"check": shlex.join(["pytest", "-m", ""])}) != \
        R.receipt_identity({"check": shlex.join(["pytest", "-m"])})

    if sys.platform == "win32" or not shutil.which("echo"):
        pytest.skip("the end-to-end half needs a POSIX `echo` binary")

    # End to end through the real tool, on a tmp_path-rooted registry.
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    home = tmp_path / "home"
    # NEUTRAL layout, not home/Ouroboros/data: the registry gets explicit
    # roots, and a live-SHAPED path under the (patched) home now trips the
    # pytest fail-closed guard in append_jsonl (issue #455 class).
    repo, data = tmp_path / "repo", tmp_path / "data"
    for directory in (home, repo, data):
        directory.mkdir(parents=True)
    monkeypatch.setattr(pathlib.Path, "home", lambda: home)
    registry = ToolRegistry(repo_dir=repo, drive_root=data)
    registry._ctx.task_id = "task1"

    registry.execute("verify_and_record", {
        "contract_kind": "explicit_command", "check": ["echo", "a b"],
        "expected": "a b", "cwd": str(repo),
    })
    registry.execute("verify_and_record", {
        "contract_kind": "explicit_command", "check": ["echo", "a", "b"],
        "expected": "a b", "cwd": str(repo),
    })
    receipts = read_verification_receipts(data, "task1")[-2:]
    assert [r["status"] for r in receipts] == ["pass", "pass"]
    # The stored texts differ, each re-tokenizes back to the argv that actually ran, and
    # each carries the stamp of the renderer that wrote it (round 8).
    assert receipts[0]["check"] != receipts[1]["check"]
    assert [r["check_rendering"] for r in receipts] == ["shlex_join", "shlex_join"]
    assert shell_tokens(receipts[0]["check"]) == ["echo", "a b"]
    assert shell_tokens(receipts[1]["check"]) == ["echo", "a", "b"]
    # Which is the point: a green on one no longer reconciles a red on the other.
    assert R.receipt_identity(receipts[0]) != R.receipt_identity(receipts[1])

    # #2919: `pytest -m ""` (no marker filter) reaches pytest through the guard and the
    # execution. Dropping the empty value made `-m` swallow the file name, deselecting both
    # tests; that argv stays a different, still-red verification.
    (repo / "test_marks.py").write_text(
        "import pytest\n\n@pytest.mark.slow\ndef test_marked():\n    pass\n\n"
        "def test_plain():\n    pass\n", encoding="utf-8")
    run = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-m"]
    for check in ([*run, "test_marks.py"], [*run, "", "test_marks.py"]):
        registry.execute("verify_and_record", {
            "contract_kind": "explicit_command", "check": check,
            "expected": "2 passed", "cwd": str(repo),
        })
    red, green = read_verification_receipts(data, "task1")[-2:]
    assert [red["status"], green["status"]] == ["fail", "pass"]
    assert shell_tokens(green["check"]) == [*run, "", "test_marks.py"]
    assert R.unreconciled_failed([red, green]) == [red]


# --------------------------------------------------------------------------------------
# P4 review round 5 — identity is ONE TYPED KEY, not a fallback chain. A chain over
# (criterion_id, check, paths) is not an equivalence relation: `{c1, check}` matched
# `{check}` and `{check}` matched `{c2, check}` while c1 and c2 are explicitly different
# criteria, so one check-only green reconciled two distinct reds and "collapse the
# candidates onto the identity they name" was not well defined. Keying makes sameness the
# kernel of a function — reflexive, symmetric, transitive — and makes an existing
# `criterion_id` authoritative structurally rather than by a rule someone must remember.
# --------------------------------------------------------------------------------------
# P4 review round 6 — the relation is sound; the PROJECTION was lying about which
# authority decided. `_reconciles_masked` never consults check text, yet the disclosure
# re-derived its own answer and told the acceptance reviewer that canonical check text
# controlled reconciliation. Host-attested evidence making a false claim about its own
# basis is the defect class this whole surface exists to eliminate. Fixed the same way
# rounds 4 and 5 were: the reporting path now READS the deciding path — one shared
# mode-aware key, `receipt_reconciliation_key`.
# --------------------------------------------------------------------------------------
# P4 review round 7 — the round-6 fix was made for ONE kind and never asked of the others.
# `expected_whitespace_normalized` still claimed canonical-command-text normalization for
# `artifact_paths`, whose set `canonical_path_set` compares byte-for-byte. The durable
# form is not a third patched branch: the per-kind answer moves INTO the closed kind
# table, so a fourth kind cannot be added without answering for itself.
# --------------------------------------------------------------------------------------
# P4 review round 8 — the `shlex.join` change was argued safe in ONE direction. An old and
# a new receipt for the SAME argv render differently and fail to reconcile, and a
# non-reconciling red stays red. But an old red and a new green from DIFFERENT argvs can
# render IDENTICALLY (`["echo","a b"]` and `["echo","a","b"]` both space-joined to
# `echo a b`), and that green cleared that red — a false green produced by the change made
# to remove one. A receipt did not record WHICH renderer wrote its check text, so the
# comparator could not tell the formats apart. It records it now.
# --------------------------------------------------------------------------------------
