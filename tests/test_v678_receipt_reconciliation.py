"""v6.78.0 (phase P4) — verification-receipt IDENTITY, reconciliation, and the
disclosures that describe them.

The pure half of phase P4, and the file the review rounds kept landing in. A receipt is
reconciled by ONE typed identity key; the outstanding sets count identities rather than
rows; every reviewer-facing projection of a receipt discloses what it bounded and names
the authority that actually decided. Each section below is the round that forced it,
kept because the defect is easier to re-introduce than to re-find. The loop/acceptance
wiring that consumes these flags lives in `test_v678_acceptance_state.py`.

Offline and layout-free: plain dicts through the real helpers, no ambient checkout state.
"""

from __future__ import annotations

import json

import pytest


# --------------------------------------------------------------------------------------
# P4 review round — the three defects in the receipt-identity projections.
# --------------------------------------------------------------------------------------


def test_bounded_path_set_discloses_its_omitted_count_and_a_durable_hash():
    """Defect 1: `paths[:20]` used to drop data with NO record that anything was
    dropped. The bound may stay; the SILENCE may not (BIBLE P1). Every bounded path
    set now reports how many it omitted, plus a hash of the FULL normalized identity
    set so the bounded list stays checkable against the whole thing."""
    import hashlib

    from ouroboros._outcome_receipts import (
        IDENTITY_PATH_LIMIT,
        receipt_canonical_identity,
        receipt_identity_projection,
        verification_receipt_ledger_row,
    )
    from ouroboros.review_evidence import _accept_verification_summary

    many = [f"artifacts/report_{i:03d}.md" for i in range(IDENTITY_PATH_LIMIT + 7)]
    receipt = {"status": "observed", "contract_kind": "artifact_observation", "paths": many}
    # The hash covers the CANONICAL set — the same object the carried items and the
    # omitted count come from (round 4), serialized injectively.
    full_identity = receipt_canonical_identity(receipt).paths_identity_source
    expected_hash = hashlib.sha256(full_identity.encode("utf-8")).hexdigest()

    proj = receipt_identity_projection(receipt)
    assert len(proj["paths"]) == IDENTITY_PATH_LIMIT
    assert proj["paths_omitted"] == 7
    assert proj["paths_identity_sha256"] == expected_hash

    # BOTH fixed reviewer projections disclose it — the ledger row...
    row = verification_receipt_ledger_row(receipt)
    assert row["paths_omitted"] == 7 and row["paths_identity_sha256"] == expected_hash
    # ...and the acceptance summary, for the latest receipt and the unreconciled red alike.
    summary = _accept_verification_summary([dict(receipt, status="fail")])
    assert summary["latest_identity"]["paths_omitted"] == 7
    assert summary["latest_identity"]["paths_identity_sha256"] == expected_hash
    assert summary["unreconciled_red_identity"]["paths_omitted"] == 7

    # An unbounded set still says so explicitly rather than leaving it to inference.
    assert receipt_identity_projection({"paths": ["a.md"]})["paths_omitted"] == 0
    # A receipt with no path set carries no path hash to claim.
    assert "paths_identity_sha256" not in receipt_identity_projection({"check": "pytest"})


def test_unreconciled_red_identity_reaches_the_reviewer_not_only_the_flag():
    """Defect 2: an EARLIER red left unreconciled by a LATER green of a DIFFERENT
    verification. The reviewer used to see `unreconciled_red=true` beside a green
    `latest_*` — a flag whose cause was nowhere in the packet. Assert the RED's own
    identity is projected, not merely that the flag is set."""
    from ouroboros.review_evidence import _accept_verification_summary

    red = {"status": "fail", "criterion_id": "c-integration",
           "check": "pytest tests/test_integration.py", "summary": "2 failed"}
    later_green = {"status": "pass", "criterion_id": "c-unit", "check": "pytest tests/test_unit.py"}
    summary = _accept_verification_summary([red, later_green])

    # The scenario is real: a different criterion's green does NOT clear the red.
    assert summary["unreconciled_red"] is True
    assert summary["latest_status"] == "pass"          # the LAST receipt is the green one
    assert summary["latest_identity"]["criterion_id"] == "c-unit"

    identity = summary["unreconciled_red_identity"]
    assert identity["criterion_id"] == "c-integration"
    assert identity["check"] == "pytest tests/test_integration.py"
    assert identity["reconciliation_identity"] == "criterion_id"
    # ...and the red identity is genuinely absent from the rest of the packet, so
    # projecting it is what makes the flag reconstructible at all.
    assert "c-integration" not in json.dumps({k: v for k, v in summary.items()
                                              if k != "unreconciled_red_identity"})

    # Same for the id-less artifact-observation class, whose identity is its path SET:
    # an earlier red observation cleared by neither a later green of another path set.
    red_obs = {"status": "fail", "paths": ["build/report.md"]}
    green_obs = {"status": "observed", "paths": ["build/other.md"]}
    obs = _accept_verification_summary([red_obs, green_obs])
    assert obs["unreconciled_red"] is True
    assert obs["unreconciled_red_identity"]["paths"] == ["build/report.md"]
    assert obs["unreconciled_red_identity"]["reconciliation_identity"] == "artifact_paths"

    # No unreconciled red -> no orphan identity block claiming one.
    cleared = _accept_verification_summary([red, dict(red, status="pass")])
    assert cleared["unreconciled_red"] is False
    assert "unreconciled_red_identity" not in cleared


def test_receipt_projections_bound_strings_through_the_ssot_helper_only():
    """Defect 3: the hand-rolled `clip()` (`t[:cap] + f'…[+N chars]'`) duplicated
    `utils.truncate_review_artifact` AND lost its anti-waste floor — on a one-character
    overflow the 11-char marker made the 'shortened' value 10 chars LONGER than the
    input. No hand-rolled slice-plus-marker may survive in the receipt modules."""
    import pathlib
    import re

    from ouroboros import _outcome_receipts as R
    from ouroboros.utils import truncate_review_artifact

    # The floor: a one-character overflow passes through WHOLE, never grows.
    for cap in (200, 300, 500):
        over = "p" * (cap + 1)
        bounded = R.receipt_identity_projection({"check": over}, check_cap=cap)["check"]
        assert bounded == over
        assert len(bounded) <= len(over)  # the old clip() returned cap + 11 here
    # A real overflow is still bounded, and DISCLOSED by the canonical marker.
    huge = "p" * 5000
    row = R.verification_receipt_ledger_row({"check": huge, "summary": huge, "expected": huge})
    for field, cap in (("check", 300), ("summary", 300), ("expected", 200)):
        assert row[field] == truncate_review_artifact(huge, limit=cap)
        assert "OMISSION NOTE" in row[field] and len(row[field]) < len(huge)

    # SSOT: the receipt modules hold no second truncation implementation.
    assert not hasattr(R, "clip")
    src = pathlib.Path(R.__file__).read_text(encoding="utf-8")
    assert re.search(r"\[:\s*cap\s*\]", src) is None, "hand-rolled string clip is back"


# --------------------------------------------------------------------------------------
# P4 review round 2 — "latest" was the wrong shape for "is anything still outstanding",
# and the disclosed-bounding fix had not reached every changed review surface.
# --------------------------------------------------------------------------------------


def test_a_newer_failure_can_never_erase_an_older_still_unreconciled_one():
    """Round-2 CRITICAL 1: a single latest-POINTER cannot answer "is anything still
    outstanding". Fail A, fail B, pass B used to report NO red at all — B displaced A,
    then B's own green cleared the pointer — so the acceptance evidence and the
    finalization nudge falsely said every verification was accounted for."""
    from ouroboros import outcomes as O
    from ouroboros._outcome_receipts import unreconciled_failed
    from ouroboros.review_evidence import _accept_verification_summary

    red_a = {"status": "fail", "check": "pytest tests/a.py"}
    red_b = {"status": "fail", "check": "pytest tests/b.py"}
    green_b = {"status": "pass", "check": "pytest tests/b.py"}

    assert unreconciled_failed([red_a, red_b, green_b]) == [red_a]
    still = O.latest_unreconciled_failed_receipt([red_a, red_b, green_b])
    assert still is not None and still["check"] == "pytest tests/a.py"

    # criterion-id variant (ids bind ahead of the command text)
    assert O.latest_unreconciled_failed_receipt([
        {"status": "fail", "criterion_id": "c1", "check": "make check"},
        {"status": "fail", "criterion_id": "c2", "check": "make check"},
        {"status": "pass", "criterion_id": "c2", "check": "make check"},
    ])["criterion_id"] == "c1"
    # path-set variant (the command-less artifact-observation class)
    assert O.latest_unreconciled_failed_receipt([
        {"status": "fail", "paths": ["report.md"]},
        {"status": "fail", "paths": ["summary.md"]},
        {"status": "observed", "paths": ["summary.md"]},
    ])["paths"] == ["report.md"]

    # BOTH outstanding reds survive, and the reviewer is told there are two — the newest
    # one's identity alone would read as if it were the only one.
    both = unreconciled_failed([red_a, red_b])
    assert both == [red_a, red_b]
    summary = _accept_verification_summary([red_a, red_b, green_b])
    assert summary["unreconciled_red"] is True and summary["unreconciled_red_count"] == 1
    assert summary["unreconciled_red_identity"]["check"] == "pytest tests/a.py"
    assert _accept_verification_summary([red_a, red_b])["unreconciled_red_count"] == 2
    # ...and a genuinely clean history still reports nothing outstanding.
    assert _accept_verification_summary([red_a, green_b, {"status": "pass", "check": "pytest tests/a.py"}])[
        "unreconciled_red_count"
    ] == 0


def test_a_cleanly_reconciled_masked_check_does_not_take_an_older_one_with_it():
    """Round-2 CRITICAL 1, masked variant: masked c1, masked c2, clean c2 used to
    report no masked check outstanding, because c2 had displaced c1 in the single
    latest-pointer before its own clean re-grounding cleared it."""
    from ouroboros import outcomes as O
    from ouroboros._outcome_receipts import unreconciled_masked
    from ouroboros.review_evidence import _accept_verification_summary

    m1 = {"status": "pass", "criterion_id": "c1", "check": "make a | tail", "check_exit_masking": True}
    m2 = {"status": "pass", "criterion_id": "c2", "check": "make b | tail", "check_exit_masking": True}
    clean2 = {"status": "pass", "criterion_id": "c2", "check": "make b"}

    assert unreconciled_masked([m1, m2, clean2]) == [m1]
    assert O.latest_unreconciled_masked_pass([m1, m2, clean2])["criterion_id"] == "c1"
    summary = _accept_verification_summary([m1, m2, clean2])
    assert summary["check_exit_masking_unreconciled"] is True
    assert summary["check_exit_masking_unreconciled_count"] == 1
    # Round 5: an id-LESS clean grounding no longer clears identified masked checks —
    # it carries no criterion key, and a key that is absent matches nothing.
    assert unreconciled_masked([m1, m2, {"status": "pass", "check": "make lint"}]) == [m1, m2]


def test_every_bounded_list_on_a_changed_review_surface_discloses_its_omissions():
    """Round-2 CRITICAL 2: the ledger projection kept hand-rolled `[:50]`/`[:10]` slices
    that destroyed evidence with no record that evidence was destroyed (BIBLE P1). All
    of them now run through the ONE shared disclosed-list projection."""
    from ouroboros._outcome_receipts import verification_receipt_ledger_row
    from ouroboros.review_evidence import _accept_verification_summary

    receipt = {
        "status": "pass",
        "check_exit_masking": True,
        "artifact_lifecycle": [{"path": f"out/{i}.bin", "exists": False} for i in range(57)],
        "artifacts_missing_after": [f"out/{i}.bin" for i in range(57)],
        "check_exit_masking_reasons": [f"reason-{i}" for i in range(14)],
    }
    row = verification_receipt_ledger_row(receipt)
    assert len(row["artifact_lifecycle"]) == 50 and row["artifact_lifecycle_omitted"] == 7
    assert len(row["artifacts_missing_after"]) == 50 and row["artifacts_missing_after_omitted"] == 7
    assert len(row["check_exit_masking_reasons"]) == 10
    assert row["check_exit_masking_reasons_omitted"] == 4
    # A list that fits still states its zero rather than leaving it to inference.
    small = verification_receipt_ledger_row({"status": "pass", "artifacts_missing_after": ["a"]})
    assert small["artifacts_missing_after_omitted"] == 0 and small["artifact_lifecycle_omitted"] == 0
    # Structural lifecycle flags survive the projection; only the path is bounded.
    assert row["artifact_lifecycle"][0] == {"path": "out/0.bin", "exists": False}
    long_path = verification_receipt_ledger_row(
        {"status": "pass", "artifact_lifecycle": [{"path": "p" * 5000, "exists": True}]},
    )["artifact_lifecycle"][0]
    assert "OMISSION NOTE" in long_path["path"] and long_path["exists"] is True

    # The acceptance summary's own masking-reason list was capped the same silent way.
    summary = _accept_verification_summary([receipt])
    assert len(summary["check_exit_masking_reasons"]) == 10
    assert summary["check_exit_masking_reasons_omitted"] == 4


def test_agent_supplied_review_prose_is_disclosed_truncated_not_sliced():
    """Same P1 rule on the acceptance-decision surface this phase rewrote: an agent
    rationale / obligation-disposition reason that ends mid-argument must say so."""
    import pathlib

    from ouroboros import loop_tool_execution as LTE

    src = pathlib.Path(LTE.__file__).read_text(encoding="utf-8")
    assert 'or "")[:500]' not in src, "hand-rolled silent slice on a review surface"
    assert src.count("truncate_review_artifact(") >= 2


# --------------------------------------------------------------------------------------
# P4 review round 3 — both findings are SECOND-ORDER effects of the rounds 1-2 fixes:
# the shared projection's `bound` callback had one field routed around it, and the
# disclosed URL bound was applied BEFORE the dedup it invalidated.


def test_agent_supplied_criterion_id_is_bounded_and_redacted_exactly_like_check():
    """Round-3 SECURITY: `criterion_id` comes from the agent's own `verify_and_record`
    call, so inserting it RAW into the shared identity projection routed it around the
    caller's `bound` — on the acceptance path the REDACTING `_accept_redact_cap` — and
    let unbounded agent text reach external reviewers unredacted (and overflow the
    pack's immutable core). It must be treated byte-for-byte like its `check` sibling."""
    from ouroboros._outcome_receipts import receipt_identity_projection
    from ouroboros.review_evidence import _accept_verification_summary
    from ouroboros.utils import truncate_review_artifact

    oversized = "crit-" + "z" * 5000
    proj = receipt_identity_projection({"criterion_id": oversized, "check": oversized})
    # Bounded through the SSOT truncator on the SAME cap as `check`, and DISCLOSED.
    assert proj["criterion_id"] == truncate_review_artifact(oversized, limit=300)
    assert proj["criterion_id"] == proj["check"]
    assert "OMISSION NOTE" in proj["criterion_id"]
    assert len(proj["criterion_id"]) < len(oversized)

    # The reviewer path: the caller's redacting `bound` must reach it too. A secret
    # pasted into the id is masked exactly as it is in the check text beside it.
    secret = "sk-ant-" + "a" * 40
    summary = _accept_verification_summary([
        {"status": "fail", "criterion_id": f"crit {secret}", "check": f"pytest {secret}"},
    ])
    identity = summary["latest_identity"]
    assert secret not in json.dumps(summary)
    assert "***REDACTED***" in identity["criterion_id"]
    assert "***REDACTED***" in identity["check"]

    # An oversized id on the reviewer path is bounded at the acceptance cap, not carried.
    big = _accept_verification_summary([{"status": "fail", "criterion_id": oversized}])
    assert len(big["latest_identity"]["criterion_id"]) < len(oversized)
    assert "OMISSION NOTE" in big["latest_identity"]["criterion_id"]


# --------------------------------------------------------------------------------------
# P4 review round 4 — ONE canonical identity. The phase carried two unreconciled notions
# of it: a normalized/sorted/set-shaped COMPARISON identity (`_reconciles`,
# `paths_identity_sha256`) and a raw/ordered/list-shaped PROJECTION identity (the carried
# items, the omitted counts, the outstanding rows). Every round-4 finding was a place
# where the two disagreed. `receipt_canonical_identity` is now the only derivation, and
# comparison, hashing, counting and projection all read it.
# --------------------------------------------------------------------------------------


def test_check_identity_is_structural_so_quoted_whitespace_is_never_collapsed():
    """Round-4 CRITICAL: the identity normalized checks with `" ".join(check.split())`,
    which collapses whitespace INSIDE quoted arguments too. Two checks that assert
    DIFFERENT things compared equal, so a later PASS could clear an unrelated failed
    verification — a false green in the very system built to prevent one."""
    from ouroboros import _outcome_receipts as R
    from ouroboros import outcomes as O

    wide = """python -c "assert value == 'a  b'" """.strip()
    narrow = """python -c "assert value == 'a b'" """.strip()

    # The two are DIFFERENT verifications: the quoted argument is not whitespace noise.
    assert R.receipt_identity({"check": wide}) != R.receipt_identity({"check": narrow})
    assert O.latest_unreconciled_failed_receipt([
        {"status": "fail", "check": wide},
        {"status": "pass", "check": narrow},
    ]) is not None
    # ...and the red IS cleared by a genuine re-run of its own check, however spaced.
    assert O.latest_unreconciled_failed_receipt([
        {"status": "fail", "check": wide},
        {"status": "pass", "check": "python   -c    \"assert value == 'a  b'\""},
    ]) is None

    # Cosmetic spacing between TOKENS still normalizes (the whole point of the rule)...
    assert R.receipt_identity({"check": "go   test"}) == R.receipt_identity({"check": "go test"})
    # ...and an argv-quoted argument equals its shell-quoted spelling.
    assert R.receipt_identity({"check": ["echo", "a  b"]}) == R.receipt_identity({"check": "echo 'a  b'"})
    # Control operators are part of the command, never folded into one another.
    assert R.receipt_identity({"check": "make a && make b"}) != R.receipt_identity(
        {"check": "make a || make b"})
    # An unlexable check (unbalanced quote) falls back to its stripped raw text: it may
    # fail to equate two spellings, but it can never equate two different commands.
    assert R.receipt_identity({"check": '  pytest "x  '}) == _check_key('pytest "x')
    assert R.receipt_identity({"check": '  pytest "x  '}) != R.receipt_identity({"check": 'pytest "y'})


def test_path_identity_keeps_every_whitespace_byte_of_a_filename():
    """Round-4 CRITICAL, path half: `" ".join(split())` conflated distinct filenames
    containing repeated spaces, so an observation of one file reconciled a red
    observation of another. Round 5, same defect at the EDGES: the set was still
    `.strip()`ed, and a leading or trailing space is a legal filename byte."""
    from ouroboros import _outcome_receipts as R
    from ouroboros import outcomes as O

    wide, narrow = "notes/my  report.md", "notes/my report.md"
    assert R.canonical_path_set([wide]) != R.canonical_path_set([narrow])
    assert O.latest_unreconciled_failed_receipt([
        {"status": "fail", "paths": [wide]},
        {"status": "observed", "paths": [narrow]},
    ]) is not None
    # Adversarial: a filename with leading AND trailing spaces is its own file.
    spaced = "  notes/report.md  "
    assert R.canonical_path_set([spaced]) == (spaced,)
    assert R.canonical_path_set([spaced]) != R.canonical_path_set([spaced.strip()])
    assert O.latest_unreconciled_failed_receipt([
        {"status": "fail", "paths": [spaced]},
        {"status": "observed", "paths": [spaced.strip()]},
    ]) is not None
    # ...and it is still reconciled by an observation of that very file.
    assert O.latest_unreconciled_failed_receipt([
        {"status": "fail", "paths": [spaced]},
        {"status": "observed", "paths": [spaced]},
    ]) is None


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


def test_repeated_masked_runs_of_one_criterion_count_as_one_open_masked_check():
    """Round-4, masked half of the same defect: the masked-receipt count had it too."""
    from ouroboros._outcome_receipts import unreconciled_masked
    from ouroboros.review_evidence import _accept_verification_summary

    m1 = {"status": "pass", "criterion_id": "c1", "check": "make a | tail", "check_exit_masking": True}
    m1_again = {"status": "pass", "criterion_id": "c1", "check": "make a | head", "check_exit_masking": True}
    m2 = {"status": "pass", "criterion_id": "c2", "check": "make b | tail", "check_exit_masking": True}

    assert unreconciled_masked([m1, m1_again]) == [m1_again]
    assert _accept_verification_summary([m1, m1_again])["check_exit_masking_unreconciled_count"] == 1
    assert _accept_verification_summary([m1, m1_again])["check_exit_masking_unreconciled"] is True
    # Two distinct criteria stay two.
    assert len(unreconciled_masked([m1, m2, m1_again])) == 2
    # An id-LESS masked pass is not folded into an id-carrying one (they may well be
    # different criteria — `_reconciles_masked`'s any-later-clean fallback is not equality).
    assert len(unreconciled_masked([m1, {"status": "pass", "check": "make z | tail",
                                         "check_exit_masking": True}])) == 2


def test_carried_paths_the_omitted_count_and_the_hash_describe_one_set():
    """Round-4: `paths_identity_sha256` hashed the normalized sorted set while `paths`
    and `paths_omitted` came from the RAW ordered list, so duplicates and whitespace
    variants claimed omissions that did not exist in the hashed identity. All three now
    derive from `receipt_canonical_identity`."""
    import hashlib

    from ouroboros._outcome_receipts import (
        IDENTITY_PATH_LIMIT,
        receipt_canonical_identity,
        receipt_identity_projection,
        verification_receipt_ledger_row,
    )

    # 23 raw entries, but only 3 distinct paths: the rest are exact duplicates (a
    # whitespace VARIANT is a different file and is deliberately not folded in).
    raw = ["a.md", "a.md", "a.md"] * 7 + ["b.md", "c.md"]
    receipt = {"status": "observed", "contract_kind": "artifact_observation", "paths": raw}
    identity = receipt_canonical_identity(receipt)
    assert identity.paths == ("a.md", "b.md", "c.md")

    proj = receipt_identity_projection(receipt)
    assert proj["paths"] == ["a.md", "b.md", "c.md"]
    # The old code carried 20 raw entries and claimed 4 omitted from a 3-element identity.
    assert proj["paths_omitted"] == 0
    assert proj["paths_identity_sha256"] == hashlib.sha256(
        identity.paths_identity_source.encode("utf-8")).hexdigest()
    # The invariant, stated once: carried + omitted == the size of the hashed identity.
    assert len(proj["paths"]) + proj["paths_omitted"] == len(identity.paths)

    # And it still holds when the identity genuinely exceeds the bound.
    big = {"paths": [f"p{i:03d}.md" for i in range(IDENTITY_PATH_LIMIT + 5)] * 2}
    big_proj = receipt_identity_projection(big)
    assert len(big_proj["paths"]) + big_proj["paths_omitted"] == IDENTITY_PATH_LIMIT + 5

    # A path may contain a newline; the hash must still separate two different sets
    # (a line-joined source would hash ["a\nb"] and ["a", "b"] identically).
    assert receipt_identity_projection({"paths": ["a\nb"]})["paths_identity_sha256"] != \
        receipt_identity_projection({"paths": ["a", "b"]})["paths_identity_sha256"]

    # The ledger row's `artifacts_missing_after` is a path SET too — same derivation.
    row = verification_receipt_ledger_row({"status": "pass", "artifacts_missing_after": raw})
    assert row["artifacts_missing_after"] == ["a.md", "b.md", "c.md"]
    assert row["artifacts_missing_after_omitted"] == 0


def test_missing_after_paths_are_deduplicated_before_they_are_rendered():
    """Round-4: `_missing_after` applied redaction and truncation BEFORE set dedup, so
    two DISTINCT long paths with the same rendered prefix collapsed into one while
    `artifacts_missing_after_omitted` still reported 0 — the exact ordering mistake
    already fixed once in `fold_retrieval_usage`. The rule, not the instance:
    canonicalize the raw set first, render and bound it second."""
    from ouroboros.review_evidence import _accept_verification_summary

    stem = "build/" + "d" * 400
    first, second = stem + "/alpha.bin", stem + "/omega.bin"
    assert len(first) == len(second) and first != second

    summary = _accept_verification_summary([
        {"status": "pass", "artifacts_missing_after": [first, second]},
    ])
    carried = summary["artifacts_missing_after"]
    # The two render byte-identically — which is what made the loss silent.
    assert len(set(carried)) == 1
    # The invariant: 2 distinct missing artifacts, fully carried OR exactly counted.
    assert len(carried) + summary["artifacts_missing_after_omitted"] == 2
    assert summary["artifacts_missing_after_any"] is True

    # A genuine repeat across receipts is still one path, not a phantom second one.
    repeated = _accept_verification_summary([
        {"status": "pass", "artifacts_missing_after": ["out/x.bin"]},
        {"status": "pass", "artifacts_missing_after": ["out/x.bin"]},
    ])
    assert repeated["artifacts_missing_after"] == ["out/x.bin"]
    assert repeated["artifacts_missing_after_omitted"] == 0
    # Redaction still reaches these paths (they are raw host command surface).
    secret = "sk-ant-" + "a" * 40
    redacted = _accept_verification_summary([
        {"status": "pass", "artifacts_missing_after": [f"out/{secret}.bin"]},
    ])
    assert secret not in json.dumps(redacted)


def test_canonical_identity_is_the_only_derivation_in_the_receipt_module():
    """The class, not the three instances: no second normalizer may reappear beside
    `receipt_canonical_identity` / `canonical_path_set`. A hand-rolled
    `" ".join(x.split())` on receipt text is exactly what made round 4's false green."""
    import ast
    import pathlib

    from ouroboros import _outcome_receipts as R

    src = pathlib.Path(R.__file__).read_text(encoding="utf-8")
    # AST, not text: the anti-pattern is DESCRIBED in the docstrings on purpose, so a
    # regex over the source would only pin the prose. No `<sep>.join(<x>.split())` CALL
    # may exist in this module — that transformation is what collapsed quoted whitespace.
    for node in ast.walk(ast.parse(src)):
        if not (isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "join"):
            continue
        inner = node.args[0] if node.args else None
        assert not (
            isinstance(inner, ast.Call) and getattr(inner.func, "attr", "") == "split"
        ), "lossy re-normalizer is back"
    # The check identity comes from the SHARED shell seam, not a private copy.
    assert "canonical_command_text" in src
    assert R.receipt_canonical_identity({"check": "go   test"}).check == "go test"


def test_claim_support_refs_disclose_their_bounds_and_dedup_before_rendering():
    """Round-4 call-site audit: `_accept_claim_support_refs` is the ELEVENTH surface —
    it bounded three lists with hand-rolled `[:5]`/`[-5:]` and rendered its path set
    before deduplicating it, the same two defects the identity projection just lost."""
    from ouroboros.review_evidence import _accept_claim_support_refs

    stem = "build/" + "e" * 400
    first, second = stem + "/alpha.bin", stem + "/omega.bin"
    contract = {"acceptance_claims": [{"id": "c1", "claim": "it builds", "support": "the log"}]}
    receipts = [
        {"status": "pass", "criterion_id": "c1", "check": f"make {i}"} for i in range(8)
    ]
    receipts[-1] = {
        "status": "pass", "criterion_id": "c1", "check": "make 7",
        "artifacts_missing_after": [first, second, "out/x.bin", "out/x.bin"],
        "artifact_lifecycle": [{"path": f"out/{i}.bin", "exists_after": False} for i in range(9)],
    }

    row = _accept_claim_support_refs(contract, receipts)[0]
    # The receipt window is bounded — and now says by how much.
    assert len(row["support_refs"]) == 5 and row["support_refs_omitted"] == 3
    ref = row["support_refs"][-1]
    # The lifecycle list is bounded through the shared helper, with its own count.
    assert len(ref["artifact_lifecycle"]) == 5 and ref["artifact_lifecycle_omitted"] == 4
    # The path set is canonicalized on the RAW values first: the duplicate and the
    # whitespace variant collapse honestly, the two prefix-colliding paths do not vanish.
    carried = ref["artifacts_missing_after"]
    assert len(carried) + ref["artifacts_missing_after_omitted"] == 3
    assert "out/x.bin" in carried
    # Redaction still reaches both lists.
    secret = "sk-ant-" + "b" * 40
    redacted = _accept_claim_support_refs(contract, [{
        "status": "pass", "criterion_id": "c1",
        "artifacts_missing_after": [f"out/{secret}.bin"],
        "artifact_lifecycle": [{"path": f"out/{secret}.bin", "exists_after": False}],
    }])
    assert secret not in json.dumps(redacted)


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
                 ["python", "-c", "assert v == 'a  b'"]):
        assert shell_tokens(shlex.join(argv)) == argv

    # ...so the two argv spellings are two DIFFERENT verifications.
    assert R.receipt_identity({"check": shlex.join(["echo", "a b"])}) != \
        R.receipt_identity({"check": shlex.join(["echo", "a", "b"])})

    if sys.platform == "win32" or not shutil.which("echo"):
        pytest.skip("the end-to-end half needs a POSIX `echo` binary")

    # End to end through the real tool, on a tmp_path-rooted registry.
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    home = tmp_path / "home"
    repo, data = home / "Ouroboros" / "repo", home / "Ouroboros" / "data"
    for directory in (repo, data):
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


# --------------------------------------------------------------------------------------
# P4 review round 5 — identity is ONE TYPED KEY, not a fallback chain. A chain over
# (criterion_id, check, paths) is not an equivalence relation: `{c1, check}` matched
# `{check}` and `{check}` matched `{c2, check}` while c1 and c2 are explicitly different
# criteria, so one check-only green reconciled two distinct reds and "collapse the
# candidates onto the identity they name" was not well defined. Keying makes sameness the
# kernel of a function — reflexive, symmetric, transitive — and makes an existing
# `criterion_id` authoritative structurally rather than by a rule someone must remember.
# --------------------------------------------------------------------------------------


def _check_key(text: str, rendering: str = "unversioned") -> tuple[str, str]:
    """The typed identity key of a receipt whose check canonicalizes to `text`, spelled
    out rather than derived — the check identity is the RENDERING paired with the text
    (round 8), and a test that computed it through the code under test could not catch
    the pairing being dropped again."""
    return ("check", json.dumps([rendering, text]))


def _keys(receipts):
    from ouroboros._outcome_receipts import receipt_identity

    return {receipt_identity(r) for r in receipts}


def test_a_check_only_green_cannot_clear_a_criterion_keyed_red():
    """Round-5 CRITICAL: two failed receipts with DIFFERENT `criterion_id`s and the same
    check text were both reconciled by one later passing receipt that omitted the id,
    even though the ids explicitly distinguish the criteria."""
    from ouroboros import _outcome_receipts as R

    check = "pytest tests/x.py"
    c1 = {"status": "fail", "criterion_id": "c1", "check": check}
    c2 = {"status": "fail", "criterion_id": "c2", "check": check}
    green = {"status": "pass", "check": check}

    # Different KINDS never match, so neither red is touched.
    assert R.unreconciled_failed([c1, c2, green]) == [c1, c2]
    # The named ids are what keeps them apart, and they keep them apart from each other.
    assert _keys([c1, c2, green]) == {
        ("criterion_id", "c1"), ("criterion_id", "c2"), _check_key(check)}
    # Each is still cleared by a green naming ITS key — however the command is spelled.
    assert R.unreconciled_failed([c1, c2, {"status": "pass", "criterion_id": "c1",
                                           "check": "pytest tests/x.py -v"}]) == [c2]
    # The documented cost, stated as a test: a re-run that OMITS the id it carried before
    # no longer clears its own red. Strictly fewer reconciliations — the SAFE direction.
    assert R.unreconciled_failed([c1, green]) == [c1]


def test_same_verification_is_an_equivalence_relation():
    """The property the whole round turns on. A fallback chain cannot satisfy it; key
    equality satisfies it by construction, which is why the fix is structural."""
    from ouroboros._outcome_receipts import _same_masked_verification, _same_verification

    check = "pytest tests/x.py"
    corpus = [
        {"criterion_id": "c1", "check": check},
        {"criterion_id": "c2", "check": check},
        {"check": check},
        {"check": "pytest  tests/x.py"},          # same canonical text as the one above
        {"paths": ["report.md"]},
        {"paths": ["report.md", "other.md"]},
        {"criterion_id": "c1"},
        {},                                        # no identity at all
    ]
    for relation in (_same_verification, _same_masked_verification):
        keyed = [r for r in corpus if relation(r, r)]
        assert keyed, "the relation must be reflexive on receipts that carry a key"
        for one in corpus:
            for other in corpus:
                assert relation(one, other) == relation(other, one)      # symmetric
                for third in corpus:
                    if relation(one, other) and relation(other, third):
                        assert relation(one, third)                      # transitive
    # ...and the identity-less receipt is equal to nothing, not even itself: it is
    # indistinguishable, not known-equal, so merging it would UNDERCOUNT open reds.
    assert _same_verification({}, {}) is False


def test_the_outstanding_set_is_order_independent():
    """Order-independence is the observable consequence of the relation being an
    equivalence, and a stronger claim than any three fixed sequences: EVERY permutation
    of the same candidates must yield the same outstanding identities and the same count.
    Under the chain, which red survived depended on which one was collected first."""
    import itertools

    from ouroboros._outcome_receipts import unreconciled_failed
    from ouroboros.review_evidence import _accept_verification_summary

    check = "pytest tests/x.py"
    candidates = [
        {"status": "fail", "criterion_id": "c1", "check": check},
        {"status": "fail", "criterion_id": "c2", "check": check},
        {"status": "fail", "check": check},            # shares the check, names no id
    ]
    expectations = [
        # (trailing reconciler, the identity keys that must remain outstanding)
        (None, {("criterion_id", "c1"), ("criterion_id", "c2"), _check_key(check)}),
        ({"status": "pass", "check": check},
         {("criterion_id", "c1"), ("criterion_id", "c2")}),
        ({"status": "pass", "criterion_id": "c1", "check": check},
         {("criterion_id", "c2"), _check_key(check)}),
        ({"status": "observed", "paths": ["report.md"]},
         {("criterion_id", "c1"), ("criterion_id", "c2"), _check_key(check)}),
    ]
    for reconciler, expected in expectations:
        for order in itertools.permutations(candidates):
            receipts = [*order] + ([reconciler] if reconciler else [])
            outstanding = unreconciled_failed(receipts)
            assert _keys(outstanding) == expected
            assert len(outstanding) == len(expected)
            assert _accept_verification_summary(
                receipts)["unreconciled_red_count"] == len(expected)


def test_the_outstanding_masked_set_is_order_independent():
    """The masked path keys on the `criterion_id` alone — same relation, same property."""
    import itertools

    from ouroboros._outcome_receipts import unreconciled_masked

    def masked(**kw):
        return {"status": "pass", "check_exit_masking": True, "check": "make x | tail", **kw}

    candidates = [masked(criterion_id="c1"), masked(criterion_id="c2"), masked()]
    expectations = [
        (None, {("criterion_id", "c1"), ("criterion_id", "c2"), ("none", "")}),
        # An id-less clean green clears only the masked receipt that names no criterion.
        ({"status": "pass", "check": "make x"},
         {("criterion_id", "c1"), ("criterion_id", "c2")}),
        # ...and the id-LESS masked receipt keeps the any-later-clean fallback: it names
        # no criterion, so there is nothing for the narrowing to protect.
        ({"status": "pass", "criterion_id": "c1", "check": "make x"},
         {("criterion_id", "c2")}),
    ]
    for reconciler, expected in expectations:
        for order in itertools.permutations(candidates):
            receipts = [*order] + ([reconciler] if reconciler else [])
            outstanding = unreconciled_masked(receipts)
            assert {
                ("criterion_id", r["criterion_id"]) if r.get("criterion_id") else ("none", "")
                for r in outstanding
            } == expected
            assert len(outstanding) == len(expected)


def test_a_quoted_operator_argument_is_not_syntax():
    """Round-5, finding 1: `canonical_command_text` dropped leading/trailing separator
    tokens AFTER `shlex` had discarded whether a token was quoted, so an argv whose final
    argument is the literal string `&&` canonicalized identically to the same argv without
    it — a live path to a false green, since a passing run of one could clear a failing
    run of the other."""
    import shlex

    from ouroboros import _outcome_receipts as R
    from ouroboros.shell_parse import canonical_command_text, shell_tokens

    # The reviewer's exact case: a literal trailing `&&`.
    assert canonical_command_text(shlex.join(["echo", "a", "&&"])) != \
        canonical_command_text(shlex.join(["echo", "a"]))
    # ...and the same collision in the middle, where the argument looks like syntax.
    assert canonical_command_text("echo a && b") != canonical_command_text(
        shlex.join(["echo", "a", "&&", "b"]))
    # A backslash-escaped operator is a literal too (the shell's own rule).
    assert canonical_command_text(r"echo \&\& x") != canonical_command_text("echo && x")
    # Genuine syntax still is syntax, and the two operators never fold together.
    assert canonical_command_text("make a && make b") != canonical_command_text(
        "make a || make b")
    # Marking quoting must not leak into the tokens the guards read.
    assert shell_tokens(shlex.join(["echo", "&&"])) == ["echo", "&&"]
    assert shell_tokens("echo && x") == ["echo", "&&", "x"]
    # Which is the point: two DIFFERENT commands are two different verifications.
    assert R.unreconciled_failed([
        {"status": "fail", "check": shlex.join(["echo", "a", "&&"])},
        {"status": "pass", "check": shlex.join(["echo", "a"])},
    ])


def test_the_projection_emits_the_canonical_check_it_compares():
    """Round-5, finding 1 third instance: the shared identity projection emitted the RAW
    `receipt["check"]` while reconciliation compared the canonical one, contradicting the
    single-derivation claim the function exists to make."""
    from ouroboros._outcome_receipts import (
        receipt_canonical_identity,
        receipt_identity_projection,
        verification_receipt_ledger_row,
    )
    from ouroboros.review_evidence import _accept_verification_summary

    receipt = {"status": "fail", "check": "pytest   tests/x.py\n"}
    canonical = receipt_canonical_identity(receipt).check
    assert canonical != receipt["check"]
    assert receipt_identity_projection(receipt)["check"] == canonical
    assert verification_receipt_ledger_row(receipt)["check"] == canonical
    summary = _accept_verification_summary([receipt])
    assert summary["latest_check"] == canonical
    assert summary["unreconciled_red_identity"]["check"] == canonical


# --------------------------------------------------------------------------------------
# P4 review round 6 — the relation is sound; the PROJECTION was lying about which
# authority decided. `_reconciles_masked` never consults check text, yet the disclosure
# re-derived its own answer and told the acceptance reviewer that canonical check text
# controlled reconciliation. Host-attested evidence making a false claim about its own
# basis is the defect class this whole surface exists to eliminate. Fixed the same way
# rounds 4 and 5 were: the reporting path now READS the deciding path — one shared
# mode-aware key, `receipt_reconciliation_key`.
# --------------------------------------------------------------------------------------


def test_an_id_less_masked_pass_discloses_the_any_clean_authority_that_clears_it():
    """Round-6 CRITICAL: a masked pass with no `criterion_id` is reconciled by ANY later
    clean grounding, and was projected as `reconciliation_identity="check"` with
    `expected_whitespace_normalized=true` — the reviewer was told the command text
    decided when nothing about the command text was consulted."""
    from ouroboros import _outcome_receipts as R
    from ouroboros.review_evidence import _accept_verification_summary

    masked = {"status": "pass", "check": "make test | tail", "check_exit_masking": True}

    # What actually decides: no criterion key, so any later clean grounding clears it —
    # under a completely different command.
    assert R.receipt_disclosed_reconciliation_key(masked) == ("none", "")
    assert R.unreconciled_masked([masked, {"status": "pass", "check": "make lint"}]) == []

    # ...and that is exactly what the two disclosures now say.
    proj = R.receipt_identity_projection(masked)
    assert proj["reconciliation_identity"] == "none"
    assert proj["expected_whitespace_normalized"] is False
    assert R.receipt_expected_whitespace_normalized(masked) is False
    assert R.verification_receipt_ledger_row(masked)["reconciliation_identity"] == "none"
    # The check text is still CARRIED (disclosure), it just no longer claims authority.
    assert proj["check"] == "make test | tail"

    summary = _accept_verification_summary([masked])
    assert summary["expected_whitespace_normalized"] is False
    assert summary["reconciliation_identity_kinds"] == ["none"]
    assert summary["latest_identity"]["reconciliation_identity"] == "none"


def test_an_identified_masked_pass_is_not_cleared_by_an_id_less_clean_receipt():
    """Round-6, the second case the prose got wrong: "an equal `criterion_id` when BOTH
    receipts carry one, else ANY" also misdescribes an IDENTIFIED masked receipt followed
    by an id-less clean one. The masked receipt's own id is the authority; the later
    receipt omitting its id does not clear it."""
    from ouroboros import _outcome_receipts as R
    from ouroboros.review_evidence import _accept_verification_summary

    masked = {"status": "pass", "criterion_id": "c1", "check": "make test | tail",
              "check_exit_masking": True}
    clean_no_id = {"status": "pass", "check": "make test"}
    clean_c1 = {"status": "pass", "criterion_id": "c1", "check": "make test"}

    # Behaviour: only the id-naming clean receipt clears it.
    assert R.unreconciled_masked([masked, clean_no_id]) == [masked]
    assert R.unreconciled_masked([masked, clean_c1]) == []

    # Disclosure: the authority is the criterion_id, never the check text — so the
    # whitespace flag stays FALSE here too, even though the receipt carries a command.
    assert R.receipt_disclosed_reconciliation_key(masked) == ("criterion_id", "c1")
    proj = R.receipt_identity_projection(masked)
    assert proj["reconciliation_identity"] == "criterion_id"
    assert proj["expected_whitespace_normalized"] is False

    # The masked receipt alone claims no text authority...
    assert _accept_verification_summary([masked])["expected_whitespace_normalized"] is False
    summary = _accept_verification_summary([masked, clean_no_id])
    assert summary["check_exit_masking_unreconciled"] is True
    # ...and the flag that IS set for the pair comes from the plain green beside it,
    # whose own key really is the check text — the kinds list says which is which.
    assert summary["expected_whitespace_normalized"] is True
    assert summary["reconciliation_identity_kinds"] == ["check", "criterion_id"]

    # The same receipt WITHOUT the masking flag is an ordinary green again, and there the
    # check text really is the authority — the mode is what differs, not the receipt text.
    plain = {k: v for k, v in masked.items() if k != "check_exit_masking"}
    plain.pop("criterion_id")
    assert R.receipt_disclosed_reconciliation_key(plain)[0] == "check"
    assert R.receipt_expected_whitespace_normalized(plain) is True


def test_the_decider_and_the_discloser_are_the_same_projection():
    """The class, not the two instances: every reconciliation decision and every
    disclosure of one must come from `receipt_reconciliation_key`, so no third caller can
    re-derive an authority that then drifts from the one that decides."""
    import ast
    import pathlib

    from ouroboros import _outcome_receipts as R

    src = pathlib.Path(R.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    deciders = {"_reconciles", "_reconciles_masked", "_same_verification",
                "_same_masked_verification", "receipt_expected_whitespace_normalized",
                "receipt_disclosed_reconciliation_key"}
    seen = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name in deciders):
            continue
        seen.add(node.name)
        called = {
            n.func.id for n in ast.walk(node)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
        assert called & {"receipt_reconciliation_key", "receipt_disclosed_reconciliation_key"}, (
            f"{node.name} must read the shared key projection, not re-derive one"
        )
        # ...and none of them may reach past it into the raw components.
        attrs = {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}
        assert not (attrs & {"key", "criterion_key"}), f"{node.name} bypasses the projection"
    assert seen == deciders

    # For every receipt shape, the mode-aware key and the relation agree about whether a
    # later grounding of the same kind clears it.
    for receipt in (
        {"status": "pass", "check": "make x | tail", "check_exit_masking": True},
        {"status": "pass", "criterion_id": "c1", "check": "make x | tail",
         "check_exit_masking": True},
        {"status": "fail", "check": "make x"},
        {"status": "fail", "criterion_id": "c1", "check": "make x"},
        {"status": "fail", "paths": ["report.md"]},
        {"status": "fail"},
    ):
        masked_mode = R.receipt_is_masked_pass(receipt)
        key = R.receipt_disclosed_reconciliation_key(receipt)
        relation = R._reconciles_masked if masked_mode else R._reconciles
        # An unrelated later grounding clears it IFF it has no key of its own to protect.
        unrelated = {"status": "pass", "check": "totally other"}
        assert relation(unrelated, receipt) is (key[0] == "none")


# --------------------------------------------------------------------------------------
# P4 review round 7 — the round-6 fix was made for ONE kind and never asked of the others.
# `expected_whitespace_normalized` still claimed canonical-command-text normalization for
# `artifact_paths`, whose set `canonical_path_set` compares byte-for-byte. The durable
# form is not a third patched branch: the per-kind answer moves INTO the closed kind
# table, so a fourth kind cannot be added without answering for itself.
# --------------------------------------------------------------------------------------


def test_the_whitespace_flag_is_derived_once_for_every_identity_kind():
    """Total over the kinds, by construction. True only for the kind whose identity really
    normalizes command text; false for every other, including a kind added later — which
    has no default to fall into, because the answer is a column of the kind table."""
    from ouroboros import _outcome_receipts as R

    # The table is the closed set of kinds, and `key` can produce nothing outside it.
    kinds = {kind for kind, _, _ in R.IDENTITY_KINDS} | {R.IDENTITY_KIND_NONE}
    assert kinds == {"criterion_id", "check", "artifact_paths", "none"}
    assert set(R.KIND_NORMALIZES_COMMAND_TEXT) == kinds          # total, no gaps
    assert R.KIND_NORMALIZES_COMMAND_TEXT == {
        "check": True, "criterion_id": False, "artifact_paths": False, "none": False}

    # One receipt per kind, and the flag agrees with the table for each.
    by_kind = {
        "criterion_id": {"status": "fail", "criterion_id": "c1", "check": "go test"},
        "check": {"status": "fail", "check": "go test"},
        "artifact_paths": {"status": "fail", "paths": ["report.md"]},
        "none": {"status": "fail"},
    }
    assert set(by_kind) == kinds
    for kind, receipt in by_kind.items():
        assert R.receipt_disclosed_reconciliation_key(receipt)[0] == kind
        assert R.receipt_expected_whitespace_normalized(receipt) is (
            R.KIND_NORMALIZES_COMMAND_TEXT[kind])
    assert R.receipt_expected_whitespace_normalized(by_kind["check"]) is True
    assert [k for k in kinds if R.KIND_NORMALIZES_COMMAND_TEXT[k]] == ["check"]

    # ROUND-7 CRITICAL, stated as behaviour: `artifact_paths` normalizes NOTHING, and the
    # disclosure must not claim otherwise — the two filenames below differ only by edge
    # whitespace and are two different files to this identity.
    spaced = {"status": "fail", "paths": ["  report.md  "]}
    assert R.receipt_identity(spaced) != R.receipt_identity(by_kind["artifact_paths"])
    for receipt in (spaced, by_kind["artifact_paths"]):
        proj = R.receipt_identity_projection(receipt)
        assert proj["reconciliation_identity"] == "artifact_paths"
        assert proj["expected_whitespace_normalized"] is False
        assert R.verification_receipt_ledger_row(receipt)[
            "expected_whitespace_normalized"] is False


def test_the_flag_reads_the_kind_table_rather_than_naming_kinds_inline():
    """The class, not the instance: a fourth kind must inherit its answer from its own
    table row. A membership test against a hard-coded kind list is exactly how rounds 6
    and 7 each shipped a flag that was right for the kinds someone remembered."""
    import ast
    import pathlib

    from ouroboros import _outcome_receipts as R

    tree = ast.parse(pathlib.Path(R.__file__).read_text(encoding="utf-8"))
    fn = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "receipt_expected_whitespace_normalized"
    )
    # It subscripts the total table...
    assert any(
        isinstance(n, ast.Subscript) and getattr(n.value, "id", "") == "KIND_NORMALIZES_COMMAND_TEXT"
        for n in ast.walk(fn)
    )
    # ...and names no kind of its own, in a comparison or a literal container.
    named = {
        n.value for n in ast.walk(fn)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    }
    assert not (named & {"check", "criterion_id", "artifact_paths", "none"}), named
    # ...and it makes no comparison of its own — a `kind in (...)` / `kind == "..."` test
    # is precisely the shape that answers only for the kinds it was written against.
    assert not any(isinstance(n, ast.Compare) for n in ast.walk(fn))


# --------------------------------------------------------------------------------------
# P4 review round 8 — the `shlex.join` change was argued safe in ONE direction. An old and
# a new receipt for the SAME argv render differently and fail to reconcile, and a
# non-reconciling red stays red. But an old red and a new green from DIFFERENT argvs can
# render IDENTICALLY (`["echo","a b"]` and `["echo","a","b"]` both space-joined to
# `echo a b`), and that green cleared that red — a false green produced by the change made
# to remove one. A receipt did not record WHICH renderer wrote its check text, so the
# comparator could not tell the formats apart. It records it now.
# --------------------------------------------------------------------------------------


def test_a_legacy_check_receipt_is_never_the_same_verification_as_a_versioned_one():
    """Round-8 CRITICAL, the reviewer's exact pair: a pre-upgrade red whose argv was
    `["echo", "a b"]` and a post-upgrade green whose argv was `["echo", "a", "b"]`. Both
    stored strings read `echo a b`. They are not known-equal, they are UNKNOWN, and
    unknown must not clear a red."""
    import shlex

    from ouroboros import _outcome_receipts as R

    legacy_red = {"status": "fail", "check": " ".join(["echo", "a b"])}
    versioned_green = {
        "status": "pass",
        "check": shlex.join(["echo", "a", "b"]),
        "check_rendering": R.CHECK_RENDERING_SHLEX_JOIN,
    }
    # The collision is real at the TEXT level — which is why text alone cannot be the key.
    assert R.receipt_canonical_identity(legacy_red).check == "echo a b"
    assert R.receipt_canonical_identity(versioned_green).check == "echo a b"
    # ...and the keys differ, so the red survives.
    assert R.receipt_identity(legacy_red) != R.receipt_identity(versioned_green)
    assert R.unreconciled_failed([legacy_red, versioned_green]) == [legacy_red]
    # Not just this pair: a versioned green NEVER clears a legacy red, even spelling the
    # same argv — the direction is honest and stated. Cross-version reconciliation is
    # strictly less likely, so an upgrade can leave standing a red that was really fixed.
    # That is a FALSE RED (a human looks twice) traded against a FALSE GREEN (the thing
    # this surface exists to prevent).
    assert R.unreconciled_failed([
        {"status": "fail", "check": "echo a"},
        {"status": "pass", "check": "echo a", "check_rendering": R.CHECK_RENDERING_SHLEX_JOIN},
    ]) != []

    # Within ONE rendering, reconciliation is untouched — including legacy↔legacy, which
    # is all that can be recovered from receipts already on disk and is exactly the
    # behaviour they had before the upgrade.
    assert R.unreconciled_failed([legacy_red, {"status": "pass", "check": "echo a b"}]) == []
    assert R.unreconciled_failed([
        dict(versioned_green, status="fail"), versioned_green,
    ]) == []
    # An unrecognised future rendering is its own namespace, with no code change needed.
    assert R.receipt_identity({"check": "echo a b", "check_rendering": "some_future_v9"}) \
        != R.receipt_identity(versioned_green)
    # The rendering is DISCLOSED, so a reviewer seeing an open red beside a byte-identical
    # green can tell why rather than reading the host as broken.
    assert R.receipt_identity_projection(legacy_red)["check_rendering"] == "unversioned"
    assert R.verification_receipt_ledger_row(versioned_green)["check_rendering"] == "shlex_join"


def test_every_receipt_writer_that_renders_a_check_stamps_its_rendering():
    """The class, not the instance: the hole was one writer changing its renderer without
    saying so, and `verify.py` has FOUR receipt writers that store a `check` (the fourth
    records a check the REMOTE target ran). A fifth must not reintroduce it silently."""
    import ast
    import pathlib

    from ouroboros.tools import verify as V

    tree = ast.parse(pathlib.Path(V.__file__).read_text(encoding="utf-8"))
    writers = [
        node.args[0] for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute) and node.func.attr == "update"
        and getattr(node.func.value, "id", "") == "receipt"
        and node.args and isinstance(node.args[0], ast.Dict)
    ]
    stamped = 0
    for payload in writers:
        keys = {k.value for k in payload.keys if isinstance(k, ast.Constant)}
        if "check" not in keys:
            continue
        stamped += 1
        assert "check_rendering" in keys, "a receipt writer stores a check with no rendering stamp"
    assert stamped == 4, f"expected 4 check-writing receipt updates, found {stamped}"


def test_the_masked_and_path_identities_have_no_mixed_version_hole():
    """The same question asked of the other two identities. Neither has the hole, and both
    reasons are structural rather than lucky."""
    from ouroboros import _outcome_receipts as R

    # MASKED: keys on `criterion_id` alone and never consults check text, so a rendering
    # difference must NOT split it — an old masked criterion is still re-grounded by the
    # new clean receipt naming it. (The id is agent-authored and stored verbatim; no
    # writer ever re-rendered it, so there is nothing to version.)
    legacy_masked = {"status": "pass", "criterion_id": "c1", "check": "make test | tail",
                     "check_exit_masking": True}
    versioned_clean = {"status": "pass", "criterion_id": "c1", "check": "make test",
                       "check_rendering": R.CHECK_RENDERING_SHLEX_JOIN}
    assert R.unreconciled_masked([legacy_masked, versioned_clean]) == []
    assert R.receipt_disclosed_reconciliation_key(legacy_masked) == ("criterion_id", "c1")

    # ARTIFACT PATHS: stored RAW by the writer and canonicalized by the READER, so both
    # eras go through today's `canonical_path_set` and compare on equal terms. The one
    # path-side change this phase made (dropping `.strip()`) was a comparator change, not
    # a stored-format change, so it applies to old and new receipts alike.
    legacy_observation = {"status": "fail", "paths": ["report.md", "notes.md"]}
    versioned_observation = {"status": "observed", "paths": ["notes.md", "report.md"],
                             "check_rendering": R.CHECK_RENDERING_SHLEX_JOIN}
    assert R.unreconciled_failed([legacy_observation, versioned_observation]) == []
    assert R.receipt_identity(legacy_observation) == R.receipt_identity(versioned_observation)


def test_receipt_identity_parts_is_a_disclosure_not_the_comparison():
    """Round 8, finding 2: the docstring claimed reconciliation matches the three
    components "COMPONENT-wise" — the pre-v6.78.0 fallback chain, i.e. exactly the
    behaviour whose removal is the point. The function exposes three texts; sameness reads
    ONE key."""
    from ouroboros import _outcome_receipts as R

    one = {"status": "fail", "criterion_id": "c1", "check": "make x", "paths": ["a.md"]}
    other = {"status": "pass", "criterion_id": "c2", "check": "make x", "paths": ["a.md"]}
    # Two of the three disclosed components are identical...
    assert R.receipt_identity_parts(one)[1:] == R.receipt_identity_parts(other)[1:]
    # ...and they are still different verifications, because only the key decides.
    assert R.receipt_identity(one) != R.receipt_identity(other)
    assert R.unreconciled_failed([one, other]) == [one]
    # The parts stay plain TEXT (the key's check value is the rendering-paired form).
    assert R.receipt_identity_parts(one) == ("c1", "make x", "a.md")
    assert R.receipt_identity_parts(one)[1] != R.receipt_canonical_identity(one).check_identity_source
