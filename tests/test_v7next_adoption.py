"""The v7next release bar, executed.

``scripts/v7next_adoption.py`` is the checker that says whether
``ADOPTION_v7next.md`` still describes the tree. Until this file existed the
checker was run by hand, which is how a whole upstream-train row (sync #2,
``TRAIN-F6b-f3fbfdbb``) could be deleted by a stale-base overwrite and leave
both validator modes at rc 0. This suite runs ``validate()`` on the live
manifest in both modes and drives one mutant per rule that the deletion
taught us to want, so the bar is executed by something automatic.

Deliberately NOT a CI-workflow change: ``.github/workflows/ci.yml`` is a
protected file and the default pytest lane already carries this file.
"""
from __future__ import annotations

import copy
import pathlib

import pytest

from scripts.v7next_adoption import (
    DEFERRED_OUT_OF_V70,
    OPERATOR,
    OWNER,
    REQUIRED_PHASE,
    REQUIRED_TRAINS,
    declared_deferral_authorities,
    manifest_prose,
    parse_rows,
    validate,
)

REPO = pathlib.Path(__file__).resolve().parent.parent
MANIFEST = REPO / "ADOPTION_v7next.md"


@pytest.fixture(scope="module")
def rows() -> list[dict[str, str]]:
    parsed, errors = parse_rows(MANIFEST.read_text(encoding="utf-8"))
    assert not errors, errors
    assert parsed, "the manifest table parsed to zero rows"
    return parsed


def _without(rows: list[dict[str, str]], row_id: str) -> list[dict[str, str]]:
    kept = [r for r in rows if r["id"] != row_id]
    assert len(kept) == len(rows) - 1, f"{row_id} is not in the manifest"
    return kept


def _mutate(rows: list[dict[str, str]], row_id: str, **cells: str) -> list[dict[str, str]]:
    out = copy.deepcopy(rows)
    for r in out:
        if r["id"] == row_id:
            r.update(cells)
            return out
    raise AssertionError(f"{row_id} is not in the manifest")


def _first_done(rows: list[dict[str, str]]) -> dict[str, str]:
    for r in rows:
        if r["status"] == "done" and "::" in r["verification hook"]:
            return r
    raise AssertionError("no done row carries a ::nodeid hook")


@pytest.fixture(scope="module")
def prose() -> str:
    return manifest_prose(MANIFEST.read_text(encoding="utf-8"))


def test_the_live_manifest_passes_both_modes(rows, prose):
    """The manifest on this tree is the thing the bar is about — table AND the
    prose around it, the way ``main()`` runs it."""
    assert validate(copy.deepcopy(rows), release=False, prose=prose) == []
    assert validate(copy.deepcopy(rows), release=True, prose=prose) == []


def test_manifest_prose_is_everything_but_the_table(prose):
    assert "Notes:" in prose
    assert not [line for line in prose.splitlines() if line.startswith("|")]


@pytest.mark.parametrize("train_id", sorted(REQUIRED_TRAINS))
@pytest.mark.parametrize("release", [False, True])
def test_deleting_an_upstream_train_row_turns_the_bar_red(rows, train_id, release):
    """The mutant that actually happened: a whole-file overwrite drops a train
    row. It must be red in BOTH modes — the deletion in 285ab66d survived
    because the default mode was the one being run."""
    errors = validate(_without(rows, train_id), release=release)
    assert any(train_id in e for e in errors), errors


@pytest.mark.parametrize("release", [False, True])
def test_repointing_a_train_row_at_another_merge_turns_the_bar_red(rows, release):
    """A train row that no longer names its own upstream tip and merge is not a
    record of that train."""
    train_id = sorted(REQUIRED_TRAINS)[0]
    errors = validate(_mutate(rows, train_id, what="upstream train, details elsewhere"),
                      release=release)
    assert any(train_id in e for e in errors), errors


def test_a_bogus_hook_nodeid_turns_the_bar_red(rows):
    """A hook may name a test that does not exist only if nothing checks it.
    Paths were already resolved; the ``::nodeid`` half was free text."""
    victim = _first_done(rows)
    bogus = "tests/test_smoke.py::test_no_such_pin_was_ever_written"
    errors = validate(_mutate(rows, victim["id"], **{"verification hook": bogus}),
                      release=False)
    assert any("test_no_such_pin_was_ever_written" in e for e in errors), errors


@pytest.mark.parametrize("hook", [
    "the suites this row moved bytes in",                      # prose only
    "tests/test_no_such_suite_was_ever_written.py",            # missing file
    "tests/../scripts/v7next_adoption.py",                     # escapes tests/
    "tests/test_smoke.py::test_no_such_pin_was_ever_written",  # bogus nodeid
])
def test_a_hook_error_does_not_claim_the_release_bar(rows, hook):
    """Hook resolution runs for every ``done`` row in BOTH modes — it is a
    property of a shipped row, not of the ``--release`` invocation. A message
    reported in the default mode must therefore not say ``release:``, or the
    reader is told to look for a switch that has nothing to do with it."""
    victim = _first_done(rows)
    errors = validate(_mutate(rows, victim["id"], **{"verification hook": hook}),
                      release=False)
    assert errors, "the hook shape was accepted in the default mode"
    assert not [e for e in errors if e.startswith("release:")], errors


def test_a_hook_nodeid_that_names_a_real_test_stays_green(rows):
    """The AST read must accept what the manifest legitimately names, including
    a data carrier (``tests/_shared.py::SETTINGS_WRITERS``) — a hook may point
    at the inventory a pin closes, not only at a function."""
    victim = _first_done(rows)
    good = ("tests/_shared.py::settings_writers + tests/_shared.py::SETTINGS_WRITERS "
            "+ tests/test_smoke.py::test_size_ratchet_transition_against_explicit_base")
    assert validate(_mutate(rows, victim["id"], **{"verification hook": good}),
                    release=False) == []


@pytest.mark.parametrize("marker", ["NOT DONE", "OPEN RESIDUAL", "not integrated yet",
                                    "still owed", "read pending"])
def test_a_done_row_that_says_it_is_not_done_turns_the_bar_red(rows, marker):
    """The contradiction this wave found six times: a row whose text says the
    work is open while its status cell says ``done``."""
    victim = _first_done(rows)
    text = f"{victim['what']} — {marker} on this tree"
    errors = validate(_mutate(rows, victim["id"], what=text), release=False)
    assert any(victim["id"] in e and "done" in e for e in errors), errors


def test_a_named_residual_clause_is_the_escape_not_a_second_status(rows):
    """A shipped row may carry an open residual — that is what ``residual:``
    declares. The rule refuses the contradiction, not the disclosure."""
    victim = _first_done(rows)
    text = f"{victim['what']} — NOT DONE for the review surfaces; residual: the migration is post-release"
    assert validate(_mutate(rows, victim["id"], what=text), release=False) == []


def test_a_post_release_row_needs_a_recorded_deferral(rows):
    """post-release is the one state that leaves the release bar. An id nobody
    recorded cannot take it."""
    victim = _first_done(rows)
    errors = validate(_mutate(rows, victim["id"], disposition="post-release",
                              status="deferred", phase="POST"), release=True)
    assert any(victim["id"] in e and "DEFERRED_OUT_OF_V70" in e for e in errors), errors


def test_a_required_row_cannot_be_parked_post_release_by_the_operator(rows, monkeypatch):
    """The property the old frozenset carried: a row of the owner-approved
    inventory leaves 7.0 only by an owner decision. Operator authority exists
    for disclosures, and must not become a way past that."""
    monkeypatch.setitem(DEFERRED_OUT_OF_V70, "ABI-8", OPERATOR)
    errors = validate(copy.deepcopy(rows), release=True)
    assert any("ABI-8" in e and "owner decision" in e for e in errors), errors


def test_prose_that_calls_a_row_rowless_turns_the_bar_red(rows):
    """The contradiction that stood two days past a green bar: the Notes said
    W4-F3/W4-F4 «get no row» after d348ea46 had made them rows. The validator
    read rows only; now the prose's ids are resolved against the table."""
    victim = rows[0]["id"]
    notes = f"Notes:\n- No-row ids: {victim} — a disclosed observation, not work owed."
    errors = validate(copy.deepcopy(rows), release=False, prose=notes)
    assert any(e.startswith("prose:") and victim in e and "No-row ids" in e
               for e in errors), errors


def test_prose_that_names_a_ghost_id_needs_a_no_row_declaration(rows):
    """The mirror: prose naming an id the table does not have is red unless the
    prose declares it rowless — the declared form is the escape, not phrasing."""
    ghost = "DEFER-NO-SUCH-ROW"
    assert all(r["id"] != ghost for r in rows)
    notes = f"Notes:\n- {ghost} was folded into another row (see D02)."
    errors = validate(copy.deepcopy(rows), release=False, prose=notes)
    assert any(e.startswith("prose:") and ghost in e for e in errors), errors
    assert validate(copy.deepcopy(rows), release=False,
                    prose=notes + f"\n- No-row ids: {ghost}.") == []


def test_prose_id_grammar_ignores_plan_decisions_and_lane_labels(rows):
    """`D-14` (a plan decision), `CPL4-C6` (a lane label) and the schema's own
    pattern words (`Dnn`, `ABI-n`) are not ids and must not be reported."""
    notes = "Notes:\n- D-14 sent CPL4-C6 here; `Dnn` and `ABI-n` are patterns."
    assert validate(copy.deepcopy(rows), release=False, prose=notes) == []


def _post_release_rows(rows, authority):
    return [r for r in rows if r["disposition"] == "post-release"
            and DEFERRED_OUT_OF_V70.get(r["id"]) == authority]


def test_an_owner_deferral_row_must_carry_the_owner_quote(rows):
    """The record and the row tell one story. An OWNER value in
    DEFERRED_OUT_OF_V70 beside a row with no ``owner verbatim «…»`` quote is
    the drift the record's own comment block showed (E2/E3, spec §6.4)."""
    victim = _post_release_rows(rows, OWNER)[0]
    unquoted = victim["what"].replace("owner verbatim «", "owner said «")
    errors = validate(_mutate(rows, victim["id"], what=unquoted), release=False)
    assert any(victim["id"] in e and "owner deferral" in e for e in errors), errors


def test_the_notes_declare_the_deferral_authorities_the_register_records(prose):
    """The Notes carry the register's mirror in the one declared form, and it
    agrees with ``DEFERRED_OUT_OF_V70`` id for id. Free prose about authority is
    not read: the Notes called W4-F4 operator-disclosed for a day after the
    register made it an owner deferral."""
    assert declared_deferral_authorities(prose) == DEFERRED_OUT_OF_V70


@pytest.mark.parametrize("mutant", ["register_moves", "declaration_omits", "declaration_invents"])
def test_a_deferral_declaration_that_disagrees_with_the_register_turns_the_bar_red(
        rows, prose, monkeypatch, mutant):
    """Both directions and both edges: the register moves under a standing
    declaration, the declaration drops a recorded id, the declaration invents one."""
    if mutant == "register_moves":
        monkeypatch.setitem(DEFERRED_OUT_OF_V70, "W4-F4", OPERATOR)
        text, needle = prose, "W4-F4 is owner while DEFERRED_OUT_OF_V70 records operator-disclosed"
    elif mutant == "declaration_omits":
        text, needle = prose.replace("W4-F4 owner, ", ""), "omits W4-F4"
    else:
        text, needle = prose.replace("W4-F4 owner,", "W4-F4 owner, DEFER-NO-SUCH-ROW owner,"), "declares DEFER-NO-SUCH-ROW"
    assert text != prose or mutant == "register_moves"
    errors = validate(copy.deepcopy(rows), release=False, prose=text)
    assert any(e.startswith("prose: Deferral authorities") and needle in e for e in errors), errors


def test_an_operator_disclosure_row_must_not_carry_an_owner_quote(rows, monkeypatch):
    """The other direction: a quoted row recorded as operator-disclosed hides
    an owner decision behind the weaker authority."""
    victim = next(r for r in _post_release_rows(rows, OWNER)
                  if r["id"] not in REQUIRED_PHASE)  # keep the required-inventory rule out of it
    monkeypatch.setitem(DEFERRED_OUT_OF_V70, victim["id"], OPERATOR)
    errors = validate(copy.deepcopy(rows), release=False)
    assert any(victim["id"] in e and OPERATOR in e and "owner quote" in e
               for e in errors), errors
