"""Keyed custody readers: addressed consumers select rows before Python decodes them.

Every check drives a REAL consumer and compares it with the input those readers had
before, every custody row decoded (``delegate_custody.replay``):

* the health invariants' delegated-run lines read ONE selected superset (own, held,
  maybe-unread) behind an ordinary-JSON guard: the same lines for every
  complete/consumed flag pair over nineteen stored values (NaN and ±Infinity, which
  ``machine_json`` writes and Python reads, included), for owner, foreign and
  unattributed readers, snapshot and directory-copy patches and nonfinite unrelated
  facts; a failed selection reads every row into the SAME state (the own-run line has
  no read of its own), and a double failure stays a failure that rewrites nothing;
* ``delegate_start(root="skill_payload")`` by a second top-level task reads payload
  holders and pending requests in ONE transaction: a holder whose REQUESTED becomes
  STARTED right after that transaction still refuses it, while a deliberately
  two-transaction read lets the start through; live, unsettled-disposed,
  closed-undisposed and pending holders refuse, settled-and-disposed and terminal-owner
  holders release, nonfinite facts never hide a holder and a failed read refuses typed;
* by-id readers (retry, orphan, continuation, hand-over, apply, sleep) answer exactly as
  over every row, keep ``None`` versus ``KeyError`` and the one existing memo seed;
* Python decodes stay equal at 10 and 5,000 unrelated closed rows (SQL still visits them).
"""

from __future__ import annotations

import copy
import inspect
import json
import math
from types import SimpleNamespace

import pytest

from ouroboros import delegate_custody as custody
from ouroboros import delegate_custody_current as current
from ouroboros import store
from tests.test_delegated_skill_payload import (  # noqa: F401  (autouse fixture applies on import)
    _exact_payload_start,
    _owned_gateway_uses_each_test_transport,
    _payload_ctx,
    _seed_skill,
    _StartStub,
)

NAN, INF = float("nan"), float("inf")
_ABSENT = object()
# Nineteen stored flag values: every JSON type, numeric zero and one in each spelling, and
# the nonfinite tokens ``machine_json`` writes (Python reads them; ``json_valid`` does not).
FLAG_VALUES = (_ABSENT, None, False, True, 0, 1, 2, 0.0, 1.0, -0.0, "", "0", "1", "x", [], {}, NAN, INF, -INF)
SETTLED = {"settled": True, "terminal_state": "failed", "settled_at": "2026-10-13T00:00:00Z"}


def _put_many(root, rows):
    """Custody rows exactly as the reducer stores them: ``put_custody`` and its hold column."""
    with store.open_store(root, write=True) as txn:
        for run_id, facts in rows:
            facts = {"run_id": run_id, **facts}
            hold = current.HELD if current._held(current._decode(facts)) else current.CLOSED
            txn.put_custody(run_id, facts, state=hold)


def _stored(root):
    with store.open_store(root, write=False) as txn:
        return [tuple(record) for record in txn.conn.execute(
            "SELECT run_id, state, facts FROM custody_runs ORDER BY run_id")]


def _decodes(monkeypatch):
    """Every RunCustody decode (the Python cost a selection avoids), by run id."""
    seen, real = [], current._decode
    monkeypatch.setattr(current, "_decode", lambda raw: seen.append(raw.get("run_id")) or real(raw))
    return seen


def _seed_health(root):
    """Rows the three health blocks can and cannot name, nonfinite facts included."""
    locator = {"run_dir": "/engine/runs/own"}
    _put_many(root, [
        # The reader's own settled run is CLOSED: only the own arm selects it.
        ("run-own", {"task_id": "own", **SETTLED, "engine_locator": locator}),
        ("run-own-acked", {"task_id": "own", **SETTLED, "engine_locator": locator, "terminal_ack_attempt": "1"}),
        ("run-own-live", {"task_id": "own"}),
        # Foreign outputs: NaN is truthy to Python and invalid to json_valid; Infinity consumed is read.
        ("run-nan-complete", {"task_id": "owner", **SETTLED, "output_artifact": "a.out", "output_complete": NAN}),
        ("run-inf-consumed", {"task_id": "owner", **SETTLED, "output_artifact": "b.out", "output_complete": True,
                              "output_consumed": INF}),
        ("run-text-complete", {"task_id": "owner", **SETTLED, "output_artifact": "c.out", "output_complete": "yes",
                               "output_consumed": ""}),
        ("run-read", {"task_id": "owner", **SETTLED, "output_artifact": "d.out", "output_complete": True,
                      "output_consumed": True}),
        # Undisposed patches are held rows: a snapshot and a directory copy; a disposed one is closed.
        ("run-snapshot", {"task_id": "owner", **SETTLED, "snapshot_id": "snap-1", "target_root": "/tmp/t1",
                          "patch_captured": True}),
        ("run-copy", {"task_id": "owner", **SETTLED, "target_root": "/tmp/t2",
                      "resource_ref": {"workspace_kind": "directory", "strategy": "copy"}}),
        ("run-disposed", {"task_id": "owner", **SETTLED, "snapshot_id": "snap-2", "patch_disposed": "applied"}),
        # An unrelated closed run whose disclosed spend is nonfinite.
        ("run-nan-cost", {"task_id": "other", **SETTLED, "cost_usd": NAN, "spend_disclosed": True}),
    ])


def _env(root):
    return SimpleNamespace(drive_root=root, drive_path=lambda rel="": root / rel,
                           repo_path=lambda rel="": root / "repo" / rel)


def _delegated_lines(root, task_id="", attempt_key="1", active_root=""):
    """The delegated-run lines of one real health build (the rest reads logs and clocks)."""
    from ouroboros.context_health import build_health_invariants

    text = build_health_invariants(_env(root), task_id=task_id, active_root=active_root, attempt_key=attempt_key)
    return [line for line in text.splitlines() if "DELEGATED" in line]


def _over_every_row(fn, *args, **kwargs):
    """The oracle: the same consumer with the selection replaced by every row, decoded."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(current, "health_runs", lambda root, task_id="": current.runs(root))
        return fn(*args, **kwargs)


_FOLDED = {"NEVER READ": "unread", "AWAITS DISPOSITION": "patches"}


def _named(lines, marker, root):
    """Run ids from own lines or the exact foreign list the aggregate line names."""
    from tests.test_health_invariants_ownership import _foreign_source

    named = {line.split("— run ")[1].split()[0] for line in lines if marker in line and "— run " in line}
    for line in lines:
        if "OTHER TASKS' DELEGATED WORK" in line and marker in _FOLDED:
            for work in _foreign_source(root, line).values():
                named |= set(work[_FOLDED[marker]])
    return named


# --- health: one selected read, equal lines, the shared full-read fallback -------------


@pytest.mark.parametrize("reader", ["own", "owner", "someone-else", ""])
def test_health_lines_equal_the_every_row_read_for_every_reader(tmp_path, reader):
    _seed_health(tmp_path)
    keyed = _delegated_lines(tmp_path, task_id=reader, active_root="/tmp/t1")
    assert keyed == _over_every_row(_delegated_lines, tmp_path, task_id=reader, active_root="/tmp/t1")
    assert _named(keyed, "NEVER READ", tmp_path) == {"run-nan-complete", "run-text-complete"}, keyed
    assert _named(keyed, "AWAITS DISPOSITION", tmp_path) == {"run-snapshot", "run-copy"}, keyed
    assert _named(keyed, "OWN DELEGATED RUN ENDED", tmp_path) == ({"run-own"} if reader == "own" else set()), keyed


def test_the_selection_skips_closed_rows_it_cannot_name(tmp_path, monkeypatch):
    _seed_health(tmp_path)
    decoded = _decodes(monkeypatch)
    selected = current.health_runs(tmp_path, "own")
    assert list(selected) == sorted(selected), "run_id order, as over every row"
    assert set(selected) == {"run-own", "run-own-acked", "run-own-live", "run-nan-complete", "run-inf-consumed",
                             "run-text-complete", "run-snapshot", "run-copy", "run-nan-cost"}
    assert sorted(decoded) == sorted(selected), "the closed read output and disposed patch are never decoded"
    assert set(current.health_runs(tmp_path, "")) == set(selected) - {"run-own", "run-own-acked"}


def _sql_keeps(complete, consumed):
    """The documented prefilter, modelled: a nonfinite row is not ordinary JSON and always
    passes; otherwise an absent/null/false/numeric-zero ``output_complete`` or a
    true/numeric-one ``output_consumed`` drops it, and every text/array/object passes."""
    if any(isinstance(value, float) and not math.isfinite(value) for value in (complete, consumed)):
        return True
    numeric = (bool, int, float)
    drops_complete = complete is _ABSENT or complete is None or (isinstance(complete, numeric) and complete == 0)
    drops_consumed = isinstance(consumed, numeric) and consumed == 1
    return not (drops_complete or drops_consumed)


def test_every_stored_flag_pair_names_the_same_unread_outputs(tmp_path, monkeypatch):
    """361 complete/consumed pairs, NaN and ±Infinity included; no strict ``=1``/``=0``."""
    rows, expected, kept = [], set(), set()
    for i, complete in enumerate(FLAG_VALUES):
        for j, consumed in enumerate(FLAG_VALUES):
            run_id = f"run-{i:02d}-{j:02d}"
            facts = {"task_id": "owner", **SETTLED, "output_artifact": f"{run_id}.out"}
            if complete is not _ABSENT:
                facts["output_complete"] = complete
            if consumed is not _ABSENT:
                facts["output_consumed"] = consumed
            rows.append((run_id, facts))
            if complete is not _ABSENT and bool(complete) and not (consumed is not _ABSENT and bool(consumed)):
                expected.add(run_id)
            if _sql_keeps(complete, consumed):
                kept.add(run_id)
    _put_many(tmp_path, rows)
    assert len(expected) == 90 and expected <= kept and len(kept) == 235
    for reader in ("owner", "reader", ""):
        keyed = _delegated_lines(tmp_path, task_id=reader)
        assert keyed == _over_every_row(_delegated_lines, tmp_path, task_id=reader), reader
        assert _named(keyed, "NEVER READ", tmp_path) == expected, reader
    decoded = _decodes(monkeypatch)
    assert set(current.health_runs(tmp_path, "reader")) == kept
    assert len(decoded) == len(kept), "only the kept rows reach Python"


def test_a_failed_selection_is_said_with_its_reader_and_replays_no_row(tmp_path, monkeypatch):
    """The three delegated-run blocks share the ONE selected read; when it fails the block says
    custody is unavailable, names the addressed reader and skips those blocks (each would replay
    every row for itself, and an empty replay proves no absence of obligations: an earlier build
    fell back to that full read; no longer)."""
    from ouroboros.context_health import build_health_invariants

    _seed_health(tmp_path)
    baseline = _delegated_lines(tmp_path, task_id="own")
    assert _named(baseline, "OWN DELEGATED RUN ENDED", tmp_path) == {"run-own"}
    assert "run-nan-complete" in _named(baseline, "NEVER READ", tmp_path)

    selections, replays = [], []

    def failing_selection(root, task_id=""):
        selections.append(task_id)
        raise store.StoreBusy("the selected read failed")

    monkeypatch.setattr(current, "health_runs", failing_selection)
    monkeypatch.setattr(custody, "replay", lambda root, rows=None: replays.append(rows) or {})
    text = build_health_invariants(_env(tmp_path), task_id="own", attempt_key="1")
    assert _delegated_lines(tmp_path, task_id="own") == [] and selections == ["own", "own"] and replays == []
    unavailable = [line for line in text.splitlines() if "CUSTODY UNAVAILABLE" in line]
    assert len(unavailable) == 1 and "StoreBusy: the selected read failed" in unavailable[0]
    assert "query_data(op='sql', db='current'" in unavailable[0] and "custody_runs" in unavailable[0]


def test_both_reads_failing_stays_a_failure_and_rewrites_no_fact(tmp_path, monkeypatch):
    from ouroboros.context_health import build_health_invariants

    _seed_health(tmp_path)
    before = _stored(tmp_path)
    assert _delegated_lines(tmp_path, task_id="own")
    assert _stored(tmp_path) == before, "a successful build rewrites nothing either"

    def unreadable(*_args, **_kwargs):
        raise store.StoreBusy("custody unreadable")

    monkeypatch.setattr(current, "health_runs", unreadable)
    monkeypatch.setattr(custody, "replay", unreadable)
    text = build_health_invariants(_env(tmp_path), task_id="own", attempt_key="1")
    assert text.startswith("## Health Invariants")
    assert [line for line in text.splitlines() if "DELEGATED" in line] == [], "no line claims an answer"
    assert _stored(tmp_path) == before


def test_malformed_facts_still_reach_the_original_decode_failure(tmp_path):
    _seed_health(tmp_path)
    with store.open_store(tmp_path, write=True) as txn:
        txn.conn.execute(
            "INSERT INTO custody_runs (run_id, task_id, project_id, invocation_id, state, facts, updated_at) "
            "VALUES ('run-torn', 'other', '', '', 'closed', '{\"run_id\": \"run-torn\", \"settled\": tru', '')")
    with pytest.raises(ValueError):
        current.health_runs(tmp_path, "own")  # a torn row is selected, never silently skipped
    assert _delegated_lines(tmp_path, task_id="own") == []
    assert _over_every_row(_delegated_lines, tmp_path, task_id="own") == []


# --- payload busy: one transaction, distinct top-level owners --------------------------


def _caller(tmp_path, monkeypatch, task_id="t-caller"):
    ctx = _payload_ctx(tmp_path, monkeypatch)
    ctx.task_id = task_id
    ctx.task_metadata = {"root_task_id": task_id}
    return ctx


def _request_holder(drive, skill, *, invocation_id="inv-holder", task_id="t-holder"):
    assert custody.record_start_requested(
        drive, run_id="", task_id=task_id, idempotency_key=f"k-{invocation_id}", invocation_id=invocation_id,
        max_seconds=60, request={"prompt": "hold the payload"}, project_id="p-holder", project_owned=False,
        route="r", root_task_id=task_id, parent_task_id="", snapshot_id="snap-holder", execution_root="/x/exec",
        baseline_sha="b1", target_root=str(skill), authority_source="skill_payload",
        resource_ref={"root": "skill_payload", "source": "external", "skill_name": "alpha",
                      "target_root": str(skill), "payload_hash": "h1"})


def _start_holder(drive, skill, *, run_id="run-holder", invocation_id="inv-holder", task_id="t-holder",
                  snapshot=True, settle=None, dispose="", owner_status=""):
    from ouroboros.task_results import write_task_result

    assert custody.record_started(drive, custody.RunCustody(
        run_id=run_id, task_id=task_id, route_id="r", invocation_id=invocation_id,
        authority_source="skill_payload", target_root=str(skill), snapshot_id="snap-holder" if snapshot else ""))
    if settle is not None:
        assert custody.emit(drive, custody.SETTLED, {"run_id": run_id, "task_id": task_id, "state": "succeeded",
                                                     **settle})
    if dispose:
        assert custody.emit(drive, custody.PATCH_DISPOSED, {"run_id": run_id, "task_id": task_id,
                                                            "disposition": dispose})
    if owner_status:
        write_task_result(drive, task_id, owner_status)
    custody._CUSTODY.clear()


def _start(ctx):
    return json.loads(_exact_payload_start(ctx, "edit the skill", root="skill_payload", bucket="external",
                                           skill_name="alpha"))


def _no_gateway(monkeypatch):
    import ouroboros.claudexor_daemon as daemon

    def refuse():
        raise AssertionError("a busy refusal lands BEFORE any gateway work")

    monkeypatch.setattr(daemon, "ensure_owned_gateway", refuse)


def _stub_gateway(monkeypatch):
    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways import claudexor as gw

    seen: dict = {}
    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _StartStub(seen))
    delegate._CUSTODY.clear()
    return seen


# (holder setup, expected holder id or "" when the caller's start proceeds)
_HOLDERS = {
    "live run": (dict(), "run-holder"),
    "unsettled but disposed": (dict(dispose="applied"), "run-holder"),
    "closed, undisposed, not held": (dict(snapshot=False, settle={}), "run-holder"),
    "settled undisposed, nonfinite spend": (dict(settle={"cost_usd": NAN, "spend_disclosed": True}), "run-holder"),
    "settled and disposed": (dict(settle={}, dispose="applied"), ""),
    "settled and disposed, nonfinite spend": (dict(settle={"cost_usd": INF, "spend_disclosed": True},
                                                   dispose="rejected"), ""),
    "live run of a terminal owner": (dict(owner_status="failed"), ""),
}


@pytest.mark.parametrize("case", sorted(_HOLDERS))
def test_foreign_payload_holders_are_inventory_not_an_admission_limit(tmp_path, monkeypatch, case):
    setup, _holder = _HOLDERS[case]
    caller = _caller(tmp_path, monkeypatch)
    drive = custody.custody_root(caller)
    skill = _seed_skill(tmp_path / "data")
    _start_holder(drive, skill, **setup)
    _stub_gateway(monkeypatch)
    assert _start(caller)["status"] == "started"
    assert "run-holder" in current.runs(drive)
    custody._CUSTODY.clear()


def test_foreign_pending_payload_invocation_is_preserved_beside_new_start(tmp_path, monkeypatch):
    caller = _caller(tmp_path, monkeypatch)
    drive = custody.custody_root(caller)
    skill = _seed_skill(tmp_path / "data")
    _request_holder(drive, skill)
    _stub_gateway(monkeypatch)
    assert _start(caller)["status"] == "started"
    assert any(row["invocation_id"] == "inv-holder" for row in custody.pending_invocations(drive))


@pytest.mark.parametrize("unreadable", [False, True])
def test_a_claim_preserves_unreadable_custody_refusal(tmp_path, monkeypatch, unreadable):
    from ouroboros.delegate_start_claims import claimed_start_request

    writes = []
    monkeypatch.setattr(custody, "custody_log_unreadable", lambda _root: unreadable)
    monkeypatch.setattr(custody, "record_start_requested", lambda *_a, **kw: writes.append(kw) or True)
    accepted, refusal = claimed_start_request(tmp_path, actor_ctx=SimpleNamespace(), task_id="caller")
    assert accepted != unreadable
    if unreadable:
        assert refusal["reason"] == "replacement_custody_unknown" and not writes
    else:
        assert len(writes) == 1


# --- by-id readers: the same answer, None/KeyError and the one memo seed ---------------


def _answer(result):
    status, entry, extra = result
    return status, extra, None if entry is None else vars(entry)


@pytest.mark.parametrize("run_id", ["run-own", "run-snapshot", "run-missing", "", None, " run-own"])
def test_retry_result_status_answers_as_over_every_row(tmp_path, run_id):
    from ouroboros.delegate_shared import retry_result_status

    _seed_health(tmp_path)
    ctx = SimpleNamespace(task_id="own")
    keyed = retry_result_status(ctx, tmp_path, run_id)
    assert _answer(keyed) == _answer(retry_result_status(ctx, tmp_path, run_id, state=custody.replay(tmp_path)))
    assert keyed[0] == {"run-own": custody.OWNED, "run-snapshot": custody.FOREIGN}.get(run_id, custody.UNKNOWN)


_STALE = custody.RunCustody(run_id="", task_id="reader", settled=True, snapshot_id="snap-stale")


@pytest.mark.parametrize("run_id, memo, owner_terminal", [
    ("run-snapshot", {}, False),
    ("run-snapshot", {}, True),
    ("run-snapshot", {"run-snapshot": _STALE}, False),   # a stale memo is replaced by the durable row
    ("run-ghost", {"run-ghost": _STALE}, False),         # no durable row: the stale memo still answers
    ("run-ghost", {}, False),
    ("run-snapshot", {"run-other": _STALE}, False),      # an unrelated entry neither moves nor multiplies
])
def test_orphan_disposition_keeps_its_one_memo_seed(tmp_path, monkeypatch, run_id, memo, owner_terminal):
    from ouroboros.delegate_shared import orphan_disposition_status
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.registry import ToolContext

    _seed_health(tmp_path)
    if owner_terminal:
        write_task_result(tmp_path, "owner", "failed")
    reader = ToolContext(repo_dir=tmp_path / "repo", drive_root=tmp_path)
    reader.task_id = "reader"
    outcomes = []
    for state in (None, "every row"):
        fresh = {rid: copy.deepcopy(entry) for rid, entry in memo.items()}
        monkeypatch.setattr(custody, "_CUSTODY", fresh)
        answer = orphan_disposition_status(reader, tmp_path, run_id,
                                           state=None if state is None else custody.replay(tmp_path))
        outcomes.append((_answer(answer), {rid: vars(entry) for rid, entry in fresh.items()}))
    assert outcomes[0] == outcomes[1]
    (status, orphan_of, _facts), seeded = outcomes[0]
    assert set(seeded) == set(memo) | ({run_id} if run_id == "run-snapshot" else set())
    if run_id == "run-ghost":
        assert status == (custody.OWNED if memo else custody.UNKNOWN)
    else:
        assert seeded[run_id]["task_id"] == "owner", "the durable row replaced any stale memo"
        assert (status, orphan_of) == ((custody.OWNED, "owner") if owner_terminal else (custody.FOREIGN, ""))


def test_a_terminal_owners_orphan_is_rejected_through_by_id_reads(tmp_path, monkeypatch):
    """The apply gate (orphan status, its re-check and the hand-over refusal) reads the one
    row it names; only the post-disposition disclosure refresh still reads every row."""
    from ouroboros.tools.subagent_integration import _integrate_delegated_patch
    from tests.test_delegated_run_isolation_orphans import _captured_flat, _disposer

    target, _ctx, _entry, _block = _captured_flat(tmp_path, monkeypatch)
    disposer = _disposer(tmp_path, monkeypatch, target, target)
    replayed_by, real_replay = [], custody.replay

    def spy(*args, **kwargs):
        replayed_by.append(inspect.stack()[1].function)
        return real_replay(*args, **kwargs)

    monkeypatch.setattr(custody, "replay", spy)
    out = _integrate_delegated_patch(disposer, "run-read", "reject", "abandoned by its terminal owner")
    assert "🚫 Rejected" in out, out
    assert set(replayed_by) <= {"refresh_disposed_reconciliation"}, replayed_by
    custody._CUSTODY.clear()
    assert custody.replay(custody.custody_root(disposer))["run-read"].patch_disposed == "rejected"


def _own_readonly(root, run_id="run-c", **facts):
    _put_many(root, [(run_id, {"task_id": "own", "settled": True, "terminal_state": "succeeded",
                               "access": "readonly", **facts})])


_CONTINUE = dict(actor={}, route=SimpleNamespace(route_id=""), authority=SimpleNamespace(access="readonly"),
                 target_root="")


def test_bind_continuation_hands_its_task_line_the_named_row_as_a_mapping(tmp_path, monkeypatch):
    from ouroboros import delegate_continuation as continuation

    _own_readonly(tmp_path)
    _put_many(tmp_path, [("run-f", {"task_id": "owner", "settled": True, "terminal_state": "succeeded",
                                    "access": "readonly"})])
    lines, real_line = [], continuation._task_line
    monkeypatch.setattr(continuation, "_task_line",
                        lambda ctx, drive, state, entry: lines.append((state, entry)) or real_line(ctx, drive, state, entry))
    ctx = SimpleNamespace(task_id="own")
    facts, code, _ = continuation.bind_continuation(ctx, tmp_path, " run-c ", **_CONTINUE)
    assert code == "" and (facts["continuation_of"], facts["task_line"]) == ("run-c", "own")
    state, entry = lines[-1]
    assert isinstance(state, dict) and list(state) == ["run-c"] and state["run-c"] is entry
    assert continuation.bind_continuation(ctx, tmp_path, "run-missing", **_CONTINUE)[1] == \
        continuation.REFUSAL_SOURCE_UNKNOWN
    assert continuation.bind_continuation(ctx, tmp_path, "run-f", **_CONTINUE)[1] == \
        continuation.REFUSAL_SOURCE_NOT_OWNED


@pytest.mark.parametrize("access", ["readonly", "workspace_write"])
def test_start_binding_keeps_the_missing_row_key_error(tmp_path, access):
    """Only a writing continuation reaches the snapshot read's subscript; both subscripts keep KeyError."""
    from ouroboros import delegate_continuation as continuation

    target = str(tmp_path / "target")
    _own_readonly(tmp_path, access=access, target_root=target)
    gateway = SimpleNamespace(engine_version="test",
                              agent_capabilities=lambda: {"runControlKeys": [continuation.ENGINE_CONTINUE_KEY]})
    common = dict(gateway=gateway, invocation_id="inv-next", text="carry on", source_binding={},
                  **{**_CONTINUE, "authority": SimpleNamespace(access=access), "target_root": target})
    ctx = SimpleNamespace(task_id="own")
    start, refusal = continuation.start_binding(ctx, tmp_path, "run-c", **common)
    assert refusal is None and start.custody == {"continuation_of": "run-c"}
    assert start.request["continueFrom"] == "run-c"
    with pytest.raises(KeyError):  # the floors strip the token; the row subscripts never did
        continuation.start_binding(ctx, tmp_path, " run-c", **common)


def test_the_snapshot_hand_over_reads_the_one_predecessor_row(tmp_path):
    from ouroboros import delegate_continuation as continuation

    _put_many(tmp_path, [
        ("run-p", {"task_id": "own", **SETTLED, "snapshot_id": "snap-p"}),
        ("run-live", {"task_id": "own", "snapshot_id": "snap-l"}),
    ])
    cases = [
        ({"continuation_of": "run-missing", "snapshot_id": "snap-p"}, continuation.REFUSAL_SOURCE_UNKNOWN),
        ({"continuation_of": "run-p", "snapshot_id": "snap-p", "invocation_id": "inv-1"}, None),
        ({"continuation_of": "run-live", "snapshot_id": "snap-l"}, continuation.REFUSAL_SOURCE_NOT_TERMINAL),
        ({"continuation_of": "run-p", "snapshot_id": "snap-other"}, continuation.REFUSAL_SNAPSHOT_RELEASED),
    ]
    for row, reason in cases:
        with continuation.snapshot_handover(tmp_path, row) as refusal:
            assert refusal.get("reason") == reason, (row, refusal)


def test_sleep_selection_reads_its_own_rows_from_the_canonical_root(tmp_path):
    from ouroboros import model_sleep
    from tests.test_model_sleep import _ctx, _result

    _result(tmp_path, "sleeper")
    _put_many(tmp_path, [
        ("run-own", {"task_id": "sleeper", "settled": True, "terminal_state": "succeeded"}),
        ("run-live", {"task_id": "sleeper"}),
        ("run-foreign", {"task_id": "other", "settled": True}),
    ])
    ctx = _ctx(tmp_path)
    ctx.drive_root = tmp_path / "child-drive"  # a child's own drive: custody lives on the canonical root
    oracle = [(run.run_id, run.terminal_state in custody.TERMINAL_STATES or bool(run.settled))
              for run in custody.replay(tmp_path).values() if run.task_id == "sleeper"]
    actual = [(rid, run.terminal_state in custody.TERMINAL_STATES or bool(run.settled))
              for rid, run in model_sleep._owned_runs(ctx).items()]
    assert actual == oracle == [("run-live", False), ("run-own", True)]
    assert model_sleep.selectors(ctx, runs=["run-own", "run-live"])["runs"] == ["run-own", "run-live"]
    for ghost in ("run-foreign", "run-missing"):
        with pytest.raises(ValueError, match=f"runs: {ghost} are not delegated runs of this task"):
            model_sleep.selectors(ctx, runs=[ghost])
    assert model_sleep.wake_reason(ctx, {"runs": ["run-live", "run-own"]}) == "run:run-own"
    assert model_sleep.wake_reason(ctx, {"runs": ["run-live"]}) == ""


# --- scale: Python decodes independent of unrelated closed history ---------------------


def _scale_store(root, unrelated):
    _seed_health(root)
    _put_many(root, [
        ("run-pay-live", {"task_id": "t-holder", "authority_source": "skill_payload", "target_root": "/payload/alpha",
                          "snapshot_id": "snap-pay"}),
        ("run-pay-done", {"task_id": "t-holder", "authority_source": "skill_payload", "target_root": "/payload/alpha",
                          "snapshot_id": "snap-done", **SETTLED, "patch_disposed": "applied"}),
    ])
    # Ordinary valid-JSON closed history: consumed outputs, foreign tasks, disposed payload runs.
    _put_many(root, [(f"old-{index:05d}", {
        "task_id": f"t-{index % 97}", "settled": True, "terminal_state": "succeeded", "output_artifact": "o.out",
        "output_complete": True, "output_consumed": True, "cost_usd": 0.01, "spend_disclosed": True,
        **({"authority_source": "skill_payload", "target_root": "/payload/beta", "snapshot_id": f"snap-{index}",
            "patch_disposed": "applied"} if index % 10 == 0 else {}),
    }) for index in range(unrelated)])


def _hand_over(root, row):
    from ouroboros import delegate_continuation as continuation

    with continuation.snapshot_handover(root, row) as refusal:
        return refusal


def _consumer_decodes(root):
    from ouroboros import delegate_continuation as continuation
    from ouroboros import model_sleep
    from ouroboros.delegate_shared import orphan_disposition_status, retry_result_status
    from ouroboros.tools.registry import ToolContext

    reader = ToolContext(repo_dir=root / "repo", drive_root=root)
    reader.task_id = "reader"
    own = SimpleNamespace(task_id="own", drive_root=root, budget_drive_root=str(root), task_metadata={})
    consumers = {
        "health": lambda: _delegated_lines(root, task_id="own"),
        "sleep": lambda: model_sleep._owned_runs(own),
        "retry": lambda: retry_result_status(own, root, "run-snapshot"),
        "orphan": lambda: orphan_disposition_status(reader, root, "run-snapshot"),
        "continuation": lambda: continuation.bind_continuation(own, root, "run-own", **_CONTINUE),
        "hand-over": lambda: _hand_over(root, {"continuation_of": "run-snapshot", "snapshot_id": "snap-1"}),
        "payload inventory": lambda: current.payload_snapshot(root),
        "every row (old input)": lambda: custody.replay(root),
    }
    counts, answers = {}, {}
    for name, consumer in consumers.items():
        custody._CUSTODY.clear()
        with pytest.MonkeyPatch.context() as patch:
            seen = []
            real = current._decode
            patch.setattr(current, "_decode", lambda raw, seen=seen, real=real: seen.append(1) or real(raw))
            answers[name] = consumer()
        counts[name] = len(seen)
    custody._CUSTODY.clear()
    assert "run-pay-live" in answers["payload inventory"]["runs"] and answers["hand-over"] == {}
    sleep_terminals = [(rid, run.terminal_state in custody.TERMINAL_STATES or bool(run.settled))
                       for rid, run in answers["sleep"].items()]
    assert sleep_terminals == [("run-own", True), ("run-own-acked", True), ("run-own-live", False)]
    return counts


def test_python_decodes_do_not_grow_with_unrelated_closed_rows(tmp_path):
    small, large = tmp_path / "ten", tmp_path / "five-thousand"
    for root, unrelated in ((small, 10), (large, 5000)):
        root.mkdir()
        _scale_store(root, unrelated)
    few, many = _consumer_decodes(small), _consumer_decodes(large)
    oracle = "every row (old input)"
    assert many[oracle] - few[oracle] == 4990, "the spy sees every decode of the old input"
    assert {name: count for name, count in many.items() if name != oracle} == \
        {name: count for name, count in few.items() if name != oracle}, (few, many)
    assert few["retry"] == few["continuation"] == 1
    assert few["orphan"] == 2, "its named row, then ``lookup``'s fresh durable read of the same row"
