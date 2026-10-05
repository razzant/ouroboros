"""Bindings come from the live ledger alone: no archive walk on any money path.

Three header shapes an installed ledger can carry — no ``binding_authority``
stamp (a pre-carriage pass), ``"unknown"`` (an operator's hand edit) and
``carried`` whose payloads are the literal ``"unknown"`` (a pass that compacted
an unknown) — all resolve from the block rows' own cap literals
(``legacy_live``) or, for a member whose rows disagree or carry no cap, from the
configured cap disclosed on the task (``legacy_default``), while the archive
walker raises. The archive stays readable for explicit history questions.
"""
from __future__ import annotations

import pytest

from ouroboros import usage_accounting as ua
from ouroboros import usage_compaction as compact
from ouroboros.usage_admission import (
    ledger_billing_binding, original_group_limit, task_billing_fields, task_money_snapshot,
)
from tests.test_batch4_compaction_authority import (
    _cold, _compact, _legacy_attempt, _rewrite, _rows, _strip_carriage, _sums,
)
from tests.test_billing_group import _spend, data_root as data_root

HEADERS = ("absent", "unknown", "carried_unknown")


def _walker_raises(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("the usage archive was opened on a money path")
    for name in ("_archive_chain", "_load_segment", "archived_attempt_ids"):
        monkeypatch.setattr(compact, name, forbidden)


def _reshape_header(root, shape):
    rows = _rows(root)
    if shape == "absent":
        _strip_carriage(root)
        return
    if shape == "unknown":
        rows[0]["binding_authority"] = "unknown"
    else:
        for row in rows:
            for field in ("original_root_binding", "original_group_binding"):
                if field in row:
                    row[field] = "unknown"
    _rewrite(root, rows)
    _cold(root)


def _history(root):
    from ouroboros.task_results import write_task_result

    write_task_result(root, "P", "failed", root_task_id="P")  # legacy roots: a result row, no pinned binding
    write_task_result(root, "Q", "failed", root_task_id="Q")
    _legacy_attempt(root, "p-first", model="m", cap=20.0)
    _legacy_attempt(root, "p-later", model="m", cap=100.0)  # one aggregate: its minimum literal, $20
    _legacy_attempt(root, "q-first", model="a", cap=5.0, rid="Q")
    _legacy_attempt(root, "q-later", model="z", cap=50.0, rid="Q")  # two aggregates, two literals


@pytest.mark.parametrize("shape", HEADERS)
def test_money_paths_never_open_the_archive_under_any_legacy_header(data_root, monkeypatch, shape):
    from ouroboros.task_results import load_task_result

    root = ua._drive_root(data_root)
    _history(root)
    _compact(root, monkeypatch)
    _reshape_header(root, shape)
    _walker_raises(monkeypatch)
    # An old root whose block carries one literal binds from it, disclosed.
    assert original_group_limit(root, "P") == {"limit_usd": 20.0, "source": "legacy_live"}
    assert ledger_billing_binding(root, "P") == {
        "billing_group_id": "P", "billing_group_limit_usd": 20.0,
        "billing_group_limit_source": "legacy_live", "billing_group_limit_revision": None}
    assert task_money_snapshot(root, {"id": "P"}, "P")["root_axis"]["limit_usd"] == 20.0
    # An old root whose block rows disagree stays open: admission binds it as its own
    # group under the configured cap and discloses that on the task.
    assert original_group_limit(root, "Q") == {"limit_usd": None, "source": "no_attempt_recorded"}
    assert ledger_billing_binding(root, "Q") == {}
    pinned = task_billing_fields({"id": "Q"}, "Q", 30.0, root, pin_initial=True)
    assert (pinned["billing_group_id"], pinned["billing_group_limit_usd"],
            pinned["billing_group_limit_source"]) == ("Q", 30.0, "legacy_default")
    assert load_task_result(root, "Q")["billing_group"]["billing_group_limit_source"] == "legacy_default"
    # Billing for a new id, then reserve -> dispatch -> settle.
    fresh = task_billing_fields({"id": "N"}, "N", 7.0, root, pin_initial=True)
    assert fresh["billing_group_limit_source"] == "initial_task_admission"
    group = {key: value for key, value in fresh.items() if key.startswith("billing_group_")}
    _spend(root, ua.UsageScope(drive_root=root, task_id="N", root_task_id="N", root_limit_usd=7.0, **group), .25)
    assert ua.usage_projection(root, root_task_id="N")["accounted_usd"] == .25
    # The old roots' spend is preserved, and late work on them spends against their groups.
    assert ua.usage_projection(root, billing_group_id="P")["accounted_usd"] == 1.0
    _spend(root, ua.UsageScope(drive_root=root, task_id="P-late", root_task_id="P", **ledger_billing_binding(root, "P")), .5)
    assert ua.usage_projection(root, billing_group_id="P")["accounted_usd"] == 1.5
    with pytest.raises(ua.BudgetExceeded):  # the live literal is a real ceiling
        _spend(root, ua.UsageScope(drive_root=root, task_id="P-late", root_task_id="P", **ledger_billing_binding(root, "P")), 19.0)


@pytest.mark.serial
@pytest.mark.parametrize("shape", HEADERS)
def test_continue_of_an_old_root_binds_live_or_default_without_the_archive(data_root, monkeypatch, shape):
    from supervisor.continuation_admission import admit_continuation
    from tests._budget_pause_exact_helpers import _install_queue
    from tests.test_owner_continue import _interrupted, NONCE

    root = ua._drive_root(data_root)
    _queue, _state, workers = _install_queue(root, monkeypatch)
    _legacy_attempt(root, "p-first", model="m", cap=20.0)
    _interrupted(root, "P", reason_code="task_exception")  # while the ledger binds P: no pinned binding written
    _legacy_attempt(root, "p-later", model="m", cap=100.0)
    _legacy_attempt(root, "q-first", model="a", cap=5.0, rid="Q")
    _interrupted(root, "Q", reason_code="task_exception")
    _legacy_attempt(root, "q-later", model="z", cap=50.0, rid="Q")
    _compact(root, monkeypatch)
    _reshape_header(root, shape)
    _walker_raises(monkeypatch)
    monkeypatch.setenv("OUROBOROS_PER_TASK_COST_USD", "40")
    accepted = admit_continuation("P", action_nonce=NONCE)
    assert accepted["ok"] and not accepted["held"]
    binding = workers.PENDING[-1]["metadata"]["continuation"]
    assert (binding["billing_group_id"], binding["billing_group_limit_usd"],
            binding["billing_group_limit_source"]) == ("P", 20.0, "legacy_live")
    accepted = admit_continuation("Q", action_nonce=NONCE)
    assert accepted["ok"] and not accepted["held"]
    binding = workers.PENDING[-1]["metadata"]["continuation"]
    assert (binding["billing_group_id"], binding["billing_group_limit_usd"],
            binding["billing_group_limit_source"]) == ("Q", 40.0, "legacy_default")


def test_the_next_pass_carries_the_live_literal_and_a_cold_read_needs_no_archive(data_root, monkeypatch):
    root = ua._drive_root(data_root)
    _legacy_attempt(root, "first", model="m", cap=20.0)
    _legacy_attempt(root, "later", model="m", cap=100.0)
    _compact(root, monkeypatch)
    _strip_carriage(root)
    for index in range(8):
        _legacy_attempt(root, f"n-{index}", model="m", cap=7.0, rid="N")
    _walker_raises(monkeypatch)
    sums = _sums(root, "P", "N")
    assert _compact(root, monkeypatch)
    carrier = next(row for row in _rows(root) if row.get("root_task_id") == "P" and "original_root_binding" in row)
    assert carrier["original_root_binding"]["billing_group_limit_source"] == "legacy_live"
    assert float(carrier["original_root_binding"]["root_limit_usd"]) == 20.0
    _cold(root)
    assert original_group_limit(root, "P") == {"limit_usd": 20.0, "source": "legacy_live"}
    assert ledger_billing_binding(root, "P")["billing_group_limit_usd"] == 20.0
    assert _sums(root, "P", "N") == sums


def test_a_carried_binding_outranks_the_literal_and_a_disputed_literal_stays_open(data_root, monkeypatch):
    root = ua._drive_root(data_root)
    _legacy_attempt(root, "first", model="z", cap=20.0)
    _legacy_attempt(root, "later", model="a", cap=100.0)  # two aggregates; a-model's $100 sorts first
    _compact(root, monkeypatch)
    assert original_group_limit(root, "P") == {"limit_usd": 20.0, "source": "ledger_first_row"}  # carried wins
    _strip_carriage(root)
    assert original_group_limit(root, "P") == {"limit_usd": None, "source": "no_attempt_recorded"}  # disputed: open
    assert ledger_billing_binding(root, "P") == {}
    for index in range(4):
        _legacy_attempt(root, f"n-{index}", model="m", cap=7.0, rid="N")
    _compact(root, monkeypatch)
    rows = _rows(root)
    assert next(row for row in rows if row.get("root_task_id") == "P" and "original_root_binding" in row)[
        "original_root_binding"] == "unbound"
    _cold(root)
    assert original_group_limit(root, "P") == {"limit_usd": None, "source": "no_attempt_recorded"}
    _legacy_attempt(root, "late", model="m", cap=30.0)  # the first original row after an open block binds it
    assert original_group_limit(root, "P") == {"limit_usd": 30.0, "source": "ledger_first_row"}


@pytest.mark.parametrize("capless_model", ["a", "z"])  # the cap-less aggregate sorts first or last
def test_a_capless_block_row_does_not_vote_in_either_order(data_root, monkeypatch, capless_model):
    """One aggregate without a cap beside one with a cap: the cap is the member's one literal, whichever sorts first."""
    from tests.test_batch4_compaction_authority import UNCAPPED

    root = ua._drive_root(data_root)
    _legacy_attempt(root, "free", model=capless_model, cap=UNCAPPED)
    _legacy_attempt(root, "capped", model="z" if capless_model == "a" else "a", cap=20.0)
    _legacy_attempt(root, "foreign", model="m", cap=7.0, rid="F")
    _compact(root, monkeypatch)
    _strip_carriage(root)
    _walker_raises(monkeypatch)
    assert original_group_limit(root, "P") == {"limit_usd": 20.0, "source": "legacy_live"}
    assert ledger_billing_binding(root, "P")["billing_group_limit_usd"] == 20.0
    assert original_group_limit(root, "F") == {"limit_usd": 7.0, "source": "legacy_live"}
    _legacy_attempt(root, "late", model="m", cap=30.0)  # a later original row does not replace the block's literal
    assert original_group_limit(root, "P") == {"limit_usd": 20.0, "source": "legacy_live"}


def test_capless_rows_joining_a_bound_member_later_keep_its_binding(data_root, monkeypatch):
    """The answer does not depend on when compaction ran: cap-less work folded into a carried member changes nothing."""
    from tests.test_batch4_compaction_authority import UNCAPPED

    root = ua._drive_root(data_root)
    for index in range(3):
        _legacy_attempt(root, f"capped-{index}", model="m", cap=20.0)
    _compact(root, monkeypatch)
    _strip_carriage(root)
    assert original_group_limit(root, "P") == {"limit_usd": 20.0, "source": "legacy_live"}
    for index in range(4):
        _legacy_attempt(root, f"free-{index}", model="a", cap=UNCAPPED)  # later cap-less work of the same root
    _compact(root, monkeypatch)  # the block is stamped carried now; the cap-less aggregate joins it
    _cold(root)
    _walker_raises(monkeypatch)
    assert original_group_limit(root, "P") == {"limit_usd": 20.0, "source": "legacy_live"}
    _strip_carriage(root)  # and the same rows read uncarried agree
    assert original_group_limit(root, "P") == {"limit_usd": 20.0, "source": "legacy_live"}


def test_a_writer_prepared_before_an_atomic_replacement_is_prepared_again(data_root, monkeypatch):
    """A compaction that lands between preparation and the lock makes the writer prepare the new generation."""
    from ouroboros import _usage_rows_memo as memo

    root = ua._drive_root(data_root)
    _legacy_attempt(root, "first", model="z", cap=20.0)
    _legacy_attempt(root, "later", model="a", cap=100.0)
    for index in range(8):
        _legacy_attempt(root, f"new-{index}", model="m", cap=7.0, rid="N")
    _cold(root)
    original = memo._prepare_writer
    prepared = []

    def prepare(path):
        result = original(path)
        prepared.append(result)
        if len(prepared) == 1:
            _compact(path, monkeypatch)  # atomic replacement after this preparation
        return result

    monkeypatch.setattr(memo, "_prepare_writer", prepare)
    assert ledger_billing_binding(root, "P")["billing_group_limit_usd"] == 20.0
    assert len(prepared) == 2
    with memo._writer_locked(root) as view:
        assert view is prepared[1]


def test_check_budget_runs_in_the_server_and_is_skipped_in_a_worker(monkeypatch, tmp_path):
    """Both directions: a worker never folds the ledger at construction; the server still checks."""
    import json
    from types import SimpleNamespace

    from ouroboros import agent_startup_checks as checks
    from ouroboros.utils import WORKER_PROCESS_ENV

    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "state.json").write_text(json.dumps({"mode": "idle"}), encoding="utf-8")
    env = SimpleNamespace(budget_drive_root=tmp_path, drive_path=lambda name: tmp_path / name)
    folds = []
    monkeypatch.setattr("ouroboros.settings_setup_contract.resolve_total_budget_usd", lambda: 50.0)
    monkeypatch.setattr("ouroboros.usage_accounting.ensure_legacy_imported", lambda root: folds.append(("import", root)))
    monkeypatch.setattr("ouroboros.usage_accounting.usage_projection",
                        lambda root, **kw: folds.append(("projection", root)) or {"accounted_usd": 1.0, "remaining_known_usd": 49.0})
    monkeypatch.setenv(WORKER_PROCESS_ENV, "1")
    assert checks.check_budget(env) == ({"status": "skipped", "reason": "worker_process"}, 0)
    assert folds == []
    monkeypatch.delenv(WORKER_PROCESS_ENV)
    result, issues = checks.check_budget(env)
    assert (result["status"], issues) == ("ok", 0)
    assert [kind for kind, _ in folds] == ["import", "projection"]


def test_group_axis_agreement_ignores_the_member_roots_own_caps(data_root, monkeypatch):
    """Two member roots with different root caps share one group cap: the group literal is unambiguous."""
    from tests.test_billing_group import _scope

    root = ua._drive_root(data_root)
    initial = dict(billing_group_limit_source="initial_task_admission", billing_group_limit_revision="r1")
    for index in range(3):
        _spend(root, _scope(root, "S", "S", group="P", group_limit=20.0, root_limit=5.0, **initial), .25)
        _spend(root, _scope(root, "T", "T", group="P", group_limit=20.0, root_limit=10.0, parent_task_id="S", **initial), .25)
    _compact(root, monkeypatch)
    _strip_carriage(root)
    _walker_raises(monkeypatch)
    assert original_group_limit(root, "P") == {"limit_usd": 20.0, "source": "legacy_live"}
    for rid, cap in (("S", 5.0), ("T", 10.0)):
        binding = ledger_billing_binding(root, rid)
        assert (binding["billing_group_id"], binding["billing_group_limit_usd"], binding["billing_group_limit_source"],
                binding["billing_group_limit_revision"]) == ("P", 20.0, "legacy_live", "r1")
        assert task_money_snapshot(root, {"id": rid}, rid)["root_axis"]["limit_usd"] == cap


@pytest.mark.serial
@pytest.mark.parametrize("compacted", [False, True])
def test_continue_keeps_the_group_the_successor_roots_own_rows_name(data_root, monkeypatch, compacted):
    """A root that spent inside group P continues inside P, never as a fresh group under today's cap."""
    import json

    from ouroboros.task_results import task_result_path
    from supervisor.continuation_admission import admit_continuation
    from tests._budget_pause_exact_helpers import _install_queue
    from tests.test_owner_continue import _interrupted, NONCE

    root = ua._drive_root(data_root)
    _queue, _state, workers = _install_queue(root, monkeypatch)
    for index in range(4):
        _legacy_attempt(root, f"p-{index}", model="m", cap=20.0, rid="P", billing_group_id="P", billing_group_limit_usd=20.0)
        _legacy_attempt(root, f"s-{index}", model="m", cap=20.0, rid="S", billing_group_id="P", billing_group_limit_usd=20.0)
    _interrupted(root, "S", reason_code="task_exception")
    path = task_result_path(root, "S")
    record = json.loads(path.read_text(encoding="utf-8"))
    record.pop("billing_group", None)  # a pre-group-era successor: no pinned binding on its result
    path.write_text(json.dumps(record), encoding="utf-8")
    if compacted:
        _compact(root, monkeypatch)
        _strip_carriage(root)
    _walker_raises(monkeypatch)
    monkeypatch.setenv("OUROBOROS_PER_TASK_COST_USD", "999")
    before = ua.usage_projection(root, billing_group_id="P")["accounted_usd"]
    accepted = admit_continuation("S", action_nonce=NONCE)
    assert accepted["ok"] and not accepted["held"]
    binding = workers.PENDING[-1]["metadata"]["continuation"]
    assert (binding["billing_group_id"], binding["billing_group_limit_usd"]) == ("P", 20.0)
    assert binding["billing_group_limit_source"] == ("legacy_live" if compacted else "ledger_first_row")
    assert ua.usage_projection(root, billing_group_id="P")["accounted_usd"] == before == 4.0


@pytest.mark.serial
def test_continue_pins_the_default_cap_on_the_predecessor_for_its_own_later_work(data_root, monkeypatch):
    """The configured cap chosen at Continue is the root's binding from then on, whatever the setting becomes."""
    from ouroboros.task_results import load_task_result
    from supervisor.continuation_admission import _billing_group, admit_continuation
    from tests._budget_pause_exact_helpers import _install_queue
    from tests.test_owner_continue import _interrupted, NONCE
    from types import SimpleNamespace

    root = ua._drive_root(data_root)
    _queue, _state, workers = _install_queue(root, monkeypatch)
    _legacy_attempt(root, "first", model="z", cap=20.0)
    _interrupted(root, "P", reason_code="task_exception")  # while the ledger binds P: no pinned binding written
    _legacy_attempt(root, "later", model="a", cap=100.0)  # two aggregates, two literals: P is open
    _compact(root, monkeypatch)
    _strip_carriage(root)
    _walker_raises(monkeypatch)
    assert ledger_billing_binding(root, "P") == {}
    monkeypatch.setenv("OUROBOROS_PER_TASK_COST_USD", "20")
    accepted = admit_continuation("P", action_nonce=NONCE)
    assert accepted["ok"] and not accepted["held"]
    chosen = workers.PENDING[-1]["metadata"]["continuation"]
    assert (chosen["billing_group_id"], chosen["billing_group_limit_usd"], chosen["billing_group_limit_source"]) == ("P", 20.0, "legacy_default")
    pinned = load_task_result(root, "P")["billing_group"]
    assert (pinned["billing_group_id"], pinned["billing_group_limit_usd"], pinned["billing_group_limit_source"]) == ("P", 20.0, "legacy_default")
    monkeypatch.setenv("OUROBOROS_PER_TASK_COST_USD", "999")  # the setting moves; the binding does not
    late = task_billing_fields({"id": "P"}, "P", 999.0, root)
    assert (late["billing_group_limit_usd"], late["billing_group_limit_source"]) == (20.0, "legacy_default")
    assert _billing_group(SimpleNamespace(DRIVE_ROOT=root), "P", load_task_result(root, "P"))["billing_group_limit_usd"] == 20.0
    group = {key: value for key, value in pinned.items() if key.startswith("billing_group_")}
    _spend(root, ua.UsageScope(drive_root=root, task_id="P-late", root_task_id="P", **group), 18.5)
    with pytest.raises(ua.BudgetExceeded):  # the predecessor's own late work shares the successor's ceiling
        _spend(root, ua.UsageScope(drive_root=root, task_id="P-late", root_task_id="P", **group), 1.0)


@pytest.mark.serial
def test_continue_refuses_when_the_default_cap_cannot_be_pinned(data_root, monkeypatch):
    """An unpinned choice is no choice: the Continue is refused (typed) and the same nonce retries later."""
    from ouroboros.task_results import load_task_result
    from supervisor.continuation_admission import admit_continuation
    from tests._budget_pause_exact_helpers import _install_queue
    from tests.test_owner_continue import _interrupted, NONCE

    root = ua._drive_root(data_root)
    _queue, _state, workers = _install_queue(root, monkeypatch)
    _legacy_attempt(root, "first", model="z", cap=20.0)
    _interrupted(root, "P", reason_code="task_exception")
    _legacy_attempt(root, "later", model="a", cap=100.0)
    _compact(root, monkeypatch)
    _strip_carriage(root)
    monkeypatch.setenv("OUROBOROS_PER_TASK_COST_USD", "20")
    from ouroboros import utils

    def locked_out(*_args, **_kwargs):
        raise TimeoutError("result lock held elsewhere")

    monkeypatch.setattr(utils, "update_json_locked", locked_out)
    assert admit_continuation("P", action_nonce=NONCE) == {"ok": False, "error": "billing_authority_unavailable"}
    assert not workers.PENDING and not load_task_result(root, "P").get("billing_group")
    monkeypatch.setattr(utils, "update_json_locked", utils.__dict__["update_json_locked"])
    monkeypatch.undo()  # the lock is free again: the same nonce admits and pins
    monkeypatch.setenv("OUROBOROS_PER_TASK_COST_USD", "20")
    _queue, _state, workers = _install_queue(root, monkeypatch)
    accepted = admit_continuation("P", action_nonce=NONCE)
    assert accepted["ok"] and load_task_result(root, "P")["billing_group"]["billing_group_limit_source"] == "legacy_default"


@pytest.mark.serial
def test_admission_transactions_read_the_ledger_before_taking_the_queue_lock(data_root, monkeypatch):
    """Continue and receipt-backed admission resolve a root's billing (a ledger read) off ``_queue_lock``."""
    import contextlib

    from supervisor import queue
    from supervisor.continuation_admission import admit_continuation
    from supervisor.task_admission import enqueue_with_admission_receipt
    from tests._budget_pause_exact_helpers import _install_queue
    from tests.test_owner_continue import _interrupted, NONCE

    root = ua._drive_root(data_root)
    _queue, _state, workers = _install_queue(root, monkeypatch)
    _interrupted(root, "pred-1")
    owned = []
    original = ua._writer_locked

    @contextlib.contextmanager
    def observed(*args, **kwargs):
        owned.append(queue._queue_lock._is_owned())
        with original(*args, **kwargs) as view:
            yield view

    monkeypatch.setattr(ua, "_writer_locked", observed)
    assert admit_continuation("pred-1", action_nonce=NONCE)["ok"]
    enqueue_with_admission_receipt({"id": "ordinary-root", "type": "task", "text": "probe", "root_task_id": "ordinary-root"},
                                   receipt_required=False)
    assert owned and not any(owned), owned  # the ledger was read, never while this thread held the queue lock


def test_the_archive_stays_readable_for_explicit_history_questions(data_root, monkeypatch):
    root = ua._drive_root(data_root)
    for index in range(4):
        _legacy_attempt(root, f"first-{index}", model="m", cap=20.0)
    _compact(root, monkeypatch)
    _strip_carriage(root)
    assert "first-0" in compact.archived_attempt_ids(root)
    assert compact.usage_attempt_recorded(root, "first-0")
