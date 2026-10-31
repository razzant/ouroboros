"""Imported journals answer every binding question exactly as the journal era did.

``tests/fixtures/usage_store/binding_checkpoints.json.gz`` holds every
authority question the journal-era binding tests asked
(``tests/test_batch4_compaction_authority.py`` and
``tests/test_usage_ledger_legacy_bindings.py`` at 84febbdd3: live journals,
journals compacted once or twice, carriage stripped, unknown or foreign,
amendments, Continue), each with the data root as it was just before the
question (journal, task results, queue snapshot), the budget environment, the
arguments and the shipped answer. Here each question is asked again of a private
copy of its imported starting state: the store's one-time import reads that journal
(aggregates with their weight, ``BindingIndex`` carriage included) and must
answer the same. Provenance: ``record_binding_checkpoints.py`` beside the
fixture.
"""
from __future__ import annotations

import gzip
import json
import pathlib
from types import SimpleNamespace

import pytest

_FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "usage_store" / "binding_checkpoints.json.gz"
_DOC = json.loads(gzip.decompress(_FIXTURE.read_bytes()).decode("utf-8"))
_CHECKPOINTS = _DOC["checkpoints"]


def _decode(value, root):
    if isinstance(value, dict):
        if "__path__" in value:
            return pathlib.Path(value["__path__"].replace("<ROOT>", str(root)))
        if "__ns__" in value:
            return SimpleNamespace(**{key: _decode(item, root) for key, item in value["__ns__"].items()})
        return {key: _decode(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item, root) for item in value]
    if isinstance(value, str):
        return value.replace("<ROOT>", str(root))
    return value


def _encode(value, root):
    if isinstance(value, pathlib.PurePath):
        return {"__path__": str(value).replace(str(root), "<ROOT>")}
    if isinstance(value, str):
        return value.replace(str(root), "<ROOT>")
    if isinstance(value, SimpleNamespace):
        return {"__ns__": {key: _encode(item, root) for key, item in vars(value).items()}}
    if isinstance(value, dict):
        return {str(key): _encode(item, root) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(item, root) for item in value]
    return value


def _billing(binding):
    return {key: value for key, value in (binding or {}).items() if key.startswith("billing_group_")}


def _on_the_known_spend_rule(name, result, expected):
    """The one deliberate change since the journal era (#1487, owner Q4-A): room is
    the limit minus KNOWN (settled) spend, and a snapshot also reports that spend.

    The fixture's bytes stay the shipped answer; this names the translation instead
    of re-recording it. A projection's ``remaining_known_usd`` is recomputed from the
    shipped answer's own ``limit_usd`` and ``settled_usd``; the snapshot's added
    ``settled_usd`` fields are checked for consistency and then set aside (rooms are
    unchanged wherever the shipped answer had no open holds)."""
    def rebase(summary):
        if isinstance(summary, dict) and summary.get("limit_usd") is not None and "remaining_known_usd" in summary:
            summary = {**summary, "remaining_known_usd": round(max(
                0.0, float(summary["limit_usd"]) - float(summary["settled_usd"])), 6)}
        return summary

    if name == "usage_projection" and isinstance(expected, dict):
        expected = rebase(expected)
        if isinstance(expected.get("by_root"), dict):
            expected = {**expected, "by_root": {key: rebase(value) for key, value in expected["by_root"].items()}}
    if name == "task_money_snapshot" and isinstance(result, dict):
        assert result["settled_usd"] <= result["accounted_usd"] + 1e-9
        result = {key: value for key, value in result.items() if key != "settled_usd"}
        for axis in ("root_axis", "group_axis"):
            if isinstance(result.get(axis), dict):
                assert result[axis]["settled_usd"] <= result[axis]["accounted_usd"] + 1e-9
                result[axis] = {key: value for key, value in result[axis].items() if key != "settled_usd"}
    return result, expected


# The task-acceptance flow's root-cap amendments retired with that flow (the review
# fold): ``effective_billing_fields`` overlaid them and is gone, its callers read the
# fields themselves, and no answer names an amendment as its source any more.
_RETIRED_OVERLAY = "effective_billing_fields"
_OVERLAY_KEYS = {f"{axis}_limit_{part}" for axis in ("root", "billing_group") for part in ("usd", "source", "revision")}


def _amended(checkpoint):
    return any("acceptance_root_cap_amendments" in _DOC["blobs"][blob] for blob in checkpoint["files"].values())


def _without_the_retired_overlay(expected):
    """A money projection's shipped answer less the provenance the amendment overlay
    added (a recorded binding keeps what its row says); the limit itself is still
    compared, so an answer whose cap came from an amendment stays red."""
    if not isinstance(expected, dict):
        return expected
    for axis in ("root", "billing_group"):
        if expected.get(f"{axis}_limit_source") == "owner_amendment":
            expected = {key: value for key, value in expected.items()
                        if key not in (f"{axis}_limit_source", f"{axis}_limit_revision")}
    return expected


def test_the_retired_overlay_changed_only_amended_bindings():
    """Every journal-era question the overlay answered without a recorded amendment got
    its own fields back, so reading the fields directly loses nothing there; an amended
    answer differed only in the overlay's own keys."""
    plain = amended = 0
    for checkpoint in _CHECKPOINTS:
        if checkpoint["function"] != _RETIRED_OVERLAY:
            continue
        fields = checkpoint["args"][2]
        changed = {key for key, value in checkpoint["result"].items() if fields.get(key) != value}
        assert changed <= _OVERLAY_KEYS, checkpoint["test"]
        if _amended(checkpoint):
            amended += bool(changed)
        else:
            assert not changed, checkpoint["test"]
            plain += 1
    assert plain and amended


def test_the_fixture_covers_every_question_kind():
    assert {checkpoint["function"] for checkpoint in _CHECKPOINTS} == {
        "original_group_limit", "ledger_billing_binding", "task_billing_fields", "task_money_snapshot",
        "effective_billing_fields", "usage_projection", "_billing_group", "admit_continuation"}
    journals = [_DOC["blobs"][checkpoint["files"]["state/usage_attempts.jsonl"]] for checkpoint in _CHECKPOINTS
                if "state/usage_attempts.jsonl" in checkpoint["files"]]
    assert any('"usage_baseline"' in journal for journal in journals), "compacted journals are replayed"
    assert any("original_root_binding" in journal for journal in journals), "carriage is replayed"


@pytest.fixture(scope="session")
def imported_binding_seeds():
    # Immutable file bytes only. No connection, mutable root or query result is
    # shared; even read-looking questions may pin a previously absent binding.
    return {}


def _materialize_imported_seed(root, checkpoint, seeds):
    from ouroboros import install_identity as identity, usage_store

    # The fixture has 184 file states and 185 (files, environment) states.
    # Keep the environment in the key: defaults can affect an import.
    key = (tuple(sorted(checkpoint["files"].items())), tuple(sorted(checkpoint["env"].items())))
    if key not in seeds:
        _materialize(root, checkpoint)
        usage_store.migrate_from_journal(root)
        usage_store.forget(root)
        # The install identity names its own root; each case's root mints its own on first write.
        snapshot = tuple((path.relative_to(root), path.read_bytes())
                         for path in sorted(root.rglob("*")) if path.is_file()
                         and path.relative_to(root) != pathlib.Path(identity.IDENTITY_PATH))
        # These golden states contain no root paths. Fail explicitly if a future
        # fixture/import changes that, rather than copying an address into another case.
        addresses = (str(root).encode("utf-8"), json.dumps(str(root))[1:-1].encode("utf-8"))
        assert not any(address in body for _path, body in snapshot for address in addresses)
        seeds[key] = snapshot
    else:
        for name, body in seeds[key]:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)


_LIVE = [(index, checkpoint) for index, checkpoint in enumerate(_CHECKPOINTS)
         if checkpoint["function"] != _RETIRED_OVERLAY]


@pytest.mark.parametrize("checkpoint", [checkpoint for _index, checkpoint in _LIVE],
                         ids=[f"{index:03d}-{checkpoint['function']}" for index, checkpoint in _LIVE])
def test_an_imported_journal_gives_the_shipped_answer(checkpoint, tmp_path, monkeypatch, imported_binding_seeds):
    from ouroboros import usage_accounting as ua
    from ouroboros import usage_admission as admission
    from ouroboros import usage_store
    from supervisor import continuation_admission as continuation

    root = tmp_path / "data"
    (root / "state").mkdir(parents=True)
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(root / "settings.json"))
    for key, value in checkpoint["env"].items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    _materialize_imported_seed(root, checkpoint, imported_binding_seeds)
    args, kwargs = _decode(checkpoint["args"], root), _decode(checkpoint["kwargs"], root)
    name = checkpoint["function"]
    try:
        if name == "admit_continuation":
            from supervisor import queue_snapshot
            from tests._budget_pause_exact_helpers import _install_queue

            _queue, _state, workers = _install_queue(root, monkeypatch)
            # The recorded process held the queue it had published (a Continue waits for the restore otherwise).
            monkeypatch.setattr(queue_snapshot, "_RESTORED", {str(_queue.QUEUE_SNAPSHOT_PATH.absolute())})
            result = continuation.admit_continuation(*args, **kwargs)
            expected = checkpoint["result"]
            stable = ("ok", "held", "error", "replay", "status", "successor_task_id")
            assert {key: result.get(key) for key in stable} == {key: expected.get(key) for key in stable}
            if checkpoint["continuation"] is not None and not expected.get("replay"):
                assert _billing(workers.PENDING[-1]["metadata"]["continuation"]) == _billing(
                    _decode(checkpoint["continuation"], root))
            return
        if name == "usage_projection":
            kwargs.setdefault("include_roots", True)  # the journal era's default
            result = ua.usage_projection(*args, **kwargs)
        elif name == "_billing_group":
            result = continuation._billing_group(*args, **kwargs)
        else:
            result = getattr(admission, name)(*args, **kwargs)
        result, expected = _on_the_known_spend_rule(name, _encode(result, root), checkpoint["result"])
        if name == "usage_projection" and _amended(checkpoint):
            expected = _without_the_retired_overlay(expected)
        minted = expected.get("billing_group_limit_revision") if isinstance(expected, dict) else None
        if isinstance(minted, str) and not any(minted in _DOC["blobs"][blob] for blob in checkpoint["files"].values()):
            # A revision the question itself minted (an initial or default pin): a fresh
            # timestamp each time, compared for presence only.
            assert isinstance(result.pop("billing_group_limit_revision"), str)
            expected = {key: value for key, value in expected.items() if key != "billing_group_limit_revision"}
        assert result == expected, checkpoint["test"]
    finally:
        usage_store.forget(root)


def _materialize(root, checkpoint):
    from ouroboros.store_import import ensure_current_store

    for name, blob in checkpoint["files"].items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_DOC["blobs"][blob].replace("<ROOT>", str(root)), encoding="utf-8")
    ensure_current_store(root)  # the boot door imports the journal era's files; a read never does


@pytest.mark.serial
def test_continue_refuses_when_the_default_cap_cannot_be_pinned(tmp_path, monkeypatch):
    """An unpinned choice is no choice: the Continue of an open old root (its
    imported block disagrees on the cap) is refused, typed, while the store's
    write lock is held elsewhere, and the same nonce admits and pins once it is free."""
    from ouroboros.store import StoreBusy
    from ouroboros.task_results import load_task_result
    from supervisor import continuation_admission
    from supervisor.continuation_admission import admit_continuation
    from tests._budget_pause_exact_helpers import _install_queue
    from tests.test_owner_continue import NONCE

    [checkpoint] = [item for item in _CHECKPOINTS if item["function"] == "admit_continuation"
                    and item["test"] == "test_continue_pins_the_default_cap_on_the_predecessor_for_its_own_later_work"]
    root = tmp_path / "data"
    _materialize(root, checkpoint)
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(root / "settings.json"))
    monkeypatch.setenv("OUROBOROS_PER_TASK_COST_USD", "20")
    _queue, _state, workers = _install_queue(root, monkeypatch)
    patch = continuation_admission.patch_document

    def locked_out(*_args, **_kwargs):
        raise StoreBusy("store write lock held elsewhere")

    monkeypatch.setattr(continuation_admission, "patch_document", locked_out)
    assert admit_continuation("P", action_nonce=NONCE) == {"ok": False, "error": "billing_authority_unavailable"}
    assert not workers.PENDING and not load_task_result(root, "P").get("billing_group")
    monkeypatch.setattr(continuation_admission, "patch_document", patch)
    accepted = admit_continuation("P", action_nonce=NONCE)
    assert accepted["ok"] and load_task_result(root, "P")["billing_group"]["billing_group_limit_source"] == "legacy_default"


@pytest.mark.serial
def test_admission_transactions_read_the_store_before_taking_the_queue_lock(tmp_path, monkeypatch):
    """Continue and receipt-backed admission resolve a root's billing (a store
    read) off ``_queue_lock``."""
    import contextlib

    from ouroboros import usage_store
    from supervisor import queue
    from supervisor.continuation_admission import admit_continuation
    from supervisor.task_admission import enqueue_with_admission_receipt
    from tests._budget_pause_exact_helpers import _install_queue
    from tests.test_owner_continue import NONCE, _interrupted

    root = tmp_path / "data"
    (root / "state").mkdir(parents=True)
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(root / "settings.json"))
    monkeypatch.setenv("TOTAL_BUDGET", "1000")
    _queue, _state, workers = _install_queue(root, monkeypatch)
    _interrupted(root, "pred-1")
    owned = []
    original = usage_store.hold

    @contextlib.contextmanager
    def observed(*args, **kwargs):
        owned.append(queue._queue_lock._is_owned())
        with original(*args, **kwargs) as txn:
            yield txn

    monkeypatch.setattr(usage_store, "hold", observed)
    assert admit_continuation("pred-1", action_nonce=NONCE)["ok"]
    enqueue_with_admission_receipt({"id": "ordinary-root", "type": "task", "text": "probe", "root_task_id": "ordinary-root"},
                                   receipt_required=False)
    assert owned and not any(owned), owned  # the store was read, never while this thread held the queue lock


def test_check_budget_runs_in_the_server_and_is_skipped_in_a_worker(monkeypatch, tmp_path):
    """Both directions: a worker never reads the money store at construction; the server still checks."""
    from ouroboros import agent_startup_checks as checks
    from ouroboros.utils import WORKER_PROCESS_ENV

    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "state.json").write_text(json.dumps({"mode": "idle"}), encoding="utf-8")
    env = SimpleNamespace(budget_drive_root=tmp_path, drive_path=lambda name: tmp_path / name)
    reads = []
    monkeypatch.setattr("ouroboros.settings_setup_contract.resolve_total_budget_usd", lambda: 50.0)
    monkeypatch.setattr("ouroboros.usage_accounting.usage_projection",
                        lambda root, **kw: reads.append(root) or {"accounted_usd": 1.0, "remaining_known_usd": 49.0})
    monkeypatch.setenv(WORKER_PROCESS_ENV, "1")
    assert checks.check_budget(env) == ({"status": "skipped", "reason": "worker_process"}, 0)
    assert reads == []
    monkeypatch.delenv(WORKER_PROCESS_ENV)
    result, issues = checks.check_budget(env)
    assert (result["status"], issues) == ("ok", 0)
    assert reads == [tmp_path]
