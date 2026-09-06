"""ABI-7a (v7.0, F14): the N−1 updater transition shim over the UNTOUCHED
boot-finalize family of ``supervisor/update_merge.py``.

Contract under test: (1) every tx write stamps the marker with the shared
``_schema_version`` key (ABI-2 idiom); (2) ``finalize_managed_update_on_boot``
stays the stable N−1→N entry point — a marker recorded by the PRE-7.0 code
(no stamp; the fixtures below write that exact byte form) is recognized and
driven through every recovery seam: pending-boot-smoke (crash before AND
after the smoke), assisted resolution (``_recover_assisted_on_boot``),
replace apply (``_recover_replace_on_boot``), rollback resume, and the
pre-apply stash crash; (3) a FUTURE-schema marker (recorded by a newer
release — this binary is the rollback target) fails closed at EVERY strict
consumer: never finalized, never rolled back, never overwritten, restart
deferred; (4) the rollback direction is additive — an N−1 reader sees a
stamped marker unchanged minus one key. The N−1/rollback fixtures are shared
property with the ABI-2 F12 suite
(tests/test_task_result_schema_quarantine.py).
"""

import json

import pytest

import supervisor.git_ops as git_ops
import supervisor.update_merge as update_merge
from ouroboros.contracts.schema_versions import SCHEMA_VERSION_KEY
from tests.test_update_merge_assisted import (
    _git,
    _init_repo,
    _materialized_conflict_tx,
    _point_at,
    _stub_worker_gates,
)


def _write_nminus1_tx(tx: dict) -> None:
    """Write the marker exactly as the pre-7.0 ``write_update_tx`` did: the
    same atomic JSON idiom, NO schema stamp."""
    from ouroboros.utils import atomic_write_json

    assert SCHEMA_VERSION_KEY not in tx
    atomic_write_json(update_merge._update_tx_marker_path(), tx, trailing_newline=True)


def _raw_marker() -> dict:
    return json.loads(update_merge._update_tx_marker_path().read_text(encoding="utf-8"))


# ------------------------------------------------------------------ the stamp


def test_write_update_tx_stamps_without_mutating_the_caller(tmp_path, monkeypatch):
    repo, head = _init_repo(tmp_path)
    _point_at(monkeypatch, tmp_path, repo, head)
    payload = {"phase": "pending_boot_smoke", "task_id": "x"}
    update_merge.write_update_tx(payload)
    assert SCHEMA_VERSION_KEY not in payload  # never mutate the caller's dict
    raw = _raw_marker()
    assert raw[SCHEMA_VERSION_KEY] == update_merge.UPDATE_TX_SCHEMA_VERSION
    status, tx = update_merge.read_update_tx_strict()
    assert status == "valid" and tx["task_id"] == "x"


def test_stamped_marker_is_one_additive_key_for_the_n_minus_1_reader(tmp_path, monkeypatch):
    """Rollback direction: an N−1 binary reads the stamped marker unchanged —
    the stamp is a single additive key its reader ignores."""
    repo, head = _init_repo(tmp_path)
    _point_at(monkeypatch, tmp_path, repo, head)
    payload = {"phase": "assisted_resolution", "task_id": "r", "pre_update_sha": "a" * 40}
    update_merge.write_update_tx(payload)
    raw = _raw_marker()
    assert {k: v for k, v in raw.items() if k != SCHEMA_VERSION_KEY} == payload


def test_unstamped_n_minus_1_marker_reads_valid(tmp_path, monkeypatch):
    repo, head = _init_repo(tmp_path)
    _point_at(monkeypatch, tmp_path, repo, head)
    _write_nminus1_tx({"phase": "assisted_resolution", "task_id": "legacy"})
    status, tx = update_merge.read_update_tx_strict()
    assert status == "valid" and tx["task_id"] == "legacy"


def test_invalid_stamps_are_corrupt_and_newer_stamps_are_future(tmp_path, monkeypatch):
    from ouroboros.utils import atomic_write_json

    repo, head = _init_repo(tmp_path)
    _point_at(monkeypatch, tmp_path, repo, head)
    marker = update_merge._update_tx_marker_path()
    for bad_stamp in ("1", True, 0, -3, None):
        atomic_write_json(marker, {SCHEMA_VERSION_KEY: bad_stamp, "phase": "x"})
        assert update_merge.read_update_tx_strict()[0] == "corrupt", bad_stamp
    atomic_write_json(
        marker,
        {SCHEMA_VERSION_KEY: update_merge.UPDATE_TX_SCHEMA_VERSION + 1, "phase": "x"},
    )
    status, tx = update_merge.read_update_tx_strict()
    assert status == "future" and tx["phase"] == "x"  # raw evidence returned


def test_explicit_null_stamp_is_corrupt_and_rollback_refuses_untouched(tmp_path, monkeypatch):
    """Adversarial fix-round 2, claim 1: an explicit ``_schema_version: null``
    is a DAMAGED stamp, not the legacy unstamped form (only key ABSENCE is) —
    it reads ``corrupt`` and the direct rollback entry point refuses typed
    BEFORE any marker write or destructive reset/checkout/clean."""
    from ouroboros.utils import atomic_write_json

    repo, head = _init_repo(tmp_path)
    _point_at(monkeypatch, tmp_path, repo, head)
    cur = _git(repo, "rev-parse", "HEAD").stdout.strip()
    marker = update_merge._update_tx_marker_path()
    atomic_write_json(
        marker,
        {
            SCHEMA_VERSION_KEY: None, "phase": "rolling_back",
            "pre_update_sha": cur, "pre_update_branch": head,
        },
        trailing_newline=True,
    )
    before = marker.read_bytes()
    assert update_merge.read_update_tx_strict() == ("corrupt", {})

    dirty = repo / "uncommitted.txt"
    dirty.write_text("owner work the rollback must not clean away\n")
    ok, msg = update_merge.rollback_managed_update("null_stamp_probe")
    assert ok is False and "pre_update_sha" in msg  # refuses on the empty corrupt tx
    assert marker.read_bytes() == before  # byte-identical, never re-phased
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == cur
    assert dirty.read_text() == "owner work the rollback must not clean away\n"


# ------------------------------------ N−1 fixtures through the boot finalizer


def test_n_minus_1_pending_boot_smoke_finalizes_on_healthy_boot(tmp_path, monkeypatch):
    """Crash AFTER the pre-restart smoke: the N−1 tx (smoke already proven)
    finalizes exactly like a stamped one, clearing intent and marker."""
    repo, head = _init_repo(tmp_path)
    _point_at(monkeypatch, tmp_path, repo, head)
    cur = _git(repo, "rev-parse", "HEAD").stdout.strip()
    git_ops._write_update_intent({"target_sha": cur})
    _write_nminus1_tx({
        "phase": "pending_boot_smoke", "merge_commit": cur,
        "pre_update_sha": cur, "pre_update_branch": head,
    })

    result = update_merge.finalize_managed_update_on_boot(supervisor_ready=True)

    assert result["finalized"] is True
    assert not git_ops._update_intent_marker_path().exists()
    assert update_merge.read_update_tx_strict()[0] == "absent"


def test_n_minus_1_pending_smoke_crash_before_smoke_replays_it(tmp_path, monkeypatch):
    """Crash BEFORE the pre-restart smoke proof: the N−1 tx still carries
    ``pending`` — boot replays the smoke, then finalizes."""
    repo, head = _init_repo(tmp_path)
    _point_at(monkeypatch, tmp_path, repo, head)
    cur = _git(repo, "rev-parse", "HEAD").stdout.strip()
    calls = []
    monkeypatch.setattr(
        update_merge, "update_restart_smoke",
        lambda: calls.append("smoke") or {"ok": True},
    )
    _write_nminus1_tx({
        "phase": "pending_boot_smoke", "pre_restart_smoke": "pending",
        "merge_commit": cur, "pre_update_sha": cur, "pre_update_branch": head,
    })

    result = update_merge.finalize_managed_update_on_boot(supervisor_ready=True)

    assert result["finalized"] is True and calls == ["smoke"]
    assert update_merge.read_update_tx_strict()[0] == "absent"


def test_n_minus_1_pending_smoke_failure_rolls_back(tmp_path, monkeypatch):
    repo, head = _init_repo(tmp_path)
    _point_at(monkeypatch, tmp_path, repo, head)
    _stub_worker_gates(monkeypatch)
    cur = _git(repo, "rev-parse", "HEAD").stdout.strip()
    monkeypatch.setattr(
        update_merge, "update_restart_smoke",
        lambda: {"ok": False, "stderr": "broken", "returncode": 1},
    )
    _write_nminus1_tx({
        "phase": "pending_boot_smoke", "pre_restart_smoke": "pending",
        "merge_commit": cur, "pre_update_sha": cur, "pre_update_branch": head,
    })

    result = update_merge.finalize_managed_update_on_boot(supervisor_ready=True)

    assert result["rolled_back"] is True
    assert update_merge.read_update_tx_strict()[0] == "absent"
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == cur


def test_n_minus_1_assisted_resolution_resumes_and_upgrades_the_marker(tmp_path, monkeypatch):
    """``_recover_assisted_on_boot`` drives an N−1 assisted tx: the in-flight
    resolution is resumed (never reset away), and the shim's first re-write
    upgrades the durable marker to the stamped form."""
    repo, head, plan, tx = _materialized_conflict_tx(tmp_path, monkeypatch)
    _stub_worker_gates(monkeypatch)
    # Re-record the marker exactly as the N−1 code left it: same fields, no stamp.
    _write_nminus1_tx(dict(tx))
    enqueued = []
    monkeypatch.setattr(
        update_merge, "enqueue_assisted_resolution_task",
        lambda tx_arg: enqueued.append(dict(tx_arg)) or "resolver",
    )

    result = update_merge.finalize_managed_update_on_boot(supervisor_ready=True)

    assert result == {"finalized": False, "resumed": True, "resolution_attempts": 1}
    assert enqueued and enqueued[0]["task_id"] == "resolver"
    assert (repo / "a.txt").read_text() == "the resolver's precious resolution\n"
    raw = _raw_marker()  # the resume re-write stamped the surviving marker
    assert raw[SCHEMA_VERSION_KEY] == update_merge.UPDATE_TX_SCHEMA_VERSION
    assert raw["phase"] == "assisted_resolution"


def test_n_minus_1_replace_apply_not_applied_is_abandoned(tmp_path, monkeypatch):
    """``_recover_replace_on_boot``: the N−1 replace tx whose checkout never
    happened (HEAD == pre) is abandoned without touching the tree."""
    repo, head = _init_repo(tmp_path)
    (repo / "b.txt").write_text("second\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "second")
    target = _git(repo, "rev-parse", "HEAD").stdout.strip()
    _git(repo, "reset", "-q", "--hard", "HEAD~1")
    pre = _git(repo, "rev-parse", "HEAD").stdout.strip()
    _point_at(monkeypatch, tmp_path, repo, head)
    _write_nminus1_tx({
        "phase": "applying_replace",
        "pre_update_sha": pre, "pre_update_branch": head, "target_sha": target,
    })

    result = update_merge.finalize_managed_update_on_boot(supervisor_ready=True)

    assert result.get("abandoned") is True and result["reason"] == "replace_not_applied"
    assert update_merge.read_update_tx_strict()[0] == "absent"
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == pre


def test_n_minus_1_rolling_back_resumes_the_rollback(tmp_path, monkeypatch):
    repo, head = _init_repo(tmp_path)
    _point_at(monkeypatch, tmp_path, repo, head)
    _stub_worker_gates(monkeypatch)
    cur = _git(repo, "rev-parse", "HEAD").stdout.strip()
    _write_nminus1_tx({
        "phase": "rolling_back", "rollback_reason": "n_minus_1_crash",
        "pre_update_sha": cur, "pre_update_branch": head,
    })

    result = update_merge.finalize_managed_update_on_boot(supervisor_ready=True)

    assert result["rolled_back"] is True
    assert update_merge.read_update_tx_strict()[0] == "absent"
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == cur


def test_n_minus_1_stash_crash_recovers_and_clears(tmp_path, monkeypatch):
    repo, head = _init_repo(tmp_path)
    _point_at(monkeypatch, tmp_path, repo, head)
    _write_nminus1_tx({"phase": "stashing_local_work", "attempt_id": "legacy-attempt"})

    result = update_merge.finalize_managed_update_on_boot(supervisor_ready=True)

    assert result["reason"] == "recovered pre-apply stash crash"
    assert update_merge.read_update_tx_strict()[0] == "absent"


# ---------------------------------------------------- future-schema refusals


def test_future_schema_marker_fails_closed_everywhere(tmp_path, monkeypatch):
    from ouroboros.utils import atomic_write_json
    from supervisor.update_candidate import UpdateTxCorrupt, update_tx_phase

    repo, head = _init_repo(tmp_path)
    _point_at(monkeypatch, tmp_path, repo, head)
    cur = _git(repo, "rev-parse", "HEAD").stdout.strip()
    marker = update_merge._update_tx_marker_path()
    future_tx = {
        SCHEMA_VERSION_KEY: update_merge.UPDATE_TX_SCHEMA_VERSION + 1,
        "phase": "pending_boot_smoke", "merge_commit": cur,
        "pre_update_sha": cur, "pre_update_branch": head,
    }
    atomic_write_json(marker, future_tx, trailing_newline=True)
    before = marker.read_bytes()

    # Boot: left for the owner — never finalized, never rolled back.
    result = update_merge.finalize_managed_update_on_boot(supervisor_ready=True)
    assert result["finalized"] is False and "newer version" in result["reason"]
    assert marker.read_bytes() == before
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == cur

    # Gate re-phasing refuses; the apply guard sees an ACTIVE tx; another
    # task cannot become the resolver.
    assert update_merge.mark_update_tx_gate_blocked("nope") is False
    assert update_merge.active_update_tx()
    blocked_tx, block_msg = update_merge.managed_assisted_tx_for("anyone")
    assert not blocked_tx and block_msg
    assert marker.read_bytes() == before

    # Phase merge-writers refuse to overwrite what they cannot interpret.
    with pytest.raises(UpdateTxCorrupt, match="newer Ouroboros"):
        update_tx_phase({"phase": "x"}, {"phase": "y"})
    assert marker.read_bytes() == before

    # Direct rollback entry point: refuses typed BEFORE any marker write or
    # destructive reset/checkout/clean — the future marker holds a
    # pre_update_sha, so a permissive read would have interpreted it and reset
    # the repository by a schema this code does not know.
    dirty = repo / "uncommitted.txt"
    dirty.write_text("owner work the rollback must not clean away\n")
    ok, msg = update_merge.rollback_managed_update("future_marker_probe")
    assert ok is False and "newer version" in msg
    assert marker.read_bytes() == before  # byte-identical, never re-phased
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == cur
    assert dirty.read_text() == "owner work the rollback must not clean away\n"

    # A requested restart is deferred rather than riding an unknown tx form.
    from ouroboros.server_restart import _safe_restart_serialized

    ok, msg = _safe_restart_serialized(
        lambda **_kwargs: pytest.fail("restart must not proceed over a future tx"),
        reason="test", unsynced_policy="ignore",
    )
    assert ok is False and "newer version" in msg
    assert marker.read_bytes() == before
