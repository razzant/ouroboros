"""The P8 lane (facts + money) against the real ``v7.6.0`` body: the authority switch exports
a small anonymised root; the v7.6.0 checkout (``git archive``, run by this interpreter with its
own ``HOME`` and ``OUROBOROS_DATA_DIR``) reads the tasks, its own obligations (the cancel
intents and the custody chain it folds) and the money; it adds and deletes a result and
appends money rows; the new release reconciles per row and folds the journal. A custody event
whose process died between its COMMIT and the chain append reaches the chain at the next boot
and is read by the old body; a pending invocation the new body recorded is read by the old
body, which binds it to a run through its own chain append (the legacy custody event the new
body folds in its legacy reconcile job); a handoff row the new body wrote after the export conflicts with
the old body's rewrite of the same result and the file wins with the conflict recorded. The
chain new -> old -> new is run twice, with the two failures around the consume and the switch
in between (the new body rebuilds from the exported files)."""
from __future__ import annotations

import io
import json
import os
import pathlib
import subprocess
import sys
import tarfile
import textwrap

import pytest

from ouroboros import delegate_custody as c
from ouroboros import delegate_custody_current as current
from ouroboros import store, store_switch, task_store, usage_store
from ouroboros.store_import import ensure_current_store, run_legacy_reconcile
from ouroboros.utils import atomic_write_json
from tests._usage_store_testing import attempt_rows_in_start_order, seed
from tests.test_store_import import fixture_root, load
from tests.test_store_switch import Owner, git
from tests.test_store_switch import target_repo as _switch_target_repo

REPO = pathlib.Path(__file__).resolve().parents[1]
OLD_TAG = "v7.6.0"
target_repo = _switch_target_repo  # ``old``: a tree without the store modules; ``new``: with them

OLD_BODY = textwrap.dedent('''
    import json, pathlib, sys
    root = pathlib.Path(sys.argv[1])
    from ouroboros import cancel_intents
    from ouroboros.delegate_custody import STARTED, emit, invocation_record, replay
    from ouroboros.task_results import list_task_results, load_task_result, task_result_path, write_task_result
    from ouroboros.usage_accounting import AttemptRequest, release_attempt, reserve_attempt, usage_projection
    out = {"ids": sorted(r.get("task_id") for r in list_task_results(root)),
           "done1": (load_task_result(root, "done1") or {}).get("status"),
           "cancel": sorted(cancel_intents.active_intents(root)),
           "runs": sorted(replay(root)),
           "pending": (invocation_record(root, "inv-pending") or {}).get("state"),
           "handoffs": (load_task_result(root, "root1") or {}).get("launch_handoffs"),
           "accounted_before": usage_projection(drive_root=root)["accounted_usd"]}
    if sys.argv[2] == "write":
        write_task_result(root, "by-old", "completed", description="written by v7.6.0", chat_id=1, task_type="task")
        assert emit(root, STARTED, {"run_id": "run-old", "task_id": "root1", "root_task_id": "root1",
                                    "invocation_id": "inv-pending"})  # the legacy append: the pending invocation starts
        task_result_path(root, "mismatch").unlink()
        write_task_result(root, "root1", "completed", old_body=True)
        res = reserve_attempt(AttemptRequest(model="stub", provider="local", reservation_usd=0.01, drive_root=root,
                                             task_id="by-old", root_task_id="by-old"))
        release_attempt(res, "not_dispatched")
        out["ids_after"] = sorted(r.get("task_id") for r in list_task_results(root))
        out["accounted_after"] = usage_projection(drive_root=root)["accounted_usd"]
    print("P8 " + json.dumps(out, default=str))
''')


@pytest.fixture(scope="module")
def v760(tmp_path_factory):
    """The v7.6.0 tree, materialised from the tag into a scratch directory (never a checkout)."""
    try:
        archive = subprocess.run(["git", "archive", OLD_TAG], cwd=REPO, check=True, capture_output=True, timeout=120).stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        pytest.skip(f"{OLD_TAG} is not resolvable in this clone: {exc}")
    tree = tmp_path_factory.mktemp("v760")
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(tree)
    assert (tree / "ouroboros" / "task_results.py").exists() and not (tree / "ouroboros" / "store.py").exists()
    return tree


def old_body(v760: pathlib.Path, root: pathlib.Path, mode: str, home: pathlib.Path) -> dict:
    home.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "HOME": str(home), "OUROBOROS_DATA_DIR": str(root), "PYTHONPATH": str(v760)}
    proc = subprocess.run([sys.executable, "-c", OLD_BODY, str(root), mode], cwd=v760, env=env,
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-4000:]
    line = [row for row in proc.stdout.splitlines() if row.startswith("P8 ")][-1]
    return json.loads(line[3:])


@pytest.fixture
def lane_root(tmp_path, monkeypatch):
    """The anonymised fixture root: three results, child drives, set files, two settled money
    attempts; a pending invocation and a launch handoff of ``root1``; one custody event
    committed but never appended (the death between COMMIT and the append)."""
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path / "data"))
    root = fixture_root(tmp_path)
    ensure_current_store(root)
    seed(root, 2)
    usage_store.migrate_from_journal(root)
    assert c.emit(root, c.START_REQUESTED, {"invocation_id": "inv-pending", "task_id": "root1", "root_task_id": "root1",
                                            "route": "codex", "surface": "delegate", "request": {"prompt": "work"}})
    task_store.put_handoffs(root, "root1", "launch_handoffs", {"claim-1": HANDOFF})
    with monkeypatch.context() as patched:
        patched.setattr(current, "append_jsonl", lambda *a, **kw: False)
        assert c.emit(root, c.STARTED, {"run_id": "run-dead", "task_id": "root1", "root_task_id": "root1"})
    assert current.outbox_depth(root) == 1 and "run-dead" not in _chain_text(root)
    yield root
    usage_store.forget(root)


HANDOFF = {"state": "claimed", "claimed_at": "2026-01-01T00:00:00Z", "local_owner": "new-body"}


def _chain_text(root: pathlib.Path) -> str:
    path = c.event_log_path(root)
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _switch_to_old(root, repo, old, *, owner=None):
    store_switch.record_switch_intent(root, target_sha=old, target_ref=OLD_TAG, reason="test", door="ui_rollback")
    return store_switch.run_switch_seam(root, repo_dir=repo, owner=owner or Owner(repo))


def test_v760_reads_the_export_and_the_new_release_reconciles_its_writes(lane_root, target_repo, v760, tmp_path):
    repo, old, new = target_repo
    root = lane_root
    # The boot door flushes the dead process's committed custody to the chain before anything else.
    boot = store_switch.start(root, repo_dir=repo, request_restart=lambda: pytest.fail("no switch pending"))
    assert boot == {"status": "importing"} and store_switch.wait_ready(root, timeout=60)["status"] == "published"
    assert current.outbox_depth(root) == 0 and "run-dead" in _chain_text(root)
    outcome = _switch_to_old(root, repo, old)
    assert outcome["status"] == "switched" and outcome["needs"] == {"facts": True, "money": True, "handles": True, "dialogue": True}
    assert store.probe(root)[0] == "absent" and (root / "state" / "usage_attempts.jsonl").exists()

    read = old_body(v760, root, "write", tmp_path / "home")
    assert read["ids"] == ["child1", "done1", "other", "root1"] and read["done1"] == "failed"  # the refused file is evidence it still lists
    assert read["cancel"] == [] and read["runs"] == ["run-dead"]  # the old body's own fold of the exported chain
    assert read["pending"] == "pending" and read["handoffs"] == {"claim-1": HANDOFF}  # the rows, as the file carries them
    assert read["ids_after"] == ["by-old", "child1", "done1", "root1"]
    assert float(read["accounted_after"]) >= float(read["accounted_before"])

    report = ensure_current_store(root)  # the new release's boot: the retired store is rebuilt from the files
    assert report["status"] == "completed", report
    assert load(root, "by-old")["description"] == "written by v7.6.0"
    assert load(root, "root1")["old_body"] is True and load(root, "root1")["status"] == "completed"
    assert load(root, "root1")["launch_handoffs"] == {"claim-1": HANDOFF}  # the old body's rewrite kept the handoff
    assert load(root, "mismatch") is None
    started = c.invocation_record(root, "inv-pending")  # the old body's append bound the pending invocation
    assert started["state"] == "started" and started["run_id"] == "run-old" and c.pending_invocations(root) == []
    assert sorted(c.replay(root)) == ["run-dead", "run-old"]
    usage_store.forget(root)
    usage_store.migrate_from_journal(root)  # the journal the old body appended, folded once
    rows = [row["attempt_id"] for row in attempt_rows_in_start_order(root)]
    assert rows[:2] == ["seed-0", "seed-1"] and len(rows) == 3


def test_a_present_store_reconciles_the_old_body_writes_per_row(lane_root, v760, tmp_path):
    """P6': the export without the switch (the store stays published, the files are written
    beside it); the old body's add, change and delete come back as one row each, not a rebuild."""
    from ouroboros.store_import import export_facts

    root = lane_root
    current.flush_outbox(root)
    assert export_facts(root)["results"] == 3
    assert store.probe(root)[0] == "present"
    task_store.put_handoffs(root, "root1", "launch_handoffs", {"claim-2": HANDOFF})  # the new body, after the export
    read = old_body(v760, root, "write", tmp_path / "home")
    assert read["ids_after"] == ["by-old", "child1", "done1", "root1"] and read["pending"] == "pending"
    atomic_write_json(root / "state/state.json", {"current_sha": OLD_TAG})
    opened = ensure_current_store(root, body_sha="current-body")
    assert opened["status"] == "unchanged" and opened["legacy_reconcile"] == "scheduled"
    assert load(root, "by-old") is None  # the boot never applies history
    report = run_legacy_reconcile(root)["root_report"]
    assert report["status"] == "reconciled", report
    assert (report["new"], report["changed"], report["deleted"]) == (1, 1, 1), report
    assert (report["conflicts"], report["legacy_custody"]) == (1, 1), report
    assert load(root, "by-old")["description"] == "written by v7.6.0"
    assert load(root, "root1")["old_body"] is True and load(root, "mismatch") is None
    # The conflicting handoff: the old body's rewrite of root1 carried the exported claim only;
    # its file won, and the store's version with claim-2 is kept as the recorded conflict.
    assert load(root, "root1")["launch_handoffs"] == {"claim-1": HANDOFF}
    with store.open_store(root, write=False) as txn:
        conflict = txn.meta("conflicts")["root1"]
    assert "claim-2" in conflict["db_body"] and "claim-1" in conflict["db_body"]
    started = c.invocation_record(root, "inv-pending")  # the legacy append, folded once by the job
    assert started["state"] == "started" and started["run_id"] == "run-old" and c.pending_invocations(root) == []
    assert ensure_current_store(root)["status"] == "unchanged" and sorted(c.replay(root)) == ["run-dead", "run-old"]


def test_the_chain_new_old_new_twice_with_failures_around_the_consume_and_the_switch(lane_root, target_repo, v760, tmp_path):
    repo, old, new = target_repo
    root = lane_root
    assert store_switch.wait_ready(root, timeout=0.01) == {"status": "not_started"}
    current.flush_outbox(root)
    # Round one: the switch fails at the owner; the export stays, the store is retired, the new
    # body boots again and rebuilds from the files it wrote.
    failed = _switch_to_old(root, repo, old, owner=Owner(repo, ok=False))
    assert failed["status"] == "switch_failed" and git(repo, "rev-parse", "HEAD") == new
    assert store.probe(root)[0] == "absent" and store_switch.read_switch_intent(root) is None
    rebuilt = store_switch.start(root, repo_dir=repo, request_restart=lambda: pytest.fail("nothing to switch"))
    assert rebuilt == {"status": "importing"} and store_switch.wait_ready(root, timeout=60)["status"] == "published"
    assert sorted(load(root, tid)["task_id"] for tid in ("root1", "child1", "done1")) == ["child1", "done1", "root1"]
    usage_store.forget(root)
    usage_store.migrate_from_journal(root)
    assert len(attempt_rows_in_start_order(root)) == 2
    # Round two: the real switch, the old body writes, the new body reconciles.
    assert _switch_to_old(root, repo, old)["status"] == "switched"
    first = old_body(v760, root, "write", tmp_path / "home")
    assert first["ids_after"] == ["by-old", "child1", "done1", "root1"] and first["runs"] == ["run-dead"]
    git(repo, "checkout", "-q", "--detach", new)
    assert ensure_current_store(root)["status"] == "completed"  # rebuilt from the files the old body left
    usage_store.forget(root)
    usage_store.migrate_from_journal(root)
    assert len(attempt_rows_in_start_order(root)) == 3
    # The death after the consume, before the switch: the intent is gone, the receipt says
    # exported; the next boot reports the switch as not performed and rebuilds from the files.
    store_switch.record_switch_intent(root, target_sha=old, target_ref=OLD_TAG, reason="test", door="ui_rollback")

    def dies_before_switching(intent):
        raise RuntimeError("the process died after the consume")

    with pytest.raises(RuntimeError):
        store_switch.run_switch_seam(root, repo_dir=repo, owner=dies_before_switching)
    assert store_switch.read_switch_intent(root) is None and store.probe(root)[0] == "absent"
    resumed = store_switch.start(root, repo_dir=repo, request_restart=lambda: pytest.fail("no switch to resume"))
    assert resumed == {"status": "importing"} and store_switch.wait_ready(root, timeout=60)["status"] == "published"
    assert load(root, "by-old")["description"] == "written by v7.6.0"
    progress = json.loads((root / store_switch.PROGRESS_REL).read_text(encoding="utf-8"))
    assert progress["phase"] == "switch_not_performed"
    usage_store.migrate_from_journal(root)  # the boot's own money step (server lifespan), after the facts
    assert len(attempt_rows_in_start_order(root)) == 3
    # Round three: the old body reads what the new body kept, and the chain closes again.
    third = _switch_to_old(root, repo, old)
    assert third["status"] == "switched", third
    second = old_body(v760, root, "read", tmp_path / "home")
    assert second["ids"] == ["by-old", "child1", "done1", "root1"] and second["runs"] == ["run-dead", "run-old"]
    assert second["pending"] == "started"  # the run the old body bound in round two, read back from the chain
    git(repo, "checkout", "-q", "--detach", new)
    assert ensure_current_store(root)["status"] == "completed"
    usage_store.forget(root)
    usage_store.migrate_from_journal(root)
    assert len(attempt_rows_in_start_order(root)) == 3
