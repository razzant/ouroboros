"""The authority switch (``ouroboros/store_switch.py``): intent and receipt, the needs read
from the target tree, preflight refusals that move nothing, the seam's order (export,
receipt, intent consumed, store retired, THEN the door's own switching owner), the
idempotent money step around every crash point, abandonment restoring both authorities,
and the chain new -> old -> new with failures around the consume and the switch."""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil
import sqlite3
import subprocess

import pytest

from ouroboros import store, store_switch, usage_store
from ouroboros.store_import import ensure_current_store
from ouroboros.task_results import task_result_path
from ouroboros.usage_ledger import LEDGER_REL
from ouroboros.utils import atomic_write_json
from tests._store_shared import release_store_handles
from tests._usage_store_testing import seed
from tests.test_store_import import drives_of, fixture_root, load, meta, result, rows, snapshot

REPO = pathlib.Path(__file__).resolve().parents[1]


def git(repo: pathlib.Path, *args: str) -> str:
    env = {**os.environ, "LC_ALL": "C", "LANG": "C", "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@x",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@x"}
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True, env=env).stdout.strip()


@pytest.fixture
def target_repo(tmp_path):
    """``old``: a tree without the store modules (a pin file included); ``new``: with them."""
    repo = tmp_path / "repo"
    (repo / "ouroboros").mkdir(parents=True)
    git(repo, "init", "-q")
    shutil.copy(REPO / "ouroboros" / "claudexor_runtime_pin.json", repo / "ouroboros" / "claudexor_runtime_pin.json")
    (repo / "ouroboros" / "task_results.py").write_text("# old\n", encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "old")
    old = git(repo, "rev-parse", "HEAD")
    for name in ("store.py", "usage_store.py", "dialogue_store.py"):
        (repo / "ouroboros" / name).write_text("# new\n", encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "new")
    return repo, old, git(repo, "rev-parse", "HEAD")


@pytest.fixture
def switch_root(tmp_path, monkeypatch):
    """A root with a published facts store (child drive included) and a money store
    imported from a two-attempt journal kept in place."""
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path / "data"))
    root = fixture_root(tmp_path)
    ensure_current_store(root)
    seed(root, 2)
    usage_store.migrate_from_journal(root)
    yield root
    usage_store.forget(root)


def intent(root: pathlib.Path, target_sha: str, door: str = "ui_rollback") -> dict:
    return store_switch.record_switch_intent(root, target_sha=target_sha, target_ref="v7.6.0", reason="test", door=door)


class Owner:
    """A switching owner that does what the real ones do to the checkout: HEAD lands on the target."""

    def __init__(self, repo, ok=True):
        self.repo, self.calls, self.ok = repo, [], ok

    def __call__(self, intent):
        self.calls.append(intent)
        if self.ok:
            git(self.repo, "checkout", "-q", "--detach", intent["target_sha"])
        return self.ok, f"switched to {intent['target_sha'][:8]}"


def state(root: pathlib.Path) -> dict:
    return {"intent": (root / store_switch.INTENT_REL).exists(), "facts_store": store.probe(root)[0],
            "money_store": (root / usage_store.STORE_REL).exists(), "journal": (root / LEDGER_REL).exists(),
            "retired_facts": sorted(p.name for p in (root / "state").glob("current.sqlite.exported-*")),
            "retired_money": sorted(p.name for p in (root / "state").glob("usage.sqlite.exported-*"))}


def receipt(root: pathlib.Path) -> dict:
    return json.loads((root / store_switch.RECEIPT_REL).read_text(encoding="utf-8"))


# ---- intent, needs, pin --------------------------------------------------------------

def test_the_intent_is_write_once_and_typed(tmp_path):
    recorded = intent(tmp_path, "a" * 40)
    assert store_switch.read_switch_intent(tmp_path) == recorded and recorded["door"] == "ui_rollback"
    with pytest.raises(RuntimeError):
        intent(tmp_path, "b" * 40)
    with pytest.raises(ValueError):
        store_switch.record_switch_intent(tmp_path / "other", target_sha="c", target_ref="r", reason="x", door="front")
    assert store_switch.resume_pending_switch(tmp_path / "other", repo_dir=tmp_path) == {"status": "nothing_pending"}


def test_needs_and_the_engine_pin_come_from_the_target_tree(target_repo):
    repo, old, new = target_repo
    assert store_switch.switch_needs(repo, old) == {"facts": True, "money": True, "handles": True, "dialogue": True}
    assert store_switch.switch_needs(repo, new) == {"facts": False, "money": False, "handles": True, "dialogue": False}
    pin = store_switch.next_engine_pin(repo, old)
    assert pin is not None and pin.version == json.loads(
        (REPO / "ouroboros" / "claudexor_runtime_pin.json").read_text(encoding="utf-8"))["release"]["version"]
    git(repo, "rm", "-q", "ouroboros/claudexor_runtime_pin.json")
    git(repo, "commit", "-q", "-m", "no pin")
    assert store_switch.next_engine_pin(repo, git(repo, "rev-parse", "HEAD")) is None


def test_every_door_calls_its_own_owner_bound_to_the_intent(target_repo, monkeypatch):
    """Each door's adapter calls the owner the door calls today, with the intent's target and
    reason; the update doors refuse, without calling it, when the managed update record
    they would act on names another commit."""
    import supervisor.git_ops as git_ops
    import supervisor.update_merge as update_merge
    import supervisor.update_recovery as update_recovery

    repo, old, new = target_repo
    calls: list = []
    records = {"update_intent": {"target_sha": old}, "tx": {"pre_update_sha": old}}

    def fake(name):
        def call(*args, **kwargs):
            calls.append((name, args, kwargs))
            git(repo, "checkout", "-q", "--detach", old)
            return True, name
        return call

    monkeypatch.setattr(update_recovery, "rollback_to_version", fake("rollback_to_version"))
    monkeypatch.setattr(update_merge, "rollback_managed_update", fake("rollback_managed_update"))
    monkeypatch.setattr(git_ops, "checkout_and_reset", fake("checkout_and_reset"))
    monkeypatch.setattr(update_merge, "read_update_tx_strict", lambda: ("valid", records["tx"]))
    monkeypatch.setattr(git_ops, "_read_update_intent", lambda: records["update_intent"])
    expected = {
        "ui_rollback": ("rollback_to_version", (old,), {"reason": "because"}),
        "vcs_rollback": ("rollback_to_version", (old,), {"reason": "because"}),
        "boot_smoke_rollback": ("rollback_managed_update", ("because",), {"reopen_writer_admission": False}),
        "update_replace": ("checkout_and_reset", (git_ops.BRANCH_DEV,),
                           {"reason": "ui_update_apply", "unsynced_policy": "rescue_and_reset"}),
    }
    for door in store_switch.DOORS:
        git(repo, "checkout", "-q", "--detach", new)
        calls.clear()
        recorded = {"target_sha": old, "target_ref": "v7.6.0", "reason": "because", "door": door}
        assert store_switch.switching_owner(door)(recorded) == (True, expected[door][0])
        assert calls == [expected[door]] and git(repo, "rev-parse", "HEAD") == old
    # the update doors are bound to the record they act on: another target refuses, nothing is called
    records["update_intent"], records["tx"] = {"target_sha": new}, {"pre_update_sha": new}
    for door in ("update_replace", "boot_smoke_rollback"):
        git(repo, "checkout", "-q", "--detach", new)
        calls.clear()
        ok, message = store_switch.switching_owner(door)({"target_sha": old, "reason": "because", "door": door})
        assert ok is False and new[:12] in message and old[:12] in message and calls == []
        assert git(repo, "rev-parse", "HEAD") == new
    with pytest.raises(ValueError):
        store_switch.switching_owner("window")


def test_the_seam_reports_an_owner_that_returns_without_moving_head(switch_root, target_repo):
    root, (repo, old, new) = switch_root, target_repo
    intent(root, old)
    report = store_switch.run_switch_seam(root, repo_dir=repo, owner=lambda recorded: (True, "said so"))
    assert report["status"] == "switch_failed" and new[:12] in report["message"] and old[:12] in report["message"]
    assert receipt(root)["switched"] == {**receipt(root)["switched"], "ok": False, "head": new}
    assert git(repo, "rev-parse", "HEAD") == new
    # the failure is reported where it happened (receipt, progress); the current code boots on the complete export
    assert json.loads((root / store_switch.PROGRESS_REL).read_text(encoding="utf-8"))["phase"] == "switch_failed"
    assert store_switch.resume_pending_switch(root, repo_dir=repo, owner=Owner(repo)) == {"status": "nothing_pending"}
    assert ensure_current_store(root)["status"] == "completed" and store.probe(root)[0] == "present"


# ---- the seam -----------------------------------------------------------------------------

def test_the_seam_exports_consumes_the_intent_retires_the_store_and_only_then_switches(switch_root, target_repo):
    root, (repo, old, _new) = switch_root, target_repo
    expected = snapshot(root)
    intent(root, old)
    seen = {}

    def owner(recorded):
        seen.update(state(root), receipt=receipt(root))
        git(repo, "checkout", "-q", "--detach", recorded["target_sha"])
        return True, "rolled back"

    report = store_switch.run_switch_seam(root, repo_dir=repo, owner=owner)
    assert report["status"] == "switched" and report["exit"] == 42
    # at the owner call: intent consumed, export completed, both stores retired, journal written
    assert seen["intent"] is False and seen["facts_store"] == "absent" and seen["money_store"] is False
    assert seen["journal"] is True and seen["receipt"]["export"]["status"] == "completed"
    assert seen["receipt"]["steps"]["money"]["status"] == "completed"
    assert seen["receipt"]["steps"]["handles"] == {"status": "no_packs", "handles": 0, "bytes": 0, "packs": 0}
    assert seen["receipt"]["steps"]["dialogue"] == {}, "the export step ran; this fixture root holds no dialogue store"
    assert len(seen["retired_facts"]) == 1 and len(seen["retired_money"]) == 1
    assert snapshot(root) == expected  # the files the old body reads are the ones the store imported
    assert receipt(root)["switched"]["ok"] is True
    phases = [row["phase"] for row in rows(root / "logs" / "supervisor.jsonl", "storage_switch")]
    assert phases == ["preflight", "exporting", "money", "handles", "dialogue", "switching", "switched"]
    assert json.loads((root / store_switch.PROGRESS_REL).read_text(encoding="utf-8"))["phase"] == "switched"
    assert store_switch.resume_pending_switch(root, repo_dir=repo) == {"status": "nothing_pending"}
    # the child drive's store and its own child's were retired with the canonical one
    for drive in drives_of(root):
        assert store.probe(drive)[0] == "absent" and list((drive / "state").glob("current.sqlite.exported-*"))
    assert sorted(receipt(root)["retired"]) == sorted(str(d) for d in (root, *drives_of(root)))


def test_a_target_with_the_store_switches_without_an_export(switch_root, target_repo):
    root, (repo, _old, new) = switch_root, target_repo
    intent(root, new)
    owner = Owner(repo)
    report = store_switch.run_switch_seam(root, repo_dir=repo, owner=owner)
    assert report["status"] == "switched" and len(owner.calls) == 1
    assert state(root) == {"intent": False, "facts_store": "present", "money_store": True, "journal": True,
                           "retired_facts": [], "retired_money": []}
    assert meta(root, "phase") == store.PUBLISHED
    assert receipt(root)["steps"].get("handles") == {"status": "no_packs", "handles": 0, "bytes": 0, "packs": 0}


def test_an_unconfirmed_stop_refuses_before_anything_moves(switch_root, target_repo):
    root, (repo, old, _new) = switch_root, target_repo
    intent(root, old)
    before = state(root)
    owner = Owner(repo)
    report = store_switch.run_switch_seam(root, repo_dir=repo, owner=owner, blockers=["worker:4242"])
    assert report["status"] == "refused" and report["reason"] == "quiescence_unconfirmed" and not owner.calls
    assert state(root) == {**before, "intent": False}
    assert meta(root, "phase") == store.PUBLISHED
    assert list((root / "state").glob("switch_intent.refused-*.json"))
    refused = [row for row in rows(root / "logs" / "supervisor.jsonl", "storage_switch") if row["phase"] == "refused"]
    assert refused[0]["reason"] == "quiescence_unconfirmed" and refused[0]["blockers"] == ["worker:4242"]
    # the quiet direction: no blockers, the same intent re-recorded, the switch runs
    intent(root, old)
    assert store_switch.run_switch_seam(root, repo_dir=repo, owner=owner)["status"] == "switched"


def test_preflight_refusals_move_nothing_and_repeat_honestly(switch_root, target_repo, monkeypatch):
    from collections import namedtuple

    root, (repo, old, _new) = switch_root, target_repo
    before = state(root)
    # the space the export needs: 0 handle bytes (before the packing stage) + the money store file
    # (the journal's bound) + the LARGEST single result file (one atomic temp file at a time), no corpus multiple
    roots, largest = [root, *drives_of(root)], 0
    for drive in roots:
        with store.open_store(drive, write=False) as txn:
            largest = max([largest, *(len(store.document_text(txn.load(t)).encode("utf-8")) for t in txn.task_ids())])
    required = largest + (root / usage_store.STORE_REL).stat().st_size
    usage = namedtuple("usage", "total used free")
    needs, blank = {"facts": True, "money": True, "handles": True}, {"steps": {}}
    with monkeypatch.context() as patched:
        patched.setattr(shutil, "disk_usage", lambda path: usage(required, 0, required))
        assert store_switch._preflight(root, needs, blank, roots) == ""  # exactly the bytes needed are enough
        patched.setattr(shutil, "disk_usage", lambda path: usage(required, 0, required - 1))
        assert str(required) in store_switch._preflight(root, needs, blank, roots)
        intent(root, old)
        report = store_switch.run_switch_seam(root, repo_dir=repo, owner=Owner(repo))
    assert report["status"] == "refused" and "free space" in report["reason"]
    # the money exporter's own refusal: a journal that changed after the import
    with (root / LEDGER_REL).open("ab") as handle:
        handle.write(b'{"appended": "by an older body"}\n')
    for _ in range(2):  # a repeated refusal leaves both authorities exactly as they were
        intent(root, old)
        report = store_switch.run_switch_seam(root, repo_dir=repo, owner=Owner(repo))
        assert report["status"] == "refused" and "changed after the import" in report["reason"]
        assert state(root) == {**before, "intent": False} and meta(root, "phase") == store.PUBLISHED


# ---- the money step around every crash point --------------------------------------------

def test_a_crash_before_the_journal_is_written_resumes_cleanly(switch_root, target_repo, monkeypatch):
    from ouroboros import usage_journal

    root, (repo, old, _new) = switch_root, target_repo
    intent(root, old)
    real = usage_journal.write_journal
    monkeypatch.setattr(usage_journal, "write_journal", lambda *a, **k: (_ for _ in ()).throw(OSError("disk went away")))
    report = store_switch.run_switch_seam(root, repo_dir=repo, owner=Owner(repo))
    assert report["status"] == "failed" and report["attempt"] == 1
    assert state(root)["money_store"] is True and (root / store_switch.INTENT_REL).exists()
    assert meta(root, "phase") == store.EXPORTING and receipt(root)["steps"]["money"]["status"] == "started"
    monkeypatch.setattr(usage_journal, "write_journal", real)
    owner = Owner(repo)
    report = store_switch.resume_pending_switch(root, repo_dir=repo, owner=owner)
    assert report["status"] == "switched" and len(owner.calls) == 1
    assert state(root)["money_store"] is False and state(root)["journal"] is True


def test_a_crash_after_the_journal_before_the_store_rename_recognises_its_own_journal(switch_root, target_repo, monkeypatch):
    root, (repo, old, _new) = switch_root, target_repo
    intent(root, old)
    real_replace = os.replace

    def refuse_store_rename(src, dst, *args, **kwargs):
        if "usage.sqlite.exported-" in str(dst):
            raise OSError("the rename never happened")
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", refuse_store_rename)
    assert store_switch.run_switch_seam(root, repo_dir=repo, owner=Owner(repo))["status"] == "failed"
    assert state(root)["money_store"] is True and state(root)["journal"] is True  # our own journal, store still here
    monkeypatch.setattr(os, "replace", real_replace)
    report = store_switch.resume_pending_switch(root, repo_dir=repo, owner=Owner(repo))
    assert report["status"] == "switched"
    assert state(root)["money_store"] is False and len(state(root)["retired_money"]) == 1
    assert receipt(root)["steps"]["money"]["journal"]["size"] == (root / LEDGER_REL).stat().st_size


def test_a_crash_after_the_store_rename_before_the_receipt_completes_from_the_facts(switch_root, target_repo):
    root, (repo, old, _new) = switch_root, target_repo
    intent(root, old)
    assert store_switch.run_switch_seam(root, repo_dir=repo, owner=Owner(repo))["status"] == "switched"
    # the receipt the crash never wrote: money started, export open, intent still recorded
    kept = receipt(root)
    kept["steps"]["money"] = {"status": "started"}
    kept["export"] = {}
    kept.pop("switched")
    atomic_write_json(root / store_switch.RECEIPT_REL, kept)
    atomic_write_json(root / store_switch.INTENT_REL, kept["intent"])
    owner = Owner(repo)
    report = store_switch.run_switch_seam(root, repo_dir=repo, owner=owner)
    assert report["status"] == "switched" and len(owner.calls) == 1
    money = receipt(root)["steps"]["money"]
    assert money["status"] == "completed" and money["store_retired_as"].startswith("usage.sqlite.exported-")


def test_abandonment_after_a_partial_export_restores_both_authorities(switch_root, target_repo):
    root, (repo, old, _new) = switch_root, target_repo
    before, journal_before = state(root), (root / LEDGER_REL).read_bytes()  # captured BEFORE the export
    (root / "state" / "state.json").write_text(json.dumps({"usage_ledger_high_water_seq": [0, 40]}), encoding="utf-8")
    intent(root, old)

    def handles_fail(_root):
        raise OSError("the handle restore died")

    report = store_switch.run_switch_seam(root, repo_dir=repo, owner=Owner(repo), restore_handles=handles_fail)
    assert report["status"] == "failed"
    assert state(root)["money_store"] is False and meta(root, "phase") == store.EXPORTING  # money exported, facts open
    assert (root / LEDGER_REL).read_bytes() != journal_before  # the export's own journal is in place
    assert json.loads((root / "state" / "state.json").read_text(encoding="utf-8"))["usage_ledger_high_water_seq"] != [0, 40]
    report = store_switch.resume_pending_switch(root, repo_dir=repo, owner=Owner(repo), blockers=["packer:7"])
    assert report["status"] == "refused"
    assert report["restored"]["money"] == {"store": "restored", "journal": "restored", "state_marker": "restored",
                                           "mark": "cleared"}
    assert state(root) == {**before, "intent": False}
    assert meta(root, "phase") == store.PUBLISHED
    assert (root / LEDGER_REL).read_bytes() == journal_before  # the import-kept journal, byte for byte
    assert not list((root / "state").glob("usage_attempts.jsonl.pre-export-*"))
    assert json.loads((root / "state" / "state.json").read_text(encoding="utf-8"))["usage_ledger_high_water_seq"] == [0, 40]
    with usage_store.read(root) as txn:  # the republished money store serves again, and takes writes
        assert len(txn.attempts()) == 2 and txn.meta("export") is None
    with usage_store.hold(root) as txn:
        assert txn.writable
    assert store.update_task(root, "done1", lambda current: {**current, "after": "abandonment"})["after"] == "abandonment"
    assert receipt(root)["abandonment"]["status"] == "completed"
    intent(root, old)  # a new switch works after the abandonment
    assert store_switch.run_switch_seam(root, repo_dir=repo, owner=Owner(repo))["status"] == "switched"


def test_abandonment_after_a_partial_facts_export_keeps_the_db_only_rows(switch_root, target_repo, monkeypatch):
    """The store's own partial writes are recognised by the manifest at abandonment; the next
    boot sees nothing out of band and keeps every row, a DB-only owner-held set row included."""
    from ouroboros import store_import

    root, (repo, old, _new) = switch_root, target_repo
    # the DB era: two rows changed in the store (their files are stale) and one DB-only custody row
    for task_id in ("done1", "root1"):
        store.update_task(root, task_id, lambda current: {**current, "db_era": True})
    with store.open_store(root, write=True) as txn:
        txn.replace_set("custody_open", {**txn.set_rows("custody_open"), "run:r9": {"task_id": "root1", "custody": "{}"}})
        assert len(txn.set_rows("custody_open")) == 3
    intent(root, old)
    real, calls = store_import.write_text_atomic, []

    def dies_on_the_second_file(path, text, **kwargs):
        calls.append(path)
        if len(calls) == 2:
            raise OSError("disk went away")
        return real(path, text, **kwargs)

    monkeypatch.setattr(store_import, "write_text_atomic", dies_on_the_second_file)
    assert store_switch.run_switch_seam(root, repo_dir=repo, owner=Owner(repo))["status"] == "failed"
    monkeypatch.setattr(store_import, "write_text_atomic", real)
    assert [p.name for p in calls] == ["done1.json", "root1.json"] and meta(root, "phase") == store.EXPORTING
    report = store_switch.resume_pending_switch(root, repo_dir=repo, owner=Owner(repo), blockers=["worker:1"])
    assert report["status"] == "refused" and report["restored"]["facts"] == [str(d) for d in (root, *drives_of(root))]
    boot = ensure_current_store(root)
    with store.open_store(root, write=False) as txn:
        assert len(txn.set_rows("custody_open")) == 3 and "run:r9" in txn.set_rows("custody_open")  # 3 stays 3
        assert txn.load("done1")["db_era"] is True and txn.load("root1")["db_era"] is True
    assert boot["status"] == "unchanged"  # the own write of done1.json is not an old body's change
    assert "db_era" not in task_result_path(root, "root1").read_text(encoding="utf-8")  # the stale file stays stale
    # adopted: done1 (written) and child1 (already identical); root1's stale file matches nothing
    assert meta(root, "export") == {**meta(root, "export"), "status": "abandoned", "adopted": 2, "planned": 3}
    assert store_switch.resume_pending_switch(root, repo_dir=repo, owner=Owner(repo)) == {"status": "nothing_pending"}


def retired_attempt_ids(root: pathlib.Path) -> list:
    """The attempt ids inside the retired money store, read with plain sqlite."""
    retired = sorted((root / "state").glob("usage.sqlite.exported-*"))[-1]
    with sqlite3.connect(retired) as conn:
        return sorted(r[0] for r in conn.execute("SELECT attempt_id FROM attempts WHERE kind != 'usage_baseline_group'"))


def test_a_money_write_during_the_export_window_is_refused_typed(switch_root, target_repo, monkeypatch):
    """From the export mark's commit until the store is retired no money write lands: a writer
    arriving while the journal is being written gets ``UsageExportInProgress`` inside its own
    transaction; the journal and the retired store hold the same attempts."""
    from ouroboros import usage_journal
    from ouroboros.usage_ledger import UsageExportInProgress

    root, (repo, old, _new) = switch_root, target_repo
    intent(root, old)
    real, refused = usage_journal.write_journal, []

    def late_writer_then_write(target, rows):
        with pytest.raises(UsageExportInProgress):
            with usage_store.hold(root) as txn:  # the late writer: refused before it can touch a row
                txn.conn.execute("DELETE FROM attempts")
        with usage_store.read(root) as txn:  # a reader is not refused
            refused.append(len(txn.attempts()))
        return real(target, rows)

    monkeypatch.setattr(usage_journal, "write_journal", late_writer_then_write)
    assert store_switch.run_switch_seam(root, repo_dir=repo, owner=Owner(repo))["status"] == "switched"
    assert refused == [2]  # the late writer's DELETE never committed
    journal_ids = sorted({str(row.get("attempt_id") or "") for row in usage_journal.read_journal_records(root)[0]
                          if row.get("kind") not in {"usage_baseline", "usage_baseline_group"}} - {""})
    assert journal_ids == retired_attempt_ids(root) and len(journal_ids) == 2


def test_a_crash_right_after_the_money_store_rename_is_completed_on_resume(switch_root, target_repo, monkeypatch):
    root, (repo, old, _new) = switch_root, target_repo
    intent(root, old)
    real_replace = os.replace

    def die_after_the_rename(src, dst, *args, **kwargs):
        real_replace(src, dst, *args, **kwargs)
        if "usage.sqlite.exported-" in str(dst):
            raise KeyboardInterrupt("died right after renaming the money store aside")

    monkeypatch.setattr(os, "replace", die_after_the_rename)
    with pytest.raises(KeyboardInterrupt):
        store_switch.run_switch_seam(root, repo_dir=repo, owner=Owner(repo))
    monkeypatch.setattr(os, "replace", real_replace)
    assert state(root)["money_store"] is False and state(root)["journal"] is True
    assert receipt(root)["steps"]["money"]["status"] == "started"  # the receipt never saw the completion
    owner = Owner(repo)
    report = store_switch.resume_pending_switch(root, repo_dir=repo, owner=owner)
    assert report["status"] == "switched" and len(owner.calls) == 1
    money = receipt(root)["steps"]["money"]
    assert money["status"] == "completed" and len(state(root)["retired_money"]) == 1
    assert money["journal"]["sha256"] == hashlib.sha256((root / LEDGER_REL).read_bytes()).hexdigest()
    assert money["pre_export_as"] and (root / "state" / money["pre_export_as"]).exists()


def partial_journal_write(root):
    """A ``write_journal`` that dies in the middle: the mark is committed, the kept journal is
    aside, a partial own journal is on disk, the store is NOT renamed."""
    from ouroboros import usage_journal

    def write(target, rows):
        (root / LEDGER_REL).write_bytes(usage_journal.journal_bytes(rows)[:40])
        raise OSError("disk died in the middle of the journal")

    return write


def test_a_crash_before_the_money_store_rename_is_completed_on_resume(switch_root, target_repo, monkeypatch):
    """A partial own journal (neither the import's digest nor the mark's) is recognised by the
    mark's set-aside record, rewritten whole, and the retry completes both authorities."""
    from ouroboros import usage_journal

    root, (repo, old, _new) = switch_root, target_repo
    journal_before = (root / LEDGER_REL).read_bytes()
    intent(root, old)
    with monkeypatch.context() as patched:
        patched.setattr(usage_journal, "write_journal", partial_journal_write(root))
        assert store_switch.run_switch_seam(root, repo_dir=repo, owner=Owner(repo))["status"] == "failed"
    pre_export = list((root / "state").glob("usage_attempts.jsonl.pre-export-*"))
    assert state(root)["money_store"] is True and len(pre_export) == 1 and pre_export[0].read_bytes() == journal_before
    assert len((root / LEDGER_REL).read_bytes()) == 40  # the partial own journal
    with usage_store.read(root) as txn:
        mark = txn.meta("export")
    assert mark["status"] == "writing" and mark["pre_export_as"] == pre_export[0].name
    owner = Owner(repo)
    report = store_switch.resume_pending_switch(root, repo_dir=repo, owner=owner)
    assert report["status"] == "switched" and len(owner.calls) == 1
    money = receipt(root)["steps"]["money"]
    assert money["status"] == "completed" and state(root)["money_store"] is False and len(state(root)["retired_money"]) == 1
    assert money["journal"]["sha256"] == hashlib.sha256((root / LEDGER_REL).read_bytes()).hexdigest() == mark["journal"]["sha256"]
    assert pre_export[0].read_bytes() == journal_before  # the import-kept journal was never overwritten by the partial
    journal_ids = sorted({str(row.get("attempt_id") or "") for row in usage_journal.read_journal_records(root)[0]
                          if row.get("kind") not in {"usage_baseline", "usage_baseline_group"}} - {""})
    assert journal_ids == retired_attempt_ids(root) and len(journal_ids) == 2


def test_a_crash_before_the_money_store_rename_is_abandoned_cleanly(switch_root, target_repo, monkeypatch):
    from ouroboros import usage_journal

    root, (repo, old, _new) = switch_root, target_repo
    before, journal_before = state(root), (root / LEDGER_REL).read_bytes()
    intent(root, old)
    with monkeypatch.context() as patched:
        patched.setattr(usage_journal, "write_journal", partial_journal_write(root))
        assert store_switch.run_switch_seam(root, repo_dir=repo, owner=Owner(repo))["status"] == "failed"
    report = store_switch.resume_pending_switch(root, repo_dir=repo, owner=Owner(repo), blockers=["packer:7"])
    assert report["status"] == "refused" and report["restored"]["money"] == {"journal": "restored", "mark": "cleared"}
    assert state(root) == {**before, "intent": False}
    assert (root / LEDGER_REL).read_bytes() == journal_before and not list((root / "state").glob("usage_attempts.jsonl.pre-export-*"))
    with usage_store.hold(root) as txn:
        assert txn.writable and txn.meta("export") is None and len(txn.attempts()) == 2
    intent(root, old)
    assert store_switch.run_switch_seam(root, repo_dir=repo, owner=Owner(repo))["status"] == "switched"


def test_a_crash_inside_the_abandonment_after_the_journal_step_resumes_it(switch_root, target_repo, monkeypatch):
    root, (repo, old, _new) = switch_root, target_repo
    journal_before = (root / LEDGER_REL).read_bytes()
    intent(root, old)

    def handles_fail(_root):
        raise OSError("the handle restore died")

    assert store_switch.run_switch_seam(root, repo_dir=repo, owner=Owner(repo), restore_handles=handles_fail)["status"] == "failed"
    real_replace = os.replace

    def die_after_the_journal_comes_back(src, dst, *args, **kwargs):
        real_replace(src, dst, *args, **kwargs)
        if pathlib.Path(dst) == root / LEDGER_REL:
            raise KeyboardInterrupt("died inside the abandonment, after the journal step")

    monkeypatch.setattr(os, "replace", die_after_the_journal_comes_back)
    with pytest.raises(KeyboardInterrupt):
        store_switch.resume_pending_switch(root, repo_dir=repo, owner=Owner(repo), blockers=["packer:7"])
    monkeypatch.setattr(os, "replace", real_replace)
    # the store is back and the journal is back, but the mark still freezes the store and the intent is in place
    assert state(root)["money_store"] is True and (root / LEDGER_REL).read_bytes() == journal_before
    assert receipt(root)["abandonment"]["status"] == "started" and (root / store_switch.INTENT_REL).exists()
    with pytest.raises(usage_store.UsageExportInProgress):
        with usage_store.hold(root):
            pass
    owner = Owner(repo)
    report = store_switch.resume_pending_switch(root, repo_dir=repo, owner=owner)  # the next boot finishes it
    assert report["status"] == "refused" and report["reason"] == "abandonment_resumed" and not owner.calls
    assert report["restored"]["money"] == {"mark": "cleared"} and receipt(root)["abandonment"]["status"] == "completed"
    assert not (root / store_switch.INTENT_REL).exists() and meta(root, "phase") == store.PUBLISHED
    with usage_store.hold(root) as txn:
        assert txn.meta("export") is None and len(txn.attempts()) == 2
    intent(root, old)  # a new switch works after the abandonment
    assert store_switch.run_switch_seam(root, repo_dir=repo, owner=Owner(repo))["status"] == "switched"


def test_a_second_failure_of_the_same_intent_is_a_refusal_never_a_loop(switch_root, target_repo):
    root, (repo, old, _new) = switch_root, target_repo
    intent(root, old)

    def handles_fail(_root):
        raise OSError("still failing")

    for attempt in (1, 2):
        report = store_switch.resume_pending_switch(root, repo_dir=repo, owner=Owner(repo), restore_handles=handles_fail)
        assert report == {"status": "failed", "error": "OSError: still failing", "attempt": attempt}
    report = store_switch.resume_pending_switch(root, repo_dir=repo, owner=Owner(repo), restore_handles=handles_fail)
    assert report["status"] == "refused" and report["reason"] == "second_failure"
    assert not (root / store_switch.INTENT_REL).exists() and meta(root, "phase") == store.PUBLISHED
    assert state(root)["money_store"] is True and store.probe(root)[0] == "present"


# ---- retirement -----------------------------------------------------------------------------

def test_the_stamp_is_cleared_before_the_store_is_renamed_aside(switch_root, target_repo, monkeypatch):
    """A death right after the rename must leave «absent, unstamped» (the next import rebuilds
    from the complete export), never «absent beside its completed stamp» (LOST)."""
    root, (repo, old, _new) = switch_root, target_repo
    intent(root, old)
    real_replace = os.replace

    def die_after_the_rename(src, dst, *args, **kwargs):
        real_replace(src, dst, *args, **kwargs)
        if pathlib.Path(dst).parent == root / "state" and "current.sqlite.exported-" in str(dst):
            raise KeyboardInterrupt("died right after renaming the canonical store aside")

    monkeypatch.setattr(os, "replace", die_after_the_rename)
    with pytest.raises(KeyboardInterrupt):
        store_switch.run_switch_seam(root, repo_dir=repo, owner=Owner(repo))
    monkeypatch.setattr(os, "replace", real_replace)
    assert store.probe(root)[0] == "absent" and not store.generation_stamped(root)
    with pytest.raises(store.StoreRetired):  # typed for a straggler until the next import
        store.update_task(root, "done1", lambda current: current)
    assert store_switch.resume_pending_switch(root, repo_dir=repo, owner=Owner(repo))["status"] == "switch_not_performed"
    report = ensure_current_store(root)
    assert report["status"] == "completed" and report["results"] == 3
    assert report["drives"] == {str(drive): "completed" for drive in drives_of(root)}


def test_a_lost_drive_refuses_the_switch_instead_of_exporting_past_it(switch_root, target_repo):
    root, (repo, old, _new) = switch_root, target_repo
    drive = root / "state" / "headless_tasks" / "root1" / "data"
    release_store_handles(store.store_path(root))  # the fixture's own reads keep it open (Windows)
    store.store_path(root).unlink()  # the canonical store lost beside its completed stamp
    intent(root, old)
    owner = Owner(repo)
    report = store_switch.run_switch_seam(root, repo_dir=repo, owner=owner)
    assert report["status"] == "refused" and "lost" in report["reason"] and not owner.calls
    assert store.probe(drive)[0] == "present" and meta(drive, "phase") == store.PUBLISHED  # never exported nor retired
    assert state(root)["money_store"] is True and state(root)["journal"] is True
    assert not list((drive / "state").glob("current.sqlite.exported-*"))


# ---- the chain new -> old -> new ---------------------------------------------------------------

def test_a_death_after_the_consume_and_before_the_switch_keeps_the_new_code_and_reimports(switch_root, target_repo):
    root, (repo, old, _new) = switch_root, target_repo
    intent(root, old)

    def dies(_intent):
        raise KeyboardInterrupt("the process died between the consume and the switch")

    with pytest.raises(KeyboardInterrupt):
        store_switch.run_switch_seam(root, repo_dir=repo, owner=dies)
    assert not (root / store_switch.INTENT_REL).exists() and store.probe(root)[0] == "absent"
    # the next boot: no intent, a retired store, a complete export -> re-import, nothing lost, the owner told
    report = store_switch.resume_pending_switch(root, repo_dir=repo, owner=Owner(repo))
    assert report["status"] == "switch_not_performed"
    assert ensure_current_store(root)["status"] == "completed"
    assert sorted(load(root, tid)["task_id"] for tid in ("root1", "child1", "done1")) == ["child1", "done1", "root1"]
    usage_store.migrate_from_journal(root)  # money re-imports its own export
    with usage_store.read(root) as txn:
        assert len(txn.attempts()) == 2
    assert store_switch.resume_pending_switch(root, repo_dir=repo, owner=Owner(repo)) == {"status": "nothing_pending"}
    # pressing the door again runs the whole chain
    usage_store.forget(root)
    intent(root, old)
    assert store_switch.run_switch_seam(root, repo_dir=repo, owner=Owner(repo))["status"] == "switched"


def test_the_old_body_runs_and_the_new_body_rebuilds_from_the_complete_export(switch_root, target_repo):
    root, (repo, old, _new) = switch_root, target_repo
    intent(root, old)
    assert store_switch.run_switch_seam(root, repo_dir=repo, owner=Owner(repo))["status"] == "switched"
    # the old body: a new task, a deleted one, a changed one, money appended to the journal
    atomic_write_json(task_result_path(root, "by-old"), result("by-old"))
    task_result_path(root, "done1").unlink()
    atomic_write_json(task_result_path(root, "child1"), result("child1", "completed", old_body=True))
    assert not store.generation_stamped(root)
    # the new body boots: a full rebuild from the files the old body left (no reconciliation needed)
    report = ensure_current_store(root)
    assert report["status"] == "completed" and report["results"] == 3
    assert load(root, "by-old")["status"] == "completed" and load(root, "done1") is None
    assert load(root, "child1")["old_body"] is True
    assert store_switch.resume_pending_switch(root, repo_dir=repo, owner=Owner(repo)) == {"status": "nothing_pending"}


def test_a_death_inside_the_facts_export_resumes_from_the_intent(switch_root, target_repo, monkeypatch):
    from ouroboros import store_import

    root, (repo, old, _new) = switch_root, target_repo
    intent(root, old)
    real = store_import.export_facts
    calls = []

    def dies_once(drive):
        calls.append(drive)
        if len(calls) == 1:
            raise OSError("died while writing the files")
        return real(drive)

    monkeypatch.setattr(store_import, "export_facts", dies_once)
    assert store_switch.run_switch_seam(root, repo_dir=repo, owner=Owner(repo))["status"] == "failed"
    assert (root / store_switch.INTENT_REL).exists() and meta(root, "phase") == store.EXPORTING
    assert ensure_current_store(root, with_children=False)["status"] == "switch_pending"  # the importer leaves it
    owner = Owner(repo)
    assert store_switch.resume_pending_switch(root, repo_dir=repo, owner=owner)["status"] == "switched"
    assert len(owner.calls) == 1 and receipt(root)["attempts"] == 2
