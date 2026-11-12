"""A failed stand attempt keeps what it knew, and the run-wide ledger keeps working around what it cannot know.

Three owners, one question each. ``IsolatedServer.start`` owns the server's stdio: a process that dies before its
first import must still say why, through its in-place re-exec. ``run_lane`` owns the attempt's record: an exception
keeps the scenario's observations without deriving a status from them, and no finalization step costs the next one
or passes a lost proof. ``RunBudget`` owns the money: an estimate is spent and never final, an unknown spend is never
zero, a live lane in that state is still working, and an ended one always leaves the flight, wakes its waiters and
stays charged. Real processes, real
SQLite and the store's real writers; the lane's own server is a double only where its process is not the subject.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import pathlib
import shutil
import sqlite3
import stat
import subprocess
import sys
import threading
import types

import pytest

from devtools.benchmarks.common.manifests import repo_provenance
from devtools.e2e_live import adoption, run_live_lanes, scenarios, traces

KEY = "sk-or-v1-e2e-live-attempt-fault-key-never-printed-0123456789"
SEED_FILES = {"VERSION": "7.0.0-test\n"}
STDIO = "server_stdio.log"


def _seed(root: pathlib.Path, files: dict) -> pathlib.Path:
    seed = root / "source"
    seed.mkdir(parents=True)
    for rel, text in files.items():
        (seed / rel).write_text(text, encoding="utf-8")
    for cmd in (["init", "-q"], ["add", "-A"], ["-c", "user.name=t", "-c", "user.email=t@e.invalid", "commit", "-q", "-m", "seed"]):
        subprocess.run(["git", *cmd], cwd=str(seed), check=True)
    return seed


class _NoopServer:
    base_url, attestation = "http://127.0.0.1:1", {}
    __init__ = start = stop = lambda self, *a, **k: None


def _attempt(tmp_path, monkeypatch, acceptance=lambda ctx: ctx.check("scenario_ok", True), *, sid="SK1", server=_NoopServer,
             files=None, budget=None, flags=(), procfs=False) -> tuple[dict, pathlib.Path, run_live_lanes.RunBudget]:
    """One real ``run_attempt`` with the default money reader unless a ledger is given; ``server=None`` is the real class."""
    monkeypatch.setattr(run_live_lanes.tempfile, "gettempdir", lambda: "/tmp/short")
    monkeypatch.setattr(run_live_lanes, "PROCFS_AVAILABLE", procfs)
    monkeypatch.setattr(run_live_lanes, "resolve_ui_client", lambda base_url: (None, "ui_unavailable:test"))
    if server is not None:
        monkeypatch.setattr(run_live_lanes, "IsolatedServer", server)
    monkeypatch.setitem(run_live_lanes.SCENARIOS, sid, dataclasses.replace(scenarios.SCENARIOS[sid], acceptance=acceptance))
    seed = _seed(tmp_path, files or SEED_FILES)
    out = tmp_path / "out"
    args = run_live_lanes.parse_args(["--scenarios", sid, "--out", str(out), "--watch-interval", "600", *flags])
    budget = budget or run_live_lanes.RunBudget(100.0, 8.0, self_mod=bool(args.self_mod))
    row = run_live_lanes.run_attempt((sid, 1), args, out, run_live_lanes.effective_settings(args, KEY),
                                     run_live_lanes.Stagger(0.0), {}, seed, budget, dispatch_index=0, key=KEY,
                                     seed_sha=repo_provenance(seed)["head"])
    return row, out / "lanes" / f"{sid}_a1", budget


# --------------------------------------------------------------------------- #
# The server's own stdio
# --------------------------------------------------------------------------- #

# Windows has no process-image replacement (``server_control.restart_current_process``): its C runtime starts the
# successor and exits the caller, so the process the stand watches would end before its successor does and the
# successor's ``rc=7`` could never be seen. There the first generation stays and ends as its successor does, the one
# process identity an in-place re-exec gives.
SERVER_REEXECS_THEN_DIES = f"""\
import os, subprocess, sys
if len(sys.argv) == 1:
    print('first generation stdout {KEY}', flush=True)
    successor = [sys.executable, 'server.py', 'second']
    if os.name == 'nt':
        raise SystemExit(subprocess.call(successor))
    os.execv(sys.executable, successor)
sys.stderr.write('ModuleNotFoundError: second generation died before its first import\\n')
raise SystemExit(7)
"""


def test_a_server_dying_before_its_imports_leaves_its_stdio_through_the_reexec_and_an_unknown_spend(tmp_path, monkeypatch):
    """The real ``IsolatedServer`` and a real process: DEVNULL left ``rc=7`` as the whole story and read the spend as $0.
    Both generations' output is kept (0600, the key still in it), the bundle holds it redacted, and a process having
    started without a usage record makes the attempt's spend unknown, its reservation still charged."""
    row, lane, budget = _attempt(tmp_path, monkeypatch, server=None, files={**SEED_FILES, "server.py": SERVER_REEXECS_THEN_DIES})
    assert row["status"] == "infra_error" and "rc=7" in row["refusal"]["message"], row

    raw = lane / STDIO
    text = raw.read_text(encoding="utf-8")
    assert "first generation stdout" in text and "second generation died before its first import" in text and KEY in text
    if os.name == "posix":
        assert stat.S_IMODE(raw.stat().st_mode) == 0o600
    bundled = (lane / traces.TRACES_DIR / STDIO).read_text(encoding="utf-8")
    assert "second generation died before its first import" in bundled and KEY not in bundled and "<redacted:" in bundled
    assert row["traces"]["published"] is True and KEY not in json.dumps(row)

    stored = json.loads((lane / "result.json").read_text(encoding="utf-8"))
    assert stored["budget"]["spent_usd"] is None and stored["budget"]["unknown_cost_rows"] is None
    assert "LaneSpendAbsent" in stored["budget"]["spend_unknown"], "absent where a process ran is unknown, never $0"
    snap = budget.snapshot()
    assert (snap["lanes_in_flight"], snap["reserved_usd"], snap["spent_usd"], snap["unknown_liability_usd"]) == (0, 0.0, 0.0, 16.0)   # SK1: two roots
    assert "SK1_a1" in snap["spend_unknown"]


def test_the_lane_settings_digest_names_the_bytes_that_reach_the_disk(tmp_path, monkeypatch):
    """``write_settings`` returns the digest a strict server checks against the file (``expected_settings_sha256``).
    Windows opens a descriptor in text mode unless ``O_BINARY`` is set, and ``os.write`` then turns each LF of the
    indented JSON into CRLF: the digest named bytes that were never on disk and the server refused its own settings
    (``isolated settings snapshot changed``). POSIX translates nothing, so a sentinel stands for the flag and a
    descriptor opened without it translates, as a Windows one does."""
    from devtools.benchmarks.common.server_runner import IsolatedServer, _settings_json_bytes

    sentinel, native, real_open, real_write = 1 << 40, getattr(os, "O_BINARY", 0), os.open, os.write
    text_mode: set = set()

    def opened(path, flags, *args, **kwargs):
        fd = real_open(path, (flags & ~sentinel) | (native if flags & sentinel else 0), *args, **kwargs)
        (text_mode.discard if flags & sentinel else text_mode.add)(fd)
        return fd

    def written(fd, data):
        if native or fd not in text_mode:  # Windows translates by itself, and only a descriptor in text mode
            return real_write(fd, data)
        real_write(fd, bytes(data).replace(b"\n", b"\r\n"))
        return len(data)

    settings = {"OUROBOROS_MODEL": "m", "TOTAL_BUDGET": 3.0, "note": "two\nlines"}
    path = tmp_path / "lane" / "settings.json"
    monkeypatch.setattr(os, "O_BINARY", sentinel, raising=False)
    monkeypatch.setattr(os, "open", opened)
    monkeypatch.setattr(os, "write", written)
    digest = run_live_lanes.write_settings(path, settings)
    monkeypatch.undo()

    assert path.read_bytes() == _settings_json_bytes(settings) and b"\r" not in path.read_bytes()
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    strict = IsolatedServer(tmp_path, tmp_path, path, settings_authoritative_env=True, expected_settings_sha256=digest)
    assert strict._read_authoritative_settings() == settings, "the strict server accepts what the lane wrote"


def test_other_drivers_keep_the_silent_default(tmp_path):
    from devtools.benchmarks.common.server_runner import IsolatedServer

    assert IsolatedServer(tmp_path, tmp_path, tmp_path / "settings.json").stdio_path is None


def test_a_start_that_failed_still_reports_a_stop_it_could_not_prove(tmp_path, monkeypatch):
    """``server`` is assigned only after ``start`` returns: the failed start's own stop proof lives on that object."""
    unconfirmed = ["server 4242: exit unconfirmed after a forced stop"]

    class _DiesUnconfirmed(_NoopServer):
        proc = None
        stop_unconfirmed: list = []

        def start(self, **_kw):
            self.proc, self.stop_unconfirmed = types.SimpleNamespace(pid=4242), unconfirmed
            raise RuntimeError("isolated server not ready in 1s")

    row, lane, _budget = _attempt(tmp_path, monkeypatch, server=_DiesUnconfirmed, flags=("--prune-clones",))
    assert row["status"] == "infra_error" and row["server_stop_unconfirmed"] == unconfirmed
    assert row["checks"]["server_stop_confirmed"] is False and (lane / "clone" / "VERSION").is_file()


# --------------------------------------------------------------------------- #
# The attempt's record
# --------------------------------------------------------------------------- #

def test_an_exception_keeps_every_observation_and_derives_nothing_from_them(tmp_path, monkeypatch):
    """Every check so far is green and the attempt still reads ``infra_error``: partial checks never decide a status."""
    def acceptance(ctx):
        ctx.check("task_side_ok", True, task_fact="kept")
        ctx.screenshots.append("shot.png")
        ctx.facts["runtime_result"] = {"status": "completed", "reason_code": ""}
        raise RuntimeError("the scenario's own wait fell over")

    row, lane, _budget = _attempt(tmp_path, monkeypatch, acceptance)
    stored = json.loads((lane / "result.json").read_text(encoding="utf-8"))
    for seen in (row, stored):
        assert seen["status"] == "infra_error" and seen["reason_code"] == "infra_error:RuntimeError"
        assert seen["checks"]["task_side_ok"] is True and all(seen["checks"].values())
        assert seen["facts"]["task_fact"] == "kept" and seen["screenshots"] == ["shot.png"]
        assert seen["runtime_outcome"]["status"] == "completed" and "runtime_result" not in seen["facts"]
    index = json.loads((lane.parent.parent / "result_index.jsonl").read_text(encoding="utf-8").strip())
    assert index["status"] == "infra_error" and index["details"]["checks"]["task_side_ok"] is True


def test_a_failed_restart_keeps_the_absorb_it_observed(tmp_path, monkeypatch):
    """``--self-mod``'s own verdict stays its own: the unhealthy restart's observation survives the exception it causes."""
    seen = {"confirmed": False, "reason": "unhealthy", "healthy": False, "wait": {"reason": "timeout"}}
    monkeypatch.setattr(run_live_lanes, "confirm_absorb", lambda *a, **k: dict(seen))

    def acceptance(ctx):
        ctx.check("commit_landed", True)
        ctx.restart()

    row, _lane, _budget = _attempt(tmp_path, monkeypatch, acceptance, sid="SM1", flags=("--self-mod",))
    assert row["status"] == "infra_error" and "unhealthy" in row["error"] and row["checks"]["commit_landed"] is True
    assert row["self_mod_absorb"]["expected"] is True and row["self_mod_absorb"]["reason"] == "unhealthy"
    assert "self_mod_absorb_confirmed" not in row["checks"], "no verdict from a wait that ended in an exception"


def _raises(*_args, **_kwargs):
    raise OSError("this step exploded")


class _StopRaises(_NoopServer):
    stop = _raises


@pytest.mark.parametrize("step", ["server_stop", "orphan_scan", "close_ui"])
def test_each_finalization_step_survives_the_failure_of_another(tmp_path, monkeypatch, step):
    """One unguarded read in the old ``finally`` lost the row, the traces and the settlement together. Containing the
    steps must not turn a lost proof into a pass either, because the verdict counts ``pass`` rows: an attempt without
    its bundle or its record ends ``infra_error``, an unproved stop or orphan scan fails its check. An unknown spend is
    no product failure, and closing the browser is cleanup: both leave the status as the checks made it."""
    def acceptance(ctx):
        ctx.check("scenario_ok", True, kept="yes")
        (ctx.data_root / "logs").mkdir(parents=True, exist_ok=True)
        (ctx.data_root / "logs" / "events.jsonl").write_text('{"type": "ok"}\n', encoding="utf-8")
        if step == "spend":
            (ctx.data_root / "state").mkdir(parents=True, exist_ok=True)
            (ctx.data_root / "state" / "usage.sqlite").write_text("not a database\n", encoding="utf-8")

    if step == "traces":
        monkeypatch.setattr(traces, "publish_lane_traces", _raises)
    if step == "traces_withheld":   # the publisher's own refusal returns; it never raises
        monkeypatch.setattr(traces, "publish_lane_traces", lambda *a, **k: {"published": False, "reason": "secret_residue",
                                                                            "files_with_residue": ["logs/events.jsonl"]})
    if step == "record_row":   # the real append fails at its open, after result.json was written
        (tmp_path / "out" / "result_index.jsonl").mkdir(parents=True)
    if step == "orphan_scan":
        from tests.system_e2e import harness
        monkeypatch.setattr(harness, "wait_until", _raises)
    if step == "close_ui":
        monkeypatch.setattr(scenarios.LaneContext, "close_ui", _raises)
    row, lane, budget = _attempt(tmp_path, monkeypatch, acceptance, server=_StopRaises if step == "server_stop" else _NoopServer,
                                 procfs=step == "orphan_scan")

    assert row["facts"]["kept"] == "yes" and budget.snapshot()["lanes_in_flight"] == 0
    stored = json.loads((lane / "result.json").read_text(encoding="utf-8"))
    assert (stored["status"], stored["reason_code"]) == (row["status"], row["reason_code"]), "the record says what the verdict counts"
    if step == "spend":
        assert row["status"] == "pass" and "finalize_errors" not in row
        assert row["budget"]["spent_usd"] is None and "DatabaseError" in row["budget"]["spend_unknown"]
        assert [gap["path"] for gap in row["traces"]["gaps"]] == ["state/usage.sqlite"] and row["traces"]["published"] is True
        assert budget.snapshot()["unknown_liability_usd"] == 16.0
        return
    if step == "traces_withheld":
        assert "finalize_errors" not in row and row["traces"]["reason"] == "secret_residue"
        assert (row["status"], row["reason_code"]) == ("infra_error", "infra_error:traces_secret_residue")
        return
    assert [error["step"] for error in row["finalize_errors"]] == [step]
    if step == "traces":
        assert row["traces"] == {"published": False, "reason": "collect_error"}
    else:
        assert row["traces"]["published"] is True and (lane / traces.TRACES_DIR / "logs" / "events.jsonl").is_file()
    expected = {"server_stop": ("fail", "checks_failed"), "orphan_scan": ("fail", "checks_failed"), "close_ui": ("pass", ""),
                "traces": ("infra_error", "infra_error:traces_collect_error"), "record_row": ("infra_error", "infra_error:record_row")}
    assert (row["status"], row["reason_code"]) == expected[step], row
    if step == "server_stop":
        assert row["checks"]["server_stop_confirmed"] is False, "an unproved stop never passes"
    if step == "orphan_scan":
        assert row["checks"]["no_orphans_after_stop"] is False and row["no_orphans_after_stop"] is None
        assert row["orphan_scan"] == "failed", "a scan that raised is not a host without procfs"


@pytest.mark.parametrize("correction", ["written", "fails"])
def test_after_a_failed_index_append_the_record_is_rewritten_once_and_the_append_never_repeated(tmp_path, monkeypatch, correction):
    """``result.json`` is written before the index append, so a failed append left it saying ``pass`` while the attempt
    counted ``infra_error``. The index path is a directory here: the real append fails at its open. The record is
    written again, once, with the attempt's final row; the append is never retried, because whether a failed append
    left a row, whole or torn, is unknown. A rewrite that fails too is named, and the file is not claimed fixed."""
    (tmp_path / "out" / "result_index.jsonl").mkdir(parents=True)
    real_write, real_index, writes, appends = run_live_lanes._write_result, run_live_lanes._index_row, [], []

    def write(lane, row):
        writes.append(row["status"])
        if correction == "fails" and len(writes) > 1:
            raise OSError(28, "No space left on device")
        real_write(lane, row)

    monkeypatch.setattr(run_live_lanes, "_write_result", write)
    monkeypatch.setattr(run_live_lanes, "_index_row", lambda out, row: appends.append(row["status"]) or real_index(out, row))
    row, lane, _budget = _attempt(tmp_path, monkeypatch)
    stored = json.loads((lane / "result.json").read_text(encoding="utf-8"))
    assert (row["status"], row["reason_code"]) == ("infra_error", "infra_error:record_row")
    assert writes == ["pass", "infra_error"] and appends == ["pass"], (writes, appends)
    assert "IsADirectoryError" in row["finalize_errors"][0]["error"] or os.name == "nt", row["finalize_errors"]
    if correction == "written":
        assert (stored["status"], stored["reason_code"]) == (row["status"], row["reason_code"])
        assert [error["step"] for error in stored["finalize_errors"]] == ["record_row"]
    else:
        assert stored["status"] == "pass" and "finalize_errors" not in stored, "a file nothing could rewrite is not claimed fixed"
        assert [error["step"] for error in row["finalize_errors"]] == ["record_row", "record_row_correction"]
        assert "No space left" in row["finalize_errors"][-1]["error"]


# --------------------------------------------------------------------------- #
# The lane's store, read as the install the stand launched
# --------------------------------------------------------------------------- #

REPO = pathlib.Path(__file__).resolve().parents[1]
ROOT_TASK, CHILD_TASK, LANE_MARKER = "a5e1ec7ed0000001", "a5e1ec7ed0000002", "E2E_LIVE_LANE_STORE"
ADOPTED, BASE = "a" * 40, "b" * 40
# The lane server's side, in its own process on ITS data root (``IsolatedServer._env`` sets OUROBOROS_DATA_DIR): the
# boot mints that install's identity, and the task rows go through the store's writers. The stand only reads them.
LANE_PROCESS = """\
import json, sys
sys.path.insert(0, sys.argv[1])
from ouroboros import install_identity
from tests._store_shared import seed_result
root, root_task, child_task, marker, adopted, base = sys.argv[2:]
identity = install_identity.initialize_install()
seed_result(root, root_task, {"status": "completed", "reason_code": "final_message", "result": marker + " done",
                              "description": marker + " root", "body_adoptions": [{"adoption_id": "lane-adoption",
                              "outcome": "adopted", "commit_sha": adopted, "previous_sha": base, "branch": "ouroboros"}]})
seed_result(root, child_task, {"status": "completed", "parent_task_id": root_task, "root_task_id": root_task,
                               "delegation_role": "subagent"})
print(json.dumps(identity))
"""


def _lane_install(booted: dict):
    """The lane server as far as the store goes: its process boots on the lane's data root; HTTP says completed."""
    class _LaneInstall(_NoopServer):
        def __init__(self, _clone, data_root, *_args, **_kwargs) -> None:
            self.data_root = pathlib.Path(data_root)

        def start(self, **_kwargs):
            done = subprocess.run([sys.executable, "-c", LANE_PROCESS, str(REPO), str(self.data_root), ROOT_TASK, CHILD_TASK,
                                   LANE_MARKER, ADOPTED, BASE], cwd=str(REPO), capture_output=True, text=True, check=True,
                                  env={**os.environ, "OUROBOROS_DATA_DIR": str(self.data_root)})
            booted.update(json.loads(done.stdout))
            return self

        def wait_task(self, task_id, timeout=0):
            return {"status": "completed"}

    return _LaneInstall


def _read_the_lane(seen: dict):
    """The stand's own store readers after the terminal, each as the scenario that uses it reaches it."""
    def acceptance(ctx):
        seen["terminal"] = ctx.wait_task(ROOT_TASK)   # SM1_a1 refused here: harness.wait_durable_result
        seen["on_task"] = adoption.task_adoption_fact(adoption._task_result(ctx.oracle, ROOT_TASK), "lane-adoption", ADOPTED)
        seen["children"] = ctx.oracle.child_task_ids(ROOT_TASK)   # SW1's lineage
        seen["child"] = ctx.oracle.task_result(CHILD_TASK)
        seen["root"] = scenarios._find_root_task(ctx, LANE_MARKER)   # SW1's UI path
    return acceptance


def test_the_stand_reads_the_lane_store_as_the_install_it_launched_and_nowhere_else(tmp_path, monkeypatch):
    """SM1_a1 (2026-11-02, ``--stub --profile full``) recorded ``http_terminal_completed`` and then ``StoreForeign``: the
    lane's server had minted the lane root's own install identity, and the stand process is another install, which the
    copied-folder rule refuses even a read (``install_identity``). The stand launched that install, so every store read
    a scenario makes after the terminal answers from the real lane (the durable row, the adoption's task record, SW1's
    children and root). Outside the lane the same readers still refuse, and the lane keeps the identity its own process
    minted while the stand keeps its own."""
    from ouroboros import install_identity
    from tests.system_e2e import harness

    booted, seen = {}, {}
    own = install_identity.canonical_root()
    row, lane, _budget = _attempt(tmp_path, monkeypatch, _read_the_lane(seen), sid="SM1", server=_lane_install(booted))
    data = lane / "data"
    minted = json.loads((data / install_identity.IDENTITY_PATH).read_text(encoding="utf-8"))
    assert minted == booted and pathlib.Path(minted["root"]) == data.resolve(), "the lane's own process minted it"

    assert (row["status"], row.get("refusal"), row["error"]) == ("pass", None, ""), row
    assert row["checks"]["http_terminal_completed"] is True and row["checks"]["durable_terminal_completed"] is True
    assert seen["terminal"]["status"] == "completed" and seen["terminal"]["result"] == f"{LANE_MARKER} done"
    assert seen["on_task"] == {"adoption_id": "lane-adoption", "outcome": "adopted", "commit_sha": ADOPTED,
                               "previous_sha": BASE, "branch": "ouroboros"}
    assert seen["children"] == [CHILD_TASK] and seen["child"]["parent_task_id"] == ROOT_TASK and seen["root"] == ROOT_TASK
    assert install_identity.canonical_root() == own, "the stand is its own install again once the lane is done"

    # The default scope on the same lane: the very consumer SM1_a1 ended in refuses before reading anything.
    oracle = harness.ArtifactOracle(data)
    ctx = scenarios.LaneContext(server=types.SimpleNamespace(wait_task=lambda task_id, timeout=0: {"status": "completed"}),
                                clone=lane / "clone", data_root=data, oracle=oracle, harness=harness, ui_resolver=None,
                                ui_reason="", shots=tmp_path / "shots", log=lambda _msg: None, task_timeout=1,
                                restart=lambda: None)
    with pytest.raises(install_identity.StoreForeign) as refused:
        ctx.wait_task(ROOT_TASK)
    assert refused.value.reason_code == "foreign_install_root" and pathlib.Path(refused.value.install_root) == data.resolve()
    assert ctx.checks == {"http_terminal_completed": True}, "SM1_a1's row: the HTTP terminal, then the refusal"
    for read in (lambda: oracle.task_result(ROOT_TASK), lambda: oracle.child_task_ids(ROOT_TASK)):
        with pytest.raises(install_identity.StoreForeign):
            read()
    assert json.loads((data / install_identity.IDENTITY_PATH).read_text(encoding="utf-8")) == booted, "never re-minted"


# --------------------------------------------------------------------------- #
# The money ledger
# --------------------------------------------------------------------------- #

def _ask(budget, job, root, *, index: int):
    box: dict = {}
    thread = threading.Thread(target=lambda: box.__setitem__("r", budget.admit(job, 1, root, dispatch_index=index)), daemon=True)
    thread.start()
    thread.join(0.3)
    return thread, box


def _reader(spend: dict, broken: set):
    def read(root):
        if root.name in broken:
            raise sqlite3.OperationalError("database is locked")
        return spend.get(root.name, 0.0), 0
    return read


def test_a_failed_read_at_settlement_ends_the_lane_wakes_the_waiter_and_stays_charged(tmp_path):
    """The old settle popped the lane, raised before ``notify_all`` and lost the reservation: the waiter parked for ever
    behind a lane already gone. It is told now, and told why: known spend would have fitted, the unknown does not."""
    broken: set = set()
    budget = run_live_lanes.RunBudget(10.0, 8.0, reader=_reader({}, broken))
    assert budget.admit(("SM1", 1), 1, tmp_path / "a", dispatch_index=0)[0]
    waiter, box = _ask(budget, ("SW1", 1), tmp_path / "b", index=1)
    assert waiter.is_alive()
    broken.add("a")
    budget.settle(("SM1", 1))
    waiter.join(5.0)
    assert not waiter.is_alive(), "parked behind a lane that already ended"
    admitted, facts = box["r"]
    assert admitted is False and facts["reason"] == "accounting_unavailable"
    assert (facts["spent_usd"], facts["unknown_liability_usd"], facts["reserved_usd"]) == (0.0, 8.0, 0.0)
    snap = budget.snapshot()
    assert snap["lanes_in_flight"] == 0 and snap["lanes_settled"] == 1
    assert "OperationalError" in snap["spend_unknown"]["SM1_a1"]["reason"]
    assert [(r["attempt"], r["reason"]) for r in snap["refusals"]] == [("SW1_a1", "accounting_unavailable")]
    budget.settle(("SM1", 1))                     # idempotent: no second charge
    assert budget.snapshot()["unknown_liability_usd"] == 8.0
    fits = run_live_lanes.RunBudget(20.0, 8.0, reader=_reader({}, {"a"}))
    assert fits.admit(("SM1", 1), 1, tmp_path / "a", dispatch_index=0)[0]
    fits.settle(("SM1", 1))
    assert fits.admit(("SW1", 1), 1, tmp_path / "b", dispatch_index=1)[0], "an unknown charge only takes its own room"


@pytest.mark.parametrize("last_good,known,liability", [(3.0, 3.0, 5.0), (12.0, 12.0, 0.0)])
def test_an_unknown_end_charges_the_reservation_or_a_higher_last_good_reading(tmp_path, last_good, known, liability):
    spend, broken = {"a": last_good}, set()
    budget = run_live_lanes.RunBudget(100.0, 8.0, reader=_reader(spend, broken))
    assert budget.admit(("SM1", 1), 1, tmp_path / "a", dispatch_index=0)[0]
    assert budget.snapshot()["spent_usd"] == last_good   # a question read it while the lane was live
    broken.add("a")
    budget.settle(("SM1", 1))
    snap = budget.snapshot()
    assert (snap["spent_usd"], snap["unknown_liability_usd"]) == (known, liability) and "SM1_a1" in snap["spend_unknown"]


def test_a_live_lane_that_cannot_be_read_is_still_working_and_never_breaks_another_question(tmp_path):
    """The old ``_spent_locked`` raised one lane's read error into another attempt's admission. The live lane is not
    finished work either: its reservation stays counted, so nothing is admitted against a zero or refused for it."""
    spend, broken = {"a": 2.0}, set()
    budget = run_live_lanes.RunBudget(20.0, 8.0, reader=_reader(spend, broken))
    assert budget.admit(("SM1", 1), 1, tmp_path / "a", dispatch_index=0)[0]
    assert budget.snapshot()["spent_usd"] == 2.0
    broken.add("a")
    admitted, facts = budget.admit(("SW1", 1), 1, tmp_path / "b", dispatch_index=1)
    assert admitted and facts["spent_usd"] == 2.0 and facts["reserved_usd"] == 8.0 and facts["unknown_liability_usd"] == 0.0
    assert "read failed in flight" in budget.snapshot()["spend_unknown"]["SM1_a1"]["reason"]
    waiter, box = _ask(budget, ("SK1", 1), tmp_path / "c", index=2)
    assert waiter.is_alive() and not box, "2 + 16 in flight + 8 > 20: it waits on the lanes still working"
    assert budget.snapshot()["lanes_in_flight"] == 2
    broken.clear()
    assert "SM1_a1" not in budget.snapshot()["spend_unknown"], "a read that recovers takes the doubt back"
    budget.settle(("SM1", 1))
    budget.settle(("SW1", 1))
    waiter.join(5.0)
    assert not waiter.is_alive() and box["r"][0] is True


def test_no_usage_record_is_zero_only_where_no_server_process_ever_ran(tmp_path):
    budget = run_live_lanes.RunBudget(100.0, 8.0)
    for name in ("never", "started"):
        assert budget.admit((name, 1), 1, tmp_path / name, dispatch_index=0)[0]
    assert budget.snapshot()["spend_unknown"] == {}, "a lane still booting has simply not written its store"
    budget.note_spawned(("started", 1))
    for name in ("never", "started"):
        budget.settle((name, 1))
    snap = budget.snapshot()
    assert (snap["spent_usd"], snap["unknown_liability_usd"]) == (0.0, 8.0) and list(snap["spend_unknown"]) == ["started_a1"]
    assert run_live_lanes.lane_spend_fact(tmp_path / "never", spawned=False) == {"spent_usd": 0.0, "unknown_cost_rows": 0}
    assert run_live_lanes.lane_spend_fact(tmp_path / "started", spawned=True)["spent_usd"] is None


def _settled(data: pathlib.Path, *costs: tuple) -> None:
    """The store's real writers: one sent attempt per ``(cost, final)``, settled at it."""
    from ouroboros import usage_accounting as ua
    from ouroboros import usage_store

    (data / "state").mkdir(parents=True, exist_ok=True)
    for cost, final in costs:
        held = ua.reserve_attempt(ua.AttemptRequest(model="m", provider="test", drive_root=data, task_id="t", root_task_id="t",
                                                    reservation_usd=0.0, global_limit_usd=1000.0))
        ua.mark_dispatched(held)
        ua.settle_attempt(held, {}, cost_usd=cost, cost_final=final)
    usage_store.forget(data)


def test_an_estimate_binds_as_known_spend_and_is_never_final_through_the_real_store(tmp_path):
    """Read as the core counts money toward a limit (``_usage_money.exceeds_limit``, #1487: confirmed prices plus
    disclosed estimates). The old query kept only final rows, so every estimated, unpriced or never-settled attempt
    vanished; reading an estimate as unknown still dropped its amount: $1.50 final beside $99 estimated stayed charged
    max($8 reservation, $1.50), and a second $8 attempt fitted a $20 cap. The estimate binds; its row still counts as
    without a final price, so a smaller estimate keeps the reservation charged. Attempts never sent stay out."""
    from ouroboros import usage_accounting as ua
    from ouroboros import usage_store

    data = tmp_path / "data"
    _settled(data, (1.5, True), (None, True), (99.0, False))

    def attempt():
        return ua.reserve_attempt(ua.AttemptRequest(model="m", provider="test", drive_root=data, task_id="t", root_task_id="t",
                                                    reservation_usd=0.0, global_limit_usd=1000.0))

    ua.mark_dispatched(attempt())                   # sent, never settled: it may have been billed
    attempt()                                       # reserved, never sent
    ua.release_attempt(attempt())                   # released, never sent
    usage_store.forget(data)

    assert run_live_lanes.lane_spend(data) == (100.5, 3)
    assert run_live_lanes.lane_spend_fact(data, spawned=True) == {"spent_usd": 100.5, "unknown_cost_rows": 3}
    budget = run_live_lanes.RunBudget(20.0, 8.0)
    assert budget.admit(("SM1", 1), 1, data, dispatch_index=0)[0]
    budget.settle(("SM1", 1))
    snap = budget.snapshot()
    assert (snap["spent_usd"], snap["unknown_cost_rows"], snap["unknown_liability_usd"]) == (100.5, 3, 0.0)
    assert "3 attempts without a final price" in snap["spend_unknown"]["SM1_a1"]["reason"]
    admitted, facts = budget.admit(("SM1", 2), 1, tmp_path / "next", dispatch_index=1)
    assert admitted is False and facts["reason"] == "budget_cap", facts

    small = tmp_path / "small"
    _settled(small, (2.0, False))
    floor = run_live_lanes.RunBudget(100.0, 8.0)
    assert floor.admit(("SM1", 1), 1, small, dispatch_index=0)[0]
    floor.settle(("SM1", 1))
    snap = floor.snapshot()
    assert (snap["spent_usd"], snap["unknown_cost_rows"], snap["unknown_liability_usd"]) == (2.0, 1, 6.0), "max(reservation, known) stays charged"


def _journal(data: pathlib.Path, rows: list) -> pathlib.Path:
    """A journal-era lane's ledger as an older release appended it, one row per transition."""
    path = data / "state" / "usage_attempts.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Its writer is byte-exact (LF rows); text mode would write CRLF on Windows and the journal is compared byte for byte.
    path.write_bytes("".join(json.dumps(row) + "\n" for row in rows).encode("utf-8"))
    return path


def _sent(attempt: str, *ends: tuple) -> list:
    return [{"attempt_id": attempt, "state": "reserved"}, {"attempt_id": attempt, "state": "dispatched"},
            *({"attempt_id": attempt, "state": state, "cost_usd": cost, "cost_final": final} for state, cost, final in ends)]


def test_lane_spend_of_a_journal_era_seed_is_its_journal_never_zero(tmp_path):
    """``--seed`` may name a commit before the store, whose lane appends a journal row per transition. It reads as the
    store holds that history: each attempt's latest row (``usage_ledger._final_rows``; the import keeps the same and
    drops the compaction header). Summing every settled row charged an estimate AND the price that replaced it ($9 + $1
    for one attempt) and refused a supported seed under its cap; skipping ``dispatched``/``unresolved`` read attempts
    that may have been billed as $0. The seed's journal is only read. The twin is the same history through the core's
    real writers, a provider's exact price replacing the estimate included."""
    from ouroboros import usage_accounting as ua
    from ouroboros import usage_store

    store = tmp_path / "store"
    _settled(store, (0.5, True), (0.25, True), (None, True), (9.0, False))

    def attempt(**binding):
        return ua.reserve_attempt(ua.AttemptRequest(model="m", provider="test", drive_root=store, task_id="t", root_task_id="t",
                                                    reservation_usd=0.0, global_limit_usd=1000.0, provider_receipt_binding=binding or None))

    refined = attempt(generation="gen-e")
    ua.mark_dispatched(refined)
    ua.settle_attempt(refined, {}, cost_usd=9.0, cost_final=False)
    receipt = {"attempt_id": refined.attempt_id, "provider": "test", "evidence_ref": "e", "binding": {"generation": "gen-e"}, "cost_usd": 1.0}
    assert ua.apply_provider_price_receipt(store, refined.attempt_id, receipt)["status"] == "applied"
    ua.mark_dispatched(attempt())                   # sent, never settled
    unresolved = attempt()
    ua.mark_dispatched(unresolved)
    ua.mark_unresolved(unresolved, "provider_outcome_unknown")
    ua.release_attempt(attempt())                   # never sent
    attempt()
    usage_store.forget(store)

    journal = tmp_path / "journal"
    path = _journal(journal, [{"attempt_id": "baseline-1", "kind": "usage_baseline", "state": "settled"},
                              *_sent("a", ("settled", 0.5, True)), *_sent("b", ("settled", 0.25, True)), *_sent("c", ("settled", None, True)),
                              *_sent("d", ("settled", 9.0, False)), *_sent("e", ("settled", 9.0, False), ("settled", 1.0, True)),
                              *_sent("f"), *_sent("g", ("unresolved", None, False)), {"attempt_id": "h", "state": "reserved"},
                              {"attempt_id": "h", "state": "released"}, {"attempt_id": "i", "state": "reserved"}])
    before = (path.read_bytes(), path.stat().st_mtime_ns)

    assert run_live_lanes.lane_spend(store) == run_live_lanes.lane_spend(journal) == (10.75, 4)
    assert run_live_lanes.lane_spend_fact(store, spawned=True) == run_live_lanes.lane_spend_fact(journal, spawned=True) == {
        "spent_usd": 10.75, "unknown_cost_rows": 4}
    admissions = []
    for root in (store, journal):                   # the seed is admitted exactly as its store twin is
        budget = run_live_lanes.RunBudget(30.0, 8.0)
        assert budget.admit(("SW1", 1), 1, root, dispatch_index=0)[0]
        budget.settle(("SW1", 1))
        admissions.append(budget.admit(("SK1", 1), 2, tmp_path / "next", dispatch_index=1))
    assert admissions[0] == admissions[1] == (True, {"cap_usd": 30.0, "spent_usd": 10.75, "unknown_liability_usd": 0.0, "reserved_usd": 0.0,
                                                    "reservation_usd": 16.0, "unknown_cost_rows": 4, "waited_sec": 0.0})

    sent_only = tmp_path / "sent_only"              # nothing settled yet: possibly billed, never $0
    _journal(sent_only, [*_sent("f"), *_sent("g", ("unresolved", None, False))])
    assert run_live_lanes.lane_spend(sent_only) == (0.0, 2)
    budget = run_live_lanes.RunBudget(30.0, 8.0)
    assert budget.admit(("SW1", 1), 1, sent_only, dispatch_index=0)[0]
    budget.settle(("SW1", 1))
    assert budget.snapshot()["unknown_liability_usd"] == 8.0, "its reservation stays charged"
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before, "the seed's journal is read, never truncated"
    assert sorted(p.name for p in (journal / "state").iterdir()) == ["usage_attempts.jsonl"], "no quarantine, no store"


_ONE_SHOT = {"attempt_id": "b", "kind": "external_unmetered", "state": "settled", "cost_usd": 3.0, "cost_final": True}
_TORN = json.dumps(_ONE_SHOT)[:-25].encode()        # its writer died mid-append: the attempt's one and only row


@pytest.mark.parametrize("good, damage, gap", [
    (2.0, _TORN, "malformed_jsonl"),
    (0.0, _TORN, "malformed_jsonl"),
    (2.0, b"[3.0]\n", "invalid_jsonl_row"),
    (2.0, b"".join(json.dumps({**_ONE_SHOT, "attempt_id": "@", "cost_usd": usd}).encode().replace(b"@", bad) + b"\n"
                   for bad, usd in ((b"\xff", 3.0), (b"\xfe", 4.0))), "unreadable_bytes"),
], ids=["torn_tail", "torn_only_row", "not_an_object", "undecodable_ids"])
def test_a_journal_row_it_cannot_read_makes_the_spend_unknown_never_skipped(tmp_path, good, damage, gap):
    """``iter_jsonl_objects`` skips a row it cannot read as a JSON object and names why in ``gap_reasons``; the fallback
    dropped the why. A torn one-shot row, its attempt's only record, read as nothing: the ended lane was charged the $2
    before it ($0 when it is the journal's only row) and released the rest of its reservation. Rows after a torn one
    counted as a whole history, and two ids differing only in undecodable bytes merged, the earlier price lost. The
    core's own reader is fatal on corruption before the tail and calls a quarantined row's money unknown
    (``usage_store.integrity_degraded``). Here such a row makes the whole reading unknown: the live lane keeps its last
    good reading, the ended one stays charged its reservation, and the journal is still only read."""
    from ouroboros.usage_ledger import UsageLedgerCorrupt

    data = tmp_path / "data"
    path = _journal(data, _sent("a", ("settled", good, True)) if good else [])
    budget = run_live_lanes.RunBudget(15.0, 8.0)
    budget.note_spawned(("SM1", 1))
    assert budget.admit(("SM1", 1), 1, data, dispatch_index=0)[0]
    snap = budget.snapshot()
    assert (snap["spent_usd"], snap["spend_unknown"]) == (good, {}), "the last good reading"
    with path.open("ab") as journal:
        journal.write(damage)
    before = (path.read_bytes(), path.stat().st_mtime_ns)

    with pytest.raises(UsageLedgerCorrupt, match=gap):
        run_live_lanes.lane_spend(data)
    fact = run_live_lanes.lane_spend_fact(data, spawned=True)
    assert (fact["spent_usd"], fact["unknown_cost_rows"]) == (None, None) and gap in fact["spend_unknown"], fact
    snap = budget.snapshot()
    assert snap["spent_usd"] == good and "read failed in flight" in snap["spend_unknown"]["SM1_a1"]["reason"]
    budget.settle(("SM1", 1))
    snap = budget.snapshot()
    assert (snap["lanes_in_flight"], snap["spent_usd"], snap["unknown_liability_usd"]) == (0, good, 8.0 - good), "max(reservation, known)"
    assert gap in snap["spend_unknown"]["SM1_a1"]["reason"]
    admitted, facts = budget.admit(("SM1", 2), 1, tmp_path / "next", dispatch_index=1)
    assert (admitted, facts.get("reason")) == (False, "accounting_unavailable"), facts   # known + 8 fits the $15 cap, + liability not
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before, "the journal is read, never truncated"
    assert sorted(p.name for p in (data / "state").iterdir()) == ["usage_attempts.jsonl"], "no quarantine, no store"


def test_a_journal_gone_before_it_opens_is_unknown_never_an_empty_read(tmp_path, monkeypatch):
    """``iter_jsonl_objects`` answers a path it cannot find with no rows, as it answers an empty journal: one moved away
    between the fallback's ``is_file`` and its read was $0 known. The fallback opens the journal itself, so that read
    fails and the spend is unknown."""
    data = tmp_path / "data"
    (data / "state").mkdir(parents=True)
    real = pathlib.Path.is_file
    monkeypatch.setattr(pathlib.Path, "is_file", lambda self: self.name == "usage_attempts.jsonl" or real(self))
    with pytest.raises(FileNotFoundError):
        run_live_lanes.lane_spend(data)
    assert run_live_lanes.lane_spend_fact(data, spawned=True)["spent_usd"] is None


def _seq(rows: list) -> list:
    """The rows numbered as the journal's writer numbers them: the core's import refuses a journal without that."""
    return [{**row, "seq": index} for index, row in enumerate(rows, 1)]


def _imported(seed: pathlib.Path, name: str) -> pathlib.Path:
    """What the current core's one-time import (``usage_store.migrate_from_journal``) builds from a copy of a
    journal-era seed's ledger: the store beside the journal, a torn final row quarantined on the way."""
    from ouroboros import usage_store

    root = seed.parent / name
    shutil.copytree(seed / "state", root / "state")
    usage_store.migrate_from_journal(root)
    usage_store.forget(root)
    return root


def _core_money(root: pathlib.Path) -> dict:
    """The core's own answer for a lane's store: the settled bucket, and whether it calls that final."""
    from ouroboros import usage_store
    from ouroboros.usage_accounting import usage_writer_snapshot

    try:
        snapshot = usage_writer_snapshot(root)
    finally:
        usage_store.forget(root)
    return {key: snapshot[key] for key in ("settled_usd", "cost_final", "integrity_degraded")}


def _state_files(root: pathlib.Path) -> dict:
    return {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in sorted((root / "state").iterdir())}


PRECISE = "0.30000000000000000001"   # a literal a float would change: the core's export writes it as a string (``durable_literals``)


def test_a_numeric_string_price_is_known_money_from_the_journal_and_from_the_store_its_import_builds(tmp_path):
    """The core's money grammar takes a numeric string (``_usage_money.amount``, the row rule's ``_number``), and its
    import keeps that row whole: the store's typed column cannot hold a string, so it stays in ``extra`` and the store's
    reader decodes it back. ``type(cost) in (int, float)`` read the journal's two string prices as attempts without a
    price, and the store's bare column read the same: $0.50 known where the core counts $2.05. Both sources read the
    core's amount now, and agree."""
    journal = tmp_path / "journal"
    _journal(journal, _seq([*_sent("a", ("settled", PRECISE, True)), *_sent("b", ("settled", "1.25", False)),
                            *_sent("c", ("settled", 0.5, True))]))
    store = _imported(journal, "store")
    conn = sqlite3.connect(f"{(store / 'state' / 'usage.sqlite').as_uri()}?mode=ro", uri=True)
    try:
        assert conn.execute("SELECT attempt_id, cost_usd FROM attempts ORDER BY attempt_id").fetchall() == [
            ("a", None), ("b", None), ("c", "0.5")], "the typed column holds the float alone; the strings are in extra"
    finally:
        conn.close()
    assert _core_money(store) == {"settled_usd": 2.05, "cost_final": False, "integrity_degraded": False}
    assert run_live_lanes.lane_spend(journal) == run_live_lanes.lane_spend(store) == (2.05, 1)


@pytest.mark.parametrize("cost,error", [("-1", "UsageLedgerCorrupt"), ("three", "UsageLedgerCorrupt"),
                                        (float("inf"), "UsageNonFiniteMoney")])
def test_a_price_the_core_refuses_makes_the_journal_reading_unknown_never_known_or_skipped(tmp_path, cost, error):
    """The same grammar's other side (``usage_ledger.validate_row_fields``): a negative or non-numeric cost makes a
    corrupt row and a non-finite one is never stored. None is read as known money, or as one more unknown row beside a
    reading that otherwise looks whole: the reading itself is unknown."""
    data = tmp_path / "data"
    _journal(data, [*_sent("a", ("settled", 2.0, True)), *_sent("b", ("settled", cost, True))])
    with pytest.raises(Exception) as caught:
        run_live_lanes.lane_spend(data)
    assert type(caught.value).__name__ == error, caught.value
    fact = run_live_lanes.lane_spend_fact(data, spawned=True)
    assert (fact["spent_usd"], fact["unknown_cost_rows"]) == (None, None) and error in fact["spend_unknown"], fact


@pytest.mark.parametrize("shape", ["released_only", "no_rows"])
@pytest.mark.parametrize("source", ["journal", "store"])
def test_a_quarantined_journal_row_is_money_nobody_can_read_never_a_known_zero(tmp_path, shape, source):
    """The core moves a torn final journal row into ``usage_attempts.quarantine.jsonl`` and from then on calls that
    ledger's money incomplete (``usage_store.integrity_degraded``: never ``cost_final``); its import keeps the file
    beside the store it builds. The stand never looked: a ledger left with released attempts only, or with no row at
    all, read as a known $0 through either source, and the ended lane released its reservation. Without a quarantine
    the same rows are a known $0. Read only: the stand quarantines, truncates and creates nothing."""
    from ouroboros import usage_journal

    rows = _seq([{"attempt_id": "h", "state": "reserved"}, {"attempt_id": "h", "state": "released"}]) if shape == "released_only" else []
    torn = json.dumps({"seq": len(rows) + 1, "attempt_id": "x", "kind": "external_unmetered", "state": "settled",
                       "cost_usd": 4.0, "cost_final": True})[:-20].encode()   # its writer died mid-append
    with _journal(tmp_path / "seed", rows).open("ab") as journal:
        journal.write(torn)
    if source == "store":
        data = _imported(tmp_path / "seed", "data")
        assert _core_money(data) == {"settled_usd": 0.0, "cost_final": False, "integrity_degraded": True}
    else:   # a journal-era seed's own reader quarantined it, and no store was built
        data = tmp_path / "seed"
        assert len(usage_journal.read_journal_records(data)[0]) == len(rows)
    _journal(tmp_path / "clean", rows)
    assert run_live_lanes.lane_spend(tmp_path / "clean") == (0.0, 0), "the same rows without a quarantine"
    before = _state_files(data)
    assert "usage_attempts.quarantine.jsonl" in before and before["usage_attempts.jsonl"][0] == b"".join(
        json.dumps(row).encode() + b"\n" for row in rows), "the core truncated the journal before the torn row"

    assert run_live_lanes.lane_spend(data) == (0.0, 1)
    assert run_live_lanes.lane_spend_fact(data, spawned=True) == {"spent_usd": 0.0, "unknown_cost_rows": 1}
    budget = run_live_lanes.RunBudget(30.0, 8.0)
    assert budget.admit(("SW1", 1), 1, data, dispatch_index=0)[0]
    budget.settle(("SW1", 1))
    snap = budget.snapshot()
    assert (snap["spent_usd"], snap["unknown_liability_usd"]) == (0.0, 8.0) and "SW1_a1" in snap["spend_unknown"], snap
    after = _state_files(data)
    assert {name: after[name] for name in before} == before, "read only: nothing quarantined or truncated, the store unchanged"
    sidecars = {name: content for name, content in after.items() if name not in before}   # a WAL store's read-only reader
    assert set(sidecars) <= {"usage.sqlite-wal", "usage.sqlite-shm"} and sidecars.get("usage.sqlite-wal", (b"",))[0] == b"", \
        "nothing created but SQLite's own reader sidecars, and no frame written"


class _Spawned(_NoopServer):
    """A started server process whose stop ``IsolatedServer.stop`` proved."""
    proc, stop_unconfirmed = types.SimpleNamespace(pid=4242), []
    stop = lambda self: []  # noqa: E731


class _SpawnedStopUnproved(_Spawned):
    stop = lambda self: ["server 4242: exit unconfirmed after a forced stop"]  # noqa: E731


@pytest.mark.parametrize("proof", ["stop_proved", "stop_unproved", "survivor"])
def test_a_lane_whose_process_may_still_run_stays_charged_its_reservation(tmp_path, monkeypatch, proof):
    """Every price read at settlement is final ($1), yet a server whose exit the stop could not prove, or a process still
    carrying the lane's data root, may go on spending under its own TOTAL_BUDGET: the ended lane stays charged its
    reservation, by the same predicate that keeps its clone under ``--prune-clones``. A proved stop releases it."""
    if proof == "survivor":
        from tests.system_e2e import harness
        monkeypatch.setattr(harness, "wait_until", lambda predicate, timeout, interval=0.5: False)
        monkeypatch.setattr(harness, "pids_with_env_value", lambda value: [os.getpid()])
    budget = run_live_lanes.RunBudget(30.0, 8.0, reader=lambda root: (1.0, 0))
    _attempt(tmp_path, monkeypatch, server=_SpawnedStopUnproved if proof == "stop_unproved" else _Spawned, budget=budget,
             procfs=proof == "survivor")
    held = proof != "stop_proved"
    snap = budget.snapshot()
    assert (snap["lanes_in_flight"], snap["spent_usd"], snap["unknown_liability_usd"]) == (0, 1.0, 15.0 if held else 0.0)   # SK1: 2 x $8
    assert ("SK1_a1" in snap["spend_unknown"]) is held
    admitted, facts = budget.admit(("SK1", 2), 2, tmp_path / "next", dispatch_index=1)
    assert (admitted, facts.get("reason")) == ((False, "accounting_unavailable") if held else (True, None)), facts


def test_an_attempt_only_the_unknown_keeps_out_is_named_in_both_artifacts(tmp_path, monkeypatch):
    """A cap the known spend never reached must not read as one, in the row, the index or the run's stop reason."""
    budget = run_live_lanes.RunBudget(20.0, 8.0, reader=_reader({}, {"gone"}))   # SK1 needs 16: it fits, but not beside the unknown 8
    assert budget.admit(("SM1", 9), 1, tmp_path / "gone", dispatch_index=0)[0]
    budget.settle(("SM1", 9))
    row, lane, _same = _attempt(tmp_path, monkeypatch, budget=budget)
    assert row["status"] == "not_run" and row["reason_code"] == "accounting_unavailable"
    assert row["refusal"]["type"] == "RunBudgetAccountingUnavailable" and row["refusal"]["code"] == "accounting_unavailable"
    assert row["budget"]["unknown_liability_usd"] == 8.0
    index = json.loads((lane.parent.parent / "result_index.jsonl").read_text(encoding="utf-8").strip())
    assert (index["status"], index["reason_code"]) == ("not_run", "accounting_unavailable")
