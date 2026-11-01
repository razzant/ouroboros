"""A bounded Windows observation: which call refuses a new process in a Job, and in which Job histories.

Equipment, not product, and never a qualification verdict. It starts only trivial disposable Python children, and
it drives the same ``Custody.spawn`` and ``Custody.stop_all`` as the qualification (with the observation patch). Each
trial owns one fresh kill-on-close Job. A trial records what was measured: ``accepted`` (the call sequence finished,
the child wrote its marker and exited 0), ``refused`` (with the failing stage and the readings taken before cleanup),
or an unknown outcome. Nothing here retries as a repair; the recheck is a measurement of the same Job, 2 s later.
A refused start also records whether killing and reaping the child worked (``cleanup``): a failed cleanup keeps the
refusal and its stage, is disclosed beside it, and makes the measurement incomplete.

    python job_reuse_probe.py --root D:\\jrp-1234abcd --python <pinned python.exe> --evidence <dir>
    python job_reuse_probe.py --plan     # prints the plan; starts nothing, on any platform

Exit 0: every planned trial was measured, its custody confirmed and every refused start cleaned up, whatever the trials
say, refusals included. Exit 2: the measurement is incomplete. Neither code says anything about the product.
``TOTAL_DEADLINE`` limits when a new trial may start; it is not a bound on when the run ends (the last admitted trial
finishes, and the workflow step has its own timeout).
"""
import argparse
import collections
import ctypes
import hashlib
import json
import os
import pathlib
import platform
import random
import subprocess
import sys
import time
from time import perf_counter_ns

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from wq_custody import Custody  # noqa: E402

SCHEMA = "update-windows-job-reuse-probe/1"
# More trials where the retained refusal happened; the others are controls for what could explain it.
ARMS = {
    "fresh_job": 3,                  # nothing ever ran in this Job
    "emptied_no_terminate": 3,       # its one member exited by itself; TerminateJobObject never called
    "terminated_empty": 3,           # TerminateJobObject on a Job that never held a process
    "terminated_tree_immediate": 15,  # a tree (2 Python, their conhosts) stopped; the next spawn follows at once
    "terminated_tree_2s": 6,         # the same, the next spawn after the ~2 s gap seen where reuse succeeded
}
GAP_SECONDS = {"terminated_tree_2s": 2.0}
MAX_RECHECKS, RECHECK_SECONDS = 2, 2.0  # the same Job once more after a refusal, first refusals only
MAX_TRIALS, TOTAL_DEADLINE = 32, 180  # TOTAL_DEADLINE: seconds during which a new trial is admitted, not a hard stop
STEP_DEADLINE = 15  # one wait (a marker, a stop, a child's exit)
SEED = 20261101
MEASURED = ("accepted", "refused")
CLEANED = {"kill": "ok", "wait": "ok"}  # what a refused start's cleanup reads when it worked

TREE = ("import subprocess, sys, time\n"
        "code = \"import sys, time; open(sys.argv[1], 'w').close(); time.sleep(45)\"\n"
        "child = subprocess.Popen([sys.executable, '-E', '-s', '-B', '-c', code, sys.argv[1]],"
        " creationflags=0x08000000)\n"
        "time.sleep(45)\n")
TOUCH = "import sys; open(sys.argv[1], 'w').close()"
NON_CLAIM = ("Observation of disposable Python children only. Refused or accepted here is not a cause of the retained "
             "refusal, and no refusal here is not proof of a repaired product.")


class Recorder:
    """The evidence stream: one flushed JSON line per event, so a killed probe still leaves what it saw."""

    def __init__(self, path):
        self.fh, self.rows, self.trial = open(path, "a", encoding="utf-8"), [], None

    def emit(self, event, **facts):
        row = {"event": event, "trial": self.trial, "perf_ns": perf_counter_ns(), **facts}
        self.fh.write(json.dumps(row, sort_keys=True, default=repr) + "\n")
        self.fh.flush()
        self.rows.append(row)


def plan(seed=SEED):
    """The finite order of trials, drift spread across arms by a recorded seed."""
    arms = [arm for arm, count in ARMS.items() for _ in range(count)]
    if len(arms) + MAX_RECHECKS > MAX_TRIALS:
        raise ValueError("the plan exceeds its bound")
    random.Random(seed).shuffle(arms)
    return [{"seq": seq, "arm": arm} for seq, arm in enumerate(arms)]


def child_env():
    """Only what a Windows Python needs to start: no inherited tokens, paths or runner variables."""
    return {key: os.environ[key] for key in ("SystemRoot", "SystemDrive", "TEMP", "TMP") if key in os.environ}


def attempt(custody, raw, python, base, label):
    """One start of a child that must write a marker and exit 0; what happened, never raising for a refusal."""
    marker, mark = base / f"{label}.marker", len(raw.rows)
    try:
        proc, _identity, _ = custody.spawn(label, [python, "-E", "-s", "-B", "-c", TOUCH, marker], env=child_env(),
                                           cwd=base, logs=base)
    except Exception as exc:
        refused = [row for row in raw.rows[mark:] if row["event"] == "spawn_refused"]
        if not refused:  # Popen itself lies outside the refusal path: it names no stage
            return {"outcome": "popen_failed", "error": repr(exc)}
        row = refused[-1]
        return {"outcome": "refused", "stage": row["stage"], "error": row["error"], "pid": row["pid"],
                "since_stop_ms": row["since_stop_ms"], "observed": row["observed"], "cleanup": row["cleanup"]}
    spawned = [row for row in raw.rows[mark:] if row["event"] == "spawned"][-1]
    try:
        returncode = proc.wait(timeout=STEP_DEADLINE)
    except subprocess.TimeoutExpired:
        returncode = None
    ran = returncode == 0 and marker.exists()
    return {"outcome": "accepted" if ran else "unverified", "returncode": returncode, "marker": marker.exists(),
            "since_stop_ms": spawned["since_stop_ms"]}


def _until(condition):
    end = time.monotonic() + STEP_DEADLINE
    while not condition():
        if time.monotonic() >= end:
            return False
        time.sleep(0.05)
    return True


def _stop(custody, reason):
    stop = custody.stop_all(reason, deadline=STEP_DEADLINE)
    return {"confirmed": stop["confirmed"], "not_gone": stop["not_gone"], "active_after": stop["active_after"],
            "census": None if stop["before"] is None else len(stop["before"]), "terminate": stop["terminate"]}


def _terminated(stop):
    """A stop that counts as the arm's history: TerminateJobObject itself succeeded (an empty Job reads active 0
    whatever it did), the count reads 0 and nothing of the census is left."""
    return stop["terminate"] == "ok" and stop["confirmed"] is True and stop["not_gone"] == []


def prepare(arm, custody, raw, python, base):
    """The Job's history before the target; ``ready`` True only when that history was measured to have happened."""
    if arm == "fresh_job":
        return {"ready": True}
    if arm == "terminated_empty":
        stop = _stop(custody, "probe: empty Job")
        return {"ready": _terminated(stop), "stop": stop}
    if arm == "emptied_no_terminate":
        member = attempt(custody, raw, python, base, "member")
        return {"ready": member["outcome"] == "accepted" and _until(lambda: custody.win.active(custody.job) == 0),
                "member": member}
    marker = base / "tree.marker"
    custody.spawn("tree", [python, "-E", "-s", "-B", "-c", TREE, marker], env=child_env(), cwd=base, logs=base)
    if not _until(marker.exists):
        return {"ready": False, "tree": "no marker by the deadline"}
    stop = _stop(custody, "probe: tree stopped")
    return {"ready": _terminated(stop) and bool(stop["census"]), "stop": stop}


def run_trial(win, raw, python, trial, root, rechecks):
    """One trial in its own Job. ``rechecks`` is the remaining allowance; the trial's row says if it spent one."""
    arm, base = trial["arm"], pathlib.Path(root) / "trials" / f"{trial['seq']:03d}-{trial['arm']}"
    base.mkdir(parents=True)
    raw.trial = trial["seq"]
    row, custody = {**trial, "outcome": "setup_unknown"}, None
    try:
        custody = Custody(win, raw, arm, [str(pathlib.Path(python).parent)])
        row["setup"] = prepare(arm, custody, raw, python, base)
        if row["setup"]["ready"] is True:
            if arm in GAP_SECONDS:
                time.sleep(GAP_SECONDS[arm])
            row["target"] = attempt(custody, raw, python, base, "target")
            row["outcome"] = row["target"]["outcome"]
            if row["outcome"] == "refused" and rechecks > 0:
                time.sleep(RECHECK_SECONDS)
                row["recheck"] = attempt(custody, raw, python, base, "recheck")
    except Exception as exc:  # the row stays unknown, with its reason, and the Job is still closed below
        row["error"] = repr(exc)
    finally:
        if custody is None:
            row["teardown"] = {"status": "unknown", "note": "no Job opened"}
        else:
            row["teardown"] = custody.close()
        raw.trial = None
    raw.emit("trial", **row)
    return row


def summarize(trials, rows, host, digests):
    """Counts only. ``measurement`` is complete when every planned trial was measured, its custody confirmed and every
    refused target or recheck cleaned up; nothing in this result is, or can become, a qualification PASS.
    ``target_refused`` looks at target starts only: a refusal of a setup start (a tree or member start into a fresh
    Job) leaves its trial unmeasured, so the run reads incomplete, and it is in ``events.jsonl``, not in the counts."""
    arms = {arm: collections.Counter() for arm in ARMS}
    stages = {arm: collections.Counter() for arm in ARMS}
    gaps = {arm: [] for arm in ARMS}
    for row in rows:
        arms[row["arm"]][row["outcome"]] += 1
        target = row.get("target") or {}
        if row["outcome"] == "refused":
            stages[row["arm"]][target.get("stage")] += 1
        if target.get("since_stop_ms") is not None:
            gaps[row["arm"]].append(target["since_stop_ms"])
    uncleaned = [{"seq": row["seq"], "arm": row["arm"], "attempt": name, "pid": found.get("pid"), "cleanup": found.get("cleanup")}
                 for row in rows for name, found in ((name, row.get(name)) for name in ("target", "recheck"))
                 if found and found.get("outcome") == "refused" and found.get("cleanup") != CLEANED]
    complete = not uncleaned and len(rows) == len(trials) and all(
        row["outcome"] in MEASURED and row["teardown"].get("status") == "confirmed" for row in rows)
    return {"schema": SCHEMA, "measurement": "complete" if complete else "incomplete", "non_claim": NON_CLAIM,
            "planned": len(trials), "ran": len(rows), "seed": SEED,
            "target_refused": any(row["outcome"] == "refused" for row in rows), "cleanup_uncertain": uncleaned,
            "outcomes": {arm: dict(counts) for arm, counts in arms.items()},
            "refusal_stages": {arm: dict(counts) for arm, counts in stages.items()},
            "since_stop_ms": {arm: [min(values), max(values)] for arm, values in gaps.items() if values},
            "refusals": [{"seq": row["seq"], "arm": row["arm"], **row["target"]} for row in rows
                         if row["outcome"] == "refused"][:8],
            "rechecks": [{"seq": row["seq"], "arm": row["arm"], **row["recheck"]} for row in rows if "recheck" in row],
            "host": host, "equipment_files": digests}


def run(win, raw, python, root, deadline=TOTAL_DEADLINE):
    trials, rows, rechecks, end = plan(), [], MAX_RECHECKS, time.monotonic() + deadline
    for trial in trials:
        if time.monotonic() >= end:
            raw.emit("deadline", deadline=deadline, ran=len(rows), planned=len(trials))
            break
        row = run_trial(win, raw, python, trial, root, rechecks)
        rechecks -= 1 if "recheck" in row else 0
        rows.append(row)
    return trials, rows


def normalized_sha256(path):
    """The digest of a text file's LF form: the hosted image's system Git config checks files out with CRLF."""
    return hashlib.sha256(pathlib.Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def verified_files(folder=HERE):
    """The checked-out equipment against its own manifest; ``{name: sha256}``, or a refusal before any start."""
    declared = json.loads((pathlib.Path(folder) / "MANIFEST.json").read_text(encoding="utf-8"))["files"]
    actual = {name: normalized_sha256(pathlib.Path(folder) / name) for name in declared}
    wrong = sorted(name for name in declared if actual[name] != declared[name])
    if wrong:
        raise SystemExit(f"equipment differs from MANIFEST.json: {wrong}; nothing was started")
    return actual


def host_facts(win):
    """Where the measurement ran; a fact that cannot be read says so."""
    facts = {"python": sys.version, "platform": platform.platform(), "image_os": os.environ.get("ImageOS"),
             "image_version": os.environ.get("ImageVersion")}
    try:  # the current-process pseudo-handle: is the driver itself in a Job (children inherit that)?
        facts["driver_in_any_job"] = win.in_job(ctypes.c_void_p(-1).value, None)
    except Exception as exc:
        facts["driver_in_any_job"] = {"read_error": repr(exc)}
    return facts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--root")
    parser.add_argument("--python")
    parser.add_argument("--evidence")
    args = parser.parse_args(argv)
    if args.plan:
        print(json.dumps({"schema": SCHEMA, "trials": plan(), "max_rechecks": MAX_RECHECKS,
                          "total_deadline_seconds": TOTAL_DEADLINE}, indent=1))
        return 0
    if os.name != "nt":
        raise SystemExit("Native Windows required; nothing measured.")
    if not (args.root and args.python and args.evidence):
        parser.error("--root, --python and --evidence are required")
    from wq_win import Win
    digests = verified_files()
    evidence = pathlib.Path(args.evidence)
    evidence.mkdir(parents=True)
    pathlib.Path(args.root).mkdir(parents=True, exist_ok=True)
    win, raw = Win(), Recorder(evidence / "events.jsonl")
    host = host_facts(win)
    raw.emit("probe_start", host=host, equipment_files=digests, trials=plan())
    trials, rows = run(win, raw, args.python, args.root)
    summary = summarize(trials, rows, host, digests)
    (evidence / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("measurement", "planned", "ran", "target_refused", "outcomes",
                                                    "refusal_stages", "since_stop_ms", "rechecks",
                                                    "cleanup_uncertain")}, indent=1))
    return 0 if summary["measurement"] == "complete" else 2


if __name__ == "__main__":
    sys.exit(main())
