"""The post-merge check's window: where both legs start, and what earlier attempts still owe.

scripts/post_merge_check.py's `base` step chooses the window with `choose_window`; its `route` and
`summary` steps read and write the receipts described here. Stdlib only, as that script is.

A hand-over record is {head, run, attempt, base, ids}: the commit, the workflow attempt and, for a
size test, the comparison base at which these IDs failed without reaching an open issue. A leg
publishes the records it still owes as its receipt, in the name of a job step (`OWED_STEP`); the
window job publishes what it carries in the same way (`REPLAY_STEP`). A value too long for a step
name travels as an artifact that the step names by digest. A receipt is a list of records;
{"partial": records} when the leg could not take over what earlier attempts recorded, so its list
adds to theirs; null when the leg cannot say what failed.

Only an attempt that measured one first-parent range on both systems and left nothing owed
qualifies its head as the next starting point, and only that closes the walk behind it. Until then
every later check keeps the base of the unqualified window and the records its attempts name. A
later check of that range gives its own result and changes nothing about theirs.
"""
from __future__ import annotations

from functools import cache
import json
import re
from typing import NamedTuple
import urllib.error

from ouroboros.update_qualification import RANGE_TITLE, _LEG, attempt_job_pages, measured_window, range_candidate
from scripts.post_merge_routing import BRANCH, SYSTEMS, _reason

REPLAY_STEP = "Post-merge replay input "  # the window job: what it carries in from earlier attempts
OWED_STEP = "Post-merge untracked "  # a leg: what it still owes a hand-over
_BASE_STEP = re.compile(r"Post-merge (confirmed|provisional) base ([0-9a-f]{40})")
_SOURCE = ("head", "run", "attempt", "base")  # where, and by which attempt, a record's IDs were seen failing
_MISSING = object()  # no step carries the value
LOST = object()  # a receipt that was never delivered or is gone for good
UNREAD = object()  # a receipt that could not be read now: nothing is known about what it holds


# --- records ---------------------------------------------------------------------------------

def records(value) -> list[dict] | None:
    """`value` as hand-over records, None when it is anything else. Each names at least one ID; a commit is a
    full SHA and a number positive, or None where the source did not know it."""
    if not isinstance(value, list):
        return None
    rows = []
    for entry in value:
        ids = entry.get("ids") if isinstance(entry, dict) else None
        if not (isinstance(ids, list) and ids and all(isinstance(nodeid, str) and nodeid for nodeid in ids)
                and all(entry.get(key) is None or isinstance(entry[key], str)
                        and re.fullmatch(r"[0-9a-f]{40}", entry[key]) for key in ("head", "base"))
                and all(entry.get(key) is None or type(entry[key]) is int and entry[key] >= 1
                        for key in ("run", "attempt"))):
            return None
        rows.append({**{key: entry.get(key) for key in _SOURCE}, "ids": sorted(set(ids))})
    return rows


def _source(row: dict) -> tuple:
    return tuple(row[key] for key in _SOURCE)


def merge_records(*lists) -> list[dict]:
    """One record per source, first seen first, with every ID any of `lists` names for it."""
    merged = {}
    for row in (row for rows in lists for row in rows):
        merged.setdefault(_source(row), set()).update(row["ids"])
    return [{**dict(zip(_SOURCE, source)), "ids": sorted(ids)} for source, ids in merged.items()]


def _obligations(rows) -> set:
    return {(_source(row), nodeid) for row in rows for nodeid in row["ids"]}


def _unseen(rows, carried: set) -> list[dict]:
    """`rows` without what a newer attempt's window already carried in."""
    kept = []
    for row in rows:
        if ids := [nodeid for nodeid in row["ids"] if (_source(row), nodeid) not in carried]:
            kept.append({**row, "ids": ids})
    return kept


def carried_records(replay, leg: str) -> list[dict] | None:
    """What a window's replay record carried in for `leg`; None when `replay` is not that record (a reference
    to its artifact is not, nor is `LOST` or `UNREAD`)."""
    if not isinstance(replay, dict) or "artifact" in replay:
        return None
    rows = replay.get("awaiting_handoff", {})
    return records(rows.get(leg, [])) if isinstance(rows, dict) else None


# --- receipts --------------------------------------------------------------------------------

def read_handoff(value, reader, part: str, run: int, *, attempt=None, max_attempt=None):
    """`value`, or the artifact it references, read by `reader(run, attempt, part, sha256)`.

    `LOST`: the reader's FileNotFoundError, which is a fact about the run (the receipt was never delivered or
    has expired), or a reference that is not one of this run's attempts. `UNREAD`: any other failure, which
    says nothing about what the receipt holds and never stands for an empty one."""
    if not isinstance(value, dict) or "artifact" not in value:
        return value
    ref = value["artifact"]
    if (not isinstance(ref, dict) or type(ref.get("run")) is not int or ref["run"] != run
            or type(ref.get("attempt")) is not int or ref["attempt"] < 1
            or attempt is not None and ref["attempt"] != attempt
            or max_attempt is not None and ref["attempt"] > max_attempt
            or not isinstance(ref.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", ref["sha256"])):
        return LOST
    try:
        return reader(run, ref["attempt"], part, ref["sha256"])
    except FileNotFoundError as error:
        print(f"::warning::the hand-over artifact of {run}/{ref['attempt']} {part} is gone ({error})")
        return LOST
    except Exception as error:  # noqa: BLE001 -- unread evidence never confirms a hand-over
        print(f"::warning::cannot read hand-over artifact for {run}/{ref['attempt']} {part} ({_reason(error)})")
        return UNREAD


def _succeeded(step) -> bool:
    return isinstance(step, dict) and step.get("status") == "completed" and step.get("conclusion") == "success"


def _step_value(job: dict, prefix: str):
    """The JSON a successful step of `job` carries in its name after `prefix`: None when that is no JSON,
    `_MISSING` without such a step."""
    for step in job.get("steps") or []:
        if _succeeded(step) and (name := step.get("name") or "").startswith(prefix):
            try:
                return json.loads(name[len(prefix):])
            except ValueError:
                return None
    return _MISSING


def _ran(job: dict) -> bool:
    """Whether a leg that did not end green got as far as running tests."""
    return job.get("conclusion") != "success" and any(
        isinstance(step, dict) and (step.get("name") or "").startswith("Run the affected tests")
        and step.get("conclusion") != "skipped" for step in job.get("steps") or [])


def _gap(run: dict, attempt: int, part: str, base=None) -> dict:
    """One loss: `part` is a system, "window" (what the window carried in) or "jobs" (the attempt's job list);
    `base` the commit that attempt measured from, where it is known."""
    return {"run": run["id"], "attempt": attempt, "part": part, "base": base}


def gap_text(gap: dict) -> str:
    """One loss in words, for the window job's log and a leg's summary."""
    part = gap.get("part")
    what = {"window": "The record of what the window carried in", "jobs": "The job list"}.get(
        part, f"The hand-over record on {SYSTEMS.get(part, part)}")
    return (f"{what} of run {gap.get('run')}, attempt {gap.get('attempt')} is gone or was never written, so what "
            "that attempt could not hand over is unknown. This check measures the range again and answers only "
            "for what it ran")


class Prior(NamedTuple):
    """What the attempts of one run recorded about its window (`prior_window`)."""
    chosen: tuple | None  # (base, confirmed) of its window job; None when no attempt recorded one
    searched: bool  # that job read the history behind it whole
    awaiting: dict  # leg -> the records this run still leaves owed
    carried: dict | None  # leg -> the records its window carried in; None without a readable replay record
    gaps: list  # receipts of this run that are gone for good (`_gap`)
    unread: bool  # a receipt this run's state depends on could not be read now


def prior_window(run: dict, get, handoff=None) -> Prior:
    """Read `run`'s attempts, the latest first, as far as its window and both legs need.

    A rerun of failed jobs lists only the legs it reran: the window job and the other leg stay in the earlier
    attempt and are read from there. A leg's latest whole receipt stands for that leg. A partial one, a lost
    one or none leaves the earlier attempts and what the window carried in standing beside it, and a leg that
    ran tests without leaving a readable receipt is a gap. Nothing here joins one attempt's leg to another's
    as a measurement: `measured_window` reads one attempt. A 410 on an earlier attempt's job list is a gap;
    a bare 404 proves no loss and only leaves what that attempt recorded unread.
    """
    chosen, snapshot, recorded, unread = None, _MISSING, run["run_attempt"], False
    owed, whole, gaps = {leg: [] for leg in SYSTEMS}, set(), []
    for attempt in range(run["run_attempt"], 0, -1):
        try:
            jobs = [job for page in attempt_job_pages({**run, "run_attempt": attempt}, get) for job in page]
        except urllib.error.HTTPError as error:
            if attempt == run["run_attempt"] or error.code not in (404, 410):
                raise
            if error.code == 410:
                gaps.append(_gap(run, attempt, "jobs"))  # gone for good: what it recorded is lost
            else:
                unread = True  # a bare 404 proves no loss: unknown, not gone
                print(f"::warning::the job list of run {run['id']}, attempt {attempt} answered HTTP 404; "
                      "what it recorded stays unknown")
            break
        for job in jobs:
            if job.get("name") == "window" and chosen is None:
                for step in job.get("steps") or []:
                    if _succeeded(step) and (match := _BASE_STEP.fullmatch(step.get("name") or "")):
                        chosen = match[2], match[1] == "confirmed"
                snapshot, recorded = read_handoff(_step_value(job, REPLAY_STEP), handoff, "window", run["id"],
                                                  attempt=attempt), attempt
            elif (match := _LEG.fullmatch(job.get("name") or "")) and match[1] not in whole:
                leg, start = match.groups()
                value = read_handoff(_step_value(job, OWED_STEP), handoff, leg, run["id"], attempt=attempt)
                partial = isinstance(value, dict) and "partial" in value
                rows = records(value["partial"] if partial else value)
                if value is UNREAD:
                    unread = True
                elif rows is not None:
                    owed[leg] = merge_records(owed[leg], rows)
                    if not partial:
                        whole.add(leg)
                elif value is not _MISSING or _ran(job):
                    gaps.append(_gap(run, attempt, leg, start))  # it measured and left no record of what it owes
        if chosen is not None and len(whole) == len(SYSTEMS):
            break
    carried = {leg: carried_records(snapshot, leg) for leg in SYSTEMS}
    searched = None not in carried.values() and snapshot.get("history_read") is True
    if None in carried.values():
        carried = None
        if snapshot is not _MISSING and len(whole) < len(SYSTEMS):  # a leg without its own receipt needed it
            if snapshot is UNREAD:
                unread = True
            else:
                gaps.append(_gap(run, recorded, "window", chosen and chosen[0]))
    awaiting = {leg: rows if leg in whole else merge_records(rows, (carried or {}).get(leg, ()))
                for leg, rows in owed.items()}
    return Prior(chosen, searched, {leg: rows for leg, rows in awaiting.items() if rows}, carried, gaps, unread)


# --- the window ------------------------------------------------------------------------------

def choose_window(runs, head: str, *, get, ancestor, initial: str = "", replay: dict,
                  handoff=None) -> tuple[str | None, str, bool]:
    """(base, reason, continuous) of the window both legs measure at `head`, the base None when earlier checks
    establish no starting point at all; `replay` receives what the window carries in.

    `runs` is the workflow's history, newest first; `ancestor(older, newer)` answers for the first-parent line.
    The walk reads each run's latest attempt until one qualified its head on both systems:
    that head closes the walk, and nothing behind it is read. An attempt before it that chose a confirmed base
    and did not qualify keeps that base (the oldest one kept wins, so a late rerun behind a cancelled check
    widens the window to its own base) and adds the records it still owes, each under its own source. The walk
    also ends at a run whose head lies before the oldest base kept: no check reads further back than the range
    it measures. A record a newer attempt's window already carried in is that attempt's to report, and is not
    taken from its older source again. A receipt that is gone for good is a labelled gap beside the records that
    are known (`history_gaps`): this window measures that range again, and its result is its own. History that
    could not be read now leaves the base provisional: `continuous` is False and no leg of this check can qualify.
    A job list that answers 410 Gone is gone for good; a bare 404 proves no loss, so it is no gap — it only leaves
    what that run recorded unknown, and any base the walk still finds stays unconfirmed.

    A push run without the range title predates these receipts, and so does everything older: the search ends
    there. Before the first confirmed base a green measurement settles its own window and an unmeasured or red
    one that starts before it stays a known gap; only an explicit `initial` base starts a confirmed range, for
    the checks that follow it, and no provisional window behind a confirmed one widens it.
    """
    readable = runs is not None
    pending, green, held, boundary = initial, None, None, None
    owed, carried, gaps = {leg: [] for leg in SYSTEMS}, {leg: set() for leg in SYSTEMS}, []
    try:
        for run in runs if readable else ():
            if not isinstance(run.get("display_title"), str) or (run["display_title"] == RANGE_TITLE and
                    not {"head_branch", "event", "head_sha", "status", "conclusion", "id", "run_attempt"} <= run.keys()):
                readable = False  # it may hide an attempt that left a red owed
                continue
            if run["display_title"] != RANGE_TITLE and run.get("head_branch") == BRANCH and run.get("event") == "push":
                break
            sha = run.get("head_sha") or ""
            if not range_candidate(run, include_interrupted=True):
                continue
            if (sha == head and not initial) or (sha != head and not ancestor(sha, head)):
                continue
            if held and ancestor(sha, held[0]):
                break  # It ended before the base this window keeps: that range closed when the base was confirmed.
            jobs = cache(get)  # Both readers stop at their witness and share each attempt page already fetched.
            try:
                receipt = measured_window(run, jobs)
                if receipt and receipt[1] and sha != head:
                    boundary = sha, (f"eligible on both systems (run {run['id']}, attempt {run['run_attempt']}, "
                                     f"{run['conclusion']})")
                    break
                prior = prior_window(run, jobs, handoff)
            except urllib.error.HTTPError as error:
                if error.code not in (404, 410):
                    raise
                if error.code == 410:
                    if run.get("conclusion") != "success":
                        gaps.append(_gap(run, run["run_attempt"], "jobs"))  # gone for good
                else:
                    readable = False  # a bare 404 proves no loss: what this run recorded is unknown
                    print(f"::warning title=Post-merge history unread::the job list of run {run['id']}, attempt "
                          f"{run['run_attempt']} answered HTTP 404; its records stay unknown, the base unconfirmed")
                continue  # An older base is reused; nothing unread counts as settled.
            gaps += prior.gaps
            readable = readable and not prior.unread
            for leg in SYSTEMS:
                owed[leg] = merge_records(owed[leg], _unseen(prior.awaiting.get(leg, ()), carried[leg]))
                carried[leg] |= _obligations((prior.carried or {}).get(leg, ()))
            chosen = prior.chosen if prior.chosen and ancestor(prior.chosen[0], sha) else None
            if chosen and chosen[1]:
                # Only a qualified descendant closes the walk: this window keeps its base, and the oldest one wins.
                if held is None or ancestor(chosen[0], held[0]):
                    held = chosen[0], f"retained starting base of run {run['id']}, attempt {run['run_attempt']}"
                continue
            if held is None:
                # Before the first confirmed base, keep a known gap even if older history is unreadable.
                if chosen and receipt and receipt[0] == chosen[0] and run["conclusion"] == "success":
                    if green is None:
                        green = sha, chosen[0]
                    elif not pending and (sha == green[1] or ancestor(green[1], sha)) and ancestor(chosen[0], green[1]):
                        green = green[0], chosen[0]
                elif chosen and (not green or ancestor(chosen[0], green[1])) and (not pending or ancestor(chosen[0], pending)):
                    pending = chosen[0]
            if chosen and prior.searched:
                break  # This provisional window already located the oldest known gap. An explicit seed may widen it.
    except Exception as error:  # noqa: BLE001 -- keep any earlier starting point already observed
        print(f"::warning title=Post-merge base::earlier checks are unreadable ({_reason(error)}); falling back")
        readable = False
    replay.update(history_read=readable, awaiting_handoff={leg: rows for leg, rows in owed.items() if rows},
                  history_gaps=gaps)
    point = boundary or held
    if point:
        if boundary and held and ancestor(held[0], boundary[0]):
            point = held
        if pending and ancestor(pending, point[0]):
            point = pending, "widened first-parent window"
        return point[0], point[1] if readable else point[1] + "; earlier history unreadable", readable
    if initial:
        return pending, "explicit first-parent window" if readable else "explicit range; earlier history unreadable", readable
    if pending:
        return pending, "earlier range still pending", False
    if green:
        return green[0], "head of the last green check (earlier coverage remains pending)", False
    return None, "no earlier check establishes a starting point", False
