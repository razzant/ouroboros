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

from ouroboros.update_qualification import (POST_MERGE_RUNS, RANGE_TITLE, _LEG, actions_pages, attempt_job_pages,
                                            measured_window, range_candidate)
from scripts.post_merge_routing import BRANCH, SYSTEMS, _reason

REPLAY_STEP = "Post-merge replay input "  # the window job: what it carries in from earlier attempts
OWED_STEP = "Post-merge untracked "  # a leg: what it still owes a hand-over
_BASE_STEP = re.compile(r"Post-merge (confirmed|provisional) base ([0-9a-f]{40})")
_SOURCE = ("head", "run", "attempt", "base")  # where, and by which attempt, a record's IDs were seen failing
_MISSING = object()  # no step carries the value
LOST = object()  # a receipt that was never delivered or is gone for good
UNREAD = object()  # a receipt that could not be read now: nothing is known about what it holds
SHORT = object()  # in a run list: the page after it was not a whole listing, so it may lack rows (`listed_runs`)


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


def _carry(replay) -> dict | None:
    """leg -> what a window's replay record carried in; None unless `replay` is that record for every leg."""
    rows = {leg: carried_records(replay, leg) for leg in SYSTEMS}
    return None if None in rows.values() else rows


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
    chosen: tuple | None  # (base, confirmed, attempt) of its window job; None when no attempt recorded one
    searched: bool  # that job read the history behind it whole: a diagnostic, it closes no walk
    awaiting: dict  # leg -> the records this run still leaves owed
    carried: dict | None  # leg -> the records its newest window job carried in; None when that record does not read
    gaps: list  # receipts of this run that are gone for good (`_gap`)
    unread: bool  # a receipt this run's state depends on could not be read now
    read: tuple = ()  # the attempts whose job list was read whole, newest first: each can be measured at no request


def prior_window(run: dict, get, handoff=None, *, first: int | None = None) -> Prior:
    """Read `run`'s attempts, the latest first, as far as its window and both legs need.

    A rerun of failed jobs lists only the legs it reran: the window job and the other leg stay in the earlier
    attempt and are read from there. A leg's latest whole receipt stands for that leg's earlier ones. It also took
    over what the window carried in, unless that window job ran in a later attempt: a re-run of every job carries
    in anew, and a receipt written before it settles none of that. The base is read from the newest window job
    that names one, apart from those records: the replay step runs before the base step, so a job cancelled between
    them carried in and chose no base. Every window job read keeps its own record standing, for each leg until a
    whole receipt of that leg from its attempt or a later one took it over: a newer window that carried in less
    retires nothing, and neither does an older receipt. An older record is asked for only while a leg still waits on
    it; one that does not read is unread or a gap of its own and lends the others nothing. A partial one, a lost
    one or none leaves the earlier attempts and what the window carried in standing beside it, and a leg that
    ran tests without leaving a readable receipt is a gap. Nothing here joins one attempt's leg to another's
    as a measurement: `measured_window` reads one attempt. An earlier attempt's job list that answers 410 is a gap;
    one that cannot be read now (a bare 404, an outage, a broken answer) proves no loss and only leaves what that
    attempt recorded unread. Neither ends the reading: what the attempts after it showed stands, and the attempts
    before it are still read for the window job and the legs that remain open. A job list that is not whole
    leaves what it lacks unread, and the receipts it shows stand: positive evidence, never an absence.

    `first` is the newest attempt to read where the run's own newest one is still open, gone or unread: the caller
    answers for that one, and every attempt read here is an earlier one.
    """
    start = run["run_attempt"] if first is None else first
    chosen, replays, unread = None, [], False  # replays: (attempt, replay step value) of each window job read, newest first
    owed, whole, gaps = {leg: [] for leg in SYSTEMS}, {}, []  # whole: leg -> the attempt of its latest whole receipt
    read = []  # the attempts whose job list was read whole, newest first
    for attempt in range(start, 0, -1):
        jobs = []
        try:
            for page in attempt_job_pages({**run, "run_attempt": attempt}, get, partial=True):
                jobs += page
            read.append(attempt)  # one attempt's jobs, whole: enough to say whether that attempt qualified
        except ValueError:  # the listing is not whole: the jobs it showed stand, what it lacks stays unknown
            unread = True
            print(f"::warning::the job list of run {run['id']}, attempt {attempt} is not whole; "
                  "the receipts it shows stand and what it lacks stays unknown")
        except Exception as error:  # noqa: BLE001 -- no failure below a receipt already read takes it back
            if attempt == run["run_attempt"] and not jobs:
                raise  # nothing of this run is read yet: what its newest attempt's failure means is the caller's
            if isinstance(error, urllib.error.HTTPError) and error.code == 410:
                gaps.append(_gap(run, attempt, "jobs"))  # gone for good: what it recorded is lost
            else:
                unread = True  # a bare 404, an outage or a broken answer proves no loss: unknown, not gone
                print(f"::warning::the job list of run {run['id']}, attempt {attempt} could not be read "
                      f"({_reason(error)}); what it recorded stays unknown")
        for job in jobs:
            if job.get("name") == "window":
                for step in job.get("steps") or [] if chosen is None else ():
                    if _succeeded(step) and (match := _BASE_STEP.fullmatch(step.get("name") or "")):
                        chosen = match[2], match[1] == "confirmed", attempt
                # The replay step runs before the base step: a job cancelled between them carried in and chose no base.
                if (value := _step_value(job, REPLAY_STEP)) is not _MISSING:
                    replays.append((attempt, value))  # read below, and only while a leg still waits on it
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
                        whole[leg] = attempt
                elif value is not _MISSING or _ran(job):
                    gaps.append(_gap(run, attempt, leg, start))  # it measured and left no record of what it owes
        if chosen is not None and len(whole) == len(SYSTEMS):
            break
    # A whole receipt took over what the window job of its own attempt or an earlier one carried in, no later one's.
    # Until then every window's record stands: a newer window that carried in less retires nothing an older one named.
    carried, searched = None, False
    for index, (attempt, value) in enumerate(replays):
        waiting = [leg for leg in SYSTEMS if whole.get(leg, 0) < attempt]  # no whole receipt took this record over
        if index and not waiting:
            continue  # retired on both systems: its artifact is not asked for
        rows = _carry(record := read_handoff(value, handoff, "window", run["id"], attempt=attempt))
        if not index:  # the newest record alone says what this run's window carried in, and how far it searched
            carried, searched = rows, rows is not None and record.get("history_read") is True
        if rows is None and waiting:  # a leg that took nothing over needed it
            if record is UNREAD:
                unread = True
            else:
                gaps.append(_gap(run, attempt, "window", chosen and chosen[0]))
        for leg in waiting if rows else ():
            owed[leg] = merge_records(owed[leg], rows[leg])
    return Prior(chosen, searched, {leg: rows for leg, rows in owed.items() if rows}, carried, gaps, unread,
                 tuple(read))


# --- the window ------------------------------------------------------------------------------

def listed_runs(pages):
    """The runs of a paginated listing (`actions_pages`), newest first. A page that is not whole announces itself with
    `SHORT` before its rows: they are known runs, but others may be missing."""
    for rows, left in pages:
        if left is None:
            yield SHORT
        yield from rows


def _open_rerun(run: dict) -> bool:
    """A range check that is being re-run: its newest attempt has not finished, while the attempts before it have and
    keep their receipts. A first attempt that has not finished has recorded nothing: it is no history yet."""
    return (run.get("status") != "completed" and type(run.get("run_attempt")) is int and run["run_attempt"] > 1
            and range_candidate({**run, "status": "completed", "conclusion": "cancelled"}, include_interrupted=True))


def _skipped_rerun(run: dict) -> bool:
    """A range check re-run into an attempt GitHub concluded `skipped`: it ran nothing and left no record, while the
    attempts before it finished and keep their receipts. A first attempt that was skipped is no history."""
    return (run.get("conclusion") == "skipped" and type(run.get("run_attempt")) is int and run["run_attempt"] > 1
            and range_candidate({**run, "conclusion": "cancelled"}, include_interrupted=True))


def _took_over(run: dict, pages, handoff) -> dict:
    """leg -> the obligations the window job of the exact attempt `run` (its row and its pages) carried in. An attempt
    that measured both systems and left nothing owed took them over itself, and that stays so whatever a newer attempt
    of its run does. No other attempt's window speaks for it, and a record that does not read settles nothing."""
    taken = {leg: set() for leg in SYSTEMS}
    try:
        jobs = [job for page in attempt_job_pages(run, pages) for job in page]
    except Exception:  # noqa: BLE001 -- what is not shown settled stays owed
        return taken
    for job in jobs:
        if job.get("name") == "window":
            record = read_handoff(_step_value(job, REPLAY_STEP), handoff, "window", run["id"],
                                  attempt=run["run_attempt"])
            for leg, rows in (_carry(record) or {}).items():
                taken[leg] |= _obligations(rows)
    return taken


def _qualifying(run: dict, pages, attempts) -> dict:
    """(run, attempt) -> (its row, `pages`) for each of `attempts` of `run` whose own job list, whole in `pages`,
    shows both systems measuring one base and leaving nothing owed. `measured_window` reads one attempt, so no
    attempt's leg is joined to another's. Each is a fact about that exact attempt, whatever a newer one of its run
    does: it took over what its own window carried in (`_took_over`). Once a newer attempt exists it is no longer
    the run's word on its head: it settles those records and qualifies nothing now."""
    found = {}
    for attempt in attempts:
        # Of an attempt the listing no longer shows, the run, the head and the number are the facts, and that it
        # finished follows from its successor. Its conclusion is not known: the value below is a placeholder that no
        # reader of one attempt's jobs uses.
        row = run if attempt == run["run_attempt"] else {
            **run, "run_attempt": attempt, "status": "completed", "conclusion": "failure"}
        try:
            receipt = measured_window(row, pages)
        except Exception:  # noqa: BLE001 -- a job list that does not read whole shows no attempt qualifying
            continue
        if receipt and receipt[1]:
            found[run["id"], attempt] = row, pages
    return found


def _reduce(facts, closed, settlers, *, ancestor, handoff, pending, resolve):
    """(owed, gaps, unread, held, pending, green) from what the runs the walk read recorded. `facts` holds, for each
    run that did not qualify, its row, its head, its `Prior` and its measurement, in the order the walk first met
    them. A run counts by where its head stands to the head that qualified (`closed`, None when none did), never by
    where a listing put the run:

    - before that head: nothing. The range behind a qualified head is closed.
    - at that head: the records seen at that very commit, and neither a base nor a vote. A head is no descendant of
      itself.
    - after it, or with no head qualified: the records it still owes, less what a newer window carried in, and the
      base it kept, the oldest confirmed one winning.

    `settlers` holds every exact attempt known to have qualified its head, by (run, attempt) with its row and pages: a
    fact about that attempt alone, which closes nothing and names no base. It took over what its own window carried in
    (`_took_over`), so another check of the same commit does not owe that record again, whichever of the two the walk
    met first. Against its own run it settles nothing: that run's later receipts speak for themselves. `resolve()`
    reads one more source of such attempts beside `closed` and says whether there was one: an attempt behind a check
    that qualified in a re-run, then a green check set aside. It is asked only while a check of that commit still
    names a record that none of the known attempts took over.
    """
    owed, carried = {leg: [] for leg in SYSTEMS}, {leg: set() for leg in SYSTEMS}
    gaps, unread, held, green, taken = [], False, None, None, {}  # taken: what each settler took over, read once

    def settled(run, sha, leg) -> set:
        """What the exact qualifying attempts of other checks of commit `sha` took over on `leg`."""
        done = set()
        for key, (row, pages) in list(settlers.items()):
            if key[0] != run["id"] and row["head_sha"] == sha:
                if key not in taken:
                    taken[key] = _took_over(row, pages, handoff)
                done |= taken[key][leg]
        return done

    for run, sha, prior, receipt in facts:
        beside = closed is not None and sha == closed
        if closed is not None and not beside and ancestor(sha, closed):
            continue
        awaiting = prior.awaiting
        if beside:
            awaiting = {leg: [row for row in rows if row["head"] == sha] for leg, rows in awaiting.items()}
        gaps += prior.gaps
        unread = unread or prior.unread
        for leg in SYSTEMS:
            rows = _unseen(awaiting.get(leg, ()), carried[leg])
            # What another check's qualifying attempt took over at this commit is not owed again. An attempt not read
            # so far may have done so too: those are read now, one at a time, and only while a record is left.
            rows = rows and _unseen(rows, settled(run, sha, leg))
            while rows and beside and resolve():
                rows = _unseen(rows, settled(run, sha, leg))
            owed[leg] = merge_records(owed[leg], rows)
            carried[leg] |= _obligations((prior.carried or {}).get(leg, ()))
        chosen = prior.chosen if not beside and prior.chosen and ancestor(prior.chosen[0], sha) else None
        if chosen and chosen[1]:
            # Only a qualified descendant closes the walk: this window keeps its base, and the oldest one wins.
            if held is None or ancestor(chosen[0], held[0]):
                held = chosen[0], f"retained starting base of run {run['id']}, attempt {chosen[2]}"
        elif held is None:
            # Before the first confirmed base, keep a known gap even if older history is unreadable.
            if chosen and receipt and receipt[0] == chosen[0] and run["conclusion"] == "success":
                if green is None:
                    green = sha, chosen[0]
                elif not pending and (sha == green[1] or ancestor(green[1], sha)) and ancestor(chosen[0], green[1]):
                    green = green[0], chosen[0]
            elif chosen and (not green or ancestor(chosen[0], green[1])) and (not pending or ancestor(chosen[0], pending)):
                pending = chosen[0]
    return owed, gaps, unread, held, pending, green


def choose_window(runs, head: str, *, get, ancestor, initial: str = "", replay: dict,
                  handoff=None, own: int = 0) -> tuple[str | None, str, bool]:
    """(base, reason, continuous) of the window both legs measure at `head`, the base None when earlier checks
    establish no starting point at all; `replay` receives what the window carries in.

    `runs` is the workflow's history, newest first; `ancestor(older, newer)` answers for the first-parent line.
    The walk reads each run's latest attempt until one qualified its head on both systems:
    that head closes the walk, and nothing behind it in the listing is read. A head is no descendant of itself,
    though: every check of that very commit is then listed once, by branch and commit, wherever the listing put it.
    A run is one observation for each attempt and state: listed again as it was read it costs nothing, a newer
    attempt is read with the pages already fetched and alone decides whether the run still qualifies its head, and
    an unfinished one, or two lists that disagree about one attempt, confirms no range. Two things are kept apart.
    Authority: only a run whose newest state read qualifies the head confirms a range from it. Once a newer attempt
    of that run is listed, whether it is unfinished, qualifies nothing, is gone or cannot be read, the earlier one
    still names the base and confirms nothing. Fact: each exact attempt known to have qualified took over what its
    own window carried in, also one found in the job lists read behind a newer attempt of its run. Another check of
    its commit does not owe that record again, whichever of the two a listing shows first; the attempt closes nothing
    by that and qualifies no newer attempt of its run. What the runs recorded is
    weighed only when the walk has ended (`_reduce`), by where each head stands to the head that qualified and not
    by where a listing put the run: another check of that commit adds what was seen at that head under its own
    source and keeps no base; a check of an older commit read before the head was known to be qualified counts for
    nothing. A green check beside the qualified one is set aside: it is read only while a check of that commit still
    names a record no known attempt took over, and otherwise costs no request. A list
    that is not whole confirms no range. A provisional window closes nothing, however whole it
    read, for a run behind it may have been re-run since. An attempt before it that chose a confirmed base
    and did not qualify keeps that base (the oldest one kept wins, so a late rerun behind a cancelled check
    widens the window to its own base) and adds the records it still owes, each under its own source. The walk
    also ends at a run whose head lies before the oldest base kept: no check reads further back than the range
    it measures. A record a newer attempt's window already carried in is that attempt's to report, and is not
    taken from its older source again. A receipt that is gone for good is a labelled gap beside the records that
    are known (`history_gaps`): this window measures that range again, and its result is its own. History that
    could not be read now leaves the base provisional: `continuous` is False and no leg of this check can qualify.
    A comparison Git could not make while the facts are weighed is such history too: nothing is closed and no base
    is taken on it, so every record and gap already read is carried in, known and not yet resolved.
    A listing page that is not whole (`SHORT`) is such history: its runs still count, none of them confirms a range.
    So is an attempt's job listing that is not whole: the receipts it shows count, what it lacks is never assumed.
    A job list that answers 410 Gone is gone for good; a bare 404 proves no loss, so it is no gap — it only leaves
    what that run recorded unknown, and any base the walk still finds stays unconfirmed. Behind a newest attempt that
    is gone or unread, the earlier attempts of a run that is not green are still read: the base and the records they
    name stand beside that loss, and none of them is a measurement; an outage there then ends the search as before.
    The same holds for a check that is being re-run, an ancestor's or, at a seeded check's head, another run's: its
    finished attempts are read and its open one is history that cannot be read now, so this window confirms no range
    over it. This check's own run (`own`) is never its history; unnamed, an open attempt at this head is taken for it.
    A check whose first attempt has not finished has recorded nothing: no request, no change.

    A re-run GitHub concluded `skipped` ran nothing: it qualifies nothing, and the attempts before it are read for
    the base and the records they name, unless the one just before it was already read; a first attempt that was
    skipped costs no request. A check that qualifies in a re-run of its legs lists no window job there: while a check
    of its commit still names a record, its earlier attempts are read, each by its own job list, and one that itself
    qualified settled what its own window carried in. Where no check read so far qualifies the head in its newest
    attempt any more, the green checks set aside beside it are read for theirs: one that qualifies by its own whole
    job list confirms the range, with or without a record left, and a list that is not whole or cannot be read
    confirms nothing.

    A push run without the range title predates these receipts, and so does everything older: the search ends
    there. Before the first confirmed base a green measurement settles its own window and an unmeasured or red
    one that starts before it stays a known gap; only an explicit `initial` base starts a confirmed range, for
    the checks that follow it, and no provisional window behind a confirmed one widens it.
    """
    readable = runs is not None
    boundary, reach, gaps = None, None, []  # the head that qualified; the oldest base an unqualified run kept
    # Collected by run, and weighed only when the walk has ended (`_reduce`): `seen` is the attempt and state each run
    # was read in, with its pages; `facts` what a run recorded in the newest state read, unless that attempt qualified.
    # Authority and fact are kept apart. `standing` names the runs whose newest state read qualifies the head: the
    # only ground for a confirmed range, and reading a newer state of a run takes it back before anything else.
    # `settlers` holds every exact (run, attempt) known to have qualified, with its pages: no newer attempt takes
    # that back, and it settles what that attempt's own window carried in, nothing more. `greens` are the green
    # checks set aside beside the head, each with its pages (`reaffirm`, `resolve`); `behind` holds, for a run that
    # qualified in a re-run, the next earlier attempt of it not yet read (`resolve`).
    seen, facts, standing, settlers, greens, behind = {}, {}, {}, {}, [], {}
    unsure = []  # the comparisons Git could not make while the facts were weighed

    def before(older, newer):
        """`ancestor` for the weighing: a comparison Git cannot make (a timeout, no process) is no answer. Nothing is
        closed and no base is taken on it, so no record already read is dropped, and no range is confirmed."""
        try:
            return ancestor(older, newer)
        except Exception as error:  # noqa: BLE001 -- no failure takes back a record already read
            if not unsure:
                print(f"::warning title=Post-merge history unread::Git could not compare {older[:12]} with "
                      f"{newer[:12]} ({_reason(error)}); no record already read is dropped and no range is confirmed")
            unsure.append(error)
            return False

    def qualify(run, pages) -> str:
        """Record `run`, in the attempt read, as having measured both systems and left nothing owed. Authority: the
        newest state read of its run qualifies the head, until a newer one is read. Fact: the exact attempt stays
        among `settlers`. A re-run's earlier attempts are not read here: `behind` keeps the next one for `resolve`."""
        standing[run["id"]] = (f"eligible on both systems (run {run['id']}, attempt "
                               f"{run['run_attempt']}, {run['conclusion']})")
        settlers[run["id"], run["run_attempt"]] = run, pages
        if run["run_attempt"] > 1:
            behind[run["id"]] = run, pages, run["run_attempt"] - 1
        return standing[run["id"]]

    def resolve() -> bool:
        """Read one more source of settlements beside the head that qualified, and say whether there was one. Asked
        only while a check of that commit still names a record no known attempt took over. First the attempts behind
        a check that qualified in a re-run, the newest first: a re-run of legs lists no window job, and the attempt
        before it may have qualified with one. Each is read by its own job list, so no leg of one attempt is joined
        to another's, and a list that does not read whole settles nothing. Then a green check set aside. Each exact
        attempt that qualified is added to `settlers`, and that is all this adds: nothing here qualifies a head, and
        what cannot be read settles nothing and takes back no confirmation."""
        if behind:
            number = next(iter(behind))
            run, pages, attempt = behind.pop(number)
            if attempt > 1:
                behind[number] = run, pages, attempt - 1
            found = _qualifying(run, pages, (attempt,))
        elif greens:
            run, pages = greens.pop(0)
            try:
                found = _qualifying(run, pages, prior_window(run, pages, handoff).read)
            except Exception as error:  # noqa: BLE001 -- what is not shown settled stays owed
                print(f"::warning::run {run['id']} could not be read ({_reason(error)}); what it took over stays "
                      "unknown, and the records other checks of its commit name stay owed")
                found = {}
        else:
            return False
        for key, value in found.items():
            settlers.setdefault(key, value)
        return True

    def reaffirm():
        """No check read so far qualifies the head that closed the walk in its newest attempt any more. A green check
        set aside beside it was never read, and the attempt listed is its newest: one that measured both systems by
        its own whole job list qualifies the head, independently of the run that was re-run and whether or not a
        record is left. A job list that is not whole or cannot be read qualifies nothing. A green check that read
        whole without qualifying stays set aside for `resolve`, its pages kept."""
        kept = []
        while greens and not standing:
            run, pages = greens.pop(0)
            try:
                receipt = measured_window(run, pages)
            except Exception as error:  # noqa: BLE001 -- a check that does not read whole confirms no range
                print(f"::warning::run {run['id']} could not be read whole ({_reason(error)}); it confirms no range")
                continue
            if receipt and receipt[1]:
                qualify(run, pages)
            else:
                kept.append((run, pages))
        greens[:0] = kept

    def history():
        """The listing down to the qualified head, then, asked for once, every check of that very commit."""
        for row in runs:
            yield row
            if boundary:
                yield from listed_runs(actions_pages(
                    get, f"{POST_MERGE_RUNS}?branch={BRANCH}&head_sha={boundary[0]}", "workflow_runs"))
                return

    try:
        for run in history() if readable else ():
            if run is SHORT:
                readable = False  # the page after it may lack rows: the runs it lists count, none confirms a range
                print("::warning title=Post-merge history unread::a run-list page was not whole, "
                      "so the base stays unconfirmed")
                continue
            if not isinstance(run.get("display_title"), str) or (run["display_title"] == RANGE_TITLE and
                    not {"head_branch", "event", "head_sha", "status", "conclusion", "id", "run_attempt"} <= run.keys()):
                readable = False  # it may hide an attempt that left a red owed
                continue
            if run["display_title"] != RANGE_TITLE and run.get("head_branch") == BRANCH and run.get("event") == "push":
                break
            sha = run.get("head_sha") or ""
            if boundary and sha != boundary[0]:
                raise ValueError("the list of one commit's checks names another commit")
            rerun = _open_rerun(run)  # its newest attempt has not finished; the attempts before it have
            now, met = (run.get("run_attempt"), run.get("status") == "completed"), seen.get(run.get("id"))
            if met and not (type(now[0]) is int and now > met[0]):
                # One attempt in one state is one observation, wherever it is listed again: only a newer one is read.
                if now[0] == met[0][0] and now != met[0]:
                    readable = False  # the two lists disagree on whether that attempt has finished
                    print(f"::warning title=Post-merge history unread::run {run['id']}, attempt {now[0]} is listed "
                          "both finished and unfinished, so the base stays unconfirmed")
                continue
            # A re-run GitHub concluded `skipped` ran nothing, and the attempts before it keep their receipts: they
            # are read behind it, unless the one just before it was already read, finished.
            skipped = not rerun and _skipped_rerun(run) and not (met and met[0] == (run["run_attempt"] - 1, True))
            if not rerun and not skipped and not range_candidate(run, include_interrupted=True):
                # A newer state of a run already read is its word on its head even where this walk reads no such
                # attempt: GitHub concludes a finished run `skipped`, which measured nothing. It qualifies nothing
                # and settles nothing of its own, but the attempt that qualified before it confirms no range now.
                if met:
                    standing.pop(run["id"], None)
                continue
            if boundary and not met and run.get("conclusion") == "success":
                # Beside a qualified check, a green check of the same commit that was never read owes nothing. It is
                # set aside at no request, and read only if a check there still names a record (`resolve`).
                greens.append((run, cache(get)))
                continue
            if run["id"] == own or (sha == head and (not initial or rerun and not own)) or (
                    sha != head and not ancestor(sha, head)):
                continue  # Its own run is no history; unnamed, an open attempt at this head may be its own.
            if reach and ancestor(sha, reach):
                break  # It ended before the base this window keeps: that range closed when the base was confirmed.
            # Both readers stop at their witness and share each attempt page already fetched. A run listed again in a
            # newer state keeps its pages: no attempt already read is asked for twice.
            jobs = met[1] if met else cache(get)
            seen[run["id"]] = now, jobs
            # This state is the run's word now. Whatever it turns out to be, no earlier attempt of the run qualifies the
            # head any longer: that attempt stays among `settlers`, a fact about what it took over.
            standing.pop(run["id"], None)
            receipt, first, fault = None, run["run_attempt"], None  # `first`: its newest attempt that can be read
            if rerun:
                first, readable = first - 1, False  # the open attempt's record is unread: it confirms no range
                print(f"::warning title=Post-merge history unread::run {run['id']} is being re-run and attempt "
                      f"{run['run_attempt']} has not finished; what its earlier attempts recorded stands, the base "
                      "unconfirmed")
            if skipped:
                first -= 1  # that attempt left no record to read: the attempts before it are the run's history
            try:
                try:
                    receipt = None if rerun else measured_window(run, jobs)
                except ValueError:  # a listing that is not whole joins no legs; `prior_window` still reads what it shows
                    receipt = None
                if receipt and receipt[1] and sha != head:
                    # This exact attempt measured both systems and left nothing owed. As the newest state of its run
                    # it qualifies the head now; as an attempt it is a fact no newer one takes back. Its pages stay:
                    # what its own window carried in is read from them when a sibling still names a record.
                    why = qualify(run, jobs)
                    boundary = boundary or (sha, why)
                    facts.pop(run["id"], None)
                    continue
                prior = prior_window(run, jobs, handoff, first=first)
            except Exception as error:  # noqa: BLE001 -- the job list of its newest attempt gave nothing to read
                code = error.code if isinstance(error, urllib.error.HTTPError) else None
                if code == 410:
                    if run.get("conclusion") != "success":
                        gaps.append(_gap(run, run["run_attempt"], "jobs"))  # gone for good
                elif code != 404:
                    fault = error  # an outage or a broken answer still ends the search, below
                else:
                    readable = False  # a bare 404 proves no loss: what this run recorded is unknown
                    print(f"::warning title=Post-merge history unread::the job list of run {run['id']}, attempt "
                          f"{run['run_attempt']} answered HTTP 404; its records stay unknown, the base unconfirmed")
                first -= 1
                if not first or run.get("conclusion") == "success":
                    if fault is not None:
                        raise
                    continue  # An older base is reused; nothing unread counts as settled. A green run owes nothing.
                # The attempts behind it keep exact endpoints: the base and the records they name stand beside the loss.
                prior = prior_window(run, jobs, handoff, first=first)
            facts[run["id"]] = run, sha, prior, receipt  # a newer state of a run replaces what was read of it before
            # An attempt behind the newest one that itself qualified is found in the job lists just read, at no
            # request: a fact about what it took over, which makes nothing of this run qualify now.
            for key, value in _qualifying(run, jobs, prior.read).items():
                settlers.setdefault(key, value)
            if not boundary and prior.chosen and prior.chosen[1] and ancestor(prior.chosen[0], sha) and (
                    reach is None or ancestor(prior.chosen[0], reach)):
                reach = prior.chosen[0]  # the oldest base an unqualified run kept: the walk reads no run before it
            if fault is not None:
                raise fault  # the search ends where the outage did, as before, with what the finished attempts name
    except Exception as error:  # noqa: BLE001 -- keep any earlier starting point already observed
        print(f"::warning title=Post-merge base::earlier checks are unreadable ({_reason(error)}); falling back")
        readable = False
    if boundary and not standing and readable:
        reaffirm()  # authority, read apart from any record: a green check of that commit may still qualify it
    try:
        owed, lost, unread, held, pending, green = _reduce(
            facts.values(), boundary and boundary[0], settlers, ancestor=before, handoff=handoff,
            pending=initial, resolve=resolve)
    except Exception as error:  # noqa: BLE001 -- what could not be weighed names no base and takes back no record
        print(f"::warning title=Post-merge base::earlier checks could not be weighed ({_reason(error)}); every record "
              "and gap already read is carried in unweighed")
        owed = {leg: merge_records(*(prior.awaiting.get(leg, ()) for _, _, prior, _ in facts.values()))
                for leg in SYSTEMS}
        lost = [gap for _, _, prior, _ in facts.values() for gap in prior.gaps]
        unread, held, pending, green = True, None, initial, None
    point = boundary or held
    if boundary:
        # Only a run whose newest state read qualifies the head confirms a range from it. An attempt a newer one has
        # replaced still names the base, and what it took over stays settled.
        point = boundary[0], next(iter(standing.values()), boundary[1])
        if not standing:
            readable = False
            print(f"::warning title=Post-merge history unread::no check of {boundary[0][:12]} qualifies it in its "
                  "newest attempt any more, so the base stays unconfirmed; what the attempt that did took over "
                  "stays settled")
    if point:
        if boundary and held and before(held[0], boundary[0]):
            point = held
        if pending and before(pending, point[0]):
            point = pending, "widened first-parent window"
    # Asked last: a comparison Git could not make, here or in the weighing, confirms no range.
    readable = readable and not unread and not unsure
    replay.update(history_read=readable, awaiting_handoff={leg: rows for leg, rows in owed.items() if rows},
                  history_gaps=gaps + lost)
    if point:
        return point[0], point[1] if readable else point[1] + "; earlier history unreadable", readable
    if initial:
        return pending, "explicit first-parent window" if readable else "explicit range; earlier history unreadable", readable
    if pending:
        return pending, "earlier range still pending", False
    if green:
        return green[0], "head of the last green check (earlier coverage remains pending)", False
    return None, "no earlier check establishes a starting point", False
