"""Portable merge qualification and exact-range GitHub observations.

The commit trailer is evidence, never approval. Author declarations remain labelled;
missing executions stay unknown. Fetching checks consult GitHub, passive readers do not.
No prose is classified: older commit bodies reach the description's author unchanged, and
a refreshed run never rewrites a description of the same range.
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from itertools import count

TRAILER = "Ouroboros-Qualification: "
CHECKS = ("full_battery", "cost_contract")
STATES = ("passed", "failed", "not_run", "unknown")
RED_LABEL = "post-merge-red"  # scripts/post_merge_routing.py labels each red it hands over with it
POST_MERGE_RUNS = "actions/workflows/post-merge.yml/runs"
RANGE_TITLE = "Post-merge range check"
MEASURED_STEP = "Post-merge range measured"
CONTINUOUS_STEP = "Post-merge continuous range measured"
# A leg job's name: (system, the base it measured from). The one grammar of post-merge.yml's job name, for this
# reader and scripts/post_merge_window.py's.
_LEG = re.compile(r"post-merge \((linux|windows)\) from ([0-9a-f]{40})")
REVIEW_STATES = ("settled", "pending", "unknown", "no_record")
REVIEW_RELATIONS = ("covers_head", "changes_after_review", "unknown", "not_declared", "delta_only")


def merge_qualification(review: dict, checks: list | None = None, *, relation: str = "") -> dict:
    """The trailer's facts. ``relation`` is the subject relation the merge request saw (its coverage
    status); with it the observed verdict keeps its record's state. Without it neither is recorded."""
    declared = {row["name"]: {**row, "source": "author_declared"} for row in checks or []}
    for name in CHECKS:
        declared.setdefault(name, {"name": name, "status": "unknown", "source": "missing",
                                   "detail": "No execution evidence was supplied."})
    record = review.get("record") or {}
    seen = {"observed": (record.get("verdict") or {}).get("aggregate", "unknown")}
    if relation:
        seen.update(state=str(record.get("state") or "unknown") if record else "no_record", relation=relation)
    seen["declared"] = (review.get("declared") or {}).get("verdict", "unknown")
    return {"schema": 1, "checks": list(declared.values()),
            "review": seen,
            "post_merge": {"status": "not_run", "detail": "The post-merge check follows this merge."}}


def _review_line(review: dict) -> str:
    """The observed verdict beside its record state and the relation the merge request saw; a receipt
    written before both were carried says so instead of inventing them."""
    if "state" in review or "relation" in review:
        seen = (f"record {review.get('state', 'not recorded')}, subject relation at the merge request "
                f"{review.get('relation', 'not recorded')}")
    else:
        seen = "record state and subject relation at the merge request not recorded"
    return f"- Review: observed {review['observed']} ({seen}); author declared {review['declared']}"


def commit_summary(qualification: dict) -> str:
    lines = ["Ouroboros qualification receipt (declarations are not host-observed PASS):"]
    for row in qualification["checks"]:
        lines.append(f"- {row['name']}: {row['status']} ({row['source']}); {row.get('detail') or 'no detail supplied'}")
    lines += [_review_line(qualification["review"]),
              "- Post-merge check: not run yet.", TRAILER + json.dumps(qualification, ensure_ascii=False)]
    return "\n".join(lines)


def commit_qualification(body: str) -> dict | None:
    for line in reversed(body.splitlines()):
        if line.startswith(TRAILER):
            try:
                value = json.loads(line[len(TRAILER):])
                if (isinstance(value, dict) and value.get("schema") == 1 and isinstance(value.get("checks"), list)
                        and isinstance(value.get("review", {}), dict) and all(
                            isinstance(row, dict) and isinstance(row.get("name"), str)
                            and row.get("status") in STATES and isinstance(row.get("detail", ""), str)
                            for row in value["checks"])):
                    return value
            except ValueError:
                pass
    return None


def _github_repo(repo_url: str) -> str | None:
    parsed = urllib.parse.urlparse(repo_url)
    parts = parsed.path.removesuffix(".git").strip("/").split("/")
    return "/".join(parts) if parsed.hostname == "github.com" and len(parts) == 2 else None


def _get_json(url: str):
    request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json",
                                                  "User-Agent": "Ouroboros-update-check"})
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.load(response)


def actions_pages(get, path: str, key: str):
    """GitHub's named array inside a paginated Actions response. Stop fetching when the reader stops."""
    seen, first_total, ids = 0, None, set()
    for page in count(1):
        data = get(f"{path}{'&' if '?' in path else '?'}per_page=100&page={page}")
        rows, total = data[key], data["total_count"]
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows) or type(total) is not int:
            raise ValueError(f"invalid {key} page")
        seen += len(rows)
        if first_total is None:
            first_total = total
        left = total - seen
        page_ids = [row["id"] for row in rows if type(row.get("id")) is int]
        unique = not ids.intersection(page_ids) and len(set(page_ids)) == len(page_ids)
        ids.update(page_ids)
        exact = unique and total == first_total and left >= 0 and len(rows) <= 100 and (not left or len(rows) == 100)
        # Positive rows from an incomplete listing survive. Only an exhaustive page can prove absence.
        yield rows, left if exact else None
        if not exact:
            raise ValueError(f"incomplete {key} listing")
        if not left:
            return


def range_candidate(run: dict, *, include_interrupted=False) -> bool:
    return (run.get("display_title") == RANGE_TITLE and run.get("head_branch") == "ouroboros"
            and run.get("event") in ("push", "workflow_dispatch") and run.get("status") == "completed"
            and (run.get("conclusion") in ("success", "failure") or
                 (include_interrupted and run.get("conclusion") in ("cancelled", "timed_out")))
            and re.fullmatch(r"[0-9a-f]{40}", run.get("head_sha") or "") is not None
            and all(type(run.get(key)) is int and run[key] > 0 for key in ("id", "run_attempt")))


def attempt_job_pages(run: dict, get, *, partial: bool = False):
    """Pages of completed jobs belonging to this head, run and attempt.

    A listing that is not whole raises ValueError before its page is yielded, so no reader takes a joint witness
    from it. `partial` yields that page's rows first and raises after them: positive evidence for a reader that
    never takes an absence from it (scripts/post_merge_window.py `prior_window`)."""
    number, attempt = run["id"], run["run_attempt"]
    for rows, left in actions_pages(get, f"actions/runs/{number}/attempts/{attempt}/jobs", "jobs"):
        if left is None and not partial:
            raise ValueError("incomplete jobs listing")
        yield [job for job in rows if (job.get("run_id"), job.get("run_attempt", attempt), job.get("head_sha"),
                                       job.get("status")) == (number, attempt, run["head_sha"], "completed")]


def measured_window(run: dict, get) -> tuple[str, bool] | None:
    """(measured base, eligible head) from both legs. The window job only chose their input."""
    if not range_candidate(run):
        return None
    legs = {}
    for jobs in attempt_job_pages(run, get):
        for job in jobs:
            match = _LEG.fullmatch(job.get("name") or "")
            if not match:
                continue
            system, base = match.groups()
            if system in legs or base in ("0" * 40, run["head_sha"]):
                return None
            markers = [step["name"] for step in job.get("steps") or [] if isinstance(step, dict)
                       and step.get("name") in (CONTINUOUS_STEP, MEASURED_STEP) and step.get("status") == "completed"
                       and step.get("conclusion") == "success"]
            if len(markers) != 1:
                return None
            legs[system] = (base, markers[0] == CONTINUOUS_STEP)
        if len(legs) == 2:
            if legs["linux"][0] == legs["windows"][0]:
                return legs["linux"][0], all(eligible for _, eligible in legs.values())
            return None
    return None


def _interrupted(run: dict) -> bool:
    return run.get("status") == "completed" and run.get("conclusion") in ("cancelled", "skipped")


def _missing_runs(commits: list[dict], runs: list[dict], absent: set[str], base_sha="", lookup=None, *,
                  full_line=False, complete_run_lists=(), incomplete_attempts=(), previous=None) -> list[str]:
    """Cover proven absences or exclusively interrupted runs in a measured (base, head].

    For a SHA with any direct rows, both its run listing and those runs' attempts must be read
    before concluding that every run was interrupted. A completed success or failure needs no cover.
    """
    # Snapshot the old witness before clearing derived fields: on a passive read these are the same rows.
    previous = previous or {}
    edges = {_run_key(r): (r['measured_base'], tuple(r['covered_shas'])) for r in previous.get('runs', ())
             if isinstance(r, dict) and _terminal_attempt(r) and r.get('measured_base') and r.get('covered_shas')}
    own = {}
    for run in runs:  # newest first, as read
        run.pop("covered_shas", None)
        own.setdefault(run.get("sha"), []).append(run)
    positions = {c["sha"]: i for i, c in enumerate(commits)}
    if base_sha:
        positions.setdefault(base_sha, len(commits))
    pending = []
    for commit in reversed(commits):
        sha = commit["sha"]
        done = [run for run in own.get(sha, ()) if run.get("status") == "completed"
                and run.get("conclusion") in ("success", "failure")]
        if sha in absent or (sha in complete_run_lists and sha in own and sha not in incomplete_attempts
                             and all(_interrupted(run) for run in own[sha])):
            pending.append(sha)
        elif pending and done:
            # Spend requests only on unresolved gaps. A later unmeasured run cannot disqualify an earlier one.
            for run in sorted(done, key=lambda run: "measured_base" not in run):
                if not pending:
                    break
                if lookup and "measured_base" not in run:
                    lookup(run)
                base = run.get("measured_base")
                # Both endpoints are first parents. A base outside a complete local suffix must predate it.
                end = positions.get(base, len(commits) if base and full_line else 0)
                while pending and positions[pending[-1]] < end:
                    run.setdefault("covered_shas", []).append(pending.pop())
    # A short read may retain an actual edge, not issue a fresh verdict from an old SHA-level fact.
    # A fresh exhaustive listing above owns its SHA, including a fresh empty result.
    keep = set(previous.get('absent_shas') or ()).difference(own, complete_run_lists)
    keep.update(sha for sha in previous.get('interrupted_shas') or () if sha in own
                and sha not in complete_run_lists and sha not in incomplete_attempts
                and all(_interrupted(row) for row in own[sha]))
    if keep and edges:
        for run in runs:
            edge = edges.get(_run_key(run))
            if (not edge or run.get('measured_base') != edge[0] or run.get('status') != 'completed'
                    or run.get('conclusion') not in ('success', 'failure') or run['sha'] not in positions):
                continue
            end = positions.get(edge[0], len(commits) if full_line else 0)
            kept = {sha for sha in edge[1] if sha in keep and sha in positions
                    and positions[run['sha']] < positions[sha] < end}
            if kept:
                run['covered_shas'] = sorted(kept.union(run.get('covered_shas', ())), key=positions.__getitem__)
    return sorted({c["sha"] for c in commits} - own.keys() - {
        sha for run in runs for sha in run.get("covered_shas", ())})


def discard_legacy_coverage(record) -> bool:
    """Project a saved update letter as every Updates reader shows it (the panel, the preflight, query_data): coverage
    saved before CI published measured bases leaves its qualification and its last good one, including during an
    outage. True when that changed the record."""
    changed = False
    for stored in (record, record.get("last_good")) if isinstance(record, dict) else ():
        qualification = stored.get("qualification") if isinstance(stored, dict) else None
        github = qualification.get("github") if isinstance(qualification, dict) else None
        runs = [r for r in github["runs"] if isinstance(r, dict) and "sha" in r] if isinstance(github, dict) and isinstance(
            github.get("runs"), list) else []
        if not any(r.get('covered_shas') for r in runs):
            continue
        interrupted = set(github.get('interrupted_shas') or ())
        own = {r['sha'] for r in runs}
        holes = _attempt_holes(runs)  # the same saved rows can disclose an unread attempt below a run's horizon
        supported = set(github.get('absent_shas') or ()).difference(own)
        supported.update(interrupted.intersection(own).difference(r['sha'] for r in runs if not _interrupted(r))
                         .difference(holes))
        if any(r.get('covered_shas') and (not r.get('measured_base') or not supported.issuperset(r['covered_shas']))
               for r in runs):
            github['missing_shas'] = _missing_runs(
                qualification.get('commits') or [], runs, set(), qualification.get('base_sha'),
                full_line=qualification.get('first_parent_complete') is True, previous=github,
                incomplete_attempts=holes)
            if 'interrupted_shas' in github:
                github['interrupted_shas'] = sorted({s for r in runs for s in r.get('covered_shas', ()) if s in own})
            changed = True
    return changed


def _date(value) -> datetime | None:
    try:
        date = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return date if date.tzinfo is not None else None
    except ValueError:
        return None


class _Budget(ValueError):
    """The read's request budget ran out: what it left unread stays unknown, never absent."""


_BUDGET = "GitHub response page budget reached."
_UNREAD_ATTEMPT = "An attempt below a run's highest known one was never read."


class _NoReader(ValueError):
    """The update source has no GitHub check reader: no request is sent, so nothing is observed and nothing erased."""


_NO_READER = "The update source has no supported GitHub check reader."


def _run_key(run: dict) -> tuple:
    return run.get("sha"), run.get("run_id"), run.get("attempt")


def _terminal_attempt(run):
    return (type(run.get('run_id')) is int and run['run_id'] > 0
            and type(run.get('attempt')) is int and run['attempt'] > 0
            and run.get('status') == 'completed' and isinstance(run.get('conclusion'), str) and bool(run['conclusion']))


def _attempt_holes(runs) -> set:
    """SHAs with an attempt below a run's highest known one that no row supplies.

    Attempts are contiguous, so grouped positive attempt counts decide completeness: a count below
    the maximum is an unread attempt, never an attempt that never existed. Counts only — no walk
    over the horizon, no request, and the exact attempt rows are only read."""
    known = {}
    for run in runs:
        number, attempt = run.get("run_id"), run.get("attempt")
        if type(number) is int and number > 0 and type(attempt) is int and attempt > 0:
            known.setdefault((run.get("sha"), number), set()).add(attempt)
    return {sha for (sha, _), attempts in known.items() if len(attempts) != max(attempts)}


def _read_earlier_attempts(runs, sources, prior, observe, get):
    """The workflow list reports the latest attempt only; older attempts have exact endpoints.

    A run is re-run only once it has settled, so an attempt below the newest one seen is settled and a running row of
    it that a listing showed first is stale. Only a settled row of that exact SHA, run and attempt stands for it: the
    one read now, the saved one, or else its own endpoint. An attempt none supplies stays unread, never observed."""
    incomplete, unread = set(), []
    for (sha, number, newest), source in tuple(sources.items()):
        if type(number) is not int or number <= 0 or type(newest) is not int or newest <= 1:
            continue
        for attempt in range(newest - 1, 0, -1):
            key = sha, number, attempt
            if key in runs and (_terminal_attempt(runs[key]) or _terminal_attempt(prior.get(key, {}))):
                continue  # settled: no request; the join stands a saved settled row and its status over a running one
            cached = prior.get(key)
            if cached and _terminal_attempt(cached):
                runs[key] = dict(cached)
                # The listing binds the workflow identity; only this exact cached attempt supplies its outcome.
                sources[key] = {**source, "run_attempt": attempt, "status": cached["status"],
                                "conclusion": cached["conclusion"]}
                continue
            try:
                row = get(f"actions/runs/{number}/attempts/{attempt}")
                if (not isinstance(row, dict) or type(row.get("id")) is not int
                        or type(row.get("run_attempt")) is not int
                        or (row["id"], row["run_attempt"], row.get("head_sha")) != (number, attempt, sha)
                        or row.get("head_branch") != source.get("head_branch", "ouroboros")
                        or row.get("event") != source.get("event", "push")
                        or ("workflow_id" in source and row.get("workflow_id") != source["workflow_id"])
                        or row.get("status") != "completed" or not isinstance(row.get("conclusion"), str)
                        or not row["conclusion"]):
                    raise ValueError("workflow attempt does not match the requested completed attempt")
                observe(row)
            except Exception as exc:
                incomplete.add(sha)
                unread.append(exc)
                if isinstance(exc, _Budget):
                    break
    return incomplete, unread


def _join_saved_attempts(runs, sources, current_keys, prior, wanted, complete_run_lists, incomplete_attempts):
    """Join the saved record's attempts to this read's by exact identity (SHA, run, attempt), in one pass.

    Observed by both: a saved settled row replaces this read's unsettled one (a settled attempt does not run again),
    and the settled status is bound to the attempt's own raw source row, so its jobs can be looked up; it stays retained.
    Saved only: a red row stays; so does a settled one while the read was short, and any row while its run was not seen
    and its SHA's listing was not whole (a whole listing without the run owns a non-red run's disappearance). A row of
    a run this read saw at other attempts is settled by nothing here and stays as last observed: an unsettled one (a
    newer attempt followed it, or the listing lags it) and a settled one above every attempt listed. The highest
    attempt known of a run is an identity horizon, never freshness: an unsettled row below it is an old observation,
    its outcome unknown, never a job running now."""
    again = {key[:2] for key in current_keys}  # the runs this read itself observed
    top = {}  # and the highest attempt it observed of each
    for sha, number, attempt in current_keys:
        if type(number) is int and type(attempt) is int:
            top[sha, number] = max(attempt, top.get((sha, number), 0))
    for key, run in prior.items():
        sha, number = run.get('sha'), run.get('run_id')
        if sha not in wanted:
            continue
        if key in runs:
            if _terminal_attempt(run) and not _terminal_attempt(runs[key]):
                runs[key] = dict(run)  # the same attempt, saved settled, is not taken back by a stale running row
                sources[key] = {**sources.get(key, {}), 'status': run['status'], 'conclusion': run['conclusion']}
                current_keys.discard(key)  # so it is the saved row that stays, marked retained
            continue
        short_history = sha not in complete_run_lists or sha in incomplete_attempts
        aside = (type(number) is int and type(run.get('attempt')) is int and (sha, number) in top
                 and (not _terminal_attempt(run) or run['attempt'] > top[sha, number]))
        if ((number is not None and run.get('conclusion') == 'failure') or (short_history and _terminal_attempt(run))
                or aside or sha not in complete_run_lists and ((sha, number) not in again if number is not None
                                                               else all(seen[0] != sha for seen in again))):
            runs[key] = dict(run)
    newest = {}  # the highest attempt known of each run, observed here or saved
    for sha, number, attempt in runs:
        if type(number) is int and type(attempt) is int:
            newest[sha, number] = max(attempt, newest.get((sha, number), 0))
    current_keys.difference_update({key for key in current_keys if not _terminal_attempt(runs[key])
                                    and newest.get(key[:2], 0) > (key[2] if type(key[2]) is int else 0)})


def post_merge_checks(commits: list[dict] | dict, repo_url: str, *, get_json=None) -> dict:
    """Read runs for first-parent integrations, newest first, including squash.

    Read the global listing to its end, or finish exact SHA listings for the whole range when cheaper.
    Dates only choose the cheaper route. Neither a date nor one direct run proves a SHA's history complete.
    A red read before stays by its run and attempt, and an attempt read as settled is never read back as running.
    A saved attempt this read did not observe stays as last observed (same link), marked retained, when its run
    was seen at other attempts, whether a newer one followed it or the listing lags it: if it had not settled, its
    outcome is unknown.
    The same ten-request budget includes older attempts;
    a short listing for a SHA or an unread attempt never supplies a new indirect verdict for that SHA.
    Public GitHub needs no credential. Offline/rate-limit failures remain unknown. A source without a GitHub
    reader sends no request and ends at the same join as a failed read: what an earlier read observed stays,
    marked retained, and the read is unavailable, never observed. The join is one repository's: the record names
    the repository it was read from (`repository`), and the same range read from another known repository keeps
    none of its runs, absences or covers. A record saved without the name (before it was kept, or by a source with no
    reader) is no known repository's: while a read supplies no current run and no whole listing it stays as it was,
    unnamed and retained; the first read that does names it and keeps only its own observations. The unnamed ones leave
    once, for want of an origin, never as a report that GitHub removed them.
    """
    # A retained qualification also supplies the range boundary, immutable earlier attempts and what was read.
    material = commits if isinstance(commits, dict) else {"commits": commits}
    commits, base_sha = material["commits"], material.get("base_sha", "")
    if not commits:
        return {"status": "observed", "runs": [], "missing_shas": []}
    get_json = get_json or _get_json
    repo = _github_repo(repo_url)  # None: no reader. Nothing returns early: the join below owns what is kept.
    wanted = {c["sha"] for c in commits}
    runs, sources, current_keys = {}, {}, set()  # exact attempts; cached outcomes never enter current_keys
    complete_run_lists = set()
    earlier = material.get("github") or {}
    # A row joins by SHA, run and attempt, which name something in one repository only: what another known repository
    # observed joins nothing here, and a known repository's read sets an unnamed record aside; the record stands again
    # only while that read supplies nothing (below). A source without a reader is no change of repository.
    origin, unnamed = earlier.get("repository"), {}
    if repo is not None:
        if not isinstance(origin, str):
            earlier, unnamed = {}, earlier
        elif origin.lower() != repo.lower():
            earlier = {}
        origin = repo
    prior = {_run_key(r): r for r in earlier.get("runs", []) if isinstance(r, dict)}
    remaining = 10

    def get(path):
        nonlocal remaining
        if repo is None:  # every request passes here, so none is sent
            raise _NoReader(_NO_READER)
        if not remaining:
            raise _Budget(_BUDGET)
        remaining -= 1
        return get_json(f"https://api.github.com/repos/{repo}/{path}")

    def observe(row):
        sha = row.get("head_sha")
        if not isinstance(sha, str) or not sha:
            raise ValueError("workflow run without a head SHA")
        if row.get("head_branch", "ouroboros") != "ouroboros" or row.get("event", "push") not in ("push", "workflow_dispatch"):
            return
        run = {"sha": sha, "status": row.get("status"), "conclusion": row.get("conclusion"),
               "url": row.get("html_url"), "run_id": row.get("id"), "attempt": row.get("run_attempt")}
        key = _run_key(run)
        # A settled attempt does not run again: a nonterminal row read first (the global list) gives way to the terminal
        # row of the same SHA, run and attempt (an exact lookup), which becomes its range proof. Two terminal rows are
        # not ranked: the first read stays.
        if sha in wanted and (key not in runs or (_terminal_attempt(run) and not _terminal_attempt(runs[key]))):
            current_keys.add(key)
            runs[key], sources[key] = run, row
            cached = prior.get(key, {})
            if range_candidate(row) and "measured_base" in cached:
                run["measured_base"] = cached["measured_base"]

    status, reason = "observed", "No matching run is not a passing check."
    try:
        oldest = _date(commits[-1].get("date"))
        for rows, left in actions_pages(get, POST_MERGE_RUNS, "workflow_runs"):
            for row in rows:
                observe(row)
            if left == 0:
                complete_run_lists = wanted
                break
            if not remaining:
                status, reason = "partial", _BUDGET
                break
            # A late dispatch can put a SHA on this page while its push is still on an older page.
            if left and oldest and any((created := _date(row.get("created_at"))) and created < oldest for row in rows):
                if len(wanted) > min(remaining, (left + 99) // 100):
                    continue
                for sha in sorted(wanted):
                    path = f"{POST_MERGE_RUNS}?branch=ouroboros&head_sha={urllib.parse.quote(sha, safe='')}"
                    for matches, rest in actions_pages(get, path, "workflow_runs"):
                        for match in matches:
                            if match.get("head_sha") != sha:
                                raise ValueError("SHA lookup returned another commit")
                            observe(match)
                        if rest == 0:
                            complete_run_lists.add(sha)
                break
    except _Budget:
        status, reason = "partial", _BUDGET
    except _NoReader:
        status, reason = "unavailable", _NO_READER
    except Exception as exc:
        status, reason = "unavailable", type(exc).__name__
    incomplete_attempts, unread = _read_earlier_attempts(runs, sources, prior, observe, get)
    # A read that supplied nothing of this range leaves the unnamed record as it was; one that did owns the record.
    if unnamed and not current_keys and not complete_run_lists:
        earlier, origin = unnamed, None
        prior = {_run_key(r): r for r in earlier.get("runs", []) if isinstance(r, dict)}
    # Join the saved record by exact attempt identity, never by walking every hypothetical attempt below a budget cut.
    # A completed current listing still owns disappearance of a non-red run.
    _join_saved_attempts(runs, sources, current_keys, prior, wanted, complete_run_lists, incomplete_attempts)
    # Once, after the exact same-repository join: grouped positive attempt counts over the exact rows.
    # A gap below a run's highest known attempt is unread metadata, never an attempt that never existed,
    # and it buys no request; it blocks the all-interrupted cover and the saved-edge carry below.
    holes = _attempt_holes(runs.values())
    incomplete_attempts |= holes
    for key in runs.keys() - current_keys:
        runs[key]['retained'] = True
    known_shas = {run['sha'] for run in runs.values()}
    fresh_absent = complete_run_lists - known_shas
    absent = fresh_absent | (wanted.intersection(earlier.get('absent_shas') or ()) - known_shas - complete_run_lists)

    def lookup(run):
        source = sources.get(_run_key(run))
        if source is not None and range_candidate(source):
            try:
                receipt = measured_window(source, get)
            except Exception as exc:  # A missing proof does not erase the direct run outcome.
                unread.append(exc)
                return
            run["measured_base"] = receipt[0] if receipt else None

    missing = _missing_runs(commits, list(runs.values()), fresh_absent, base_sha, lookup,
                            full_line=material.get('first_parent_complete') is True, previous=earlier,
                            complete_run_lists=complete_run_lists, incomplete_attempts=incomplete_attempts)
    # This certificate is written only for a target with a direct row AND an actual proven edge.
    covered_interrupted = {s for run in runs.values() for s in run.get('covered_shas', ()) if s in known_shas}
    if unread and status == "observed":  # neither unread attempt metadata nor unread jobs prove completeness
        status, reason = "partial", _BUDGET if any(isinstance(exc, _Budget) for exc in unread) else type(unread[0]).__name__
    elif holes & complete_run_lists and status == "observed":  # a whole listing with a known gap is still partial
        status, reason = "partial", _UNREAD_ATTEMPT
    return {"status": status, "reason": reason, "runs": list(runs.values()), "absent_shas": sorted(absent),
            "interrupted_shas": sorted(covered_interrupted), "missing_shas": missing,
            **({"repository": origin} if isinstance(origin, str) else {})}


def post_merge_reds(repo_url: str, *, get_json=None) -> dict:
    """The update source's open ``post-merge-red`` issues, read by each completed update check (at boot
    and on each manual check).

    Each is a test or lint finding that stayed red after merges into the development branch, with the
    merge it was bisected to when the bisection finished, and is still open: a report still tracked on
    the body this install runs or will run, whose fix or its confirmation may not yet be reflected in
    the issue. One public page, no credential; a full page stays ``partial``. A failed read is
    ``unavailable`` and lists nothing; ``kept_reds`` keeps the stored observation.
    """
    repo = _github_repo(repo_url)
    if repo is None:
        return {"read": "unavailable", "reason": "The update source has no supported GitHub issue reader.",
                "issues": []}
    try:
        rows = (get_json or _get_json)(f"https://api.github.com/repos/{repo}/issues?labels={RED_LABEL}"
                                       "&state=open&per_page=100")
        issues = [{"number": row["number"], "title": str(row.get("title") or ""),
                   "opened": str(row.get("created_at") or "")} for row in rows if "pull_request" not in row]
    except Exception as exc:
        return {"read": "unavailable", "reason": type(exc).__name__, "repository": repo, "issues": [],
                "observed_at": ""}
    return {"read": "partial" if len(rows) >= 100 else "observed", "repository": repo, "issues": issues,
            "observed_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def kept_reds(read: dict | None, stored: dict | None) -> dict | None:
    """What a check records: its own read, or after a failed read the stored observation of the same
    repository with its time, never "none open". Called under the letter lock with the record on disk,
    so a late failure never replaces a fresher observation that another check wrote meanwhile."""
    if read is None:
        return stored
    if (read.get("read") == "unavailable" and isinstance(stored, dict) and stored.get("observed_at")
            and stored.get("repository") == read.get("repository")):
        return {**read, "issues": stored.get("issues") or [], "observed_at": stored["observed_at"]}
    return read


def range_qualification(material: dict, repo_url: str) -> dict:
    commits = material.get("first_parent_commits", material.get("commits")) or []
    facts = {"base_sha": material["base_sha"], "target_sha": material["target_sha"],
             "commits": [{"sha": c["sha"], "subject": c["subject"], "date": c.get("date"),
                         "receipt": c.get("qualification", commit_qualification(c.get("body") or "")),
                         "body_read": bool(c.get("receipt_read") or c.get("body_read", True))}
                         for c in commits]}
    if material.get("first_parent_complete") is True:
        facts["first_parent_complete"] = True
    return {**facts, "github": post_merge_checks(facts, repo_url)}


def compact_qualification(value):
    """A context form of qualification receipts, never a new qualification verdict.

    Checks count by outcome; each review counts as one combination (observed verdict, record
    state, request-time relation, declaration), never as separate totals; a receipt without state
    and relation counts as ``not_recorded``. Unknown evidence and new fields stay verbatim, bound
    to their commit. Exact receipts live in the host capture.
    """
    import copy
    from collections import Counter

    if not isinstance(value, dict):
        return value
    result = copy.deepcopy(value)
    commits = result.get('commits')
    if isinstance(commits, list):
        checks, reviews, post_merge, whole = Counter(), Counter(), Counter(), []
        for commit in commits:
            receipt = commit.get('receipt') if isinstance(commit, dict) else None
            if (not isinstance(receipt, dict) or set(commit) - {'sha', 'subject', 'date', 'receipt', 'body_read'}
                    or set(receipt) - {'schema', 'checks', 'review', 'post_merge'} or receipt.get('schema') != 1
                    or not isinstance(receipt.get('checks'), list) or not isinstance(receipt.get('review'), dict)
                    or not isinstance(receipt.get('post_merge'), dict)
                    or commit.get('body_read') is False):
                whole.append(commit)
                continue
            unknown = {}
            for check in receipt.get('checks') or ():
                if (not isinstance(check, dict) or set(check) - {'name', 'status', 'source', 'detail'}
                        or check.get('status') not in ('passed', 'failed', 'not_run')
                        or check.get('source') != 'author_declared'):
                    unknown.setdefault('checks', []).append(check)
                else:
                    checks[(check.get('name', ''), check['status'], check['source'])] += 1
            review = receipt.get('review') or {}
            recorded = {'state', 'relation'} & set(review)
            if (set(review) - {'observed', 'declared', 'state', 'relation'}
                    or any(review.get(k) not in ('PASS', 'FAIL') for k in ('observed', 'declared'))
                    or (recorded and (review.get('state') not in REVIEW_STATES
                                      or review.get('relation') not in REVIEW_RELATIONS))):
                unknown['review'] = review
            else:
                reviews[(review['observed'], review.get('state', 'not_recorded'),
                         review.get('relation', 'not_recorded'), review['declared'])] += 1
            post = receipt.get('post_merge') or {}
            if set(post) - {'status', 'detail'} or post.get('status') not in ('passed', 'failed', 'not_run'):
                unknown['post_merge'] = post
            else:
                post_merge[post['status']] += 1
            if unknown:
                whole.append({'sha': commit.get('sha'), 'receipt': unknown})
        result['commits'] = {'count': len(commits),
            'checks': [{'name': name, 'status': status, 'source': source, 'count': count}
                       for (name, status, source), count in sorted(checks.items())],
            'reviews': [{'observed': observed, 'state': state, 'relation': relation, 'declared': declared,
                         'count': count} for (observed, state, relation, declared), count in sorted(reviews.items())],
            'post_merge': dict(post_merge), 'unabridged': whole,
            'fact': 'Declarations are not observed PASS; missing or unconfirmed evidence remains unknown.',
            'details': 'retained full runtime section: official_update.letter.qualification'}
    github = result.get('github')
    if isinstance(github, dict) and isinstance(github.get('runs'), list):
        counts, retained, whole = Counter(), Counter(), []
        for run in github['runs']:  # a measured base and the commits a run covers are shown by `missing_shas`
            if (not isinstance(run, dict)
                    or set(run) - {'sha', 'status', 'conclusion', 'url', 'run_id', 'attempt', 'measured_base', 'covered_shas', 'retained'}
                    or ('retained' in run and type(run['retained']) is not bool)
                    or not isinstance(run.get('measured_base', ''), (str, type(None)))
                    or not isinstance(run.get('covered_shas', []), list)
                    or run.get('status') != 'completed'
                    or run.get('conclusion') not in ('success', 'failure', 'cancelled', 'skipped', 'timed_out')):
                whole.append(run)
            else:
                counts[run['conclusion']] += 1
                if run.get('retained') is True:
                    retained[run['conclusion']] += 1
        github['runs'] = {'count': len(github['runs']), 'completed': dict(counts), 'unabridged': whole,
                          **({'retained': dict(retained)} if retained else {}),
                          'details': 'retained full runtime section: official_update.letter.qualification.github.runs'}
    return result


def refresh_qualification(authored: dict, repo_url: str, *, previous: dict | None = None) -> tuple[dict, bool]:
    """Refresh facts without making an outage erase observations; prose never follows them.

    The boolean names new check evidence relative to ``authored``, the recorded facts being
    refreshed. Availability is shown separately, with the latest retained observations.
    """
    if not previous or any(previous.get(k) != authored.get(k) for k in ("base_sha", "target_sha")):
        previous = authored
    # The read keeps what ``previous`` observed that it cannot see now (``post_merge_checks``): a failed read
    # establishes no new absence, and a newly observed run supersedes an absence, including a prior coverage claim.
    github = post_merge_checks({**previous, "commits": authored["commits"]}, repo_url)
    old = authored.get("github") or {}

    def rows(github):
        # Whether the same immutable outcome needed reading again is not a new test result.
        return sorted(json.dumps({k: v for k, v in run.items() if k != 'retained'}, sort_keys=True)
                      for run in github.get('runs', []))

    changed = rows(github) != rows(old) or github.get("missing_shas", []) != old.get("missing_shas", [])
    return {**authored, "github": github}, changed


def qualification_summary(qualification: dict | None) -> dict | None:
    """A range's qualification in counts for the Runtime context, with the reader of the whole record.

    The panel, preflight and the letter's author keep the full record (owner 8B); a task sees the range, how
    many commits carry a receipt, each check's outcomes by their source (``author_declared``: the author's
    word, not a host run), each review combination as one row (observed verdict, record state, request-time
    relation, declaration; ``not_recorded`` for older receipts, unknown fields kept), the post-merge read, and
    how many of this install's own commits in the range it recorded over a review FAIL or no verdict.
    """
    if not isinstance(qualification, dict):
        return qualification
    commits = [c for c in qualification.get("commits") or [] if isinstance(c, dict)]
    receipts = [c["receipt"] for c in commits if isinstance(c.get("receipt"), dict)]
    checks: dict = {}
    reviews: dict = {}
    for receipt in receipts:
        for row in receipt.get("checks") or []:
            outcome = checks.setdefault(str(row.get("name")), {})
            key = f"{row.get('status')} ({row.get('source') or 'unknown'})"
            outcome[key] = outcome.get(key, 0) + 1
        review = receipt.get("review") if isinstance(receipt.get("review"), dict) else {}
        # One row per combination, so a verdict never meets another commit's state or relation.
        row = {"observed": "unknown", "state": "not_recorded", "relation": "not_recorded", "declared": "unknown"}
        # Runtime summaries are a projection, but not a lossy one: future JSON fields, including
        # explicit null/empty values and nested shapes, remain their original values.
        row.update({str(k): v for k, v in review.items()})
        key = json.dumps(row, sort_keys=True)
        reviews[key] = reviews.get(key, 0) + 1
    github = qualification.get("github") if isinstance(qualification.get("github"), dict) else {}
    conclusions, retained = {}, {}
    for run in github.get('runs') or []:
        # A row kept from before is what was last seen: unsettled, it is no job running now, its outcome unknown.
        key = str(run.get('conclusion') or (run.get('status') if run.get('retained') is not True else '') or 'unknown')
        conclusions[key] = conclusions.get(key, 0) + 1
        if run.get('retained') is True:
            retained[key] = retained.get(key, 0) + 1
    landed = qualification.get("landed_over_review")  # absent when this install's rows were not read: unknown
    return {"base_sha": qualification.get("base_sha"), "target_sha": qualification.get("target_sha"),
            "commits": len(commits), "with_receipt": len(receipts), "checks": checks,
            "reviews": _review_summary_rows(reviews),
            **({"landed_over_review": len(landed)} if isinstance(landed, list) else {}),
            "post_merge": {"read": github.get("status", "unknown"), "reason": github.get("reason"),
                           "runs": conclusions, "missing": len(github.get("missing_shas") or []),
                           **({'retained': retained} if retained else {})},
            "full": "query_data(op='json', kind='update_letter')"}


def _review_summary_rows(reviews: dict) -> list[dict]:
    """Add the projection's aggregate count without overwriting an unknown source field."""
    rows = []
    for encoded, aggregate_count in reviews.items():
        row = json.loads(encoded)
        key = "count"
        while key in row:
            key = f"_{key}"
        row[key] = aggregate_count
        rows.append(row)
    return rows


def runtime_letter(letter: dict | None) -> dict | None:
    """The letter as the Runtime context carries it: whole, but its qualification summarized."""
    if not letter or not letter.get("qualification"):
        return letter
    return {**letter, "qualification": qualification_summary(letter["qualification"])}


def preflight_description(plan: dict, *, drive_root=None) -> dict:
    """Read the saved description for the exact pins; never fetch, write or author."""
    from ouroboros import update_letter as letter

    base, target = str(plan.get("base_sha") or ""), str(plan.get("target_sha") or "")
    if not base or not target:
        return {"letter": None, "qualification": None}
    record = letter.read_record(drive_root)
    view = letter.project_letter(record, head_sha=base, latest_sha=target)
    if view and not view["description_current"]:
        view = {**view, "text": ""}
    qualification = (record or {}).get("qualification")
    if qualification and (qualification.get("base_sha"), qualification.get("target_sha")) != (base, target):
        qualification = None
    return {"letter": view, "qualification": qualification}
