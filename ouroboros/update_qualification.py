"""Portable merge qualification and exact-range GitHub observations.

The commit trailer is evidence, never approval. Author declarations remain labelled;
missing executions stay unknown. Fetching checks consult GitHub, passive readers do not.
No prose is classified: older commit bodies reach the description's author unchanged, and
a refreshed run never rewrites a description of the same range.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone

TRAILER = "Ouroboros-Qualification: "
CHECKS = ("full_battery", "cost_contract")
STATES = ("passed", "failed", "not_run", "unknown")
RED_LABEL = "post-merge-red"  # scripts/post_merge_routing.py labels each red it hands over with it


def merge_qualification(review: dict, checks: list | None = None) -> dict:
    declared = {row["name"]: {**row, "source": "author_declared"} for row in checks or []}
    for name in CHECKS:
        declared.setdefault(name, {"name": name, "status": "unknown", "source": "missing",
                                   "detail": "No execution evidence was supplied."})
    record = review.get("record") or {}
    return {"schema": 1, "checks": list(declared.values()),
            "review": {"observed": (record.get("verdict") or {}).get("aggregate", "unknown"),
                       "declared": (review.get("declared") or {}).get("verdict", "unknown")},
            "post_merge": {"status": "not_run", "detail": "The post-merge check follows this merge."}}


def commit_summary(qualification: dict) -> str:
    lines = ["Ouroboros qualification receipt (declarations are not host-observed PASS):"]
    for row in qualification["checks"]:
        lines.append(f"- {row['name']}: {row['status']} ({row['source']}); {row.get('detail') or 'no detail supplied'}")
    review = qualification["review"]
    lines += [f"- Review: observed {review['observed']}; author declared {review['declared']}",
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


def _missing_runs(commits: list[dict], runs: dict, absent: set[str]) -> list[str]:
    """First-parent order is newest first; completed checks cover skipped pushes.

    Match post_merge_check.choose_base: cancelled/skipped runs are not coverage.
    Only proven absences or observed replaced runs receive coverage. Unread pages
    prove nothing; direct observations, including earlier reds, stay visible.
    """
    for run in runs.values():
        run.pop("covered_shas", None)
    covering, missing = None, []
    for commit in commits:
        sha = commit["sha"]
        run = runs.get(sha)
        if run and run.get("status") == "completed" and run.get("conclusion") in ("success", "failure"):
            covering = run
        elif covering is not None and (run is not None or sha in absent):
            covering.setdefault("covered_shas", []).append(sha)
        elif run is None:
            missing.append(sha)
    return sorted(missing)


def _date(value) -> datetime | None:
    try:
        date = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return date if date.tzinfo is not None else None
    except ValueError:
        return None


def post_merge_checks(commits: list[dict], repo_url: str, *, get_json=None) -> dict:
    """Read runs for first-parent integrations, newest first, including squash.

    Page through the oldest integration before inferring skipped-run coverage.
    Its own row, exhausted history or a run predating it establishes absence;
    a physical 10-page response budget stays partial, never proof of absence.
    Public GitHub needs no credential. Offline/rate-limit failures remain unknown.
    """
    if not commits:
        return {"status": "observed", "runs": [], "missing_shas": []}
    get_json = get_json or _get_json
    repo = _github_repo(repo_url)
    if repo is None:
        return {"status": "unavailable", "reason": "The update source has no supported GitHub check reader.",
                "runs": [], "missing_shas": sorted(c["sha"] for c in commits)}
    wanted = {c["sha"] for c in commits}
    oldest = _date(commits[-1].get("date"))
    runs = {}
    status, reason = "partial", "GitHub response page budget reached."
    try:
        for page in range(1, 11):
            data = get_json(f"https://api.github.com/repos/{repo}/actions/workflows/"
                            f"post-merge.yml/runs?per_page=100&page={page}")
            rows = data["workflow_runs"]
            for row in rows:
                sha = row.get("head_sha")
                if sha in wanted and sha not in runs:
                    runs[sha] = {"sha": sha, "status": row.get("status"), "conclusion": row.get("conclusion"),
                                 "url": row.get("html_url"), "run_id": row.get("id"),
                                 "attempt": row.get("run_attempt")}
            if (len(rows) < 100 or commits[-1]["sha"] in runs or
                    (oldest is not None and any((created := _date(row.get("created_at"))) is not None
                                               and created < oldest for row in rows))):
                status, reason = "observed", "No matching run is not a passing check."
                break
    except Exception as exc:
        status, reason = "unavailable", type(exc).__name__
    absent = wanted - runs.keys() if status == "observed" else set()
    return {"status": status, "reason": reason, "runs": list(runs.values()), "absent_shas": sorted(absent),
            "missing_shas": _missing_runs(commits, runs, absent)}


def post_merge_reds(repo_url: str, *, get_json=None) -> dict:
    """The update source's open ``post-merge-red`` issues, read by each completed update check (at boot
    and on each manual check).

    Each is a test or lint finding that stayed red after merges into the development branch, with the
    merge it was bisected to when the bisection finished, and is still open: unfixed work on the body
    this install runs or will run. One public page, no credential; a full page stays ``partial``. A
    failed read is ``unavailable`` and lists nothing; ``kept_reds`` keeps the stored observation.
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
    return {"base_sha": material["base_sha"], "target_sha": material["target_sha"],
            "commits": [{"sha": c["sha"], "subject": c["subject"], "date": c.get("date"),
                         "receipt": c.get("qualification", commit_qualification(c.get("body") or "")),
                         "body_read": c.get("body_read", True)}
                        for c in commits],
            "github": post_merge_checks(commits, repo_url)}


def compact_qualification(value):
    """A context form of qualification receipts, never a new qualification verdict.

    Count declared/observed outcomes separately; unknown evidence and new fields
    stay verbatim, bound to their commit. Exact receipts live in the host capture.
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
            if (set(review) - {'observed', 'declared'} or any(review.get(k) not in ('PASS', 'FAIL')
                                                           for k in ('observed', 'declared'))):
                unknown['review'] = review
            else:
                reviews[(review['observed'], review['declared'])] += 1
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
            'reviews': [{'observed': observed, 'declared': declared, 'count': count}
                        for (observed, declared), count in sorted(reviews.items())],
            'post_merge': dict(post_merge), 'unabridged': whole,
            'fact': 'Declarations are not observed PASS; missing or unconfirmed evidence remains unknown.',
            'details': 'retained full runtime section: official_update.letter.qualification'}
    github = result.get('github')
    if isinstance(github, dict) and isinstance(github.get('runs'), list):
        counts, whole = Counter(), []
        for run in github['runs']:
            if (not isinstance(run, dict) or set(run) - {'sha', 'status', 'conclusion', 'url', 'run_id', 'attempt'}
                    or run.get('status') != 'completed'
                    or run.get('conclusion') not in ('success', 'failure', 'cancelled', 'skipped', 'timed_out')):
                whole.append(run)
            else:
                counts[run['conclusion']] += 1
        github['runs'] = {'count': len(github['runs']), 'completed': dict(counts), 'unabridged': whole,
                          'details': 'retained full runtime section: official_update.letter.qualification.github.runs'}
    return result


def refresh_qualification(authored: dict, repo_url: str, *, previous: dict | None = None) -> tuple[dict, bool]:
    """Refresh facts without making an outage erase observations; prose never follows them.

    The boolean names new check evidence relative to ``authored``, the recorded facts being
    refreshed. Availability is shown separately, with the latest retained observations.
    """
    if not previous or any(previous.get(k) != authored.get(k) for k in ("base_sha", "target_sha")):
        previous = authored
    github = post_merge_checks(authored["commits"], repo_url)
    if github.get("status") in ("unavailable", "partial"):
        runs = {r["sha"]: dict(r) for r in previous.get("github", {}).get("runs", [])}
        runs.update({r["sha"]: dict(r) for r in github.get("runs", [])})
        # A failed read cannot establish new absences. Retain only proven ones;
        # a newly observed run supersedes absence, including a prior coverage claim.
        absent = set(previous.get("github", {}).get("absent_shas", [])) - runs.keys()
        github = {**github, "absent_shas": sorted(absent),
                  "missing_shas": _missing_runs(authored["commits"], runs, absent), "runs": list(runs.values())}
    old = authored.get("github") or {}
    changed = ({r["sha"]: r for r in github.get("runs", [])} != {r["sha"]: r for r in old.get("runs", [])}
               or github.get("missing_shas", []) != old.get("missing_shas", []))
    return {**authored, "github": github}, changed


def qualification_summary(qualification: dict | None) -> dict | None:
    """A range's qualification in counts for the Runtime context, with the reader of the whole record.

    The panel, preflight and the letter's author keep the full record (owner 8B); a task sees the range, how
    many commits carry a receipt, each check's outcomes by their source (``author_declared``: the author's
    word, not a host run) and the reviews observed and declared, the post-merge read with its runs and gaps.
    """
    if not isinstance(qualification, dict):
        return qualification
    commits = [c for c in qualification.get("commits") or [] if isinstance(c, dict)]
    receipts = [c["receipt"] for c in commits if isinstance(c.get("receipt"), dict)]
    checks: dict = {}
    reviews: dict = {"observed": {}, "declared": {}}
    for receipt in receipts:
        for row in receipt.get("checks") or []:
            outcome = checks.setdefault(str(row.get("name")), {})
            key = f"{row.get('status')} ({row.get('source') or 'unknown'})"
            outcome[key] = outcome.get(key, 0) + 1
        for side, verdicts in reviews.items():
            verdict = str((receipt.get("review") or {}).get(side) or "unknown")
            verdicts[verdict] = verdicts.get(verdict, 0) + 1
    github = qualification.get("github") if isinstance(qualification.get("github"), dict) else {}
    conclusions: dict = {}
    for run in github.get("runs") or []:
        key = str(run.get("conclusion") or run.get("status") or "unknown")
        conclusions[key] = conclusions.get(key, 0) + 1
    return {"base_sha": qualification.get("base_sha"), "target_sha": qualification.get("target_sha"),
            "commits": len(commits), "with_receipt": len(receipts), "checks": checks, "reviews": reviews,
            "post_merge": {"read": github.get("status", "unknown"), "reason": github.get("reason"),
                           "runs": conclusions, "missing": len(github.get("missing_shas") or [])},
            "full": "query_data(op='json', kind='update_letter')"}


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
