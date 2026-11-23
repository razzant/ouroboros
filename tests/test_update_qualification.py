"""Qualification is portable evidence of the exact proposed range, never a veto."""
import json
from urllib.parse import parse_qs, urlparse

import pytest

from ouroboros import update_letter as letter, update_qualification as qualification
from ouroboros.tools import github
from tests.test_pr_merge_receipts import BASE, HEAD, ROUTE, _receipts, sent, world  # noqa: F401
from tests.test_update_letter import _capture_for, _commit, _git, _record, _status, history_repo  # noqa: F401
from tests.test_update_letter_merge_range import merged_range  # noqa: F401


def test_preflight_reads_exact_saved_range_without_writing_or_calling_a_model(tmp_path, monkeypatch):
    running, head, target = "a" * 40, "c" * 40, "b" * 40
    monkeypatch.setattr(letter, "_default_git", lambda: lambda _: (1, "", ""))
    monkeypatch.setattr(letter, "collect_range_material", lambda base, target, **k: {
        "base_sha": base, "target_sha": target, "commits": [{"sha": target, "subject": "change"}]})
    monkeypatch.setattr(qualification, "post_merge_checks", lambda *a: {"status": "observed", "runs": []})
    check = _status(current_sha=head, running_sha=running)
    plan = {"base_sha": head, "target_sha": target, "update_channel": "stable", "target_ref": "managed/main"}
    for _ in range(2):
        letter.refresh_after_check(check, drive_root=tmp_path)
        saved = letter.record_path(tmp_path).read_bytes()
        result = qualification.preflight_description(plan, drive_root=tmp_path)
        assert not result["letter"] or not result["letter"]["text"]
        assert result["qualification"] is None
        assert letter.record_path(tmp_path).read_bytes() == saved
    assert letter.save_description("letter", running, target, 0, author={"task_id": "t", "model": "m"},
                                   drive_root=tmp_path)["outcome"] == "saved"
    saved = letter.record_path(tmp_path).read_bytes()
    result = qualification.preflight_description({**plan, "base_sha": running}, drive_root=tmp_path)
    assert result["letter"]["text"] == "letter" and result["letter"]["description_current"]
    assert result["qualification"]["base_sha"] == running
    assert letter.record_path(tmp_path).read_bytes() == saved


@pytest.mark.parametrize("squash", [False, True])
def test_qualification_checks_only_first_parent_integrations(merged_range, monkeypatch, squash):  # noqa: F811
    repo, base, target, _local, feature, nested = merged_range
    if squash:
        _git(repo, "checkout", "-q", "-b", "squashed", base)
        _git(repo, "merge", "--squash", target)
        target = _commit(repo, "Squashed integration")
    material = letter.collect_range_material(base, target, git=_capture_for(repo))
    urls = []

    def get(url):
        urls.append(url)
        return {"total_count": 100, "workflow_runs": [
            {"head_sha": target, "status": "completed", "conclusion": "success"},
            *[{"head_sha": "outside", "status": "completed", "conclusion": "failure"}] * 99]}

    monkeypatch.setattr(qualification, "_get_json", get)
    fact = qualification.range_qualification(material, "https://github.com/owner/repo")
    assert [c["sha"] for c in fact["commits"]] == [target]
    assert fact["first_parent_complete"]
    assert fact["github"]["status"] == "observed" and fact["github"]["missing_shas"] == []
    assert len(urls) == 1
    # The letter still receives branch-internal subjects and bodies.
    assert {feature, nested}.issubset({c["sha"] for c in material["commits"]}) is (not squash)


def _attempt(head, status="completed", conclusion="success", *, attempt=1):
    return {"id": 77, "run_attempt": attempt, "head_sha": head, "head_branch": "ouroboros", "event": "push",
            "display_title": qualification.RANGE_TITLE, "status": status, "conclusion": conclusion}


def test_initial_range_only_accepts_first_parents_of_the_dispatched_commit(merged_range, monkeypatch):  # noqa: F811
    from scripts import post_merge_check

    repo, base, head, local, feature, nested = merged_range
    monkeypatch.setattr(post_merge_check, "REPO_ROOT", repo)
    assert post_merge_check.is_ancestor(base, head)
    assert not any(post_merge_check.is_ancestor(other, head) for other in (head, local, feature, nested, base[:12]))


def _jobs(run, base, scope="chain"):
    step = qualification.CONTINUOUS_STEP if scope == "chain" else qualification.MEASURED_STEP
    return [{"name": f"post-merge ({leg}) from {base}", "head_sha": run["head_sha"], "run_id": run["id"],
             "run_attempt": run["run_attempt"], "status": "completed", "conclusion": run["conclusion"],
             "steps": [{"name": step, "status": "completed", "conclusion": "success"}]}
            for leg in ("linux", "windows")]


@pytest.mark.parametrize("case", ["green", "measured red", "push only", "mixed continuity", "no selection", "one leg", "other attempt",
                                  "different bases", "unreadable", "legacy", "still running", "cancelled"])
def test_update_keeps_direct_results_and_covers_only_the_joint_attempt_range(case):
    newer, tip, cancelled, missing, base, older = (f"{i:040x}" for i in range(6, 0, -1))
    run = _attempt(tip, "in_progress" if case == "still running" else "completed",
                   "cancelled" if case == "cancelled" else None if case == "still running" else
                   "failure" if case in ("measured red", "push only", "mixed continuity", "no selection") else "success")
    if case == "legacy":
        run.pop("display_title")
    rows = [run, {"head_sha": cancelled, "status": "completed", "conclusion": "cancelled"}]
    seen = []

    def get(url):
        seen.append(url)
        if "/workflows/" in url:
            return {"total_count": len(rows), "workflow_runs": rows}
        assert url.endswith("/actions/runs/77/attempts/1/jobs?per_page=100&page=1")
        if case == "unreadable":
            raise OSError("offline")
        jobs = _jobs(run, base, "push" if case == "push only" else "chain")
        if case == "one leg":
            jobs.pop()
        elif case == "mixed continuity":
            jobs[0]["steps"][0]["name"] = qualification.MEASURED_STEP
        elif case == "other attempt":
            jobs[0]["run_attempt"] -= 1
        elif case == "different bases":
            jobs[0]["name"] = jobs[0]["name"].replace(base, older)
        elif case == "no selection":
            jobs[0]["steps"][0]["conclusion"] = "skipped"
        return {"total_count": len(jobs), "jobs": jobs}

    commits = [{"sha": sha} for sha in (newer, tip, cancelled, missing, base, older)]
    fact = qualification.post_merge_checks(commits, "https://github.com/owner/repo", get_json=get)
    covered = case in ("green", "measured red", "push only", "mixed continuity")
    assert fact["runs"][0]["conclusion"] == run["conclusion"] and fact["runs"][1]["conclusion"] == "cancelled"
    assert fact["runs"][0].get("covered_shas", []) == ([cancelled, missing] if covered else [])
    assert fact["missing_shas"] == sorted([newer, base, older] + ([] if covered else [missing]))
    assert len(seen) == (1 if case in ("legacy", "still running", "cancelled") else 2)


def test_an_exact_sha_lookup_finds_an_older_direct_red_before_covering_a_gap():
    head, missing, base = (s * 40 for s in "cba")
    rows = [_attempt(head), *[{"head_sha": "outside", "created_at": "2020-01-01T00:00:00Z"}] * 99]
    calls = []

    def get(url):
        query = parse_qs(urlparse(url).query)
        calls.append(query)
        if "head_sha" in query:
            assert query['branch'] == ['ouroboros']
            if query['head_sha'] == [head]:
                return {'total_count': 1, 'workflow_runs': [_attempt(head)]}
            assert query['head_sha'] == [missing]
            return {"total_count": 1, "workflow_runs": [{**_attempt(missing), "id": 12, "conclusion": "failure"}]}
        return {"total_count": 501, "workflow_runs": rows}

    material = {"base_sha": base, "commits": [{"sha": head}, {"sha": missing, "date": "2021-01-01T00:00:00Z"}]}
    fact = qualification.post_merge_checks(material, "https://github.com/owner/repo", get_json=get)
    assert len(calls) == 3 and fact["missing_shas"] == fact["absent_shas"] == []
    assert [(r["sha"], r["conclusion"]) for r in fact["runs"]] == [(head, "success"), (missing, "failure")]
    assert not any("covered_shas" in r for r in fact["runs"])


@pytest.mark.parametrize("full_line", [True, False])
def test_a_measurement_starting_before_the_install_covers_only_a_proved_local_suffix(full_line, monkeypatch):
    head, skipped, local, measured = (s * 40 for s in "dcba")
    run = _attempt(head)
    monkeypatch.setattr(qualification, "_get_json", lambda url: {"total_count": 2, "jobs": _jobs(run, measured)}
                        if "/jobs?" in url else {"total_count": 1, "workflow_runs": [run]})
    material = {"base_sha": local, "target_sha": head, "first_parent_complete": full_line,
                "first_parent_commits": [{"sha": s, "subject": s} for s in (head, skipped)]}
    fact = qualification.range_qualification(material, "https://github.com/owner/repo")
    for observed in (fact["github"], qualification.refresh_qualification(fact, "https://github.com/owner/repo")[0]["github"]):
        assert observed["runs"][0]["measured_base"] == measured
        assert observed["runs"][0].get("covered_shas", []) == ([skipped] if full_line else [])
        assert observed["missing_shas"] == ([] if full_line else [skipped])


@pytest.mark.parametrize("older_run", ["failure", "success", None])
def test_paging_observes_older_runs_before_inferring_coverage(monkeypatch, older_run):
    commits = [{"sha": f"{i:040x}", "subject": f"Integration {i}"} for i in range(150, 0, -1)]
    older = commits[130]["sha"]
    rows = [{"head_sha": c["sha"], "status": "completed",
             "conclusion": older_run if c["sha"] == older else "success"}
            for c in commits if c["sha"] != older or older_run is not None]
    witness = _attempt(commits[129]["sha"])
    rows[129] = witness
    pages = []

    def get(url):
        if "/jobs?" in url:
            pages.append("jobs")
            return {"total_count": 2, "jobs": _jobs(witness, commits[131]["sha"])}
        page = int(parse_qs(urlparse(url).query)["page"][0])
        pages.append(page)
        return {"total_count": len(rows), "workflow_runs": rows[(page - 1) * 100:page * 100]}

    monkeypatch.setattr(qualification, "_get_json", get)
    material = {"base_sha": "base", "target_sha": commits[0]["sha"], "first_parent_commits": commits}
    fact = qualification.range_qualification(material, "https://github.com/owner/repo")

    def assert_evidence(observed):
        runs = {r["sha"]: r for r in observed["runs"]}
        covered = {sha for r in runs.values() for sha in r.get("covered_shas", [])}
        assert observed["status"] == "observed" and observed["missing_shas"] == []
        assert observed["absent_shas"] == ([older] if older_run is None else [])
        assert covered == ({older} if older_run is None else set())
        assert {r["sha"] for r in runs.values() if r["conclusion"] == "failure"} == (
            {older} if older_run == "failure" else set())
        if older_run is not None:
            assert runs[older]["conclusion"] == older_run

    assert_evidence(fact["github"])
    assert pages == ([1, 2, "jobs"] if older_run is None else [1, 2])
    # Later upstream activity moves the same range farther down the API listing.
    rows[:0] = [{"head_sha": "outside", "status": "completed", "conclusion": "failure"}] * 60
    pages.clear()
    refreshed, changed = qualification.refresh_qualification(fact, "https://github.com/owner/repo")
    assert_evidence(refreshed["github"])
    assert pages == [1, 2, 3] and not changed  # the same immutable attempt reuses its receipt


@pytest.mark.parametrize("ending", ["exhausted", "budget", "unavailable"])
def test_coverage_requires_reading_through_the_range(ending, monkeypatch):
    commits = [{"sha": sha, "subject": sha, "date": "2026-10-23T03:00:00+03:00"}
               for sha in ("newest", "middle", "oldest")]
    green = {"head_sha": "newest", "status": "completed", "conclusion": "success",
             "created_at": "2026-10-23T00:01:00Z"}
    filler = {**green, "head_sha": "outside"}
    pages = []

    def get(url):
        page = int(parse_qs(urlparse(url).query)["page"][0])
        pages.append(page)
        if page == 1:
            return {"total_count": 101 if ending == "exhausted" else 1001, "workflow_runs": [green, *[filler] * 99]}
        if ending == "unavailable":
            raise OSError("offline")
        if ending == "exhausted":
            return {"total_count": 101, "workflow_runs": [filler]}
        return {"total_count": 1001, "workflow_runs": [filler] * 100}

    monkeypatch.setattr(qualification, "_get_json", get)
    fact = qualification.range_qualification(
        {"base_sha": "base", "target_sha": "newest", "first_parent_commits": commits},
        "https://github.com/owner/repo")
    observed = fact["github"]
    complete = ending == "exhausted"
    absent = ["middle", "oldest"]
    assert pages == (list(range(1, 11)) if ending == "budget" else [1, 2])
    assert observed["status"] == ("observed" if complete else "partial" if ending == "budget" else "unavailable")
    assert observed["missing_shas"] == absent
    assert observed["absent_shas"] == (absent if complete else [])
    assert not observed["runs"][0].get("covered_shas")  # the old workflow never published measured ranges
    # Refresh must use the saved date boundary, and cannot invent coverage after a short read.
    pages.clear()
    refreshed, changed = qualification.refresh_qualification(fact, "https://github.com/owner/repo")
    assert refreshed == fact and not changed
    assert pages == (list(range(1, 11)) if ending == "budget" else [1, 2])


def test_refresh_preserves_proven_absence_until_a_direct_run_is_observed(monkeypatch):
    newest, earlier, oldest, base = (x * 40 for x in "dcba")
    commits = [{"sha": sha, "subject": sha} for sha in (newest, earlier, oldest)]
    rows = [_attempt(newest), {"head_sha": oldest, "status": "completed", "conclusion": "success"}]
    monkeypatch.setattr(qualification, "_get_json", lambda url: {"total_count": 2, "jobs": _jobs(rows[0], base)}
                        if "/jobs?" in url else {"total_count": 2, "workflow_runs": rows})
    authored = qualification.range_qualification({"base_sha": base, "target_sha": newest,
                                                 "commits": commits}, "https://github.com/owner/repo")
    assert authored["github"]["runs"][0]["covered_shas"] == [earlier]
    previous = authored
    for conclusion in (None, None, "failure"):
        def get(url):
            if "page=2" in url or conclusion is None:
                raise OSError("offline")
            return {"total_count": 101, "workflow_runs": [{"head_sha": earlier, "status": "completed", "conclusion": conclusion},
                                      *[{"head_sha": "outside"}] * 99]}

        monkeypatch.setattr(qualification, "_get_json", get)
        previous, changed = qualification.refresh_qualification(authored, "https://github.com/owner/repo",
                                                               previous=previous)
        runs = {r["sha"]: r for r in previous["github"]["runs"]}
        assert previous["github"]["status"] == "unavailable"
        assert changed is (conclusion is not None)
        assert runs[newest].get("covered_shas", []) == ([earlier] if conclusion is None else [])
        if conclusion is not None:
            assert runs[earlier]["conclusion"] == "failure"
            assert previous["github"]["absent_shas"] == []

    # Older saved coverage has no proof of absence; an outage cannot ratify it.
    legacy = json.loads(json.dumps(authored))
    legacy["github"].pop("absent_shas")
    monkeypatch.setattr(qualification, "_get_json", lambda _: (_ for _ in ()).throw(OSError()))
    refreshed, changed = qualification.refresh_qualification(legacy, "https://github.com/owner/repo")
    assert changed and refreshed["github"]["missing_shas"] == [earlier]
    assert not any(r.get("covered_shas") for r in refreshed["github"]["runs"])


def test_a_new_attempt_cannot_reuse_the_previous_attempts_measured_base(monkeypatch):
    head, middle, base = (x * 40 for x in "cba")
    run = _attempt(head)
    calls = []

    def get(url):
        calls.append(url)
        if "/workflows/" in url:
            return {"total_count": 1, "workflow_runs": [run]}
        assert url.endswith("/attempts/1/jobs?per_page=100&page=1")
        return {"total_count": 2, "jobs": _jobs(run, base)}

    monkeypatch.setattr(qualification, "_get_json", get)
    material = {"base_sha": base, "target_sha": head, "commits": [{"sha": s, "subject": s} for s in (head, middle)]}
    first = qualification.range_qualification(material, "https://github.com/owner/repo")
    assert first["github"]["runs"][0]["covered_shas"] == [middle]
    run.update(run_attempt=2, conclusion="failure")
    calls.clear()
    refreshed, changed = qualification.refresh_qualification(first, "https://github.com/owner/repo")
    newer, older = refreshed["github"]["runs"]
    assert changed and (newer["attempt"], newer["conclusion"]) == (2, "failure")
    assert "measured_base" not in newer and "covered_shas" not in newer
    assert older == {**first['github']['runs'][0], 'retained': True} and refreshed['github']['missing_shas'] == []
    assert len(calls) == 1 and "/workflows/" in calls[0], "the cached exact receipt settles the gap without another request"


def test_later_unmeasured_dispatch_does_not_hide_same_head_measured_push():
    head, gap, base = (s * 40 for s in "cba")
    push = _attempt(head, conclusion="failure")
    dispatch = {**_attempt(head), "id": 78, "event": "workflow_dispatch"}
    paths = []

    def get(url):
        path = urlparse(url).path
        paths.append(path)
        if path.endswith("/workflows/post-merge.yml/runs"):
            return {"total_count": 2, "workflow_runs": [dispatch, push]}
        if path.endswith("/78/attempts/1/jobs"):
            return {"total_count": 0, "jobs": []}
        assert path.endswith("/77/attempts/1/jobs")
        return {"total_count": 2, "jobs": _jobs(push, base)}

    material = {"base_sha": base, "commits": [{"sha": head}, {"sha": gap}]}
    fact = qualification.post_merge_checks(material, "https://github.com/owner/repo", get_json=get)
    newer, older = fact["runs"]
    assert [(r["run_id"], r["conclusion"]) for r in fact["runs"]] == [(78, "success"), (77, "failure")]
    assert newer["measured_base"] is None and "covered_shas" not in newer
    assert older["measured_base"] == base and older["covered_shas"] == [gap]
    assert (fact["status"], fact["absent_shas"], fact["missing_shas"]) == ("observed", [gap], [])
    assert len(paths) == 3
    paths.clear()
    assert qualification.post_merge_checks({**material, "github": fact}, "https://github.com/owner/repo", get_json=get) == fact
    assert len(paths) == 1


@pytest.mark.parametrize("ending", ["complete", "budget", "wrong head"])
def test_a_fresh_rerun_read_keeps_exact_historical_results_and_reports_unread_attempts(ending):
    head, base = (s * 40 for s in "ba")
    newest = 11 if ending == "budget" else 2
    direct = {**_attempt(head, attempt=newest), "workflow_id": 3,
              "html_url": "https://github.com/owner/repo/actions/runs/77"}
    red_attempt = newest - 1
    paths = []

    def get(url):
        path = urlparse(url).path
        paths.append(path)
        if path.endswith("/workflows/post-merge.yml/runs"):
            return {"total_count": 1, "workflow_runs": [direct]}
        attempt = int(path.rsplit("/", 1)[1])
        assert path.endswith(f"/actions/runs/77/attempts/{attempt}")
        return {**direct, "run_attempt": attempt, "head_sha": base if ending == "wrong head" else head,
                "conclusion": "failure" if attempt == red_attempt else "success"}

    repo = "https://github.com/owner/repo"
    material = {"base_sha": base, "commits": [{"sha": head}]}
    fact = qualification.post_merge_checks(material, repo, get_json=get)
    status = "observed" if ending == "complete" else "partial"
    assert fact["status"] == status and fact["absent_shas"] == fact["missing_shas"] == []
    assert len(paths) == (10 if ending == "budget" else 2)
    expected = [2] if ending == "wrong head" else [2, 1] if ending == "complete" else list(range(11, 1, -1))
    assert [r["attempt"] for r in fact["runs"]] == expected
    assert fact["runs"][0]["conclusion"] == "success"
    assert [r for r in fact["runs"] if r["conclusion"] == "failure"] == ([] if ending == "wrong head" else [
        {"sha": head, "status": "completed", "conclusion": "failure", "url": direct["html_url"],
         "run_id": 77, "attempt": red_attempt}])
    if ending == "budget":
        assert fact["reason"] == "GitHub response page budget reached."
    elif ending == "complete":
        paths.clear()
        cached = qualification.post_merge_checks({**material, 'github': fact}, repo, get_json=get)
        assert cached == {**fact, 'runs': [fact['runs'][0], {**fact['runs'][1], 'retained': True}]}
        assert len(paths) == 1


@pytest.mark.parametrize("dated, ending, status, request_count", [
    (False, "red push", "observed", 2),
    (False, "offline", "unavailable", 2),
    (False, "budget", "partial", 10),
    (True, "red push", "observed", 3),
    (True, "short", "unavailable", 2),
    (True, "only cancelled", "observed", 3),
    (True, "unread attempt", "partial", 4),
    (True, "red attempt", "observed", 4),
])
def test_a_later_run_at_the_same_commit_never_hides_its_earlier_direct_red(dated, ending, status, request_count):
    tip, earlier, base = (s * 40 for s in "cba")
    attempt = 2 if ending in ("unread attempt", "red attempt") else 1
    dispatch = {**_attempt(earlier, conclusion="cancelled", attempt=attempt), "id": 9, "event": "workflow_dispatch"}
    green = {**_attempt(tip), "id": 7}
    push = {**_attempt(earlier, conclusion="failure"), "id": 5}
    total = 1001 if ending == 'budget' else (301 if dated else 101)
    requests = []

    def get(url):
        requests.append(url)
        path = urlparse(url).path
        query = parse_qs(urlparse(url).query)
        if not path.endswith("/workflows/post-merge.yml/runs"):
            assert path.endswith("/actions/runs/9/attempts/1")
            if ending == "unread attempt":
                raise OSError("offline")
            assert ending == "red attempt"
            return {**dispatch, "run_attempt": 1, "conclusion": "failure"}
        page = int(query["page"][0])
        if "head_sha" in query:
            assert dated and query['branch'] == ['ouroboros'] and page == 1
            if query['head_sha'] == [tip]:
                return {'total_count': 1, 'workflow_runs': [green]}
            assert query['head_sha'] == [earlier]
            rows = [dispatch, push] if ending == "red push" else [dispatch]
            return {"total_count": 2 if ending == "short" else len(rows), "workflow_runs": rows}
        if page == 1 or ending == "budget":
            first = [dispatch, green] if page == 1 else []
            fillers = [{"id": page * 100 + n, "head_sha": "outside", "created_at": "2026-11-01T00:00:00Z"}
                       for n in range(100 - len(first))]
            return {"total_count": total, "workflow_runs": first + fillers}
        assert not dated and page == 2
        if ending == "offline":
            raise OSError("offline")
        assert ending == "red push"
        return {"total_count": total, "workflow_runs": [push]}

    commits = [{"sha": tip}, {"sha": earlier, **({"date": "2026-11-02T00:00:00Z"} if dated else {})}]
    cached = {"sha": tip, "run_id": 7, "attempt": 1, "status": "completed", "conclusion": "success",
              "measured_base": base}
    saved = {"repository": "owner/repo", "runs": [cached]}  # as a read of owner/repo saved it
    fact = qualification.post_merge_checks({"base_sha": base, "commits": commits, "github": saved},
                                           "https://github.com/owner/repo", get_json=get)
    expected = [(earlier, 9, attempt, "cancelled"), (tip, 7, 1, "success")]
    if ending == "red push":
        expected.append((earlier, 5, 1, "failure"))
    if ending == "red attempt":
        expected.append((earlier, 9, 1, "failure"))
    assert [(r["sha"], r["run_id"], r["attempt"], r["conclusion"]) for r in fact["runs"]] == expected
    assert fact["runs"][1]["measured_base"] == base
    assert [sha for run in fact["runs"] for sha in run.get("covered_shas", [])] == (
        [earlier] if ending == "only cancelled" else [])
    assert (fact["status"], fact["missing_shas"], fact["absent_shas"]) == (status, [], [])
    assert len(requests) == request_count and not any("/jobs?" in url for url in requests)


def test_a_red_read_before_stays_by_its_run_and_attempt_after_a_rerun_and_a_complete_listing_without_it(monkeypatch):
    head, base = "b" * 40, "a" * 40
    red = {**_attempt(head, conclusion="failure"), "id": 5, "run_attempt": 1}
    listing = [red]
    monkeypatch.setattr(qualification, "_get_json", lambda url: {"total_count": len(listing), "workflow_runs": listing})
    material = {"base_sha": base, "target_sha": head, "commits": [{"sha": head, "subject": "change"}]}
    authored = qualification.range_qualification(material, "https://github.com/owner/repo")
    assert [(r["attempt"], r["conclusion"]) for r in authored["github"]["runs"]] == [(1, "failure")]
    listing[:] = [{**red, "run_attempt": 2, "conclusion": "success"}]  # the run, re-run green
    rerun, changed = qualification.refresh_qualification(authored, "https://github.com/owner/repo")
    assert changed and [(r["attempt"], r["conclusion"]) for r in rerun["github"]["runs"]] == [(2, "success"), (1, "failure")]
    listing[:] = []  # a complete listing that no longer holds the run (retention, deletion)
    gone, changed = qualification.refresh_qualification(rerun, "https://github.com/owner/repo")
    assert changed and [(r["attempt"], r["conclusion"]) for r in gone["github"]["runs"]] == [(1, "failure")]
    assert gone["github"]["status"] == "observed" and gone["github"]["missing_shas"] == gone["github"]["absent_shas"] == []


def test_a_settled_attempt_is_never_read_back_as_running_after_the_listing_that_saw_it_first(monkeypatch):
    """The global list shows one attempt running; the exact lookup of its SHA, in the same read, shows that attempt
    failed. The settled row is the observation, and a stale 'running' of the same repository, SHA, run and attempt
    takes it back neither offline nor from a whole listing."""
    head, base = "b" * 40, "a" * 40
    link = "https://github.com/owner/repo/actions/runs/77"
    running = {**_attempt(head, "in_progress", None), "html_url": link}
    settled = {**running, "status": "completed", "conclusion": "failure"}
    requests = []

    def get(url):
        requests.append(url)
        query = parse_qs(urlparse(url).query)
        if "head_sha" in query:
            assert query["branch"] == ["ouroboros"] and query["head_sha"] == [head]
            return {"total_count": 1, "workflow_runs": [settled]}
        assert query["page"] == ["1"]
        fillers = [{"id": 1_000 + n, "head_sha": "outside", "created_at": "2026-11-01T00:00:00Z"} for n in range(99)]
        return {"total_count": 301, "workflow_runs": [running, *fillers]}

    repo = "https://github.com/owner/repo"
    monkeypatch.setattr(qualification, "_get_json", get)
    material = {"base_sha": base, "target_sha": head,
                "commits": [{"sha": head, "subject": "change", "date": "2026-11-02T00:00:00Z"}]}
    authored = qualification.range_qualification(material, repo)
    red = {"sha": head, "status": "completed", "conclusion": "failure", "url": link, "run_id": 77, "attempt": 1}
    seen = authored["github"]
    assert [parse_qs(urlparse(url).query).get("head_sha") for url in requests] == [None, [head]]
    assert (seen["status"], seen["runs"], seen["missing_shas"], seen["absent_shas"]) == ("observed", [red], [], [])
    # The next refresh cannot reach GitHub: the red stays, retained, and is not a new result.
    monkeypatch.setattr(qualification, "_get_json", _fail)
    offline, changed = qualification.refresh_qualification(authored, repo)
    assert not changed and offline["github"]["status"] == "unavailable"
    assert offline["github"]["runs"] == [{**red, "retained": True}]
    # A whole listing that still shows the attempt running is stale for that attempt: the settled row stays.
    stale = {"total_count": 1, "workflow_runs": [running]}
    monkeypatch.setattr(qualification, "_get_json", lambda url: stale)
    held, changed = qualification.refresh_qualification(offline, repo)
    assert not changed and held["github"]["status"] == "observed" and held["github"]["missing_shas"] == []
    assert held["github"]["runs"] == [{**red, "retained": True}]


@pytest.mark.parametrize("ending", ["settled", "foreign", "saved", "older", "offline", "nonterminal", "cut"])
def test_an_attempt_a_later_one_followed_is_settled_never_the_running_row_the_list_showed_first(ending, monkeypatch):
    """A run is re-run only once it has settled, so an attempt below the newest is settled, and the running row of it
    that the global list showed before the lookup of its SHA found the later attempt is stale. Only a settled row of
    that exact repository, SHA, run and attempt stands for it: the saved one, else its own endpoint. What neither
    supplies keeps the read partial, and nothing is read as settled, red or observed on the list's word."""
    head, base = "b" * 40, "a" * 40
    link, repo = "https://github.com/owner/repo/actions/runs/77", "https://github.com/owner/repo"
    stale_no = 2 if ending == "older" else 1  # the attempt shown running; the lookup of the SHA finds the next
    requests = []

    def api(attempt, conclusion):  # what GitHub answers for one attempt of the run
        return {**_attempt(head, "completed" if conclusion else "in_progress", conclusion, attempt=attempt),
                "html_url": link}

    def row(attempt, conclusion, **more):  # the same attempt as the reader records it
        return {"sha": head, "status": "completed" if conclusion else "in_progress", "conclusion": conclusion,
                "url": link, "run_id": 77, "attempt": attempt, **more}

    def get(url):
        requests.append(url)
        path, query = urlparse(url).path, parse_qs(urlparse(url).query)
        if path.endswith(f"/actions/runs/77/attempts/{stale_no}"):  # the exact endpoint of the stale attempt
            if ending == "offline":
                raise OSError("offline")
            return api(stale_no, None if ending == "nonterminal" else "failure")
        assert path.endswith("/workflows/post-merge.yml/runs")
        if "head_sha" in query:  # the lookup of the SHA: the run, at its later attempt
            assert query["branch"] == ["ouroboros"] and query["head_sha"] == [head]
            return {"total_count": 1, "workflow_runs": [api(stale_no + 1, "success")]}
        page = int(query["page"][0])  # the list: the stale attempt, then runs of other commits
        dated = "2026-11-01T00:00:00Z" if page == (9 if ending == "cut" else 1) else "2026-11-03T00:00:00Z"
        rows = [api(stale_no, None)] if page == 1 else []
        rows += [{"id": 1_000 * page + n, "head_sha": "outside", "created_at": dated} for n in range(100 - len(rows))]
        return {"total_count": 1_000 if ending == "cut" else 301, "workflow_runs": rows}

    saved = {"saved": {"repository": "owner/repo", "runs": [row(1, "failure")]},
             "older": {"repository": "owner/repo", "runs": [row(1, "failure")]},
             "foreign": {"repository": "example/fork", "runs": [row(1, "success")]}}.get(ending, {})
    known = {"settled": [row(1, "failure"), row(2, "success")],  # endings in which the earlier outcome is known
             "saved": [row(1, "failure", retained=True), row(2, "success")],
             "older": [row(2, "failure"), row(3, "success"), row(1, "failure", retained=True)]}
    known["foreign"] = known["settled"]  # another repository's saved row stands for nothing here
    monkeypatch.setattr(qualification, "_get_json", get)
    material = {"base_sha": base, "target_sha": head, "github": saved,
                "commits": [{"sha": head, "subject": "change", "date": "2026-11-02T00:00:00Z"}]}
    first, _ = qualification.refresh_qualification(material, repo)
    seen = first["github"]
    assert (seen["status"], seen["missing_shas"], seen["absent_shas"]) == (
        "observed" if ending in known else "partial", [], [])
    assert seen["runs"] == known.get(ending, [row(1, None, retained=True), row(2, "success")])  # the old observation
    assert (len(requests), sum("/attempts/" in url for url in requests)) == {
        "saved": (2, 0), "cut": (10, 0)}.get(ending, (3, 1))
    if ending in known:  # the red stays through an outage, retained, and is no new result
        monkeypatch.setattr(qualification, "_get_json", _fail)
        offline, changed = qualification.refresh_qualification(first, repo)
        assert not changed and offline["github"]["status"] == "unavailable"
        assert offline["github"]["runs"] == [{**run, "retained": True} for run in known[ending]]
    else:  # nothing is read as red or settled; the read says what it could not read
        assert seen["reason"] == {"offline": "OSError", "nonterminal": "ValueError",
                                  "cut": "GitHub response page budget reached."}[ending]


def _post_merge_row(head, link, attempt, conclusion, **more):
    """One attempt of run 77 as the reader records it."""
    return {"sha": head, "status": "completed" if conclusion else "in_progress", "conclusion": conclusion,
            "url": link, "run_id": 77, "attempt": attempt, **more}


def _running_then_later(head, link, *, later, base=None, endpoint=None, total=301):
    """GitHub as the reader meets it: the list shows attempt 1 of run 77 running, the lookup of its SHA finds attempt 2
    (`later`), the other commits have no run, and attempt 1's own endpoint (`endpoint`) and jobs (named for `base`)
    answer only where the test expects that request."""
    requests = []

    def api(attempt, conclusion):
        return {**_attempt(head, "completed" if conclusion else "in_progress", conclusion, attempt=attempt),
                "html_url": link}

    def get(url):
        requests.append(url)
        path, query = urlparse(url).path, parse_qs(urlparse(url).query)
        if path.endswith("/attempts/1/jobs"):
            assert base, "no jobs are read here"
            return {"total_count": 2, "jobs": _jobs(api(1, "failure"), base)}
        if path.endswith("/actions/runs/77/attempts/1"):
            assert endpoint, "attempt 1 is not read here"
            return api(1, endpoint)
        assert path.endswith("/workflows/post-merge.yml/runs")
        if "head_sha" in query:
            return {"total_count": 1, "workflow_runs": [api(2, later)]} if query["head_sha"] == [head] else {
                "total_count": 0, "workflow_runs": []}
        fillers = [{"id": 1_000 + n, "head_sha": "outside", "created_at": "2026-11-01T00:00:00Z"} for n in range(99)]
        return {"total_count": total, "workflow_runs": [api(1, None), *fillers]}

    return get, requests


@pytest.mark.parametrize("saved", ["failure", "success"])
def test_a_saved_settled_attempt_the_list_showed_running_still_lets_its_exact_jobs_cover_a_new_absence(saved):
    """Attempt 1 is saved settled without coverage. This read lists it running first, finds a cancelled attempt 2 and no
    run for the commit before it. The saved outcome, not the listing's raw row, decides that attempt 1's own jobs are
    looked up inside the same ten requests, so its measured base covers the absence. Its outcome stays what was read
    (failure and success stay distinct), it is marked retained rather than fresh, and an outage keeps all of it."""
    head, gap, base = "b" * 40, "c" * 40, "a" * 40
    link, repo = "https://github.com/owner/repo/actions/runs/77", "https://github.com/owner/repo"
    get, requests = _running_then_later(head, link, later="cancelled", base=base)
    commits = [{"sha": head}, {"sha": gap, "date": "2026-11-02T00:00:00Z"}]
    record = {"repository": "owner/repo", "runs": [_post_merge_row(head, link, 1, saved)]}
    fact = qualification.post_merge_checks({"base_sha": base, "commits": commits, "github": record}, repo, get_json=get)
    covering = _post_merge_row(head, link, 1, saved, retained=True, measured_base=base, covered_shas=[gap])
    assert fact["runs"] == [covering, _post_merge_row(head, link, 2, "cancelled")]
    assert (fact["status"], fact["absent_shas"], fact["missing_shas"]) == ("observed", [gap], [])
    assert len(requests) == 4 and sum("/attempts/1/jobs" in url for url in requests) == 1
    assert not any(urlparse(url).path.endswith("/attempts/1") for url in requests), "the saved row spares the endpoint"
    down = qualification.post_merge_checks({"base_sha": base, "commits": commits, "github": fact}, repo, get_json=_fail)
    assert down["status"] == "unavailable" and (down["absent_shas"], down["missing_shas"]) == ([gap], [])
    assert down["runs"] == [covering, {**_post_merge_row(head, link, 2, "cancelled"), "retained": True}]


@pytest.mark.parametrize("owner", ["example/fork", None])
def test_a_saved_settled_attempt_of_another_or_no_known_repository_is_not_this_ones(owner):
    """The same listing, but the saved success of attempt 1 was read in another repository, or in none that is known:
    it stands for nothing here. The exact endpoint is read within the budget and its failure is the fresh observation."""
    head, gap, base = "b" * 40, "c" * 40, "a" * 40
    link, repo = "https://github.com/owner/repo/actions/runs/77", "https://github.com/owner/repo"
    get, requests = _running_then_later(head, link, later="cancelled", base=base, endpoint="failure")
    commits = [{"sha": head}, {"sha": gap, "date": "2026-11-02T00:00:00Z"}]
    record = {**({"repository": owner} if owner else {}), "runs": [_post_merge_row(head, link, 1, "success")]}
    fact = qualification.post_merge_checks({"base_sha": base, "commits": commits, "github": record}, repo, get_json=get)
    assert fact["repository"] == "owner/repo"
    assert [run for run in fact["runs"] if run["attempt"] == 1] == [
        _post_merge_row(head, link, 1, "failure", measured_base=base, covered_shas=[gap])]
    assert sum(urlparse(url).path.endswith("/actions/runs/77/attempts/1") for url in requests) == 1
    assert len(requests) == 5 and (fact["status"], fact["absent_shas"], fact["missing_shas"]) == ("observed", [gap], [])


def test_an_earlier_attempt_the_budget_left_unread_is_no_job_running_now_but_the_open_latest_is():
    """The list shows attempt 1 running; the lookups of nine commits then find attempt 2 still open and spend the budget,
    so attempt 1's own endpoint cannot be read. Attempt 1 keeps its exact link and old observation, marked previously
    observed, and the summary counts only attempt 2 as running; its final outcome is unknown, the read partial."""
    head = "b" * 40
    link, repo = "https://github.com/owner/repo/actions/runs/77", "https://github.com/owner/repo"
    others = [f"{n:040x}" for n in range(1, 9)]
    commits = [{"sha": head}, *({"sha": sha} for sha in others[:-1]), {"sha": others[-1], "date": "2026-11-02T00:00:00Z"}]
    get, requests = _running_then_later(head, link, later=None, total=1_000)
    fact = qualification.post_merge_checks({"base_sha": "a" * 40, "commits": commits}, repo, get_json=get)
    assert (fact["status"], fact["reason"]) == ("partial", "GitHub response page budget reached.")
    assert fact["runs"] == [_post_merge_row(head, link, 1, None, retained=True), _post_merge_row(head, link, 2, None)]
    assert len(requests) == 10 and not any("/attempts/" in url for url in requests)
    summary = qualification.qualification_summary({"github": fact})["post_merge"]
    assert summary["runs"] == {"in_progress": 1, "unknown": 1} and summary["retained"] == {"unknown": 1}


def _github_attempts(head, link, *, listed, lookup=None, endpoint='unexpected', base=None, total=1):
    """GitHub as the reader meets it for run 77 of `head`. The list shows attempt `listed` (number, conclusion) first
    (with total > 1, then older runs of other commits, so every SHA is looked up); `lookup` (default: the same row)
    answers for the head and nothing for the other commits. Attempt 1's own endpoint answers `endpoint` (a conclusion,
    None while it runs, or an exception to raise) and its jobs name `base` (or raise it); a request the test does not
    expect fails it."""
    requests = []

    def api(attempt, conclusion):
        return {**_attempt(head, 'completed' if conclusion else 'in_progress', conclusion, attempt=attempt),
                'html_url': link}

    def get(url):
        requests.append(url)
        path, query = urlparse(url).path, parse_qs(urlparse(url).query)
        if path.endswith('/attempts/1/jobs'):
            assert base, 'no jobs are read here'
            if isinstance(base, Exception):
                raise base
            return {'total_count': 2, 'jobs': _jobs(api(1, 'failure'), base)}
        if path.endswith('/actions/runs/77/attempts/1'):
            assert endpoint != 'unexpected', 'attempt 1 is not read here'
            if isinstance(endpoint, Exception):
                raise endpoint
            return api(1, endpoint)
        assert path.endswith('/workflows/post-merge.yml/runs')
        if 'head_sha' in query:
            rows = [api(*(lookup or listed))] if query['head_sha'] == [head] else []
            return {'total_count': len(rows), 'workflow_runs': rows}
        fillers = [{'id': 1_000 + n, 'head_sha': 'outside', 'created_at': '2026-11-01T00:00:00Z'}
                   for n in range(99 if total > 1 else 0)]
        return {'total_count': total, 'workflow_runs': [api(*listed), *fillers]}

    return get, requests


@pytest.mark.parametrize('ending', ['transport', 'ambiguous', 'settled', 'foreign', 'unnamed'])
def test_a_saved_unsettled_attempt_stays_as_last_observed_when_only_a_newer_one_is_listed(ending):
    """Attempt 1 of run 77 was saved running; this read lists only attempt 2 (the list shows a run's latest attempt) and
    reads attempt 1's own endpoint. Unless that read settles it, attempt 1 stays with its exact link and last status,
    retained: its final outcome is unknown, not running now, in the summary and through an outage. A saved row of
    another or of no known repository stands for nothing here."""
    head = 'b' * 40
    link, repo = 'https://github.com/owner/repo/actions/runs/77', 'https://github.com/owner/repo'
    answer = {'ambiguous': None, 'settled': 'failure'}.get(ending, OSError('offline'))
    get, requests = _github_attempts(head, link, listed=(2, 'success'), endpoint=answer)
    named = {'foreign': {'repository': 'example/fork'}, 'unnamed': {}}.get(ending, {'repository': 'owner/repo'})
    material = {'base_sha': 'a' * 40, 'commits': [{'sha': head}],
                'github': {**named, 'runs': [_post_merge_row(head, link, 1, None)]}}
    fact = qualification.post_merge_checks(material, repo, get_json=get)
    newer, kept = _post_merge_row(head, link, 2, 'success'), _post_merge_row(head, link, 1, None, retained=True)
    expected = {'transport': [newer, kept], 'ambiguous': [newer, kept],
                'settled': [newer, _post_merge_row(head, link, 1, 'failure')]}.get(ending, [newer])
    assert fact['runs'] == expected
    assert (fact['status'], fact['missing_shas'], fact['absent_shas']) == (
        'observed' if ending == 'settled' else 'partial', [], [])
    assert len(requests) == 2 and sum(urlparse(url).path.endswith('/attempts/1') for url in requests) == 1
    if kept in expected:
        assert fact['reason'] == ('ValueError' if ending == 'ambiguous' else 'OSError')
        summary = qualification.qualification_summary({'github': fact})['post_merge']
        assert summary['runs'] == {'success': 1, 'unknown': 1} and summary['retained'] == {'unknown': 1}
        down = qualification.post_merge_checks({**material, 'github': fact}, repo, get_json=_fail)
        assert down['status'] == 'unavailable'
        assert down['runs'] == [{**row, 'retained': True} for row in expected]


def test_a_saved_unsettled_attempt_the_budget_left_unread_stays_as_last_observed_beside_the_open_latest():
    """The list shows attempt 2 still open; the lookups of nine commits then spend all ten requests, so saved attempt 1
    cannot be read. It stays with its exact link, retained and unknown, and only attempt 2 counts as running."""
    head = 'b' * 40
    link, repo = 'https://github.com/owner/repo/actions/runs/77', 'https://github.com/owner/repo'
    others = [f'{n:040x}' for n in range(1, 9)]
    commits = [{'sha': head}, *({'sha': sha} for sha in others[:-1]),
               {'sha': others[-1], 'date': '2026-11-02T00:00:00Z'}]
    get, requests = _github_attempts(head, link, listed=(2, None), total=1_000)
    saved = {'repository': 'owner/repo', 'runs': [_post_merge_row(head, link, 1, None)]}
    fact = qualification.post_merge_checks({'base_sha': 'a' * 40, 'commits': commits, 'github': saved}, repo,
                                           get_json=get)
    assert (fact['status'], fact['reason']) == ('partial', 'GitHub response page budget reached.')
    assert fact['runs'] == [_post_merge_row(head, link, 2, None), _post_merge_row(head, link, 1, None, retained=True)]
    assert len(requests) == 10 and not any('/attempts/' in url for url in requests)
    assert head not in fact['missing_shas'], 'a commit with rows is never listed as unobserved'
    summary = qualification.qualification_summary({'github': fact})['post_merge']
    assert summary['runs'] == {'in_progress': 1, 'unknown': 1} and summary['retained'] == {'unknown': 1}


@pytest.mark.parametrize('ending', ['whole', 'running'])
def test_a_lagging_listing_neither_drops_the_known_settled_later_attempt_nor_makes_the_earlier_one_current(ending):
    """Attempt 2 was read settled; this whole listing still shows only attempt 1. The highest attempt known is an
    identity horizon, not freshness: saved attempt 2 stays, retained; a listed attempt 1 that had not settled is an old
    observation, outcome unknown, never the run's current state, while one that settled stays this read's own."""
    head = 'b' * 40
    link, repo = 'https://github.com/owner/repo/actions/runs/77', 'https://github.com/owner/repo'
    whole = ending == 'whole'
    get, requests = _github_attempts(head, link, listed=(1, 'failure' if whole else None))
    material = {'base_sha': 'a' * 40, 'commits': [{'sha': head}],
                'github': {'repository': 'owner/repo', 'runs': [_post_merge_row(head, link, 2, 'success')]}}
    fact = qualification.post_merge_checks(material, repo, get_json=get)
    earlier = _post_merge_row(head, link, 1, 'failure') if whole else _post_merge_row(head, link, 1, None, retained=True)
    assert fact['runs'] == [earlier, _post_merge_row(head, link, 2, 'success', retained=True)]
    assert (fact['status'], fact['missing_shas'], fact['absent_shas']) == ('observed', [], []) and len(requests) == 1
    summary = qualification.qualification_summary({'github': fact})['post_merge']
    assert summary['runs'] == ({'failure': 1, 'success': 1} if whole else {'unknown': 1, 'success': 1})
    assert summary['retained'] == ({'success': 1} if whole else {'unknown': 1, 'success': 1})
    down = qualification.post_merge_checks({**material, 'github': fact}, repo, get_json=_fail)
    assert down['status'] == 'unavailable' and down['runs'] == [{**run, 'retained': True} for run in fact['runs']]


@pytest.mark.parametrize('ending', ['failure', 'success', 'transport', 'budget'])
def test_a_saved_settled_attempt_over_a_stale_listing_row_binds_its_status_to_its_own_jobs_lookup(ending):
    """Attempt 1 was saved settled without a measured base; the list shows it running, no newer attempt, and a commit
    before it has no run. The saved outcome is bound to the attempt's own source, so the existing jobs lookup (within
    the ten requests) can cover the absence. A failed or unaffordable lookup leaves it missing and the read partial,
    and the saved row's own cover claim is never kept."""
    head, gap, base = 'b' * 40, 'c' * 40, 'a' * 40
    link, repo = 'https://github.com/owner/repo/actions/runs/77', 'https://github.com/owner/repo'
    others, dated = [f'{n:040x}' for n in range(1, 9)], '2026-11-02T00:00:00Z'
    budget = ending == 'budget'
    commits = ([{'sha': head}, *({'sha': sha} for sha in others[:-1]), {'sha': others[-1], 'date': dated}] if budget
               else [{'sha': head}, {'sha': gap, 'date': dated}])
    get, requests = _github_attempts(head, link, listed=(1, None), total=1_000 if budget else 1,
                                     base=OSError('offline') if ending == 'transport' else base)
    settled = 'success' if ending == 'success' else 'failure'
    saved = {'repository': 'owner/repo', 'runs': [_post_merge_row(head, link, 1, settled, covered_shas=[gap])]}
    fact = qualification.post_merge_checks({'base_sha': base, 'commits': commits, 'github': saved}, repo, get_json=get)
    covered = ending in ('failure', 'success')
    unmeasured = others if budget else [gap]
    row = _post_merge_row(head, link, 1, settled, retained=True,
                          **({'measured_base': base, 'covered_shas': [gap]} if covered else {}))
    assert fact['runs'] == [row]
    assert (fact['absent_shas'], fact['missing_shas']) == (unmeasured, [] if covered else unmeasured)
    assert (fact['status'], fact['reason']) == (
        ('observed', 'No matching run is not a passing check.') if covered
        else ('partial', 'GitHub response page budget reached.' if budget else 'OSError'))
    assert len(requests) == (10 if budget else 2)
    assert sum('/attempts/1/jobs' in url for url in requests) == int(not budget)


@pytest.mark.parametrize("cut", ["attempt", "sha query"])
def test_a_read_the_request_budget_cut_short_is_partial_never_observed_or_unavailable(cut):
    """The ten requests also pay for attempt metadata and per-commit queries: whatever they leave unread keeps
    the read partial, and coverage it would have proved stays unproven."""
    head, gap, base = "c" * 40, "b" * 40, "a" * 40
    oldest = "2026-11-01T00:00:00Z"

    def get(url):
        query = parse_qs(urlparse(url).query)
        page = int(query["page"][0])
        if "head_sha" in query:  # the gap's own runs: more pages than the budget leaves
            return {"total_count": 10_000, "workflow_runs": [
                {"id": 10_000 + page * 100 + n, "head_sha": gap, "status": "completed", "conclusion": "cancelled"}
                for n in range(100)]}
        assert "/jobs?" not in url, "the attempt read has no request left"
        rows = [*([_attempt(head)] if page == 1 else []), *[
            {"id": 1_000 + page * 100 + n, "head_sha": "outside", "created_at": "2026-10-01T00:00:00Z" if cut == "sha query"
             else "2026-11-02T00:00:00Z"} for n in range(100 - (page == 1))]]
        return {"total_count": 1_000 if cut == "attempt" else 10_000, "workflow_runs": rows}

    commits = [{"sha": head}, {"sha": gap, "date": oldest}]
    fact = qualification.post_merge_checks({"base_sha": base, "commits": commits}, "https://github.com/owner/repo",
                                           get_json=get)
    assert (fact["status"], fact["reason"]) == ("partial", "GitHub response page budget reached.")
    # A complete listing still proves the gap has no run; its cut-short query proves nothing about the rest.
    assert fact["missing_shas"] == fact["absent_shas"] == ([gap] if cut == "attempt" else [])
    assert not any(r.get("covered_shas") for r in fact["runs"])


@pytest.mark.parametrize('ending', ['exact', 'global', 'budget'])
def test_successful_late_dispatch_does_not_finish_the_history_of_its_sha(ending):
    head, newer = (s * 40 for s in 'cb')
    dispatch = {**_attempt(head), 'id': 20, 'event': 'workflow_dispatch', 'created_at': '2026-11-04T00:00:00Z'}
    second = {**_attempt(newer), 'id': 21}
    red = {**_attempt(head, conclusion='failure'), 'id': 10}
    commits = [{'sha': newer}, {'sha': head, 'date': '2026-11-05T00:00:00Z'}]
    if ending == 'budget':
        commits[:0] = [{'sha': f'{n:040x}'} for n in range(10)]
    calls = []

    def get(url):
        calls.append(url)
        query = parse_qs(urlparse(url).query)
        page = int(query['page'][0])
        if 'head_sha' in query:
            assert ending == 'exact' and query['branch'] == ['ouroboros'] and page == 1
            sha = query['head_sha'][0]
            assert sha in (head, newer)
            rows = [dispatch, red] if sha == head else [second]
            return {'total_count': len(rows), 'workflow_runs': rows}
        if ending == 'global' and page == 2:
            return {'total_count': 101, 'workflow_runs': [red]}
        assert page == 1 or ending == 'budget'
        rows = [dispatch, second] if page == 1 else []
        rows.extend({'id': 100 * page + i, 'head_sha': 'outside'} for i in range(100 - len(rows)))
        return {'total_count': {'global': 101, 'exact': 501, 'budget': 1500}[ending], 'workflow_runs': rows}

    fact = qualification.post_merge_checks(commits, 'https://github.com/owner/repo', get_json=get)
    assert len(calls) == {'exact': 3, 'global': 2, 'budget': 10}[ending]
    actual = [(r['run_id'], r['conclusion']) for r in fact['runs'] if r['sha'] == head]
    assert actual == ([(20, 'success')] if ending == 'budget' else [(20, 'success'), (10, 'failure')])
    assert fact['status'] == ('partial' if ending == 'budget' else 'observed')
    assert fact['absent_shas'] == [] and not any(r.get('covered_shas') for r in fact['runs'])


@pytest.mark.parametrize('conclusion', ['failure', 'success', 'cancelled'])
def test_terminal_attempt_one_survives_zero_budget_below_attempt_twelve(conclusion):
    head = 'b' * 40
    link = 'https://github.com/owner/repo/actions/runs/77'
    cached = {'sha': head, 'status': 'completed', 'conclusion': conclusion, 'url': link, 'run_id': 77, 'attempt': 1}
    stale = {**cached, 'attempt': 12, 'status': 'in_progress', 'conclusion': None}
    calls = []

    def get(url):
        calls.append(url)
        page = int(parse_qs(urlparse(url).query)['page'][0])
        rows = [{**_attempt(head, attempt=12), 'html_url': link}] if page == 1 else []
        rows.extend({'id': 100 * page + i, 'head_sha': 'outside'} for i in range(100 - len(rows)))
        return {'total_count': 1000, 'workflow_runs': rows}

    material = {'commits': [{'sha': head}], 'github': {'repository': 'owner/repo', 'runs': [stale, cached]}}
    fact = qualification.post_merge_checks(material, 'https://github.com/owner/repo', get_json=get)
    assert len(calls) == 10 and not any('/attempts/' in url for url in calls)
    assert (fact['status'], fact['reason']) == ('partial', 'GitHub response page budget reached.')
    assert fact['runs'] == [{**cached, 'attempt': 12, 'conclusion': 'success'}, {**cached, 'retained': True}]
    compact = qualification.compact_qualification({'github': fact})['github']['runs']
    assert compact['retained'] == {conclusion: 1} and not compact['unabridged']
    assert qualification.qualification_summary({'github': fact})['post_merge']['retained'] == {conclusion: 1}


@pytest.mark.parametrize('mode', ['offline', 'partial', 'red', 'nonterminal', 'unread attempt', 'legacy', 'never proved'])
def test_only_an_actual_interrupted_cover_survives_a_short_read(mode, monkeypatch):
    tip, middle, base = (s * 40 for s in 'cba')
    witness = _attempt(tip)
    stopped = {**_attempt(middle, conclusion='cancelled'), 'id': 5}
    repo = 'https://github.com/owner/repo'
    commits = [{'sha': s, 'subject': s} for s in (tip, middle)]

    def initial(url):
        if '/jobs?' in url:
            if mode == 'never proved':
                raise OSError('jobs offline')
            return {'total_count': 2, 'jobs': _jobs(witness, base)}
        return {'total_count': 2, 'workflow_runs': [witness, stopped]}

    monkeypatch.setattr(qualification, '_get_json', initial)
    first = qualification.range_qualification({'commits': commits, 'base_sha': base, 'target_sha': tip}, repo)
    assert first['github']['runs'][0].get('covered_shas', []) == ([] if mode == 'never proved' else [middle])
    if mode == 'legacy':
        first['github'].pop('interrupted_shas')
        passive = {'qualification': json.loads(json.dumps(first))}
        assert qualification.discard_legacy_coverage(passive)
        assert not passive['qualification']['github']['runs'][0].get('covered_shas')

    def later(url):
        if mode in ('offline', 'legacy'):
            raise OSError('offline')
        if '/jobs?' in url:
            return {'total_count': 2, 'jobs': _jobs(witness, base)}
        if url.endswith('/attempts/2'):
            raise OSError('unknown intervening outcome')
        if parse_qs(urlparse(url).query)['page'] == ['2']:
            raise OSError('no second page')
        direct = ({**stopped, 'id': 8, 'conclusion': 'failure'} if mode == 'red' else
                  {**stopped, 'run_attempt': 2, 'status': 'in_progress', 'conclusion': None} if mode == 'nonterminal' else
                  {**stopped, 'run_attempt': 3} if mode == 'unread attempt' else stopped)
        return {'total_count': 101, 'workflow_runs': [witness, direct, *[{'head_sha': 'outside'}] * 98]}

    monkeypatch.setattr(qualification, '_get_json', later)
    updated, changed = qualification.refresh_qualification(first, repo)
    seen = updated['github']
    keep = mode in ('offline', 'partial')
    assert seen['status'] == 'unavailable' and not seen['absent_shas'] and not seen['missing_shas']
    assert seen['interrupted_shas'] == ([middle] if keep else [])
    assert [(r['run_id'], r['attempt'], r['covered_shas']) for r in seen['runs'] if r.get('covered_shas')] == (
        [(77, 1, [middle])] if keep else [])
    if keep:
        assert not changed
    elif mode != 'never proved':
        assert changed


def test_a_saved_higher_attempt_over_a_stale_listing_blocks_the_all_interrupted_cover():
    """The saved record of the same repository knows attempt 3 of run 5, with attempt 2 never read, while this
    whole listing still shows the stale interrupted attempt 1. After the join, grouped attempt counts show the
    gap below the highest known attempt: it is unread, so the commit is not covered as all-interrupted, the read
    over a complete listing is partial, and the saved attempt 3 stays retained beside the listed one — all with
    no request beyond the listing, because a saved horizon never buys one."""
    tip, middle, base = (s * 40 for s in 'cba')
    link, repo = 'https://github.com/owner/repo/actions/runs/77', 'https://github.com/owner/repo'
    witness = _attempt(tip)
    stale = {**_attempt(middle, conclusion='cancelled'), 'id': 5, 'run_attempt': 1}
    requests = []

    def get(url):
        requests.append(url)
        if '/jobs?' in url:
            return {'total_count': 2, 'jobs': _jobs(witness, base)}
        assert url.endswith('/workflows/post-merge.yml/runs?per_page=100&page=1')
        return {'total_count': 2, 'workflow_runs': [witness, stale]}

    saved = {'repository': 'owner/repo', 'runs': [_post_merge_row(middle, link, 3, 'cancelled', run_id=5)]}
    material = {'base_sha': base, 'commits': [{'sha': tip}, {'sha': middle}], 'github': saved}
    fact = qualification.post_merge_checks(material, repo, get_json=get)
    assert len(requests) == 1 and not any('/attempts/' in url for url in requests)
    assert [(r['run_id'], r['attempt'], r['conclusion'], r.get('retained')) for r in fact['runs']] == [
        (77, 1, 'success', None), (5, 1, 'cancelled', None), (5, 3, 'cancelled', True)]
    assert 'measured_base' not in fact['runs'][0] and 'covered_shas' not in fact['runs'][0]
    assert (fact['status'], fact['reason'], fact['interrupted_shas'], fact['missing_shas'], fact['absent_shas']) == (
        'partial', "An attempt below a run's highest known one was never read.", [], [], [])

    # Ordinary check: the saved record also supplying attempt 2 reconciles the span, and the cover stands.
    saved2 = {'repository': 'owner/repo', 'runs': [_post_merge_row(middle, link, attempt, 'cancelled', run_id=5)
                                                    for attempt in (2, 3)]}
    requests.clear()
    reconciled = qualification.post_merge_checks(
        {'base_sha': base, 'commits': [{'sha': tip}, {'sha': middle}], 'github': saved2}, repo, get_json=get)
    assert len(requests) == 2 and sum('/jobs?' in url for url in requests) == 1
    assert reconciled['runs'][0]['measured_base'] == base and reconciled['runs'][0]['covered_shas'] == [middle]
    assert (reconciled['status'], reconciled['interrupted_shas'], reconciled['missing_shas']) == ('observed', [middle], [])


@pytest.mark.parametrize("range_fields", [{}, {"base_sha": "base"}])
def test_reading_a_saved_unmeasured_range_preserves_direct_reds_offline(tmp_path, monkeypatch, range_fields):
    commits = [{"sha": s} for s in ("head", "replaced", "absent")]
    runs = [{"sha": "head", "status": "completed", "conclusion": "failure", "covered_shas": ["replaced", "absent"]},
            {"sha": "replaced", "status": "completed", "conclusion": "cancelled"}]
    facts = {**range_fields, "commits": commits, "github": {"runs": runs, "absent_shas": ["absent"], "missing_shas": []}}
    path = letter.record_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    key = {"base_sha": "a" * 40, "target_sha": "b" * 40}  # a record the strict reader accepts
    path.write_text(json.dumps({"key": key, "qualification": facts, "last_good": {"qualification": facts}}))
    saved = path.read_bytes()
    monkeypatch.setattr(qualification, "_get_json", lambda *_: pytest.fail("a passive read cannot fetch"))
    for record in (letter.read_record(tmp_path), letter.read_record(tmp_path, strict=True)):
        for source in (record, record["last_good"]):
            observed = source["qualification"]["github"]
            assert observed["missing_shas"] == ["absent"] and "covered_shas" not in observed["runs"][0]
            assert [r["conclusion"] for r in observed["runs"]] == ["failure", "cancelled"]
    assert path.read_bytes() == saved


def _run_listing(hub):
    """``_get_json`` of one repository for a check: no open reds, and for the post-merge runs ``hub["listing"]``
    whole, or an outage while it is None. ``hub["reads"]`` counts the run reads."""
    def get(url):
        if "/issues?" in url:
            return []
        hub["reads"] = hub.get("reads", 0) + 1
        if hub["listing"] is None:
            raise OSError("offline")
        return {"total_count": len(hub["listing"]), "workflow_runs": list(hub["listing"])}
    return get


def test_github_availability_and_new_runs_never_touch_the_saved_text(tmp_path, monkeypatch):
    """Through the reader itself: only GitHub's answers are synthetic, so an outage keeps the saved run because the
    reader's join kept it, not because a stub above that join handed it back."""
    base, target = "a" * 40, "b" * 40
    material = {"base_sha": base, "target_sha": target, "commits": [{"sha": target, "subject": "change"}]}
    monkeypatch.setattr(letter, "collect_range_material", lambda *a, **k: dict(material))
    monkeypatch.setattr(letter, "_default_git", lambda: lambda _: (1, "", ""))
    hub = {"listing": [{"head_sha": target, "status": "completed", "conclusion": "success"}]}
    monkeypatch.setattr(qualification, "_get_json", _run_listing(hub))
    status = _status(official_repo_url="https://github.com/owner/repo")
    letter.refresh_after_check(status, drive_root=tmp_path)
    letter.save_description("the saved text", base, target, 0, author={"task_id": "t", "model": "m"}, drive_root=tmp_path)
    first = letter.read_record(tmp_path)
    seen = first["qualification"]["github"]
    assert seen["status"] == "observed" and [(run["sha"], run["conclusion"]) for run in seen["runs"]] == [(target, "success")]
    hub["listing"] = None
    offline = letter.refresh_after_check(status, drive_root=tmp_path)
    assert (offline["text"], offline["revision"], offline["written_at"]) == ("the saved text", 1, first["written_at"])
    assert (offline["qualification"]["github"]["status"], offline["qualification"]["github"]["reason"]) == (
        "unavailable", "OSError")
    assert offline["qualification"]["github"]["runs"] == [{**seen["runs"][0], "retained": True}]
    hub["listing"] = [{"head_sha": target, "status": "completed", "conclusion": "failure"}]
    changed = letter.refresh_after_check(status, drive_root=tmp_path)
    assert [(run["conclusion"], run.get("retained")) for run in changed["qualification"]["github"]["runs"]] == [
        ("failure", None)]
    assert (changed["text"], changed["revision"]) == ("the saved text", 1)
    assert letter.project_letter(changed, head_sha=base, latest_sha=target)["description_current"] is True


def test_a_source_without_a_github_reader_requests_nothing_and_keeps_what_was_read_before(monkeypatch):
    """An update source with no GitHub reader is a read that cannot happen, never an empty one: it ends at the reader's
    own join, so the run, the proven absence and the measured cover an earlier read recorded stay, marked retained."""
    head, gap, base = (s * 40 for s in "cba")
    witness = _attempt(head)
    material = {"base_sha": base, "commits": [{"sha": head}, {"sha": gap}]}
    fact = qualification.post_merge_checks(
        material, "https://github.com/owner/repo",
        get_json=lambda url: {"total_count": 2, "jobs": _jobs(witness, base)} if "/jobs?" in url else {
            "total_count": 1, "workflow_runs": [witness]})
    assert (fact["status"], fact["absent_shas"], fact["missing_shas"]) == ("observed", [gap], [])
    assert fact["runs"][0]["covered_shas"] == [gap]

    def refuse(url):
        pytest.fail(f"a source without a reader sends no request: {url}")

    monkeypatch.setattr(qualification, "_get_json", refuse)
    for source in ("https://gitlab.com/owner/repo", ""):
        kept = qualification.post_merge_checks({**material, "github": fact}, source, get_json=refuse)
        assert (kept["status"], kept["reason"]) == ("unavailable", "The update source has no supported GitHub check reader.")
        assert kept["runs"] == [{**fact["runs"][0], "retained": True}]
        assert (kept["absent_shas"], kept["interrupted_shas"], kept["missing_shas"]) == ([gap], [], [])
        fresh = qualification.post_merge_checks(material, source, get_json=refuse)  # nothing read before: nothing invented
        assert (fresh["status"], fresh["runs"], fresh["absent_shas"], fresh["missing_shas"]) == (
            "unavailable", [], [], sorted([head, gap]))
    refreshed, changed = qualification.refresh_qualification(
        {**material, "target_sha": head, "github": fact}, "https://gitlab.com/owner/repo")
    assert not changed and refreshed["github"]["runs"] == [{**fact["runs"][0], "retained": True}]


def test_rows_join_only_within_the_repository_that_observed_them(monkeypatch):
    """A red, a proven absence and the cover measured over it were read in owner/repo. Runs join by SHA, run and
    attempt, which name something in one repository only. A failed read of that repository (whatever the case of its
    address) and a source without a reader are no change of repository: every row stays, marked retained, under the
    name of the repository that observed it. The same range read from another known repository starts over, whether
    its own read fails or answers whole, also through the refresh the update check runs."""
    head, gap, base = (s * 40 for s in "cba")
    witness = _attempt(head, conclusion="failure")
    material = {"base_sha": base, "target_sha": head, "commits": [{"sha": head}, {"sha": gap}]}
    official, fork = "https://github.com/owner/repo", "https://github.com/example/fork"
    seen = qualification.post_merge_checks(
        material, official, get_json=lambda url: {"total_count": 2, "jobs": _jobs(witness, base)} if "/jobs?" in url
        else {"total_count": 1, "workflow_runs": [witness]})
    assert (seen["repository"], seen["status"], seen["absent_shas"], seen["missing_shas"]) == (
        "owner/repo", "observed", [gap], [])
    assert seen["runs"][0]["conclusion"] == "failure" and seen["runs"][0]["covered_shas"] == [gap]
    saved, kept = {**material, "github": seen}, [{**seen["runs"][0], "retained": True}]
    for source in (official, "https://github.com/Owner/Repo.git", "https://gitlab.com/owner/repo", ""):
        same = qualification.post_merge_checks(saved, source, get_json=_fail)
        assert (same["status"], same["runs"], same["absent_shas"], same["missing_shas"]) == (
            "unavailable", kept, [gap], []), source
        assert same["repository"].lower() == "owner/repo", source
    for answer in (_fail, lambda url: {"total_count": 0, "workflow_runs": []}):
        other = qualification.post_merge_checks(saved, fork, get_json=answer)
        assert (other["repository"], other["runs"], other["interrupted_shas"]) == ("example/fork", [], [])
        assert other["missing_shas"] == sorted([head, gap])
        assert other["absent_shas"] == ([] if answer is _fail else sorted([head, gap]))
    monkeypatch.setattr(qualification, "_get_json", _fail)
    moved, changed = qualification.refresh_qualification(saved, fork)
    assert changed and moved["github"]["runs"] == [] and moved["github"]["repository"] == "example/fork"
    stayed, changed = qualification.refresh_qualification(saved, official)
    assert not changed and stayed["github"]["runs"] == kept
    # A record saved before the name was kept claims no repository: nothing proves another origin, so its rows
    # stay as they always did, and it stays unnamed.
    legacy = {**material, "github": {key: value for key, value in seen.items() if key != "repository"}}
    unnamed = qualification.post_merge_checks(legacy, fork, get_json=_fail)
    assert unnamed["runs"] == kept and "repository" not in unnamed
    # Rows no repository is known to have observed (saved before the name was kept, or the status of a read with no
    # reader) are no known repository's evidence. They stay as they were while a read supplies nothing of this range;
    # the first read that supplies a run or a whole listing names the record and keeps only its own observations.
    no_reader = "https://gitlab.com/owner/repo"

    def refresh(record, source, reader):
        monkeypatch.setattr(qualification, "_get_json", reader)
        return qualification.refresh_qualification({**material, "github": record}, source)

    def whole(url):  # owner/repo answers whole: the failed run, measured over the gap
        return {"total_count": 2, "jobs": _jobs(witness, base)} if "/jobs?" in url else {
            "total_count": 1, "workflow_runs": [witness]}

    def empty(url):  # example/fork answers whole: no run at all
        return {"total_count": 0, "workflow_runs": []}

    def silent(url):  # owner/repo answers a page of other commits' runs, then nothing
        if "page=2" in url:
            raise OSError("offline")
        return {"total_count": 101, "workflow_runs": [{"id": 2_000 + n, "head_sha": "outside"} for n in range(100)]}

    def flood(url):  # owner/repo answers pages of other commits' runs until the request budget ends the read
        page = int(parse_qs(urlparse(url).query)["page"][0])
        return {"total_count": 10_000, "workflow_runs": [
            {"id": 3_000 + page * 100 + n, "head_sha": "outside"} for n in range(100)]}

    def short(url):  # owner/repo names a new run on its first page, then nothing
        if "page=2" in url:
            raise OSError("offline")
        return {"total_count": 101, "workflow_runs": [{**witness, "id": 78}, *[
            {"id": 1_000 + n, "head_sha": "outside"} for n in range(99)]]}

    # 1. An outage before any observation saves a status nothing can own. owner/repo's red then names the record, and
    # example/fork's whole, empty listing of the same material starts over instead of keeping that red.
    blank = qualification.post_merge_checks(material, no_reader, get_json=_fail)
    assert (blank["status"], blank["runs"], blank["absent_shas"]) == ("unavailable", [], []) and "repository" not in blank
    first, changed = refresh(blank, official, whole)
    assert changed and first["github"] == seen
    second, changed = refresh(first["github"], fork, empty)
    assert changed and second["github"]["runs"] == [] and second["github"]["repository"] == "example/fork"
    assert second["github"]["absent_shas"] == second["github"]["missing_shas"] == sorted([head, gap])
    assert second["github"]["interrupted_shas"] == []
    # 2. Rows saved without a name stay, unnamed and retained, through every read that supplies nothing of this range:
    # an outage, a spent request budget, a source with no reader, a read that never names the range's commits.
    for source, reader in ((fork, _fail), (official, _fail), (official, silent), (official, flood), (no_reader, _fail),
                           ("", _fail)):
        held, changed = refresh(legacy["github"], source, reader)
        assert not changed and held["github"]["runs"] == kept and "repository" not in held["github"], source
        assert held["github"]["status"] in ("unavailable", "partial"), source
        assert (held["github"]["absent_shas"], held["github"]["missing_shas"]) == ([gap], []), source
    # 3. Mixed: owner/repo's short read names a different run than the unnamed red. The record is then owner/repo's, with
    # that run alone; the unnamed red, its absence and its cover reach neither it nor example/fork's later whole read.
    mixed, changed = refresh(legacy["github"], official, short)
    start = mixed["github"]
    assert changed and start["repository"] == "owner/repo" and start["status"] == "unavailable"
    assert [(run["run_id"], run["conclusion"]) for run in start["runs"]] == [(78, "failure")]
    assert not any(run.get("retained") or run.get("covered_shas") for run in start["runs"])
    assert (start["absent_shas"], start["interrupted_shas"], start["missing_shas"]) == ([], [], [gap])
    other, changed = refresh(start, fork, empty)
    assert changed and other["github"]["repository"] == "example/fork" and other["github"]["runs"] == []
    assert other["github"]["absent_shas"] == other["github"]["missing_shas"] == sorted([head, gap])
    assert other["github"]["interrupted_shas"] == []
    # From then on the name is kept: an outage of owner/repo retains its own run under it.
    down, changed = refresh(start, official, _fail)
    assert not changed and down["github"]["runs"] == [{**start["runs"][0], "retained": True}]
    assert down["github"]["repository"] == "owner/repo"


@pytest.mark.parametrize("state", ["passed", "failed", "not_run"])
def test_merge_commit_carries_checks_and_missing_evidence_without_blocking(world, state):  # noqa: F811
    checks = [{"name": "full_battery", "status": state, "detail": "exact candidate"}]
    out = github._pr_merge(world.ctx, number=7, expected_head_sha=HEAD, method="merge", checks=checks,
                           review_verdict="AUTHOR ACCEPTED")
    assert "merge: merged" in out
    call = next(c for c in world.gh.calls if sent(c))
    assert call == ["api", ROUTE, "--hostname", "github.com", "--method", "PUT", "--input", "-"]
    body = world.gh.merge_inputs[0]  # the commit_message: GitHub appends it to its automatic title
    receipt = qualification.commit_qualification(body)
    assert receipt == _receipts(world)[0]["qualification"]
    assert receipt["checks"][0] == {**checks[0], "source": "author_declared"}
    assert receipt["checks"][1]["name"] == "cost_contract"
    assert receipt["checks"][1]["status"] == "unknown"
    assert "No execution evidence" in body and "Post-merge check: not run yet" in body
    assert receipt["review"] == {"observed": "unknown", "state": "no_record", "relation": "unknown",
                                 "declared": "AUTHOR ACCEPTED"}
    assert "Summary of the change" in body


def test_rebase_discloses_that_it_creates_no_merge_commit(world):  # noqa: F811
    github._pr_merge(world.ctx, number=7, expected_head_sha=HEAD, method="rebase")
    assert _receipts(world)[0]["commit_receipt"] == "no_merge_commit_for_rebase"
    assert next(c for c in world.gh.calls if sent(c)) == [
        "api", ROUTE, "--hostname", "github.com", "--method", "PUT", "--input", "-"]
    assert world.gh.merge_payloads == [{"sha": HEAD, "merge_method": "rebase"}] and world.gh.merge_inputs == [None]


def test_check_shape_refusal_then_valid_missing_checks_still_merge(world):  # noqa: F811
    out = github._pr_merge(world.ctx, number=7, expected_head_sha=HEAD, method="squash", checks=[{}])
    assert "TOOL_ARG_ERROR" in out and not world.gh.calls
    assert "merge: merged" in github._pr_merge(world.ctx, number=7, expected_head_sha=HEAD, method="squash")


def test_malformed_trailer_is_unknown_and_valid_one_is_read():
    receipt = qualification.merge_qualification({}, [{"name": "full_battery", "status": "passed"}])
    for changes in ({"checks": [None]}, {"checks": [{"name": "full_battery", "status": "maybe"}]}, {"review": []}):
        body = qualification.TRAILER + json.dumps({**receipt, **changes})
        assert qualification.commit_qualification(body) is None
    assert qualification.commit_qualification(qualification.commit_summary(receipt)) == receipt


def test_the_trailer_keeps_record_state_and_request_relation_and_legacy_says_not_recorded():
    record = {"record": {"state": "pending", "verdict": {"aggregate": "PASS"}}}
    new = qualification.merge_qualification(record, relation="covers_head")
    assert new["review"] == {"observed": "PASS", "state": "pending", "relation": "covers_head", "declared": "unknown"}
    assert qualification.merge_qualification({}, relation="not_declared")["review"] == {
        "observed": "unknown", "state": "no_record", "relation": "not_declared", "declared": "unknown"}
    assert ("- Review: observed PASS (record pending, subject relation at the merge request covers_head); "
            "author declared unknown") in qualification.commit_summary(new)
    assert qualification.commit_qualification(qualification.commit_summary(new)) == new
    legacy = qualification.merge_qualification(record)  # a receipt from before the relation: nothing invented
    assert legacy["review"] == {"observed": "PASS", "declared": "unknown"}
    assert ("- Review: observed PASS (record state and subject relation at the merge request not recorded); "
            "author declared unknown") in qualification.commit_summary(legacy)
    assert "settled" not in qualification.commit_summary(legacy)


def _receipt_commit(n, review):
    return {"sha": f"{n:040x}", "subject": f"change {n}", "date": "2026-11-04T10:00:00+00:00", "body_read": True,
            "receipt": {"schema": 1, "checks": [], "review": review, "post_merge": {"status": "not_run"}}}


def test_review_combinations_stay_bound_to_their_commit_in_every_summary():
    pass_different = {"observed": "PASS", "state": "settled", "relation": "changes_after_review", "declared": "PASS"}
    fail_same = {"observed": "FAIL", "state": "settled", "relation": "covers_head", "declared": "PASS"}
    legacy = {"observed": "PASS", "declared": "PASS"}
    future = {**fail_same, "observed": "PASS", "landing": "same"}
    guessed = {**pass_different, "relation": "approved"}
    value = {"base_sha": "a" * 40, "target_sha": "b" * 40, "commits": [
        _receipt_commit(n, review) for n, review in enumerate((pass_different, fail_same, legacy, future, guessed))]}
    compact = qualification.compact_qualification(value)["commits"]
    assert compact["reviews"] == [
        {"observed": "FAIL", "state": "settled", "relation": "covers_head", "declared": "PASS", "count": 1},
        {"observed": "PASS", "state": "not_recorded", "relation": "not_recorded", "declared": "PASS", "count": 1},
        {"observed": "PASS", "state": "settled", "relation": "changes_after_review", "declared": "PASS", "count": 1}]
    assert compact["unabridged"] == [{"sha": f"{3:040x}", "receipt": {"review": future}},
                                     {"sha": f"{4:040x}", "receipt": {"review": guessed}}]
    assert qualification.qualification_summary(value)["reviews"] == [{**review, "count": 1} for review in (
        pass_different, fail_same,
        {"observed": "PASS", "state": "not_recorded", "relation": "not_recorded", "declared": "PASS"}, future, guessed)]


def test_qualification_summary_preserves_unknown_review_count_field():
    review = {"observed": "PASS", "state": "settled", "relation": "covers_head", "declared": "PASS",
              "count": "source-count"}
    value = {"base_sha": "a" * 40, "target_sha": "b" * 40, "commits": [_receipt_commit(0, review)]}
    row = qualification.qualification_summary(value)["reviews"][0]
    assert row["count"] == "source-count"
    assert row["_count"] == 1


def test_the_context_form_counts_a_measured_run_and_keeps_only_an_unknown_field_whole():
    """A run the reader measured carries its measured base and the commits it covers; the context form still counts it
    by conclusion (coverage stays in missing_shas), and only a field no reader knows keeps a run whole."""
    runs = [{"sha": "a" * 40, "status": "completed", "conclusion": "failure", "url": None, "run_id": 5, "attempt": 1,
             "measured_base": "b" * 40, "covered_shas": ["c" * 40]},
            {"sha": "c" * 40, "status": "completed", "conclusion": "cancelled", "measured_base": None},
            {"sha": "d" * 40, "status": "completed", "conclusion": "success", "landing": "new"}]
    compact = qualification.compact_qualification({"github": {"status": "observed", "runs": runs, "missing_shas": []}})
    assert compact["github"]["runs"]["completed"] == {"failure": 1, "cancelled": 1}
    assert compact["github"]["runs"]["unabridged"] == [runs[2]] and compact["github"]["missing_shas"] == []


def test_github_checks_include_red_intermediate_sha_but_exclude_other_ranges():
    commits = [{"sha": "a"}, {"sha": "b"}]
    runs = [{"head_sha": sha, "conclusion": result, "status": "completed", "html_url": f"https://github.com/r/{sha}"}
            for sha, result in (("b", "success"), ("a", "failure"), ("outside", "failure"))]
    urls = []
    def get(url):
        urls.append(url)
        return {"total_count": len(runs), "workflow_runs": runs}
    fact = qualification.post_merge_checks(commits, "https://github.com/owner/repo.git", get_json=get)
    assert [(r["sha"], r["conclusion"]) for r in fact["runs"]] == [("b", "success"), ("a", "failure")]
    assert fact["status"] == "observed" and fact["missing_shas"] == []
    assert len(urls) == 1 and "/actions/workflows/post-merge.yml/runs?" in urls[0]
    empty = qualification.post_merge_checks(commits, "https://github.com/owner/repo", get_json=lambda _: {"total_count": 0, "workflow_runs": []})
    assert empty["missing_shas"] == ["a", "b"] and not empty["runs"]
    unavailable = qualification.post_merge_checks(commits, "https://github.com/owner/repo",
                                                   get_json=lambda _: (_ for _ in ()).throw(OSError()))
    assert unavailable["status"] == "unavailable" and not unavailable["runs"]


def test_check_writes_exact_real_git_range_and_preflight_reads_unqualified_items(history_repo, tmp_path, monkeypatch):  # noqa: F811
    repo, base, old = history_repo["repo"], history_repo["base"], history_repo["c4"]
    checks = [{"name": "full_battery", "status": "failed", "detail": "Author accepted two FAILs"},
              {"name": "cost_contract", "status": "not_run"}]
    receipt = qualification.merge_qualification({"declared": {"verdict": "AUTHOR ACCEPTED"}}, checks)
    (repo / "qualification.txt").write_text("candidate", encoding="utf-8")
    target = _commit(repo, "Incoming qualification\n\n" + qualification.commit_summary(receipt))
    monkeypatch.setattr(letter, "_default_git", lambda: _capture_for(repo))
    monkeypatch.setattr(qualification, "post_merge_checks", lambda commits, _: {
        "status": "observed", "runs": [{"sha": target, "conclusion": "failure", "url": "https://github.com/run/1"}]})
    drive = tmp_path / "letter"
    letter.atomic_write_json(letter.record_path(drive), _record(key={**_record()["key"], "base_sha": base, "target_sha": old}))
    # This install recorded two of its own commits over a review outcome: one comes back in this range, one is its base.
    letter.atomic_write_json(drive / "state" / "advisory_overrides.json", {"count": 2, "landed": [
        {"sha": base, "reason": "review_quorum", "task_id": "t0", "at": "2026-11-01T00:00:00+00:00"},
        {"sha": target, "reason": "critical_findings", "task_id": "t1", "at": "2026-11-02T00:00:00+00:00",
         "disposition": "accepted", "rationale": "Read the finding."}]})
    letter.refresh_after_check(_status(current_sha=old, running_sha=base, latest_sha=target), drive_root=drive)
    material = json.loads(letter.material_path(drive).read_text(encoding="utf-8"))["text"]
    assert all(word in material for word in (target, "Author accepted two FAILs", "not_run"))
    assert letter.save_description("Incoming qualification has accepted FAILs.", base, target, 0,
                                   author={"task_id": "t", "model": "m"}, drive_root=drive)["outcome"] == "saved"
    result = qualification.preflight_description({"base_sha": base, "target_sha": target,
                                                  "update_channel": "stable", "target_ref": "managed/main"}, drive_root=drive)
    assert result["letter"]["key"]["target_sha"] == target and result["letter"]["description_current"]
    assert result["qualification"]["target_sha"] == target
    assert {c["sha"] for c in result["qualification"]["commits"]} == set(_git(repo, "rev-list", f"{base}..{target}").splitlines())
    assert result["qualification"]["github"]["runs"][0]["conclusion"] == "failure"
    # The range's facts name that commit with its recorded outcome, for the window, the preflight and the author;
    # the Runtime summary carries the count. Rows that cannot be read leave the key out: unknown, never none.
    assert result["qualification"]["landed_over_review"] == [{
        "sha": target, "subject": "Incoming qualification", "reason": "critical_findings",
        "outcome": 'review FAIL: critical findings; author continued: accepted — "Read the finding."'}]
    assert qualification.qualification_summary(result["qualification"])["landed_over_review"] == 1
    again = letter.refresh_after_check(_status(current_sha=old, running_sha=base, latest_sha=target), drive_root=drive)
    assert again["qualification"]["landed_over_review"] == result["qualification"]["landed_over_review"], (
        "a later check of the same range keeps its rows")
    assert "landed over a review FAIL or no verdict" in material
    for name, counter, expected in (("none", None, []), ("unread", "{not json", "absent")):
        other = tmp_path / name
        if counter is not None:
            (other / "state").mkdir(parents=True)
            (other / "state" / "advisory_overrides.json").write_text(counter, encoding="utf-8")
        facts = letter.refresh_after_check(_status(current_sha=old, running_sha=base, latest_sha=target),
                                           drive_root=other)["qualification"]
        assert facts.get("landed_over_review", "absent") == expected, name
        assert ("landed_over_review" in qualification.qualification_summary(facts)) is (expected == [])
    # A long range keeps only its newest bodies, yet reads the receipt of every first-parent commit: the count of
    # merges that went in over a review outcome is never short by the commits behind the body bound.
    (repo / "later.txt").write_text("later", encoding="utf-8")
    newest = _commit(repo, "A later commit\n\nIts body is the one the bound keeps.")
    long = letter.collect_range_material(base, newest, git=_capture_for(repo), max_bodies=1)
    assert [c["body_read"] for c in long["commits"] if c["sha"] in (newest, target)] == [True, False]
    behind = next(c for c in qualification.range_qualification(long, "")["commits"] if c["sha"] == target)
    assert behind["body_read"] and behind["receipt"]["checks"][0]["detail"] == "Author accepted two FAILs"
    # Green evidence stays green; host code does not invent a failure from prose.
    green = qualification.merge_qualification({"record": {"verdict": {"aggregate": "PASS"}}},
        [{"name": name, "status": "passed"} for name in qualification.CHECKS])
    assert all(row["status"] == "passed" for row in green["checks"])
    assert qualification.commit_qualification("No tests failed according to this prose") is None


def test_new_post_merge_red_refreshes_same_range_and_offline_never_becomes_green(tmp_path, monkeypatch):
    base, target = "a" * 40, "b" * 40
    material = {"base_sha": base, "target_sha": target, "commits": [{"sha": target, "subject": "change", "body": ""}]}
    collected = []
    monkeypatch.setattr(letter, "collect_range_material", lambda *a, **k: collected.append(a) or dict(material))
    monkeypatch.setattr(letter, "_default_git", lambda: lambda _: (1, "", ""))
    hub = {"listing": [{"head_sha": target, "status": "completed", "conclusion": "success"}]}
    monkeypatch.setattr(qualification, "_get_json", _run_listing(hub))  # below the reader's join, which keeps the red
    status = _status(official_repo_url="https://github.com/owner/repo")
    for _ in range(2):
        letter.refresh_after_check(status, drive_root=tmp_path)
    assert len(collected) == 1 and hub["reads"] == 2
    hub["listing"] = [{"head_sha": target, "status": "completed", "conclusion": "failure"}]
    assert letter.refresh_after_check(status, drive_root=tmp_path)["qualification"]["github"]["runs"][0]["conclusion"] == "failure"
    hub["listing"] = None
    letter.refresh_after_check(status, drive_root=tmp_path)
    record = json.loads(letter.record_path(tmp_path).read_text(encoding="utf-8"))
    assert (record["qualification"]["github"]["status"], hub["reads"]) == ("unavailable", 4)
    assert [(run["conclusion"], run["retained"]) for run in record["qualification"]["github"]["runs"]] == [
        ("failure", True)], "an outage never turns red green"
    assert len(collected) == 1 and record["state"] == "none"


def test_preflight_letter_does_not_show_prior_range(tmp_path):
    old = _record()
    letter.atomic_write_json(letter.record_path(tmp_path), old)
    result = qualification.preflight_description({"base_sha": "a" * 40, "target_sha": "c" * 40}, drive_root=tmp_path)
    assert result["letter"]["text"] == "" and result["letter"]["description_current"] is False
    result = qualification.preflight_description({"base_sha": "a" * 40, "target_sha": "b" * 40}, drive_root=tmp_path)
    assert result["letter"]["text"] == "letter" and result["letter"]["description_current"]


@pytest.mark.parametrize("failure", ["", "material_read"])
def test_new_red_facts_land_beside_the_saved_prose_which_stays_current(tmp_path, monkeypatch, failure):
    """Owner 28.10: a same-range CI change is a new fact about the range, never a reason to mark the
    description stale; the preflight shows the text and the red run side by side."""
    base, target = "a" * 40, "b" * 40
    green = {"base_sha": base, "target_sha": target, "commits": [{"sha": target, "subject": "change"}],
             "github": {"status": "observed", "runs": [{"sha": target, "conclusion": "success"}]}}
    red = {"status": "observed", "runs": [{"sha": target, "conclusion": "failure"}]}
    letter.atomic_write_json(letter.record_path(tmp_path), _record(qualification=green, text="Old green description",
                                                                   revision=1))
    monkeypatch.setattr(qualification, "post_merge_checks", lambda *a: red)
    monkeypatch.setattr(letter, "_default_git", lambda: lambda _: (1, "", ""))

    def collect(*args, **kwargs):
        if failure == "material_read":
            raise letter.MaterialUnavailable("Git object unavailable")
        return {**green, "commits": [{"sha": target, "subject": "change", "body": ""}]}

    monkeypatch.setattr(letter, "collect_range_material", collect)
    letter.refresh_after_check(_status(), drive_root=tmp_path)
    result = qualification.preflight_description({"base_sha": base, "target_sha": target,
        "update_channel": "stable", "target_ref": "managed/main"}, drive_root=tmp_path)
    assert result["letter"]["description_current"] is True and result["letter"]["text"] == "Old green description"
    assert result["qualification"]["github"]["runs"] == red["runs"]
    assert letter.read_record(tmp_path)["revision"] == 1


def _fail(_url):
    raise OSError("offline")


def test_open_post_merge_reds_are_read_and_an_outage_never_reads_as_none_open():
    urls = []

    def get(url):
        urls.append(url)
        return [{"number": 2390, "title": "Post-merge red: tests/test_a.py::test_x", "created_at": "2026-10-28T08:00:00Z"},
                {"number": 7, "title": "a pull request", "pull_request": {}}]

    seen = qualification.post_merge_reds("https://github.com/owner/repo.git", get_json=get)
    assert urls == ["https://api.github.com/repos/owner/repo/issues?labels=post-merge-red&state=open&per_page=100"]
    assert seen["read"] == "observed" and seen["repository"] == "owner/repo" and seen["observed_at"]
    assert seen["issues"] == [{"number": 2390, "title": "Post-merge red: tests/test_a.py::test_x",
                               "opened": "2026-10-28T08:00:00Z"}]
    # An outage keeps the last observation and its time; without one it lists nothing and dates nothing.
    first = qualification.post_merge_reds("https://github.com/owner/repo", get_json=_fail)
    assert (first["read"], first["issues"], first["observed_at"]) == ("unavailable", [], "")
    down = qualification.kept_reds(first, seen)
    assert (down["read"], down["issues"], down["observed_at"]) == ("unavailable", seen["issues"], seen["observed_at"])
    assert qualification.kept_reds(first, None) == first and qualification.kept_reds(None, seen) == seen
    assert qualification.kept_reds(seen, down) == seen
    # Another source never wears the old list: a fork's failed read keeps nothing of the official observation.
    fork = qualification.kept_reds(qualification.post_merge_reds("https://github.com/example/fork", get_json=_fail), seen)
    assert (fork["repository"], fork["issues"], fork["observed_at"]) == ("example/fork", [], "")
    full = qualification.post_merge_reds("https://github.com/owner/repo",
                                         get_json=lambda _: [{"number": n} for n in range(100)])
    assert full["read"] == "partial" and len(full["issues"]) == 100
    other = qualification.post_merge_reds("https://gitlab.com/owner/repo",
                                          get_json=lambda _: pytest.fail("no GitHub reader for another host"))
    assert other["read"] == "unavailable" and other["issues"] == []


def test_every_completed_check_shows_the_open_reds_to_an_up_to_date_install(tmp_path, monkeypatch):
    """No update, so no range: the reds still reach the Runtime fact, and a later outage keeps them."""
    monkeypatch.setattr(letter, "collect_range_material", lambda *a, **k: pytest.fail("no range without an update"))
    monkeypatch.setattr(letter, "_default_git", lambda: lambda _: (1, "", ""))
    monkeypatch.setattr(qualification, "_get_json", lambda _url: [
        {"number": 2390, "title": "Post-merge red: tests/test_a.py::test_x", "created_at": "t0"}])
    status = _status(available=False, latest_sha="", official_repo_url="https://github.com/razzant/ouroboros")
    record = letter.refresh_after_check(status, drive_root=tmp_path)
    assert [row["number"] for row in record["post_merge_reds"]["issues"]] == [2390]
    cache = {"managed_update_cache": {"latest_sha": "", "available": False, "behind": 0, "ahead": 0, "checked_at": "t0",
                                      "checked_head_sha": status["current_sha"]}}
    fact = letter.official_update_projection("a" * 40, drive_root=tmp_path, state=cache)
    assert fact["status"] == "up_to_date" and fact["letter"] is None
    assert fact["post_merge_reds"]["read"] == "observed" and fact["post_merge_reds"]["issues"][0]["number"] == 2390
    monkeypatch.setattr(qualification, "_get_json", _fail)
    kept = letter.refresh_after_check(status, drive_root=tmp_path)["post_merge_reds"]
    assert kept["read"] == "unavailable" and kept["issues"] == record["post_merge_reds"]["issues"]
    monkeypatch.setattr(qualification, "_get_json", lambda _url: [])
    assert letter.refresh_after_check(status, drive_root=tmp_path)["post_merge_reds"]["issues"] == []
    monkeypatch.setenv("OUROBOROS_UPDATE_CHANNEL", "development")
    switched = letter.official_update_projection("a" * 40, drive_root=tmp_path, state={"managed_update_cache": {
        **cache["managed_update_cache"], "update_channel": "stable"}})
    assert switched["status"] == "unchecked" and switched["post_merge_reds"]["read"] == "observed"


def test_a_failed_read_keeps_the_observation_the_record_holds_under_the_lock():
    """The check reads GitHub before the record lock; the facts land on the record read under it, so a
    failed read keeps whatever observation the record holds by then."""
    held = {"read": "observed", "repository": "razzant/ouroboros", "observed_at": "t1",
            "issues": [{"number": 2412, "title": "Post-merge red: #2397 theme", "opened": "t1"}]}
    failed = qualification.post_merge_reds("https://github.com/razzant/ouroboros", get_json=_fail)
    key = {"base_sha": "a" * 40, "target_sha": "", "update_channel": "stable", "target_ref": "managed/main"}
    record = letter._with_check_facts({"schema": 1, "state": "none", "post_merge_reds": held}, key,
                                      {"post_merge_reds": failed}, running_sha="a" * 40, checkout_sha="a" * 40,
                                      target_version="7.9.0")
    assert record["post_merge_reds"]["read"] == "unavailable"
    assert (record["post_merge_reds"]["issues"], record["post_merge_reds"]["observed_at"]) == (held["issues"], "t1")
