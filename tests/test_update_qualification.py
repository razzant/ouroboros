"""Qualification is portable evidence of the exact proposed range, never a veto."""
import json
from urllib.parse import parse_qs, urlparse

import pytest

from ouroboros import update_letter as letter, update_qualification as qualification
from ouroboros.tools import github
from tests.test_pr_merge_receipts import BASE, HEAD, _receipts, world  # noqa: F401
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
        return {"workflow_runs": [
            {"head_sha": target, "status": "completed", "conclusion": "success"},
            *[{"head_sha": "outside", "status": "completed", "conclusion": "failure"}] * 99]}

    monkeypatch.setattr(qualification, "_get_json", get)
    fact = qualification.range_qualification(material, "https://github.com/owner/repo")
    assert [c["sha"] for c in fact["commits"]] == [target]
    assert fact["github"]["status"] == "observed" and fact["github"]["missing_shas"] == []
    assert len(urls) == 1
    # The letter still receives branch-internal subjects and bodies.
    assert {feature, nested}.issubset({c["sha"] for c in material["commits"]}) is (not squash)


@pytest.mark.parametrize("status,conclusion,covers", [
    ("completed", "success", True), ("completed", "failure", True),
    ("in_progress", None, False), ("completed", "cancelled", False),
])
@pytest.mark.parametrize("replaced_pending", [False, True])
def test_later_finished_run_covers_earlier_merges_only(status, conclusion, covers, replaced_pending):
    commits = [{"sha": sha} for sha in ("newest", "checked", "earlier")]
    cancelled = [{"head_sha": "earlier", "status": "completed", "conclusion": "cancelled"}] if replaced_pending else []
    fact = qualification.post_merge_checks(commits, "https://github.com/owner/repo", get_json=lambda _: {
        "workflow_runs": [{"head_sha": "checked", "status": status, "conclusion": conclusion}, *cancelled,
                          {"head_sha": "outside", "status": "completed", "conclusion": "success"}]})
    assert fact["missing_shas"] == (["newest"] if covers or replaced_pending else ["earlier", "newest"])
    assert fact["runs"][0].get("covered_shas", []) == (["earlier"] if covers else [])
    if replaced_pending:
        assert fact["runs"][1]["conclusion"] == "cancelled"  # the direct observation survives coverage


@pytest.mark.parametrize("older_run", ["failure", "success", None])
def test_paging_observes_older_runs_before_inferring_coverage(monkeypatch, older_run):
    commits = [{"sha": f"{i:040x}", "subject": f"Integration {i}"} for i in range(150, 0, -1)]
    older = commits[130]["sha"]
    rows = [{"head_sha": c["sha"], "status": "completed",
             "conclusion": older_run if c["sha"] == older else "success"}
            for c in commits if c["sha"] != older or older_run is not None]
    pages = []

    def get(url):
        page = int(parse_qs(urlparse(url).query)["page"][0])
        pages.append(page)
        return {"workflow_runs": rows[(page - 1) * 100:page * 100]}

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
    assert pages == [1, 2]
    # Later upstream activity moves the same range farther down the API listing.
    rows[:0] = [{"head_sha": "outside", "status": "completed", "conclusion": "failure"}] * 60
    pages.clear()
    refreshed, changed = qualification.refresh_qualification(fact, "https://github.com/owner/repo")
    assert_evidence(refreshed["github"])
    assert pages == [1, 2, 3] and not changed


@pytest.mark.parametrize("ending", ["exhausted", "oldest", "predates", "budget", "unavailable"])
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
            return {"workflow_runs": [green, *[filler] * 99]}
        if ending == "unavailable":
            raise OSError("offline")
        if ending == "exhausted":
            return {"workflow_runs": []}
        tail = ({**green, "head_sha": "oldest"} if ending == "oldest" else
                {**filler, "created_at": "2026-10-22T23:59:59Z"} if ending == "predates" else filler)
        return {"workflow_runs": [*[filler] * 99, tail]}

    monkeypatch.setattr(qualification, "_get_json", get)
    fact = qualification.range_qualification(
        {"base_sha": "base", "target_sha": "newest", "first_parent_commits": commits},
        "https://github.com/owner/repo")
    observed = fact["github"]
    complete = ending in ("exhausted", "oldest", "predates")
    absent = ["middle"] if ending == "oldest" else ["middle", "oldest"]
    assert pages == (list(range(1, 11)) if ending == "budget" else [1, 2])
    assert observed["status"] == ("observed" if complete else "partial" if ending == "budget" else "unavailable")
    assert observed["missing_shas"] == ([] if complete else absent)
    assert observed["absent_shas"] == (absent if complete else [])
    assert observed["runs"][0].get("covered_shas", []) == (absent if complete else [])
    # Refresh must use the saved date boundary, and cannot invent coverage after a short read.
    pages.clear()
    refreshed, changed = qualification.refresh_qualification(fact, "https://github.com/owner/repo")
    assert refreshed == fact and not changed
    assert pages == (list(range(1, 11)) if ending == "budget" else [1, 2])


def test_refresh_preserves_proven_absence_until_a_direct_run_is_observed(monkeypatch):
    commits = [{"sha": sha, "subject": sha} for sha in ("newest", "earlier", "oldest")]
    rows = [{"head_sha": sha, "status": "completed", "conclusion": "success"} for sha in ("newest", "oldest")]
    monkeypatch.setattr(qualification, "_get_json", lambda _: {"workflow_runs": rows})
    authored = qualification.range_qualification({"base_sha": "base", "target_sha": "newest",
                                                 "commits": commits}, "https://github.com/owner/repo")
    assert authored["github"]["runs"][0]["covered_shas"] == ["earlier"]
    previous = authored
    for conclusion in (None, None, "failure"):
        def get(url):
            if "page=2" in url or conclusion is None:
                raise OSError("offline")
            return {"workflow_runs": [{"head_sha": "earlier", "status": "completed", "conclusion": conclusion},
                                      *[{"head_sha": "outside"}] * 99]}

        monkeypatch.setattr(qualification, "_get_json", get)
        previous, changed = qualification.refresh_qualification(authored, "https://github.com/owner/repo",
                                                               previous=previous)
        runs = {r["sha"]: r for r in previous["github"]["runs"]}
        assert previous["github"]["status"] == "unavailable"
        assert changed is (conclusion is not None)
        assert runs["newest"].get("covered_shas", []) == (["earlier"] if conclusion is None else [])
        if conclusion is not None:
            assert runs["earlier"]["conclusion"] == "failure"
            assert previous["github"]["absent_shas"] == []

    # Older saved coverage has no proof of absence; an outage cannot ratify it.
    legacy = json.loads(json.dumps(authored))
    legacy["github"].pop("absent_shas")
    monkeypatch.setattr(qualification, "_get_json", lambda _: (_ for _ in ()).throw(OSError()))
    refreshed, changed = qualification.refresh_qualification(legacy, "https://github.com/owner/repo")
    assert changed and refreshed["github"]["missing_shas"] == ["earlier"]
    assert not any(r.get("covered_shas") for r in refreshed["github"]["runs"])


def test_github_availability_and_new_runs_never_touch_the_saved_text(tmp_path, monkeypatch):
    base, target = "a" * 40, "b" * 40
    material = {"base_sha": base, "target_sha": target, "commits": [{"sha": target, "subject": "change"}]}
    monkeypatch.setattr(letter, "collect_range_material", lambda *a, **k: dict(material))
    monkeypatch.setattr(letter, "_default_git", lambda: lambda _: (1, "", ""))
    observed = {"status": "observed", "runs": [{"sha": target, "status": "completed", "conclusion": "success"}]}
    monkeypatch.setattr(qualification, "post_merge_checks", lambda *a: dict(observed))
    letter.refresh_after_check(_status(), drive_root=tmp_path)
    letter.save_description("the saved text", base, target, 0, author={"task_id": "t", "model": "m"}, drive_root=tmp_path)
    first = letter.read_record(tmp_path)
    runs = observed["runs"]
    observed.update(status="unavailable", reason="HTTPError", runs=[])
    offline = letter.refresh_after_check(_status(), drive_root=tmp_path)
    assert (offline["text"], offline["revision"], offline["written_at"]) == ("the saved text", 1, first["written_at"])
    assert offline["qualification"]["github"]["status"] == "unavailable"
    assert offline["qualification"]["github"]["runs"] == runs
    observed.update(status="observed", reason="", runs=[{"sha": target, "status": "completed", "conclusion": "failure"}])
    changed = letter.refresh_after_check(_status(), drive_root=tmp_path)
    assert changed["qualification"]["github"]["runs"][0]["conclusion"] == "failure"
    assert (changed["text"], changed["revision"]) == ("the saved text", 1)
    assert letter.project_letter(changed, head_sha=base, latest_sha=target)["description_current"] is True


@pytest.mark.parametrize("state", ["passed", "failed", "not_run"])
def test_merge_commit_carries_checks_and_missing_evidence_without_blocking(world, state):  # noqa: F811
    checks = [{"name": "full_battery", "status": state, "detail": "exact candidate"}]
    out = github._pr_merge(world.ctx, number=7, expected_head_sha=HEAD, method="merge", checks=checks,
                           review_verdict="AUTHOR ACCEPTED")
    assert "merge: merged" in out
    call = next(c for c in world.gh.calls if c[:2] == ["pr", "merge"])
    assert call[6:] == ["--body-file", "-"]
    body = world.gh.merge_inputs[0]
    receipt = qualification.commit_qualification(body)
    assert receipt == _receipts(world)[0]["qualification"]
    assert receipt["checks"][0] == {**checks[0], "source": "author_declared"}
    assert receipt["checks"][1]["name"] == "cost_contract"
    assert receipt["checks"][1]["status"] == "unknown"
    assert "No execution evidence" in body and "Post-merge check: not run yet" in body
    assert receipt["review"] == {"observed": "unknown", "declared": "AUTHOR ACCEPTED"}
    assert "Summary of the change" in body


def test_rebase_discloses_that_it_creates_no_merge_commit(world):  # noqa: F811
    github._pr_merge(world.ctx, number=7, expected_head_sha=HEAD, method="rebase")
    assert _receipts(world)[0]["commit_receipt"] == "no_merge_commit_for_rebase"
    assert next(c for c in world.gh.calls if c[:2] == ["pr", "merge"]) == [
        "pr", "merge", "7", "--rebase", "--match-head-commit", HEAD]
    assert world.gh.merge_inputs == [None]


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


def test_github_checks_include_red_intermediate_sha_but_exclude_other_ranges():
    commits = [{"sha": "a"}, {"sha": "b"}]
    runs = [{"head_sha": sha, "conclusion": result, "status": "completed", "html_url": f"https://github.com/r/{sha}"}
            for sha, result in (("b", "success"), ("a", "failure"), ("outside", "failure"))]
    urls = []
    def get(url):
        urls.append(url)
        return {"workflow_runs": runs}
    fact = qualification.post_merge_checks(commits, "https://github.com/owner/repo.git", get_json=get)
    assert [(r["sha"], r["conclusion"]) for r in fact["runs"]] == [("b", "success"), ("a", "failure")]
    assert fact["status"] == "observed" and fact["missing_shas"] == []
    assert len(urls) == 1 and "/actions/workflows/post-merge.yml/runs?" in urls[0]
    empty = qualification.post_merge_checks(commits, "https://github.com/owner/repo", get_json=lambda _: {"workflow_runs": []})
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
    observed = {"status": "observed", "runs": [{"sha": target, "conclusion": "success"}]}
    monkeypatch.setattr(qualification, "post_merge_checks", lambda *a: dict(observed))
    for _ in range(2):
        letter.refresh_after_check(_status(), drive_root=tmp_path)
    assert len(collected) == 1
    observed["runs"] = [{"sha": target, "conclusion": "failure"}]
    assert letter.refresh_after_check(_status(), drive_root=tmp_path)["qualification"]["github"]["runs"][0]["conclusion"] == "failure"
    observed.update(status="unavailable", runs=[])
    letter.refresh_after_check(_status(), drive_root=tmp_path)
    record = json.loads(letter.record_path(tmp_path).read_text(encoding="utf-8"))
    assert record["qualification"]["github"]["status"] == "unavailable"
    assert record["qualification"]["github"]["runs"][0]["conclusion"] == "failure", "an outage never turns red green"
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
    cache = {"managed_update_cache": {"latest_sha": "", "available": False, "behind": 0, "ahead": 0, "checked_at": "t0"}}
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

