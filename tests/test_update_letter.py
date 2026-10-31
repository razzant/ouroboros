"""The update description: range material, the facts-only check, storage and projection.

The writer, the owner request and the tool live in tests/test_update_description.py."""

from __future__ import annotations

import json
import pathlib
import subprocess
import pytest

from ouroboros import update_letter as ul


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()


def _capture_for(repo):
    def capture(cmd):
        proc = subprocess.run(cmd, cwd=repo, capture_output=True, text=True)
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()

    return capture


_HEADER = "# Demo\n\n## Version History\n\n| Version | Date | Description |\n|---|---|---|\n"


def _write_readme(repo, rows):
    (repo / "README.md").write_text(_HEADER + "".join(f"| {v} | {d} | {t} |\n" for v, d, t in rows), encoding="utf-8")


def _commit(repo, message):
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def history_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    _write_readme(repo, [("1.0.0", "2026-01-01", "first")])
    (repo / "VERSION").write_text("1.0.0\n")
    base = _commit(repo, "release: 1.0.0")
    _write_readme(repo, [("1.1.0", "2026-01-02", "second with an escaped \\| pipe"), ("1.0.0", "2026-01-01", "first")])
    (repo / "VERSION").write_text("1.1.0\n")
    c1 = _commit(repo, "release: 1.1.0\n\nBody of the second release.")
    # Two rows in one commit plus one malformed row (two cells only).
    (repo / "README.md").write_text(
        _HEADER
        + "| 1.2.0 | 2026-01-03 | third |\n"
        + "| 1.1.1 | 2026-01-03 | third-fix |\n"
        + "| 1.3.0 | only two cells |\n"
        + "| 1.1.0 | 2026-01-02 | second with an escaped \\| pipe |\n"
        + "| 1.0.0 | 2026-01-01 | first |\n",
        encoding="utf-8",
    )
    (repo / "VERSION").write_text("1.2.0\n")
    c2 = _commit(repo, "release: 1.2.0")
    # Roll-off: the 1.0.0 row leaves the table (capped), a new row arrives, a tag lands.
    _write_readme(repo, [
        ("1.4.0", "2026-01-04", "fourth"),
        ("1.2.0", "2026-01-03", "third"),
        ("1.1.1", "2026-01-03", "third-fix"),
        ("1.1.0", "2026-01-02", "second with an escaped \\| pipe"),
    ])
    (repo / "VERSION").write_text("1.4.0\n")
    c3 = _commit(repo, "release: 1.4.0")
    _git(repo, "tag", "-a", "v1.4.0", "-m", "v1.4.0")
    (repo / "notes.txt").write_text("tail\n")
    c4 = _commit(repo, "tail work\n\nDetails of the untagged tail.")
    return {"repo": repo, "base": base, "c1": c1, "c2": c2, "c3": c3, "c4": c4}


# ---------------------------------------------------------------------------
# material
# ---------------------------------------------------------------------------


def test_local_history_preview_is_bounded_and_names_the_complete_range():
    local = [{'sha': f'{i:040x}', 'date': '2026-10-21', 'subject': f'local subject {i}'} for i in range(50)]
    material = {'base_sha': 'f' * 40, 'target_sha': 'e' * 40, 'local_commits': local}
    text = ul.material_text(material)
    assert text.count('- ') == ul.LOCAL_SUBJECT_PREVIEW
    assert '30 older local subjects omitted' in text
    assert f"git log {material['target_sha']}..{material['base_sha']}" in text
    assert 'not incoming changes' in text and 'do not prove unique code' in text
    assert len(material['local_commits']) == 50  # the collector's complete input is not changed

def test_material_recovers_rows_from_commit_diffs_and_lists_range_commits(history_repo):
    repo = history_repo["repo"]
    material = ul.collect_range_material(history_repo["base"], history_repo["c4"], git=_capture_for(repo))

    assert [row["version"] for row in material["releases"]] == ["1.4.0", "1.2.0", "1.1.1", "1.1.0"]
    assert material["omitted_rows"] == 1  # the two-cell 1.3.0 row is disclosed, not silently dropped
    by_version = {row["version"]: row for row in material["releases"]}
    assert by_version["1.1.0"]["text"] == "second with an escaped | pipe"
    assert by_version["1.4.0"]["commit"] == history_repo["c3"]
    assert [c["sha"] for c in material["commits"]] == [
        history_repo["c4"], history_repo["c3"], history_repo["c2"], history_repo["c1"],
    ]
    assert material["commits"][0]["body"] == "Details of the untagged tail."
    assert material["bodies_omitted"] == 0
    assert material["versions"] == {"base": "1.0.0", "target": "1.4.0"}
    assert set(material) == {"base_sha", "target_sha", "commits", "bodies_omitted",
                             "omitted_commit_chunks", "releases", "omitted_rows",
                             "omitted_row_commits", "rows_summarized", "versions",
                             "local_commits", "omitted_local_chunks", "first_parent_commits"}
    assert material["first_parent_commits"] == material["commits"]  # this history is linear
    assert material["local_commits"] == [] and material["omitted_local_chunks"] == 0


def test_material_keeps_every_commit_subject_and_bounds_only_the_bodies(history_repo):
    # EVERY subject reaches the author however long the range is; the bound is on
    # bodies and on the text of the oldest release rows, and both are disclosed.
    material = ul.collect_range_material(
        history_repo["base"], history_repo["c4"], git=_capture_for(history_repo["repo"]), max_bodies=1,
    )
    assert [c["sha"] for c in material["commits"]] == [
        history_repo["c4"], history_repo["c3"], history_repo["c2"], history_repo["c1"],
    ]
    assert material["commits"][0]["body"] == "Details of the untagged tail."
    assert [c["body"] for c in material["commits"][1:]] == ["", "", ""]
    assert material["bodies_omitted"] == 3  # the three older bodies were never even read
    rendered = ul.material_text(material)
    # Both ends of the range are named; only the older BODY is gone, and it says so.
    assert "tail work" in rendered and "release: 1.1.0" in rendered
    assert "Details of the untagged tail." in rendered
    assert "Body of the second release." not in rendered
    assert "bodies of the 3 oldest commit(s) were not read" in rendered
    assert "1 unreadable history row(s) omitted" in rendered

    capped = ul.collect_range_material(
        history_repo["base"], history_repo["c4"], git=_capture_for(history_repo["repo"]), max_rows=3,
    )
    assert [row["version"] for row in capped["releases"]] == ["1.4.0", "1.2.0", "1.1.1", "1.1.0"]
    assert capped["releases"][-1]["text"] == "" and capped["rows_summarized"] == 1
    summarized = ul.material_text(capped)
    assert "- 1.1.0 (2026-01-02, added in " in summarized  # version, date, provenance, no text
    assert "second with an escaped | pipe" not in summarized
    assert "oldest 1 row(s) above carry version and date only" in summarized


def test_material_ignores_tables_outside_the_version_history(history_repo):
    # A README carries other pipe tables. Their rows are not releases, and reporting them
    # as dropped would tell the author a release was withheld when none was.
    repo = history_repo["repo"]
    (repo / "README.md").write_text(
        "# Demo\n\n## Providers\n\n| Provider | Key |\n|---|---|\n| openrouter | set |\n| local | n/a |\n"
        "\n## Version History\n\n| Version | Date | Description |\n|---|---|---|\n"
        "| 1.5.0 | 2026-01-08 | a real release |\n",
        encoding="utf-8",
    )
    c5 = _commit(repo, "readme: a provider table beside the history")
    material = ul.collect_range_material(history_repo["c4"], c5, git=_capture_for(repo))
    assert [row["version"] for row in material["releases"]] == ["1.5.0"]
    assert material["omitted_rows"] == 0, "another table's rows are not withheld releases"
    assert material["omitted_row_commits"] == []


def test_material_reads_a_release_row_with_the_repository_version_grammar(history_repo):
    # ONE grammar for what a release version looks like: a real rc row is a release, and a
    # dashed word is not a version at all. Retyping the pattern here got both backwards.
    repo = history_repo["repo"]
    _write_readme(repo, [("4.50.0rc1", "2026-01-07", "a real pre-release row"),
                         ("4.50.0-foo", "2026-01-07", "not a version at all")])
    c5 = _commit(repo, "readme: an rc row and a lookalike")
    material = ul.collect_range_material(history_repo["c4"], c5, git=_capture_for(repo))
    assert [row["version"] for row in material["releases"]] == ["4.50.0rc1"]
    assert material["omitted_rows"] == 1 and material["omitted_row_commits"] == [c5]


def test_material_ignores_a_merge_second_parent_diff(history_repo):
    # `-m` would diff the merge against its SECOND parent too and re-emit a row that the
    # first-parent line already had before the range — an old release presented as added.
    repo = history_repo["repo"]
    # A side branch from BEFORE the 1.4.0 row existed, with unrelated work.
    _git(repo, "checkout", "-q", "-b", "side", history_repo["c2"])
    (repo / "side.txt").write_text("side\n")
    side = _commit(repo, "side work")
    _git(repo, "checkout", "-q", "main")
    _git(repo, "merge", "-q", "--no-ff", "-m", "merge side", side)
    merged = _git(repo, "rev-parse", "HEAD")
    material = ul.collect_range_material(history_repo["c4"], merged, git=_capture_for(repo))
    assert {c["subject"] for c in material["commits"]} == {"merge side", "side work"}
    assert material["releases"] == [], "the merge added no README row on the first-parent line"


def test_material_reworded_row_is_first_wins_newest_first(history_repo):
    repo = history_repo["repo"]
    _write_readme(repo, [("1.4.0", "2026-01-04", "fourth, reworded"), ("1.2.0", "2026-01-03", "third")])
    c5 = _commit(repo, "reword 1.4.0 row")
    material = ul.collect_range_material(history_repo["base"], c5, git=_capture_for(repo))
    assert material["releases"][0]["text"] == "fourth, reworded"
    assert [row["version"] for row in material["releases"]] == ["1.4.0", "1.2.0", "1.1.1", "1.1.0"]


def test_material_unreadable_range_is_a_typed_failure_not_an_empty_one(history_repo):
    # A git that CANNOT read the range must never be recorded as "this update has
    # nothing to say": the two are opposite facts about the same state file.
    def broken(cmd):
        return (128, "", "fatal: bad object")

    with pytest.raises(ul.MaterialUnavailable):
        ul.collect_range_material(history_repo["base"], history_repo["c4"], git=broken)

    def readme_broken(cmd):
        if "README.md" in cmd:
            return (128, "", "fatal: bad object")
        return _capture_for(history_repo["repo"])(cmd)

    with pytest.raises(ul.MaterialUnavailable):
        ul.collect_range_material(history_repo["base"], history_repo["c4"], git=readme_broken)


def test_material_discloses_an_added_row_whose_version_is_not_one(history_repo):
    # A row the parser cannot read as a release is a row the author will never see, so it
    # is counted — while the table's own header and separator stay silent, carrying nothing.
    repo = history_repo["repo"]
    (repo / "README.md").write_text(
        "# Demo\n\n## Version History\n\n| Version | Date | Description |\n|---|---|---|\n"
        "| 6.114 | 2026-01-05 | opens like a version but is not one |\n"
        "| 1.4.0 | 2026-01-04 | fourth |\n",
        encoding="utf-8",
    )
    c5 = _commit(repo, "readme: a row with an unparsed version")
    material = ul.collect_range_material(history_repo["c4"], c5, git=_capture_for(repo))
    assert [row["version"] for row in material["releases"]] == []
    assert material["omitted_rows"] == 1, "the unreadable row is disclosed, the furniture is not"
    # …and the omission names the commit it can be read in (BIBLE P1: resolvable, not a count).
    assert material["omitted_row_commits"] == [c5]
    assert f"1 unreadable history row(s) omitted; read them in {c5}" in ul.material_text(material)

    # A release row is recognised by its own first cell, so a row that opens with a word
    # belongs to some other table and is passed over rather than reported as a lost release.
    (repo / "README.md").write_text(
        "# Demo\n\n## Version History\n\n| Version | Date | Description |\n|---|---|---|\n"
        "| version | 2026-01-06 | a row that only looks like a header |\n",
        encoding="utf-8",
    )
    c6 = _commit(repo, "readme: a row that looks like the header")
    lookalike = ul.collect_range_material(c5, c6, git=_capture_for(repo))
    assert lookalike["omitted_rows"] == 0 and lookalike["releases"] == []


def test_material_text_carries_full_provenance(history_repo):
    # Full commit shas and the commit each row came from: an omission has to stay resolvable.
    material = ul.collect_range_material(
        history_repo["base"], history_repo["c4"], git=_capture_for(history_repo["repo"]),
    )
    rendered = ul.material_text(material)
    assert history_repo["c4"] in rendered and history_repo["c1"] in rendered
    assert f"added in {history_repo['c3']}" in rendered
    assert history_repo["c4"][:8] + " " not in rendered.replace(history_repo["c4"], ""), "no bare 8-char prefixes"


def test_material_text_discloses_unreadable_commit_records_with_no_valid_commit_left():
    # Every git record malformed: the omission is said, the range is never called empty.
    material = {"commits": [], "omitted_commit_chunks": 2, "bodies_omitted": 0,
                "releases": [], "omitted_rows": 0, "omitted_row_commits": [], "rows_summarized": 0}
    text = ul.material_text(material)
    assert "2 commit record(s) git returned unreadably" in text
    assert "(no commits in this range)" not in text
    assert ul.material_text({**material, "omitted_commit_chunks": 0}) == "(no commits in this range)"


def test_material_text_discloses_malformed_rows_with_no_valid_row_left(history_repo):
    # Every candidate row malformed: there is nothing to print and the omission is
    # exactly what the author still has to be told.
    material = {"commits": [], "releases": [], "omitted_rows": 3, "rows_summarized": 0,
                "bodies_omitted": 0, "omitted_row_commits": ["f" * 40]}
    rendered = ul.material_text(material)
    assert "3 unreadable history row(s) omitted; read them in " + "f" * 40 in rendered


def test_material_empty_range_and_non_ancestor_base(history_repo):
    repo = history_repo["repo"]
    empty = ul.collect_range_material(history_repo["c4"], history_repo["c4"], git=_capture_for(repo))
    assert empty["commits"] == [] and empty["releases"] == []
    reverse = ul.collect_range_material(history_repo["c4"], history_repo["base"], git=_capture_for(repo))
    assert reverse["commits"] == [] and reverse["releases"] == []


def test_split_row_keeps_three_cells_only():
    assert ul._split_row("+| 1.2.3 | 2026-01-01 | a \\| b |") == ("1.2.3", "2026-01-01", "a | b")
    assert ul._split_row("+| 1.2.3 | only two |") is None


# ---------------------------------------------------------------------------
# check facts
# ---------------------------------------------------------------------------

def _status(**over):
    base = {
        "check_ok": True, "available": True, "current_sha": "a" * 40, "latest_sha": "b" * 40,
        "update_channel": "stable", "target_ref": "managed/main", "behind": 3, "ahead": 0,
        "checked_at": "2026-09-03T18:00:00+00:00",
    }
    base.update(over)
    return base


def _material():
    return {
        "base_sha": "a" * 40, "target_sha": "b" * 40,
        "commits": [{"sha": "b" * 40, "date": "2026-09-03", "subject": "s", "body": ""}],
        "releases": [{"version": "6.114.0", "date": "2026-09-01", "text": "row", "commit": "b" * 40}],
        "bodies_omitted": 0, "omitted_rows": 0,
        "versions": {"base": "6.113.5", "target": "6.114.0"},
    }


def _no_model(monkeypatch):
    """Any model construction or send during a check is a failure of the contract."""
    def refuse(*_a, **_k):
        raise AssertionError("a check must never call a model")

    monkeypatch.setattr("ouroboros.llm.LLMClient.__init__", refuse)
    monkeypatch.setattr("ouroboros.llm_observability.chat_observed", refuse)


def _git_versions(argv):
    if argv[:2] == ["git", "show"]:
        return 0, "6.114.0", ""
    return 1, "", ""


def _runs(conclusion):
    return {"status": "observed", "reason": "", "runs": [{"sha": "b" * 40, "status": "completed",
                                                          "conclusion": conclusion, "url": "u"}],
            "absent_shas": [], "missing_shas": []}


def test_refresh_writes_nothing_without_a_successful_check(tmp_path, monkeypatch):
    drive = tmp_path / "data"
    monkeypatch.setattr(ul, "collect_range_material", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no material")))
    assert ul.refresh_after_check(_status(check_ok=False), drive_root=drive) is None
    assert ul.refresh_after_check(_status(check_ok=None), drive_root=drive) is None
    assert not ul.record_path(drive).exists()


def test_refresh_records_the_checked_head_even_without_an_update(tmp_path, monkeypatch):
    drive = tmp_path / "data"
    monkeypatch.setattr(ul, "collect_range_material", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no material")))
    record = ul.refresh_after_check(_status(available=False, latest_sha=""), drive_root=drive)
    assert record["state"] == "none" and record["checked_head_sha"] == "a" * 40
    assert ul.project_letter(record, head_sha="a" * 40, latest_sha="") is None
    fact = ul.official_update_projection("a" * 40, drive_root=drive, state={"managed_update_cache": {
        "latest_sha": "", "available": False, "behind": 0, "ahead": 2, "checked_at": "t0"}})
    assert fact["status"] == "up_to_date" and fact["letter"] is None


def test_a_check_records_material_and_facts_without_a_model_or_a_description(tmp_path, monkeypatch):
    # A first check of a new range: the material goes beside the record, the record stays small,
    # and nothing is authored — the panel offers Write, the Runtime context points at the sources.
    drive = tmp_path / "data"
    _no_model(monkeypatch)
    monkeypatch.setattr(ul, "_default_git", lambda: _git_versions)
    monkeypatch.setattr(ul, "collect_range_material", lambda base, target, **k: _material())
    monkeypatch.setattr("ouroboros.update_qualification.post_merge_checks", lambda commits, url: _runs("success"))
    record = ul.refresh_after_check(_status(), drive_root=drive)
    assert record["state"] == "none" and record["text"] == "" and _revision(record) == 0
    assert record["checked_key"]["target_sha"] == "b" * 40 and record["checked_target_version"] == "6.114.0"
    assert record["qualification"]["target_sha"] == "b" * 40
    material = json.loads(ul.material_path(drive).read_text())
    assert material["key"] == record["checked_key"] and material["commits"] == 1
    assert "Commits in this range" in material["text"] and 'vcs_read(root="system_repo", action="show"' in material["text"]
    assert len(ul.record_path(drive).read_bytes()) < 8000, "the hot record never carries the material"
    view = ul.project_letter(record, head_sha="a" * 40, latest_sha="b" * 40)
    assert view["state"] == "none" and view["text"] == "" and view["checked_key"]["target_sha"] == "b" * 40
    cache = {"managed_update_cache": {"latest_sha": "b" * 40, "available": True, "behind": 3, "checked_at": "t",
                                      "update_channel": "stable"}}
    monkeypatch.setenv("OUROBOROS_UPDATE_CHANNEL", "stable")
    fact = ul.official_update_projection("a" * 40, drive_root=drive, state=cache)
    assert fact["status"] == "update_available" and fact["letter"]["state"] == "none"
    assert fact["description_sources"] == ul.DESCRIPTION_SOURCES
    assert fact["target"] == {"version": "6.114.0", "sha": "b" * 40}


def _revision(record):
    return record.get("revision") or 0


def test_a_same_range_check_refreshes_runs_and_never_stales_the_description(tmp_path, monkeypatch):
    # Owner 28.10: a new post-merge run is a fact about the same range, not a reason to call the
    # description stale or to write it again; the material is reused, not collected again.
    drive = tmp_path / "data"
    _no_model(monkeypatch)
    monkeypatch.setattr(ul, "_default_git", lambda: _git_versions)
    collected = []
    monkeypatch.setattr(ul, "collect_range_material", lambda base, target, **k: collected.append(target) or _material())
    monkeypatch.setattr("ouroboros.update_qualification.post_merge_checks", lambda commits, url: _runs("in_progress"))
    ul.refresh_after_check(_status(), drive_root=drive)
    record = dict(ul.read_record(drive), state="ready", text="what changes", revision=4, written_at="t-written",
                  key=ul._key_from_status(_status()))
    ul.atomic_write_json(ul.record_path(drive), record)
    monkeypatch.setattr("ouroboros.update_qualification.post_merge_checks", lambda commits, url: _runs("failure"))
    after = ul.refresh_after_check(_status(), drive_root=drive)
    assert collected == ["b" * 40], "the same range's material is reused"
    assert after["qualification"]["github"]["runs"][0]["conclusion"] == "failure"
    assert (after["text"], after["revision"], after["written_at"]) == ("what changes", 4, "t-written")
    view = ul.project_letter(after, head_sha="a" * 40, latest_sha="b" * 40)
    assert view["description_current"] is True and view["relation"] == "pending"


def test_a_moved_target_keeps_the_old_text_with_its_own_range(tmp_path, monkeypatch):
    drive = tmp_path / "data"
    (drive / "state").mkdir(parents=True)
    ul.atomic_write_json(ul.record_path(drive), _record(text="about b", revision=2))
    _no_model(monkeypatch)
    monkeypatch.setattr(ul, "_default_git", lambda: _git_versions)
    monkeypatch.setattr(ul, "collect_range_material",
                        lambda base, target, **k: dict(_material(), target_sha=target))
    monkeypatch.setattr("ouroboros.update_qualification.post_merge_checks", lambda commits, url: _runs("success"))
    record = ul.refresh_after_check(_status(latest_sha="c" * 40), drive_root=drive)
    assert record["text"] == "about b" and record["key"]["target_sha"] == "b" * 40 and record["revision"] == 2
    assert record["checked_key"]["target_sha"] == "c" * 40 and record["qualification"]["target_sha"] == "c" * 40
    assert json.loads(ul.material_path(drive).read_text())["key"]["target_sha"] == "c" * 40
    view = ul.project_letter(record, head_sha="a" * 40, latest_sha="c" * 40)
    assert view["relation"] == "superseded" and view["description_current"] is False
    assert view["key"]["target_sha"] == "b" * 40 and view["target_version"] == "6.114.0"


def test_an_unreadable_range_is_a_check_fact_and_the_description_stays(tmp_path, monkeypatch):
    drive = tmp_path / "data"
    (drive / "state").mkdir(parents=True)
    ul.atomic_write_json(ul.record_path(drive), _record(text="the description of 6.114.0", revision=1))
    monkeypatch.setattr(ul, "_default_git", lambda: _git_versions)
    monkeypatch.setattr(ul, "collect_range_material",
                        lambda *a, **k: (_ for _ in ()).throw(ul.MaterialUnavailable("git log failed (rc=128)")))
    record = ul.refresh_after_check(_status(latest_sha="c" * 40), drive_root=drive)
    assert record["state"] == "ready" and record["text"] == "the description of 6.114.0" and record["revision"] == 1
    assert record["material_error"]["kind"] == "material_unavailable" and "git log failed" in record["material_error"]["text"]
    view = ul.project_letter(record, head_sha="a" * 40, latest_sha="c" * 40)
    assert view["error_kind"] == "material_unavailable" and view["latest_failed_key"]["target_sha"] == "c" * 40
    assert view["text"] == "the description of 6.114.0" and view["relation"] == "superseded"
    assert not ul.material_path(drive).exists()
    # A later readable check clears the fact; the description is still untouched.
    monkeypatch.setattr(ul, "collect_range_material", lambda base, target, **k: dict(_material(), target_sha=target))
    monkeypatch.setattr("ouroboros.update_qualification.post_merge_checks", lambda commits, url: _runs("success"))
    record = ul.refresh_after_check(_status(latest_sha="c" * 40), drive_root=drive)
    assert "material_error" not in record and record["text"] == "the description of 6.114.0"


def test_check_facts_land_on_whatever_the_record_holds_when_they_are_written(tmp_path, monkeypatch):
    # Git and GitHub are read outside the record lock; a description saved meanwhile (a direct
    # turn, a wake or the requested task) is compared on write and stays whole.
    drive = tmp_path / "data"
    monkeypatch.setattr(ul, "_default_git", lambda: _git_versions)
    ul.refresh_after_check(_status(available=False, latest_sha="a" * 40), drive_root=drive)
    ul.atomic_write_json(ul.record_path(drive), dict(ul.read_record(drive), checked_key=ul._key_from_status(_status())))

    def collect(base, target, **_k):
        saved = ul.save_description("written while the check read Git", base, target, 0,
                                    author={"task_id": "turn1", "model": "m"}, drive_root=drive, git=_git_versions)
        assert saved["outcome"] == "saved"
        return _material()

    monkeypatch.setattr(ul, "collect_range_material", collect)
    monkeypatch.setattr("ouroboros.update_qualification.post_merge_checks", lambda commits, url: _runs("success"))
    record = ul.refresh_after_check(_status(), drive_root=drive)
    assert record["text"] == "written while the check read Git" and record["revision"] == 1
    assert record["author"]["task_id"] == "turn1" and record["qualification"]["target_sha"] == "b" * 40


def test_a_check_waits_for_the_record_lock_and_writes_nothing_past_it(tmp_path, monkeypatch):
    # The lock is the file the worker's tool takes too: a held one is never bypassed.
    from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock

    drive = tmp_path / "data"
    (drive / "state").mkdir(parents=True)
    ul.atomic_write_json(ul.record_path(drive), _record(text="kept"))
    before = ul.record_path(drive).read_bytes()
    monkeypatch.setattr(ul, "RECORD_LOCK_WAIT_SEC", 0.2)
    monkeypatch.setattr(ul, "_default_git", lambda: _git_versions)
    lock = ul.record_path(drive).parent / ul.LOCK_REL.name
    fd = acquire_exclusive_file_lock(lock, timeout_sec=1.0)
    try:
        assert ul.refresh_after_check(_status(available=False, latest_sha="a" * 40), drive_root=drive)["text"] == "kept"
        assert ul.record_path(drive).read_bytes() == before
    finally:
        release_exclusive_file_lock(lock, fd)
    assert ul.refresh_after_check(_status(available=False, latest_sha="a" * 40), drive_root=drive)["checked_key"]


def test_refresh_is_single_flight(tmp_path, monkeypatch):
    # A held check is waited for (bounded by the fetch ceiling), never doubled; past the bound
    # the current record is returned as it is.
    drive = tmp_path / "data"
    monkeypatch.setattr(ul, "collect_range_material", lambda *a, **k: (_ for _ in ()).throw(AssertionError("busy")))
    monkeypatch.setattr("ouroboros.update_channels.get_managed_update_fetch_timeout_sec", lambda: 0.2)
    assert ul._CHECK_FLIGHT.acquire(blocking=False)
    try:
        assert ul.refresh_after_check(_status(), drive_root=drive) is None
    finally:
        ul._CHECK_FLIGHT.release()


def test_mark_checked_records_the_official_target_version(tmp_path, monkeypatch):
    # An up-to-date check (latest == HEAD, no description) still names the official target's
    # version, so the Runtime fact can say which version "up to date" means.
    drive = tmp_path / "data"
    monkeypatch.setattr(ul, "_default_git",
                        lambda: (lambda argv: (0, "6.114.0\n", "") if argv[:2] == ["git", "show"] else (1, "", "")))
    record = ul.refresh_after_check(_status(available=False, latest_sha="a" * 40, behind=0), drive_root=drive)
    assert record["state"] == "none" and record["target_version"] == "6.114.0"
    assert record["key"]["target_sha"] == "a" * 40 and record["checked_head_sha"] == "a" * 40
    cache = {"managed_update_cache": {"latest_sha": "a" * 40, "available": False, "behind": 0, "checked_at": "t"}}
    fact = ul.official_update_projection("a" * 40, drive_root=drive, state=cache)
    assert fact["status"] == "up_to_date" and fact["target"] == {"version": "6.114.0", "sha": "a" * 40}
    assert fact["letter"] is None


def test_refresh_never_raises(tmp_path, monkeypatch):
    drive = tmp_path / "data"
    monkeypatch.setattr(ul, "collect_range_material", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("git down")))
    assert ul.refresh_after_check(_status(), drive_root=drive) is None

# ---------------------------------------------------------------------------
# projection
# ---------------------------------------------------------------------------

def _record(**over):
    record = {"schema": 1, "key": {"base_sha": "a" * 40, "target_sha": "b" * 40, "update_channel": "stable",
                                   "target_ref": "managed/main"},
              "checked_head_sha": "a" * 40, "state": "ready", "text": "letter", "author_version": "6.113.5",
              "target_version": "6.114.0", "model": "m", "written_at": "t", "attempt_id": "att",
              "error_kind": "", "error_text": "", "last_good": None}
    record.update(over)
    return record


@pytest.mark.parametrize("head, latest, relation", [
    ("a" * 40, "b" * 40, "pending"),
    ("b" * 40, "b" * 40, "applied"),
    ("b" * 40, "c" * 40, "applied"),
    ("a" * 40, "c" * 40, "superseded"),
    ("d" * 40, "b" * 40, "other"),
])
def test_project_letter_relations(head, latest, relation):
    view = ul.project_letter(_record(), head_sha=head, latest_sha=latest)
    assert view["relation"] == relation and view["state"] == "ready" and view["text"] == "letter"


def test_project_letter_applied_from_the_recorded_ancestry_fact():
    # A divergent install applies the official target as a merge commit: HEAD never equals
    # the target, and the CHECK (which has git) records that the target is inside HEAD.
    merged = "e" * 40
    rec = _record(checked_head_sha=merged, target_in_head=True)
    assert ul.project_letter(rec, head_sha=merged, latest_sha="b" * 40)["relation"] == "applied"
    # The fact is about the head the check described; a HEAD that moved on is not covered.
    assert ul.project_letter(rec, head_sha="f" * 40, latest_sha="b" * 40)["relation"] == "other"
    # Without the fact a moved HEAD stays "other" (no git on the hot path).
    assert ul.project_letter(_record(checked_head_sha=merged), head_sha=merged, latest_sha="b" * 40)["relation"] == "other"


def test_the_check_records_the_shown_target_in_head_and_both_surfaces_read_it(tmp_path, monkeypatch):
    # One proof, recorded where git is (the check), read by the panel and the Runtime
    # fact alike — so the two can never describe one install differently.
    drive = tmp_path / "data"
    (drive / "state").mkdir(parents=True)
    ul.record_path(drive).write_text(json.dumps(_record()))
    merged, target = "e" * 40, "b" * 40
    calls = []

    def git(argv):
        calls.append(argv)
        if argv[:3] == ["git", "merge-base", "--is-ancestor"]:
            return (0, "", "") if argv[3:] == [target, merged] else (1, "", "")
        return (0, "6.114.0", "") if argv[:2] == ["git", "show"] else (1, "", "")

    monkeypatch.setattr(ul, "_default_git", lambda: git)
    monkeypatch.setattr(ul, "collect_range_material", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no update")))
    record = ul.refresh_after_check(_status(current_sha=merged, available=False, latest_sha=target, behind=0), drive_root=drive)
    assert record["target_in_head"] is True and record["checked_head_sha"] == merged
    assert any(a[:3] == ["git", "merge-base", "--is-ancestor"] for a in calls), "the check asks git, the readers do not"
    calls.clear()
    panel = ul.project_letter_for_panel({"current_sha": merged, "latest_sha": ""}, drive_root=drive)
    cache = {"managed_update_cache": {"latest_sha": target, "available": False, "behind": 0,
                                      "checked_at": "t", "update_channel": "stable"}}
    fact = ul.official_update_projection(merged, drive_root=drive, state=cache)
    assert panel["relation"] == "applied" and fact["letter"]["relation"] == "applied"
    assert calls == [], "no git on the reading paths"
    # A HEAD that is NOT a descendant keeps the honest "other" on both surfaces.
    ul.record_path(drive).write_text(json.dumps(_record()))
    stranger = "d" * 40
    ul.refresh_after_check(_status(current_sha=stranger, available=False, latest_sha=target, behind=0), drive_root=drive)
    assert ul.project_letter_for_panel({"current_sha": stranger, "latest_sha": ""}, drive_root=drive)["relation"] == "other"
    assert ul.official_update_projection(stranger, drive_root=drive, state=cache)["letter"]["relation"] == "other"


def test_a_kept_letter_is_related_by_ITS_target_on_both_surfaces(tmp_path, monkeypatch):
    # a->b letter, a failed rewrite for a->c, then c applied as merge H (so b is inside H).
    # The text shown is the kept one about b; the check records the ancestry for THAT
    # letter, and both surfaces call it applied, with b's own version beside b's own sha.
    drive = tmp_path / "data"
    (drive / "state").mkdir(parents=True)
    kept = _record(text="the letter about 6.114.0")
    failed = _record(key=dict(_record()["key"], target_sha="c" * 40), state="failed", text="",
                     error_kind="provider_unavailable", target_version="6.115.0", last_good=kept)
    ul.record_path(drive).write_text(json.dumps(failed))
    merged = "e" * 40
    monkeypatch.setattr(ul, "_default_git", lambda: (lambda argv: (0, "", "") if argv[:3] == ["git", "merge-base", "--is-ancestor"] else (1, "", "")))
    monkeypatch.setattr(ul, "collect_range_material", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no update")))
    ul.refresh_after_check(_status(current_sha=merged, available=False, latest_sha="c" * 40, behind=0), drive_root=drive)
    panel = ul.project_letter_for_panel({"current_sha": merged, "latest_sha": "c" * 40, "available": False}, drive_root=drive)
    assert panel["relation"] == "applied" and panel["text"] == "the letter about 6.114.0"
    assert panel["target_version"] == "6.114.0" and panel["has_last_good"] is True
    cache = {"managed_update_cache": {"latest_sha": "c" * 40, "available": False, "behind": 0,
                                      "checked_at": "t2", "update_channel": "stable"}}
    fact = ul.official_update_projection(merged, drive_root=drive, state=cache)
    assert fact["letter"]["relation"] == "applied" and fact["letter"]["text"] == "the letter about 6.114.0"
    # The version travels with the sha it belongs to: the failed range's 6.115.0 is never
    # paired with the kept letter's target.
    assert fact["target"] == {"version": "", "sha": "c" * 40}


def test_official_update_projection_applied_only_while_the_check_describes_this_head(tmp_path):
    drive = tmp_path / "data"
    (drive / "state").mkdir(parents=True)
    merged = "e" * 40
    ul.record_path(drive).write_text(json.dumps(_record(checked_head_sha=merged, target_in_head=True)))
    cache = {"managed_update_cache": {"latest_sha": "b" * 40, "available": False, "behind": 0,
                                      "checked_at": "t1", "update_channel": "stable"}}
    assert ul.official_update_projection(merged, drive_root=drive, state=cache)["letter"]["relation"] == "applied"
    # HEAD moved on after that check: the recorded fact is about another head.
    moved = ul.official_update_projection("f" * 40, drive_root=drive, state=cache)
    assert moved["status"] == "moved_since_check" and moved["letter"]["relation"] == "other"


def test_official_update_projection_applied_after_divergent_merge(tmp_path):
    drive = tmp_path / "data"
    (drive / "state").mkdir(parents=True)
    ul.record_path(drive).write_text(json.dumps(_record(checked_head_sha="e" * 40, target_in_head=True)))
    cache = {"managed_update_cache": {"latest_sha": "b" * 40, "available": False, "behind": 0, "ahead": 2,
                                      "checked_at": "t1", "update_channel": "stable"}}
    fact = ul.official_update_projection("e" * 40, drive_root=drive, state=cache)
    assert fact["status"] == "up_to_date" and fact["letter"]["relation"] == "applied"
    assert fact["target"] == {"version": "6.114.0", "sha": "b" * 40}


def test_official_update_projection_after_a_channel_switch_is_unchecked(tmp_path, monkeypatch):
    # The cached check described Stable; the owner switched to Development. That check says
    # nothing about the active channel, so the fact is unchecked — the letter stays, as history.
    drive = tmp_path / "data"
    (drive / "state").mkdir(parents=True)
    ul.record_path(drive).write_text(json.dumps(_record()))
    cache = {"managed_update_cache": {"latest_sha": "b" * 40, "available": False, "behind": 0,
                                      "checked_at": "t1", "update_channel": "stable"}}
    monkeypatch.setenv("OUROBOROS_UPDATE_CHANNEL", "development")
    fact = ul.official_update_projection("a" * 40, drive_root=drive, state=cache)
    assert fact["status"] == "unchecked" and fact["target"] is None and fact["update_channel"] == "development"
    assert fact["letter"]["text"] == "letter" and fact["letter"]["relation"] in ("superseded", "other")
    monkeypatch.setenv("OUROBOROS_UPDATE_CHANNEL", "stable")
    assert ul.official_update_projection("a" * 40, drive_root=drive, state=cache)["status"] == "up_to_date"


def test_project_letter_failed_with_last_good_shows_previous_text_and_provenance():
    record = _record(state="failed", text="", error_kind="timeout", error_text="slow", written_at="t-fail",
                     author_version="6.114.0", last_good=_record(text="older letter", written_at="t-good"))
    view = ul.project_letter(record, head_sha="a" * 40, latest_sha="b" * 40)
    assert view["state"] == "failed" and view["text"] == "older letter" and view["has_last_good"] is True
    assert view["error_kind"] == "timeout"
    assert view["written_at"] == "t-good" and view["author_version"] == "6.113.5"
    assert ul.project_letter(None, head_sha="a" * 40, latest_sha="") is None


def test_project_letter_kept_last_good_is_related_by_its_own_range():
    # The target moved (b -> c) and the rewrite for a->c failed: the kept letter is
    # about a->b, so it reads as superseded, never as the pending letter for a->c.
    kept = _record(text="older letter", written_at="t-good")
    record = _record(key=dict(_record()["key"], target_sha="c" * 40), state="failed", text="",
                     error_kind="provider_unavailable", error_text="503", target_version="6.115.0",
                     last_good=kept)
    view = ul.project_letter(record, head_sha="a" * 40, latest_sha="c" * 40)
    assert view["relation"] == "superseded" and view["text"] == "older letter" and view["has_last_good"] is True
    assert view["target_version"] == "6.114.0" and view["key"]["target_sha"] == "b" * 40
    assert view["error_kind"] == "provider_unavailable"
    # Once the kept letter's own target runs, it is "applied" - the failed range never was.
    assert ul.project_letter(record, head_sha="b" * 40, latest_sha="c" * 40)["relation"] == "applied"


def test_official_update_projection_states(tmp_path):
    drive = tmp_path / "data"
    (drive / "state").mkdir(parents=True)
    head = "a" * 40
    assert ul.official_update_projection(head, drive_root=drive, state={})["status"] == "unchecked"
    cache = {"managed_update_cache": {"latest_sha": "b" * 40, "available": True, "behind": 3, "ahead": 0,
                                      "checked_at": "t0", "update_channel": "stable"}}
    # A cache from a check this code never recorded (no record at all): never invented.
    assert ul.official_update_projection(head, drive_root=drive, state=cache)["status"] == "moved_since_check"
    ul.record_path(drive).write_text(json.dumps(_record()))
    fact = ul.official_update_projection(head, drive_root=drive, state=cache)
    assert fact["status"] == "update_available" and fact["letter"]["relation"] == "pending"
    assert fact["target"] == {"version": "6.114.0", "sha": "b" * 40} and fact["status_as_of"] == "t0"
    assert fact["running"]["sha"] == head and fact["behind"] == 3
    applied = ul.official_update_projection("b" * 40, drive_root=drive, state=cache)
    assert applied["status"] == "up_to_date" and applied["letter"]["relation"] == "applied"
    moved = ul.official_update_projection("e" * 40, drive_root=drive, state=cache)
    assert moved["status"] == "moved_since_check" and moved["letter"]["relation"] == "other"
    # A newer official target than the letter's: the target version is not the letter's.
    newer = dict(cache); newer["managed_update_cache"] = dict(cache["managed_update_cache"], latest_sha="c" * 40)
    superseded = ul.official_update_projection(head, drive_root=drive, state=newer)
    assert superseded["target"] == {"version": "", "sha": "c" * 40}
    assert superseded["letter"]["relation"] == "superseded"


def test_official_update_projection_unresolved_head_is_unknown_not_moved(tmp_path):
    # context.py hands the projection its "unknown" sentinel when git could not be read;
    # comparing that with real SHAs would claim the body moved when nothing did.
    drive = tmp_path / "data"
    (drive / "state").mkdir(parents=True)
    ul.record_path(drive).write_text(json.dumps(_record()))
    cache = {"managed_update_cache": {"latest_sha": "b" * 40, "available": True, "behind": 3, "checked_at": "t"}}
    for head in ("unknown", ""):
        fact = ul.official_update_projection(head, drive_root=drive, state=cache)
        assert fact == {"status": "unknown", "error": "head_unresolved"}, head


def test_official_update_projection_never_raises(tmp_path):
    fact = ul.official_update_projection("a" * 40, drive_root=tmp_path / "missing", state={"managed_update_cache": "bad"})
    assert fact["status"] == "unchecked"


def test_boot_check_records_the_facts_before_the_readiness_broadcast(monkeypatch):
    import server
    import supervisor.git_ops as git_ops
    import supervisor.update_merge as update_merge

    calls = []
    monkeypatch.setattr(server, "_wait_for_supervisor_update_finalize", lambda: False)
    monkeypatch.setattr(update_merge, "finalize_managed_update_on_boot",
                        lambda supervisor_ready, **_kwargs: {"finalized": False, "rolled_back": False})
    monkeypatch.setattr(git_ops, "compute_managed_update_status", lambda fetch: _status())
    monkeypatch.setattr(ul, "refresh_after_check", lambda status, **k: calls.append(("letter", status["latest_sha"])))
    monkeypatch.setattr(server, "broadcast_ws_sync", lambda payload: calls.append((payload["type"], "")))

    server._boot_managed_update_tasks()

    assert calls == [("letter", "b" * 40), ("update_status_ready", "")]
