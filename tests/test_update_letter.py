"""The update letter: range material, the accounted LIGHT call, storage, and projection."""

from __future__ import annotations

import json
import pathlib
import subprocess
import threading
import time

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

def test_material_recovers_rows_from_commit_diffs_and_lists_first_parent_commits(history_repo):
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
                             "omitted_row_commits", "rows_summarized", "versions"}


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
    assert [c["subject"] for c in material["commits"]] == ["merge side"]
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
# generation
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
        "commits": [{"sha": "b" * 40, "date": "2026-09-03", "subject": "s", "body": ""}],
        "releases": [{"version": "6.114.0", "date": "2026-09-01", "text": "row", "commit": "b" * 40}],
        "bodies_omitted": 0, "omitted_rows": 0,
        "versions": {"base": "6.113.5", "target": "6.114.0"},
    }


@pytest.fixture
def letter_env(tmp_path, monkeypatch):
    drive = tmp_path / "data"
    repo = tmp_path / "repo"
    (drive / "state").mkdir(parents=True)
    (drive / "memory").mkdir()
    repo.mkdir()
    monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", "test/light")
    monkeypatch.delenv("USE_LOCAL_LIGHT", raising=False)
    monkeypatch.setattr("ouroboros.provider_models.model_has_credentials", lambda model: True)
    calls = []

    class FakePlan:
        initial_mode = "max"

        def __init__(self, task):
            self.task = task

        def messages_for(self, mode):
            return [{"role": "system", "content": f"identity:{mode}"}, {"role": "user", "content": self.task["text"]}]

    def fake_plan(env, memory, task):
        calls.append(("context", task))
        return FakePlan(task)

    monkeypatch.setattr(ul, "_fit_plan", fake_plan)
    return {"drive": drive, "repo": repo, "calls": calls}


def test_write_letter_ready_record_carries_attempt_and_versions(letter_env, monkeypatch):
    seen = {}

    def fake_chat(client, *, drive_root, **kwargs):
        seen.update(kwargs)
        return {"content": "  One short paragraph.  "}, {"ledger_attempt_ids": ["att-1", "att-2"]}

    monkeypatch.setattr(ul, "_chat", fake_chat)
    record = ul.write_letter(_status(), _material(), drive_root=letter_env["drive"])

    assert record["state"] == "ready" and record["text"] == "One short paragraph."
    assert record["attempt_id"] == "att-2" and record["attempt_ids"] == ["att-1", "att-2"]
    assert record["model"] == "test/light"
    assert record["key"]["base_sha"] == "a" * 40 and record["checked_head_sha"] == "a" * 40
    assert record["target_version"] == "6.114.0" and record["error_kind"] == ""
    assert seen["model"] == "test/light" and seen["max_tokens"] == ul.UPDATE_LETTER_MAX_TOKENS
    assert seen["reasoning_effort"] == "low" and seen["tools"] is None
    task = letter_env["calls"][0][1]
    assert task["id"] == ul.SYSTEM_TASK_ID and task["model"] == "test/light"
    assert "[UPDATE LETTER REQUEST]" in task["text"] and "6.114.0" in task["text"]
    assert "ONE short paragraph" in seen["messages"][-1]["content"]


def test_write_letter_stores_the_model_s_shape_without_policing_it(letter_env, monkeypatch):
    # The shape is the mind's ceiling, not a host gate (owner decision; reviewed three
    # times): an oddly shaped answer is stored and shown, never edited and never thrown
    # away. What bounds it is the output budget, and the panel's sanitizing renderer.
    odd = "# A heading\n\n- one\n- two\n\nAnd a second paragraph."
    monkeypatch.setattr(ul, "_chat", lambda *a, **k: ({"content": odd}, {"ledger_attempt_ids": ["att"]}))
    record = ul.write_letter(_status(), _material(), drive_root=letter_env["drive"])
    assert record["state"] == "ready" and record["text"] == odd
    # Nothing in the module promises the host enforces one paragraph.
    source = (pathlib.Path(ul.__file__)).read_text(encoding="utf-8")
    floor = source.split("Floor (host code):", 1)[1].split("Ceiling", 1)[0]
    assert "paragraph" not in floor, "the floor must not claim a shape the host does not enforce"


def test_write_letter_shares_one_ceiling_between_the_slot_wait_and_the_call(letter_env, monkeypatch):
    # The slot wait spends the SAME budget the provider call does: a busy slot must not
    # hand the transport a second full window.
    monkeypatch.setattr(ul, "_letter_timeout_sec", lambda: 10.0)
    import ouroboros.model_concurrency as mc
    from contextlib import contextmanager

    @contextmanager
    def slow_slot(model, use_local, deadline_ts=None):
        monkeypatch.setattr(ul.time, "time", lambda: real_now + 7.0)
        yield

    real_now = ul.time.time()
    monkeypatch.setattr(mc, "model_call_slot", slow_slot)
    seen = {}

    def fake_chat(client, *, drive_root, **kwargs):
        seen.update(kwargs)
        return {"content": "A paragraph."}, {"ledger_attempt_ids": ["att"]}

    monkeypatch.setattr(ul, "_chat", fake_chat)
    record = ul.write_letter(_status(), _material(), drive_root=letter_env["drive"])
    assert record["state"] == "ready"
    assert 2.5 < seen["timeout"] < 3.5, f"the call gets what the slot left, not 10s: {seen['timeout']}"


def test_write_letter_reports_a_ceiling_spent_on_the_slot_as_a_timeout(letter_env, monkeypatch):
    monkeypatch.setattr(ul, "_letter_timeout_sec", lambda: 10.0)
    import ouroboros.model_concurrency as mc
    from contextlib import contextmanager

    real_now = ul.time.time()

    @contextmanager
    def exhausting_slot(model, use_local, deadline_ts=None):
        monkeypatch.setattr(ul.time, "time", lambda: real_now + 11.0)
        yield

    monkeypatch.setattr(mc, "model_call_slot", exhausting_slot)
    monkeypatch.setattr(ul, "_chat", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no call past the ceiling")))
    record = ul.write_letter(_status(), _material(), drive_root=letter_env["drive"])
    assert record["state"] == "failed" and record["error_kind"] == "timeout"
    assert "waiting for a model slot" in record["error_text"]


def test_write_letter_reads_the_global_budget_from_its_one_resolver(letter_env, monkeypatch):
    # An absent TOTAL_BUDGET is the product default, a non-positive value is the owner's
    # "no limit": both answers belong to settings_setup_contract.resolve_total_budget_usd,
    # the resolver every other reader uses — never an inline env read with its own fallback.
    monkeypatch.setattr("ouroboros.settings_setup_contract.resolve_total_budget_usd", lambda: 123.5)
    seen = {}

    class Scope:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    import ouroboros.usage_accounting as ua
    monkeypatch.setattr(ua, "UsageScope", Scope)
    monkeypatch.setattr(ul, "_chat", lambda *a, **k: ({"content": "A paragraph."}, {"ledger_attempt_ids": ["att"]}))
    record = ul.write_letter(_status(), _material(), drive_root=letter_env["drive"])
    assert record["state"] == "ready" and seen["global_limit_usd"] == 123.5


def test_write_letter_without_light_credentials_fails_typed_and_never_calls(letter_env, monkeypatch):
    monkeypatch.setattr("ouroboros.provider_models.model_has_credentials", lambda model: False)
    monkeypatch.setattr(ul, "_chat", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not call")))
    record = ul.write_letter(_status(), _material(), drive_root=letter_env["drive"])
    assert record["state"] == "failed" and record["error_kind"] == "no_credentials"
    assert "test/light" in record["error_text"] and record["text"] == ""


@pytest.mark.parametrize("exc, kind", [
    (TimeoutError("slow"), "timeout"),
    (RuntimeError("boom"), "provider_unavailable"),
])
def test_write_letter_failures_are_typed(letter_env, monkeypatch, exc, kind):
    def fake_chat(client, *, drive_root, **kwargs):
        raise exc

    monkeypatch.setattr(ul, "_chat", fake_chat)
    record = ul.write_letter(_status(), _material(), drive_root=letter_env["drive"])
    assert record["state"] == "failed" and record["error_kind"] == kind
    assert "boom" in record["error_text"] or "slow" in record["error_text"]


def test_write_letter_budget_exhausted_is_typed(letter_env, monkeypatch):
    from ouroboros.usage_accounting import BudgetExceeded

    def fake_chat(client, *, drive_root, **kwargs):
        raise BudgetExceeded("global budget exhausted")

    monkeypatch.setattr(ul, "_chat", fake_chat)
    record = ul.write_letter(_status(), _material(), drive_root=letter_env["drive"])
    assert record["error_kind"] == "budget_exhausted"


def test_write_letter_empty_response_is_typed(letter_env, monkeypatch):
    monkeypatch.setattr(ul, "_chat", lambda client, *, drive_root, **kw: ({"content": "   "}, {}))
    record = ul.write_letter(_status(), _material(), drive_root=letter_env["drive"])
    assert record["state"] == "failed" and record["error_kind"] == "empty_response"


# ---------------------------------------------------------------------------
# refresh seam
# ---------------------------------------------------------------------------

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


def test_refresh_writes_record_and_keeps_last_good_on_failure(tmp_path, monkeypatch):
    drive = tmp_path / "data"
    monkeypatch.setattr(ul, "collect_range_material", lambda base, target, **k: _material())
    ready = {"schema": 1, "key": ul._key_from_status(_status()), "checked_head_sha": "a" * 40,
             "state": "ready", "text": "first letter", "author_version": "6.113.5",
             "target_version": "6.114.0", "model": "m", "written_at": "t1", "attempt_id": "att-1",
             "error_kind": "", "error_text": "", "last_good": None}
    monkeypatch.setattr(ul, "write_letter", lambda status, material, **k: dict(ready))
    record = ul.refresh_after_check(_status(), drive_root=drive)
    assert record["state"] == "ready" and ul.read_record(drive)["text"] == "first letter"

    failed = dict(ready, state="failed", text="", error_kind="timeout", written_at="t2")
    monkeypatch.setattr(ul, "write_letter", lambda status, material, **k: dict(failed))
    record = ul.refresh_after_check(_status(), drive_root=drive)
    assert record["state"] == "failed" and record["last_good"]["text"] == "first letter"
    stored = json.loads(ul.record_path(drive).read_text())
    assert stored["last_good"]["text"] == "first letter"

    # An applied update (available=False) leaves the letter untouched.
    kept = ul.refresh_after_check(_status(available=False), drive_root=drive)
    assert kept["last_good"]["text"] == "first letter"

    # A newer target whose letter fails still carries the older good letter (D-KEEP).
    moved = dict(failed, key=ul._key_from_status(_status(latest_sha="c" * 40)), target_version="6.115.0")
    monkeypatch.setattr(ul, "write_letter", lambda status, material, **k: dict(moved))
    record = ul.refresh_after_check(_status(latest_sha="c" * 40), drive_root=drive)
    assert record["last_good"]["text"] == "first letter"
    view = ul.project_letter(record, head_sha="a" * 40, latest_sha="c" * 40)
    assert view["text"] == "first letter" and view["target_version"] == "6.114.0"
    assert view["state"] == "failed" and view["has_last_good"] is True


def test_refresh_records_a_typed_failure_when_the_range_cannot_be_read(tmp_path, monkeypatch):
    drive = tmp_path / "data"
    (drive / "state").mkdir(parents=True)
    good = _record(text="the letter about 6.114.0")
    ul.atomic_write_json(ul.record_path(drive), good)
    monkeypatch.setattr(ul, "collect_range_material",
                        lambda *a, **k: (_ for _ in ()).throw(ul.MaterialUnavailable("git log failed (rc=128)")))
    monkeypatch.setattr(ul, "write_letter",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no model call without material")))
    record = ul.refresh_after_check(_status(), drive_root=drive)
    assert record["state"] == "failed" and record["error_kind"] == "material_unavailable"
    assert "git log failed" in record["error_text"]
    # D-KEEP: the previous good letter survives an unreadable range.
    assert record["last_good"]["text"] == "the letter about 6.114.0"
    view = ul.project_letter(record, head_sha="a" * 40, latest_sha="b" * 40)
    assert view["text"] == "the letter about 6.114.0" and view["has_last_good"] is True


def test_mark_checked_takes_the_same_lock_as_the_writer(tmp_path, monkeypatch):
    # A no-update check that read the record before a letter landed must not write its
    # stale copy back over it: every record write goes through the one lock.
    drive = tmp_path / "data"
    (drive / "state").mkdir(parents=True)
    monkeypatch.setattr(ul, "_letter_timeout_sec", lambda: 0.2)
    monkeypatch.setattr(ul, "_default_git", lambda: (lambda argv: (0, "6.114.0", "")))
    assert ul._REFRESH_LOCK.acquire(blocking=False)
    try:
        assert ul.refresh_after_check(_status(available=False), drive_root=drive) is None
        assert not ul.record_path(drive).exists(), "the letterless mark must not bypass the lock"
    finally:
        ul._REFRESH_LOCK.release()
    assert ul.refresh_after_check(_status(available=False), drive_root=drive)["state"] == "none"


def test_refresh_is_single_flight(tmp_path, monkeypatch):
    # A held lock is waited for (bounded by the letter timeout), never bypassed by a
    # second physical write; past the bound the current record is returned as it is.
    drive = tmp_path / "data"
    monkeypatch.setattr(ul, "collect_range_material", lambda *a, **k: (_ for _ in ()).throw(AssertionError("busy")))
    monkeypatch.setattr(ul, "_letter_timeout_sec", lambda: 0.2)
    assert ul._REFRESH_LOCK.acquire(blocking=False)
    try:
        assert ul.refresh_after_check(_status(), drive_root=drive) is None
    finally:
        ul._REFRESH_LOCK.release()


def test_refresh_waits_for_and_shares_the_in_flight_letter(tmp_path, monkeypatch):
    # Two fetching checks for the same key (two dashboards clicking Check): the second
    # returns the letter the first just wrote instead of paying for a second one.
    drive = tmp_path / "data"
    (drive / "state").mkdir(parents=True)
    monkeypatch.setattr(ul, "collect_range_material",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("the waiter must share, not rewrite")))
    monkeypatch.setattr(ul, "_letter_timeout_sec", lambda: 5.0)
    status = _status()
    key = ul._key_from_status(status)
    assert ul._REFRESH_LOCK.acquire(blocking=False)

    def writer():
        time.sleep(0.2)
        record = _record(key=key)
        ul.atomic_write_json(ul.record_path(drive), record)
        ul._note_written(key, drive, record)
        ul._REFRESH_LOCK.release()

    threading.Thread(target=writer, daemon=True).start()
    record = ul.refresh_after_check(status, drive_root=drive)
    assert record["text"] == "letter" and record["key"] == key and not ul._REFRESH_LOCK.locked()

    # The share is per DATA ROOT too: the same key against another root is not the same letter.
    other = tmp_path / "other"
    (other / "state").mkdir(parents=True)
    monkeypatch.setattr(ul, "collect_range_material", lambda *a, **k: _material())
    monkeypatch.setattr(ul, "write_letter", lambda *a, **k: _record(key=key, text="the other root's letter"))
    assert ul._REFRESH_LOCK.acquire(blocking=False)
    threading.Thread(target=lambda: (time.sleep(0.2), ul._REFRESH_LOCK.release()), daemon=True).start()
    assert ul.refresh_after_check(status, drive_root=other)["text"] == "the other root's letter"


def test_mark_checked_records_the_official_target_version(tmp_path, monkeypatch):
    # An up-to-date check (latest == HEAD, no letter) still names the official target's
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


class _Projection:
    def __init__(self, fits, label):
        self.fits_known_window = fits
        self.label = label

    def system_message(self):
        return {"role": "system", "content": self.label}


class _Plan:
    def __init__(self, preferred, max_fits, low_fits):
        self.initial_mode = preferred
        self.max_projection = _Projection(max_fits, "max")
        self.low_projection = _Projection(low_fits, "low")

    def projection(self, mode):
        return self.low_projection if mode == "low" else self.max_projection

    def messages_for(self, mode):
        return [self.projection(mode).system_message(), {"role": "user", "content": "req"}]


def test_write_letter_sends_the_owner_mode_and_retries_low_only_on_an_actual_overflow(letter_env, monkeypatch):
    # DEVELOPMENT "Context mode": predicted pressure never swaps in Low — the owner's mode is
    # sent, and ONE same-route Low retry follows an ACTUAL provider overflow.
    class Overflow(RuntimeError):
        ledger_attempt_ids = ["att-max"]

    monkeypatch.setattr(ul, "_fit_plan", lambda env, memory, task: _Plan("max", False, True))
    real_classify = ul._classify
    monkeypatch.setattr(ul, "_classify", lambda exc: ("context_overflow", "overflow") if isinstance(exc, Overflow) else real_classify(exc))
    sent = []

    def fake_chat(client, *, drive_root, **kwargs):
        sent.append(kwargs["messages"][0]["content"])
        if len(sent) == 1:
            raise Overflow("prompt is too long")
        return {"content": "A Low paragraph."}, {"ledger_attempt_ids": ["att-low"]}

    monkeypatch.setattr(ul, "_chat", fake_chat)
    record = ul.write_letter(_status(), _material(), drive_root=letter_env["drive"])
    assert sent == ["max", "low"], "Max first (even though the plan says it will not fit), Low only after the overflow"
    assert record["state"] == "ready" and record["attempt_ids"] == ["att-max", "att-low"]

    # A second overflow is the typed failure — never a third, strictly-smaller call.
    sent.clear()
    monkeypatch.setattr(ul, "_chat", lambda *a, **k: (_ for _ in ()).throw(Overflow("still too long")))
    record = ul.write_letter(_status(), _material(), drive_root=letter_env["drive"])
    assert record["state"] == "failed" and record["error_kind"] == "context_overflow"

    # Owner Low: nothing smaller to retry with, so the overflow is typed at once.
    monkeypatch.setattr(ul, "_fit_plan", lambda env, memory, task: _Plan("low", True, True))
    calls = []
    monkeypatch.setattr(ul, "_chat", lambda *a, **k: calls.append(1) or (_ for _ in ()).throw(Overflow("too long")))
    record = ul.write_letter(_status(), _material(), drive_root=letter_env["drive"])
    assert record["error_kind"] == "context_overflow" and len(calls) == 1


def test_write_letter_treats_an_http_200_body_error_as_the_provider_s_verdict(letter_env, monkeypatch):
    # OpenRouter serves 429/5xx/context_length_exceeded INSIDE an HTTP 200; llm.py keeps that
    # in usage["provider_error"]. Never "the model returned no text" — and a body overflow
    # earns the same ONE Low retry a raised overflow does.
    monkeypatch.setattr(ul, "_fit_plan", lambda env, memory, task: _Plan("max", False, True))
    overflow = {"code": "context_length_exceeded", "type": "invalid_request_error",
                "kind": "provider_error", "message": "prompt too long"}
    sent = []

    def body_overflow_then_low(client, *, drive_root, **kwargs):
        sent.append(kwargs["messages"][0]["content"])
        if len(sent) == 1:
            return {"content": ""}, {"ledger_attempt_ids": ["att-max"], "provider_error": dict(overflow)}
        return {"content": "A Low paragraph."}, {"ledger_attempt_ids": ["att-low"]}

    monkeypatch.setattr(ul, "_chat", body_overflow_then_low)
    record = ul.write_letter(_status(), _material(), drive_root=letter_env["drive"])
    assert sent == ["max", "low"] and record["state"] == "ready"
    assert record["attempt_ids"] == ["att-max", "att-low"] and record["attempt_id"] == "att-low"

    # Overflow again on the Low call: typed, never a third call.
    sent.clear()

    def body_overflow_always(client, *, drive_root, **kwargs):
        sent.append(kwargs["messages"][0]["content"])
        return {"content": ""}, {"ledger_attempt_ids": [f"att-{len(sent)}"], "provider_error": dict(overflow)}

    monkeypatch.setattr(ul, "_chat", body_overflow_always)
    record = ul.write_letter(_status(), _material(), drive_root=letter_env["drive"])
    assert sent == ["max", "low"] and record["state"] == "failed"
    assert record["error_kind"] == "context_overflow" and record["attempt_ids"] == ["att-1", "att-2"]

    # A raised overflow whose Low retry is served with a body overflow: still one retry.
    class Overflow(RuntimeError):
        ledger_attempt_ids = ["att-raised"]

    real_classify = ul._classify
    monkeypatch.setattr(ul, "_classify", lambda exc: ("context_overflow", "x") if isinstance(exc, Overflow) else real_classify(exc))
    sent.clear()

    def raise_then_body_overflow(client, *, drive_root, **kwargs):
        sent.append(kwargs["messages"][0]["content"])
        if len(sent) == 1:
            raise Overflow("too long")
        return {"content": ""}, {"ledger_attempt_ids": ["att-low"], "provider_error": dict(overflow)}

    monkeypatch.setattr(ul, "_chat", raise_then_body_overflow)
    record = ul.write_letter(_status(), _material(), drive_root=letter_env["drive"])
    assert sent == ["max", "low"] and record["error_kind"] == "context_overflow"
    assert record["attempt_ids"] == ["att-raised", "att-low"]

    # A rate-limit / transient body error is a provider failure carrying its code, no retry.
    sent.clear()
    limited = {"code": 429, "kind": "rate_limit", "message": "rate limited"}
    monkeypatch.setattr(ul, "_chat", lambda client, *, drive_root, **kw: (sent.append(kw["messages"][0]["content"])
                        or ({"content": ""}, {"ledger_attempt_ids": ["att"], "provider_error": dict(limited)})))
    record = ul.write_letter(_status(), _material(), drive_root=letter_env["drive"])
    assert sent == ["max"] and record["state"] == "failed"
    assert record["error_kind"] == "provider_unavailable" and "429" in record["error_text"]
    assert record["attempt_ids"] == ["att"] and record["text"] == ""


@pytest.mark.parametrize("body, kind", [
    # A generic 400 whose MESSAGE is the overflow (the shape OpenRouter forwards for many routes).
    ({"code": 400, "type": "invalid_request_error", "kind": "provider_error",
      "message": "prompt is too long: 250000 tokens > 128000 maximum"}, "context_overflow"),
    # Output/body-size rejections take precedence: shrinking the prompt cannot fix them.
    ({"code": 400, "type": "invalid_request_error", "kind": "provider_error",
      "message": "max_tokens 65536 exceeds maximum context length 32768"}, "provider_unavailable"),
    # A rate limit that happens to mention the context window is still a rate limit —
    # by the transport's structured kind, whichever code shape it came with.
    ({"code": 429, "kind": "rate_limit",
      "message": "rate limit exceeded for this context window tier"}, "provider_unavailable"),
    ({"code": "rate_limit_exceeded", "kind": "provider_transient",
      "message": "rate limit exceeded for the context window tier"}, "provider_unavailable"),
    # A token count containing "429" is not a rate limit: no text guard re-reads it as one.
    ({"code": 400, "type": "invalid_request_error", "kind": "provider_error",
      "message": "prompt is too long: 429000 tokens > 128000 maximum"}, "context_overflow"),
    ({"code": 502, "kind": "provider_transient", "message": "upstream unavailable"}, "provider_unavailable"),
    # A structured transient verdict wins over overflow-shaped wording: an outage, not an overflow.
    ({"code": 503, "kind": "provider_transient",
      "message": "context window shard temporarily unavailable"}, "provider_unavailable"),
    ({"code": "context_length_exceeded", "message": ""}, "context_overflow"),
    (None, ""), ("not a dict", ""),
])
def test_body_error_kind_uses_the_shared_overflow_vocabulary(body, kind):
    assert ul._body_error_kind(body) == kind


def test_write_letter_retries_low_on_a_generic_400_whose_message_is_the_overflow(letter_env, monkeypatch):
    monkeypatch.setattr(ul, "_fit_plan", lambda env, memory, task: _Plan("max", False, True))
    sent = []

    def generic_400_then_low(client, *, drive_root, **kwargs):
        sent.append(kwargs["messages"][0]["content"])
        if len(sent) == 1:
            return {"content": ""}, {"ledger_attempt_ids": ["att-max"], "provider_error": {
                "code": 400, "type": "invalid_request_error", "kind": "provider_error",
                "message": "prompt is too long: 250000 tokens > 128000 maximum"}}
        return {"content": "A Low paragraph."}, {"ledger_attempt_ids": ["att-low"]}

    monkeypatch.setattr(ul, "_chat", generic_400_then_low)
    record = ul.write_letter(_status(), _material(), drive_root=letter_env["drive"])
    assert sent == ["max", "low"] and record["state"] == "ready" and record["attempt_ids"] == ["att-max", "att-low"]


def test_write_letter_keeps_the_attempt_ids_of_a_call_that_raised(letter_env, monkeypatch):
    # usage_accounting attaches the accounted attempt ids to the exception of a failed call;
    # the record keeps them, so a failed letter still points at what it cost.
    class Boom(RuntimeError):
        ledger_attempt_ids = ["att-boom"]

    monkeypatch.setattr(ul, "_chat", lambda *a, **k: (_ for _ in ()).throw(Boom("boom")))
    record = ul.write_letter(_status(), _material(), drive_root=letter_env["drive"])
    assert record["state"] == "failed" and "boom" in record["error_text"]
    assert record["attempt_ids"] == ["att-boom"] and record["attempt_id"] == "att-boom"


def test_write_letter_treats_an_output_budget_cut_as_a_typed_failure(letter_env, monkeypatch):
    # A reply stopped by the output budget is a partial cognitive artifact (BIBLE P1):
    # never stored as ready, always named for what it is.
    shapes = [
        ({"content": "This update brings…", "finish_reason": "length"}, {"ledger_attempt_ids": ["att"]}),
        ({"content": "This update brings…", "stop_reason": "max_tokens"}, {"ledger_attempt_ids": ["att"]}),
        # llm.py keeps the OpenAI-compatible marker in usage["response_finish_reason"]
        ({"content": "This update brings…"}, {"ledger_attempt_ids": ["att"], "response_finish_reason": "length"}),
    ]
    for msg, usage in shapes:
        monkeypatch.setattr(ul, "_chat", lambda *a, _m=msg, _u=usage, **k: (dict(_m), dict(_u)))
        record = ul.write_letter(_status(), _material(), drive_root=letter_env["drive"])
        assert record["state"] == "failed" and record["error_kind"] == "output_truncated", (msg, usage)
        assert str(ul.UPDATE_LETTER_MAX_TOKENS) in record["error_text"] and record["text"] == ""
    monkeypatch.setattr(ul, "_chat", lambda *a, **k: ({"content": "Done.", "finish_reason": "stop"}, {"ledger_attempt_ids": ["att"]}))
    assert ul.write_letter(_status(), _material(), drive_root=letter_env["drive"])["state"] == "ready"


def test_write_letter_local_only_install_inherits_the_local_route(letter_env, monkeypatch):
    # A local-only install leaves the LIGHT slot empty, so it inherits the local Main route.
    # Asking the remote credential gate about it would refuse a letter it can write.
    monkeypatch.delenv("USE_LOCAL_LIGHT", raising=False)
    monkeypatch.setattr("ouroboros.provider_models.review_model_uses_local", lambda model: True)
    monkeypatch.setattr(
        "ouroboros.provider_models.model_has_credentials",
        lambda model: (_ for _ in ()).throw(AssertionError("a local route asks no remote gate")),
    )
    seen = {}

    def fake_chat(client, *, drive_root, **kwargs):
        seen.update(kwargs)
        return {"content": "A local paragraph."}, {"ledger_attempt_ids": ["att"]}

    monkeypatch.setattr(ul, "_chat", fake_chat)
    record = ul.write_letter(_status(), _material(), drive_root=letter_env["drive"])
    assert record["state"] == "ready" and seen["use_local"] is True


def test_write_letter_local_light_route_skips_the_credential_gate(letter_env, monkeypatch):
    monkeypatch.setenv("USE_LOCAL_LIGHT", "true")
    monkeypatch.setattr("ouroboros.provider_models.model_has_credentials",
                        lambda model: (_ for _ in ()).throw(AssertionError("local route must not ask")))
    seen = {}

    def fake_chat(client, *, drive_root, **kwargs):
        seen.update(kwargs)
        return {"content": "local paragraph"}, {}

    monkeypatch.setattr(ul, "_chat", fake_chat)
    record = ul.write_letter(_status(), _material(), drive_root=letter_env["drive"])
    assert record["state"] == "ready" and seen["use_local"] is True
    assert letter_env["calls"][0][1]["use_local_model"] is True


def test_write_letter_context_overflow_is_typed(letter_env, monkeypatch):
    from ouroboros.llm import LocalContextTooLargeError

    def fake_chat(client, *, drive_root, **kwargs):
        raise LocalContextTooLargeError("too big")

    monkeypatch.setattr(ul, "_chat", fake_chat)
    record = ul.write_letter(_status(), _material(), drive_root=letter_env["drive"])
    assert record["state"] == "failed" and record["error_kind"] == "context_overflow"


def test_boot_check_writes_the_letter_before_the_readiness_broadcast(monkeypatch):
    import server
    import supervisor.git_ops as git_ops
    import supervisor.update_merge as update_merge

    calls = []
    monkeypatch.setattr(server, "_wait_for_supervisor_update_finalize", lambda: False)
    monkeypatch.setattr(update_merge, "finalize_managed_update_on_boot",
                        lambda supervisor_ready: {"finalized": False, "rolled_back": False})
    monkeypatch.setattr(git_ops, "compute_managed_update_status", lambda fetch: _status())
    monkeypatch.setattr(ul, "refresh_after_check", lambda status, **k: calls.append(("letter", status["latest_sha"])))
    monkeypatch.setattr(server, "broadcast_ws_sync", lambda payload: calls.append((payload["type"], "")))

    server._boot_managed_update_tasks()

    assert calls == [("letter", "b" * 40), ("update_status_ready", "")]
    # And at boot the local model server is started BEFORE the check that may need it.
    source = pathlib.Path(server.__file__).read_text(encoding="utf-8")
    assert source.index('name="local-model-autostart"') < source.index('name="boot-managed-update"')
