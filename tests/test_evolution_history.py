"""Dashboard → Evolution's history owner (``ouroboros/evolution_history.py``) over a real throwaway Git repository."""
from __future__ import annotations

import json
import os
import subprocess
import threading

import pytest

from ouroboros import evolution_history as eh

README_HEAD = "# Demo\n\n## Version History\n\n| Version | Date | Description |\n|---|---|---|\n"


def _git(repo, *args, env=None):
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True,
                          env={**os.environ, **(env or {})}).stdout.strip()


def _commit(repo, message, when, files):
    for name, text in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    _git(repo, "add", "-A")
    stamp = {"GIT_AUTHOR_DATE": when, "GIT_COMMITTER_DATE": when}
    _git(repo, "commit", "-q", "-m", message, env=stamp)
    return _git(repo, "rev-parse", "HEAD"), stamp


@pytest.fixture()
def repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "T")
    first, stamp = _commit(repo, "one", "2026-02-16T10:00:00+00:00", {
        "BIBLE.md": "b" * 2048, "prompts/SYSTEM.md": "s" * 1000, "a.py": "one\ntwo\n", "tests/t.py": "x",
        "README.md": README_HEAD})
    _git(repo, "tag", "-a", "v1.0.0", "-m", "v1.0.0: First light", env=stamp)
    _git(repo, "tag", "-a", "remote-v1.0.0", "-m", "Ouroboros v1.0.0", env=stamp)   # an alias of the same commit
    second, stamp = _commit(repo, "two", "2026-06-01T10:00:00+00:00", {
        "b.py": "three\n", "README.md": README_HEAD
        + "| 1.1.0 | 2026-06-01 | **Memory as one ladder** — **feat: memory keeps pages.** Pages \\| notes stay. |\n"})
    _git(repo, "tag", "-a", "v1.1.0", "-m", "Ouroboros v1.1.0", env=stamp)
    third, stamp = _commit(repo, "three", "2026-06-02T10:00:00+00:00", {
        "a.py": "one\n", "README.md": README_HEAD + "| 1.1.1 | 2026-06-02 | **fix: a fix.** Small. |\n"})
    _git(repo, "tag", "v1.1.1", env=stamp)                                            # lightweight: no annotation
    _git(repo, "tag", "-a", "remote-v1.1.0", "-m", "elsewhere", third, env=stamp)     # same name, different commit
    return repo


def test_versions_fold_aliases_only_when_they_name_the_same_commit(repo):
    digest, versions = eh.list_versions(str(repo))
    assert len(digest) == 64
    assert [(v["tag"], v["aliases"]) for v in versions] == [
        ("v1.0.0", ["remote-v1.0.0"]), ("v1.1.0", []), ("remote-v1.1.0", []), ("v1.1.1", [])]
    assert [v["annotation"] for v in versions] == ["v1.0.0: First light", "Ouroboros v1.1.0", "elsewhere", ""]
    assert versions[2]["commit"] == versions[3]["commit"] != versions[1]["commit"]
    _git(repo, "tag", "-a", "v1.2.0", "-m", "new")
    assert eh.list_versions(str(repo))[0] != digest, "a new tag changes the identity digest"


def test_refresh_measures_each_commit_once_from_its_own_objects(repo, tmp_path, monkeypatch):
    data = tmp_path / "data"
    digest, versions = eh.list_versions(str(repo))
    snapshot = eh.refresh(str(repo), str(data), digest, versions, None)
    owner = eh.EvolutionHistory()
    out = owner.projection(str(repo), str(data))
    assert out["refresh"]["state"] == "idle" and out["saved_at"] == snapshot["saved_at"]
    rows = {v["tag"]: v for v in out["versions"]}
    # Lines are newline-terminated lines of every tracked .py blob, an unterminated last line included.
    assert [rows[t]["code_lines"] for t in ("v1.0.0", "v1.1.0", "v1.1.1")] == [3, 4, 3]
    assert (rows["v1.0.0"]["bible_bytes"], rows["v1.0.0"]["system_bytes"]) == (2048, 1000)
    assert rows["v1.1.0"]["release"] == {"source": "readme", "title": "Memory as one ladder", "kind": "feat",
                                         "heading": "memory keeps pages.", "body": "Pages | notes stay."}
    assert rows["v1.1.0"]["milestone"] is True and rows["v1.1.0"]["level"] == "minor"
    assert rows["v1.0.0"]["release"] == {"source": "tag", "title": "", "kind": "", "heading": "First light", "body": ""}
    assert rows["v1.0.0"]["milestone"] is True, "a MAJOR release is a milestone by P9"
    assert rows["v1.1.1"]["release"]["kind"] == "fix" and rows["v1.1.1"]["milestone"] is False
    assert rows["remote-v1.1.0"]["milestone"] is True, "a MINOR release is a milestone too (owner 21.10, A)"
    # The earlier `points` shape stays for older clients: one per tag name, honest numbers, null memory fields.
    assert out["cached"] is True
    assert [point["tag"] for point in out["points"]] == ["v1.0.0", "remote-v1.0.0", "v1.1.0", "remote-v1.1.0", "v1.1.1"]
    assert out["points"][0] == {"tag": "v1.0.0", "date": rows["v1.0.0"]["date"], "code_lines": 3, "bible_kb": 2.0,
                                "system_kb": 0.98, "identity_kb": None, "scratchpad_kb": None, "memory_kb": None}
    # A version whose README at its own commit has no row for it falls back to its tag annotation.
    assert rows["remote-v1.1.0"]["release"] == {"source": "tag", "title": "", "kind": "", "heading": "elsewhere", "body": ""}
    calls = []
    real = eh._measure
    monkeypatch.setattr(eh, "_measure", lambda *a: calls.append(a[1]) or real(*a))
    again = eh.refresh(str(repo), str(data), digest, versions, eh.read_snapshot(str(data)))
    assert calls == [] and again["commits"] == snapshot["commits"], "a measured commit is never read again"
    newest = versions[-1]["commit"]
    python = {line.split()[2] for line in _git(repo, "ls-tree", "-r", newest).splitlines() if line.endswith(".py")}
    assert set(snapshot["blob_lines"]) == python, "only the newest tree's counts are kept for the next refresh"


def test_an_older_or_broken_saved_copy_is_not_read_as_history(tmp_path):
    path = tmp_path / eh.SNAPSHOT_REL
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"schema": 1, "points": {"v1": {"tag": "v1", "code_lines": 9}}}), encoding="utf-8")
    assert eh.read_snapshot(str(tmp_path)) is None
    path.write_text("[]", encoding="utf-8")
    assert eh.read_snapshot(str(tmp_path)) is None


def test_the_installation_series_keep_their_own_units_and_gaps(repo, tmp_path):
    data = tmp_path / "data"
    memory = data / "memory"
    memory.mkdir(parents=True)
    (memory / "identity_journal.jsonl").write_text("\n".join(json.dumps(row) for row in [
        {"ts": "2026-05-01T00:00:00+00:00", "new_len": 120, "new_content": "x" * 120},
        "not json",
        {"ts": "2026-06-01T12:00:00+00:00", "new_len": 150},
        {"ts": "2026-07-01T00:00:00+00:00", "old_len": 150},
    ]) + "\nnot json\n", encoding="utf-8")
    (memory / "scratchpad_journal.jsonl").write_text(json.dumps(
        {"ts": "2026-06-02T11:00:00+00:00", "type": "block_appended", "content_len": 40}) + "\n", encoding="utf-8")
    # The files themselves differ from every journal figure: they are never read for this.
    (memory / "identity.md").write_text("i" * 999, encoding="utf-8")
    (memory / "scratchpad.md").write_text("## header\n" + "s" * 999, encoding="utf-8")
    digest, versions = eh.list_versions(str(repo))
    eh.refresh(str(repo), str(data), digest, versions, None)
    install = eh.EvolutionHistory().projection(str(repo), str(data))["install"]
    # v1.1.0 was tagged at 10:00, before that day's 12:00 write: each version holds the last record before it.
    assert install["identity"]["at_versions"] == [None, 120, 150, 150]
    assert install["identity"]["latest"] == {"at": "2026-06-01T12:00:00+00:00", "chars": 150}
    assert install["identity"]["first_at"] == "2026-05-01T00:00:00+00:00"
    assert install["scratchpad"]["at_versions"] == [None, None, None, None], "before its first append it is unknown"
    assert install["scratchpad"]["latest"]["chars"] == 40


def test_the_saved_copy_answers_at_once_while_a_refresh_runs_and_a_failure_keeps_it(repo, tmp_path, monkeypatch):
    data = tmp_path / "data"
    owner = eh.EvolutionHistory()
    gate = threading.Event()
    real = eh.refresh

    def slow(*args):
        assert gate.wait(10), "the test released the refresh"
        return real(*args)

    monkeypatch.setattr(eh, "refresh", slow)
    first = owner.projection(str(repo), str(data))
    assert first["refresh"]["state"] == "running" and first["versions"] == [] and first["saved_at"] == ""
    assert first["points"] == [] and first["cached"] is False, "no saved copy yet: nothing cached"
    assert owner.projection(str(repo), str(data))["refresh"]["state"] == "running", "one refresh at a time"
    gate.set()
    owner.join(10)
    saved = owner.projection(str(repo), str(data))
    assert saved["refresh"]["state"] == "idle" and len(saved["versions"]) == 4

    _git(repo, "tag", "-a", "v2.0.0", "-m", "Ouroboros v2.0.0")
    monkeypatch.setattr(eh, "refresh", lambda *a: (_ for _ in ()).throw(RuntimeError("disk full")))
    assert owner.projection(str(repo), str(data))["refresh"]["state"] == "running"
    owner.join(10)
    failed = owner.projection(str(repo), str(data))
    assert failed["refresh"]["state"] == "failed" and "disk full" in failed["refresh"]["error"]
    assert failed["versions"] == saved["versions"], "the previous copy stays on screen"
    owner.join(10)
    assert owner.projection(str(repo), str(data))["refresh"]["state"] == "failed", "the same tags are not retried by themselves"
    monkeypatch.setattr(eh, "refresh", real)
    assert owner.projection(str(repo), str(data), force=True)["refresh"]["state"] == "running"
    owner.join(10)
    assert len(owner.projection(str(repo), str(data))["versions"]) == 5


def test_a_repository_git_cannot_list_is_a_failure_not_an_empty_history(tmp_path):
    out = eh.EvolutionHistory().projection(str(tmp_path / "missing"), str(tmp_path / "data"))
    assert out["refresh"]["state"] == "failed" and out["refresh"]["error"]
    assert out["versions"] == [] and out["saved_at"] == ""


def test_release_rows_keep_their_words_and_never_read_a_legacy_heading_as_a_title():
    words = lambda text: eh.release_words({"version": "1.0.0", "row": {"text": text}})  # noqa: E731
    assert words("**fix: everything merged since 7.7.0.** Desktop shell.")["title"] == ""
    assert words("**feat: a** — **b**")["title"] == "", "a bold `type:` opener is a heading, not a title"
    assert words("**v7: a modular runtime.** Body")["kind"] == "v7"
    assert words("plain text without bold") == {"source": "readme", "title": "", "kind": "", "heading": "",
                                                "body": "plain text without bold"}
    assert eh.release_row(README_HEAD + "| 1.0.0 | d | **x** |\n", "1.0.0") == {"date": "d", "text": "**x**"}
    assert eh.release_row(README_HEAD + "| 1.0.0 | d | **x** |\n", "1.0.1") is None
    assert eh.release_row("# no history\n", "1.0.0") is None
    assert [eh.version_level(v) for v in ("7.0.0", "7.6.0", "7.7.1", "6.9.0-rc.1")] == ["major", "minor", "patch", "pre"]


def test_a_missing_blob_leaves_its_version_unmeasured_and_a_forced_refresh_recovers_it(repo, tmp_path, monkeypatch):
    data = tmp_path / "data"
    digest, versions = eh.list_versions(str(repo))
    newest = versions[-1]["commit"]
    real = eh._cat_file
    # a.py's blob at the newest commit ("one\n") exists nowhere earlier; the object store "loses" it.
    lost = {line.split()[2] for line in _git(repo, "ls-tree", "-r", newest).splitlines() if line.endswith("\ta.py")}

    def lose_newest_python_blob(repo_dir, names):
        found = real(repo_dir, names)
        for name in lost & set(found):
            found[name] = None
        return found

    monkeypatch.setattr(eh, "_cat_file", lose_newest_python_blob)
    snapshot = eh.refresh(str(repo), str(data), digest, versions, None)
    assert snapshot["unmeasured"] == 2, "both versions on the newest commit (remote-v1.1.0, v1.1.1)"
    assert "missing from the object store" in snapshot["unmeasured_error"]
    assert newest not in snapshot["commits"], "an unreadable blob is never counted as an empty file"
    owner = eh.EvolutionHistory()
    out = owner.projection(str(repo), str(data))
    rows = {v["tag"]: v for v in out["versions"]}
    assert rows["v1.1.1"]["code_lines"] is None and rows["v1.0.0"]["code_lines"] == 3, "the measured history stays"
    assert out["refresh"]["unmeasured"] == 2 and out["refresh"]["unmeasured_error"].startswith("remote-v1.1.0: ")
    assert out["refresh"]["state"] == "idle", "same tags: no automatic retry"
    monkeypatch.setattr(eh, "_cat_file", real)
    assert owner.projection(str(repo), str(data), force=True)["refresh"]["state"] == "running"
    owner.join(10)
    healed = owner.projection(str(repo), str(data))
    assert healed["refresh"]["unmeasured"] == 0 and healed["refresh"]["unmeasured_error"] == ""
    assert {v["tag"]: v["code_lines"] for v in healed["versions"]}["v1.1.1"] == 3


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_an_appended_journal_row_is_read_without_reparsing_the_journal(repo, tmp_path, monkeypatch, newline):
    data = tmp_path / "data"
    journal = data / "memory" / "scratchpad_journal.jsonl"
    journal.parent.mkdir(parents=True)

    def row(ts, n):
        return (json.dumps({"ts": ts, "type": "block_appended", "content_len": n}) + newline).encode("utf-8")

    journal.write_bytes(row("2026-06-01T00:00:00+00:00", 10))
    digest, versions = eh.list_versions(str(repo))
    eh.refresh(str(repo), str(data), digest, versions, None)
    owner = eh.EvolutionHistory()
    assert owner.projection(str(repo), str(data))["install"]["scratchpad"]["latest"]["chars"] == 10
    reads = []
    real = eh._journal_series
    monkeypatch.setattr(eh, "_journal_series", lambda path, field, start=0, known=None: reads.append(start) or real(path, field, start, known))
    with journal.open("ab") as handle:
        handle.write(row("2026-06-03T00:00:00+00:00", 30))
        handle.write(row("2026-06-02T00:00:00+00:00", 20)[:-12])  # a row still being written
    install = owner.projection(str(repo), str(data))["install"]["scratchpad"]
    assert install["latest"] == {"at": "2026-06-03T00:00:00+00:00", "chars": 30}
    assert reads == [len(row("2026-06-01T00:00:00+00:00", 10))], "only the appended bytes were read"
    with journal.open("ab") as handle:
        handle.write(row("2026-06-02T00:00:00+00:00", 20)[-12:])  # its newline arrives
    series = owner.projection(str(repo), str(data))["install"]["scratchpad"]
    assert series["latest"]["chars"] == 30, "an out-of-order row joins in time order"
    # Versions dated 1 June 10:00 and 2 June 10:00 hold 10, 20, 20: the late 2 June row joined in its time order.
    assert series["at_versions"][1:] == [10, 20, 20], "the rows stand at the dates they were recorded"
    journal.write_bytes(row("2026-06-05T00:00:00+00:00", 7))  # replaced and shorter: read whole
    assert owner.projection(str(repo), str(data))["install"]["scratchpad"]["latest"]["chars"] == 7


# The shape of v6.2.0's annotation (no README row at its commit), with what a framing must survive: pipes, a
# backslash, a format placeholder, markup, a line that reads like the next tag's record, blank lines, indentation.
LEGACY = ("v1.2.0: Critical Bugfixes + LLM-First Dedup\n\nCritical fixes:\n- Fix worker_id==0 hard-timeout bug | a \\| pipe\n"
          "- %(refname) and <b>markup</b> stay text\n\nv1.3.0\ttag\n  indented — Ünïcode, кириллица\n")


def test_a_version_without_a_readme_row_speaks_with_its_whole_tag_annotation(repo, tmp_path):
    message = tmp_path / "message.txt"
    message.write_text(LEGACY, encoding="utf-8")
    fourth, stamp = _commit(repo, "four", "2026-06-03T10:00:00+00:00", {"c.py": "four\n"})
    _git(repo, "tag", "-a", "v1.2.0", "-F", str(message), env=stamp)
    _git(repo, "tag", "-a", "v1.3.0", "-m", "Ouroboros v1.3.0\n\nOnly a body.", env=stamp)
    _git(repo, "tag", "v1.4.0", env=stamp)                                          # lightweight, on the same commit
    _digest, versions = eh.list_versions(str(repo))
    annotations = {v["tag"]: v["annotation"] for v in versions}
    assert annotations["v1.2.0"] == LEGACY.rstrip("\n"), "the whole message as git keeps it, each record in its own frame"
    assert annotations["v1.3.0"] == "Ouroboros v1.3.0\n\nOnly a body."
    assert annotations["v1.4.0"] == "" and annotations["v1.0.0"] == "v1.0.0: First light"
    data = tmp_path / "data"
    digest, versions = eh.list_versions(str(repo))
    eh.refresh(str(repo), str(data), digest, versions, None)
    rows = {v["tag"]: v for v in eh.EvolutionHistory().projection(str(repo), str(data))["versions"]}
    assert rows["v1.2.0"]["release"] == {
        "source": "tag", "title": "", "kind": "", "heading": "Critical Bugfixes + LLM-First Dedup",
        "body": LEGACY.split("\n\n", 1)[1].rstrip("\n")}, "the body reads as written, its lines and characters kept"
    assert rows["v1.3.0"]["release"] == {"source": "tag", "title": "", "kind": "", "heading": "", "body": "Only a body."}
    assert rows["v1.4.0"]["release"]["source"] == "" and rows["v1.2.0"]["commit"] == fourth
    # A copy saved before bodies were read (its annotation the subject alone) still reads; a wrapped subject is one line.
    assert eh.release_words({"version": "1.2.0", "annotation": "v1.2.0: Critical Bugfixes"})["heading"] == "Critical Bugfixes"
    assert eh.release_words({"version": "1.2.0", "annotation": "a wrapped\nsubject\n\nbody"})["heading"] == "a wrapped subject"


# Exact Version History row from the public v6.18.0 README; no Git history is needed at test time.
PIPE_ROW = (
    "| 6.18.0 | 2026-06-05 | **feat(frontend-browser): mobile-grade browser checks and UI polish.** "
    "Browser tools now support explicit `engine=chromium|webkit` and Playwright device descriptors, "
    "with packaged Chromium/WebKit install paths and CI smoke coverage for both engines. "
    "The chat composer uses a responsive glass toolbar: desktop controls stay inside the frosted field, "
    "while mobile lifts Consilium and Low/Max above the textarea and keeps Send inside. "
    "Chat-wide drag/drop stages attachments until Send; nested subagent child cards stay visible but "
    "collapsed by default with role-first headings; widget cards can be reordered via a persisted "
    "owner-local UI preference endpoint. |"
)


def test_release_description_keeps_literal_pipes_through_refresh_and_projection(repo, tmp_path):
    commit, stamp = _commit(repo, "historical pipe row", "2026-06-05T10:00:00+00:00",
                            {"README.md": README_HEAD + PIPE_ROW + "\n"})
    _git(repo, "tag", "-a", "v6.18.0", "-m", "Ouroboros v6.18.0", env=stamp)
    digest, versions = eh.list_versions(str(repo))
    data = tmp_path / "pipe-data"
    eh.refresh(str(repo), str(data), digest, versions, None)
    row = next(v for v in eh.EvolutionHistory().projection(str(repo), str(data))["versions"]
               if v["commit"] == commit)
    assert row["release"]["source"] == "readme"
    assert row["release"]["body"] == PIPE_ROW.split("** ", 1)[1].removesuffix(" |")
    assert eh.release_row(README_HEAD + "| 1.0.0 | d | a | b \\| c |\n", "1.0.0")["text"] == "a | b | c"


def test_schema_two_cached_missing_row_is_rebuilt_without_new_tags(repo, tmp_path):
    data = tmp_path / "old-cache"
    digest, versions = eh.list_versions(str(repo))
    snapshot = eh.refresh(str(repo), str(data), digest, versions, None)
    snapshot["schema"] = 2
    for row in snapshot["versions"]:
        row["row"] = None
    (data / eh.SNAPSHOT_REL).write_text(json.dumps(snapshot), encoding="utf-8")
    owner = eh.EvolutionHistory()
    assert owner.projection(str(repo), str(data))["refresh"]["state"] == "running"
    owner.join(10)
    rows = owner.projection(str(repo), str(data))["versions"]
    assert next(row for row in rows if row["tag"] == "v1.1.0")["release"]["source"] == "readme"


def _checkpoint(n, kind=""):
    row = {"schema_version": 1, "ts": f"2026-06-01T00:00:{n % 60:02d}+00:00", "task_id": f"evo-{n}",
           "result_status": "failed", "loop_outcome": {"result_status": "failed"}}
    return json.dumps({**row, "kind": kind} if kind else row) + "\n"


def test_the_checkpoints_follow_their_journal_and_never_remember_an_unreadable_one(tmp_path, monkeypatch):
    from ouroboros.evolution_checkpoints import CHECKPOINTS_REL

    data = tmp_path / "data"
    path = data / CHECKPOINTS_REL
    path.parent.mkdir(parents=True)
    owner = eh.EvolutionHistory()
    assert owner.checkpoints(str(data)) == [], "no journal: nothing recorded"
    path.write_text("".join(_checkpoint(n, "cycle_outcome" if n % 4 == 3 else "") for n in range(200)), encoding="utf-8")
    reads = []
    real = eh._appended_rows
    monkeypatch.setattr(eh, "_appended_rows", lambda p, start: reads.append(start) or real(p, start))
    shown = owner.checkpoints(str(data))
    assert [row["task_id"] for row in shown] == [f"evo-{n}" for n in range(200) if n % 4 != 3][-100:]
    assert "result_status" not in shown[0] and "result_status" not in shown[0]["loop_outcome"], "the public form"
    assert owner.checkpoints(str(data)) == shown and reads == [0], "an unchanged journal is not read again"
    size = path.stat().st_size
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_checkpoint(200) + _checkpoint(201, "cycle_outcome") + _checkpoint(202)[:-9])  # the last is half-written
    grown = owner.checkpoints(str(data))
    assert reads[-1] == size, "only the appended bytes are read"
    assert [row["task_id"] for row in grown][-2:] == ["evo-198", "evo-200"] and len(grown) == 100
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_checkpoint(202)[-9:])
    assert owner.checkpoints(str(data))[-1]["task_id"] == "evo-202", "a row read once its newline lands"
    path.write_text(_checkpoint(7), encoding="utf-8")                                # replaced and shorter: read whole
    assert [row["task_id"] for row in owner.checkpoints(str(data))] == ["evo-7"] and reads[-1] == 0
    flaky = {"fail": True}

    def unreadable_once(p, start):
        if flaky.pop("fail", False):
            raise PermissionError("denied")
        return real(p, start)

    monkeypatch.setattr(eh, "_appended_rows", unreadable_once)
    previous = path.stat()
    path.write_text(_checkpoint(8), encoding="utf-8")
    # Same size, same inode: a coarse clock (Windows) can give the rewrite the old mtime too. The
    # changed time is the fact this path needs, set exactly rather than waited for.
    os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns + 1_000_000_000))
    assert owner.checkpoints(str(data)) == [], "unreadable: nothing known this time"
    assert [row["task_id"] for row in owner.checkpoints(str(data))] == ["evo-8"], \
        "the same unchanged journal is read again, never remembered as empty"
