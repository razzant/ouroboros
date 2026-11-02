"""Release diagnostics read a chosen Git source and never grant review authority."""
from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from ouroboros import commit_admission as admission
from ouroboros.tools import preflight_review, review
from ouroboros.tools.release_sync import release_metadata_findings

pytestmark = pytest.mark.serial  # real Git processes and isolated runtime globals


def _git(repo, *args):
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def _release(version="1.2.3"):
    return {
        "VERSION": version + "\n",
        "README.md": f"[![Version {version}](https://img.shields.io/badge/version-{version}-green.svg)]\n"
                     f"## Version History\n| {version} | today | release |\n",
        "pyproject.toml": f'[project]\nversion = "{version}"\n',
        "docs/ARCHITECTURE.md": f"# Ouroboros v{version}\n",
        "web/package.json": json.dumps({"version": version}),
        "uv.lock": f'[[package]]\nname = "ouroboros"\nversion = "{version}"\nsource = {{ editable = "." }}\n',
        "web/modules/api_types.js": f"GATEWAY_CONTRACT_VERSION = '{version}'\n",
    }


def _write(repo, files):
    for name, value in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8", newline="")


@pytest.fixture
def candidate(tmp_path, monkeypatch):
    from ouroboros import config
    from supervisor import git_ops, queue, state, workers

    repo, drive = tmp_path / "repo", tmp_path / "data"
    repo.mkdir()
    drive.mkdir()
    # Environment is scrubbed before imports by the test invocation; rebind all
    # imported mutable roots as well. Never initialize or contact a live launcher.
    for key, value in {"OUROBOROS_APP_ROOT": tmp_path, "OUROBOROS_REPO_DIR": repo,
                       "OUROBOROS_DATA_DIR": drive, "OUROBOROS_SETTINGS_PATH": drive / "settings.json",
                       "OUROBOROS_SUBAGENT_PROJECTS_ROOT": tmp_path / "projects"}.items():
        monkeypatch.setenv(key, str(value))
    for module, attrs in ((config, {"APP_ROOT": tmp_path, "REPO_DIR": repo, "DATA_DIR": drive,
                                   "SETTINGS_PATH": drive / "settings.json"}),
                          (git_ops, {"REPO_DIR": repo, "DRIVE_ROOT": drive}),
                          (workers, {"REPO_DIR": repo, "DRIVE_ROOT": drive, "DATA_DIR": drive}),
                          (state, {"DRIVE_ROOT": drive, "DATA_DIR": drive,
                                   "STATE_PATH": drive / "state/state.json",
                                   "STATE_LAST_GOOD_PATH": drive / "state/state.last_good.json",
                                   "STATE_LOCK_PATH": drive / "locks/state.lock"}),
                          (queue, {"DRIVE_ROOT": drive, "DATA_DIR": drive,
                                   "QUEUE_SNAPSHOT_PATH": drive / "state/queue_snapshot.json"})):
        for name, value in attrs.items():
            monkeypatch.setattr(module, name, value, raising=False)
    _git(repo, "init")
    _git(repo, "config", "core.autocrlf", "false")
    _git(repo, "config", "user.name", "Fixture")
    _git(repo, "config", "user.email", "fixture@example.invalid")
    _write(repo, {**_release(), "change.py": "value = 1\n"})
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "fixture")
    return SimpleNamespace(repo_dir=repo, drive_root=drive, task_id="diagnostic-test",
                           emit_progress_fn=lambda *_: None)


def test_crlf_carriers_have_same_text_semantics_in_index_and_worktree(candidate):
    repo = candidate.repo_dir
    _git(repo, "config", "core.autocrlf", "false")
    for name, text in _release("1.2.4").items():
        (repo / name).write_bytes(text.replace("\n", "\r\n").encode("utf-8"))
    _git(repo, "add", ".")
    for source in ("worktree", "index"):
        result = admission.release_metadata_diagnostics(repo, ["VERSION"], source=source)
        assert result["status"] == "clean", result


def test_sm1_committed_crlf_export_preserves_carrier_lines(candidate):
    from devtools.e2e_live.scenarios import _git_show, release_carriers_desync_at
    repo = candidate.repo_dir
    _git(repo, "config", "core.autocrlf", "false")
    for name, text in _release("1.2.4").items():
        (repo / name).write_bytes(text.replace("\n", "\r\n").encode("utf-8"))
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "CRLF carriers")
    assert _git_show(repo, "HEAD", "uv.lock") == _release("1.2.4")["uv.lock"]
    assert release_carriers_desync_at(repo, "HEAD") == ""


@pytest.mark.parametrize("reader", ["index", "preflight"])
@pytest.mark.parametrize("delete_readme", [False, True])
def test_numbered_release_needs_a_surviving_staged_readme(candidate, reader, delete_readme):
    repo = candidate.repo_dir
    _write(repo, _release("1.2.4"))
    if delete_readme:
        (repo / "README.md").unlink()
    _git(repo, "add", "-A")
    if reader == "index":
        report = admission.release_metadata_diagnostics(repo, source="index")
        assert report["form"] == "numbered"
        message = admission.format_release_metadata_preflight(report)
    else:
        # The real preflight classifies active/deleted paths and supplies its index reader.
        message = review._preflight_check("release", review._build_preflight_staged(str(repo)), repo)
    if delete_readme:
        assert "Missing from staged: README.md" in message
        assert "PREFLIGHT_UNAVAILABLE" in message and "index:README.md" in message
    else:
        assert not message


def test_unicode_worktree_discovery_is_not_locale_decoded(candidate):
    repo = candidate.repo_dir
    _git(repo, "config", "core.quotepath", "false")
    (repo / "А.py").write_text("value = 1\n", encoding="utf-8")
    assert "А.py" in admission.changed_worktree_paths(repo, strict=True)


def _broken(repo):
    files = _release("1.2.4")
    files["README.md"] = _release()["README.md"] + "".join(
        f"| 1.1.{patch} | today | old patch |\n" for patch in range(1, 7))
    files["pyproject.toml"] = 'version = "0.0.0"\n'
    _write(repo, files)


def _snapshot(root):
    """Bytes and mtimes catch index refreshes and unchanged-content rewrites too."""
    return {p.relative_to(root).as_posix(): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in root.rglob("*") if p.is_file()}


def _diagnose(ctx, source, **kwargs):
    return json.loads(preflight_review._handle_preflight_review(
        ctx, commit_message="release", deterministic_only=True, source=source, **kwargs))


@pytest.mark.parametrize("source", ["worktree", "index"])
@pytest.mark.parametrize("outcome", ["clean", "blocked", "unavailable"])
def test_diagnostic_has_no_effects_even_with_pending_managed_review(candidate, monkeypatch, source, outcome):
    ctx = candidate
    # Exercise the real book helper too, including remote and merge-base reads.
    _write(ctx.repo_dir, {"ouroboros/reference_books.py": "# book contract\n"})
    _git(ctx.repo_dir, "remote", "add", "canonical", "https://github.com/razzant/ouroboros.git")
    _git(ctx.repo_dir, "update-ref", "refs/remotes/canonical/ouroboros", "HEAD")
    if outcome == "blocked":
        _broken(ctx.repo_dir)
    else:
        _write(ctx.repo_dir, _release("1.2.4"))
    if outcome == "unavailable":
        (ctx.repo_dir / "pyproject.toml").write_bytes(b"\xff")
    _git(ctx.repo_dir, "add", ".")
    # Actual durable bytes are present: a dry run must neither read nor rewrite
    # pending custody, obligations or review freshness, even for a managed caller.
    _write(ctx.drive_root, {"state/advisory_review.json": '{"pending":true,"obligations":["keep"]}',
                            "logs/events.jsonl": '{"type":"keep"}\n',
                            "state/delegate_custody.jsonl": '{"invocation":"pending"}\n'})
    ctx.task_metadata = {"managed_update": {"transaction_id": "pending"}}
    ctx._preflight_test_proof = object()
    before, context_before = _snapshot(ctx.repo_dir.parent), dict(vars(ctx))

    def forbidden(*args, **kwargs):
        pytest.fail("diagnostics crossed into review, preparation or test work")

    for target in ("ouroboros.tools.review_change._handle_review_change",
                   "ouroboros.tools.review_change.run_review_change",
                   "ouroboros.tools.commit_gate.preflight_reviewer_error",
                   "ouroboros.tools.commit_gate.run_commit_preflight",
                   "ouroboros.tools.commit_gate.deterministic_preflight",
                   "ouroboros.tools.review_helpers.check_worktree_readiness",
                   "ouroboros.tools.review_helpers._run_review_preflight_tests",
                   "ouroboros.review_state.compute_snapshot_hash", "ouroboros.review_state.load_state",
                   "ouroboros.review_state.update_state"):
        monkeypatch.setattr(target, forbidden)
    monkeypatch.setattr("ouroboros.tools.release_sync.sync_release_metadata", forbidden)
    monkeypatch.setattr("ouroboros.provider_models.model_has_credentials", forbidden)
    monkeypatch.setattr("ouroboros.tools.review._fingerprint_staged_diff", forbidden, raising=False)
    monkeypatch.setattr("ouroboros.tools.registry._authorized_managed_update_resolver", forbidden)
    real_run = admission.subprocess.run

    def only_reads(argv, *args, **kwargs):
        assert argv[0] == "git" and not set(argv[1:]) & {
            "add", "write-tree", "commit", "reset", "update-index", "update-ref", "fetch", "push", "checkout"}
        return real_run(argv, *args, **kwargs)

    monkeypatch.setattr(admission.subprocess, "run", only_reads)
    # A named row changes nothing: the diagnostics never validate, seat or pay it.
    result = _diagnose(ctx, source, reviewer="api-scout", goal="release", scope="carriers")
    assert result["status"] == outcome, result
    assert result["source"] == source and result["review_freshness"] is False
    assert result["book_balance"]["source"] == "worktree"
    assert "refs/remotes/canonical/ouroboros" in result["book_balance"]["note"]
    assert "snapshot_hash" not in result and "review_reference" not in result
    assert _snapshot(ctx.repo_dir.parent) == before
    assert vars(ctx) == context_before


@pytest.mark.parametrize("source", ["worktree", "index"])
@pytest.mark.parametrize("geometry", ["published", "candidate"])
def test_free_tool_json_reports_cumulative_books_with_an_explicit_byte_source(candidate, source, geometry):
    """A free call sees committed contribution growth even on a clean candidate;
    staged and worktree book bytes subsequently diverge without mislabelling."""
    ctx = candidate
    repo = ctx.repo_dir
    chapter = "docs/architecture/01-one.md"
    body = "# One\n\nAn authored introduction.\n"
    _write(repo, {"ouroboros/reference_books.py": "# book contract\n", chapter: body,
                  "docs/ARCHITECTURE.md": "# Architecture\n\nThe map.\n\n## Chapters\n\n"
                                          "- [One](architecture/01-one.md)\n"})
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "book base")
    base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "remote", "add", "official-renamed", "git@github.com:razzant/ouroboros.git")
    _git(repo, "update-ref", "refs/remotes/official-renamed/ouroboros", base)
    _write(repo, {chapter: body + "Committed growth.\n"})
    _git(repo, "commit", "-qam", "grow")
    if geometry == "candidate":
        linked = repo.parent / "linked"
        _git(repo, "worktree", "add", "-qb", "candidate/task", str(linked), "HEAD")
        ctx.serving_repo_dir, ctx.repo_dir = repo, linked
        ctx.task_metadata = {"body_candidate": {"path": str(linked)}}
        repo = linked
        assert (repo / ".git").is_file()
    else:
        _git(repo, "remote", "add", "origin", "https://github.com/contributor/ouroboros.git")
        _git(repo, "update-ref", "refs/remotes/origin/feature", "HEAD")
        _git(repo, "branch", "--set-upstream-to=origin/feature")
    clean = _diagnose(ctx, source)
    note = clean["book_balance"]["note"]
    assert "+0 B vs HEAD" in note and "+18 B vs refs/remotes/official-renamed/ouroboros" in note
    assert base in note and "nothing owed" not in note
    # Index and worktree deliberately carry different book growth.
    _write(repo, {chapter: body + "Committed growth.\nStaged.\n"})
    _git(repo, "add", chapter)
    _write(repo, {chapter: body + "Committed growth.\nWorktree differs.\n"})
    expected_release = admission.release_metadata_diagnostics(
        repo, source=source, neutral_allowed=source == "index" or geometry == "candidate")
    result = _diagnose(ctx, source)
    assert all(result[key] == value for key, value in expected_release.items())
    assert result["review_freshness"] is False and result["deterministic_only"] is True
    assert result["source"] == source and result["book_balance"]["source"] == "worktree"
    note = result["book_balance"]["note"]
    assert "+18 B vs HEAD" in note and "+36 B vs refs/remotes/official-renamed/ouroboros" in note
    assert "local commits are not blocked" in note
    assert _diagnose(ctx, source, paths=["change.py"])["book_balance"]["note"] == ""


@pytest.mark.parametrize("source", ["worktree", "index"])
def test_free_diagnostic_without_paths_measures_bible_and_the_prompts(candidate, source):
    """Omitted paths mean every book and ruled text, so a grown prompt is never 'nothing owed'."""
    repo = candidate.repo_dir
    _write(repo, {"ouroboros/reference_books.py": "# book contract\n", "BIBLE.md": "# BIBLE\n",
                  "prompts/SYSTEM.md": "system\n"})
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "texts")
    _git(repo, "remote", "add", "official", "https://github.com/razzant/ouroboros.git")
    _git(repo, "update-ref", "refs/remotes/official/ouroboros", "HEAD")
    _write(repo, {"prompts/SYSTEM.md": "system, grown\n"})
    note = _diagnose(candidate, source)["book_balance"]["note"]
    assert "Prompts 14 B: +7 B vs HEAD" in note and "BIBLE.md 8 B: +0 B vs HEAD" in note
    assert "nothing owed" not in note and "local commits are not blocked" in note


@pytest.mark.parametrize("source", ["worktree", "index"])
def test_free_book_diagnostic_distinguishes_foreign_and_unknown_baseline(candidate, source):
    # A foreign project can have the same docs paths without the body contract.
    assert _diagnose(candidate, source)["book_balance"]["note"] == ""
    _write(candidate.repo_dir, {"ouroboros/reference_books.py": "# book contract\n"})
    _git(candidate.repo_dir, "add", ".")
    _git(candidate.repo_dir, "commit", "-qm", "body contract")
    result = _diagnose(candidate, source)
    assert result["status"] == "not_applicable"
    assert "contribution unknown" in result["book_balance"]["note"]
    assert "nothing owed" not in result["book_balance"]["note"]


@pytest.mark.parametrize("source", ["worktree", "index"])
def test_reports_simultaneous_missing_row_history_overflow_and_desync(candidate, source):
    _broken(candidate.repo_dir)
    _git(candidate.repo_dir, "add", ".")
    result = _diagnose(candidate, source)
    assert result["status"] == "blocked" and result["unavailable"] == []
    findings = "\n".join(result["findings"])
    assert "no table row" in findings
    assert "patch rows (limit 5)" in findings
    assert "pyproject.toml" in findings and "README.md badge" in findings
    staged = review.format_name_status_for_preflight(_git(candidate.repo_dir, "diff", "--cached", "--name-status"))
    legacy = review._preflight_check("release", staged, candidate.repo_dir) if source == "index" else admission.release_metadata_preflight(candidate.repo_dir, "release", None)
    assert all(finding in legacy for finding in result["findings"])


@pytest.mark.parametrize("clean_source", ["worktree", "index"])
def test_partial_staging_uses_selected_source(candidate, clean_source):
    repo = candidate.repo_dir
    _write(repo, _release("1.2.4")) if clean_source == "index" else _broken(repo)
    _git(repo, "add", ".")
    _write(repo, _release("1.2.4")) if clean_source == "worktree" else _broken(repo)
    for source in ("worktree", "index"):
        report = _diagnose(candidate, source)
        assert report["status"] == ("clean" if source == clean_source else "blocked"), report
    assert (admission.release_metadata_preflight(repo, "release", None) is None) == (clean_source == "worktree")
    assert (admission.release_metadata_preflight(repo, "release", None, source="index") is None) == (clean_source == "index")


@pytest.mark.parametrize("version", ["", "not-a-release"])
def test_malformed_version_does_not_hide_independent_history_failure(candidate, version):
    _broken(candidate.repo_dir)
    (candidate.repo_dir / "VERSION").write_text(version, encoding="utf-8")
    result = _diagnose(candidate, "worktree")
    findings = "\n".join(result["findings"])
    assert result["status"] == "blocked"
    assert "empty or malformed" in findings and "patch rows (limit 5)" in findings
    assert "no table row" not in findings  # no invented version or dependent finding


@pytest.mark.parametrize("source", ["worktree", "index"])
@pytest.mark.parametrize("path", ["VERSION", "README.md", "pyproject.toml"])
def test_unreadable_source_preserves_independent_findings(candidate, monkeypatch, source, path):
    _broken(candidate.repo_dir)
    _git(candidate.repo_dir, "add", ".")
    read = admission.read_release_file

    def fail_one(repo, relative, **kwargs):
        if relative == path:
            raise PermissionError("fixture unavailable")
        return read(repo, relative, **kwargs)

    monkeypatch.setattr(admission, "read_release_file", fail_one)
    result = _diagnose(candidate, source)
    assert result["status"] == "unavailable"
    assert any(f"{source}:{path}" in message for message in result["unavailable"])
    assert result["findings"]  # independent readable metadata still diagnosed
    assert "PREFLIGHT_UNAVAILABLE" in admission.format_release_metadata_preflight(result)
    if path == "VERSION":
        assert not any("no table row" in message for message in result["findings"])


def test_index_read_failure_and_unmerged_entries_are_unavailable(candidate):
    repo = candidate.repo_dir
    _write(repo, _release("1.2.4"))
    _git(repo, "add", ".")
    blob = _git(repo, "rev-parse", ":pyproject.toml")
    # Install an unmerged optional carrier in this disposable fixture index.
    subprocess.run(["git", "update-index", "--index-info"], cwd=repo, check=True,
                   input=(f"0 {'0' * 40}\tpyproject.toml\n100644 {blob} 1\tpyproject.toml\n100644 {blob} 2\tpyproject.toml\n").encode("utf-8"))
    report = _diagnose(candidate, "index")
    assert report["status"] == "unavailable"
    assert any("pyproject.toml" in item for item in report["unavailable"])


@pytest.mark.parametrize("source", ["worktree", "index"])
def test_git_discovery_failure_is_never_clean(candidate, monkeypatch, source):
    real = admission.subprocess.run

    def fail_discovery(argv, **kwargs):
        if "status" in argv or "diff" in argv:
            return subprocess.CompletedProcess(argv, 128, stdout=b"", stderr=b"fixture failed")
        return real(argv, **kwargs)

    # The real run(check=True) raises for index discovery; emulate that contract.
    def checked(argv, **kwargs):
        result = fail_discovery(argv, **kwargs)
        if kwargs.get("check"):
            result.check_returncode()
        return result

    monkeypatch.setattr(admission.subprocess, "run", checked)
    report = _diagnose(candidate, source)
    assert report["status"] == "unavailable" and report["unavailable"]


def test_scope_preserves_contributor_and_candidate_forms_and_empty_paths(candidate):
    repo = candidate.repo_dir
    empty = _diagnose(candidate, "index")
    assert empty["status"] == "not_applicable" and empty["form"] == "none"
    _write(repo, {"change.py": "value = 2\n", "docs/notes.md": "notes\n"})
    _git(repo, "add", ".")
    # Code with VERSION untouched is the version-neutral form. P9 admits it for a contribution
    # (the prepared index lane) and for a body candidate commit; a serving-checkout commit still
    # needs the numbered release, including a documentation-only selection.
    staged = _diagnose(candidate, "index")
    assert staged["status"] == "not_applicable" and staged["form"] == "neutral"
    report = _diagnose(candidate, "worktree", paths=["change.py"])
    assert report["status"] == "blocked" and report["form"] == "neutral"
    assert len(report["findings"]) == 1 and "VERSION is not in scope" in report["findings"][0]
    assert "no table row" not in str(report)
    bound = admission.release_metadata_diagnostics(repo, ["change.py"], source="worktree", neutral_allowed=True)
    assert bound["status"] == "not_applicable" and bound["form"] == "neutral" and bound["findings"] == []
    docs = _diagnose(candidate, "worktree", paths=["docs/notes.md"])
    assert docs["status"] == "blocked" and docs["form"] == "neutral"
    assert "VERSION is not in scope" in docs["findings"][0]
    assert _diagnose(candidate, "index", paths=["VERSION"])["status"] == "not_applicable"


@pytest.mark.parametrize("source", ["worktree", "index"])
def test_neutral_form_keeps_every_carrier_span_byte_identical_to_head(candidate, source):
    """A partial carrier move without VERSION is neither form; a numbered release syncs all."""
    repo = candidate.repo_dir

    def _diagnose_candidate_commit():
        return admission.release_metadata_diagnostics(repo, source=source, neutral_allowed=True)

    _write(repo, {"change.py": "value = 2\n",
                  "pyproject.toml": '[project]\nversion = "1.2.4"\n'})  # one carrier moved, VERSION untouched
    _git(repo, "add", ".")
    report = _diagnose_candidate_commit()
    assert report["form"] == "neutral" and report["status"] == "blocked"
    assert len(report["findings"]) == 1 and "pyproject.toml" in report["findings"][0]
    assert "byte-identical to HEAD" in report["findings"][0]
    # The same carrier file touched WITHOUT moving its span (another key) stays neutral and clean.
    _write(repo, {"pyproject.toml": '[project]\nversion = "1.2.3"\nname = "ouroboros"\n'})
    _git(repo, "add", ".")
    report = _diagnose_candidate_commit()
    assert report["form"] == "neutral" and report["status"] == "not_applicable" and report["findings"] == []
    # Touching VERSION selects the numbered form: every carrier must follow in the same diff.
    _write(repo, {"VERSION": "1.2.4\n"})
    _git(repo, "add", ".")
    report = _diagnose_candidate_commit()
    assert report["form"] == "numbered" and report["status"] == "blocked"
    assert any("pyproject" in finding or "1.2.4" in finding for finding in report["findings"])
    _write(repo, _release("1.2.4"))
    _git(repo, "add", ".")
    report = _diagnose_candidate_commit()
    assert report["form"] == "numbered" and report["status"] == "clean" and report["findings"] == []


@pytest.mark.parametrize("change", ["deleted_with_code", "deleted_alone", "renamed_away"])
def test_neutral_form_reports_a_removed_carrier_file_through_every_consumer(candidate, change):
    """A carrier file the neutral diff deletes or renames away is a carrier change.

    Consumers: contribution preflight over the real staged name-status, the
    prepared index lane (its own staged discovery) and the bound-candidate worktree
    diagnostic. The removed file must still reach the comparison with HEAD.
    """
    repo = candidate.repo_dir
    if change == "renamed_away":
        _git(repo, "mv", "pyproject.toml", "pyproject.old.toml")
    else:
        _git(repo, "rm", "-q", "pyproject.toml")
    if change != "deleted_alone":
        _write(repo, {"change.py": "value = 2\n"})
    _git(repo, "add", "-A")

    staged = review._preflight_check("neutral change", review._build_preflight_staged(str(repo)), repo)
    prepared = _diagnose(candidate, "index")
    bound = admission.release_metadata_diagnostics(repo, source="worktree", neutral_allowed=True)

    assert staged and staged.startswith("⚠️ PREFLIGHT_BLOCKED") and "pyproject.toml" in staged, staged
    for report in (prepared, bound):
        assert (report["form"], report["status"]) == ("neutral", "blocked"), report
        assert len(report["findings"]) == 1 and "pyproject.toml" in report["findings"][0], report


def test_non_carrier_deletion_is_a_real_neutral_change(candidate):
    repo = candidate.repo_dir
    _git(repo, "rm", "-q", "change.py")
    assert review._preflight_check("drop module", review._build_preflight_staged(str(repo)), repo) is None
    prepared = _diagnose(candidate, "index")
    assert (prepared["status"], prepared["form"]) == ("not_applicable", "neutral")
    serving = admission.release_metadata_diagnostics(repo, source="index", neutral_allowed=False)
    assert (serving["status"], serving["form"]) == ("blocked", "neutral")
    assert "change.py" in serving["findings"][0]
    _write(repo, {"other.py": "value = 3\n"})
    _git(repo, "add", "-A")
    # Code with every carrier at HEAD's bytes stays the clean neutral form.
    assert review._preflight_check("replace module", review._build_preflight_staged(str(repo)), repo) is None
    prepared = _diagnose(candidate, "index")
    assert (prepared["status"], prepared["form"]) == ("not_applicable", "neutral")


def test_serving_refusal_needs_no_head_probe_even_for_documentation(candidate, tmp_path):
    """An unreadable HEAD must not mask the known serving-checkout release refusal."""
    repo = candidate.repo_dir
    _git(repo, "mv", "pyproject.toml", "docs/pyproject.old.md")
    serving = admission.release_metadata_diagnostics(repo, source="worktree", neutral_allowed=False)
    assert (serving["form"], serving["status"]) == ("neutral", "blocked"), serving
    assert "VERSION is not in scope" in serving["findings"][0] and "docs/pyproject.old.md" in serving["findings"][0]

    unborn = tmp_path / "unborn"
    unborn.mkdir()
    _git(unborn, "init", "-q")
    _write(unborn, {"change.py": "value = 2\n", "docs/notes.md": "notes\n"})
    _git(unborn, "add", "-A")
    for path in ("change.py", "docs/notes.md"):
        report = admission.release_metadata_diagnostics(unborn, [path], source="worktree", neutral_allowed=False)
        assert (report["form"], report["status"], report["unavailable"]) == ("neutral", "blocked", []), report
        assert "VERSION is not in scope" in report["findings"][0]


_NEW_LOCK = ('{\n  "name": "ouroboros-web",\n  "version": "1.2.3",\n  "lockfileVersion": 3,\n'
             '  "packages": {\n    "": {\n      "name": "ouroboros-web",\n      "version": "1.2.3"\n    }\n  }\n}\n')


def _three_consumers(ctx):
    """Contribution preflight, the prepared index lane and the bound-candidate worktree."""
    repo = ctx.repo_dir
    return (review._preflight_check("neutral change", review._build_preflight_staged(str(repo)), repo),
            _diagnose(ctx, "index"),
            admission.release_metadata_diagnostics(repo, source="worktree", neutral_allowed=True))


def test_neutral_form_reports_a_newly_introduced_carrier_file_through_every_consumer(candidate):
    """HEAD lacks web/package-lock.json (an optional older carrier); adding it with its
    declared span while VERSION stays put introduces a carrier, so it is no neutral form."""
    repo = candidate.repo_dir
    _write(repo, {"change.py": "value = 2\n", "web/package-lock.json": _NEW_LOCK})
    _git(repo, "add", "-A")

    staged, prepared, bound = _three_consumers(candidate)

    assert staged and staged.startswith("⚠️ PREFLIGHT_BLOCKED") and "web/package-lock.json" in staged, staged
    for report in (prepared, bound):
        assert (report["form"], report["status"]) == ("neutral", "blocked"), report
        assert len(report["findings"]) == 1, report
        assert "introduces" in report["findings"][0] and "web/package-lock.json" in report["findings"][0]


def test_optional_carrier_absent_on_both_sides_stays_optional(candidate):
    repo = candidate.repo_dir
    _write(repo, {"change.py": "value = 2\n"})
    _git(repo, "add", "-A")
    staged, prepared, bound = _three_consumers(candidate)
    assert staged is None
    for report in (prepared, bound):
        assert (report["form"], report["status"], report["unavailable"]) == ("neutral", "not_applicable", [])
    # Naming the absent optional carrier explicitly does not invent a change either.
    named = admission.release_metadata_diagnostics(
        repo, ["change.py", "web/package-lock.json"], source="worktree", neutral_allowed=True)
    assert (named["form"], named["status"], named["findings"]) == ("neutral", "not_applicable", [])


def test_neutral_carrier_comparison_without_a_readable_base_is_unavailable(candidate, tmp_path):
    """An unborn HEAD has no tree to compare with: unavailable evidence, never a clean neutral."""
    repo = tmp_path / "unborn"
    repo.mkdir()
    _git(repo, "init", "-q")
    _write(repo, {"change.py": "value = 2\n", "web/package-lock.json": _NEW_LOCK})
    _git(repo, "add", "-A")

    staged, prepared, bound = _three_consumers(SimpleNamespace(**{**vars(candidate), "repo_dir": repo}))

    assert staged and staged.startswith("⚠️ PREFLIGHT_UNAVAILABLE"), staged
    for report in (prepared, bound):
        assert (report["form"], report["status"], report["findings"]) == ("neutral", "unavailable", []), report
        assert any("web/package-lock.json" in item for item in report["unavailable"]), report


@pytest.mark.parametrize("source", ["", "other"])
def test_explicit_source_is_required_before_any_context_access(source):
    result = _diagnose(object(), source)
    assert result["failure_code"] == "PREFLIGHT_SOURCE_REQUIRED"


def test_schema_and_alias_offer_the_same_diagnostic_contract():
    entries = {entry.name: entry for entry in preflight_review.get_tools()}
    canonical, alias = entries["preflight_review"], entries["advisory_review"]
    assert canonical.schema["parameters"] == alias.schema["parameters"]
    params = canonical.schema["parameters"]["properties"]
    assert params["deterministic_only"]["default"] is False
    assert params["source"]["enum"] == ["worktree", "index"]
    assert "no reviewer" in params["deterministic_only"]["description"]


@pytest.mark.parametrize("unreadable, reason", [(True, "infra_failure"), (False, "preflight")])
def test_the_commit_gate_tells_an_unreadable_carrier_from_a_candidate_defect(candidate, monkeypatch, unreadable, reason):
    """The staged index's release metadata blocks every commit before any test run or
    look; an unreadable carrier is the check's own failure (``infra_failure``), a
    malformed one the candidate's defect (``preflight``)."""
    from ouroboros.tools import git

    _broken(candidate.repo_dir)
    if unreadable:
        (candidate.repo_dir / "VERSION").write_bytes(b"\xff")
    _git(candidate.repo_dir, "add", ".")
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    monkeypatch.setattr("ouroboros.tools.review_helpers.check_worktree_readiness", lambda *a, **kw: [])
    monkeypatch.setattr(git, "_run_review_preflight_tests", lambda *a, **kw: pytest.fail("no tests after a failed check"))
    outcome = git._preflight_and_tests_gate(candidate, "release", 0, classification_paths=["VERSION", "README.md"])
    assert outcome["status"] == "blocked" and outcome["block_reason"] == reason
    assert ("PREFLIGHT_UNAVAILABLE" in outcome["message"]) is unreadable


def test_author_continuation_formats_real_name_status_and_keeps_checks(candidate):
    from ouroboros.tools.commit_gate import bind_author_commit_candidate

    repo = candidate.repo_dir
    _broken(repo)
    _git(repo, "add", ".")
    candidate._author_commit_source = SimpleNamespace(block_reason="critical_findings", status="reviewed")
    candidate._author_commit_decision = {"disposition": "accepted", "rationale": "Inspected original findings."}
    candidate._author_commit_reference = {"attempt": 1}
    result = bind_author_commit_candidate(candidate, "release", {"fingerprint": "current"})
    assert "no table row" in result and "patch rows (limit 5)" in result and "pyproject.toml" in result
    _write(repo, _release("1.2.4"))
    _git(repo, "add", ".")
    assert bind_author_commit_candidate(candidate, "release", {"fingerprint": "corrected"}) is None
    assert candidate._author_commit_record["subject_hash"] == "corrected"
    assert candidate._author_commit_record["reviewer_signal"] == "critical_findings"


def test_release_validators_keep_optional_absence_and_detect_present_malformed_carriers():
    values = _release()
    assert release_metadata_findings(values) == []
    values["web/package.json"] = "malformed"
    values["uv.lock"] = ""
    findings = "\n".join(release_metadata_findings(values))
    assert "web/package.json" in findings and "uv.lock is empty" in findings


@pytest.mark.parametrize("source", ["worktree", "index"])
def test_missing_required_readme_is_unavailable_while_optional_older_carriers_are_absent(candidate, source):
    repo = candidate.repo_dir
    (repo / "VERSION").write_text("1.2.4\n", encoding="utf-8")
    (repo / "README.md").unlink()
    _git(repo, "add", "-A")
    result = _diagnose(candidate, source)
    assert result["status"] == "unavailable"
    assert any("README.md is missing" in item for item in result["unavailable"])
    assert not any("package-lock.json" in item for item in result["unavailable"])
    assert any("pyproject.toml" in item for item in result["findings"])


@pytest.mark.parametrize("path,content,expected", [
    ("ouroboros/new.py", "value = 1\n", "Architecture book"),
    ("tests/conftest.py", "def test_misplaced(): pass\n", "contains test functions"),
])
def test_author_formatter_also_preserves_non_release_staged_checks(candidate, path, content, expected):
    from ouroboros.tools.commit_gate import bind_author_commit_candidate

    candidate._author_commit_source = SimpleNamespace(block_reason="critical_findings", status="reviewed")
    candidate._author_commit_decision = {"disposition": "accepted", "rationale": "Read the review."}
    candidate._author_commit_reference = {"attempt": 1}
    candidate.serving_repo_dir = candidate.repo_dir.parent / "serving"
    candidate.task_metadata = {"body_candidate": {"path": str(candidate.repo_dir)}}
    _write(candidate.repo_dir, {path: content})
    _git(candidate.repo_dir, "add", ".")
    result = bind_author_commit_candidate(candidate, "candidate", {"fingerprint": "current"})
    assert expected in result


@pytest.mark.parametrize("message,reason", [
    ("⚠️ PREFLIGHT_UNAVAILABLE: index evidence unavailable", "infra_failure"),
    ("⚠️ PREFLIGHT_BLOCKED: malformed carrier", "preflight"),
])
def test_author_stage_cycle_preserves_preflight_failure_kind(candidate, monkeypatch, message, reason):
    from ouroboros.tools import commit_gate, git_review_cycle

    facade = git_review_cycle._git()
    candidate._author_commit_source = object()
    monkeypatch.setattr(facade, "_stage_candidate_for_review", lambda *a, **kw: ([], [], None))
    monkeypatch.setattr(facade, "protected_paths_in", lambda paths: [])
    monkeypatch.setattr(facade, "_current_runtime_mode", lambda: "pro")
    monkeypatch.setattr(facade, "_fingerprint_staged_diff", lambda repo: {"ok": True})
    monkeypatch.setattr(commit_gate, "bind_author_commit_candidate", lambda *a: message)
    result = git_review_cycle._run_reviewed_stage_cycle(candidate, "release", 0.0)
    assert result == {"status": "blocked", "message": message, "block_reason": reason}


def test_index_checks_the_whole_staged_candidate_even_with_narrow_paths(candidate):
    repo = candidate.repo_dir
    _write(repo, _release("1.2.4"))
    _git(repo, "add", ".")
    report = admission.release_metadata_diagnostics(repo, ["VERSION"], source="index")
    assert report["status"] == "clean"
    assert report["findings"] == []


@pytest.mark.parametrize("path,change", [
    ("README.md", "version"), ("README.md", "delete"), ("README.md", "rename"),
    ("docs/ARCHITECTURE.md", "version"), ("docs/ARCHITECTURE.md", "delete"),
])
def test_documentation_carriers_preserved_through_all_consumers(candidate, path, change):
    repo = candidate.repo_dir
    if change == "version":
        target = repo / path
        target.write_text(target.read_text().replace("1.2.3", "1.2.4"))
    elif change == "delete":
        (repo / path).unlink()
    else:
        (repo / path).rename(repo / "history.md")
    _git(repo, "add", "-A")
    staged, prepared, bound = _three_consumers(candidate)
    assert staged and "PREFLIGHT_BLOCKED" in staged and path in staged, staged
    for report in (prepared, bound):
        assert report["status"] == "blocked" and any(path in f for f in report["findings"]), report
    serving = admission.release_metadata_diagnostics(repo, source="worktree", neutral_allowed=False)
    assert serving["status"] == "blocked" and "VERSION is not in scope" in serving["findings"][0]
    assert ("history.md" if change == "rename" else path) in serving["findings"][0]


def test_documentation_prose_uses_the_explicit_neutral_forms(candidate):
    repo = candidate.repo_dir
    for path in ("README.md", "docs/ARCHITECTURE.md"):
        target = repo / path
        target.write_text(target.read_text(encoding="utf-8") + "## Usage\n\nOrdinary prose.\n", encoding="utf-8")
    _git(repo, "add", "-A")
    staged, prepared, bound = _three_consumers(candidate)
    assert staged is None
    for report in (prepared, bound):
        assert (report["status"], report["form"]) == ("not_applicable", "neutral"), report
    serving = admission.release_metadata_diagnostics(repo, source="worktree", neutral_allowed=False)
    assert serving["status"] == "blocked" and "VERSION is not in scope" in serving["findings"][0]


@pytest.mark.parametrize("consumer", ["commit", "author_continuation"])
@pytest.mark.parametrize("bound", [False, True])
@pytest.mark.parametrize("change", ["docs", "code", "delete_docs", "delete_code"])
def test_actual_commit_consumers_require_release_unless_bound(candidate, monkeypatch, consumer, bound, change):
    from ouroboros.tools import commit_gate

    repo = candidate.repo_dir
    _write(repo, {"docs/notes.md": "Original notes.\n"})
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "fixture notes")
    if bound:
        linked = repo.parent / "linked"
        _git(repo, "worktree", "add", "-qb", "candidate/task", str(linked), "HEAD")
        candidate.serving_repo_dir, candidate.repo_dir = repo, linked
        candidate.task_metadata = {"body_candidate": {"path": str(linked)}}
        repo = linked
    candidate._author_commit_source = SimpleNamespace(block_reason="critical_findings", status="reviewed")
    candidate._author_commit_decision = {"disposition": "accepted", "rationale": "Read the review."}
    candidate._author_commit_reference = {"attempt": 1}
    monkeypatch.setattr("ouroboros.tools.review_helpers.check_worktree_readiness", lambda *a, **kw: [])

    def preflight():
        if consumer == "commit":
            return commit_gate.deterministic_preflight(candidate, "change", None)
        return commit_gate.bind_author_commit_candidate(candidate, "change", {"fingerprint": "current"})

    assert not preflight(), "An unchanged index is not a serving commit."
    path = "docs/notes.md" if change.endswith("docs") else "change.py"
    if change.startswith("delete"):
        (repo / path).unlink()
    else:
        _write(repo, {path: "Changed notes.\n" if change == "docs" else "value = 2\n"})
    _git(repo, "add", "-A")
    result = preflight()
    if bound:
        assert not result
    else:
        assert result and "VERSION is not in scope" in result and path in result
    prepared = _diagnose(candidate, "index")
    assert (prepared["status"], prepared["form"]) == ("not_applicable", "neutral")
    _write(repo, _release("1.2.4"))
    _git(repo, "add", "-A")
    assert not preflight(), "A coherent numbered release remains admitted in either checkout."


@pytest.mark.parametrize("source", ["worktree", "index"])
def test_deleting_version_is_not_a_numbered_release(candidate, source):
    _git(candidate.repo_dir, "rm", "-q", "VERSION")
    for allowed in (False, True):
        report = admission.release_metadata_diagnostics(candidate.repo_dir, source=source, neutral_allowed=allowed)
        assert (report["form"], report["status"]) == ("neutral", "blocked"), report
        assert not report["unavailable"]
        assert "VERSION" in " ".join(report["findings"])
    error = review._preflight_check("delete version", review._build_preflight_staged(str(candidate.repo_dir)),
                                   candidate.repo_dir, neutral_allowed=True)
    assert error and "VERSION" in error


@pytest.mark.parametrize("mode,enforcement", [("pro", "advisory"), ("cyber_pro", "advisory"),
                                             ("cyber_pro", "blocking")])
@pytest.mark.parametrize("path", ["docs/notes.md", "change.py"])
def test_author_continuation_applies_mode_without_erasing_release_finding(candidate, monkeypatch, mode, enforcement, path):
    from ouroboros.review_state import load_state
    from ouroboros.tools import commit_gate, git
    from ouroboros.tools.registry import ToolContext

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", mode)
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", enforcement)
    monkeypatch.setattr("ouroboros.tools.review_helpers.check_worktree_readiness", lambda *a, **kw: [])
    repo = candidate.repo_dir
    _write(repo, {path: "Updated notes.\n" if path.endswith(".md") else "value = 2\n"})
    _git(repo, "add", "-A")
    ctx = ToolContext(repo_dir=repo, drive_root=candidate.drive_root, task_id=candidate.task_id,
                      emit_progress_fn=lambda *_a: None)
    git._reset_commit_review_state(ctx)
    fingerprint = git._fingerprint_staged_diff(repo)["fingerprint"]
    git._record_commit_attempt(ctx, "change", "reviewing", pre_review_fingerprint=fingerprint, paid=True)
    git._record_commit_attempt(ctx, "change", "reviewed", phase="review_only", pre_review_fingerprint=fingerprint)
    source = load_state(ctx.drive_root).attempts[-1]
    reference = {"surface": "commit", **{key: getattr(source, key) for key in
        ("repo_key", "task_id", "tool_name", "attempt", "pre_review_fingerprint")}}
    git._reset_commit_review_state(ctx)
    ctx._author_commit_source, ctx._author_commit_reference = source, reference
    ctx._author_commit_decision = {"disposition": "accepted", "rationale": "I accept the recorded facts."}
    monkeypatch.setattr(git, "_run_review_preflight_tests", lambda *a, **kw: pytest.fail("metadata refusal did not run tests"))
    monkeypatch.setattr(git, "_run_parallel_review", lambda *a, **kw: pytest.fail("author continuation buys no panel"))
    assert not commit_gate._neutral_commit_allowed(ctx), "Cyber changes action authority, never the release form."
    assert "VERSION is not in scope" in commit_gate.deterministic_preflight(ctx, "change", [path])
    head = _git(repo, "rev-parse", "HEAD")
    outcome = git._run_reviewed_stage_cycle(ctx, "change", 0, paths=[path], require_release_tag=False)
    if mode == "pro":
        assert outcome["status"] == "blocked" and outcome["block_reason"] == "preflight", outcome
        assert "VERSION is not in scope" in outcome["message"]
    else:
        assert outcome["status"] == "passed", outcome
        assert ctx._commit_review_status == "author_continued"
        assert "VERSION is not in scope" in " ".join(str(row) for row in ctx._review_advisory)
        assert ctx._preflight_execution["status"] == "not_run"
        assert "VERSION is not in scope" in ctx._preflight_execution["reason"]
        events = [json.loads(line) for line in (ctx.drive_logs() / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        assert any(row.get("type") == "review_advisory_override" and row.get("decision_authority") == "cyber_pro"
                   and row.get("block_reason") == "preflight" and "VERSION is not in scope" in row["message_head"]
                   for row in events)
    assert _git(repo, "rev-parse", "HEAD") == head, "A stage cycle authorizes; it does not itself commit."


@pytest.mark.parametrize("consumer", ["commit", "author_continuation"])
def test_managed_update_keeps_official_version_under_strict_resolver_authority(candidate, monkeypatch, consumer):
    from ouroboros.tools import commit_gate
    from supervisor import git_ops, update_merge

    repo = candidate.repo_dir
    branch, base = _git(repo, "branch", "--show-current"), _git(repo, "rev-parse", "HEAD")
    monkeypatch.setattr(git_ops, "BRANCH_DEV", branch)
    _git(repo, "checkout", "-qb", "official")
    _write(repo, {"change.py": "value = 2\n"})
    _git(repo, "commit", "-qam", "official change at the same release version")
    target = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", branch)
    _write(repo, {"change.py": "value = 3\n"})
    _git(repo, "commit", "-qam", "local customization")
    before = _git(repo, "rev-parse", "HEAD")
    ok, message, baseline = update_merge.materialize_assisted_merge_live(branch, before, target, base)
    assert ok and baseline, message
    _write(repo, {"change.py": "value = 4\n"})
    _git(repo, "add", "-A")
    tx = {"phase": "assisted_resolution", "task_id": candidate.task_id, "pre_update_sha": before,
          "target_sha": target, "m0_tree": baseline, "conflict_paths": ["change.py"]}
    update_merge.write_update_tx(tx)
    candidate.task_metadata = {"managed_update": {
        "authority_fingerprint": update_merge.assisted_authority_fingerprint(tx)}}
    candidate._author_commit_source = SimpleNamespace(block_reason="critical_findings", status="reviewed")
    candidate._author_commit_decision = {"disposition": "accepted", "rationale": "Read the review."}
    candidate._author_commit_reference = {"attempt": 1}
    monkeypatch.setattr("ouroboros.tools.review_helpers.check_worktree_readiness", lambda *a, **kw: [])

    def preflight():
        if consumer == "commit":
            return commit_gate.deterministic_preflight(candidate, "integrate update", None)
        return commit_gate.bind_author_commit_candidate(candidate, "integrate update", {"fingerprint": "current"})

    assert "VERSION" not in _git(repo, "diff", "--cached", "--name-only").splitlines()
    assert _git(repo, "show", f"{target}:VERSION") == _git(repo, "show", "HEAD:VERSION") == "1.2.3"
    assert not preflight(), "The authorized merge adopts the official version without inventing another release."
    candidate.task_id = "unrelated-task"
    assert "VERSION is not in scope" in preflight()
    candidate.task_id = tx["task_id"]
    _write(repo, {"pyproject.toml": '[project]\nversion = "1.2.4"\n'})
    _git(repo, "add", "pyproject.toml")
    assert "pyproject.toml" in preflight(), "Managed authority does not waive carrier-span coherence."


@pytest.mark.parametrize("carrier", ["readme_badge", "readme_history", "readme_download_refs", "architecture_header"])
@pytest.mark.parametrize("change", ["edit", "remove"])
def test_each_documentation_span_is_protected_without_a_code_change(candidate, carrier, change):
    from ouroboros.tools.release_sync import VERSION_CARRIER_SPANS, locate_carrier_span
    repo = candidate.repo_dir
    if carrier == "readme_download_refs":
        target = repo / "README.md"
        target.write_text(target.read_text() + "[download-macos-arm64]: https://example.invalid/v1.2.3/app\n")
        _git(repo, "add", ".")
        _git(repo, "commit", "-qm", "fixture download span")
    span = next(span for span in VERSION_CARRIER_SPANS if span.carrier_id == carrier)
    target = repo / span.path
    text = target.read_text()
    _, location = locate_carrier_span(text, span)
    assert location
    start, end = location
    replacement = text[start:end].replace("1.2.3", "1.2.4") if change == "edit" else ""
    target.write_text(text[:start] + replacement + text[end:])
    _git(repo, "add", "-A")
    staged, prepared, bound = _three_consumers(candidate)
    assert staged and "PREFLIGHT_BLOCKED" in staged and carrier in staged
    for report in (prepared, bound):
        assert report["status"] == "blocked" and carrier in " ".join(report["findings"])


def test_docs_only_introduction_of_download_span_is_not_prose(candidate):
    target = candidate.repo_dir / "README.md"
    target.write_text(target.read_text() + "[download-macos-arm64]: https://example.invalid/v1.2.4/app\n")
    _git(candidate.repo_dir, "add", "-A")
    staged, prepared, bound = _three_consumers(candidate)
    assert staged and "PREFLIGHT_BLOCKED" in staged and "readme_download_refs" in staged
    assert all(report["status"] == "blocked" for report in (prepared, bound))
