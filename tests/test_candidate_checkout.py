"""The browser candidate preserves dirty bytes and never writes the source Git state."""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import stat
import subprocess

import pytest

from tests import candidate_checkout as candidate

pytestmark = pytest.mark.serial


def test_ready_flag_does_not_hide_failed_supervisor():
    candidate.require_running_supervisor({"supervisor_ready": True, "workers_total": 1})
    for state in ({"supervisor_ready": True, "workers_total": 0},
                  {"supervisor_ready": True, "workers_total": 1, "supervisor_error": "init failed"}):
        with pytest.raises(candidate.CandidateError, match="CANDIDATE_SERVER_UNAVAILABLE"):
            candidate.require_running_supervisor(state)


def test_installed_project_cannot_supply_code_missing_from_candidate(tmp_path):
    """The LAUNCHED interpreter's import roots decide, not this process's venv.

    The positive branch runs in a fresh dependency-only environment (a bare venv
    made from this interpreter, stdlib only), because the venv running THIS test
    may legitimately carry an editable `ouroboros` install (`uv sync` without
    `--no-install-project`, as the ordinary CI action does) — exactly what the
    probe must refuse for a candidate server, and no evidence about the probe.
    """
    import sys
    import venv

    checkout = tmp_path / "checkout"
    (checkout / "ouroboros").mkdir(parents=True)
    (checkout / "ouroboros" / "__init__.py").write_text("", encoding="utf-8")
    venv.EnvBuilder(with_pip=False, symlinks=os.name != "nt").create(tmp_path / "bare-venv")
    scripts = "Scripts" if os.name == "nt" else "bin"
    python = str(tmp_path / "bare-venv" / scripts / ("python.exe" if os.name == "nt" else "python"))
    env = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
    env["PYTHONNOUSERSITE"] = "1"
    candidate.require_candidate_interpreter(python, env, checkout)
    if sys.executable != python:
        # Whether or not THIS venv installed the project, the probe answers from the child.
        try:
            candidate.require_candidate_interpreter(sys.executable, env, checkout)
        except candidate.CandidateError as exc:
            assert "--no-install-project" in str(exc)
    # A distribution only the child sees — as the Windows base interpreter's own
    # site-packages would be, or an editable install reached through PYTHONPATH.
    site = tmp_path / "base-site" / "ouroboros-9.9.dist-info"
    site.mkdir(parents=True)
    (site / "METADATA").write_text("Metadata-Version: 2.1\nName: ouroboros\nVersion: 9.9\n",
                                   encoding="utf-8")
    with pytest.raises(candidate.CandidateError, match="--no-install-project"):
        candidate.require_candidate_interpreter(python, {**env, "PYTHONPATH": str(site.parent)}, checkout)
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "ouroboros").mkdir(parents=True)
    (elsewhere / "ouroboros" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "bare").mkdir()
    with pytest.raises(candidate.CandidateError, match="outside the checkout"):
        candidate.require_candidate_interpreter(python, {**env, "PYTHONPATH": str(elsewhere)},
                                                tmp_path / "bare")


def git(repo, *args, input=None, env=None):
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True,
                          input=input, env=env).stdout


@pytest.fixture
def source(tmp_path):
    repo = tmp_path / "source"
    repo.mkdir()
    git(repo, "init", "-b", "candidate-test")
    git(repo, "config", "user.name", "Fixture")
    git(repo, "config", "user.email", "fixture@example.invalid")
    (repo / "edited").write_bytes(b"HEAD\n")
    (repo / "deleted").write_bytes(b"delete me\n")
    (repo / "recreated").write_bytes(b"old\n")
    (repo / ".gitignore").write_text("ignored/\n", encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "synthetic test baseline")
    return repo


def test_git_input_is_file_backed_with_identical_bytes_and_failure_semantics(source, monkeypatch):
    """Keep the real batched protocol without a parent-to-child writer pipe."""
    blob = git(source, "rev-parse", "HEAD:edited").strip()
    empty = git(source, "hash-object", "--stdin", input=b"").strip()
    absent = b"f" * len(blob)
    real_run, seen = subprocess.run, []

    def run(argv, **kwargs):
        stdin = kwargs.get("stdin")
        regular = stdin is not None and stat.S_ISREG(os.fstat(stdin.fileno()).st_mode)
        seen.append((kwargs.get("input"), regular, kwargs.get("timeout")))
        return real_run(argv, **kwargs)

    monkeypatch.setattr(candidate.subprocess, "run", run)
    rows = [blob, absent] * 2000
    assert candidate._git(source, "cat-file", "--batch-check", data=b"\n".join(rows) + b"\n") == (
        blob + b" blob 5\n" + absent + b" missing\n") * 2000
    assert candidate._git(source, "hash-object", "--stdin", data=b"").strip() == empty
    with pytest.raises(candidate.CandidateError, match=r"CANDIDATE_GIT_FAILED cat-file: \S"):
        candidate._git(source, "cat-file", "--batch-check", "--no-such-option", data=b"x\n")
    assert seen == [(None, True, 120)] * 3


def test_unproven_process_or_inner_marker_retains_the_copy_in_place(source, tmp_path):
    """Deletion needs a proven-gone holder; retention is marked for enclosing layers."""
    from ouroboros.test_environment import RETENTION_MARKER, retain_tree
    from tests._shared import retire_synthetic_retention

    proven = tmp_path / "proven"
    with candidate.candidate_checkout(source, proven) as copy:
        copy.hold()
        copy.release()  # The holder proved its process tree gone.
    assert not proven.exists()

    held, failing, nested = tmp_path / "held", tmp_path / "failing", (tmp_path / "nested").resolve()
    unproven = "1 process tree(s) started from this checkout were never proven gone"
    try:
        with pytest.raises(candidate.CandidateError, match="CANDIDATE_RETAINED.*never proven gone"):
            with candidate.candidate_checkout(source, held) as copy:
                copy.hold()  # Started, never proven gone: an ordinary exit is no proof.
        assert "never proven gone" in (held / RETENTION_MARKER).read_text(encoding="utf-8")

        with pytest.raises(RuntimeError, match="reap failed"):
            with candidate.candidate_checkout(source, failing) as copy:
                copy.hold()
                raise RuntimeError("reap failed")  # The original failure stays the error.
        assert (failing / RETENTION_MARKER).is_file()

        with pytest.raises(candidate.CandidateError, match="retention marker"):
            with candidate.candidate_checkout(source, nested) as copy:
                (nested / "server-data").mkdir()
                retain_tree(nested / "server-data", "inner layer could not prove its tree gone")
        assert (nested / "server-data" / RETENTION_MARKER).is_file()

        with candidate.candidate_checkout(source, tmp_path / "unmatched") as copy:
            with pytest.raises(candidate.CandidateError, match="release without a matching hold"):
                copy.release()
        assert not (tmp_path / "unmatched").exists()
    finally:
        # Every hold above was synthetic: no process was started from these copies.
        retire_synthetic_retention(held, unproven)
        retire_synthetic_retention(failing, unproven)
        retire_synthetic_retention(nested / "server-data", "inner layer could not prove its tree gone")
        retire_synthetic_retention(nested, "retention marker(s) beneath: "
                                   + str(nested / "server-data" / RETENTION_MARKER))


def test_staged_unstaged_new_binary_deleted_and_executable_bytes_survive(source, tmp_path):
    (source / "edited").write_bytes(b"staged\n")
    git(source, "add", "edited")
    (source / "edited").write_bytes(b"unstaged\r\n\x80\xff\x00")
    (source / "deleted").unlink()
    git(source, "rm", "recreated")
    (source / "recreated").write_bytes(b"new untracked after staged deletion\n")
    name = "new file\nwith tabs\t" if os.name != "nt" else "new file"
    (source / name).write_bytes(b"new\x00binary\xfe")
    (source / "executable").write_bytes(b"#!/bin/sh\nexit 0\n")
    (source / "executable").chmod(0o755)
    (source / "ignored").mkdir()
    (source / "ignored" / "artifact").write_bytes(b"not a Git candidate input")
    before = candidate.observe_candidate(source)
    target = tmp_path / "checkout"
    with candidate.candidate_checkout(source, target) as captured:
        assert captured.state == before and captured.overlay == {}
        assert git(target, "show", ":edited") == b"staged\n"
        assert (target / "edited").read_bytes() == b"unstaged\r\n\x80\xff\x00"
        assert not (target / "deleted").exists()
        assert not (target / "ignored").exists()
        assert (target / "recreated").read_bytes() == (source / "recreated").read_bytes()
        assert (target / "executable").stat().st_mode == (source / "executable").stat().st_mode
        assert candidate.observe_candidate(target).content_identity == before.content_identity
    assert candidate.observe_candidate(source) == before
    assert not target.exists()


def test_source_change_during_copy_is_explicit_failure(source, tmp_path, monkeypatch):
    copy = candidate._copy_candidate

    def racing_copy(*args):
        copy(*args)
        (source / "late-new").write_bytes(b"arrived during capture")

    monkeypatch.setattr(candidate, "_copy_candidate", racing_copy)
    with pytest.raises(candidate.CandidateError, match="CANDIDATE_CHANGED"):
        with candidate.candidate_checkout(source, tmp_path / "checkout"):
            pytest.fail("mixed candidate was admitted")
    assert (source / "late-new").read_bytes() == b"arrived during capture"


def test_empty_index_after_staged_deletions_is_preserved(source, tmp_path):
    git(source, "rm", "-r", ".")
    with candidate.candidate_checkout(source, tmp_path / "checkout") as captured:
        assert not captured.state.entries
        assert all(value is None for value in captured.state.files.values())


def test_empty_intent_to_add_transition_invalidates_copy(source, tmp_path):
    (source / "intent").write_bytes(b"")
    git(source, "add", "-N", "intent")
    before = candidate.observe_candidate(source)
    target = tmp_path / "checkout"
    with pytest.raises(candidate.CandidateError, match="CANDIDATE_CHANGED"):
        with candidate.candidate_checkout(source, target) as captured:
            candidate.verify_checkout(target, captured)
            git(target, "add", "intent")
            after = candidate.observe_candidate(target)
            assert before.entries == after.entries  # Blob, mode, path and stage are identical.
            assert before.status != after.status  # Staging intent is not.
            candidate.verify_checkout(target, captured)
    assert candidate.observe_candidate(source) == before


def test_split_index_is_refused(source, tmp_path):
    git(source, "update-index", "--split-index")
    with pytest.raises(candidate.CandidateError, match="split index"):
        with candidate.candidate_checkout(source, tmp_path / "checkout"):
            pytest.fail("unsupported index was accepted")


def test_git_status_inside_the_copy_is_not_drift_but_a_staged_change_is(source, tmp_path):
    """A served process runs `git status`; that rewrites index stat data, not the candidate."""
    (source / "edited").write_bytes(b"staged\n")
    git(source, "add", "edited")
    checkout = tmp_path / "checkout"
    with candidate.candidate_checkout(source, checkout) as captured:
        raw_index = (checkout / ".git" / "index").read_bytes()
        # Same bytes, new mtime: exactly what a served process sees after touching
        # its own tree; `git status` then rewrites the cached stat data in the index.
        os.utime(checkout / "edited", ns=(1_000_000_000_000_000_000, 1_000_000_000_000_000_000))
        # The launcher hands tests GIT_OPTIONAL_LOCKS=0; a served process runs with
        # Git's default, which takes the lock and writes the refreshed index.
        plain = {key: value for key, value in os.environ.items() if key != "GIT_OPTIONAL_LOCKS"}
        git(checkout, "status", "--porcelain", env=plain)
        git(checkout, "update-index", "--really-refresh", env=plain)
        assert (checkout / ".git" / "index").read_bytes() != raw_index, "fixture did not exercise the refresh"
        candidate.verify_checkout(checkout, captured)
        git(checkout, "update-index", "--add", "--cacheinfo", "100644",
            git(checkout, "hash-object", "-w", "--stdin", input=b"other\n").strip().decode(), "edited")
        with pytest.raises(candidate.CandidateError, match=r"metadata=\['entries', 'staged_diff'\]"):
            candidate.verify_checkout(checkout, captured)
        (checkout / "edited").write_bytes(b"content drift\n")
        with pytest.raises(candidate.CandidateError, match=r"paths=\['edited'\]"):
            candidate.verify_checkout(checkout, captured)
        git(checkout, "reset", "--quiet", "--", "edited")
        git(checkout, "update-index", "--add", "--cacheinfo", "100644",
            git(checkout, "hash-object", "--stdin", input=b"staged\n").strip().decode(), "edited")
        (checkout / "edited").write_bytes(b"staged\n")


@pytest.mark.parametrize("mutation", ["source", "checkout"])
def test_mutation_after_start_invalidates_success(source, tmp_path, mutation):
    checkout = tmp_path / "checkout"
    with pytest.raises(candidate.CandidateError, match="CANDIDATE_CHANGED"):
        with candidate.candidate_checkout(source, checkout):
            selected = source if mutation == "source" else checkout
            (selected / "edited").write_bytes(b"unexpected modification")
    assert not checkout.exists()


@pytest.mark.parametrize("flag", ["--assume-unchanged", "--skip-worktree"])
def test_unsupported_index_flags_fail_instead_of_omitting_inputs(source, tmp_path, flag):
    git(source, "update-index", flag, "edited")
    with pytest.raises(candidate.CandidateError, match="CANDIDATE_UNSUPPORTED"):
        with candidate.candidate_checkout(source, tmp_path / "checkout"):
            pytest.fail("unsupported input was accepted")


def test_symlink_is_refused_without_dereferencing_foreign_bytes(source, tmp_path):
    foreign = tmp_path / "sentinel"
    foreign.write_bytes(b"private")
    try:
        (source / "link").symlink_to(foreign)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(candidate.CandidateError, match="CANDIDATE_UNSUPPORTED"):
        with candidate.candidate_checkout(source, tmp_path / "checkout"):
            pytest.fail("symlink was silently copied or omitted")
    assert foreign.read_bytes() == b"private"


@pytest.mark.skipif(os.name == "nt", reason="POSIX FIFO")
def test_special_file_is_refused_before_a_blocking_read(source, tmp_path):
    os.mkfifo(source / "fifo")
    with pytest.raises(candidate.CandidateError, match="CANDIDATE_UNSUPPORTED"):
        candidate.observe_candidate(source)


def test_two_independent_copies_preserve_the_same_source(source, tmp_path):
    before = candidate.observe_candidate(source)

    def run(number):
        target = tmp_path / str(number) / "checkout"
        with candidate.candidate_checkout(source, target) as snapshot:
            assert (target / ".git").is_dir()
            assert (target / "edited").read_bytes() == b"HEAD\n"
            return snapshot.identity

    with ThreadPoolExecutor(max_workers=2) as executor:
        assert list(executor.map(run, range(2))) == [before.identity] * 2
    assert candidate.observe_candidate(source) == before


def test_origin_proof_bytes_are_unique_per_checkout_and_absent_from_the_source(source, tmp_path):
    (source / "VERSION").write_text("9.9.9\n", encoding="utf-8")
    (source / "web").mkdir()
    (source / "web" / "index.html").write_bytes(b"<!doctype html>\n")
    before = candidate.observe_candidate(source)
    identities, sentinels, versions = set(), set(), set()
    for number in range(2):
        target = tmp_path / str(number) / "checkout"
        with candidate.candidate_checkout(source, target, origin_proof=True) as checkout:
            # Proof bytes live in the COPY only; the source keeps what we observed.
            assert (target / candidate.SENTINEL_PATH).read_bytes() == checkout.sentinel_bytes
            assert not (source / candidate.SENTINEL_PATH).exists()
            assert (source / "VERSION").read_text(encoding="utf-8") == "9.9.9\n"
            # A parseable build-metadata suffix, not an invented version.
            assert checkout.version_text.startswith("9.9.9+candidate.")
            assert (target / "VERSION").read_text(encoding="utf-8").strip() == checkout.version_text
            identities.add(checkout.identity)
            sentinels.add(checkout.sentinel_bytes)
            versions.add(checkout.version_text)
            candidate.verify_checkout(target, checkout)
    assert len(identities) == 1, "the selected source bytes are one identity"
    assert len(sentinels) == 2 and len(versions) == 2, "proof bytes must not repeat across checkouts"
    assert candidate.observe_candidate(source) == before


def test_origin_proof_refuses_a_candidate_without_the_bytes_it_must_prove(source, tmp_path):
    with pytest.raises(candidate.CandidateError, match="CANDIDATE_UNSUPPORTED"):
        with candidate.candidate_checkout(source, tmp_path / "checkout", origin_proof=True):
            pytest.fail("a candidate without VERSION/web cannot carry the origin proof")
    assert not (tmp_path / "checkout").exists()


def test_proof_bytes_removed_from_the_checkout_invalidate_success(source, tmp_path):
    (source / "VERSION").write_text("9.9.9\n", encoding="utf-8")
    (source / "web").mkdir()
    (source / "web" / "index.html").write_bytes(b"<!doctype html>\n")
    target = tmp_path / "checkout"
    with pytest.raises(candidate.CandidateError, match="CANDIDATE_CHANGED"):
        with candidate.candidate_checkout(source, target, origin_proof=True):
            (target / candidate.SENTINEL_PATH).unlink()
    assert not target.exists()


def test_a_stopped_copy_takes_new_overlay_bytes_that_its_next_verification_binds(source, tmp_path):
    (source / "VERSION").write_text("9.9.9\n", encoding="utf-8")
    (source / "web").mkdir()
    (source / "web" / "index.html").write_bytes(b"<!doctype html>\n")
    target = tmp_path / "checkout"
    with candidate.candidate_checkout(source, target, origin_proof=True) as checkout:
        restarted = checkout.version_text.replace("9.9.9+", "9.9.10+", 1)
        checkout.hold()  # a server started from the copy may still read it
        with pytest.raises(candidate.CandidateError, match="CANDIDATE_CUSTODY"):
            checkout.rewrite(candidate.VERSION_PATH, f"{restarted}\n".encode())
        checkout.release()
        with pytest.raises(candidate.CandidateError, match="CANDIDATE_UNSUPPORTED"):
            checkout.rewrite("edited", b"source bytes are not per-checkout\n")
        checkout.rewrite(candidate.VERSION_PATH, f"{restarted}\n".encode())
        assert checkout.version_text == restarted
        assert (target / "VERSION").read_text(encoding="utf-8") == f"{restarted}\n"
        candidate.verify_checkout(target, checkout)
    assert (source / "VERSION").read_text(encoding="utf-8") == "9.9.9\n"
    assert not target.exists()


@pytest.mark.parametrize("suffix", [".cmd", ".bat", ".exe", ".com"])
def test_path_execute_bits_do_not_change_handle_identity(source, monkeypatch, suffix):
    real_lstat = Path.lstat

    class NamedStat:
        def __init__(self, observed):
            self.observed = observed

        def __getattr__(self, name):
            return getattr(self.observed, name)

        @property
        def st_mode(self):
            return self.observed.st_mode | 0o111

    def path_lstat(path):
        observed = real_lstat(path)
        return NamedStat(observed) if path.suffix == suffix else observed

    monkeypatch.setattr(Path, "lstat", path_lstat)
    payload = b"@echo off\r\nexit /b 0\r\n"
    name = "install" + suffix
    (source / name).write_bytes(payload)
    assert candidate._read(source, name)[0] == payload


def test_origin_proof_uses_observable_sibling_mode(source, tmp_path, monkeypatch):
    (source / "VERSION").write_bytes(b"9.9.9\n")
    (source / "web").mkdir()
    (source / "web" / "index.html").write_bytes(b"<!doctype html>\n")
    for name, value in candidate.observe_candidate(source).files.items():
        if value is not None:
            os.chmod(source / name, 0o666)
    real_chmod = Path.chmod
    monkeypatch.setattr(Path, "chmod", lambda path, mode, **kw: real_chmod(
        path, 0o666 if mode & stat.S_IWRITE else 0o444, **kw))
    with candidate.candidate_checkout(source, tmp_path / "copy", origin_proof=True) as checkout:
        for name, value in checkout.overlay.items():
            assert candidate._read(checkout.path, name)[:2] == value


def _origin_source(source):
    (source / "VERSION").write_text("9.9.9\n", encoding="utf-8")
    (source / "web").mkdir()
    (source / "web" / "index.html").write_bytes(b"<!doctype html>\n")


def test_cases_reuse_one_copy_under_their_own_custody_and_overlay(source, tmp_path, monkeypatch):
    """One materialization serves many cases; each gives back the overlay it rewrote."""
    _origin_source(source)
    with candidate.candidate_checkout(source, tmp_path / "copy", origin_proof=True) as owner:
        monkeypatch.setattr(candidate, "_copy_candidate", lambda *args: pytest.fail("a case copied the source again"))
        with candidate.case_checkout(source, owner) as first:
            first.hold()
            first.release()  # Proven gone: the stopped copy may take another VERSION.
            first.rewrite(candidate.VERSION_PATH, b"9.9.10+restart\n")
            candidate.verify_checkout(owner.path, first)
        assert candidate._read(owner.path, candidate.VERSION_PATH)[:2] == owner.overlay[candidate.VERSION_PATH]
        with candidate.case_checkout(source, owner) as second, candidate.case_checkout(source, owner) as third:
            assert second.path == first.path and second.version_text == owner.version_text
            second.hold()  # Its process may read the shared bytes, so no other case rewrites them now.
            with pytest.raises(candidate.CandidateError, match="CANDIDATE_CUSTODY"):
                third.rewrite(candidate.VERSION_PATH, b"9.9.11\n")
            assert owner.unproven == 1
            second.release()
        assert owner.unproven == 0
    assert not (tmp_path / "copy").exists()


@pytest.mark.parametrize("when", ["between", "during"])
def test_a_case_refuses_a_source_that_left_the_copied_content(source, tmp_path, when):
    _origin_source(source)
    with pytest.raises(candidate.CandidateError, match="CANDIDATE_CHANGED during browser verification"):
        with candidate.candidate_checkout(source, tmp_path / "copy", origin_proof=True) as owner:
            if when == "during":
                with pytest.raises(candidate.CandidateError, match="CANDIDATE_CHANGED during browser verification"):
                    with candidate.case_checkout(source, owner):
                        (source / "edited").write_bytes(b"edited while the case ran\n")
            else:
                (source / "edited").write_bytes(b"edited between cases\n")
            with pytest.raises(candidate.CandidateError, match=r"since the shared checkout was copied: .*'edited'"):
                with candidate.case_checkout(source, owner):
                    pytest.fail("a case ran bytes the source no longer holds")
    assert not (tmp_path / "copy").exists()


@pytest.mark.parametrize("left", ["candidate file", "ignored file"])
def test_state_a_case_leaves_in_the_copy_fails_it_and_the_next_case(source, tmp_path, left):
    _origin_source(source)
    match = r"paths=\['edited'\]" if left == "candidate file" else r"ignored files in the shared checkout: \['ignored/'\]"
    with candidate.candidate_checkout(source, tmp_path / "copy", origin_proof=True) as owner:
        with pytest.raises(candidate.CandidateError, match=match):
            with candidate.case_checkout(source, owner):
                if left == "candidate file":
                    (owner.path / "edited").write_bytes(b"written beside the code\n")
                else:
                    (owner.path / "ignored").mkdir()
                    (owner.path / "ignored" / "server.log").write_bytes(b"runtime state\n")
        with pytest.raises(candidate.CandidateError, match=match):
            with candidate.case_checkout(source, owner):
                pytest.fail("a case booted beside an earlier case's state")
        if left == "candidate file":
            (owner.path / "edited").write_bytes(owner.state.files["edited"][0])
        else:
            (owner.path / "ignored" / "server.log").unlink()
            (owner.path / "ignored").rmdir()
        with candidate.case_checkout(source, owner):
            pass  # The owner's bytes again: cases proceed.
    assert not (tmp_path / "copy").exists()


def test_an_unproven_case_keeps_its_hold_on_the_owner_and_marks_retention_now(source, tmp_path, capsys):
    from ouroboros.test_environment import RETENTION_MARKER
    from tests._shared import retire_synthetic_retention

    _origin_source(source)
    copy, one = tmp_path / "owner" / "copy", "1 process tree(s) started from this checkout were never proven gone"
    try:
        with pytest.raises(candidate.CandidateError, match="CANDIDATE_RETAINED"):
            with candidate.candidate_checkout(source, copy, origin_proof=True) as owner:
                with pytest.raises(candidate.CandidateError, match="CANDIDATE_RETAINED.*never proven gone"):
                    with candidate.case_checkout(source, owner) as case:
                        case.hold()  # Started, never proven gone: an ordinary exit is no proof.
                assert owner.unproven == 1 and one in (copy.parent / RETENTION_MARKER).read_text(encoding="utf-8")
                with pytest.raises(candidate.CandidateError, match="CANDIDATE_CUSTODY"):
                    with candidate.case_checkout(source, owner):
                        pytest.fail("a later case overlapped an unproven process")
                assert owner.unproven == 1
        assert (copy / RETENTION_MARKER).is_file()
    finally:
        # Every hold above was synthetic: no process was started from this copy.
        retire_synthetic_retention(copy.parent, one)
        retire_synthetic_retention(copy, one)


def test_case_restores_git_config_after_success_and_failure(source, tmp_path):
    _origin_source(source)
    with candidate.candidate_checkout(source, tmp_path / "copy", origin_proof=True) as owner:
        config = owner.path / ".git" / "config"
        before = config.read_bytes()
        for failed in (False, True):
            try:
                with candidate.case_checkout(source, owner) as case:
                    case.hold()
                    git(owner.path, "remote", "add", "managed", "https://example.invalid/repo")
                    case.release()
                    if failed:
                        raise RuntimeError("test body failed")
            except RuntimeError as exc:
                assert failed and str(exc) == "test body failed"
            assert config.read_bytes() == before
            with candidate.case_checkout(source, owner):
                assert b"managed" not in git(owner.path, "remote")


def test_case_rejects_ref_drift_and_does_not_restore_config_until_proven_gone(source, tmp_path):
    from ouroboros.test_environment import RETENTION_MARKER
    from tests._shared import retire_synthetic_retention

    _origin_source(source)
    copy = tmp_path / "copy"
    with candidate.candidate_checkout(source, copy, origin_proof=True) as owner:
        with pytest.raises(candidate.CandidateError, match="refs changed"):
            with candidate.case_checkout(source, owner):
                git(copy, "tag", "case-tag")
        with pytest.raises(candidate.CandidateError, match="refs differ"):
            with candidate.case_checkout(source, owner):
                pytest.fail("a later case inherited changed refs")
        git(copy, "tag", "-d", "case-tag")  # Synthetic test restores its own known mutation.
        config = copy / ".git" / "config"
        before = config.read_bytes()
        try:
            with pytest.raises(RuntimeError, match="reap failed"):
                with candidate.case_checkout(source, owner) as case:
                    case.hold()  # Synthetic: no process is launched.
                    git(copy, "remote", "add", "managed", "https://example.invalid/repo")
                    changed = config.read_bytes()
                    raise RuntimeError("reap failed")
            assert config.read_bytes() == changed != before
            assert (copy.parent / RETENTION_MARKER).exists()
        finally:
            case.release()
            config.write_bytes(before)
            retire_synthetic_retention(copy.parent, "1 process tree(s) started from this checkout were never proven gone")
