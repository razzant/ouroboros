"""HTTP contract for GET /api/tasks/{task_id}/diff.

The endpoint answers ONE typed lifecycle for both task shapes:

- a WORKSPACE task projects its durable ``workspace.patch`` artifact (with its
  manifest as the base/head and blocker source) — a task whose artifacts are not
  finalized yet is ``pending``, never a 404 and never a fabricated "no changes";
- a SELF-REPO task has no historical patch, so it projects the paths the
  mutation-attribution authority attributed to the task window against the
  CURRENT repo, discloses baseline drift as a boolean, passes attribution
  blockers through, and refuses (typed ``projection_changed_during_read``)
  rather than serving a patch that does not belong to the disclosed baseline.

The "adversarial regressions" section at the end pins the hostile reproductions an
adversarial review landed against this endpoint: pathspec-magic candidate names,
an option-shaped ``base_commit``, non-UTF-8 patch bytes, and a new file whose name
git would C-quote. Each of those once produced a wrong answer (a leaked
unattributed edit, a file written outside the repo, a 503, a silently missing
file); the assertions below are written from the FIXED behaviour so a regression
fails loudly instead of degrading quietly.

Every test is hermetic: a tmp drive root, a tmp git repo, no supervisor.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess

from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros.gateway import task_diff as gateway_task_diff
from ouroboros.gateway.tasks import api_task_diff
from ouroboros.headless import task_artifacts_dir
from ouroboros.task_results import STATUS_COMPLETED, STATUS_RUNNING, write_task_result


def _client(drive_root: pathlib.Path, repo_dir: pathlib.Path) -> TestClient:
    app = Starlette(routes=[Route("/api/tasks/{task_id}/diff", api_task_diff, methods=["GET"])])
    app.state.drive_root = drive_root
    app.state.repo_dir = repo_dir
    return TestClient(app)


def _git(root: pathlib.Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=str(root), capture_output=True, text=True, check=True,
        env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
             "GIT_COMMITTER_EMAIL": "t@t", "PATH": __import__("os").environ.get("PATH", ""),
             "HOME": str(root), "LC_ALL": "C"},
    )
    return proc.stdout.strip()


def _init_repo(root: pathlib.Path) -> str:
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-q", "-b", "main")
    (root / "loop.py").write_text("a = 1\nb = 2\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "base")
    return _git(root, "rev-parse", "HEAD")


def _write_artifact(drive_root: pathlib.Path, task_id: str, name: str, body: str) -> pathlib.Path:
    path = task_artifacts_dir(drive_root, task_id) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def _baseline_evidence(root: pathlib.Path, base_commit: str, **extra) -> dict:
    return {
        "effect_state": "observed_window",
        "baseline": {
            "baseline_hash": "hash-1",
            "surfaces": [{
                "surface_type": "system_repo",
                "canonical_root": str(root.resolve()),
                "git": {
                    "base_commit": base_commit,
                    "base_tree": "",
                    "dirty_paths": [],
                    "dirty_fingerprints": {},
                },
            }],
        },
        **extra,
    }


# --- workspace source -------------------------------------------------------

def test_workspace_ready_serves_full_patch_artifact_bytes(tmp_path):
    drive_root = tmp_path / "drive"
    patch_text = "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-old\n+new\n"
    patch_path = _write_artifact(drive_root, "ws-ready", "workspace.patch", patch_text)
    manifest_path = _write_artifact(drive_root, "ws-ready", "workspace_patch.json", json.dumps({
        "status": "ready_with_changes", "base_head": "aaa", "current_head": "aaa", "errors": [],
    }))
    write_task_result(
        drive_root, "ws-ready", STATUS_COMPLETED,
        workspace_root=str(tmp_path / "ws"),
        artifact_status="ready_with_changes",
        artifacts=[
            {"kind": "workspace_patch", "name": "workspace.patch", "path": str(patch_path)},
            {"kind": "workspace_patch_manifest", "name": "workspace_patch.json", "path": str(manifest_path)},
        ],
    )
    with _client(drive_root, tmp_path / "repo") as client:
        payload = client.get("/api/tasks/ws-ready/diff").json()
    assert payload["status"] == "ready"
    assert payload["source"] == "workspace_patch"
    assert payload["patch"] == patch_text
    assert payload["patch_sha256"] == hashlib.sha256(patch_text.encode("utf-8")).hexdigest()
    assert payload["base_commit"] == "aaa"
    assert payload["head_advanced"] is False
    assert payload["blockers"] == []


def test_workspace_running_without_finalized_artifacts_is_pending(tmp_path):
    drive_root = tmp_path / "drive"
    write_task_result(
        drive_root, "ws-pending", STATUS_RUNNING,
        workspace_root=str(tmp_path / "ws"), artifact_status="pending",
    )
    with _client(drive_root, tmp_path / "repo") as client:
        response = client.get("/api/tasks/ws-pending/diff")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "pending"
    assert payload["source"] == "workspace_patch"
    assert payload["patch"] == ""


def test_workspace_ready_no_changes_manifest_is_empty(tmp_path):
    drive_root = tmp_path / "drive"
    manifest_path = _write_artifact(drive_root, "ws-empty", "workspace_patch.json", json.dumps({
        "status": "ready_no_changes", "base_head": "bbb", "current_head": "bbb", "errors": [],
    }))
    write_task_result(
        drive_root, "ws-empty", STATUS_COMPLETED,
        workspace_root=str(tmp_path / "ws"), artifact_status="ready_no_changes",
        artifacts=[{"name": "workspace_patch.json", "path": str(manifest_path)}],
    )
    with _client(drive_root, tmp_path / "repo") as client:
        payload = client.get("/api/tasks/ws-empty/diff").json()
    assert payload["status"] == "empty"
    assert payload["patch"] == ""
    assert payload["base_commit"] == "bbb"


def test_workspace_head_change_is_blocked_with_manifest_blockers(tmp_path):
    drive_root = tmp_path / "drive"
    manifest_path = _write_artifact(drive_root, "ws-failed", "workspace_patch.json", json.dumps({
        "status": "failed", "base_head": "ccc", "current_head": "ddd",
        "errors": [{"type": "workspace_head_changed", "message": "HEAD moved"}],
    }))
    write_task_result(
        drive_root, "ws-failed", STATUS_COMPLETED,
        workspace_root=str(tmp_path / "ws"), artifact_status="failed",
        artifacts=[{"name": "workspace_patch.json", "path": str(manifest_path)}],
    )
    with _client(drive_root, tmp_path / "repo") as client:
        payload = client.get("/api/tasks/ws-failed/diff").json()
    assert payload["status"] == "blocked"
    assert payload["blockers"] == ["workspace_head_changed"]
    assert payload["head_advanced"] is True


def test_terminal_workspace_task_without_any_artifact_is_blocked_not_empty(tmp_path):
    """A finished workspace task with no patch artifact must not claim "no changes"."""
    drive_root = tmp_path / "drive"
    write_task_result(
        drive_root, "ws-none", STATUS_COMPLETED,
        workspace_root=str(tmp_path / "ws"), artifact_status="ready_with_changes",
    )
    with _client(drive_root, tmp_path / "repo") as client:
        payload = client.get("/api/tasks/ws-none/diff").json()
    assert payload["status"] == "blocked"
    assert payload["blockers"] == ["artifact_not_declared"]


def test_artifact_path_outside_task_dir_is_refused_as_a_typed_blocker(tmp_path):
    """The shared resolver's containment guard is what the diff path relies on."""
    drive_root = tmp_path / "drive"
    outside = tmp_path / "elsewhere" / "workspace.patch"
    outside.parent.mkdir(parents=True, exist_ok=True)
    outside.write_text("diff --git a/x b/x\n", encoding="utf-8")
    write_task_result(
        drive_root, "ws-escape", STATUS_COMPLETED,
        workspace_root=str(tmp_path / "ws"), artifact_status="ready_with_changes",
        artifacts=[{"name": "workspace.patch", "path": str(outside)}],
    )
    with _client(drive_root, tmp_path / "repo") as client:
        payload = client.get("/api/tasks/ws-escape/diff").json()
    assert payload["status"] == "blocked"
    assert payload["blockers"] == ["artifact_outside_task_dir"]
    assert payload["patch"] == ""


# --- self-repo source -------------------------------------------------------

def test_self_repo_terminal_snapshot_drives_the_patch(tmp_path):
    repo = tmp_path / "repo"
    base = _init_repo(repo)
    (repo / "loop.py").write_text("a = 1\nb = 3\n", encoding="utf-8")
    (repo / "new_file.py").write_text("fresh = True\n", encoding="utf-8")
    drive_root = tmp_path / "drive"
    write_task_result(
        drive_root, "self-terminal", STATUS_COMPLETED,
        mutation_evidence=_baseline_evidence(
            repo, base,
            effect_state="quiescent",
            terminal_candidate_snapshot={
                "captured_at": "now", "baseline_hash": "hash-1",
                "surfaces": [{
                    "surface_type": "system_repo",
                    "canonical_root": str(repo.resolve()),
                    "candidates": ["loop.py", "new_file.py"],
                    "excluded_preexisting_dirty": [],
                    "blockers": [],
                    "head_advanced": False,
                }],
            },
        ),
    )
    with _client(drive_root, repo) as client:
        payload = client.get("/api/tasks/self-terminal/diff").json()
    assert payload["status"] == "ready"
    assert payload["source"] == "mutation_baseline"
    assert payload["base_commit"] == base
    assert payload["head_advanced"] is False
    assert payload["blockers"] == []
    assert "-b = 2" in payload["patch"] and "+b = 3" in payload["patch"]
    # The untracked new file arrives through its own --no-index section.
    assert "new_file.py" in payload["patch"] and "+fresh = True" in payload["patch"]
    assert payload["patch_sha256"] == hashlib.sha256(payload["patch"].encode("utf-8")).hexdigest()


def test_terminal_drift_is_measured_at_READ_time_not_from_the_snapshot(tmp_path):
    """The patch is taken against the CURRENT repo, so drift must be read now.

    The snapshot recorded ``head_advanced: False`` when the task ended; HEAD moved
    afterwards. Trusting the recorded flag would describe a repo that has since
    moved on, and the owner would review a projection whose baseline silently
    disagrees with the disclosure.
    """
    repo = tmp_path / "repo"
    base = _init_repo(repo)
    (repo / "loop.py").write_text("a = 1\nb = 8\n", encoding="utf-8")
    drive_root = tmp_path / "drive"
    write_task_result(
        drive_root, "self-late-drift", STATUS_COMPLETED,
        mutation_evidence=_baseline_evidence(
            repo, base,
            effect_state="quiescent",
            terminal_candidate_snapshot={
                "captured_at": "now", "baseline_hash": "hash-1",
                "surfaces": [{
                    "surface_type": "system_repo",
                    "canonical_root": str(repo.resolve()),
                    "candidates": ["loop.py"],
                    "blockers": [],
                    "head_advanced": False,
                }],
            },
        ),
    )
    (repo / "unrelated.py").write_text("later = True\n", encoding="utf-8")
    _git(repo, "add", "unrelated.py")
    _git(repo, "commit", "-qm", "someone else moved HEAD")
    with _client(drive_root, repo) as client:
        payload = client.get("/api/tasks/self-late-drift/diff").json()
    assert payload["head_advanced"] is True
    assert payload["status"] == "ready"
    assert payload["base_commit"] == base
    assert "+b = 8" in payload["patch"]
    # Only the attributed path is projected; the unrelated commit is not claimed.
    assert "unrelated.py" not in payload["patch"]


def test_self_repo_running_uses_the_live_attribution_authority(tmp_path):
    repo = tmp_path / "repo"
    base = _init_repo(repo)
    (repo / "loop.py").write_text("a = 1\nb = 9\n", encoding="utf-8")
    drive_root = tmp_path / "drive"
    write_task_result(
        drive_root, "self-running", STATUS_RUNNING,
        mutation_evidence=_baseline_evidence(repo, base),
    )
    with _client(drive_root, repo) as client:
        payload = client.get("/api/tasks/self-running/diff").json()
    assert payload["status"] == "ready"
    assert payload["head_advanced"] is False
    assert payload["blockers"] == []
    assert "+b = 9" in payload["patch"]


def test_self_repo_head_drift_is_disclosed_as_a_boolean_not_a_refusal(tmp_path):
    """Decision 33: drift discloses a boolean AND still shows the projection.

    The task committed one attributed change and left another dirty. The live
    authority reports ``baseline_stale`` (HEAD moved), which this endpoint renders
    as ``head_advanced`` while still serving the current projection — a refusal
    here would hide real work behind an evidence footnote.
    """
    repo = tmp_path / "repo"
    base = _init_repo(repo)
    (repo / "loop.py").write_text("a = 1\nb = 4\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "task commit")
    (repo / "later.py").write_text("still_working = True\n", encoding="utf-8")
    drive_root = tmp_path / "drive"
    write_task_result(
        drive_root, "self-drift", STATUS_RUNNING,
        mutation_evidence=_baseline_evidence(repo, base),
    )
    with _client(drive_root, repo) as client:
        payload = client.get("/api/tasks/self-drift/diff").json()
    assert payload["head_advanced"] is True
    assert payload["status"] == "ready"
    assert "baseline_stale" in payload["blockers"]
    assert "+still_working = True" in payload["patch"]


def test_running_task_whose_work_is_fully_committed_is_blocked_not_empty(tmp_path):
    """Honest limit of the LIVE authority (decision 33).

    ``attributed_git_candidates`` attributes working-tree-dirty paths only; a
    running task that already committed everything has no live candidate set. The
    answer is ``blocked`` with the drift disclosed — never ``empty``, which would
    claim the task changed nothing. The committed paths become visible once the
    task terminalizes and its ``terminal_candidate_snapshot`` is recorded.
    """
    repo = tmp_path / "repo"
    base = _init_repo(repo)
    (repo / "loop.py").write_text("a = 1\nb = 7\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "all committed")
    drive_root = tmp_path / "drive"
    write_task_result(
        drive_root, "self-committed", STATUS_RUNNING,
        mutation_evidence=_baseline_evidence(repo, base),
    )
    with _client(drive_root, repo) as client:
        payload = client.get("/api/tasks/self-committed/diff").json()
    assert payload["status"] == "blocked"
    assert payload["head_advanced"] is True
    assert payload["blockers"] == ["baseline_stale"]


def test_self_repo_without_a_baseline_is_blocked_never_empty(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    drive_root = tmp_path / "drive"
    write_task_result(drive_root, "self-nobaseline", STATUS_RUNNING)
    with _client(drive_root, repo) as client:
        payload = client.get("/api/tasks/self-nobaseline/diff").json()
    assert payload["status"] == "blocked"
    assert payload["blockers"] == ["baseline_missing"]
    assert payload["patch"] == ""


def test_terminal_self_repo_without_snapshot_is_blocked(tmp_path):
    repo = tmp_path / "repo"
    base = _init_repo(repo)
    drive_root = tmp_path / "drive"
    write_task_result(
        drive_root, "self-nosnapshot", STATUS_COMPLETED,
        mutation_evidence=_baseline_evidence(repo, base),
    )
    with _client(drive_root, repo) as client:
        payload = client.get("/api/tasks/self-nosnapshot/diff").json()
    assert payload["status"] == "blocked"
    assert payload["blockers"] == ["terminal_snapshot_missing"]
    assert payload["base_commit"] == base


def test_self_repo_clean_projection_with_no_candidates_is_empty(tmp_path):
    repo = tmp_path / "repo"
    base = _init_repo(repo)
    drive_root = tmp_path / "drive"
    write_task_result(
        drive_root, "self-clean", STATUS_RUNNING,
        mutation_evidence=_baseline_evidence(repo, base),
    )
    with _client(drive_root, repo) as client:
        payload = client.get("/api/tasks/self-clean/diff").json()
    assert payload["status"] == "empty"
    assert payload["blockers"] == []


def test_projection_race_retries_once_then_refuses(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    base = _init_repo(repo)
    (repo / "loop.py").write_text("a = 1\nb = 5\n", encoding="utf-8")
    drive_root = tmp_path / "drive"
    write_task_result(
        drive_root, "self-race", STATUS_RUNNING,
        mutation_evidence=_baseline_evidence(repo, base),
    )
    seen = []
    monkeypatch.setattr(
        gateway_task_diff, "_projection_fingerprint",
        lambda root, candidates, index_path=None: seen.append(1) or f"fp-{len(seen)}",
    )
    with _client(drive_root, repo) as client:
        payload = client.get("/api/tasks/self-race/diff").json()
    assert payload["status"] == "blocked"
    assert "projection_changed_during_read" in payload["blockers"]
    assert payload["patch"] == ""
    # Exactly one retry: before/after for the first read, before/after for the retry.
    assert len(seen) == 4


def test_projection_race_that_settles_on_the_retry_answers_ready(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    base = _init_repo(repo)
    (repo / "loop.py").write_text("a = 1\nb = 6\n", encoding="utf-8")
    drive_root = tmp_path / "drive"
    write_task_result(
        drive_root, "self-settle", STATUS_RUNNING,
        mutation_evidence=_baseline_evidence(repo, base),
    )
    values = iter(["fp-a", "fp-b", "fp-c", "fp-c"])
    monkeypatch.setattr(
        gateway_task_diff, "_projection_fingerprint", lambda root, candidates, index_path=None: next(values),
    )
    with _client(drive_root, repo) as client:
        payload = client.get("/api/tasks/self-settle/diff").json()
    assert payload["status"] == "ready"
    assert "projection_changed_during_read" not in payload["blockers"]
    assert "+b = 6" in payload["patch"]


# --- lifecycle --------------------------------------------------------------

def test_unknown_task_id_is_the_only_404(tmp_path):
    with _client(tmp_path / "drive", tmp_path / "repo") as client:
        response = client.get("/api/tasks/absent-task/diff")
    assert response.status_code == 404
    assert response.json() == {"error": "task not found", "task_id": "absent-task"}


def test_malformed_task_id_is_a_400(tmp_path):
    with _client(tmp_path / "drive", tmp_path / "repo") as client:
        response = client.get("/api/tasks/not a task id/diff")
    assert response.status_code == 400


def test_response_carries_exactly_the_declared_envelope(tmp_path):
    drive_root = tmp_path / "drive"
    write_task_result(
        drive_root, "envelope", STATUS_RUNNING,
        workspace_root=str(tmp_path / "ws"), artifact_status="pending",
    )
    with _client(drive_root, tmp_path / "repo") as client:
        payload = client.get("/api/tasks/envelope/diff").json()
    assert set(payload) == {
        "status", "source", "base_commit", "head_advanced", "blockers", "patch", "patch_sha256",
    }


# --- adversarial regressions ------------------------------------------------
# Ported from the reviewer's hostile reproductions. Every assertion here is the
# FIXED behaviour; each test names the wrong answer it prevents.

def _terminal_evidence(root: pathlib.Path, base_commit: str, candidates: list[str]) -> dict:
    """Evidence for a TERMINAL self-repo task with a persisted candidate row."""
    return _baseline_evidence(
        root, base_commit,
        effect_state="quiescent",
        terminal_candidate_snapshot={
            "captured_at": "now", "baseline_hash": "hash-1",
            "surfaces": [{
                "surface_type": "system_repo",
                "canonical_root": str(root.resolve()),
                "candidates": list(candidates),
                "excluded_preexisting_dirty": [],
                "blockers": [],
            }],
        },
    )


def test_pathspec_magic_candidate_name_cannot_widen_the_projection(tmp_path):
    """A candidate literally named `:(top)` must stay a FILENAME, never magic.

    Candidate paths are task-attributed filenames, so a task can choose them. Left
    as ordinary pathspecs, `:(top)` widens `git diff -- <candidates>` to the whole
    repository and the owner's OWN unattributed edit lands in a patch labelled as
    the task's work — the single worst lie this screen could tell.
    """
    repo = tmp_path / "repo"
    base = _init_repo(repo)
    (repo / "owner_wip.py").write_text("owner = 'clean'\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "owner file")
    base = _git(repo, "rev-parse", "HEAD")
    # The owner's dirty edit: NOT attributed to the task.
    (repo / "owner_wip.py").write_text("owner = 'SECRET-NOT-ATTRIBUTED'\n", encoding="utf-8")
    (repo / ":(top)").write_text("x\n", encoding="utf-8")
    drive_root = tmp_path / "drive"
    write_task_result(
        drive_root, "magic", STATUS_COMPLETED,
        mutation_evidence=_terminal_evidence(repo, base, [":(top)"]),
    )
    with _client(drive_root, repo) as client:
        payload = client.get("/api/tasks/magic/diff").json()
    assert "SECRET-NOT-ATTRIBUTED" not in payload["patch"], payload["patch"]
    assert "owner_wip.py" not in payload["patch"]
    # The literal file the task did create is still projected honestly.
    assert ":(top)" in payload["patch"]
    assert payload["status"] == "ready"


def test_pathspec_exclude_magic_candidate_cannot_invert_the_projection(tmp_path):
    """The `:!x` form is the same hole from the other side (exclusion magic)."""
    repo = tmp_path / "repo"
    base = _init_repo(repo)
    (repo / "loop.py").write_text("a = 1\nb = 42\n", encoding="utf-8")
    drive_root = tmp_path / "drive"
    write_task_result(
        drive_root, "magic-exclude", STATUS_COMPLETED,
        mutation_evidence=_terminal_evidence(repo, base, [":!loop.py"]),
    )
    with _client(drive_root, repo) as client:
        payload = client.get("/api/tasks/magic-exclude/diff").json()
    # As a literal, `:!loop.py` matches nothing — so the honest answer is "no
    # attributed change", not "everything except loop.py".
    assert "b = 42" not in payload["patch"]
    assert payload["status"] == "empty"


def test_option_shaped_base_commit_never_reaches_git_argv(tmp_path):
    """`base_commit` sits BEFORE `--`, so an option-shaped value is a write hole.

    `git diff --output=<path>` writes the patch to a file of the caller's choosing
    and leaves stdout empty — the endpoint would then answer a reassuring "no
    changes" for a request that just wrote outside the repo. A baseline that is not
    a hex object name is refused before argv is built.
    """
    repo = tmp_path / "repo"
    _init_repo(repo)
    victim = tmp_path / "victim.txt"
    victim.write_text("PRECIOUS\n", encoding="utf-8")
    (repo / "loop.py").write_text("a = 1\nb = 9\n", encoding="utf-8")
    drive_root = tmp_path / "drive"
    write_task_result(
        drive_root, "argv", STATUS_COMPLETED,
        mutation_evidence=_terminal_evidence(repo, f"--output={victim}", ["loop.py"]),
    )
    with _client(drive_root, repo) as client:
        payload = client.get("/api/tasks/argv/diff").json()
    assert victim.read_text(encoding="utf-8") == "PRECIOUS\n", "argv injection landed"
    assert payload["status"] == "blocked"
    assert "base_commit_unknown" in payload["blockers"]
    assert payload["patch"] == ""


def test_non_hex_base_commit_is_refused_rather_than_resolved_as_a_ref(tmp_path):
    """Only hex object names are accepted; a ref expression is not a baseline."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "loop.py").write_text("a = 1\nb = 11\n", encoding="utf-8")
    drive_root = tmp_path / "drive"
    write_task_result(
        drive_root, "refname", STATUS_COMPLETED,
        mutation_evidence=_terminal_evidence(repo, "HEAD~1", ["loop.py"]),
    )
    with _client(drive_root, repo) as client:
        payload = client.get("/api/tasks/refname/diff").json()
    assert payload["status"] == "blocked"
    assert "base_commit_unknown" in payload["blockers"]


def test_non_utf8_patch_bytes_are_served_with_replacement_not_a_503(tmp_path):
    """A latin-1 byte in a tracked file must not fail an owner-facing read.

    git's stdout is bytes; decoding it as strict UTF-8 turned any repo holding
    legacy-encoded content into a 503 for the whole Changes screen.
    """
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "latin.txt").write_bytes(b"caf\xe9 one\ntwo\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "latin")
    base = _git(repo, "rev-parse", "HEAD")
    (repo / "latin.txt").write_bytes(b"caf\xe9 one\nTWO\n")
    drive_root = tmp_path / "drive"
    write_task_result(
        drive_root, "latin", STATUS_COMPLETED,
        mutation_evidence=_terminal_evidence(repo, base, ["latin.txt"]),
    )
    with _client(drive_root, repo) as client:
        response = client.get("/api/tasks/latin/diff")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert "latin.txt" in payload["patch"]
    assert "+TWO" in payload["patch"] and "-two" in payload["patch"]
    # The undecodable byte is a visible replacement character, not a crash.
    assert "�" in payload["patch"]
    # C7: the digest covers the SERVED text, so it still verifies after replacement.
    assert payload["patch_sha256"] == hashlib.sha256(
        payload["patch"].encode("utf-8"),
    ).hexdigest()


def test_untracked_file_needing_git_quoting_is_present_in_the_patch(tmp_path):
    """A new file named `héllo.txt` must reach the patch, not vanish silently.

    The newline-separated `ls-files` default hands back the C-quoted
    `"h\\303\\251llo.txt"`, which is not a path on disk, so the per-file
    `--no-index` diff produced nothing and the file disappeared with NO blocker.
    """
    repo = tmp_path / "repo"
    base = _init_repo(repo)
    (repo / "héllo.txt").write_text("fresh\n", encoding="utf-8")
    (repo / "plain new.txt").write_text("fresh\n", encoding="utf-8")
    drive_root = tmp_path / "drive"
    write_task_result(
        drive_root, "quoted", STATUS_COMPLETED,
        mutation_evidence=_terminal_evidence(repo, base, ["héllo.txt", "plain new.txt"]),
    )
    with _client(drive_root, repo) as client:
        payload = client.get("/api/tasks/quoted/diff").json()
    assert payload["status"] == "ready"
    assert "héllo.txt" in payload["patch"], payload["patch"]
    assert "plain new.txt" in payload["patch"]
    assert payload["blockers"] == []


def test_an_untracked_candidate_that_yields_no_diff_is_disclosed(tmp_path):
    """An omission is a blocker; "no output" must never read as "no change"."""
    repo = tmp_path / "repo"
    base = _init_repo(repo)
    (repo / "ghost.py").write_text("new = True\n", encoding="utf-8")
    drive_root = tmp_path / "drive"
    write_task_result(
        drive_root, "ghost", STATUS_COMPLETED,
        mutation_evidence=_terminal_evidence(repo, base, ["ghost.py"]),
    )
    real_capture = gateway_task_diff._git_capture

    def _blank_no_index(root, args):
        if "--no-index" in args:
            return 0, ""
        return real_capture(root, args)

    with _client(drive_root, repo) as client:
        gateway_task_diff._git_capture = _blank_no_index
        try:
            payload = client.get("/api/tasks/ghost/diff").json()
        finally:
            gateway_task_diff._git_capture = real_capture
    assert "untracked_patch_unavailable" in payload["blockers"]


def test_untracked_projection_beyond_the_cap_is_refused_not_truncated(tmp_path):
    """S3: a huge new-file set is refused as a whole, with a typed blocker.

    One git subprocess per untracked file is the cost model here, so the count is
    bounded. Truncating to the first N would render as a COMPLETE diff, so the
    untracked projection is dropped entirely and named instead.
    """
    repo = tmp_path / "repo"
    base = _init_repo(repo)
    (repo / "loop.py").write_text("a = 1\nb = 3\n", encoding="utf-8")
    names = [f"new_{i:04d}.py" for i in range(gateway_task_diff.DIFF_MAX_UNTRACKED_SECTIONS + 1)]
    for name in names:
        (repo / name).write_text("x = 1\n", encoding="utf-8")
    drive_root = tmp_path / "drive"
    write_task_result(
        drive_root, "flood", STATUS_COMPLETED,
        mutation_evidence=_terminal_evidence(repo, base, ["loop.py", *names]),
    )
    with _client(drive_root, repo) as client:
        payload = client.get("/api/tasks/flood/diff").json()
    assert "untracked_projection_capped" in payload["blockers"]
    # The TRACKED half is real and complete, so it is still served.
    assert "+b = 3" in payload["patch"]
    assert "new_0000.py" not in payload["patch"]


def test_patch_over_the_byte_cap_is_blocked_and_never_clipped(tmp_path):
    """S4: too large to serve is a disclosed refusal, not a shortened patch."""
    drive_root = tmp_path / "drive"
    body = "diff --git a/big.txt b/big.txt\n" + ("+padding line\n" * 700_000)
    assert len(body.encode("utf-8")) > gateway_task_diff.DIFF_MAX_PATCH_BYTES
    patch_path = _write_artifact(drive_root, "ws-big", "workspace.patch", body)
    write_task_result(
        drive_root, "ws-big", STATUS_COMPLETED,
        workspace_root=str(tmp_path / "ws"), artifact_status="ready_with_changes",
        artifacts=[{"name": "workspace.patch", "path": str(patch_path)}],
    )
    with _client(drive_root, tmp_path / "repo") as client:
        payload = client.get("/api/tasks/ws-big/diff").json()
    assert payload["status"] == "blocked"
    assert payload["blockers"] == ["patch_too_large"]
    assert payload["patch"] == ""
    assert payload["patch_sha256"] == ""


def test_workspace_excluded_paths_are_disclosed_as_a_blocker(tmp_path):
    """C4: the capture dropped paths, so the patch is not the whole story."""
    drive_root = tmp_path / "drive"
    patch_text = "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-old\n+new\n"
    patch_path = _write_artifact(drive_root, "ws-excl", "workspace.patch", patch_text)
    manifest_path = _write_artifact(drive_root, "ws-excl", "workspace_patch.json", json.dumps({
        "status": "ready_with_changes", "base_head": "aaa", "current_head": "aaa", "errors": [],
        "counts": {"tracked_excluded": 1, "untracked_excluded": 2},
    }))
    write_task_result(
        drive_root, "ws-excl", STATUS_COMPLETED,
        workspace_root=str(tmp_path / "ws"), artifact_status="ready_with_changes",
        artifacts=[
            {"name": "workspace.patch", "path": str(patch_path)},
            {"name": "workspace_patch.json", "path": str(manifest_path)},
        ],
    )
    with _client(drive_root, tmp_path / "repo") as client:
        payload = client.get("/api/tasks/ws-excl/diff").json()
    assert payload["status"] == "ready"
    assert payload["blockers"] == ["workspace_paths_excluded"]


def test_workspace_patch_digest_covers_the_served_text_after_replacement(tmp_path):
    """C7: one digest rule on BOTH sources — the bytes the owner received."""
    drive_root = tmp_path / "drive"
    raw = b"diff --git a/l.txt b/l.txt\n--- a/l.txt\n+++ b/l.txt\n@@ -1 +1 @@\n-caf\xe9\n+cafe\n"
    path = task_artifacts_dir(drive_root, "ws-latin") / "workspace.patch"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    write_task_result(
        drive_root, "ws-latin", STATUS_COMPLETED,
        workspace_root=str(tmp_path / "ws"), artifact_status="ready_with_changes",
        artifacts=[{"name": "workspace.patch", "path": str(path)}],
    )
    with _client(drive_root, tmp_path / "repo") as client:
        payload = client.get("/api/tasks/ws-latin/diff").json()
    assert payload["status"] == "ready"
    assert "�" in payload["patch"]
    assert payload["patch_sha256"] == hashlib.sha256(
        payload["patch"].encode("utf-8"),
    ).hexdigest()
    assert payload["patch_sha256"] != hashlib.sha256(raw).hexdigest()


def test_staging_a_candidate_changes_the_projection_fingerprint(tmp_path):
    """C6: `git add` moves work index-ward without touching HEAD or any mtime.

    That DOES change what `git diff <base> -- <path>` reports, so this worktree's
    own index stat belongs in the binding fingerprint — otherwise a read racing
    a stage/unstage is silently accepted as unchanged.
    """
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "loop.py").write_text("a = 1\nb = 21\n", encoding="utf-8")
    before = gateway_task_diff._projection_fingerprint(repo, ["loop.py"])
    _git(repo, "add", "loop.py")
    assert gateway_task_diff._projection_fingerprint(repo, ["loop.py"]) != before


def test_projection_fingerprint_tracks_a_linked_worktrees_own_index(tmp_path):
    """C6, worktree leg: in a LINKED worktree `.git` is a FILE, not a directory.

    `<root>/.git/index` therefore does not exist there, so a naive stat raised and
    the index half of the fingerprint collapsed to the same "absent" constant on
    every read — a guard that looked present while catching nothing. The path is
    resolved through git (`rev-parse --git-path index`) instead.
    """
    repo = tmp_path / "repo"
    _init_repo(repo)
    linked = tmp_path / "linked"
    _git(repo, "worktree", "add", "-q", str(linked), "-b", "probe")
    # The precondition the old code got wrong: `.git` here is a file.
    assert (linked / ".git").is_file()
    assert not (linked / ".git" / "index").exists()

    index = gateway_task_diff._git_index_path(linked)
    assert index is not None
    assert index.exists()
    assert index.resolve() != (linked / ".git" / "index")

    (linked / "loop.py").write_text("a = 1\nb = 21\n", encoding="utf-8")
    before = gateway_task_diff._projection_fingerprint(linked, ["loop.py"])
    _git(linked, "add", "loop.py")
    # The staging is invisible to HEAD and to the candidate's mtime; only the
    # worktree's own index moved, and the fingerprint must see it.
    assert gateway_task_diff._projection_fingerprint(linked, ["loop.py"]) != before


def test_projection_fingerprint_keeps_the_absent_marker_outside_a_repo(tmp_path):
    """An unresolvable index is the honest absent marker, never a crash."""
    outside = tmp_path / "not-a-repo"
    outside.mkdir()
    assert gateway_task_diff._git_index_path(outside) is None
    # Still a stable digest: a non-repo root answers a fingerprint, not an error.
    assert gateway_task_diff._projection_fingerprint(outside, ["loop.py"]) == \
        gateway_task_diff._projection_fingerprint(outside, ["loop.py"])


def test_git_invocations_pin_literal_pathspecs_and_readable_paths(tmp_path):
    """S1/C2 at the invocation level: the env and the `-c` policy are the fix."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    seen: dict = {}
    real_run = gateway_task_diff.subprocess.run

    def _record(argv, **kwargs):
        seen["argv"] = list(argv)
        seen["kwargs"] = dict(kwargs)
        return real_run(argv, **kwargs)

    gateway_task_diff.subprocess.run = _record
    try:
        rc, out = gateway_task_diff._git_capture(repo, ["rev-parse", "HEAD"])
    finally:
        gateway_task_diff.subprocess.run = real_run
    env = seen["kwargs"]["env"]
    assert env["GIT_LITERAL_PATHSPECS"] == "1"
    assert env["LC_ALL"] == "C"
    assert seen["argv"][:3] == ["git", "-c", "core.quotepath=off"]
    # C1: git returns BYTES here; this module owns the (replacing) decode, so no
    # `text=`/`encoding=` may be handed to subprocess.
    assert not seen["kwargs"].get("text")
    assert "encoding" not in seen["kwargs"] and "errors" not in seen["kwargs"]
    assert rc == 0 and isinstance(out, str)


def test_no_index_sections_use_the_literal_dev_null(tmp_path):
    """C8: git understands the literal "/dev/null", not the host's os.devnull."""
    repo = tmp_path / "repo"
    base = _init_repo(repo)
    (repo / "fresh.py").write_text("new = 1\n", encoding="utf-8")
    calls: list = []
    real_capture = gateway_task_diff._git_capture

    def _record(root, args):
        calls.append(list(args))
        return real_capture(root, args)

    gateway_task_diff._git_capture = _record
    try:
        patch, blockers = gateway_task_diff._build_projection_patch(repo, base, ["fresh.py"])
    finally:
        gateway_task_diff._git_capture = real_capture
    no_index = [args for args in calls if "--no-index" in args]
    assert no_index and no_index[0][-2:] == ["/dev/null", "fresh.py"]
    assert blockers == []
    assert "+new = 1" in patch


def test_concurrent_diff_reads_are_gated_to_the_declared_slot_count(tmp_path):
    """S3: the process-wide gate QUEUES concurrent reads instead of fanning out."""
    import asyncio

    async def _exercise():
        gate = gateway_task_diff.diff_gate()
        assert gate is gateway_task_diff.diff_gate(), "the gate must be process-wide"
        live = 0
        peak = 0

        async def _worker():
            nonlocal live, peak
            async with gate:
                live += 1
                peak = max(peak, live)
                await asyncio.sleep(0.01)
                live -= 1

        await asyncio.gather(*[_worker() for _ in range(8)])
        return peak

    peak = asyncio.run(_exercise())
    assert peak == gateway_task_diff.DIFF_WORKER_SLOTS
