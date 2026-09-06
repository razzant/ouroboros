"""W2-F3 pins: ``/api/state`` serves the runtime repo identity honestly.

On a source-mode install nothing ever stamps ``current_branch``/``current_sha``
into state.json, so the endpoint answered ``sha: ""`` / ``branch: null`` while
the process demonstrably ran from a real git checkout. The fix reads the ACTUAL
checkout (read-only, stdlib file reads — worktree pointers, loose and packed
refs, detached HEAD) as the fallback; supervisor-stamped state values keep
winning, and nothing is invented when both sides are silent.
"""

from __future__ import annotations

import asyncio
import json
import types

import pytest
from starlette.requests import Request

from ouroboros.gateway.state import _git_checkout_identity, _runtime_repo_identity


SHA = "1234567890abcdef1234567890abcdef12345678"
OTHER = "fedcba0987654321fedcba0987654321fedcba09"


def _plain_checkout(root, *, branch="ouroboros", sha=SHA, packed=False):
    gitdir = root / ".git"
    gitdir.mkdir(parents=True)
    (gitdir / "HEAD").write_text(f"ref: refs/heads/{branch}\n", encoding="utf-8")
    if packed:
        (gitdir / "packed-refs").write_text(
            "# pack-refs with: peeled fully-peeled sorted \n"
            f"{OTHER} refs/heads/unrelated\n"
            f"{sha} refs/heads/{branch}\n"
            f"^{OTHER}\n",
            encoding="utf-8",
        )
    else:
        ref = gitdir / "refs" / "heads" / branch
        ref.parent.mkdir(parents=True)
        ref.write_text(sha + "\n", encoding="utf-8")
    return root


def test_reads_a_plain_checkout_with_a_loose_ref(tmp_path):
    repo = _plain_checkout(tmp_path / "repo")
    assert _git_checkout_identity(repo) == ("ouroboros", SHA)


def test_reads_a_packed_ref_and_skips_peeled_lines(tmp_path):
    repo = _plain_checkout(tmp_path / "repo", packed=True)
    assert _git_checkout_identity(repo) == ("ouroboros", SHA)


def test_reads_a_linked_worktree_through_its_pointer_and_commondir(tmp_path):
    main = _plain_checkout(tmp_path / "main", branch="feature/lane")
    worktree_gitdir = tmp_path / "main" / ".git" / "worktrees" / "wt"
    worktree_gitdir.mkdir(parents=True)
    (worktree_gitdir / "HEAD").write_text("ref: refs/heads/feature/lane\n", encoding="utf-8")
    (worktree_gitdir / "commondir").write_text("../..\n", encoding="utf-8")
    worktree = tmp_path / "wt"
    worktree.mkdir()
    (worktree / ".git").write_text(f"gitdir: {worktree_gitdir}\n", encoding="utf-8")
    # The main checkout must not shadow the branch ref lookup.
    assert main.exists()
    assert _git_checkout_identity(worktree) == ("feature/lane", SHA)


def test_detached_head_has_a_sha_and_honestly_no_branch(tmp_path):
    gitdir = tmp_path / "repo" / ".git"
    gitdir.mkdir(parents=True)
    (gitdir / "HEAD").write_text(SHA + "\n", encoding="utf-8")
    assert _git_checkout_identity(tmp_path / "repo") == (None, SHA)


def test_unreadable_layouts_degrade_to_unknown_not_invented(tmp_path):
    assert _git_checkout_identity(tmp_path / "nowhere") == (None, "")
    repo = tmp_path / "notgit"
    repo.mkdir()
    (repo / ".git").write_text("not a gitdir pointer\n", encoding="utf-8")
    assert _git_checkout_identity(repo) == (None, "")


def test_supervisor_stamped_state_values_win_over_the_checkout(tmp_path, monkeypatch):
    import ouroboros.config as config

    monkeypatch.setattr(config, "REPO_DIR", _plain_checkout(tmp_path / "repo"))
    st = {"current_branch": "managed-branch", "current_sha": OTHER}
    assert _runtime_repo_identity(st) == ("managed-branch", OTHER)


def test_source_mode_state_gap_is_filled_from_the_checkout(tmp_path, monkeypatch):
    import ouroboros.config as config

    monkeypatch.setattr(config, "REPO_DIR", _plain_checkout(tmp_path / "repo"))
    # Exactly the source-mode shape: keys seeded by load_state, values None.
    assert _runtime_repo_identity({"current_branch": None, "current_sha": None}) == (
        "ouroboros", SHA,
    )


@pytest.mark.parametrize("stamped", [False, True])
def test_api_state_serves_the_honest_identity(tmp_path, monkeypatch, stamped):
    from ouroboros.gateway.state import api_state
    import ouroboros.config as config
    from ouroboros import usage_accounting as ua
    from supervisor import queue, state, workers

    root = tmp_path / "data"
    (root / "state").mkdir(parents=True)
    (root / "logs").mkdir(parents=True)
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(root / "settings.json"))
    ua.ensure_legacy_imported(root)
    monkeypatch.setattr(config, "REPO_DIR", _plain_checkout(tmp_path / "repo"))
    monkeypatch.setattr(state, "TOTAL_BUDGET_LIMIT", 0.0)
    monkeypatch.setattr(state, "load_state", lambda: (
        {"current_branch": "managed-branch", "current_sha": OTHER}
        if stamped else {"current_branch": None, "current_sha": None}
    ))
    monkeypatch.setattr(workers, "WORKERS", {})
    monkeypatch.setattr(workers, "PENDING", [])
    monkeypatch.setattr(workers, "RUNNING", {})
    monkeypatch.setattr(queue, "get_evolution_status_snapshot", lambda **_kwargs: {})
    request = Request({
        "type": "http", "method": "GET", "path": "/api/state", "headers": [],
        "query_string": b"", "scheme": "http", "server": ("test", 80),
        "client": ("test", 1),
        "app": types.SimpleNamespace(
            state=types.SimpleNamespace(drive_root=root, app_start=0.0),
        ),
    })

    response = asyncio.run(api_state(request))
    payload = json.loads(response.body)
    assert response.status_code == 200
    if stamped:
        assert payload["branch"] == "managed-branch"
        assert payload["sha"] == OTHER[:8]
    else:
        assert payload["branch"] == "ouroboros"
        assert payload["sha"] == SHA[:8]
