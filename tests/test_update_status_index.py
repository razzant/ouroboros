"""A progress-driven status read must not compete with the update's index writer."""

import os
import subprocess



def test_passive_update_status_does_not_refresh_the_git_index(tmp_path, monkeypatch):
    from supervisor import git_ops
    from supervisor.git_ops_updates import compute_managed_update_status

    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)

    git("init", "-b", "ouroboros")
    tracked = repo / "tracked.txt"
    tracked.write_text("unchanged content\n", encoding="utf-8")
    git("add", "tracked.txt")
    git("-c", "user.name=Update test", "-c", "user.email=update@test.invalid", "commit", "-m", "fixture")
    # The content stays identical but the cached stat data needs refreshing.
    # Ordinary `git status` writes that refresh under index.lock, which can
    # collide with the concurrent update's checkout/reset/stash commands.
    info = tracked.stat()
    os.utime(tracked, ns=(info.st_atime_ns, info.st_mtime_ns - 10_000_000_000))
    index = repo / ".git" / "index"
    before = index.read_bytes()
    monkeypatch.delenv("GIT_OPTIONAL_LOCKS", raising=False)
    monkeypatch.setattr(git_ops, "REPO_DIR", repo)

    status = compute_managed_update_status(fetch=False)

    assert status["dirty"] is False
    assert index.read_bytes() == before, "passive status acquired the update writer's optional index lock"
