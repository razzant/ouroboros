"""The public stats tool writes task files, never a remote-only code commit."""
from __future__ import annotations

import hashlib
import json
import subprocess
from types import SimpleNamespace

import pytest

from ouroboros.artifacts import task_artifact_dir_path
from ouroboros.tools import evolution_stats
from ouroboros.tools.registry import ToolContext, ToolRegistry


def _git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args]).decode().strip()


def _context(tmp_path, monkeypatch):
    repo = tmp_path / "source"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "ouroboros")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "user.email", "test@example.invalid")
    (repo / "sample.py").write_bytes(b"first\nsecond\n")
    (repo / "BIBLE.md").write_text("Purpose café\n", encoding="utf-8", newline="\n")
    (repo / "SYSTEM.md").write_text("Identity\n", encoding="utf-8", newline="\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "v1.0.0: initial")
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", "-q", str(remote)], check=True)
    _git(repo, "remote", "add", "origin", str(remote))
    _git(repo, "push", "-q", "origin", "ouroboros")
    monkeypatch.setattr(evolution_stats, "resolve_repo_dir", lambda: repo)
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    ctx = ToolContext(repo_dir=repo, drive_root=tmp_path / "data", task_id="stats-export")
    reg = ToolRegistry(repo_dir=repo, drive_root=ctx.drive_root)
    reg.set_context(ctx)
    return repo, remote, ctx, reg


def test_named_tool_exports_readable_immutable_files_without_github(tmp_path, monkeypatch):
    import requests
    repo, remote, ctx, reg = _context(tmp_path, monkeypatch)

    # Git emits UTF-8 even when Windows' default text pipe codec is CP1252.
    git_run = evolution_stats.subprocess.run

    def windows_default(*args, **kwargs):
        kwargs.setdefault("encoding", "cp1252")
        return git_run(*args, **kwargs)

    monkeypatch.setattr(evolution_stats, "subprocess", SimpleNamespace(run=windows_default))

    def no_network(*a, **k):
        raise AssertionError("local stats must not publish or fetch over HTTP")

    monkeypatch.setattr(requests.sessions.Session, "request", no_network)
    before = (_git(repo, "rev-parse", "HEAD"), _git(repo, "status", "--porcelain"),
              _git(remote, "show-ref"))
    times = iter(["2026-10-21T01:00:00Z", "2026-10-21T01:00:00Z", "2026-10-21T01:00:01Z"])
    clock = evolution_stats.time
    monkeypatch.setattr(evolution_stats, "time", SimpleNamespace(
        time=clock.time, gmtime=clock.gmtime, strftime=lambda *a: next(times)))
    assert reg.get_schema_by_name("generate_evolution_stats") is not None
    outputs = [reg.execute_result("generate_evolution_stats", {}) for _ in range(3)]
    assert all(result.code == "OK" for result in outputs), [r.text for r in outputs]
    payloads = [json.loads(result.text) for result in outputs]
    assert all(p["publication"] == "local_only" for p in payloads)
    refs = [p["artifact"] for p in payloads]
    assert refs[0] == refs[1] and refs[0]["path"] != refs[2]["path"]
    directory = task_artifact_dir_path(ctx.drive_root, ctx.task_id)
    delivered = []
    ctx.current_chat_id = 11
    monkeypatch.setattr("ouroboros.tools.owner_delivery.deliver_owner_event",
                        lambda _ctx, event: delivered.append(event) or "live")
    for ref, payload in zip(refs, payloads):
        content = (directory / ref["path"]).read_bytes()
        assert len(content) == ref["bytes"]
        assert hashlib.sha256(content).hexdigest() == ref["sha256"]
        data = json.loads(content)
        assert data["total_commits_sampled"] == 1
        assert data["max_py_lines"] == 2
        assert data["max_bible_bytes"] == len("Purpose café\n".encode())
        assert data["max_system_bytes"] == 9
        assert data["points"][0]["version"] == "1.0.0"
        read = reg.execute_result("read_file", {"root": "artifact_store", "path": ref["path"]})
        assert read.code == "OK" and '"py_lines": 2' in read.text
        sent = reg.execute_result("send_file", {"file_path": payload["file_path"]})
        assert sent.code == "OK", sent.text
        assert delivered[-1]["type"] == "send_document"
        assert delivered[-1]["file_ref"]["sha256"] == ref["sha256"]
    assert before == (_git(repo, "rev-parse", "HEAD"), _git(repo, "status", "--porcelain"),
                      _git(remote, "show-ref"))
    assert not (repo / "docs" / "evolution.json").exists()


def test_missing_task_is_refused_before_collecting(monkeypatch):
    monkeypatch.setattr(evolution_stats, "_collect_data", lambda: pytest.fail("must not collect"))
    assert "EVOLUTION_EXPORT_UNAVAILABLE" in evolution_stats.generate_evolution_stats()
