"""Where a publication goes and what lists it: origin resolution, the v2 hub entry, git destinations."""

from __future__ import annotations

import json
import types

import pytest

from ouroboros.skill_install_record import write_install_record
from ouroboros.skill_publish_destination import (
    PublishListingError,
    captured_release_digest,
    hub_entry_listing,
    resolve_publish_destination,
    task_destination,
)
from ouroboros.skill_publish_result import SkillPublishDestinationError
from ouroboros.tools import skill_publish
from tests.test_skill_publish_transaction import BASE_SHA, COMMIT_SHA, _install_transaction_fakes, _snapshot, _submit

ROWS = [
    {"id": "builtin", "name": "Built in", "kind": "builtin", "trust": "trusted", "confirmed": True},
    {"id": "ouroboros", "name": "Ouroboros", "kind": "hub", "url": "https://github.com/razzant/OuroborosHub",
     "ref": "main", "part": "official", "trust": "trusted", "confirmed": True},
    {"id": "ouroboros-community", "name": "Ouroboros Community", "kind": "hub",
     "url": "https://github.com/razzant/OuroborosHub", "ref": "main", "part": "community", "trust": "review",
     "confirmed": True},
    {"id": "mine", "name": "My skills", "kind": "git", "url": "https://github.com/alice/skills", "ref": "main",
     "path": "packs", "trust": "trusted", "confirmed": True},
    {"id": "elsewhere", "name": "Elsewhere", "kind": "git", "url": "ssh://git.example.invalid/x", "trust": "review",
     "confirmed": True},
]


def _record(drive, name, source_id, kind, release):
    write_install_record(drive, name, {"source": {"id": source_id, "kind": kind, "name": source_id},
                                       "release": {"version": "1.0.0", "digest": "0" * 64, **release},
                                       "basis": "review"})


def test_the_origin_is_where_the_installed_release_came_from(tmp_path):
    _record(tmp_path, "hosted", "ouroboros-community", "hub", {"commit": "c" * 40, "path": "skills/hosted"})
    _record(tmp_path, "pointed", "ouroboros-community", "hub",
            {"repo": "https://github.com/author/scholar", "path": "pkg/scholar", "commit": "d" * 40})
    _record(tmp_path, "packed", "mine", "git", {"commit": "e" * 40, "path": "group/packed"})
    hosted = resolve_publish_destination(tmp_path, "hosted", rows=ROWS)
    assert (hosted.kind, hosted.repository, hosted.skill_folder("hosted")) == ("hub", "razzant/OuroborosHub",
                                                                                 "skills/hosted")
    pointed = resolve_publish_destination(tmp_path, "pointed", rows=ROWS)
    assert (pointed.kind, pointed.pointer, pointed.repository) == ("git", True, "author/scholar")
    assert pointed.skill_folder("pointed") == "pkg/scholar"
    packed = resolve_publish_destination(tmp_path, "packed", {"source_id": "mine", "mode": "origin"}, rows=ROWS)
    assert (packed.repository, packed.skill_folder("packed")) == ("alice/skills", "group/packed")
    made_here = resolve_publish_destination(tmp_path, "made_here", rows=ROWS)
    assert (made_here.kind, made_here.source_id) == ("hub", "ouroboros")


def test_the_owner_names_a_source_or_a_new_hub_entry(tmp_path):
    own = resolve_publish_destination(tmp_path, "fresh", {"source_id": "mine", "mode": "entry"}, rows=ROWS)
    assert (own.kind, own.branch, own.skill_folder("fresh")) == ("git", "main", "packs/fresh")
    assert resolve_publish_destination(tmp_path, "fresh", "mine", rows=ROWS).skill_folder("fresh") == "packs/fresh"
    _record(tmp_path, "packed", "mine", "git", {"path": "group/packed"})
    # The same source the release came from keeps its exact folder.
    assert resolve_publish_destination(tmp_path, "packed", "mine", rows=ROWS).skill_folder("packed") == "group/packed"
    community = resolve_publish_destination(tmp_path, "fresh", {"source_id": "ouroboros-community",
                                                                "mode": "entry"}, rows=ROWS)
    assert community.kind == "hub" and not community.new_entry
    beside = resolve_publish_destination(tmp_path, "fresh", {"source_id": "ouroboros-community",
                                                             "mode": "new_entry"}, rows=ROWS)
    assert beside.kind == "hub" and beside.new_entry
    assert resolve_publish_destination(tmp_path, "fresh", "hub", rows=ROWS).source_id == "ouroboros"
    for requested, code in (({"source_id": "nope"}, "destination_unknown"),
                            ({"source_id": "mine", "mode": "new_entry"}, "destination_not_hub"),
                            ({"source_id": "mine", "mode": "push"}, "destination_unknown"),
                            ("elsewhere", "hub_destination_invalid"), ("builtin", "destination_not_git")):
        with pytest.raises(SkillPublishDestinationError) as refused:
            resolve_publish_destination(tmp_path, "fresh", requested, rows=ROWS)
        assert refused.value.reason_code == code


def test_the_publish_task_carries_the_owner_choice():
    chosen = {"source_id": "mine", "mode": "entry"}
    ctx = types.SimpleNamespace(task_metadata={"skill_publish_target": {"skill": "x", "repository": "a/b",
                                                                        "destination": chosen}})
    assert task_destination(ctx) == chosen
    assert task_destination(types.SimpleNamespace(task_metadata={})) is None
    assert task_destination(types.SimpleNamespace()) is None


def _reader(files):
    return lambda path: files.get(path)


_CATEGORIES = json.dumps({"schema_version": 1, "categories": [{"id": "city", "label": "City"},
                                                              {"id": "writing", "label": "Writing"}]}).encode()


def test_a_new_hub_entry_is_canonical_and_bound_to_the_published_files():
    snapshot = _snapshot()
    files, previous = hub_entry_listing(_reader({"categories.json": _CATEGORIES}), snapshot, skill="demo",
                                        login="alice", category=["writing"], new_entry=False)
    raw = files["entries/demo.json"]
    entry = json.loads(raw)
    assert previous is None and raw == (json.dumps(entry, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    assert entry["digest"] == captured_release_digest(snapshot) and len(entry["digest"]) == 64
    assert entry["owner"] == "alice" and entry["category"] == ["writing"] and entry["withdrawn"] is None
    assert entry["source"] == {"kind": "hosted", "path": "skills/demo"} and entry["schema_version"] == 2
    assert entry["type"] == "instruction" and entry["when_to_use"] == "Use for a test."


def test_an_existing_hub_entry_keeps_its_owner_and_refusals_are_typed():
    existing = {"schema_version": 2, "name": "demo", "description": "old", "category": ["city"], "owner": "bob",
                "version": "0.9.0", "source": {"kind": "hosted", "path": "skills/demo"}, "digest": "1" * 64,
                "withdrawn": None, "author": "Bob"}
    read = _reader({"categories.json": _CATEGORIES, "entries/demo.json": json.dumps(existing).encode()})
    files, previous = hub_entry_listing(read, _snapshot(), skill="demo", login="alice", category=None,
                                        new_entry=False)
    entry = json.loads(files["entries/demo.json"])
    assert previous == "0.9.0" and entry["owner"] == "bob" and entry["category"] == ["city"]
    assert entry["author"] == "Bob" and entry["version"] == "1.0.0"
    cases = [
        (read, {"new_entry": True}, "entry_exists"),
        (_reader({"categories.json": _CATEGORIES, "entries/demo.json": json.dumps(
            {**existing, "version": "1.0.0"}).encode()}), {}, "catalog_version_exists"),
        (_reader({"categories.json": _CATEGORIES, "entries/demo.json": json.dumps(
            {**existing, "source": {"kind": "git", "repo": "https://github.com/b/r", "path": "", "commit": "f" * 40}}
        ).encode()}), {}, "pointer_entry"),
        (_reader({"categories.json": _CATEGORIES}), {"category": ["games"]}, "category_unknown"),
        (_reader({"categories.json": _CATEGORIES}), {"category": None}, "category_required"),
        (_reader({"categories.json": b"not json"}), {}, "upstream_catalog_invalid"),
    ]
    for reader, overrides, code in cases:
        kwargs = {"skill": "demo", "login": "alice", "category": ["city"], "new_entry": False, **overrides}
        with pytest.raises(PublishListingError) as refused:
            hub_entry_listing(reader, _snapshot(), **kwargs)
        assert refused.value.reason_code == code


def _destination(monkeypatch, destination):
    monkeypatch.setattr(skill_publish, "resolve_publish_destination", lambda *_a, **_k: destination)


def test_a_v2_hub_receives_one_entry_and_its_files_without_the_generated_catalog(monkeypatch, tmp_path):
    from ouroboros.skill_publish_destination import PublishDestination

    ctx, events, captured = _install_transaction_fakes(monkeypatch, tmp_path, snapshot=_snapshot())
    upstream = {"hub.json": b'{"schema_version": 2}', "categories.json": _CATEGORIES}
    monkeypatch.setattr(skill_publish, "fetch_upstream_file", lambda _c, _o, _r, sha, path: (
        upstream.get(path) if sha == BASE_SHA else None))
    monkeypatch.setattr(skill_publish, "fetch_tree_paths", lambda *_a: ["SKILL.md", "old.py"])
    monkeypatch.setattr(skill_publish, "fetch_upstream_catalog",
                        lambda *_a: pytest.fail("a v2 hub regenerates its own v1 catalog"))
    _destination(monkeypatch, PublishDestination("hub", "hub", "project", "main", "skills", "ouroboros", "Ouroboros"))
    result = _submit(ctx, category=["city"])
    assert result["ok"] is True, result
    committed = {row["path"] for row in captured["additions"]}
    assert committed == {"skills/demo/skill.json", "skills/demo/SKILL.md", "entries/demo.json"}
    assert captured["deletions"] == [{"path": "skills/demo/old.py"}]
    assert (result["files_published"], result["listed"]) == ("pull_request", "when_merged")
    assert result["release_digest"] == captured_release_digest(_snapshot())
    assert result["destination_kind"] == "hub"


@pytest.mark.parametrize("can_push", [False, True])
def test_a_git_destination_merges_at_once_only_when_this_account_may_push(monkeypatch, tmp_path, can_push):
    from ouroboros.skill_publish_destination import PublishDestination

    ctx, _events, captured = _install_transaction_fakes(monkeypatch, tmp_path, snapshot=_snapshot())
    _destination(monkeypatch, PublishDestination("git", "hub", "project", "", "packs", "mine", "My skills"))
    monkeypatch.setattr(skill_publish, "fetch_repository", lambda *_a: {"default_branch": "trunk", "can_push": can_push})
    branches = []
    monkeypatch.setattr(skill_publish, "fetch_branch_head",
                        lambda _c, _o, _r, branch: branches.append(branch) or (BASE_SHA, branch))
    monkeypatch.setattr(skill_publish, "fetch_tree_paths", lambda *_a: ["SKILL.md", "stale.txt"])
    merges = []
    monkeypatch.setattr(skill_publish, "merge_pull_request",
                        lambda _c, owner, repo, number, sha: merges.append((owner, repo, number, sha)) or {
                            "merged": True, "commit": "9" * 40})
    result = _submit(ctx)
    assert result["ok"] is True, result
    assert branches == ["trunk"]  # the repository's default branch
    assert {row["path"] for row in captured["additions"]} == {"packs/demo/skill.json", "packs/demo/SKILL.md"}
    assert captured["deletions"] == [{"path": "packs/demo/stale.txt"}]
    if can_push:
        assert merges == [("hub", "project", 7, COMMIT_SHA)]
        assert (result["files_published"], result["listed"], result["merged_commit"]) == ("merged", "listed", "9" * 40)
        assert "merged at once" in captured["pr_body"]
    else:
        assert merges == [] and (result["files_published"], result["listed"]) == ("pull_request", "when_merged")
        assert "The pull request is the only requested public effect." in captured["pr_body"]


def test_a_refused_merge_keeps_the_open_request_as_the_result(monkeypatch, tmp_path):
    from ouroboros.skill_publish_destination import PublishDestination

    ctx, _events, _captured = _install_transaction_fakes(monkeypatch, tmp_path, snapshot=_snapshot())
    _destination(monkeypatch, PublishDestination("git", "hub", "project", "main", "", "mine", "My skills",
                                                 skill_path="demo"))
    monkeypatch.setattr(skill_publish, "fetch_repository", lambda *_a: {"default_branch": "main", "can_push": True})
    monkeypatch.setattr(skill_publish, "merge_pull_request", lambda *_a: {"merged": False, "error": "protected"})
    result = _submit(ctx)
    assert result["ok"] is True and result["receipt"]["number"] == 7
    assert (result["files_published"], result["listed"], result["merge_error"]) == ("pull_request", "when_merged",
                                                                                    "protected")


def test_a_pointer_origin_publishes_files_and_names_the_entry_update(monkeypatch, tmp_path):
    from ouroboros.skill_publish_destination import PublishDestination

    ctx, _events, captured = _install_transaction_fakes(monkeypatch, tmp_path, snapshot=_snapshot())
    _destination(monkeypatch, PublishDestination("git", "hub", "project", "", "", "ouroboros-community",
                                                 "hub/project", pointer=True, skill_path=""))
    monkeypatch.setattr(skill_publish, "fetch_tree_paths", lambda *_a: pytest.fail("a root folder never deletes"))
    result = _submit(ctx)
    assert result["ok"] is True
    assert {row["path"] for row in captured["additions"]} == {"skill.json", "SKILL.md"} and captured["deletions"] == []
    assert result["listed"] == "needs_entry_update" and result["entry_repo"] == "https://github.com/hub/project"


def test_the_task_destination_reaches_the_resolver(monkeypatch, tmp_path):
    from ouroboros.skill_publish_destination import PublishDestination

    ctx, _events, _captured = _install_transaction_fakes(monkeypatch, tmp_path, snapshot=_snapshot())
    chosen = {"source_id": "ouroboros-community", "mode": "new_entry"}
    ctx.task_metadata = {"skill_publish_target": {"skill": "demo", "repository": "hub/project", "destination": chosen}}
    seen = []
    monkeypatch.setattr(skill_publish, "resolve_publish_destination", lambda _d, _s, requested: seen.append(
        requested) or PublishDestination("hub", "hub", "project", "main", "skills", "ouroboros", "Ouroboros"))
    assert _submit(ctx)["ok"] is True and seen == [chosen]
    assert _submit(ctx, destination="mine")["ok"] is True and seen[-1] == "mine"  # an explicit argument wins


def test_projected_attempt_metadata_keeps_both_results():
    from ouroboros.skill_publish_result import extract_skill_publish_result_metadata

    envelope = json.loads(skill_publish._PublishAttempt(skill="demo").result(
        ok=False, status="blocked", reason_code="x", extra_fields={"files_published": "merged", "listed": "listed",
                                                                   "destination_kind": "git"}))
    attempt = extract_skill_publish_result_metadata(json.dumps(envelope))["skill_publish_attempt"]
    assert (attempt["files_published"], attempt["listed"], attempt["destination_kind"]) == ("merged", "listed", "git")


def test_merge_requests_the_exact_head_and_reports_refusals(monkeypatch):
    from ouroboros import skill_publish_github as github
    from ouroboros.skill_publish_client import GitHubError

    calls = []

    def fake(method, path, **kwargs):
        calls.append((method, path, kwargs))
        if "/7/" in path:
            return {"merged": True, "sha": "A" * 40}
        raise GitHubError("Base branch was modified", http_status=405)

    monkeypatch.setattr(github, "request", fake)
    assert github.merge_pull_request(types.SimpleNamespace(), "o", "r", 7, "b" * 40) == {
        "merged": True, "commit": "a" * 40}
    assert calls[0] == ("PUT", "/repos/o/r/pulls/7/merge",
                        {"body": {"sha": "b" * 40, "merge_method": "squash"}, "timeout": 60})
    refused = github.merge_pull_request(types.SimpleNamespace(), "o", "r", 8, "b" * 40)
    assert refused["merged"] is False and refused["http_status"] == 405 and "405" in refused["error"]


@pytest.mark.parametrize("requested", [None, "mine", {"source_id": "mine", "mode": "origin"}])
def test_repository_root_origin_is_distinct_from_an_unknown_folder(tmp_path, requested):
    _record(tmp_path, "root_skill", "mine", "git", {"path": ""})
    destination = resolve_publish_destination(tmp_path, "root_skill", requested, rows=ROWS)
    assert destination.skill_folder("root_skill") == ""
    _record(tmp_path, "unknown", "mine", "git", {})
    assert resolve_publish_destination(tmp_path, "unknown", requested, rows=ROWS).skill_folder("unknown") == "packs/unknown"


@pytest.mark.parametrize("url", [
    "https://github.com/owner/repo", "https://github.com/owner/repo.git",
    "git@github.com:owner/repo", "git@github.com:owner/repo.git",
    "ssh://git@github.com/owner/repo", "ssh://git@github.com/owner/repo.git",
])
def test_github_publication_accepts_https_and_ssh(url):
    from ouroboros.skill_publish_destination import github_repository

    assert github_repository(url) == ("owner", "repo")


@pytest.mark.parametrize("url", ["https://gitlab.com/owner/repo", "git@gitlab.com:owner/repo",
                                  "ssh://git@gitlab.com/owner/repo"])
def test_publication_to_other_hosts_explains_the_remaining_limit(url):
    from ouroboros.skill_publish_destination import github_repository

    with pytest.raises(SkillPublishDestinationError, match="Publishing currently supports GitHub repositories only") as err:
        github_repository(url)
    assert err.value.reason_code == "hub_destination_invalid"
