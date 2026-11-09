"""Where a publication goes and what lists it: origin resolution, the v2 hub entry, git destinations."""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
import types
import urllib.parse

import pytest

from ouroboros.artifacts import task_artifact_dir_path
from ouroboros.config import SKILL_SOURCE_OUROBOROSHUB
from ouroboros.skill_install_record import write_install_record
from ouroboros.skill_loader import SkillReviewState, load_skill
from ouroboros.skill_publish_client import GitHubError
from ouroboros.skill_publish_destination import (
    PublishListingError,
    captured_release_digest,
    entry_bytes,
    git_blob_oid,
    hub_entry_listing,
    resolve_publish_destination,
    task_destination,
)
from ouroboros.skill_publish_result import SkillPublishDestinationError
from ouroboros.tools import skill_publish
from tests.test_skill_publish_transaction import (
    BASE_SHA,
    COMMIT_SHA,
    _install_transaction_fakes,
    _installed_from_destination,
    _snapshot,
    _submit,
)

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
    monkeypatch.setattr(skill_publish, "fetch_tree_blobs", lambda *_a, **_k: {"SKILL.md": "a" * 40, "old.py": "b" * 40})
    _installed_from_destination(tmp_path)
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
    assert result["destination_comparison"]["state"] == "clear"
    assert {(row["path"], row["upstream"]) for row in result["destination_comparison"]["paths"]} >= {
        ("skills/demo/SKILL.md", "unchanged_since_base"), ("skills/demo/old.py", "unchanged_since_base")}


@pytest.mark.parametrize("can_push", [False, True])
def test_a_git_destination_merges_at_once_only_when_this_account_may_push(monkeypatch, tmp_path, can_push):
    from ouroboros.skill_publish_destination import PublishDestination

    ctx, _events, captured = _install_transaction_fakes(monkeypatch, tmp_path, snapshot=_snapshot())
    _destination(monkeypatch, PublishDestination("git", "hub", "project", "", "packs", "mine", "My skills"))
    monkeypatch.setattr(skill_publish, "fetch_repository", lambda *_a: {"default_branch": "trunk", "can_push": can_push})
    branches = []
    monkeypatch.setattr(skill_publish, "fetch_branch_head",
                        lambda _c, _o, _r, branch: branches.append(branch) or (BASE_SHA, branch))
    monkeypatch.setattr(skill_publish, "fetch_tree_blobs", lambda *_a, **_k: {"SKILL.md": "a" * 40, "stale.txt": "b" * 40})
    _installed_from_destination(tmp_path, folder="packs/demo")
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
    monkeypatch.setattr(skill_publish, "fetch_tree_blobs", lambda *_a, **_k: {"README.md": "a" * 40, "src/x.py": "b" * 40})
    result = _submit(ctx)
    assert result["ok"] is True
    assert {row["path"] for row in captured["additions"]} == {"skill.json", "SKILL.md"} and captured["deletions"] == []
    assert {row["path"] for row in result["destination_comparison"]["paths"]} == {"skill.json", "SKILL.md"}
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
    "https://github.com/owner/repo", "git@github.com:owner/repo", "ssh://git@github.com/owner/repo.git",
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


def test_a_hub_inside_a_repository_folder_is_never_a_publication_target(tmp_path):
    folder_hub = {"id": "team-hub", "name": "Team hub", "kind": "hub", "url": "https://github.com/team/tools",
                  "path": "catalog", "trust": "review", "confirmed": True}
    assert resolve_publish_destination(tmp_path, "fresh", rows=[folder_hub, *ROWS]).source_id == "ouroboros"
    # Without an Official row the first hub is the default: never the one inside a folder.
    assert resolve_publish_destination(tmp_path, "fresh", rows=[folder_hub, ROWS[2]]).source_id == "ouroboros-community"
    with pytest.raises(SkillPublishDestinationError) as refused:
        resolve_publish_destination(tmp_path, "fresh", "team-hub", rows=[folder_hub, *ROWS])
    assert refused.value.reason_code == "hub_folder_unsupported" and "catalog" in str(refused.value)
    # A record names its view: the row it came from is found by id first, never another view of the repository.
    other_view = {**ROWS[3], "id": "mine-next", "ref": "next"}
    write_install_record(tmp_path, "packed", {"source": {"id": "mine", "kind": "git", "name": "My skills",
                                                         "url": ROWS[3]["url"], "ref": "main", "commit": "",
                                                         "path": "packs"},
                                              "release": {"version": "1", "digest": "0" * 64, "path": "packs/packed"},
                                              "basis": "review"})
    origin = resolve_publish_destination(tmp_path, "packed", rows=[other_view, *ROWS])
    assert (origin.source_id, origin.branch, origin.skill_folder("packed")) == ("mine", "main", "packs/packed")


def test_a_name_added_again_for_another_address_is_not_the_origin_of_an_older_install(tmp_path):
    # "mine" was removed and added again for alice/skills; this release came from alice/old-skills.
    write_install_record(tmp_path, "packed", {"source": {"id": "mine", "kind": "git", "name": "My skills",
                                                         "url": "https://github.com/alice/old-skills", "ref": "main",
                                                         "commit": "", "path": "packs"},
                                              "release": {"version": "1", "digest": "0" * 64, "path": "group/packed"},
                                              "basis": "review"})
    assert resolve_publish_destination(tmp_path, "packed", rows=ROWS).source_id == "ouroboros", "no origin is left"
    chosen = resolve_publish_destination(tmp_path, "packed", "mine", rows=ROWS)
    assert (chosen.repository, chosen.skill_folder("packed")) == ("alice/skills", "packs/packed")


def test_an_older_record_publishes_in_place_only_on_its_own_row(tmp_path):
    """A record from before views matches every view of its repository for updates; it goes back into its own
    folder only on its own row, as before. Another view the owner picks gets the skill in that view's folder."""
    write_install_record(tmp_path, "packed", {"source": {"id": "mine", "kind": "git", "name": "My skills",
                                                         "url": ROWS[3]["url"]},
                                              "release": {"version": "1", "digest": "0" * 64, "path": "group/packed"},
                                              "basis": "review"})
    team = {**ROWS[3], "id": "team", "name": "Team", "path": "team"}
    rows = [*ROWS, team]
    chosen = resolve_publish_destination(tmp_path, "packed", {"source_id": "team", "mode": "entry"}, rows=rows)
    assert (chosen.repository, chosen.skill_folder("packed")) == ("alice/skills", "team/packed")
    assert resolve_publish_destination(tmp_path, "packed", "mine", rows=rows).skill_folder("packed") == "group/packed"
    # A record naming its view goes back in place on any row reading exactly that view.
    write_install_record(tmp_path, "viewed", {"source": {"id": "gone", "kind": "git", "name": "Gone",
                                                         "url": ROWS[3]["url"], "ref": "main", "commit": "",
                                                         "path": "packs"},
                                              "release": {"version": "1", "digest": "0" * 64, "path": "packs/viewed"},
                                              "basis": "review"})
    assert resolve_publish_destination(tmp_path, "viewed", "mine", rows=rows).skill_folder("viewed") == "packs/viewed"
    assert resolve_publish_destination(tmp_path, "viewed", "team", rows=rows).skill_folder("viewed") == "team/viewed"


def test_a_pinned_view_takes_back_only_what_came_from_it(tmp_path):
    pinned = {**ROWS[3], "id": "pinned", "name": "Pinned", "commit": "a" * 40}
    pinned.pop("ref")
    rows = [*ROWS, pinned]
    with pytest.raises(SkillPublishDestinationError, match="Pinned stays at commit aaaaaaa") as refused:
        resolve_publish_destination(tmp_path, "fresh", "pinned", rows=rows)
    assert refused.value.reason_code == "destination_pinned"
    write_install_record(tmp_path, "kept", {"source": {"id": "pinned", "kind": "git", "name": "Pinned",
                                                       "url": ROWS[3]["url"], "ref": "", "commit": "a" * 40,
                                                       "path": "packs"},
                                            "release": {"version": "1", "digest": "0" * 64, "path": "packs/kept"},
                                            "basis": "review"})
    back = resolve_publish_destination(tmp_path, "kept", "pinned", rows=rows)
    assert (back.branch, back.skill_folder("kept")) == ("", "packs/kept"), "its origin: the default branch, in place"


def test_the_default_hub_is_never_a_pinned_view(tmp_path):
    """The hub chosen by default or by its alias obeys the refusal an explicit choice meets: a hub pinned to a commit
    is passed over, exactly as a hub read from a folder; the release that came from it still goes back to it."""
    pinned_hub = {"id": "stable-hub", "name": "Stable hub", "kind": "hub", "url": "https://github.com/team/hub",
                  "commit": "a" * 40, "trust": "review", "confirmed": True}
    with pytest.raises(SkillPublishDestinationError, match="Stable hub stays at commit aaaaaaa") as refused:
        resolve_publish_destination(tmp_path, "fresh", "stable-hub", rows=[pinned_hub])
    assert refused.value.reason_code == "destination_pinned"
    for requested in (None, "hub", {"source_id": "hub", "mode": "new_entry"}):
        alone = resolve_publish_destination(tmp_path, "fresh", requested, rows=[pinned_hub])
        assert (alone.source_id, alone.repository) == ("hub", "razzant/OuroborosHub"), requested
        beside = resolve_publish_destination(tmp_path, "fresh", requested, rows=[pinned_hub, *ROWS])
        assert beside.source_id == "ouroboros", requested
    _record(tmp_path, "kept", "stable-hub", "hub", {"commit": "a" * 40, "path": "skills/kept"})
    back = resolve_publish_destination(tmp_path, "kept", rows=[pinned_hub, *ROWS])
    assert (back.kind, back.source_id, back.repository, back.branch) == ("hub", "stable-hub", "team/hub", "")


def test_a_branch_named_in_full_is_that_branch_and_a_tag_is_refused_by_name(monkeypatch, tmp_path):
    """A revision may name its branch in full (``refs/heads/main``): publication reads and targets that branch. A tag
    never moves, so a source reading one would never read a publication: it is refused by name, before anything is
    written, and no other branch is chosen for it."""
    from ouroboros import skill_publish_github
    from ouroboros.skill_publish_client import GitHubError
    from ouroboros.skill_sources import with_added_source

    rows, full = with_added_source([], kind="git", url="https://github.com/team/skills", name="Main", trust="review",
                                   revision="refs/heads/main")
    rows, _hub = with_added_source(rows, kind="hub", url="https://github.com/team/hub", name="Team hub",
                                   trust="review", revision="refs/heads/stable")
    target = resolve_publish_destination(tmp_path, "fresh", full["id"], rows=rows)
    assert (target.repository, target.branch, target.skill_folder("fresh")) == ("team/skills", "main", "fresh")
    assert resolve_publish_destination(tmp_path, "fresh", "hub", rows=rows).branch == "stable"
    reads = []
    monkeypatch.setattr(skill_publish_github, "request",
                        lambda method, path, **_kwargs: reads.append(path) or {"object": {"sha": BASE_SHA}})
    assert skill_publish_github.fetch_branch_head(None, "team", "skills", target.branch) == (BASE_SHA, "main")
    assert reads == ["/repos/team/skills/git/refs/heads/main"]
    rows, tagged = with_added_source(rows, kind="git", url="https://github.com/team/skills", name="Release",
                                     trust="review", revision="refs/tags/v1.0.0")
    with pytest.raises(SkillPublishDestinationError, match="Release stays at the tag v1.0.0") as refused:
        resolve_publish_destination(tmp_path, "fresh", tagged["id"], rows=rows)
    assert refused.value.reason_code == "destination_tag"

    # A bare name that the repository holds as a tag is refused when its branch is read.
    rows, bare = with_added_source(rows, kind="git", url="https://github.com/hub/project", name="Bare",
                                   trust="review", revision="v1.0.0")
    ctx, events, captured = _install_transaction_fakes(monkeypatch, tmp_path, snapshot=_snapshot())
    monkeypatch.setattr(skill_publish, "resolve_publish_destination", lambda drive, skill, destination:
                        resolve_publish_destination(drive, skill, destination, rows=rows))
    monkeypatch.setattr(skill_publish, "fetch_branch_head", skill_publish_github.fetch_branch_head)
    reads.clear()

    def github(method, path, **_kwargs):
        reads.append(path)
        if path == "/repos/hub/project/git/refs/tags/v1.0.0":
            return {"ref": "refs/tags/v1.0.0", "object": {"type": "tag", "sha": "c" * 40}}
        raise GitHubError("Not Found", http_status=404)

    monkeypatch.setattr(skill_publish_github, "request", github)
    result = _submit(ctx, destination=bare["id"])
    assert (result["ok"], result["status"], result["reason_code"]) == (False, "blocked", "destination_tag"), result
    assert "v1.0.0 is a tag of hub/project" in result["repair_hint"]
    assert reads == ["/repos/hub/project/git/refs/heads/v1.0.0", "/repos/hub/project/git/refs/tags/v1.0.0"]
    assert not [event for event in events if event[0] == "mutation"] and captured["additions"] == []


def test_a_refusal_that_names_its_cause_reaches_the_preflight_and_the_tool(monkeypatch, tmp_path):
    """Publishing into a hub read from a folder is refused with its own reason, never as «not a GitHub repository»."""
    import ouroboros.config as config
    from ouroboros.gateway import skill_publish as preflight
    from ouroboros.skill_publish_destination import _row_destination

    folder_hub = {"id": "team-hub", "name": "Team hub", "kind": "hub", "url": "https://github.com/team/tools",
                  "path": "catalog", "trust": "review", "confirmed": True}
    monkeypatch.setattr(config, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setenv("OUROBOROS_SKILL_SOURCES", json.dumps([folder_hub, *ROWS]))
    _record(tmp_path, "hosted", "team-hub", "hub", {"commit": "c" * 40, "path": "catalog/skills/hosted"})
    outcome = preflight.build_skill_publish_preflight(tmp_path, "hosted", repo_path="", github_token_configured=True)
    assert (outcome.status_code, outcome.payload["reason_code"]) == (409, "hub_folder_unsupported")
    assert outcome.payload["summary"].startswith("Publishing to a hub inside a repository folder (catalog)")
    assert "not a GitHub repository" not in outcome.payload["summary"] + outcome.payload["repair_hint"]

    ctx, _events, _captured = _install_transaction_fakes(monkeypatch, tmp_path, snapshot=_snapshot())

    def refuse(_drive, _skill, _requested):
        return _row_destination(folder_hub)

    monkeypatch.setattr(skill_publish, "resolve_publish_destination", refuse)
    result = _submit(ctx, destination="team-hub")
    assert (result["ok"], result["reason_code"]) == (False, "hub_folder_unsupported")
    assert "inside a repository folder (catalog)" in result["repair_hint"]


# --- before any effect: the destination's copy of this skill against the release the local copy came from --

_HUB_ROW = {"id": "hub-main", "name": "Hub", "kind": "hub", "url": "https://github.com/hub/project", "ref": "main",
            "part": "official", "trust": "trusted", "confirmed": True}
_REAL_TRANSPORT = ("github_login", "fetch_repository", "fetch_branch_head", "fetch_tree_blobs", "fetch_upstream_file",
                   "fetch_upstream_catalog", "fetch_blob", "prepare_publish_repository", "ensure_branch",
                   "commit_payload", "create_pr_receipt", "merge_pull_request")


def _manifest(version: str, description: str = "Keeps a store.") -> bytes:
    return (f"---\nname: demo\ndescription: {description}\nversion: {version}\ntype: instruction\n"
            "when_to_use: Use for a test.\n---\nUse the store.\n").encode()


_RELEASE = {"SKILL.md": _manifest("1.0.0"), "lib/store.py": b"def store():\n    return 1\n",
            "tests/test_store.py": b"def test_store():\n    assert True\n"}


def _entry(version: str, **changes) -> dict:
    return {"schema_version": 2, "name": "demo", "description": "Keeps a store.", "category": ["writing"],
            "owner": "alice", "version": version, "source": {"kind": "hosted", "path": "skills/demo"},
            "digest": hashlib.sha256(version.encode()).hexdigest(), "withdrawn": None, "type": "instruction",
            "when_to_use": "Use for a test.", **changes}


def _hub(skill=_RELEASE, entry=None) -> dict:
    """A v2 hub with this skill (``skill`` None: not yet), a neighbouring skill and the hub's own files."""
    files = {"hub.json": b'{"schema_version": 2}', "categories.json": _CATEGORIES, "README.md": b"# Hub\n",
             "skills/other/SKILL.md": b"---\nname: other\nversion: 3.0.0\n---\n",
             "entries/other.json": entry_bytes({**_entry("3.0.0"), "name": "other"})}
    if skill is not None:
        files.update({f"skills/demo/{path}": data for path, data in skill.items()})
        files["entries/demo.json"] = entry_bytes(entry or _entry("1.0.0"))
    return files


class _GitHub:
    """One synthetic destination repository ``hub/project`` (branch ``main``) and the account ``alice``,
    answering the publisher's real HTTPS transport with Git's own blob ids; writes are recorded, never real."""

    def __init__(self, files, *, can_push=False):
        self.blobs, self.commits, self.writes, self.head, self.can_push = {}, {}, [], "", can_push
        self.push(files)

    def push(self, files):
        tree = {}
        for path, data in files.items():
            tree[path] = git_blob_oid(data)
            self.blobs[tree[path]] = data
        self.head = hashlib.sha1(json.dumps([self.head, sorted(tree.items())]).encode()).hexdigest()
        self.commits[self.head] = tree
        return self.head

    def files(self):
        return {path: self.blobs[oid] for path, oid in self.commits[self.head].items()}

    def request(self, method, path, *, body=None, timeout=30):
        repo = "/repos/hub/project"
        if method != "GET":
            self.writes.append((method, path, body))
        if (method, path) == ("GET", "/user"):
            return {"login": "alice"}
        if (method, path) == ("GET", repo):
            return {"default_branch": "main", "permissions": {"push": self.can_push}}
        if (method, path) == ("GET", f"{repo}/git/refs/heads/main"):
            return {"object": {"sha": self.head}}
        if method == "GET" and path.startswith(f"{repo}/git/trees/"):
            commit = path.rsplit("/", 1)[1].partition("?")[0]
            if commit not in self.commits:
                raise GitHubError("No commit found for SHA", http_status=422)
            return {"sha": commit, "truncated": False, "tree": [
                {"path": name, "mode": "100644", "type": "blob", "sha": oid}
                for name, oid in sorted(self.commits[commit].items())]}
        if method == "GET" and path.startswith(f"{repo}/git/blobs/"):
            return {"encoding": "base64", "content": base64.encodebytes(self.blobs[path.rsplit("/", 1)[1]]).decode()}
        if method == "GET" and path.startswith(f"{repo}/contents/"):
            name, _, commit = path[len(f"{repo}/contents/"):].partition("?ref=")
            oid = self.commits.get(commit, {}).get(urllib.parse.unquote(name))
            if oid is None:
                raise GitHubError("Not Found", http_status=404)
            return {"content": base64.b64encode(self.blobs[oid]).decode()}
        if (method, path) == ("GET", "/repos/alice/project"):
            return {"name": "project"}
        if method == "GET" and "/git/ref/heads/" in path:
            raise GitHubError("Not Found", http_status=404)
        if (method, path) == ("POST", "/repos/alice/project/git/refs"):
            return {"object": {"sha": body["sha"]}}
        if (method, path) == ("POST", "/graphql"):
            return {"data": {"createCommitOnBranch": {"commit": {
                "oid": COMMIT_SHA, "url": "https://github.com/alice/project/commit/" + COMMIT_SHA}}}}
        if (method, path) == ("POST", f"{repo}/pulls"):
            return {"html_url": "https://github.com/hub/project/pull/7", "number": 7}
        if (method, path) == ("PUT", f"{repo}/pulls/7/merge"):
            return {"merged": True, "sha": "9" * 40}
        raise AssertionError(f"unexpected GitHub request {method} {path}")


def _local(tmp_path, files):
    """The installed payload with exactly ``files``, freshly reviewed clean."""
    payload = tmp_path / "skills" / "ouroboroshub" / "demo"
    shutil.rmtree(payload, ignore_errors=True)
    for path, data in files.items():
        (payload / path).parent.mkdir(parents=True, exist_ok=True)
        (payload / path).write_bytes(data)
    loaded = load_skill(payload, tmp_path)
    loaded.source = SKILL_SOURCE_OUROBOROSHUB
    loaded.review = SkillReviewState(status="clean", content_hash=loaded.content_hash)
    return loaded


def _installed_at(tmp_path, commit, **release):
    write_install_record(tmp_path, "demo", {
        "source": {"id": "hub-main", "kind": "hub", "name": "Hub", "url": _HUB_ROW["url"], "ref": "main",
                   "commit": "", "path": ""},
        "release": {"version": "1.0.0", "digest": "0" * 64, "commit": commit, "path": "skills/demo", **release},
        "basis": "trusted"})


def _publishing(monkeypatch, tmp_path, github, loaded):
    """The real publisher over ``github``: capture, destination resolution, comparison, transport and receipt
    are the product's; only the scanner and the optional PR prose model stay the transaction fakes."""
    from ouroboros import skill_publish_github
    from ouroboros.skill_publish_snapshot import capture_skill_publish_snapshot

    ctx, _events, captured = _install_transaction_fakes(monkeypatch, tmp_path, snapshot=_snapshot())
    for name in _REAL_TRANSPORT:
        monkeypatch.setattr(skill_publish, name, getattr(skill_publish_github, name))
    monkeypatch.setattr(skill_publish_github, "request", github.request)
    monkeypatch.setattr(skill_publish, "capture_skill_publish_snapshot", capture_skill_publish_snapshot)
    monkeypatch.setattr(skill_publish, "_validate_local_skill", lambda *_a: ("demo", loaded))
    monkeypatch.setattr(skill_publish, "resolve_publish_destination", lambda drive, skill, requested: (
        resolve_publish_destination(drive, skill, requested, rows=[_HUB_ROW])))
    return ctx, captured


def _written(github):
    """``(files added, paths deleted, PR body, the commit the branch was built on)`` of the one publication."""
    branch = next(body for _method, path, body in github.writes if path.endswith("/git/refs"))
    change = next(body for _method, path, body in github.writes if path == "/graphql")["variables"]["input"]
    pull = next(body for _method, path, body in github.writes if path.endswith("/pulls"))
    files = change["fileChanges"]
    return ({row["path"]: base64.b64decode(row["contents"]) for row in files["additions"]},
            [row["path"] for row in files.get("deletions", [])], pull["body"], branch["sha"])


def _rows(comparison, kind="paths"):
    name, change = ("path", "effect") if kind == "paths" else ("field", "change")
    return {(row[name], row[change], row["upstream"]) for row in comparison[kind]}


@pytest.mark.parametrize("format", ["v1", "v2"])
def test_removed_listing_needs_a_decision_before_republication(monkeypatch, tmp_path, format):
    files = _hub()
    if format == "v1":
        files.pop("hub.json")
        files.pop("entries/demo.json")
        files["catalog.json"] = json.dumps({"skills": [{"slug": "demo", "name": "demo", "version": "1.0.0",
            "description": "Keeps a store.", "type": "instruction", "when_to_use": "Use for a test.",
            "files": [{"path": p, "sha256": hashlib.sha256(b).hexdigest(), "size": len(b)}
                      for p, b in _RELEASE.items()]}]}).encode()
    github = _GitHub(files)
    base = github.head
    if format == "v1":
        files["catalog.json"] = b'{"skills": []}'
    else:
        files.pop("entries/demo.json")
    github.push(files)
    loaded = _local(tmp_path, {**_RELEASE, "SKILL.md": _manifest("1.1.0")})
    _installed_at(tmp_path, base)
    ctx, _captured = _publishing(monkeypatch, tmp_path, github, loaded)
    result = _submit(ctx, category=["writing"])
    assert not github.writes and result["reason_code"] == "destination_decision_required"
    comparison = result["destination_comparison"]
    assert any(r["upstream"] == "changed_since_base" for r in comparison["fields"])
    result = _submit(ctx, category=["writing"], expected_destination_state=comparison["destination_state"],
                     replacement_rationale="Restore this listing as part of the authorized publication.")
    assert result["ok"] is True and result["destination_comparison"]["state"] == "decided"


def test_a_destination_unchanged_since_the_install_takes_local_edits_and_deletions_at_once(monkeypatch, tmp_path):
    """Ordinary publication stays autonomous: the hub's own files and a neighbouring skill moved since the install,
    this skill's files and entry did not, so every change in its footprint is the local copy's own."""
    github = _GitHub(_hub())
    base = github.head
    github.push({**github.files(), "README.md": b"# Hub, newer\n", "skills/other/SKILL.md": b"changed neighbour\n",
                 "entries/other.json": entry_bytes({**_entry("3.1.0"), "name": "other"})})
    loaded = _local(tmp_path, {"SKILL.md": _manifest("1.1.0"), "lib/store.py": b"def store():\n    return 2\n",
                               "lib/feature.py": b"FEATURE = True\n"})
    _installed_at(tmp_path, base)
    ctx, _captured = _publishing(monkeypatch, tmp_path, github, loaded)
    result = _submit(ctx)
    assert result["ok"] is True, result
    comparison = result["destination_comparison"]
    assert (comparison["state"], comparison["base_commit"], comparison["head_commit"]) == ("clear", base, github.head)
    assert _rows(comparison) == {("skills/demo/SKILL.md", "replaced", "unchanged_since_base"),
                                 ("skills/demo/lib/store.py", "replaced", "unchanged_since_base"),
                                 ("skills/demo/lib/feature.py", "added", "unchanged_since_base"),
                                 ("skills/demo/tests/test_store.py", "deleted", "unchanged_since_base")}
    assert _rows(comparison, "fields") == {("version", "replaced", "unchanged_since_base"),
                                           ("digest", "replaced", "unchanged_since_base")}
    added, deleted, body, built_on = _written(github)
    assert deleted == ["skills/demo/tests/test_store.py"] and built_on == github.head
    assert set(added) == {"skills/demo/SKILL.md", "skills/demo/lib/store.py", "skills/demo/lib/feature.py",
                          "entries/demo.json"}
    assert "Replaced destination content" not in body and "replacement_rationale" not in result
    assert not list(task_artifact_dir_path(tmp_path, "task-1").glob("skill-publish-comparison-*"))


def test_newer_destination_work_is_never_replaced_without_a_decision_bound_to_it(monkeypatch, tmp_path):
    """The incident's shape: the hub fixed the skill after the install; the local copy keeps the old file, lacks the
    new one and the newer entry text. Nothing is written until the mind decides on exactly that comparison."""
    from ouroboros.skill_install_record import install_record_path

    github = _GitHub(_hub())
    base = github.head
    github.push({**github.files(), "skills/demo/lib/store.py": b"def store():\n    return 1  # fixed upstream\n",
                 "skills/demo/CHANGELOG.md": b"## 1.0.1\n- fixed the store upstream\n",
                 "entries/demo.json": entry_bytes(_entry("1.0.1", description="Keeps a store, fixed."))})
    local = {**_RELEASE, "SKILL.md": _manifest("1.1.0"), "lib/feature.py": b"FEATURE = True\n"}
    loaded = _local(tmp_path, local)
    _installed_at(tmp_path, base)
    payload = tmp_path / "skills" / "ouroboroshub" / "demo"
    inputs = {path: path.read_bytes() for path in [*payload.rglob("*"), install_record_path(tmp_path, "demo")]
              if path.is_file()}
    ctx, captured = _publishing(monkeypatch, tmp_path, github, loaded)

    refused = _submit(ctx)
    assert (refused["ok"], refused["status"], refused["reason_code"], refused["completed_stage"]) == (
        False, "blocked", "destination_decision_required", "upstream_read")
    assert github.writes == [] and captured["llm_calls"] == 0 and "receipt" not in refused
    comparison = refused["destination_comparison"]
    assert (comparison["state"], comparison["base_commit"], comparison["base_version"]) == (
        "decision_required", base, "1.0.0")
    assert _rows(comparison) == {("skills/demo/lib/store.py", "replaced", "changed_since_base"),  # reverts the fix
                                 ("skills/demo/CHANGELOG.md", "deleted", "changed_since_base"),  # drops newer file
                                 ("skills/demo/SKILL.md", "replaced", "unchanged_since_base"),
                                 ("skills/demo/lib/feature.py", "added", "unchanged_since_base")}
    assert {("description", "replaced", "changed_since_base"),
            ("version", "replaced", "changed_since_base")} <= _rows(comparison, "fields")
    report = (task_artifact_dir_path(tmp_path, "task-1") / comparison["artifact"]["path"]).read_text(encoding="utf-8")
    assert comparison["destination_state"] in report and "fixed the store upstream" in report
    assert "+    return 1  # fixed upstream" in report and "-    return 1  # fixed upstream" in report
    assert {path: path.read_bytes() for path in inputs} == inputs  # payload and install record untouched
    assert not (tmp_path / "state" / "skills" / "demo" / "ouroboroshub.json").exists()

    # This skill's copy at the destination moved: the earlier decision no longer describes it.
    github.push({**github.files(), "skills/demo/CHANGELOG.md": b"## 1.0.2\n- another upstream fix\n"})
    stale = _submit(ctx, expected_destination_state=comparison["destination_state"], replacement_rationale="Ship it.")
    assert stale["reason_code"] == "destination_state_changed" and github.writes == []
    state = stale["destination_comparison"]["destination_state"]
    assert state != comparison["destination_state"]
    unexplained = _submit(ctx, expected_destination_state=state)
    assert unexplained["reason_code"] == "replacement_rationale_required" and github.writes == []
    # So did the local snapshot (re-reviewed after an edit).
    edited = _local(tmp_path, {**local, "lib/feature.py": b"FEATURE = 2\n"})
    monkeypatch.setattr(skill_publish, "_validate_local_skill", lambda *_a: ("demo", edited))
    moved = _submit(ctx, expected_destination_state=state, replacement_rationale="Ship it.")
    assert moved["reason_code"] == "destination_state_changed" and github.writes == []
    state = moved["destination_comparison"]["destination_state"]
    # A commit outside this skill's footprint leaves the decision valid.
    github.push({**github.files(), "README.md": b"# Hub, again\n", "skills/other/SKILL.md": b"neighbour again\n"})
    rationale = "Ship the local feature; it supersedes the upstream store fix and its changelog."
    published = _submit(ctx, expected_destination_state=state, replacement_rationale=rationale)
    assert published["ok"] is True, published
    assert (published["destination_comparison"]["state"], published["replacement_rationale"]) == ("decided", rationale)
    added, deleted, body, built_on = _written(github)
    assert deleted == ["skills/demo/CHANGELOG.md"] and built_on == github.head
    assert added["skills/demo/lib/store.py"] == _RELEASE["lib/store.py"]
    assert "## Replaced destination content" in body and "- `skills/demo/lib/store.py` (replaced)" in body
    assert "- `entries/demo.json` field `description` (replaced)" in body and f"Author rationale: {rationale}" in body
    assert published["receipt"]["number"] == 7


def test_a_requested_merge_and_an_earlier_publication_never_stand_in_for_the_base(monkeypatch, tmp_path):
    """``merge`` is written when a merge is asked for, before any edit; a publication receipt survives reinstalls.
    Neither shows that the local copy holds the destination's newer content."""
    from ouroboros.marketplace.provenance import write_publication_record
    from ouroboros.skill_install_record import update_install_record

    github = _GitHub(_hub())
    base = github.head
    newer = _entry("1.0.1", description="Keeps a store, fixed.")
    github.push({**github.files(), "skills/demo/lib/store.py": b"def store():\n    return 3\n",
                 "entries/demo.json": entry_bytes(newer)})
    loaded = _local(tmp_path, {**_RELEASE, "SKILL.md": _manifest("1.1.0")})
    _installed_at(tmp_path, base)
    update_install_record(tmp_path, "demo", skipped_release_digest=newer["digest"], merge={
        "digest": newer["digest"], "version": "1.0.1", "requested_at": "2026-11-05T20:00:00+00:00",
        "requested_by": "ouroboros"})
    write_publication_record(tmp_path, "demo", {
        "slug": "demo", "version": "1.0.1", "content_hash": "c" * 64, "repository": "hub/project", "pr_number": 5,
        "pr_url": "https://github.com/hub/project/pull/5", "published_at": "2026-11-05T20:02:00+00:00"})
    ctx, _captured = _publishing(monkeypatch, tmp_path, github, loaded)
    result = _submit(ctx)
    assert (result["reason_code"], result["destination_comparison"]["base_commit"]) == (
        "destination_decision_required", base)
    assert ("skills/demo/lib/store.py", "replaced", "changed_since_base") in _rows(result["destination_comparison"])
    assert github.writes == []


def test_a_new_skill_publishes_without_a_decision(monkeypatch, tmp_path):
    github = _GitHub(_hub(skill=None))
    loaded = _local(tmp_path, _RELEASE)
    ctx, _captured = _publishing(monkeypatch, tmp_path, github, loaded)
    result = _submit(ctx, category=["writing"])
    assert result["ok"] is True, result
    comparison = result["destination_comparison"]
    assert (comparison["state"], comparison["base_reason"], comparison["fields"]) == ("clear", "no_install_record", [])
    assert {row["effect"] for row in comparison["paths"]} == {"added"}


@pytest.mark.parametrize(("record", "reason"), [("none", "no_install_record"), ("legacy", "base_unknown"),
                                                ("gone", "base_unreadable"), ("elsewhere", "installed_elsewhere")])
def test_a_copy_without_a_usable_base_replaces_destination_content_only_on_a_decision(monkeypatch, tmp_path,
                                                                                     record, reason):
    """Self-authored, sidecar-era, rewritten-history or other-origin copies keep a working path: the same
    comparison and an explicit decision, never a silent overwrite and never a permanent refusal."""
    local = {**_RELEASE, "SKILL.md": _manifest("2.0.0")}
    del local["tests/test_store.py"]
    loaded = _local(tmp_path, local)
    if record == "legacy":  # an OuroborosHub sidecar names its release digest, never a commit
        write_install_record(tmp_path, "demo", {
            "source": {"id": "ouroboroshub-legacy", "kind": "hub", "name": "Hub", "url": _HUB_ROW["url"]},
            "release": {"version": "1.0.0", "digest": "0" * 64}, "basis": "review",
            "converted_from": ".ouroboroshub.json"})
    elif record == "gone":
        _installed_at(tmp_path, "f" * 40)
    elif record == "elsewhere":
        write_install_record(tmp_path, "demo", {
            "source": {"id": "fork", "kind": "hub", "name": "Fork", "url": "https://github.com/someone/fork",
                       "ref": "main", "commit": "", "path": ""},
            "release": {"version": "1.0.0", "digest": "0" * 64, "commit": "e" * 40, "path": "skills/demo"},
            "basis": "review"})
    github = _GitHub(_hub())
    ctx, _captured = _publishing(monkeypatch, tmp_path, github, loaded)
    refused = _submit(ctx)
    comparison = refused["destination_comparison"]
    assert (refused["reason_code"], comparison["base_commit"], comparison["base_reason"]) == (
        "destination_decision_required", "", reason)
    assert _rows(comparison) == {("skills/demo/SKILL.md", "replaced", "unknown"),
                                 ("skills/demo/tests/test_store.py", "deleted", "unknown")}
    assert github.writes == []
    published = _submit(ctx, expected_destination_state=comparison["destination_state"],
                        replacement_rationale="Release 2.0.0 of this skill replaces the published 1.0.0.")
    assert published["ok"] is True, published
    _added, deleted, body, _built_on = _written(github)
    assert deleted == ["skills/demo/tests/test_store.py"]
    assert "with no recorded common base with the local copy" in body


@pytest.mark.parametrize(("change", "replaced"), [
    ({"category": ["city"]}, ""), ({"owner": "bob"}, ""),
    ({"withdrawn": {"reason": "Withdrawn by the hub."}}, "withdrawn"), ({"description": "Better words."}, "description"),
])
def test_entry_fields_the_destination_changed_need_a_decision_only_when_this_publication_replaces_them(
        monkeypatch, tmp_path, change, replaced):
    github = _GitHub(_hub())
    base = github.head
    github.push({**github.files(), "entries/demo.json": entry_bytes(_entry("1.0.0", **change))})
    loaded = _local(tmp_path, {**_RELEASE, "SKILL.md": _manifest("1.1.0")})
    _installed_at(tmp_path, base)
    ctx, _captured = _publishing(monkeypatch, tmp_path, github, loaded)
    result = _submit(ctx)
    changed = [row["field"] for row in result["destination_comparison"]["fields"]
               if row["upstream"] == "changed_since_base"]
    if replaced:
        assert result["reason_code"] == "destination_decision_required" and changed == [replaced]
        assert {row["upstream"] for row in result["destination_comparison"]["paths"]} == {"unchanged_since_base"}
        assert github.writes == []
    else:  # the publication keeps the hub's own choice, so nothing of it is replaced
        assert result["ok"] is True and changed == [], result
        written = json.loads(_written(github)[0]["entries/demo.json"])
        assert {key: written[key] for key in change} == change


def test_an_older_hub_compares_this_skills_catalog_row_not_the_whole_catalog(monkeypatch, tmp_path):
    def row(version, **extra):
        return {"slug": "demo", "name": "demo", "description": "Keeps a store.", "version": version,
                "type": "instruction", "when_to_use": "Use for a test.", **extra,
                "files": [{"path": path, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
                          for path, data in sorted(_RELEASE.items())]}

    def catalog(own, other):
        return json.dumps({"skills": [own, {"slug": "other", "version": other, "files": []}]}).encode()

    github = _GitHub({"catalog.json": catalog(row("1.0.0"), "3.0.0"),
                      **{f"skills/demo/{path}": data for path, data in _RELEASE.items()}})
    base = github.head
    github.push({**github.files(), "catalog.json": catalog(row("1.0.0"), "3.1.0")})  # only a neighbour's row moved
    loaded = _local(tmp_path, {**_RELEASE, "SKILL.md": _manifest("1.1.0")})
    _installed_at(tmp_path, base)
    ctx, _captured = _publishing(monkeypatch, tmp_path, github, loaded)
    result = _submit(ctx)
    assert result["ok"] is True, result
    assert (result["destination_comparison"]["state"], result["destination_comparison"]["metadata_path"]) == (
        "clear", "catalog.json")
    written = json.loads(_written(github)[0]["catalog.json"])
    assert {item["slug"]: item["version"] for item in written["skills"]} == {"demo": "1.1.0", "other": "3.1.0"}
    github.writes.clear()
    github.push({**github.files(), "catalog.json": catalog(row("1.0.0", description="Hub wording."), "3.1.0")})
    refused = _submit(ctx)
    assert refused["reason_code"] == "destination_decision_required" and github.writes == []
    assert ("description", "replaced", "changed_since_base") in _rows(refused["destination_comparison"], "fields")


def test_a_repository_root_destination_compares_and_writes_only_the_files_it_sends(monkeypatch, tmp_path):
    """A hub entry pointing at an author's repository root: the repository's own files are never deleted or
    compared; only what the publication sends is."""
    github = _GitHub({"README.md": b"# The author's repository\n", "LICENSE": b"MIT\n", **_RELEASE}, can_push=True)
    base = github.head
    github.push({**github.files(), "README.md": b"# The author's repository, newer\n", "docs/guide.md": b"Guide\n"})
    local = {**_RELEASE, "SKILL.md": _manifest("1.1.0")}
    del local["tests/test_store.py"]
    loaded = _local(tmp_path, local)
    _installed_at(tmp_path, base, repo=_HUB_ROW["url"], path="")
    ctx, _captured = _publishing(monkeypatch, tmp_path, github, loaded)
    result = _submit(ctx)
    assert result["ok"] is True, result
    comparison = result["destination_comparison"]
    assert (comparison["state"], comparison["metadata_path"]) == ("clear", "")
    assert _rows(comparison) == {("SKILL.md", "replaced", "unchanged_since_base")}
    added, deleted, _body, _built_on = _written(github)
    assert deleted == [] and set(added) == {"SKILL.md", "lib/store.py"}
    assert (result["files_published"], result["listed"]) == ("merged", "needs_entry_update")
    github.writes.clear()
    github.push({**github.files(), "lib/store.py": b"def store():\n    return 4\n"})
    refused = _submit(ctx)
    assert refused["reason_code"] == "destination_decision_required" and github.writes == []
    assert ("lib/store.py", "replaced", "changed_since_base") in _rows(refused["destination_comparison"])


@pytest.mark.parametrize("update_at", ["capture", "scan"])
def test_publication_keeps_the_origin_of_its_snapshot_during_a_real_update(monkeypatch, tmp_path, update_at):
    from concurrent.futures import ThreadPoolExecutor
    import threading

    from ouroboros import skill_lifecycle_queue as lane, skill_source_install as installer
    from ouroboros.skill_install_record import read_install_record
    from ouroboros.skill_source_catalog import tree_release_digest

    github = _GitHub(_hub())
    base = github.head
    newer = {**_RELEASE, "SKILL.md": _manifest("1.0.1"), "lib/store.py": b"def store():\n    return 9\n"}
    github.push(_hub(newer, _entry("1.0.1")))
    loaded = _local(tmp_path, {**_RELEASE, "SKILL.md": _manifest("1.1.0")})
    _installed_at(tmp_path, base)
    ctx, _captured = _publishing(monkeypatch, tmp_path, github, loaded)
    folder = tmp_path / "offered"
    for path, data in newer.items():
        (folder / path).parent.mkdir(parents=True, exist_ok=True)
        (folder / path).write_bytes(data)
    item = {"id": "hub-main/demo", "name": "demo", "version": "1.0.1", "digest": tree_release_digest(folder)}
    row = {**_HUB_ROW, "trust": "review"}  # no model or dependency installation during this test
    offer = {"commit": github.head, "entries": [{"name": "demo", "source": {"kind": "hosted", "path": "skills/demo"}}]}
    monkeypatch.setattr(installer, "find_item", lambda *_a, **_k: (row, item, offer))
    monkeypatch.setattr(installer, "entry_payload_dir", lambda *_a, **_k: folder)
    monkeypatch.delenv("OUROBOROS_SKILLS_REPO_PATH", raising=False)
    queued = threading.Event()
    monkeypatch.setattr(lane, "_notify_chat_progress", lambda job, phase: (
        queued.set() if job.kind == "update" and phase == "queued" else None))
    capture, scan = skill_publish.capture_skill_publish_snapshot, skill_publish.scan_named_bytes
    pending = []
    with ThreadPoolExecutor(max_workers=1) as pool:
        def start_update():
            pending.append(pool.submit(installer.run_in_lane, tmp_path, "update", "demo", lambda:
                installer.update_installed(tmp_path, "demo", mode="replace", digest=item["digest"],
                                           local_digest=loaded.content_hash, rows=[row])))
            assert queued.wait(10), "Update must enter the real lifecycle queue"

        def capture_and_queue(current):
            frozen = capture(current)
            if update_at == "capture":
                start_update()
                assert not pending[0].done(), "Update must wait until local capture and provenance are together"
            return frozen

        def scan_after_update(named_bytes, **kwargs):
            assert lane._LANE_HANDLE.get() is None, "scanning must not retain the lifecycle lane"
            if not pending:
                start_update()
            assert pending[0].result(timeout=10)["needs_review"]
            return scan(named_bytes, **kwargs)

        monkeypatch.setattr(skill_publish, "capture_skill_publish_snapshot", capture_and_queue)
        monkeypatch.setattr(skill_publish, "scan_named_bytes", scan_after_update)
        result = _submit(ctx)
    assert read_install_record(tmp_path, "demo")["release"]["commit"] == github.head
    assert (loaded.skill_dir / "lib/store.py").read_bytes() == newer["lib/store.py"]
    assert result["reason_code"] == "destination_decision_required", result
    assert result["destination_comparison"]["base_commit"] == base
    assert ("skills/demo/lib/store.py", "replaced", "changed_since_base") in _rows(result["destination_comparison"])
    assert github.writes == []


def test_a_decision_reaches_the_publisher_through_the_tool_registry(monkeypatch, tmp_path):
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.tools.tool_result import _HOST_NOTE_SEPARATOR

    github = _GitHub(_hub())
    base = github.head
    github.push({**github.files(), "skills/demo/CHANGELOG.md": b"## 1.0.1\n"})
    loaded = _local(tmp_path, {**_RELEASE, "SKILL.md": _manifest("1.1.0")})
    _installed_at(tmp_path, base)
    _publishing(monkeypatch, tmp_path, github, loaded)
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_SYNTHETIC1234567890")
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_a, **_k: (True, ""))
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_id = "task-1"

    def call(**arguments):
        text = registry.execute("submit_skill_to_hub", {"skill": "demo", "confirm_public_submission": True,
                                                        **arguments})
        return json.loads(text.partition(_HOST_NOTE_SEPARATOR)[0])

    refused = call()
    assert refused["reason_code"] == "destination_decision_required", refused
    published = call(expected_destination_state=refused["destination_comparison"]["destination_state"],
                     replacement_rationale="The changelog belongs to the superseded release line.")
    assert published["ok"] is True and published["destination_comparison"]["state"] == "decided", published
    assert _written(github)[1] == ["skills/demo/CHANGELOG.md"]
