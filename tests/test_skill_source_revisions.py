"""A repository source read at a revision and from a folder, through probe → add → catalog → install → update.

Real local repositories: the remote's default branch is not ``main``; branches, an annotated tag, a commit
pin, a subfolder, a hub inside a folder and the ``.agents/skills`` standard folder are all fetched by the
ordinary transport. Install records name the view they came from, so two views of one repository stay apart.
"""

from __future__ import annotations

import json
import os
import shutil
import sys

import pytest

from ouroboros import skill_source_catalog as catalog
from ouroboros import skill_source_install as installer
from ouroboros import skill_sources as sources
from ouroboros.skill_install_record import read_install_record, write_install_record
from tests._skill_source_fixtures import author_repo, commit_all, entry, git, join_source_refreshes, write_skill


@pytest.fixture
def drive(tmp_path, monkeypatch):
    root = tmp_path / "data"
    (root / "skills").mkdir(parents=True)
    monkeypatch.delenv("OUROBOROS_SKILLS_REPO_PATH", raising=False)
    yield root
    join_source_refreshes()


class Remote:
    """A bare repository whose HEAD names ``trunk``, fed from a work tree."""

    def __init__(self, root):
        self.work, self.bare = root / "work", root / "remote.git"
        self.work.mkdir(parents=True)
        git("init", "-q", "-b", "trunk", cwd=self.work)
        git("init", "-q", "--bare", str(self.bare), cwd=root)
        git("symbolic-ref", "HEAD", "refs/heads/trunk", cwd=self.bare)
        self.url = str(self.bare)

    def push(self, message, branch="trunk"):
        commit = commit_all(self.work, message)
        git("push", "-q", "--force", self.url, f"HEAD:refs/heads/{branch}", cwd=self.work)
        return commit

    def tag(self, name):  # annotated: a fetch of it names the tag object, the checkout its commit
        git("tag", "-a", name, "-m", name, cwd=self.work)
        git("push", "-q", self.url, f"refs/tags/{name}", cwd=self.work)


@pytest.fixture
def remote(tmp_path):
    repo = Remote(tmp_path / "origin")
    write_skill(repo.work / "skills", "alpha", "1.0.0")
    write_skill(repo.work / ".agents" / "skills", "beta", "1.0.0")
    write_skill(repo.work / ".agents" / "skills", "alpha", "9.9.9")  # the same name: skills/alpha wins
    write_skill(repo.work / "pack", "gamma", "1.0.0")
    write_skill(repo.work / "pack" / ".agents" / "skills", "delta", "1.0.0")
    repo.trunk = repo.push("trunk one")
    return repo


def _row(rows, source_id):
    return next(row for row in rows if row["id"] == source_id)


def _item(drive, rows, item_id):
    return next(i for i in catalog.catalog_view(drive, rows=rows)["items"] if i["id"] == item_id)


def test_the_default_branch_is_the_remote_head_and_standard_folders_join_the_walk(drive, remote):
    probe = catalog.probe_source(drive, url=remote.url)
    assert (probe["kind"], probe["commit"]) == ("git", remote.trunk), "an empty revision reads the remote's HEAD"
    assert sorted(probe["found"]) == ["alpha", "beta", "gamma"]
    assert probe["problems"] == [".agents/skills/alpha: not offered, skills/alpha already offers the name 'alpha'"]
    rows, row = sources.with_added_source([], kind="git", url=remote.url, name="Team", trust="trusted")
    assert not ({"ref", "commit", "path"} & set(row)), "no revision is stored: it keeps following the default branch"
    assert not catalog.refresh_source(drive, row).get("error")
    alpha = _item(drive, rows, "team/alpha")
    assert alpha["version"] == "1.0.0", "the first folder of a name is the one offered"
    _snapshot, status = catalog.snapshot_for(drive, row)
    assert status["problems"] == probe["problems"], "the source state carries what it could not offer"


@pytest.mark.parametrize("kind", ["branch", "tag", "commit"])
def test_revision_and_folder_survive_add_catalog_install_and_update(drive, remote, kind):
    git("checkout", "-q", "-b", "next", cwd=remote.work)
    write_skill(remote.work / "pack", "gamma", "2.0.0", body="Next.")
    pinned = remote.push("next one", branch="next")
    remote.tag("v2")
    revision = {"branch": "next", "tag": "v2", "commit": pinned}[kind]
    probe = catalog.probe_source(drive, url=remote.url, revision=revision, subfolder="pack")
    assert probe["commit"] == pinned and sorted(probe["found"]) == ["delta", "gamma"], probe
    rows, row = sources.with_added_source([], kind=probe["kind"], url=remote.url, name="Pack", trust="trusted",
                                          revision=revision, subfolder="/pack/")
    assert row["path"] == "pack" and row.get("commit" if kind == "commit" else "ref") == revision
    assert not catalog.refresh_source(drive, row).get("error")
    gamma = _item(drive, rows, "pack/gamma")
    assert gamma["version"] == "2.0.0" and gamma.get("updated_at")
    installer.install_item(drive, "pack/gamma", rows=rows)
    record = read_install_record(drive, "gamma")
    assert record["release"]["path"] == "pack/gamma" and record["release"]["commit"] == pinned
    assert {key: record["source"][key] for key in ("ref", "commit", "path")} == {
        "ref": "" if kind == "commit" else revision, "commit": pinned if kind == "commit" else "", "path": "pack"}
    detail = catalog.item_detail(drive, "pack/gamma", rows=rows)
    assert "Next." in detail["about"]
    # A newer commit on the branch is an update; a tag or a pin stays where it is.
    write_skill(remote.work / "pack", "gamma", "3.0.0", body="Newer.")
    newer = remote.push("next two", branch="next")
    assert not catalog.refresh_source(drive, row).get("error")
    offered = _item(drive, rows, "pack/gamma")
    assert offered["installed"]["update_available"] is (kind == "branch")
    if kind != "branch":
        return
    result = installer.update_installed(drive, "gamma", mode="replace", digest=offered["digest"], rows=rows)
    assert result["mode"] == "replace" and result["acceptance"]["accepted"], result
    record = read_install_record(drive, "gamma")
    assert (record["release"]["commit"], record["release"]["path"], record["release"]["version"]) == (
        newer, "pack/gamma", "3.0.0")


def test_a_hub_inside_a_folder_serves_hosted_and_pointer_entries_from_that_folder(drive, remote, tmp_path):
    from ouroboros.skill_source_catalog import tree_release_digest

    hub = remote.work / "catalog"
    folder = write_skill(hub / "skills", "hosted", "1.2.0")
    bare, commit, digest = author_repo(tmp_path / "authors", "scholar")
    (hub / "hub.json").write_text(json.dumps({"schema_version": 2, "name": "Team hub", "maintainer": "team"}))
    (hub / "entries").mkdir()
    for name, value in {"hosted": entry("hosted", tree_release_digest(folder), version="1.2.0"),
                        "scholar": entry("scholar", digest, source={"kind": "git", "repo": str(bare), "path": "",
                                                                    "commit": commit})}.items():
        (hub / "entries" / f"{name}.json").write_text(json.dumps(value))
    remote.push("a hub in a folder")
    probe = catalog.probe_source(drive, url=remote.url, subfolder="catalog")
    assert (probe["kind"], probe["suggested_name"], probe["owner"], sorted(probe["found"])) == (
        "hub", "Team hub", "team", ["hosted", "scholar"])
    rows, row = sources.with_added_source([], kind="hub", url=remote.url, name="Team hub", trust="trusted",
                                          subfolder="catalog")
    assert not catalog.refresh_source(drive, row).get("error")
    snapshot = catalog.read_snapshot(drive, row)
    assert snapshot["path"] == "catalog" and snapshot["signals"] is None
    assert all(e.get("updated_at") for e in snapshot["entries"]), "hub entries are dated inside the hub folder"
    installer.install_item(drive, "team-hub/hosted", rows=rows)
    assert read_install_record(drive, "hosted")["release"]["path"] == "catalog/skills/hosted"
    installer.install_item(drive, "team-hub/scholar", rows=rows)
    assert read_install_record(drive, "scholar")["release"]["commit"] == commit


def test_two_views_of_one_repository_keep_their_installs_apart(drive, remote):
    git("checkout", "-q", "-b", "next", cwd=remote.work)
    write_skill(remote.work / "skills", "alpha", "2.0.0")
    remote.push("next", branch="next")
    rows, trunk = sources.with_added_source([], kind="git", url=remote.url, name="Trunk", trust="trusted")
    rows, nxt = sources.with_added_source(rows, kind="git", url=remote.url, name="Next", trust="trusted",
                                          revision="next")
    for row in rows:
        assert not catalog.refresh_source(drive, row).get("error")
    installer.install_item(drive, "next/alpha", rows=rows)
    assert _item(drive, rows, "next/alpha")["installed"]["state"] == "installed"
    assert _item(drive, rows, "trunk/alpha")["installed"]["state"] == "name_taken", "another view, another release"
    with pytest.raises(installer.SkillInstallError) as refused:
        installer.plan_install(drive, "trunk/alpha", rows=rows)
    assert refused.value.code == "name_taken"
    record = read_install_record(drive, "alpha")
    assert catalog.row_for_record(record, rows)["id"] == "next"
    # Removing its row leaves no other view to update from; the record is not re-homed on the other branch.
    with pytest.raises(installer.SkillInstallError) as gone:
        installer.update_installed(drive, "alpha", mode="replace", rows=[trunk])
    assert gone.value.code == "source_gone"
    # A record written before views existed names only its repository and keeps matching it.
    legacy = {**record, "source": {key: record["source"][key] for key in ("id", "kind", "name", "url")}}
    write_install_record(drive, "alpha", legacy)
    assert catalog.record_matches_row(legacy, trunk) and catalog.record_matches_row(legacy, nxt)
    assert not catalog.record_matches_row(record, trunk) and catalog.record_matches_row(record, nxt)


def test_a_name_added_again_for_another_view_does_not_inherit_the_installs_of_the_removed_source(drive, remote):
    from ouroboros.skill_loader import find_skill

    git("checkout", "-q", "-b", "next", cwd=remote.work)
    write_skill(remote.work / "skills", "alpha", "2.0.0", body="Next.")
    remote.push("next", branch="next")
    git("checkout", "-q", "trunk", cwd=remote.work)
    rows, team = sources.with_added_source([], kind="git", url=remote.url, name="Team", trust="trusted")
    assert not catalog.refresh_source(drive, team).get("error")
    installer.install_item(drive, "team/alpha", rows=rows)
    record = read_install_record(drive, "alpha")
    # The owner removes Team and adds the name again for the next branch: the same id now reads another view.
    rows, again = sources.with_added_source(sources.without_source(rows, "team"), kind="git", url=remote.url,
                                            name="Team", trust="trusted", revision="next")
    assert again["id"] == "team" and not catalog.refresh_source(drive, again).get("error")
    offered = _item(drive, rows, "team/alpha")
    assert offered["version"] == "2.0.0" and offered["installed"]["state"] == "name_taken"
    assert catalog.row_for_record(record, rows) is None
    with pytest.raises(installer.SkillInstallError) as taken:
        installer.plan_install(drive, "team/alpha", rows=rows)
    assert taken.value.code == "name_taken", "a skill of the same name from another view is not an update"
    with pytest.raises(installer.SkillInstallError) as gone:
        installer.update_installed(drive, "alpha", mode="replace", digest=offered["digest"], rows=rows)
    assert gone.value.code == "source_gone" and read_install_record(drive, "alpha")["release"] == record["release"]
    listing = catalog.install_listing(drive, find_skill(drive, "alpha"), catalog.cached_snapshots(drive, rows))
    assert listing["install"]["source"] == {"name": "Team", "kind": "git"}, "the id names another view now"
    assert listing["install"]["update_available"] is None
    # A record from before views names only its repository: its id still matches, as documented.
    assert catalog.record_matches_row({**record, "source": {"id": "team", "kind": "git", "name": "Team",
                                                            "url": "https://git.example.invalid/x"}}, again)
    # The view it came from, connected again under any name, is its source: updates flow from there.
    rows, shared = sources.with_added_source(rows, kind="git", url=remote.url, name="Shared", trust="trusted")
    write_skill(remote.work / "skills", "alpha", "1.1.0", body="Trunk again.")
    newer = remote.push("trunk two")
    assert not catalog.refresh_source(drive, shared).get("error")
    assert catalog.row_for_record(record, rows)["id"] == "shared"
    update = _item(drive, rows, "shared/alpha")
    assert update["installed"]["update_available"] and _item(drive, rows, "team/alpha")["installed"]["state"] == \
        "name_taken"
    # The listing names that same source, the one the catalog, Update and Replace match, under the recorded name.
    listing = catalog.install_listing(drive, find_skill(drive, "alpha"), catalog.cached_snapshots(drive, rows))
    assert listing["install"]["source"] == {"id": "shared", "name": "Team", "kind": "git"}
    assert listing["install"]["update_available"] == {"version": "1.1.0", "digest": update["digest"]}
    assert listing["install"]["modified"] is False, "the copy's own bytes still decide whether it is modified"
    with pytest.raises(installer.SkillInstallError) as same:
        installer.plan_install(drive, "shared/alpha", replace=True, rows=rows)
    assert same.value.code == "update_instead"
    installer.update_installed(drive, "alpha", mode="replace", digest=update["digest"], rows=rows)
    updated = read_install_record(drive, "alpha")
    assert (updated["source"]["id"], updated["release"]["commit"], updated["release"]["version"]) == (
        "shared", newer, "1.1.0")


def test_views_are_validated_where_they_are_chosen():
    url = "https://git.example.invalid/team/skills"
    rows, root = sources.with_added_source([], kind="git", url=url, name="Root", trust="review")
    rows, _pinned = sources.with_added_source(rows, kind="git", url=url, name="Pinned", trust="review",
                                              revision="A" * 40)
    assert _pinned["commit"] == "a" * 40 and "ref" not in _pinned
    rows, _folder = sources.with_added_source(rows, kind="git", url=url, name="Folder", trust="review",
                                              subfolder="skills/team")
    for revision, subfolder in (("", ""), ("a" * 40, ""), ("", "skills/team/")):
        with pytest.raises(sources.SkillSourceError, match="already reads this repository"):
            sources.with_added_source(rows, kind="git", url=url, name="Again", trust="review", revision=revision,
                                      subfolder=subfolder)
    for revision in ("main:refs/heads/x", "-x", "+main", "a b", "x..y", "x~1", "HEAD^", "@{1}"):
        with pytest.raises(sources.SkillSourceError, match="is not a branch, tag or commit name"):
            sources.view_fields(revision)
    with pytest.raises(sources.SkillSourceError, match="inside the repository"):
        sources.view_fields("", "a/../../b")
    assert sources.view_fields("refs/heads/feature/x", "/skills/") == {"ref": "refs/heads/feature/x", "path": "skills"}
    with pytest.raises(sources.SkillSourceError, match="read as it is"):
        sources.with_added_source([], kind="local", path="/x", name="Mine", trust="review", subfolder="skills")
    # The public hub comes back as its two rows only for its own view; another branch of it is another source.
    public = "https://github.com/razzant/OuroborosHub"
    restored, first = sources.with_added_source([], kind="hub", url=public, name="x", trust="review")
    assert [r["id"] for r in restored] == [sources.HUB_OFFICIAL_ID, sources.HUB_COMMUNITY_ID] and first["ref"] == "main"
    _rows, beta = sources.with_added_source(restored, kind="hub", url=public, name="Hub beta", trust="review",
                                            revision="beta")
    assert (beta["id"], beta["ref"], beta.get("part")) == ("hub-beta", "beta", None)
    assert root["id"] == "root" and "path" not in root


def test_a_connected_folder_loads_its_standard_folder_and_keeps_the_first_name(drive, tmp_path, monkeypatch):
    from ouroboros import skill_loader as loader

    folder = tmp_path / "project"
    write_skill(folder / "skills", "alpha", "1.0.0")
    write_skill(folder / ".agents" / "skills", "alpha", "9.9.9")
    write_skill(folder / ".agents" / "skills", "beta", "1.0.0")
    rows, row = sources.with_added_source([], kind="local", path=str(folder), name="Project", trust="review")
    monkeypatch.setenv(sources.SOURCES_SETTING, json.dumps(rows))
    inventory = {(c.name, c.skill_dir) for c in loader._skill_location_inventory(drive)}
    assert inventory == {("alpha", (folder / "skills" / "alpha").resolve()),
                         ("beta", (folder / ".agents" / "skills" / "beta").resolve())}
    assert not loader.skill_identity_collision_names(drive), "the second alpha is not a new collision"
    view = catalog.catalog_view(drive, rows=rows)
    assert {i["name"]: i["installed"]["state"] for i in view["items"]} == {"alpha": "installed", "beta": "installed"}
    status = next(s for s in view["sources"] if s["id"] == row["id"])
    assert status["problems"] == [".agents/skills/alpha: not offered, skills/alpha already offers the name 'alpha'"]
    assert catalog.probe_source(drive, path=str(folder), revision="main")["error"].startswith("A folder")
    # The first folder wins only inside the connected folder. A name another location already uses (an installed
    # skill, another connected folder) is the ordinary collision of any two locations: neither copy is preferred.
    write_skill(drive / "skills" / "ouroboroshub", "beta", "2.0.0")
    assert loader.skill_identity_collision_names(drive) == {"beta"}


@pytest.mark.skipif(sys.platform == "win32" or getattr(os, "geteuid", lambda: 1)() == 0,
                    reason="POSIX folder permissions; root reads any folder")
def test_an_unreadable_standard_folder_follows_the_reader_and_an_absent_one_is_normal(drive, tmp_path, monkeypatch):
    from ouroboros import skill_loader as loader

    folder = tmp_path / "project"
    write_skill(folder / "skills", "alpha", "1.0.0")
    standard = write_skill(folder / ".agents" / "skills", "beta", "1.0.0").parent
    rows, row = sources.with_added_source([], kind="local", path=str(folder), name="Project", trust="review")
    monkeypatch.setenv(sources.SOURCES_SETTING, json.dumps(rows))
    standard.chmod(0)
    try:
        assert {c.name for c in loader._skill_location_inventory(drive)} == {"alpha"}, "the runtime stays fail-soft"
        with pytest.raises(PermissionError):
            loader.discover_skills(drive, strict=True)
        _snapshot, status = catalog.snapshot_for(drive, row)
        assert [p.split(" (")[0] for p in status["problems"]] == [".agents/skills: could not be read"]
        probe = catalog.probe_source(drive, path=str(folder))
        assert probe["found"] == ["alpha"] and probe["problems"] == status["problems"]
    finally:
        standard.chmod(0o755)
    shutil.rmtree(folder / ".agents")
    assert [s.name for s in loader.discover_skills(drive, strict=True)] == ["alpha"]
    assert "problems" not in catalog.snapshot_for(drive, row)[1]


def test_a_view_behind_the_last_fetch_of_its_repository_reads_it_again(drive, remote):
    """Folder views of one repository and revision share its fetch: one view's refresh leaves the other's snapshot
    behind a fresh fetch, and reading that other view reads it again."""
    rows, top = sources.with_added_source([], kind="git", url=remote.url, name="Top", trust="trusted")
    rows, pack = sources.with_added_source(rows, kind="git", url=remote.url, name="Pack", trust="trusted",
                                           subfolder="pack")
    for row in (top, pack):
        assert not catalog.refresh_source(drive, row).get("error")
    write_skill(remote.work / "pack", "gamma", "2.0.0")
    newer = remote.push("trunk two")
    assert catalog.refresh_source(drive, top)["commit"] == newer  # the shared fetch is fresh now
    served, _status = catalog.snapshot_for(drive, pack)
    assert served["commit"] == remote.trunk  # the earlier read is served while the view reads again
    join_source_refreshes()
    caught_up, _status = catalog.snapshot_for(drive, pack)
    assert caught_up["commit"] == newer and {e["name"]: e["version"] for e in caught_up["entries"]}["gamma"] == "2.0.0"
    join_source_refreshes()


def test_a_view_behind_the_shared_clone_reads_its_files_at_its_own_commit(drive, remote):
    """While a folder view's snapshot is behind the clone another view moved, its detail and its install read the
    files of the snapshot's commit, so the version, the text and the digest describe one release."""
    from ouroboros import skill_source_git as transport
    from ouroboros.platform_layer import release_exclusive_file_lock

    rows, top = sources.with_added_source([], kind="git", url=remote.url, name="Top", trust="trusted")
    rows, pack = sources.with_added_source(rows, kind="git", url=remote.url, name="Pack", trust="trusted",
                                           subfolder="pack")
    for row in (top, pack):
        assert not catalog.refresh_source(drive, row).get("error")
    write_skill(remote.work / "pack", "gamma", "2.0.0", body="Newer files.")
    newer = remote.push("trunk two")
    assert catalog.refresh_source(drive, top)["commit"] == newer  # the shared clone holds the newer files now
    lock = transport.repo_cache_dir(drive, pack) / ".lock"
    held = transport._locked(lock, 2)  # the view's own refresh waits on the ordinary lock meanwhile
    assert held is not None
    try:
        detail = catalog.item_detail(drive, "pack/gamma", rows=rows)
        result = installer.install_item(drive, "pack/gamma", rows=rows)
    finally:
        release_exclusive_file_lock(lock, held)
        join_source_refreshes()
    assert detail["item"]["version"] == "1.0.0" and "About it." in detail["about"], detail
    assert "Newer" not in detail["about"] and not detail["about_error"]
    assert result["acceptance"]["accepted"], result
    record = read_install_record(drive, "gamma")
    assert (record["release"]["commit"], record["release"]["version"]) == (remote.trunk, "1.0.0")
    assert "Newer" not in (drive / "skills" / "ouroboroshub" / "gamma" / "SKILL.md").read_text(encoding="utf-8")
    # Caught up, the same view reads the shared clone again and offers the newer release as an update.
    offered = _item(drive, rows, "pack/gamma")
    assert offered["version"] == "2.0.0" and offered["installed"]["update_available"]


def test_install_keeps_selected_files_when_another_view_refreshes(drive, remote, monkeypatch):
    """The installer uses its selected folder after resolution returns; a refresh may happen in that interval."""
    rows, top = sources.with_added_source([], kind="git", url=remote.url, name="Top", trust="trusted")
    rows, pack = sources.with_added_source(rows, kind="git", url=remote.url, name="Pack", trust="trusted",
                                         subfolder="pack")
    for row in (top, pack):
        assert not catalog.refresh_source(drive, row).get("error")
    write_skill(remote.work / "pack", "gamma", "2.0.0", body="Newer files.")
    newer = remote.push("trunk two")
    resolve = installer.entry_payload_dir

    def refresh_after_resolution(*args, **kwargs):
        folder = resolve(*args, **kwargs)
        assert catalog.refresh_source(drive, top)["commit"] == newer
        return folder

    monkeypatch.setattr(installer, "entry_payload_dir", refresh_after_resolution)
    result = installer.install_item(drive, "pack/gamma", rows=rows)
    assert result["acceptance"]["accepted"], result
    record = read_install_record(drive, "gamma")
    assert (record["release"]["commit"], record["release"]["version"]) == (remote.trunk, "1.0.0")
    installed = drive / "skills" / "ouroboroshub" / "gamma" / "SKILL.md"
    assert "About it." in installed.read_text() and "Newer" not in installed.read_text()


def test_a_build_hub_address_follows_its_default_branch_unless_a_raw_address_names_one(drive, remote):
    factory = [row for row in sources.derived_sources({}) if row["kind"] == "hub"]
    assert [row["ref"] for row in factory] == ["main", "main"], "the factory raw GitHub address names main"
    raw = sources.derived_sources({"OUROBOROS_HUB_CATALOG_URL":
                                   "https://raw.githubusercontent.com/team/hub/stable/catalog.json"})
    assert {(row["url"], row["ref"]) for row in raw if row["kind"] == "hub"} == {("https://github.com/team/hub",
                                                                                 "stable")}
    hubs = [row for row in sources.derived_sources({"OUROBOROS_HUB_CATALOG_URL": remote.url}) if row["kind"] == "hub"]
    assert [(row["url"], "ref" in row) for row in hubs] == [(remote.url, False)] * 2
    assert not catalog.refresh_source(drive, hubs[0]).get("error")
    assert catalog.read_snapshot(drive, hubs[0])["commit"] == remote.trunk, "the remote's HEAD branch is trunk"
    saved = sources.load_sources({sources.SOURCES_SETTING: json.dumps([{**row, "ref": "main"} for row in hubs])})
    assert [row["ref"] for row in saved] == ["main", "main"], "a saved revision stays as saved"


def test_suggestions_are_offers_that_connect_nothing():
    rows = sources.derived_sources({})
    assert sources.source_suggestions({}, rows) == [], "the open-source build suggests nothing"
    offered = sources.source_suggestions({sources.SUGGESTIONS_SETTING: json.dumps([
        {"name": "Team skills", "url": "https://git.example.invalid/team/skills.git", "ref": "stable",
         "path": "skills", "about": "Our curated skills"},
        {"name": "Public hub", "url": rows[1]["url"], "ref": "main"},
        {"name": "Hub beta", "url": rows[1]["url"], "ref": "beta"},
        {"name": "Broken", "url": "https://u:p@git.example.invalid/x"},
        {"name": "Bad ref", "url": "https://git.example.invalid/y", "ref": "a:b"},
        "not an object",
    ])}, rows)
    assert offered == [
        {"name": "Team skills", "url": "https://git.example.invalid/team/skills.git", "ref": "stable",
         "path": "skills", "about": "Our curated skills", "connected": False},
        {"name": "Public hub", "url": rows[1]["url"], "ref": "main", "about": "", "connected": True},
        {"name": "Hub beta", "url": rows[1]["url"], "ref": "beta", "about": "", "connected": False},
    ], "another branch of a connected repository is not connected"
    assert sources.source_suggestions({sources.SUGGESTIONS_SETTING: "{not json"}, rows) == []
    assert sources.load_sources({sources.SUGGESTIONS_SETTING: json.dumps([offered[0]])}) == rows


@pytest.mark.parametrize("original,alias,name", [
    ("pack", "./pack", "gamma"), ("pack", "pack//./", "gamma"), ("", ".", "alpha"),
])
def test_reconnected_folder_alias_keeps_install_and_update_identity(drive, remote, original, alias, name):
    rows, row = sources.with_added_source([], kind="git", url=remote.url, name="Team", trust="trusted",
                                          subfolder=original)
    assert not catalog.refresh_source(drive, row).get("error")
    installer.install_item(drive, f"team/{name}", rows=rows)
    record = read_install_record(drive, name)
    rows, reconnected = sources.with_added_source([], kind="git", url=remote.url, name="Shared", trust="trusted",
                                                 subfolder=alias)
    assert not catalog.refresh_source(drive, reconnected).get("error")
    assert _item(drive, rows, f"shared/{name}")["installed"]["state"] == "installed"
    # Previously saved records can carry the alias too; reading it must not rewrite provenance.
    record["source"]["path"] = alias
    write_install_record(drive, name, record)
    assert catalog.record_matches_row(record, row)
    folder = remote.work / (original or "skills")
    write_skill(folder, name, "2.0.0", body="Updated through the same folder.")
    newer = remote.push("next release")
    assert not catalog.refresh_source(drive, reconnected).get("error")
    item = _item(drive, rows, f"shared/{name}")
    assert item["installed"]["update_available"]
    assert read_install_record(drive, name)["source"]["path"] == alias
    result = installer.update_installed(drive, name, mode="replace", digest=item["digest"], rows=rows)
    assert result["acceptance"]["accepted"]
    assert read_install_record(drive, name)["release"]["commit"] == newer


def test_cached_release_files_work_with_spaces_in_the_data_path(drive, remote):
    drive = drive / "owner data"
    (drive / "skills").mkdir(parents=True)
    rows, row = sources.with_added_source([], kind="git", url=remote.url, name="Team", trust="trusted",
                                          subfolder="pack")
    assert not catalog.refresh_source(drive, row).get("error")
    detail = catalog.item_detail(drive, "team/gamma", rows=rows)
    assert not detail["about_error"] and "About it." in detail["about"]
    assert installer.install_item(drive, "team/gamma", rows=rows)["acceptance"]["accepted"]
    write_skill(remote.work / "pack", "gamma", "2.0.0")
    newer = remote.push("next release")
    assert not catalog.refresh_source(drive, row).get("error")
    item = _item(drive, rows, "team/gamma")
    result = installer.update_installed(drive, "gamma", mode="replace", digest=item["digest"], rows=rows)
    assert result["acceptance"]["accepted"]
    assert read_install_record(drive, "gamma")["release"]["commit"] == newer
