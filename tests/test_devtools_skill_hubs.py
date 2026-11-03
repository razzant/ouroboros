"""The GitVerse catalog converter writes a hub the ordinary source lifecycle reads, installs and updates.

A GitVerse-shaped catalog (``<Section>/properties/<language>/<name>.json`` with ``skill_link``,
``install-cmd`` and ``prompt``) points at real local author repositories. The converter's commits and
digests are observed from those repositories; each run writes a new folder, whose files the test copies
into the hub repository and commits as an operator would, and that repository is consumed by
refresh → catalog → install → update. Metadata commands never run and never reach entries.
"""

from __future__ import annotations

import json
import shutil

import pytest

from devtools.skill_hubs import gitverse_hub
from ouroboros import skill_source_catalog as catalog
from ouroboros import skill_source_install as installer
from ouroboros import skill_sources as sources
from ouroboros.skill_install_record import read_install_record
from tests._skill_source_fixtures import commit_all, git, join_source_refreshes, publish, write_skill


@pytest.fixture
def drive(tmp_path, monkeypatch):
    root = tmp_path / "data"
    (root / "skills").mkdir(parents=True)
    monkeypatch.delenv("OUROBOROS_SKILLS_REPO_PATH", raising=False)
    yield root
    join_source_refreshes()


def _author(root, owner, repo, build):
    work = root / "work" / owner / repo
    work.mkdir(parents=True)
    git("init", "-q", cwd=work)
    build(work)
    commit = commit_all(work, "release")
    publish(work, root / "git" / owner / f"{repo}.git")
    return work, commit


def _metadata(catalog_root, section, language, name, **fields):
    path = catalog_root / section / "properties" / language / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"name": name, **fields}, ensure_ascii=False), encoding="utf-8")


@pytest.fixture
def market(tmp_path):
    root = tmp_path / "gitverse"
    marker = tmp_path / "install-cmd-ran"
    command = f"touch {marker}"
    writer, writer_commit = _author(root, "team", "skills", lambda w: (
        write_skill(w, "writer", "1.0.0"), write_skill(w / "skills", "reviewer", "0.2.0")))
    _single, single_commit = _author(root, "solo", "lone-skill", lambda w: (w / "SKILL.md").write_text(
        "---\nname: lone-skill\ndescription: One skill at the top\nversion: 0.1.0\ntype: instruction\n---\nTop.\n"))
    cat = root / "catalog"
    _metadata(cat, "Writing", "ru", "writer", description="Пишет тексты", skill_link="team/skills/writer",
              category="Писательство", labels=["Тексты"], **{"install-cmd": command, "prompt": f"run: {command}"})
    _metadata(cat, "Writing", "en", "writer", description="Writes texts", skill_link="team/skills/writer",
              category="Writing", **{"install-cmd": command})
    _metadata(cat, "Review", "en", "reviewer", description="Reviews", skill_link="team/skills",
              category="Review")
    _metadata(cat, "Misc", "en", "lone-skill", description="Alone", skill_link="solo/lone-skill", category="Misc")
    _metadata(cat, "Misc", "en", "ghost", description="Missing", skill_link="team/skills/ghost", category="Misc")
    _metadata(cat, "Misc", "en", "gone", description="No repository", skill_link="nobody/none/gone")
    (cat / "Misc" / "properties" / "en" / "broken.json").write_text("{not json", encoding="utf-8")
    return {"root": root, "catalog": cat, "marker": marker, "writer": (writer, writer_commit),
            "single_commit": single_commit, "template": f"{(root / 'git').as_uri()}/{{owner}}/{{repo}}.git"}


def _hub_repo(tmp_path, out, message):
    """The operator copies the reviewed output into the hub repository's working tree and commits it there."""
    work = tmp_path / "hub-repo"
    if not (work / ".git").is_dir():
        work.mkdir()
        git("init", "-q", cwd=work)
    shutil.copytree(out, work, dirs_exist_ok=True)
    commit_all(work, message)
    publish(work, tmp_path / "hub.git")
    return str(tmp_path / "hub.git")


def _files(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def test_the_converted_catalog_installs_and_updates_through_the_ordinary_lifecycle(drive, market, tmp_path):
    out = tmp_path / "hub"
    report = gitverse_hub.convert(market["catalog"], out, tmp_path / "work", repo_url=market["template"],
                                  name="GitVerse skills", maintainer="ops")
    assert report["written"] == ["lone-skill", "reviewer", "writer"]
    assert {item["skill_link"] for item in report["skipped"]} == {"nobody/none/gone", "team/skills/ghost"}
    assert any("broken.json" in problem for problem in report["problems"])
    assert not market["marker"].exists(), "install-cmd and prompt are data, never run"
    writer = json.loads((out / "entries" / "writer.json").read_text(encoding="utf-8"))
    assert writer["source"] == {"kind": "git", "repo": market["template"].format(owner="team", repo="skills"),
                                "path": "writer", "commit": market["writer"][1]}
    assert writer["description"] == "Writes texts" and writer["category"] == ["writing"]
    assert not ({"install-cmd", "prompt", "install_specs"} & set(writer))
    lone = json.loads((out / "entries" / "lone-skill.json").read_text(encoding="utf-8"))
    assert (lone["source"]["path"], lone["source"]["commit"]) == ("", market["single_commit"])
    reviewer = json.loads((out / "entries" / "reviewer.json").read_text(encoding="utf-8"))
    assert reviewer["source"]["path"] == "skills/reviewer", "no path in skill_link: the documented skills/<name>"
    hub = json.loads((out / "hub.json").read_text(encoding="utf-8"))
    assert hub["converted_from"] == {}, "a folder input claims no origin"

    url = _hub_repo(tmp_path, out, "converted")
    probe = catalog.probe_source(drive, url=url)
    assert (probe["kind"], probe["owner"], sorted(probe["found"])) == ("hub", "ops", report["written"])
    rows, row = sources.with_added_source([], kind="hub", url=url, name="GitVerse", trust="trusted")
    assert not catalog.refresh_source(drive, row).get("error")
    view = catalog.catalog_view(drive, rows=rows)
    assert {i["name"] for i in view["items"]} == set(report["written"]) and not view["sources"][0].get("problems")
    assert {c["id"]: c["label"] for c in view["categories"]}["writing"] == "Writing"
    result = installer.install_item(drive, "gitverse/writer", rows=rows)
    assert result["acceptance"]["accepted"], result
    record = read_install_record(drive, "writer")
    assert record["release"]["digest"] == writer["digest"] and record["release"]["commit"] == market["writer"][1]

    # The author releases again; a new run into a new folder, copied into the hub repository, moves the pointer.
    work, _ = market["writer"]
    write_skill(work, "writer", "1.1.0", body="Newer.")
    newer = commit_all(work, "second release")
    publish(work, market["root"] / "git" / "team" / "skills.git")
    again = gitverse_hub.convert(market["catalog"], tmp_path / "hub-2", tmp_path / "work-2",
                                 repo_url=market["template"])
    assert again["repositories"][writer["source"]["repo"]] == newer
    _hub_repo(tmp_path, tmp_path / "hub-2", "converted again")
    assert not catalog.refresh_source(drive, row).get("error")
    offered = next(i for i in catalog.catalog_view(drive, rows=rows)["items"] if i["name"] == "writer")
    assert offered["installed"]["update_available"] and offered["version"] == "1.1.0"
    installer.update_installed(drive, "writer", mode="replace", digest=offered["digest"], rows=rows)
    assert read_install_record(drive, "writer")["release"]["commit"] == newer
    assert not market["marker"].exists()


def test_skills_sharing_a_repository_link_stay_apart_and_a_repeated_file_is_reported(market, tmp_path):
    """A ``skill_link`` may name only a repository; each metadata file name is its own skill there, read from
    one checkout of that repository. A second file for one skill and language is named, never silently used."""
    _author(market["root"], "team", "pack", lambda w: (write_skill(w / "skills", "alpha", "1.0.0"),
                                                      write_skill(w / "skills", "beta", "2.0.0")))
    cat = tmp_path / "catalog"
    for section, name in (("Writing", "alpha"), ("Writing", "beta"), ("Review", "beta")):
        _metadata(cat, section, "en", name, description=f"{name} in {section}", skill_link="team/pack",
                  category=section)
    _metadata(cat, "Writing", "ru", "alpha", description="альфа", skill_link="team/pack", category="Письмо")
    report = gitverse_hub.convert(cat, tmp_path / "hub", tmp_path / "work", repo_url=market["template"])
    assert report["written"] == ["alpha", "beta"] and report["skipped"] == [], report
    assert report["problems"] == ["Writing/properties/en/beta.json: not used, Review/properties/en/beta.json already "
                                  "describes this skill in en"]
    assert list(report["repositories"]) == [market["template"].format(owner="team", repo="pack")]
    assert len(list((tmp_path / "work" / "repos").iterdir())) == 1, "one checkout serves every skill of a repository"
    alpha = json.loads((tmp_path / "hub" / "entries" / "alpha.json").read_text(encoding="utf-8"))
    beta = json.loads((tmp_path / "hub" / "entries" / "beta.json").read_text(encoding="utf-8"))
    assert (alpha["source"]["path"], alpha["description"], alpha["version"]) == ("skills/alpha", "alpha in Writing",
                                                                                 "1.0.0")
    assert alpha["converted_from"] == ["Writing/properties/en/alpha.json", "Writing/properties/ru/alpha.json"]
    assert (beta["source"]["path"], beta["description"], beta["converted_from"]) == (
        "skills/beta", "beta in Review", ["Review/properties/en/beta.json"])


def test_the_cli_writes_a_new_folder_and_never_changes_an_earlier_hub(market, tmp_path, capsys):
    out = tmp_path / "hub"
    report_file = tmp_path / "report.json"
    code = gitverse_hub.main(["--catalog", str(market["catalog"]), "--out", str(out), "--work", str(tmp_path / "w"),
                              "--repo-url", market["template"], "--language", "ru", "--report", str(report_file)])
    assert code == 0
    assert sorted(p.stem for p in (out / "entries").glob("*.json")) == ["lone-skill", "reviewer", "writer"]
    writer = json.loads((out / "entries" / "writer.json").read_text(encoding="utf-8"))
    assert (writer["description"], writer["category"]) == ("Пишет тексты", ["писательство"]), \
        "the chosen language's metadata names the entry's description and category"
    categories = json.loads((out / "categories.json").read_text(encoding="utf-8"))["categories"]
    assert {c["id"]: c["label"] for c in categories}["писательство"] == "Писательство"
    assert json.loads(report_file.read_text(encoding="utf-8"))["written"] == ["lone-skill", "reviewer", "writer"]
    # An earlier hub, with the operator's own files, is refused before anything is read or written; a run that
    # reads less (here: no author repository at all) could otherwise have removed its entries.
    (out / "entries" / "retired.json").write_text("{}", encoding="utf-8")
    (out / "README.md").write_text("operator notes", encoding="utf-8")
    before = _files(out)
    unreadable = ["--repo-url", "file:///nowhere/{owner}/{repo}.git"]
    with pytest.raises(SystemExit):
        gitverse_hub.main(["--catalog", str(market["catalog"]), "--out", str(out), "--work", str(tmp_path / "w2"),
                           *unreadable])
    assert "must be absent or empty" in capsys.readouterr().err
    with pytest.raises(ValueError, match="must be absent or empty"):
        gitverse_hub.convert(market["catalog"], out, tmp_path / "w3", repo_url=unreadable[1])
    assert _files(out) == before and not (tmp_path / "w2").exists() and not (tmp_path / "w3").exists()
    assert not market["marker"].exists()
