"""Real Git range regressions for the update description's material (issue #961)."""

import pytest

from ouroboros import update_letter as ul
from tests.test_update_description import _registry
from tests.test_update_letter import _capture_for, _commit, _git, _status, _write_readme

pytestmark = pytest.mark.serial


@pytest.fixture
def merged_range(tmp_path):
    repo = tmp_path / "history"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "user.email", "test@example.com")
    (repo / "VERSION").write_text("1.0.0\n")
    _write_readme(repo, [("1.0.0", "2026-01-01", "already released")])
    base = _commit(repo, "shared base")
    _git(repo, "checkout", "-q", "-b", "feature")
    _write_readme(repo, [("1.1.0", "2026-01-02", "temporary branch row")])
    feature = _commit(repo, "Project retry binding\n\nPreserve the Project through retries.")
    _git(repo, "checkout", "-q", "-b", "nested")
    (repo / "nested.txt").write_text("nested\n")
    nested = _commit(repo, "Swarm planning transfer\n\nTransfer the planning obligation.")
    _git(repo, "checkout", "-q", "feature")
    _git(repo, "merge", "-q", "--no-ff", "-m", "merge nested", "nested")
    # The temporary row never reaches main; the surviving row does.
    _write_readme(repo, [("1.2.0", "2026-01-03", "released Project behavior"),
                         ("1.0.0", "2026-01-01", "already released")])
    _commit(repo, "prepare release")
    _git(repo, "checkout", "-q", "main")
    _git(repo, "merge", "-q", "--no-ff", "-m", "generic merge", "feature")
    target = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "-b", "local", base)
    (repo / "local.txt").write_text("local\n")
    local = _commit(repo, "local-only customization")
    return repo, base, target, local, feature, nested


@pytest.mark.parametrize("divergent", [False, True])
def test_full_graph_range_excludes_base_and_local_commits(merged_range, divergent):
    repo, base, target, local, feature, nested = merged_range
    start = local if divergent else base
    material = ul.collect_range_material(start, target, git=_capture_for(repo))
    shas = [c["sha"] for c in material["commits"]]
    assert set(shas) == set(_git(repo, "rev-list", f"{start}..{target}").splitlines())
    assert len(shas) == len(set(shas))
    assert feature in shas and nested in shas
    assert base not in shas and local not in shas
    assert [c['sha'] for c in material['local_commits']] == ([local] if divergent else [])
    bodies = {c["sha"]: c["body"] for c in material["commits"]}
    assert bodies[feature] == "Preserve the Project through retries."
    assert bodies[nested] == "Transfer the planning obligation."
    assert [(r["version"], r["commit"]) for r in material["releases"]] == [("1.2.0", target)]
    assert "temporary branch row" not in ul.material_text(material)


@pytest.mark.parametrize("limit", [0, 1])
def test_merge_range_preserves_body_and_row_bounds(merged_range, limit):
    repo, base, target, *_ = merged_range
    material = ul.collect_range_material(base, target, git=_capture_for(repo), max_bodies=limit, max_rows=limit)
    expected = _git(repo, "rev-list", f"{base}..{target}").splitlines()
    assert {c["sha"] for c in material["commits"]} == set(expected)
    assert material["bodies_omitted"] == len(expected) - limit
    assert all(not c["body"] for c in material["commits"][limit:])
    assert material["rows_summarized"] == 1 - limit
    assert bool(material["releases"][0]["text"]) == bool(limit)
    text = ul.material_text(material)
    assert all(sha in text for sha in expected)
    assert f"bodies of the {len(expected) - limit} oldest commit(s)" in text


def test_real_merge_material_reaches_its_reader_beside_the_record(merged_range, tmp_path, monkeypatch):
    """A fetching check writes the range's material where query_data(kind='update_material') reads it;
    the author's request names that reader, never a paragraph template."""
    import json

    repo, base, target, local, feature, nested = merged_range
    monkeypatch.setattr(ul, "_default_git", lambda: _capture_for(repo))
    monkeypatch.setattr("ouroboros.update_qualification.post_merge_checks",
                        lambda commits, url: {"status": "unavailable", "reason": "offline", "runs": [], "missing_shas": []})
    drive = tmp_path / "drive"
    record = ul.refresh_after_check(_status(current_sha=local, latest_sha=target), drive_root=drive)
    assert record["state"] == "none" and record["text"] == ""
    assert record["checked_key"]["base_sha"] == local and record["checked_key"]["target_sha"] == target
    document = json.loads(ul.material_path(drive).read_text(encoding="utf-8"))
    text = document["text"]
    for sha, subject in ((feature, "Project retry binding"), (nested, "Swarm planning transfer")):
        assert sha in text and subject in text
    assert "Preserve the Project through retries." in text
    assert "Transfer the planning obligation." in text
    assert "shared base" not in text
    assert text.index("Existing local history") < text.index("local-only customization")
    assert local in text and "not incoming changes" in text
    budget = document["text_bytes"]
    assert budget == record["material"]["text_bytes"] > len(text.encode("utf-8")), "the reader counts canonical JSON"
    request = ul.author_request_text(record, {"mode": "write", "key": record["checked_key"], "revision_at_request": 0})
    assert f'kind="update_material", select="text", max_bytes={budget})' in request
    assert "ONE short paragraph" not in request and "Summarize the whole range" not in request
    reg = _registry(tmp_path / "repo", drive, task_id="main-turn")
    reads = {size: reg.execute_result("query_data", {"op": "json", "kind": "update_material", "select": "text",
                                                     "max_bytes": size}) for size in (budget, budget - 1)}
    assert json.loads(reads[budget].text)["data"] == text and reads[budget - 1].status == "error"


@pytest.mark.parametrize("failed_read", ["subjects", "bodies", "readme", "first_parent"])
def test_each_material_read_failure_is_typed(merged_range, failed_read):
    repo, base, target, *_ = merged_range

    def capture(cmd):
        if failed_read == "first_parent" and cmd[:2] == ["git", "rev-list"]:
            return 128, "", "fixture read failure"
        kind = "readme" if "README.md" in cmd else "bodies" if "-n" in cmd else "subjects"
        if cmd[:2] == ["git", "log"] and kind == failed_read:
            return 128, "", "fixture read failure"
        return _capture_for(repo)(cmd)

    with pytest.raises(ul.MaterialUnavailable):
        ul.collect_range_material(base, target, git=capture)


def test_previously_consumed_branch_commits_stay_excluded(merged_range):
    repo, base, target, local, feature, nested = merged_range
    # Local already contains a side-branch commit, but not the official merge.
    _git(repo, "merge", "-q", "--no-ff", "-m", "consume feature", feature)
    consumed_base = _git(repo, "rev-parse", "HEAD")
    material = ul.collect_range_material(consumed_base, target, git=_capture_for(repo))
    shas = [c["sha"] for c in material["commits"]]
    assert set(shas) == set(_git(repo, "rev-list", f"{consumed_base}..{target}").splitlines())
    assert feature not in shas and base not in shas and local not in shas
    assert nested in shas and target in shas


def test_long_branch_body_is_bounded_without_losing_its_identity(merged_range):
    repo, base, target, *_ = merged_range
    _git(repo, "checkout", "-q", "-b", "long-body", target)
    (repo / "long.txt").write_text("body\n")
    large = _commit(repo, "long branch detail\n\n" + "x" * (ul.COMMIT_BODY_MAX_CHARS + 500))
    _git(repo, "checkout", "-q", "main")
    _git(repo, "merge", "-q", "--no-ff", "-m", "merge long body", large)
    final = _git(repo, "rev-parse", "HEAD")
    material = ul.collect_range_material(base, final, git=_capture_for(repo))
    entry = next(c for c in material["commits"] if c["sha"] == large)
    assert len(entry["body"]) <= ul.COMMIT_BODY_MAX_CHARS
    assert "OMISSION" in entry["body"]
    assert large in ul.material_text(material) and entry["subject"] == "long branch detail"
