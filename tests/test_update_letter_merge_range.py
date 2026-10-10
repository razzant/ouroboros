"""Real Git range regressions for update-letter model input (issue #961)."""

import pytest

from ouroboros import update_letter as ul
from tests.test_update_letter import _capture_for, _commit, _git, _status, _write_readme, letter_env  # noqa: F401

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


def test_real_merge_material_reaches_write_letter_messages(merged_range, letter_env, monkeypatch):  # noqa: F811
    repo, base, target, local, feature, nested = merged_range
    monkeypatch.setattr(ul, "_default_git", lambda: _capture_for(repo))
    seen = []

    def chat(client, **kwargs):
        seen.extend(kwargs["messages"])
        return {"content": "A short letter."}, {"ledger_attempt_ids": ["test-attempt"]}

    monkeypatch.setattr(ul, "_chat", chat)
    record = ul.refresh_after_check(_status(current_sha=local, latest_sha=target), drive_root=letter_env["drive"])
    assert record["state"] == "ready"
    text = seen[-1]["content"]
    for sha, subject in ((feature, "Project retry binding"), (nested, "Swarm planning transfer")):
        assert sha in text and subject in text
    assert "Preserve the Project through retries." in text
    assert "Transfer the planning obligation." in text
    assert "shared base" not in text and "local-only customization" not in text
    assert "ONE short paragraph" in text and "Summarize the whole range" in text
    assert record["key"]["base_sha"] == local and record["key"]["target_sha"] == target


@pytest.mark.parametrize("failed_read", ["subjects", "bodies", "readme"])
def test_each_material_read_failure_is_typed(merged_range, failed_read):
    repo, base, target, *_ = merged_range

    def capture(cmd):
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


def test_nano_real_fit_keeps_material_for_toolless_author(merged_range, letter_env, monkeypatch):  # noqa: F811
    import json
    from types import SimpleNamespace

    from ouroboros import context_fit
    from ouroboros.context_budget import OWNER_NANO_TARGET_TOKENS

    repo, base, target, *_ = merged_range
    material = ul.collect_range_material(base, target, git=_capture_for(repo))
    plans = []
    monkeypatch.setattr(context_fit, "reference_doc_sections", lambda *a, **k: ([], ""))

    def real_plan(env, memory, task):
        # Production fitting, with deterministic context/evidence instead of owner memory
        # or a network capability probe. Pressure exercises Nano input fitting.
        core = context_fit.ContextCore(
            base_prompt="p", bible_md="b", architecture_md="a", development_md="d",
            semi_stable_text="s" * (OWNER_NANO_TARGET_TOKENS * 8), dynamic_text="y",
            user_content_json=json.dumps(task["text"]), docs_need_development=False,
        )
        route = lambda task, **kw: (
            {"provider": "test", "model": "test/light", "base_url": "", "use_local": False},
            SimpleNamespace(route_fp="test", status="confirmed", stale=False, window_tokens=1_000_000),
        )
        plan = context_fit.build_context_fit_plan(env, core, task, preferred_mode="nano", route_resolver=route)
        plans.append(plan)
        return plan

    sent = []
    monkeypatch.setattr(ul, "_fit_plan", real_plan)
    monkeypatch.setattr(ul, "_chat", lambda client, **kw: (
        sent.append(kw) or {"content": "A paragraph."}, {"ledger_attempt_ids": ["test-nano"]}))
    record = ul.write_letter(_status(current_sha=base, latest_sha=target), material, drive_root=letter_env["drive"])
    assert record["state"] == "ready"
    plan = plans[0]
    assert plan.initial_mode == "nano"
    assert "[Exact task input source]" not in plan.messages_for("nano")[-1]["content"]
    assert sent[0]["messages"][0] == plan.projection("nano").system_message()
    assert sent[0]["tools"] is None
    text = sent[0]["messages"][-1]["content"]
    assert text == json.loads(plan.user_content_json)
    assert all(c["sha"] in text and c["subject"] in text for c in material["commits"])
