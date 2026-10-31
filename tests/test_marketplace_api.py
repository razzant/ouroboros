"""Tests for the ClawHub marketplace HTTP API adapter layer."""

from __future__ import annotations

import asyncio
import json
import shutil
from types import SimpleNamespace

import pytest

from ouroboros.gateway import marketplace as marketplace_api
from ouroboros.marketplace.clawhub import ClawHubArchive, ClawHubSkillSummary
from ouroboros.marketplace.fetcher import StagedSkill
from ouroboros.marketplace.install_specs import install_specs_hash


class _Request:
    def __init__(self, query_params):
        self.query_params = query_params


class _BodyRequest:
    def __init__(self, body=None, path_params=None, query_params=None):
        self._body = body if body is not None else {}
        self.path_params = path_params or {}
        self.query_params = query_params or {}

    async def json(self):
        return self._body


def _json_response_payload(response):
    return json.loads(response.body.decode("utf-8"))


def _stub_marketplace_roots(monkeypatch, tmp_path):
    monkeypatch.setattr(marketplace_api, "_request_drive_root", lambda _req: tmp_path)
    monkeypatch.setattr(marketplace_api, "_request_repo_dir", lambda _req: tmp_path / "repo")


def _run_lifecycle_inline(monkeypatch):
    async def _fake_lifecycle_job(**kwargs):
        return await kwargs["runner"]()

    async def _fake_blocking(func, *args, **kwargs):
        kwargs.pop("log_label", None)
        return func(*args, **kwargs)

    monkeypatch.setattr(marketplace_api, "run_lifecycle_job", _fake_lifecycle_job)
    monkeypatch.setattr(marketplace_api, "run_blocking_preserving_cancellation", _fake_blocking)


def _patch_clawhub_install_pipeline(monkeypatch, tmp_path, *, auto_specs):
    from ouroboros.marketplace import install as install_mod
    from ouroboros.marketplace.adapter import AdapterResult

    monkeypatch.setattr(
        install_mod,
        "_registry_info",
        lambda slug: ClawHubSkillSummary(slug=slug, latest_version="1.0.0"),
    )
    monkeypatch.setattr(
        install_mod,
        "_registry_download",
        lambda slug, *, version=None: ClawHubArchive(slug=slug, version=version or "1.0.0", content=b"archive", sha256="sha"),
    )

    def _fake_stage(_content, *, slug, version, expected_sha256, staging_root):
        staging_dir = tmp_path / f"staging-{slug}-{version}"
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        staging_dir.mkdir(parents=True)
        (staging_dir / "SKILL.md").write_text(
            "---\n"
            f"name: {slug}\n"
            "description: Demo\n"
            "version: 1.0.0\n"
            "type: instruction\n"
            "when_to_use: Test skill.\n"
            "---\n"
            "Demo body.\n",
            encoding="utf-8",
        )
        (staging_dir / ".clawhub.json").write_text(
            json.dumps({"schema_version": 1, "source": "clawhub", "slug": slug, "sanitized_name": "demo"}),
            encoding="utf-8",
        )
        return StagedSkill(slug=slug, version=version, sha256=expected_sha256 or "sha", staging_dir=staging_dir)

    monkeypatch.setattr(install_mod, "_stage_archive", _fake_stage)
    monkeypatch.setattr(
        install_mod,
        "adapt_openclaw_skill",
        lambda *_args, **_kwargs: AdapterResult(
            ok=True,
            sanitized_name="demo",
            target_dirname="demo",
            provenance={
                "source": "clawhub",
                "slug": "demo",
                "sanitized_name": "demo",
                "install_specs": {
                    "schema_version": 1,
                    "auto": list(auto_specs),
                    "manual": [],
                    "specs_hash": install_specs_hash(auto_specs),
                },
            },
        ),
    )
    def reviewed(drive, _repo, name):
        from ouroboros.skill_loader import load_skill, save_review_state, SkillReviewState
        loaded = load_skill(drive / "skills" / "clawhub" / name, drive)
        save_review_state(drive, name, SkillReviewState(status="clean", content_hash=loaded.content_hash))
        return "clean", [], ""
    monkeypatch.setattr(install_mod, "_run_skill_review", reviewed)
    return install_mod


def test_marketplace_api_search_drops_params_with_query(monkeypatch):
    captured = {}

    def _fake_search(query, **kwargs):
        captured["query"] = query
        captured["kwargs"] = kwargs
        return {
            "results": [],
            "next_cursor": "",
            "path": "search",
            "attempts": [],
        }

    monkeypatch.setattr(marketplace_api, "_registry_search", _fake_search)
    response = asyncio.run(
        marketplace_api.api_marketplace_search(
            _Request(
                {
                    "q": "deep research",
                    "limit": "7",
                    "offset": "50",
                    "cursor": "abc",
                    "official": "1",
                }
            )
        )
    )

    assert response.status_code == 200
    assert captured["query"] == "deep research"
    assert captured["kwargs"]["limit"] == 7
    assert "offset" not in captured["kwargs"]
    assert captured["kwargs"]["cursor"] is None
    assert captured["kwargs"]["official_only"] is False
    assert captured["kwargs"]["timeout_sec"] == 15
    assert "enrich_search_results" not in captured["kwargs"]
    payload = _json_response_payload(response)
    assert payload["official"] is True
    assert payload["offset"] == 0
    assert payload["cursor"] is None
    assert payload["registry_path"] == "search"


def test_marketplace_api_search_filters_official_after_enrichment(monkeypatch):
    def _fake_search(query, **_kwargs):
        return {
            "results": [
                ClawHubSkillSummary(slug="official", badges={"official": True}),
                ClawHubSkillSummary(slug="community", badges={}),
            ],
            "next_cursor": "",
            "path": "search",
            "attempts": [],
        }

    monkeypatch.setattr(marketplace_api, "_registry_search", _fake_search)
    response = asyncio.run(
        marketplace_api.api_marketplace_search(
            _Request({"q": "deep research", "official": "1"})
        )
    )

    assert response.status_code == 200
    payload = _json_response_payload(response)
    assert payload["official"] is True
    assert [r["slug"] for r in payload["results"]] == ["official"]


def test_marketplace_api_browse_keeps_official_and_cursor(monkeypatch):
    captured = {}

    def _fake_search(query, **kwargs):
        captured["query"] = query
        captured["kwargs"] = kwargs
        return {
            "results": [],
            "next_cursor": "next",
            "path": "packages",
            "attempts": [],
        }

    monkeypatch.setattr(marketplace_api, "_registry_search", _fake_search)
    response = asyncio.run(
        marketplace_api.api_marketplace_search(
            _Request({"limit": "5", "cursor": "abc", "official": "1"})
        )
    )

    assert response.status_code == 200
    assert captured["query"] == ""
    assert captured["kwargs"]["limit"] == 5
    assert captured["kwargs"]["cursor"] == "abc"
    assert captured["kwargs"]["official_only"] is True
    assert captured["kwargs"]["timeout_sec"] == 5
    payload = _json_response_payload(response)
    assert payload["official"] is True
    assert payload["cursor"] == "abc"
    assert payload["next_cursor"] == "next"


def test_marketplace_installed_collision_does_not_read_lifecycle_provenance(
    monkeypatch, tmp_path
):
    from ouroboros.contracts.skill_manifest import SkillManifest
    from ouroboros.skill_loader import LoadedSkill

    skill_dir = tmp_path / "skills" / "clawhub" / "demo"
    skill_dir.mkdir(parents=True)
    collision = LoadedSkill(
        name="demo",
        skill_dir=skill_dir,
        manifest=SkillManifest(
            name="demo", description="", version="", type="instruction"
        ),
        content_hash="",
        load_error="Skill name collision: clawhub and user_repo",
        source="clawhub",
        identity_collision=True,
    )
    monkeypatch.setattr(
        "ouroboros.skill_loader.discover_skills",
        lambda *_args, **_kwargs: [collision],
    )
    monkeypatch.setattr(
        marketplace_api,
        "read_provenance",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("ambiguous identity must not read provenance")
        ),
    )

    rows = marketplace_api._installed_skills_for_source(tmp_path, "clawhub")

    assert len(rows) == 1
    assert rows[0]["name"] == "demo"
    assert rows[0]["load_error"] == collision.load_error
    assert rows[0]["provenance"] == {}
    assert not (tmp_path / "state").exists()


def test_clawhub_uninstall_clears_deps_state(tmp_path):
    from ouroboros.marketplace.install import uninstall_skill

    target = tmp_path / "skills" / "clawhub" / "demo"
    target.mkdir(parents=True)
    (target / ".clawhub.json").write_text("{}", encoding="utf-8")
    deps = tmp_path / "state" / "skills" / "demo" / "deps.json"
    deps.parent.mkdir(parents=True)
    deps.write_text(json.dumps({"status": "installed", "specs_hash": "abc"}), encoding="utf-8")

    result = uninstall_skill(tmp_path, sanitized_name="demo")

    assert result.ok
    assert not deps.exists()


def test_clawhub_fresh_install_rolls_back_payload_on_dependency_failure(monkeypatch, tmp_path):
    auto_specs = [{"kind": "pip", "package": "ddgs", "bins": []}]
    install_mod = _patch_clawhub_install_pipeline(monkeypatch, tmp_path, auto_specs=auto_specs)
    monkeypatch.setattr(
        "ouroboros.marketplace.isolated_deps.install_isolated_dependencies",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("pip failed")),
    )

    result = install_mod.install_skill(tmp_path, tmp_path / "repo", slug="demo", auto_review=True)

    assert result.ok is False
    assert result.deps_status == "failed"
    assert not (tmp_path / "skills" / "clawhub" / "demo").exists()
    assert not (tmp_path / "state" / "skills" / "demo" / "deps.json").exists()
    assert not (tmp_path / "state" / "skills" / "demo" / "clawhub.json").exists()


@pytest.mark.parametrize('basis', ['trusted', 'owner_skip'])
@pytest.mark.parametrize('floor_ok', [False, True])
@pytest.mark.parametrize('mode', ['advanced', 'cyber_pro'])
def test_registry_acceptance_floor_precedes_package_execution(monkeypatch, tmp_path, basis, floor_ok, mode):
    from ouroboros.skill_source_install import accept_and_turn_on

    monkeypatch.setenv('OUROBOROS_RUNTIME_MODE', mode)
    monkeypatch.setenv('OUROBOROS_SKILLS_REPO_PATH', '')
    storage = _patch_clawhub_install_pipeline(monkeypatch, tmp_path,
        auto_specs=[{'kind': 'pip', 'package': 'ddgs', 'bins': []}])
    order = []
    monkeypatch.setattr('ouroboros.marketplace.isolated_deps._install_python_packages',
        lambda *args, **kwargs: order.append('deps') or [])

    def floor(*args, **kwargs):
        order.append('preflight')
        return {'ok': floor_ok, 'error': '' if floor_ok else 'controlled invalid payload'}

    monkeypatch.setattr('ouroboros.skill_review.run_acceptance_preflight', floor)
    result = storage.install_skill(tmp_path, tmp_path / 'repo', slug='demo', auto_review=False,
        install_record={'source': {'id': 'clawhub', 'kind': 'registry', 'name': 'ClawHub'},
                        'basis': basis, 'official': False, 'turn_on_when_ready': False})
    assert result.ok and order == [], 'landing must not execute packages before the acceptance floor'
    outcome = accept_and_turn_on(tmp_path, 'demo', repo_path='', repo_dir=tmp_path / 'repo')
    assert order == (['preflight', 'deps'] if floor_ok else ['preflight'])
    assert outcome['accepted'] is floor_ok
    if not floor_ok:
        assert outcome['setup_error'] and not (tmp_path / 'state/skills/demo/deps.json').exists()


def test_clawhub_target_override_rejects_foreign_identity_before_landing(monkeypatch, tmp_path):
    install_mod = _patch_clawhub_install_pipeline(monkeypatch, tmp_path, auto_specs=[])
    checkout = tmp_path / "checkout"
    foreign = checkout / "demo"
    foreign.mkdir(parents=True)
    (foreign / "SKILL.md").write_text(
        "---\nname: demo\ndescription: User copy\nversion: 1.0.0\ntype: skill\n---\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(checkout))

    result = install_mod.install_skill(
        tmp_path,
        tmp_path / "repo",
        slug="demo",
        auto_review=True,
        overwrite=True,
        target_name_override="demo",
    )

    assert result.ok is False
    assert "collision" in result.error.lower()
    assert not (tmp_path / "skills" / "clawhub" / "demo").exists()
    assert not (tmp_path / "state" / "skills" / "demo").exists()


def test_clawhub_fresh_install_purely_dedupes_foreign_user_repo(monkeypatch, tmp_path):
    install_mod = _patch_clawhub_install_pipeline(monkeypatch, tmp_path, auto_specs=[])
    checkout = tmp_path / "checkout"
    foreign = checkout / "demo"
    foreign.mkdir(parents=True)
    (foreign / "SKILL.md").write_text(
        "---\nname: demo\ndescription: User copy\nversion: 1.0.0\ntype: skill\n---\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(checkout))

    result = install_mod.install_skill(
        tmp_path,
        tmp_path / "repo",
        slug="demo",
        auto_review=False,
    )

    assert result.ok is True
    assert result.sanitized_name == "demo-clawhub"
    assert (tmp_path / "skills" / "clawhub" / "demo-clawhub" / "SKILL.md").is_file()
    assert not (tmp_path / "state" / "skills" / "demo").exists()


def test_clawhub_update_rejects_collision_before_provenance_or_unload(monkeypatch, tmp_path):
    from ouroboros.marketplace import install as install_mod
    from ouroboros.marketplace.provenance import write_provenance

    target = tmp_path / "skills" / "clawhub" / "demo"
    target.mkdir(parents=True)
    original = "---\nname: demo\n---\nold payload\n"
    (target / "SKILL.md").write_text(original, encoding="utf-8")
    write_provenance(
        tmp_path,
        "demo",
        {"schema_version": 1, "source": "clawhub", "slug": "demo"},
    )
    checkout = tmp_path / "checkout"
    foreign = checkout / "demo"
    foreign.mkdir(parents=True)
    (foreign / "SKILL.md").write_text("---\nname: demo\n---\n", encoding="utf-8")
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(checkout))
    monkeypatch.setattr(
        "ouroboros.extension_loader.is_extension_live",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("collision must block before unload")
        ),
    )

    result = install_mod.update_skill(
        tmp_path,
        tmp_path / "repo",
        sanitized_name="demo",
    )

    assert result.ok is False
    assert "collision" in result.error.lower()
    assert (target / "SKILL.md").read_text(encoding="utf-8") == original


def test_clawhub_landing_failure_restores_previous_payload(monkeypatch, tmp_path):
    auto_specs = []
    install_mod = _patch_clawhub_install_pipeline(monkeypatch, tmp_path, auto_specs=auto_specs)
    target = tmp_path / "skills" / "clawhub" / "demo"
    target.mkdir(parents=True)
    (target / "SKILL.md").write_text("old payload", encoding="utf-8")
    provenance = tmp_path / "state" / "skills" / "demo" / "clawhub.json"
    provenance.parent.mkdir(parents=True)
    provenance.write_text(
        json.dumps({"source": "clawhub", "slug": "demo", "sanitized_name": "demo", "old": True}),
        encoding="utf-8",
    )

    def fake_land(_staged, target_dir, *, overwrite):
        assert overwrite is True
        shutil.rmtree(target_dir)
        raise RuntimeError("partial landing failed")

    monkeypatch.setattr(install_mod, "_land_staged_into_data_plane", fake_land)

    result = install_mod.install_skill(
        tmp_path,
        tmp_path / "repo",
        slug="demo",
        auto_review=False,
        overwrite=True,
    )

    assert result.ok is False
    assert "Could not land skill into data plane" in result.error
    assert (target / "SKILL.md").read_text(encoding="utf-8") == "old payload"
    assert json.loads(provenance.read_text(encoding="utf-8"))["old"] is True


def test_clawhub_retry_accepts_existing_valid_dependency_fingerprint(monkeypatch, tmp_path):
    auto_specs = [{"kind": "pip", "package": "ddgs", "bins": []}]
    specs_hash = install_specs_hash(auto_specs)
    install_mod = _patch_clawhub_install_pipeline(monkeypatch, tmp_path, auto_specs=auto_specs)
    target = tmp_path / "skills" / "clawhub" / "demo"
    env_root = target / ".ouroboros_env"
    env_root.mkdir(parents=True)
    (target / "SKILL.md").write_text(
        "---\nname: demo\ndescription: Demo\nversion: 1.0.0\ntype: instruction\n"
        "install_specs:\n  - kind: pip\n    package: ddgs\n---\nInstalled.\n", encoding="utf-8")
    (target / ".clawhub.json").write_text(
        json.dumps({"schema_version": 1, "source": "clawhub", "slug": "demo", "sanitized_name": "demo",
                    "install_specs": {"auto": auto_specs}}),
        encoding="utf-8",
    )
    fingerprint = {"status": "installed", "specs_hash": specs_hash}
    (env_root / "fingerprint.json").write_text(json.dumps(fingerprint), encoding="utf-8")
    deps = tmp_path / "state" / "skills" / "demo" / "deps.json"
    deps.parent.mkdir(parents=True)
    deps.write_text(json.dumps({"status": "failed", "specs_hash": specs_hash}), encoding="utf-8")
    (tmp_path / "state" / "skills" / "demo" / "clawhub.json").write_text(
        json.dumps({"source": "clawhub", "slug": "demo", "sanitized_name": "demo"}),
        encoding="utf-8",
    )
    called = {"deps": 0}
    monkeypatch.setattr(
        "ouroboros.marketplace.isolated_deps.install_isolated_dependencies",
        lambda *_args, **_kwargs: called.__setitem__("deps", called["deps"] + 1),
    )

    result = install_mod.install_skill(tmp_path, tmp_path / "repo", slug="demo", auto_review=True)

    assert result.ok is True
    assert result.deps_status == "installed"
    assert result.deps_fingerprint["specs_hash"] == specs_hash
    assert called["deps"] == 0
    persisted = json.loads(deps.read_text(encoding="utf-8"))
    assert persisted["status"] == "installed"
    assert persisted["specs_hash"] == specs_hash


def test_clawhub_retry_repairs_partial_payload_missing_sidecar(monkeypatch, tmp_path):
    auto_specs = [{"kind": "pip", "package": "ddgs", "bins": []}]
    specs_hash = install_specs_hash(auto_specs)
    install_mod = _patch_clawhub_install_pipeline(monkeypatch, tmp_path, auto_specs=auto_specs)
    target = tmp_path / "skills" / "clawhub" / "demo"
    env_root = target / ".ouroboros_env"
    env_root.mkdir(parents=True)
    (target / "SKILL.md").write_text("partial install", encoding="utf-8")
    fingerprint = {"status": "installed", "specs_hash": specs_hash}
    (env_root / "fingerprint.json").write_text(json.dumps(fingerprint), encoding="utf-8")
    deps = tmp_path / "state" / "skills" / "demo" / "deps.json"
    deps.parent.mkdir(parents=True)
    deps.write_text(json.dumps({"status": "failed", "specs_hash": specs_hash}), encoding="utf-8")
    called = {"deps": 0}

    def fake_install_deps(*_args, **_kwargs):
        called["deps"] += 1
        return {"status": "installed", "specs_hash": specs_hash}

    monkeypatch.setattr("ouroboros.marketplace.isolated_deps.install_isolated_dependencies", fake_install_deps)

    result = install_mod.install_skill(tmp_path, tmp_path / "repo", slug="demo", auto_review=True)

    assert result.ok is True
    assert result.deps_status == "installed"
    assert called["deps"] == 1
    assert (target / ".clawhub.json").is_file()
    assert "Demo body." in (target / "SKILL.md").read_text(encoding="utf-8")


def test_a_blocked_review_never_queues_an_automatic_repair(monkeypatch, tmp_path):
    """Fixing a skill is the owner's explicit request; an install that reviews to blockers queues nothing."""
    _stub_marketplace_roots(monkeypatch, tmp_path)
    _run_lifecycle_inline(monkeypatch)
    sent = []
    monkeypatch.setattr("supervisor.message_bus.get_bridge", lambda: SimpleNamespace(
        ui_send=lambda *a, **k: sent.append(a), broadcast=lambda *a, **k: sent.append(a)))
    blocked = SimpleNamespace(
        ok=True, sanitized_name="demo", error="", target_dir=None, summary=None, archive=None, adapter=None,
        review_status="blockers", review_findings=[{"verdict": "FAIL"}], review_error="", deps_status="",
        deps_error="", deps_fingerprint={}, provenance={})
    monkeypatch.setattr(marketplace_api, "install_skill", lambda *a, **k: blocked)
    response = asyncio.run(marketplace_api.api_marketplace_install(_BodyRequest({"slug": "demo"})))
    assert response.status_code == 200 and _json_response_payload(response)["review_status"] == "blockers"
    assert sent == [] and not (tmp_path / "state" / "skills" / "demo" / "auto_repair.json").exists()
    assert not any(name.startswith("_maybe_enqueue") for name in vars(marketplace_api))
