"""Tests for the ClawHub marketplace HTTP API adapter layer."""

from __future__ import annotations

import asyncio
import json
import shutil
from types import SimpleNamespace

from ouroboros.gateway import marketplace as marketplace_api
from ouroboros.marketplace.clawhub import ClawHubArchive, ClawHubSkillSummary
from ouroboros.marketplace.fetcher import StagedSkill
from ouroboros.marketplace.install_specs import install_specs_hash
from ouroboros.marketplace.ouroboroshub import HubInstallResult, HubSkillSummary


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
            "type: skill\n"
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
    monkeypatch.setattr(install_mod, "_run_skill_review", lambda *_args, **_kwargs: ("clean", [], ""))
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


def test_ouroboroshub_install_response_shape_after_review_and_deps(monkeypatch, tmp_path):
    _stub_marketplace_roots(monkeypatch, tmp_path)
    _run_lifecycle_inline(monkeypatch)

    target_dir = tmp_path / "skills" / "ouroboroshub" / "demo"
    target_dir.mkdir(parents=True)
    (target_dir / ".ouroboroshub.json").write_text("{}", encoding="utf-8")
    summary = HubSkillSummary(slug="demo", name="Demo", version="1.0.0")
    provenance = {"source": "ouroboroshub", "slug": "demo"}

    def _fake_install(slug, *, overwrite=False):
        assert slug == "demo"
        assert overwrite is False
        return HubInstallResult(
            ok=True,
            sanitized_name="demo",
            target_dir=target_dir,
            summary=summary,
            provenance=provenance,
        )

    monkeypatch.setattr(marketplace_api.ouroboroshub, "install", _fake_install)
    monkeypatch.setattr(
        marketplace_api,
        "_run_skill_review",
        lambda _drive, _repo, name: ("clean", [{"message": "ok"}], ""),
    )
    monkeypatch.setattr(
        marketplace_api,
        "_reconcile_deps_after_review",
        lambda _drive, name: ("installed", ""),
    )

    response = asyncio.run(
        marketplace_api.api_ouroboroshub_install(
            _BodyRequest({"slug": "demo", "auto_review": True})
        )
    )

    assert response.status_code == 200
    assert _json_response_payload(response) == {
        "ok": True,
        "sanitized_name": "demo",
        "error": "",
        "provenance": provenance,
        "summary": summary.to_dict(),
        "target_dir": str(target_dir),
        "review_status": "clean",
        "review_findings": [{"message": "ok"}],
        "review_error": "",
        "deps_status": "installed",
        "deps_error": "",
    }


def test_ouroboroshub_api_rejects_foreign_identity_before_lifecycle_job(
    monkeypatch, tmp_path
):
    _stub_marketplace_roots(monkeypatch, tmp_path)
    checkout = tmp_path / "checkout"
    foreign = checkout / "demo"
    foreign.mkdir(parents=True)
    (foreign / "SKILL.md").write_text("---\nname: demo\n---\n", encoding="utf-8")
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(checkout))

    async def _unexpected_job(**_kwargs):
        raise AssertionError("lifecycle job must not start for ambiguous identity")

    monkeypatch.setattr(marketplace_api, "run_lifecycle_job", _unexpected_job)
    monkeypatch.setattr(
        marketplace_api.ouroboroshub,
        "install",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("installer must not run for ambiguous identity")
        ),
    )

    response = asyncio.run(
        marketplace_api.api_ouroboroshub_install(_BodyRequest({"slug": "demo"}))
    )

    assert response.status_code == 409
    assert "collision" in _json_response_payload(response)["error"].lower()
    assert not (tmp_path / "state" / "skills" / "demo").exists()


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
        install_mod,
        "install_isolated_dependencies",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("pip failed")),
    )

    result = install_mod.install_skill(tmp_path, tmp_path / "repo", slug="demo", auto_review=True)

    assert result.ok is False
    assert result.deps_status == "failed"
    assert not (tmp_path / "skills" / "clawhub" / "demo").exists()
    assert not (tmp_path / "state" / "skills" / "demo" / "deps.json").exists()
    assert not (tmp_path / "state" / "skills" / "demo" / "clawhub.json").exists()


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
    (target / "SKILL.md").write_text("installed", encoding="utf-8")
    (target / ".clawhub.json").write_text(
        json.dumps({"schema_version": 1, "source": "clawhub", "slug": "demo", "sanitized_name": "demo"}),
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
        install_mod,
        "install_isolated_dependencies",
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

    monkeypatch.setattr(install_mod, "install_isolated_dependencies", fake_install_deps)

    result = install_mod.install_skill(tmp_path, tmp_path / "repo", slug="demo", auto_review=True)

    assert result.ok is True
    assert result.deps_status == "installed"
    assert called["deps"] == 1
    assert (target / ".clawhub.json").is_file()
    assert "Demo body." in (target / "SKILL.md").read_text(encoding="utf-8")


def test_marketplace_auto_repair_enqueues_once_per_payload_hash(monkeypatch, tmp_path):
    skill_dir = tmp_path / "skills" / "clawhub" / "demo"
    skill_dir.mkdir(parents=True)
    skill = SimpleNamespace(name="demo", content_hash="abcdef123456", skill_dir=skill_dir)
    calls = []
    broadcasts = []

    class Bridge:
        def ui_send(self, text, **kwargs):
            calls.append((text, kwargs))

        def broadcast(self, payload):
            broadcasts.append(payload)

    monkeypatch.setattr("ouroboros.config.get_skills_repo_path", lambda: "")
    monkeypatch.setattr("ouroboros.skill_loader.find_skill", lambda *_args, **_kwargs: skill)
    monkeypatch.setattr("supervisor.message_bus.get_bridge", lambda: Bridge())

    first = marketplace_api._maybe_enqueue_marketplace_auto_repair(
        tmp_path,
        skill_name="demo",
        source="clawhub",
        reason="review blockers",
        review_findings=[{"message": "fix me"}],
    )
    second = marketplace_api._maybe_enqueue_marketplace_auto_repair(
        tmp_path,
        skill_name="demo",
        source="clawhub",
        reason="review blockers",
        review_findings=[{"message": "fix me"}],
    )

    assert first is True
    assert second is False
    assert len(calls) == 1
    assert calls[0][1]["task_constraint"]["mode"] == "normal"
    assert calls[0][1]["task_constraint"]["allow_review"] is True
    assert calls[0][1]["task_constraint"]["allow_enable"] is False
    assert broadcasts and broadcasts[0]["system_type"] == "skill_repair"
    marker = json.loads((tmp_path / "state" / "skills" / "demo" / "auto_repair.json").read_text(encoding="utf-8"))
    assert marker["attempted_hashes"] == ["abcdef123456"]


def test_ouroboroshub_update_rejects_missing_install(monkeypatch, tmp_path):
    _stub_marketplace_roots(monkeypatch, tmp_path)
    _run_lifecycle_inline(monkeypatch)
    called = {"install": 0}

    monkeypatch.setattr(
        marketplace_api.ouroboroshub,
        "install",
        lambda *_args, **_kwargs: called.__setitem__("install", called["install"] + 1),
    )

    response = asyncio.run(
        marketplace_api.api_ouroboroshub_update(
            _BodyRequest(path_params={"name": "demo"})
        )
    )

    assert response.status_code == 400
    assert called["install"] == 0
    assert _json_response_payload(response) == {
        "ok": False,
        "sanitized_name": "demo",
        "error": "demo is not installed",
        "provenance": {},
        "summary": None,
    }


def test_ouroboroshub_update_rejects_unmarked_payload(monkeypatch, tmp_path):
    _stub_marketplace_roots(monkeypatch, tmp_path)
    _run_lifecycle_inline(monkeypatch)
    target_dir = tmp_path / "skills" / "ouroboroshub" / "demo"
    target_dir.mkdir(parents=True)
    called = {"install": 0}

    monkeypatch.setattr(
        marketplace_api.ouroboroshub,
        "install",
        lambda *_args, **_kwargs: called.__setitem__("install", called["install"] + 1),
    )

    response = asyncio.run(
        marketplace_api.api_ouroboroshub_update(
            _BodyRequest(path_params={"name": "demo"})
        )
    )

    assert response.status_code == 400
    assert called["install"] == 0
    assert _json_response_payload(response) == {
        "ok": False,
        "sanitized_name": "demo",
        "error": "missing OuroborosHub provenance marker",
        "provenance": {},
        "summary": None,
        "target_dir": str(target_dir),
    }


def test_ouroboroshub_update_rejects_wrong_provenance_marker(monkeypatch, tmp_path):
    _stub_marketplace_roots(monkeypatch, tmp_path)
    _run_lifecycle_inline(monkeypatch)
    target_dir = tmp_path / "skills" / "ouroboroshub" / "demo"
    target_dir.mkdir(parents=True)
    (target_dir / ".ouroboroshub.json").write_text(
        json.dumps({"source": "clawhub", "slug": "demo", "sanitized_name": "demo"}),
        encoding="utf-8",
    )
    called = {"install": 0}

    monkeypatch.setattr(
        marketplace_api.ouroboroshub,
        "install",
        lambda *_args, **_kwargs: called.__setitem__("install", called["install"] + 1),
    )

    response = asyncio.run(
        marketplace_api.api_ouroboroshub_update(
            _BodyRequest(path_params={"name": "demo"})
        )
    )

    assert response.status_code == 400
    assert called["install"] == 0
    assert _json_response_payload(response) == {
        "ok": False,
        "sanitized_name": "demo",
        "error": "invalid OuroborosHub provenance marker",
        "provenance": {},
        "summary": None,
        "target_dir": str(target_dir),
    }


def test_ouroboroshub_update_response_shape_on_dependency_failure(monkeypatch, tmp_path):
    _stub_marketplace_roots(monkeypatch, tmp_path)
    _run_lifecycle_inline(monkeypatch)

    target_dir = tmp_path / "skills" / "ouroboroshub" / "demo"
    target_dir.mkdir(parents=True)
    (target_dir / "old.txt").write_text("old-live-version", encoding="utf-8")
    (target_dir / ".ouroboroshub.json").write_text(
        json.dumps({
            "schema_version": 1,
            "source": "ouroboroshub",
            "slug": "demo",
            "sanitized_name": "demo",
        }),
        encoding="utf-8",
    )
    summary = HubSkillSummary(slug="demo", name="Demo", version="1.0.0")
    provenance = {"source": "ouroboroshub", "slug": "demo"}

    monkeypatch.setattr(
        "ouroboros.extension_loader.is_extension_live",
        lambda _name, _drive: False,
    )
    monkeypatch.setattr("ouroboros.extension_loader.unload_extension", lambda _name: None)

    def _fake_install(slug, *, overwrite=False):
        assert slug == "demo"
        assert overwrite is True
        shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True)
        (target_dir / ".ouroboroshub.json").write_text(
            json.dumps({
                "schema_version": 1,
                "source": "ouroboroshub",
                "slug": "demo",
                "sanitized_name": "demo",
            }),
            encoding="utf-8",
        )
        (target_dir / "new.txt").write_text("new-broken-version", encoding="utf-8")
        return HubInstallResult(
            ok=True,
            sanitized_name="demo",
            target_dir=target_dir,
            summary=summary,
            provenance=provenance,
        )

    monkeypatch.setattr(
        marketplace_api.ouroboroshub,
        "install",
        _fake_install,
    )
    monkeypatch.setattr(
        marketplace_api,
        "_run_skill_review",
        lambda _drive, _repo, name: ("clean", [{"message": "ok"}], ""),
    )
    monkeypatch.setattr(
        marketplace_api,
        "_reconcile_deps_after_review",
        lambda _drive, name: ("failed", "dependency boom"),
    )

    response = asyncio.run(
        marketplace_api.api_ouroboroshub_update(
            _BodyRequest(path_params={"name": "demo"})
        )
    )

    assert response.status_code == 400
    assert _json_response_payload(response) == {
        "ok": False,
        "sanitized_name": "demo",
        "error": "dependency boom",
        "provenance": provenance,
        "summary": summary.to_dict(),
        "target_dir": str(target_dir),
        "review_status": "clean",
        "review_findings": [{"message": "ok"}],
        "review_error": "",
        "deps_status": "failed",
        "deps_error": "dependency boom",
        "rolled_back": True,
    }
    assert (target_dir / "old.txt").read_text(encoding="utf-8") == "old-live-version"
    assert not (target_dir / "new.txt").exists()


def test_ouroboroshub_update_exception_after_unload_restores_live_payload(monkeypatch, tmp_path):
    _stub_marketplace_roots(monkeypatch, tmp_path)
    _run_lifecycle_inline(monkeypatch)

    target_dir = tmp_path / "skills" / "ouroboroshub" / "demo"
    target_dir.mkdir(parents=True)
    marker = {
        "schema_version": 1,
        "source": "ouroboroshub",
        "slug": "demo",
        "sanitized_name": "demo",
    }
    (target_dir / "old.txt").write_text("old-live-version", encoding="utf-8")
    (target_dir / ".ouroboroshub.json").write_text(json.dumps(marker), encoding="utf-8")
    calls = {"unload": 0, "reconcile": 0}

    monkeypatch.setattr("ouroboros.extension_loader.is_extension_live", lambda _name, _drive: True)
    monkeypatch.setattr("ouroboros.extension_loader.unload_extension", lambda _name: calls.__setitem__("unload", calls["unload"] + 1))
    monkeypatch.setattr(
        "ouroboros.extension_loader.reconcile_extension",
        lambda *_args, **_kwargs: calls.__setitem__("reconcile", calls["reconcile"] + 1) or {"action": "loaded"},
    )

    def _fake_install(_slug, *, overwrite=False):
        assert overwrite is True
        shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True)
        (target_dir / "new.txt").write_text("new-broken-version", encoding="utf-8")
        raise RuntimeError("download exploded")

    monkeypatch.setattr(marketplace_api.ouroboroshub, "install", _fake_install)

    response = asyncio.run(
        marketplace_api.api_ouroboroshub_update(
            _BodyRequest(path_params={"name": "demo"})
        )
    )

    payload = _json_response_payload(response)
    assert response.status_code == 400
    assert payload["ok"] is False
    assert payload["rolled_back"] is True
    assert "download exploded" in payload["error"]
    assert calls == {"unload": 1, "reconcile": 1}
    assert (target_dir / "old.txt").read_text(encoding="utf-8") == "old-live-version"
    assert not (target_dir / "new.txt").exists()


def test_ouroboroshub_update_review_blockers_restore_live_payload(monkeypatch, tmp_path):
    _stub_marketplace_roots(monkeypatch, tmp_path)
    _run_lifecycle_inline(monkeypatch)

    target_dir = tmp_path / "skills" / "ouroboroshub" / "demo"
    target_dir.mkdir(parents=True)
    marker = {
        "schema_version": 1,
        "source": "ouroboroshub",
        "slug": "demo",
        "sanitized_name": "demo",
    }
    (target_dir / "old.txt").write_text("old-live-version", encoding="utf-8")
    (target_dir / ".ouroboroshub.json").write_text(json.dumps(marker), encoding="utf-8")
    calls = {"unload": 0, "reconcile": 0}
    summary = HubSkillSummary(slug="demo", name="Demo", version="1.0.0")

    monkeypatch.setattr("ouroboros.extension_loader.is_extension_live", lambda _name, _drive: True)
    monkeypatch.setattr("ouroboros.extension_loader.unload_extension", lambda _name: calls.__setitem__("unload", calls["unload"] + 1))
    monkeypatch.setattr(
        "ouroboros.extension_loader.reconcile_extension",
        lambda *_args, **_kwargs: calls.__setitem__("reconcile", calls["reconcile"] + 1) or {"action": "loaded"},
    )

    def _fake_install(_slug, *, overwrite=False):
        assert overwrite is True
        shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True)
        (target_dir / ".ouroboroshub.json").write_text(json.dumps(marker), encoding="utf-8")
        (target_dir / "new.txt").write_text("new-blocked-version", encoding="utf-8")
        return HubInstallResult(
            ok=True,
            sanitized_name="demo",
            target_dir=target_dir,
            summary=summary,
            provenance=marker,
        )

    monkeypatch.setattr(marketplace_api.ouroboroshub, "install", _fake_install)
    monkeypatch.setattr(
        marketplace_api,
        "_run_skill_review",
        lambda _drive, _repo, name: ("blockers", [{"message": "blocked"}], ""),
    )

    response = asyncio.run(
        marketplace_api.api_ouroboroshub_update(
            _BodyRequest(path_params={"name": "demo"})
        )
    )

    payload = _json_response_payload(response)
    assert response.status_code == 400
    assert payload["review_status"] == "blockers"
    assert payload["rolled_back"] is True
    assert calls == {"unload": 1, "reconcile": 1}
    assert (target_dir / "old.txt").read_text(encoding="utf-8") == "old-live-version"
    assert not (target_dir / "new.txt").exists()


def test_uninstall_handler_waits_for_the_queue_lock_off_the_event_loop(monkeypatch, tmp_path):
    """While the post-uninstall schedule resync waits for a held supervisor
    queue lock, another request (GET /api/widgets) completes on the same loop;
    the resync still runs, after the uninstall step, on a worker thread."""
    import threading

    import supervisor.queue as supervisor_queue
    from ouroboros.gateway.widgets import api_widgets

    _stub_marketplace_roots(monkeypatch, tmp_path)
    _run_lifecycle_inline(monkeypatch)
    order: list[str] = []
    resync_threads: list[int] = []
    entered = threading.Event()
    held = threading.Event()
    release = threading.Event()
    holding = {"value": False}

    def uninstall_skill(_drive_root, *, sanitized_name):
        order.append("uninstall")
        return SimpleNamespace(ok=True, sanitized_name=sanitized_name, error="")

    def resync_skill_schedules(_drive_root=None):
        resync_threads.append(threading.get_ident())
        entered.set()
        with supervisor_queue._queue_lock:  # the real mirror waits here too
            order.append("resync")

    monkeypatch.setattr(marketplace_api, "uninstall_skill", uninstall_skill)
    monkeypatch.setattr(supervisor_queue, "resync_skill_schedules", resync_skill_schedules)

    def hold_lock():
        with supervisor_queue._queue_lock:
            holding["value"] = True
            held.set()
            release.wait(timeout=10)
            holding["value"] = False

    async def scenario():
        loop_thread = threading.get_ident()
        uninstall = asyncio.create_task(
            marketplace_api.api_marketplace_uninstall(_BodyRequest(path_params={"name": "demo"}))
        )
        await asyncio.get_running_loop().run_in_executor(None, entered.wait, 5)
        assert entered.is_set(), "the handler never reached the schedule resync"
        other = await asyncio.wait_for(api_widgets(None), 5)
        still_held = holding["value"]
        release.set()
        response = await uninstall
        return loop_thread, other, still_held, response

    holder = threading.Thread(target=hold_lock, daemon=True)
    holder.start()
    assert held.wait(timeout=5)
    try:
        loop_thread, other, still_held, response = asyncio.run(scenario())
    finally:
        release.set()
        holder.join(timeout=5)

    assert other.status_code == 200
    assert still_held, "the other request finished only after the lock was released: the loop was blocked"
    assert response.status_code == 200
    assert _json_response_payload(response)["ok"] is True
    assert order == ["uninstall", "resync"]
    assert resync_threads and resync_threads[0] != loop_thread
