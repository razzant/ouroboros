from __future__ import annotations

import pathlib
import shutil
import json

from ouroboros.marketplace import ouroboroshub


def test_ouroboroshub_stages_under_target_root(monkeypatch, tmp_path):
    hub_root = tmp_path / "hub"
    monkeypatch.setattr(ouroboroshub, "get_ouroboroshub_skills_dir", lambda: hub_root)
    summary = ouroboroshub.HubSkillSummary(slug="demo", name="demo", version="1.0.0", files=[{"path": "SKILL.md", "sha256": "x", "size": 1}])
    monkeypatch.setattr(ouroboroshub, "load_catalog", lambda: {"raw_base_url": "https://raw.githubusercontent.com/razzant/OuroborosHub/main"})
    monkeypatch.setattr(ouroboroshub, "_summaries", lambda _catalog: [summary])
    seen = {}

    def fake_download(_summary, _raw_base, staging_dir):
        seen["staging"] = pathlib.Path(staging_dir)
        (staging_dir / "SKILL.md").write_text("---\nname: demo\n---\n", encoding="utf-8")

    monkeypatch.setattr(ouroboroshub, "_download_skill_files", fake_download)
    result = ouroboroshub.install("demo")
    assert result.ok
    seen["staging"].relative_to(hub_root / ".staging")


def test_ouroboroshub_rejects_foreign_identity_before_download(monkeypatch, tmp_path):
    hub_root = tmp_path / "data" / "skills" / "ouroboroshub"
    checkout = tmp_path / "checkout"
    foreign = checkout / "demo"
    foreign.mkdir(parents=True)
    (foreign / "SKILL.md").write_text("---\nname: demo\n---\n", encoding="utf-8")
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(checkout))
    monkeypatch.setattr(ouroboroshub, "get_ouroboroshub_skills_dir", lambda: hub_root)
    summary = ouroboroshub.HubSkillSummary(
        slug="demo", name="demo", version="1.0.0",
        files=[{"path": "SKILL.md", "sha256": "x", "size": 1}],
    )
    monkeypatch.setattr(
        ouroboroshub,
        "load_catalog",
        lambda: {"raw_base_url": "https://raw.githubusercontent.com/razzant/OuroborosHub/main"},
    )
    monkeypatch.setattr(ouroboroshub, "_summaries", lambda _catalog: [summary])
    monkeypatch.setattr(
        ouroboroshub,
        "_download_skill_files",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("download must not run")),
    )

    result = ouroboroshub.install("demo")

    assert result.ok is False
    assert "collision" in result.error.lower()
    assert not (hub_root / "demo").exists()
    assert not (tmp_path / "data" / "state" / "skills" / "demo").exists()


def test_ouroboroshub_persists_catalog_dependency_specs(monkeypatch, tmp_path):
    hub_root = tmp_path / "hub"
    monkeypatch.setattr(ouroboroshub, "get_ouroboroshub_skills_dir", lambda: hub_root)
    summary = ouroboroshub.HubSkillSummary(
        slug="duckduckgo",
        name="duckduckgo",
        version="1.0.0",
        files=[{"path": "SKILL.md", "sha256": "x", "size": 1}],
        install_specs=[{"kind": "pip", "package": "ddgs"}],
    )
    monkeypatch.setattr(ouroboroshub, "load_catalog", lambda: {"raw_base_url": "https://raw.githubusercontent.com/razzant/OuroborosHub/main"})
    monkeypatch.setattr(ouroboroshub, "_summaries", lambda _catalog: [summary])

    def fake_download(_summary, _raw_base, staging_dir):
        (staging_dir / "SKILL.md").write_text("---\nname: duckduckgo\n---\n", encoding="utf-8")

    monkeypatch.setattr(ouroboroshub, "_download_skill_files", fake_download)

    result = ouroboroshub.install("duckduckgo")

    assert result.ok
    assert result.provenance["install_specs"]["auto"][0]["package"] == "ddgs"
    assert (hub_root / "duckduckgo" / ".ouroboroshub.json").is_file()


def test_ouroboroshub_preserves_dict_dependency_specs(monkeypatch, tmp_path):
    hub_root = tmp_path / "hub"
    monkeypatch.setattr(ouroboroshub, "get_ouroboroshub_skills_dir", lambda: hub_root)
    summary = ouroboroshub.HubSkillSummary(
        slug="duckduckgo",
        name="duckduckgo",
        version="1.0.0",
        files=[{"path": "SKILL.md", "sha256": "x", "size": 1}],
        install_specs={"python": ["ddgs"]},
    )
    monkeypatch.setattr(ouroboroshub, "load_catalog", lambda: {"raw_base_url": "https://raw.githubusercontent.com/razzant/OuroborosHub/main"})
    monkeypatch.setattr(ouroboroshub, "_summaries", lambda _catalog: [summary])

    def fake_download(_summary, _raw_base, staging_dir):
        (staging_dir / "SKILL.md").write_text("---\nname: duckduckgo\n---\n", encoding="utf-8")

    monkeypatch.setattr(ouroboroshub, "_download_skill_files", fake_download)

    result = ouroboroshub.install("duckduckgo")

    assert result.ok
    assert result.provenance["install_specs"]["auto"][0]["package"] == "ddgs"
    assert summary.to_dict()["install_specs"] == {"python": ["ddgs"]}


def test_ouroboroshub_retry_requires_valid_marker_for_installed_fast_path(monkeypatch, tmp_path):
    hub_root = tmp_path / "data" / "skills" / "ouroboroshub"
    monkeypatch.setattr(ouroboroshub, "get_ouroboroshub_skills_dir", lambda: hub_root)
    summary = ouroboroshub.HubSkillSummary(
        slug="duckduckgo",
        name="duckduckgo",
        version="1.0.0",
        files=[{"path": "SKILL.md", "sha256": "x", "size": 1}],
        install_specs=[{"kind": "pip", "package": "ddgs"}],
    )
    monkeypatch.setattr(ouroboroshub, "load_catalog", lambda: {"raw_base_url": "https://raw.githubusercontent.com/razzant/OuroborosHub/main"})
    monkeypatch.setattr(ouroboroshub, "_summaries", lambda _catalog: [summary])
    auto_specs, _manual_specs, _warnings = ouroboroshub.normalize_declared_dependency_specs(summary.install_specs)
    specs_hash = ouroboroshub.install_specs_hash(auto_specs)
    target = hub_root / "duckduckgo"
    env_root = target / ".ouroboros_env"
    env_root.mkdir(parents=True)
    (target / "SKILL.md").write_text("installed", encoding="utf-8")
    (env_root / "fingerprint.json").write_text(json.dumps({"status": "installed", "specs_hash": specs_hash}), encoding="utf-8")
    deps = tmp_path / "data" / "state" / "skills" / "duckduckgo" / "deps.json"
    deps.parent.mkdir(parents=True)
    deps.write_text(json.dumps({"status": "failed", "specs_hash": specs_hash}), encoding="utf-8")

    def fake_download(_summary, _raw_base, staging_dir):
        (staging_dir / "SKILL.md").write_text("---\nname: duckduckgo\n---\n", encoding="utf-8")

    monkeypatch.setattr(ouroboroshub, "_download_skill_files", fake_download)

    result = ouroboroshub.install("duckduckgo")

    assert result.ok is True
    assert result.provenance["source"] == "ouroboroshub"
    assert (target / ".ouroboroshub.json").is_file()


def test_ouroboroshub_retry_accepts_valid_marker_fast_path(monkeypatch, tmp_path):
    hub_root = tmp_path / "data" / "skills" / "ouroboroshub"
    monkeypatch.setattr(ouroboroshub, "get_ouroboroshub_skills_dir", lambda: hub_root)
    summary = ouroboroshub.HubSkillSummary(
        slug="duckduckgo",
        name="duckduckgo",
        version="1.0.0",
        files=[{"path": "SKILL.md", "sha256": "x", "size": 1}],
        install_specs=[{"kind": "pip", "package": "ddgs"}],
    )
    monkeypatch.setattr(ouroboroshub, "load_catalog", lambda: {"raw_base_url": "https://raw.githubusercontent.com/razzant/OuroborosHub/main"})
    monkeypatch.setattr(ouroboroshub, "_summaries", lambda _catalog: [summary])
    auto_specs, _manual_specs, _warnings = ouroboroshub.normalize_declared_dependency_specs(summary.install_specs)
    specs_hash = ouroboroshub.install_specs_hash(auto_specs)
    target = hub_root / "duckduckgo"
    env_root = target / ".ouroboros_env"
    env_root.mkdir(parents=True)
    marker = {"schema_version": 1, "source": "ouroboroshub", "slug": "duckduckgo", "sanitized_name": "duckduckgo"}
    (target / "SKILL.md").write_text("installed", encoding="utf-8")
    (target / ".ouroboroshub.json").write_text(json.dumps(marker), encoding="utf-8")
    (env_root / "fingerprint.json").write_text(json.dumps({"status": "installed", "specs_hash": specs_hash}), encoding="utf-8")
    deps = tmp_path / "data" / "state" / "skills" / "duckduckgo" / "deps.json"
    deps.parent.mkdir(parents=True)
    deps.write_text(json.dumps({"status": "failed", "specs_hash": specs_hash}), encoding="utf-8")

    def fail_download(*_args, **_kwargs):
        raise AssertionError("valid retry fast-path should not download")

    monkeypatch.setattr(ouroboroshub, "_download_skill_files", fail_download)

    result = ouroboroshub.install("duckduckgo")

    assert result.ok is True
    assert result.provenance == marker
    assert json.loads(deps.read_text(encoding="utf-8"))["status"] == "installed"


def test_ouroboroshub_uninstall_clears_deps_state(monkeypatch, tmp_path):
    data_root = tmp_path / "data"
    hub_root = data_root / "skills" / "ouroboroshub"
    monkeypatch.setattr(ouroboroshub, "get_ouroboroshub_skills_dir", lambda: hub_root)
    target = hub_root / "demo"
    target.mkdir(parents=True)
    (target / ".ouroboroshub.json").write_text(
        json.dumps({"schema_version": 1, "source": "ouroboroshub", "slug": "demo", "sanitized_name": "demo"}),
        encoding="utf-8",
    )
    deps = data_root / "state" / "skills" / "demo" / "deps.json"
    deps.parent.mkdir(parents=True)
    deps.write_text(json.dumps({"status": "installed", "specs_hash": "abc"}), encoding="utf-8")

    result = ouroboroshub.uninstall("demo")

    assert result.ok
    assert not deps.exists()


def test_ouroboroshub_atomic_land_restores_old_on_move_failure(monkeypatch, tmp_path):
    target = tmp_path / "demo"
    target.mkdir()
    (target / "old.txt").write_text("old", encoding="utf-8")
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "new.txt").write_text("new", encoding="utf-8")

    def boom(_src, _dst):
        raise OSError("boom")

    monkeypatch.setattr(shutil, "move", boom)
    try:
        ouroboroshub.land_staged_tree(staging, target, replacement_suffix="replaced-ouroboroshub")
    except OSError:
        pass
    assert (target / "old.txt").read_text(encoding="utf-8") == "old"
    assert not (target / "new.txt").exists()


_RAW_BASE = "https://raw.githubusercontent.com/razzant/OuroborosHub/main"


def test_ouroboroshub_display_read_serves_ttl_cache(monkeypatch):
    ouroboroshub._catalog_cache_clear()
    try:
        cached = {"raw_base_url": _RAW_BASE, "skills": [{"slug": "demo", "version": "1.0.0"}]}
        ouroboroshub._catalog_cache_inject(cached)
        monkeypatch.setattr(
            ouroboroshub,
            "_fetch_bytes",
            lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("display read must not fetch")),
        )
        catalog = ouroboroshub.load_catalog(fresh=False)
        assert catalog["skills"][0]["slug"] == "demo"
        results = ouroboroshub.search("", fresh=False)
        assert [item.slug for item in results] == ["demo"]
    finally:
        ouroboroshub._catalog_cache_clear()


def test_ouroboroshub_display_cache_expires_after_ttl(monkeypatch):
    ouroboroshub._catalog_cache_clear()
    try:
        stale = {"raw_base_url": _RAW_BASE, "skills": [{"slug": "stale"}]}
        ouroboroshub._catalog_cache_inject(stale, age_sec=ouroboroshub._CATALOG_CACHE_TTL_SEC + 1)
        calls = {"fetch": 0}

        def fake_fetch(_url, *, max_bytes, timeout_sec=15):
            calls["fetch"] += 1
            return json.dumps({"raw_base_url": _RAW_BASE, "skills": [{"slug": "fresh"}]}).encode("utf-8")

        monkeypatch.setattr(ouroboroshub, "_fetch_bytes", fake_fetch)
        catalog = ouroboroshub.load_catalog(fresh=False)
        assert calls["fetch"] == 1
        assert catalog["skills"][0]["slug"] == "fresh"
        # The refetch refreshed the memo: the next display read is served from it.
        monkeypatch.setattr(
            ouroboroshub,
            "_fetch_bytes",
            lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("memo should be fresh again")),
        )
        assert ouroboroshub.load_catalog(fresh=False)["skills"][0]["slug"] == "fresh"
    finally:
        ouroboroshub._catalog_cache_clear()


def test_ouroboroshub_default_and_info_reads_bypass_cache(monkeypatch):
    """install/info/verifier reads never consume the display memo (§7.1a)."""
    ouroboroshub._catalog_cache_clear()
    try:
        poisoned = {"raw_base_url": _RAW_BASE, "skills": [{"slug": "poisoned"}]}
        ouroboroshub._catalog_cache_inject(poisoned)
        calls = {"fetch": 0}

        def fake_fetch(_url, *, max_bytes, timeout_sec=15):
            calls["fetch"] += 1
            return json.dumps({"raw_base_url": _RAW_BASE, "skills": [{"slug": "network-truth"}]}).encode("utf-8")

        monkeypatch.setattr(ouroboroshub, "_fetch_bytes", fake_fetch)
        catalog = ouroboroshub.load_catalog()
        assert calls["fetch"] == 1
        assert catalog["skills"][0]["slug"] == "network-truth"
        # info() (the official-hub verifier's read) resolves from the network,
        # even while the poisoned display memo is still valid.
        ouroboroshub._catalog_cache_inject(poisoned)
        summary = ouroboroshub.info("network-truth")
        assert summary.slug == "network-truth"
        assert calls["fetch"] == 2
    finally:
        ouroboroshub._catalog_cache_clear()


def test_ouroboroshub_install_reads_catalog_fresh(monkeypatch, tmp_path):
    ouroboroshub._catalog_cache_clear()
    try:
        hub_root = tmp_path / "data" / "skills" / "ouroboroshub"
        monkeypatch.setattr(ouroboroshub, "get_ouroboroshub_skills_dir", lambda: hub_root)
        ouroboroshub._catalog_cache_inject({"raw_base_url": _RAW_BASE, "skills": []})
        calls = {"fetch": 0}

        def fake_fetch(_url, *, max_bytes, timeout_sec=15):
            calls["fetch"] += 1
            return json.dumps(
                {
                    "raw_base_url": _RAW_BASE,
                    "skills": [
                        {"slug": "demo", "version": "1.0.0", "files": [{"path": "SKILL.md", "sha256": "x", "size": 1}]}
                    ],
                }
            ).encode("utf-8")

        monkeypatch.setattr(ouroboroshub, "_fetch_bytes", fake_fetch)

        def fake_download(_summary, _raw_base, staging_dir):
            (staging_dir / "SKILL.md").write_text("---\nname: demo\n---\n", encoding="utf-8")

        monkeypatch.setattr(ouroboroshub, "_download_skill_files", fake_download)

        result = ouroboroshub.install("demo")

        assert result.ok is True
        assert calls["fetch"] == 1
    finally:
        ouroboroshub._catalog_cache_clear()


def test_ouroboroshub_catalog_rows_carry_canonical_identity_facts():
    entries = ouroboroshub._summaries(
        {"skills": [{"slug": "demo"}, {"slug": "demo!"}, {"slug": "other"}]}
    )
    by_slug = {entry.slug: entry for entry in entries}
    assert by_slug["demo"].sanitized_name == "demo"
    assert by_slug["demo!"].sanitized_name == "demo"
    assert by_slug["demo"].identity_conflict is True
    assert by_slug["demo!"].identity_conflict is True
    assert by_slug["other"].identity_conflict is False
    row = by_slug["demo"].to_dict()
    assert row["sanitized_name"] == "demo"
    assert row["identity_conflict"] is True
    assert by_slug["other"].to_dict()["identity_conflict"] is False


def test_ouroboroshub_install_refuses_catalog_identity_conflict(monkeypatch, tmp_path):
    hub_root = tmp_path / "data" / "skills" / "ouroboroshub"
    monkeypatch.setattr(ouroboroshub, "get_ouroboroshub_skills_dir", lambda: hub_root)
    monkeypatch.setattr(
        ouroboroshub,
        "load_catalog",
        lambda: {
            "raw_base_url": _RAW_BASE,
            "skills": [
                {"slug": "demo", "version": "1.0.0", "files": [{"path": "SKILL.md", "sha256": "x", "size": 1}]},
                {"slug": "demo!", "version": "2.0.0"},
            ],
        },
    )
    monkeypatch.setattr(
        ouroboroshub,
        "_download_skill_files",
        lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("conflicted install must not download")),
    )

    result = ouroboroshub.install("demo")

    assert result.ok is False
    assert result.code == "catalog_identity_conflict"
    assert "catalog identity conflict" in result.error
    assert not (hub_root / "demo").exists()


def test_ouroboroshub_rejects_windows_and_review_opaque_paths():
    for value in (
        "..\\evil",
        "..\\..\\evil",
        "C:\\evil",
        "node_modules/dep/index.js",
        ".ouroboros_env/bin/tool",
        "__pycache__/plugin.cpython-39.pyc",
        "plugin.pyc",
        "native.so",
    ):
        try:
            ouroboroshub._safe_rel(value)
        except Exception:
            continue
        raise AssertionError(f"expected unsafe path rejection for {value!r}")
    # Q15=A: WebAssembly is a reviewable, content-hash-bound asset (it runs only
    # inside the sandboxed widget frame), not a generated or native artifact.
    assert ouroboroshub._safe_rel("wasm/core.wasm") == pathlib.PurePosixPath("wasm/core.wasm")
