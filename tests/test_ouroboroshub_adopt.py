"""Adopt transaction (§7.3) tests: eligibility, CAS, rollback matrix, success,
and the gateway dispatch (body validation, precheck skip, typed statuses)."""

from __future__ import annotations

import asyncio
import json
import shutil

from ouroboros.gateway import marketplace as marketplace_api
from ouroboros.marketplace import ouroboroshub

_RAW_BASE = "https://raw.githubusercontent.com/razzant/OuroborosHub/main"
_QUINTET = (
    "review.json",
    "review_job.json",
    "deps.json",
    "grants.json",
    "accepted_rebuttals.json",
)
_HEX64 = "a" * 64


class _Progress:
    def __init__(self) -> None:
        self.stages: list[str] = []

    def set(self, message: str) -> None:
        self.stages.append(message)


class _BodyRequest:
    def __init__(self, body=None, path_params=None, query_params=None):
        self._body = body if body is not None else {}
        self.path_params = path_params or {}
        self.query_params = query_params or {}

    async def json(self):
        return self._body


def _json_response_payload(response):
    return json.loads(response.body.decode("utf-8"))


async def _fake_run_blocking(func, *args, **kwargs):
    kwargs.pop("log_label", None)
    return func(*args, **kwargs)


def _apply_result(status="clean", error="", deps_status="installed", deps_error="", side_effect=None):
    async def _apply(payload, _skill_name):
        if side_effect is not None:
            side_effect()
        payload.update({"review_status": status, "review_findings": [], "review_error": error})
        payload.update({"deps_status": deps_status, "deps_error": deps_error})
        if deps_status == "failed":
            payload["ok"] = False
            payload["error"] = deps_error
        return (status, error, deps_status)

    return _apply


def _setup_hub(monkeypatch, tmp_path, *, skills=None):
    drive = tmp_path / "data"
    (drive / "skills").mkdir(parents=True, exist_ok=True)
    hub_root = drive / "skills" / "ouroboroshub"
    monkeypatch.setattr(ouroboroshub, "get_ouroboroshub_skills_dir", lambda: hub_root)
    empty_checkout = tmp_path / "empty-checkout"
    empty_checkout.mkdir(exist_ok=True)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(empty_checkout))
    catalog_skills = skills if skills is not None else [
        {"slug": "demo", "version": "1.0.0", "files": [{"path": "SKILL.md", "sha256": "x", "size": 1}]}
    ]
    monkeypatch.setattr(
        ouroboroshub, "load_catalog", lambda: {"raw_base_url": _RAW_BASE, "skills": catalog_skills}
    )

    def fake_download(_summary, _raw_base, staging_dir):
        (staging_dir / "SKILL.md").write_text("---\nname: demo\n---\nhub payload\n", encoding="utf-8")

    monkeypatch.setattr(ouroboroshub, "_download_skill_files", fake_download)
    monkeypatch.setattr("ouroboros.extension_loader.is_extension_live", lambda *_a, **_kw: False)
    monkeypatch.setattr("ouroboros.extension_loader.unload_extension", lambda *_a, **_kw: None)
    return drive


_LOCAL_BODY = "---\nname: demo\n---\nlocal payload\n"


def _make_occupant(drive, bucket="external", name="demo", body=_LOCAL_BODY):
    skill_dir = drive / "skills" / bucket / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(body, encoding="utf-8")
    return skill_dir


def _live_hash(drive, skill_dir):
    from ouroboros.skill_loader import load_skill

    return load_skill(skill_dir, drive).content_hash


def _seed_state_quintet(drive, name="demo"):
    """Seed four quintet files (accepted_rebuttals.json stays absent) + enabled.json."""
    state = drive / "state" / "skills" / name
    state.mkdir(parents=True, exist_ok=True)
    originals = {}
    for filename in _QUINTET:
        if filename == "accepted_rebuttals.json":
            originals[filename] = None
            continue
        blob = json.dumps({"sentinel": filename}).encode("utf-8")
        (state / filename).write_bytes(blob)
        originals[filename] = blob
    (state / "enabled.json").write_text('{"enabled": true}\n', encoding="utf-8")
    return state, originals


def _run_adopt(drive, expected, *, slug="demo", apply=None):
    return asyncio.run(
        ouroboroshub.run_hub_adopt(
            slug,
            drive_root=drive,
            expected_content_hash=expected,
            progress=_Progress(),
            run_blocking=_fake_run_blocking,
            apply_review_and_deps=apply if apply is not None else _apply_result(),
        )
    )


def _assert_source_restored(drive, originals=None, name="demo", body=_LOCAL_BODY):
    source = drive / "skills" / "external" / name
    assert (source / "SKILL.md").read_text(encoding="utf-8") == body
    assert not (drive / "skills" / "ouroboroshub" / name).exists()
    if originals is not None:
        state = drive / "state" / "skills" / name
        for filename, blob in originals.items():
            path = state / filename
            if blob is None:
                assert not path.exists(), f"{filename} must be removed by rollback"
            else:
                assert path.read_bytes() == blob, f"{filename} must be byte-restored"


# --- eligibility matrix -----------------------------------------------------


def test_adopt_refuses_without_local_occupant(monkeypatch, tmp_path):
    drive = _setup_hub(monkeypatch, tmp_path)

    payload = _run_adopt(drive, _HEX64)

    assert payload["ok"] is False
    assert payload["code"] == "adopt_not_eligible"
    assert payload["reason"] == "no_local_occupant"
    assert "rolled_back" not in payload


def test_adopt_refuses_non_external_occupants(monkeypatch, tmp_path):
    for bucket, reason in (
        ("ouroboroshub", "already_hub"),
        ("native", "native_seed"),
        ("clawhub", "clawhub_unsupported_v1"),
    ):
        base = tmp_path / bucket
        base.mkdir()
        drive = _setup_hub(monkeypatch, base)
        occupant = _make_occupant(drive, bucket=bucket)

        payload = _run_adopt(drive, _HEX64)

        assert payload["ok"] is False, bucket
        assert payload["code"] == "adopt_not_eligible", bucket
        assert payload["reason"] == reason, bucket
        assert (occupant / "SKILL.md").is_file(), bucket
        assert "rolled_back" not in payload, bucket


def test_adopt_refuses_user_repo_occupant(monkeypatch, tmp_path):
    drive = _setup_hub(monkeypatch, tmp_path)
    checkout = tmp_path / "checkout"
    foreign = checkout / "demo"
    foreign.mkdir(parents=True)
    (foreign / "SKILL.md").write_text(_LOCAL_BODY, encoding="utf-8")
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(checkout))

    payload = _run_adopt(drive, _HEX64)

    assert payload["ok"] is False
    assert payload["code"] == "adopt_not_eligible"
    assert payload["reason"] == "user_repo"
    assert (foreign / "SKILL.md").is_file()


def test_adopt_refuses_on_cas_mismatch(monkeypatch, tmp_path):
    drive = _setup_hub(monkeypatch, tmp_path)
    occupant = _make_occupant(drive)
    live = _live_hash(drive, occupant)
    assert live and live != "0" * 64

    payload = _run_adopt(drive, "0" * 64)

    assert payload["ok"] is False
    assert payload["code"] == "adopt_cas_mismatch"
    assert payload["live_content_hash"] == live
    assert (occupant / "SKILL.md").read_text(encoding="utf-8") == _LOCAL_BODY
    assert "rolled_back" not in payload


def test_adopt_refuses_when_catalog_unavailable(monkeypatch, tmp_path):
    drive = _setup_hub(monkeypatch, tmp_path)
    _make_occupant(drive)
    monkeypatch.setattr(
        ouroboroshub,
        "load_catalog",
        lambda: (_ for _ in ()).throw(ouroboroshub.OuroborosHubError("catalog fetch failed")),
    )

    payload = _run_adopt(drive, _HEX64)

    assert payload["ok"] is False
    assert payload["code"] == "catalog_unavailable"


def test_adopt_refuses_catalog_identity_conflict(monkeypatch, tmp_path):
    drive = _setup_hub(
        monkeypatch,
        tmp_path,
        skills=[
            {"slug": "demo", "version": "1.0.0", "files": [{"path": "SKILL.md", "sha256": "x", "size": 1}]},
            {"slug": "demo!", "version": "2.0.0"},
        ],
    )
    occupant = _make_occupant(drive)

    payload = _run_adopt(drive, _live_hash(drive, occupant))

    assert payload["ok"] is False
    assert payload["code"] == "catalog_identity_conflict"
    assert (occupant / "SKILL.md").read_text(encoding="utf-8") == _LOCAL_BODY


# --- rollback matrix ---------------------------------------------------------


def test_adopt_install_failure_restores_source_and_state(monkeypatch, tmp_path):
    drive = _setup_hub(monkeypatch, tmp_path)
    occupant = _make_occupant(drive)
    _state, originals = _seed_state_quintet(drive)
    expected = _live_hash(drive, occupant)
    monkeypatch.setattr(
        ouroboroshub,
        "install",
        lambda _slug, **_kw: ouroboroshub.HubInstallResult(False, "demo", error="download boom"),
    )

    payload = _run_adopt(drive, expected)

    assert payload["ok"] is False
    assert payload["rolled_back"] is True
    assert payload["error"] == "download boom"
    _assert_source_restored(drive, originals)


def test_adopt_install_exception_restores_source_and_state(monkeypatch, tmp_path):
    drive = _setup_hub(monkeypatch, tmp_path)
    occupant = _make_occupant(drive)
    _state, originals = _seed_state_quintet(drive)
    expected = _live_hash(drive, occupant)
    monkeypatch.setattr(
        ouroboroshub,
        "install",
        lambda _slug, **_kw: (_ for _ in ()).throw(RuntimeError("landing exploded")),
    )

    payload = _run_adopt(drive, expected)

    assert payload["ok"] is False
    assert payload["rolled_back"] is True
    assert "landing exploded" in payload["error"]
    _assert_source_restored(drive, originals)


def test_adopt_review_error_rolls_back(monkeypatch, tmp_path):
    drive = _setup_hub(monkeypatch, tmp_path)
    occupant = _make_occupant(drive)
    _state, originals = _seed_state_quintet(drive)
    expected = _live_hash(drive, occupant)

    payload = _run_adopt(
        drive, expected, apply=_apply_result("pending", "review_skill raised: boom", "not_required")
    )

    assert payload["ok"] is False
    assert payload["rolled_back"] is True
    _assert_source_restored(drive, originals)


def test_adopt_review_blockers_roll_back_under_blocking_enforcement(monkeypatch, tmp_path):
    drive = _setup_hub(monkeypatch, tmp_path)
    occupant = _make_occupant(drive)
    _state, originals = _seed_state_quintet(drive)
    expected = _live_hash(drive, occupant)
    monkeypatch.setattr(
        "ouroboros.skill_loader.review_status_allows_execution",
        lambda status: status in {"pass", "clean"},
    )

    payload = _run_adopt(drive, expected, apply=_apply_result("blockers", "", "not_required"))

    assert payload["ok"] is False
    assert payload["rolled_back"] is True
    assert "blockers" in payload["error"]
    _assert_source_restored(drive, originals)


def test_adopt_deps_failure_restores_state_clobbered_by_the_attempt(monkeypatch, tmp_path):
    drive = _setup_hub(monkeypatch, tmp_path)
    occupant = _make_occupant(drive)
    state, originals = _seed_state_quintet(drive)
    expected = _live_hash(drive, occupant)

    def clobber_state():
        # Simulate the failed attempt's review write + auto-grant rebinding +
        # a rebuttals file created where none existed before.
        (state / "review.json").write_text('{"status": "pass", "content_hash": "new"}\n', encoding="utf-8")
        (state / "grants.json").write_text('{"granted_keys": ["NEW_KEY"], "content_hash": "new"}\n', encoding="utf-8")
        (state / "accepted_rebuttals.json").write_text("[]\n", encoding="utf-8")

    payload = _run_adopt(
        drive,
        expected,
        apply=_apply_result("clean", "", "failed", "pip boom", side_effect=clobber_state),
    )

    assert payload["ok"] is False
    assert payload["rolled_back"] is True
    assert payload["error"] == "pip boom"
    assert payload["deps_status"] == "failed"
    _assert_source_restored(drive, originals)
    assert (state / "enabled.json").read_text(encoding="utf-8") == '{"enabled": true}\n'


def test_adopt_reload_failure_is_terminal(monkeypatch, tmp_path):
    drive = _setup_hub(monkeypatch, tmp_path)
    occupant = _make_occupant(drive)
    _state, originals = _seed_state_quintet(drive)
    expected = _live_hash(drive, occupant)
    calls = {"reconcile": 0}

    def fake_reconcile(*_args, **_kwargs):
        calls["reconcile"] += 1
        if calls["reconcile"] == 1:
            return {"action": "extension_load_error", "load_error": "boom", "reason": "load_error"}
        return {"action": "extension_loaded"}

    monkeypatch.setattr("ouroboros.extension_loader.is_extension_live", lambda *_a, **_kw: True)
    monkeypatch.setattr("ouroboros.extension_loader.reconcile_extension", fake_reconcile)

    payload = _run_adopt(drive, expected)

    assert payload["ok"] is False
    assert payload["rolled_back"] is True
    assert "extension reload failed after adopt" in payload["error"]
    # First reconcile is the failed reload; the second restores the source live.
    assert calls["reconcile"] == 2
    _assert_source_restored(drive, originals)


# --- success path ------------------------------------------------------------


def test_adopt_success_lands_hub_payload_and_retains_pre_adopt_snapshot(monkeypatch, tmp_path):
    drive = _setup_hub(monkeypatch, tmp_path)
    occupant = _make_occupant(drive)
    state, _originals = _seed_state_quintet(drive)
    expected = _live_hash(drive, occupant)

    payload = _run_adopt(drive, expected)

    assert payload["ok"] is True
    assert payload["adopted"] is True
    assert payload["sanitized_name"] == "demo"
    assert payload["review_status"] == "clean"
    assert payload["deps_status"] == "installed"
    assert "rolled_back" not in payload
    target = drive / "skills" / "ouroboroshub" / "demo"
    assert payload["target_dir"] == str(target)
    assert "hub payload" in (target / "SKILL.md").read_text(encoding="utf-8")
    sidecar = json.loads((target / ".ouroboroshub.json").read_text(encoding="utf-8"))
    assert sidecar["source"] == "ouroboroshub"
    assert sidecar["slug"] == "demo"
    assert sidecar["sanitized_name"] == "demo"
    # Source payload moved to the depth-1 retained snapshot; bucket is clean.
    assert not occupant.exists()
    snapshot = drive / "skills" / "external" / ".rollback" / "demo.pre-adopt"
    assert (snapshot / "SKILL.md").read_text(encoding="utf-8") == _LOCAL_BODY
    leftovers = [p.name for p in (drive / "skills" / "external" / ".rollback").iterdir()]
    assert leftovers == ["demo.pre-adopt"]
    # Non-quintet state survives untouched.
    assert (state / "enabled.json").read_text(encoding="utf-8") == '{"enabled": true}\n'


def test_adopt_pre_adopt_snapshot_depth_one_replaces_older_copy(monkeypatch, tmp_path):
    drive = _setup_hub(monkeypatch, tmp_path)
    first = _make_occupant(drive, body="---\nname: demo\n---\nfirst local\n")
    payload = _run_adopt(drive, _live_hash(drive, first))
    assert payload["adopted"] is True
    # Simulate the owner starting over: drop the hub copy, author a new local one.
    shutil.rmtree(drive / "skills" / "ouroboroshub" / "demo")
    second = _make_occupant(drive, body="---\nname: demo\n---\nsecond local\n")

    payload = _run_adopt(drive, _live_hash(drive, second))

    assert payload["adopted"] is True
    rollback_root = drive / "skills" / "external" / ".rollback"
    snapshot = rollback_root / "demo.pre-adopt"
    assert (snapshot / "SKILL.md").read_text(encoding="utf-8") == "---\nname: demo\n---\nsecond local\n"
    assert [p.name for p in rollback_root.iterdir()] == ["demo.pre-adopt"]


# --- gateway dispatch --------------------------------------------------------


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


def test_adopt_body_validation_matrix(monkeypatch, tmp_path):
    _stub_marketplace_roots(monkeypatch, tmp_path)

    async def _unexpected_job(**_kwargs):
        raise AssertionError("invalid adopt body must not start a lifecycle job")

    monkeypatch.setattr(marketplace_api, "run_lifecycle_job", _unexpected_job)
    cases = [
        ({}, "adopt_expected_hash_missing"),
        ({"expected_content_hash": "zz"}, "adopt_expected_hash_invalid"),
        ({"expected_content_hash": "A" * 64}, "adopt_expected_hash_invalid"),
        ({"expected_content_hash": "a" * 63}, "adopt_expected_hash_invalid"),
        ({"expected_content_hash": "a" * 65}, "adopt_expected_hash_invalid"),
        ({"expected_content_hash": "g" + "a" * 63}, "adopt_expected_hash_invalid"),
        ({"expected_content_hash": _HEX64, "auto_review": False}, "adopt_requires_auto_review"),
        ({"expected_content_hash": _HEX64, "overwrite": True}, "adopt_overwrite_conflict"),
    ]
    for extra, code in cases:
        body = {"slug": "demo", "adopt": True, **extra}

        response = asyncio.run(marketplace_api.api_ouroboroshub_install(_BodyRequest(body)))

        assert response.status_code == 400, code
        payload = _json_response_payload(response)
        assert payload["ok"] is False, code
        assert payload["sanitized_name"] == "demo", code
        assert payload["code"] == code, (code, payload)


def test_adopt_with_explicit_auto_review_true_is_accepted(monkeypatch, tmp_path):
    _stub_marketplace_roots(monkeypatch, tmp_path)
    _run_lifecycle_inline(monkeypatch)

    async def fake_adopt(_slug, **_kwargs):
        return {"ok": True, "sanitized_name": "demo", "adopted": True}

    monkeypatch.setattr(marketplace_api.ouroboroshub, "run_hub_adopt", fake_adopt)

    response = asyncio.run(
        marketplace_api.api_ouroboroshub_install(
            _BodyRequest({"slug": "demo", "adopt": True, "expected_content_hash": _HEX64, "auto_review": True})
        )
    )

    assert response.status_code == 200
    assert _json_response_payload(response)["adopted"] is True


def test_adopt_skips_pre_lifecycle_identity_precheck(monkeypatch, tmp_path):
    """A foreign external occupant + adopt:true reaches the transaction instead
    of the non-adopt path's pre-lifecycle 409 (§7.3 precheck-skip ordering)."""
    _stub_marketplace_roots(monkeypatch, tmp_path)
    _run_lifecycle_inline(monkeypatch)
    occupant = tmp_path / "skills" / "external" / "demo"
    occupant.mkdir(parents=True)
    (occupant / "SKILL.md").write_text(_LOCAL_BODY, encoding="utf-8")
    monkeypatch.setattr(
        marketplace_api.ouroboroshub,
        "install_identity_error",
        lambda *_a, **_kw: (_ for _ in ()).throw(
            AssertionError("adopt must skip the gateway pre-lifecycle identity precheck")
        ),
    )
    captured = {}

    async def fake_adopt(slug, **kwargs):
        captured["slug"] = slug
        captured["expected_content_hash"] = kwargs.get("expected_content_hash")
        return {"ok": True, "sanitized_name": "demo", "adopted": True}

    monkeypatch.setattr(marketplace_api.ouroboroshub, "run_hub_adopt", fake_adopt)

    response = asyncio.run(
        marketplace_api.api_ouroboroshub_install(
            _BodyRequest({"slug": "demo", "adopt": True, "expected_content_hash": _HEX64})
        )
    )

    assert response.status_code == 200
    assert _json_response_payload(response)["adopted"] is True
    assert captured == {"slug": "demo", "expected_content_hash": _HEX64}


def test_adopt_typed_codes_map_to_http_statuses(monkeypatch, tmp_path):
    _stub_marketplace_roots(monkeypatch, tmp_path)
    _run_lifecycle_inline(monkeypatch)
    cases = [
        ({"ok": False, "sanitized_name": "demo", "error": "cas", "code": "adopt_cas_mismatch", "live_content_hash": "b" * 64}, 409),
        ({"ok": False, "sanitized_name": "demo", "error": "occupied", "code": "adopt_not_eligible", "reason": "already_hub"}, 409),
        ({"ok": False, "sanitized_name": "demo", "error": "conflict", "code": "catalog_identity_conflict"}, 409),
        ({"ok": False, "sanitized_name": "demo", "error": "offline", "code": "catalog_unavailable"}, 502),
        ({"ok": False, "sanitized_name": "demo", "error": "plain failure"}, 400),
    ]
    for payload, status in cases:
        async def fake_adopt(_slug, _payload=payload, **_kwargs):
            return dict(_payload)

        monkeypatch.setattr(marketplace_api.ouroboroshub, "run_hub_adopt", fake_adopt)

        response = asyncio.run(
            marketplace_api.api_ouroboroshub_install(
                _BodyRequest({"slug": "demo", "adopt": True, "expected_content_hash": _HEX64})
            )
        )

        assert response.status_code == status, payload
        assert _json_response_payload(response).get("code") == payload.get("code")


def test_install_endpoint_maps_catalog_identity_conflict_to_409(monkeypatch, tmp_path):
    _stub_marketplace_roots(monkeypatch, tmp_path)
    _run_lifecycle_inline(monkeypatch)
    monkeypatch.setattr(
        marketplace_api.ouroboroshub,
        "load_catalog",
        lambda: {
            "raw_base_url": _RAW_BASE,
            "skills": [{"slug": "demo", "version": "1.0.0"}, {"slug": "demo!", "version": "2.0.0"}],
        },
    )
    monkeypatch.setattr(
        marketplace_api.ouroboroshub,
        "_download_skill_files",
        lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("conflicted install must not download")),
    )
    monkeypatch.setattr(
        marketplace_api.ouroboroshub, "get_ouroboroshub_skills_dir", lambda: tmp_path / "skills" / "ouroboroshub"
    )

    response = asyncio.run(
        marketplace_api.api_ouroboroshub_install(_BodyRequest({"slug": "demo"}))
    )

    assert response.status_code == 409
    payload = _json_response_payload(response)
    assert payload["ok"] is False
    assert payload["code"] == "catalog_identity_conflict"


def test_extensions_index_rows_expose_loader_content_hash(monkeypatch, tmp_path):
    """§7.2: /api/extensions rows carry the loader content hash; collision rows
    (whose loader hash never existed) carry the empty string."""
    import ouroboros.gateway.extensions as extensions_api
    import supervisor.queue as supervisor_queue
    from ouroboros.contracts.skill_manifest import SkillManifest
    from ouroboros.skill_loader import LoadedSkill, SkillReviewState

    drive_root = tmp_path / "drive"
    skill_dir = drive_root / "skills" / "external" / "demo"
    skill_dir.mkdir(parents=True)
    loaded = LoadedSkill(
        name="demo",
        skill_dir=skill_dir,
        manifest=SkillManifest(name="demo", description="", version="1.0.0", type="instruction"),
        content_hash="f" * 64,
        enabled=False,
        review=SkillReviewState(status="pending"),
        load_error="",
        source="external",
    )
    collision_dir = drive_root / "skills" / "clawhub" / "demo2"
    collision_dir.mkdir(parents=True)
    collision = LoadedSkill(
        name="demo2",
        skill_dir=collision_dir,
        manifest=SkillManifest(name="demo2", description="", version="", type="instruction"),
        content_hash="",
        load_error="Skill name collision: clawhub and user_repo",
        source="clawhub",
        identity_collision=True,
    )
    monkeypatch.setattr(extensions_api, "discover_skills", lambda *_a, **_kw: [loaded, collision])
    monkeypatch.setattr(
        extensions_api,
        "snapshot",
        lambda: {"tools": [], "routes": [], "ws_handlers": [], "ui_tabs": []},
    )
    monkeypatch.setattr(supervisor_queue, "sync_skill_schedules", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        "ouroboros.tools.github.github_token_from_env_or_settings", lambda: ""
    )

    payload = extensions_api._build_extensions_index(drive_root, repo_path="")

    rows = {row["name"]: row for row in payload["skills"]}
    assert rows["demo"]["content_hash"] == "f" * 64
    assert rows["demo2"]["content_hash"] == ""


def test_catalog_endpoint_serves_cached_rows_with_identity_facts(monkeypatch, tmp_path):
    _stub_marketplace_roots(monkeypatch, tmp_path)
    ouroboroshub._catalog_cache_clear()
    try:
        ouroboroshub._catalog_cache_inject(
            {
                "raw_base_url": _RAW_BASE,
                "skills": [{"slug": "demo"}, {"slug": "demo!"}, {"slug": "other"}],
            }
        )
        monkeypatch.setattr(
            ouroboroshub,
            "_fetch_bytes",
            lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("display endpoint must serve the memo")),
        )

        response = asyncio.run(
            marketplace_api.api_ouroboroshub_catalog(_BodyRequest(query_params={}))
        )

        assert response.status_code == 200
        rows = {row["slug"]: row for row in _json_response_payload(response)["results"]}
        assert rows["demo"]["sanitized_name"] == "demo"
        assert rows["demo"]["identity_conflict"] is True
        assert rows["demo!"]["sanitized_name"] == "demo"
        assert rows["demo!"]["identity_conflict"] is True
        assert rows["other"]["identity_conflict"] is False
    finally:
        ouroboroshub._catalog_cache_clear()


def test_adopt_cas_reverified_on_aside_tree(monkeypatch, tmp_path):
    """E-fix BC-1: a payload mutated after the CAS read is refused, not adopted."""
    from ouroboros.skill_loader import compute_content_hash

    drive = _setup_hub(monkeypatch, tmp_path)
    skill_dir = _make_occupant(drive)
    expected = compute_content_hash(skill_dir)

    def _mutating_unload(name):
        # Mutate INSIDE the CAS->move window (the unload hook runs between them).
        (skill_dir / "SKILL.md").write_text(_LOCAL_BODY + "# edited\n", encoding="utf-8")

    monkeypatch.setattr("ouroboros.extension_loader.unload_extension", _mutating_unload)
    monkeypatch.setattr("ouroboros.extension_loader.is_extension_live", lambda *a, **k: False)

    outcome = ouroboroshub._adopt_begin("demo", drive, expected)

    assert isinstance(outcome, dict), outcome
    assert outcome.get("code") == "adopt_cas_mismatch"
    # Source restored in place; no aside residue holds the payload.
    assert skill_dir.is_dir()
    assert (skill_dir / "SKILL.md").read_text(encoding="utf-8").endswith("# edited\n")
    rollback_root = skill_dir.parent / ".rollback"
    leftovers = list(rollback_root.glob("demo.adopt.*")) if rollback_root.is_dir() else []
    assert leftovers == []


def test_rollback_failure_reports_rolled_back_false(monkeypatch, tmp_path):
    """Final-gate fix: a failed restore may never claim rolled_back:true."""
    drive = _setup_hub(monkeypatch, tmp_path)
    skill_dir = _make_occupant(drive)
    expected = _live_hash(drive, skill_dir)
    monkeypatch.setattr(
        ouroboroshub, "_adopt_rollback",
        lambda ctx: ["source_restore: OSError: boom (source preserved at X)"],
    )

    outcome = asyncio.run(ouroboroshub.run_hub_adopt(
        "demo",
        drive_root=drive,
        expected_content_hash=expected,
        progress=_Progress(),
        run_blocking=_fake_run_blocking,
        apply_review_and_deps=_apply_result(deps_status="failed", deps_error="boom"),
    ))

    assert outcome["ok"] is False
    assert outcome["rolled_back"] is False
    assert any("source_restore" in e for e in outcome.get("rollback_errors", [])), outcome
    assert "ROLLBACK INCOMPLETE" in outcome["error"]


def test_adopt_rollback_discloses_concurrent_source_recreation(tmp_path):
    """Aside AND source both present: keep the newer occupant, report failure."""
    source = tmp_path / "skills" / "external" / "demo"
    aside = tmp_path / "skills" / "external" / ".rollback" / "demo.adopt.x"
    source.mkdir(parents=True)
    aside.mkdir(parents=True)
    ctx = ouroboroshub._AdoptContext(
        name="demo",
        drive_root=tmp_path,
        source_dir=source,
        aside_dir=aside,
        dest_dir=tmp_path / "skills" / "ouroboroshub" / "demo",
        state_snapshot={},
        was_live=False,
    )
    errors = ouroboroshub._adopt_rollback(ctx)
    assert any("recreated concurrently" in e for e in errors), errors
    assert source.exists() and aside.exists()


def test_adopt_rollback_counts_failed_live_reconcile(monkeypatch, tmp_path):
    """A rollback whose live-extension reconcile fails is not rolled_back:true."""
    source = tmp_path / "skills" / "external" / "demo"
    aside = tmp_path / "skills" / "external" / ".rollback" / "demo.adopt.x"
    aside.mkdir(parents=True)
    (aside / "SKILL.md").write_text(_LOCAL_BODY, encoding="utf-8")
    monkeypatch.setattr(
        ouroboroshub, "_reconcile_extension_quiet",
        lambda name, root: "live_reconcile: ImportError: boom",
    )
    ctx = ouroboroshub._AdoptContext(
        name="demo",
        drive_root=tmp_path,
        source_dir=source,
        aside_dir=aside,
        dest_dir=tmp_path / "skills" / "ouroboroshub" / "demo",
        state_snapshot={},
        was_live=False,
        desired_live=True,
    )
    errors = ouroboroshub._adopt_rollback(ctx)
    assert any("live_reconcile" in e for e in errors), errors
    assert source.exists() and not aside.exists()


def test_adopt_rollback_collects_missing_source_error(tmp_path):
    """Unit: aside gone AND source gone is a collected error, never silence."""
    ctx = ouroboroshub._AdoptContext(
        name="demo",
        drive_root=tmp_path,
        source_dir=tmp_path / "skills" / "external" / "demo",
        aside_dir=tmp_path / "skills" / "external" / ".rollback" / "demo.adopt.x",
        dest_dir=tmp_path / "skills" / "ouroboroshub" / "demo",
        state_snapshot={},
        was_live=False,
    )
    errors = ouroboroshub._adopt_rollback(ctx)
    assert any("source_restore" in e for e in errors), errors


def test_retention_failure_disclosed_on_success(monkeypatch, tmp_path):
    """Final-gate fix: retention failure -> adopted:true + pre_adopt_retained:false."""
    drive = _setup_hub(monkeypatch, tmp_path)
    skill_dir = _make_occupant(drive)
    expected = _live_hash(drive, skill_dir)
    real_finalize = ouroboroshub._adopt_finalize
    monkeypatch.setattr(ouroboroshub, "_adopt_finalize", lambda ctx: (False, "OSError: keep failed (source preserved at X)"))
    outcome = asyncio.run(ouroboroshub.run_hub_adopt(
        "demo",
        drive_root=drive,
        expected_content_hash=expected,
        progress=_Progress(),
        run_blocking=_fake_run_blocking,
        apply_review_and_deps=_apply_result(),
    ))
    assert outcome.get("adopted") is True
    assert outcome.get("pre_adopt_retained") is False
    assert "keep failed" in str(outcome.get("retention_error"))
    monkeypatch.setattr(ouroboroshub, "_adopt_finalize", real_finalize)


def test_enabled_but_not_live_occupant_gets_strict_reconcile(monkeypatch, tmp_path):
    """Final-gate fix: reconcile keys on ENABLED state, not only was_live."""
    from ouroboros.skill_loader import save_enabled

    drive = _setup_hub(monkeypatch, tmp_path)
    skill_dir = _make_occupant(drive)
    save_enabled(drive, "demo", True)
    expected = _live_hash(drive, skill_dir)
    calls = []

    def _reconcile(name, _drive, _settings):
        # The hub payload fails to load; the restored original loads fine —
        # so the rollback's own reconcile succeeds and rolled_back stays true.
        calls.append(name)
        if len(calls) == 1:
            return {"action": "extension_load_error", "load_error": "import boom"}
        return {"action": "loaded"}

    monkeypatch.setattr("ouroboros.extension_loader.reconcile_extension", _reconcile)
    outcome = asyncio.run(ouroboroshub.run_hub_adopt(
        "demo",
        drive_root=drive,
        expected_content_hash=expected,
        progress=_Progress(),
        run_blocking=_fake_run_blocking,
        apply_review_and_deps=_apply_result(),
    ))
    assert len(calls) >= 2, "reconcile must run forward AND during rollback for an enabled occupant"
    assert outcome["ok"] is False
    assert outcome["rolled_back"] is True
    assert "reload failed" in outcome["error"]
    assert skill_dir.is_dir()


def test_adopt_reuses_prelude_catalog_fetch(monkeypatch, tmp_path):
    """Final-gate fix: install inside adopt must not refetch the catalog."""
    drive = _setup_hub(monkeypatch, tmp_path)
    skill_dir = _make_occupant(drive)
    expected = _live_hash(drive, skill_dir)
    fetches = {"n": 0}
    orig = ouroboroshub.load_catalog

    def _counting(*a, **k):
        fetches["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(ouroboroshub, "load_catalog", _counting)
    outcome = asyncio.run(ouroboroshub.run_hub_adopt(
        "demo",
        drive_root=drive,
        expected_content_hash=expected,
        progress=_Progress(),
        run_blocking=_fake_run_blocking,
        apply_review_and_deps=_apply_result(),
    ))
    assert outcome.get("adopted") is True
    assert fetches["n"] == 1, f"expected exactly one catalog fetch, got {fetches['n']}"
