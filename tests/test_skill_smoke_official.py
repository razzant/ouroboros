# tests/test_skill_smoke_official.py — official OuroborosHub skill install smoke.
#
# The `skill_smoke` lane: real-network install of the nine pinned official
# skills + preflight/deps/command probes. Canonical lane description (purpose,
# triggers, red-means-investigate posture): docs/DEVELOPMENT.md "Pytest marker
# lanes". No fallback-skip on network failure (owner directive); bounded
# transient retries are the sole flake mitigation. Never add this file to
# _SERIAL_TEST_FILES or mark it `serial`: the lane runs only in its dedicated
# serial CI job, and the `and not skill_smoke` markexprs in quick/full-test
# are the barrier that keeps it out of those passes — the no-serial rule keeps
# the lane assignment single and unambiguous (defense-in-depth).
#
# Command smoke deliberately uses direct plugin.py registration against a fake
# PluginAPI (the tests/test_unix_computer_use_skill.py pattern) instead of the
# full extension_loader/enable chain (owner decision). weather/backlog_manager
# are the no-dependency class production also imports in-process; duckduckgo
# carries isolated deps (production dispatches it out-of-process) — the
# in-process call here is a smoke-level deviation accepted for CI simplicity
# on disposable runners over sha256-verified official payloads.

from __future__ import annotations

import hashlib
import http.client
import importlib.util
import json
import pathlib
import subprocess
import time
from typing import Any, Dict

import pytest

from ouroboros.config import DATA_DIR, get_ouroboroshub_skills_dir
from ouroboros.contracts.skill_manifest import parse_skill_manifest_text
from ouroboros.extension_isolated_deps import isolated_site_dirs_scope
from ouroboros.marketplace import ouroboroshub
from ouroboros.marketplace.install_specs import install_specs_hash
from ouroboros.marketplace.isolated_deps import (
    install_isolated_dependencies,
    python_runtime_binary,
    read_deps_state,
)
from ouroboros.skill_dependencies import auto_install_specs_for_skill
from ouroboros.skill_loader import _sanitize_skill_name, find_skill
from ouroboros.tools.registry import ToolContext
from ouroboros.tools.skill_preflight import _handle_skill_preflight

pytestmark = pytest.mark.skill_smoke

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

OFFICIAL_SKILLS = [
    "telegram-bridge",
    "a2a",
    "duckduckgo",
    "perplexity",
    "nanobanana",
    "weather",
    "backlog_manager",
    "music_gen",
    "video_gen",
]
# The only skills in the set that declare auto-installable dependency specs.
DEPS_SKILLS = ("a2a", "duckduckgo")
# Import proof per deps skill, run inside the isolated venv's own python.
DEPS_IMPORT_CHECK = {
    "duckduckgo": "import ddgs",
    "a2a": "import google.protobuf, a2a",
}

_TRANSIENT_RETRY_SLEEP_SEC = 10

# Deterministic install() failure markers — never retried. Each literal is
# owned by ouroboros/marketplace/ouroboroshub.py (sha mismatch :210, missing
# sha :205, unsafe/opaque paths in _safe_rel, staging escape :215, missing
# SKILL.md :219, unknown slug :265, already-installed :287); a rewording there
# must update this tuple or the affected error gains one wasted retry.
_DETERMINISTIC_INSTALL_ERRORS = (
    "sha256 mismatch",
    "missing sha256",
    "unsafe catalog file path",
    "review-opaque dependency directory",
    "generated or binary artifact",
    "escapes staging dir",
    "did not include SKILL.md",
    "skill not found",
    "already installed",
    "has no files",
    "is not an object",
)

# urllib transport errors are OSError subclasses; http.client raises its own
# transient shapes (IncompleteRead/BadStatusLine) outside that hierarchy.
_TRANSIENT_EXC = (OSError, http.client.HTTPException)


def _retry_transient(fn, *, describe: str, attempts: int = 2):
    """Retry transport-class failures only. Deterministic failures
    (OuroborosHubError over bad catalog data is a RuntimeError) re-raise
    immediately — red is the honest outcome.
    """
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except _TRANSIENT_EXC as exc:
            last_exc = exc
            if attempt < attempts:
                print(f"{describe}: attempt {attempt} failed ({exc!r}); retrying once")
                time.sleep(_TRANSIENT_RETRY_SLEEP_SEC)
    raise AssertionError(f"{describe} failed after {attempts} attempts: {last_exc!r}")


# ---------------------------------------------------------------------------
# Session fixtures: one live catalog fetch, memoized installs/deps.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def hub_catalog():
    """Fetch the LIVE catalog exactly once, then serve every install from a cache.

    The single real load_catalog() exercises _fetch_bytes + the host allowlist +
    raw_base_url derivation. The digest/version log line makes a red run
    reconstructible against the exact catalog revision that was tested.
    """
    catalog = _retry_transient(ouroboroshub.load_catalog, describe="load_catalog")
    digest = hashlib.sha256(
        json.dumps(catalog, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    slugs = {
        str(item.get("slug") or ""): str(item.get("version") or "")
        for item in catalog.get("skills") or []
        if isinstance(item, dict)
    }
    # "normalized": hashed over the client-parsed dict (incl. the injected
    # raw_base_url), not the raw catalog.json bytes — do not compare to
    # `sha256sum catalog.json`.
    print(f"OuroborosHub normalized-catalog digest sha256={digest}")
    print("catalog versions: " + ", ".join(f"{s}={slugs.get(s, '?')}" for s in OFFICIAL_SKILLS))
    missing = [slug for slug in OFFICIAL_SKILLS if slug not in slugs]
    assert not missing, f"official catalog is missing expected skills: {missing}"

    patch = pytest.MonkeyPatch()
    cached = catalog

    def _cached_catalog() -> Dict[str, Any]:
        return json.loads(json.dumps(cached))

    patch.setattr(ouroboroshub, "load_catalog", _cached_catalog)
    try:
        yield cached
    finally:
        patch.undo()


@pytest.fixture(scope="session")
def install_skill(hub_catalog):
    """Memoizing installer: each slug is installed at most once per session.

    Only a fully successful atomic install is cached (landing is atomic via
    land_staged_tree; a failed attempt removes its own staging). A transient
    failure gets one retry; a sha256 mismatch is deterministic hub corruption
    and is never retried.
    """
    results: Dict[str, ouroboroshub.HubInstallResult] = {}

    def _ensure(slug: str) -> ouroboroshub.HubInstallResult:
        if slug in results:
            return results[slug]
        result = ouroboroshub.install(slug)
        if not result.ok and not any(m in result.error for m in _DETERMINISTIC_INSTALL_ERRORS):
            print(f"install({slug}): non-deterministic failure ({result.error}); retrying once")
            time.sleep(_TRANSIENT_RETRY_SLEEP_SEC)
            result = ouroboroshub.install(slug)
        assert result.ok, f"install({slug}) failed: {result.error}"
        results[slug] = result
        return result

    return _ensure


@pytest.fixture(scope="session")
def ensure_deps(install_skill):
    """Memoized real isolated-deps install (venv + pip) per deps skill."""
    fingerprints: Dict[str, Dict[str, Any]] = {}

    def _ensure(slug: str) -> Dict[str, Any]:
        if slug in fingerprints:
            return fingerprints[slug]
        install_skill(slug)
        sanitized = _sanitize_skill_name(slug)
        loaded = find_skill(DATA_DIR, sanitized)
        assert loaded is not None, f"installed skill {sanitized!r} not discoverable"
        auto = auto_install_specs_for_skill(DATA_DIR, loaded)
        assert auto, f"{slug}: expected declared auto install specs"
        try:
            fingerprint = install_isolated_dependencies(DATA_DIR, sanitized, loaded.skill_dir, auto)
        except Exception as exc:  # noqa: BLE001 — pip failure text is opaque
            # PyPI/pip flakes are indistinguishable from real resolver errors
            # by text; one bounded retry is safe (venv reuse is idempotent and
            # the failed deps.json state is overwritten on success).
            print(f"{slug}: isolated deps install failed ({exc!r}); retrying once")
            time.sleep(_TRANSIENT_RETRY_SLEEP_SEC)
            fingerprint = install_isolated_dependencies(DATA_DIR, sanitized, loaded.skill_dir, auto)
        assert fingerprint.get("status") == "installed", (
            f"{slug}: isolated deps install failed: {json.dumps(fingerprint, ensure_ascii=False)[:4000]}"
        )
        fingerprints[slug] = fingerprint
        return fingerprint

    return _ensure


def _catalog_entry(hub_catalog: Dict[str, Any], slug: str) -> Dict[str, Any]:
    for item in hub_catalog.get("skills") or []:
        if isinstance(item, dict) and item.get("slug") == slug:
            return item
    raise AssertionError(f"slug {slug} not in catalog")


def _skill_dir(slug: str) -> pathlib.Path:
    return get_ouroboroshub_skills_dir() / _sanitize_skill_name(slug)


def _load_manifest(slug: str):
    text = (_skill_dir(slug) / "SKILL.md").read_text(encoding="utf-8")
    return parse_skill_manifest_text(text)


# ---------------------------------------------------------------------------
# Tier 1 — install: payload landed, sha256 re-verified, provenance sidecar.
# ---------------------------------------------------------------------------


@pytest.mark.timeout(300)
@pytest.mark.parametrize("slug", OFFICIAL_SKILLS)
def test_install_lands_payload_with_verified_provenance(slug, hub_catalog, install_skill):
    result = install_skill(slug)
    sanitized = _sanitize_skill_name(slug)
    target = _skill_dir(slug)
    assert result.target_dir == target and target.is_dir()
    assert (target / "SKILL.md").is_file()

    entry = _catalog_entry(hub_catalog, slug)
    for item in entry.get("files") or []:
        rel = str(item.get("path") or "")
        expected_sha = str(item.get("sha256") or "").lower()
        on_disk = target / pathlib.Path(*pathlib.PurePosixPath(rel).parts)
        assert on_disk.is_file(), f"{slug}: catalog file missing on disk: {rel}"
        actual_sha = hashlib.sha256(on_disk.read_bytes()).hexdigest()
        assert actual_sha == expected_sha, f"{slug}: sha256 mismatch for {rel}"

    provenance = json.loads((target / ".ouroboroshub.json").read_text(encoding="utf-8"))
    assert provenance["schema_version"] == 1
    assert provenance["source"] == "ouroboroshub"
    assert provenance["slug"] == slug
    assert provenance["sanitized_name"] == sanitized
    assert provenance["version"] == str(entry.get("version") or "")
    assert provenance["files"] == entry.get("files")
    if slug in DEPS_SKILLS:
        specs = provenance.get("install_specs") or {}
        assert specs.get("auto"), f"{slug}: expected auto install specs in provenance"
    else:
        assert not (provenance.get("install_specs") or {}).get("auto"), (
            f"{slug}: unexpected auto install specs — the smoke lane's deps coverage "
            f"(DEPS_SKILLS) no longer matches the catalog"
        )

    staging = get_ouroboroshub_skills_dir() / ".staging"
    leftovers = [p.name for p in staging.glob("ouroboroshub_skill_*")] if staging.is_dir() else []
    assert not leftovers, f"install left staging directories behind: {leftovers}"


# ---------------------------------------------------------------------------
# Tier 2 — manifest parses and matches the per-skill contract.
# ---------------------------------------------------------------------------


@pytest.mark.timeout(60)
@pytest.mark.parametrize("slug", OFFICIAL_SKILLS)
def test_manifest_parses_and_matches_contract(slug, install_skill):
    install_skill(slug)
    manifest = _load_manifest(slug)
    target = _skill_dir(slug)

    if slug == "video_gen":
        assert manifest.is_script()
        script_names = {
            str(s.get("name") if isinstance(s, dict) else s) for s in manifest.scripts
        }
        assert "generate.py" in script_names
        assert not manifest.entry
        assert "OPENROUTER_API_KEY" in manifest.env_from_settings
        return

    assert manifest.is_extension(), f"{slug}: expected type extension"
    assert manifest.entry == "plugin.py"
    assert (target / "plugin.py").is_file()

    keyed = {"perplexity", "nanobanana", "music_gen"}
    if slug in keyed:
        assert "OPENROUTER_API_KEY" in manifest.env_from_settings
    if slug == "telegram-bridge":
        # The protected bot token is REQUESTED by the manifest; this lane never
        # grants it and never enables the skill, so nothing can read it.
        assert "TELEGRAM_BOT_TOKEN" in manifest.env_from_settings
        assert manifest.subscribe_events
    if slug in {"duckduckgo", "weather", "backlog_manager", "a2a"}:
        assert not manifest.env_from_settings, f"{slug}: expected a keyless manifest"
    if slug == "a2a":
        # Companion daemon is validated STATICALLY only — never spawned here
        # (owner decision): install/preflight paths start no processes.
        assert len(manifest.companion_processes) == 1
        companion = manifest.companion_processes[0]
        assert companion.get("name") == "a2a_server"
        assert companion.get("runtime") == "python3"
        # Contract-level shape only (a hub-side flag addition must not red the
        # release gate): python3 runs the daemon script, which exists on disk.
        # command[1] is the parser-pinned script position (skill_manifest.py
        # rejects inline -c/-m forms), so trailing flags stay legal.
        command = companion.get("command") or []
        assert len(command) >= 2 and command[0] == "python3", command
        assert str(command[1]).endswith("a2a_daemon.py"), command
        assert (target / "scripts" / "a2a_daemon.py").is_file()


# ---------------------------------------------------------------------------
# Tier 3 — offline skill_preflight is clean (compile + widget + permissions).
# ---------------------------------------------------------------------------


@pytest.mark.timeout(120)
@pytest.mark.parametrize("slug", OFFICIAL_SKILLS)
def test_skill_preflight_is_clean(slug, install_skill):
    install_skill(slug)
    ctx = ToolContext(repo_dir=REPO_ROOT, drive_root=DATA_DIR)
    out = _handle_skill_preflight(ctx, skill=_sanitize_skill_name(slug))
    assert not out.startswith("⚠️"), out
    payload = json.loads(out)
    assert payload["ok"] is True, out
    assert payload["files_failed"] == 0, out
    assert all(f.get("ok") for f in payload["manifest"]), out


# ---------------------------------------------------------------------------
# Tier 4 — real pip into the per-skill isolated venv (a2a, duckduckgo).
# ---------------------------------------------------------------------------


@pytest.mark.timeout(900)
@pytest.mark.parametrize("slug", DEPS_SKILLS)
def test_isolated_deps_install_real_pip(slug, ensure_deps):
    ensure_deps(slug)
    sanitized = _sanitize_skill_name(slug)
    loaded = find_skill(DATA_DIR, sanitized)
    assert loaded is not None
    auto = auto_install_specs_for_skill(DATA_DIR, loaded)
    state = read_deps_state(DATA_DIR, sanitized, loaded.skill_dir)
    assert state.get("status") == "installed", state
    assert state.get("specs_hash") == install_specs_hash(auto), state

    venv_python = python_runtime_binary(loaded.skill_dir)
    assert venv_python is not None and venv_python.is_file(), (
        f"{slug}: isolated venv python missing under {loaded.skill_dir}"
    )
    check = DEPS_IMPORT_CHECK[slug]
    proof = subprocess.run(
        [str(venv_python), "-c", check],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    assert proof.returncode == 0, (
        f"{slug}: venv import check {check!r} failed:\n{proof.stderr[-4000:]}"
    )
    # Provenance: log the resolved package set so a red run caused by PyPI
    # drift is reconstructible from CI output alone.
    freeze = subprocess.run(
        [str(venv_python), "-m", "pip", "freeze"],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    print(f"{slug} resolved deps:\n{freeze.stdout.strip()}")


@pytest.mark.timeout(300)
def test_non_deps_skills_declare_no_auto_specs(install_skill):
    """Guard: DEPS_SKILLS stays in sync with what the catalog actually declares."""
    for slug in OFFICIAL_SKILLS:
        if slug in DEPS_SKILLS:
            continue
        install_skill(slug)
        loaded = find_skill(DATA_DIR, _sanitize_skill_name(slug))
        assert loaded is not None
        assert not auto_install_specs_for_skill(DATA_DIR, loaded), (
            f"{slug} now declares auto install specs — add it to DEPS_SKILLS coverage"
        )


# ---------------------------------------------------------------------------
# Tier 5 — command smoke for the keyless skills (direct plugin registration).
# ---------------------------------------------------------------------------


class _FakePluginAPI:
    """Minimal recording PluginAPI stand-in (tests/test_unix_computer_use_skill.py
    pattern). Only the surfaces these three plugins actually use exist —
    an unexpected call fails loudly with AttributeError instead of silently
    no-opping."""

    def __init__(self, state_dir: pathlib.Path, data_dir: pathlib.Path) -> None:
        self.state_dir = state_dir
        self.data_dir = data_dir
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.routes: Dict[str, Dict[str, Any]] = {}
        self.ui_tabs: Dict[str, Dict[str, Any]] = {}
        self.logs: list[tuple[str, str]] = []

    def get_state_dir(self) -> str:
        return str(self.state_dir)

    def get_runtime_info(self) -> Dict[str, Any]:
        return {"data_dir": str(self.data_dir)}

    def register_tool(self, name: str, handler: Any = None, **metadata: Any) -> None:
        self.tools[name] = {"handler": handler, "metadata": metadata}

    def register_route(self, path: str, handler: Any = None, **metadata: Any) -> None:
        self.routes[path] = {"handler": handler, "metadata": metadata}

    def register_ui_tab(self, tab_id: str, title: str = "", **metadata: Any) -> None:
        self.ui_tabs[tab_id] = {"title": title, "metadata": metadata}

    def log(self, level: str, message: str) -> None:
        self.logs.append((level, message))


def _register_plugin(slug: str, tmp_path: pathlib.Path) -> _FakePluginAPI:
    plugin_path = _skill_dir(slug) / "plugin.py"
    # Unique module name per slug: a bare "plugin" name would collide in
    # sys.modules across skills and silently reuse the first loaded module.
    spec = importlib.util.spec_from_file_location(f"skill_smoke_plugin_{slug}", plugin_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    api = _FakePluginAPI(state_dir=tmp_path / "state", data_dir=tmp_path / "data")
    api.state_dir.mkdir(parents=True, exist_ok=True)
    api.data_dir.mkdir(parents=True, exist_ok=True)
    module.register(api)
    return api


def _tool_handler(api: _FakePluginAPI, slug: str, name: str):
    assert name in api.tools, (
        f"{slug}: expected tool {name!r}; registered tools: {sorted(api.tools)}"
    )
    return api.tools[name]["handler"]


@pytest.mark.timeout(120)
def test_duckduckgo_search_returns_results(ensure_deps, tmp_path):
    ensure_deps("duckduckgo")
    skill_dir = _skill_dir("duckduckgo")
    # isolated_site_dirs_scope is the production sys.path bridge over the
    # Tier-4 venv (ddgs is also a core dependency, so this doubles as a check
    # that the bridge itself does not break plugin imports). On exit the scope
    # purges venv-resident modules from sys.modules and removes the injected
    # sys.path entries, so nothing leaks past this test.
    with isolated_site_dirs_scope(skill_dir, enabled=True):
        api = _register_plugin("duckduckgo", tmp_path)
        handler = _tool_handler(api, "duckduckgo", "search")
        payload = json.loads(handler(query="python programming language", max_results=5))
        if "error" in payload:
            # DDG rate-limits shared CI runner IPs; one retry, then honest red.
            print(f"duckduckgo first attempt error: {payload['error']}; retrying once")
            time.sleep(_TRANSIENT_RETRY_SLEEP_SEC)
            payload = json.loads(handler(query="python programming language", max_results=5))
    assert "error" not in payload, f"duckduckgo search failed: {payload}"
    assert payload["count"] >= 1, payload
    assert all(str(r.get("url", "")).startswith("http") for r in payload["results"]), payload


@pytest.mark.timeout(120)
def test_weather_fetch_returns_conditions(install_skill, tmp_path):
    install_skill("weather")
    api = _register_plugin("weather", tmp_path)
    handler = _tool_handler(api, "weather", "fetch")
    payload = json.loads(handler(city="London"))
    if "error" in payload:
        print(f"weather first attempt error: {payload['error']}; retrying once")
        time.sleep(_TRANSIENT_RETRY_SLEEP_SEC)
        payload = json.loads(handler(city="London"))
    assert "error" not in payload, f"weather fetch failed: {payload}"
    assert isinstance(payload.get("temp_c"), int), payload
    assert str(payload.get("condition") or "").strip(), payload
    assert payload.get("forecast_rows"), payload
    assert payload.get("weather_summary"), payload


@pytest.mark.timeout(60)
def test_backlog_manager_summary_offline(install_skill, tmp_path):
    install_skill("backlog_manager")
    api = _register_plugin("backlog_manager", tmp_path)
    handler = _tool_handler(api, "backlog_manager", "summary")
    summary = handler()
    assert summary.startswith("Backlog Manager: 0 visible items"), summary
    assert {"list", "add", "update", "move"} <= set(api.routes), sorted(api.routes)
    assert "backlog" in api.ui_tabs, sorted(api.ui_tabs)


# a2a command smoke is deliberately absent: its tools need a live A2A peer and
# its companion daemon, which this lane must not start (owner decision). Its
# coverage is Tier 2 static companion contract + Tier 3 daemon compile +
# Tier 4 real dependency install.
