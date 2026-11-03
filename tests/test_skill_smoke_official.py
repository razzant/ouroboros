# tests/test_skill_smoke_official.py — official OuroborosHub skill install smoke.
#
# The `skill_smoke` lane: real-network install of the pinned official skills
# through the hub's git source (one live fetch of the hub repository) +
# preflight/deps/command probes + the Tier 6 review→grant→
# enable-persistence flow (CI review step only). Canonical lane description (purpose,
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
import os
import pathlib
import subprocess
import time
from typing import Any, Dict

import pytest

from ouroboros.config import DATA_DIR
from ouroboros.contracts.skill_manifest import parse_skill_manifest_text
from ouroboros.extension_isolated_deps import isolated_site_dirs_scope
from ouroboros.marketplace.install_specs import install_specs_hash
from ouroboros.marketplace.isolated_deps import (
    install_isolated_dependencies,
    python_runtime_binary,
    read_deps_state,
)
from ouroboros.settings_defaults import SETTINGS_DEFAULTS
from ouroboros.skill_dependencies import auto_install_specs_for_skill
from ouroboros.skill_loader import (
    _sanitize_skill_name,
    find_skill,
    grant_status_for_skill,
    load_enabled,
    save_enabled,
)
from ouroboros.skill_readiness import skill_readiness_for_execution
from ouroboros.tools.registry import ToolContext
from ouroboros.tools.skill_preflight import _handle_skill_preflight

pytestmark = pytest.mark.skill_smoke

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

OFFICIAL_SKILLS = [
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

# Deterministic install refusals (SkillInstallError codes) — never retried: a digest
# mismatch is hub corruption, an unknown or withdrawn entry is a catalog fact.
_DETERMINISTIC_INSTALL_CODES = frozenset({
    "digest_mismatch", "not_found", "withdrawn", "no_release", "payload_refused", "payload_unreadable",
    "name_taken", "update_instead",
})

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


def _hub_rows():
    from ouroboros.skill_sources import load_sources

    return [row for row in load_sources() if row["kind"] == "hub"]


@pytest.fixture(scope="session")
def hub_catalog():
    """Fetch the LIVE hub repository exactly once through its git source.

    The fetch exercises the real transport (shallow clone, snapshot). The v1
    ``catalog.json`` the hub keeps for older clients carries per-file hashes,
    which Tier 1 re-verifies on disk; its digest line makes a red run
    reconstructible against the exact hub commit that was tested.
    """
    from ouroboros.skill_source_catalog import refresh_source
    from ouroboros.skill_source_git import clone_dir

    rows = _hub_rows()
    assert rows, "no hub source is connected"
    meta = {}
    for attempt in (1, 2):
        meta = refresh_source(DATA_DIR, rows[0])
        if not meta.get("error"):
            break
        print(f"hub fetch attempt {attempt} failed ({meta.get('error')}); retrying once")
        time.sleep(_TRANSIENT_RETRY_SLEEP_SEC)
    assert not meta.get("error"), f"hub fetch failed: {meta.get('error')}"
    catalog = json.loads((clone_dir(DATA_DIR, rows[0]) / "catalog.json").read_text(encoding="utf-8"))
    slugs = {
        str(item.get("slug") or ""): str(item.get("version") or "")
        for item in catalog.get("skills") or []
        if isinstance(item, dict)
    }
    print(f"OuroborosHub commit={meta.get('commit')}")
    print("catalog versions: " + ", ".join(f"{s}={slugs.get(s, '?')}" for s in OFFICIAL_SKILLS))
    missing = [slug for slug in OFFICIAL_SKILLS if slug not in slugs]
    assert not missing, f"hub catalog is missing expected skills: {missing}"
    yield catalog


@pytest.fixture(scope="session")
def install_skill(hub_catalog):
    """Memoizing installer through the shared source lifecycle: each slug at most once per session.

    The item comes from whichever hub row offers it (Official or Community). A transient fetch failure
    gets one retry; deterministic refusals (a digest mismatch first) are never retried.
    """
    from ouroboros.skill_source_install import SkillInstallError, install_item

    results: Dict[str, Dict[str, Any]] = {}

    def _install(slug: str) -> Dict[str, Any]:
        last: Exception | None = None
        for row in _hub_rows():
            try:
                return install_item(DATA_DIR, f"{row['id']}/{slug}", turn_on_when_ready=False)
            except SkillInstallError as exc:
                if exc.code != "not_found":
                    raise
                last = exc
        raise AssertionError(f"{slug} is offered by no hub row: {last}")

    def _ensure(slug: str) -> Dict[str, Any]:
        if slug in results:
            return results[slug]
        try:
            result = _install(slug)
        except SkillInstallError as exc:
            if exc.code in _DETERMINISTIC_INSTALL_CODES:
                raise AssertionError(f"install({slug}) refused: {exc.code}: {exc}") from exc
            print(f"install({slug}): transient failure ({exc}); retrying once")
            time.sleep(_TRANSIENT_RETRY_SLEEP_SEC)
            result = _install(slug)
        assert result["ok"], f"install({slug}) failed: {result}"
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
    return DATA_DIR / "skills" / "ouroboroshub" / _sanitize_skill_name(slug)


def _load_manifest(slug: str):
    text = (_skill_dir(slug) / "SKILL.md").read_text(encoding="utf-8")
    return parse_skill_manifest_text(text)


# ---------------------------------------------------------------------------
# Tier 1 — install: payload landed, sha256 re-verified, provenance sidecar.
# ---------------------------------------------------------------------------


@pytest.mark.timeout(300)
@pytest.mark.parametrize("slug", OFFICIAL_SKILLS)
def test_install_lands_payload_with_verified_provenance(slug, hub_catalog, install_skill):
    from ouroboros.skill_identity import is_release_digest
    from ouroboros.skill_install_record import read_install_record

    result = install_skill(slug)
    sanitized = _sanitize_skill_name(slug)
    target = _skill_dir(slug)
    assert result["name"] == sanitized and target.is_dir()
    assert (target / "SKILL.md").is_file()

    entry = _catalog_entry(hub_catalog, slug)
    for item in entry.get("files") or []:
        rel = str(item.get("path") or "")
        expected_sha = str(item.get("sha256") or "").lower()
        on_disk = target / pathlib.Path(*pathlib.PurePosixPath(rel).parts)
        assert on_disk.is_file(), f"{slug}: catalog file missing on disk: {rel}"
        actual_sha = hashlib.sha256(on_disk.read_bytes()).hexdigest()
        assert actual_sha == expected_sha, f"{slug}: sha256 mismatch for {rel}"

    # The install record replaces the payload sidecar: nothing per-install lands in the payload.
    assert not (target / ".ouroboroshub.json").exists()
    record = read_install_record(DATA_DIR, sanitized)
    assert record is not None and record["entry"] == slug
    assert record["release"]["version"] == str(entry.get("version") or "")
    assert is_release_digest(record["release"]["digest"]) and record["release"]["commit"]
    assert record["basis"] in {"trusted", "review"}
    loaded = find_skill(DATA_DIR, sanitized)
    declared = auto_install_specs_for_skill(DATA_DIR, loaded)
    if slug in DEPS_SKILLS:
        assert declared, f"{slug}: expected auto install specs declared by its manifest"
    else:
        assert not declared, (
            f"{slug}: unexpected auto install specs — the smoke lane's deps coverage "
            f"(DEPS_SKILLS) no longer matches the hub"
        )

    staging = DATA_DIR / "skills" / "ouroboroshub" / ".staging"
    leftovers = [p.name for p in staging.iterdir()] if staging.is_dir() else []
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


# ---------------------------------------------------------------------------
# Tier 6 — production review→grant flow + enable-persistence prerequisites
# (real LLM skill review).
#
# Owner directive: CI must exercise the production flow "agent installs a
# skill → runs Ouroboros's OWN skill review → only if the review is OK the
# skill gets enabled" for a representative subset of official skills, with
# the review running on a cheap single reviewer slot (production users keep
# the strong default reviewer models — untouched here).
#
# Runs ONLY in the CI job's dedicated review step (ubuntu shard, secret in
# env, fresh pytest process): downloaded plugin code (Tier 5) must never
# share a process with OPENROUTER_API_KEY, so the job runs Tiers 1-5 and
# Tier 6 as two separate pytest invocations and only the Tier 6 step carries
# the secret. Missing key = hard red, not a skip (owner directive).
#
# One ordinary author task owns the INITIAL public skill_review and (only on
# evidence-grounded disagreement) ONE review_rebuttal. Its two paid-cycle cap
# counts infrastructure failures too: these may leave RED without reassessment.
# The independent persisted gate decides, never the author's final prose.
# Fresh HOME/state, disabled payload, no turn-on intent, env-only provider key;
# the server tree stops before enable-persistence checks below.
# Author kickoff/completion add paid calls even when the initial review is clean.
# ---------------------------------------------------------------------------

# Owner-picked subset: a2a (mandated) + duckduckgo (core keyless search;
# exercises the review→deps production ordering) + perplexity (keyed skill;
# exercises OPENROUTER_API_KEY auto-grant). The subset carried telegram-bridge
# for the protected-token and chat-permission path until that skill was retired
# from the official catalog; no remaining official skill requests a protected
# token, so that tier is unexercised here.
REVIEW_SKILLS = ("a2a", "duckduckgo", "perplexity")
_REVIEW_MODEL = "deepseek/deepseek-v4.1-flash"
# Shipped reviewer transport window; the ordinary task covers two windows
# plus author reasoning. Infrastructure errors stay RED without a new task.
_REVIEW_WINDOW_SEC = float(SETTINGS_DEFAULTS["OUROBOROS_LLM_TRANSPORT_READ_TIMEOUT_SEC"])
_REVIEW_WAVES = 2
_REVIEW_AUTHOR_TIMEOUT_SEC = int(_REVIEW_WAVES * _REVIEW_WINDOW_SEC + 600)
_REVIEW_TEST_TIMEOUT_SEC = int((_REVIEW_WAVES + 1) * _REVIEW_WINDOW_SEC + 900)
# Expected auto-granted settings keys per skill (grants are request-keyed,
# not value-keyed; empty values grant fine and only bite at runtime).
_REVIEW_EXPECTED_KEYS = {
    "a2a": set(),
    "duckduckgo": set(),
    "perplexity": {"OPENROUTER_API_KEY"},
}


# Session scope on purpose: session fixtures instantiate before any
# function-scoped fixture, and among session fixtures signature order wins —
# so listing this BEFORE install_skill makes the missing-key red fire before
# the live catalog fetch. Only Tier 6 requests it; Tiers 1-5 stay keyless.
@pytest.fixture(scope="session")
def review_secret():
    assert os.environ.get("OPENROUTER_API_KEY"), (
        "Tier 6 needs OPENROUTER_API_KEY in the environment (CI: the review "
        "step's secret env; locally: export a key). A missing key is a hard "
        "red by owner directive — forks/mirrors without the secret stay red."
    )


def _actor_reported_valuation_usd(review: Dict[str, Any]) -> float:
    """Non-authoritative reviewer telemetry, kept only for live-smoke diagnostics."""
    return sum(float(rec.get("cost_usd") or 0.0) for rec in review.get("raw_actor_records") or [])


@pytest.mark.timeout(_REVIEW_TEST_TIMEOUT_SEC)
# review_secret (session) is listed before install_skill (session): among
# same-scope fixtures signature order wins, so a fork without the secret reds
# on one crisp message before the live catalog fetch.
@pytest.mark.parametrize("slug", REVIEW_SKILLS)
def test_review_grants_and_enable(slug, review_secret, install_skill, tmp_path):
    install_skill(slug)
    name = _sanitize_skill_name(slug)
    started = time.time()

    from tests.candidate_checkout import candidate_checkout
    from tests.skill_smoke_author import run_author_review
    from ouroboros.marketplace.isolated_deps import read_deps_state

    with candidate_checkout(REPO_ROOT, tmp_path / "repo") as candidate:
        evidence = run_author_review(
            candidate=candidate, source_data=DATA_DIR, root=tmp_path / "author",
            name=name, model=_REVIEW_MODEL, timeout_sec=_REVIEW_AUTHOR_TIMEOUT_SEC,
            credential=("OPENROUTER_API_KEY", os.environ["OPENROUTER_API_KEY"]))
    data = evidence["data_root"]
    review = evidence["review"]
    status = review["status"]
    loaded = find_skill(data, name)
    deps_status = read_deps_state(data, name, loaded.skill_dir).get("status", "not_required")

    # Production ordering: dependencies reconcile only AFTER the executable
    # review, inside the same gateway wrapper.
    if slug in DEPS_SKILLS:
        assert deps_status == "installed", f"{slug}: deps_status={deps_status!r}"
    else:
        assert deps_status == "not_required", f"{slug}: deps_status={deps_status!r}"

    # Persisted review binds to the reloaded payload (fresh find_skill, not
    # the pre-review in-memory object).
    loaded = find_skill(data, name)
    assert loaded is not None and not loaded.load_error
    review = evidence["review"]
    assert review.get("content_hash") == loaded.content_hash
    models = list(review.get("reviewer_models") or [])
    assert len(models) == 1, models
    # Reviewer identity may carry the stable slot attribution suffix introduced
    # by the shared review substrate, while the model id remains the contract.
    model_id = models[0].split(" [", 1)[0].rsplit("#", 1)[0]
    assert model_id == _REVIEW_MODEL.replace("::", "/"), models
    # The official_hub reduced-noise profile is retired: Official is the trusted
    # source row's install basis now, so an explicit review is an ordinary one.
    assert review.get("review_profile") in (None, ""), review.get("review_profile")

    # Auto-grant ran inside review_skill (blocking enforcement ⇒ only on
    # clean/warnings). Grants are request-keyed: a requested key grants with an
    # empty settings value.
    grants = grant_status_for_skill(data, loaded)
    assert grants.get("all_granted") is True, grants
    expected_keys = _REVIEW_EXPECTED_KEYS[slug]
    granted_keys = set(grants.get("granted_keys") or [])
    assert granted_keys == expected_keys, (
        f"{slug}: granted_keys={sorted(granted_keys)} expected={sorted(expected_keys)}"
    )
    # Enable persistence prerequisites: the production toggle-gate facts plus
    # the top static readiness gate. The key-bearing server is already stopped;
    # enabling a2a here cannot poll, import its plugin or spawn a companion.
    save_enabled(data, name, True)
    assert load_enabled(data, name) is True
    reloaded = find_skill(data, name)
    assert reloaded is not None, f"{slug}: skill vanished from discovery after enable"
    readiness = skill_readiness_for_execution(data, reloaded)
    assert readiness.ready, (
        f"{slug}: readiness not ready — blockers={readiness.blockers} "
        f"agent_fixable={readiness.agent_fixable_blockers} "
        f"owner_action={readiness.owner_action_blockers}"
    )

    print(
        f"{slug}: review={status} slots={models} "
        f"actor_reported_valuation=${_actor_reported_valuation_usd(review):.4f} "
        f"elapsed={time.time() - started:.1f}s"
    )
