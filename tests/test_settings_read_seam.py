"""The settings READ path: one normalization, applied by every reader.

A settings document on disk is written by whatever release the owner last used, so
reading one starts by translating it into today's vocabulary: coerce every known key
to its declared type, fold the deprecated per-subsystem retention keys into the
unified one, drop the keys a release retired, promote the renamed model slots (and
the singular scope-review pin), and repair secret placeholders. Every one of those
steps exists to PRESERVE an owner customization written under a former key.

That normalization used to live inside `load_settings`. `_owner_read_settings_raw` —
the reader behind every owner endpoint and behind the context-fit route resolver —
merged the shipped defaults over the RAW document instead and got none of it. On its
own that was a wrong read; combined with the read-modify-write those endpoints
perform it was destructive, because the defaults the merge invented were written back
as if the owner had chosen them, and the migration that would have rescued the legacy
value then found the new key already present and left it alone. Forever.

`config.normalize_settings_raw` is now that step, and both readers apply it. These
tests pin the golden it must keep producing, the property that lets a locked
read-modify-write apply it on every save (idempotence), the fact that a read writes
nothing, and the closed inventory of readers and writers that keeps the seam single.
"""

from __future__ import annotations

import json
import pathlib

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

# One owner-authored document, written entirely under keys a release renamed or
# retired. Every value differs from both its legacy default and its current one,
# so nothing here can be mistaken for "the shipped value".
LEGACY_OWNER_DOCUMENT = {
    "TOTAL_BUDGET": 77.0,
    "OUROBOROS_MODEL_CODE": "owner/heavy-choice",
    "OUROBOROS_VISION_MODEL": "owner/vision-choice",
    "OUROBOROS_MODEL_FALLBACK": "owner/fallback-choice",
    "USE_LOCAL_CODE": True,
    "OUROBOROS_SCOPE_REVIEW_MODEL": "owner/scope-pin",
    "OUROBOROS_SUBAGENT_WORKTREE_RETENTION_DAYS": 30,
    "OUROBOROS_SUBAGENT_CAPABILITY_DEPTH_LIMIT": 1,
}

# key -> the value the READ path must produce for the document above.
MIGRATED_OWNER_VALUES = {
    "OUROBOROS_MODEL_HEAVY": "owner/heavy-choice",
    "OUROBOROS_MODEL_VISION": "owner/vision-choice",
    "OUROBOROS_MODEL_FALLBACKS": "owner/fallback-choice",
    "USE_LOCAL_HEAVY": True,
    "OUROBOROS_SCOPE_REVIEW_MODELS": "owner/scope-pin",
    "OUROBOROS_GC_RETENTION_DAYS": 30,
}

RETIRED_GHOST = "OUROBOROS_SUBAGENT_CAPABILITY_DEPTH_LIMIT"


@pytest.fixture
def isolated_settings(tmp_path, monkeypatch):
    """A real settings file nobody else shares, with the ratchet env neutralised."""
    from ouroboros import config as cfg

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    settings_path = data_dir / "settings.json"
    monkeypatch.setattr(cfg, "DATA_DIR", data_dir, raising=True)
    monkeypatch.setattr(cfg, "SETTINGS_PATH", settings_path, raising=True)
    for key in cfg.SETTINGS_DEFAULTS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("OUROBOROS_MODEL_FALLBACK", raising=False)
    cfg.reset_runtime_mode_baseline_for_tests()
    yield settings_path
    cfg.reset_runtime_mode_baseline_for_tests()


def _seed(settings_path: pathlib.Path, document: dict) -> None:
    settings_path.write_text(json.dumps(document, indent=2), encoding="utf-8")


def _owner_app(handler_name: str, route: str, drive_root: pathlib.Path) -> Starlette:
    from ouroboros.gateway import settings as settings_mod

    app = Starlette(routes=[
        Route(route, endpoint=getattr(settings_mod, handler_name), methods=["POST"])])
    app.state.drive_root = drive_root
    return app


def test_load_settings_migrates_every_renamed_key_and_drops_the_retired_one(isolated_settings):
    """The golden the seam must keep producing: five raw-stage migrations, in the
    order that makes each of them work (the singular scope pin is promoted BEFORE
    the defaults supply the plural that would otherwise win)."""
    from ouroboros import config as cfg

    _seed(isolated_settings, LEGACY_OWNER_DOCUMENT)
    loaded = cfg.load_settings()

    for key, expected in MIGRATED_OWNER_VALUES.items():
        assert loaded[key] == expected, key
    for legacy in ("OUROBOROS_MODEL_CODE", "OUROBOROS_VISION_MODEL", "OUROBOROS_MODEL_FALLBACK",
                   "USE_LOCAL_CODE", "OUROBOROS_SUBAGENT_WORKTREE_RETENTION_DAYS"):
        assert legacy not in loaded, f"{legacy} survived its rename"
    assert RETIRED_GHOST not in loaded, "a retired key is still served to consumers"
    assert loaded["TOTAL_BUDGET"] == 77.0


def test_load_settings_coerces_declared_types_before_the_defaults_merge(isolated_settings):
    """The coercion half of the same stage: strings off disk reach consumers as the
    type the default declares, and a value that cannot be coerced falls back."""
    from ouroboros import config as cfg

    _seed(isolated_settings, {
        "OUROBOROS_MAX_WORKERS": "12",
        "TOTAL_BUDGET": "40.5",
        "MCP_ENABLED": "yes",
        "OUROBOROS_RUNTIME_MODE": "PRO",
        "OUROBOROS_SKILLS_REPO_PATH": "  ",
        "MCP_SERVERS": '[{"name": "one"}]',
        "OUROBOROS_TOOL_TIMEOUT_SEC": "not a number",
    })
    loaded = cfg.load_settings()

    assert loaded["OUROBOROS_MAX_WORKERS"] == 12
    assert loaded["TOTAL_BUDGET"] == 40.5
    assert loaded["MCP_ENABLED"] is True
    assert loaded["OUROBOROS_RUNTIME_MODE"] == "pro"
    assert loaded["OUROBOROS_SKILLS_REPO_PATH"] == ""
    assert loaded["MCP_SERVERS"] == [{"name": "one"}]
    assert loaded["OUROBOROS_TOOL_TIMEOUT_SEC"] == 600


def test_reading_settings_writes_nothing_to_disk(isolated_settings):
    """A read is a read on both readers: same bytes, same mtime, no lock left behind."""
    from ouroboros import config as cfg
    from ouroboros.gateway.owner_settings import _owner_read_settings_raw

    _seed(isolated_settings, LEGACY_OWNER_DOCUMENT)
    before = isolated_settings.read_bytes()
    before_mtime = isolated_settings.stat().st_mtime_ns

    for _ in range(3):
        cfg.load_settings()
        cfg.load_settings_lock_held(_settings_lock_held=False)
        _owner_read_settings_raw()

    assert isolated_settings.read_bytes() == before
    assert isolated_settings.stat().st_mtime_ns == before_mtime
    assert not pathlib.Path(str(isolated_settings) + ".lock").exists()


def test_the_one_read_that_writes_is_the_context_compatibility_migration(isolated_settings):
    """The single, deliberate exception, pinned rather than assumed.

    A document that carries a context mode WITHOUT the false provenance marker is
    ambiguous for the BIBLE P3 scope gate, and the one-window compatibility migration
    resolves it by writing the canonical pair back — under the settings lock, through
    the live-data guard. So `load_settings` on such a file performs exactly ONE write
    and is stable from then on. `_owner_read_settings_raw` performs NONE even on the
    same file: it uses the non-persisting normalizer, so an owner GET is always a read.

    This is easy to miss because a fixture whose document has no context keys makes any
    "a read writes nothing" assertion pass vacuously — which is how a live settings file
    got rewritten by what looked like a read-only smoke."""
    from ouroboros import config as cfg
    from ouroboros.gateway.owner_settings import _owner_read_settings_raw

    ambiguous = {"TOTAL_BUDGET": 10.0, "OUROBOROS_CONTEXT_MODE": "low"}

    _seed(isolated_settings, ambiguous)
    before = isolated_settings.read_bytes()
    for _ in range(3):
        _owner_read_settings_raw()
    assert isolated_settings.read_bytes() == before, "an owner read migrated the file"

    settings = cfg.load_settings()
    after_first = isolated_settings.read_bytes()
    assert after_first != before, "the compatibility migration did not converge"
    stored = json.loads(after_first.decode("utf-8"))
    assert stored["OUROBOROS_CONTEXT_MODE_AUTO_LOW"] == "false"
    assert stored["OUROBOROS_CONTEXT_MODE"] == "max", "ambiguous Low is not owner Low"
    assert stored["TOTAL_BUDGET"] == 10.0, "the migration rewrote more than the pair"
    assert settings["OUROBOROS_CONTEXT_MODE"] == "max"

    for _ in range(3):
        cfg.load_settings()
    assert isolated_settings.read_bytes() == after_first, "the migration is not idempotent"


def test_owner_read_settings_raw_applies_the_same_normalization_as_load_settings(
        isolated_settings):
    """The seam: "raw" means "without the RATCHETS", never "without the migrations".
    Both readers answer the same owner values for the same document."""
    from ouroboros import config as cfg
    from ouroboros.gateway.owner_settings import _owner_read_settings_raw

    _seed(isolated_settings, LEGACY_OWNER_DOCUMENT)
    raw = _owner_read_settings_raw()
    loaded = cfg.load_settings()

    for key, expected in MIGRATED_OWNER_VALUES.items():
        assert raw[key] == expected, key
        assert raw[key] == loaded[key], key
    assert "OUROBOROS_MODEL_CODE" not in raw
    assert "OUROBOROS_VISION_MODEL" not in raw
    assert RETIRED_GHOST not in raw


def test_one_owner_endpoint_write_preserves_every_owner_customization(isolated_settings):
    """The defect, gone: turning auto-grant off changes auto-grant and nothing else,
    and the retired ghost leaves the file on the way through."""
    from ouroboros import config as cfg

    _seed(isolated_settings, LEGACY_OWNER_DOCUMENT)
    before = cfg.load_settings()

    app = _owner_app("api_owner_auto_grant", "/api/owner/auto-grant", isolated_settings.parent)
    response = TestClient(app).post("/api/owner/auto-grant", json={"enabled": False})
    assert response.status_code == 200, response.text

    after = cfg.load_settings()
    assert after["OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS"] == "false", "the intended change"
    changed = {key for key in before if before[key] != after.get(key)}
    assert changed == {"OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS"}, changed
    for key, expected in MIGRATED_OWNER_VALUES.items():
        assert after[key] == expected, key
    stored = json.loads(isolated_settings.read_text(encoding="utf-8"))
    assert RETIRED_GHOST not in stored, "the retired ghost survived a full rewrite"


def test_every_owner_endpoint_reaches_the_same_normalized_read(isolated_settings):
    """The fix is one seam, not six patches: each single-decision owner endpoint,
    and the generic save, take their document from ``_owner_read_settings_raw`` —
    directly, or through the locked read-modify-write primitive built on it.

    The names are the SYNCHRONOUS bodies: every settings writer hands its body to a
    worker thread (the event loop must not freeze for a save), so the async endpoint
    is a two-line delegator and the document work lives in its ``_sync`` companion —
    the generic save one level deeper again, in the body its lock wrapper calls."""
    import ast

    from ouroboros.gateway import owner_settings as owner_mod
    from ouroboros.gateway import settings as settings_mod

    def _callers(module, callee: str) -> set:
        source = pathlib.Path(module.__file__).read_text(encoding="utf-8")
        return {
            node.name
            for node in ast.walk(ast.parse(source))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and any(
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Name)
                and inner.func.id == callee
                for inner in ast.walk(node)
            )
        }

    assert _callers(owner_mod, "_owner_read_settings_raw") == {"_owner_update_settings"}
    readers = _callers(settings_mod, "_owner_read_settings_raw") | _callers(
        settings_mod, "_owner_update_settings")
    assert readers == {
        "_api_owner_runtime_mode_sync",
        "_api_owner_auto_grant_sync",
        "_api_owner_context_mode_sync",
        "_api_owner_scope_review_floor_sync",
        "_api_owner_safety_mode_sync",
        "_api_settings_post_locked",
    }, readers


def test_normalize_settings_raw_is_idempotent(isolated_settings):
    """Property: a reader may apply the normalization to an already-normalized
    document — which is what the owner endpoints' locked read-modify-write does on
    every save — and get the same document back. A migration that fired twice would
    otherwise re-promote a value it had already consumed."""
    from ouroboros import config as cfg

    documents = [
        LEGACY_OWNER_DOCUMENT,
        {},
        dict(cfg.SETTINGS_DEFAULTS),
        {"OUROBOROS_MODEL_HEAVY": "already/new", "OUROBOROS_MODEL_CODE": "old/loser"},
        {"OUROBOROS_SCOPE_REVIEW_MODEL": "pin", "OUROBOROS_SCOPE_REVIEW_MODELS": "a,b"},
        {"OUROBOROS_SUBAGENT_WORKTREE_RETENTION_DAYS": 30,
         "OUROBOROS_SERVICE_LOG_RETENTION_DAYS": 14},
        {"OUROBOROS_MAX_WORKERS": "3", "MCP_SERVERS": '[{"name": "one"}]', "unknown_key": {"a": 1}},
        {RETIRED_GHOST: 9},
    ]
    for document in documents:
        once = cfg.normalize_settings_raw(document)
        assert cfg.normalize_settings_raw(once) == once, document
        assert cfg.normalize_settings_raw(dict(once)) == once, document
        # Pure: the caller's mapping is never mutated and nothing reaches the disk.
        snapshot = dict(document)
        cfg.normalize_settings_raw(document)
        assert document == snapshot
    assert not isolated_settings.exists()


def test_a_stale_owner_read_cannot_overwrite_a_change_it_never_saw(isolated_settings):
    """The unlocked read-modify-write, closed: a decision taken from an earlier read
    is bound to the document that read saw."""
    from ouroboros import config as cfg
    from ouroboros.gateway.owner_settings import (
        SettingsPreconditionFailed,
        _owner_update_settings,
        settings_document_digest,
    )

    _seed(isolated_settings, {"TOTAL_BUDGET": 10.0})
    stale = settings_document_digest()
    _seed(isolated_settings, {"TOTAL_BUDGET": 10.0, "OUROBOROS_MAX_ROUNDS": 42})

    with pytest.raises(SettingsPreconditionFailed):
        _owner_update_settings(lambda current: {**current, "TOTAL_BUDGET": 99.0}, stale)
    assert cfg.load_settings()["OUROBOROS_MAX_ROUNDS"] == 42, "the other change was reverted"
    assert cfg.load_settings()["TOTAL_BUDGET"] == 10.0

    _owner_update_settings(lambda current: {**current, "TOTAL_BUDGET": 99.0},
                           settings_document_digest())
    assert cfg.load_settings()["TOTAL_BUDGET"] == 99.0
    assert cfg.load_settings()["OUROBOROS_MAX_ROUNDS"] == 42


def test_a_transform_that_returns_nothing_writes_nothing(isolated_settings):
    """A no-change decision must not rewrite the file — the rewrite would race a
    concurrent save for zero information gain."""
    from ouroboros.gateway.owner_settings import _owner_update_settings

    _seed(isolated_settings, {"TOTAL_BUDGET": 10.0})
    before = isolated_settings.read_bytes()
    before_mtime = isolated_settings.stat().st_mtime_ns

    _owner_update_settings(lambda _current: None)

    assert isolated_settings.read_bytes() == before
    assert isolated_settings.stat().st_mtime_ns == before_mtime


def test_all_three_writers_serialize_a_document_to_the_same_bytes(isolated_settings):
    """One serializer: the config saver, the owner-endpoint writer's atomic helper and
    the packaged bootstrap saver produce identical text for identical content. They
    disagreed on ``ensure_ascii``, so the same document had two spellings on disk."""
    from ouroboros import config as cfg
    from ouroboros.packaged_cli import _save_settings
    from ouroboros.utils import atomic_write_json

    document = {"TOTAL_BUDGET": 10.0, "OUROBOROS_EVOLUTION_PERSISTENT_OBJECTIVE": "\u043f\u0440\u0438\u043e\u0440\u0438\u0442\u0435\u0442"}

    cfg.save_settings(dict(document))
    by_config = isolated_settings.read_text(encoding="utf-8")
    isolated_settings.unlink()

    atomic_write_json(isolated_settings, cfg.prepare_settings_for_persist(dict(document)),
                      trailing_newline=False)
    by_owner_endpoint = isolated_settings.read_text(encoding="utf-8")
    isolated_settings.unlink()

    _save_settings(isolated_settings, dict(document))
    by_packaged_cli = isolated_settings.read_text(encoding="utf-8")

    assert by_config == by_owner_endpoint == by_packaged_cli
    assert document["OUROBOROS_EVOLUTION_PERSISTENT_OBJECTIVE"] in by_config, (
        "the shared serializer escaped a non-ASCII owner value")


def test_the_packaged_bootstrap_writes_the_path_the_prologue_reads(monkeypatch, tmp_path):
    """The packaged saver owns its own path, and the persistence prologue proves its
    ratchets against ``config.SETTINGS_PATH``. That is only honest while the two are
    the same file, which the packaged runtime resolves by construction: both derive
    from ``Path.home() / "Ouroboros"`` when no path override is set."""
    import ast
    import inspect
    import pathlib as _pathlib

    from ouroboros import config as cfg
    from ouroboros import packaged_cli

    # Both derivations, side by side, from their own source: config's module-level
    # default chain and the packaged runtime's data dir.
    config_source = _pathlib.Path(cfg.__file__).read_text(encoding="utf-8")
    assert 'APP_ROOT = pathlib.Path(os.environ.get("OUROBOROS_APP_ROOT", HOME / "Ouroboros"))' in config_source
    assert 'DATA_DIR = pathlib.Path(os.environ.get("OUROBOROS_DATA_DIR", APP_ROOT / "data"))' in config_source
    assert 'SETTINGS_PATH = pathlib.Path(os.environ.get("OUROBOROS_SETTINGS_PATH", DATA_DIR / "settings.json"))' in config_source
    assert "HOME = pathlib.Path.home()" in config_source

    resolver = inspect.getsource(packaged_cli.resolve_packaged_runtime)
    assert 'app_root = pathlib.Path.home() / "Ouroboros"' in resolver
    assert 'data_dir=app_root / "data"' in resolver
    # ...and the saver is wired to that data dir, not to some other root.
    bootstrap = inspect.getsource(packaged_cli._bootstrap_runtime)
    assert '_save_settings(runtime.data_dir / "settings.json", settings)' in bootstrap
    assert ast.parse(resolver.strip()) is not None

def test_the_three_settings_writers_are_exactly_these_three():
    """No fourth writer: the persisting surfaces are the config saver, the owner
    endpoint seam, and the packaged CLI bootstrap saver — and all three go through
    the same persistence prologue and the same serializer."""
    import ast

    repo = pathlib.Path(__file__).resolve().parents[1]
    writers: set[str] = set()
    for relpath in ("ouroboros/config.py", "ouroboros/gateway/owner_settings.py",
                    "ouroboros/packaged_cli.py", "ouroboros/context_mode_compat.py",
                    "ouroboros/colab_bootstrap.py"):
        source = (repo / relpath).read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for call in ast.walk(node):
                if not isinstance(call, ast.Call):
                    continue
                text = ast.get_source_segment(source, call) or ""
                targets_settings = "settings" in text.lower() or "SETTINGS_PATH" in text
                writes = any(
                    marker in text
                    for marker in ("atomic_write_json(", "os.replace(", ".write_text(")
                )
                if writes and targets_settings:
                    writers.add(f"{relpath}::{node.name}")
    assert writers == {
        "ouroboros/config.py::save_settings",
        # The owner endpoints' write lives in the locked read-modify-write primitive.
        "ouroboros/gateway/owner_settings.py::_owner_update_settings",
        "ouroboros/packaged_cli.py::_save_settings",
        # Not settings documents: the one-window raw context pair migration, written
        # under the load lock, and the Colab bootstrap's own generated file.
        "ouroboros/context_mode_compat.py::normalize_and_persist_context_mode_compat",
        "ouroboros/colab_bootstrap.py::write_colab_settings",
    }, writers


# ---------------------------------------------------------------------------
# The retired-key seam. It had no test at all: nothing proved a retired key ever
# leaves the owner's file, and nothing proved a live key cannot be retired by
# accident. Both halves matter — `settings.json` is the owner's document, so an
# unrecognized key is deliberately KEPT, and only membership in this list makes a
# key's absence intentional rather than data loss.
# ---------------------------------------------------------------------------


def test_a_retired_key_is_absent_from_the_defaults_that_offer_it():
    """Retirement is a two-part statement. A key still in ``SETTINGS_DEFAULTS``
    would be dropped by the read and re-supplied by the defaults merge on the very
    same call — a loop that reads as "retired" and behaves as "live"."""
    from ouroboros import config as cfg

    assert cfg.RETIRED_SETTING_KEYS, "the seam exists"
    overlap = set(cfg.RETIRED_SETTING_KEYS) & set(cfg.SETTINGS_DEFAULTS)
    assert not overlap, overlap
    assert not set(cfg.RETIRED_SETTING_KEYS) & set(cfg.settings_env_keys())


def test_a_retired_key_is_dropped_by_every_reader(isolated_settings):
    from ouroboros import config as cfg
    from ouroboros.gateway.owner_settings import _owner_read_settings_raw

    stored = {key: "9999" for key in cfg.RETIRED_SETTING_KEYS}
    stored["TOTAL_BUDGET"] = 12.0
    _seed(isolated_settings, stored)

    for reader in (cfg.load_settings, _owner_read_settings_raw):
        settings = reader()
        assert settings["TOTAL_BUDGET"] == 12.0
        for key in cfg.RETIRED_SETTING_KEYS:
            assert key not in settings, f"{reader.__name__} still serves {key}"


def test_a_retired_key_leaves_the_file_on_the_next_owner_write(isolated_settings):
    """A read that drops the ghost is only half the retirement: the file the owner
    keeps must stop carrying it too. It does, without a migration step, because
    every writer persists what a reader produced — including the owner-endpoint
    path, which is the one that previously wrote the ghost straight back."""
    from ouroboros import config as cfg

    stored = {key: "9999" for key in cfg.RETIRED_SETTING_KEYS}
    stored["TOTAL_BUDGET"] = 12.0
    _seed(isolated_settings, stored)

    app = _owner_app("api_owner_auto_grant", "/api/owner/auto-grant", isolated_settings.parent)
    response = TestClient(app).post("/api/owner/auto-grant", json={"enabled": True})
    assert response.status_code == 200, response.text

    on_disk = json.loads(isolated_settings.read_text(encoding="utf-8"))
    assert on_disk["TOTAL_BUDGET"] == 12.0
    for key in cfg.RETIRED_SETTING_KEYS:
        assert key not in on_disk, f"{key} survived an owner-endpoint write"


def test_the_three_retired_timeout_knobs_are_gone_from_every_owner_surface():
    """The two flat wall-clock timeouts and the planning heartbeat-staleness knob
    stopped governing anything a release ago and spent their deprecation window
    announcing it. Nothing may still offer them: not the defaults, not the
    environment projection, not the hot-reload classification, not the docs."""
    import pathlib as _pathlib

    from ouroboros import config as cfg
    from ouroboros.gateway import settings as settings_mod

    retired = (
        "OUROBOROS_SOFT_TIMEOUT_SEC",
        "OUROBOROS_HARD_TIMEOUT_SEC",
        "OUROBOROS_PLAN_TASK_SWARM_HEARTBEAT_STALE_SEC",
    )
    for key in retired:
        assert key in cfg.RETIRED_SETTING_KEYS, key
        assert key not in cfg.SETTINGS_DEFAULTS, key
        assert key not in settings_mod._IMMEDIATE_KEYS, key
        assert key not in settings_mod._RESTART_REQUIRED_KEYS, key

    architecture = (_pathlib.Path(__file__).resolve().parents[1] / "docs" / "ARCHITECTURE.md")
    table_rows = [
        line for line in architecture.read_text(encoding="utf-8").splitlines()
        if any(line.startswith(f"| {key} |") for key in retired)
    ]
    assert table_rows == [], table_rows


def test_a_live_setting_cannot_be_retired_by_accident():
    """The inverse tripwire: three keys that look retired and are not. A behavioural
    no-op is not the test — `until_deadline` lifts a real cap, the singular fallback
    env alias is a live benchmark contract, and the frozen-compat stall threshold is a
    true no-op the contract surface still declares."""
    from ouroboros import config as cfg
    from ouroboros.contracts import __name__ as _contracts_package  # noqa: F401

    assert "OUROBOROS_MODEL_FALLBACK" not in cfg.RETIRED_SETTING_KEYS
    assert cfg.parse_fallback_chain is not None
    for key in ("OUROBOROS_TASK_IDLE_TIMEOUT_SEC", "OUROBOROS_TASK_ABS_CEILING_SEC"):
        assert key in cfg.SETTINGS_DEFAULTS
        assert key not in cfg.RETIRED_SETTING_KEYS
