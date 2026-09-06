"""The settings READ path: one normalization, applied by every reader.

A settings document on disk is written by whatever release the owner last used, so
reading one starts by translating it into today's vocabulary: coerce every known key
to its declared type, fold the deprecated per-subsystem retention keys into the
unified one, seed the shared review-cycle cap from the retired acceptance-pass count,
drop the keys a release retired, promote the renamed model slots, and repair secret
placeholders. Every one of those steps exists to PRESERVE an owner customization
written under a former key.

That normalization used to live inside `load_settings`. `_owner_read_settings_raw` —
the reader behind every owner endpoint, and at the time behind the context-fit route
resolver too — merged the shipped defaults over the RAW document instead and got none
of it. On its own that was a wrong read; combined with the read-modify-write those
endpoints perform it was destructive, because the defaults the merge invented were
written back as if the owner had chosen them, and the migration that would have rescued
the legacy value then found the new key already present and left it alone. Forever.

`config.normalize_settings_raw` is now that step, and every reader applies it: the
loader, the owner reader (through the loader's verified read primitive, so a pinned
snapshot that changed refuses both), and the Colab re-run over the Drive document. It
carries the VOCABULARY normalization only; the context-fit route resolver reads the
provider-normalized EFFECTIVE document — the route the loop runs — not the owner-raw
one. These tests pin the golden it must keep producing, the property that lets a
locked read-modify-write apply it on every save (idempotence), the fact that a read
writes nothing, the one serializer's bytes on disk from every writer in the tree's
closed inventory (`tests._shared.SETTINGS_WRITERS`; the inventory itself is closed by
the whole-tree tripwire in tests/test_runtime_mode_elevation.py over the same list), and
the inventory of the owner endpoints that read — closed over the two modules that own
the owner write seam, which is where an endpoint could grow a second reader;
`gateway/onboarding.py` calls the same reader for its preview, legitimately and through
the same seam.
"""

from __future__ import annotations

import json
import pathlib

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from tests._shared import SETTINGS_WRITERS, calls_function

# One owner-authored document, written entirely under keys a release renamed or
# retired. Every value differs from both its legacy default and its current one,
# so nothing here can be mistaken for "the shipped value".
LEGACY_OWNER_DOCUMENT = {
    "TOTAL_BUDGET": 77.0,
    "OUROBOROS_MODEL_CODE": "owner/heavy-choice",
    "OUROBOROS_VISION_MODEL": "owner/vision-choice",
    "OUROBOROS_MODEL_FALLBACK": "owner/fallback-choice",
    "USE_LOCAL_CODE": True,
    "OUROBOROS_SUBAGENT_WORKTREE_RETENTION_DAYS": 30,
    "OUROBOROS_ACCEPTANCE_MAX_IMPROVEMENT_PASSES": 3,
    "OUROBOROS_SUBAGENT_CAPABILITY_DEPTH_LIMIT": 1,
}

# key -> the value the READ path must produce for the document above.
MIGRATED_OWNER_VALUES = {
    "OUROBOROS_MODEL_HEAVY": "owner/heavy-choice",
    "OUROBOROS_MODEL_VISION": "owner/vision-choice",
    "OUROBOROS_MODEL_FALLBACKS": "owner/fallback-choice",
    "USE_LOCAL_HEAVY": True,
    "OUROBOROS_GC_RETENTION_DAYS": 30,
    # cycles = passes + 1: the retired acceptance-pass count seeds the shared cap.
    "OUROBOROS_REVIEW_MAX_CYCLES": "4",
}

RETIRED_GHOST = "OUROBOROS_SUBAGENT_CAPABILITY_DEPTH_LIMIT"

# The document's keys a release RENAMED (as opposed to retired): every one must be
# gone from a normalized read, its value promoted to the key above.
RENAMED_LEGACY_KEYS = ("OUROBOROS_MODEL_CODE", "OUROBOROS_VISION_MODEL", "OUROBOROS_MODEL_FALLBACK",
                       "USE_LOCAL_CODE", "OUROBOROS_SUBAGENT_WORKTREE_RETENTION_DAYS",
                       "OUROBOROS_ACCEPTANCE_MAX_IMPROVEMENT_PASSES")


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
    """The golden the seam must keep producing: the raw-stage migrations, in the
    order that makes each of them work — the retired acceptance-pass count seeds the
    review-cycle cap BEFORE the retired purge drops it, and the purge runs BEFORE the
    slot rename so a retired spelling is never promoted into a live key."""
    from ouroboros import config as cfg

    _seed(isolated_settings, LEGACY_OWNER_DOCUMENT)
    loaded = cfg.load_settings()

    for key, expected in MIGRATED_OWNER_VALUES.items():
        assert loaded[key] == expected, key
    for legacy in RENAMED_LEGACY_KEYS:
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
    assert "OUROBOROS_ACCEPTANCE_MAX_IMPROVEMENT_PASSES" not in raw
    assert RETIRED_GHOST not in raw


def test_the_context_fit_route_is_the_provider_normalized_effective_route(
        isolated_settings, monkeypatch):
    """The read seam carries the vocabulary normalization only. The PROVIDER
    normalization is a separate derivation that is never persisted, so a consumer
    that needs the route the loop actually runs must re-derive it over the effective
    document. A direct-provider install with no explicit model has no main slot at
    all outside that derivation: read from the owner-raw document, the context-fit
    probe resolved a window for an OpenRouter model the loop never runs, and `fits`
    was computed against the wrong route on every ordinary task.

    The expectation comes from the LOOP side rather than from the resolver's own
    expression — an equality written as `_active_main_route(apply_runtime_provider_
    defaults(load_settings()))` restates the implementation and stays green if the
    implementation and the expectation drift together. `apply_task_start_settings()`
    is what a task start projects into the environment, so the model it leaves there
    IS the model the next task runs. That projection is rolled back before the route
    is resolved, so the resolver has to reach the same answer on its own instead of
    reading the environment the projection left behind."""
    import os

    from ouroboros.context_fit import _failed_route_evidence, resolve_context_fit_route
    from ouroboros.gateway.owner_settings import _owner_read_settings_raw
    from ouroboros.gateway.settings import _active_main_route
    from ouroboros.provider_models import provider_for_model
    from ouroboros.subagent_runtime import apply_task_start_settings

    monkeypatch.setattr(os, "environ", dict(os.environ))
    _seed(isolated_settings, {"ANTHROPIC_API_KEY": "sk-ant-test"})

    pristine = dict(os.environ)
    apply_task_start_settings()
    loop_model = os.environ["OUROBOROS_MODEL"]
    os.environ.clear()
    os.environ.update(pristine)

    assert provider_for_model(loop_model) == "anthropic", (
        "the fixture is a direct-provider install")
    # The fixture is not vacuous: the owner-raw document answers a different route.
    assert _active_main_route(_owner_read_settings_raw())["provider"] != "anthropic"

    route, _evidence = resolve_context_fit_route({"model": ""}, allow_fetch=False)
    assert route["model"] == loop_model, "the probe sized a model the loop never runs"
    assert route["provider"] == "anthropic"
    failed_route, _failed = _failed_route_evidence({"model": ""})
    assert failed_route == route


def test_a_pinned_snapshot_that_changed_refuses_every_reader(isolated_settings, monkeypatch):
    """One read primitive on both readers: under a strict benchmark pin the owner
    reader refuses a changed snapshot exactly as the loader does, instead of quietly
    serving the unverified file — and the context-fit route, which reads through the
    loader, does not resolve a route from an unverified document either."""
    from hashlib import sha256

    from ouroboros import config as cfg
    from ouroboros.context_fit import resolve_context_fit_route
    from ouroboros.gateway.owner_settings import _owner_read_settings_raw

    _seed(isolated_settings, LEGACY_OWNER_DOCUMENT)
    monkeypatch.setenv(cfg.SETTINGS_INTEGRITY_ENV, "0" * 64)
    for reader in (cfg.load_settings, _owner_read_settings_raw):
        with pytest.raises(cfg.SettingsIntegrityError):
            reader()
    with pytest.raises(cfg.SettingsIntegrityError):
        resolve_context_fit_route({"model": ""}, allow_fetch=False)

    monkeypatch.setenv(cfg.SETTINGS_INTEGRITY_ENV, sha256(isolated_settings.read_bytes()).hexdigest())
    assert _owner_read_settings_raw()["TOTAL_BUDGET"] == 77.0
    assert cfg.load_settings()["TOTAL_BUDGET"] == 77.0
    route, _evidence = resolve_context_fit_route({"model": ""}, allow_fetch=False)
    assert route["model"]


def test_the_colab_re_run_reads_the_drive_document_through_the_same_normalization(tmp_path):
    """The third reader (spec 4.3.5-7, the Colab fixture): the quickstart re-reads the
    Drive ``settings.json`` it wrote last session and hands it to ``build_colab_settings``
    as ``existing``. That is an install's settings document like any other, so the same
    raw-stage normalization runs BEFORE the shipped defaults are merged (the launch knobs
    then win), and what ``write_colab_settings`` puts back on Drive is the one on-disk
    spelling. Folding only the slot rename kept the retired ghost, replaced the folded
    retention and review-cycle customizations with their defaults, and wrote all of it
    back to Drive as the owner's choices — the owner-endpoint defect, one reader over."""
    from ouroboros import config as cfg
    from ouroboros.colab_bootstrap import build_colab_settings, write_colab_settings

    out = build_colab_settings({}, existing=dict(LEGACY_OWNER_DOCUMENT))

    for key, expected in MIGRATED_OWNER_VALUES.items():
        assert out[key] == expected, key
    for legacy in RENAMED_LEGACY_KEYS:
        assert legacy not in out, f"{legacy} survived the re-run"
    assert RETIRED_GHOST not in out, "a retired key is written back to Drive"

    written = write_colab_settings(tmp_path / "drive", out)
    assert written.read_text(encoding="utf-8") == cfg.serialize_settings(out)
    assert json.loads(written.read_text(encoding="utf-8")) == out


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
    """The fix is one seam, not five patches: each single-decision owner endpoint,
    and the generic save, take their document from ``_owner_read_settings_raw`` —
    directly, or through the locked read-modify-write primitive built on it. The set is
    closed over the two modules that own the owner write seam, which is where a second
    reader could appear; `gateway/onboarding.py` calls the same reader for its subagent
    preview, which is that reader used as intended rather than a bypass of it.

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
        # Both retired comma spellings: purged, never promoted into anything.
        {"OUROBOROS_SCOPE_REVIEW_MODEL": "pin", "OUROBOROS_SCOPE_REVIEW_MODELS": "a,b"},
        # An owner-authored cycle cap wins over the retired pass count that would seed it.
        {"OUROBOROS_ACCEPTANCE_MAX_IMPROVEMENT_PASSES": "3", "OUROBOROS_REVIEW_MAX_CYCLES": "2"},
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


def test_an_unreadable_settings_file_never_compares_equal(monkeypatch):
    """Exactly two digests can ever COMPARE EQUAL: a real digest and the absent
    sentinel. Folding every read failure of one exception class into one stable
    token would let a swap between two DIFFERENT unreadable files satisfy the
    staleness check — fail-OPEN, and reachable, because a reader silently falls
    back to defaults when it cannot read while the atomic rename still lands.

    The refusal is injected rather than provoked with chmod 0o000: on Windows chmod
    only toggles the read-only bit, so the unreadable branch would never run."""
    from ouroboros import config as cfg
    from ouroboros.gateway.owner_settings import settings_document_digest

    class _Unreadable:
        def read_bytes(self):
            raise PermissionError("injected: settings unreadable")

    monkeypatch.setattr(cfg, "SETTINGS_PATH", _Unreadable(), raising=False)
    first = settings_document_digest()
    second = settings_document_digest()

    assert first.startswith("unreadable:") and second.startswith("unreadable:")
    assert first != second, "an unreadable file must refuse, never satisfy equality"


def test_every_settings_writer_puts_the_one_serializers_bytes_on_disk(isolated_settings):
    """One serializer, one spelling. Each of the five writers, driven through its real
    entry point, leaves exactly ``serialize_settings(document).encode("utf-8")`` on disk:
    the three prologue-routed savers produce identical bytes for identical content, and
    the two exempt writers — the context-pair migration, which is the one READ that
    writes, and the Colab generator for the Drive root — produce the serializer's bytes
    for what they persist. The writers once disagreed on ``ensure_ascii``, and a writer
    that re-derived the JSON text itself agreed with the serializer only by coincidence:
    mutate the serializer and that writer keeps its old spelling while the others follow.
    So besides the bytes, each writer is required to CALL the serializer.

    The comparison is on BYTES against the serializer's own output, so a trailing newline
    or a platform newline translation that crept in after the serializer fails it. The
    half no Linux run can observe is pinned on the MECHANISM: ``Path.write_text`` and a
    text-mode ``open`` translate ``\n`` to ``\r\n`` on Windows and the byte-exact helpers
    never do, so every commit a writer makes must be one of those helpers (or
    ``Path.write_bytes``, the config saver's rename-less fallback) and nothing else that
    reaches the disk — a positive rule, because denying one spelling of a text-mode
    write let a text-mode ``open`` through.

    The spelling itself is pinned as a golden too. Once every writer calls the serializer
    the agreement above cannot see the serializer change at all — a serializer that
    started appending a newline or escaping non-ASCII would be green on every writer at
    once — and the spelling is what the digest precondition, a pinned benchmark snapshot
    and the migration's convergence observe. The golden is the one half of this pin that
    still reads the serializer."""
    import ast
    import re

    from ouroboros import config as cfg
    from ouroboros.colab_bootstrap import write_colab_settings
    from ouroboros.gateway.owner_settings import _owner_update_settings
    from ouroboros.packaged_cli import _save_settings

    document = {"TOTAL_BUDGET": 10.0, "OUROBOROS_EVOLUTION_PERSISTENT_OBJECTIVE": "приоритет"}
    on_disk: dict[str, bytes] = {}

    cfg.save_settings(dict(document))
    on_disk["config.save_settings"] = isolated_settings.read_bytes()
    isolated_settings.unlink()

    _owner_update_settings(lambda _current: dict(document))
    on_disk["owner_settings._owner_update_settings"] = isolated_settings.read_bytes()
    isolated_settings.unlink()

    _save_settings(isolated_settings, dict(document))
    on_disk["packaged_cli._save_settings"] = isolated_settings.read_bytes()

    # The migration is the one read that writes: an ambiguous Low makes the load persist.
    _seed(isolated_settings, {**document, "OUROBOROS_CONTEXT_MODE": "low"})
    cfg.load_settings()
    on_disk["context_mode_compat.normalize_and_persist_context_mode_compat"] = (
        isolated_settings.read_bytes())

    on_disk["colab_bootstrap.write_colab_settings"] = write_colab_settings(
        isolated_settings.parent / "drive", dict(document)).read_bytes()

    routed = [on_disk[name] for name in ("config.save_settings",
                                         "owner_settings._owner_update_settings",
                                         "packaged_cli._save_settings")]
    assert routed[0] == routed[1] == routed[2]
    for name, written in on_disk.items():
        assert written == cfg.serialize_settings(json.loads(written)).encode("utf-8"), (
            f"{name}: the file is not exactly the serializer's bytes for the document it "
            "holds: something between serialize_settings and the disk re-derived the text, "
            "added a trailing newline or translated a newline")
        assert "приоритет".encode("utf-8") in written, f"{name} escaped a non-ASCII value"

    byte_exact = {"write_text_atomic(", "write_bytes_atomic(", "atomic_write_json(", ".write_bytes("}
    repo = pathlib.Path(__file__).resolve().parents[1]
    for relpath, name in SETTINGS_WRITERS:
        source = (repo / relpath).read_text(encoding="utf-8")
        (node,) = [n for n in ast.walk(ast.parse(source))
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]
        assert calls_function(node, "serialize_settings"), (
            f"{relpath}::{name} persists a settings document without calling the one "
            "serializer: its bytes agree with the others only while nobody changes either.")
        commits = re.findall(
            r"write_text_atomic\(|write_bytes_atomic\(|atomic_write_json\(|\.write_bytes\("
            r"|\.write_text\(|\.write\(|json\.dump\(|\bopen\(",
            ast.get_source_segment(source, node) or "")
        assert commits and set(commits) <= byte_exact, (
            f"{relpath}::{name} commits a settings document through {sorted(set(commits))}: a "
            "text-mode write turns every newline into CRLF on Windows while the other writers "
            "keep LF. Commit through utils.write_text_atomic (or write_bytes), which is "
            "byte-exact everywhere.")

    assert cfg.serialize_settings(document) == (
        '{\n  "TOTAL_BUDGET": 10.0,\n  "OUROBOROS_EVOLUTION_PERSISTENT_OBJECTIVE": "приоритет"\n}'
    ), "the one on-disk spelling changed: UTF-8 (ensure_ascii=False), two-space indent, no trailing newline"


def test_the_packaged_bootstrap_writes_the_path_the_prologue_reads(isolated_settings, tmp_path,
                                                                    monkeypatch):
    """The packaged saver owns its own path, and the persistence prologue proves its
    ratchets against ``config.SETTINGS_PATH``. That is only honest while the two are
    the same file, which holds by construction in a process that carries no path
    override: both derive from ``Path.home() / "Ouroboros"``. Asserted as the property
    (the two resolutions, computed), not as the text of either resolver. The packaged
    runtime ignores the environment by design (the inner CLI child is handed the
    packaged paths explicitly), so the identity is CONDITIONAL on the outer process
    having no override — disclosed in the saver's docstring, dormant while the
    bootstrap callback has no caller."""
    from ouroboros import config as cfg
    from ouroboros import packaged_cli

    for key in ("OUROBOROS_APP_ROOT", "OUROBOROS_REPO_DIR", "OUROBOROS_DATA_DIR",
                "OUROBOROS_SETTINGS_PATH"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(packaged_cli, "_find_bundle_root", lambda _paths: tmp_path)
    monkeypatch.setattr(packaged_cli, "_find_embedded_python", lambda _root: tmp_path / "python")
    monkeypatch.setattr(packaged_cli, "_read_version", lambda _root: "0.0.0")

    runtime = packaged_cli.resolve_packaged_runtime()
    assert runtime.data_dir / "settings.json" == cfg.resolve_data_dir() / "settings.json"

    # ...and the bootstrap wires its saver to exactly that file: the context it hands
    # `bootstrap_repo` names the file, and its saver persists there in seam bytes.
    contexts: list = []
    monkeypatch.setattr(packaged_cli, "check_git", lambda _windows: True)
    monkeypatch.setattr(packaged_cli, "bootstrap_repo", contexts.append)
    scratch = packaged_cli.PackagedRuntime(
        bundle_root=tmp_path, embedded_python=tmp_path / "python", app_root=tmp_path,
        repo_dir=tmp_path / "repo", data_dir=tmp_path / "data", app_version="0.0.0")
    packaged_cli._bootstrap_runtime(scratch)
    (context,) = contexts
    assert context.settings_path == scratch.data_dir / "settings.json"
    context.save_settings({"TOTAL_BUDGET": 1.0})
    assert context.settings_path.read_bytes() == cfg.serialize_settings(
        cfg.prepare_settings_for_persist({"TOTAL_BUDGET": 1.0})).encode("utf-8")

    # ...and it takes the write GUARDS on that same path. A bootstrap save is still a
    # settings write: under a strict benchmark pin it must refuse like the other two
    # writers instead of overwriting the snapshot the pin exists to hold still.
    before = context.settings_path.read_bytes()
    monkeypatch.setenv(cfg.SETTINGS_INTEGRITY_ENV, "0" * 64)
    with pytest.raises(cfg.SettingsIntegrityError):
        context.save_settings({"TOTAL_BUDGET": 2.0})
    assert context.settings_path.read_bytes() == before


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


def test_a_retired_key_is_absent_from_every_surface_that_would_react_to_it():
    """The half that neither reading nor writing covers: the surfaces that CLASSIFY a
    key. The settings endpoint answers "applied immediately" or "restart required" per
    saved key, and the ARCHITECTURE settings table gives each key a default — both would
    then be talking about a knob no reader will ever serve again. Pinned over the whole
    retired list rather than over the names one retirement happened to add, so the next
    retirement is covered by membership instead of by remembering.

    A documentation ROW is not the defect and is not forbidden: telling an owner where
    the value they wrote under the old key went is the point of a retirement, and three
    retired keys carry exactly such a row today. What may not survive is the DEFAULT
    column offering a value, because that is the table saying the key is still a knob."""
    from ouroboros import config as cfg
    from ouroboros.gateway import settings as settings_mod

    documented = (pathlib.Path(__file__).resolve().parents[1]
                  / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8").splitlines()
    for key in cfg.RETIRED_SETTING_KEYS:
        assert key not in settings_mod._IMMEDIATE_KEYS, key
        assert key not in settings_mod._RESTART_REQUIRED_KEYS, key
        for row in [line for line in documented if line.startswith(f"| {key} |")]:
            default = row.split("|")[2].strip()
            assert default.startswith("(") and default.endswith(")"), (
                f"{key} is retired, but its settings-table row still offers the default "
                f"{default!r} — a default no reader will ever serve. State its status "
                f"instead, the way `(retired)` and `(env-only)` do.")


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


def test_a_live_setting_cannot_be_retired_by_accident():
    """The inverse tripwire: three keys that look retired and are not. A behavioural
    no-op is not the test — the activity-model pair replaced the retired flat
    wall-clock timeouts and governs real deadlines, and the singular fallback env
    alias is a live benchmark contract read by ``parse_fallback_chain``."""
    from ouroboros import config as cfg

    assert "OUROBOROS_MODEL_FALLBACK" not in cfg.RETIRED_SETTING_KEYS
    assert cfg.parse_fallback_chain is not None
    for key in ("OUROBOROS_TASK_IDLE_TIMEOUT_SEC", "OUROBOROS_TASK_ABS_CEILING_SEC"):
        assert key in cfg.SETTINGS_DEFAULTS
        assert key not in cfg.RETIRED_SETTING_KEYS


def test_a_retired_comma_list_triad_is_not_dropped_silently(isolated_settings, caplog):
    """Review M2, the first 7.0 boot. An install that configured its review
    panel with the comma-list keys loses that configuration on upgrade: the
    keys are in RETIRED_SETTING_KEYS, so the raw-stage normalization purges
    them before migration, and the install runs the SHIPPED default reviewer
    panel instead. That is the ratified migration ("move the config to the
    structured OUROBOROS_REVIEWER_SLOTS BEFORE the upgrade"), but it happened
    without a word: nothing in the runtime told the owner which keys went, or
    what replaced them.

    The notice is ONE line per process per dropped set — the seam is on every
    settings read, so a per-call emission would be a log flood.
    """
    import logging

    from ouroboros import config as cfg

    cfg._RETIREMENT_NOTICE_SEEN.clear()
    document = {
        "OUROBOROS_REVIEW_MODELS": "a/one,b/two",
        "OUROBOROS_SCOPE_REVIEW_MODEL": "c/three",
        "OUROBOROS_ADVISORY_REVIEW_ROUTE": "native",
        "TOTAL_BUDGET": 10.0,
    }

    with caplog.at_level(logging.WARNING, logger="ouroboros.config"):
        loaded = cfg.normalize_settings_raw(document)
        repeat = cfg.normalize_settings_raw(dict(document))

    # Behavior is unchanged: the keys are still dropped, nothing else is.
    assert "OUROBOROS_REVIEW_MODELS" not in loaded
    assert "OUROBOROS_SCOPE_REVIEW_MODEL" not in loaded
    assert "OUROBOROS_ADVISORY_REVIEW_ROUTE" not in loaded
    assert loaded["TOTAL_BUDGET"] == 10.0
    assert repeat == loaded

    notices = [r.getMessage() for r in caplog.records if "retired" in r.getMessage()]
    assert len(notices) == 1, notices
    for key in document:
        if key != "TOTAL_BUDGET":
            assert key in notices[0], key
    assert "OUROBOROS_REVIEWER_SLOTS" in notices[0]
    assert "shipped" in notices[0].lower()


def test_the_retirement_notice_names_the_successor_the_table_states(
    isolated_settings, caplog,
):
    """The FIRST of the notice's two non-comma shapes. Any dropped set without
    comma-list keys used to get one fixed sentence: "no replacement setting:
    what they used to configure is fixed behavior in this release". That is
    false for a retired key this retirement table gives a successor — the flat
    wall-clock pair was superseded by the activity model, and the acceptance
    pass count is migrated into the shared review-cycle cap — so an owner told
    there is no replacement stops looking for the setting that took over.
    RETIRED_SETTING_SUCCESSORS is the decision table the notice reads, and a key
    absent from it gets the neutral clause instead (sibling test), never an
    invented successor.
    """
    import logging

    from ouroboros import config as cfg
    from ouroboros.settings_defaults import (
        RETIRED_SETTING_KEYS,
        RETIRED_SETTING_SUCCESSORS,
    )

    # The map classifies INSIDE the retirement tuple: a successor for a key that
    # is not retired would name a migration nothing performs.
    assert set(RETIRED_SETTING_SUCCESSORS) <= set(RETIRED_SETTING_KEYS)
    # And a MIGRATED key is not a reportable loss: the acceptance pass count is
    # consumed into the shared cap before the purge computes the dropped set, so
    # an entry for it would promise a notice line nothing can emit.
    assert "OUROBOROS_ACCEPTANCE_MAX_IMPROVEMENT_PASSES" not in RETIRED_SETTING_SUCCESSORS

    # The document is DERIVED from the table, not spelled out: the D04 grep gate
    # (tests/test_legacy_timeout_retirement.py) lets the retired wall-clock pair
    # appear only in the retirement SSOT and its own audits, and this pin is
    # about the CLASS "a retired key whose successor the table states", not about
    # one key's spelling.
    successor_bearing = sorted(RETIRED_SETTING_SUCCESSORS)
    assert successor_bearing, "the notice's successor shape needs a member"
    document = dict.fromkeys(successor_bearing, 900)
    document["OUROBOROS_ACCEPTANCE_MAX_IMPROVEMENT_PASSES"] = 3
    document["TOTAL_BUDGET"] = 10.0

    cfg._RETIREMENT_NOTICE_SEEN.clear()
    with caplog.at_level(logging.WARNING, logger="ouroboros.config"):
        loaded = cfg.normalize_settings_raw(document)

    for key in successor_bearing:
        assert key not in loaded, key
    assert loaded["TOTAL_BUDGET"] == 10.0
    # Migrated, hence unreported: cycles = passes + 1, and the notice is silent.
    assert loaded["OUROBOROS_REVIEW_MAX_CYCLES"] == "4"
    notices = [r.getMessage() for r in caplog.records if "retired" in r.getMessage()]
    assert len(notices) == 1, notices
    assert "OUROBOROS_ACCEPTANCE_MAX_IMPROVEMENT_PASSES" not in notices[0]
    for key in successor_bearing:
        for successor in RETIRED_SETTING_SUCCESSORS[key]:
            assert successor in notices[0], successor
    assert "fixed behavior" not in notices[0]


def test_the_retirement_notice_invents_no_successor_when_the_table_states_none(
    isolated_settings, caplog,
):
    """The SECOND shape, and the reason the two are separate clauses rather than
    one sentence about the whole dropped set. The observability retention knob
    really has no successor setting (manifests and blobs are preserved
    indefinitely by contract) and the plan-task swarm timeouts have none this
    table states per key, so the notice says they are gone and points at the
    release notes — without promising a replacement key, and without the older
    claim that their effect is now fixed behavior, which for the swarm timeouts
    was never established.
    """
    import logging

    from ouroboros import config as cfg
    from ouroboros.settings_defaults import RETIRED_SETTING_SUCCESSORS

    cfg._RETIREMENT_NOTICE_SEEN.clear()
    with caplog.at_level(logging.WARNING, logger="ouroboros.config"):
        cfg.normalize_settings_raw({
            "OUROBOROS_OBSERVABILITY_RETENTION_DAYS": 30,
            "OUROBOROS_PLAN_TASK_SWARM_TIMEOUT_SEC": 60,
        })

    notices = [r.getMessage() for r in caplog.records if "retired" in r.getMessage()]
    assert len(notices) == 1, notices
    assert "OUROBOROS_OBSERVABILITY_RETENTION_DAYS" in notices[0]
    assert "OUROBOROS_PLAN_TASK_SWARM_TIMEOUT_SEC" in notices[0]
    assert "removed, not honored" in notices[0]
    assert "release notes" in notices[0]
    assert "fixed behavior" not in notices[0]
    for successors in RETIRED_SETTING_SUCCESSORS.values():
        for successor in successors:
            assert successor not in notices[0], successor


def test_the_retirement_notice_stays_quiet_for_a_document_without_ghosts(
    isolated_settings, caplog,
):
    import logging

    from ouroboros import config as cfg

    cfg._RETIREMENT_NOTICE_SEEN.clear()
    with caplog.at_level(logging.WARNING, logger="ouroboros.config"):
        cfg.normalize_settings_raw({"TOTAL_BUDGET": 10.0})

    assert [r.getMessage() for r in caplog.records] == []
