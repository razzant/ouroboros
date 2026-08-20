"""Who may author a mode decision: the shared writer prologue and the env floor.

Split verbatim out of ``tests/test_runtime_mode_elevation.py`` by theme. This module
owns the prologue every settings writer routes through, the rule that a generic POST
authors no mode decision while an owner endpoint authors its own key, and the
env-forwarded modes that survive startup but can never author a lowering.

Hermetic — no network, no supervisor boot. Uses temp dirs for ``DATA_DIR`` /
``SETTINGS_PATH`` overrides via monkeypatching ``ouroboros.config`` module-level
constants.
"""

from __future__ import annotations

import json

import pytest


from tests._runtime_mode_elevation_shared import (
    _seed_disk,
)
from tests._runtime_mode_elevation_shared import isolated_settings as _isolated_settings

# The fixture is requested by name as a test parameter, so it is re-bound through a
# module attribute: a direct import of a name that reappears as a parameter is an F811
# redefinition under the CI ruff gate.
isolated_settings = _isolated_settings


def _own_ratchet_env(monkeypatch) -> None:
    """Own EVERY env key the code under test reads or writes for the mode ratchets.

    Owning only the neighbouring key is how a leak from one test poisoned two others into
    looking like a production-guard bug; these tests start from a known-empty env instead.
    """
    from ouroboros import config as cfg

    for key in (
        "OUROBOROS_CONTEXT_MODE",
        "OUROBOROS_CONTEXT_MODE_AUTO_LOW",
        "OUROBOROS_SAFETY_MODE",
        "OUROBOROS_RUNTIME_MODE",
        cfg.BOOT_RUNTIME_MODE_ENV_KEY,
    ):
        monkeypatch.delenv(key, raising=False)


def test_every_settings_writer_routes_through_the_shared_prologue():
    """Tripwire for the shape that produced three review rounds in a row.

    Rounds three, four and five each fixed the disk-authored-key rule on ONE path while a sibling
    path kept bypassing it (sibling keys, then the projection, then the generic owner POST). The
    rule now lives in ``config.prepare_settings_for_persist``, and this test enumerates every
    function that writes the settings file so a NEW writer cannot quietly reintroduce the shape:
    it must either route through the prologue or be added here with a reason.
    """
    import ast
    import pathlib
    import re

    # (module, function) -> why it may write settings.json without the prologue
    exempt = {
        ("ouroboros/context_mode_compat.py", "normalize_and_persist_context_mode_compat"):
            "one-window startup migration: while the settings lock is held, atomically rewrites "
            "only the raw document's context compatibility pair. Routing through the prologue "
            "would merge defaults and turn unrelated absence into authorship.",
        ("ouroboros/tools/registry_guard_process.py", "_restore_owner_files"):
            "immune-system ROLLBACK: rewrites the exact bytes snapshotted before an agent shell "
            "command. It authors no value, and filtering a restore would corrupt it — an "
            "owner-authored default would be dropped instead of restored.",
        ("ouroboros/usage_legacy_import.py", "_legacy_snapshot"):
            "reads/hashes the settings file for the usage archive; its writes target the archive.",
        ("ouroboros/tools/core.py", "_data_write"):
            "names SETTINGS_PATH only to REFUSE agent writes to it.",
        ("ouroboros/colab_bootstrap.py", "write_colab_settings"):
            "generates a settings document for ANOTHER root (the Colab Drive data dir) from "
            "scratch. The prologue proves its ratchets against the value on THIS process's "
            "disk, so routing a foreign path through it would answer the wrong file.",
    }
    # Keys are POSIX-normalised: `str(WindowsPath(...))` is backslash-separated, so on Windows
    # every `exempt` lookup below would miss and every hardcoded assertion at the end would
    # fail — turning the tripwire into either a red matrix or, worse, a guard that flags the
    # exempted writers while silently vouching for nothing.
    writers = {}
    for path in sorted(pathlib.Path("ouroboros").rglob("*.py")) + [pathlib.Path("server.py")]:
        src = path.read_text(encoding="utf-8")
        if ("SETTINGS_PATH" not in src and "atomic_write_json" not in src
                and "settings.json" not in src):
            continue
        for node in ast.walk(ast.parse(src)):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            seg = ast.get_source_segment(src, node) or ""
            # A writer that takes its destination as a PARAMETER never names SETTINGS_PATH,
            # so the path-literal trigger alone left the packaged bootstrap saver invisible
            # to this tripwire for as long as it existed. Its name is the other honest
            # signal for "this function persists a settings document".
            settings_write = (
                ("SETTINGS_PATH" in seg or "settings" in node.name.lower())
                and re.search(r"\.write_text\(|atomic_write_json\(|json\.dump\(", seg)
            ) or "atomic_write_json(settings_path" in seg
            if settings_write:
                writers[(path.as_posix(), node.name)] = "prepare_settings_for_persist" in seg

    unrouted = {k for k, routed in writers.items() if not routed and k not in exempt}
    assert not unrouted, (
        f"these functions write the settings file without going through the single enforcement "
        f"point config.prepare_settings_for_persist: {sorted(unrouted)}. Route them through it "
        f"(naming any key they genuinely author in `authored_keys`), or add them to `exempt` with "
        f"a reason. Do not re-implement the silence/ratchet rule at the call site."
    )
    # The three real writers must still BE routed — deleting the call must fail this test.
    # The owner endpoints' write lives in the locked read-modify-write primitive that
    # `_owner_write_settings` is now one caller of.
    assert writers.get(("ouroboros/config.py", "save_settings")) is True
    assert writers.get(("ouroboros/gateway/owner_settings.py", "_owner_update_settings")) is True
    assert writers.get(("ouroboros/packaged_cli.py", "_save_settings")) is True


def test_generic_settings_post_does_not_author_a_mode_decision(isolated_settings, monkeypatch):
    """A POST about a model slot must not author a context mode (the round-five sibling path).

    ``api_settings_post`` builds its payload from ``_owner_read_settings_raw`` — SETTINGS_DEFAULTS
    merged over the file — and persists through ``_owner_write_settings``, which had no filter. On a
    disk-silent instance the unrelated save therefore wrote ``max`` and ended a forwarded ablation
    override, exactly the defect the write-path fix was supposed to remove one round earlier.
    """
    import os as _os

    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros import config as cfg
    from ouroboros.gateway import settings as settings_mod
    from ouroboros.gateway.settings import api_settings_post

    monkeypatch.setattr(_os, "environ", dict(_os.environ))
    _own_ratchet_env(monkeypatch)
    _os.environ["OUROBOROS_CONTEXT_MODE"] = "low"  # forwarded by the benchmark launcher
    _os.environ["OUROBOROS_SAFETY_MODE"] = "light"
    _seed_disk(isolated_settings, {"TOTAL_BUDGET": "10"})
    monkeypatch.setattr(settings_mod, "apply_runtime_provider_defaults", lambda s: (s, False, []))
    monkeypatch.setattr(settings_mod, "_start_supervisor_if_needed_for_request", lambda *_a, **_k: False)
    cfg.apply_settings_to_env(cfg.load_settings())

    app = Starlette(routes=[Route("/api/settings", endpoint=api_settings_post, methods=["POST"])])
    app.state.drive_root = isolated_settings.parent
    app.state.repo_dir = isolated_settings.parent
    resp = TestClient(app).post("/api/settings", json={"TOTAL_BUDGET": "25"})

    assert resp.status_code == 200, resp.text
    stored = json.loads(isolated_settings.read_text(encoding="utf-8"))
    assert float(stored["TOTAL_BUDGET"]) == 25.0, "the POST's actual subject must still be saved"
    assert "OUROBOROS_CONTEXT_MODE" not in stored, "a generic POST authored a mode decision"
    assert "OUROBOROS_SAFETY_MODE" not in stored
    assert cfg.get_context_mode() == "low", "the forwarded ablation mode ended on an unrelated save"
    assert cfg.get_safety_mode() == "light"


def test_owner_endpoint_authors_its_own_key_even_at_the_default(isolated_settings, monkeypatch):
    """The other half of the rule: a caller that NAMES a key authors it, default value or not.

    Silence is only preserved for keys nobody claimed. The dedicated owner endpoint passes
    ``authored_keys``, so an owner selecting the shipped default on a disk-silent instance persists
    it — and it then overrides a contradicting forwarded env value.
    """
    import os as _os

    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros import config as cfg
    from ouroboros.gateway.settings import api_owner_safety_mode

    monkeypatch.setattr(_os, "environ", dict(_os.environ))
    _own_ratchet_env(monkeypatch)
    _os.environ["OUROBOROS_SAFETY_MODE"] = "light"
    _seed_disk(isolated_settings, {"TOTAL_BUDGET": "10"})

    app = Starlette(routes=[Route("/api/owner/safety-mode", endpoint=api_owner_safety_mode, methods=["POST"])])
    app.state.drive_root = isolated_settings.parent
    resp = TestClient(app).post("/api/owner/safety-mode", json={"mode": "full"})  # "full" IS the default

    assert resp.status_code == 200, resp.text
    stored = json.loads(isolated_settings.read_text(encoding="utf-8"))
    assert stored["OUROBOROS_SAFETY_MODE"] == "full", "the owner's explicit choice was dropped as a gap-filler"
    cfg.apply_settings_to_env(cfg.load_settings())
    assert cfg.get_safety_mode() == "full", "the owner's stored choice must beat the forwarded env value"


def test_env_forwarded_modes_survive_the_documented_startup_path(isolated_settings, monkeypatch):
    """Env CONFIGURES an isolated server; env may not AUTHOR a persisted lowering. Two concerns.

    The startup path is ``apply_settings_to_env(load_settings())`` (server.py, agent.py). Because
    load_settings does not let env author these keys, the dict it returns carries a DEFAULT wherever
    settings.json is silent — and projecting that default overwrote (or popped) a value the launcher
    forwarded on purpose. ``devtools/benchmarks/terminal_bench/harbor_installed_agent.py`` runs the
    container with NO settings.json at all and forwards ``OUROBOROS_CONTEXT_MODE`` (plus the
    explicit false tombstone for owner-Low runs); ``server_runner._patch_settings_ports`` and
    ``run_gaia._resolve_provider_keys`` both document the same "settings.json over env" clobber; and
    ``run_clb`` forwards the context mode and ``OUROBOROS_SAFETY_MODE`` the same way. Projection must
    therefore say only what the FILE says, and stay silent where the file says nothing.
    """
    import os as _os

    from ouroboros import config as cfg

    # Own the WHOLE environment: apply_settings_to_env writes ~122 keys, so a real copy is the only
    # honest ownership boundary here (the same technique the auto-low regression test uses).
    monkeypatch.setattr(_os, "environ", dict(_os.environ))
    _own_ratchet_env(monkeypatch)
    _os.environ["OUROBOROS_CONTEXT_MODE"] = "low"
    _os.environ["OUROBOROS_SAFETY_MODE"] = "light"

    # 1. No settings.json at all — the harbor container shape.
    assert not isolated_settings.exists()
    cfg.apply_settings_to_env(cfg.load_settings())
    assert cfg.get_context_mode() == "low", "startup clobbered an env-forwarded context mode"
    assert cfg.get_safety_mode() == "light", "startup clobbered an env-forwarded safety mode"
    assert "OUROBOROS_CONTEXT_MODE_AUTO_LOW" not in _os.environ
    # A bare forwarded `low` is still NOT an owner-declared scope-review skip.
    assert cfg.get_owner_context_mode() == "max"

    # 2. A settings.json that simply does not carry these keys — the seeded-benchmark shape.
    _seed_disk(isolated_settings, {"TOTAL_BUDGET": "10"})
    cfg.apply_settings_to_env(cfg.load_settings())
    assert cfg.get_context_mode() == "low"
    assert cfg.get_safety_mode() == "light"

    # 3. Disk CONTRADICTS env -> the owner-authored file wins, in both directions.
    _seed_disk(isolated_settings, {"OUROBOROS_CONTEXT_MODE": "max", "OUROBOROS_SAFETY_MODE": "full"})
    cfg.apply_settings_to_env(cfg.load_settings())
    assert cfg.get_context_mode() == "max"
    assert cfg.get_safety_mode() == "full"

    # 4. An explicit forwarded false accompanies a benchmark/operator Low declaration.
    _os.environ["OUROBOROS_CONTEXT_MODE"] = "low"
    _os.environ["OUROBOROS_CONTEXT_MODE_AUTO_LOW"] = "false"
    _seed_disk(isolated_settings, {"TOTAL_BUDGET": "10"})
    cfg.apply_settings_to_env(cfg.load_settings())
    assert cfg.get_context_mode() == "low"
    assert cfg.get_owner_context_mode() == "low"


def test_agent_save_cannot_end_a_forwarded_mode_mid_run(isolated_settings, monkeypatch):
    """An AGENT-reachable save must not quietly re-label a running context ablation.

    ``set_tool_timeout`` (``tools/control.py``) is a tool the agent can call itself, and it does
    ``load_settings -> save_settings -> apply_settings_to_env``. With the mode forwarded by env and
    absent from settings.json, the loaded dict carried the DEFAULT, the save authored that default
    onto disk, and from then on disk spoke: the ablation run continued under ``max`` while its
    artifact still claimed ``low``. Nothing fails, the numbers look fine, and the label is wrong —
    which is exactly the defect class this release exists to remove. Silence on disk therefore stays
    silence on the write path too, symmetrically with the projection rule.
    """
    import os as _os

    from ouroboros import config as cfg
    from ouroboros.tools.control import _set_tool_timeout

    monkeypatch.setattr(_os, "environ", dict(_os.environ))  # the tool writes os.environ via apply
    _own_ratchet_env(monkeypatch)
    _os.environ["OUROBOROS_CONTEXT_MODE"] = "low"
    _os.environ["OUROBOROS_SAFETY_MODE"] = "light"
    _seed_disk(isolated_settings, {"TOTAL_BUDGET": "10"})  # seeded run, no mode keys stored
    cfg.apply_settings_to_env(cfg.load_settings())
    assert cfg.get_context_mode() == "low"

    result = _set_tool_timeout(None, 600)  # the agent's own mid-run save

    assert result.startswith("OK")
    stored = json.loads(isolated_settings.read_text(encoding="utf-8"))
    assert stored["OUROBOROS_TOOL_TIMEOUT_SEC"] == 600, "the tool's actual job must still happen"
    assert "OUROBOROS_CONTEXT_MODE" not in stored, "an agent save must not author a mode decision"
    assert "OUROBOROS_SAFETY_MODE" not in stored
    assert cfg.get_context_mode() == "low", "the run's forwarded context mode ended mid-run"
    assert cfg.get_safety_mode() == "light", "the run's forwarded safety mode ended mid-run"
    # The owner path is still the author — but only for keys it NAMES, which is what the dedicated
    # endpoint passes (end-to-end coverage: test_owner_endpoint_authors_its_own_key_even_at_the_default).
    from ouroboros.gateway.settings import _CONTEXT_MODE_KEYS, _owner_write_settings

    _owner_write_settings({**stored, "OUROBOROS_CONTEXT_MODE": "max"})
    assert "OUROBOROS_CONTEXT_MODE" not in json.loads(isolated_settings.read_text(encoding="utf-8")), (
        "an owner write that does not claim the key must not author it either"
    )
    _owner_write_settings({**stored, "OUROBOROS_CONTEXT_MODE": "max"}, authored_keys=_CONTEXT_MODE_KEYS)
    cfg.apply_settings_to_env(cfg.load_settings())
    assert cfg.get_context_mode() == "max"


def test_env_declared_context_mode_cannot_author_a_lowering(isolated_settings, monkeypatch):
    """A ratchet reads its PREVIOUS value off DISK for every key it guards — env never.

    Round-3 regression: the bypass closed for the context provenance tombstone was still open one key
    over. With no ``OUROBOROS_CONTEXT_MODE`` stored, an inherited/forwarded env ``low`` made the
    guard compare ``low -> low`` instead of ``max -> low``, so any caller could persist the
    lowered cognitive horizon without ``allow_context_lowering``. Absent from disk resolves
    FAIL-CLOSED to ``max``: the gate stays on and lowering needs the owner path, never the reverse.
    """
    from ouroboros import config as cfg

    _own_ratchet_env(monkeypatch)
    _seed_disk(isolated_settings, {"TOTAL_BUDGET": "10"})  # no mode key stored at all
    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", "low")

    with pytest.raises(PermissionError, match="OUROBOROS_CONTEXT_MODE lowering refused"):
        cfg.save_settings({"OUROBOROS_CONTEXT_MODE": "low"})
    assert "OUROBOROS_CONTEXT_MODE" not in json.loads(isolated_settings.read_text(encoding="utf-8"))

    # And env cannot author the NEXT value either: an ordinary load->save round-trip in the same
    # process must not launder env's `low` onto disk — nor raise a PermissionError nobody authored.
    loaded = cfg.load_settings()
    assert loaded["OUROBOROS_CONTEXT_MODE"] == "max"
    cfg.save_settings(loaded)
    stored = json.loads(isolated_settings.read_text(encoding="utf-8"))
    assert "OUROBOROS_CONTEXT_MODE" not in stored, (
        "env's `low` must not be laundered onto disk — and the default filling the gap is not "
        "authorship either, so a silent file stays silent (see test_agent_save_cannot_end_a_"
        "forwarded_mode_mid_run)"
    )

    # The owner authorisation is untouched.
    from ouroboros.gateway.settings import _owner_write_settings

    _owner_write_settings({"OUROBOROS_CONTEXT_MODE": "low"}, allow_context_lowering=True)
    assert json.loads(isolated_settings.read_text(encoding="utf-8"))["OUROBOROS_CONTEXT_MODE"] == "low"


def test_env_declared_safety_mode_cannot_author_a_lowering(isolated_settings, monkeypatch):
    """Same shape, third key: the safety-coverage ratchet also read its previous value via env."""
    from ouroboros import config as cfg

    _own_ratchet_env(monkeypatch)
    _seed_disk(isolated_settings, {"TOTAL_BUDGET": "10"})
    monkeypatch.setenv("OUROBOROS_SAFETY_MODE", "off")

    with pytest.raises(PermissionError, match="OUROBOROS_SAFETY_MODE lowering refused"):
        cfg.save_settings({"OUROBOROS_SAFETY_MODE": "off"})
    assert "OUROBOROS_SAFETY_MODE" not in json.loads(isolated_settings.read_text(encoding="utf-8"))

    loaded = cfg.load_settings()
    assert loaded["OUROBOROS_SAFETY_MODE"] == "full"  # absent -> fail-closed to FULL coverage
    cfg.save_settings(loaded)

    from ouroboros.gateway.settings import _owner_write_settings

    _owner_write_settings({"OUROBOROS_SAFETY_MODE": "off"}, allow_safety_lowering=True)
    assert json.loads(isolated_settings.read_text(encoding="utf-8"))["OUROBOROS_SAFETY_MODE"] == "off"


def test_env_boot_baseline_may_tighten_the_elevation_floor_but_never_raise_it(
    isolated_settings, monkeypatch
):
    """The runtime-mode baseline is the same shape on a fourth key.

    ``OUROBOROS_BOOT_RUNTIME_MODE`` exists so a fresh subprocess inherits the parent's ratchet —
    it keeps an out-of-process settings edit from BECOMING the baseline. A subprocess exporting it
    upward must not be able to raise its own floor and persist the elevation, so the baseline is
    the STRICTEST of the inherited pin and disk.
    """
    from ouroboros import config as cfg

    _own_ratchet_env(monkeypatch)
    _seed_disk(isolated_settings, {"OUROBOROS_RUNTIME_MODE": "light"})
    monkeypatch.setenv(cfg.BOOT_RUNTIME_MODE_ENV_KEY, "pro")

    with pytest.raises(PermissionError, match="elevation refused"):
        cfg.save_settings({"OUROBOROS_RUNTIME_MODE": "pro"})
    assert json.loads(isolated_settings.read_text(encoding="utf-8"))["OUROBOROS_RUNTIME_MODE"] == "light"

    # The pin's real job is preserved: it still TIGHTENS a higher on-disk mode.
    _seed_disk(isolated_settings, {"OUROBOROS_RUNTIME_MODE": "pro"})
    monkeypatch.setenv(cfg.BOOT_RUNTIME_MODE_ENV_KEY, "light")
    with pytest.raises(PermissionError, match="elevation refused"):
        cfg.save_settings({"OUROBOROS_RUNTIME_MODE": "advanced"})

    # And an honest same-mode save still works under an inherited pin.
    _seed_disk(isolated_settings, {"OUROBOROS_RUNTIME_MODE": "advanced"})
    monkeypatch.setenv(cfg.BOOT_RUNTIME_MODE_ENV_KEY, "advanced")
    cfg.save_settings({"OUROBOROS_RUNTIME_MODE": "advanced", "TOTAL_BUDGET": "7"})
    assert json.loads(isolated_settings.read_text(encoding="utf-8"))["TOTAL_BUDGET"] == "7"


def test_private_owner_write_settings_keeps_context_lowering_guard(isolated_settings, monkeypatch):
    from ouroboros.gateway import settings as settings_mod

    _seed_disk(isolated_settings, {"OUROBOROS_CONTEXT_MODE": "max"})
    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", "max")

    with pytest.raises(PermissionError):
        settings_mod._owner_write_settings({"OUROBOROS_CONTEXT_MODE": "low"})


def test_merge_settings_payload_preserves_other_keys():
    """Sanity: dropping runtime_mode didn't accidentally drop everything else."""
    from ouroboros.gateway import settings as server_mod

    old = {"OUROBOROS_RUNTIME_MODE": "advanced", "TOTAL_BUDGET": "10.0"}
    body = {"TOTAL_BUDGET": "20.0", "OUROBOROS_REVIEW_ENFORCEMENT": "blocking"}
    merged = server_mod._merge_settings_payload(old, body)
    assert merged["TOTAL_BUDGET"] == "20.0"
    assert merged["OUROBOROS_REVIEW_ENFORCEMENT"] == "blocking"
    assert merged["OUROBOROS_RUNTIME_MODE"] == "advanced"
