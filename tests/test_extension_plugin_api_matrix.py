"""ABI-1 PluginAPI 2.0 negotiation matrix (v7next Ф3.1-B).

Pins the full manifest-negotiation contract: absent field ≡ the LEGACY
generation by construction ("1.3", deliberately NOT 1.4); declared versions
are held to major-strict / minor-as-minimum; capabilities form a closed set
validated against the actual execution mode; every refusal is typed and
educational; and the per-version surface fingerprint fails closed in BOTH
directions (unknown version, and live-surface drift without a version bump).
"""

from __future__ import annotations

import pytest

from ouroboros.contracts import plugin_api as papi
from ouroboros.contracts.plugin_api import (
    ExecutionMode,
    LEGACY_PLUGIN_API_GENERATION,
    PLUGIN_API_SURFACE_FINGERPRINTS,
    PLUGIN_API_VERSION,
    api_generation,
    extension_new_pass_admission_error,
    negotiate_plugin_api,
    plugin_api_surface_fingerprint,
)
from ouroboros.contracts.skill_manifest import (
    SkillManifestError,
    parse_skill_manifest_text,
)

from tests._extension_loader_shared import _prepare_extension
from tests._extension_loader_shared import (  # noqa: F401  (autouse fixture applies on import)
    _clear_loader_state,
)


def _manifest(plugin_api_line: str = "") -> object:
    return parse_skill_manifest_text(
        "---\n"
        "name: matrix_ext\n"
        "description: d\n"
        "version: 1.0.0\n"
        "type: extension\n"
        "entry: plugin.py\n"
        "permissions: [tool]\n"
        f"{plugin_api_line}"
        "---\n"
        "body\n"
    )


# --- parse shape (fail-closed structure, tolerant absence) -------------------


def test_plugin_api_field_is_optional_and_absent_means_legacy_generation():
    manifest = _manifest()
    assert manifest.plugin_api is None
    assert api_generation(manifest) == LEGACY_PLUGIN_API_GENERATION == "1.3"
    result = negotiate_plugin_api(manifest)
    assert result.ok and not result.declared
    assert result.generation == "1.3"


def test_plugin_api_field_parses_string_and_mapping_forms():
    as_string = _manifest('plugin_api: "2.0"\n')
    assert as_string.plugin_api == {"version": "2.0", "capabilities": []}
    as_mapping = _manifest(
        "plugin_api:\n  version: \"2.0\"\n  capabilities: [register_tool, log]\n"
    )
    assert as_mapping.plugin_api == {
        "version": "2.0", "capabilities": ["register_tool", "log"],
    }


@pytest.mark.parametrize("bad_line", [
    "plugin_api: [2, 0]\n",
    "plugin_api:\n  version: \"2.0\"\n  extra_key: true\n",
    "plugin_api:\n  capabilities: [register_tool]\n",
])
def test_plugin_api_structural_damage_fails_manifest_parse(bad_line):
    with pytest.raises(SkillManifestError):
        _manifest(bad_line)


def test_invalid_declaration_surfaces_as_manifest_validate_warning():
    manifest = _manifest('plugin_api: "1.0"\n')
    warnings = manifest.validate()
    assert any("plugin_api" in w for w in warnings)


# --- negotiation: major strict / minor minimum / typed education -------------


def test_exact_host_version_negotiates_to_the_host_generation():
    result = negotiate_plugin_api(_manifest('plugin_api: "2.0"\n'))
    assert result.ok and result.declared
    assert result.generation == PLUGIN_API_VERSION


def test_lower_major_is_refused_with_grandfather_education():
    result = negotiate_plugin_api(_manifest('plugin_api: "1.3"\n'))
    assert not result.ok
    assert "major is strict" in result.error
    assert "OMIT the field" in result.error


def test_higher_major_is_refused():
    result = negotiate_plugin_api(_manifest('plugin_api: "3.0"\n'))
    assert not result.ok and "major" in result.error


def test_higher_minor_minimum_is_refused_with_upgrade_education():
    result = negotiate_plugin_api(_manifest('plugin_api: "2.7"\n'))
    assert not result.ok
    assert "minimum minor" in result.error and "upgrade" in result.error


def test_malformed_version_string_is_refused_typed():
    result = negotiate_plugin_api(_manifest('plugin_api: "two.zero"\n'))
    assert not result.ok and "major.minor" in result.error


# --- capabilities: closed set + execution-mode availability ------------------


def test_unknown_capability_is_refused_and_names_the_closed_set():
    result = negotiate_plugin_api(_manifest(
        "plugin_api:\n  version: \"2.0\"\n  capabilities: [register_rootkit]\n"
    ))
    assert not result.ok
    assert "register_rootkit" in result.error and "closed" in result.error


def test_mode_unavailable_capability_is_refused_with_companion_education():
    manifest = _manifest(
        "plugin_api:\n  version: \"2.0\"\n  capabilities: [subscribe_event]\n"
    )
    in_proc = negotiate_plugin_api(manifest, mode=ExecutionMode.IN_PROCESS)
    assert in_proc.ok and in_proc.capabilities == ("subscribe_event",)
    out_proc = negotiate_plugin_api(manifest, mode=ExecutionMode.OUT_OF_PROCESS)
    assert not out_proc.ok and "companion_process" in out_proc.error


# --- surface fingerprint: fail-closed in both directions ---------------------


def test_recorded_fingerprint_matches_the_live_surface():
    assert PLUGIN_API_SURFACE_FINGERPRINTS[PLUGIN_API_VERSION] == plugin_api_surface_fingerprint()


def test_unknown_version_key_fails_closed(monkeypatch):
    monkeypatch.setattr(
        papi, "PLUGIN_API_SURFACE_FINGERPRINTS",
        {PLUGIN_API_VERSION: plugin_api_surface_fingerprint(), "2.9": "unused"},
    )
    monkeypatch.setattr(papi, "PLUGIN_API_VERSION", "2.9")
    # Declared "2.0" is inside major/minor range for host 2.9 but has no
    # recorded fingerprint entry in this simulated table -> refused.
    monkeypatch.setitem(papi.PLUGIN_API_SURFACE_FINGERPRINTS, "2.9", papi.plugin_api_surface_fingerprint())
    del papi.PLUGIN_API_SURFACE_FINGERPRINTS["2.0"]
    result = papi.negotiate_plugin_api(_manifest('plugin_api: "2.0"\n'))
    assert not result.ok and "no recorded surface fingerprint" in result.error


def test_live_surface_drift_without_version_bump_fails_closed(monkeypatch):
    monkeypatch.setitem(
        papi.PLUGIN_API_SURFACE_FINGERPRINTS, PLUGIN_API_VERSION, "0" * 64,
    )
    drifted = papi.negotiate_plugin_api(_manifest('plugin_api: "2.0"\n'))
    assert not drifted.ok and "drifted" in drifted.error
    # The drift refusal is host-integrity: even the grandfather path refuses.
    legacy = papi.negotiate_plugin_api(_manifest())
    assert not legacy.ok and "drifted" in legacy.error


# --- admission predicate shape (full issuance-path coverage lives in
# tests/test_plugin_api_admission.py) ----------------------------------------


def test_admission_error_only_for_fieldless_or_invalid_extensions():
    assert extension_new_pass_admission_error(None) == ""
    fieldless = _manifest()
    assert "plugin_api" in extension_new_pass_admission_error(fieldless)
    declared = _manifest('plugin_api: "2.0"\n')
    assert extension_new_pass_admission_error(declared) == ""
    invalid = _manifest('plugin_api: "1.0"\n')
    assert "major is strict" in extension_new_pass_admission_error(invalid)
    instruction = parse_skill_manifest_text("# just_instructions\nbody\n")
    assert extension_new_pass_admission_error(instruction) == ""


# --- loader integration: negotiation gates the load, generation rides
# the published surfaces ------------------------------------------------------


def test_declared_2_0_extension_loads_and_carries_the_generation(tmp_path):
    from ouroboros import extension_loader
    from ouroboros.extension_surface_names import extension_surface_name

    loaded, _repo, drive_root = _prepare_extension(
        tmp_path,
        "gen2ext",
        plugin_body=(
            "def register(api):\n"
            "    api.register_tool('t1', lambda **kw: 'ok', description='d', schema={})\n"
        ),
        permissions=["tool"],
        extra_frontmatter='plugin_api: "2.0"\n',
    )
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root, _force_in_process=True)
    assert err is None, err
    entry = extension_loader.get_tool(extension_surface_name("gen2ext", "t1"))
    assert entry is not None and entry.get("plugin_api_generation") == "2.0"
    with extension_loader._lock:
        bundle = extension_loader._extensions["gen2ext"]
        assert bundle.plugin_api_generation == "2.0"


def test_grandfathered_fieldless_extension_still_loads_as_legacy(tmp_path):
    from ouroboros import extension_loader
    from ouroboros.extension_surface_names import extension_surface_name

    loaded, _repo, drive_root = _prepare_extension(
        tmp_path,
        "legacyext",
        plugin_body=(
            "def register(api):\n"
            "    api.register_tool('t1', lambda **kw: 'ok', description='d', schema={})\n"
        ),
        permissions=["tool"],
    )
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root, _force_in_process=True)
    assert err is None, err
    entry = extension_loader.get_tool(extension_surface_name("legacyext", "t1"))
    assert entry is not None
    assert entry.get("plugin_api_generation") == LEGACY_PLUGIN_API_GENERATION


def test_refused_negotiation_blocks_the_load_before_import(tmp_path):
    from ouroboros import extension_loader

    loaded, _repo, drive_root = _prepare_extension(
        tmp_path,
        "badgen",
        plugin_body="RAN = True\ndef register(api):\n    pass\n",
        permissions=["tool"],
        extra_frontmatter='plugin_api: "9.9"\n',
    )
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root, _force_in_process=True)
    assert err is not None and "negotiation refused" in err
    import sys

    from ouroboros.extension_import_staging import _module_key

    assert _module_key("badgen") not in sys.modules, (
        "plugin.py was imported although negotiation refused the load"
    )


# --- surface fingerprint: SHAPE, not just names ------------------------------


def _mutant_module(tmp_path, old: str, new: str):
    """Load a copy of the contract with one surface edit applied.

    The copy keeps the RECORDED fingerprint table verbatim — that is the whole
    point: a shape change without a version bump must make the live surface
    disagree with what the release recorded, so negotiation refuses.
    """
    import importlib.util
    import pathlib
    import sys

    source = pathlib.Path(papi.__file__).read_text(encoding="utf-8")
    assert source.count(old) == 1, old
    target = tmp_path / "plugin_api_mutant.py"
    target.write_text(source.replace(old, new), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("plugin_api_mutant", target)
    module = importlib.util.module_from_spec(spec)
    # @dataclass resolves its own module through sys.modules while the class
    # body executes, so the entry must exist before exec_module.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


@pytest.mark.parametrize(("old", "new", "what"), (
    ("def get_settings(self, keys: Sequence[str]) -> Dict[str, Any]:",
     "def get_settings(self, keys: List[str]) -> Dict[str, Any]:",
     "a method's parameter annotation"),
    ("    server_port: int\n", "    server_port: str\n",
     "a RuntimeInfo value type"),
), ids=("method-parameter-annotation", "runtime_info-value-type"))
def test_a_shape_change_without_a_version_bump_fails_closed(tmp_path, old, new, what):
    """The digest hashed method NAMES and RuntimeInfo KEYS only, so every
    signature-level break — the exact class this module's header says requires
    a version bump — left the recorded digest byte-identical and negotiated
    happily. Both directions of the fingerprint contract were therefore blind
    to the changes an extension author actually collides with."""
    mutant = _mutant_module(tmp_path, old, new)

    assert mutant.plugin_api_surface_fingerprint() != plugin_api_surface_fingerprint(), what
    refused = mutant.negotiate_plugin_api(_manifest('plugin_api: "2.0"\n'))
    assert not refused.ok and "drifted" in refused.error, what


def test_the_fingerprint_covers_every_public_method_signature_and_runtime_key():
    """Naming the payload's two halves so a future narrowing is visible: the
    digest must move for ANY public method's signature text and for ANY
    RuntimeInfo annotation, not merely for an added or removed name."""
    import inspect

    methods = sorted(
        m for m in dir(papi.PluginAPI)
        if not m.startswith("_") and callable(getattr(papi.PluginAPI, m, None))
    )
    assert methods, "the surface cannot be empty"
    for name in methods:
        assert str(inspect.signature(getattr(papi.PluginAPI, name)))
    assert set(papi.RuntimeInfo.__annotations__) >= {"server_port", "capabilities"}
