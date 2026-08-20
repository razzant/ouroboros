"""Build-time and runtime parity tests for the frozen tool inventory."""

from __future__ import annotations

import importlib
import importlib.machinery
import json
import os
import pathlib
import subprocess
import sys

import pytest

from ouroboros.tool_module_inventory import (
    FROZEN_TOOL_MANIFEST_NAME,
    TOOL_PACKAGE,
    ToolModuleInventoryError,
    build_frozen_tool_manifest,
    discover_tool_module_inventory,
    load_frozen_tool_modules,
    parse_frozen_tool_manifest,
    render_frozen_tool_manifest,
    tool_modules_for_runtime,
    verify_frozen_tool_manifest,
)
from ouroboros.tools import registry_core
from ouroboros.tools.registry import ToolRegistry

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "ouroboros" / "tools"


def _write_module(root: pathlib.Path, name: str, source: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{name}.py").write_text(source, encoding="utf-8", newline="\n")


def _isolated_env(root: pathlib.Path) -> dict[str, str]:
    app_root = root / "app"
    repo_dir = root / "repo"
    data_dir = root / "data"
    repo_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "OUROBOROS_APP_ROOT": str(app_root),
            "OUROBOROS_REPO_DIR": str(repo_dir),
            "OUROBOROS_DATA_DIR": str(data_dir),
            "OUROBOROS_SETTINGS_PATH": str(data_dir / "settings.json"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": os.pathsep.join(part for part in (str(REPO_ROOT), env.get("PYTHONPATH", "")) if part),
        }
    )
    return env


def _registry_projection_process(
    root: pathlib.Path,
    *,
    manifest: pathlib.Path | None = None,
) -> dict:
    code = r"""
import json
import pathlib
import sys

mode, manifest, repo_dir, data_dir = sys.argv[1:]
if mode == "frozen":
    sys.frozen = True
from ouroboros.tools import registry as registry_module
from ouroboros.tools import registry_core
if mode == "frozen":
    registry_core._FROZEN_TOOL_MANIFEST_PATH = pathlib.Path(manifest)
registry = registry_module.ToolRegistry(pathlib.Path(repo_dir), pathlib.Path(data_dir))
projection = []
for name, entry in registry._base_catalog.entries.items():
    handler = entry.handler
    projection.append({
        "name": name,
        "origin": registry._base_catalog.origins[name],
        "schema": entry.schema,
        "handler_module": str(getattr(handler, "__module__", "")),
        "handler_qualname": str(
            getattr(handler, "__qualname__", "")
            or getattr(handler, "__name__", "")
            or type(handler).__qualname__
        ),
        "is_code_tool": entry.is_code_tool,
        "timeout_sec": entry.timeout_sec,
        "mutates_worktree": entry.mutates_worktree,
    })
print("REGISTRY_PROJECTION=" + json.dumps({
    "modules": list(registry_module.ToolRegistry._FROZEN_TOOL_MODULES),
    "projection": projection,
}, ensure_ascii=True, sort_keys=True))
"""
    repo_dir = root / "repo"
    data_dir = root / "data"
    mode = "frozen" if manifest is not None else "source"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            mode,
            str(manifest or ""),
            str(repo_dir),
            str(data_dir),
        ],
        cwd=REPO_ROOT,
        env=_isolated_env(root),
        check=True,
        capture_output=True,
        text=True,
    )
    prefix = "REGISTRY_PROJECTION="
    rows = [line for line in completed.stdout.splitlines() if line.startswith(prefix)]
    assert len(rows) == 1, completed.stdout
    return json.loads(rows[0][len(prefix) :])


def test_inventory_discovers_direct_owners_and_complete_package_closure(tmp_path):
    _write_module(
        tmp_path,
        "helper",
        "get_tools: object\nVALUES = [get_tools for get_tools in ()]\n",
    )
    _write_module(tmp_path, "alpha", "def get_tools():\n    return []\n")
    _write_module(
        tmp_path,
        "zeta",
        "def local():\n    def get_tools():\n        return []\n    return get_tools\n",
    )
    _write_module(tmp_path, "registry", "VALUE = 2\n")
    _write_module(tmp_path, "_private", "VALUE = 3\n")
    _write_module(tmp_path, "__init__", "VALUE = 4\n")

    inventory = discover_tool_module_inventory(tmp_path)

    assert inventory.package_modules == (
        f"{TOOL_PACKAGE}._private",
        f"{TOOL_PACKAGE}.alpha",
        f"{TOOL_PACKAGE}.helper",
        f"{TOOL_PACKAGE}.registry",
        f"{TOOL_PACKAGE}.zeta",
    )
    assert inventory.tool_modules == ("alpha",)


@pytest.mark.parametrize(
    "source, message",
    (
        ("async def get_tools():\n    return []\n", "async_function"),
        ("from elsewhere import get_tools\n", "import"),
        ("get_tools = lambda: []\n", "assignment"),
        ("if True:\n    def get_tools():\n        return []\n", "function"),
        (
            "def get_tools():\n    return []\ndef get_tools():\n    return []\n",
            "function, function",
        ),
        ("def get_tools():\n    return []\ndel get_tools\n", "function, deletion"),
        (
            "@staticmethod\ndef get_tools():\n    return []\n",
            "decorated_function",
        ),
    ),
)
def test_inventory_rejects_ambiguous_get_tools_bindings(tmp_path, source, message):
    _write_module(tmp_path, "ambiguous", source)

    with pytest.raises(ToolModuleInventoryError, match=message):
        discover_tool_module_inventory(tmp_path)


@pytest.mark.parametrize(
    "source, message",
    (
        ("from elsewhere import *\n", "wildcard import"),
        (
            "def __getattr__(name):\n    return object()\n",
            "module-level __getattr__",
        ),
        ("globals()['get_tools'] = lambda: []\n", "module-level globals"),
        ("exec('get_tools = lambda: []')\n", "module-level exec"),
        (
            "import sys\nsys.modules[__name__].get_tools = lambda: []\n",
            "attribute target 'get_tools'",
        ),
        (
            "import sys\nsys.modules[__name__].__dict__['get_tools'] = lambda: []\n",
            "subscript target 'get_tools'",
        ),
        ("__getattr__ = lambda name: None\n", "__getattr__ assignment"),
        (
            "import sys\nsetattr(sys.modules[__name__], '__getattr__', lambda name: None)\n",
            "setattr.*__getattr__",
        ),
    ),
)
def test_inventory_rejects_dynamic_module_surfaces(tmp_path, source, message):
    _write_module(tmp_path, "dynamic", source)

    with pytest.raises(ToolModuleInventoryError, match=message):
        discover_tool_module_inventory(tmp_path)


@pytest.mark.parametrize("with_init", (False, True))
def test_inventory_rejects_direct_subpackages(tmp_path, with_init):
    _write_module(tmp_path, "alpha", "def get_tools():\n    return []\n")
    package = tmp_path / "nested"
    package.mkdir()
    if with_init:
        (package / "__init__.py").write_text(
            "def get_tools():\n    return []\n",
            encoding="utf-8",
        )

    with pytest.raises(ToolModuleInventoryError, match="direct tool subpackages"):
        discover_tool_module_inventory(tmp_path)


def test_inventory_rejects_native_extension_modules(tmp_path):
    _write_module(tmp_path, "alpha", "def get_tools():\n    return []\n")
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    (tmp_path / f"native{suffix}").write_bytes(b"")

    with pytest.raises(ToolModuleInventoryError, match="non-source tool module"):
        discover_tool_module_inventory(tmp_path)


def test_source_scan_degrades_one_structurally_invalid_module(tmp_path, monkeypatch):
    _write_module(tmp_path, "alpha", "def get_tools():\n    return []\n")
    _write_module(tmp_path, "broken", "def invalid(:\n")
    monkeypatch.delattr(sys, "frozen", raising=False)

    modules, errors = tool_modules_for_runtime(tmp_path)

    assert modules == ("alpha",)
    assert len(errors) == 1
    assert "cannot parse tool module" in errors[0]


def test_source_registry_logs_inventory_error_and_loads_healthy_owner(
    tmp_path,
    monkeypatch,
    caplog,
):
    monkeypatch.setattr(
        registry_core,
        "tool_modules_for_runtime",
        lambda *_args: (("core",), ("broken helper",)),
    )

    registry = ToolRegistry(tmp_path, tmp_path)

    assert "read_file" in registry._base_catalog.entries
    assert "Failed to inspect tool module: broken helper" in caplog.text


def test_manifest_is_canonical_and_round_trips():
    raw = render_frozen_tool_manifest(("alpha", "zeta"))

    assert raw == (b'{"modules":["alpha","zeta"],"package":"ouroboros.tools","schema_version":1}\n')
    assert parse_frozen_tool_manifest(raw) == ("alpha", "zeta")


@pytest.mark.parametrize(
    "payload, message",
    (
        ({"modules": ["alpha"], "package": TOOL_PACKAGE}, "invalid schema"),
        (
            {"modules": ["alpha"], "package": "other", "schema_version": 1},
            "wrong package",
        ),
        (
            {"modules": ["alpha"], "package": TOOL_PACKAGE, "schema_version": True},
            "unsupported schema version",
        ),
        (
            {"modules": ["zeta", "alpha"], "package": TOOL_PACKAGE, "schema_version": 1},
            "lexically sorted",
        ),
        (
            {"modules": ["alpha", "alpha"], "package": TOOL_PACKAGE, "schema_version": 1},
            "duplicate/case-colliding",
        ),
    ),
)
def test_manifest_rejects_invalid_data(payload, message):
    raw = (json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n").encode("ascii")

    with pytest.raises(ToolModuleInventoryError, match=message):
        parse_frozen_tool_manifest(raw)


def test_manifest_rejects_noncanonical_json():
    raw = b'{"schema_version": 1, "package": "ouroboros.tools", "modules": ["alpha"]}\n'

    with pytest.raises(ToolModuleInventoryError, match="not canonical"):
        parse_frozen_tool_manifest(raw)


def test_build_materializes_exact_current_inventory(tmp_path):
    manifest = tmp_path / FROZEN_TOOL_MANIFEST_NAME

    inventory = build_frozen_tool_manifest(TOOLS_DIR, manifest)

    assert manifest.read_bytes() == render_frozen_tool_manifest(inventory.tool_modules)
    assert load_frozen_tool_modules(manifest) == inventory.tool_modules
    assert f"{TOOL_PACKAGE}.registry" in inventory.package_modules
    assert f"{TOOL_PACKAGE}.registry_core" in inventory.package_modules
    assert f"{TOOL_PACKAGE}.registry_guard_process" in inventory.package_modules
    assert f"{TOOL_PACKAGE}.tool_catalog" in inventory.package_modules
    assert f"{TOOL_PACKAGE}.tool_resolution" in inventory.package_modules
    assert f"{TOOL_PACKAGE}.tool_result" in inventory.package_modules
    assert f"{TOOL_PACKAGE}.extension_dispatch" in inventory.package_modules
    assert "tool_result" not in inventory.tool_modules
    assert "registry_guard_process" not in inventory.tool_modules
    assert "registry_core" not in inventory.tool_modules
    assert "tool_resolution" not in inventory.tool_modules


def test_missing_frozen_manifest_fails_closed(tmp_path):
    with pytest.raises(ToolModuleInventoryError, match="cannot read frozen tool manifest"):
        load_frozen_tool_modules(tmp_path / "missing.json")


@pytest.mark.parametrize("payload", (None, b"not-json\n"))
def test_fresh_frozen_registry_rejects_missing_or_invalid_manifest(tmp_path, payload):
    manifest = tmp_path / FROZEN_TOOL_MANIFEST_NAME
    if payload is not None:
        manifest.write_bytes(payload)
    code = r"""
import pathlib
import sys
sys.frozen = True
from ouroboros.tools import registry as registry_module
from ouroboros.tools import registry_core
registry_core._FROZEN_TOOL_MANIFEST_PATH = pathlib.Path(sys.argv[1])
registry_module.ToolRegistry(pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(manifest), str(tmp_path / "repo"), str(tmp_path / "data")],
        cwd=REPO_ROOT,
        env=_isolated_env(tmp_path / "process"),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "ToolModuleInventoryError" in completed.stderr


def test_source_and_fresh_frozen_registries_have_exact_ordered_catalog_parity(tmp_path):
    manifest = tmp_path / FROZEN_TOOL_MANIFEST_NAME
    inventory = build_frozen_tool_manifest(TOOLS_DIR, manifest)

    source = _registry_projection_process(tmp_path / "source")
    frozen = _registry_projection_process(tmp_path / "frozen", manifest=manifest)

    assert source == frozen
    assert source["modules"] == list(inventory.tool_modules)


def test_inventoried_owner_import_failure_degrades_only_that_owner(
    tmp_path,
    monkeypatch,
    caplog,
):
    real_import = importlib.import_module
    monkeypatch.setattr(
        registry_core,
        "tool_modules_for_runtime",
        lambda *_args: (("missing_owner", "core"), ()),
    )
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: (
            (_ for _ in ()).throw(ImportError("missing owner"))
            if name == "ouroboros.tools.missing_owner"
            else real_import(name)
        ),
    )

    registry = ToolRegistry(tmp_path, tmp_path)

    assert "read_file" in registry._base_catalog.entries
    assert "Failed to load tool module missing_owner" in caplog.text


def test_archive_verification_requires_complete_package_closure(tmp_path):
    manifest = tmp_path / FROZEN_TOOL_MANIFEST_NAME
    inventory = build_frozen_tool_manifest(TOOLS_DIR, manifest)
    archive = tmp_path / "archive.txt"
    archive.write_text(
        "Options in 'Ouroboros'\n" + "\n".join(inventory.package_modules) + "\n",
        encoding="utf-8",
    )

    assert verify_frozen_tool_manifest(TOOLS_DIR, manifest, archive) == inventory

    archive.write_text(
        "\n".join(inventory.package_modules[1:]) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ToolModuleInventoryError, match="archive is missing tool modules"):
        verify_frozen_tool_manifest(TOOLS_DIR, manifest, archive)


def test_pyinstaller_spec_derives_manifest_and_hiddenimports_before_analysis():
    source = (REPO_ROOT / "Ouroboros.spec").read_text(encoding="utf-8")
    build_pos = source.index("_tool_module_inventory = _build_frozen_tool_manifest(")
    analysis_pos = source.index("a = Analysis(")

    assert build_pos < analysis_pos
    assert '_extra_datas.append((str(_frozen_tool_manifest_path), "ouroboros"))' in source
    assert "_extra_hiddenimports.extend(_tool_module_inventory.package_modules)" in source
    assert '_pathlib.Path("build") / "generated" / _FROZEN_TOOL_MANIFEST_NAME' in source


def test_release_smokes_verify_manifest_and_pyz_closure_on_every_platform():
    source = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert source.count("ouroboros.tool_module_inventory verify-artifact") == 4
    for path in (
        "Contents/Resources/ouroboros/_frozen_tool_modules.v1.json",
        "Ouroboros/_internal/ouroboros/_frozen_tool_modules.v1.json",
        "usr/lib/ouroboros/_internal/ouroboros/_frozen_tool_modules.v1.json",
    ):
        assert path in source
    assert "Ouroboros.app/Contents/MacOS/$APP_EXECUTABLE" in source
    assert "Ouroboros/Ouroboros" in source
    assert "usr/lib/ouroboros/Ouroboros" in source
    assert r"Ouroboros\Ouroboros.exe" in source
