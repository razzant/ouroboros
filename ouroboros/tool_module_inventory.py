"""Build-time inventory for first-party tool modules.

The source ``get_tools()`` definitions are authoritative. PyInstaller uses this
module to derive both its import closure and the transient manifest consumed by
the frozen registry; no checked-in module list mirrors that source surface.
"""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import keyword
import pathlib
import sys
from dataclasses import dataclass
from typing import Iterable, Sequence

from ouroboros.utils import write_text_atomic

TOOL_PACKAGE = "ouroboros.tools"
FROZEN_TOOL_MANIFEST_NAME = "_frozen_tool_modules.v1.json"
_MANIFEST_SCHEMA_VERSION = 1


class ToolModuleInventoryError(ValueError):
    """The tool package or its frozen manifest is not structurally valid."""


@dataclass(frozen=True)
class ToolModuleInventory:
    """Deterministic package and direct ``get_tools`` owner projections."""

    package_modules: tuple[str, ...]
    tool_modules: tuple[str, ...]


class _GetToolsBindingVisitor(ast.NodeVisitor):
    """Find module-scope bindings without descending into local scopes."""

    def __init__(self) -> None:
        self.bindings: list[tuple[str, ast.AST]] = []
        self.has_wildcard_import = False
        self.dynamic_authoring: list[str] = []

    def _record_binding(self, name: str | None, kind: str, node: ast.AST) -> None:
        if name == "get_tools":
            self.bindings.append((kind, node))
        elif name == "__getattr__":
            self.dynamic_authoring.append(f"module-level __getattr__ {kind}")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        kind = "decorated_function" if node.decorator_list else "function"
        self._record_binding(node.name, kind, node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._record_binding(node.name, "async_function", node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._record_binding(node.name, "class", node)

    def visit_Lambda(self, _node: ast.Lambda) -> None:
        return

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store):
            self._record_binding(node.id, "assignment", node)
        elif isinstance(node.ctx, ast.Del):
            self._record_binding(node.id, "deletion", node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)) and node.attr in {
            "get_tools",
            "__getattr__",
        }:
            self.dynamic_authoring.append(f"module-scope attribute target {node.attr!r}")
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        key = node.slice.value if isinstance(node.slice, ast.Constant) else None
        if isinstance(node.ctx, (ast.Store, ast.Del)) and key in {
            "get_tools",
            "__getattr__",
        }:
            self.dynamic_authoring.append(f"module-scope subscript target {key!r}")
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self.visit(node.target)
            self.visit(node.value)

    def _visit_comprehension(self, values: Iterable[ast.AST], generators) -> None:
        for value in values:
            self.visit(value)
        for generator in generators:
            self.visit(generator.iter)
            for condition in generator.ifs:
                self.visit(condition)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension((node.elt,), node.generators)

    visit_SetComp = visit_ListComp
    visit_GeneratorExp = visit_ListComp

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension((node.key, node.value), node.generators)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in {
            "exec",
            "globals",
            "locals",
            "vars",
        }:
            self.dynamic_authoring.append(f"module-level {node.func.id}()")
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "setattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value in {"get_tools", "__getattr__"}
        ):
            self.dynamic_authoring.append(
                f"module-level setattr(..., {node.args[1].value!r}, ...)"
            )
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            bound = alias.asname or alias.name.split(".", 1)[0]
            self._record_binding(bound, "import", node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name == "*":
                self.has_wildcard_import = True
                continue
            self._record_binding(alias.asname or alias.name, "import", node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        self._record_binding(node.name, "assignment", node)
        for statement in node.body:
            self.visit(statement)

    def visit_MatchAs(self, node: ast.MatchAs) -> None:
        self._record_binding(node.name, "assignment", node)
        if node.pattern is not None:
            self.visit(node.pattern)

    def visit_MatchStar(self, node: ast.MatchStar) -> None:
        self._record_binding(node.name, "assignment", node)

    def visit_MatchMapping(self, node: ast.MatchMapping) -> None:
        self._record_binding(node.rest, "assignment", node)
        for pattern in node.patterns:
            self.visit(pattern)


def _module_tree(path: pathlib.Path) -> ast.Module:
    try:
        source = path.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ToolModuleInventoryError(f"cannot read tool module {path}: {exc}") from exc
    try:
        return ast.parse(source, filename=path.as_posix())
    except SyntaxError as exc:
        raise ToolModuleInventoryError(f"cannot parse tool module {path}: {exc}") from exc


def _owns_get_tools(path: pathlib.Path, tree: ast.Module) -> bool:
    visitor = _GetToolsBindingVisitor()
    for statement in tree.body:
        visitor.visit(statement)

    if visitor.has_wildcard_import:
        raise ToolModuleInventoryError(
            f"tool module {path} has a module-scope wildcard import; get_tools ownership is not statically provable"
        )
    if visitor.dynamic_authoring:
        raise ToolModuleInventoryError(
            f"tool module {path} uses dynamic module authoring "
            f"({', '.join(visitor.dynamic_authoring)}); get_tools ownership is not statically provable"
        )

    if not visitor.bindings:
        return False
    direct = [node for kind, node in visitor.bindings if kind == "function" and node in tree.body]
    if len(visitor.bindings) == 1 and len(direct) == 1:
        return True
    kinds = ", ".join(kind for kind, _node in visitor.bindings)
    raise ToolModuleInventoryError(
        f"tool module {path} must own exactly one top-level synchronous get_tools function; found {kinds}"
    )


def _validate_module_names(names: Iterable[str]) -> tuple[str, ...]:
    modules = tuple(str(name) for name in names)
    if not modules:
        raise ToolModuleInventoryError("frozen tool manifest must contain at least one module")
    if modules != tuple(sorted(modules)):
        raise ToolModuleInventoryError("frozen tool modules must be lexically sorted")
    folded: set[str] = set()
    for name in modules:
        if not name or name.startswith("_") or name == "registry" or not name.isidentifier() or keyword.iskeyword(name):
            raise ToolModuleInventoryError(f"invalid frozen tool module name: {name!r}")
        key = name.casefold()
        if key in folded:
            raise ToolModuleInventoryError(f"duplicate/case-colliding frozen tool module name: {name!r}")
        folded.add(key)
    return modules


def _scan_tool_module_inventory(
    tools_dir: pathlib.Path,
) -> tuple[ToolModuleInventory, tuple[str, ...]]:
    root = pathlib.Path(tools_dir)
    try:
        entries = sorted(root.iterdir(), key=lambda item: item.name)
    except OSError as exc:
        raise ToolModuleInventoryError(f"cannot enumerate tool package {root}: {exc}") from exc
    if not entries:
        raise ToolModuleInventoryError(f"tool package contains no Python modules: {root}")

    errors: list[str] = []
    invalid_names: set[str] = set()
    paths: list[pathlib.Path] = []
    for entry in entries:
        if entry.name == "__pycache__":
            continue
        if entry.is_dir():
            if entry.name.isidentifier() and not keyword.iskeyword(entry.name):
                invalid_names.add(entry.name)
                errors.append(f"direct tool subpackages are unsupported: {entry}")
            continue
        module_name = inspect.getmodulename(entry.name)
        if entry.suffix != ".py" and module_name:
            invalid_names.add(module_name)
            errors.append(f"importable non-source tool module is unsupported: {entry}")
            continue
        if entry.suffix == ".py":
            paths.append(entry)

    package_names: list[str] = []
    tool_names: list[str] = []
    seen_casefold: set[str] = set()
    for path in paths:
        if path.name == "__init__.py":
            continue
        if path.is_symlink() or not path.is_file():
            errors.append(f"tool package entry must be a regular file: {path}")
            continue
        name = path.stem
        if name in invalid_names:
            continue
        if not name.isidentifier() or keyword.iskeyword(name):
            errors.append(f"invalid tool package module name: {name!r}")
            continue
        folded = name.casefold()
        if folded in seen_casefold:
            errors.append(f"case-colliding tool package module: {name!r}")
            continue
        seen_casefold.add(folded)

        package_names.append(f"{TOOL_PACKAGE}.{name}")
        try:
            owns_tools = _owns_get_tools(path, _module_tree(path))
        except ToolModuleInventoryError as exc:
            errors.append(str(exc))
            continue
        if owns_tools:
            if name.startswith("_") or name == "registry":
                errors.append(f"reserved tool package module cannot export get_tools: {path}")
                continue
            tool_names.append(name)

    if not tool_names:
        errors.append(f"tool package contains no valid direct get_tools owners: {root}")
    inventory = ToolModuleInventory(tuple(package_names), tuple(sorted(tool_names)))
    return inventory, tuple(errors)


def discover_tool_module_inventory(tools_dir: pathlib.Path) -> ToolModuleInventory:
    """Strictly scan direct package modules without importing tool code."""

    inventory, errors = _scan_tool_module_inventory(tools_dir)
    if errors:
        raise ToolModuleInventoryError("; ".join(errors))
    _validate_module_names(inventory.tool_modules)
    return inventory


def render_frozen_tool_manifest(modules: Sequence[str]) -> bytes:
    """Render canonical versioned JSON for the packaged frozen registry."""

    names = _validate_module_names(modules)
    payload = {
        "modules": list(names),
        "package": TOOL_PACKAGE,
        "schema_version": _MANIFEST_SCHEMA_VERSION,
    }
    return (json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n").encode("ascii")


def parse_frozen_tool_manifest(raw: bytes) -> tuple[str, ...]:
    """Parse a canonical manifest and reject unknown or ambiguous data."""

    if not isinstance(raw, bytes):
        raise ToolModuleInventoryError("frozen tool manifest must be bytes")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ToolModuleInventoryError(f"invalid frozen tool manifest JSON: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "modules",
        "package",
        "schema_version",
    }:
        raise ToolModuleInventoryError("frozen tool manifest has an invalid schema")
    if payload["package"] != TOOL_PACKAGE:
        raise ToolModuleInventoryError("frozen tool manifest names the wrong package")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != _MANIFEST_SCHEMA_VERSION:
        raise ToolModuleInventoryError("frozen tool manifest has an unsupported schema version")
    if not isinstance(payload["modules"], list) or not all(isinstance(name, str) for name in payload["modules"]):
        raise ToolModuleInventoryError("frozen tool manifest modules must be a string list")
    modules = _validate_module_names(payload["modules"])
    if raw != render_frozen_tool_manifest(modules):
        raise ToolModuleInventoryError("frozen tool manifest is not canonical JSON")
    return modules


def build_frozen_tool_manifest(
    tools_dir: pathlib.Path,
    output_path: pathlib.Path,
) -> ToolModuleInventory:
    """Derive and atomically materialize the manifest used by one build."""

    inventory = discover_tool_module_inventory(tools_dir)
    target = pathlib.Path(output_path)
    raw = render_frozen_tool_manifest(inventory.tool_modules)
    write_text_atomic(target, raw.decode("ascii"), fsync=True)
    verify_frozen_tool_manifest(tools_dir, target)
    return inventory


def load_frozen_tool_modules(path: pathlib.Path | None = None) -> tuple[str, ...]:
    """Load the build-generated manifest beside the packaged module."""

    manifest_path = (
        pathlib.Path(path) if path is not None else pathlib.Path(__file__).with_name(FROZEN_TOOL_MANIFEST_NAME)
    )
    try:
        raw = manifest_path.read_bytes()
    except OSError as exc:
        raise ToolModuleInventoryError(f"cannot read frozen tool manifest {manifest_path}: {exc}") from exc
    return parse_frozen_tool_manifest(raw)


def tool_modules_for_runtime(
    tools_dir: pathlib.Path,
    manifest_path: pathlib.Path | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Select source discovery or the packaged manifest for this process."""

    if getattr(sys, "frozen", False):
        return load_frozen_tool_modules(manifest_path), ()
    inventory, errors = _scan_tool_module_inventory(tools_dir)
    return inventory.tool_modules, errors


def verify_frozen_tool_manifest(
    tools_dir: pathlib.Path,
    manifest_path: pathlib.Path,
    archive_listing_path: pathlib.Path | None = None,
) -> ToolModuleInventory:
    """Verify packaged manifest bytes, membership, and optional PYZ closure."""

    inventory = discover_tool_module_inventory(tools_dir)
    modules = load_frozen_tool_modules(manifest_path)
    if modules != inventory.tool_modules:
        raise ToolModuleInventoryError(
            f"frozen tool manifest membership drifted: expected {inventory.tool_modules!r}, got {modules!r}"
        )
    if archive_listing_path is not None:
        try:
            archive_names = {
                line.strip()
                for line in pathlib.Path(archive_listing_path).read_text(encoding="utf-8-sig").splitlines()
                if line.strip()
            }
        except (OSError, UnicodeDecodeError) as exc:
            raise ToolModuleInventoryError(f"cannot read PyInstaller archive listing: {exc}") from exc
        missing = sorted(set(inventory.package_modules) - archive_names)
        if missing:
            raise ToolModuleInventoryError(f"PyInstaller archive is missing tool modules: {missing}")
    return inventory


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "verify-artifact"))
    parser.add_argument("manifest", type=pathlib.Path)
    parser.add_argument("tools_dir", type=pathlib.Path)
    parser.add_argument("archive_listing", type=pathlib.Path, nargs="?")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "verify-artifact" and args.archive_listing is None:
        parser.error("verify-artifact requires archive_listing")
    if args.command == "verify" and args.archive_listing is not None:
        parser.error("verify does not accept archive_listing")
    inventory = verify_frozen_tool_manifest(args.tools_dir, args.manifest, args.archive_listing)
    print(f"frozen tool inventory OK ({len(inventory.tool_modules)} owners)")
    return 0


__all__ = [
    "FROZEN_TOOL_MANIFEST_NAME",
    "ToolModuleInventory",
    "ToolModuleInventoryError",
    "build_frozen_tool_manifest",
    "discover_tool_module_inventory",
    "load_frozen_tool_modules",
    "parse_frozen_tool_manifest",
    "render_frozen_tool_manifest",
    "tool_modules_for_runtime",
    "verify_frozen_tool_manifest",
]


if __name__ == "__main__":
    raise SystemExit(main())
