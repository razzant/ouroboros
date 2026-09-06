"""Structural contracts for the semantic-no-op extension loader extraction."""

from __future__ import annotations

import ast
import pathlib

from ouroboros import (
    extension_child_catalog,
    extension_import_staging,
    extension_liveness,
    extension_loader,
    extension_plugin_api,
    extension_registry_state,
    extension_surface_names,
)

REPO = pathlib.Path(__file__).parents[1]

_LEAVES = (
    extension_registry_state,
    extension_surface_names,
    extension_child_catalog,
    extension_import_staging,
    extension_liveness,
    extension_plugin_api,
)

_MOVED_OWNERS = {
    "_ExtensionLoadFailure": extension_registry_state,
    "_ExtensionRegistrations": extension_registry_state,
    "_PluginAPIConfig": extension_registry_state,
    "_extension_modules": extension_registry_state,
    "_extensions": extension_registry_state,
    "_lifecycle_lock_for": extension_registry_state,
    "_lifecycle_locks": extension_registry_state,
    "_load_failures": extension_registry_state,
    "_lock": extension_registry_state,
    "_record_companion_name": extension_registry_state,
    "_routes": extension_registry_state,
    "_settings_sections": extension_registry_state,
    "_tools": extension_registry_state,
    "_ui_tabs": extension_registry_state,
    "_unloading": extension_registry_state,
    "_ws_handlers": extension_registry_state,
    "extension_generation_digest": extension_registry_state,
    # Upstream ``fa397986`` / ``afe1fc4f`` added these two one-lock reads to the
    # loader; they read only ``_ui_tabs``/``_extensions``, so they are owned here
    # and the loader re-exports the very same objects for its gateway callers.
    "live_module_sources": extension_registry_state,
    "live_widget_projection": extension_registry_state,
    "_EXTENSION_NAME_PREFIX": extension_surface_names,
    "_EXTENSION_NAME_RE": extension_surface_names,
    "_EXTENSION_SHORT_MAX": extension_surface_names,
    "_EXTENSION_SKILL_TOKEN_MAX": extension_surface_names,
    "_assert_namespace_path": extension_surface_names,
    "_assert_tool_name": extension_surface_names,
    "_extension_skill_token": extension_surface_names,
    "_widget_geometry_from_render": extension_surface_names,
    "_widget_span_from_render": extension_surface_names,
    "extension_name_prefix": extension_surface_names,
    "extension_surface_name": extension_surface_names,
    "parse_extension_surface_name": extension_surface_names,
    "_out_of_process_handler_proxy": extension_child_catalog,
    "_stage_out_of_process_surfaces": extension_child_catalog,
    "_validate_child_catalog_namespace": extension_child_catalog,
    "_validate_child_route_descriptor": extension_child_catalog,
    "_validate_child_settings_descriptor": extension_child_catalog,
    "_validate_child_tool_descriptor": extension_child_catalog,
    "_validate_child_ui_descriptor": extension_child_catalog,
    "_validate_child_ws_descriptor": extension_child_catalog,
    "_IMPORT_SWEEP_GRACE_SEC": extension_import_staging,
    "_module_key": extension_import_staging,
    "_plugin_entry_path": extension_import_staging,
    "_purge_extension_bytecode": extension_import_staging,
    "_stage_extension_import_tree": extension_import_staging,
    "_sweep_stale_extension_imports": extension_import_staging,
    "_apply_deps_block": extension_liveness,
    "_apply_durable_extension_health": extension_liveness,
    "_deps_block_reason": extension_liveness,
    "_extension_runtime_state": extension_liveness,
    "_revert_enabled_after_load_error": extension_liveness,
    "is_extension_live": extension_liveness,
    "runtime_state_for_loaded_skill": extension_liveness,
    "runtime_state_for_skill_name": extension_liveness,
    "PluginAPIImpl": extension_plugin_api,
    "_reject_extension_child_side_effect": extension_plugin_api,
    "current_execution_mode": extension_plugin_api,
    "mint_skill_token": extension_plugin_api,
    "set_ws_broadcaster": extension_plugin_api,
}

# The loader keeps the lifecycle it is named for: catalog installation for an
# out-of-process child, companion spawning, reconcile, load, unload, reload and
# the read-only snapshots the host and the tool registry consume. (Fix-round-4:
# ``_stage_out_of_process_surfaces`` moved to the child-catalog leaf it
# composes; the loader re-exports it.)
# Merge with upstream ``23ab428f``: ``_finalize_extension_reconcile`` is the one
# arrival — upstream ``01a9df8e`` added the reconcile receipt as
# ``_request_server_reconcile_if_worker`` and ``2f1e2c23`` folded the durable
# health record into that same exit, so it closes the lifecycle the loader owns.
# The widget projections upstream added beside it (``live_widget_projection``
# ``fa397986``, ``live_module_sources`` ``afe1fc4f``) are NOT here: they are
# one-lock reads over ``_ui_tabs``/``_extensions`` and live in the leaf that
# owns those registries, re-exported by the facade for their gateway callers.
_STAYED = (
    "__all__", "_finalize_extension_reconcile", "_publish_out_of_process_registration",
    "_run_unload_callback", "_unload_extension_locked",
    "ensure_companions_running", "get_tool", "list_companion_names", "list_routes",
    "list_ws_handlers", "load_extension", "log", "reconcile_extension", "reload_all",
    "snapshot", "unload_extension",
)


def test_extension_leaves_never_import_the_loader_they_serve():
    """The dependency is one-way, which is what keeps the import graph a DAG."""
    for module in _LEAVES:
        tree = ast.parse(pathlib.Path(module.__file__).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module != "ouroboros.extension_loader", module.__name__
            if isinstance(node, ast.Import):
                assert all(a.name != "ouroboros.extension_loader" for a in node.names), module.__name__


def test_extension_facade_reexports_every_moved_identity():
    """``extension_loader`` keeps the exact objects, so the server, the gateway,
    the tool registry, skill exec and the tests that reach into the registries
    see no identity change."""
    for name, owner in _MOVED_OWNERS.items():
        assert hasattr(extension_loader, name), name
        assert getattr(extension_loader, name) is getattr(owner, name), name
    owned = {name for module in _LEAVES for name in vars(module)}
    assert set(_MOVED_OWNERS) <= owned


def test_the_broadcaster_slot_has_exactly_one_binding():
    """``_ws_broadcaster`` is rebound by ``set_ws_broadcaster``, so it is the one
    moved name the loader deliberately does NOT re-export: a facade copy would be
    a snapshot that silently stops tracking its owner."""
    assert not hasattr(extension_loader, "_ws_broadcaster")
    previous = extension_plugin_api._ws_broadcaster
    try:
        sentinel = object()
        extension_loader.set_ws_broadcaster(sentinel)
        assert extension_plugin_api._ws_broadcaster is sentinel
    finally:
        extension_plugin_api.set_ws_broadcaster(previous)


def test_the_loader_kept_only_the_extension_lifecycle():
    tree = ast.parse(pathlib.Path(extension_loader.__file__).read_text(encoding="utf-8"))
    defined = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.append(node.name)
        elif isinstance(node, ast.Assign):
            defined.extend(t.id for t in node.targets if isinstance(t, ast.Name))
    assert sorted(defined) == sorted(_STAYED)


def test_the_public_loader_surface_is_unchanged():
    # Merge with upstream ``23ab428f``: the two widget reads are the only names
    # the extraction did not already publish — ``live_widget_projection``
    # (``fa397986``) and ``live_module_sources`` (``afe1fc4f``), which the
    # gateway reaches as ``extension_loader.<name>``. Their bodies live in
    # ``extension_registry_state``; the facade re-export is what this pins.
    assert extension_loader.__all__ == [
        "PluginAPIImpl", "is_extension_live", "load_extension", "reconcile_extension",
        "ensure_companions_running", "unload_extension", "reload_all",
        "runtime_state_for_skill_name", "snapshot",
        "live_widget_projection", "live_module_sources",
        "get_tool", "list_ws_handlers",
        "list_routes", "list_companion_names", "current_execution_mode",
    ]
    for name in extension_loader.__all__:
        assert hasattr(extension_loader, name), name


def test_extension_extraction_size_bounds_have_meaningful_headroom():
    counts = {
        module.__name__: len(
            pathlib.Path(module.__file__).read_text(encoding="utf-8").splitlines()
        )
        for module in (extension_loader, *_LEAVES)
    }
    assert counts["ouroboros.extension_loader"] <= 1000
    assert all(count <= 1000 for count in counts.values())
    assert 600 <= counts["ouroboros.extension_plugin_api"] <= 1000
