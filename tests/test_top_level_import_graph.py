"""Top-level runtime imports remain an acyclic tracked-core graph."""

import graphlib
import pathlib

import pytest

from ouroboros.code_intelligence import (
    _resolve_python_import,
    collect_top_level_python_imports,
)
from ouroboros.review import candidate_repo_paths

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_top_level_import_collector_covers_executed_compound_statements():
    source = """
import root_module
import typing
import typing_extensions as typing_ext
from typing import TYPE_CHECKING
from typing import TYPE_CHECKING as TC
from package import submodule
from . import sibling
from .nested import leaf

try:
    import try_body
except ImportError:
    from try_handler import value
else:
    import try_else
finally:
    import try_finally

if FLAG:
    import if_body
else:
    import if_else

if TYPE_CHECKING:
    import typing_only
else:
    import runtime_else

if not typing.TYPE_CHECKING:
    import runtime_not_typing
else:
    import typing_not_else

if TC:
    import aliased_typing_only
else:
    import aliased_runtime

if typing_ext.TYPE_CHECKING:
    import extension_typing_only
else:
    import extension_runtime

if runtime_flags.TYPE_CHECKING:
    import arbitrary_attribute_body

for item in ():
    import for_body
else:
    import for_else

while FLAG:
    import while_body
else:
    import while_else

with context():
    import with_body

match value:
    case 1:
        import match_one
    case _:
        import match_default

class LoadedAtImport:
    import class_body
    if FLAG:
        import class_nested

    def method(self):
        import method_local

    async def async_method(self):
        import async_method_local

def function_local():
    import function_body

async def async_function_local():
    import async_function_body

deferred = lambda: __import__("lambda_body")
"""

    assert collect_top_level_python_imports(
        source,
        pathlib.PurePosixPath("pkg/module.py"),
    ) == sorted({
        "root_module",
        "typing",
        "typing.TYPE_CHECKING",
        "typing_extensions",
        "package",
        "package.submodule",
        "pkg",
        "pkg.sibling",
        "pkg.nested",
        "pkg.nested.leaf",
        "try_body",
        "try_handler",
        "try_handler.value",
        "try_else",
        "try_finally",
        "if_body",
        "if_else",
        "runtime_else",
        "runtime_not_typing",
        "aliased_runtime",
        "extension_runtime",
        "arbitrary_attribute_body",
        "for_body",
        "for_else",
        "while_body",
        "while_else",
        "with_body",
        "match_one",
        "match_default",
        "class_body",
        "class_nested",
    })


@pytest.mark.serial
def test_tracked_core_top_level_import_graph_is_acyclic():
    paths = sorted(
        path
        for path in candidate_repo_paths(REPO_ROOT)
        if path.endswith(".py")
        and (path.startswith("ouroboros/") or path.startswith("supervisor/") or path == "server.py")
    )
    assert len(paths) >= 271, "tracked-core census unexpectedly shrank"

    graph = {path: set() for path in paths}
    for path in paths:
        source = (REPO_ROOT / path).read_text(encoding="utf-8")
        modules = collect_top_level_python_imports(source, pathlib.PurePosixPath(path))
        graph[path].update(
            resolved
            for module in modules
            if (resolved := _resolve_python_import(REPO_ROOT, module)) in graph
        )

    try:
        order = tuple(graphlib.TopologicalSorter(graph).static_order())
    except graphlib.CycleError as exc:
        pytest.fail(f"top-level import cycle detected: {exc.args[1]}")
    assert len(order) == len(graph)
