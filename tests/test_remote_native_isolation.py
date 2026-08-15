"""RWS v2 §3.3 structural isolation gate for the execd bundle.

Forward: nothing reachable from the transferred transport/native modules may
import Home authority.  Reverse: those same modules may not import Home POLICY
authorities — one brain means Home decides and execd executes.

Both directions come out of one static closure over EVERY import scope, and a
violation must name the module AND the concrete import edge, because "closure
violated" is not actionable when the bundle is 17 modules deep.

Two controls, deliberately:

* the STATIC closure reads source, so it sees an import that is merely written —
  including a function-local one, which is where two real violations were hiding
  while the gate read module scope only;
* the clean-subprocess SMOKE at the bottom of this file runs each kernel import
  for real in a fresh interpreter with a meta-path finder that refuses every
  forbidden module. `tool_capabilities` used to name this smoke as the reason
  function-local imports were out of scope; it did not exist. Now it does.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

from ouroboros.tool_capabilities import (
    FORBIDDEN_REMOTE_IMPORT_PREFIXES,
    REMOTE_NATIVE_CLOSURE_SEEDS,
    assert_remote_native_import_closure,
    remote_native_import_closure,
)
from ouroboros.workspace_native_contract import (
    MANDATORY_REMOTE_NATIVE_OPERATIONS,
    REMOTE_NATIVE_KERNEL_MODULES,
)

REPO = pathlib.Path(__file__).resolve().parents[1]

# Everything Lane 1a transferred has to be INSIDE the gate, not merely near it.
TRANSFERRED_BUNDLE_MODULES = (
    "ouroboros.execd",
    "ouroboros.execd_spool",
    "ouroboros.execd_state",
    "ouroboros.execd_task_files",
    "ouroboros.remote_protocol",
    "ouroboros.workspace_diagnostics",
    "ouroboros.workspace_native",
    "ouroboros.workspace_native_contract",
    "ouroboros.workspace_payload_native",
    "ouroboros.workspace_query_native",
    "ouroboros.workspace_snapshot_native",
)

_FAKE_OPERATION_MODULES = {
    name: "ouroboros.workspace_native" for name in MANDATORY_REMOTE_NATIVE_OPERATIONS
}


# ── the live repository ─────────────────────────────────────────────────


def test_live_execd_import_closure_is_clean():
    audit = assert_remote_native_import_closure(REPO)
    assert audit["forbidden"] == {}
    assert audit["missing_modules"] == []


@pytest.mark.parametrize("module", TRANSFERRED_BUNDLE_MODULES)
def test_every_transferred_module_is_inside_the_gate(module):
    audit = remote_native_import_closure(REPO)
    assert module in audit["modules"]


def test_seed_set_covers_the_transferred_modules():
    assert set(TRANSFERRED_BUNDLE_MODULES) <= set(REMOTE_NATIVE_CLOSURE_SEEDS)


def test_declared_kernel_modules_are_all_reached_by_the_closure():
    """A renamed module must not survive in the declared bundle allowlist."""

    audit = remote_native_import_closure(REPO)
    assert set(REMOTE_NATIVE_KERNEL_MODULES) <= set(audit["modules"])


def test_a_stale_declared_kernel_module_fails_the_gate(monkeypatch):
    import ouroboros.workspace_native_contract as contract

    monkeypatch.setattr(
        contract,
        "REMOTE_NATIVE_KERNEL_MODULES",
        frozenset({*REMOTE_NATIVE_KERNEL_MODULES, "ouroboros.remote_task_files"}),
    )
    with pytest.raises(ValueError) as excinfo:
        assert_remote_native_import_closure(REPO)
    assert "ouroboros.remote_task_files" in str(excinfo.value)


def test_a_missing_seed_fails_before_a_bundle_is_built():
    with pytest.raises(ValueError) as excinfo:
        assert_remote_native_import_closure(
            REPO, extra_roots=["ouroboros.not_a_real_module"]
        )
    assert "missing modules" in str(excinfo.value)
    assert "ouroboros.not_a_real_module" in str(excinfo.value)


# ── synthetic repositories ──────────────────────────────────────────────


def _fake_repo(tmp_path: pathlib.Path, sources: dict[str, str]) -> pathlib.Path:
    """A minimal tree holding every seed, plus the given module sources."""

    root = tmp_path / "repo"
    (root / "ouroboros").mkdir(parents=True)
    (root / "ouroboros" / "__init__.py").write_text("", encoding="utf-8")
    for seed in REMOTE_NATIVE_CLOSURE_SEEDS:
        (root / pathlib.Path(*seed.split("."))).with_suffix(".py").write_text(
            "", encoding="utf-8"
        )
    for module, source in sources.items():
        path = (root / pathlib.Path(*module.split("."))).with_suffix(".py")
        if module.endswith(".__init__"):
            path = root / pathlib.Path(*module.split(".")[:-1]) / "__init__.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        for parent in path.parents:
            if parent == root:
                break
            (parent / "__init__.py").touch()
        path.write_text(source, encoding="utf-8")
    return root


def _audit(root: pathlib.Path) -> dict:
    return remote_native_import_closure(
        root, operation_modules=_FAKE_OPERATION_MODULES
    )


def _assert_fails(root: pathlib.Path) -> str:
    # A synthetic tree declares no kernel bundle; only the closure is under test.
    with pytest.raises(ValueError) as excinfo:
        assert_remote_native_import_closure(
            root,
            operation_modules=_FAKE_OPERATION_MODULES,
            declared_kernel_modules=(),
        )
    return str(excinfo.value)


def test_a_direct_home_artifact_import_names_the_module_and_the_edge(tmp_path):
    root = _fake_repo(
        tmp_path,
        {
            "ouroboros.execd": "from ouroboros.artifacts import record\n",
            "ouroboros.artifacts": "",
        },
    )
    audit = _audit(root)
    rows = audit["forbidden"]["home_task_or_artifact_state"]
    assert [row["module"] for row in rows] == ["ouroboros.artifacts"]
    assert rows[0]["edge"] == "ouroboros.execd -> ouroboros.artifacts"
    message = _assert_fails(root)
    assert "ouroboros.artifacts" in message
    assert "ouroboros.execd -> ouroboros.artifacts" in message


@pytest.mark.parametrize(
    "module,source",
    [
        ("ouroboros.tool_access", "ouroboros.tool_access"),
        ("ouroboros.protected_artifacts", "ouroboros.protected_artifacts"),
        ("ouroboros.observability", "ouroboros.observability"),
        ("ouroboros.workspace_executor", "ouroboros.workspace_executor"),
        ("ouroboros.safety", "ouroboros.safety"),
    ],
)
def test_reverse_gate_forbids_each_named_home_policy_authority(tmp_path, module, source):
    root = _fake_repo(
        tmp_path,
        {
            "ouroboros.workspace_native": f"import {source}\n",
            module: "",
        },
    )
    audit = _audit(root)
    rows = audit["forbidden"]["home_policy_authority"]
    assert module in [row["module"] for row in rows]
    assert f"ouroboros.workspace_native -> {module}" in [row["edge"] for row in rows]


@pytest.mark.parametrize(
    "package,expected_category",
    [
        ("ouroboros.gateway", "server_or_gateway"),
        ("ouroboros.gateways", "server_or_gateway"),
        ("ouroboros.supervisor", "server_or_gateway"),
        ("supervisor", "server_or_gateway"),
        ("ouroboros.tools", "registry"),
    ],
)
def test_reverse_gate_forbids_whole_home_packages(tmp_path, package, expected_category):
    root = _fake_repo(
        tmp_path,
        {
            "ouroboros.execd_state": f"from {package}.queue import persist\n",
            f"{package}.__init__": "",
            f"{package}.queue": "",
        },
    )
    audit = _audit(root)
    modules = [row["module"] for row in audit["forbidden"][expected_category]]
    assert package in modules
    assert f"{package}.queue" in modules


def test_a_transitive_import_names_the_immediate_edge_and_the_whole_chain(tmp_path):
    root = _fake_repo(
        tmp_path,
        {
            "ouroboros.execd": "from ouroboros import middle_helper\n",
            "ouroboros.middle_helper": "from ouroboros import deeper_helper\n",
            "ouroboros.deeper_helper": "import ouroboros.tool_access\n",
            "ouroboros.tool_access": "",
        },
    )
    rows = _audit(root)["forbidden"]["home_policy_authority"]
    assert rows[0]["edge"] == "ouroboros.deeper_helper -> ouroboros.tool_access"
    assert rows[0]["path"] == (
        "ouroboros.execd -> ouroboros.middle_helper -> ouroboros.deeper_helper"
        " -> ouroboros.tool_access"
    )
    message = _assert_fails(root)
    assert "ouroboros.deeper_helper -> ouroboros.tool_access" in message


def test_a_function_local_import_is_a_violation_too(tmp_path):
    """The gap two real violations were living in.

    The gate read module scope only and this module's comment claimed "a
    clean-subprocess invocation smoke covers those per native operation" — no such
    smoke existed. So `remote_ssh_config` reached `ouroboros.config` and
    `ouroboros.utils` (a declared bundle member that really travels) reached
    `ouroboros.observability`, both from inside function bodies, and the gate was
    green. Deferring a scope to a compensating control that does not exist is not
    a scoping decision; it is an unchecked hole.
    """

    root = _fake_repo(
        tmp_path,
        {
            "ouroboros.execd": (
                "def late():\n    from ouroboros import artifacts\n    return artifacts\n"
            ),
            "ouroboros.artifacts": "",
        },
    )
    audit = _audit(root)
    rows = audit["forbidden"]["home_task_or_artifact_state"]
    assert [row["module"] for row in rows] == ["ouroboros.artifacts"]
    assert rows[0]["edge"] == "ouroboros.execd -> ouroboros.artifacts"
    assert rows[0]["scope"] == "function_local"
    # Not TRAVERSED, though: a conditional edge does not drag its own closure in.
    assert "ouroboros.artifacts" not in audit["modules"]
    assert "ouroboros.artifacts" in audit["function_local_edges"]["ouroboros.execd"]
    assert "ouroboros.execd -> ouroboros.artifacts" in _assert_fails(root)


def test_a_function_local_edge_is_not_followed_transitively(tmp_path):
    """Why the two scopes are treated differently, stated as behaviour.

    Following function-local imports transitively reaches 226 modules in the live
    repo — effectively the whole program, because a late import is this codebase's
    ordinary cycle-breaker. A gate that always fails teaches nothing, so the
    conditional edge is judged where it is SPELLED and its own imports are left
    alone.
    """

    root = _fake_repo(
        tmp_path,
        {
            "ouroboros.execd": "def late():\n    from ouroboros import lazy_helper\n",
            "ouroboros.lazy_helper": "from ouroboros import artifacts\n",
            "ouroboros.artifacts": "",
        },
    )
    audit = _audit(root)
    assert audit["forbidden"] == {}
    assert "ouroboros.lazy_helper" not in audit["modules"]


def test_a_function_local_import_inside_a_class_body_method_is_seen(tmp_path):
    root = _fake_repo(
        tmp_path,
        {
            "ouroboros.workspace_native": (
                "class Runner:\n"
                "    def run(self):\n"
                "        from ouroboros.tool_access import decide\n"
                "        return decide\n"
            ),
            "ouroboros.tool_access": "",
        },
    )
    rows = _audit(root)["forbidden"]["home_policy_authority"]
    assert rows[0]["module"] == "ouroboros.tool_access"
    assert rows[0]["scope"] == "function_local"


def test_conditional_module_scope_imports_are_inside_the_boundary(tmp_path):
    """A `try:`/`if:` wrapper is not an escape hatch — it still runs on import."""

    root = _fake_repo(
        tmp_path,
        {
            "ouroboros.execd": (
                "try:\n    from ouroboros import observability\n"
                "except ImportError:\n    observability = None\n"
            ),
            "ouroboros.observability": "",
        },
    )
    rows = _audit(root)["forbidden"]["home_policy_authority"]
    assert rows[0]["edge"] == "ouroboros.execd -> ouroboros.observability"


def test_relative_imports_resolve_to_the_same_forbidden_module(tmp_path):
    root = _fake_repo(
        tmp_path,
        {
            "ouroboros.workspace_query_native": "from .tool_access import decide\n",
            "ouroboros.tool_access": "",
        },
    )
    rows = _audit(root)["forbidden"]["home_policy_authority"]
    assert rows[0]["module"] == "ouroboros.tool_access"
    assert rows[0]["edge"] == "ouroboros.workspace_query_native -> ouroboros.tool_access"


@pytest.mark.parametrize(
    "source,expected_module,expected_scope",
    [
        (
            "import importlib\n"
            "def late():\n    return importlib.import_module('ouroboros.artifacts')\n",
            "ouroboros.artifacts",
            "function_local",
        ),
        (
            "def late():\n    return __import__('ouroboros.tool_access')\n",
            "ouroboros.tool_access",
            "function_local",
        ),
        (
            # The bare-name importer: `func` is a Name, and matching only
            # `<x>.import_module` plus the builtin `__import__` left it unchecked.
            "from importlib import import_module\n"
            "def late():\n    return import_module('ouroboros.observability')\n",
            "ouroboros.observability",
            "function_local",
        ),
        (
            "from importlib import import_module as imp\n"
            "def late():\n    return imp('ouroboros.tool_access')\n",
            "ouroboros.tool_access",
            "function_local",
        ),
        (
            # Module scope: it runs merely by importing, so it is the harder form.
            "import importlib\nARTIFACTS = importlib.import_module('ouroboros.artifacts')\n",
            "ouroboros.artifacts",
            "module",
        ),
        (
            "from importlib import import_module\n"
            "def late():\n    return import_module('.tool_access', 'ouroboros')\n",
            "ouroboros.tool_access",
            "function_local",
        ),
    ],
)
def test_a_dynamic_import_naming_a_home_module_is_a_violation(
    tmp_path, source, expected_module, expected_scope
):
    """A literal module name is an import edge whichever statement spells it.

    The gate read `ast.Import`/`ast.ImportFrom` only, so a bundle-core module could
    name a Home authority in plain source through `importlib.import_module(...)`,
    `__import__(...)`, or a bare/aliased `import_module` — and pass BOTH this static
    gate and the clean-subprocess smoke, which never executes the branch that
    imports. Every form here was verified undetected before the fix.
    """

    root = _fake_repo(tmp_path, {"ouroboros.execd": source, expected_module: ""})
    audit = _audit(root)
    rows = [
        row
        for category_rows in audit["forbidden"].values()
        for row in category_rows
        if row["module"] == expected_module
    ]
    assert rows, audit["forbidden"]
    assert rows[0]["scope"] == expected_scope
    assert f"ouroboros.execd -> {expected_module}" in _assert_fails(root)


@pytest.mark.parametrize(
    "source",
    [
        # A genuinely dynamic name: nothing in the source says which module.
        "from importlib import import_module\ndef late(name):\n    return import_module(name)\n",
        "import importlib\ndef late(name):\n    return importlib.import_module('ouroboros.' + name)\n",
        # A stdlib module is not a Home authority.
        "import importlib\ndef late():\n    return importlib.import_module('json')\n",
        # A relative spelling with no resolvable literal anchor.
        "from importlib import import_module\ndef late(pkg):\n    return import_module('.artifacts', pkg)\n",
    ],
)
def test_the_dynamic_import_boundary_is_named_and_real(tmp_path, source):
    """The residue the scan admits to missing, pinned so the admission stays true.

    A module name assembled or passed at runtime cannot be resolved without running
    the program. `_local_imports_by_scope`'s docstring says so; if one of these ever
    starts being caught, that BOUNDARY note is stale and must be narrowed.
    """

    root = _fake_repo(tmp_path, {"ouroboros.execd": source, "ouroboros.artifacts": ""})
    assert _audit(root)["forbidden"] == {}


def test_a_clean_synthetic_bundle_passes(tmp_path):
    root = _fake_repo(
        tmp_path,
        {"ouroboros.execd": "from ouroboros import execd_state, execd_spool\n"},
    )
    audit = assert_remote_native_import_closure(
        root,
        operation_modules=_FAKE_OPERATION_MODULES,
        declared_kernel_modules=(),
    )
    assert audit["forbidden"] == {}


def test_every_forbidden_category_names_at_least_one_prefix():
    assert FORBIDDEN_REMOTE_IMPORT_PREFIXES
    for category, prefixes in FORBIDDEN_REMOTE_IMPORT_PREFIXES.items():
        assert prefixes, category
        assert all(isinstance(prefix, str) and prefix for prefix in prefixes), category


def test_a_prefix_match_does_not_leak_to_a_similarly_named_module(tmp_path):
    """`ouroboros.tool_capabilities` is not `ouroboros.tool_access`."""

    root = _fake_repo(
        tmp_path,
        {
            "ouroboros.execd": "from ouroboros import tool_accessory\n",
            "ouroboros.tool_accessory": "",
        },
    )
    assert _audit(root)["forbidden"] == {}


# ── the compensating control: a real clean-subprocess import smoke ───────
#
# The static gate reads source. This runs the import for real, in a fresh
# interpreter with nothing of Home already in `sys.modules`, behind a meta-path
# finder that REFUSES any forbidden module. It answers a question AST cannot: does
# importing this kernel module actually work when no Home module is on the path,
# and does the real import machinery — `__init__` side effects, conditional
# branches, `importlib` calls, C-level hooks — stay inside the boundary?
#
# Its own limit, stated plainly: it observes IMPORT time. A function-local import
# inside an uncalled function does not fire here; that scope is the static gate's
# job, and the two controls are complementary rather than redundant.

_ISOLATION_SMOKE = '''
import sys

sys.path.insert(0, sys.argv[1])
FORBIDDEN = tuple(sys.argv[3].split(","))
TARGET = sys.argv[2]


class _Refuser:
    """Refuse a forbidden module the moment anything asks for it."""

    def find_module(self, fullname, path=None):
        return self.find_spec(fullname, path)

    def find_spec(self, fullname, path=None, target=None):
        for prefix in FORBIDDEN:
            if fullname == prefix or fullname.startswith(prefix + "."):
                raise ImportError(
                    "FORBIDDEN_IMPORT:" + fullname + ":requested_while_importing:" + TARGET
                )
        return None


sys.meta_path.insert(0, _Refuser())
__import__(TARGET)
print("CLEAN:" + TARGET)
'''


def _forbidden_prefixes() -> str:
    return ",".join(
        sorted(
            prefix
            for prefixes in FORBIDDEN_REMOTE_IMPORT_PREFIXES.values()
            for prefix in prefixes
        )
    )


def _clean_import(module: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-E", "-c", _ISOLATION_SMOKE, str(REPO), module, _forbidden_prefixes()],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=str(REPO.parent),
    )


@pytest.mark.parametrize("module", sorted(REMOTE_NATIVE_KERNEL_MODULES))
def test_each_kernel_module_imports_clean_in_a_fresh_interpreter(module):
    """The control `tool_capabilities` used to CLAIM and never had."""

    result = _clean_import(module)
    assert result.returncode == 0, (
        f"{module} could not be imported with no Home module on the path:\n"
        f"{result.stderr[-4000:]}"
    )
    assert f"CLEAN:{module}" in result.stdout


@pytest.mark.parametrize("module", sorted(REMOTE_NATIVE_CLOSURE_SEEDS))
def test_each_closure_seed_imports_clean_in_a_fresh_interpreter(module):
    result = _clean_import(module)
    assert result.returncode == 0, (
        f"{module} could not be imported with no Home module on the path:\n"
        f"{result.stderr[-4000:]}"
    )


def test_the_smoke_itself_fails_when_a_forbidden_module_is_reached():
    """Never trust a green control that was never seen red.

    `ouroboros.remote_workspace` is Home-side and imports the transfer/policy
    authorities at module scope, so the refuser must fire on it. If this ever
    passes, the smoke above is decorative.
    """

    result = _clean_import("ouroboros.remote_workspace")
    assert result.returncode != 0
    assert "FORBIDDEN_IMPORT:" in result.stderr
