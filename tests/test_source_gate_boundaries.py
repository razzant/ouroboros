"""CLASS GATE 3 — a source-inspecting gate must state what it cannot see.

The defect: a gate reads the source, catches the LITERAL spelling of the thing it forbids,
and passes on every indirect one — `getattr`, an import alias, `importlib`, a `**kwargs`
splat, a relative import, a two-line variable hop.  It was found in the platform gate, in
the isolation gate and in the export channels' grep-proof, which means it is a class and
not three bugs.  The dangerous part is not the miss: it is that a gate which misses
silently reads as PROOF, so the invariant looks defended and nobody looks again.

The structural close is the pattern `tests/test_platform_guard.py` established, and it has
two halves that only work together:

1. the gate's own docstring carries a ``BOUNDARY`` paragraph naming, in concrete syntax,
   the forms it does not catch and where that residue is covered instead;
2. a meta-test asserts the named blind spots are STILL blind, so the admission cannot go
   stale in the other direction — if one starts being caught, the paragraph must narrow.

This file makes half 1 mandatory for every source gate over the remote-workspaces feature.
Each gate then owns half 2 next to itself, where the scanner it describes actually lives
(``test_remote_panic_descriptors``, ``test_cli_entrypoint``, ``test_remote_export_policy``
each carry their own "sees the indirect spellings" / "blind spots are really blind" pair).

BOUNDARY of THIS gate, held to its own standard: it discovers gates by reading each test
function's source for a source-reading primitive, a REPO-rooted anchor, and a feature
module's name.  A gate that reaches the source through a helper in ``conftest.py``, or
names its target with a computed path, is invisible here — so is a gate that inspects the
feature through a module absent from ``FEATURE_MODULE_STEMS``, and so is one that roots its
read in a fixture rather than in the repo.  It also cannot judge whether a
``BOUNDARY`` paragraph is TRUE, only that one exists; truth is what the per-gate blind-spot
tests are for, and this gate asserts those exist for the three hardened scanners by name.
"""

from __future__ import annotations

import ast
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parent.parent

# Module STEMS whose inspection makes a test a "feature source gate" — deliberately the
# feature's own distinctive names. Generic filenames the feature happens to contain
# (`registry.py`, `verify.py`, `connections.py`, `projects.py`) are excluded on purpose:
# they matched tests that police unrelated invariants, and a discovery that drags those in
# turns the boundary requirement into paperwork.
FEATURE_MODULE_STEMS: frozenset[str] = frozenset(
    {p.stem for p in REPO.glob("ouroboros/remote_*.py")}
    | {p.stem for p in REPO.glob("ouroboros/execd*.py")}
    | {p.stem for p in REPO.glob("ouroboros/workspace_*.py")}
    | {
        "connection_store",
        "cli_connections",
        "cli_projects",
        "export_policy_contract",
        "tool_capabilities",
        "dispatch_args",
        "dispatch_execute",
        "dispatch_prepare",
        "shell_guards",
        "shell_guards_runtime",
        "shell_guards_target",
    }
)

# Primitives that make a test an inspector of source rather than of behaviour.
SOURCE_READERS: tuple[str, ...] = ("read_text", "getsource", "ast.parse", "ast.walk", "NodeVisitor")

# ...but reading BYTES is not inspecting SOURCE. A behavioural test reads files it wrote
# under `tmp_path` all the time. A gate is distinguished by rooting its read in the REPO
# or in a module object, so one of these tokens must appear too. Without this the
# discovery pulled in 27 ordinary behavioural tests and the boundary requirement would
# have become a tax on tests that inspect nothing.
REPO_ROOTED: tuple[str, ...] = ("__file__", "REPO", "parents[", "parent.parent", "getsource")

# Gates that read a feature file INCIDENTALLY while policing something else, so a
# BOUNDARY paragraph about the remote feature would be noise. Each says what it is really
# about; being on this list is a claim a reviewer can check, not a mute button.
NON_FEATURE_INSPECTORS: dict[tuple[str, str], str] = {
    (
        "tests/test_remote_contract_compatibility.py",
        "test_the_shipped_bundle_declares_this_builds_contract_set",
    ): (
        "about a BUILD ARTIFACT, not about source: it reads the `contract_set_version` "
        "field of `assets/execd/manifest.json`, the JSON manifest the packager writes "
        "beside the execd tarballs, and compares one integer against "
        "`remote_contracts.CONTRACT_SET_VERSION`. It parses no Python, walks no AST and "
        "asserts nothing about how any module is written — the discovery matches it only "
        "because a JSON read under `REPO` looks like a source read from the outside"
    ),
    (
        "tests/test_launcher_sync.py",
        "test_start_agent_unix_uses_process_group_and_writes_server_record",
    ): (
        "about the launcher's process-group discipline and its server record; it names a "
        "workspace module only in passing and asserts nothing about placement"
    ),
    (
        "tests/test_tool_capabilities.py",
        "test_tool_policy_defines_no_local_tool_sets",
    ): (
        "an SSOT-location rule (tool sets live in one module) that predates the feature; "
        "it reads the capabilities module because that IS the SSOT, not to police routing"
    ),
    (
        "tests/test_tool_capabilities.py",
        "test_loop_execution_imports_from_capabilities",
    ): (
        "the same SSOT-location rule for the loop's import; a placement boundary paragraph "
        "would describe something this test never looks at"
    ),
    (
        "tests/test_remote_connection_security.py",
        "test_the_three_observed_host_id_readers_look_in_the_same_three_places",
    ): (
        "its Python half is a real behavioural triple-check of both readers; only the "
        "browser mirror is read as text, and that half's limits are named in the test's "
        "own docstring (a `const observedHostId = (p) => {}` rewrite, or "
        "`payload[\"handshake\"]` bracket access, is invisible to its regex)"
    ),
}

# The scanners hardened in this pass, each of which must keep BOTH halves of the pattern.
HARDENED_SCANNERS: tuple[tuple[str, str, str], ...] = (
    (
        "tests/test_remote_panic_descriptors.py",
        "_borrowed_descriptor_closes",
        "test_the_named_descriptor_blind_spots_are_really_blind",
    ),
    (
        "tests/test_cli_entrypoint.py",
        "_cli_module_reaches",
        "test_the_cli_import_scan_states_its_own_boundary",
    ),
    (
        # Two lanes hardened this scanner independently and the merge kept BOTH sets of
        # forms; the union lives in `blob_channel_sites`, which also owns the BOUNDARY
        # paragraph and is keyed by (file, function) rather than by file.
        "tests/test_remote_export_policy.py",
        "blob_channel_sites",
        "test_the_gate_states_its_own_boundary",
    ),
)


def _names_a_feature_module(segment: str) -> bool:
    """True when the test names a feature module — as a filename OR as a module.

    Both spellings occur and both must count: a gate may write
    ``"ouroboros/remote_ssh.py"`` or reach the same bytes through
    ``pathlib.Path(remote_ssh.__file__).read_text()``. Matching only the filename form
    missed a real gate (`test_bounded_disclosure`'s four-slice fence) that uses the
    second, which is the same "the gate saw one spelling" defect this file is about.
    """

    return any(
        re.search(rf"\b{re.escape(stem)}\b", segment) for stem in FEATURE_MODULE_STEMS
    )


def _feature_source_gates():
    """Yield (rel, lineno, test_name, boundary_text_reachable) per discovered gate."""

    for path in sorted(REPO.glob("tests/test_*.py")):
        text = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        rel = str(path.relative_to(REPO))
        module_doc = ast.get_docstring(tree) or ""
        helper_docs = {
            node.name: (ast.get_docstring(node) or "")
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not node.name.startswith("test_"):
                continue
            segment = ast.get_source_segment(text, node) or ""
            if not any(reader in segment for reader in SOURCE_READERS):
                continue
            if not any(anchor in segment for anchor in REPO_ROOTED):
                continue
            if not _names_a_feature_module(segment):
                continue
            # A boundary may live on the test, on the module, or on a helper the test
            # calls — the reference pattern puts it wherever the scanner is.
            reachable = (ast.get_docstring(node) or "") + module_doc
            for helper, doc in helper_docs.items():
                if helper in segment:
                    reachable += doc
            yield rel, node.lineno, node.name, reachable


def test_every_feature_source_gate_states_its_boundary():
    """A gate that reads the feature's source says what its reading cannot see.

    Failing here has exactly two honest fixes: write the BOUNDARY paragraph (naming the
    indirect forms in concrete syntax, as `test_platform_guard.py` does), or — if the gate
    is really about something else — declare it in NON_FEATURE_INSPECTORS with what it IS
    about. Weakening this rule is never the fix.
    """

    undeclared: list[str] = []
    for rel, lineno, name, reachable in _feature_source_gates():
        if "BOUNDARY" in reachable:
            continue
        if (rel, name) in NON_FEATURE_INSPECTORS:
            continue
        undeclared.append(f"{rel}:{lineno}: {name}")
    assert not undeclared, (
        "source gate over the remote feature with no stated boundary — a gate that misses "
        "silently reads as proof:\n" + "\n".join(undeclared)
    )


def test_the_discovery_is_not_vacuous():
    """The scan must still find gates, or this file is a green light over nothing.

    The count is a floor, not a pin: gates should be able to appear without editing this
    test, but the discovery collapsing to zero (a renamed primitive, a moved module) must
    be loud rather than silently reassuring.
    """

    found = list(_feature_source_gates())
    assert len(found) >= 8, (
        f"only {len(found)} feature source gates discovered — the discovery heuristic has "
        "stopped matching the codebase; re-derive it before trusting this file"
    )
    documented = [row for row in found if "BOUNDARY" in row[3]]
    assert len(documented) >= 5, (
        f"only {len(documented)} of {len(found)} state a boundary; the pattern is supposed "
        "to be spreading, not receding"
    )


def test_declared_non_feature_inspectors_still_exist_and_still_read_source():
    """The exemption list cannot cover tests that no longer exist or no longer inspect.

    A stale exemption is how a list like this turns into an unread allowlist; if a gate on
    it was deleted or rewritten into a behavioural test, the row must go so the next real
    source gate with that name cannot inherit the pass.
    """

    discovered = {(rel, name) for rel, _lineno, name, _doc in _feature_source_gates()}
    stale = [
        f"{rel}::{name}"
        for (rel, name), _reason in NON_FEATURE_INSPECTORS.items()
        if (rel, name) not in discovered
    ]
    assert not stale, (
        "NON_FEATURE_INSPECTORS names gates the discovery no longer finds — remove them:\n"
        + "\n".join(stale)
    )
    for (rel, name), reason in NON_FEATURE_INSPECTORS.items():
        assert len(reason) > 40, f"{rel}::{name} needs a real reason, got {reason!r}"


def test_each_hardened_scanner_keeps_both_halves_of_the_pattern():
    """A BOUNDARY paragraph without a blind-spot test is half a control.

    The paragraph says what is not caught; only an assertion can keep that true. Both must
    exist for each scanner hardened in this pass, and the scanner must still be there.
    """

    missing: list[str] = []
    for rel, scanner, blind_spot_test in HARDENED_SCANNERS:
        path = REPO / rel
        assert path.exists(), f"{rel} is gone; update HARDENED_SCANNERS"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        functions = {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        if scanner not in functions:
            missing.append(f"{rel}: scanner {scanner} no longer exists")
            continue
        doc = ast.get_docstring(functions[scanner]) or ""
        if "BOUNDARY" not in doc:
            missing.append(f"{rel}::{scanner} lost its BOUNDARY paragraph")
        if blind_spot_test not in functions:
            missing.append(
                f"{rel}: {blind_spot_test} is gone — nothing now keeps "
                f"{scanner}'s admission from going stale"
            )
    assert not missing, "\n".join(missing)
