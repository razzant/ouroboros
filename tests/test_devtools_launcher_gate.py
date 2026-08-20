"""The structural gate every migrated benchmark launcher passes.

Split verbatim out of ``tests/test_devtools_benchmarks.py`` by theme. This module owns the
synthetic launchers the gate is proved against, the manifest seams a launcher may not
publish from, the write forms whose destination the gate must place, and the pre-admission
reads and refusal authorities it resolves through helpers.
"""

from __future__ import annotations

import ast
import inspect

import pytest


from tests._devtools_benchmarks_shared import REPO_ROOT
from tests._devtools_benchmarks_shared import _isolate_bench_runs_root as __isolate_bench_runs_root

# Fixtures are requested by name as test parameters, so they are re-bound through a
# module attribute: a direct import of a name that reappears as a parameter is an F811
# redefinition under the CI ruff gate.
_isolate_bench_runs_root = __isolate_bench_runs_root


# A synthetic launcher-shaped module for pinning the pre-admission resolver itself.
# Deliberately not a real launcher: the gate's BEHAVIOUR is what must not regress.
_GUARD_PROBE_SOURCE = '''
def _looks_innocent(path):
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=path)

def _two_levels_down(path):
    return _looks_innocent(path)

def _three_levels_down(path):
    return _two_levels_down(path)

def _pure(a, b):
    return f"{a}/{b}"

def _steps_aside(root):
    root.mkdir(parents=True, exist_ok=True)
    return None

def main():
    args = parse_args()
    if args.collect_only:
        _steps_aside(args.out)
        return 0
    label = _pure(args.a, args.b)
    provenance = _looks_innocent(args.repo)
    manifest = admit_benchmark_run(args.out, label=label, extra=provenance)
    return finish(manifest)
'''

# A synthetic launcher that violates BOTH invariants, in the exact shapes round 6 found:
# `ensure_outside_repo` (an IMPORTED helper that mkdirs what it validates) called before
# admission, and an output path confined against a module-level constant while the run's
# provenance is attested against the checkout the launcher was HANDED.
_VIOLATING_LAUNCHER_SOURCE = '''
import pathlib
from devtools.benchmarks.common.manifests import admit_benchmark_run, finalize_run_manifest
from devtools.benchmarks.common.run_roots import ensure_outside_repo

REPO = pathlib.Path(__file__).resolve().parents[3]


def main():
    args = parse_args()
    repo_dir = pathlib.Path(args.repo_dir).expanduser()
    out = ensure_outside_repo(pathlib.Path(args.out_dir), REPO)
    manifest = admit_benchmark_run(out / "run_manifest.json", run_root=out, repo_dir=repo_dir)
    with finalize_run_manifest(out / "run_manifest.json", manifest) as final:
        return 0
'''

# The same launcher with both invariants honoured: the pure `assert_*` form (no mkdir) before
# admission, and the handed-in checkout as the confinement authority.
_CLEAN_LAUNCHER_SOURCE = _VIOLATING_LAUNCHER_SOURCE.replace(
    "import ensure_outside_repo", "import assert_outside_repo",
).replace(
    "out = ensure_outside_repo(pathlib.Path(args.out_dir), REPO)",
    "out = assert_outside_repo(pathlib.Path(args.out_dir), repo_dir)",
)

# INVARIANT C. A synthetic launcher that publishes its manifest from inside the seam, in the
# exact shape the real ones had: a helper named for the RECORDS it keeps, whose body happens to
# write the manifest too. The name says nothing; only the body does.
_SEAM_PUBLICATION_DEFECT_SOURCE = '''
import pathlib
from devtools.benchmarks.common.manifests import (
    admit_benchmark_run, finalize_run_manifest, write_json,
)
from devtools.benchmarks.common.run_roots import assert_outside_repo


def _write_records(run_dir, manifest, outcome):
    write_json(run_dir / "task_outcome.json", outcome)
    write_json(run_dir / "task_run_manifest.json", manifest)
    return outcome


def main():
    args = parse_args()
    repo_dir = pathlib.Path(args.repo_dir).expanduser()
    out = assert_outside_repo(pathlib.Path(args.out_dir), repo_dir)
    manifest = admit_benchmark_run(out / "run_manifest.json", run_root=out, repo_dir=repo_dir)
    with finalize_run_manifest(out / "run_manifest.json", manifest) as final:
        final["outcome"] = "completed"
        return _write_records(out, manifest, {"ok": True})
'''

# The corrected twin: the records helper keeps its OUTCOME sidecar and stops publishing the
# manifest, which the seam writes on every exit path anyway.
_SEAM_PUBLICATION_FIXED_SOURCE = _SEAM_PUBLICATION_DEFECT_SOURCE.replace(
    '    write_json(run_dir / "task_run_manifest.json", manifest)\n', "")

# The same publication with the filename moved one line up into a local — the `run_pro` shape,
# which a check that only read the call site would wave through.
_SEAM_PUBLICATION_INDIRECT_SOURCE = _SEAM_PUBLICATION_DEFECT_SOURCE.replace(
    '    write_json(run_dir / "task_run_manifest.json", manifest)',
    '    manifest_path = run_dir / "task_run_manifest.json"\n'
    '    write_json(manifest_path, manifest)')

def test_the_launcher_gate_forbids_publishing_a_manifest_inside_the_seam():
    """INVARIANT C, pinned against a violator, its corrected twin and its indirect form.

    `finalize_run_manifest` merges the terminal outcome/exit_code/refusal into the manifest when
    its context EXITS. Anything written from inside publishes a PRE-MERGE record — for a refusal,
    the admission seam's generic payload saying exit_code 1 while the process will exit 2. Two
    review rounds fixed this in `run_cu_bridge_agent` and a by-hand sweep still missed
    `run_step_agent` and `run_pro`, because the sweep asked "is there a second copy that can go
    stale?" when the hazard is "is anything published before the merge?" — true of a single-path
    launcher too. Hence a gate.

    Judged by EFFECT: the helper is called `_write_records`, the real ones `_write_task_records`
    and `_write_cu_outcome`. No name-based check finds any of the three.
    """
    from devtools.benchmarks.common import launcher_audit

    # The offending helper is not named anywhere in the gate -- resolution is the rule.
    assert "_write_records" not in launcher_audit.WRITE_PRIMITIVES
    assert not (launcher_audit.WRITE_PRIMITIVES
                & {"_write_task_records", "_write_cu_outcome", "_write_records"})

    violations = launcher_audit.audit_source(_SEAM_PUBLICATION_DEFECT_SOURCE, name="seam.py")
    assert len(violations) == 1, violations
    assert "publishes a manifest from INSIDE an active finalize_run_manifest" in violations[0]
    assert "_write_records -> write_json" in violations[0]

    # ...the same defect with the filename bound to a local one line earlier is still caught...
    indirect = launcher_audit.audit_source(_SEAM_PUBLICATION_INDIRECT_SOURCE, name="seam.py")
    assert len(indirect) == 1 and "_write_records -> write_json" in indirect[0], indirect

    # ...and the corrected twin passes, so the invariant is not simply always-red.
    assert launcher_audit.audit_source(_SEAM_PUBLICATION_FIXED_SOURCE, name="seam.py") == []

def test_every_migrated_launcher_routes_through_both_manifest_seams():
    """Fix the CLASS, not the cases: the seams are pointless if a launcher can pair
    `benchmark_run_manifest()` with its own `write_json()` again (no durable refusal) or skip the
    finalization block (no final outcome). Named files, so a new launcher cannot join silently and
    the launchers whose migration belongs to a LATER phase cannot be silently claimed."""
    # v6.76.0 promoted these three helpers out of this test module and into the shared gate;
    # this test uses that SSOT rather than keeping a second, weaker copy of the same walk.
    from devtools.benchmarks.common.launcher_audit import (
        _dotted_callee, calls_before as _calls_before,
        denied_pre_admission_call as _denied_pre_admission_call,
    )

    bench = REPO_ROOT / "devtools" / "benchmarks"
    migrated = [
        bench / "programbench" / "run_programbench.py",
        bench / "programbench" / "run_programbench_e2e.py",
        bench / "swe_bench" / "swebench_predictions.py",
        bench / "swe_bench_pro" / "pro_predictions.py",
        bench / "harness_bench_fast" / "run_harness_bench_fast.py",
        bench / "swe_bench_pro" / "e1v2" / "run_pro.py",
        bench / "swe_bench_pro" / "e1v2" / "auto_run.py",
        bench / "gaia" / "run_gaia.py",
        bench / "terminal_bench" / "run_tb.py",
        bench / "terminal_bench" / "run_harbor_smoke.py",
        bench / "continual_learning" / "run_clb.py",
        bench / "osworld" / "run_step_agent.py",
        bench / "osworld" / "run_cu_bridge_agent.py",
        bench / "osworld" / "osworld_adapter_skeleton.py",
        bench / "editbench" / "run_editbench.py",
    ]
    for path in migrated:
        source = path.read_text(encoding="utf-8")
        assert "admit_benchmark_run(" in source, f"{path.name} bypasses the admission seam"
        assert "finalize_run_manifest(" in source, f"{path.name} records no final outcome"
        assert "benchmark_run_manifest(" not in source, (
            f"{path.name} calls the builder directly again: its refusal would never be persisted"
        )
        # Python evaluates ARGUMENTS before entering the callee, so a gate called inside the
        # admission call's argument list refuses BEFORE the manifest can be written — the durable
        # refusal defeated by evaluation order. Attestation belongs after admission.
        call = source.split("admit_benchmark_run(", 1)[1].split("\n    )\n", 1)[0]
        assert "runtime_attestation(" not in call, (
            f"{path.name} evaluates runtime_attestation inside the admission argument list"
        )
        # ADMISSION IS THE OUTER BOUNDARY. Everything a launcher does before it must be argument
        # parsing and pure local derivation: no filesystem assertion, no docker, no subprocess, no
        # network, no state mutation. Walked with `ast` over the function that performs admission
        # AND, when that is not `main()`, over the statements of `main()` that precede it.
        tree = ast.parse(source)
        functions = {node.name: node for node in ast.walk(tree)
                     if isinstance(node, ast.FunctionDef)}
        owner = next(
            node.name for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and any(isinstance(inner, ast.Call)
                    and _dotted_callee(inner.func).endswith("admit_benchmark_run")
                    for inner in ast.walk(node))
        )
        prefix = _calls_before(functions[owner], "admit_benchmark_run")
        if owner != "main":
            prefix += _calls_before(functions["main"], owner)
        for dotted in prefix:
            denied = _denied_pre_admission_call(dotted)
            assert not denied, (
                f"{path.name}: {dotted}() runs BEFORE admit_benchmark_run() in {owner}() -- a "
                f"refusal there leaves no durable manifest (denied token: {denied})"
            )
    # The pending set is EMPTY on this tree: CL-Bench and the three OSWorld launchers migrated
    # in v6.76.0, GAIA and both Terminal-Bench launchers in v6.79.0. Asserted against the gate's
    # own list so the two enumerations cannot drift apart silently.
    from devtools.benchmarks.common import launcher_audit

    assert launcher_audit.PENDING_LAUNCHERS == ()
    assert sorted(path.relative_to(bench).as_posix() for path in migrated) == sorted(
        launcher_audit.MIGRATED_LAUNCHERS
    )

# One synthetic per CALL FORM a write primitive can wear. The destination model is derived from
# each primitive's real signature, so this matrix is what proves the derivation covers the forms
# rather than asserting it. The first two are the ones a reviewer found missing from the
# hand-written position table it replaced.
_SEAM_FORM_TEMPLATE = '''
import json
import os
import pathlib
import shutil
from devtools.benchmarks.common.manifests import (
    admit_benchmark_run, finalize_run_manifest, write_json, write_jsonl,
)
from devtools.benchmarks.common.run_roots import assert_outside_repo
from ouroboros.utils import atomic_write_json, write_text_atomic


def main():
    args = parse_args()
    repo_dir = pathlib.Path(args.repo_dir).expanduser()
    out = assert_outside_repo(pathlib.Path(args.out_dir), repo_dir)
    manifest = admit_benchmark_run(out / "run_manifest.json", run_root=out, repo_dir=repo_dir)
    with finalize_run_manifest(out / "run_manifest.json", manifest) as final:
        final["outcome"] = "completed"
        {statement}
        return 0
'''

_SEAM_WRITE_FORMS = (
    # (label, statement, the callee the report must name)
    ("os.rename publishes to argument ONE",
     'os.rename(tmp, out / "run_manifest.json")', "os.rename"),
    ("standalone write_text takes the path positionally",
     'write_text(out / "run_manifest.json", body)', "write_text"),
    ("standalone write_bytes takes the path positionally",
     'write_bytes(out / "run_manifest.json", blob)', "write_bytes"),
    ("receiver-style write_text names its destination as the receiver",
     '(out / "run_manifest.json").write_text(body)', "write_text"),
    ("receiver-style rename publishes to its target argument",
     'tmp.rename(out / "run_manifest.json")', "rename"),
    ("os.replace publishes to argument ONE",
     'os.replace(tmp, out / "run_manifest.json")', "os.replace"),
    ("shutil.move publishes to argument ONE",
     'shutil.move(tmp, out / "run_manifest.json")', "shutil.move"),
    ("json.dump publishes to its fp argument",
     'json.dump(manifest, open(out / "run_manifest.json", "w"))', "json.dump"),
    ("the destination may arrive as a KEYWORD",
     'write_json(path=out / "run_manifest.json", payload=manifest)', "write_json"),
    ("write_jsonl", 'write_jsonl(out / "run_manifest.json", rows)', "write_jsonl"),
    ("atomic_write_json", 'atomic_write_json(out / "run_manifest.json", manifest)',
     "atomic_write_json"),
    ("write_text_atomic", 'write_text_atomic(out / "run_manifest.json", text)',
     "write_text_atomic"),
    # ...and the local hop, which is how `run_pro` spelled it.
    ("the destination bound to a local one line earlier",
     'manifest_path = out / "run_manifest.json"\n        write_json(manifest_path, manifest)',
     "write_json"),
)

@pytest.mark.parametrize("label, statement, callee", _SEAM_WRITE_FORMS,
                         ids=[form[2] + "/" + form[0][:28] for form in _SEAM_WRITE_FORMS])
def test_invariant_c_places_the_destination_of_every_write_form(label, statement, callee):
    """Every call form a write primitive wears is caught, and the coverage is PROVEN per form.

    The first cut of Invariant C carried a hand-enumerated position table, and it was wrong in
    exactly the way hand-enumerated tables are: `rename` was mapped to argument 0 although
    `os.rename(src, dst)` publishes to argument 1, and standalone `write_text(path, ...)` had no
    positional destination at all — so an in-seam `os.rename(tmp, .../run_manifest.json)` passed
    silently. A gate whose whole subject is incomplete models of where a write goes cannot carry
    one. Destinations now come from each primitive's REAL signature, and this matrix is the proof
    that the derivation covers the forms rather than an assertion that it does.
    """
    from devtools.benchmarks.common import launcher_audit

    source = _SEAM_FORM_TEMPLATE.format(statement=statement)
    violations = launcher_audit.audit_source(source, name="form.py")
    assert len(violations) == 1, (label, violations)
    assert "publishes a manifest from INSIDE an active finalize_run_manifest" in violations[0]
    assert callee in violations[0], (label, violations[0])
    assert launcher_audit.UNRESOLVED_WRITE not in violations[0]

    # The same form writing a NON-manifest artefact is not a publication -- per form, so the
    # matrix cannot pass by being uniformly red.
    benign = launcher_audit.audit_source(
        source.replace("run_manifest.json", "task_outcome.json").replace(
            'admit_benchmark_run(out / "task_outcome.json"',
            'admit_benchmark_run(out / "run_manifest.json"').replace(
            'finalize_run_manifest(out / "task_outcome.json"',
            'finalize_run_manifest(out / "run_manifest.json"'),
        name="form.py")
    assert benign == [], (label, benign)

def test_invariant_c_derives_destinations_from_real_signatures_not_a_hand_written_table():
    """The positions come from the callable, so they cannot drift out of step with it."""
    import os

    from devtools.benchmarks.common import launcher_audit

    # Each primitive resolves to at least one REAL signature...
    for leaf in launcher_audit.WRITE_PRIMITIVES:
        assert launcher_audit.primitive_signatures(leaf), leaf

    # ...and those signatures are the live ones, not a copy. `rename` is the case in point: two
    # different callables share the name, and the union of both is what closes the hole.
    assert ("src", "dst") in {positional for positional, _every
                              in launcher_audit.primitive_signatures("rename")}
    assert ("self", "target") in {positional for positional, _every
                                  in launcher_audit.primitive_signatures("rename")}
    assert tuple(inspect.signature(os.rename).parameters)[:2] == ("src", "dst")

def test_invariant_c_fails_closed_on_a_write_form_it_cannot_place(monkeypatch):
    """An unplaceable write is REPORTED, never assumed harmless.

    A write whose destination no signature can name is the state the hand-written table was
    silently in for every form it omitted. Failing closed converts that silence into a report:
    the gate says it cannot tell, instead of saying there is nothing there.
    """
    from devtools.benchmarks.common import launcher_audit

    source = _SEAM_FORM_TEMPLATE.format(statement='write_json(out / "run_manifest.json", manifest)')
    assert launcher_audit.audit_source(source, name="closed.py")      # placed: a plain violation

    # Strip the primitive's home so nothing can place it, exactly as an unmodelled form is.
    monkeypatch.setitem(launcher_audit._PRIMITIVE_HOMES, "write_json", ())
    launcher_audit.primitive_signatures.cache_clear()
    try:
        violations = launcher_audit.audit_source(source, name="closed.py")
    finally:
        # Drop the patched answer BEFORE monkeypatch restores the table, so no later test in this
        # process sees a cached "unplaceable" verdict for a primitive that is placeable again.
        launcher_audit.primitive_signatures.cache_clear()
    assert len(violations) == 1, violations
    assert launcher_audit.UNRESOLVED_WRITE in violations[0]
    assert "no real signature places its destination" in violations[0]

def test_the_launcher_gate_does_not_confuse_a_recorded_manifest_path_with_a_publication():
    """Recording a manifest PATH in a payload is not writing to it — the vacuity guard.

    CL-Bench's `collect_results` writes `results.json` whose payload lists pointers to the
    external runner's sidecar manifests (`.../cl_bench/*/run_manifest.json`). A first cut of
    Invariant C inspected every argument of the write and reported that as a publication. Only
    the DESTINATION counts; an always-red gate is as useless as a vacuously green one.
    """
    from devtools.benchmarks.common import launcher_audit

    pointer_payload = _SEAM_PUBLICATION_FIXED_SOURCE.replace(
        '    write_json(run_dir / "task_outcome.json", outcome)',
        '    write_json(run_dir / "results.json",\n'
        '               {"sidecars": sorted(str(p) for p in run_dir.glob("*/run_manifest.json"))})')
    assert launcher_audit.audit_source(pointer_payload, name="pointers.py") == []

def test_the_launcher_gate_catches_a_synthetic_violator_of_both_invariants():
    """The gate is pinned against a launcher that BREAKS it, not only against clean ones.

    Round 6 found `ensure_outside_repo` running before admission in four launchers, and the
    guard had missed it for six rounds because it is IMPORTED: the resolver followed only local
    definitions, so an imported mutator was invisible unless somebody had thought to name it in
    the denylist. A denylist is a list of yesterday's bugs. This asserts the RESOLUTION: the
    two `ensure_*` names are NOT in the denylist, and the violation is still reported — by
    reading, one module over, what the helper's body actually does.
    """
    from devtools.benchmarks.common import launcher_audit

    assert not (launcher_audit.PRE_ADMISSION_DENIED_NAMES
                & {"ensure_outside_repo", "ensure_file_output_outside_repo"})
    assert launcher_audit.denied_pre_admission_call("ensure_outside_repo") == ""

    violations = launcher_audit.audit_source(_VIOLATING_LAUNCHER_SOURCE, name="synthetic.py")
    # INVARIANT A, caught through the imported hop and reported as `helper -> what it does`.
    assert any("BEFORE admit_benchmark_run()" in v and "ensure_outside_repo -> mkdir" in v
               for v in violations), violations
    # INVARIANT B: the run is attested against `--repo-dir` but confined against `REPO`.
    assert any("confines paths ONLY against module scope" in v and "REPO" in v
               for v in violations), violations
    assert len(violations) == 2

    # ...and the corrected launcher passes, so the gate is not simply always-red.
    assert launcher_audit.audit_source(_CLEAN_LAUNCHER_SOURCE, name="synthetic.py") == []

def test_the_launcher_gate_reproduces_both_round_six_confinement_defects():
    """Invariant B, on the two real shapes: a helper that resolves its own authority, and a
    launcher that validates its out-dir against its own checkout instead of the executed one."""
    from devtools.benchmarks.common import launcher_audit

    # The `confined_claims_dir` shape: the authority came from `repo_root_from_devtools()`, so
    # `--repo-dir /other/clone --claim-dir /other/clone/.claims` wrote lock and marker state
    # into the execution checkout.
    claims_defect = '''
from devtools.benchmarks.common.manifests import admit_benchmark_run, finalize_run_manifest
from devtools.benchmarks.common.run_roots import assert_outside_repo, repo_root_from_devtools


def confined_claims_dir(claims_dir):
    return assert_outside_repo(claims_dir, repo_root_from_devtools())


def main():
    args = parse_args()
    repo_dir = args.repo_dir
    claims = confined_claims_dir(args.claim_dir)
    manifest = admit_benchmark_run(args.out, repo_dir=repo_dir)
    with finalize_run_manifest(args.out, manifest) as final:
        return 0
'''
    violations = launcher_audit.audit_source(claims_defect, name="claims_defect.py")
    assert any("confined_claims_dir() confines paths ONLY against module scope" in v
               and "repo_root_from_devtools" in v for v in violations), violations

    # The `run_clb.main` shape: `--out-dir` validated against the launcher's own REPO, so
    # admission artefacts could land inside the execution clone being attested.
    clb_defect = '''
import pathlib
from devtools.benchmarks.common.manifests import admit_benchmark_run, finalize_run_manifest
from devtools.benchmarks.common.run_roots import assert_outside_repo

REPO = pathlib.Path(__file__).resolve().parents[3]


def main():
    args = parse_args()
    execution_clone = pathlib.Path(args.ouroboros_clone)
    out = assert_outside_repo(pathlib.Path(args.out_dir), REPO)
    manifest = admit_benchmark_run(out / "run_manifest.json", repo_dir=execution_clone)
    with finalize_run_manifest(out / "run_manifest.json", manifest) as final:
        return 0
'''
    violations = launcher_audit.audit_source(clb_defect, name="clb_defect.py")
    assert any("main() confines paths ONLY against module scope" in v and "REPO" in v
               for v in violations), violations
    # Confining against BOTH checkouts — which is what run_clb.py does now — is accepted: the
    # invariant is agreement with the attested checkout, not a ban on constants.
    fixed = clb_defect.replace(
        "    out = assert_outside_repo(pathlib.Path(args.out_dir), REPO)",
        "    out = pathlib.Path(args.out_dir)\n"
        "    for authority in (REPO, execution_clone):\n"
        "        out = assert_outside_repo(out, authority)",
    )
    assert launcher_audit.audit_source(fixed, name="clb_fixed.py") == []

def test_the_launcher_gate_leaves_static_launchers_alone():
    """A launcher that attests a STATICALLY derived root and confines against that same root is
    CONSISTENT, and flagging it would push the gate straight back toward per-case exemptions.
    The in-repo prediction writers (`swebench_predictions`, `pro_predictions`) are exactly this
    shape, and there is no other checkout for them to be wrong about."""
    from devtools.benchmarks.common import launcher_audit

    static_launcher = '''
import pathlib
from devtools.benchmarks.common.manifests import admit_benchmark_run, finalize_run_manifest
from devtools.benchmarks.common.run_roots import assert_file_output_outside_repo

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


def main():
    args = parse_args()
    output = assert_file_output_outside_repo(pathlib.Path(args.output), REPO_ROOT)
    manifest = admit_benchmark_run(args.manifest_output, repo_dir=REPO_ROOT)
    with finalize_run_manifest(args.manifest_output, manifest) as final:
        return 0
'''
    assert launcher_audit.audit_source(static_launcher, name="static.py") == []

def test_pre_admission_resolver_sees_through_helpers_and_past_step_aside_branches():
    """Pin the RESOLVER, not just its current verdict.

    Two rounds in a row, pre-admission work slipped past it by living one level down inside a
    local helper the denylist does not name (`_ensure_vmrun_on_path` probing the filesystem,
    `_install_optional_dependency_stubs` mutating `sys.modules`, `repo_provenance` shelling out
    to git, `_read_task_ids` running `uv run ... list` with a 60s timeout). So the guard is
    maintained by what a helper DOES. The complement matters too: a branch that always leaves
    the function is not on the path to admission — those are the deliberate step-aside paths
    that exist to leave no footprint — and flagging them would push the guard back toward the
    per-case exemptions it is supposed to replace.
    """
    from devtools.benchmarks.common import launcher_audit

    unit = launcher_audit._Unit(ast.parse(_GUARD_PROBE_SOURCE), "probe.py")
    prefix = launcher_audit.calls_before(unit.functions["main"], "admit_benchmark_run")

    # The helper that hides a subprocess IS caught, and the report names the helper.
    assert launcher_audit.resolve_denied("_looks_innocent", unit) == "_looks_innocent -> subprocess"
    # ...which is exactly what walking main()'s pre-admission statements now reports.
    denied = [d for d in (launcher_audit.resolve_denied(c, unit) for c in prefix) if d]
    assert denied == ["_looks_innocent -> subprocess"]
    # A pure helper is not flagged.
    assert launcher_audit.resolve_denied("_pure", unit) == ""
    # The step-aside branch (`if args.collect_only: ...; return 0`) never reaches admission, so
    # its mutating helper is not on the guarded path -- though the helper itself is still
    # recognised as mutating, so the exclusion is about the PATH, not about the denylist.
    assert "_steps_aside" not in prefix
    assert launcher_audit.resolve_denied("_steps_aside", unit) == "_steps_aside -> mkdir"
    # The branch TEST runs on the way past, so it is still walked.
    assert "parse_args" in prefix
    # TWO hops are resolved, and a hop now CROSSES MODULES — both are the round-6 fix. The old
    # guard resolved ONE hop of LOCAL definitions only, which is why an imported helper whose
    # own body called another imported helper was invisible twice over. A three-hop chain is
    # still out of the gate's reach and stays a review question; asserted so the real depth is
    # documented rather than implied.
    assert launcher_audit.resolve_denied("_two_levels_down", unit) == \
        "_two_levels_down -> _looks_innocent -> subprocess"
    assert launcher_audit.resolve_denied("_three_levels_down", unit) == ""

def test_the_gate_catches_pre_admission_reads_parses_probes_and_nested_admission_args():
    """Round 7: the gate documented a WIDER class than it enforced.

    It denied MUTATION, but the invariant it states is that nothing which can FAIL may precede
    the persisted manifest — and a run that dies parsing its dataset leaves no manifest at all,
    so it is invisible rather than merely footprint-free, which is strictly worse. Four migrated
    launchers were still doing exactly that (`_records`/`_rows` reading `--input`,
    `preflight_model_slots` reading settings, `read_csv_order`/`load_pro_rows` reading the task
    order and downloading the dataset), and a fifth shape hid in plain sight: a call nested in
    the admission call's own ARGUMENT LIST, which Python evaluates before entering the callee.

    The four shapes are pinned here as synthetic launchers, then the corrected launcher is
    asserted to PASS, so the widening cannot be satisfied by a gate that is always red.
    """
    from devtools.benchmarks.common import launcher_audit

    def audit(body, name):
        return launcher_audit.audit_source(
            "import pathlib\n"
            "from devtools.benchmarks.common.manifests import "
            "admit_benchmark_run, finalize_run_manifest\n"
            "from devtools.benchmarks.common.run_roots import assert_outside_repo\n"
            "\nREPO = pathlib.Path(__file__).resolve().parents[3]\n\n" + body,
            name=name,
        )

    # 1. A DATASET READ one hop down, the `_records`/`_rows` shape.
    read = audit('''
def _records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main():
    args = parse_args()
    rows = _records(pathlib.Path(args.input))
    manifest = admit_benchmark_run(args.out, repo_dir=REPO, requested_task_ids=rows)
    with finalize_run_manifest(args.out, manifest) as final:
        return 0
''', "read.py")
    assert any("_records() runs BEFORE" in v and "_records -> read_text" in v
               for v in read), read

    # 2. A PARSE that opens the file itself, the `read_csv_order` shape.
    parse = audit('''
def read_csv_order(path):
    with path.open(encoding="utf-8") as handle:
        return sorted(csv.DictReader(handle), key=lambda row: int(row["idx"]))


def main():
    args = parse_args()
    order = read_csv_order(pathlib.Path(args.csv))
    manifest = admit_benchmark_run(args.out, repo_dir=REPO, requested_task_ids=order)
    with finalize_run_manifest(args.out, manifest) as final:
        return 0
''', "parse.py")
    assert any("read_csv_order -> open" in v for v in parse), parse

    # 3. A MODEL-SLOT PROBE that reads settings and refuses, the `preflight_model_slots` shape.
    #    Reported by the read; the refusal is what made it fatal.
    probe = audit('''
def preflight_model_slots(settings_path):
    settings = json.loads(pathlib.Path(settings_path).read_text(encoding="utf-8"))
    if not settings:
        raise SystemExit("model slot preflight failed")
    return settings


def main():
    args = parse_args()
    slots = preflight_model_slots(args.settings)
    manifest = admit_benchmark_run(args.out, repo_dir=REPO, harness=slots)
    with finalize_run_manifest(args.out, manifest) as final:
        return 0
''', "probe.py")
    assert any("preflight_model_slots -> read_text" in v for v in probe), probe

    # 4. A CALL NESTED IN THE ADMISSION ARGUMENTS, the `_collect_attestations` shape. Evaluated
    #    before `admit_benchmark_run` is even entered, and previously invisible because the
    #    walk STOPPED at the statement holding the admission call.
    nested = audit('''
def _collect_attestations(paths):
    return [json.loads(pathlib.Path(raw).read_text(encoding="utf-8")) for raw in paths]


def main():
    args = parse_args()
    manifest = admit_benchmark_run(
        args.out, repo_dir=REPO,
        extra={"runtime_attestations": _collect_attestations(args.attestation)},
    )
    with finalize_run_manifest(args.out, manifest) as final:
        return 0
''', "nested.py")
    assert any("_collect_attestations() runs BEFORE" in v and "read_text" in v
               for v in nested), nested

    # 5. A DEFERRED NON-STDLIB IMPORT, the `load_pro_rows`/`_load_instances` shape. Not a call
    #    at all, so no callee-name rule could ever have seen it; its ImportError (or an offline
    #    hub) killed the process with nothing on disk.
    dataset = audit('''
def load_pro_rows(ids):
    from datasets import load_dataset
    return load_dataset("ScaleAI/SWE-bench_Pro", split="test")


def main():
    args = parse_args()
    rows = load_pro_rows(args.ids)
    manifest = admit_benchmark_run(args.out, repo_dir=REPO, requested_task_ids=rows)
    with finalize_run_manifest(args.out, manifest) as final:
        return 0
''', "dataset.py")
    assert any("load_pro_rows -> deferred import datasets" in v for v in dataset), dataset

    # THE CORRECTED SHAPE PASSES. Declared selector at admission, resolved ids amended after —
    # the chicken-and-egg has one answer and this is it.
    fixed = audit('''
def _records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main():
    args = parse_args()
    manifest = admit_benchmark_run(
        args.out, repo_dir=REPO, requested_task_ids=[], extra={"input": str(args.input)},
    )
    with finalize_run_manifest(args.out, manifest) as final:
        rows = _records(pathlib.Path(args.input))
        manifest["requested_task_ids"] = [row["instance_id"] for row in rows]
        manifest["requested_count"] = len(rows)
        return 0
''', "fixed.py")
    assert fixed == [], fixed

def test_the_gate_separates_argv_shaped_refusals_from_state_shaped_ones():
    """Where the widened invariant draws its line, pinned so it is not re-litigated.

    Argument parsing and pure path arithmetic MUST precede admission — they compute the
    manifest's own path — and their refusals are a deterministic function of argv. A bare
    existence probe is the one permitted middle: it reads no content, cannot fail on malformed
    input, and is what lets `scored_claim_state` answer "another lane already scored this" and
    step aside leaving zero footprint. The combination is what is denied: a helper that PROBES
    and can also REFUSE produces a refusal no argv can explain, which is exactly the class that
    needs a durable manifest.
    """
    from devtools.benchmarks.common import launcher_audit

    source = '''
import pathlib


def refuse_live_repo_clone(clone):
    resolved = pathlib.Path(clone).expanduser().resolve(strict=False)
    if resolved == LIVE:
        raise SystemExit("--ouroboros-clone must never be the live repo")
    return resolved


def scored_claim_state(claims_dir, key):
    if (claims_dir / f"{key}.scored").exists():
        return "already_scored"
    return ""


def check_clone(clone):
    if not (clone / "devtools").exists():
        raise SystemExit("not an Ouroboros checkout")
'''
    unit = launcher_audit._Unit(ast.parse(source), "line.py")
    # Pure-argv refusal: allowed before admission.
    assert launcher_audit.resolve_denied("refuse_live_repo_clone", unit) == ""
    # Probe that only RETURNS: allowed, and this is deliberate, not an oversight.
    assert launcher_audit.resolve_denied("scored_claim_state", unit) == ""
    # Probe + refusal: denied.
    assert launcher_audit.resolve_denied("check_clone", unit) == \
        "check_clone -> refuses on probed state"
    # The probe names are recognised, and none of them is denied on its own.
    assert "exists" in launcher_audit.STATE_PROBE_NAMES
    assert not (launcher_audit.STATE_PROBE_NAMES & launcher_audit.PRE_ADMISSION_DENIED_NAMES)
    # A stdlib deferred import is not a dependency on the state of the world.
    assert launcher_audit.resolve_denied("_is_default_desktop_server", launcher_audit._Unit(
        ast.parse('''
def _is_default_desktop_server(url):
    from urllib.parse import urlparse
    return urlparse(url).port == 8765
'''), "stdlib.py")) == ""

def test_the_gate_catches_a_refusal_authority_derived_from___file__():
    """Invariant B's second shape, found by a live CL-Bench smoke rather than by review.

    `run_clb.refuse_live_repo_clone` compared `--ouroboros-clone` against `REPO`, a
    `__file__`-derived module constant, so running a PINNED SEED's own launcher and handing it
    that same seed — the recipe METHODOLOGY prescribes — was refused, while the live repo the
    guard exists to protect went unmentioned. The two trees coincide only in the development
    workspace. Same class as the `confined_claims_dir` finding, different syntax (a comparison
    rather than a call), which is why the call-shaped detector missed it.
    """
    from devtools.benchmarks.common import launcher_audit

    defect = '''
import pathlib
from devtools.benchmarks.common.manifests import admit_benchmark_run, finalize_run_manifest
from devtools.benchmarks.common.run_roots import assert_outside_repo

REPO = pathlib.Path(__file__).resolve().parents[3]


def refuse_live_repo_clone(clone):
    resolved = pathlib.Path(clone).expanduser().resolve(strict=False)
    if resolved == REPO.resolve(strict=False):
        raise SystemExit("--ouroboros-clone must be a dedicated CLONE, never the live repo")
    return resolved


def main():
    args = parse_args()
    execution_clone = refuse_live_repo_clone(pathlib.Path(args.ouroboros_clone))
    out = assert_outside_repo(pathlib.Path(args.out_dir), execution_clone)
    manifest = admit_benchmark_run(out / "run_manifest.json", repo_dir=execution_clone)
    with finalize_run_manifest(out / "run_manifest.json", manifest) as final:
        return 0
'''
    violations = launcher_audit.audit_source(defect, name="refusal_defect.py")
    assert any("refuse_live_repo_clone() REFUSES against ['REPO']" in v
               and "__file__" in v for v in violations), violations

    # Refusing against the LIVE runtime instead — what run_clb.py does now — passes.
    fixed = defect.replace(
        "    if resolved == REPO.resolve(strict=False):\n"
        '        raise SystemExit("--ouroboros-clone must be a dedicated CLONE, never the live repo")',
        "    for live in live_repo_roots():\n"
        "        if resolved == live.expanduser().resolve(strict=False):\n"
        '            raise SystemExit("--ouroboros-clone must never be the LIVE repo")',
    )
    assert launcher_audit.audit_source(fixed, name="refusal_fixed.py") == []

def test_the_gate_resolves_imported_first_party_helpers_only():
    """The resolver opens FIRST-PARTY modules only. Stdlib and third-party callees stay
    unresolved (the gate must not depend on what happens to be installed) and are covered by
    the name/prefix denylist instead."""
    from devtools.benchmarks.common import launcher_audit

    source = '''
from devtools.benchmarks.common.run_roots import (
    assert_outside_repo, ensure_file_output_outside_repo, ensure_outside_repo,
)
from json import dumps
import shutil


def _wrapper(path, repo):
    from devtools.benchmarks.common.manifests import write_json
    return write_json(path, {})
'''
    unit = launcher_audit._Unit(ast.parse(source), "imports.py")
    assert unit.imports["ensure_outside_repo"] == "devtools.benchmarks.common.run_roots"
    # A first-party import is opened and its body read: BOTH `ensure_*` helpers are caught by
    # what they do, one and two modules-hops away, with neither of them in the denylist.
    assert launcher_audit.resolve_denied("ensure_outside_repo", unit) == \
        "ensure_outside_repo -> mkdir"
    assert launcher_audit.resolve_denied("ensure_file_output_outside_repo", unit) == \
        "ensure_file_output_outside_repo -> ensure_outside_repo -> mkdir"
    # The pure `assert_*` form is what a pre-admission caller must use, and it is NOT flagged.
    assert launcher_audit.resolve_denied("assert_outside_repo", unit) == ""
    # A FUNCTION-LEVEL import is in the map too — the OSWorld launchers import their shared
    # claim helpers inside the functions that use them, and an import the resolver cannot see
    # is an imported mutator it cannot follow.
    assert unit.imports["write_json"] == "devtools.benchmarks.common.manifests"
    # A stdlib import is not opened; nothing is claimed about it.
    assert launcher_audit.resolve_denied("dumps", unit) == ""
    # ...but the name/prefix denylist still covers third-party mutators without resolving them:
    # the name hit wins when there is one, and the prefix catches whole families.
    assert launcher_audit.denied_pre_admission_call("shutil.rmtree") == "rmtree"
    assert launcher_audit.denied_pre_admission_call("shutil.copytree") == "shutil"
    assert launcher_audit.denied_pre_admission_call("docker_pull_if_missing") == \
        "docker_pull_if_missing"

def test_every_migrated_launcher_passes_the_structural_gate():
    """THE GATE. Every launcher under the admission contract, both invariants, one report.

    Fix the CLASS, not the cases. Six review rounds produced eighteen criticals whose per-round
    count went UP, because each round patched the call sites it happened to find. This answers
    the question for the whole family at once, and a launcher that joins the family later joins
    the gate with it. The seams themselves are pointless if a launcher can pair
    `benchmark_run_manifest()` with its own `write_json()` again (no durable refusal) or skip
    the finalization block (no final outcome), so those are checked here too.
    """
    from devtools.benchmarks.common import launcher_audit

    assert launcher_audit.audit_all_launchers() == []
    # Named files, so a new launcher cannot join silently and the launchers whose migration
    # belongs to a LATER phase cannot be silently claimed.
    for path in launcher_audit.launcher_paths():
        assert path.is_file(), path
    for rel in launcher_audit.PENDING_LAUNCHERS:
        source = (launcher_audit.BENCH_ROOT / rel).read_text(encoding="utf-8")
        assert "benchmark_run_manifest(" in source
