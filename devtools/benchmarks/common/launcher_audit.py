"""ONE deterministic structural gate over EVERY benchmark launcher, for three invariants.

Six review rounds found the same mistakes in four different launchers each, so the answer is not
more patches but a gate answering the question for the whole family at once. It reads sources
with ``ast`` — no imports, no execution, no network — cheap enough to run as an ordinary test and
deterministic enough to be a release gate.

INVARIANT A — ADMISSION IS THE OUTER BOUNDARY.
    Nothing that can FAIL may happen before ``admit_benchmark_run()`` has PERSISTED its
    manifest. Not just mutation: a run that dies while reading its dataset leaves no manifest at
    all, so it is INVISIBLE rather than merely footprint-free — strictly worse for a provenance
    system. Enforced by walking the statements preceding the admission call, INCLUDING that
    call's own argument list (Python evaluates arguments first, so a probe nested there runs
    while nothing is on disk), denying every callee whose EFFECT is a mutation, a content
    read/parse, a network reach, a deferred non-stdlib import, or a refusal depending on probed
    state.

    The line it draws, stated once so it is not re-litigated per launcher: ARGUMENT-SHAPED work
    may precede admission, WORLD-SHAPED work may not. Argv parsing and pure path arithmetic are a
    deterministic function of the command line and diagnosable from it; reading a file, importing
    a dataset library or asking the network are not, so those refusals are exactly the ones
    needing a durable record. A bare EXISTENCE probe is the one permitted middle, and becomes a
    violation the moment the helper holding it can RAISE.

    One ownership primitive may precede admission: the canonical host-local
    campaign lock. It exists specifically so a losing launcher cannot race and
    overwrite the winning launcher's manifest. The exemption is resolved to one
    exact first-party module and is shape-checked: hashing the absolute run-root,
    opening the host temp lock, and flocking it are allowed; any additional
    denied read, mutation, network call, or deferred dependency revokes it.

INVARIANT B — CONFINEMENT IS COMPUTED FROM THE ACTIVE CHECKOUT.
    A launcher whose provenance is attested against a checkout it was HANDED (``--repo-dir``,
    ``--ouroboros-clone``) must confine its output paths against that same checkout, never a
    statically derived repo root, or lock, marker and admission artefacts land inside the very
    seed being attested. Attesting a STATIC root and confining against that same root AGREES and
    is not flagged: the invariant is agreement, not a ban on constants.

    A second shape needs its own detector: a REFUSAL comparing a path against a
    ``__file__``-derived module root coincides with "am I pointed at the LIVE repo" only in the
    development layout — inside a pinned seed it refuses the documented recipe while leaving the
    live repo unprotected. "Wherever I execute from" is never the authority; the live runtime, or
    the handed checkout, is.

INVARIANT C — THE FINALIZATION SEAM'S EXIT IS THE ONLY PUBLISHER.
    ``finalize_run_manifest`` merges the terminal ``outcome``/``exit_code``/``refusal`` into the
    manifest when its context EXITS. A launcher writing a manifest from INSIDE that context
    publishes a PRE-MERGE record: for a refusal, the admission seam's generic payload saying
    ``exit_code`` 1 while the process will really exit 2. The final artefact is corrected a
    moment later — which is what let this survive two review rounds — but the intermediate state
    is observable (OSWorld runs multi-lane, and the canonical per-task manifest exists precisely
    to serve a concurrent reader) and an interruption inside the window leaves the wrong record
    durably. Every such write is also redundant: the seam writes the same path on EVERY exit
    path. A by-hand sweep missed one launcher by asking "is there a second copy that can go
    stale?" when the hazard is "is anything published before the merge?" — true of a single-path
    launcher too. Hence a gate, not a sweep.

    Judged by EFFECT, like A: a call is a publication when following its body (local or
    imported) reaches a write primitive whose DESTINATION names a run-manifest artefact. The
    offenders are ``_write_task_records`` and ``_write_cu_outcome`` — named for the records they
    keep, not for the manifest they published — so a name-based check finds neither, and would
    miss the next helper wrapping ``write_json``.

Names are a list of yesterday's bugs; RESOLUTION is the rule, and it crosses module boundaries:
``ensure_outside_repo``, which mkdirs the directory it validates, is caught by what its body
DOES, not by anyone remembering to name it. All three read source text, so a SYNTHETIC violating
launcher goes through the same entry point (``audit_source``) as the real ones — a gate passing
only because today's code is clean tells us nothing about tomorrow's.
"""

from __future__ import annotations

import ast
import functools
import importlib
import inspect
import pathlib
import sys
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
BENCH_ROOT = REPO_ROOT / "devtools" / "benchmarks"

ADMISSION_CALLEE = "admit_benchmark_run"

# Launchers under the admission contract. Named, so a new launcher cannot join silently and the
# ones whose migration belongs to a LATER phase cannot be silently claimed.
MIGRATED_LAUNCHERS: tuple[str, ...] = (
    "programbench/run_programbench.py",
    "programbench/run_programbench_e2e.py",
    "swe_bench/swebench_predictions.py",
    "swe_bench_pro/pro_predictions.py",
    "harness_bench_fast/run_harness_bench_fast.py",
    "swe_bench_pro/e1v2/run_pro.py",
    "swe_bench_pro/e1v2/auto_run.py",
    "continual_learning/run_clb.py",
    "cybergym/run_cybergym.py",
    "osworld/run_step_agent.py",
    "osworld/run_cu_bridge_agent.py",
    "osworld/osworld_adapter_skeleton.py",
    "gaia/run_gaia.py",
    "terminal_bench/run_tb.py",
    "terminal_bench/run_harbor_smoke.py",
    "editbench/run_editbench.py",
)

# GAIA and both Terminal-Bench launchers were the last pre-seam holdouts and migrated in
# v6.79.0, so the pending set is now EMPTY. It stays declared: a launcher added later that is
# not yet under the contract belongs here, named, rather than simply missing from both lists.
PENDING_LAUNCHERS: tuple[str, ...] = ()


def launcher_paths() -> list[pathlib.Path]:
    return [BENCH_ROOT / rel for rel in MIGRATED_LAUNCHERS]


# --------------------------------------------------------------------------- #
# Invariant A: the pre-admission denylist and its cross-module resolver
# --------------------------------------------------------------------------- #

PRE_ADMISSION_DENIED_NAMES = frozenset({
    # Deliberately NOT here: `ensure_outside_repo` / `ensure_file_output_outside_repo`, which
    # mutate (mkdir what they validate) AND are imported -- the combination that defeated the
    # old local-only resolver. Leaving them unnamed is the point: the resolver reads their
    # bodies, so the NEXT imported mutator nobody enumerated is caught the same way.
    "assert_seed_is_git_directory", "ensure_util_image", "dump_state", "runtime_attestation",
    "snapshot", "restore", "reflections", "seed_stamp", "run_one", "run_instance",
    "preflight_cleanroom_container", "prepare_seeded_workspace", "create_submission_tarball",
    "run_official_eval", "write_json", "write_jsonl", "write_result_index", "append_result_index",
    "write_text", "write_bytes", "mkdir", "unlink", "rmtree", "touch", "rename", "chmod",
    "urlopen", "urlretrieve", "Popen", "check_output", "check_call",
    # READS AND PARSES. None of these mutates anything, and every one can take the run down
    # before a manifest exists. `read_text` one hop inside a `_rows`-style helper is how four
    # launchers still did dataset discovery outside the boundary they claimed to enforce.
    "read_text", "read_bytes", "read_json", "read_jsonl", "open", "load", "load_dataset",
    "iterdir", "glob", "rglob", "walk", "listdir", "scandir",
})
# Matched as a prefix on any dotted segment, so `docker_pull_if_missing` / `subprocess.run` /
# `shutil.rmtree` / `requests.get` are all caught without enumerating them.
PRE_ADMISSION_DENIED_PREFIXES = ("subprocess", "docker", "shutil", "requests", "httpx",
                                 "aiohttp", "socket")

# EXISTENCE probes. Not denied alone — they read no content and cannot fail on malformed input,
# which is what makes a zero-footprint step-aside legitimate. They ARE denied inside a helper
# that can raise (see `_helper_effects`): probing may not refuse; refusing may not probe.
STATE_PROBE_NAMES = frozenset({
    "exists", "is_file", "is_dir", "is_symlink", "is_mount", "stat", "lstat", "samefile",
})

# Modules the resolver will open. First-party only: stdlib and third-party callees stay
# unresolved and are covered by the name/prefix denylist, which keeps the gate hermetic.
RESOLVABLE_PACKAGES = ("devtools.", "ouroboros.")
# A function-level import inside a pre-admission helper is a deferred dependency, deferred
# because it is heavy or optional, and its ImportError is a pre-manifest death. Stdlib and
# first-party imports are exempt: those are a style choice, not a dependency on the world.
_STDLIB_MODULES = frozenset(getattr(sys, "stdlib_module_names", ()))
_FIRST_PARTY_ROOTS = frozenset(package.rstrip(".") for package in RESOLVABLE_PACKAGES)
_PRE_ADMISSION_LOCK_MODULE = (
    "devtools.benchmarks.cybergym.cybergym_result_index"
)
_PRE_ADMISSION_LOCK_NAME = "acquire_campaign_execution_lock"
_PRE_ADMISSION_LOCK_REQUIRED_CALLS = frozenset({
    "encode", "flock", "gettempdir", "open", "sha256",
})


def _dotted_callee(node: ast.expr) -> str:
    parts: list[str] = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


def denied_pre_admission_call(dotted: str) -> str:
    """The denied token in ``dotted``, or ``""``. Pure name/prefix matching, no resolution."""
    segments = dotted.split(".")
    hits = PRE_ADMISSION_DENIED_NAMES.intersection(segments)
    if hits:
        return sorted(hits)[0]
    for segment in segments:
        for prefix in PRE_ADMISSION_DENIED_PREFIXES:
            if segment.startswith(prefix):
                return segment
    return ""


def _terminates(statement: ast.stmt) -> bool:
    """True if ``statement`` is a branch that ALWAYS leaves the function (return/raise).

    Calls inside such a branch are not on the path to admission: they are the deliberate
    step-aside paths that exist precisely to leave NO footprint and have no run to record
    against. The branch's TEST expression is still walked, because that runs on the way past.
    """
    if isinstance(statement, (ast.Return, ast.Raise)):
        return True
    if isinstance(statement, ast.If) and statement.body:
        return _terminates(statement.body[-1]) and not statement.orelse
    return False


def _walk_calls(statement: ast.stmt) -> list[str]:
    """Dotted callees of ``statement``, skipping the bodies of always-terminating branches."""
    if isinstance(statement, ast.If) and _terminates(statement):
        return [_dotted_callee(node.func) for node in ast.walk(statement.test)
                if isinstance(node, ast.Call)]
    return [_dotted_callee(node.func) for node in ast.walk(statement)
            if isinstance(node, ast.Call)]


def _admission_argument_calls(statement: ast.stmt, stop_callee: str) -> list[str]:
    """Dotted callees evaluated INSIDE the ``stop_callee(...)`` argument list.

    Python evaluates arguments before entering the callee, so anything nested here runs while the
    manifest is still unwritten — pre-admission work that merely LOOKS like admission, which is
    where `pro_predictions` read every ``--attestation`` file. Only the argument subtrees are
    walked; the rest of the statement is genuinely post-admission.
    """
    found: list[str] = []
    for node in ast.walk(statement):
        if not (isinstance(node, ast.Call) and _dotted_callee(node.func).endswith(stop_callee)):
            continue
        for argument in [*node.args, *(keyword.value for keyword in node.keywords)]:
            found.extend(_dotted_callee(inner.func) for inner in ast.walk(argument)
                         if isinstance(inner, ast.Call))
    return found


def calls_before(function: ast.FunctionDef, stop_callee: str) -> list[str]:
    """Dotted callee names in the statements of ``function`` that PRECEDE ``stop_callee``.

    "Precede" includes the admission call's own arguments, which are evaluated first.
    """
    seen: list[str] = []
    for statement in function.body:
        if any(isinstance(node, ast.Call) and _dotted_callee(node.func).endswith(stop_callee)
               for node in ast.walk(statement)):
            return seen + _admission_argument_calls(statement, stop_callee)
        seen.extend(_walk_calls(statement))
    raise LauncherAuditError(f"{function.name}() never calls {stop_callee}")


class LauncherAuditError(RuntimeError):
    """The audited source does not have the shape the gate can reason about."""


class _Unit:
    """One parsed module: its function definitions and the modules its names came from."""

    __slots__ = ("tree", "functions", "imports", "module_assigns", "import_bound", "name",
                 "file_roots")

    def __init__(self, tree: ast.Module, name: str) -> None:
        self.tree = tree
        self.name = name
        self.functions: dict[str, ast.FunctionDef] = {
            node.name: node for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }  # type: ignore[misc]
        # Function-level imports count: an import the resolver cannot see is an imported mutator
        # it cannot follow.
        self.imports: dict[str, str] = {}
        self.import_bound: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and not node.level:
                for alias in node.names:
                    bound = alias.asname or alias.name
                    self.import_bound.add(bound)
                    self.imports[bound] = node.module
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    self.import_bound.add((alias.asname or alias.name).split(".")[0])
        self.module_assigns: set[str] = set()
        # Module constants derived from ``__file__``: "wherever this file happens to live".
        # Invariant B's second shape reports one of these being used as a REFUSAL authority.
        self.file_roots: set[str] = set()
        for statement in tree.body:
            targets = (statement.targets if isinstance(statement, ast.Assign)
                       else [statement.target] if isinstance(statement, ast.AnnAssign) else [])
            from_file = any(isinstance(node, ast.Name) and node.id == "__file__"
                            for node in ast.walk(statement))
            for target in targets:
                for node in ast.walk(target):
                    if isinstance(node, ast.Name):
                        self.module_assigns.add(node.id)
                        if from_file:
                            self.file_roots.add(node.id)


@functools.lru_cache(maxsize=None)
def _unit_for_module(module: str) -> _Unit | None:
    if not module.startswith(RESOLVABLE_PACKAGES):
        return None
    path = REPO_ROOT.joinpath(*module.split("."))
    for candidate in (path.with_suffix(".py"), path / "__init__.py"):
        if candidate.is_file():
            return _Unit(ast.parse(candidate.read_text(encoding="utf-8")), module)
    return None


def _imported_modules(node: ast.Import | ast.ImportFrom) -> list[str]:
    if isinstance(node, ast.ImportFrom):
        return [node.module] if node.module and not node.level else []
    return [alias.name for alias in node.names]


def _raises(function: ast.FunctionDef) -> bool:
    """True if ``function``'s own body can refuse (``raise`` / ``sys.exit``)."""
    for node in ast.walk(function):
        if isinstance(node, ast.Raise):
            return True
        if isinstance(node, ast.Call) and _dotted_callee(node.func).split(".")[-1] == "exit":
            return True
    return False


def _helper_effects(function: ast.FunctionDef) -> str:
    """Effects of a helper BODY that no callee name can express, or ``""``.

    Two of them:

    * a DEFERRED non-stdlib import — a dataset library whose ``ImportError`` kills the process
      while nothing is on disk. Name-based resolution is blind: it is not a call at all.
    * a REFUSAL THAT DEPENDS ON PROBED STATE — the helper both raises and asks the filesystem
      whether something exists. Either alone is fine pre-admission; together they are a refusal
      no argv can explain, precisely the class needing a durable manifest.
    """
    for node in ast.walk(function):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        for module in _imported_modules(node):
            root = module.split(".")[0]
            if root in _STDLIB_MODULES or root in _FIRST_PARTY_ROOTS:
                continue
            return f"deferred import {module}"
    if _raises(function) and any(
            isinstance(node, ast.Call)
            and _dotted_callee(node.func).split(".")[-1] in STATE_PROBE_NAMES
            for node in ast.walk(function)):
        return "refuses on probed state"
    return ""


def resolve_denied(dotted: str, unit: _Unit, *, depth: int = 2) -> str:
    """Denied token for ``dotted``, resolving ONE hop through helper bodies.

    The hop crosses module boundaries: a local callee resolves from this module's source, an
    IMPORTED first-party one from THAT module's — which is what `ensure_outside_repo` needed and
    never got. Reported as ``helper -> token`` so the hop is named, not just the leaf. ONE hop BY
    DESIGN, asserted in the tests: a two-hop chain is out of reach and stays a review question.
    """
    direct = denied_pre_admission_call(dotted)
    if direct:
        return direct
    if depth <= 0:
        return ""
    leaf = dotted.split(".")[-1]
    target = unit.functions.get(leaf)
    target_unit = unit
    if target is None:
        imported = _unit_for_module(unit.imports.get(leaf, ""))
        if imported is None:
            return ""
        target = imported.functions.get(leaf)
        target_unit = imported
        if target is None:
            return ""
    # NOTE the confinement primitives get NO exemption. `assert_outside_repo` clears the effect
    # check on its own merits (pure path arithmetic, probes nothing); `ensure_outside_repo` must
    # stay caught by the `mkdir` in its body, and a name-keyed exemption would hand it back.
    effect = _helper_effects(target)
    if effect:
        return f"{target.name} -> {effect}"
    for node in ast.walk(target):
        if not isinstance(node, ast.Call):
            continue
        inner = resolve_denied(_dotted_callee(node.func), target_unit, depth=depth - 1)
        if inner:
            return f"{target.name} -> {inner}"
    return ""


def _safe_pre_admission_lock_helper(target: ast.FunctionDef, unit: _Unit) -> bool:
    """Validate the one exact temp-lock implementation, including its sole open."""
    if _helper_effects(target):
        return False
    call_nodes = [node for node in ast.walk(target) if isinstance(node, ast.Call)]
    calls = {_dotted_callee(node.func).split(".")[-1] for node in call_nodes}
    if not _PRE_ADMISSION_LOCK_REQUIRED_CALLS.issubset(calls):
        return False
    open_calls = [
        node for node in call_nodes
        if _dotted_callee(node.func).split(".")[-1] == "open"
    ]
    if len(open_calls) != 1 or not isinstance(open_calls[0].func, ast.Attribute):
        return False
    receiver = open_calls[0].func.value
    receiver_calls = {
        _dotted_callee(node.func).split(".")[-1]
        for node in ast.walk(receiver)
        if isinstance(node, ast.Call)
    }
    receiver_names = {
        node.id for node in ast.walk(receiver) if isinstance(node, ast.Name)
    }
    if "gettempdir" not in receiver_calls or "root_digest" not in receiver_names:
        return False
    for node in call_nodes:
        denied = resolve_denied(_dotted_callee(node.func), unit, depth=1)
        if denied and node is not open_calls[0]:
            return False
    return True


def _approved_pre_admission_lock(dotted: str, unit: _Unit) -> bool:
    """Whether ``dotted`` resolves unshadowed to the canonical lock helper."""
    leaf = dotted.split(".")[-1]
    if leaf != _PRE_ADMISSION_LOCK_NAME or unit.imports.get(leaf) != _PRE_ADMISSION_LOCK_MODULE:
        return False
    if leaf in unit.functions or any(
        (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id == leaf)
        or (isinstance(node, ast.arg) and node.arg == leaf)
        for node in ast.walk(unit.tree)
    ):
        return False
    imported = _unit_for_module(_PRE_ADMISSION_LOCK_MODULE)
    target = imported.functions.get(leaf) if imported is not None else None
    return bool(imported is not None and target is not None
                and _safe_pre_admission_lock_helper(target, imported))


def _pre_admission_violations(unit: _Unit) -> list[str]:
    owners = [node.name for node in ast.walk(unit.tree)
              if isinstance(node, ast.FunctionDef)
              and any(isinstance(inner, ast.Call)
                      and _dotted_callee(inner.func).endswith(ADMISSION_CALLEE)
                      for inner in ast.walk(node))]
    if not owners:
        return [f"{unit.name}: bypasses the admission seam ({ADMISSION_CALLEE} is never called)"]
    owner = owners[0]
    prefix = calls_before(unit.functions[owner], ADMISSION_CALLEE)
    if owner != "main" and "main" in unit.functions:
        prefix += calls_before(unit.functions["main"], owner)
    violations: list[str] = []
    for dotted in prefix:
        if _approved_pre_admission_lock(dotted, unit):
            continue
        denied = resolve_denied(dotted, unit)
        if denied:
            violations.append(
                f"{unit.name}: {dotted}() runs BEFORE {ADMISSION_CALLEE}() in {owner}() -- a "
                f"refusal there leaves no durable manifest (denied token: {denied})"
            )
    return violations


# --------------------------------------------------------------------------- #
# Invariant B: confinement authority must come from the attested checkout
# --------------------------------------------------------------------------- #

# The path-confinement primitives. ``_outside`` is the OSWorld skeleton's local form of the same
# predicate; naming it here keeps that launcher inside the invariant instead of outside it.
CONFINEMENT_PRIMITIVES = frozenset({
    "assert_outside_repo", "ensure_outside_repo",
    "assert_file_output_outside_repo", "ensure_file_output_outside_repo",
    "_outside",
})
# Helpers deriving a repo root from the module's own location: the same value whatever checkout
# is executing, which is why it cannot be the sole authority for a launcher handed one.
STATIC_ROOT_HELPERS = frozenset({
    "repo_root_from_devtools", "workspace_root_from_devtools",
})


def _local_assigns(function: ast.FunctionDef) -> dict[str, list[ast.expr]]:
    """Name -> the expressions it is bound from, inside one function (assignments and loops)."""
    bound: dict[str, list[ast.expr]] = {}

    def _add(target: ast.expr, value: ast.expr) -> None:
        for node in ast.walk(target):
            if isinstance(node, ast.Name):
                bound.setdefault(node.id, []).append(value)

    for node in ast.walk(function):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                _add(target, node.value)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)) and node.value is not None:
            _add(node.target, node.value)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            _add(node.target, node.iter)
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            _add(node.optional_vars, node.context_expr)
    return bound


def _authority_tokens(expr: ast.expr, unit: _Unit,
                      assigns: dict[str, list[ast.expr]],
                      seen: tuple[str, ...] = ()) -> set[tuple[str, str]]:
    """Classify every value the authority expression is built from as ``static`` or ``dyn``.

    ``static`` is module scope: a module-level constant, or a helper deriving a root from
    ``__file__``. ``dyn`` came from the function's own inputs. Names bound by imports are modules
    and functions, not authorities, so they are skipped.
    """
    tokens: set[tuple[str, str]] = set()
    for node in ast.walk(expr):
        if isinstance(node, ast.Call):
            # `ast.walk` continues into the arguments regardless: a wrapped authority
            # (`Path(args.repo_dir).resolve()`) must still contribute its inner names.
            callee = _dotted_callee(node.func).split(".")[-1]
            if callee in STATIC_ROOT_HELPERS:
                tokens.add(("static", callee))
        if not isinstance(node, ast.Name):
            continue
        name = node.id
        if name in STATIC_ROOT_HELPERS or name in unit.import_bound or name in unit.functions:
            continue
        if name in assigns and name not in seen:
            inner: set[tuple[str, str]] = set()
            for value in assigns[name]:
                inner |= _authority_tokens(value, unit, assigns, seen + (name,))
            # A local resolving to nothing recognisable is still input-derived, not a constant:
            # report it dynamic rather than collapsing to "no tokens", which a vacuous all()
            # would read as static.
            tokens |= inner or {("dyn", name)}
        elif name in unit.module_assigns:
            tokens.add(("static", name))
        else:
            tokens.add(("dyn", name))
    return tokens


def _authority_arg(call: ast.Call) -> ast.expr | None:
    for keyword in call.keywords:
        if keyword.arg in ("repo_dir", "repo_root", "forbidden"):
            return keyword.value
    return call.args[1] if len(call.args) > 1 else None


def _confinement_calls(function: ast.FunctionDef) -> list[ast.Call]:
    return [node for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and _dotted_callee(node.func).split(".")[-1] in CONFINEMENT_PRIMITIVES]


def _attested_authority(unit: _Unit) -> set[tuple[str, str]]:
    """Tokens of the ``repo_dir=`` the launcher attests in its admission call."""
    for function in unit.functions.values():
        for node in ast.walk(function):
            if not (isinstance(node, ast.Call)
                    and _dotted_callee(node.func).endswith(ADMISSION_CALLEE)):
                continue
            for keyword in node.keywords:
                if keyword.arg == "repo_dir":
                    return _authority_tokens(keyword.value, unit, _local_assigns(function))
    return set()


def _refusal_authority_violations(unit: _Unit) -> list[str]:
    """Refusals whose authority is a ``__file__``-derived module root.

    ``if <path> == REPO: raise`` asks "am I pointed at the tree I execute from?", which matches
    "at the LIVE repo" only when running from the live workspace. Same class as the
    ``confined_claims_dir`` finding, different syntax, hence its own detector.
    """
    violations: list[str] = []
    for name, function in sorted(unit.functions.items()):
        for node in ast.walk(function):
            if not (isinstance(node, ast.If)
                    and any(isinstance(inner, ast.Raise) for inner in ast.walk(node))):
                continue
            roots = sorted({inner.id for inner in ast.walk(node.test)
                            if isinstance(inner, ast.Name) and inner.id in unit.file_roots})
            if roots:
                violations.append(
                    f"{unit.name}: {name}() REFUSES against {roots} -- a module root derived "
                    f"from __file__, i.e. the tree this launcher happens to execute from. "
                    f"Refuse against the LIVE runtime (or the handed checkout) instead: the "
                    f"two coincide only in the development layout"
                )
    return violations


def _confinement_violations(unit: _Unit) -> list[str]:
    attested = _attested_authority(unit)
    if not any(kind == "dyn" for kind, _ in attested):
        # The run's provenance is attested against a statically derived root, so confining
        # against that same root AGREES with the record. Nothing to enforce.
        return []
    violations: list[str] = _refusal_authority_violations(unit)
    for name, function in sorted(unit.functions.items()):
        calls = _confinement_calls(function)
        if not calls:
            continue
        assigns = _local_assigns(function)
        seen_dynamic = False
        static_only: list[str] = []
        params = {arg.arg for arg in function.args.args + function.args.kwonlyargs}
        for call in calls:
            authority = _authority_arg(call)
            tokens = _authority_tokens(authority, unit, assigns) if authority is not None else set()
            if any(kind == "dyn" for kind, _ in tokens):
                seen_dynamic = True
                if params & {"repo_dir", "repo_root"} and not (
                        {token for kind, token in tokens if kind == "dyn"}
                        & (params | {"args", "config"})):
                    violations.append(
                        f"{unit.name}: {name}() takes a checkout parameter "
                        f"({sorted(params & {'repo_dir', 'repo_root'})}) but confines against "
                        f"{sorted(token for _k, token in tokens)} instead of using it"
                    )
            elif tokens:
                static_only.append(_dotted_callee(call.func).split(".")[-1])
        if not seen_dynamic and static_only:
            violations.append(
                f"{unit.name}: {name}() confines paths ONLY against module scope "
                f"({', '.join(sorted(set(static_only)))} -> "
                f"{sorted(token for _k, token in _authority_tokens(_authority_arg(calls[0]), unit, assigns))}) "
                f"while the run is attested against a checkout it was handed -- pass that "
                f"checkout in as the confinement authority"
            )
    return violations


# --------------------------------------------------------------------------- #
# Invariant C: only the finalization seam's EXIT may publish a manifest
# --------------------------------------------------------------------------- #

FINALIZE_CALLEE = "finalize_run_manifest"

# The protected artefact, matched on the FILENAME a launcher names rather than on the helper
# that writes it. Every manifest in the family is a `*run_manifest*.json`.
MANIFEST_ARTEFACT = "run_manifest"

# Marker for a write the gate could not place. Fail CLOSED: reported, never assumed harmless.
UNRESOLVED_WRITE = "UNRESOLVED write form"

# WHERE each write primitive lives -- what a write IS, nothing about argument layout, which is
# read from each primitive's REAL signature below. A short leaf set is the same design as
# Invariant A: everything above it resolves by reading bodies, so the next helper wrapping one of
# these is caught unnamed. `finalize_run_manifest`'s own write is deliberately NOT reachable here
# -- it writes the path it was HANDED and names no artefact, which is the whole difference
# between publishing on exit and publishing early.
#
# A hand-enumerated position table is a list of the call forms somebody thought of; the first cut
# proved it by mapping `rename` to argument 0 when `os.rename(src, dst)` publishes to argument 1.
# A gate whose subject is incomplete models of where a write goes may not carry one.
_PRIMITIVE_HOMES: dict[str, tuple[tuple[str, str], ...]] = {
    # leaf -> ((kind, home), ...). "py": introspect the STDLIB owner. "ast": read the
    # first-party source with the gate's own resolver, so nothing first-party is imported.
    "write_json":        (("ast", "devtools.benchmarks.common.manifests"),),
    "write_jsonl":       (("ast", "devtools.benchmarks.common.manifests"),),
    "atomic_write_json": (("ast", "ouroboros.utils"),),
    "write_text_atomic": (("ast", "ouroboros.utils"),),
    "write_text":        (("py", "pathlib.Path"),),
    "write_bytes":       (("py", "pathlib.Path"),),
    "dump":              (("py", "json"),),
    # Two real callables share each of these leaves, so BOTH signatures are consulted and their
    # destinations unioned: picking one of the two shapes wearing the name was the hole.
    "rename":            (("py", "os"), ("py", "pathlib.Path")),
    "replace":           (("py", "os"), ("py", "pathlib.Path")),
    "move":              (("py", "shutil"),),
}
WRITE_PRIMITIVES = frozenset(_PRIMITIVE_HOMES)

# Parameter names denoting a FILE the call acts on, matched against the REAL signature's own
# names rather than argument positions. `src` counts alongside `dst`: a move publishes at both
# endpoints, and guessing which end matters is guessing again.
_DESTINATION_PARAMETERS = frozenset({
    "self", "path", "dst", "dest", "destination", "target", "src", "source",
    "file", "filename", "fp",
})


def _stdlib_signature(home: str, leaf: str) -> list[tuple[tuple[str, ...], frozenset[str]]]:
    """``leaf``'s real parameter names on a STDLIB owner, via ``inspect``.

    Stdlib only on purpose: the gate must not import a benchmark package to audit it.
    """
    module_name, _, attribute = home.partition(".")
    if module_name not in _STDLIB_MODULES:
        return []
    try:
        owner: Any = importlib.import_module(module_name)
    except ImportError:                                     # pragma: no cover - stdlib is present
        return []
    if attribute:
        owner = getattr(owner, attribute, None)
    target = getattr(owner, leaf, None)
    if target is None:
        return []
    try:
        parameters = inspect.signature(target).parameters
    except (TypeError, ValueError):                         # no introspectable signature
        return []
    positional = tuple(name for name, parameter in parameters.items()
                       if parameter.kind in (parameter.POSITIONAL_ONLY,
                                             parameter.POSITIONAL_OR_KEYWORD))
    return [(positional, frozenset(parameters))]


def _source_signature(home: str, leaf: str) -> list[tuple[tuple[str, ...], frozenset[str]]]:
    """``leaf``'s real parameter names read from first-party SOURCE, no import."""
    unit = _unit_for_module(home)
    function = unit.functions.get(leaf) if unit is not None else None
    if function is None:
        return []
    arguments = function.args
    positional = tuple(argument.arg
                       for argument in [*arguments.posonlyargs, *arguments.args])
    every = set(positional) | {argument.arg for argument in arguments.kwonlyargs}
    return [(positional, frozenset(every))]


@functools.lru_cache(maxsize=None)
def primitive_signatures(leaf: str) -> tuple[tuple[tuple[str, ...], frozenset[str]], ...]:
    """Every real ``(positional names, all names)`` form ``leaf`` can denote.

    Derived from the actual callables and sources, never enumerated, so the gate follows a
    signature that changes instead of drifting away from it.
    """
    forms: list[tuple[tuple[str, ...], frozenset[str]]] = []
    for kind, home in _PRIMITIVE_HOMES.get(leaf, ()):
        forms.extend(_stdlib_signature(home, leaf) if kind == "py"
                     else _source_signature(home, leaf))
    return tuple(forms)


def destination_expressions(call: ast.Call, leaf: str) -> list[ast.expr] | None:
    """Expressions naming a file ``call`` acts on, or ``None`` when the form is UNRESOLVED.

    ``None`` is the FAIL-CLOSED answer: the leaf says this call writes, but no real signature
    told us where, so the caller reports it instead of assuming it harmless. An unmodelled form
    passing quietly is how the hand-written position table grew a hole.

    Binding is decided by the signature, not guessed: a first parameter named ``self`` means an
    attribute call's receiver is the destination and the rest shift by one, while the same
    primitive called flat keeps its positions. No ``self`` never shifts, so ``os.rename(src,
    dst)`` reads as the module function it is despite also being spelled as an attribute access.
    """
    forms = primitive_signatures(leaf)
    found: list[ast.expr] = []
    resolved = False
    attribute_call = isinstance(call.func, ast.Attribute)
    for positional, every in forms:
        indices = [index for index, name in enumerate(positional)
                   if name in _DESTINATION_PARAMETERS]
        if not indices:
            continue
        resolved = True
        bound = attribute_call and bool(positional) and positional[0] == "self"
        for index in indices:
            if bound and index == 0:
                found.append(call.func.value)
                continue
            position = index - 1 if bound else index
            if 0 <= position < len(call.args):
                found.append(call.args[position])
        keyword_names = every & _DESTINATION_PARAMETERS
        found.extend(keyword.value for keyword in call.keywords
                     if keyword.arg in keyword_names)
    return found if resolved else None


def _names_manifest_artefact(call: ast.Call,
                             assigns: dict[str, list[ast.expr]]) -> bool | None:
    """True if a DESTINATION of ``call`` names a run-manifest file. ``None`` == unresolved form.

    Only destinations are inspected, never the payload: CL-Bench's `collect_results` records
    POINTERS to sidecar manifests in the `results.json` it writes, and a first cut reading every
    argument called that a publication. Recording a path is not writing to it.

    The local hop matters the other way: ``manifest_path = run_dir / "run_manifest.json"`` then
    ``write_json(manifest_path, manifest)`` is the same publication with the literal moved one
    line up — the shape of the `run_pro` instance this invariant found.
    """
    def _literal(expr: ast.expr) -> bool:
        return any(isinstance(node, ast.Constant) and isinstance(node.value, str)
                   and MANIFEST_ARTEFACT in node.value for node in ast.walk(expr))

    destinations = destination_expressions(call, _dotted_callee(call.func).split(".")[-1])
    if destinations is None:
        return None
    for destination in destinations:
        if _literal(destination):
            return True
        for node in ast.walk(destination):
            if isinstance(node, ast.Name) and any(_literal(source)
                                                  for source in assigns.get(node.id, [])):
                return True
    return False


def _resolve_function(leaf: str, unit: _Unit) -> tuple[ast.FunctionDef | None, _Unit]:
    """``leaf``'s definition, from this module or from the first-party module it came from."""
    target = unit.functions.get(leaf)
    if target is not None:
        return target, unit
    imported = _unit_for_module(unit.imports.get(leaf, ""))
    if imported is None:
        return None, unit
    return imported.functions.get(leaf), imported


def publishes_manifest(call: ast.Call, unit: _Unit,
                       assigns: dict[str, list[ast.expr]] | None = None,
                       *, depth: int = 3) -> str:
    """Description of a manifest publication effected by ``call``, or ``""``.

    Resolves through helper bodies as ``resolve_denied`` does, for the same reason: the offenders
    are named for the records they keep, so only their BODIES say a manifest gets written.
    """
    assigns = {} if assigns is None else assigns
    callee = _dotted_callee(call.func)
    leaf = callee.split(".")[-1]
    if leaf in WRITE_PRIMITIVES:
        # A primitive DECIDES here; there is no point walking a writer's own body. `None` is the
        # fail-closed verdict for a write form no real signature could place.
        named = _names_manifest_artefact(call, assigns)
        if named is None:
            return f"{callee} [{UNRESOLVED_WRITE}]"
        return callee if named else ""
    if depth <= 0:
        return ""
    target, target_unit = _resolve_function(leaf, unit)
    if target is None:
        return ""
    inner_assigns = _local_assigns(target)
    for node in ast.walk(target):
        if not isinstance(node, ast.Call):
            continue
        inner = publishes_manifest(node, target_unit, inner_assigns, depth=depth - 1)
        if inner:
            return f"{target.name} -> {inner}"
    return ""


def _seam_publication_violations(unit: _Unit) -> list[str]:
    """Manifest publications reachable from INSIDE an active finalization seam."""
    # Keyed by the offending call's line so a `with` in a nested def is reported once, and
    # attributed to the INNERMOST enclosing function.
    found: dict[int, tuple[int, str, str]] = {}
    for name, function in sorted(unit.functions.items()):
        assigns = _local_assigns(function)
        for node in ast.walk(function):
            if not isinstance(node, (ast.With, ast.AsyncWith)):
                continue
            if not any(isinstance(item.context_expr, ast.Call)
                       and _dotted_callee(item.context_expr.func).endswith(FINALIZE_CALLEE)
                       for item in node.items):
                continue
            for statement in node.body:
                for call in [inner for inner in ast.walk(statement)
                             if isinstance(inner, ast.Call)]:
                    published = publishes_manifest(call, unit, assigns)
                    if not published:
                        continue
                    if call.lineno not in found or function.lineno > found[call.lineno][0]:
                        found[call.lineno] = (function.lineno, name, published)
    reports: list[str] = []
    for lineno, (_def_line, name, published) in sorted(found.items()):
        if UNRESOLVED_WRITE in published:
            reports.append(
                f"{unit.name}: {name}() performs a write at line {lineno} ({published}) from "
                f"INSIDE an active {FINALIZE_CALLEE}, and no real signature places its "
                f"destination -- so the gate cannot tell whether it publishes a manifest. "
                f"Reported rather than assumed harmless: give the primitive a home in "
                f"_PRIMITIVE_HOMES, or move the write out of the seam"
            )
            continue
        reports.append(
            f"{unit.name}: {name}() publishes a manifest from INSIDE an active {FINALIZE_CALLEE} "
            f"at line {lineno} ({published}) -- the seam merges the terminal outcome/exit_code/"
            f"refusal only when the context EXITS, so this writes a pre-merge record that a "
            f"concurrent reader can observe and an interruption makes durable. The seam writes "
            f"the same path on every exit path already: delete the early write"
        )
    return reports


# --------------------------------------------------------------------------- #
# Seam shape (kept with the invariants: ONE gate, one report)
# --------------------------------------------------------------------------- #

def _seam_violations(source: str, name: str) -> list[str]:
    violations: list[str] = []
    if f"{ADMISSION_CALLEE}(" not in source:
        violations.append(f"{name}: bypasses the admission seam")
        return violations
    if "finalize_run_manifest(" not in source:
        violations.append(f"{name}: records no final outcome")
    if "benchmark_run_manifest(" in source:
        violations.append(
            f"{name}: calls the builder directly again: its refusal would never be persisted")
    # Python evaluates ARGUMENTS before entering the callee, so a gate called inside the
    # admission call's argument list refuses BEFORE the manifest can be written — the durable
    # refusal defeated by evaluation order. Attestation belongs after admission.
    call = source.split(f"{ADMISSION_CALLEE}(", 1)[1].split("\n    )\n", 1)[0]
    if "runtime_attestation(" in call:
        violations.append(
            f"{name}: evaluates runtime_attestation inside the admission argument list")
    return violations


def audit_source(source: str, *, name: str = "<synthetic>") -> list[str]:
    """Every invariant violation in one launcher source. Empty list == the gate passes."""
    unit = _Unit(ast.parse(source), name)
    violations = _seam_violations(source, name)
    if any("bypasses the admission seam" in item for item in violations):
        return violations
    return (violations + _pre_admission_violations(unit) + _confinement_violations(unit)
            + _seam_publication_violations(unit))


def audit_launcher(path: pathlib.Path) -> list[str]:
    return audit_source(pathlib.Path(path).read_text(encoding="utf-8"),
                        name=pathlib.Path(path).name)


def audit_all_launchers() -> list[str]:
    """The gate: every migrated launcher, both invariants, one report."""
    violations: list[str] = []
    for path in launcher_paths():
        violations.extend(audit_launcher(path))
    return violations
