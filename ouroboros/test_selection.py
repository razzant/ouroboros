"""Commit/PR test scope: a contract floor, direct imports and authored widening.

Selection is computed from the candidate's files, including dirty and untracked
inputs. It is a minimum, not a claim about dynamic or indirect consumers. It names
modules; the runners' lane exclusions still decide which of their tests execute.
"""
from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
from typing import Iterable


# Minimal witnesses of the contracts verification and every install rely on, and the inventory
# pin: book chapters, runtime text, tracked paths and UI modules reach it as data no import traces (#3188).
# Replacing a witness is a verifier change and requires full qualification.
# The whole Node lane is unconditional in both runners, outside this Python list.
FLOOR_TESTS = {
    "tests/test_contracts.py": "frozen ABI and tool/context protocols",
    "tests/test_gateway_parity.py": "HTTP/WS wire schemas and router parity",
    "tests/test_packaging_sync.py": "release identity and public install carriers",
    "tests/test_process_custody.py": "every long-lived child has a lifetime owner",
    "tests/test_hermetic_data_root.py": "tests cannot bind the live data roots",
    "tests/test_preflight_process_containment.py": "a leaked test process cannot pass the gate",
    "tests/test_smoke.py": "the shipped Python modules import",
    "tests/test_safety_policy.py": "every built-in tool has an explicit safety policy",
    "tests/test_generated_inventories.py": "docs/inventories/ match their regeneration from books, code, paths and UI",
}

SELECTION_RULE = "floor + changed tests + direct importers of changed code, tests and test helpers + authored widening"
# Bind the policy this process loaded, even if its source changes later on disk.
_POLICY_SOURCE = Path(__file__).read_bytes()
_APPARATUS = frozenset({
    "pyproject.toml", "pytest.ini", ".pytest.ini", "setup.cfg", "tox.ini",
    "uv.lock", "requirements.txt", "requirements-runtime.lock",
    "web/package.json", "web/package-lock.json", "site/pnpm-lock.yaml",
    "ouroboros/preflight_runner.py", "ouroboros/preflight_probe.py", "ouroboros/preflight_node.py",
    "ouroboros/test_bytecode.py",
    "ouroboros/preflight_receipts.py", "ouroboros/commit_admission.py",
    "ouroboros/test_selection.py", "ouroboros/preflight_inventory.py",
    "ouroboros/preflight_qualification.py", "scripts/run_tests.py", "scripts/safe_test.py",
    "ouroboros/test_environment.py", ".github/workflows/ci.yml",
    "ouroboros/process_containment.py", "ouroboros/process_lifeline.py",
    "ouroboros/process_overload.py", "ouroboros/test_parallelism.py", "ouroboros/host_memory.py",
    "tests/default_lane.py", "tests/browser_lane.py", "tests/ci_evidence.py", "tests/ci_verdicts.py",
    "tests/ci_accepted.py", "tests/stack_samples.py", "tests/_test_folders.py",
})
# What switches, attests, boots and restores a body: a selected run cannot qualify
# adopting a change to the machinery that adopts it. A minimum, not a risk classifier.
_CUTOVER = frozenset({
    "server.py", "launcher.py", "ouroboros/__init__.py", "ouroboros/body_switch.py",
    "ouroboros/body_adoption.py", "supervisor/update_recovery.py",
    "ouroboros/launcher_bootstrap.py", "supervisor/git_ops_reset.py", "supervisor/update_merge.py",
})


def _git(repo: Path, *args: str) -> bytes:
    # Git inspection is bounded transport, never a test-runner timeout.
    from ouroboros.git_observation import host_git_env

    return subprocess.run(["git", *args], cwd=repo, capture_output=True,
                          check=True, timeout=30, env=host_git_env()).stdout


def changed_paths(repo: Path, base: str) -> tuple[str, tuple[str, ...]]:
    """Resolve once; --no-renames keeps both sides of a module rename in scope."""
    from ouroboros.git_observation import numstat_paths

    oid = _git(repo, "rev-parse", "--verify", "--end-of-options", f"{base}^{{commit}}").decode().strip()
    cmd = ["git", "diff", "--no-ext-diff", "--no-textconv", "--no-renames", "--numstat", "-z", oid, "--"]
    paths = numstat_paths(_git(repo, *cmd[1:]), cmd, lambda args: _git(repo, *args[1:]))
    paths.extend(os.fsdecode(p) for p in _git(repo, "ls-files", "--others", "--exclude-standard", "-z").split(b"\0") if p)
    return oid, tuple(sorted(set(paths)))


def is_test_module(path: str) -> bool:
    p = PurePosixPath(path)
    return bool(p.parts) and p.parts[0] == "tests" and p.name.startswith("test_") and p.suffix == ".py"


def is_test_helper(path: str) -> bool:
    """Python under tests/ that pytest neither collects nor loads as its configuration."""
    p = PurePosixPath(path)
    return (p.parts[0] == "tests" and p.suffix == ".py" and p.name != "conftest.py"
            and not is_test_module(path) and path not in _APPARATUS)


def apparatus_change(path: str) -> bool:
    p = PurePosixPath(path)
    # Under tests/, pytest configuration and fixture data are shared apparatus; a helper selects its importers.
    return (path in _APPARATUS or path in FLOOR_TESTS
            or (p.parts[0] == "tests" and not is_test_module(path) and not is_test_helper(path))
            or (p.name.startswith("requirements") and p.suffix in {".txt", ".lock"}))


# Release carriers bumped by every numbered release (P9): only a change beyond the
# release version makes them apparatus, or selection would never apply to a release.
_RELEASE_CARRIERS = frozenset({"pyproject.toml", "uv.lock", "web/package.json", "web/package-lock.json"})
_PYPROJECT_VERSION = re.compile(r'^version\s*=\s*"([^"]+)"', re.M)


def _release_versions(repo: Path, base: str) -> set[str]:
    """Both spellings of the base and candidate release version (VERSION and its PEP 440 form)."""
    versions = set()
    for name in ("VERSION", "pyproject.toml"):
        texts = []
        try:
            texts.append(_git(repo, "show", f"{base}:{name}").decode("utf-8"))
        except (subprocess.SubprocessError, UnicodeError):
            pass
        try:
            texts.append((repo / name).read_text(encoding="utf-8"))
        except OSError:
            pass
        for text in texts:
            if name == "VERSION":
                versions.add(text.strip())
            elif match := _PYPROJECT_VERSION.search(text):
                versions.add(match.group(1))
    return {version for version in versions if version}


def _version_only_change(repo: Path, base: str, path: str, versions: set[str]) -> bool:
    try:
        # Both sides with universal newlines, as read_text gives the checkout: a blob a
        # Windows checkout stored with CRLF bytes must not turn a version bump into apparatus.
        old = _git(repo, "show", f"{base}:{path}").decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        new = (repo / path).read_text(encoding="utf-8")
    except (OSError, UnicodeError, subprocess.SubprocessError):
        return False
    for version in sorted(versions, key=len, reverse=True):
        old, new = old.replace(version, "\0"), new.replace(version, "\0")
    return bool(versions) and old == new


def _module(path: str) -> str:
    parts = list(PurePosixPath(path).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def direct_imports(source: str, path: str) -> set[str]:
    """Include imports in functions/TYPE_CHECKING; do not follow transitive edges."""
    imports: set[str] = set()
    package = path.removesuffix(".py").split("/")[:-1]
    for node in ast.walk(ast.parse(source, filename=path)):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            prefix = ".".join(package[:len(package) - node.level + 1]) if node.level else ""
            module = ".".join(part for part in (prefix, node.module or "") if part)
            if module:
                imports.add(module)
                imports.update(f"{module}.{alias.name}" for alias in node.names if alias.name != "*")
    return imports


# Every proof check re-selects; parsing ~2,000 test modules each time cost seconds per commit.
# Keyed by content, so an edited module is parsed again; the bound only caps memory.
_IMPORT_SCANS: dict[tuple[str, bytes], frozenset[str]] = {}
_IMPORT_SCAN_ENTRIES = 50_000


def _scanned_imports(repo: Path, path: str) -> frozenset[str]:
    source = (repo / path).read_text(encoding="utf-8")
    key = (path, hashlib.blake2b(source.encode("utf-8"), digest_size=16).digest())
    if key not in _IMPORT_SCANS:
        if len(_IMPORT_SCANS) >= _IMPORT_SCAN_ENTRIES:
            _IMPORT_SCANS.clear()
        _IMPORT_SCANS[key] = frozenset(direct_imports(source, path))
    return _IMPORT_SCANS[key]


def _imports_of(repo: Path, path: str) -> frozenset[str]:
    try:
        return _scanned_imports(repo, path)
    except (SyntaxError, UnicodeError) as exc:
        # An unavailable graph grants no extra automatic escalation rule.
        raise ValueError(f"import scan unavailable for {path}; repair it or request full_tests with test_reason") from exc


def _bound(imports: Iterable[str], modules: Iterable[str]) -> list[str]:
    """The modules that a set of imports names, a package or a member of it included."""
    return sorted(name for name in modules if any(i == name or i.startswith(name + ".") for i in imports))


def _reaching_helper(repo: Path, helper: str, sources: Iterable[str]) -> set[str]:
    """The non-test Python under tests/ that reaches ``helper`` through imports, itself included.

    A helper runs nothing alone: the test modules importing any of these answer for it.
    """
    reached = {helper}
    while True:
        names = {_module(path) for path in reached}
        more = {path for path in sources if path not in reached and _bound(_imports_of(repo, path), names)}
        if not more:
            return reached
        reached |= more


def validate_widening(repo: Path, extra_tests=(), full_tests=False, reason="", *, skipped=None) -> tuple[str, ...]:
    """The widening a commit asks; ``skipped`` (a list) collects the extra_tests entries the tests preflight cannot
    run, disclosed instead of refusing the call (06.11); without it such an entry refuses, as before."""
    if type(full_tests) is not bool or not isinstance(extra_tests, (list, tuple)):
        raise ValueError("extra_tests must be a list of paths and full_tests must be a boolean")
    if not isinstance(reason, str):
        raise ValueError("test_reason must be a string")
    if (extra_tests or full_tests) and not reason.strip():
        raise ValueError("test_reason is required when widening with extra_tests or full_tests")
    normalized = set()
    for path in extra_tests:
        if not isinstance(path, str) or "\\" in path or ".." in PurePosixPath(path).parts:
            problem = f"extra_tests needs repository-relative test modules: {path!r}"
        elif PurePosixPath(path).is_absolute() or not is_test_module(path) or not (repo / path).is_file():
            problem = f"extra_tests module is missing or invalid: {path!r}"
        elif not (repo / path).resolve().is_relative_to(repo.resolve()):
            problem = f"extra_tests module escapes the candidate: {path!r}"
        else:
            normalized.add(PurePosixPath(path).as_posix())
            continue
        if skipped is None:
            raise ValueError(problem)
        skipped.append(str(path))
    return tuple(sorted(normalized))


@dataclass(frozen=True)
class TestSelection:
    __test__ = False
    tier: str
    base: str
    files: tuple[str, ...]
    reasons: tuple[tuple[str, str], ...]
    changed: tuple[str, ...]
    policy: str
    available: int
    extra_tests: tuple[str, ...] = ()
    full_tests: bool = False
    test_reason: str = ""

    def scope(self) -> dict:
        """The selection inputs a durable selected record binds beside its workload."""
        return {"base": self.base, "policy": self.policy, "changed": list(self.changed),
                "extra_tests": list(self.extra_tests), "full_tests": self.full_tests, "test_reason": self.test_reason}

    def record(self) -> dict:
        record = {**asdict(self), "rule": SELECTION_RULE, "omitted_files": self.available - len(self.files)}
        # A full run would list every module; its count and the rule say the same far smaller.
        record["files"] = record["expected_files"] = list(self.files) if self.tier == "selected" else []
        record["file_count"] = len(self.files)
        return record


def select_from_paths(repo: Path, paths: Iterable[str], *, base: str,
                      extra_tests=(), full_tests=False, test_reason="") -> TestSelection:
    """Pure candidate scan seam, also used for historical shadow comparisons."""
    repo = Path(repo)
    extra = validate_widening(repo, extra_tests, full_tests, test_reason)
    missing = [path for path in FLOOR_TESTS if not (repo / path).is_file()]
    if missing:
        raise ValueError("required gate floor is missing: " + ", ".join(missing))
    changed = tuple(sorted(set(paths)))
    python = sorted(p.relative_to(repo).as_posix() for p in (repo / "tests").rglob("*.py"))
    files = tuple(path for path in python if is_test_module(path))
    sources = tuple(path for path in python if not is_test_module(path))
    reasons = [(path, f"floor: {invariant}") for path, invariant in FLOOR_TESTS.items()]
    versions = _release_versions(repo, base) if _RELEASE_CARRIERS & set(changed) else set()
    apparatus = "verification apparatus changed"
    full = [(p, apparatus) for p in changed if apparatus_change(p)
            and not (p in _RELEASE_CARRIERS and _version_only_change(repo, base, p, versions))]
    full += [(p, "body cutover/recovery changed") for p in changed if p in _CUTOVER]
    if full_tests:
        full.append(("*", "author requested full: " + test_reason.strip()))
    selected = set(FLOOR_TESTS)
    # A changed test module or helper is imported like product code; a helper's importers answer for it,
    # unless pytest configuration loads it for every test or nothing traced imports it (a child script).
    modules = {_module(p) for p in changed if p.endswith(".py")}
    helpers = {p: _reaching_helper(repo, p, sources) for p in changed
               if not full and is_test_helper(p) and (repo / p).is_file()}
    full += [(p, apparatus) for p, reached in helpers.items() if any(apparatus_change(r) for r in reached)]
    modules.update(_module(r) for reached in helpers.values() for r in reached)
    imported: set[str] = set()
    for path in files:
        if path in changed:
            selected.add(path)
            reasons.append((path, "changed test module"))
        if not full and modules:
            bound = _bound(_imports_of(repo, path), modules)
            if bound:
                selected.add(path)
                imported.update(bound)
                reasons.append((path, "direct import: " + ", ".join(bound)))
    if not full:
        full = [(p, apparatus) for p, reached in helpers.items() if not {_module(r) for r in reached} & imported]
    reasons.extend((p, "author widening: " + test_reason.strip()) for p in extra)
    selected.update(extra)
    return TestSelection("full" if full else "selected", base, files if full else tuple(sorted(selected)),
                         tuple(sorted(reasons + full)), changed, policy_digest(), len(files), extra, full_tests,
                         test_reason.strip())


def policy_digest() -> str:
    """The loaded selection policy; its source digest binds policy changes before any version label."""
    return hashlib.sha256(_POLICY_SOURCE + json.dumps(FLOOR_TESTS, sort_keys=True).encode()).hexdigest()


def select_tests(repo: Path, *, base="HEAD", extra_tests=(), full_tests=False, test_reason="") -> TestSelection:
    oid, paths = changed_paths(Path(repo), base)
    return select_from_paths(Path(repo), paths, base=oid, extra_tests=extra_tests,
                             full_tests=full_tests, test_reason=test_reason)


def inherit_widening(repo: Path, request: dict, earlier: dict) -> dict:
    """``request`` joined with what its unadopted predecessors declared: the indirect consumers
    still present, any broad impact (full) and the reasons for both."""
    kept = [p for p in earlier.get("extra_tests", ()) if is_test_module(p) and (Path(repo) / p).is_file()]
    reasons = (str(request.get("test_reason") or "").strip(), str(earlier.get("test_reason") or "").strip())
    return {"extra_tests": tuple(sorted({*request.get("extra_tests", ()), *kept})),
            "full_tests": bool(request.get("full_tests") or earlier.get("full_tests")),
            "test_reason": "; ".join(dict.fromkeys(reason for reason in reasons if reason))}


def refresh_selection(repo: Path, selection: TestSelection) -> TestSelection:
    return select_tests(repo, base=selection.base, extra_tests=selection.extra_tests,
                        full_tests=selection.full_tests, test_reason=selection.test_reason)


def widening_properties() -> dict:
    """Tool-schema SSOT: the model chooses consumers the import graph cannot see."""
    return {
        "extra_tests": {"type": "array", "items": {"type": "string"}, "default": [],
                        "description": "Add Python test modules (tests/test_*.py of this repository) for indirect/dynamic, prompt or skill consumers beyond the floor and direct imports; explain in test_reason. Other entries are skipped with a note."},
        "full_tests": {"type": "boolean", "default": False,
                       "description": "Request the full battery for broad or unresolved impact; explain in test_reason. Managed updates and adoption/evolution publication require full; local adoption may use an exact whole-delta selection. See DEVELOPMENT: The tiered commit and CI gates for publication limits."},
        "test_reason": {"type": "string", "default": "",
                        "description": "Why extra_tests or full_tests widens verification; retained with the commit's test evidence."},
    }


def bind_commit_test_request(ctx, kwargs) -> str:
    ctx._commit_call_notes = []  # a commit call's notes start here (``review_collection.note_commit_call``)
    request = {key: kwargs.pop(key, default) for key, default in
               (("extra_tests", []), ("full_tests", False), ("test_reason", ""))}
    skipped: list = []
    try:
        request["extra_tests"] = validate_widening(Path(ctx.repo_dir), **{
            "extra_tests": request["extra_tests"], "full_tests": request["full_tests"], "reason": request["test_reason"]},
            skipped=skipped)
    except ValueError as exc:
        return f"⚠️ TOOL_ARG_ERROR: {exc}. Nothing was staged or reviewed."
    if skipped:  # what the tests preflight cannot run is disclosed, never a refusal of the commit (06.11)
        ctx._commit_call_notes.append("extra_tests skipped (only Python test modules tests/test_*.py of this "
                                      f"repository run in the tests preflight): {', '.join(skipped)}.")
    ctx._commit_test_request = request
    return ""


def commit_test_options(ctx, *, force=False, post_commit=False) -> dict:
    """The author's request, selected; ``{}`` (full) when forced or never requested.

    A bound body candidate stays selected even when forced (its evolution run): that proof
    can qualify only local adoption. Its post-commit run selects against the whole
    serving→candidate delta, not ``HEAD~1``. The managed resolver is never bound.
    """
    from ouroboros import body_candidate

    request, bound = getattr(ctx, "_commit_test_request", None), body_candidate.is_bound(ctx)
    if request is None or (force and not bound):
        return {}
    return {"selected": True, **(body_candidate.qualification_request(ctx, request) if bound and post_commit
                                 else request)}
