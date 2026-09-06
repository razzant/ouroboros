"""Repository size-ratchet inventory and complexity metrics."""

from __future__ import annotations

import ast
import contextlib
import hashlib
import os
import pathlib
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Tuple

from ouroboros.tools.review_helpers import _VENDORED_NAMES, _VENDORED_SUFFIXES
from ouroboros.tools.review_helpers import iter_repo_pack_entries  # noqa: F401 - public compatibility re-export

TARGET_MODULE_LINES = 1000
MAX_MODULE_LINES = 1600
BAND_MODULE_MAX_LINES = 1500
MAX_MODULE_BYTES = 200_000
TARGET_FUNCTION_LINES = 150
MAX_FUNCTION_LINES = 300
# Owner decision 2026-08-21: keep this only as a high-water alarm with ample
# product headroom; module and per-function ratchets remain the primary gates.
MAX_TOTAL_FUNCTIONS = 9500

SIZE_RATCHET_MANIFEST_PATH = "ouroboros/size_ratchet_manifest.py"

# Module inventory covers tests/devtools. Only generated, vendored, or environment
# directories remain outside the source ratchet.
_MODULE_SKIP_DIR_NAMES = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".venv",
        "__pycache__",
        "assets",
        "build",
        "dist",
        "node_modules",
        "python-standalone",
        "venv",
    }
)
# Function inventory intentionally preserves the pre-v7 runtime-health scope.
_FUNCTION_SKIP_DIR_NAMES = _MODULE_SKIP_DIR_NAMES | frozenset({"devtools", "tests"})
FUNCTION_COUNT_EXCLUDED_FILES = frozenset({"app.py", "demo_app.py", "launcher.py"})


@dataclass(frozen=True, order=True)
class GatedModule:
    """One strict-UTF-8 source module in the deterministic size inventory."""

    path: str
    line_count: int
    utf8_bytes: int
    _source_text: str = field(default="", repr=False, compare=False)


@dataclass(frozen=True, order=True)
class GatedFunction:
    """One Python function keyed by exact path and lexical qualname."""

    path: str
    qualname: str
    line_start: int
    line_count: int


@dataclass(frozen=True)
class SizeRatchetManifest:
    """Parsed data-only size ratchet manifest."""

    baseline_source_sha: str
    giant_paths: frozenset[str]
    function_debt: frozenset[tuple[str, str]]
    band_baseline_paths: frozenset[str]
    band_paths: Mapping[str, str | None]
    byte_baseline_debt: Mapping[str, int]
    byte_debt: Mapping[str, int]


@dataclass(frozen=True)
class SizeRatchetInventory:
    """Exact live debt derived from one candidate tree."""

    modules: tuple[GatedModule, ...]
    functions: tuple[GatedFunction, ...]
    giant_paths: frozenset[str]
    function_debt: frozenset[tuple[str, str]]
    band_paths: frozenset[str]
    byte_debt: Mapping[str, int]


def _exact_repo_relative_path(raw: str | pathlib.Path) -> str:
    text = str(raw)
    posix = pathlib.PurePosixPath(text)
    if (
        "\\" in text
        or posix.is_absolute()
        or not posix.parts
        or any(part in {"", ".", ".."} for part in posix.parts)
        or text != posix.as_posix()
    ):
        raise ValueError(f"path is not a canonical repo-relative path: {raw!r}")
    return posix.as_posix()


def _section_repo_relative_path(raw: str | pathlib.Path) -> str:
    """Normalize the legacy ``collect_sections`` display prefix only."""
    text = str(raw).replace("\\", "/")
    if text.startswith("repo/"):
        text = text[len("repo/") :]
    return _exact_repo_relative_path(text)


def is_gated_js_module(path: str) -> bool:
    """Return whether an exact repo-relative path is first-party web JavaScript."""
    rel = _exact_repo_relative_path(path)
    posix = pathlib.PurePosixPath(rel)
    if not (rel.startswith("web/") and rel.endswith(".js")):
        return False
    if any(part in _MODULE_SKIP_DIR_NAMES for part in posix.parts[:-1]):
        return False
    name = posix.name
    return name not in _VENDORED_NAMES and not any(name.endswith(suffix) for suffix in _VENDORED_SUFFIXES)


def _is_gated_module_path(path: str) -> bool:
    posix = pathlib.PurePosixPath(path)
    if any(part in _MODULE_SKIP_DIR_NAMES for part in posix.parts[:-1]):
        return False
    return path.endswith(".py") or is_gated_js_module(path)


def _filesystem_repo_paths(repo_dir: pathlib.Path) -> Iterator[str]:
    for raw_root, dirs, files in os.walk(repo_dir, topdown=True, followlinks=False):
        dirs[:] = sorted(name for name in dirs if name not in _MODULE_SKIP_DIR_NAMES)
        root = pathlib.Path(raw_root)
        for name in sorted(files):
            yield (root / name).relative_to(repo_dir).as_posix()


def candidate_repo_paths(repo_dir: pathlib.Path) -> tuple[str, ...]:
    """Return cached and nonignored untracked paths for a Git candidate tree."""
    root = pathlib.Path(repo_dir).resolve()
    try:
        top_level = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=root,
            check=False,
            capture_output=True,
        )
    except OSError:
        top_level = None
    if top_level is None or top_level.returncode != 0:
        return tuple(_filesystem_repo_paths(root))

    git_root = pathlib.Path(top_level.stdout.decode("utf-8").strip()).resolve()
    if git_root != root:
        return tuple(_filesystem_repo_paths(root))
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    paths = (item.decode("utf-8") for item in result.stdout.split(b"\0") if item)
    return tuple(sorted(path for path in paths if root.joinpath(*pathlib.PurePosixPath(path).parts).exists()))


def iter_gated_modules(
    repo_dir: pathlib.Path,
    *,
    repo_paths: Iterable[str] | None = None,
) -> Iterator[GatedModule]:
    """Yield the canonical source inventory.

    The default Git candidate sees cached and nonignored untracked paths. A
    deterministic filesystem fallback supports non-Git fixtures, while callers
    may inject an exact merge-base inventory.
    """
    root = pathlib.Path(repo_dir).resolve()
    injected = repo_paths is not None
    raw_paths = candidate_repo_paths(root) if repo_paths is None else repo_paths
    normalized: set[str] = set()
    for raw_path in raw_paths:
        rel = _exact_repo_relative_path(raw_path)
        if _is_gated_module_path(rel):
            normalized.add(rel)

    for rel in sorted(normalized):
        path = root.joinpath(*pathlib.PurePosixPath(rel).parts)
        if not path.exists():
            if injected:
                raise ValueError(f"injected gated source path does not exist: {rel}")
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"gated source path must be a regular file: {rel}")
        try:
            path.resolve().relative_to(root)
        except ValueError as exc:
            raise ValueError(f"gated source path escapes repository: {rel}") from exc
        raw = path.read_bytes()
        text = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        yield GatedModule(rel, len(text.splitlines()), len(text.encode("utf-8")), text)


def _iter_lexical_functions(tree: ast.AST, path: str) -> Iterator[GatedFunction]:
    def visit(node: ast.AST, scope: tuple[str, ...]) -> Iterator[GatedFunction]:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            qualname = ".".join((*scope, node.name))
            if node.end_lineno is None:
                raise ValueError(f"missing function end line: {path}:{qualname}")
            yield GatedFunction(
                path=path,
                qualname=qualname,
                line_start=node.lineno,
                line_count=node.end_lineno - node.lineno + 1,
            )
            nested_scope = (*scope, node.name, "<locals>")
            for child in node.body:
                yield from visit(child, nested_scope)
            return
        if isinstance(node, ast.ClassDef):
            class_scope = (*scope, node.name)
            for child in node.body:
                yield from visit(child, class_scope)
            return
        if isinstance(node, ast.Lambda):
            return
        for child in ast.iter_child_nodes(node):
            yield from visit(child, scope)

    yield from visit(tree, ())


# (path, sha1 of the module text) -> its exact function inventory. Parsing is a
# pure function of that key, and one validation run inventories several trees
# (live tree, staged index, HEAD/parent refs for the pairwise transition) that
# share almost every module blob unchanged: each distinct text is parsed once,
# not once per tree. Bounded: cleared when it outgrows the working set.
_MODULE_FUNCTIONS_CACHE: dict[tuple[str, str], tuple[GatedFunction, ...]] = {}
_MODULE_FUNCTIONS_CACHE_LIMIT = 8192


def _module_functions(module: GatedModule) -> tuple[GatedFunction, ...]:
    key = (module.path, hashlib.sha1(module._source_text.encode("utf-8")).hexdigest())
    cached = _MODULE_FUNCTIONS_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        tree = ast.parse(module._source_text, filename=module.path)
    except SyntaxError as exc:
        raise ValueError(f"gated Python module has invalid syntax: {module.path}") from exc
    functions = tuple(_iter_lexical_functions(tree, module.path))
    if len(_MODULE_FUNCTIONS_CACHE) >= _MODULE_FUNCTIONS_CACHE_LIMIT:
        _MODULE_FUNCTIONS_CACHE.clear()
    _MODULE_FUNCTIONS_CACHE[key] = functions
    return functions


def _iter_gated_functions_from_modules(modules: Iterable[GatedModule]) -> Iterator[GatedFunction]:
    seen_keys: set[tuple[str, str]] = set()
    for module in modules:
        posix = pathlib.PurePosixPath(module.path)
        if not module.path.endswith(".py"):
            continue
        if posix.name in FUNCTION_COUNT_EXCLUDED_FILES:
            continue
        if any(part in _FUNCTION_SKIP_DIR_NAMES for part in posix.parts[:-1]):
            continue
        for function in _module_functions(module):
            key = (function.path, function.qualname)
            if key in seen_keys:
                raise ValueError(f"duplicate function ratchet key: {key!r}")
            seen_keys.add(key)
            yield function


def iter_gated_functions(
    repo_dir: pathlib.Path,
    *,
    repo_paths: Iterable[str] | None = None,
) -> Iterator[GatedFunction]:
    """Yield exact Python function debt while preserving the legacy skip scope."""
    yield from _iter_gated_functions_from_modules(iter_gated_modules(repo_dir, repo_paths=repo_paths))


def collect_size_ratchet_inventory(
    repo_dir: pathlib.Path,
    *,
    repo_paths: Iterable[str] | None = None,
) -> SizeRatchetInventory:
    """Collect one exact inventory for gates, health, census, and regeneration."""
    injected = tuple(repo_paths) if repo_paths is not None else None
    modules = tuple(iter_gated_modules(repo_dir, repo_paths=injected))
    functions = tuple(_iter_gated_functions_from_modules(modules))
    return SizeRatchetInventory(
        modules=modules,
        functions=functions,
        giant_paths=frozenset(item.path for item in modules if item.line_count > MAX_MODULE_LINES),
        function_debt=frozenset(
            (item.path, item.qualname) for item in functions if item.line_count > MAX_FUNCTION_LINES
        ),
        band_paths=frozenset(
            item.path for item in modules if TARGET_MODULE_LINES < item.line_count <= BAND_MODULE_MAX_LINES
        ),
        byte_debt={item.path: item.utf8_bytes for item in modules if item.utf8_bytes > MAX_MODULE_BYTES},
    )


@contextlib.contextmanager
def _git_source_snapshot(repo_dir: pathlib.Path, ref: str) -> Iterator[tuple[pathlib.Path, tuple[str, ...]]]:
    root = pathlib.Path(repo_dir).resolve()
    with tempfile.TemporaryDirectory(prefix="ouroboros-size-ratchet-") as raw_temp:
        snapshot = pathlib.Path(raw_temp)
        tree = subprocess.run(
            ["git", "ls-tree", "-rz", "--full-tree", ref],
            cwd=root,
            check=True,
            capture_output=True,
        )
        entries: list[tuple[str, bytes]] = []
        for record in tree.stdout.split(b"\0"):
            if not record:
                continue
            try:
                metadata, raw_path = record.split(b"\t", 1)
                mode, object_type, object_id = metadata.split(b" ", 2)
            except ValueError as exc:
                raise ValueError(f"git ls-tree returned a malformed entry at {ref}") from exc
            path = _exact_repo_relative_path(raw_path.decode("utf-8"))
            if not _is_gated_module_path(path):
                continue
            if object_type != b"blob" or mode not in {b"100644", b"100755"}:
                raise ValueError(f"gated source at {ref} must be a regular file: {path}")
            entries.append((path, object_id))

        batch = subprocess.run(
            ["git", "cat-file", "--batch"],
            cwd=root,
            check=True,
            input=b"".join(object_id + b"\n" for _path, object_id in entries),
            capture_output=True,
        ).stdout
        cursor = 0
        paths: list[str] = []
        for path, expected_id in entries:
            header_end = batch.find(b"\n", cursor)
            if header_end < 0:
                raise ValueError(f"git cat-file omitted the header for {path} at {ref}")
            header = batch[cursor:header_end].split(b" ")
            if len(header) != 3 or header[0] != expected_id or header[1] != b"blob":
                raise ValueError(f"git cat-file returned the wrong object for {path} at {ref}")
            try:
                size = int(header[2])
            except ValueError as exc:
                raise ValueError(f"git cat-file returned an invalid size for {path} at {ref}") from exc
            blob_start = header_end + 1
            blob_end = blob_start + size
            if blob_end >= len(batch) or batch[blob_end : blob_end + 1] != b"\n":
                raise ValueError(f"git cat-file returned a truncated blob for {path} at {ref}")
            destination = snapshot.joinpath(*pathlib.PurePosixPath(path).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(batch[blob_start:blob_end])
            paths.append(path)
            cursor = blob_end + 1
        if cursor != len(batch):
            raise ValueError(f"git cat-file returned unexpected trailing data at {ref}")
        yield snapshot, tuple(paths)


def collect_size_ratchet_inventory_at_ref(repo_dir: pathlib.Path, ref: str) -> SizeRatchetInventory:
    """Collect the canonical inventory from immutable Git blobs at ``ref``."""
    with _git_source_snapshot(repo_dir, ref) as (snapshot, paths):
        return collect_size_ratchet_inventory(snapshot, repo_paths=paths)


def module_is_grandfathered(path: str) -> bool:
    """Return whether the exact repo-relative path is current module debt."""
    return _exact_repo_relative_path(path) in GIANT_PATHS


def function_is_grandfathered(path: str, qualname: str) -> bool:
    """Return whether the exact (repo-relative path, qualname) key is debt."""
    return (_exact_repo_relative_path(path), str(qualname)) in FUNCTION_DEBT


def _literal_assignments(text: str) -> dict[str, Any]:
    tree = ast.parse(text, filename=SIZE_RATCHET_MANIFEST_PATH)
    values: dict[str, Any] = {}
    for index, node in enumerate(tree.body):
        if (
            index == 0
            and isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            continue
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            raise ValueError("size manifest is data-only: expected literal assignments")
        name = node.targets[0].id
        if name in values:
            raise ValueError(f"duplicate size manifest assignment: {name}")
        try:
            if isinstance(node.value, ast.Dict):
                keys = [ast.literal_eval(key) for key in node.value.keys]
                if len(keys) != len(set(keys)):
                    raise ValueError(f"size manifest dict contains duplicate keys: {name}")
            values[name] = ast.literal_eval(node.value)
        except (TypeError, ValueError) as exc:
            if str(exc).startswith("size manifest dict contains duplicate keys"):
                raise
            raise ValueError(f"size manifest assignment must be literal: {name}") from exc
    return values


def parse_size_ratchet_manifest(text: str) -> SizeRatchetManifest:
    """Parse the checked-in data-only manifest without executing it."""
    values = _literal_assignments(text)
    required = {
        "BASELINE_SOURCE_SHA",
        "GIANT_PATHS",
        "FUNCTION_DEBT",
        "BAND_BASELINE_PATHS",
        "BAND_PATHS",
        "BYTE_BASELINE_DEBT",
        "BYTE_DEBT",
    }
    missing = sorted(required - values.keys())
    if missing:
        raise ValueError(f"size manifest missing assignments: {', '.join(missing)}")
    extra = sorted(values.keys() - required)
    if extra:
        raise ValueError(f"size manifest has unexpected assignments: {', '.join(extra)}")

    def manifest_path(raw: str, label: str) -> str:
        try:
            return _exact_repo_relative_path(raw)
        except ValueError as exc:
            raise ValueError(f"{label} path is not canonical POSIX repo-relative: {raw!r}") from exc

    def exact_paths(items: Any, label: str) -> frozenset[str]:
        if not isinstance(items, tuple):
            raise ValueError(f"{label} must be a tuple")
        if not all(isinstance(item, str) for item in items):
            raise ValueError(f"{label} paths must be strings")
        paths = tuple(manifest_path(item, label) for item in items)
        if len(paths) != len(set(paths)):
            raise ValueError(f"{label} contains duplicate or colliding paths")
        for raw, path in zip(items, paths):
            if raw != path:
                raise ValueError(f"{label} path is not canonical POSIX repo-relative: {raw!r}")
        return frozenset(paths)

    giant_paths = exact_paths(values["GIANT_PATHS"], "GIANT_PATHS")
    band_baseline_paths = exact_paths(values["BAND_BASELINE_PATHS"], "BAND_BASELINE_PATHS")
    raw_functions = values["FUNCTION_DEBT"]
    if not isinstance(raw_functions, tuple):
        raise ValueError("FUNCTION_DEBT must be a tuple")
    function_items: list[tuple[str, str]] = []
    for item in raw_functions:
        if (
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not isinstance(item[1], str)
            or not item[1]
        ):
            raise ValueError("FUNCTION_DEBT entries must be (path, nonempty qualname) string pairs")
        rel = manifest_path(item[0], "FUNCTION_DEBT")
        function_items.append((rel, item[1]))
    if len(function_items) != len(set(function_items)):
        raise ValueError("FUNCTION_DEBT contains duplicate or colliding keys")
    for raw, (path, _qualname) in zip(raw_functions, function_items):
        if raw[0] != path:
            raise ValueError(f"FUNCTION_DEBT path is not canonical POSIX repo-relative: {raw[0]!r}")
    function_debt = frozenset(function_items)

    raw_band = values["BAND_PATHS"]
    if not isinstance(raw_band, dict):
        raise ValueError("BAND_PATHS must be a dict")
    band_paths: dict[str, str | None] = {}
    for path, rationale in raw_band.items():
        if not isinstance(path, str):
            raise ValueError("BAND_PATHS keys must be strings")
        rel = manifest_path(path, "BAND_PATHS")
        if rel in band_paths:
            raise ValueError(f"BAND_PATHS contains canonical path collision: {rel}")
        if path != rel:
            raise ValueError(f"BAND_PATHS path is not canonical POSIX repo-relative: {path!r}")
        if rationale is not None and (not isinstance(rationale, str) or not rationale.strip()):
            raise ValueError(f"BAND_PATHS rationale must be nonblank or None: {rel}")
        band_paths[rel] = rationale

    def byte_map(raw: Any, label: str) -> dict[str, int]:
        if not isinstance(raw, dict):
            raise ValueError(f"{label} must be a dict")
        parsed: dict[str, int] = {}
        for path, count in raw.items():
            if not isinstance(path, str):
                raise ValueError(f"{label} keys must be strings")
            rel = manifest_path(path, label)
            if rel in parsed:
                raise ValueError(f"{label} contains canonical path collision: {rel}")
            if path != rel:
                raise ValueError(f"{label} path is not canonical POSIX repo-relative: {path!r}")
            if isinstance(count, bool) or not isinstance(count, int) or count <= MAX_MODULE_BYTES:
                raise ValueError(f"{label} must store exact byte counts above {MAX_MODULE_BYTES}: {rel}")
            parsed[rel] = count
        return parsed

    sha = values["BASELINE_SOURCE_SHA"]
    if not isinstance(sha, str) or len(sha) != 40 or any(char not in "0123456789abcdef" for char in sha):
        raise ValueError("BASELINE_SOURCE_SHA must be a lowercase full commit SHA")
    return SizeRatchetManifest(
        baseline_source_sha=sha,
        giant_paths=giant_paths,
        function_debt=function_debt,
        band_baseline_paths=band_baseline_paths,
        band_paths=band_paths,
        byte_baseline_debt=byte_map(values["BYTE_BASELINE_DEBT"], "BYTE_BASELINE_DEBT"),
        byte_debt=byte_map(values["BYTE_DEBT"], "BYTE_DEBT"),
    )


def load_size_ratchet_manifest(path: pathlib.Path) -> SizeRatchetManifest:
    return parse_size_ratchet_manifest(path.read_text(encoding="utf-8"))


_CHECKED_IN_MANIFEST = load_size_ratchet_manifest(pathlib.Path(__file__).with_name("size_ratchet_manifest.py"))
GIANT_PATHS = _CHECKED_IN_MANIFEST.giant_paths
FUNCTION_DEBT = _CHECKED_IN_MANIFEST.function_debt
# Compatibility names remain public during the v7 migration; their keys are now exact.
GRANDFATHERED_OVERSIZED_MODULES = GIANT_PATHS
GRANDFATHERED_OVERSIZED_FUNCTIONS = FUNCTION_DEBT


def validate_manifest_transition(
    current: SizeRatchetManifest,
    previous: SizeRatchetManifest,
    *,
    adjacent: bool = True,
) -> list[str]:
    """Validate shrink-only debt and rationale authority against the parent tree.

    ``adjacent=False`` is the PAIRWISE (interval) form: byte-equality of a
    surviving band rationale is only sound between adjacent manifests — across
    an interval a path may legally retire and re-enter with a fresh rationale
    (every intermediate step green) — so the interval form requires a nonblank
    tip rationale instead of equality (Q18-A: interval archaeology is out of
    scope; interval FALSE POSITIVES are not)."""
    errors: list[str] = []
    if current.baseline_source_sha != previous.baseline_source_sha:
        errors.append("BASELINE_SOURCE_SHA is immutable")
    if current.band_baseline_paths != previous.band_baseline_paths:
        errors.append("BAND_BASELINE_PATHS is immutable")
    if dict(current.byte_baseline_debt) != dict(previous.byte_baseline_debt):
        errors.append("BYTE_BASELINE_DEBT is immutable")

    for path in sorted(current.giant_paths - previous.giant_paths):
        errors.append(f"new module debt above {MAX_MODULE_LINES} lines: {path}")
    added_functions = current.function_debt - previous.function_debt
    removed_functions = previous.function_debt - current.function_debt
    # A same-qualname relocation — the function left exactly one path and appeared
    # at exactly one other in the same transition — moves existing debt, it does
    # not create it: the count is unchanged and the ratchet still names the
    # function. A fresh >300-line function, a swap onto a different qualname, or
    # an ambiguous many-to-one move is still refused.
    relocated_functions = {
        (path, qualname)
        for path, qualname in added_functions
        if sum(1 for _p, q in removed_functions if q == qualname) == 1
        and sum(1 for _p, q in added_functions if q == qualname) == 1
    }
    for path, qualname in sorted(added_functions - relocated_functions):
        errors.append(f"new function debt above {MAX_FUNCTION_LINES} lines: {path}:{qualname}")

    previous_band = set(previous.band_paths)
    for path in sorted(set(current.band_paths) - previous_band):
        rationale = current.band_paths[path]
        if not isinstance(rationale, str) or not rationale.strip():
            errors.append(f"new or re-entered 1001-1500 path needs a nonblank rationale: {path}")
    for path in sorted(set(current.band_paths) & previous_band):
        old = previous.band_paths[path]
        new = current.band_paths[path]
        if adjacent:
            if new != old:
                errors.append(f"surviving band rationale is immutable: {path}")
        elif old is not None and (new is None or not new.strip()):
            # Interval form: a rewrite is legal (retire+re-enter), but DROPPING
            # a recorded rationale — to None or to a blank string — across the
            # interval launders the justification away.
            errors.append(f"band rationale dropped across the interval: {path}")

    for path in sorted(set(current.byte_debt) - set(previous.byte_debt)):
        errors.append(f"new or re-entered module debt above {MAX_MODULE_BYTES} UTF-8 bytes: {path}")
    for path in sorted(set(current.byte_debt) & set(previous.byte_debt)):
        if current.byte_debt[path] > previous.byte_debt[path]:
            errors.append(f"byte debt grew: {path} {previous.byte_debt[path]} -> {current.byte_debt[path]}")
    return errors


def _manifest_inventory_errors(
    manifest: SizeRatchetManifest,
    inventory: SizeRatchetInventory,
) -> list[str]:
    errors: list[str] = []

    def compare_set(label: str, live: frozenset[Any], recorded: frozenset[Any]) -> None:
        for item in sorted(live - recorded):
            errors.append(f"{label} missing live entry: {item!r}")
        for item in sorted(recorded - live):
            errors.append(f"{label} contains stale entry: {item!r}")

    compare_set("GIANT_PATHS", inventory.giant_paths, manifest.giant_paths)
    compare_set("FUNCTION_DEBT", inventory.function_debt, manifest.function_debt)
    compare_set("BAND_PATHS", inventory.band_paths, frozenset(manifest.band_paths))
    if dict(inventory.byte_debt) != dict(manifest.byte_debt):
        errors.append(f"BYTE_DEBT differs from live exact counts: live={dict(inventory.byte_debt)!r}")
    if len(inventory.functions) > MAX_TOTAL_FUNCTIONS:
        errors.append(
            f"total function count exceeds {MAX_TOTAL_FUNCTIONS}: {len(inventory.functions)}"
        )
    return errors


def _bootstrap_baseline_errors(
    manifest: SizeRatchetManifest,
    inventory: SizeRatchetInventory,
) -> list[str]:
    """A fresh bootstrap seeds its immutable baselines from its own candidate tree."""
    errors: list[str] = []
    if manifest.band_baseline_paths != inventory.band_paths:
        errors.append("BAND_BASELINE_PATHS differs from the bootstrap candidate inventory")
    if dict(manifest.byte_baseline_debt) != dict(inventory.byte_debt):
        errors.append("BYTE_BASELINE_DEBT differs from the bootstrap candidate inventory")
    return errors


def _git_show_manifest(repo_dir: pathlib.Path, ref: str, manifest_path: str) -> str | None:
    result = subprocess.run(
        ["git", "show", f"{ref}:{manifest_path}"],
        cwd=repo_dir,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout if result.returncode == 0 else None


def _staged_manifest_inventory(
    repo_dir: pathlib.Path,
    index_tree: str,
    manifest_path: str,
) -> tuple[str | None, SizeRatchetManifest | None, SizeRatchetInventory | None]:
    """Read one immutable staged projection and label any typed failure."""
    staged_text = _git_show_manifest(repo_dir, index_tree, manifest_path)
    if staged_text is None:
        return None, None, None
    try:
        staged = parse_size_ratchet_manifest(staged_text)
        inventory = collect_size_ratchet_inventory_at_ref(repo_dir, index_tree)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"staged: {exc}") from exc
    return staged_text, staged, inventory


def _commit_parents(repo_dir: pathlib.Path, commit: str) -> list[str]:
    """Exact parent SHAs of ``commit`` in parent order (empty for a root commit)."""
    listed = subprocess.run(
        ["git", "rev-list", "--parents", "-n", "1", commit],
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    return listed[1:]


def resolve_committed_manifest_text(
    repo_dir: pathlib.Path,
    *,
    manifest_path: str = SIZE_RATCHET_MANIFEST_PATH,
) -> str | None:
    """Merge-aware committed manifest authority for the checkout's ``HEAD``.

    The committed manifest from ``HEAD``'s tree; when that tree lacks it, the
    first parent (in parent order) whose tree carries it — a merge that landed
    the manifest through EITHER parent line keeps its authority, with no
    first-parent history replay. ``None`` means no committed manifest exists on
    ``HEAD`` or any of its parents: a bootstrap, accepted from the current tree
    (interval archaeology is an owner-accepted tradeoff — the official
    repository CI enforces the pairwise base-vs-tip transition instead).
    """
    root = pathlib.Path(repo_dir).resolve()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    text = _git_show_manifest(root, head, manifest_path)
    if text is not None:
        return text
    for parent in _commit_parents(root, head):
        text = _git_show_manifest(root, parent, manifest_path)
        if text is not None:
            return text
    return None


def _staged_tree_without_index_lock(root: pathlib.Path) -> str:
    """Return the staged tree id (``git write-tree``) without taking ``index.lock``.

    ``git write-tree`` rewrites the cache-tree extension into the index and
    therefore holds ``.git/index.lock`` with ``LOCK_DIE_ON_ERROR``: any
    concurrent ``git status`` / ``git diff`` refresh of the same checkout
    (parallel test workers, an agent shell, an editor) makes it die with exit
    128 even though the staged content is perfectly readable. The validator only
    needs the tree id, so it writes the tree from a private copy of the index
    (``GIT_INDEX_FILE``): same staged snapshot, no lock on the live index.
    An index without a checkout-level index file (bare/odd layouts) falls back
    to the plain command.
    """
    index_path = subprocess.run(
        ["git", "rev-parse", "--git-path", "index"],
        cwd=root, check=True, capture_output=True, text=True,
    ).stdout.strip()
    index_file = pathlib.Path(index_path)
    if not index_file.is_absolute():
        index_file = root / index_file
    if not index_file.is_file():
        return subprocess.run(
            ["git", "write-tree"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
    with tempfile.TemporaryDirectory(prefix="ouroboros-ratchet-index-") as tmp:
        private_index = pathlib.Path(tmp) / "index"
        private_index.write_bytes(index_file.read_bytes())
        env = dict(os.environ)
        env["GIT_INDEX_FILE"] = str(private_index)
        return subprocess.run(
            ["git", "write-tree"], cwd=root, check=True, capture_output=True, text=True, env=env
        ).stdout.strip()


def _validate_manifest_candidate(
    root: pathlib.Path,
    current_text: str,
    *,
    manifest_path: str,
    include_staged: bool,
) -> list[str]:
    """Shared exactness + merge-aware transition core for live and in-memory candidates."""
    current = parse_size_ratchet_manifest(current_text)
    inventory = collect_size_ratchet_inventory(root)
    errors = _manifest_inventory_errors(current, inventory)

    previous_text = resolve_committed_manifest_text(root, manifest_path=manifest_path)
    previous = parse_size_ratchet_manifest(previous_text) if previous_text is not None else None
    if previous is None:
        errors.extend(f"bootstrap: {error}" for error in _bootstrap_baseline_errors(current, inventory))
    elif current_text != previous_text:
        errors.extend(validate_manifest_transition(current, previous))

    if not include_staged:
        return errors

    head_tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    index_tree = _staged_tree_without_index_lock(root)
    if index_tree == head_tree:
        return errors
    staged_text, staged, staged_inventory = _staged_manifest_inventory(root, index_tree, manifest_path)
    if staged_text is None:
        if previous is None:
            errors.append("staged: size-ratchet bootstrap manifest is missing from the changed index")
        else:
            errors.append("staged: size-ratchet manifest was removed after bootstrap")
        return errors
    assert staged is not None and staged_inventory is not None
    errors.extend(f"staged: {error}" for error in _manifest_inventory_errors(staged, staged_inventory))
    if previous is None:
        errors.extend(
            f"staged bootstrap: {error}" for error in _bootstrap_baseline_errors(staged, staged_inventory)
        )
    elif staged_text != previous_text:
        errors.extend(f"staged: {error}" for error in validate_manifest_transition(staged, previous))
    return errors


def validate_size_ratchet(
    repo_dir: pathlib.Path,
    *,
    manifest_path: str = SIZE_RATCHET_MANIFEST_PATH,
) -> list[str]:
    """Validate live and staged candidates against the merge-aware committed authority.

    Enforcement contract: the OFFICIAL repository's CI ``size_ratchet`` lane
    BLOCKS on these findings (tip exactness plus the pairwise base-vs-tip
    transition); every local surface (default pytest lanes exclude the marker;
    ``check_worktree_readiness`` and ``codebase_health`` report the findings)
    only WARNS. There is no committed-history replay: the previous manifest
    resolves from ``HEAD`` or any of its parents, and a checkout with no
    committed manifest anywhere bootstraps from its own tree — a locally
    evolved fork is never trapped by structural debt it inherited.
    """
    root = pathlib.Path(repo_dir).resolve()
    current_path = root.joinpath(*pathlib.PurePosixPath(manifest_path).parts)
    current_text = current_path.read_text(encoding="utf-8")
    return _validate_manifest_candidate(root, current_text, manifest_path=manifest_path, include_staged=True)


def validate_size_ratchet_candidate(
    repo_dir: pathlib.Path,
    candidate_text: str,
    *,
    manifest_path: str = SIZE_RATCHET_MANIFEST_PATH,
) -> list[str]:
    """Validate one rendered in-memory manifest BEFORE it is written to disk.

    Pure candidate mode for ``scripts/regenerate_size_ratchet.py``: the same
    live-tree exactness, bootstrap, and shrink-only transition checks as
    ``validate_size_ratchet``, with no staged-index checks and no filesystem
    write first. The manifest module's own metrics sit far below every debt
    threshold today (an observation, not a structural guarantee), which keeps
    them out of the debt sets and makes validation against the current live
    inventory exact even while the old manifest bytes are still on disk.
    """
    root = pathlib.Path(repo_dir).resolve()
    return _validate_manifest_candidate(root, candidate_text, manifest_path=manifest_path, include_staged=False)


def validate_size_ratchet_transition_against_base(
    repo_dir: pathlib.Path,
    base_ref: str | None,
    *,
    manifest_path: str = SIZE_RATCHET_MANIFEST_PATH,
) -> list[str]:
    """Pairwise (interval-form) shrink-only check of HEAD's manifest vs a base.

    CI passes the event base (PR base.sha / push event.before) via
    ``OURO_SIZE_RATCHET_BASE_REF``. Base semantics — every hole here is a
    laundering vector, so degradations are deliberate and narrow: all-zeros
    (new-branch/tag push) degrades to HEAD's parent, never skips; a resolvable
    base WITHOUT the manifest FAILS (delete-then-rebootstrap laundering; true
    first adoption re-runs with a post-adoption base); empty/unresolvable
    (force-push loss) degrades to HEAD's parent whose manifest is first
    validated against the parent's own tree (fabricated bases are refused);
    no parent manifest at all = bootstrap, transition skipped."""
    root = pathlib.Path(repo_dir).resolve()
    tip_text = _git_show_manifest(root, "HEAD", manifest_path)
    if tip_text is None:
        return ["pairwise: HEAD does not carry the size-ratchet manifest"]

    ref = (base_ref or "").strip()
    if ref and not ref.strip("0"):
        ref = ""  # all-zeros: no event base — fall through to the parent degradation
    base_text: str | None = None
    validate_parent_tree = False
    if ref:
        resolved = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        if resolved.returncode == 0:
            base_text = _git_show_manifest(root, resolved.stdout.strip(), manifest_path)
            if base_text is None:
                return [
                    f"pairwise: base {resolved.stdout.strip()[:12]} does not carry the "
                    "size-ratchet manifest — a deletion inside the interval retires the "
                    "ratchet's memory; for genuine first adoption re-run with a "
                    "post-adoption base"
                ]
    if base_text is None:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
        for parent in _commit_parents(root, head):
            base_text = _git_show_manifest(root, parent, manifest_path)
            if base_text is not None:
                validate_parent_tree = True  # the degraded parent base is always tree-verified
                parent_ref = parent
                break
    if base_text is None:
        return []
    previous = parse_size_ratchet_manifest(base_text)
    if validate_parent_tree:
        # BEFORE the identical-text shortcut: a fabricated parent manifest that
        # pre-declares the tip's debt is byte-identical to the tip's manifest —
        # equality is exactly what the laundering constructs.
        try:
            parent_inventory = collect_size_ratchet_inventory_at_ref(root, parent_ref)
            parent_errors = _manifest_inventory_errors(previous, parent_inventory)
        except Exception as exc:
            parent_errors = [f"pairwise-parent: inventory failed: {type(exc).__name__}: {exc}"]
        if parent_errors:
            return [
                "pairwise: the degraded parent-base manifest does not match the parent's own tree "
                "(a fabricated base cannot authorize a transition): " + "; ".join(parent_errors[:3])
            ]
    if base_text == tip_text:
        return []
    return validate_manifest_transition(
        parse_size_ratchet_manifest(tip_text), previous, adjacent=False
    )


def _metrics_from_inventory(inventory: SizeRatchetInventory) -> Dict[str, Any]:
    modules = inventory.modules
    functions = inventory.functions
    func_lens = [item.line_count for item in functions]
    py_files = [item for item in modules if item.path.endswith(".py")]
    js_files = [item for item in modules if item.path.endswith(".js")]
    hard_modules = [item for item in modules if item.line_count > MAX_MODULE_LINES]
    hard_functions = [item for item in functions if item.line_count > MAX_FUNCTION_LINES]
    return {
        "total_files": len(modules),
        "py_files": len(py_files),
        "js_files": len(js_files),
        "total_lines": sum(item.line_count for item in modules),
        "total_bytes": sum(item.utf8_bytes for item in modules),
        "total_functions": len(functions),
        "avg_function_length": round(sum(func_lens) / max(1, len(func_lens)), 1) if func_lens else 0,
        "max_function_length": max(func_lens) if func_lens else 0,
        "largest_files": [
            (item.path, item.line_count) for item in sorted(modules, key=lambda x: x.line_count, reverse=True)[:10]
        ],
        "longest_functions": [
            (item.path, item.line_start, item.line_count)
            for item in sorted(functions, key=lambda x: x.line_count, reverse=True)[:10]
        ],
        "target_drift_functions": [
            (item.path, item.line_start, item.line_count)
            for item in functions
            if item.line_count > TARGET_FUNCTION_LINES
        ],
        "grandfathered_functions": [
            (item.path, item.line_start, item.line_count)
            for item in hard_functions
            if (item.path, item.qualname) in FUNCTION_DEBT
        ],
        "oversized_functions": [
            (item.path, item.line_start, item.line_count)
            for item in hard_functions
            if (item.path, item.qualname) not in FUNCTION_DEBT
        ],
        "target_drift_modules": [
            (item.path, item.line_count) for item in modules if item.line_count > TARGET_MODULE_LINES
        ],
        "grandfathered_modules": [(item.path, item.line_count) for item in hard_modules if item.path in GIANT_PATHS],
        "oversized_modules": [(item.path, item.line_count) for item in hard_modules if item.path not in GIANT_PATHS],
    }


def compute_repo_complexity_metrics(repo_dir: pathlib.Path) -> Dict[str, Any]:
    """Compute health metrics from the same production inventory as the hard gate."""
    return _metrics_from_inventory(collect_size_ratchet_inventory(repo_dir))


def compute_complexity_metrics(sections: List[Tuple[str, str]]) -> Dict[str, Any]:
    """Compatibility helper for callers that already hold in-memory sections."""
    modules: list[GatedModule] = []
    functions: list[GatedFunction] = []
    for raw_path, content in sections:
        rel = _section_repo_relative_path(raw_path)
        if not _is_gated_module_path(rel):
            continue
        raw = content.encode("utf-8")
        modules.append(GatedModule(rel, len(content.splitlines()), len(raw)))
        posix = pathlib.PurePosixPath(rel)
        if (
            rel.endswith(".py")
            and posix.name not in FUNCTION_COUNT_EXCLUDED_FILES
            and not any(part in _FUNCTION_SKIP_DIR_NAMES for part in posix.parts[:-1])
        ):
            try:
                tree = ast.parse(content, filename=rel)
            except SyntaxError as exc:
                raise ValueError(f"gated Python module has invalid syntax: {rel}") from exc
            functions.extend(_iter_lexical_functions(tree, rel))
    inventory = SizeRatchetInventory(
        modules=tuple(sorted(modules)),
        functions=tuple(sorted(functions)),
        giant_paths=frozenset(item.path for item in modules if item.line_count > MAX_MODULE_LINES),
        function_debt=frozenset(
            (item.path, item.qualname) for item in functions if item.line_count > MAX_FUNCTION_LINES
        ),
        band_paths=frozenset(
            item.path for item in modules if TARGET_MODULE_LINES < item.line_count <= BAND_MODULE_MAX_LINES
        ),
        byte_debt={item.path: item.utf8_bytes for item in modules if item.utf8_bytes > MAX_MODULE_BYTES},
    )
    return _metrics_from_inventory(inventory)


def collect_sections(
    repo_dir: pathlib.Path,
    drive_root: pathlib.Path,
) -> Tuple[List[Tuple[str, str]], Dict[str, Any]]:
    """Compatibility collector backed by the untruncated production inventory."""
    del drive_root
    root = pathlib.Path(repo_dir).resolve()
    modules = tuple(iter_gated_modules(root))
    sections = [(f"repo/{module.path}", module._source_text) for module in modules]
    return sections, {
        "files": len(sections),
        "chars": sum(len(content) for _path, content in sections),
        "omitted": 0,
    }
