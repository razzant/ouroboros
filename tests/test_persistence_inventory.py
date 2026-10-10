"""CPL-4 verify pair: every factual data/-path writer has a row in docs/PERSISTENCE.md.

The scanner AST-walks every runtime module (``ouroboros/``, ``supervisor/``,
``server.py``, ``launcher.py``) and collects every data-relative path it
constructs: ``/``-join chains rooted at a data-root expression, chains whose
leading literal is a known top-level data entity, ``pathlib.Path("literal")``
chain bases, ``drive_path("literal")`` calls, and source-handle stores with
literal or constant category, source id and extension arguments.

A segment is resolved to the name the SOURCE STATES — a module or imported
string constant, a literal ``Path`` chain, the literal text of an f-string, a
prefix a runtime helper returns, a variable or loop target inside one function
— and only then, having run out of facts, collapses to ``*``. That order is
the point: a durable file whose name sits in a module constant used to
normalize to ``*`` and vanish from the forward check, which is how twelve live
entities held no inventory row while this test passed.

Contract (count-anchored both ways):

- every scanned path the scan can NAME must be covered by a backticked path
  pattern in the first column of a PERSISTENCE.md inventory row;
- a family wildcard proves nothing about a name: an unresolved scan segment is
  certified only by a row segment that spells its own wildcard, never by a
  literal row (``state/*`` is not evidence about ``state/state.json``). The
  complete set of still-unresolvable spellings is audited in
  ``UNRESOLVED_SPELLINGS`` and asserted by equality;
- every inventory row must still name something the scan sees (no stale rows).
  This direction uses a STRICTER matcher than the forward one: a scan path
  shorter than the row does not certify it, because the population also holds
  bare top-level tokens (``state``, ``logs``, ``archive`` …) that would keep
  any invented exact row alive — that is what made the check vacuous. The
  complete set of rows admitted with no in-tree writer is
  ``STALE_ROW_EXEMPTIONS``, asserted by equality;
- the total number of distinct scanned paths is pinned — adding a NEW
  data-relative path fails here until PERSISTENCE.md gets its row and the pin
  moves.
"""

from __future__ import annotations

import ast
import fnmatch
import functools
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parent.parent
DOC = REPO / "docs" / "PERSISTENCE.md"

# --- scanner ---------------------------------------------------------------

DATA_ROOT_MARKERS = (
    "DATA_DIR", "data_dir", "data_root", "drive_root", "canonical_data_root",
    "state_drive_root", "budget_drive_root", "state_dir",
)
DATA_REL_CALL_NAMES = frozenset({"drive_path"})
PATH_CTORS = frozenset({"Path", "PurePath", "PurePosixPath", "PureWindowsPath"})
DIR_LISTING_CALLS = frozenset({"iterdir", "glob", "rglob"})
TOP_LEVEL = frozenset({
    "state", "logs", "memory", "skills", "task_results", "observability",
    "locks", "archive", "settings.json", "uploads", "services", "artifacts",
    "task_drives", "tmp_scripts", "playwright-browsers", "cache", "tmp",
    "projects", "task_trees",
})

# Chains whose sub-root the AST cannot reach even through constants, returned
# prefixes and directory listings — the callee takes the parent directory as a
# PARAMETER, so only a call-site audit says where it lives. Each alias names
# the canonical data-relative location, audited against every caller. Keep
# this table SHORT: a growing list means writers are drifting away from
# recognizable root expressions, and an alias is a human promise, not a fact
# the scan re-derives.
SUBROOT_ALIASES = {
    # _run_go_skill's only caller passes skill_state_dir(drive_root, loaded.name).
    # These are per-skill caches, not directories at the canonical data root.
    "go": "state/skills/*/go",
    "go-cache": "state/skills/*/go-cache",
    # mint_skill_token(state_dir=...): all three call sites (extension_process_
    # runner.py:291, extension_plugin_api.py:870/959 via PluginAPI._state_dir)
    # pass a skill_state_dir(drive_root, name).
    "auth_token.json": "state/skills/*/auth_token.json",
    # PluginAPI.skill_job_dir builds on the same _state_dir: every
    # _PluginAPIConfig in extension_loader.py is constructed with
    # skill_state_dir[_path](drive_root, skill.name).
    "jobs/*": "state/skills/*/jobs/*",
    "jobs/*/*": "state/skills/*/jobs/*/*",
    # sweep_uninstalled_skill_state (skill_uninstall_state.py:65) writes
    # `state_dir / UNINSTALL_TOMBSTONE_FILENAME` for each entry of its own
    # `drive_root / "state" / "skills"` listing, and write_uninstall_tombstone
    # (:39) spells the SAME file as `skill_state_dir(drive_root, name) /
    # UNINSTALL_TOMBSTONE_FILENAME` — which the scan already resolves to the
    # aliased path, so the two spellings collapse onto one entity.
    "uninstalled.json": "state/skills/*/uninstalled.json",
    # _stage_extension_import_tree (extension_import_staging.py:84-88) builds
    # `state_dir / "__extension_imports" / f"{pid}-{uuid}"` plus its `skill`
    # subtree; its ONE caller (extension_loader.py:649) passes the
    # `state_dir = skill_state_dir(drive_root, skill.name)` bound at :581, and
    # the sweep (:120) spells the same directory through skill_state_dir.
    "__extension_imports/*-*": "state/skills/*/__extension_imports/*-*",
    "__extension_imports/*-*/skill": "state/skills/*/__extension_imports/*-*/skill",
    "__extension_imports/*-*/skill/*": "state/skills/*/__extension_imports/*-*/skill/*",
}

# The same human promise for a parameter whose NAME is the audited root, read
# BEFORE the chain is built (so the plane is named right instead of being
# mis-rooted and relocated afterwards). Keep it as short as the table above.
PARAM_SUBROOTS = {
    # sanitize_task_for_event(task, drive_logs, ...) — utils.py:1160 writes
    # `drive_logs / "tasks" / f"task_{id}.txt"`. Every `drive_logs` in the tree
    # is `env.drive_path("logs")` (agent.py:442/795, agent_startup_checks.py:860)
    # or `ctx.drive_logs`/`drive_root / "logs"` (commit_gate.py:907) threaded
    # down unchanged; nothing else binds the name.
    "drive_logs": "logs",
}


def _call_name(node: ast.Call) -> str:
    fn = node.func
    return fn.attr if isinstance(fn, ast.Attribute) else (
        fn.id if isinstance(fn, ast.Name) else "")


def _literal_segment(node: ast.expr) -> str | None:
    """A path segment the source states literally: ``"x"`` or an f-string.

    An f-string keeps its literal text and collapses each interpolation to
    ``*`` (``f"{pid}-{uuid}.json"`` -> ``*-*.json``): the constant part is a
    fact about the file NAME, and losing it was how whole families read as a
    bare ``*``.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        text = "".join(
            part.value if isinstance(part, ast.Constant) and isinstance(part.value, str)
            else "*"
            for part in node.values
        )
        return re.sub(r"[*]+", "*", text)
    return None


def _literal_path_chain(node: ast.expr) -> str | None:
    """``Path("task_results") / "artifacts"`` -> ``task_results/artifacts``.

    ``None`` when any segment is not literal — a partially literal constant is
    not a name the inventory can be held to.
    """
    parts, cur = [], node
    while isinstance(cur, ast.BinOp) and isinstance(cur.op, ast.Div):
        seg = _literal_segment(cur.right)
        if seg is None:
            return None
        parts.append(seg)
        cur = cur.left
    if isinstance(cur, ast.Call) and _call_name(cur) in PATH_CTORS and cur.args:
        seg = _literal_segment(cur.args[0])
        if seg is None:
            return None
        parts.append(seg)
    elif isinstance(cur, ast.Constant) and isinstance(cur.value, str):
        parts.append(cur.value)
    else:
        return None
    parts.reverse()
    return _normalize(parts) or None


def _module_constants(tree: ast.AST, *, exact_strings: bool = False) -> dict[str, str]:
    """``NAME`` -> its literal string/path value, when the file binds it once.

    A name the file rebinds (or binds to something non-literal) resolves to
    nothing: an ambiguous constant must not become a claimed file name.
    ``exact_strings`` limits source-store arguments to module string literals,
    excluding path expressions, partially known f-strings and function locals.
    """
    seen: dict[str, set] = {}
    for node in (_own_nodes(tree) if exact_strings else ast.walk(tree)):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value = node.value
        literal = None
        if exact_strings:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                literal = value.value
        elif value is not None:
            literal = _literal_segment(value)
            if literal is None:
                literal = _literal_path_chain(value)
        for target in targets:
            if isinstance(target, ast.Name):
                seen.setdefault(target.id, set()).add(literal)
    return {
        name: next(iter(values))
        for name, values in seen.items()
        if len(values) == 1 and next(iter(values)) is not None
    }


def _own_nodes(scope: ast.AST) -> list[ast.AST]:
    """Every node of one scope, NOT descending into nested function bodies.

    Return statements of a nested helper describe that helper, not its parent;
    attributing them to the parent made both prefixes ambiguous and dropped.
    """
    out: list[ast.AST] = []

    def walk(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            out.append(child)
            walk(child)

    walk(scope)
    return out


def _scanned_files(root: pathlib.Path) -> list[pathlib.Path]:
    return sorted(
        p
        for p in list(root.glob("ouroboros/**/*.py"))
        + list(root.glob("supervisor/**/*.py"))
        + [root / "server.py", root / "launcher.py"]
        if "__pycache__" not in p.parts
    )


def _root_is_data(node: ast.expr) -> bool:
    for sub in ast.walk(node):
        name = None
        if isinstance(sub, ast.Name):
            name = sub.id
        elif isinstance(sub, ast.Attribute):
            name = sub.attr
        if name and any(marker in name for marker in DATA_ROOT_MARKERS):
            return True
    return False


def _normalize(parts) -> str:
    out = []
    for part in parts:
        if part == "*":
            out.append("*")
            continue
        for seg in str(part).split("/"):
            seg = seg.strip()
            if seg:
                out.append(seg)
    return "/".join(out)


def _is_named(rel: str) -> bool:
    """A path that names something — not an empty or all-wildcard chain."""
    return bool(rel) and bool(rel.replace("*", "").replace("/", "").strip())


class _PathResolver:
    """Resolves the data-relative paths one source tree constructs.

    Three resolution planes, each a fact the source states:

    * **constants** — a segment spelled as a module constant (own file first,
      then the repo-wide map, which covers imported names and ``mod.NAME``);
    * **returned prefixes** — a function whose returns all resolve to one
      data-relative path lends that prefix to its callers, file-local first
      then repo-wide-if-unambiguous (fixed point, so ``skill_state_dir`` ->
      ``skill_state_dir_path`` chains resolve);
    * **local flow** — inside one function, a variable assigned a resolved
      path, and a loop variable bound to ``<resolved>.iterdir()/.glob()``;
    * **parents** — ``.parent`` of any path the planes above already named is
      that path minus its leaf, so ``<helper>().parent / "file.json"`` lands in
      the helper's own directory instead of reading as an unrooted leaf.

    Anything still unresolved stays ``*`` — a family, never a name.
    """

    def __init__(self, root: pathlib.Path) -> None:
        self.trees: dict[pathlib.Path, ast.AST] = {}
        self.consts: dict[pathlib.Path, dict[str, str]] = {}
        self.source_consts: dict[pathlib.Path, dict[str, str]] = {}
        repo_wide: dict[str, set] = {}
        for path in _scanned_files(root):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (OSError, SyntaxError):
                continue
            self.trees[path] = tree
            self.consts[path] = _module_constants(tree)
            self.source_consts[path] = _module_constants(tree, exact_strings=True)
            for name, value in self.consts[path].items():
                repo_wide.setdefault(name, set()).add(value)
        self.repo_consts = {
            name: next(iter(values))
            for name, values in repo_wide.items()
            if len(values) == 1
        }
        self.source_modules = {
            ".".join(path.relative_to(root).with_suffix("").parts): values
            for path, values in self.source_consts.items()
        }
        self.functions = {
            path: [n for n in ast.walk(tree)
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            for path, tree in self.trees.items()
        }
        self.prefixes: dict[str, str] = {}
        self.file_prefixes: dict[pathlib.Path, dict[str, str]] = {}
        # The file whose scopes are being resolved right now: name/attribute
        # prefixes are file-local facts (see ``_base_prefix``).
        self.current: pathlib.Path | None = None
        # Scope -> its own nodes. Keyed by identity and owned by THIS resolver,
        # so it dies with the parsed trees it describes; the prefix fixed point
        # would otherwise re-walk every scope once per round.
        self._nodes: dict[int, list] = {}
        self._resolve_prefixes()

    def own_nodes(self, scope: ast.AST) -> list[ast.AST]:
        cached = self._nodes.get(id(scope))
        if cached is None:
            cached = _own_nodes(scope)
            self._nodes[id(scope)] = cached
        return cached

    # -- segments and chains ------------------------------------------------

    def _segment(self, node: ast.expr, consts: dict[str, str]) -> str:
        literal = _literal_segment(node)
        if literal is not None:
            return literal
        if isinstance(node, ast.Name):
            value = consts.get(node.id) or self.repo_consts.get(node.id)
            if value:
                return value
        if isinstance(node, ast.Attribute):
            value = self.repo_consts.get(node.attr)
            if value:
                return value
        return "*"

    def _chain(self, node: ast.expr, consts: dict[str, str]):
        parts, cur = [], node
        while isinstance(cur, ast.BinOp) and isinstance(cur.op, ast.Div):
            parts.append(self._segment(cur.right, consts))
            cur = cur.left
        if isinstance(cur, ast.Call) and _call_name(cur) in PATH_CTORS and cur.args:
            literal = _literal_segment(cur.args[0])
            if literal is not None:
                parts.append(literal)
        parts.reverse()
        return cur, parts

    def _base_prefix(self, base: ast.expr, consts: dict[str, str],
                     local: dict[str, str]) -> str | None:
        """The data-relative prefix a chain base already stands for.

        A CALL takes the current file's function of that name first, then a
        repo-wide name that resolves to ONE path. A bare name or attribute
        takes ONLY the current file: ``self._state_dir`` is instance state
        whose name may collide with an unrelated module's helper (it does —
        ``workspace_executor._state_dir`` — and resolving it repo-wide claimed
        the wrong plane for the per-skill jobs dir).

        ``.parent`` is the one attribute read as an operation rather than a
        name: the parent of a path this resolver already named is itself a
        named plane (``state/reviewer_slot_last_execution.json`` -> ``state``),
        which is how ``<helper>().parent / "file.json"`` writers become
        visible. A single-segment path has no named parent — that is the data
        root — so it yields nothing rather than an empty prefix.
        """
        own = self.file_prefixes.get(self.current, {})
        if isinstance(base, ast.Attribute) and base.attr == "parent":
            inner, ok = self.resolve(base.value, consts, local)
            if ok and _is_named(inner) and "/" in inner:
                return inner.rsplit("/", 1)[0]
            return None
        if isinstance(base, ast.Call):
            name = _call_name(base)
            return own.get(name) or self.prefixes.get(name)
        if isinstance(base, ast.Name):
            return (local.get(base.id) or own.get(base.id)
                    or PARAM_SUBROOTS.get(base.id))
        if isinstance(base, ast.Attribute):
            return own.get(base.attr)
        return None

    def resolve(self, node: ast.expr, consts: dict[str, str],
                local: dict[str, str]) -> tuple[str, bool]:
        """``(data-relative path, is-data-relative)`` for a path expression."""
        if isinstance(node, ast.Name):
            known = local.get(node.id)
            return (known, True) if known else ("", False)
        base, parts = self._chain(node, consts)
        rel = _normalize(parts)
        prefix = self._base_prefix(base, consts, local)
        if prefix:
            return (f"{prefix}/{rel}" if rel else prefix), True
        if _root_is_data(base):
            return rel, True
        if rel and rel.split("/")[0] in TOP_LEVEL:
            return rel, True
        return rel, False

    # -- scopes -------------------------------------------------------------

    def _locals(self, scope: ast.AST, consts: dict[str, str],
                nodes: list[ast.AST]) -> dict[str, str]:
        local: dict[str, str] = {}
        if self.current.name == "image_preparation.py" and getattr(scope, "name", "") == "retain_original":
            # Both callers (context.build_user_content and tools.vision's loader)
            # pass their data owner's uploads/views. Bind only this parameter,
            # never a generic directory name in other writers.
            local["directory"] = "uploads/views"
        created_dirs = {node.func.value.id for node in nodes
                        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "mkdir" and isinstance(node.func.value, ast.Name)}
        for node in nodes:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                    and isinstance(node.targets[0], ast.Name):
                value = node.value
                if (node.targets[0].id in created_dirs and isinstance(value, ast.Call)
                        and isinstance(value.func, ast.Attribute) and value.func.attr == "resolve"
                        and isinstance(value.func.value, ast.BinOp) and isinstance(value.func.value.op, ast.Div)
                        and self._segment(value.func.value.right, consts) in TOP_LEVEL):
                    # A constructor creates this resolved directory. Its declared
                    # store prefix survives canonicalization; an arbitrary locator
                    # reader's resolve() is not a declaration of another store.
                    value = value.func.value
                rel, ok = self.resolve(value, consts, local)
                if ok and _is_named(rel):
                    local[node.targets[0].id] = rel
            elif isinstance(node, ast.For) and isinstance(node.target, ast.Name) \
                    and isinstance(node.iter, ast.Call) \
                    and _call_name(node.iter) in DIR_LISTING_CALLS \
                    and isinstance(node.iter.func, ast.Attribute):
                rel, ok = self.resolve(node.iter.func.value, consts, local)
                if not (ok and _is_named(rel)):
                    continue
                leaf = "*"
                if node.iter.args:
                    literal = _literal_segment(node.iter.args[0])
                    if literal:
                        leaf = literal
                local[node.target.id] = f"{rel}/{leaf}"
        return local

    def _scopes(self, path: pathlib.Path):
        consts = self.consts[path]
        for fn in self.functions[path]:
            nodes = self.own_nodes(fn)
            yield fn, consts, nodes, self._locals(fn, consts, nodes)
        module_nodes = self.own_nodes(self.trees[path])
        yield (self.trees[path], consts, module_nodes,
               self._locals(self.trees[path], consts, module_nodes))

    def _resolve_prefixes(self) -> None:
        """Fixed point over ``<function> -> <data-relative prefix>``.

        Two maps, because a name can be honest locally and ambiguous globally:
        the file map answers for a function DEFINED in the calling file (both
        ``managed_runtime_root``s resolve, each to its own runtime root), the
        repo map only for a name that resolves to ONE path tree-wide (that is
        what makes an imported ``skill_state_dir`` usable everywhere).
        """
        for _round in range(6):
            per_file: dict[pathlib.Path, dict[str, set]] = {}
            for path in self.trees:
                self.current = path
                consts = self.consts[path]
                found: dict[str, set] = {}
                for fn in self.functions[path]:
                    nodes = self.own_nodes(fn)
                    local = self._locals(fn, consts, nodes)
                    for node in nodes:
                        if not isinstance(node, ast.Return) or node.value is None:
                            continue
                        rel, ok = self.resolve(node.value, consts, local)
                        if ok and _is_named(rel):
                            found.setdefault(fn.name, set()).add(rel)
                per_file[path] = found
            repo_wide: dict[str, set] = {}
            for found in per_file.values():
                for name, rels in found.items():
                    repo_wide.setdefault(name, set()).update(rels)
            settled = {
                name: next(iter(rels))
                for name, rels in repo_wide.items() if len(rels) == 1
            }
            file_settled = {
                path: {
                    name: next(iter(rels))
                    for name, rels in found.items() if len(rels) == 1
                }
                for path, found in per_file.items()
            }
            if settled == self.prefixes and file_settled == self.file_prefixes:
                return
            self.prefixes = settled
            self.file_prefixes = file_settled

    # -- the scan -----------------------------------------------------------

    def _source_handle_path(self, node: ast.Call, scope: ast.AST) -> str | None:
        """Name literal source-store calls by the artifact writer's contract.

        Unlike a path f-string, a dynamic source id passes through sanitation
        and truncation before becoming a filename. Only exact strings can be
        resolved here; a guessed wildcard must not stand in for those bytes.
        """
        shadowed = ({arg.arg for arg in ast.walk(scope.args) if isinstance(arg, ast.arg)}
                    if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)) else set())
        if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
            shadowed.update(node.id for node in self.own_nodes(scope)
                            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store))
        own_bindings = {node.id for node in self.own_nodes(self.trees[self.current])
                        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}
        imports = {}
        for item in self.own_nodes(self.trees[self.current]) + self.own_nodes(scope):
            if isinstance(item, ast.ImportFrom) and item.level == 0:
                for alias in item.names:
                    imports.setdefault(alias.asname or alias.name, set()).add((item.module, alias.name))

        def exact_string(value: ast.expr) -> str | None:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                return value.value
            if isinstance(value, ast.Name) and value.id not in shadowed:
                if value.id in self.source_consts[self.current]:
                    return self.source_consts[self.current][value.id]
                if value.id not in own_bindings and len(imports.get(value.id, ())) == 1:
                    module, name = next(iter(imports[value.id]))
                    return self.source_modules.get(module, {}).get(name)
            return None

        args = {arg.arg: exact_string(arg.value) for arg in node.keywords}
        category, source_id, extension = (args.get(key) for key in
                                          ("category", "source_id", "extension"))
        if category is None or source_id is None or extension is None:
            return None
        # artifacts.store_actor_source_bytes owns this layout and sanitation.
        category = category.strip()
        if category not in {"context_checkpoints", "tool_results", "delegated_activity"}:
            return None
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", source_id or "source").strip("._")
        safe_id = safe_id[:160] or "source"
        safe_extension = re.sub(r"[^A-Za-z0-9]+", "", extension or "bin")[:12] or "bin"
        return (f"task_results/artifacts/*/source_handles/{category}/"
                f"{safe_id}-*.{safe_extension}")

    def paths(self) -> set[str]:
        found: set[str] = set()
        for path in self.trees:
            self.current = path
            for scope, consts, nodes, local in self._scopes(path):
                consumed: set[int] = set()
                for node in nodes:
                    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                        if id(node) in consumed:
                            continue
                        cur = node
                        while isinstance(cur, ast.BinOp) and isinstance(cur.op, ast.Div):
                            if isinstance(cur.left, ast.BinOp):
                                consumed.add(id(cur.left))
                            cur = cur.left
                        rel, ok = self.resolve(node, consts, local)
                        if ok and _is_named(rel):
                            found.add(SUBROOT_ALIASES.get(rel, rel))
                    elif isinstance(node, ast.Call) \
                            and _call_name(node) in DATA_REL_CALL_NAMES and node.args:
                        literal = _literal_segment(node.args[0])
                        if literal:
                            rel = _normalize([literal])
                            if rel:
                                found.add(SUBROOT_ALIASES.get(rel, rel))
                    elif isinstance(node, ast.Call) and _call_name(node) == "store_actor_source_bytes":
                        rel = self._source_handle_path(node, scope)
                        if rel:
                            found.add(rel)
        return found


@functools.lru_cache(maxsize=4)
def scan_data_paths(root: pathlib.Path = REPO) -> frozenset[str]:
    paths = set(_PathResolver(root).paths())
    # The linked-knowledge owner derives the project shelf from a validated
    # project id, then appends its history beside that shelf.  Keep these
    # canonical dynamic siblings visible to the inventory audit even though
    # the AST resolver cannot expand the address object's validated prefix.
    # rebuild_knowledge_index writes shelf / INDEX_FILE for either shelf;
    # context's shared reader no longer spells the project index path itself.
    if (root / "ouroboros" / "knowledge.py").exists():
        paths.update({
            "projects/*/knowledge/*.md",
            "projects/*/knowledge/index-full.md",
            "projects/*/knowledge_history.jsonl",
        })
    return frozenset(paths)


# The pinned scan-population size. Moving it is deliberate: a new distinct
# data-relative path requires a PERSISTENCE.md row AND this bump in one diff.
# 124 -> 271: the stage-2 fix wave taught the scanner to resolve what the
# source states — module/imported string constants, literal ``Path`` chains,
# f-string names, returned prefixes of runtime helpers and one-scope local
# flow — instead of collapsing every non-literal segment to ``*``. The old
# population was not 124 entities but 124 SPELLINGS, most of them family
# wildcards that hid exact durable files behind them (12 of those files had no
# inventory row and the forward check could not see them).
# 271 -> 281: ``<helper>().parent`` became a named root (see ``_base_prefix``),
# which surfaced ``state/reviewer_slot_api_fallback.json`` — a live durable
# disclosure record that had no inventory row — plus nine already-documented
# spellings under ``state/cx/`` and ``projects/*``.
# 281 -> 284: giving the no-stale-rows direction a matcher that actually bites
# left three rows unbacked; two were real writers the scan simply could not
# see, and resolving them (module-scope local flow for the stdlib rotating
# handlers, ``PARAM_SUBROOTS`` for the oversized-task-text sink) added
# ``logs/server.log``, ``logs/launcher.log`` and ``logs/tasks/*``.
# 284 -> 283: aliasing the parameter-rooted ``uninstalled.json`` onto
# ``state/skills/*/uninstalled.json`` merged it with the spelling the scan
# already resolved; the three ``__extension_imports`` spellings moved plane
# without changing count.
# 283 -> 285: absorbing upstream added two durable planes — the skill-review
# root-task projection with its gaps ledger (``state/skill_review_root_tasks*``)
# and the per-project retirement locks (``state/delegate_project_retirements/``)
# — while the retired acceptance api-fallback record left the population.
# 290 -> 291: the supervisor's off-lock projection of live direct-chat roots
# (``state/direct_roots.json``, ``supervisor/direct_roots.py``) is the one new
# durable plane of the structural-health train; it has its own row in section 2.
# 291 -> 290: the Background Consciousness redesign retired the observation inbox
# and its startup fold; the archive segments ``archive/consciousness_observations_<ts>``
# left the population and the one-time ``archive/consciousness_observations.jsonl``
# move target (``ouroboros/consciousness.py``) took their place.
# 290 -> 291: the retained focus source (``task_results/artifacts/*/source_handles/
# context_checkpoints``, ``ouroboros/task_finalization.py``'s digest glob) is the
# one new durable plane of cross-focus awareness; it has its own row in section 2.
# 291 -> 292: the streamed bytes blob's temp name under observability/blobs (the existing
# ``observability/{calls,blobs,salvaged}/**`` row covers it).
# 292 -> 293: the Presence previous-turn pointer (``state/presence_turn_gate/last-<sha256>.json``),
# one rebuildable projection per conversation written by presence_runner at the end of an executed
# turn; it has its own row in section 2.
# 293 -> 295: the disposable test-environment caches (``cache/pip``, ``cache/uv``; test root only).
# 295 -> 294: TZ-3 removed the destructive memory journal rewrite and its
# ``.compact.tmp`` sibling path; PERSISTENCE.md keeps the journals, now
# read-only observed and never age-digested.
# 294 -> 296 (upstream 7.5.0): the merged extra-CA bundle is content-addressed under
# ``state/extra-ca-bundle/`` (the directory and its ``*.pem`` members, so a changed
# owner PEM rotates every cache); one section-2 row covers both.
# 296 -> 297 (Presence resilience): Presence recovery inspects the retained quarantine
# members (``task_results/quarantine/*``) before deciding whether an event ever started.
# 297 -> 296 (TZ-3 PR-1): the ``knowledge_journal.jsonl`` size-telemetry writer is
# removed (its only reader was this inventory); ``knowledge_history.jsonl`` keeps the
# complete captures, now host-stamped.
# 296 -> 300 (TZ-1 child-drive custody): ``task_results/<id>.custody.lock`` (the per-task
# custody lock) and the settlement's ``state/custody_staging`` / ``state/custody_trash``
# entries; one PERSISTENCE.md row covers all three.
# 300 -> 303: state-initialization witness plus named review source and review_inputs.
# 303 -> 305: immutable retention names exact text-CAS manifest versions and the
# existing blob copy destination; both stay under the documented observability store.
# 308 -> 312 (memory chronicle store): ``memory/chronicle`` with its append-only
# ``records.jsonl``, the disposable ``index.sqlite3`` and the ``.publication.lock``; one
# section-6 row each (the directory is answered by the deeper rows).
# 312 -> 313 (memory legacy import): ``memory/.consolidation.lock``, the legacy
# writer's flock that the one-time chronicle import also takes, now spelled from the data
# root; its own section-6 row.
# 313 -> 312 (memory view): the request no longer reads ``memory/dialogue_summary.md``
# (the retired flat summary is a chronicle legacy record, imported with the blocks; its row merged
# into theirs).
# 312 -> 311 (memory writers): the removed memory pressure upkeep was the only code
# spelling ``memory/knowledge/overview.md``; the mind still writes the note through the knowledge
# shelf, whose section-6 row covers it.
# 311 -> 312 (memory writers): ``memory/chronicle/.fallback.lock``, the fallback memory
# writer's no-wait lock between two roots' drafts; its own section-6 row (its refusal receipts live in the
# journal's scan state, the ``records.jsonl`` row).
# 312 -> 314 (target merge 8d612997c): the two paths the target added on its own
# 308 -> 310 step land beside the memory change's four.
# 314 -> 315 (C4 restart/exit): ``state/owned_processes.json``, the installation's ownership set the
# exit stop reads instead of walking data/state; its own section-2 row.
# 315 -> 318 (C3 current obligations): state/obligations/*.json, its shared
# state/obligations.lock, and state/migrations.json; section-2 rows own all three.
# 318 -> 320 (owned-stop review fix): ``state/owned_processes.pending`` and its lock-free
# ``*.*.json`` registrations written when the custody lock is held; one section-2 row.
# 320 -> 321 (obligations rebuild): ``state/obligations/rebuild.owed``, left by a transition whose
# set write failed so the next start rebuilds the sets; it joins the obligations section-2 row.
# 321 -> 322 (#1536 reentry): ``task_results/artifacts/*/source_handles/
# context_checkpoints/presence-reentry-*.json``, retained by presence_continuation
# through store_actor_source_bytes and resolved by its digest reader; its own section-7 row.
# 322 -> 337 (#1536 source-store call facts): literal/constant arguments expose
# presence-transport-queue plus these existing context checkpoints: acceptance,
# acceptance-artifact-inventory, acceptance-operation, acceptance-packet,
# acceptance_historical, attachments, completion, historical-acceptance-intent,
# historical_owner_request, plan-spec, task_input, terminal-delivery and view;
# tool_results also gains acceptance_tool_trajectory. Existing section-7 rows
# cover these immutable sources; no stale-row or unresolved-path exemption.
# Own-body candidates add two existing-owner state paths alongside the retained source store.
# 339 -> 345 (review ledger): ``state/review_ledger`` with its per-wave ``<record_id>.json``
# records, the hot ``index.jsonl`` and its rotated ``index.<stamp>[_n].jsonl`` segments
# (``ouroboros/review_ledger.py``; one section-4 row), plus ``locks/review_ledger.lock``,
# which the existing ``locks/**`` sidecar row covers.
# 345 -> 347 (2026-10-07): the isolated review checkout of a frozen review subject,
# ``state/review_checkouts/<token>`` and its ``repo`` worktree
# (``ouroboros/tools/review_subject.py`` ``isolated_checkout``; one section-4 row).
# 347 -> 346 (2026-10-08): the scope-delivery migration disclosure
# ``state/scope_delivery_migration.json`` is gone with the scope role — the one
# brief has two parts and no separate scope wave to migrate (its PERSISTENCE.md row
# was removed with the writer).
# 346 -> 348 (review pool, PR-3): the review-lane -> review-pool migration receipt
# ``state/review_migrations/<ts>-slots-to-pool.json`` and its directory
# (``ouroboros/review_pool_receipts.py`` ``persist_receipts``, written by the process that
# saves the migrated document — FIX6b moved it there from the supervisor boot; one section-2 row).
# 348 -> 349 (#1610): first/latest usable author inputs retained by
# ``context_input_selection._historical_anchor`` in
# ``task_results/artifacts/*/source_handles/context_checkpoints/historical-author-input-*.json``
# (one section-7 row; evidence, not current task authority).
# 349 -> 351 (#1563 saved work): the rolling per-attempt working checkpoint,
# ``task_results/artifacts/<task>/source_handles/working`` and its ``*.json`` file
# (``ouroboros/working_checkpoint.py``; one row beside the source-handle rows).
# Event waits add one exact immutable wait-handoff source family below.
EXPECTED_SCAN_PATHS = 352

# Scanned paths that must always be present — guards the scanner itself
# against a silent regression that would shrink coverage while keeping counts
# plausible. The constant-named files are the stage-2 sentinels: each one was
# invisible to the scan while non-literal segments collapsed to ``*``.
SENTINELS = frozenset({
    "settings.json",
    "state/state.json",
    "state/queue_snapshot.json",
    "state/process_ledger.jsonl",
    "state/consciousness_observations.jsonl",
    "state/skills/*/review_history.jsonl",
    "logs/events.jsonl",
    "logs/chat.jsonl",
    "memory/identity.md",
    "task_results/artifacts/*",
    "task_results/artifacts/*/source_handles/context_checkpoints/presence-reentry-*.json",
    "task_results/artifacts/*/source_handles/context_checkpoints/presence-transport-queue-*.json",
    "task_trees/*/blackboard.jsonl",
    "uploads",
    # stage-2: names that live in constants, helper returns or f-strings
    "state/claudexor_rotation_provisioning.json",
    "state/delegate_terminal_refresh_cursor.json",
    "state/extension_generation.json",
    "state/post_task_evolution_counter.json",
    "state/post_task_evolution_request.json",
    "state/presence_bindings.json",
    "state/request_wire_compatibility.json",
    "state/subagent_last_delegation.json",
    "state/review_continuations/*.json",
    "state/skills/*/repair_admission.json",
    "state/skills/*/auth_token.json",
    "logs/chat_annotations.jsonl",
})

# The COMPLETE audited set of paths the scan cannot name, and why. A family
# wildcard proves nothing about any exact file (see ``_seg_match``), so these
# are not "covered" — they are disclosed, one entry per unresolvable spelling,
# and this set is asserted by EQUALITY: a new unresolvable spelling fails here
# until it is either made resolvable (name the segment in a constant) or
# audited in. Every entry below is a parameterized reader/writer over planes
# that DO carry inventory rows.
UNRESOLVED_SPELLINGS = frozenset({
    # `state`/`logs` reached with the file name as a parameter or loop value:
    # delegate_recovery.py (the two stop flags), usage_ledger.py (ledger /
    # quarantine / lock names), gateway/logs.py + memory.py + supervisor/
    # state.py (bounded log tail, rotation helper).
    "state/*",
    "logs/*",
    # skill payload roots resolved from a validated relpath (bucket + name):
    # config.py, contracts/skill_payload_policy.py, skill_payload_binding.py,
    # tools/core.py — the `skills/<source>/<name>/**` row owns them.
    "skills/*",
    "skills/*/*",
})

# --- doc parsing -----------------------------------------------------------

_BACKTICK_RE = re.compile(r"`([^`]+)`")
_PLACEHOLDER_RE = re.compile(r"<[^>/]+>")
_BRACES_RE = re.compile(r"\{([^{}]+)\}")


def _expand_braces(token: str) -> list[str]:
    match = _BRACES_RE.search(token)
    if not match:
        return [token]
    head, tail = token[: match.start()], token[match.end():]
    out: list[str] = []
    for option in match.group(1).split(","):
        out.extend(_expand_braces(head + option.strip() + tail))
    return out


def _pattern_tokens(cell: str) -> list[str]:
    tokens: list[str] = []
    for raw in _BACKTICK_RE.findall(cell):
        for piece in raw.split():
            piece = _PLACEHOLDER_RE.sub("*", piece).strip().rstrip("/")
            if not piece or piece in {"+", "("}:
                continue
            tokens.extend(_expand_braces(piece))
    return tokens


def doc_rows(text: str) -> list[list[str]]:
    """Every inventory-table row as its list of first-cell path patterns."""
    rows: list[list[str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("| `"):
            continue
        first_cell = stripped.strip("|").split("|", 1)[0]
        tokens = _pattern_tokens(first_cell)
        if tokens:
            rows.append(tokens)
    return rows


# --- matching --------------------------------------------------------------


def _seg_match(scan_seg: str, pat_seg: str) -> bool:
    """One segment against one row pattern segment.

    A scan segment carrying a wildcard is a FAMILY the scan could not name
    ("some file in state/"), so only a row segment that spells its own
    wildcard may certify it. Letting a literal row absorb it made both
    directions vacuous: ``state/*`` "proved" ``state/state.json`` documented
    and, backwards, kept every exact row alive with no writer behind it.
    """
    if "*" in scan_seg:
        return "*" in pat_seg and (
            fnmatch.fnmatchcase(pat_seg, scan_seg)
            or fnmatch.fnmatchcase(scan_seg, pat_seg)
        )
    return pat_seg == "*" or fnmatch.fnmatchcase(scan_seg, pat_seg)


def _match(scan: list[str], pat: list[str], scan_prefix_ok: bool = True) -> bool:
    """Segment match with multi-segment ``**`` and directional prefix rules.

    A pattern that runs out first always matches: a directory row covers its
    children. A SCAN path that runs out first is directional. Forward
    (``scan_prefix_ok=True``, "is this scanned path documented?") accepts it —
    ``state`` is answered by the deeper rows under ``state/``. Backward
    (``scan_prefix_ok=False``, "does anything still write this row?") refuses
    it: the population contains bare top-level tokens, so accepting a shorter
    scan path let ``state`` keep ANY invented ``state/<name>.json`` row alive
    and the no-stale-rows direction proved nothing. A trailing ``**`` still
    matches zero segments, so ``state/skills/**`` needs depth 2, not 3.
    """
    if not pat:
        return True  # pattern is a prefix: a directory row covers children
    if pat[0] == "**":
        return any(_match(scan[i:], pat[1:], scan_prefix_ok)
                   for i in range(len(scan) + 1))
    if not scan:
        return scan_prefix_ok  # scan path is a prefix: named by a deeper row
    return _seg_match(scan[0], pat[0]) and _match(scan[1:], pat[1:], scan_prefix_ok)


def _covers(scan_path: str, pattern: str, scan_prefix_ok: bool = True) -> bool:
    scan_segs = scan_path.split("/")
    pat_segs = pattern.split("/")
    if _match(scan_segs, pat_segs, scan_prefix_ok):
        return True
    # Bare filename tokens (contain a dot, single segment) also cover a path
    # by basename — the row names the file inside its family directory. Only a
    # NAMED leaf may be matched this way: a wildcard leaf would be certified by
    # any dotted token (``*.lock`` used to answer for every unresolved path).
    if len(pat_segs) == 1 and "." in pat_segs[0] and "*" not in scan_segs[-1]:
        return _seg_match(scan_segs[-1], pat_segs[0])
    return False


# --- tests -----------------------------------------------------------------


def test_scan_is_populated_and_pinned():
    paths = scan_data_paths()
    assert "uploads/views/*_*" in paths, "the shared original-retention writer must remain scanned"
    missing_sentinels = sorted(SENTINELS - paths)
    assert not missing_sentinels, (
        f"scanner regressed — sentinel paths vanished: {missing_sentinels}"
    )
    assert len(paths) == EXPECTED_SCAN_PATHS, (
        f"distinct data-relative paths changed: {len(paths)} != {EXPECTED_SCAN_PATHS}. "
        "A new/removed writer path must land together with its PERSISTENCE.md "
        f"row and this pin. Full set:\n" + "\n".join(sorted(paths))
    )


def test_skill_cache_paths_keep_their_actual_state_owner():
    paths = scan_data_paths()
    assert {"state/skills/*/go", "state/skills/*/go-cache", "state/skills/*/dependency_cache"} <= paths
    assert "go" not in paths and "go-cache" not in paths


def test_every_scanned_path_has_an_inventory_row():
    """Every path the scan can NAME has a row; the rest is audited, not assumed."""
    text = DOC.read_text(encoding="utf-8")
    patterns = [token for row in doc_rows(text) for token in row]
    assert patterns, "PERSISTENCE.md inventory tables not found"
    uncovered = {
        path for path in scan_data_paths()
        if not any(_covers(path, pattern) for pattern in patterns)
    }
    undocumented = sorted(uncovered - UNRESOLVED_SPELLINGS)
    assert not undocumented, (
        "data/-relative paths written by the runtime but absent from "
        f"docs/PERSISTENCE.md: {undocumented}"
    )
    resolved = sorted(UNRESOLVED_SPELLINGS - uncovered)
    assert not resolved, (
        "these spellings are no longer unresolvable-and-uncovered — drop them "
        f"from UNRESOLVED_SPELLINGS: {resolved}"
    )


# Rows the backward check must NOT demand a writer for, each because the row
# itself states there is none in this tree. Asserted by equality below, so an
# entry that acquires a writer (or a row that quietly loses one) surfaces.
STALE_ROW_EXEMPTIONS = frozenset({
    # PERSISTENCE.md documents this plane as an orphan of a removed feature:
    # "none in this tree ... nothing reads or recreates it". Confirmed by grep —
    # the string `project_source_locks` appears in no .py file.
    "state/project_source_locks",
    # The retired compactor's archive: "none since the usage store" — kept as
    # evidence, read only by the explicit history audit (model_send_seal).
    "archive/usage_ledger/segment_*.jsonl",
})


def stale_rows(text: str, paths=None) -> list[str]:
    """Inventory rows for which no scanned path reaches the row's own depth."""
    paths = scan_data_paths() if paths is None else paths
    return [
        row[0] for row in doc_rows(text)
        if not any(_covers(path, token, scan_prefix_ok=False)
                   for token in row for path in paths)
    ]


def test_every_inventory_row_is_still_real():
    """No row survives without a scanned path behind it.

    The backward direction drops the forward direction's scan-prefix rule: a
    scan path SHORTER than the row must not certify it, or every exact row is
    kept alive by the bare top-level token (``state``, ``logs``, ``skills``,
    ``archive``, …) the scan also produces — which is what made this check
    vacuous. ``test_a_fabricated_inventory_row_is_caught_as_stale`` is the
    mutant that proves it bites now.

    Making it bite cost two resolution facts and one exemption. The stdlib
    rotating handlers (`logs/server.log`, `logs/launcher.log`) and the
    oversized-task-text sink (`logs/tasks/*`) were resolved instead of
    exempted — a real writer must be SEEN, not excused. The remaining row is
    an orphan plane the document itself says nothing in this tree writes, and
    it is exempted by name in ``STALE_ROW_EXEMPTIONS``.
    """
    text = DOC.read_text(encoding="utf-8")
    stale = stale_rows(text)
    assert not set(stale) - STALE_ROW_EXEMPTIONS, (
        "PERSISTENCE.md rows no scanned writer path matches (stale?): "
        f"{sorted(set(stale) - STALE_ROW_EXEMPTIONS)}"
    )
    acquired = sorted(STALE_ROW_EXEMPTIONS - set(stale))
    assert not acquired, (
        "these rows have an in-tree writer now — drop them from "
        f"STALE_ROW_EXEMPTIONS: {acquired}"
    )


def test_a_fabricated_inventory_row_is_caught_as_stale():
    """A row for a file nothing writes must be reported, not absorbed.

    The scan population contains bare top-level tokens, and the forward
    matcher lets a SHORTER scan path count as "named by a deeper row" — so
    ``state`` used to certify any invented ``state/<anything>.json`` row and
    the no-stale-rows direction proved nothing at all.
    """
    text = DOC.read_text(encoding="utf-8")
    fabricated = "state/does_not_exist.json"
    mutant = text + (
        f"\n| `{fabricated}` | fabricated writer | none | none | none |\n"
    )
    assert set(stale_rows(mutant)) - STALE_ROW_EXEMPTIONS == {fabricated}
    # ... and the honest document adds nothing beyond its audited exemption.
    assert not set(stale_rows(text)) - STALE_ROW_EXEMPTIONS


# --- red-first pins (stage-2 fix wave, lane persistence-inventory) ----------


def test_constant_named_durable_files_are_visible_to_the_scan():
    """A durable file whose NAME lives in a module constant must be scanned.

    Collapsing every non-literal segment to ``*`` hid whole entities from the
    forward check: the name is a compile-time constant, so the scan can and
    must resolve it.
    """
    paths = scan_data_paths()
    hidden = sorted(p for p in (
        "state/claudexor_rotation_provisioning.json",
        "state/delegate_terminal_refresh_cursor.json",
        "state/extension_generation.json",
        "state/post_task_evolution_counter.json",
        "state/post_task_evolution_request.json",
        "state/presence_bindings.json",
        "state/request_wire_compatibility.json",
        "state/subagent_last_delegation.json",
        "logs/chat_annotations.jsonl",
    ) if p not in paths)
    assert not hidden, f"constant-named durable files invisible to the scan: {hidden}"


def test_unresolved_wildcard_never_certifies_an_exact_row():
    """An unresolved scan segment is a FAMILY, never proof about a NAME.

    ``state/*`` means "some file in state/ the scan could not name"; letting it
    match the exact row ``state/state.json`` made both directions of the
    contract vacuous — the wildcard absorbed every literal row.
    """
    assert not _covers("state/*", "state/state.json")
    assert not _covers("state/skills/*/*", "state/skills/*/review.json")
    # Family patterns still certify family scans (both spell the wildcard).
    assert _covers("state/*", "state/*")
    assert _covers("task_results/*", "task_results/*.json")


def test_parent_of_a_helper_returned_path_is_a_named_root(tmp_path):
    """``<helper>().parent / "name"`` is a NAMED plane, not an unrooted leaf.

    The parent of a path the scan already resolved is a fact the source states,
    so a sibling written through it must enter the population under its own
    name. The live writer that first exposed this (the acceptance api-fallback
    record beside ``state/reviewer_slot_last_execution.json``) was retired with
    the acceptance-rows decision, so the mechanism is pinned on a synthetic
    module shaped exactly like that writer — while the base stayed unresolved,
    such a durable record was invisible to the forward check and held no
    inventory row.
    """
    module = tmp_path / "ouroboros" / "synthetic_writer.py"
    module.parent.mkdir(parents=True)
    module.write_text(
        "import pathlib\n"
        "\n"
        "def _last_execution_path(drive_root):\n"
        '    return pathlib.Path(drive_root) / "state" / "reviewer_slot_last_execution.json"\n'
        "\n"
        "def record_sibling(drive_root, payload):\n"
        '    path = _last_execution_path(drive_root).parent / "sibling_disclosure.json"\n'
        '    path.write_text(payload, encoding="utf-8")\n',
        encoding="utf-8",
    )
    paths = scan_data_paths(tmp_path)
    assert "state/reviewer_slot_last_execution.json" in paths
    assert "state/sibling_disclosure.json" in paths

def test_no_parameter_rooted_spelling_lands_at_the_data_ROOT():
    """A path whose root came in as a parameter must not read as top-level.

    ``state_dir`` matches ``DATA_ROOT_MARKERS``, so a chain hanging off it is
    correctly seen as data-relative but WRONGLY placed at the data root:
    ``uninstalled.json`` and the ``__extension_imports/*-*`` family are
    per-skill state under ``state/skills/<name>/``. At the root they were
    certified by the basename / bare-token fallbacks — covered by accident,
    documented nowhere they actually live. Every such spelling belongs in
    ``SUBROOT_ALIASES`` with its call-site audit.
    """
    misrooted = sorted(
        path for path in scan_data_paths()
        if path.split("/")[0] in {"uninstalled.json", "__extension_imports"}
    )
    assert not misrooted, f"parameter-rooted spellings left at the data root: {misrooted}"


def test_resolved_observability_root_matches_real_writer_outputs(tmp_path):
    """Canonical root spelling must not turn blobs/calls into data-root stores."""
    from ouroboros import observability

    root = tmp_path / "data"
    call = observability.persist_call(root, task_id="inventory", call_id="call",
                                      call_type="llm_response", payload={"answer": "retained"})
    manifest = observability.read_call_manifest_ref(root, call["manifest_ref"], task_id="inventory")
    assert observability.read_blob_ref(root, manifest["full_payload_ref"]) == {"answer": "retained"}
    paths = scan_data_paths()
    for ref in (call["manifest_ref"], manifest["full_payload_ref"]):
        relative = pathlib.Path(ref["path"]).relative_to(root.resolve()).as_posix()
        assert relative.startswith("observability/")
        assert any(fnmatch.fnmatchcase(relative, path) for path in paths)
    assert {"observability/blobs/*.*.gz", "observability/calls/*/*.json"} <= paths
    assert not any(path.startswith(("blobs/", "calls/")) for path in paths)


def test_literal_source_handle_calls_match_real_writer_outputs(tmp_path):
    """Call-site names match real writes, including sanitized/capped extensions."""
    from ouroboros.artifacts import read_actor_source_bytes, store_actor_source_bytes

    cases = [
        {"category": " context_checkpoints ", "source_id": " .queue / Ω * .", "extension": ".j-s_o+n"},
        {"category": "tool_results", "source_id": "inventory-tool", "extension": "abcdefghijklmnop"},
        {"category": "context_checkpoints", "source_id": "...", "extension": "!"},
        {"category": "context_checkpoints", "source_id": "presence-transport-queue", "extension": "json"},
    ]
    module = tmp_path / "ouroboros" / "synthetic_writer.py"
    module.parent.mkdir(parents=True)
    source = "from ouroboros.artifacts import store_actor_source_bytes\n"
    calls = []
    for index, args in enumerate(cases):
        source += "".join(f"{key.upper()}_{index} = {value!r}\n" for key, value in args.items())
        calls.append(
            f"def write_{index}(root):\n"
            f'    return store_actor_source_bytes(root, "inventory", category=CATEGORY_{index}, '
            f'source_id=SOURCE_ID_{index}, extension=EXTENSION_{index}, data=b"retained")\n'
        )
    constants = module.with_name("source_constants.py")
    constants.write_text(source, encoding="utf-8")
    source = "from ouroboros.source_constants import CATEGORY_0 as IMPORTED_CATEGORY\n" + source
    calls[0] = calls[0].replace("category=CATEGORY_0", "category=IMPORTED_CATEGORY")
    module.write_text(source + "".join(calls), encoding="utf-8")
    paths = scan_data_paths(tmp_path)
    assert len(paths) == len(cases)
    root = tmp_path / "data"
    for args in cases:
        ref = store_actor_source_bytes(root, "inventory", data=b"retained", **args)
        assert read_actor_source_bytes(root, "inventory", ref) == b"retained"
        relative = f'task_results/artifacts/inventory/{ref["path"]}'
        assert any(fnmatch.fnmatchcase(relative, path) for path in paths), (relative, paths)

    queue_path = "task_results/artifacts/*/source_handles/context_checkpoints/presence-transport-queue-*.json"
    row = f"| `{queue_path}` | writer | schema | retained | loss |\n"
    assert queue_path in paths and stale_rows(row, paths) == []
    # Removing just this writer must stale its exact row even while sibling
    # source stores remain. Reparse because the fixture source has changed.
    module.write_text(source + "".join(calls[:-1]), encoding="utf-8")
    assert stale_rows(row, _PathResolver(tmp_path).paths()) == [queue_path]


def test_dynamic_source_handle_arguments_do_not_certify_named_rows(tmp_path):
    module = tmp_path / "ouroboros" / "synthetic_writer.py"
    module.parent.mkdir(parents=True)
    module.with_name("foreign_constants.py").write_text(
        'DYNAMIC_ID = "presence-transport-queue"\n'
        'REBOUND_ID = "presence-transport-queue"\n'
        'UNKNOWN_ID = "presence-transport-queue"\n', encoding="utf-8")
    module.write_text(
        'from ouroboros.foreign_constants import DYNAMIC_ID\n'
        'from ouroboros.missing_module import UNKNOWN_ID\n'
        'CATEGORY = "context_checkpoints"\n'
        'DYNAMIC_ID = f"queue-{unknown}"\n'
        'REBOUND_ID = "first"\nREBOUND_ID = "second"\n'
        'CONDITIONAL_ID = "presence-transport-queue"\nif unknown:\n    CONDITIONAL_ID = unknown\n'
        'SHADOWED_ID = "presence-transport-queue"\n'
        'def write(root, source_id, category, extension, SHADOWED_ID):\n'
        '    store_actor_source_bytes(root, "inventory", category=CATEGORY, source_id=source_id, extension="json")\n'
        '    store_actor_source_bytes(root, "inventory", category=CATEGORY, source_id=DYNAMIC_ID, extension="json")\n'
        '    store_actor_source_bytes(root, "inventory", category=CATEGORY, source_id=f"queue-{source_id}", extension="json")\n'
        '    store_actor_source_bytes(root, "inventory", category=CATEGORY, source_id=REBOUND_ID, extension="json")\n'
        '    store_actor_source_bytes(root, "inventory", category=CATEGORY, source_id=CONDITIONAL_ID, extension="json")\n'
        '    store_actor_source_bytes(root, "inventory", category=CATEGORY, source_id=UNKNOWN_ID, extension="json")\n'
        '    store_actor_source_bytes(root, "inventory", category=CATEGORY, source_id=SHADOWED_ID, extension="json")\n'
        '    store_actor_source_bytes(root, "inventory", category=category, source_id="queue", extension="json")\n'
        '    store_actor_source_bytes(root, "inventory", category=CATEGORY, source_id="queue", extension=extension)\n'
        '    store_actor_source_bytes(root, "inventory", category="invalid", source_id="queue", extension="json")\n',
        encoding="utf-8",
    )
    assert scan_data_paths(tmp_path) == frozenset()
