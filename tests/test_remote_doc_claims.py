"""CLASS GATE 6 — a doc claim about the remote feature must be true of the code.

The defect: the documents promise a control the code does not have.  Three instances were
found in this feature, and the worst was in ``prompts/SYSTEM.md`` — resident context, so a
stale sentence does not merely misinform a reader, it STEERS the agent.  It told the model
that ``verify_and_record`` and ``claude_code_edit`` were unwired for remote placement and
to hand-roll a weaker substitute, while both had complete, gate-covered remote paths.

Prose cannot be gated.  A large share of the claims can: this file asserts the
programmatically checkable subset and, for the rest, the audit's list is in the commit
message rather than pretending to be a test.

Checked here:
* every module ARCHITECTURE's map names exists;
* the tools README says run "on the target" are EXACTLY the routing table's keys;
* every test file the docs name as a gate exists;
* the SSH timeout table in the docs matches ``remote_ssh_config`` row for row;
* every CLI subcommand README lists is really registered;
* every connection endpoint ARCHITECTURE names exists as a symbol;
* no document calls a tool unwired for remote placement while a remote route exists.

WHICH AXIS THIS COVERS — the question the Guard Proof Rule says to ask, asked here after
this file let FOUR doc lies through at once. It had been hardened along exactly ONE axis:
DOC → CODE, for a claim that names an identifier matching a hard-coded naming pattern.
Two whole axes were blind, and each cost two of the four:

* CODE → DOC. Every check above reads what a document SAYS. A map that OMITS a real
  module, directory or field says nothing at all, and silence is unreadable — so
  `tools/dispatch_policy.py` and `state/remote_reconciliation/` were simply absent from
  the maps that claim to be complete, and no rule here could look for them. This is the
  same shape as the platform gate, which grew four rounds of new SPELLINGS for a fixed
  list of names and never a new KIND of dependency.
* IDENTIFIER KINDS OUTSIDE THE PATTERNS. `(remote|execd|workspace|connection|cli)_*\\.py`
  plus three hand-named modules; a fixed tuple of seven subcommand names; a regex for
  `api_connections_*`. A plain-named module, a directory, a new subcommand, a
  `module.SYMBOL` reference — none of them matched anything, so an invented one passed.

Both are now closed MECHANICALLY, in both directions: paths and symbols the docs name
must exist, and the module map, the state map and the connection schema must be complete
against the code. `CLAIM_KINDS` below is the register, and every kind is either bound to
a check in this file or carries a written reason it cannot be — the shape
`tests/test_execd_packaging_guard_proofs.py` uses for the same purpose.

WHAT STAYS UNCOVERED, on purpose and in writing: a BEHAVIOURAL promise ("streams instead
of buffering", "filters every path", "comes back as a Home artifact") and a PROCESS
promise ("run Bootstrap again after a restart"). Those are claims about what the code
DOES, not about what it CONTAINS, and no reading of the text can settle them — the two
that shipped were caught by a human comparing a paragraph to a docstring. Do not
describe this gate as making the documents true; it makes their IDENTIFIERS and their
INVENTORIES true, which is where the mechanizable half of the lying happens.

FURTHER BOUNDARY: these checks read the docs as TEXT, so a reworded claim escapes and a
claim spread across two sentences is not assembled.
"""

from __future__ import annotations

import argparse
import ast
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parent.parent
ARCHITECTURE = REPO / "docs" / "ARCHITECTURE.md"
README = REPO / "README.md"
SYSTEM_PROMPT = REPO / "prompts" / "SYSTEM.md"
DEVELOPMENT = REPO / "docs" / "DEVELOPMENT.md"
DOCUMENTS = (ARCHITECTURE, DEVELOPMENT, README, SYSTEM_PROMPT)

#: Top-level directories a backticked path may be rooted in for the existence check.
#: Deliberately NOT every repo directory: `site/` prose uses illustrative paths for a
#: USER's project folder (`site/.ssh/config`), which are not claims about this tree.
SOURCE_ROOTS = (
    "ouroboros",
    "supervisor",
    "tests",
    "scripts",
    "prompts",
    "web",
    "packaging",
    "devtools",
    "docs",
    ".github",
)
#: Directories whose module inventory the ARCHITECTURE source map claims to carry.
MAPPED_PACKAGE_DIRS = (
    "ouroboros",
    "ouroboros/tools",
    "ouroboros/gateway",
    "supervisor",
    "scripts",
)
#: A sentence that DENIES a path's existence must not be read as claiming it. The
#: `file://` bridge paragraph exists precisely to say `remote_file_bridge` is not there,
#: and `site/public/assets/` is documented as gitignored. Keep this vocabulary short:
#: every entry widens the escape hatch, and "no" alone is far too common to include.
ABSENCE_MARKERS = (
    "there is no",
    "does not exist",
    "gitignored",
    "deferred",
    "never created",
    "not in the repository",
    "no longer",
)
#: Modules that predate this gate and have no entry in the ARCHITECTURE source map.
#: Not a waiver in principle — a map debt, kept explicit so no NEW module can join it,
#: and liveness-checked below so an entry cannot outlive the gap it names.
UNMAPPED_WITH_REASON: dict[str, str] = {
    "ouroboros/evolution_fingerprint.py": "pre-existing map gap (evolution lane)",
    "ouroboros/llm_observability.py": "pre-existing map gap (observability lane)",
    "ouroboros/skill_owner_attestation.py": "pre-existing map gap (skills lane)",
    "ouroboros/version.py": "pre-existing map gap (one constant, read from VERSION)",
    "ouroboros/tools/browser.py": "pre-existing map gap (browser tool lane)",
    "ouroboros/tools/compact_context.py": "pre-existing map gap (context lane)",
    "ouroboros/tools/control_delegation.py": "pre-existing map gap (delegation lane)",
    "ouroboros/tools/evolution_stats.py": "pre-existing map gap (evolution lane)",
    "ouroboros/tools/knowledge.py": "pre-existing map gap (knowledge lane)",
    "ouroboros/tools/memory_tools.py": "pre-existing map gap (memory lane)",
    "ouroboros/tools/search.py": "pre-existing map gap (search lane)",
    "ouroboros/tools/tool_discovery.py": "pre-existing map gap (discovery lane)",
    "ouroboros/tools/vision.py": "pre-existing map gap (vision lane)",
    "scripts/pyi_rth_pythonnet.py": "PyInstaller runtime hook, not an architecture unit",
    "scripts/release_proof.py": "pre-existing map gap (release proof capsule)",
    # Arrived with the rebase range; upstream's own map does not carry them either.
    "ouroboros/_usage_rows_memo.py": "pre-existing map gap (usage accounting lane)",
    "ouroboros/launcher_onboarding.py": "pre-existing map gap (launcher lane)",
    "ouroboros/size_ratchet_manifest.py": "generated size-ratchet data, not an architecture unit",
    "ouroboros/gateway/onboarding_host.py": "pre-existing map gap (onboarding lane)",
    "ouroboros/gateway/task_events.py": "pre-existing map gap (gateway events lane)",
    "scripts/bench_delegated_snapshot.py": "benchmark script, not an architecture unit",
    "scripts/regenerate_size_ratchet.py": "generator for the size-ratchet data above",
}


def _text(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _fenced_blocks(body: str) -> list[str]:
    blocks: list[str] = []
    current: list[str] | None = None
    for line in body.splitlines():
        if line.startswith("```"):
            if current is None:
                current = []
            else:
                blocks.append("\n".join(current))
                current = None
            continue
        if current is not None:
            current.append(line)
    return blocks


def _map_region(anchor: str) -> str:
    """The one fenced ASCII-tree block containing `anchor`."""

    matches = [block for block in _fenced_blocks(_text(ARCHITECTURE)) if anchor in block]
    assert len(matches) == 1, (
        f"expected exactly one ARCHITECTURE map block containing {anchor!r}, "
        f"found {len(matches)} — the map moved and this gate cannot find it"
    )
    return matches[0]


def _sentence_around(body: str, position: int) -> str:
    start = max(body.rfind(". ", 0, position), body.rfind("\n\n", 0, position)) + 1
    end = body.find(". ", position)
    return body[start : end if end != -1 else min(len(body), position + 400)].lower()


def _module_symbols(path: pathlib.Path) -> set[str]:
    """Every name a module binds or keys on, read by AST so nothing is imported.

    Non-docstring identifier-shaped string LITERALS count, because half of what the
    documents name this way is a dict key or a state-file field
    (`state.evolution_owner_stopped`, `task_contract.answer_protocol`) rather than a
    Python binding. Docstrings are excluded on purpose: a name that appears only in a
    module's own prose is exactly the unchecked claim this file is about.
    """

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):  # pragma: no cover - a broken module fails elsewhere
        return set()
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list) or not body:
            continue
        first = body[0]
        if (
            isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            docstrings.add(id(first.value))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.alias):
            names.add((node.asname or node.name).split(".")[0])
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
            and node.value.isidentifier()
        ):
            names.add(node.value)
    return names


def _modules_by_basename() -> dict[str, pathlib.Path]:
    """Unambiguous module basenames only.

    `state`, `review` and `control` each name two modules, so a bare
    `state.something` in a document does not say WHICH one and cannot be judged —
    a stated boundary rather than a coin flip.
    """

    seen: dict[str, list[pathlib.Path]] = {}
    for directory in (*MAPPED_PACKAGE_DIRS, "ouroboros/contracts"):
        for module in (REPO / directory).glob("*.py"):
            if module.name != "__init__.py":
                seen.setdefault(module.stem, []).append(module)
    return {stem: found[0] for stem, found in seen.items() if len(found) == 1}


def _resolve_documented_path(candidate: str) -> tuple[str, pathlib.Path | None, str]:
    """Split a backticked token into (path, resolved file, trailing symbol).

    `ouroboros/remote_refusal_actions.REFUSAL_ACTIONS` is a symbol reference wearing a
    path; `tests/test_x` is a module written without its suffix. Both are claims, and
    both were unreadable to the naming-pattern rules this file started with.
    """

    trimmed = candidate.rstrip("/")
    for base in (trimmed, f"{trimmed}.py"):
        if (REPO / base).exists():
            return base, (REPO / base if (REPO / base).is_file() else None), ""
    head, _, tail = trimmed.rpartition(".")
    if head and tail and re.match(r"[A-Za-z_]", tail):
        for base in (head, f"{head}.py"):
            if (REPO / base).is_file():
                return base, REPO / base, tail
    return trimmed, None, ""


def _documented_repo_paths(body: str):
    """Every backticked token that claims to be a path in one of SOURCE_ROOTS."""

    for match in re.finditer(r"`([A-Za-z0-9_.\-/]+)`", body):
        candidate = match.group(1)
        if "/" not in candidate or candidate.split("/")[0] not in SOURCE_ROOTS:
            continue
        yield match.start(), candidate, _resolve_documented_path(candidate)


def test_every_remote_module_the_architecture_map_names_exists():
    """A map entry for a module nobody wrote is a promise with no code behind it."""

    architecture = _text(ARCHITECTURE)
    named = set(re.findall(r"\b((?:remote|execd|workspace|connection|cli)_[a-z0-9_]+)\.py\b", architecture))
    named |= set(re.findall(r"\b(export_policy_contract|tool_capabilities|projects_registry)\.py\b", architecture))
    assert named, "the architecture map named no remote module — the regex has gone stale"
    missing = sorted(
        stem
        for stem in named
        if not (REPO / "ouroboros" / f"{stem}.py").exists()
        and not (REPO / "ouroboros" / "tools" / f"{stem}.py").exists()
        and not (REPO / "ouroboros" / "gateway" / f"{stem}.py").exists()
        and not (REPO / "supervisor" / f"{stem}.py").exists()
    )
    assert not missing, f"ARCHITECTURE names modules that do not exist: {missing}"


def test_the_readme_target_tool_list_is_exactly_the_routing_table():
    """The owner-facing list of target-executing tools cannot drift from the table.

    Both directions matter. A tool in the docs but not the table is a promise; a tool in
    the table but not the docs is a capability the owner has no way to know about.
    """

    from ouroboros.tool_capabilities import REMOTE_NATIVE_TOOL_OPERATION

    readme = _text(README)
    routed = set(REMOTE_NATIVE_TOOL_OPERATION)
    documented = {name for name in routed if f"`{name}`" in readme}
    assert documented == routed, (
        "README's target-executing tool list and REMOTE_NATIVE_TOOL_OPERATION disagree; "
        f"undocumented: {sorted(routed - documented)}"
    )


def test_the_system_prompt_names_the_same_target_tools():
    """The resident prompt is the highest-blast-radius document of the three."""

    from ouroboros.tool_capabilities import REMOTE_NATIVE_TOOL_OPERATION

    prompt = _text(SYSTEM_PROMPT)
    missing = sorted(
        name
        for name in REMOTE_NATIVE_TOOL_OPERATION
        if f"`{name}`" not in prompt and name not in prompt
    )
    # The prompt groups tools in prose ("the service tools"), so an exact set match would
    # gate wording rather than truth; what must hold is that it names most of them and
    # contradicts none, which the next test enforces.
    assert len(missing) <= 6, f"the prompt names too few routed tools: {missing}"


# Tools with a real remote production path. Derived from the routing table plus the two
# hybrid forks, so a new fork cannot be forgotten here.
def _remotely_capable_tools() -> set[str]:
    from ouroboros.tool_capabilities import REMOTE_NATIVE_TOOL_OPERATION

    capable = set(REMOTE_NATIVE_TOOL_OPERATION)
    shell = _text(REPO / "ouroboros" / "tools" / "shell.py")
    verify = _text(REPO / "ouroboros" / "tools" / "verify.py")
    if "_remote_claude_code_edit" in shell:
        capable.add("claude_code_edit")
    if "_verify_on_remote_target" in verify:
        capable.add("verify_and_record")
    return capable


def test_no_document_calls_a_remotely_capable_tool_unwired():
    """The instance that steered the model away from a working faculty.

    A document may not say a tool is unwired for remote placement when a remote route
    exists for it. This check was born with a two-fragment exemption for the stale claims
    the docs lane was already removing, plus a companion test that failed the moment they
    were gone. The docs lane has landed, both fragments are absent from every document,
    and so the exemption and its companion are deleted: the check is UNCONDITIONAL, which
    also means a reworded regression no longer has an exemption to hide behind.
    """

    capable = _remotely_capable_tools()
    violations: list[str] = []
    for path in (README, SYSTEM_PROMPT, DEVELOPMENT, ARCHITECTURE):
        body = _text(path)
        if not body:
            continue
        for match in re.finditer(
            r"[^.]*\b(?:not wired|unwired|remain unfinished|is not available)\b[^.]*\.", body
        ):
            sentence = match.group(0)
            named = {tool for tool in capable if f"`{tool}`" in sentence}
            if not named:
                continue
            violations.append(f"{path.name}: {sorted(named)} in {sentence.strip()[:120]!r}")
    assert not violations, (
        "a document claims a tool is unwired for remote placement while a remote route "
        "exists — in prompts/SYSTEM.md this actively steers the agent:\n"
        + "\n".join(violations)
    )


def test_every_gate_the_docs_name_as_a_test_exists():
    """A doc that cites a gate by filename must cite one that is real."""

    body = _text(ARCHITECTURE) + _text(DEVELOPMENT) + _text(README)
    named = set(re.findall(r"tests/(test_[a-z0-9_]+\.py)", body))
    assert named, "no test file is cited by the docs — the regex has gone stale"
    missing = sorted(name for name in named if not (REPO / "tests" / name).exists())
    assert not missing, f"the docs cite gates that do not exist: {missing}"


def test_the_documented_ssh_timeout_table_matches_the_code():
    """Seven rows of operational timing, in a table an operator will trust."""

    from ouroboros.remote_ssh_config import _SSH_TIMEOUTS

    body = _text(ARCHITECTURE)
    if "OUROBOROS_SSH_" not in body:
        return  # the table moved; the module-map test still covers the module itself
    for kind, row in _SSH_TIMEOUTS.items():
        env_name, default, hard_max = row
        assert env_name in body, f"{env_name} is undocumented"
        assert f"{default}" in body, (
            f"the documented default for {kind!r} does not match the code's {default}"
        )
        assert f"{hard_max}" in body, (
            f"the documented hard maximum for {kind!r} does not match the code's {hard_max}"
        )


def test_the_connections_cli_subcommand_list_is_exactly_the_registered_set():
    """Both directions, against the parser itself rather than a fixed tuple.

    This check used to iterate a hard-coded tuple of seven names, so it could only
    ever judge those seven: `ouroboros connections purge` documented in the README
    matched no name in the tuple and passed, and a subcommand REGISTERED but never
    documented was invisible from the other side. The parser is the authority, so
    ask the parser.
    """

    from ouroboros.cli_connections import add_connections_parser

    parser = argparse.ArgumentParser()
    add_connections_parser(parser.add_subparsers(dest="command"))
    connections = parser._subparsers._group_actions[0].choices["connections"]  # noqa: SLF001
    registered = set(connections._subparsers._group_actions[0].choices)  # noqa: SLF001
    assert registered, "the connections parser registered nothing — extraction is stale"

    readme = _text(README)
    documented = set(re.findall(r"ouroboros connections ([a-z-]+)", readme))
    assert documented == registered, (
        "README's connections subcommand list and the registered parser disagree; "
        f"documented-only: {sorted(documented - registered)}; "
        f"registered-only: {sorted(registered - documented)}"
    )


# ── CODE → DOC: the axis that let two of the four lies through ───────────────


def test_the_architecture_source_map_names_every_module_it_maps():
    """A map that silently omits a module is a completeness claim that is false.

    `tools/dispatch_policy.py` escaped every doc-claim rule in this file by being
    ABSENT: nothing in the text was wrong, because nothing in the text was there.
    Every rule that reads what a document SAYS is structurally unable to see this,
    which is why it needs a rule that reads what the CODE has.
    """

    region = _map_region("launcher.py (PyWebView)")
    unmapped: list[str] = []
    for directory in MAPPED_PACKAGE_DIRS:
        for module in sorted((REPO / directory).glob("*.py")):
            if module.name == "__init__.py":
                continue
            relative = f"{directory}/{module.name}"
            if module.name in region or relative in UNMAPPED_WITH_REASON:
                continue
            unmapped.append(relative)
    assert not unmapped, (
        "the ARCHITECTURE source map claims to carry these directories but names "
        f"neither of these modules: {unmapped} — add a map entry, or an entry to "
        "UNMAPPED_WITH_REASON saying why the map does not carry it"
    )
    # The exemption list needs its own liveness check, or a waiver for a module that
    # is now mapped quietly pardons the next one that is not.
    settled = sorted(
        relative
        for relative in UNMAPPED_WITH_REASON
        if not (REPO / relative).exists() or pathlib.Path(relative).name in region
    )
    assert not settled, (
        f"UNMAPPED_WITH_REASON names modules that are mapped or gone: {settled}"
    )
    assert all(len(reason) > 20 for reason in UNMAPPED_WITH_REASON.values())


def test_the_architecture_data_map_names_every_state_location_the_code_creates():
    """`state/remote_reconciliation/` was missing from the data map the same way.

    The authority is the code that CONSTRUCTS the path: every `"state" / "<name>"`
    join in a production module is a durable location an operator will look for.
    """

    region = _map_region("usage_attempts.quarantine.jsonl")
    sources = sorted((REPO / "ouroboros").rglob("*.py")) + sorted(
        (REPO / "supervisor").glob("*.py")
    )
    created: dict[str, str] = {}
    for source in sources:
        for match in re.finditer(
            r'"state"\s*/\s*"([a-z0-9_]+)"', source.read_text(encoding="utf-8")
        ):
            created.setdefault(match.group(1), source.relative_to(REPO).as_posix())
    assert created, "no state subdirectory was found in the code — the regex is stale"
    unmapped = sorted(
        f"state/{name} (built in {origin})"
        for name, origin in created.items()
        if not re.search(rf"\b{re.escape(name)}\b", region)
    )
    assert not unmapped, (
        f"the ARCHITECTURE data map does not name these state locations: {unmapped}"
    )


def test_the_data_map_lists_every_field_the_connection_store_writes():
    """A field list is an inventory, so it is checkable in both directions.

    The map described `remote_connections.json` as "id, display name, ssh_alias,
    expected_host_id + trust history, lifecycle, bootstrapped_at + bootstrap_build"
    while the store had written `bootstrap_contract_set` — the field the whole
    compatibility gate reads — plus three timestamps, for a release. Scope is
    deliberately narrow: this feature's own store, whose schema is one function.
    """

    import tempfile

    from ouroboros import connection_store

    with tempfile.TemporaryDirectory(prefix="doc-claims-store-") as temporary:
        path = pathlib.Path(temporary) / "remote_connections.json"
        row = connection_store.add_connection(
            name="doc claims", ssh_alias="doc-claims-alias", path=path
        )
        row = connection_store.record_bootstrap(
            row["id"], build="0.0.0", contract_set=1, path=path
        )
    stored = set(row)
    line = next(
        candidate
        for candidate in _map_region("usage_attempts.quarantine.jsonl").splitlines()
        if "remote_connections.json" in candidate
    )
    missing = sorted(field for field in stored if f"`{field}`" not in line)
    assert not missing, (
        f"the data map's remote_connections.json entry omits stored fields: {missing}"
    )


# ── DOC → CODE, for any name rather than a matching one ──────────────────────


def test_every_repo_path_the_docs_name_exists():
    """A backticked repo path is a claim, whatever the module happens to be called.

    The naming-pattern rule above only ever judged `remote_*`/`execd_*`/… names, so
    an invented `ouroboros/ghost_policy.py` or `state/…`-shaped directory sailed
    through. Roots are restricted (see SOURCE_ROOTS) and a sentence that DENIES the
    path is exempt (see ABSENCE_MARKERS), both of which are stated boundaries, not
    accidents.
    """

    violations: list[str] = []
    for document in DOCUMENTS:
        body = _text(document)
        for position, candidate, (resolved, _module, _symbol) in _documented_repo_paths(body):
            if (REPO / resolved).exists():
                continue
            sentence = _sentence_around(body, position)
            if any(marker in sentence for marker in ABSENCE_MARKERS):
                continue
            violations.append(f"{document.name}: {candidate}")
    assert not violations, (
        "the docs name repository paths that do not exist: " + ", ".join(violations)
    )


def test_every_symbol_the_docs_attribute_to_a_module_exists():
    """All three spellings the documents use, not just the qualified ones.

    `module.py::name`, `path/module.NAME` and the bare `module.name` — the last is by
    far the commonest here (190 references) and was judged by nothing at all, which
    is how `task_lifecycle.finalize_cancel_on_queue_miss` survived in ARCHITECTURE
    while the function has always been `_finalize_cancel_requested_on_miss`.
    """

    #: File suffixes that make a `word.word` token a FILENAME rather than a symbol.
    suffixes = {
        "py", "json", "jsonl", "md", "js", "mjs", "txt", "yml", "yaml", "sh", "ps1",
        "html", "css", "lock", "ico", "icns", "png", "svg", "jpg", "spec", "cfg",
        "toml", "pid", "bundle", "tar", "gz", "sha256", "xml", "csv", "log", "sql",
        "ini", "plist", "bat", "exe", "cmd", "dmg", "zip", "so", "dylib", "dll",
    }
    modules = _modules_by_basename()
    violations: list[str] = []
    wanted: list[tuple[pathlib.Path, str]] = []
    for document in DOCUMENTS:
        body = _text(document)
        for _position, _candidate, (_resolved, module, symbol) in _documented_repo_paths(body):
            if module is not None and symbol:
                wanted.append((module, symbol))
        for match in re.finditer(r"`?([A-Za-z0-9_/]+\.py)::([A-Za-z_][A-Za-z0-9_.]*)`?", body):
            for base in ("", "ouroboros/", "ouroboros/tools/", "ouroboros/gateway/", "supervisor/", "scripts/", "tests/"):
                module = REPO / (base + match.group(1))
                if module.is_file():
                    wanted.append((module, match.group(2)))
                    break
        for match in re.finditer(r"`([a-z_][a-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)`", body):
            head, tail = match.group(1), match.group(2)
            if head in modules and tail not in suffixes:
                wanted.append((modules[head], tail))
    assert wanted, "no module symbol reference was found — the patterns have gone stale"
    for module, symbol in wanted:
        names = _module_symbols(module)
        unresolved = [part for part in symbol.split(".") if part not in names]
        if unresolved:
            violations.append(f"{module.relative_to(REPO)} has no {unresolved}")
    assert not violations, "the docs name symbols that do not exist: " + ", ".join(
        sorted(set(violations))
    )


# ── the register: every claim kind is covered here or reasoned here ──────────
#
# Modelled on `tests/test_execd_packaging_guard_proofs.py::UNPROVEN_WITH_REASON`. A kind
# maps EITHER to the test in this module that judges it, OR to a written reason no test
# can. The point is that the uncovered set is a decision on the record rather than an
# oversight, and that a covering test cannot be deleted without this failing.
CLAIM_KINDS: dict[str, str] = {
    "existence: module named in a map or in prose": (
        "test_every_repo_path_the_docs_name_exists"
    ),
    "existence: directory": "test_every_repo_path_the_docs_name_exists",
    "existence: cited test file": "test_every_gate_the_docs_name_as_a_test_exists",
    "existence: CLI subcommand": (
        "test_the_connections_cli_subcommand_list_is_exactly_the_registered_set"
    ),
    "existence: gateway endpoint": (
        "test_every_connection_endpoint_the_architecture_names_exists"
    ),
    "existence: module named with a feature prefix": (
        "test_every_remote_module_the_architecture_map_names_exists"
    ),
    "capability: the resident prompt names the routed tools": (
        "test_the_system_prompt_names_the_same_target_tools"
    ),
    "completeness: module map": (
        "test_the_architecture_source_map_names_every_module_it_maps"
    ),
    "completeness: state-location map": (
        "test_the_architecture_data_map_names_every_state_location_the_code_creates"
    ),
    "completeness: stored schema field list": (
        "test_the_data_map_lists_every_field_the_connection_store_writes"
    ),
    "completeness: target-executing tool list": (
        "test_the_readme_target_tool_list_is_exactly_the_routing_table"
    ),
    "reference: symbol or constant": (
        "test_every_symbol_the_docs_attribute_to_a_module_exists"
    ),
    "number: documented timing ceiling": (
        "test_the_documented_ssh_timeout_table_matches_the_code"
    ),
    "capability: a tool called unwired for remote placement": (
        "test_no_document_calls_a_remotely_capable_tool_unwired"
    ),
    "behaviour: what the code DOES with data it holds": (
        "REASON: not mechanizable from text. 'Streams every blob instead of buffering' "
        "and 'filters every path against the export policy' are claims about control "
        "flow, and the two that shipped were both false while every identifier in the "
        "sentence was real. Matching the phrasing would gate wording rather than truth "
        "and would be defeated by a rewrite. This is what code review is for; the "
        "compensating control is that the identifiers such a sentence names are now all "
        "checked, so the sentence cannot ALSO invent a module to attribute itself to."
    ),
    "process: an owner procedure the docs prescribe": (
        "REASON: not mechanizable from text. 'A Home restart requires Bootstrap again' "
        "was false because compatibility is durable owner state, and deciding that "
        "needs the meaning of `connection_store.record_bootstrap`, not its name. A "
        "phrase-matching rule here would be a wording gate that the next paraphrase "
        "walks past. The compensating control is a single stated owner: DEVELOPMENT.md "
        "names the module that owns each half of the evidence, so the contradiction is "
        "checkable by a reader in one hop."
    ),
    "existence: a directory under the runtime data tree named in loose prose": (
        "REASON: not checkable against this repository. `~/Ouroboros/state/...` does not "
        "exist in the tree at all, so the filesystem cannot answer, and the only code-"
        "side authority — literal `\"state\" / \"<name>\"` joins — knows 6 of the ~10 "
        "locations the data map carries, so using it in the map→code direction would "
        "refuse true entries. The code→doc direction IS covered "
        "(test_the_architecture_data_map_names_every_state_location_the_code_creates), "
        "which is the direction the shipped lie went. To gate the other one, give the "
        "state layout a single code-side register the way the SSH timings have one."
    ),
    "number: any documented constant outside the SSH timing table": (
        "REASON: not mechanizable in general. A bare integer in prose cannot be bound "
        "to the constant it means without a machine-readable link between them; the SSH "
        "table is covered because it is a TABLE, keyed by the env var name, which is "
        "exactly the structure a general prose number lacks. Put a new ceiling in a "
        "keyed table if it is to be gated."
    ),
}


def test_every_claim_kind_is_covered_here_or_reasoned_here():
    """The register may not name a test that does not exist, or reason around one."""

    present = {
        node.name
        for node in ast.parse(pathlib.Path(__file__).read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef)
    }
    for kind, coverage in CLAIM_KINDS.items():
        if coverage.startswith("REASON:"):
            assert len(coverage) > 120, f"{kind}: a reason this short is not a reason"
            continue
        assert coverage in present, (
            f"{kind} claims to be covered by {coverage}, which this module does not "
            "define — either the check was deleted or the register is lying, and "
            "both are the defect this file exists to prevent"
        )
    reasoned = {k for k, v in CLAIM_KINDS.items() if v.startswith("REASON:")}
    assert reasoned, "every kind claims to be mechanizable, which is not credible"
    # Every check in this module must be accounted for by some kind, so a new rule
    # cannot be added without saying which axis it closes.
    claimed = {v for v in CLAIM_KINDS.values() if not v.startswith("REASON:")}
    orphans = sorted(
        name
        for name in present
        if name.startswith("test_") and name not in claimed
        and name != "test_every_claim_kind_is_covered_here_or_reasoned_here"
    )
    assert not orphans, (
        f"these checks belong to no declared claim kind: {orphans} — name the axis "
        "in CLAIM_KINDS so the next reader can ask what is still blind"
    )


def test_every_connection_endpoint_the_architecture_names_exists():
    """Each named gateway handler is a real symbol, not a plan."""

    from ouroboros.gateway import connections

    body = _text(ARCHITECTURE)
    named = set(re.findall(r"\b(api_connections?_[a-z_]+)\b", body))
    assert named, "ARCHITECTURE names no connection endpoint — the regex has gone stale"
    missing = sorted(name for name in named if not hasattr(connections, name))
    assert not missing, (
        f"ARCHITECTURE names endpoints gateway.connections does not define: {missing}"
    )
