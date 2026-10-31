"""Reference-book structure, executable inventories and prompt identifiers."""

import os
import pathlib
import re

from ouroboros.tools.registry import ToolRegistry
from ouroboros.reference_books import (
    BOOK_ENTRYPOINTS,
    compose_book,
    load_reference_book,
    read_book_section,
)

REPO = pathlib.Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _read(rel: str) -> str:
    book_id = next((key for key, path in BOOK_ENTRYPOINTS.items() if path == rel), None)
    if book_id:
        return compose_book(load_reference_book(REPO, book_id))
    return (REPO / rel).read_text(encoding="utf-8")


def _names_basename(text: str, basename: str) -> bool:
    """Whether ``text`` names this module file as its own token.

    The boundary is stated as "not a file-name character" rather than a list
    of allowed delimiters: the component map introduces modules after a space,
    a backtick AND an opening parenthesis (``(clawhub.py
    registry client``), so an allow-list of delimiters would report a module
    the document does name. What must NOT precede the basename is a character
    that could be part of a longer file name — a word character, a dot or a
    hyphen — which is exactly how ``test_s3_task_control_browser.py`` used to
    answer for ``browser.py``. A trailing word character is refused too, so
    ``x.py`` never answers for ``x.pyi``. A basename inside an explicit path
    names that path only; the caller checks the full requested path separately.
    Bare module-tree rows remain ambiguous when several modules share a name.
    """
    return re.search(
        r"(?<![\w.\-/\\])" + re.escape(basename) + r"(?!\w)", text
    ) is not None


def test_component_basename_does_not_borrow_another_explicit_path():
    assert not _names_basename("`ouroboros/tools/knowledge.py`", "knowledge.py")
    assert not _names_basename(r"`ouroboros\tools\knowledge.py`", "knowledge.py")
    assert _names_basename("├── knowledge.py ← topic storage", "knowledge.py")
    assert _names_basename("(clawhub.py registry client)", "clawhub.py")
    assert not _names_basename("test_s3_task_control_browser.py", "browser.py")


def test_recent_abi_retirements_section_carries_the_abi_70_window():
    """Section 11.4 carries the compatibility facts an upgrading operator needs.

    ABI 7.0 is the largest retirement window in the project's history — five
    gateway aliases, the reviewer comma-list configuration keys, two wall-clock
    timeout keys, a plugin-API major, and a durable task-row schema stamp with
    no legacy converter. Every key of `RETIRED_COMMA_LIST_SETTING_KEYS` must be
    named there, because those are the ones whose migration must happen BEFORE
    the upgrade. Older release chronology is git history, not a book fact.
    """
    section = _architecture_section("11.4 Recent ABI Retirements")

    from ouroboros.settings_defaults import RETIRED_COMMA_LIST_SETTING_KEYS

    assert "ABI 7.0" in section
    missing = [key for key in RETIRED_COMMA_LIST_SETTING_KEYS if key not in section]
    assert not missing, f"11.4 does not name the retired comma-list keys: {missing}"
    assert "OUROBOROS_REVIEWER_SLOTS" in section, "the migration target must be named"


def test_model_send_design_note_matches_the_observability_contract():
    assert (REPO / "ouroboros" / "model_send_seal.py").exists()


def test_settings_docs_name_every_key_owner_and_what_startup_persists():
    """Keep the live settings and their rows in the tool-read reference table."""
    arch = _read("docs/ARCHITECTURE.md")

    from ouroboros.settings_defaults import SETTINGS_DEFAULTS

    for key in ("OUROBOROS_CONTEXT_MODE_AUTO_LOW", "OUROBOROS_CLAWHUB_REGISTRY_URL"):
        assert key in SETTINGS_DEFAULTS
        assert f"| {key} |" in arch, f"{key} is missing from the settings table"


def test_architecture_does_not_claim_usage_response_is_the_only_usage_reader():
    """Shared normalization and adapter response reads remain separate consumers."""
    for module in ("ouroboros/usage_accounting.py", "ouroboros/loop_llm_call.py"):
        assert "from ouroboros._usage_response import" in _read(module), module
    assert 'resp_dict.get("usage")' in _read("ouroboros/llm_openai_compatible.py")


def test_architecture_component_map_covers_every_live_runtime_module():
    """README calls ARCHITECTURE.md the full component map — so prove it.

    Every non-``__init__`` module of the tracked runtime population must be
    named in the component map (by path or by basename). A module nobody wrote
    a row for is invisible to the one document a contributor is told to read
    before editing, and the omission is silent: no other gate reads this
    document against the tree. Population comes from the domain manifest's own
    SSOT helper (``scripts/domain_graph.tracked_population``) so this pin and
    the domain gates can never disagree about what "live module" means.

    The basename must appear as its OWN token, not as a substring. A plain
    ``name in arch`` accepted a basename buried inside a longer name — so
    ``browser.py`` was "documented" by ``test_s3_task_control_browser.py``,
    ``health.py`` by ``extension_health.py`` and ``vision.py`` by
    ``delegate_supervision.py``, while those three modules had no row of any
    kind.
    """
    from scripts.domain_graph import tracked_population

    arch = _read("docs/ARCHITECTURE.md")
    missing = sorted(
        path for path in tracked_population(REPO)
        if not path.endswith("__init__.py")
        and path not in arch
        and not _names_basename(arch, pathlib.PurePosixPath(path).name)
    )
    assert not missing, (
        "docs/ARCHITECTURE.md names no owner for these live modules: "
        f"{missing}"
    )


def test_architecture_subject_map_row_resolves_its_owner_section():
    """The map preserves the module and a resolvable owner section, not a fixed
    inventory of private helpers inside the registry row."""
    from ouroboros.reference_books import load_reference_book, read_book_section

    arch = _read("docs/ARCHITECTURE.md")
    rows = [line for line in arch.splitlines() if line.strip().startswith("review_subject.py ")]
    assert len(rows) == 1, rows
    assert "§6 Subject operation" in rows[0]
    section = read_book_section(load_reference_book(REPO, "architecture"), "Subject operation")
    assert section.sources[0].path == "docs/architecture/06-agent-core.md"


def _change_review_items() -> dict:
    """``number -> item`` over the Change Review Checklist and the Ouroboros Body
    Layer tables (one continued numbering), read from the live CHECKLISTS.md."""
    text = (REPO / "docs/CHECKLISTS.md").read_text(encoding="utf-8")
    start = text.index("## Change Review Checklist")
    end = text.index("## Shared Contract Ownership")
    return {int(n): name for n, name in re.findall(r"^\| (\d+) \| ([a-z_]+) \|", text[start:end], flags=re.M)}


def test_checklist_item_numbers_cited_outside_the_checklist_name_the_current_items():
    """The change-review checklist was renumbered (core 1-9, body layer 10-29).
    Every place that cites an item BY NUMBER AND NAME — the engineering chapters,
    the architecture book, runtime comments, test docstrings — must agree with the
    live table; a number that names the wrong item sends a reviewer to the wrong
    rule. The standing archive keeps its historical numbers by its own note."""
    items = _change_review_items()
    assert items[7] == "capability_regression" and items[11] == "development_compliance" and len(items) == 29
    names = "|".join(sorted(items.values(), key=len, reverse=True))
    forms = (
        # "item 17 (`subagent_isolation`)", "item 21 `source_completeness`", "item 15 (self_consistency)", "item 27, `gateway_parity`"
        re.compile(rf"\bitems?\s+(\d{{1,2}})(?:\([a-z]\))?,?\s+\(?`?({names})`?", flags=re.I),
        # "cache_friendliness item 28"
        re.compile(rf"\b({names})\s+item\s+(\d{{1,2}})\b"),
        # "`self_consistency` (item 15)"
        re.compile(rf"`({names})`\s+\(item\s+(\d{{1,2}})\)"),
    )
    roots = ("docs/CHECKLISTS.md", "docs/development", "docs/architecture", "ouroboros", "tests")
    skip = {pathlib.Path(__file__).resolve(), (REPO / "docs/CHECKLISTS_ARCHIVE.md").resolve()}
    wrong = []
    for root in roots:
        path = REPO / root
        files = [path] if path.is_file() else [*path.rglob("*.md"), *path.rglob("*.py")]
        for file in files:
            if file.resolve() in skip:
                continue
            text = file.read_text(encoding="utf-8", errors="replace")
            for line_no, line in enumerate(text.splitlines(), 1):
                for pattern in forms:
                    for match in pattern.finditer(line):
                        number, name = match.groups() if pattern is forms[0] else reversed(match.groups())
                        if items.get(int(number)) != name:
                            wrong.append(f"{file.relative_to(REPO)}:{line_no}: item {number} is not `{name}`")
    assert not wrong, "\n".join(wrong)
    # Sub-item citations carry no name; the engineering chapters cite the body layer's
    # development_compliance (11) and self_consistency (15) letters, never the old 2/13.
    dev = _read("docs/DEVELOPMENT.md")
    assert not re.search(r"CHECKLISTS items? (?:2|13)\([a-z]\)", dev)


def test_phase3_governance_language_is_pinned_without_new_qa_surface():
    development = _read("docs/DEVELOPMENT.md")
    # The heading is selected by DOC_RESIDUE_SKIPPED_SUBSECTIONS below.
    assert "External facts: unknown is not no" in development
    assert "| Location | Fact | Evidence & re-probe | If stale |" in development


# Identifiers the prompts legitimately name in backticks that are NOT tools:
# parameter names, resource roots, write surfaces, typed outcome/status tokens
# and runtime-context keys. A NEW snake_case identifier in a prompt must either
# be a real tool (or background-whitelisted tool) or be classified here on
# purpose — that classification step is the governance the prompt audit wants:
# a phantom or renamed tool name can no longer hide in the runtime prompts
# (`advisory_review` and the CONSCIOUSNESS "You can" catalog rotted that way).
# Scope: backticked names in all three prompts plus the bare snake_case names
# CONSCIOUSNESS.md writes without backticks; BIBLE.md is deliberately out of scope.
PROMPT_NON_TOOL_IDENTIFIERS = frozenset({
    # resource roots / write surfaces / write roots
    "active_workspace", "artifact_store", "external_workspace", "runtime_data",
    "skill_payload", "subagent_projects", "system_repo", "task_drive", "user_files",
    "write_root", "write_surface",
    # tool parameters named as cross-tool policy
    "project_id", "project_name", "recommended_use", "retry_of", "review_rebuttal", "subagent_id", "working_note",
    "extra_tests", "full_tests", "test_reason",
    # typed outcomes / statuses / runtime-context keys
    "needs_manual_target", "started_uncustodied", "owner_client",
    # safety policy class names (ouroboros/safety.py TOOL_POLICY values) and
    # owner-setting values named as policy
    "check_conditional", "check", "off", "low",
    "cyber_pro",
    # package managers / interpreters named as acquisition or process choices
    "pip", "pip3", "uv", "brew", "apt", "python", "python3", "sudo", "grep", "env",
    # git branches / remotes / skill buckets / write surfaces named as policy
    "ouroboros", "main", "managed", "origin", "external", "genesis", "deliverables",
    # fenced-block languages the owner chat renders natively
    "mermaid", "chart",
    # backlog item status value in CONSCIOUSNESS.md
    "done",
    # runtime-context fact (build_runtime_section -> update_letter.official_update_projection)
    "official_update",
})


def _prompt_backticked_identifiers(text: str) -> set:
    """Backticked lowercase identifiers, single-word ones included (a renamed
    single-word tool such as `escalate` must be caught too)."""
    found = set()
    for token in re.findall(r"`([^`]+)`", text):
        head = token.split("(", 1)[0]
        if re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*", head):
            found.add(head)
    return found


def _prompt_bare_identifiers(text: str) -> set:
    """snake_case tokens written WITHOUT backticks (CONSCIOUSNESS.md's style);
    tokens that are part of a path or filename (`a/b_c`, `x_y.json`) are skipped."""
    return {
        m.group(1)
        for m in re.finditer(r"(?<![\w/.`-])([a-z][a-z0-9]*(?:_[a-z0-9]+)+)(?![\w/.`-])", text)
    }


def test_prompt_tool_names_resolve_to_registered_tools(tmp_path):
    """Every backticked snake_case identifier in the three runtime prompts is
    either a registered tool (public schema) or a documented non-tool identifier
    (for the wake-up template also one of its own render placeholders).
    Completeness is deliberately NOT required (the schemas are the catalog);
    this only forbids phantoms and stale spellings, the drift class the prompt
    audit found in every prompt. The wake-up runs on the full registry, so its
    universe is Main's."""
    from ouroboros.consciousness_wake import PLACEHOLDERS

    root = pathlib.Path(__file__).resolve().parent.parent
    registry = ToolRegistry(repo_dir=tmp_path / "repo", drive_root=tmp_path / "data")
    registered = {schema["function"]["name"] for schema in registry.schemas()}
    universe = registered | PROMPT_NON_TOOL_IDENTIFIERS
    from ouroboros.presence_context import PLACEHOLDERS as FOCUS_PLACEHOLDERS
    wake_universe = universe | set(PLACEHOLDERS)
    for rel, allowed in (
        ("prompts/SYSTEM.md", universe),
        ("prompts/SAFETY.md", universe),
        ("prompts/CONSCIOUSNESS.md", wake_universe),
        ("prompts/FOCUS.md", universe | set(FOCUS_PLACEHOLDERS)),
    ):
        text = (root / rel).read_text(encoding="utf-8")
        unresolved = _prompt_backticked_identifiers(text) - allowed
        assert not unresolved, (
            f"{rel} names identifiers that are neither registered tools nor "
            f"classified non-tool identifiers: {sorted(unresolved)}"
        )
    # The wake template also names tools without backticks; its bare snake_case
    # tokens (the render placeholders aside) must resolve the same way.
    bare = _prompt_bare_identifiers((root / "prompts" / "CONSCIOUSNESS.md").read_text(encoding="utf-8"))
    unresolved_bare = bare - wake_universe
    assert not unresolved_bare, (
        f"prompts/CONSCIOUSNESS.md names bare identifiers that are neither registered tools "
        f"nor classified non-tool identifiers: {sorted(unresolved_bare)}"
    )


# --- Documentation contract enforcement (DEVELOPMENT.md "Documentation contract") ---
#
# Three deterministic checks keep the two resident docs a present-tense map:
#   1. residue: version stamps, decision codenames and "used to / previously"
#      narrative may only SHRINK per `## ` section (baseline below, like the code
#      size ratchet — a section that reaches zero may never grow again);
#   2. the endpoint table mirrors the executable route registries;
#   3. the settings table mirrors `config.SETTINGS_DEFAULTS` (env-only and retired
#      rows are declared, not guessed).
# Language-tagged code fences (```yaml, ```python …) are examples and are not
# scanned; the plain ``` fence holding the §1 module tree IS scanned. The first
# ARCHITECTURE line carries the release version by contract and is skipped, as
# are DEVELOPMENT's "External facts: unknown is not no" (dated provenance is the
# rule there) and the "Documentation contract" section that quotes the markers.

DOC_RESIDUE_PATTERNS = {
    "version_stamp": r"\((?:v\d+\.\d+(?:\.\d+)?(?:-rc\.\d+)?)\)",
    "version_narrative": r"\b(?:since|before|pre-)v\d+",
    "narrative": r"\b(?:used to|previously|formerly|was deleted|replaces the earlier|gate[- ]round|round \d+)\b",
    "codename_paren": r"\((?:GR|AR|BR|CR|D|Q|S|HQ|C|B)\d+[^)]{0,24}\)",
    "codename_word": r"\bPoltergeist\b|\bphase [A-C]\d?\b|owner(?:-| )(?:decision|ratif)",
    # Owner-batch answer codes and reviewer-round labels resolve only inside the chat that minted
    # them; the decision CONTENT belongs in the book, the label in the commit or PR.
    "owner_code": r"\bowner (?:R\d+|batch \d+|fork \d+|\d+[A-Za-z]?\s*=\s*[A-Z]\b)",
    "owner_paren": r"\(owner (?:R\d+|\d+[A-Za-z]?\b|fork \d|batch \d|answer )[^)]{0,40}\)",
    # A residual may cite its tracker once, in the one resolvable form `(issue #NNN)`.
    "bare_issue_ref": r"(?<!\(issue )#\d{3,}\b",
    "cyrillic": r"[А-Яа-яЁё]",
}
DOC_RESIDUE_SKIPPED_SUBSECTIONS = {
    "docs/DEVELOPMENT.md": ("External facts: unknown is not no", "Documentation contract"),
}


def doc_residue_counts(rel: str, text: str, *, is_entrypoint: bool = True) -> dict:
    """Per-`## ` section counts of residue markers (see DOC_RESIDUE_PATTERNS)."""
    counts: dict = {}
    section = "(preamble)"
    fence_lang = None
    skipping = False
    skipped = DOC_RESIDUE_SKIPPED_SUBSECTIONS.get(rel, ())
    for lineno, line in enumerate(text.split("\n"), 1):
        stripped = line.lstrip()
        if stripped.startswith("```"):
            fence_lang = (stripped[3:].strip() or "") if fence_lang is None else None
            continue
        if fence_lang:  # language-tagged example block
            continue
        if fence_lang is None and line.startswith("## "):
            section, skipping = line.strip(), False
        if fence_lang is None and line.startswith("### "):
            skipping = any(name in line for name in skipped)
        if skipping or (is_entrypoint and rel == "docs/ARCHITECTURE.md" and lineno == 1):
            continue
        for kind, pattern in DOC_RESIDUE_PATTERNS.items():
            hits = len(re.findall(pattern, line))
            if hits:
                counts.setdefault(section, {k: 0 for k in DOC_RESIDUE_PATTERNS})[kind] += hits
    return counts


# Zero-residue ceilings recorded after the 07b53365 cleanup: an absent section or kind is a
# zero ceiling, so both documents are held at zero everywhere. A section row exists only
# while a re-baseline records a temporary non-zero ceiling, and such a row may only shrink;
# an absent section stays at zero.
DOC_RESIDUE_BASELINE = {
    "docs/ARCHITECTURE.md": {},
    "docs/DEVELOPMENT.md": {},
}


def test_resident_docs_residue_only_shrinks():
    for rel, baseline in DOC_RESIDUE_BASELINE.items():
        book_id = next(key for key, path in BOOK_ENTRYPOINTS.items() if path == rel)
        book = load_reference_book(REPO, book_id)
        for source in (book.entrypoint, *book.chapters):
            # Scan each physical body once, so a chapter's H1 cannot inherit
            # the prior file's skipped subsection or fenced-example state.
            current = doc_residue_counts(rel, source.text, is_entrypoint=source.source_path == rel)
            for section, counts in current.items():
                allowed = baseline.get(section, {})
                for kind, hits in counts.items():
                    assert hits <= allowed.get(kind, 0), (
                        f"{source.source_path} {section!r}: {kind} residue grew to {hits} (baseline "
                        f"{allowed.get(kind, 0)}); replace the node's description instead of "
                        "appending history (DEVELOPMENT.md 'Documentation contract')"
                    )


def _architecture_section(title: str) -> str:
    return read_book_section(load_reference_book(REPO, "architecture"), title).text


def test_architecture_endpoint_table_mirrors_route_registries(tmp_path):
    """Every mounted browser/CLI route and Host Service route has exactly one table
    row in ARCHITECTURE §4, and no row names a route that is not mounted."""
    from ouroboros.gateway.endpoint_index import HTTP_ENDPOINTS
    from ouroboros.gateway import files as gateway_files
    from ouroboros.server_web import ServedUI

    section = _architecture_section("4. Server API Endpoints")
    rows = re.findall(r"^\| (GET|POST|PUT|PATCH|DELETE|ANY|WS|STATIC) \| `([^`]+)` \|", section, re.M)
    host_prefix = "127.0.0.1:${OUROBOROS_HOST_SERVICE_PORT:-8767}"
    documented_public = {f"{m} {p}" for m, p in rows if not p.startswith(host_prefix)}
    documented_host = {f"{m} {p[len(host_prefix):]}" for m, p in rows if p.startswith(host_prefix)}

    expected_public = set(HTTP_ENDPOINTS)
    for route in gateway_files.file_browser_routes():
        for method in sorted(getattr(route, "methods", None) or []):
            if method in ("HEAD", "OPTIONS"):
                continue
            expected_public.add(f"{method} {route.path}")
    # Server-level surfaces include every static mount of the existing serving owner.
    # Use a disposable source tree, without importing or starting the real server.
    fixture_repo = tmp_path / "public-repo"
    (fixture_repo / "web").mkdir(parents=True)
    expected_public |= {"GET /", "WS /ws"}
    expected_public.update(f"STATIC {mount.path}/*" for mount in ServedUI(fixture_repo).mounts())
    # /api/extensions/{skill}/{rest:path} is mounted for every verb and documented once as ANY
    expected_public = {e for e in expected_public if not e.endswith("/api/extensions/{skill}/{rest:path}")}
    expected_public.add("ANY /api/extensions/{skill}/{rest:path}")

    from starlette.routing import Route, WebSocketRoute

    from ouroboros.gateway.host_service import create_host_service_app

    expected_host = set()
    for route in create_host_service_app(tmp_path).routes:
        if isinstance(route, WebSocketRoute):
            expected_host.add(f"WS {route.path}")
        elif isinstance(route, Route):
            if route.methods is None:
                expected_host.add(f"ANY {route.path}")
            else:
                expected_host.update(
                    f"{method} {route.path}"
                    for method in route.methods
                    if method not in {"HEAD", "OPTIONS"}
                )
        else:
            raise AssertionError(f"Unhandled Host Service route type: {type(route).__name__}")

    assert documented_public == expected_public, (
        f"missing rows: {sorted(expected_public - documented_public)}; "
        f"stale rows: {sorted(documented_public - expected_public)}"
    )
    assert documented_host == expected_host, (
        f"missing host rows: {sorted(expected_host - documented_host)}; "
        f"stale host rows: {sorted(documented_host - expected_host)}"
    )
    assert len(rows) == len(documented_public) + len(documented_host), "duplicate endpoint rows"


# Rows the settings table documents on purpose although `config.SETTINGS_DEFAULTS`
# has no such key: operator env-only levers (never a settings.json carrier), the
# retired alias whose migration the table still explains (pinned by test_review_cycles),
# and the review-lane keys the review-pool migration consumes at load
# (`settings_defaults.REVIEW_POOL_MIGRATED_SETTING_KEYS`; tests/test_review_pool_migration.py).
SETTINGS_TABLE_ENV_ONLY_ROWS = frozenset({
    "OUROBOROS_TRUST_NONLOCAL_BIND_WITHOUT_PASSWORD", "OUROBOROS_DISABLE_MANAGED_UPDATES",
    "OUROBOROS_PRESENTATION", "OUROBOROS_DESKTOP_BACKGROUND", "OUROBOROS_USER_FILES_ROOT", "OUROBOROS_OBSERVABILITY_KEEP_RAW",
    "OUROBOROS_OBSERVABILITY_RETENTION_DAYS", "OUROBOROS_REVIEW_MODEL_TIMEOUT_SEC",
    "OUROBOROS_REVIEW_MAX_TOKENS", "OUROBOROS_PREFLIGHT_TIMEOUT_SEC", "OUROBOROS_PREFLIGHT_SERIAL",
    "OUROBOROS_PREFLIGHT_TEST_WORKERS", "OUROBOROS_BUNDLE_DIR",
    "OUROBOROS_EXTERNAL_HOST_UPDATE", "OUROBOROS_EXTERNAL_HOST_RESULT",
})
SETTINGS_TABLE_RETIRED_ROWS = frozenset({
    "OUROBOROS_ACCEPTANCE_MAX_IMPROVEMENT_PASSES",
    "OUROBOROS_REVIEWER_SLOTS", "OUROBOROS_EFFORT_REVIEW", "OUROBOROS_EFFORT_SCOPE_REVIEW",
    "OUROBOROS_EFFORT_DEEP_SELF_REVIEW", "OUROBOROS_MODEL_DEEP_SELF_REVIEW",
})


def _normalize_default_cell(cell: str) -> str:
    value = cell.strip()
    value = re.sub(r"^`(.*)`$", r"\1", value)
    if value in ('""', "(empty)", "(unset)", "unset", ""):
        return ""
    value = re.sub(r'^"(.*)"$', r"\1", value)
    return value.split()[0].lower() if value else ""


def test_architecture_settings_table_mirrors_config_defaults():
    """Every `config.SETTINGS_DEFAULTS` key has one row in ARCHITECTURE's Default
    settings table with the shipped default; every other row is a declared env-only
    lever or retired alias."""
    from ouroboros import config

    table = _architecture_section("Default settings")
    rows = re.findall(r"^\| ([A-Z][A-Z0-9_]+) \| ([^|]*?) \|", table, re.M)
    keys = [k for k, _ in rows]
    assert len(keys) == len(set(keys)), f"duplicate settings rows: {sorted(k for k in set(keys) if keys.count(k) > 1)}"
    documented = set(keys)
    expected = set(config.SETTINGS_DEFAULTS)
    assert expected <= documented, f"settings missing from the table: {sorted(expected - documented)}"
    undeclared = documented - expected - SETTINGS_TABLE_ENV_ONLY_ROWS - SETTINGS_TABLE_RETIRED_ROWS
    assert not undeclared, f"table rows that are neither shipped defaults nor declared env-only/retired: {sorted(undeclared)}"
    mismatched = [
        (key, cell, str(config.SETTINGS_DEFAULTS[key]))
        for key, cell in rows
        if key in expected and _normalize_default_cell(cell) != str(config.SETTINGS_DEFAULTS[key]).lower()
    ]
    assert not mismatched, f"documented default differs from config.SETTINGS_DEFAULTS: {mismatched}"
