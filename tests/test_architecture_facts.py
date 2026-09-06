"""CPL-3 architecture-fact suite (plan §7.3).

Pins the five pure queries of ``ouroboros/code_intelligence_architecture.py``
on the REAL repository carriers — the domain manifest, the generated facade /
frozen-contract inventories, ``docs/PERSISTENCE.md`` and the
``runtime_mode_policy`` protected inventories — plus their completeness
against those carriers (a manifest row, facade row, persistence row or
frozen-contract row the queries cannot reach = red), and the model-facing
seam: the existing ``query_code`` tool's ``op=architecture``.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from ouroboros.code_intelligence_architecture import (
    ARCHITECTURE_FACTS,
    FROZEN_INVENTORY_RELPATH,
    PERSISTENCE_DOC_RELPATH,
    _frozen_contract_rows,
    _persistence_rows,
    architecture_fact_rows,
    domain_dependencies,
    facade_consumers,
    facade_reexports,
    load_domain_manifest,
    owner_of,
    paths_from_diff,
    persistence_entities_written_by,
    protected_contracts_affected,
)

REPO = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def manifest():
    return load_domain_manifest(REPO)


@pytest.fixture(scope="module")
def reexports(manifest):
    return facade_reexports(REPO, manifest)


# ---------------------------------------------------------------------------
# owner_of
# ---------------------------------------------------------------------------

def test_owner_of_resolves_a_module_path_a_dotted_module_and_a_symbol():
    (llm,) = owner_of(REPO, "ouroboros/llm.py")
    assert (llm.domain, llm.via) == ("D02", "module_path")
    assert llm.domain_title  # the human vocabulary rides along

    (state,) = owner_of(REPO, "supervisor.state")
    assert (state.module, state.domain, state.via) == (
        "supervisor/state.py", "D08", "dotted_module",
    )

    owners = owner_of(REPO, "protected_path_category")
    assert any(
        (o.module, o.domain, o.via) == ("ouroboros/runtime_mode_policy.py", "D13", "symbol_definition")
        for o in owners
    ), owners


def test_owner_of_outside_the_population_is_ownerless_not_an_error():
    assert owner_of(REPO, "web/modules/chat.js") == ()
    assert owner_of(REPO, "docs/ARCHITECTURE.md") == ()


def test_owner_of_covers_every_manifest_module(manifest):
    """Completeness against the manifest: every population row answers with
    exactly its pinned domain."""
    assert len(manifest.modules) >= 400  # the whole runtime population, not a sample
    for path, domain in manifest.modules.items():
        rows = owner_of(REPO, path)
        assert [(r.module, r.domain) for r in rows] == [(path, domain)], path


# ---------------------------------------------------------------------------
# domain_dependencies
# ---------------------------------------------------------------------------

def test_domain_dependencies_reports_the_manifest_baseline_edges(manifest):
    deps = domain_dependencies(REPO, "d02")  # case-normalized
    assert deps.domain == "D02" and deps.title == manifest.domains["D02"]
    assert set(deps.outgoing) == {
        d.split("->")[1] for d in manifest.graph_allowed if d.startswith("D02->")
    }
    assert set(deps.incoming) == {
        d.split("->")[0] for d in manifest.graph_allowed if d.endswith("->D02")
    }


def test_domain_dependencies_cover_the_whole_direction_matrix(manifest):
    """Completeness: the union of per-domain answers reproduces [graph].allowed
    and [graph].lazy_only exactly — both directions."""
    strict, lazy = set(), set()
    for domain in manifest.domains:
        deps = domain_dependencies(REPO, domain)
        strict.update(f"{domain}->{dst}" for dst in deps.outgoing)
        strict.update(f"{src}->{domain}" for src in deps.incoming)
        lazy.update(f"{domain}->{dst}" for dst in deps.lazy_outgoing)
        lazy.update(f"{src}->{domain}" for src in deps.lazy_incoming)
    assert strict == set(manifest.graph_allowed)
    assert lazy == set(manifest.lazy_only)


def test_domain_dependencies_unknown_domain_teaches_the_vocabulary():
    with pytest.raises(ValueError, match="unknown domain.*D01"):
        domain_dependencies(REPO, "D99")


# ---------------------------------------------------------------------------
# facade_consumers
# ---------------------------------------------------------------------------

def test_facade_scan_matches_the_generated_facade_inventory(reexports):
    """Completeness against the gen/verify-pinned carrier: the runtime scan
    finds exactly the facade modules docs/v7next/FACADE_INVENTORY.md pins."""
    inventory_text = (REPO / "docs/v7next/FACADE_INVENTORY.md").read_text(encoding="utf-8")
    pinned = set()
    for line in inventory_text.splitlines():
        if line.startswith("| `") and line.count("|") >= 4:
            first = line.split("|")[1].strip()
            if first.startswith("`") and first.endswith("`"):
                pinned.add(first.strip("`"))
    assert pinned, "the facade inventory carrier parsed empty"
    assert set(reexports) == pinned


def test_facade_module_query_lists_its_import_consumers():
    rows = facade_consumers(REPO, "ouroboros/llm.py")
    consumers = {r.consumer for r in rows}
    assert "ouroboros/agent.py" in consumers  # from ouroboros.llm import LLMClient
    assert all(r.facade == "ouroboros/llm.py" and r.line > 0 for r in rows)


def test_facade_symbol_query_narrows_to_the_reexported_name(reexports):
    rows = facade_consumers(REPO, "add_usage")
    assert rows, "add_usage is a re-exported facade binding with real consumers"
    for row in rows:
        assert row.name == "add_usage"
        assert "add_usage" in reexports[row.facade]
    assert "ouroboros/loop_llm_call.py" in {r.consumer for r in rows}


def test_facade_query_on_a_non_facade_is_a_teaching_refusal():
    with pytest.raises(ValueError, match="not a facade module"):
        facade_consumers(REPO, "ouroboros/runtime_mode_policy.py")
    with pytest.raises(ValueError, match="no facade re-exports"):
        facade_consumers(REPO, "definitely_not_an_exported_name")


# ---------------------------------------------------------------------------
# persistence_entities_written_by
# ---------------------------------------------------------------------------

def test_writer_module_query_names_its_entities():
    entities = " | ".join(
        r.entity for r in persistence_entities_written_by(REPO, "supervisor/state.py")
    )
    assert "state/state.json" in entities
    ledger = persistence_entities_written_by(REPO, "ouroboros/usage_ledger.py")
    assert any("usage_attempts.jsonl" in r.entity for r in ledger)
    # A dotted spelling of the same writer answers identically.
    assert persistence_entities_written_by(REPO, "supervisor.state") == \
        persistence_entities_written_by(REPO, "supervisor/state.py")


def test_writer_function_name_query_matches_writer_prose():
    rows = persistence_entities_written_by(REPO, "save_settings")
    assert [r.entity for r in rows] == ["`settings.json`"]


def test_persistence_parser_reaches_every_table_row(manifest):
    """Completeness against the carrier: the parser yields one row per
    Path|Writer table line of docs/PERSISTENCE.md (nothing silently dropped),
    and every exact .py writer span resolves against the tree."""
    rows = _persistence_rows(REPO)
    text = (REPO / PERSISTENCE_DOC_RELPATH).read_text(encoding="utf-8")
    raw_rows = 0
    in_table = False
    for line in text.splitlines():
        if line.startswith("| Path | Writer |"):
            in_table = True
            continue
        if not line.startswith("|"):
            in_table = False
            continue
        if in_table and not set(line.replace("|", "").strip()) <= {"-", " ", ":"}:
            raw_rows += 1
    assert raw_rows == len(rows) and raw_rows >= 50, (raw_rows, len(rows))

    span_re = re.compile(r"`([^`]+\.py)`")
    for _section, _entity, writer in rows:
        for span in span_re.findall(writer):
            if any(ch in span for ch in "*<>"):
                continue  # glob/placeholder spans are labels, not paths
            resolved = (REPO / span).is_file() or any(
                module.endswith("/" + span) for module in manifest.modules
            )  # the doc shortens sibling writers to bare filenames in listings
            assert resolved, f"PERSISTENCE.md names a missing writer: {span}"


# ---------------------------------------------------------------------------
# protected_contracts_affected
# ---------------------------------------------------------------------------

def test_protected_diff_names_categories_and_contracts():
    diff = (
        "diff --git a/ouroboros/gateway/contracts.py b/ouroboros/gateway/contracts.py\n"
        "--- a/ouroboros/gateway/contracts.py\n"
        "+++ b/ouroboros/gateway/contracts.py\n"
        "@@ -1 +1 @@\n-1\n+2\n"
        "diff --git a/BIBLE.md b/BIBLE.md\n"
        "--- a/BIBLE.md\n+++ b/BIBLE.md\n@@ -1 +1 @@\n-1\n+2\n"
        "diff --git a/README.md b/README.md\n"
        "--- a/README.md\n+++ b/README.md\n@@ -1 +1 @@\n-1\n+2\n"
    )
    impact = protected_contracts_affected(REPO, diff)
    assert set(impact.paths) == {"ouroboros/gateway/contracts.py", "BIBLE.md", "README.md"}
    categories = {(p.path, p.category) for p in impact.protected}
    assert ("BIBLE.md", "safety-critical") in categories
    assert ("ouroboros/gateway/contracts.py", "frozen-contract") in categories
    assert not any(p.path == "README.md" for p in impact.protected)
    labels = {c.contract for c in impact.contracts}
    assert "gateway/contracts.py" in labels  # the §11.1 row itself
    assert "ProviderTestRequest" in labels   # a contract owned by the touched file


def test_protected_path_list_form_and_release_invariants():
    impact = protected_contracts_affected(
        REPO, ["supervisor/git_ops_reset.py", "ouroboros/safety.py"],
    )
    categories = dict((p.path, p.category) for p in impact.protected)
    assert categories == {
        "supervisor/git_ops_reset.py": "release-invariant",
        "ouroboros/safety.py": "safety-critical",
    }


def test_every_frozen_contract_row_is_reachable_from_its_owner_file():
    """Completeness against the generated inventory: feeding each row's own
    owner file back into the query must name that row."""
    triples = _frozen_contract_rows(REPO)
    assert len({label for label, _f, _r in triples}) >= 20
    for label, file_part, role in triples:
        impact = protected_contracts_affected(REPO, [file_part])
        assert (label, role) in {(c.contract, c.role) for c in impact.contracts}, (
            label, file_part, role,
        )


def test_protected_inventories_are_fully_categorized():
    from ouroboros.runtime_mode_policy import (
        FROZEN_CONTRACT_PATHS,
        RELEASE_INVARIANT_PATHS,
        SAFETY_CRITICAL_PATHS,
    )

    every = sorted(SAFETY_CRITICAL_PATHS | FROZEN_CONTRACT_PATHS | RELEASE_INVARIANT_PATHS)
    impact = protected_contracts_affected(REPO, every)
    assert {p.path for p in impact.protected} == set(every)


def test_paths_from_diff_accepts_lists_diffs_and_separated_strings():
    assert paths_from_diff("a.py, b.py  c.py") == ("a.py", "b.py", "c.py")
    assert paths_from_diff(["./x/y.py", "x/y.py"]) == ("x/y.py",)
    with pytest.raises(ValueError, match="no changed paths"):
        protected_contracts_affected(REPO, "   ")


# ---------------------------------------------------------------------------
# The model-facing seam: query_code op=architecture
# ---------------------------------------------------------------------------

def _ctx(tmp_path):
    from ouroboros.tools.registry import ToolContext

    return ToolContext(repo_dir=REPO, drive_root=tmp_path)


def test_query_code_architecture_op_serves_the_five_facts(tmp_path):
    from ouroboros.tools.query_code import _query_code

    out = _query_code(_ctx(tmp_path), op="architecture", query="owner_of ouroboros/llm.py")
    assert "D02" in out and "module_path" in out

    out = _query_code(_ctx(tmp_path), op="architecture", query="domain_dependencies D02")
    assert "imports:" in out and "imported by:" in out

    out = _query_code(
        _ctx(tmp_path), op="architecture",
        query="protected_contracts_affected BIBLE.md, ouroboros/gateway/contracts.py",
    )
    assert "safety-critical" in out and "frozen contract" in out


def test_query_code_architecture_op_refuses_bad_facts_and_foreign_roots(tmp_path):
    from ouroboros.tools.query_code import _query_code

    out = _query_code(_ctx(tmp_path), op="architecture", query="who_owns ouroboros/llm.py")
    assert "TOOL_ARG_ERROR" in out and "owner_of" in out  # the refusal teaches the facts

    out = _query_code(_ctx(tmp_path), op="architecture", query="owner_of")
    assert "TOOL_ARG_ERROR" in out and "argument" in out

    out = _query_code(
        _ctx(tmp_path), op="architecture", query="owner_of ouroboros/llm.py",
        root="user_files", path="/tmp",
    )
    # A non-code root never serves architecture facts: either the binding layer
    # or the op's own root guard refuses, typed.
    assert ("TOOL_ARG_ERROR" in out or "TOOL_ACCESS_BLOCKED" in out)
    assert "D02" not in out


def test_architecture_fact_vocabulary_is_closed():
    assert ARCHITECTURE_FACTS == (
        "owner_of",
        "domain_dependencies",
        "facade_consumers",
        "persistence_entities_written_by",
        "protected_contracts_affected",
    )
    with pytest.raises(ValueError, match="owner_of"):
        architecture_fact_rows(REPO, "unknown_fact x")
    assert FROZEN_INVENTORY_RELPATH.endswith("FROZEN_CONTRACTS_INVENTORY.md")
