"""CPL-1 verify half (plan §7.1): the production domain manifest
``ouroboros/domains.toml`` is complete over the tracked runtime population and
its generated baseline sections match the live tree.

The generator half is ``python scripts/check_domains.py --write``; these tests
are the CI pin that staleness = red:

- a tracked runtime module without a manifest row (or a stale row) is red;
- a new cross-domain strict import direction outside ``[graph].allowed`` is red;
- cycle-group growth beyond the pinned SCC ceiling is red;
- lazy/dynamic classification drift is red;
- a new cross-domain literal-copy function body is red (the banned "fix" of a
  dependency edge);
- ``docs/DOMAIN_MAP.md`` not byte-identical to its regeneration is red.

Synthetic-tree tests prove the detection branches actually fire (a gate whose
red path is unreachable is coverage in form only).
"""
from __future__ import annotations

import importlib.util
import pathlib
import textwrap

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "check_domains", REPO_ROOT / "scripts" / "check_domains.py")
check_domains = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_domains)

from scripts.domain_graph import (  # noqa: E402
    LAZY,
    STRICT,
    Manifest,
    build_import_graph,
    computed_graph_sections,
    cycle_groups_of,
    duplicate_bodies,
    load_manifest,
    tracked_population,
)


@pytest.fixture(scope="module")
def manifest() -> Manifest:
    return load_manifest()


@pytest.fixture(scope="module")
def graph(manifest):
    return build_import_graph(manifest, REPO_ROOT)


@pytest.fixture(scope="module")
def sections(manifest, graph):
    return computed_graph_sections(manifest, graph)


# ---------------------------------------------------------------------------
# Live-tree pins (the verify half of the gen/verify pair)
# ---------------------------------------------------------------------------

def test_manifest_covers_the_tracked_population_exactly(manifest):
    tracked = tracked_population(REPO_ROOT)
    missing = sorted(tracked - set(manifest.modules))
    stale = sorted(set(manifest.modules) - tracked)
    assert not missing, (
        "tracked runtime modules without a domain row (add them to "
        f"ouroboros/domains.toml [modules]): {missing}")
    assert not stale, (
        f"manifest rows that are no longer tracked modules (remove them): {stale}")


def test_every_module_maps_to_a_known_domain(manifest):
    unknown = sorted({d for d in manifest.modules.values() if d not in manifest.domains})
    assert not unknown, f"manifest names unknown domains: {unknown}"


def test_strict_direction_matrix_matches_the_pinned_baseline(manifest, sections):
    computed = set(sections["allowed"])
    pinned = set(manifest.graph_allowed)
    new = sorted(computed - pinned)
    gone = sorted(pinned - computed)
    assert not new, (
        "NEW cross-domain strict dependency directions (an owner decision — make "
        "it visible by regenerating the manifest: python scripts/check_domains.py "
        f"--write): {new}")
    assert not gone, (
        "the tree dropped pinned directions — bank the shrink with "
        f"`python scripts/check_domains.py --write`: {gone}")


def test_cycle_groups_match_the_pinned_ceiling(manifest, sections):
    assert sections["cycle_groups"] == manifest.cycle_groups, (
        "strict-quotient cycle groups drifted from the pinned SCC ceiling "
        f"(computed {sections['cycle_groups']} != pinned {manifest.cycle_groups}). "
        "Growth = break the new edge instead of widening the ceiling; shrinkage = "
        "bank it with `python scripts/check_domains.py --write`. The target is [].")


def test_lazy_and_dynamic_classification_matches_the_pin(manifest, sections):
    assert sections["lazy_only"] == manifest.lazy_only, (
        "lazy-only cross-domain pairs drifted — regenerate the manifest")
    assert sections["dynamic_pairs"] == manifest.dynamic_pairs, (
        "dynamic-import cross-domain pairs drifted — regenerate the manifest")


def test_no_cross_domain_literal_copies_beyond_the_baseline(manifest):
    rows = duplicate_bodies(manifest, REPO_ROOT)
    pinned = set(manifest.duplicates_allowed)
    new = [r for r in rows if r not in pinned]
    gone = sorted(pinned - set(rows))
    assert not new, (
        "cross-domain literal-copy function bodies (the banned cycle 'fix' — "
        f"import the single owner instead): {new}")
    assert not gone, (
        f"literal-copy baseline rows no longer observed — bank the shrink: {gone}")


def test_domain_map_is_byte_identical_to_its_regeneration(manifest):
    rendered = check_domains.render_domain_map(manifest)
    on_disk = check_domains.DOMAIN_MAP_PATH.read_text(encoding="utf-8")
    assert on_disk == rendered, (
        "docs/DOMAIN_MAP.md is stale — regenerate with "
        "`python scripts/check_domains.py --write`")


def test_checker_cli_is_green_on_this_tree(capsys):
    assert check_domains.main([]) == 0
    out = capsys.readouterr().out
    assert "OK: domain manifest complete" in out


# ---------------------------------------------------------------------------
# Synthetic-tree tests: the red branches are reachable
# ---------------------------------------------------------------------------

def _mini_manifest(tmp_path: pathlib.Path, modules: dict[str, str]) -> Manifest:
    return Manifest(
        path=tmp_path / "domains.toml", raw_bytes=b"", meta={},
        domains={"DA": "Alpha", "DB": "Beta"},
        modules=modules, proposed=set(),
    )


def test_strict_import_produces_a_cross_domain_pair(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "pkg" / "a.py").write_text("from pkg import b\n", encoding="utf-8")
    (tmp_path / "pkg" / "b.py").write_text("X = 1\n", encoding="utf-8")
    m = _mini_manifest(tmp_path, {
        "pkg/__init__.py": "DA", "pkg/a.py": "DA", "pkg/b.py": "DB"})
    g = build_import_graph(m, tmp_path)
    sections = computed_graph_sections(m, g)
    assert sections["allowed"] == ["DA->DB"]
    assert sections["cycle_groups"] == []


def test_lazy_import_is_classified_out_of_the_strict_graph(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "pkg" / "a.py").write_text(
        "def f():\n    from pkg import b\n    return b\n", encoding="utf-8")
    (tmp_path / "pkg" / "b.py").write_text("X = 1\n", encoding="utf-8")
    m = _mini_manifest(tmp_path, {
        "pkg/__init__.py": "DA", "pkg/a.py": "DA", "pkg/b.py": "DB"})
    g = build_import_graph(m, tmp_path)
    assert not g.quotient(STRICT)
    assert list(g.quotient(LAZY)) == [("DA", "DB")]
    assert computed_graph_sections(m, g)["lazy_only"] == ["DA->DB"]


def test_mutual_imports_form_a_cycle_group(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "pkg" / "a.py").write_text("from pkg import b\n", encoding="utf-8")
    (tmp_path / "pkg" / "b.py").write_text("from pkg import a\n", encoding="utf-8")
    m = _mini_manifest(tmp_path, {
        "pkg/__init__.py": "DA", "pkg/a.py": "DA", "pkg/b.py": "DB"})
    g = build_import_graph(m, tmp_path)
    assert cycle_groups_of(["DA", "DB"], g.quotient(STRICT)) == [["DA", "DB"]]


_COPY_BODY = textwrap.dedent(
    '''
    def shared_fix(value):
        total = 0
        for item in value:
            if item is None:
                continue
            if isinstance(item, str):
                total += len(item)
            else:
                total += int(item)
        if total > 100:
            return total - 100
        return total
    ''')


def test_literal_copy_across_domains_is_detected(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "pkg" / "a.py").write_text(_COPY_BODY, encoding="utf-8")
    (tmp_path / "pkg" / "b.py").write_text(_COPY_BODY, encoding="utf-8")
    m = _mini_manifest(tmp_path, {
        "pkg/__init__.py": "DA", "pkg/a.py": "DA", "pkg/b.py": "DB"})
    rows = duplicate_bodies(m, tmp_path)
    assert len(rows) == 1
    assert "pkg/a.py::shared_fix" in rows[0] and "pkg/b.py::shared_fix" in rows[0]


def test_literal_copy_within_one_domain_is_not_flagged(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "pkg" / "a.py").write_text(_COPY_BODY, encoding="utf-8")
    (tmp_path / "pkg" / "b.py").write_text(_COPY_BODY, encoding="utf-8")
    m = _mini_manifest(tmp_path, {
        "pkg/__init__.py": "DA", "pkg/a.py": "DA", "pkg/b.py": "DA"})
    assert duplicate_bodies(m, tmp_path) == []


def test_small_bodies_are_below_the_literal_copy_floor(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    small = "def tiny():\n    return 1\n"
    (tmp_path / "pkg" / "a.py").write_text(small, encoding="utf-8")
    (tmp_path / "pkg" / "b.py").write_text(small, encoding="utf-8")
    m = _mini_manifest(tmp_path, {
        "pkg/__init__.py": "DA", "pkg/a.py": "DA", "pkg/b.py": "DB"})
    assert duplicate_bodies(m, tmp_path) == []


def test_generated_block_replacement_is_idempotent():
    block = check_domains.render_generated_block(
        {"allowed": ["DA->DB"], "cycle_groups": [], "lazy_only": [],
         "dynamic_pairs": []}, [])
    base = "# human header\n[modules]\n\"a.py\" = \"DA\"\n"
    once = check_domains.manifest_with_generated_block(base, block)
    twice = check_domains.manifest_with_generated_block(once, block)
    assert once == twice
    assert once.count(check_domains.GENERATED_BEGIN) == 1
