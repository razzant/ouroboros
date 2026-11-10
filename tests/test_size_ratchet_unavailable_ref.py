"""An unavailable size-ratchet comparison ref is partial evidence, never a PASS.

When the committed manifest or the staged tree cannot be read, live exactness
still runs and one ``partial:`` finding names what was skipped —
never a silent bootstrap or skip that empties the result. The pairwise CI
transition refuses to pass without a readable parent. A malformed live
manifest (the subject, not a comparison) still raises.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros.review import (
    MAX_MODULE_LINES,
    SizeRatchetRefUnavailable,
    _staged_manifest_inventory,
    parse_size_ratchet_manifest,
    validate_size_ratchet,
    validate_size_ratchet_candidate,
    validate_size_ratchet_transition_against_base,
)
from ouroboros.tools.health import _codebase_health
from ouroboros.tools.review_helpers import check_worktree_readiness
from scripts import regenerate_size_ratchet as regenerate
from tests import test_smoke as ci_consumer
from tests.test_repo_health_smoke import _bootstrap_repo, _git, _manifest, _write_lines, _write_manifest


MANIFEST = "ouroboros/size_ratchet_manifest.py"
GROWTH = "new module debt above 1600 lines: new.py"
DRIFT = "GIANT_PATHS contains stale entry: 'stale.py'"


def _local_findings(repo: Path, findings: list[str]) -> None:
    assert validate_size_ratchet(repo) == findings
    warnings = check_worktree_readiness(repo, information=[])
    report = _codebase_health(SimpleNamespace(repo_dir=repo))
    for finding in findings:
        assert f"official CI will enforce: {finding}" in warnings
        assert f"  - {finding}" in report
    assert "validator unavailable" not in report
    assert "matches the live tree and is shrink-only against the committed authority" not in report
    print(f"LOCAL_CONSUMERS {repo.name}: {findings!r}\nREADINESS {warnings!r}\nHEALTH\n{report}")


def _drop_object(repo: Path, oid: str) -> None:
    loose = repo / ".git" / "objects" / oid[:2] / oid[2:]
    loose.chmod(0o600)  # Git writes loose objects read-only; Windows refuses to unlink those.
    loose.unlink()


def _commit(repo: Path, message: str) -> None:
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", message)


def _ratchet_then_growth(repo: Path, *, commit_growth: bool = False) -> None:
    """HEAD commits a debt-free manifest; the next tree self-authorizes new module debt."""
    baseline = _bootstrap_repo(repo)
    _write_manifest(repo, _manifest(sha=baseline))
    _commit(repo, "bootstrap ratchet")
    _write_lines(repo / "new.py", MAX_MODULE_LINES + 1)
    _write_manifest(repo, _manifest(giant_paths=frozenset({"new.py"}), sha=baseline))
    if commit_growth:
        _commit(repo, "grow debt")


@pytest.mark.parametrize("object_ref", [f"HEAD:{MANIFEST}", "HEAD^{tree}", "HEAD"])
def test_unreadable_committed_manifest_is_partial_at_every_local_consumer(
    tmp_path: Path, monkeypatch, object_ref: str,
) -> None:
    valid = tmp_path / "valid"
    _ratchet_then_growth(valid)
    _local_findings(valid, [GROWTH])

    missing = tmp_path / "missing"
    _ratchet_then_growth(missing)
    _drop_object(missing, _git(missing, "rev-parse", object_ref))

    # Before: the unreadable blob passed as "no committed manifest", so the
    # growth bootstrapped into an empty list and a green health line.
    [finding] = validate_size_ratchet(missing)
    assert finding.startswith("partial: committed size-ratchet manifest unavailable (SizeRatchetRefUnavailable: ")
    assert finding.endswith("; not checked: shrink-only transition and staged index")
    _local_findings(missing, [finding])
    current = (missing / MANIFEST).read_text(encoding="utf-8")
    assert validate_size_ratchet_candidate(missing, current) == [finding.replace(" and staged index", "")]
    (missing / MANIFEST).write_text(current.replace('"new.py",', '"new.py",\n    "stale.py",'), encoding="utf-8")
    _local_findings(missing, [DRIFT, finding])
    before = (missing / MANIFEST).read_bytes()
    monkeypatch.setattr(regenerate, "REPO_ROOT", missing)
    assert regenerate.main([]) == 2
    assert (missing / MANIFEST).read_bytes() == before


def test_unparseable_committed_manifest_keeps_live_findings(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    baseline = _bootstrap_repo(repo)
    older_schema = regenerate._render(_manifest(sha=baseline)).replace("BYTE_DEBT = {\n}\n", "")
    (repo / "ouroboros").mkdir()
    (repo / MANIFEST).write_text(older_schema, encoding="utf-8")
    _commit(repo, "older-schema manifest")
    _write_lines(repo / "new.py", MAX_MODULE_LINES + 1)
    _write_manifest(repo, _manifest(giant_paths=frozenset({"stale.py"}), sha=baseline))

    *live, gap = validate_size_ratchet(repo)

    assert live == ["GIANT_PATHS missing live entry: 'new.py'", "GIANT_PATHS contains stale entry: 'stale.py'"]
    assert gap.startswith(
        "partial: committed size-ratchet manifest unavailable "
        "(ValueError: size manifest missing assignments: BYTE_DEBT)"
    )
    _local_findings(repo, [*live, gap])
    (repo / MANIFEST).write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="size manifest"):
        validate_size_ratchet(repo)


def test_unmerged_index_keeps_committed_checks_and_names_the_staged_gap(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    baseline = _bootstrap_repo(repo, files={"small.py": "x = 1\n", "conflict.txt": "base\n"})
    _write_manifest(repo, _manifest(sha=baseline))
    _commit(repo, "bootstrap ratchet")
    trunk = _git(repo, "branch", "--show-current")
    _git(repo, "checkout", "-q", "-b", "side")
    (repo / "conflict.txt").write_text("side\n", encoding="utf-8")
    _commit(repo, "side")
    _git(repo, "checkout", "-q", trunk)
    (repo / "conflict.txt").write_text("trunk\n", encoding="utf-8")
    _commit(repo, "trunk")
    subprocess.run(["git", "merge", "-q", "side"], cwd=repo, capture_output=True, check=False)
    assert _git(repo, "ls-files", "-u")

    # An unmerged index must not silently disappear from an otherwise clean result.
    [gap] = validate_size_ratchet(repo)
    assert gap.startswith("partial: staged index tree unavailable (CalledProcessError: ")
    assert "unmerged" in gap
    assert gap.endswith("; not checked: staged exactness and transition")

    _write_lines(repo / "new.py", MAX_MODULE_LINES + 1)
    _write_manifest(repo, _manifest(giant_paths=frozenset({"new.py"}), sha=baseline))
    _local_findings(repo, [GROWTH, gap])
    current = (repo / MANIFEST).read_text(encoding="utf-8")
    (repo / MANIFEST).write_text(current.replace('"new.py",', '"new.py",\n    "stale.py",'), encoding="utf-8")
    findings = validate_size_ratchet(repo)
    assert DRIFT in findings and GROWTH in findings and gap in findings
    _local_findings(repo, findings)


def test_pairwise_transition_refuses_to_pass_without_a_readable_parent(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    _ratchet_then_growth(source, commit_growth=True)
    unresolvable = "deadbeef" * 5
    valid_base = _git(source, "rev-parse", "HEAD~1")
    monkeypatch.setattr(ci_consumer, "REPO", source)
    for base in (valid_base, unresolvable, "0" * 40):
        assert validate_size_ratchet_transition_against_base(source, base) == [GROWTH]
        monkeypatch.setenv("OURO_SIZE_RATCHET_BASE_REF", base)
        with pytest.raises(AssertionError, match=GROWTH) as blocked:
            ci_consumer.test_size_ratchet_transition_against_explicit_base()
        print(f"CI_BLOCKED base={base}: {blocked.value}")

    # Before: a shallow boundary hid the parent, so the degraded base found no
    # manifest and the transition was skipped as an empty (passing) list.
    shallow = tmp_path / "shallow"
    subprocess.run(["git", "clone", "-q", "--depth", "1", source.as_uri(), str(shallow)], check=True)
    # Local HEAD exactness needs no parent; CI's fallback comparison does.
    assert validate_size_ratchet(shallow) == []
    monkeypatch.setattr(ci_consumer, "REPO", shallow)
    for base in (None, unresolvable):
        monkeypatch.setenv("OURO_SIZE_RATCHET_BASE_REF", base or "")
        with pytest.raises(SizeRatchetRefUnavailable, match=r"comparison ref \w+ is unavailable") as blocked:
            ci_consumer.test_size_ratchet_transition_against_explicit_base()
        print(f"CI_SHALLOW_BLOCKED base={base}: {blocked.value}")

    _drop_object(source, _git(source, "rev-parse", f"HEAD~1:{MANIFEST}"))
    monkeypatch.setattr(ci_consumer, "REPO", source)
    for base in (valid_base, unresolvable):
        monkeypatch.setenv("OURO_SIZE_RATCHET_BASE_REF", base)
        with pytest.raises(SizeRatchetRefUnavailable, match=r"comparison ref \w+ is unavailable") as blocked:
            ci_consumer.test_size_ratchet_transition_against_explicit_base()
        print(f"CI_BROKEN_BLOB_BLOCKED base={base}: {blocked.value}")


@pytest.mark.parametrize("missing_path", ["staged.py", MANIFEST, "nested"])
def test_missing_staged_objects_keep_live_findings(tmp_path: Path, missing_path: str) -> None:
    repo = tmp_path / "repo"
    _ratchet_then_growth(repo)
    (repo / "staged.py").write_text("staged_only = 42\n", encoding="utf-8")
    (repo / "nested").mkdir()
    (repo / "nested" / "small.py").write_text("nested_only = 43\n", encoding="utf-8")
    _git(repo, "add", "staged.py", "nested")
    if missing_path == MANIFEST:
        (repo / MANIFEST).write_text(_git(repo, "show", f"HEAD:{MANIFEST}") + "\n# staged copy\n", encoding="utf-8")
        _git(repo, "add", MANIFEST)
    tree = _git(repo, "write-tree")  # Cache before dropping the fixture's unique blob.
    _drop_object(repo, _git(repo, "rev-parse", f"{tree}:{missing_path}"))
    error_type = subprocess.CalledProcessError if missing_path == "nested" else SizeRatchetRefUnavailable
    with pytest.raises(error_type):
        _staged_manifest_inventory(repo, tree, MANIFEST)
    # Preserve the authority's provenance while introducing live exactness drift.
    sha = parse_size_ratchet_manifest(_git(repo, "show", f"HEAD:{MANIFEST}")).baseline_source_sha
    _write_manifest(repo, _manifest(giant_paths=frozenset({"new.py", "stale.py"}), sha=sha))
    findings = validate_size_ratchet(repo)
    assert DRIFT in findings and GROWTH in findings
    assert findings[-1].startswith(f"partial: staged index inventory unavailable ({error_type.__name__}:")
    _local_findings(repo, findings)


def test_shallow_missing_manifest_does_not_bootstrap(tmp_path: Path) -> None:
    source = tmp_path / "source"
    baseline = _bootstrap_repo(source)
    _write_manifest(source, _manifest(sha=baseline))
    current = (source / MANIFEST).read_text(encoding="utf-8")
    _commit(source, "bootstrap")
    (source / MANIFEST).unlink()
    _commit(source, "remove manifest")
    shallow = tmp_path / "shallow"
    subprocess.run(["git", "clone", "-q", "--depth", "1", source.as_uri(), str(shallow)], check=True)
    (shallow / MANIFEST).parent.mkdir(exist_ok=True)
    (shallow / MANIFEST).write_text(current, encoding="utf-8")
    [gap] = validate_size_ratchet(shallow)
    assert gap.startswith("partial: committed size-ratchet manifest unavailable")
    _local_findings(shallow, [gap])


def test_readable_absence_and_valid_bases_still_pass(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    baseline = _bootstrap_repo(repo)
    _write_manifest(repo, _manifest(sha=baseline))
    assert validate_size_ratchet(repo) == []  # Actual readable absence bootstraps.
    _commit(repo, "bootstrap")
    base = _git(repo, "rev-parse", "HEAD")
    (repo / "small.py").write_text("small = 2\n", encoding="utf-8")
    _commit(repo, "ordinary change")
    monkeypatch.setattr(ci_consumer, "REPO", repo)
    for ref in (base, "deadbeef" * 5):
        monkeypatch.setenv("OURO_SIZE_RATCHET_BASE_REF", ref)
        ci_consumer.test_size_ratchet_manifest_matches_live_tree()
        ci_consumer.test_size_ratchet_transition_against_explicit_base()
        print(f"CI_ALLOWED base={ref}")
    assert "matches the live tree and is shrink-only against the committed authority" in _codebase_health(
        SimpleNamespace(repo_dir=repo)
    )
