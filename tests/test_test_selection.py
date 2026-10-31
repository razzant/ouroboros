"""The deterministic minimum and author widening on candidate-owned source."""
from pathlib import Path
import subprocess

import pytest

from ouroboros import test_selection as selection


def write(repo, path, text="def test_ok(): pass\n", newline=None):
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8", newline=newline)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setattr(selection, "FLOOR_TESTS", {"tests/test_floor.py": "a verification invariant"})
    write(tmp_path, "tests/test_floor.py")
    write(tmp_path, "tests/test_direct.py", "from ouroboros import leaf\n")
    write(tmp_path, "tests/test_nested.py", "def test_x():\n    import ouroboros.leaf as x\n")
    write(tmp_path, "tests/test_unrelated.py", "import ouroboros.leafish\n")
    write(tmp_path, "tests/test_indirect.py", "from ouroboros import consumer\n")
    write(tmp_path, "ouroboros/consumer.py", "from ouroboros import leaf\n")
    return tmp_path


def test_leaf_selects_only_direct_importers_and_changed_tests_above_floor(repo):
    result = selection.select_from_paths(repo, ["ouroboros/leaf.py", "tests/test_indirect.py"], base="base")
    assert result.tier == "selected"
    assert set(result.files) == {"tests/test_floor.py", "tests/test_direct.py", "tests/test_nested.py", "tests/test_indirect.py"}
    assert result.record()["omitted_files"] == 1
    assert result.policy and result.base == "base"


@pytest.mark.parametrize("path", [
    "tests/conftest.py", "tests/_shared.py", "tests/support.py", "tests/nested/conftest.py",
    "tests/test_floor.py", "pyproject.toml", "uv.lock", "requirements-runtime.lock", "site/pnpm-lock.yaml",
    "ouroboros/preflight_runner.py", "ouroboros/test_selection.py", "scripts/run_tests.py",
    "ouroboros/process_containment.py", "ouroboros/process_lifeline.py",
    "ouroboros/process_overload.py", "ouroboros/test_parallelism.py", "tests/fixtures/sample.json",
])
def test_verification_changes_require_full(repo, path):
    result = selection.select_from_paths(repo, [path], base="base")
    assert result.tier == "full"
    assert len(result.files) == result.available
    assert (path, "verification apparatus changed") in result.reasons


def test_no_product_hub_or_fraction_threshold(repo):
    for i in range(12):
        write(repo, f"tests/test_hub_{i}.py", "from ouroboros.leaf import api\n")
    result = selection.select_from_paths(repo, ["ouroboros/leaf.py"], base="base")
    assert result.tier == "selected"
    assert len(result.files) == 15


@pytest.mark.parametrize("path", ["server.py", "launcher.py", "ouroboros/__init__.py", "ouroboros/body_switch.py",
                                  "ouroboros/body_adoption.py", "supervisor/update_recovery.py",
                                  "ouroboros/launcher_bootstrap.py", "supervisor/git_ops_reset.py",
                                  "supervisor/update_merge.py"])
def test_body_cutover_and_recovery_changes_require_full(repo, path):
    # The package init carries the adoption hook: it is cutover, not an ordinary hub of imports.
    result = selection.select_from_paths(repo, [path], base="base")
    assert result.tier == "full" and (path, "body cutover/recovery changed") in result.reasons
    assert selection.select_from_paths(repo, ["ouroboros/body_candidate.py"], base="base").tier == "selected"


def test_inherited_widening_keeps_present_consumers_declared_full_and_both_reasons(repo):
    earlier = {"extra_tests": ["tests/test_indirect.py", "tests/test_deleted.py"], "full_tests": False,
               "test_reason": "reads leaf by path"}
    joined = selection.inherit_widening(repo, {"extra_tests": ("tests/test_direct.py",), "full_tests": False,
                                               "test_reason": "dynamic consumer"}, earlier)
    assert joined == {"extra_tests": ("tests/test_direct.py", "tests/test_indirect.py"), "full_tests": False,
                      "test_reason": "dynamic consumer; reads leaf by path"}
    assert selection.inherit_widening(repo, dict(extra_tests=(), full_tests=False, test_reason=""),
                                      {"full_tests": True, "test_reason": "broad"})["full_tests"] is True
    assert selection.inherit_widening(repo, dict(extra_tests=(), full_tests=False, test_reason=""), {}) == {
        "extra_tests": (), "full_tests": False, "test_reason": ""}


def test_author_can_widen_dynamic_consumers_or_request_full(repo):
    result = selection.select_from_paths(repo, ["prompts/SYSTEM.md"], base="base",
                                         extra_tests=["tests/test_indirect.py"], test_reason="dynamic consumer")
    assert result.files == ("tests/test_floor.py", "tests/test_indirect.py")
    assert ("tests/test_indirect.py", "author widening: dynamic consumer") in result.reasons
    full = selection.select_from_paths(repo, [], base="base", full_tests=True, test_reason="unknown impact")
    assert full.tier == "full" and ("*", "author requested full: unknown impact") in full.reasons


@pytest.mark.parametrize("extra,full,reason", [
    (["tests/test_direct.py"], False, ""), ([], True, ""),
    (["tests/missing.py"], False, "consumer"), (["../tests/test_direct.py"], False, "consumer"),
    (["/tests/test_direct.py"], False, "consumer"), (["-k"], False, "consumer"),
    ("tests/test_direct.py", False, "consumer"), ([], "yes", "consumer"), ([""], False, "consumer"), ([], False, None),
])
def test_invalid_widening_fails_before_execution(repo, extra, full, reason):
    with pytest.raises(ValueError):
        selection.select_from_paths(repo, [], base="base", extra_tests=extra, full_tests=full, test_reason=reason)


def test_a_missing_floor_is_a_block_even_for_full(repo):
    (repo / "tests/test_floor.py").unlink()
    with pytest.raises(ValueError, match="required gate floor is missing"):
        selection.select_from_paths(repo, ["uv.lock"], base="base")


def test_an_unparseable_import_graph_requires_explicit_widening(repo):
    write(repo, "tests/test_direct.py", "not python !!!")
    with pytest.raises(ValueError, match="import scan unavailable"):
        selection.select_from_paths(repo, ["ouroboros/leaf.py"], base="base")
    assert selection.select_from_paths(repo, ["ouroboros/leaf.py"], base="base",
                                       full_tests=True, test_reason="inspect collection failure").tier == "full"


def test_repeated_selection_parses_only_modules_whose_content_changed(repo, monkeypatch):
    parsed = []
    scan = selection.direct_imports
    monkeypatch.setattr(selection, "_IMPORT_SCANS", {})
    monkeypatch.setattr(selection, "direct_imports", lambda source, path: parsed.append(path) or scan(source, path))
    first = selection.select_from_paths(repo, ["ouroboros/leaf.py"], base="base")
    assert len(parsed) == 5  # every test module once
    assert selection.select_from_paths(repo, ["ouroboros/leaf.py"], base="base") == first and len(parsed) == 5
    write(repo, "tests/test_unrelated.py", "from ouroboros import leaf\n")  # an edit is read again
    assert "tests/test_unrelated.py" in selection.select_from_paths(repo, ["ouroboros/leaf.py"], base="base").files
    assert parsed[5:] == ["tests/test_unrelated.py"]


def test_git_candidate_includes_dirty_untracked_deleted_and_renamed_inputs(repo):
    def git(*args):
        return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True).stdout.decode().strip()
    git("init", "-q")
    git("config", "user.email", "selection@example.invalid")
    git("config", "user.name", "Selection tests")
    write(repo, "ouroboros/leaf.py", "before = 1\n")
    git("add", ".")
    git("commit", "-qm", "base")
    base = git("rev-parse", "HEAD")
    git("mv", "ouroboros/leaf.py", "ouroboros/renamed.py")
    write(repo, "tests/test_new.py")
    write(repo, "tests/test_indirect.py", "def test_changed(): pass\n")
    result = selection.select_tests(repo)
    assert result.base == base
    assert set(result.changed) == {"ouroboros/leaf.py", "ouroboros/renamed.py", "tests/test_new.py", "tests/test_indirect.py"}
    assert {"tests/test_direct.py", "tests/test_nested.py", "tests/test_new.py"} <= set(result.files)
    assert selection.refresh_selection(repo, result) == result
    with pytest.raises(subprocess.CalledProcessError):
        selection.select_tests(repo, base="missing-base")


@pytest.mark.parametrize("eol", ["\n", "\r\n"], ids=["lf", "crlf"])
@pytest.mark.parametrize("autocrlf", ["false", "true"])
def test_a_release_bump_alone_stays_selected_but_other_carrier_edits_run_full(repo, eol, autocrlf):
    def write_carrier(path, text):
        target = repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(text.replace("\n", eol).encode("utf-8"))

    def git(*args):
        return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True).stdout.decode().strip()
    git("init", "-q")
    git("config", "user.email", "selection@example.invalid")
    git("config", "user.name", "Selection tests")
    git("config", "core.autocrlf", autocrlf)
    git("config", "core.safecrlf", "false")
    project = '[project]\nname = "ouroboros"\nversion = "{}"\ndependencies = [{}]\n'
    write_carrier("VERSION", "7.7.0-rc.1\n")
    write_carrier("pyproject.toml", project.format("7.7.0rc1", ""))
    write_carrier("web/package.json", '{\n  "version": "7.7.0-rc.1"\n}\n')
    git("add", ".")
    git("commit", "-qm", "base")
    write_carrier("VERSION", "7.7.0-rc.2\n")  # P9: every numbered release bumps the carriers
    write_carrier("pyproject.toml", project.format("7.7.0rc2", ""))
    write_carrier("web/package.json", '{\n  "version": "7.7.0-rc.2"\n}\n')
    assert selection.select_tests(repo).tier == "selected"
    write_carrier("pyproject.toml", project.format("7.7.0rc2", '"requests"'))
    result = selection.select_tests(repo)
    assert result.tier == "full" and ("pyproject.toml", "verification apparatus changed") in result.reasons


def test_floor_exists_in_this_checkout():
    root = Path(__file__).resolve().parents[1]
    assert all((root / path).is_file() and invariant for path, invariant in selection.FLOOR_TESTS.items())


def test_reviewed_tool_binds_widening_and_force_uses_full(repo, monkeypatch):
    from types import SimpleNamespace
    from ouroboros.tools import git, review_helpers

    ctx = SimpleNamespace(repo_dir=repo)
    seen = []
    monkeypatch.setattr(git, "_repo_commit_push", lambda c, message, *a, **kw: seen.append(dict(c._commit_test_request)) or "ok")
    monkeypatch.setattr(git, "name_review_record", lambda c, result: result)
    assert git._commit_reviewed(ctx, "commit", extra_tests=["tests/test_indirect.py"], test_reason="runtime lookup") == "ok"
    assert seen == [{"extra_tests": ("tests/test_indirect.py",), "full_tests": False, "test_reason": "runtime lookup"}]
    runs = []
    monkeypatch.setenv("OUROBOROS_PRE_PUSH_TESTS", "1")
    monkeypatch.setattr("ouroboros.preflight_runner.run_hermetic_pytest", lambda *a, **kw: runs.append(kw))
    review_helpers._run_review_preflight_tests(ctx)
    review_helpers._run_review_preflight_tests(ctx, force=True)
    assert runs[0]["selected"] is True and runs[0]["extra_tests"] == ("tests/test_indirect.py",)
    assert not runs[1].get("selected")
    assert "base" not in runs[0]  # an unbound pre-commit run keeps its HEAD base
    schema = next(t.schema for t in git.get_tools() if t.name == "commit_reviewed")
    assert {"extra_tests", "full_tests", "test_reason"} <= schema["parameters"]["properties"].keys()


def test_a_bound_candidates_post_commit_and_forced_runs_stay_selected_with_its_qualification_scope(monkeypatch):
    from types import SimpleNamespace
    from ouroboros import body_candidate

    request = {"extra_tests": (), "full_tests": False, "test_reason": ""}
    ctx = SimpleNamespace(_commit_test_request=request)
    scoped = {**request, "base": "fork", "extra_tests": ("tests/test_inherited.py",), "test_reason": "earlier"}
    monkeypatch.setattr(body_candidate, "qualification_request", lambda c, r: dict(scoped))
    monkeypatch.setattr(body_candidate, "is_bound", lambda c: False)
    assert selection.commit_test_options(ctx, post_commit=True) == {"selected": True, **request}
    assert selection.commit_test_options(ctx, force=True, post_commit=True) == {}  # serving evolution, managed
    monkeypatch.setattr(body_candidate, "is_bound", lambda c: True)
    assert selection.commit_test_options(ctx) == {"selected": True, **request}  # pre-commit: review admission
    assert selection.commit_test_options(ctx, post_commit=True) == {"selected": True, **scoped}
    assert selection.commit_test_options(ctx, force=True, post_commit=True) == {"selected": True, **scoped}
    assert selection.commit_test_options(SimpleNamespace(), force=True, post_commit=True) == {}  # never requested
