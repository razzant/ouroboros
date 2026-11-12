"""The deterministic minimum and author widening on candidate-owned source."""
from pathlib import Path

import pytest

from ouroboros import test_selection as selection
import subprocess


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
    "tests/conftest.py", "pyproject.toml", "tests/fixtures/sample.json",
    "ouroboros/preflight_probe.py", "ouroboros/test_bytecode.py",  # the gate's plugin and its warm: apparatus too
    "ouroboros/host_memory.py",  # the memory reading behind test admission and the overload stop
])
def test_verification_changes_require_full(repo, path):
    result = selection.select_from_paths(repo, [path], base="base")
    assert result.tier == "full"
    assert len(result.files) == result.available
    assert (path, "verification apparatus changed") in result.reasons


def test_a_changed_helper_selects_what_reaches_it_through_other_helpers_and_nothing_else(repo):
    write(repo, "tests/_world.py", "world = 1\n")
    write(repo, "tests/_stand.py", "from tests._world import world\n")
    write(repo, "tests/fixtures/rows.py", "from tests import _stand\n")
    write(repo, "tests/_other.py", "other = 1\n")
    importers = {
        "tests/test_direct.py": "from tests._world import world\n",
        "tests/test_through.py": "from tests._stand import world\n",
        "tests/test_deep.py": "def test_x():\n    from tests.fixtures import rows\n",
    }
    for path, source in {**importers, "tests/test_other.py": "from tests._other import other\n"}.items():
        write(repo, path, source)
    result = selection.select_from_paths(repo, ["tests/_world.py"], base="base")
    assert result.tier == "selected"
    assert set(result.files) == {"tests/test_floor.py", *importers}
    assert ("tests/test_through.py", "direct import: tests._stand") in result.reasons


@pytest.mark.parametrize("configuration", ["tests/conftest.py", "tests/system_e2e/conftest.py", "tests/ci_evidence.py"])
def test_a_helper_pytest_configuration_loads_is_full_and_a_test_only_helper_is_not(repo, configuration):
    """The fixture a configuration file reaches runs for every test, through another helper too."""
    write(repo, "tests/_guard.py", "guard = 1\n")
    write(repo, "tests/_bridge.py", "from tests._guard import guard\n")
    write(repo, "tests/test_guarded.py", "from tests._guard import guard\n")
    assert selection.select_from_paths(repo, ["tests/_guard.py"], base="base").tier == "selected"
    write(repo, configuration, "from tests._bridge import guard\n")
    result = selection.select_from_paths(repo, ["tests/_guard.py"], base="base")
    assert result.tier == "full" and ("tests/_guard.py", "verification apparatus changed") in result.reasons


def test_a_child_script_nothing_imports_is_full_and_a_deleted_helper_selects_stale_importers(repo):
    """A test starts it by path: its consumers cannot be traced, and the floor alone would prove nothing."""
    write(repo, "tests/_child.py", "print('started by path')\n")
    write(repo, "tests/_only_for_child.py", "value = 1\n")
    write(repo, "tests/s1_host.py", "from tests._only_for_child import value\n")
    for path in ("tests/_child.py", "tests/_only_for_child.py"):
        assert selection.select_from_paths(repo, [path], base="base").tier == "full"
    write(repo, "tests/test_stale.py", "from tests._gone import helper\n")
    result = selection.select_from_paths(repo, ["tests/_gone.py"], base="base")
    assert result.tier == "selected" and "tests/test_stale.py" in result.files


def test_a_broken_helper_refuses_the_graph_only_when_a_helper_must_be_traced(repo):
    write(repo, "tests/_broken.py", "def broken(:\n")
    write(repo, "tests/_fine.py", "fine = 1\n")
    assert selection.select_from_paths(repo, ["ouroboros/leaf.py"], base="base").tier == "selected"
    with pytest.raises(ValueError, match="import scan unavailable for tests/_broken.py"):
        selection.select_from_paths(repo, ["tests/_fine.py"], base="base")
    assert selection.select_from_paths(repo, ["tests/_fine.py", "pyproject.toml"], base="base").tier == "full"


def test_no_product_hub_or_fraction_threshold(repo):
    for i in range(12):
        write(repo, f"tests/test_hub_{i}.py", "from ouroboros.leaf import api\n")
    result = selection.select_from_paths(repo, ["ouroboros/leaf.py"], base="base")
    assert result.tier == "selected"
    assert len(result.files) == 15


def test_a_changed_test_module_selects_its_direct_importers_and_no_other_consumer(repo):
    write(repo, "tests/test_hub.py")
    write(repo, "tests/test_hub_extra.py")
    importers = {
        "tests/test_by_from.py": "from tests.test_hub import helper\n",
        "tests/test_by_module.py": "import tests.test_hub as hub\n",
        "tests/test_by_package.py": "from tests import test_hub\n",
        "tests/test_by_function.py": "def test_x():\n    from tests.test_hub import helper\n",
        **{f"tests/test_fan_{i}.py": "from tests.test_hub import helper\n" for i in range(12)},  # no fan-out threshold
    }
    for path, source in importers.items():
        write(repo, path, source)
    write(repo, "tests/test_lookalike.py", "from tests.test_hub_extra import helper\n")  # shares the hub's prefix only
    write(repo, "tests/test_transitive.py", "from tests.test_by_from import helper\n")  # imports an importer
    result = selection.select_from_paths(repo, ["tests/test_hub.py"], base="base")
    assert result.tier == "selected"
    assert set(result.files) == {"tests/test_floor.py", "tests/test_hub.py", *importers}
    assert ("tests/test_hub.py", "changed test module") in result.reasons
    assert all((path, "direct import: tests.test_hub") in result.reasons for path in importers)


def test_reasons_name_every_changed_module_an_importer_binds_and_an_unimported_test_stays_alone(repo):
    write(repo, "tests/test_hub.py")
    write(repo, "tests/test_both.py", "from ouroboros import leaf\nfrom tests.test_hub import helper\n")
    ordinary = selection.select_from_paths(repo, ["tests/test_unrelated.py"], base="base")
    assert ordinary.files == ("tests/test_floor.py", "tests/test_unrelated.py")
    result = selection.select_from_paths(repo, ["ouroboros/leaf.py", "tests/test_hub.py"], base="base")
    assert ("tests/test_direct.py", "direct import: ouroboros.leaf") in result.reasons
    assert ("tests/test_both.py", "direct import: ouroboros.leaf, tests.test_hub") in result.reasons


def test_an_unreadable_module_blocks_a_test_only_change_unless_full_is_requested(repo):
    write(repo, "tests/test_hub.py")
    write(repo, "tests/test_broken.py", "not python !!!")
    with pytest.raises(ValueError, match="import scan unavailable for tests/test_broken.py"):
        selection.select_from_paths(repo, ["tests/test_hub.py"], base="base")
    full = selection.select_from_paths(repo, ["tests/test_hub.py"], base="base", full_tests=True, test_reason="repair pending")
    assert full.tier == "full"


@pytest.mark.parametrize("apparatus", ["tests/conftest.py", "tests/test_floor.py", "server.py"])
def test_a_changed_test_module_never_narrows_a_full_escalation(repo, apparatus):
    write(repo, "tests/test_hub.py")
    write(repo, "tests/test_by_from.py", "from tests.test_hub import helper\n")
    result = selection.select_from_paths(repo, ["tests/test_hub.py", apparatus], base="base")
    assert result.tier == "full" and len(result.files) == result.available


@pytest.mark.parametrize("path", ["server.py", "ouroboros/__init__.py", "supervisor/update_merge.py"])
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
    ([], "yes", "consumer"), ([], False, None),
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


@pytest.mark.parametrize("action", ["rename", "remove"])
def test_a_removed_or_renamed_test_module_selects_the_importers_it_would_break(repo, action):
    def git(*args):
        return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True).stdout.decode().strip()
    write(repo, "tests/test_hub.py")
    write(repo, "tests/test_by_from.py", "from tests.test_hub import helper\n")
    git("init", "-q")
    git("config", "user.email", "selection@example.invalid")
    git("config", "user.name", "Selection tests")
    git("add", ".")
    git("commit", "-qm", "base")
    if action == "rename":
        git("mv", "tests/test_hub.py", "tests/test_hub_renamed.py")
    else:
        git("rm", "-q", "tests/test_hub.py")
    result = selection.select_tests(repo)
    assert "tests/test_hub.py" in result.changed
    assert "tests/test_by_from.py" in result.files  # its import of the old name no longer resolves
    assert "tests/test_hub.py" not in result.files  # a removed module has nothing to run
    assert set(result.files) == {"tests/test_floor.py", "tests/test_by_from.py", *(
        ["tests/test_hub_renamed.py"] if action == "rename" else [])}


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


def test_a_change_an_inventory_reads_as_data_runs_its_pin_and_stays_selected():
    """An inventory reads its sources as data, so changing them selects its pin through no import: a book
    chapter (#3187), a new folder named like a data-layout entry (#3235) and a control's text (#3207) each
    left the target red while the selection of its merge never named the pin. The pin runs for every change
    now, and these changes still select nothing else and never escalate to full."""
    root = Path(__file__).resolve().parents[1]
    for change in ("docs/architecture/01-high-level-architecture.md", "docs/llm-rating/data/latest.json",
                   "web/modules/onboarding_wizard.js"):
        result = selection.select_from_paths(root, [change], base="base")
        assert "tests/test_generated_inventories.py" in result.files, change
        assert result.tier == "selected" and set(result.files) == set(selection.FLOOR_TESTS), change


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


def test_extra_tests_the_preflight_cannot_run_are_skipped_with_a_note_not_a_refusal(repo):
    """06.11: the schema names what the tests preflight runs; another entry is disclosed, never a refused commit."""
    skipped = []
    kept = selection.validate_widening(repo, ["tests/test_direct.py", "web/tests/chat.test.js", "tests/test_missing.py"],
                                       reason="consumers", skipped=skipped)
    assert kept == ("tests/test_direct.py",) and skipped == ["web/tests/chat.test.js", "tests/test_missing.py"]
    with pytest.raises(ValueError):  # a caller without ``skipped`` refuses, as before
        selection.validate_widening(repo, ["web/tests/chat.test.js"], reason="consumers")

