import json
import os
from pathlib import Path
import re
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest

from scripts.browser_ci_scope import documentation_only
from tests import browser_lane

REPO = Path(__file__).resolve().parents[1]
WORKFLOWS = REPO / ".github/workflows"


@pytest.mark.parametrize("paths,expected", [
    (["README.md", "docs/development/14-build-and-ci.md"], True),
    (["docs/new.rst"], True),
    ([".github/PULL_REQUEST_TEMPLATE.md"], True),
    (["docs/install/index.html"], False),
    (["site/src/index.md"], False),
    (["assets/logo.md"], False),
    (["README.md", "server.py"], False),
    (["requirements.txt"], False),
    (["Makefile"], False),
    (["Ouroboros.spec"], False),
    (["prompts/SYSTEM.md"], False),
    (["skills/new/SKILL.md"], False),
    (["tests/test_new.py"], False),
    ([".github/workflows/ci.yml"], False),
    (["unknown/new-code.ext"], False),
    ([], False),
])
def test_only_complete_documentation_changes_skip_browser(paths, expected):
    assert documentation_only(paths) is expected


def _workflow(name):
    import yaml

    loaded = yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))
    return loaded, loaded.get("on", loaded.get(True))


def test_one_browser_lane_serves_pull_requests_manual_tags_and_ouroboros_pushes():
    shared, triggers = _workflow("ui-browser.yml")
    assert list(triggers) == ["workflow_call"], "the shared lane runs only when called"
    assert "secrets." not in str(shared["jobs"])
    job = shared["jobs"]["ui-shard"]
    full = next(step for step in job["steps"] if "--require-ui-browser" in step.get("run", ""))
    assert "pytest tests/ -m ui_browser" in full["run"]
    assert "safe_test.py" in full["run"]
    push, push_triggers = _workflow("ui-browser-push.yml")
    for producer in (job, shared["jobs"]["ui-manifest"], push["jobs"]["ui-diagnostic"]):
        setup = next(step for step in producer["steps"]  # Exact candidate, never an editable install.
                     if step.get("uses") == "./.github/actions/setup-python-env")
        assert setup["with"]["install-project"] == "false"

    ci, ci_triggers = _workflow("ci.yml")
    caller = ci["jobs"]["ui-smoke"]
    assert caller["uses"] == "./.github/workflows/ui-browser.yml"
    assert "github.event_name == 'pull_request'" in caller["if"]
    assert "github.event_name == 'workflow_dispatch'" in caller["if"]
    assert "startsWith(github.ref, 'refs/tags/v')" in caller["if"]
    # A push reaches the lane through its own workflow, never through ci.yml's
    # push trigger.
    assert "push" not in caller["if"] and "schedule" not in caller["if"]
    assert "schedule" not in ci_triggers  # owner, 2026-10-05: no nightly run nobody reads

    assert list(push_triggers) == ["push", "workflow_dispatch"]
    assert push_triggers["workflow_dispatch"]["inputs"]["diagnostic"]["options"] == [
        "full", "viewport", "inflight"]
    assert push_triggers["push"] == {"branches": ["ouroboros"]}
    assert push["jobs"]["ui-smoke"]["uses"] == caller["uses"]
    # Both callers hand the lane nothing: it has no input to select a part of it with.
    assert "with" not in caller and "with" not in push["jobs"]["ui-smoke"]
    assert triggers["workflow_call"] is None


def test_a_later_docs_only_push_cannot_supersede_an_untested_code_push():
    """Push A changes code, push B (A..B) only docs. B's lane skips browsers for B's own
    range, so A's lane must finish: a concurrency group would cancel it (in progress) or
    replace it (pending), leaving B green over code no browser ever ran."""
    push, _ = _workflow("ui-browser-push.yml")
    groups = [push.get("concurrency")] + [job.get("concurrency") for job in push["jobs"].values()]
    assert groups == [None] * len(groups), groups
    shared, _ = _workflow("ui-browser.yml")
    assert all(job.get("concurrency") is None for job in [shared, *shared["jobs"].values()])
    code_push, docs_push = ["server.py", "README.md"], ["README.md"]
    assert documentation_only(docs_push) and not documentation_only(code_push)


def _item(name, marked=True):
    return SimpleNamespace(nodeid=name,
                           get_closest_marker=lambda marker: marked and marker == "ui_browser")


class _Config(SimpleNamespace):
    required, shard = True, None

    def getoption(self, name):
        return self.shard if name == "--ui-browser-shard" else self.required

    def getini(self, name):
        return {"python_files": ["test_*.py"], "norecursedirs": [".*", "venv"]}[name]


def _config(**kwargs):
    plugins = {}
    manager = SimpleNamespace(
        get_plugin=plugins.get,
        register=lambda plugin, name: plugins.__setitem__(name, plugin),
    )
    kwargs.setdefault("option", SimpleNamespace(collectonly=False))
    deselected = kwargs.setdefault("deselected", [])
    kwargs.setdefault("hook", SimpleNamespace(pytest_deselected=lambda items: deselected.extend(items)))
    # No tests/ beneath this root: only the explicitly reported modules exist.
    kwargs.setdefault("rootpath", Path(__file__).parent / "no-such-lane-root")
    return _Config(args=["tests/"], pluginmanager=manager, **kwargs)


def _guard(config, items):
    hook = browser_lane.pytest_collection_modifyitems(config, items)
    next(hook)
    return hook


@pytest.mark.parametrize("narrowing", [
    {"ignore": ["tests/test_skill_publish_browser.py"]}, {"ignore_glob": ["*publish*"]},
    {"lf": True}, {"deselect": ["tests/test_x.py::one"]}, {"keyword": "publish"},
    {"override_ini": ["python_files=test_ui_*.py"]}, {"override_ini": ["norecursedirs=tests"]},
])
def test_collection_guard_refuses_controls_that_narrow_collection(narrowing):
    config = _config(option=SimpleNamespace(collectonly=False, **narrowing))
    browser_lane.pytest_configure(config)
    # The dropped modules never reach `items`, so the marker comparison alone agrees.
    hook = _guard(config, [_item("first")])
    with pytest.raises(pytest.UsageError, match="UI_BROWSER_INCOMPLETE: collection narrowed"):
        next(hook)


def test_collection_guard_accounts_for_every_module_on_disk(tmp_path):
    for name in ("tests/test_a.py", "tests/nested/test_b.py", "tests/helper.py",
                 "tests/.cache/test_hidden.py"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text("", encoding="utf-8")
    config = _config(rootpath=tmp_path)
    browser_lane.pytest_configure(config)
    modules = config.pluginmanager.get_plugin("ui_browser_collected_modules")
    for nodeid in ("tests", "tests/test_a.py"):
        modules.pytest_collectreport(SimpleNamespace(nodeid=nodeid, skipped=False))
    with pytest.raises(pytest.UsageError, match=r"never collected \(1\): tests/nested/test_b\.py"):
        next(_guard(config, [_item("tests/test_a.py::one")]))
    modules.pytest_collectreport(SimpleNamespace(nodeid="tests/nested/test_b.py", skipped=False))
    with pytest.raises(StopIteration):
        next(_guard(config, [_item("tests/test_a.py::one")]))

    skipped = _config(rootpath=tmp_path)
    browser_lane.pytest_configure(skipped)
    modules = skipped.pluginmanager.get_plugin("ui_browser_collected_modules")
    modules.pytest_collectreport(SimpleNamespace(nodeid="tests/test_a.py", skipped=False))
    modules.pytest_collectreport(SimpleNamespace(
        nodeid="tests/nested/test_b.py", skipped=True,
        longrepr=("tests/nested/test_b.py", 1, "Skipped: could not import 'optional'")))
    with pytest.raises(pytest.UsageError, match=r"skipped tests/nested/test_b\.py: .*optional"):
        next(_guard(skipped, [_item("tests/test_a.py::one")]))


def test_windows_settings_launcher_skip_is_registered_for_its_exact_case_only():
    case = ("tests/test_ui_candidate_server.py::"
            "test_real_server_uses_disposable_candidate_and_keeps_identity_through_restart")
    reason = "Skipped: POSIX test launcher; production browser behavior is shared"
    assert browser_lane.permitted_skip(case + "[settings]", reason)
    assert not browser_lane.permitted_skip(case + "[direct]", reason), "direct still runs on Windows"
    assert not browser_lane.permitted_skip(
        "tests/test_ui_candidate_server.py::test_concurrent_servers_keep_distinct_roots_and_owner_sentinels",
        reason)
    assert not browser_lane.permitted_skip(case + "[settings]", "Skipped: no browser executable")


def test_native_qt_opt_in_skip_precedes_optional_desktop_imports(tmp_path, monkeypatch):
    import sys

    from tests import test_widget_stream_download_ui as widget

    monkeypatch.delenv("PYWEBVIEW_GUI", raising=False)
    # The browser lane installs no desktop extra: both imports are unavailable there.
    monkeypatch.setitem(sys.modules, "webview", None)
    monkeypatch.setitem(sys.modules, "qtpy", None)
    with pytest.raises(pytest.skip.Exception) as skipped:
        widget.test_native_widget_exports(None, tmp_path, monkeypatch)
    assert browser_lane.permitted_skip(
        "tests/test_widget_stream_download_ui.py::test_native_widget_exports", str(skipped.value))


@pytest.mark.parametrize("partial", [False, True])
def test_collection_guard_accepts_full_lane_and_refuses_deselection(partial):
    config = _config()
    browser_lane.pytest_configure(config)
    items = [_item("first"), _item("second")]
    hook = browser_lane.pytest_collection_modifyitems(config, items)
    next(hook)
    if partial:
        items.pop()
        with pytest.raises(pytest.UsageError, match="UI_BROWSER_INCOMPLETE"):
            next(hook)
    else:
        with pytest.raises(StopIteration):
            next(hook)
        assert config._required_ui_nodes == {"first", "second"}
        assert browser_lane._reconciliation(config).required == {"first", "second"}
        assert config._ui_browser_lane == {"full": ["first", "second"]}, "an unsharded lane records no shard"
        assert config.deselected == []


@pytest.mark.parametrize("index,slice_", [(1, ["a", "d"]), (2, ["b", "e"]), (3, ["c"])])
def test_shard_keeps_every_nth_node_of_the_sorted_lane_in_collection_order(index, slice_):
    config = _config(shard=f"{index}/3")
    browser_lane.pytest_configure(config)
    items = [_item(name) for name in "dacbe"]  # Collection order is not the sorted order.
    with pytest.raises(StopIteration):
        next(_guard(config, items))
    assert [item.nodeid for item in items] == [name for name in "dacbe" if name in slice_]
    assert sorted(item.nodeid for item in config.deselected) == sorted(set("abcde") - set(slice_))
    assert config._ui_browser_lane == {"full": list("abcde"), "shard": [index, 3], "assigned": slice_}
    # Reconciliation covers this shard's slice only: another shard's node is never "not executed" here.
    assert config._required_ui_nodes == set(slice_)
    assert browser_lane._reconciliation(config).required == set(slice_)


@pytest.mark.parametrize("option", [{"keyword": "publish"}, {"deselect": ["tests/test_x.py::one"]}, {}])
def test_shard_option_never_relaxes_the_full_lane_guard(option):
    config = _config(shard="1/2", option=SimpleNamespace(collectonly=False, **option))
    browser_lane.pytest_configure(config)
    items = [_item("first"), _item("second"), _item("third")]
    hook = _guard(config, items)
    if not option:
        items.pop()  # Deselected by something other than the shard itself.
    with pytest.raises(pytest.UsageError, match="UI_BROWSER_INCOMPLETE"):
        next(hook)
    assert not hasattr(config, "_ui_browser_lane"), "a refused lane leaves no lane facts to export"


@pytest.mark.parametrize("value,required,accepted", [
    ("1/1", True, True), ("3/3", True, True), ("2/4", True, True),
    ("2/4", False, False),  # A shard without the full-lane guard proves nothing.
    ("0/3", True, False), ("4/3", True, False), ("1/0", True, False), ("-1/3", True, False),
    ("1", True, False), ("1/3/5", True, False), ("a/b", True, False), ("1 /3", True, False),
    ("", True, False),
])
def test_shard_option_is_refused_when_malformed_or_without_the_lane_guard(value, required, accepted):
    config = _config(shard=value, required=required)
    if accepted:
        browser_lane.pytest_configure(config)
        assert browser_lane._shard(config) == tuple(int(part) for part in value.split("/"))
    else:
        with pytest.raises(pytest.UsageError, match="UI_BROWSER_SHARD"):
            browser_lane.pytest_configure(config)


def test_shard_beyond_the_lane_is_refused_instead_of_running_nothing():
    config = _config(shard="3/3")
    browser_lane.pytest_configure(config)
    with pytest.raises(pytest.UsageError, match=r"UI_BROWSER_INCOMPLETE: shard 3/3 of 2 nodes is empty"):
        next(_guard(config, [_item("first"), _item("second")]))


def test_collection_guard_refuses_an_empty_lane():
    hook = browser_lane.pytest_collection_modifyitems(_config(), [])
    next(hook)
    with pytest.raises(pytest.UsageError, match="UI_BROWSER_INCOMPLETE"):
        next(hook)


def test_browser_skip_cannot_become_a_successful_required_run():
    item = SimpleNamespace(config=_config(), nodeid="tests/test_x.py::missing_engine")
    report = SimpleNamespace(skipped=True, outcome="skipped",
                             longrepr=("tests/test_x.py", 10, "Skipped: no browser executable"))
    hook = browser_lane.pytest_runtest_makereport(item, None)
    next(hook)
    with pytest.raises(StopIteration):
        hook.send(SimpleNamespace(get_result=lambda: report))
    assert report.outcome == "failed"
    assert "UI_BROWSER_SKIPPED" in report.longrepr


def test_registered_platform_skip_survives_with_its_recorded_reason():
    nodeid = "tests/test_settings_restart_browser.py::test_pending_survives_reconnect"
    item = SimpleNamespace(config=_config(), nodeid=nodeid)
    report = SimpleNamespace(
        skipped=True, outcome="skipped", when="setup", nodeid=nodeid,
        longrepr=("tests/test_settings_restart_browser.py", 17,
                  "Skipped: POSIX test launcher; production browser behavior is shared"))
    hook = browser_lane.pytest_runtest_makereport(item, None)
    next(hook)
    with pytest.raises(StopIteration):
        hook.send(SimpleNamespace(get_result=lambda: report))
    assert report.outcome == "skipped"
    assert "POSIX test launcher" in report.ui_browser_platform_skip

    reconciliation = browser_lane.LaneReconciliation({nodeid})
    reconciliation.pytest_runtest_logreport(report)
    assert reconciliation.missing == [], "a permitted skip is still an executed node"
    assert reconciliation.platform_skips[nodeid].endswith("shared")


def test_collected_nodes_that_never_ran_fail_the_lane():
    reconciliation = browser_lane.LaneReconciliation({"a::one", "b::two"})
    reconciliation.pytest_runtest_logreport(
        SimpleNamespace(when="setup", outcome="passed", nodeid="a::one", skipped=False))
    assert reconciliation.missing == ["a::one", "b::two"], "setup alone is not a run"
    reconciliation.pytest_runtest_logreport(
        SimpleNamespace(when="call", outcome="passed", nodeid="a::one", skipped=False))
    assert reconciliation.missing == ["b::two"]

    session = SimpleNamespace(config=_config(), exitstatus=0)
    session.config.pluginmanager.register(reconciliation, "ui_browser_lane_reconciliation")
    browser_lane.pytest_sessionfinish(session, 0)
    assert session.exitstatus == 1, "an unexecuted collected node must not exit green"


def test_xdist_controller_receives_collection_without_collecting_itself():
    config = _config()
    browser_lane.pytest_configure(config)
    tracker = browser_lane._reconciliation(config)
    tracker.pytest_xdist_node_collection_finished(None, ["tests/a.py::one"])
    session = SimpleNamespace(config=config, exitstatus=0)
    browser_lane.pytest_sessionfinish(session, 0)
    assert session.exitstatus == 1
    tracker.pytest_runtest_logreport(SimpleNamespace(
        when="call", outcome="passed", nodeid="tests/a.py::one"))
    session.exitstatus = 0
    browser_lane.pytest_sessionfinish(session, 0)
    assert session.exitstatus == 0


def test_required_engine_is_launched_and_missing_engine_fails(monkeypatch):
    from contextlib import nullcontext
    import playwright.sync_api

    monkeypatch.setenv("OUROBOROS_RUN_UI_SMOKE", "1")
    monkeypatch.setenv("OUROBOROS_EXPECT_BROWSER_ENGINES", "chromium,webkit")
    launches = []

    def launch(**kwargs):
        launches.append(kwargs)
        return SimpleNamespace(close=lambda: None)

    engine = SimpleNamespace(launch=launch)
    monkeypatch.setattr(playwright.sync_api, "sync_playwright", lambda: nullcontext(
        SimpleNamespace(chromium=engine, webkit=engine)))
    session = SimpleNamespace(config=_config())
    browser_lane.pytest_collection_finish(session)
    assert launches == [{"headless": True}] * 2
    engine.launch = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("missing executable"))
    with pytest.raises(pytest.UsageError, match="UI_BROWSER_UNAVAILABLE"):
        browser_lane.pytest_collection_finish(session)


RUN_FACTS = {"GITHUB_SHA": "5" * 40, "GITHUB_RUN_ID": "7007", "GITHUB_RUN_ATTEMPT": "1"}
LANE = {
    "test_lane_alpha.py": """
        import pytest
        @pytest.mark.ui_browser
        @pytest.mark.parametrize("engine", ["chromium", "webkit"])
        def test_page(engine):
            pass
        @pytest.mark.ui_browser
        def test_plain():
            pass
        def test_outside_the_lane():
            raise AssertionError("-m ui_browser deselects this")
    """,
    "test_lane_beta.py": """
        import pytest
        pytestmark = pytest.mark.ui_browser
        @pytest.mark.parametrize("case", range(4))
        def test_case(case):
            pass
    """,
}


def _lane_process(tmp_path, name, *options, guarded=True):
    """One real pytest process over a synthetic `tests/` lane.

    The real plugins collect, guard, slice, reconcile and export; only the engine
    launch is replaced, so the proof needs no installed browser.
    """
    project = tmp_path / "project"
    if not project.exists():
        (project / "tests").mkdir(parents=True)
        (project / "pytest.ini").write_text("[pytest]\nmarkers = ui_browser: synthetic lane\n",
                                            encoding="utf-8")
        for module, source in LANE.items():
            (project / "tests" / module).write_text(textwrap.dedent(source), encoding="utf-8")
    work = tmp_path / name
    work.mkdir(parents=True)
    argv = ["-p", "tests.conftest", "-o", "addopts=", "--rootdir", str(project),
            "--confcutdir", str(project), "-m", "ui_browser", "--ci-evidence-dir", str(work / "host"),
            *(["--require-ui-browser"] if guarded else []), *options, "tests/"]
    entry = work / "run_lane.py"
    entry.write_text(
        "import contextlib, os, sys, types\n"
        f"sys.path.insert(0, {str(REPO)!r})\n"
        f"os.chdir({str(project)!r})\n"
        f"os.environ.update({RUN_FACTS!r})\n"
        "import playwright.sync_api\n"
        "engine = types.SimpleNamespace(launch=lambda **_: types.SimpleNamespace(close=lambda: None))\n"
        "playwright.sync_api.sync_playwright = lambda: contextlib.nullcontext(\n"
        "    types.SimpleNamespace(chromium=engine, webkit=engine))\n"
        "import pytest\n"
        f"raise SystemExit(pytest.main({argv!r}))\n", encoding="utf-8")
    env = {**os.environ, "OUROBOROS_RUN_UI_SMOKE": "1",
           "OUROBOROS_EXPECT_BROWSER_ENGINES": "chromium,webkit"}
    result = subprocess.run(
        [sys.executable, "-I", "-S", str(REPO / "scripts/safe_test.py"), "--temp-parent", str(tmp_path),
         "--", sys.executable, str(entry)],
        cwd=REPO, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
    return result, work  # A console code page may not be UTF-8: assertions read ASCII only.


def _lane_results(work):
    return json.loads((work / "host" / "results.json").read_text(encoding="utf-8"))


def _called(results):
    return [row["nodeid"] for row in results["reports"] if row["phase"] == "call"]


def _reconcile(tmp_path, manifest, shards, count):
    summary = tmp_path / "reconcile.md"
    summary.unlink(missing_ok=True)
    result = subprocess.run(
        [sys.executable, "-I", "-S", str(REPO / "tests/ci_evidence.py"), "reconcile-shards",
         "--manifest", str(manifest), "--shards", str(shards), "--count", str(count),
         "--sha", RUN_FACTS["GITHUB_SHA"], "--run-id", RUN_FACTS["GITHUB_RUN_ID"],
         "--summary", str(summary)],
        cwd=tmp_path, capture_output=True, text=True, encoding="utf-8", timeout=60)
    return result, summary.read_text(encoding="utf-8")


@pytest.mark.serial
def test_real_shards_partition_the_lane_and_reconcile_against_the_unsharded_manifest(tmp_path):
    collected, manifest = _lane_process(tmp_path, "manifest", "--collect-only", "-q")
    assert collected.returncode == 0, collected.stdout + collected.stderr
    lane = _lane_results(manifest)["ui_browser"]
    full = lane["full"]
    assert lane == {"full": full} and len(full) == 7 and full == sorted(set(full))
    assert not any("test_outside_the_lane" in node for node in full)
    assert _lane_results(manifest)["reports"] == [] and "UI_BROWSER_SHARD" not in collected.stdout

    slices = []
    for index in (1, 2, 3):
        run, work = _lane_process(tmp_path, f"shards/ui-shard-{index}", f"--ui-browser-shard={index}/3")
        assert run.returncode == 0, run.stdout + run.stderr
        results = _lane_results(work)
        assigned = results["ui_browser"]["assigned"]
        assert results["ui_browser"] == {"full": full, "shard": [index, 3], "assigned": assigned}
        assert f"UI_BROWSER_SHARD {index}/3: assigned {len(assigned)} of 7" in run.stdout
        assert sorted(_called(results)) == assigned and results["tests_collected"] == len(assigned)
        assert results["session_exit_code"] == 0 and "UI_BROWSER_NOT_EXECUTED" not in run.stdout
        slices.append(assigned)
    # Pairwise disjoint and complete at once: the sorted concatenation is the lane itself.
    assert sorted(node for assigned in slices for node in assigned) == full
    assert [len(assigned) for assigned in slices] == [3, 2, 2]

    proven, summary = _reconcile(tmp_path, manifest, tmp_path / "shards", 3)
    assert proven.returncode == 0, proven.stdout + proven.stderr
    assert "UI_BROWSER_RECONCILE complete" in proven.stdout and "GAP" not in summary

    (tmp_path / "shards" / "ui-shard-2" / "host" / "results.json").unlink()
    silent, summary = _reconcile(tmp_path, manifest, tmp_path / "shards", 3)
    assert silent.returncode == 1 and "shard 2/3: no proof" in silent.stdout
    for node in slices[1]:  # The nodes no shard proved are named, the proven ones are not.
        assert node in silent.stdout and node in summary
    assert not any(node in silent.stdout for node in slices[0] + slices[2])


@pytest.mark.serial
def test_real_shard_still_refuses_a_narrowed_lane_and_an_unguarded_shard(tmp_path):
    narrowed, work = _lane_process(tmp_path, "narrowed", "--ui-browser-shard=1/3", "-k", "test_case")
    assert narrowed.returncode == 4, narrowed.stdout + narrowed.stderr
    assert re.search(r"ERROR: UI_BROWSER_INCOMPLETE: collection narrowed \S+ -k$", narrowed.stderr, re.M)
    refused = _lane_results(work)
    assert refused["reports"] == [] and "ui_browser" not in refused, "a refused lane exports no lane facts"

    unguarded, work = _lane_process(tmp_path, "unguarded", "--ui-browser-shard=1/3", guarded=False)
    assert unguarded.returncode == 4, unguarded.stdout + unguarded.stderr
    assert "UI_BROWSER_SHARD" in unguarded.stderr and not (work / "host" / "results.json").exists()


@pytest.mark.serial
def test_real_shard_out_of_session_budget_names_only_its_own_unexecuted_nodes(tmp_path):
    # pytest-timeout's cooperative session budget: checked after each test, it ends the
    # session through its normal finish. An already spent budget runs exactly one test.
    run, work = _lane_process(tmp_path, "shards/ui-shard-1", "--ui-browser-shard=1/2",
                              "--session-timeout=0.001")
    assert run.returncode == 1, run.stdout + run.stderr
    results = _lane_results(work)
    lane = results["ui_browser"]
    assert lane["shard"] == [1, 2] and len(lane["assigned"]) == 4 and len(lane["full"]) == 7
    executed = _called(results)
    assert len(executed) == 1 and executed[0] in lane["assigned"]
    unexecuted = sorted(set(lane["assigned"]) - set(executed))
    assert f"UI_BROWSER_NOT_EXECUTED (3): {', '.join(unexecuted)}" in run.stdout
    assert "session-timeout" in run.stdout and results["session_exit_code"] == 1
    other_shard = set(lane["full"]) - set(lane["assigned"])
    named = run.stdout.split("UI_BROWSER_NOT_EXECUTED", 1)[1].splitlines()[0]
    assert not any(node in named for node in other_shard)

    _, manifest = _lane_process(tmp_path, "manifest", "--collect-only", "-q")
    red, summary = _reconcile(tmp_path, manifest, tmp_path / "shards", 2)
    assert red.returncode == 1
    assert f"shard 1/2: assigned but not executed (3): {', '.join(unexecuted)}" in red.stdout
    assert "shard 2/2: no proof" in red.stdout and "INCOMPLETE" in summary
