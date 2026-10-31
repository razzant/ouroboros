"""Opt-in proof that the complete UI marker lane and its engines actually ran.

`--require-ui-browser` turns a green exit into evidence. The whole `ui_browser`
marker lane must be collected, the declared engines must launch BEFORE the first
test, every collected node must reach a terminal outcome, and the only tolerated
skips are the reviewed platform gates registered below — each one recorded by
node id and reason in the run summary, so tolerated never means invisible.

Completeness is judged against the test modules on disk, not only against what
was collected: `--ignore`, `--ignore-glob`, `--last-failed` file skipping or an
overridden collection setting drop modules BEFORE any marker comparison could
see them, so those controls are refused and every module must be collected.

`--ui-browser-shard=K/N` splits the run, never the proof of the lane: each shard
collects and checks the complete lane exactly as above, then keeps its slice of the
sorted node ids and reconciles that slice. When the run's weight table records UI
nodes (`OUROBOROS_CI_DURATIONS`, `ui_nodes`), the slices pack the sorted lane by
recorded time (`node_weights`; a node the table lacks weighs the table's median);
otherwise shard K keeps every N-th node starting at the K-th. The lane facts
(complete sorted lane, its weights, shard, assigned slice) are left on the config
for the evidence export; `tests/ci_evidence.py reconcile-shards` proves the slices
of one run add up to the lane an unsharded collection sees. Shards are separate processes
that agree on node IDS: a parametrization must give each case a stable id and order
(no iteration over a set, no unsorted glob), or one id names different cases per shard.
"""
import fnmatch
import os
from pathlib import Path
import re
import statistics

import pytest

from tests.ci_verdicts import durations_table


# Reviewed platform gates: conditions a runner legitimately lacks. Registered as
# (node-id prefix, reason pattern) so a NEW skip cannot arrive unnoticed; adding
# an entry is a diff-time decision, like a reference-book byte budget. Every
# match is reported as UI_BROWSER_PLATFORM_SKIP with the reason the test gave.
PLATFORM_SKIPS = (
    ("tests/test_settings_restart_browser.py", r"POSIX test launcher"),
    # The same POSIX launcher, for exactly this case: its `[direct]` sibling runs everywhere.
    ("tests/test_ui_candidate_server.py::"
     "test_real_server_uses_disposable_candidate_and_keeps_identity_through_restart[settings]",
     r"POSIX test launcher"),
    ("tests/test_chat_history_recovery_browser.py",
     r"POSIX archive permissions|identity bypasses unreadable-file permissions"),
    ("tests/test_widget_stream_download_ui.py", r"native Qt probe is explicitly selected"),
    # Headless Chromium on Linux applies no synthetic touch pinch even to a plain page (CI 40682936207); macOS does.
    ("tests/test_shell_drawer_browser.py::test_a_pinch_magnifies_the_room", r"applies no synthetic touch pinch"),
)

# Controls that narrow collection itself. Post-collection deselection is caught by
# the marker comparison too; these are refused by name so the error says why.
_NARROWING_OPTIONS = {"ignore": "--ignore", "ignore_glob": "--ignore-glob",
                      "deselect": "--deselect", "keyword": "-k", "lf": "--last-failed",
                      "stepwise": "--stepwise"}
_COLLECTION_INI = {"python_files", "python_classes", "python_functions", "testpaths",
                   "norecursedirs"}


def pytest_addoption(parser):
    parser.addoption("--require-ui-browser", action="store_true",
                     help="Require the full ui_browser lane, installed engines, and no silent skips")
    parser.addoption("--ui-browser-shard", default=None, metavar="K/N",
                     help="With --require-ui-browser: run slice K of N of the complete sorted lane")


def _required(config) -> bool:
    return bool(config.getoption("--require-ui-browser"))


def _shard(config):
    """(K, N) of `--ui-browser-shard=K/N`, 1-based, or None for the whole lane."""
    value = config.getoption("--ui-browser-shard")
    if value is None:
        return None
    match = re.fullmatch(r"([1-9][0-9]*)/([1-9][0-9]*)", str(value))
    if match is None or int(match[1]) > int(match[2]) or not _required(config):
        raise pytest.UsageError("UI_BROWSER_SHARD: --ui-browser-shard takes K/N with 1 <= K <= N "
                                f"and needs --require-ui-browser, got {value!r}")
    return int(match[1]), int(match[2])


def _skip_reason(report) -> str:
    """The recorded reason, whatever shape this report's longrepr carries."""
    longrepr = getattr(report, "longrepr", "")
    if isinstance(longrepr, tuple) and len(longrepr) == 3:
        longrepr = longrepr[2]
    return str(longrepr)


def permitted_skip(nodeid: str, reason: str) -> bool:
    return any(nodeid.startswith(prefix) and re.search(pattern, reason)
               for prefix, pattern in PLATFORM_SKIPS)


class LaneReconciliation:
    """Terminal outcomes as seen by the process that receives every report.

    Registered only for a required run. Under xdist the controller receives the
    workers' reports, so the executed set is complete there and each worker's own
    partial view never decides the verdict.
    """

    def __init__(self, required):
        self.required = set(required)
        self.executed = set()
        self.platform_skips = {}

    @pytest.hookimpl(optionalhook=True)
    def pytest_xdist_node_collection_finished(self, node, ids):
        # Workers enforce the complete-marker contract; the controller does not
        # collect and learns that same set from xdist's collection receipt.
        self.required.update(ids)

    def pytest_runtest_logreport(self, report):
        # A node counts as executed once it produced a call phase or failed/skipped
        # in setup; a setup that merely passed is not yet evidence of a run.
        if report.when == "call" or report.outcome != "passed":
            self.executed.add(report.nodeid)
        reason = getattr(report, "ui_browser_platform_skip", "")
        if reason:
            self.platform_skips[report.nodeid] = reason

    @property
    def missing(self):
        return sorted(self.required - self.executed)


class CollectedModules:
    """The test modules this process actually imported, and those skipped whole."""

    def __init__(self):
        self.collected = set()
        self.skipped = {}

    def pytest_collectreport(self, report):
        if report.nodeid.endswith(".py"):
            if report.skipped:
                self.skipped[report.nodeid] = _skip_reason(report)
            else:
                self.collected.add(report.nodeid)


def _reconciliation(config):
    return config.pluginmanager.get_plugin("ui_browser_lane_reconciliation")


def node_weights(full: list[str]) -> tuple[list | None, str]:
    """([[node, positive weight]] in lane order, weights_source) from the run's weight table.

    (None, "") without a table: shards keep every N-th node. A table that records
    no UI node, or cannot be read, keeps those positions too and says why.
    """
    table, why = durations_table()
    if table is None:
        return None, "positions: " + why if why else ""
    recorded, run_id = table.get("ui_nodes") or {}, table["source"]["run_id"]
    if not recorded:
        return None, f"positions: the durations of run {run_id} record no UI node"
    median = statistics.median_low(recorded.values())
    return [[node, max(1, recorded.get(node, median))] for node in full], f"durations:{run_id}"


def _module_files(config) -> set:
    """What the committed collection settings select beneath tests/ on disk."""
    root = Path(config.rootpath)
    patterns, skipped_dirs = config.getini("python_files"), config.getini("norecursedirs")
    found = set()
    for directory, dirs, files in os.walk(root / "tests"):
        dirs[:] = [name for name in dirs if name != "__pycache__"
                   and not any(fnmatch.fnmatch(name, pattern) for pattern in skipped_dirs)]
        found.update((Path(directory) / name).relative_to(root).as_posix() for name in files
                     if any(fnmatch.fnmatch(name, pattern) for pattern in patterns))
    return found


def _collection_gaps(config) -> list:
    """Why the collected modules are not the whole lane; empty when they are."""
    option = config.option
    gaps = [flag for name, flag in _NARROWING_OPTIONS.items() if getattr(option, name, None)]
    gaps += ["-o " + entry.split("=", 1)[0].strip()
             for entry in (getattr(option, "override_ini", None) or [])
             if entry.split("=", 1)[0].strip() in _COLLECTION_INI]
    modules = config.pluginmanager.get_plugin("ui_browser_collected_modules")
    if modules is None:
        return gaps + ["module collection was not observed"]
    gaps += [f"skipped {nodeid}: {reason}" for nodeid, reason in sorted(modules.skipped.items())]
    missing = sorted(_module_files(config) - modules.collected - set(modules.skipped))
    if missing:
        gaps.append(f"never collected ({len(missing)}): " + ", ".join(missing[:10]))
    return gaps


def pytest_configure(config):
    _shard(config)  # A malformed or unguarded shard is refused before collection starts.
    if not _required(config):
        return
    # Every collecting process tracks its modules; under xdist that is each worker.
    config.pluginmanager.register(CollectedModules(), "ui_browser_collected_modules")
    if not hasattr(config, "workerinput"):
        config.pluginmanager.register(LaneReconciliation(()), "ui_browser_lane_reconciliation")


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_collection_modifyitems(config, items):
    expected = {item.nodeid for item in items if item.get_closest_marker("ui_browser")}
    yield
    if not _required(config):
        return
    gaps = _collection_gaps(config)
    if gaps:
        raise pytest.UsageError("UI_BROWSER_INCOMPLETE: collection narrowed — " + "; ".join(gaps))
    actual = {item.nodeid for item in items}
    if not expected or actual != expected or config.args != ["tests/"]:
        raise pytest.UsageError("UI_BROWSER_INCOMPLETE: require tests/ and the complete -m ui_browser lane")
    # The complete lane is recorded BEFORE slicing: every shard and the unsharded
    # manifest collection must report the same list for their slices to add up.
    lane = {"full": sorted(expected)}
    weights, source = node_weights(lane["full"])
    if source:
        lane["weights_source"] = source
    if weights is not None:
        lane["node_weights"] = weights
    assigned, shard = lane["full"], _shard(config)
    if shard is not None:
        index, count = shard
        if weights is None:
            assigned = lane["full"][index - 1::count]
        else:
            # At collection, not at import: conftest registers this plugin before tests.ci_evidence,
            # whose assertion rewriting an earlier import would forestall.
            from tests.ci_evidence import node_shards

            owners = node_shards([weight for _, weight in weights], count)
            assigned = [node for node, owner in zip(lane["full"], owners) if owner == index]
        if not assigned:
            raise pytest.UsageError(f"UI_BROWSER_INCOMPLETE: shard {index}/{count} of "
                                    f"{len(expected)} nodes is empty")
        kept = set(assigned)
        deselected = [item for item in items if item.nodeid not in kept]
        items[:] = [item for item in items if item.nodeid in kept]
        config.hook.pytest_deselected(items=deselected)
        lane.update(shard=[index, count], assigned=assigned)
    config._ui_browser_lane = lane
    # Reconciliation covers this process's slice, so "not executed" names its own nodes only.
    config._required_ui_nodes = set(assigned)
    reconciliation = _reconciliation(config)
    if reconciliation is not None:
        reconciliation.required.update(assigned)


def pytest_collection_finish(session):
    if not _required(session.config) or session.config.option.collectonly:
        return
    if os.environ.get("OUROBOROS_RUN_UI_SMOKE") != "1":
        raise pytest.UsageError("UI_BROWSER_DISABLED: set OUROBOROS_RUN_UI_SMOKE=1")
    from playwright.sync_api import sync_playwright

    engines = os.environ.get("OUROBOROS_EXPECT_BROWSER_ENGINES", "chromium,webkit").split(",")
    with sync_playwright() as playwright:
        for name in engines:
            if name not in {"chromium", "webkit", "firefox"}:
                raise pytest.UsageError(f"UI_BROWSER_ENGINE_UNKNOWN: {name!r}")
            try:
                browser = getattr(playwright, name).launch(headless=True)
                browser.close()
            except Exception as exc:
                raise pytest.UsageError(f"UI_BROWSER_UNAVAILABLE ({name}): {exc}") from exc


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if not _required(item.config) or not report.skipped or hasattr(report, "wasxfail"):
        return
    reason = _skip_reason(report)
    if permitted_skip(item.nodeid, reason):
        # A plain attribute survives xdist report serialization, so the controller
        # can print what a worker tolerated.
        report.ui_browser_platform_skip = reason
        return
    report.outcome = "failed"
    report.longrepr = f"UI_BROWSER_SKIPPED: {item.nodeid}: {reason}"


def pytest_terminal_summary(terminalreporter, exitstatus, config):  # noqa: ARG001
    lane = getattr(config, "_ui_browser_lane", {})
    if "shard" in lane:
        terminalreporter.write_line("UI_BROWSER_SHARD {}/{}: assigned {} of {}".format(
            *lane["shard"], len(lane["assigned"]), len(lane["full"])))
    reconciliation = _reconciliation(config)
    if reconciliation is None or config.option.collectonly:
        return
    for nodeid, reason in sorted(reconciliation.platform_skips.items()):
        terminalreporter.write_line(f"UI_BROWSER_PLATFORM_SKIP {nodeid}: {reason}")
    missing = reconciliation.missing
    if missing:
        terminalreporter.write_line(
            f"UI_BROWSER_NOT_EXECUTED ({len(missing)}): " + ", ".join(missing))


def pytest_sessionfinish(session, exitstatus):  # noqa: ARG001
    config = session.config
    if not _required(config) or hasattr(config, "workerinput") or config.option.collectonly:
        return
    reconciliation = _reconciliation(config)
    # No reconciliation means collection never reached the guard: an empty or
    # narrowed lane must not exit green.
    if reconciliation is None or not reconciliation.required or reconciliation.missing:
        session.exitstatus = 1
