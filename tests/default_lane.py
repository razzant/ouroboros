"""Whole-file shards of CI's default suite, preserving its two existing passes.

A file's weight is its recorded time when the run's weight table records it on
this system (`OUROBOROS_CI_DURATIONS`, written by `tests/ci_evidence.py durations`:
integer milliseconds of every report of the file in both default passes). A file
the table does not record weighs its collected default node count times the
table's median milliseconds per node on this system. Without a readable table that
records a collected file here, a file weighs its number of collected default nodes.
The lane block names the source (`weights_source`). Weights cover both passes,
before the serial split, so both passes assign a mixed file to the same runner.
Greedy largest-first packing (path and shard-index tie breaks) needs no hash seed;
every shard and the manifest of a run read the same table and agree on the
weights. Opt-in lanes never contribute weight or join the selection.

A shard may hold no file with a test of a pass (few files carry `serial`). It
deselects everything, records `assigned: []` and ends with exit status 0 instead
of pytest's 5: the empty slice of the manifest's packing is its whole proof.

An independent, unsharded --default-suite-manifest collection witnesses each
pass. tests/ci_evidence.py reconciles every shard against those manifests. Skips
retain ordinary pytest semantics; this selector neither adds nor removes them.
"""
from collections import Counter
import re
import statistics
import sys

import pytest

from ouroboros.preflight_runner import LANE_EXCLUSION_EXPR
from tests.browser_lane import _COLLECTION_INI, _NARROWING_OPTIONS
from tests.ci_evidence import file_shards
from tests.ci_verdicts import durations_table, platform_name


def pytest_addoption(parser):
    parser.addoption("--default-suite-shard", default=None, metavar="K/N",
                     help="Run whole-file shard K/N of the default parallel or serial pass")
    parser.addoption("--default-suite-manifest", action="store_true",
                     help="Export an unsharded collect-only witness of a default pass")


def _enabled(config):
    return config.getoption("--default-suite-shard") or config.getoption("--default-suite-manifest")


def _shard(config):
    value = config.getoption("--default-suite-shard")
    if value is None:
        return None
    match = re.fullmatch(r"([1-9][0-9]*)/([1-9][0-9]*)", value)
    if match is None or int(match[1]) > int(match[2]):
        raise pytest.UsageError("DEFAULT_SUITE_SHARD: require K/N with 1 <= K <= N")
    return int(match[1]), int(match[2])


def file_weights(counts: dict[str, int]) -> tuple[dict[str, int], str]:
    """(positive weight per file, weights_source) for one collection's default node counts."""
    table, why = durations_table()
    if table is None:
        return dict(counts), "nodes: " + why if why else "nodes"
    run_id, system = table["source"]["run_id"], platform_name(sys.platform)
    recorded = {path: ms for path, ms in table["platforms"].get(system, {}).items() if counts.get(path)}
    if not recorded:
        return dict(counts), f"nodes: the durations of run {run_id} record no collected file on {system}"
    per_node = statistics.median(ms / counts[path] for path, ms in recorded.items())
    return ({path: max(1, recorded[path] if path in recorded else round(count * per_node))
             for path, count in counts.items()}, f"durations:{run_id}")


def pytest_configure(config):
    if not _enabled(config):
        return
    shard = _shard(config)
    manifest = config.getoption("--default-suite-manifest")
    if manifest and (shard or not config.option.collectonly):
        raise pytest.UsageError("DEFAULT_SUITE_MANIFEST: require unsharded --collect-only")
    expressions = {f"{prefix}serial and {LANE_EXCLUSION_EXPR}": name
                   for prefix, name in (("not ", "parallel"), ("", "serial"))}
    if config.option.markexpr not in expressions:
        raise pytest.UsageError("DEFAULT_SUITE_INCOMPLETE: require the unchanged default pass expression")
    config._default_suite_pass = expressions[config.option.markexpr]
    narrowed = any(getattr(config.option, name, None) for name in _NARROWING_OPTIONS)
    overridden = any(entry.split("=", 1)[0].strip() in _COLLECTION_INI
                     for entry in (config.option.override_ini or []))
    if config.args != ["tests/"] or narrowed or overridden:
        raise pytest.UsageError("DEFAULT_SUITE_INCOMPLETE: require tests/ without collection narrowing")


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_collection_modifyitems(config, items):
    collected = list(items) if _enabled(config) else []
    yield  # conftest adds file-level serial markers; pytest applies the normal -m selection.
    if not _enabled(config):
        return
    excluded = [term.removeprefix("not ") for term in LANE_EXCLUSION_EXPR.split(" and ")]
    default = [item for item in collected
               if not any(item.get_closest_marker(marker) for marker in excluded)]
    serial = config._default_suite_pass == "serial"
    expected = sorted(item.nodeid for item in default
                      if bool(item.get_closest_marker("serial")) == serial)
    if not expected or sorted(item.nodeid for item in items) != expected:
        raise pytest.UsageError("DEFAULT_SUITE_INCOMPLETE: collected selection differs from the default pass")
    weights, source = file_weights(dict(sorted(Counter(item.nodeid.split("::", 1)[0] for item in default).items())))
    # Paths are values, never object keys: the public redactor treats keys containing
    # "credential" or "token" as secrets, even when their values are harmless counts.
    lane = {"full": expected, "file_weights": [[path, weight] for path, weight in weights.items()],
            "pass": config._default_suite_pass, "weights_source": source}
    shard = _shard(config)
    if shard:
        index, count = shard
        owners = file_shards(weights, count)
        assigned = [node for node in expected if owners[node.split("::", 1)[0]] == index]
        kept = set(assigned)  # Possibly empty: then this shard proves it ran nothing of this pass.
        deselected = [item for item in items if item.nodeid not in kept]
        items[:] = [item for item in items if item.nodeid in kept]
        config.hook.pytest_deselected(items=deselected)
        lane.update(shard=list(shard), assigned=assigned)
    config._default_suite_lane = lane


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node, error):
    """xdist's controller does not collect: retain the workers' agreed lane proof."""
    if not _enabled(node.config):
        return
    lane = getattr(node, "workeroutput", {}).get("default_suite_lane")
    if lane is None:
        return  # A dead worker makes pytest red and cannot supply a complete projection.
    prior = getattr(node.config, "_default_suite_lane", lane)
    if prior != lane:
        node.config._default_suite_mismatch = True
    node.config._default_suite_lane = lane


def pytest_sessionfinish(session):
    config = session.config
    lane = getattr(config, "_default_suite_lane", None)
    if hasattr(config, "workeroutput"):
        if lane is not None:
            config.workeroutput["default_suite_lane"] = lane
    elif getattr(config, "_default_suite_mismatch", False):
        print("DEFAULT_SUITE_INCOMPLETE: workers disagree about the complete lane")
        session.exitstatus = 1
    elif (lane is not None and lane.get("assigned") == []
          and session.exitstatus == pytest.ExitCode.NO_TESTS_COLLECTED):
        session.exitstatus = pytest.ExitCode.OK  # An empty shard ran all it was assigned: nothing.
