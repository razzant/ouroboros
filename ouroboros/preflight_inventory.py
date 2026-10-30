"""Host-written pytest inventory for selected gate attempts (never receipt authority)."""
from __future__ import annotations

from collections import Counter
import contextlib
import hashlib
import json
from pathlib import Path
import uuid


MODULE = "ouroboros_selection_inventory"
SELECT = "--select-test"
# Loaded explicitly by each controller; worker hooks forward their collection via
# xdist. No candidate module imports or persisted selection manifest are needed.
# A selected scope is ONE walk of tests/ that keeps the modules named by SELECT
# before any is imported: pytest walks the tree once per module given as an
# argument (30 modules, the same 848 nodes: 8.4-10.1 CPU-s per collecting process
# that way, 4.6-5.0 from one walk).
SOURCE = '''import json
from pathlib import Path
import pytest

_path = Path(__RESULT_PATH__)
_worker = False
_selected = None
_reached = set()

def _emit(event, **fields):
    if not _worker:
        with _path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(event=event, **fields)) + "\\n")

def pytest_addoption(parser):
    parser.addoption("__SELECT__", action="append", default=[], metavar="PATH",
                     help="collect only these test modules, from one walk of the directories given")

def _selection(config):
    global _selected
    if _selected is None:
        _selected = frozenset(config.invocation_params.dir / path for path in config.getoption("select_test"))
    return _selected

def pytest_ignore_collect(collection_path, config):
    selected = _selection(config)
    if not selected or not (collection_path.name.startswith("test_") and collection_path.suffix == ".py"):
        return None
    if collection_path in selected:
        _reached.add(collection_path)
        return None
    return True

def pytest_configure(config):
    global _worker
    _worker = hasattr(config, "workerinput")
    _emit("start")

def pytest_collection_finish(session):
    missing = sorted(str(path) for path in _selection(session.config) - _reached)
    if missing:
        raise pytest.UsageError("selected test modules the collection never reached: " + ", ".join(missing))
    if not _worker and session.items:
        _emit("collection", ids=[item.nodeid for item in session.items])

@pytest.hookimpl(optionalhook=True)
def pytest_xdist_node_collection_finished(node, ids):
    _emit("collection", ids=list(ids))

def pytest_runtest_logreport(report):
    if report.when == "teardown" or report.skipped or report.failed:
        _emit("report", nodeid=report.nodeid, when=report.when, outcome=report.outcome)

@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_sessionfinish(session, exitstatus):
    yield
    _emit("finish", exitcode=int(session.exitstatus))
'''


def selection_passes(selection, *, probe_module="", temp_root=None, inventory_dir=None):
    """Probe modules live in scratch; ``inventory_dir`` (default temp_root) keeps the full JSONL.

    The argv names every selected module (it is the workload a proof binds) and never
    begins with ``tests/`` for a selected scope, so no reader takes it for the full one.
    """
    from ouroboros import preflight_runner as pr

    scope = ["tests/"] if selection.tier == "full" else [*(f"{SELECT}={path}" for path in selection.files), "tests/"]
    passes = []
    for spec in pr._preflight_pass_specs(probe_module=probe_module):
        module = MODULE
        if temp_root is not None:
            module += "_" + uuid.uuid4().hex
            probe = pr._probe_dir(Path(temp_root))
            probe.mkdir(parents=True, exist_ok=True)
            result = Path(inventory_dir or temp_root) / f"inventory-{spec.label}.jsonl"
            result.write_text("", encoding="utf-8")
            (probe / f"{module}.py").write_text(SOURCE.replace("__RESULT_PATH__", repr(str(result)))
                                                .replace("__SELECT__", SELECT), encoding="utf-8")
        args = [arg for arg in spec.args if arg != "tests/"]
        passes.append(pr._make_pass(spec.label, [*scope, *args, "-p", module]))
    return passes


def read_inventory(temp_root, label):
    try:
        rows = [json.loads(line) for line in
                (Path(temp_root) / f"inventory-{label}.jsonl").read_text(encoding="utf-8").splitlines()]
        collections = [row["ids"] for row in rows if row["event"] == "collection"]
        reports = [row for row in rows if row["event"] == "report"]
        result = {"expected": collections[0] if collections else [], "collections": collections,
                  "executed": [r["nodeid"] for r in reports if r["when"] == "teardown"],
                  "skipped": [r["nodeid"] for r in reports if r["outcome"] == "skipped"],
                  "failed": [r["nodeid"] for r in reports if r["outcome"] == "failed"],
                  "exitcode": rows[-1].get("exitcode") if rows else None}
        expected, executed = result["expected"], result["executed"]
        if not all(isinstance(value, list) and all(isinstance(item, str) for item in value)
                   for value in (expected, executed, result["skipped"], result["failed"])):
            raise ValueError("invalid node inventory")
        problem = ""
        if any(collection != expected for collection in result["collections"]):
            problem = "worker collections disagree"
        elif len(set(expected)) != len(expected) or Counter(executed) != Counter(expected):
            problem = "expected and executed test ids differ (missing or duplicate execution)"
        elif result["exitcode"] not in (0, 5) or result["failed"]:
            problem = "test session did not finish green"
        result.pop("collections")
        return result, problem
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {}, f"test inventory unavailable: {exc}"


_LISTED_IDS = 50  # bounded excerpt of ids worth reading; counts and the digest cover the rest


def compact_inventory(inventory):
    """Counts, a digest of the expected ids and only the ids worth reading.

    The complete inventory stays with the run (and in the CLI's selection.json); a
    durable record repeated in events, exhibits, ledgers and handoffs never carries it.
    """
    compact = {}
    for label, row in (inventory or {}).items():
        expected, executed = row.get("expected", []), row.get("executed", [])
        compact[label] = {
            "expected": len(expected), "executed": len(executed), "skipped": len(row.get("skipped", [])),
            "digest": hashlib.sha256("\n".join(sorted(expected)).encode("utf-8")).hexdigest(),
            "failed": row.get("failed", [])[:_LISTED_IDS],
            "missing": sorted((Counter(expected) - Counter(executed)).elements())[:_LISTED_IDS],
            "unexpected": sorted((Counter(executed) - Counter(expected)).elements())[:_LISTED_IDS],
            "exitcode": row.get("exitcode"),
        }
    return compact


def scope_record(proof, inventory=None):
    selection = getattr(proof, "selection", None)
    rule = "full default suite" if proof.tier == "full" else "custom pytest selection"
    return {"tier": proof.tier, **(selection.record() if selection else {"rule": rule}),
            "candidate": proof.tree, "head": proof.head, "index_tree": proof.index_tree,
            **({"inventory": compact_inventory(inventory)} if inventory is not None else {})}


def inventory_problem(selection, inventory):
    """Every floor module must collect; opt-in-only affected modules may be excluded."""
    from ouroboros.test_selection import FLOOR_TESTS

    collected = {node.split("::", 1)[0] for row in inventory.values() for node in row.get("expected", [])}
    missing = sorted(set(FLOOR_TESTS) - collected)
    return "floor modules collected no tests: " + ", ".join(missing) if missing else ""


def note_lane_process(exc: BaseException, container, proc) -> None:
    """What a contained lane that raised had acquired, for ``launching``: ``True`` once
    ``spawn`` returned its process, ``False`` while the container never created one,
    ``None`` (unknown) when it created one but raised before handing it over, or cannot
    say. Its first reap's answer travels too: a missing handle proves nothing either way.
    Never raises: the lane's own exception is what propagates."""
    created = getattr(container, "root_created", None)
    try:
        teardown = container.reap()
        if not teardown and proc is not None and proc.poll() is None:
            proc.wait(timeout=5)
    except Exception as failure:
        teardown = f"the lane's teardown is unconfirmed: {failure}"
    try:
        exc.preflight_lane_process = True if proc is not None else False if created is False else None
        exc.preflight_lane_teardown = teardown
    except (AttributeError, TypeError):  # an exception that takes no attributes stays unknown
        pass


@contextlib.contextmanager
def launching(evidence: dict):
    """One contained lane's launch in the runner, keeping its ``execution`` true to what it
    acquired: ``attempted`` while the lane runs, ``executed`` once it returns (a pytest pass
    returns only after its spawn did; the node lane then states its own phase). A lane that
    raises leaves ``executed`` once its process was acquired, the earlier phase when it
    never created one, and ``attempted`` (unknown) otherwise. Only a raising lane's own
    teardown answer may later replace the runner's armed "did not return"."""
    before, evidence["execution"] = evidence.get("execution", "not_run"), "attempted"
    try:
        yield
    except BaseException as exc:
        acquired = getattr(exc, "preflight_lane_process", None)
        evidence["execution"] = "executed" if acquired is True else before if acquired is False else "attempted"
        if isinstance(teardown := getattr(exc, "preflight_lane_teardown", None), str):
            evidence["lane_teardown"] = teardown
        raise
    evidence["execution"] = "executed"


def attempt_outcome(run) -> str:
    """What a run that minted no green proof observed, read from the runner's own
    execution phase and lane exits (its evidence dict, or the ``result.json`` it wrote):
    ``not_started`` (no lane process was created), ``failed`` (a lane exited red),
    ``interrupted`` (a lane process was acquired but the run stopped before an accepted
    verdict: a timeout, a containment, worker, inventory or empty-collection refusal, or
    a later lane that never started) or ``start_unknown`` (the first lane raised before
    the runner could tell whether its process was created). The phase says ``executed``
    only for an acquired process (``launching``); a recorded lane is
    one that returned. A missing terminal is the reader's ``no_terminal``, never
    ``running``."""
    from ouroboros.preflight_runner import _PYTEST_EXIT_NO_TESTS

    execution = str(run.get("execution") or "not_run")
    if execution == "not_run":
        return "not_started"
    lanes = [lane for lane in run.get("lanes") or [] if isinstance(lane, dict)]
    red = [lane for lane in lanes if lane.get("returncode") not in (0, _PYTEST_EXIT_NO_TESTS, None)
           and not lane.get("containment_error")]
    return "failed" if red else "interrupted" if execution == "executed" or lanes else "start_unknown"


def record_failed_attempt(ctx, selection, subject, inventory, evidence=None):
    """Keep a run that minted no green proof visible without granting it any proof authority.

    Both tiers: ``subject`` is the tested checkout when the run bound one (a full run's
    too), ``evidence`` the runner's own record of its path, execution phase and lanes.
    The row stays on the ctx as the reviewer exhibit's holder (``preflight_receipts``)."""
    if ctx is None or getattr(ctx, "_preflight_tests_passed", False):
        return
    from ouroboros.commit_admission import workload_fingerprint
    from ouroboros.preflight_receipts import canonical_data_root
    from ouroboros.utils import append_jsonl, utc_now_iso

    run = evidence if isinstance(evidence, dict) else {}
    event = {"type": "preflight_test_attempt", "ts": utc_now_iso(), "status": "failed",
             "outcome": attempt_outcome(run),
             **(scope_record(subject, inventory) if subject else selection.record() if selection else {}),
             **({"workload_fingerprint": workload_fingerprint(subject.workload)} if subject else {}),
             "lanes": [{key: lane.get(key) for key in ("lane", "pass", "returncode") if key in lane}
                       | ({"containment_error": lane["containment_error"]} if lane.get("containment_error") else {})
                       for lane in run.get("lanes") or [] if isinstance(lane, dict)],
             "evidence": str(run["path"]) if run.get("path") else None}
    ctx._preflight_test_disclosure = event
    try:
        root = canonical_data_root(ctx)
        if root:
            append_jsonl(Path(root) / "logs/events.jsonl", event)
    except Exception:
        pass  # Diagnostics cannot replace the actual failure.
