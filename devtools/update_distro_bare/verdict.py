"""One cell's result from its raw evidence, on the runner; absent, skipped or unexplained evidence is never a pass.

    python3 -I -B verdict.py MANIFEST CELL EVIDENCE_DIR OUT.json       exit 0 only for an overall PASS

Stdlib only. A row is one of these:

- PASS: the reported pass of a selected node, a clean census, or a phase that exited 0 (``phases.txt``, which
  ``cell.sh`` writes from each phase container's exit status).
- FAIL: a product or census assertion failed, a node was skipped, or the report holds an unselected node.
- INCOMPLETE: the equipment did not finish. A phase exited nonzero without a report (builder, driver or container),
  or with every reported row passing (a session-level stop such as the subject conftest's exit status), or a node hit
  the per-test timeout, or the census could not read some bytes and saw no change elsewhere. Nothing here is a product
  statement.
- PATH_NOT_COVERED: preparation took another legitimate path than the cell predicts and everything else held for the
  path it took. The predicted branch is not covered; this is neither a product failure nor a pass.
- NOT_RUN: no evidence for the row.
- ACCOUNTED: a phase exited nonzero and another row of that phase says why. It adds nothing, and that row keeps
  the overall from being a pass.

CONDITION_NOT_MET means the cell's apt state was not what it exists to observe, so nothing can be said about the product.
Each row's ``proves`` is the narrow meaning a reader may quote.
"""
from __future__ import annotations

import json
import os
import re
import sys
import xml.etree.ElementTree as ET

NOTE = ("A disposable Ubuntu container on this interpreter; the row proves only what it names, not a real server's "
        "boot or serving, a VM, another distro or another Python.")
RANK = {"PASS": 0, "ACCOUNTED": 0, "PATH_NOT_COVERED": 1, "NOT_RUN": 2, "INCOMPLETE": 3, "FAIL": 4}
OUTCOME_RANK = {"passed": 0, "skipped": 1, "timeout": 2, "failed": 3, "error": 3}
PATH_NODE = "test_preparation_takes_the_path_this_interpreter_really_has"


def _load(path: str):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def junit(path: str) -> "dict | None":
    """``file.py::name`` -> worst outcome reported for it; ``None`` when the report is absent or unreadable."""
    try:
        tree = ET.parse(path)
    except (OSError, ET.ParseError):
        return None
    outcomes = {}
    for case in tree.getroot().iter("testcase"):
        module = (case.get("classname") or "").rsplit(".", 1)[-1]
        name = (case.get("name") or "").split("[", 1)[0]
        kinds = {child.tag: child.get("message") or "" for child in case}
        outcome = ("error" if "error" in kinds else "failed" if "failure" in kinds
                   else "skipped" if "skipped" in kinds else "passed")
        if "Timeout (>" in kinds.get("failure", "") + kinds.get("error", ""):  # "Timeout (>900.0s) from pytest-timeout"
            outcome = "timeout"
        key = f"{module}.py::{name}"
        if OUTCOME_RANK[outcome] >= OUTCOME_RANK[outcomes.get(key, "passed")]:
            outcomes[key] = outcome
    return outcomes


def phases(evidence: str) -> dict:
    """Phase name -> its container's exit status, from ``phases.txt``; a phase with no line has none."""
    found = {}
    try:
        with open(os.path.join(evidence, "phases.txt"), encoding="utf-8") as fh:
            for line in fh:
                match = re.fullmatch(r"phase (\S+) rc=(-?\d+)", line.strip())
                if match:
                    found[match.group(1)] = int(match.group(2))
    except OSError:
        pass
    return found


def _test_row(outcomes, node: str, proves: str, folder: str, expected_pip: str) -> dict:
    module = node.split("::", 1)[0].rsplit("/", 1)[-1]
    name = node.split("::", 1)[1]
    outcome = None if outcomes is None else outcomes.get(f"{module}::{name}")
    status = {"passed": "PASS", None: "NOT_RUN", "timeout": "INCOMPLETE"}.get(outcome, "FAIL")
    observed = outcome
    if name == PATH_NODE and status == "FAIL":
        record = _load(os.path.join(folder, "preparation.json")) or {}
        if record.get("held_for_taken_path") is True and record.get("taken") != expected_pip:
            status, observed = "PATH_NOT_COVERED", {"outcome": outcome, "taken": record.get("taken"),
                                                    "expected": expected_pip}
    return {"row": node, "status": status, "observed": observed, "proves": proves}


def _census_status(census) -> str:
    """A clean census passes; one that saw a change fails; one that only could not read some bytes is incomplete."""
    if census is None:
        return "NOT_RUN"
    if census.get("clean"):
        return "PASS"
    moved = any(census.get(name) for name in ("added", "removed", "changed")) or not all(
        census.get(name) for name in ("same_roots", "same_host", "same_dpkg"))
    return "INCOMPLETE" if census.get("unread") and not moved else "FAIL"


def _exit_row(phase_name: str, rc, report_present: bool, phase_rows: list, folder: str) -> dict:
    if rc is None:
        status = "NOT_RUN"
    elif rc == 0:
        status = "PASS"
    elif not report_present:
        status = "INCOMPLETE"
    elif any(row["status"] != "PASS" for row in phase_rows):
        status = "ACCOUNTED"
    else:
        status = "INCOMPLETE"
    try:
        with open(os.path.join(folder, "driver_rc.txt"), encoding="utf-8") as fh:
            driver = fh.read().strip()
    except OSError:
        driver = None
    return {"row": f"{phase_name}:exit", "status": status, "observed": {"rc": rc, "driver_rc": driver},
            "proves": "the phase's container ran its census, driver and census to the end and exited 0"}


def judge(manifest: dict, cell_name: str, evidence: str) -> dict:
    cell = manifest["cells"][cell_name]
    conditions = _load(os.path.join(evidence, "conditions.json"))
    exits = phases(evidence)
    rows, overall = [], "PASS"
    if conditions is None:
        overall = "NOT_RUN"
    elif conditions.get("condition_not_met"):
        overall = "CONDITION_NOT_MET"
    else:
        for phase_name, phase in cell["phases"].items():
            if not phase.get("nodes"):
                continue
            folder = os.path.join(evidence, phase_name)
            outcomes = junit(os.path.join(folder, "junit.xml"))
            phase_rows = [_test_row(outcomes, node, manifest["node_meaning"][node.split("::", 1)[1]], folder,
                                    cell["expect"]["expected_pip"]) for node in phase["nodes"]]
            extra = sorted(set(outcomes or {}) - {
                f"{n.split('::', 1)[0].rsplit('/', 1)[-1]}::{n.split('::', 1)[1]}" for n in phase["nodes"]})
            if extra:
                phase_rows.append({"row": f"{phase_name}:selection", "status": "FAIL", "observed": extra,
                                   "proves": "the report holds exactly the selected nodes"})
            census = _load(os.path.join(folder, "census_diff.json"))
            phase_rows.append({"row": f"{phase_name}:global_census", "status": _census_status(census),
                               "observed": census and {name: census.get(name) for name in (
                                   "files_before", "added", "removed", "changed", "unread", "same_dpkg",
                                   "bytecode_added", "bytecode_changed")},
                               "proves": manifest["node_meaning"]["global_census"]})
            phase_rows.append(_exit_row(phase_name, exits.get(phase_name), outcomes is not None, phase_rows, folder))
            rows += phase_rows
        overall = max((row["status"] for row in rows), key=RANK.__getitem__) if rows else "NOT_RUN"
        if RANK[overall] == 0 and any(row["status"] == "ACCOUNTED" for row in rows):
            overall = "INCOMPLETE"  # a nonzero exit is never a pass, even if the row that accounts for it went missing
    return {"cell": cell_name, "overall": overall, "subject": manifest["subject"]["commit"], "rows": rows,
            "conditions": conditions and {key: conditions.get(key) for key in (
                "real", "version", "euid", "marker", "marker_sha256", "marker_owner", "ensurepip_works", "own_pip",
                "ensurepip_module", "ensurepip_bundled", "packages", "root_mount", "condition_not_met")},
            "limit": NOTE, "unmeasured": manifest["not_proved"]}


def main(argv: list) -> int:
    with open(argv[0], encoding="utf-8") as fh:
        manifest = json.load(fh)
    result = judge(manifest, argv[1], argv[2])
    with open(argv[3], "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    print(result["overall"], [f"{row['row']}={row['status']}" for row in result["rows"]])
    return 0 if result["overall"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
