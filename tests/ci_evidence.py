"""Opt-in, safe CI result facts; raw JUnit is deliberately never read or uploaded.

Registered after tests/conftest.py establishes isolation. The summary entrypoint
uses only stdlib and trusts the supplied Actions producer outcome, not case totals.
events.jsonl is the controller's incremental journal: it names the tests in flight
when a session is killed before its final results.json export.

Everything here is diagnostic except `reconcile-shards`: sharded UI and default lanes
are proven complete only by their shards' projections, so there a missing or invalid
projection is a failure, never an unknown. A pass that scripts/ci_retry.py ran keeps
its first session's projection where it was; each retry round has its own in
`retry-<k>`, and `verdict.json` beside them says what the rounds decided.

`durations` turns one run's projections into the weight table the next runs shard
by (tests/default_lane.py, tests/browser_lane.py); `run-verdict` lists every flaky,
accepted, failed and never-run test of a run, completely. Both, and the readers of a
verdict and of a session's export, live in the stdlib module tests/ci_verdicts.py.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import threading
from time import monotonic

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    # `python3 -I -S tests/ci_evidence.py`: nothing else puts the checkout on sys.path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.ci_verdicts import (  # noqa: E402 -- stdlib only, as everything above `import pytest` below
    DURATIONS_ENV, PHASES as _PHASES, ROUND_DIR as _ROUND_DIR, SECTIONS as _SECTIONS, attempt_text as _attempt,
    cell as _cell, durations_command, error_text as _error, final as _final, journal_rows as _journal_rows,
    run_verdict_command, system_of as _system, valid as _valid,
    verdict_counts as _verdict_counts, verdict_file as _verdict_file, verdict_passes as _verdict_passes, write_json,
)


def output_dir(config) -> Path | None:
    value = getattr(getattr(config, "option", None), "ci_evidence_dir", None)
    return Path(value).resolve() if value else None


def _case_outcomes(reports: list[dict]) -> dict[str, str]:
    cases = {}
    for row in reports:
        node = row["nodeid"]
        status = row["outcome"]
        if status == "failed":
            cases[node] = "failed"
        elif cases.get(node) != "failed":
            if status == "skipped":
                cases[node] = "skipped"
            elif row["phase"] == "call":
                if cases.get(node) != "skipped":
                    cases[node] = "passed"
            else:
                cases.setdefault(node, "not_run")
    return cases


def _journal(root: Path) -> tuple[list[str], list[dict]] | None:
    """Tests in flight and failed reports from a session without a final export.

    Invalid report rows are skipped.
    """
    rows = _journal_rows(root)
    if rows is None:
        return None
    running, failed = {}, []
    for row in rows:
        if row.get("event") == "start":
            running[row["nodeid"]] = None
        elif row.get("event") == "finish":
            running.pop(row["nodeid"], None)
        elif row.get("event") == "report" and _valid(row):
            if row["phase"] == "crash":
                running.pop(row["nodeid"], None)  # A crashed worker never sends finish.
            if row["outcome"] == "failed":
                failed.append(row)
    return list(running), failed


def _journal_projection(root: Path) -> dict | None:
    """A killed session's stand-in for its export: the lane blocks its journal recorded at
    collection (with that session's identity and system) and the reports it got to; None
    when the journal names no lane. Its exit status is unknown."""
    projection = None
    for row in _journal_rows(root) or []:
        if (row.get("event") == "lane" and isinstance(row.get("block"), str)
                and isinstance(row.get("lane"), dict) and isinstance(row.get("github"), dict)):
            projection = projection or {"session_exit_code": None, "reports": [], "github": row["github"],
                                        "environment": row.get("environment") or {}}
            projection[row["block"]] = row["lane"]
        elif row.get("event") == "report" and _valid(row) and projection is not None:
            projection["reports"].append(row)
    return projection


def _executed(reports) -> set[str]:
    """The lane guard's own rule (tests/browser_lane.py LaneReconciliation): a node is
    executed once it has a call phase or any non-passed report; a crash row is one."""
    return {row["nodeid"] for row in reports if row["phase"] in ("call", "crash") or row["outcome"] != "passed"}


def _round_executed(root: Path) -> set[str]:
    """What one retry round executed: by its final export, else by its journal."""
    data, error = _final(root)
    if not error:
        return _executed(data["reports"])
    return _executed([row for row in _journal_rows(root) or [] if row.get("event") == "report" and _valid(row)])


def _flight_table(running: list[str]) -> list[str]:
    if not running:
        return []
    return ["", "| Test in flight when the session ended |", "| --- |"] + [
        f"| {_cell(node)} |" for node in running]


def _failure_table(failed: list[dict]) -> list[str]:
    if not failed:
        return []
    return ["", "| Test | Phase | Error type |", "| --- | --- | --- |"] + [
        f"| {_cell(row['nodeid'])} | {_cell(row['phase'])} | "
        f"{_cell(_error(row) or 'see test step')} |" for row in failed]


def render_summary(root: Path, *, producer_outcome: str, artifact_outcome: str,
                   artifact_url: str = "", scope: str = "full", label: str = "") -> str:
    """Reporter failure cannot replace producer truth; missing reports stay unknown."""
    data, error = _final(root)
    available = not error
    # The selected CLI writes scope beside the pass directories before execution.
    # Its actual tier may escalate to full; never describe a subset as full.
    try:
        selection = json.loads((Path(root).parent / "selection.json").read_text(encoding="utf-8"))
        if scope == "full" and selection.get("tier") in ("selected", "full"):
            scope = selection["tier"]
    except (OSError, ValueError, AttributeError):
        pass
    lines = [
        "## CI test evidence" + (f" — {_cell(label)}" if label else "")
        + (" — PARTIAL DIAGNOSTIC" if scope != "full" else ""),
        "",
        f"Producer step: **{_cell(producer_outcome)}**. Selection: **{_cell(scope)}**.",
        "Testcase results and diagnostic availability are separate from that process outcome.",
        "",
    ]
    if available:
        identity = data.get("github", {})
        if identity:
            lines.append(f"Commit: `{_cell(identity.get('sha', 'unknown'))}`; "
                         f"run {_cell(identity.get('run_id', 'unknown'))}, "
                         f"attempt {_cell(identity.get('run_attempt', 'unknown'))}.")
        counts = Counter(_case_outcomes(data["reports"]).values())
        lines.append("Cases: " + ", ".join(f"{name}={counts[name]}" for name in
                                         ("passed", "failed", "skipped", "not_run")) + ".")
        lines.append(f"Observed pytest session exit: {_cell(data.get('session_exit_code', 'unknown'))}.")
        collected = data.get("tests_collected")
        if isinstance(collected, int) and collected > sum(counts.values()):
            # A stopped session (dead worker, -x, session timeout) leaves its queue unreported.
            lines.append(f"**{collected - sum(counts.values())} collected test(s) produced no report**: "
                         "the session stopped before running them.")
        lines.extend(_failure_table([row for row in data["reports"] if row["outcome"] == "failed"]))
        # An interrupted session still exports; only its journal names what it left running.
        lines.extend(_flight_table((_journal(root) or ([], []))[0]))
        if data.get("collection_failures"):
            lines.append(f"Collection failures: {len(data['collection_failures'])}.")
    else:
        lines.append("Case outcomes: **unknown**. No passing result is inferred.")
        journal = _journal(root)
        if journal is not None:
            running, failed = journal
            lines.extend(["", "The session ended before its final export. Its incremental journal "
                          f"recorded {len(running)} test(s) in flight and {len(failed)} failed report(s)."])
            lines.extend(_flight_table(running))
            lines.extend(_failure_table(failed))
    # scripts/ci_retry.py: the cases above are the first session's; its rounds decided the pass.
    verdict, unreadable = _verdict_file(root)
    if verdict is not None:
        counts = _verdict_counts(verdict)
        lines.extend(["", f"Retry verdict after {len(verdict['rounds'])} round(s): **{_cell(verdict['verdict'])}** — "
                      + ", ".join(f"{name} {counts[key]}" for key, name, _ in _SECTIONS)
                      + (f". {_cell(verdict['reason'])}" if verdict.get("reason") else "") + "."])
    elif unreadable:
        lines.extend(["", f"Retry verdict: {_cell(unreadable.removeprefix('its retry verdict is '))}."])
    providers = []
    for path in sorted(root.glob("provider-*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(row, dict):
                raise ValueError("invalid provider projection")
            providers.append(row)
        except (OSError, ValueError):
            error = "a provider projection is unreadable"
    expected_providers = {row["canary_id"] for row in data.get("reports", [])
                          if row.get("canary_id")}
    recorded_providers = {row.get("canary_id") for row in providers}
    if expected_providers - recorded_providers:
        error = "provider result projections are missing"
    if any(row.get("diagnostics_errors") for row in providers):
        error = "provider evidence contains recorded capture gaps"
    for path in sorted(root.glob("browser/*/evidence.json")):
        try:
            browser = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(browser, dict):
                raise ValueError("invalid browser projection")
            if browser.get("diagnostics_incomplete"):
                error = "browser evidence contains recorded capture gaps"
        except (OSError, ValueError):
            error = "a browser projection is unreadable"
    if providers:
        lines.extend(["", "| Provider check | Outcome | Recorded cause |",
                      "| --- | --- | --- |"])
        for row in providers:
            lines.append(f"| {_cell(row.get('canary_id', row.get('nodeid', 'unknown')))} | "
                         f"{_cell(row.get('outcome', 'unknown'))} | "
                         f"{_cell(row.get('violation') or row.get('classification') or 'none recorded')} |")
    if artifact_url.startswith("https://"):
        lines.extend(["", f"[Download safe evidence]({artifact_url})"])
    incomplete = error or artifact_outcome != "success" or not artifact_url
    if incomplete:
        lines.extend(["", "**diagnostics_incomplete**: " +
                      _cell(error or f"artifact outcome={artifact_outcome}; artifact URL available={bool(artifact_url)}") +
                      ". The original producer outcome above is unchanged."])
    lines.extend(["", "Raw JUnit, exception bodies, credentials and private runtime stores are not in this export."])
    return "\n".join(lines) + "\n"


def _escaped(value, *, property_value: bool = False) -> str:
    """Workflow-command data: it cannot end the command line, nor a property its field."""
    text = str(value).replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    return text.replace(":", "%3A").replace(",", "%2C") if property_value else text


def _annotation(title: str, message: str, command: str = "error") -> str:
    return f"::{command} title={_escaped(title, property_value=True)}::{_escaped(message)}"


def render_annotations(roots, *, limit: int = 10) -> list[str]:
    """At most `limit` commands over all directories: errors first, failed tests leading.

    A pass with a retry verdict (scripts/ci_retry.py) is annotated from it: what still
    blocks is an error, a flaky or accepted failure only a warning.
    """
    failed, collection, running, never, warnings = [], [], [], [], []
    for root in roots:
        verdict, _ = _verdict_file(root)
        if verdict is not None:
            issues = {row["nodeid"]: row["issue"] for row in verdict["accepted"]}
            failed += [(row["nodeid"], "; ".join(_attempt(attempt) for attempt in row.get("attempts") or []
                                                 if isinstance(attempt, dict)))
                       for row in verdict["failed"] if row["nodeid"] not in issues]
            never += verdict["not_run"]
            warnings += [_annotation("Flaky test (passed on its retry)", row["nodeid"] + (
                f" (first: {_attempt(row['first'])})" if isinstance(row.get("first"), dict) else ""), "warning")
                for row in verdict["flaky"]]
            warnings += [_annotation(f"Accepted failure (#{issue})", node, "warning") for node, issue in issues.items()]
            continue
        data, error = _final(root)
        if not error:
            failed += [(row["nodeid"], _attempt(row)) for row in data["reports"] if row["outcome"] == "failed"]
            rows = data.get("collection_failures")
            collection += [row["nodeid"] for row in (rows if isinstance(rows, list) else [])
                           if isinstance(row, dict) and isinstance(row.get("nodeid"), str)]
            running += (_journal(root) or ([], []))[0]
        elif (journal := _journal(root)) is not None:
            running += journal[0]
            failed += [(row["nodeid"], _attempt(row)) for row in journal[1]]
    lines = [_annotation("Failed test", f"{node} ({detail})" if detail else node) for node, detail in failed]
    lines += [_annotation("Collection failed", node) for node in collection]
    lines += [_annotation("Test in flight when the session was killed", node) for node in running]
    lines += [_annotation("Test never ran", node) for node in never]
    lines += warnings
    shown = lines[:max(limit, 0)]
    if len(lines) > len(shown):
        shown.append(f"::notice::{len(lines) - len(shown)} more failed tests are listed in the job summary")
    return shown


def _named(nodes, *, shown: int = 20) -> str:
    nodes = sorted(nodes)
    return ", ".join(nodes[:shown]) + (f" … and {len(nodes) - shown} more" if len(nodes) > shown else "")


def file_shards(weights: dict[str, int], count: int) -> dict[str, int]:
    """Whole-file greedy packing, deterministic even when input insertion order differs."""
    loads, owners = [0] * count, {}
    for path in sorted(weights, key=lambda path: (-weights[path], path)):
        index = min(range(count), key=lambda index: (loads[index], index))
        owners[path] = index + 1
        loads[index] += weights[path]
    return owners


def node_shards(weights: list[int], count: int) -> list[int]:
    """The 1-based shard of each position: greedy largest-first packing, ties by position.

    Positions, not ids, break ties: an exported id may be redacted after the plugin
    packed it, while its position in the recorded lane stays the plugin's.
    """
    loads, owners = [0] * count, [0] * len(weights)
    for position in sorted(range(len(weights)), key=lambda position: (-weights[position], position)):
        index = min(range(count), key=lambda index: (loads[index], index))
        owners[position] = index + 1
        loads[index] += weights[position]
    return owners


def _pairs(rows) -> bool:
    """A non-empty `[[name, positive int]]` weight list."""
    return isinstance(rows, list) and bool(rows) and all(
        isinstance(row, list) and len(row) == 2 and isinstance(row[0], str) and type(row[1]) is int and row[1] > 0
        for row in rows)


def _lane_shape(lane, lane_key="ui_browser") -> bool:
    def strings(value):
        return isinstance(value, list) and all(isinstance(node, str) for node in value)
    if not isinstance(lane, dict) or not strings(lane.get("full")):
        return False
    if lane_key == "ui_browser":
        weights = lane.get("node_weights")
        if weights is not None and not (_pairs(weights) and [row[0] for row in weights] == lane["full"]):
            return False
    else:
        weights = lane.get("file_weights")
        if not _pairs(weights):
            return False
        files = dict(weights)
        if len(files) != len(weights) or any(node.split("::", 1)[0] not in files for node in lane["full"]):
            return False
    shard = lane.get("shard")
    return "shard" not in lane or (
        isinstance(shard, list) and len(shard) == 2 and all(type(part) is int for part in shard)
        and strings(lane.get("assigned")))


def _lane_proofs(root: Path, *, sha: str, run_id: str, gaps: list, others: list | None = None,
                 lane_key="ui_browser", system: str = "") -> list[dict]:
    """Lane projections beneath root that belong to this commit and run.

    Each results.json is judged where it lies, never by its directory name. One
    without the requested lane block is another producer's or a session the
    guard refused before it recorded a lane: it proves no lane node and is collected in
    `others` for its session status only. An unreadable
    one, or a lane projection of another commit or run, is a gap. A retry round's
    projection (`retry-<k>`, scripts/ci_retry.py) belongs to the verdict beside its
    pass, never to the lane: its directory is not searched, and a proof whose pass
    holds a verdict of this commit and run has executed what any of its rounds did.
    A pass whose session was killed before its final export is judged by the lane
    blocks its journal recorded (`_journal_projection`): its exit status is unknown,
    so only a passing verdict makes it a proof (`_red_session`).
    With `system`, a projection made on another operating system is another lane's.
    """
    proofs = []
    for directory, dirs, files in os.walk(root):
        dirs[:] = sorted(name for name in dirs if not _ROUND_DIR.fullmatch(name))
        if "results.json" in files:
            name = (Path(directory) / "results.json").relative_to(root).as_posix()
            data, error = _final(Path(directory))
            if error:
                gaps.append(f"{name}: {error}")
                continue
        elif "events.jsonl" in files and (data := _journal_projection(Path(directory))) is not None:
            name = (Path(directory) / "events.jsonl").relative_to(root).as_posix()
        else:
            continue
        if system and _system(data) != system:
            continue
        identity = data.get("github") if isinstance(data.get("github"), dict) else {}
        attempt = identity.get("run_attempt")
        verdict, why = _verdict_file(Path(directory))
        if verdict is not None and (verdict["github"].get("sha") != sha or verdict["github"].get("run_id") != run_id):
            verdict, why = None, (f"its retry verdict belongs to commit {verdict['github'].get('sha')} "
                                  f"run {verdict['github'].get('run_id')}")
        if lane_key not in data:
            if others is not None:
                others.append({"name": name, "exit": data.get("session_exit_code"), "killed": name.endswith(".jsonl"),
                               "verdict": verdict, "verdict_error": why})
            continue
        lane = data[lane_key]
        if not _lane_shape(lane, lane_key):
            gaps.append(f"{name}: invalid {lane_key} block")
        elif identity.get("sha") != sha or identity.get("run_id") != run_id:
            gaps.append(f"{name}: belongs to commit {identity.get('sha')} run "
                        f"{identity.get('run_id')}, expected commit {sha} run {run_id}")
        elif not (isinstance(attempt, str) and attempt.isdecimal()):
            gaps.append(f"{name}: run attempt is unknown")
        else:
            executed = _executed(data["reports"])
            for row in verdict["rounds"] if verdict is not None else []:
                if row["dir"] != ".":
                    executed |= _round_executed(Path(directory) / row["dir"])
            proofs.append({"name": name, "attempt": int(attempt), "lane": lane, "executed": executed,
                           "duplicates": [node for node, seen in Counter(
                               row["nodeid"] for row in data["reports"]
                               if row["phase"] == "call" and not row.get("is_subtest", False)).items() if seen > 1],
                           "exit": data.get("session_exit_code"), "killed": name.endswith(".jsonl"),
                           "verdict": verdict, "verdict_error": why})
    return proofs


def _red_session(label: str, proof: dict) -> list[str]:
    """The gap of a proof whose session did not end with exit status 0, or was killed.

    None when its pass holds a passing retry verdict of this commit and run: its
    rounds retried what failed once and ran what the session left (scripts/ci_retry.py).
    """
    status = proof["exit"]
    if type(status) is int and status == 0:
        return []
    verdict = proof.get("verdict")
    if verdict is not None and _verdict_passes(verdict):
        return []
    counts = _verdict_counts(verdict) if verdict is not None else {}
    detail = (f"; its retry verdict is red ({counts['failed']} failed, {counts['not_run']} not run)"
              if verdict is not None else f"; {proof['verdict_error']}" if proof.get("verdict_error") else "")
    if proof.get("killed"):
        return [f"{label}: its session ended before its final export{detail}"]
    return [f"{label}: its session ended with exit status {status if type(status) is int else 'unknown'}{detail}"]


def reconcile_shards(manifest_root: Path, shards_root: Path, *, count: int, sha: str,
                     run_id: str, lane_key="ui_browser", system: str = "") -> tuple[list[str], list[str]]:
    """(gaps, notes) of one run's sharded lane; any gap makes the lane RED.

    AUTHORITATIVE, unlike every other reader in this module: the manifest is the
    unsharded collection's witness of the lane, each shard must prove it saw that
    same lane, took exactly its slice and executed all of it. A proof whose session
    ended with a non-zero exit status is a gap as well: what GitHub reports as the
    result of a matrix job after one leg is re-run is undocumented, so a red session
    is refused here and never left to the job results alone. The one exception is a
    passing retry verdict beside the proof (`_red_session`). With `system`, only that
    operating system's projections make up the lane.
    """
    gaps, notes = [], []
    if not sha or not run_id or count < 1:
        return ["the expected commit, run id and a positive shard count are required"], notes
    full = None
    witnesses = _lane_proofs(manifest_root, sha=sha, run_id=run_id, gaps=gaps, lane_key=lane_key, system=system)
    gaps += [f"{proof['name']}: the manifest carries a shard block"
             for proof in witnesses if "shard" in proof["lane"]]
    witnesses = [proof for proof in witnesses if "shard" not in proof["lane"]]
    if not witnesses:
        gaps.append("manifest: no valid projection of the unsharded lane")
    else:
        manifest = max(witnesses, key=lambda proof: proof["attempt"])
        full = manifest["lane"]["full"]
        source = manifest["lane"].get("weights_source")
        notes.append(f"manifest: {len(full)} nodes, from attempt {manifest['attempt']}"
                     + (f"; weights {source}" if source else ""))
        same = sum(proof["attempt"] == manifest["attempt"] for proof in witnesses)
        if same > 1:  # Two witnesses of one attempt cannot both be the manifest.
            gaps.append(f"manifest: {same} projections from attempt {manifest['attempt']}")
        gaps += _red_session("manifest", manifest)
        if not full or len(set(full)) != len(full):
            gaps.append("manifest: the lane is empty" if not full else "manifest: duplicate node ids: "
                        + _named(node for node, seen in Counter(full).items() if seen > 1))
            full = None
    # UI nodes are packed by their recorded time when the manifest carries node weights.
    packed = (node_shards([weight for _, weight in manifest["lane"]["node_weights"]], count)
              if full is not None and lane_key == "ui_browser" and "node_weights" in manifest["lane"] else None)
    by_shard, others = {}, []
    for proof in _lane_proofs(shards_root, sha=sha, run_id=run_id, gaps=gaps, others=others,
                              lane_key=lane_key, system=system):
        index, total = proof["lane"].get("shard", (0, 0))
        if total != count or not 1 <= index <= count:
            gaps.append(f"{proof['name']}: " + (f"declares shard {index}/{total}, expected one of {count}"
                                                if total else "carries no shard block"))
        else:
            by_shard.setdefault(index, []).append(proof)
    proven, winners = set(), set()
    for index in range(1, count + 1):
        label = f"shard {index}/{count}"
        proofs = sorted(by_shard.get(index, []), key=lambda proof: proof["attempt"])
        if not proofs:
            gaps.append(f"{label}: no proof")
            continue
        proof = proofs[-1]
        winners.add(proof["name"].split("/", 1)[0])
        attempts = [candidate["attempt"] for candidate in proofs]
        if attempts.count(proof["attempt"]) > 1:
            gaps.append(f"{label}: {attempts.count(proof['attempt'])} proofs from attempt {proof['attempt']}")
        gaps += _red_session(label, proof)
        if proof["duplicates"]:
            gaps.append(f"{label}: duplicate executions: " + _named(proof["duplicates"]))
        assigned, executed, verdict = proof["lane"]["assigned"], proof["executed"], proof["verdict"]
        notes.append(f"{label}: proof from attempt {proof['attempt']}"
                     + (f" (attempts present: {', '.join(map(str, attempts))})" if len(proofs) > 1 else "")
                     + f"; assigned {len(assigned)}, executed {len(executed & set(assigned))}"
                     + (f"; retry verdict {verdict['verdict']} over {len(verdict['rounds'])} round(s): "
                        f"FLAKY {len(verdict['flaky'])}, ACCEPTED {len(verdict['accepted'])}" if verdict else ""))
        if full is not None and proof["lane"]["full"] != full:
            theirs, ours = set(proof["lane"]["full"]), set(full)
            gaps.append(f"{label}: its lane differs from the manifest"
                        + (f"; only in the shard: {_named(theirs - ours)}" if theirs - ours else "")
                        + (f"; only in the manifest: {_named(ours - theirs)}" if ours - theirs else "")
                        + ("; same nodes in another order" if theirs == ours else ""))
        # Preserve recorded order: redaction can change a node's lexical position.
        # UI slices by node position, or packs positions by recorded time; the default
        # suite assigns whole files by weight. An empty slice is a valid assignment.
        if full is not None:
            if lane_key == "ui_browser":
                expected = (full[index - 1::count] if packed is None
                            else [node for node, owner in zip(full, packed) if owner == index])
                if proof["lane"].get("node_weights") != manifest["lane"].get("node_weights"):
                    gaps.append(f"{label}: node weights differ from the manifest")
            else:
                weights = manifest["lane"]["file_weights"]
                owners = file_shards(dict(weights), count)
                expected = [node for node in full if owners[node.split("::", 1)[0]] == index]
                if proof["lane"]["file_weights"] != weights:
                    sources = (proof["lane"].get("weights_source"), manifest["lane"].get("weights_source"))
                    gaps.append(f"{label}: file weights differ from the manifest"
                                + (" (weights {} here, {} in the manifest)".format(*sources)
                                   if sources[0] != sources[1] else ""))
            if assigned != expected:
                gaps.append(f"{label}: its assignment is not slice {index} of {count} of the manifest lane")
        if set(assigned) - executed:
            gaps.append(f"{label}: assigned but not executed ({len(set(assigned) - executed)}): "
                        + _named(set(assigned) - executed))
        if executed - set(assigned):
            gaps.append(f"{label}: executed outside its assignment ({len(executed - set(assigned))}): "
                        + _named(executed - set(assigned)))
        proven |= executed & set(assigned)
    # A producer that shares a winning proof's artifact (browser tools) proves no lane node; its
    # red session is refused for the same reason a red shard is. One in the artifact of a
    # superseded attempt belongs to that attempt and is superseded with it.
    for other in others:
        if other["name"].split("/", 1)[0] in winners:
            gaps += _red_session(other["name"], other)
    if full is not None and set(full) - proven:
        gaps.append(f"lane: {len(set(full) - proven)} of {len(full)} manifest nodes are executed by no shard: "
                    + _named(set(full) - proven))
    return gaps, notes


def _reconcile(args) -> int:
    try:
        gaps, notes = reconcile_shards(args.manifest, args.shards, count=args.count, sha=args.sha,
                                       run_id=args.run_id, lane_key=args.lane, system=args.os)
    except Exception as error:  # An unreadable proof tree is a failed proof, not a skipped one.
        gaps, notes = [f"reconciliation failed ({type(error).__name__})"], []
    verdict = (f"INCOMPLETE — {len(gaps)} gap(s)" if gaps
               else f"complete — every node of the lane is executed by one of {args.count} shards")
    label = ("UI browser lane" if args.lane == "ui_browser" else args.lane.replace("_", " ") + " lane") + (
        f" ({args.os})" if args.os else "")
    lines = [f"{args.lane.upper()}_RECONCILE {verdict}"] + notes + [f"GAP {gap}" for gap in gaps]
    # GitHub shows at most ten error annotations per step; the log and summary list every gap.
    lines += [_annotation(f"{label} incomplete", gap) for gap in gaps[:10]]
    sys.stdout.buffer.write("".join(line + "\n" for line in lines).encode("utf-8", "replace"))
    sys.stdout.flush()
    try:
        with args.summary.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(
                [f"## {label} reconciliation — {_cell(verdict)}", "",
                 f"Commit `{_cell(args.sha)}`, run {_cell(args.run_id)}. "
                 "A session that ended red is a gap; its shard job names the failed tests.",
                 ""] + [f"- {_cell(note)}" for note in notes]
                + [f"- **GAP** {_cell(gap)}" for gap in gaps]) + "\n")
    except OSError as error:  # The verdict stands without its rendering.
        print(f"::warning::{label} reconciliation summary is unavailable ({type(error).__name__}).")
    return 1 if gaps else 0


def _main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    summarize = commands.add_parser("summarize")
    summarize.add_argument("--evidence-dir", type=Path, required=True)
    summarize.add_argument("--summary", type=Path, required=True)
    summarize.add_argument("--producer-outcome", required=True)
    summarize.add_argument("--artifact-outcome", required=True)
    summarize.add_argument("--artifact-url", default="")
    summarize.add_argument("--scope", default="full")
    summarize.add_argument("--label", default="")
    annotate = commands.add_parser("annotate")
    annotate.add_argument("--evidence-dir", type=Path, action="append", required=True)
    annotate.add_argument("--limit", type=int, default=10)
    reconcile = commands.add_parser("reconcile-shards")
    reconcile.add_argument("--lane", choices=("ui_browser", "default_parallel", "default_serial"),
                           default="ui_browser")
    reconcile.add_argument("--os", choices=("darwin", "linux", "windows"), default="",
                           help="Only this system's projections make up the lane")
    for flag, kind in (("--manifest", Path), ("--shards", Path), ("--count", int),
                       ("--sha", str), ("--run-id", str), ("--summary", Path)):
        reconcile.add_argument(flag, type=kind, required=True)
    durations = commands.add_parser("durations")
    for flag, kind in (("--evidence-root", Path), ("--run-id", str), ("--sha", str), ("--out", Path)):
        durations.add_argument(flag, type=kind, required=True)
    verdicts = commands.add_parser("run-verdict")
    for flag in ("--evidence-root", "--summary", "--out"):
        verdicts.add_argument(flag, type=Path, required=True)
    verdicts.add_argument("--required-lanes", nargs="*", default=[], metavar="PATTERN",
                          help="Label patterns (fnmatch) that must each have a verdict")
    argv = sys.argv[1:] if argv is None else list(argv)
    try:
        args = parser.parse_args(argv)
    except SystemExit as stop:
        if not stop.code or argv[:1] not in (["annotate"], ["run-verdict"]):
            raise
        # Annotations and the run's verdict list are advisory: no usage error fails a step.
        print("::warning::" + ("CI annotations are" if argv[0] == "annotate" else "The run's test verdicts are")
              + " unavailable (usage error).")
        return 0
    if args.command == "reconcile-shards":
        return _reconcile(args)
    if args.command == "durations":
        return durations_command(args)
    if args.command == "run-verdict":
        return run_verdict_command(args)
    if args.command == "annotate":
        try:
            lines = render_annotations(args.evidence_dir, limit=args.limit)
            # Runners decode step output as UTF-8 whatever the console code page is.
            sys.stdout.buffer.write("".join(line + "\n" for line in lines).encode("utf-8", "replace"))
        except Exception as error:
            print(f"::warning::CI annotations are unavailable ({type(error).__name__}).")
        return 0
    text = render_summary(args.evidence_dir, producer_outcome=args.producer_outcome,
                          artifact_outcome=args.artifact_outcome, artifact_url=args.artifact_url,
                          scope=args.scope, label=args.label)
    with args.summary.open("a", encoding="utf-8") as handle:
        handle.write(text)
    if "diagnostics_incomplete" in text:
        print("::warning::CI diagnostics are incomplete; original test outcome is unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

# pytest is intentionally below the stdlib-only CLI entrypoint and loaded only
# through conftest's post-isolation plugin registration.
import pytest


def _machine() -> dict:
    """What ran the session. Hosted runners come in faster and slower hardware classes, and
    the same commit runs about 1.5x apart on them, so durations compare within one class:
    the processor, a fixed pure-Python workload's time, and the SQLite the stores ran."""
    import sqlite3
    import subprocess

    model = ""
    try:
        if sys.platform.startswith("linux"):
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines():
                if line.lower().startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
        elif sys.platform == "darwin":
            model = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                   text=True, timeout=5, check=False).stdout.strip()
        else:
            model = os.environ.get("PROCESSOR_IDENTIFIER", "") or platform.processor()
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    started, total = monotonic(), 0
    for value in range(2_000_000):
        total += value * value
    return {"cpu": model, "cpus": os.cpu_count(), "calibration_sec": round(monotonic() - started, 3),
            "sqlite": sqlite3.sqlite_version}
try:
    from _pytest.subtests import SubtestReport
except ModuleNotFoundError as error:
    if error.name != "_pytest.subtests":
        raise
    SubtestReport = ()  # pytest <9 has no built-in subtest reports; ordinary calls still count.


def pytest_addoption(parser):
    parser.addoption("--ci-evidence-dir", default=None,
                     help="Dedicated public projection directory; raw JUnit must be elsewhere.")


def pytest_configure(config):
    # An xdist worker writes where its controller reads (pytest_configure_node below), even when
    # tests/conftest.py derived that directory from this process's own session root.
    handed = getattr(config, "workerinput", {}).get("ci_evidence_dir")
    if handed:
        config.option.ci_evidence_dir = handed
    if output_dir(config) is not None:
        config.pluginmanager.register(_Results(config), "ci_safe_results")


@pytest.hookimpl(optionalhook=True)
def pytest_configure_node(node):
    directory = output_dir(node.config)
    if directory is not None:
        node.workerinput["ci_evidence_dir"] = str(directory)


def _timeout_callback(directory, report):
    """The timeout whose pytest-timeout callback the dead worker recorded for exactly this test of
    this run, read once. A stale run, another worker or test, or a torn record names no cause."""
    node = getattr(report, "node", None)
    worker = getattr(getattr(node, "gateway", None), "id", "")
    path = Path(directory) / f".timeout-{worker}.json"
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        record = None
    try:
        path.unlink()
    except OSError:
        pass
    expected = {"worker": worker, "testrunuid": getattr(node, "workerinput", {}).get("testrunuid"),
                "nodeid_sha256": hashlib.sha256(report.nodeid.encode("utf-8")).hexdigest()}
    if not (isinstance(record, dict) and expected["testrunuid"]
            and all(record.get(key) == value for key, value in expected.items())):
        return None
    seconds = record.get("timeout_seconds")
    return float(seconds) if type(seconds) in (int, float) and math.isfinite(seconds) and seconds > 0 else None


def pytest_collection_finish(session):  # noqa: ARG001
    """The run's weight table shaped this session's own lanes, which are collected now
    (tests/default_lane.py, tests/browser_lane.py). The sessions its tests start are
    other lanes: none of them inherits it."""
    os.environ.pop(DURATIONS_ENV, None)


def _redact(value, secrets):
    from ouroboros.observability import redact_projection
    from ouroboros.secret_masking import redact_known_values

    return redact_projection(redact_known_values(value, secrets)).value


def _identity() -> dict:
    return {name: os.environ.get("GITHUB_" + name.upper(), "") for name in ("sha", "run_id", "run_attempt")}


def _lanes(config) -> dict:
    """The lane blocks this session proves, by projection key (tests/browser_lane.py, tests/default_lane.py)."""
    blocks = {}
    lane = getattr(config, "_ui_browser_lane", None)
    if lane:
        blocks["ui_browser"] = lane
    lane = getattr(config, "_default_suite_lane", None)
    if lane:
        blocks["default_" + lane["pass"]] = lane
    return blocks


class _Results:
    def __init__(self, config):
        from ouroboros.secret_masking import MASKED_SECRET_SETTING_KEYS

        self.config = config
        self.reports = []
        self.collection_failures = []
        self.secrets = tuple(os.environ[key] for key in MASKED_SECRET_SETTING_KEYS if os.environ.get(key))
        self.safe_text = {}
        self.started_at = monotonic()
        self.journal = None
        self.lanes = {}  # Lane blocks an xdist worker handed over at collection.
        if not hasattr(config, "workerinput"):  # xdist re-emits worker events on the controller.
            try:
                path = output_dir(config) / "events.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                # Unbuffered: each row reaches the OS before a kill can lose it.
                self.journal = path.open("wb", buffering=0)
            except Exception as error:
                self._journal_off(error)

    def _journal_off(self, error=None):
        journal, self.journal = self.journal, None
        try:
            if journal is not None:
                journal.close()
        except Exception as close_error:
            error = error or close_error
        if error is not None:  # One line, then silence: the session outcome is untouched.
            from ouroboros.test_evidence import diagnostics_incomplete
            diagnostics_incomplete(type(error).__name__)
            print(f"CI_DIAGNOSTICS_INCOMPLETE: incremental journal disabled ({type(error).__name__})")

    def _event(self, event, nodeid, **facts):
        if self.journal is None:
            return
        try:
            # A parametrized nodeid can carry a secret: every value is redacted
            # before its first write; the cache keeps that to once per distinct text.
            row = {"event": event, "nodeid": nodeid,
                   "elapsed_seconds": monotonic() - self.started_at, **facts}
            for key, value in row.items():
                if not isinstance(value, str):  # Only approved numeric timing facts bypass redaction.
                    continue
                if value not in self.safe_text:
                    self.safe_text[value] = _redact(value, self.secrets)
                row[key] = self.safe_text[value]
            self.journal.write((json.dumps(row) + "\n").encode("ascii"))
        except Exception as error:
            self._journal_off(error)

    def pytest_unconfigure(self):
        self._journal_off()  # A session that never reached sessionfinish still releases the file.

    @pytest.hookimpl(tryfirst=True)
    def pytest_collection_finish(self, session):
        """Journal each lane block as soon as collection decided it, with this run's identity.

        A session killed before its final export (pytest-timeout's thread method ends
        the whole process on Windows) still names what it was assigned: scripts/ci_retry.py
        continues from it, and reconcile-shards proves its shard with the retry verdict.
        An xdist worker cannot write the controller's journal, and the blocks in its final
        output never arrive when the controller aborts on another worker's crash
        (tests/conftest.py): it leaves them beside the journal before it reports its
        collection, which runs after this hook.
        """
        lanes = _lanes(session.config)
        worker = getattr(session.config, "workerinput", {}).get("workerid")
        if worker is None:
            self._journal_lanes(lanes)
            return
        try:
            for block, lane in lanes.items():
                write_json(output_dir(session.config) / f".lane-{block}-{worker}.json", _redact(lane, self.secrets))
        except Exception as error:  # Diagnostics alone never alter the session.
            print(f"CI_DIAGNOSTICS_INCOMPLETE: lane hand-over failed ({type(error).__name__})")

    @pytest.hookimpl(optionalhook=True)
    def pytest_xdist_node_collection_finished(self, node, ids):  # noqa: ARG002
        """The xdist controller takes the blocks a worker left (above): it journals them and
        exports them should that worker's own output never arrive."""
        suffix = f"-{node.gateway.id}.json"
        lanes = {}
        for path in sorted(output_dir(self.config).glob(f".lane-*{suffix}")):
            try:
                lanes[path.name[len(".lane-"):-len(suffix)]] = json.loads(path.read_text(encoding="utf-8"))
                path.unlink()
            except (OSError, ValueError):
                continue
        lanes = {block: lane for block, lane in lanes.items() if block not in self.lanes}
        self.lanes.update(lanes)
        self._journal_lanes(lanes)

    def _journal_lanes(self, lanes):
        for block, lane in lanes.items():
            if self.journal is None:
                return
            try:
                row = {"event": "lane", "nodeid": "", "elapsed_seconds": monotonic() - self.started_at,
                       "block": block, "lane": lane, "github": _identity(),
                       "environment": {"platform": platform.system()}}
                self.journal.write((json.dumps(_redact(row, self.secrets)) + "\n").encode("ascii"))
            except Exception as error:
                self._journal_off(error)

    def pytest_runtest_logstart(self, nodeid):
        self._event("start", nodeid)

    def pytest_runtest_logfinish(self, nodeid):
        self._event("finish", nodeid)

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item, call):
        outcome = yield
        report = outcome.get_result()
        # The exception TYPE is useful; repr/str/longrepr can echo credentials.
        report.ci_error_type = type(call.excinfo.value).__name__ if call.excinfo else ""
        canary = getattr(getattr(item, "callspec", None), "params", {}).get("canary")
        report.ci_canary_id = getattr(canary, "canary_id", "")

    @pytest.hookimpl(optionalhook=True)
    def pytest_timeout_set_timer(self, item, settings):
        """Arm pytest-timeout's thread method as the plugin does, around the plugin's own callback.

        That callback prints its banner to an xdist worker's stdout, which execnet discards, and then
        ends the worker with os._exit(1) (#1949). On entry the wrapper leaves this worker's record for
        the controller (_timeout_callback); should the callback return, a debugger suppressed it and
        the record goes. Other methods, a run without xdist and a plugin without that callback keep
        the plugin's own timer.
        """
        import pytest_timeout

        callback, worker = getattr(pytest_timeout, "timeout_timer", None), getattr(self.config, "workerinput", None)
        if settings.method != "thread" or worker is None or not callable(callback):
            return None
        record = output_dir(self.config) / f".timeout-{worker['workerid']}.json"
        facts = json.dumps({"worker": worker["workerid"], "testrunuid": worker.get("testrunuid"),
                            "nodeid_sha256": hashlib.sha256(item.nodeid.encode("utf-8")).hexdigest(),
                            "timeout_seconds": settings.timeout})

        def invoked():
            try:
                record.write_text(facts, encoding="utf-8")
            except Exception:  # A diagnostic never delays or skips the timeout itself.
                pass
            try:
                callback(item, settings)
            finally:
                try:
                    record.unlink(missing_ok=True)
                except OSError:
                    pass

        timer = threading.Timer(settings.timeout, invoked)
        timer.name = f"pytest_timeout {item.nodeid}"

        def cancel():
            timer.cancel()
            timer.join()

        item.cancel_timeout = cancel
        timer.start()
        return True

    def pytest_runtest_logreport(self, report):
        duration = float(getattr(report, "duration", 0.0))
        # xdist reports a dead worker's test with when="???" and no worker-side attributes.
        crashed = report.when not in _PHASES
        row = {
            "nodeid": report.nodeid, "phase": "crash" if crashed else report.when,
            # Preserve the report's kind, never its potentially private subtest context.
            "is_subtest": isinstance(report, SubtestReport),
            "outcome": report.outcome,
            "duration_seconds": duration if math.isfinite(duration) else None,
            "error_type": "WorkerCrash" if crashed else getattr(report, "ci_error_type", ""),
            "canary_id": getattr(report, "ci_canary_id", ""),
        }
        cause = {}
        if crashed:  # The dead worker: the commit gate matches it to worker_crash (ouroboros/preflight_runner.py).
            row["worker"] = cause["worker"] = getattr(getattr(getattr(report, "node", None), "gateway", None), "id", "")
        seconds = _timeout_callback(output_dir(self.config), report) if crashed else None
        if seconds is not None:  # A fact about this test only; the worker's end stays unexplained beyond it.
            row["timeout_callback_seconds"] = cause["timeout_callback_seconds"] = seconds
            reporter = self.config.pluginmanager.get_plugin("terminalreporter")
            if reporter is not None:  # For readers, on its own line after xdist's "F"; this row is the record.
                reporter.write_line(f"\n[{report.node.gateway.id}] pytest-timeout callback invoked ({seconds:g}s) "
                                    f"while running {report.nodeid!r}; worker subsequently lost")
        self.reports.append(row)
        self._event("report", row["nodeid"], phase=row["phase"], outcome=row["outcome"],
                    error_type=row["error_type"], duration_seconds=row["duration_seconds"],
                    is_subtest=row["is_subtest"], **cause)

    def pytest_collectreport(self, report):
        if report.failed:
            self.collection_failures.append({"nodeid": report.nodeid, "outcome": "failed"})

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_sessionfinish(self, session, exitstatus):
        yield  # Read the final session status after ordinary guards have run.
        if hasattr(self.config, "workerinput"):
            return  # xdist's controller receives the worker reports.
        try:
            facts = {
                "format": 1, "session_exit_code": int(session.exitstatus),
                "tests_collected": session.testscollected, "reports": self.reports,
                "collection_failures": self.collection_failures,
                "environment": {"python": platform.python_version(), "platform": platform.system(),
                                "pytest": pytest.__version__, "machine": _machine()},
                "raw_exception_text": "omitted",
                "raw_junit": "private_not_copied",
                "github": _identity(),
            }
            # The complete lane, and this session's shard and slice: the workers' final word,
            # else what one handed over at collection (an aborted controller never gets the former).
            facts.update({**self.lanes, **_lanes(self.config)})
            crash = getattr(self.config, "_ouroboros_worker_crash", None)
            if crash:
                facts["worker_crash"] = crash
            write_json(output_dir(self.config) / "results.json", _redact(facts, self.secrets))
        except Exception as error:
            # Diagnostics alone never alter session.exitstatus or expose a body.
            from ouroboros.test_evidence import diagnostics_incomplete
            diagnostics_incomplete(type(error).__name__)
            print(f"CI_DIAGNOSTICS_INCOMPLETE: safe result export failed ({type(error).__name__})")
        finally:
            self._journal_off()
