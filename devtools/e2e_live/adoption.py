"""SM1's direct adoption, read from durable artifacts: the exact reviewed commit, the real test run that qualified it,
its restart, and what the server then serves.

A self-change lives on the task's body candidate until ``request_restart(adopt_commit=...)`` names it. ``body_adoption``
owns the transition and it happens AFTER the root task has finished: the restart request reaches the supervisor with
the task's own end (``agent_task_pipeline``), so nothing here or in a script may wait inside the task for it. This
contract is separate from ``--self-mod``'s post-task evolution, whose absorbed-cycle count says nothing about it.

The identity chain is the check: the commit the landing ``commit_reviewed`` produced (its result names a review record,
whose landing names the commit, because the result text itself carries no SHA) must be the candidate row's newest
reviewed commit, the one the restart request's own ``adopt_commit`` names, the handoff's candidate and qualification
head, the restart verification's SHA, and what the server and its checkout then serve. The handoff is matched by its
id and that SHA: a later evolution cycle's adoption is a different one. Nothing here writes a receipt, and a refusal
or unconfirmed state stays a fact of its own.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any

from devtools.benchmarks.common.manifests import runtime_attestation
from devtools.e2e_live.traces import FORK_ROOTS

_REVIEW_RECORD_RE = re.compile(r"review_record_id: (rl-[0-9A-Za-z-]+)")
_RESTART_REFUSAL_RE = re.compile(r"(RESTART_[A-Z]+)(?:: (ADOPTION_[A-Z_]+))?")
_RESTART_RECEIPT_RE = re.compile(r"RESTART_[A-Z]+: request \S")   # the host's receipt of a request it holds, not yet accepted
_LANDED_PREFIX = "OK: committed to "
_RESTART_ACCEPTED = "Restart requested."
_RESTART_HANDOVER_EXIT = 42   # ``server_control``: the process handed over to a successor; it did not die


def _git(cwd: pathlib.Path, *args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=str(cwd), check=False, capture_output=True, text=True)
    return (proc.stdout or "").strip()


def _drive_roots(data_root: pathlib.Path) -> list[pathlib.Path]:
    return [pathlib.Path(data_root), *sorted(pathlib.Path(data_root).glob(FORK_ROOTS))]


def _record_landing(record_id: str, data_root: pathlib.Path) -> dict:
    """The newest commit the review record says landed, through the ledger's own readers."""
    from ouroboros.review_doctrine import public_commit_landings
    from ouroboros.review_ledger import load_record

    for root in _drive_roots(data_root):
        try:
            record = load_record(root, record_id)
        except (OSError, ValueError) as exc:
            return {"record_id": record_id, "error": f"{type(exc).__name__}: {exc}"[:300]}
        landings = public_commit_landings(record) if record else []
        if landings:
            newest = landings[-1]
            return {"record_id": record_id, **{key: newest.get(key, "") for key in (
                "commit_sha", "tree_sha", "parent_sha", "reviewed_subject")}}
    return {"record_id": record_id, "error": "the record names no landed commit"}


def gate_identity(settled_rows: list, data_root: pathlib.Path) -> dict:
    """What the task's newest landing ``commit_reviewed`` produced, from its settled tool rows; ``{}`` when none landed."""
    landing = next((row for row in reversed(settled_rows) if str(row.get("tool") or "") == "commit_reviewed"
                    and str(row.get("result_preview") or "").startswith(_LANDED_PREFIX)), None)
    if landing is None:
        return {}
    meta = landing.get("tool_result_meta") if isinstance(landing.get("tool_result_meta"), dict) else {}
    receipt = meta.get("review_delivery") if isinstance(meta.get("review_delivery"), dict) else {}
    found = _REVIEW_RECORD_RE.search(str(landing.get("result_preview") or ""))
    record_id = str(receipt.get("record_id") or "") or (found.group(1) if found else "")
    return _record_landing(record_id, data_root) if record_id else {"error": "the landing names no review record"}


def conversation_gate_identity(text: str, data_root: pathlib.Path) -> dict:
    """The same answer for a scripted author, which reads the conversation: its newest landing result names the record."""
    at = text.rfind(_LANDED_PREFIX)
    found = _REVIEW_RECORD_RE.search(text, at) if at >= 0 else None
    return _record_landing(found.group(1), data_root) if found else {}


def restart_answer_in_conversation(text: str) -> tuple[bool, str]:
    """``(accepted, its line)``: the newest answer to ``request_restart`` after the newest landing result."""
    tail = text[max(text.rfind(_LANDED_PREFIX), 0):]
    refused, accepted = list(_RESTART_REFUSAL_RE.finditer(tail)), tail.rfind(_RESTART_ACCEPTED)
    if refused and refused[-1].start() > accepted:
        return False, tail[refused[-1].start():].splitlines()[0][:300]
    return (True, _RESTART_ACCEPTED) if accepted >= 0 else (False, "no answer to request_restart in the conversation")


def _candidate_rows(data_root: pathlib.Path) -> list[dict]:
    """Every body candidate record under any drive root of the lane."""
    rows = []
    for root in _drive_roots(data_root):
        for path in sorted((root / "state" / "private_copies").glob("*.json")):
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(row, dict) and row.get("kind") == "body_candidate":
                rows.append(row)
    return rows


def candidate_reviewed_heads(data_root: pathlib.Path) -> set[str]:
    return {str(row["reviewed_commits"][-1]) for row in _candidate_rows(data_root) if row.get("reviewed_commits")}


def candidate_fact(data_root: pathlib.Path, task_id: str) -> dict:
    """The task's own body candidate record, ``{}`` when it has none."""
    rows = [row for row in _candidate_rows(data_root) if row.get("task_id") == task_id]
    if not rows:
        return {}
    reviewed = rows[-1].get("reviewed_commits") or []
    return {"candidate_id": str(rows[-1].get("candidate_id") or ""), "branch": str(rows[-1].get("branch") or ""),
            "path": str(rows[-1].get("path") or ""), "base_sha": str(rows[-1].get("base_sha") or ""),
            "reviewed_commits": len(reviewed), "reviewed_head": str(reviewed[-1]) if reviewed else ""}


def names_commit(text: Any, sha: str) -> bool:
    """``text`` spells ``sha``: the whole SHA or an abbreviation of it, which ``request_restart`` resolves to that same
    commit (``git rev-parse``, at least git's four hex digits). A ref or another commit names something else."""
    text = str(text or "").strip().lower()
    return bool(sha) and len(text) >= 4 and all(char in "0123456789abcdef" for char in text) and sha.lower().startswith(text)


def _task_attempt(row: dict) -> int:
    return int(row.get("task_attempt") or 1)   # as the host keys a restart request: ``int(ctx.task_attempt or 1)``


def restart_request_fact(settled_rows: list) -> dict:
    """Infer the request in the newest restart-bearing attempt and its newest answer; a refusal keeps its typed code.

    ``control_runtime._request_restart`` sends one request per task attempt: the first call the host answers with
    acceptance or a receipt (``RESTART_<STATUS>: request <id>``) sent it, and every later call of that attempt only reads
    that request's receipt, its own ``reason`` and ``adopt_commit`` ignored. A call refused before anything was sent
    (``⚠️ RESTART_BLOCKED``) leaves the attempt free, and a new attempt holds only its own request (``owner_wait``). So
    ``adopt_commit`` is that first answered call's, never the newest call's nor any call that merely named a commit."""
    rows = [row for row in settled_rows if str(row.get("tool") or "") == "request_restart"]
    if not rows:
        return {"requested": False}
    attempt = max(map(_task_attempt, rows))
    mine = [row for row in rows if _task_attempt(row) == attempt]
    held = [row for row in mine if str(row.get("result_preview") or "").startswith(_RESTART_ACCEPTED)
            or _RESTART_RECEIPT_RE.search(str(row.get("result_preview") or ""))]
    request, answer = (held[0], held[-1]) if held else (mine[-1], mine[-1])
    preview = str(answer.get("result_preview") or "")
    args = request.get("args") if isinstance(request.get("args"), dict) else {}
    accepted = preview.startswith(_RESTART_ACCEPTED)
    refused = None if accepted else _RESTART_REFUSAL_RE.search(preview)
    return {"requested": True, "accepted": accepted, "adopt_commit": str(args.get("adopt_commit") or ""),
            "refusal": (refused.group(2) or refused.group(1)) if refused else "", "text": preview[:500],
            "task_attempt": attempt, "calls": len(mine)}


def handoff_fact(authorized: list, not_armed: list, unconfirmed: list, closed: list, sha: str) -> dict:
    """The newest handoff authorized for exactly ``sha`` and how it ended; another commit's handoff never speaks for it."""
    mine = [row for row in authorized if row.get("candidate") == sha]
    if not sha or not mine:
        return {"authorized": False}
    adoption_id = mine[-1].get("adoption_id")

    def of_it(rows: list) -> list:
        return [row for row in rows if row.get("adoption_id") == adoption_id and row.get("candidate") == sha]

    fact: dict[str, Any] = {"authorized": True, "adoption_id": adoption_id, "base": str(mine[-1].get("base") or ""),
                            "branch": str(mine[-1].get("branch") or ""),
                            "not_armed": [row.get("blockers") for row in of_it(not_armed)],
                            "unconfirmed": [str(row.get("reason") or "") for row in of_it(unconfirmed)]}
    ended = of_it(closed)
    if ended:
        fact.update(outcome=str(ended[-1].get("outcome") or ""), tests=ended[-1].get("tests") or {},
                    phases=[str(event.get("phase") or "") for event in ended[-1].get("events") or [] if isinstance(event, dict)])
    return fact


def restart_verify_fact(events: list, sha: str) -> dict:
    rows = [row for row in events if sha and row.get("expected_sha") == sha]
    return {key: rows[-1].get(key) for key in ("ok", "expected_sha", "observed_sha", "error") if key in rows[-1]} if rows else {}


def task_adoption_fact(task_result: dict, adoption_id: Any, sha: str) -> dict:
    """The adoption as the authorizing task's own record states it: what binds the archived handoff to this task. Its
    row can lag the handoff: an unconfirmed boot leaves an ``unconfirmed`` one (``body_adoption._waiting``) until the
    boot that adopts has archived and verified, and only then does ``_publish_on_task`` replace it."""
    from ouroboros.body_adoption import ADOPTIONS_KEY

    rows = [row for row in task_result.get(ADOPTIONS_KEY) or [] if isinstance(row, dict)
            and row.get("adoption_id") == adoption_id and row.get("commit_sha") == sha]
    return {key: rows[-1].get(key) for key in ("adoption_id", "outcome", "commit_sha", "previous_sha", "branch")} if rows else {}


def qualification_fact(tests: Any, sha: str, tree: str) -> dict:
    """The evidence ``adoption_test_evidence`` accepted, as observed, bound to this commit and tree. Selected or full and
    proof or receipt are what actually happened, recorded; the product's rules already decided that they qualify."""
    tests = tests if isinstance(tests, dict) else {}
    observed = {key: tests[key] for key in ("tier", "source", "head", "tree", "key", "phase", "base", "changed",
                                            "rule", "file_count", "omitted_files") if key in tests}
    return {**observed, "bound": bool(sha) and tests.get("head") == sha and bool(tree) and tests.get("tree") == tree
            and tests.get("tier") in ("full", "selected")}


def served_stylesheets(base_url: str, clone: pathlib.Path, sha: str, stylesheet: str, routes: tuple[str, ...]) -> list[dict]:
    """Each route's bytes as a client receives them, against the exact blob of the adopted commit."""
    expected = subprocess.run(["git", "show", f"{sha}:{stylesheet}"], cwd=str(clone), check=False, capture_output=True)
    rows = []
    for route in routes:
        fact: dict[str, Any] = {"route": route, "matches": False}
        try:
            with urllib.request.urlopen(base_url.rstrip("/") + route, timeout=30) as resp:  # noqa: S310 - the lane's loopback server
                body = resp.read()
            fact.update(status=int(resp.status), bytes=len(body), sha256=hashlib.sha256(body).hexdigest(),
                        matches=expected.returncode == 0 and bool(body) and body == expected.stdout)
        except (urllib.error.URLError, OSError) as exc:
            fact["error"] = f"{type(exc).__name__}: {exc}"[:300]
        rows.append(fact)
    return rows


def _serving_fact(clone: pathlib.Path, sha: str) -> dict:
    """The serving checkout: HEAD, branch, and clean only as ``git status`` proves it (a failed status prints nothing)."""
    head = _git(clone, "rev-parse", "HEAD")
    status = subprocess.run(["git", "status", "--porcelain"], cwd=str(clone), check=False, capture_output=True, text=True)
    entries = [line for line in (status.stdout or "").splitlines() if line.strip()]
    fact = {"head": head, "branch": _git(clone, "symbolic-ref", "-q", "--short", "HEAD"), "at_commit": bool(sha) and head == sha,
            "clean": status.returncode == 0 and not [line for line in entries if not line.split(None, 1)[-1].startswith(".ouroboros/")],
            "porcelain": entries[:20]}
    if status.returncode != 0:
        fact["status_error"] = f"git status exit {status.returncode}: {(status.stderr or '').strip()}"[:300]
    return fact


def _settlement_dead_end(handoff: dict, verify: dict, server: Any) -> str:
    """Why this adoption can no longer settle as asked, or ``""`` while it still can: read from what the product wrote.
    An unconfirmed boot is not one: the product keeps that handoff for the next start (``body_adoption._waiting``), whose
    ``adopted`` still settles it, so only the deadline names it."""
    if handoff.get("outcome") and handoff["outcome"] != "adopted":
        return f"adoption_closed:{handoff['outcome']}"
    if verify and not verify.get("ok"):
        return "restart_verify_failed"
    code = getattr(getattr(server, "proc", None), "poll", lambda: None)()
    return f"server_exited:{code}" if code not in (None, _RESTART_HANDOVER_EXIT) else ""


def _task_result(oracle: Any, task_id: str) -> dict:
    """The task's record, ``{}`` while its store cannot answer: the next generation may hold it during its boot."""
    try:
        return oracle.task_result(task_id)
    except Exception:  # noqa: BLE001 - read again on the next poll; the deadline bounds it
        return {}


def observe_direct_adoption(ctx: Any, task_id: str, settled_rows: list, gate: dict, *, timeout: float,
                            stylesheet: str, routes: tuple[str, ...], poll: float = 1.0) -> dict:
    """Everything observable about this task's adoption, called once its task is terminal. Waits only on durable
    files and tolerant HTTP, so the re-exec gap's refused connections are not a failure; a dead end ends the wait at
    once and ``timeout`` bounds the rest. ``reason`` is ``adopted`` or the first thing that stopped it."""
    sha = str(gate.get("commit_sha") or "")
    facts: dict[str, Any] = {"gate": dict(gate), "candidate": candidate_fact(ctx.data_root, task_id),
                             "request": restart_request_fact(settled_rows), "handoff": {"authorized": False},
                             "restart_verify": {}, "on_task": {}, "qualification": {}, "served": {}}
    tree = _git(ctx.clone, "rev-parse", "--verify", f"{sha}^{{tree}}") if sha else ""
    if not sha:
        facts["reason"] = "no_reviewed_commit"
    elif not facts["request"]["requested"]:
        facts["reason"] = "adoption_not_requested"
    elif not facts["request"]["accepted"]:
        facts["reason"] = f"restart_refused:{facts['request']['refusal'] or 'unconfirmed'}"
    elif not facts["request"]["adopt_commit"]:
        facts["reason"] = "restart_without_adoption"
    elif not names_commit(facts["request"]["adopt_commit"], sha):   # whatever settles next is not this request's
        facts["reason"] = "restart_for_other_commit"
    else:
        deadline = time.monotonic() + float(timeout)
        while True:
            oracle = ctx.oracle
            handoff = handoff_fact(*(oracle.supervisor_rows(f"body_adoption_{kind}") for kind in (
                "authorized", "not_armed", "unconfirmed", "closed")), sha)
            verify = restart_verify_fact(oracle.events("restart_verify"), sha)
            on_task = task_adoption_fact(_task_result(oracle, task_id), handoff.get("adoption_id"), sha) if handoff["authorized"] else {}
            facts.update(handoff=handoff, restart_verify=verify, on_task=on_task)
            why = _settlement_dead_end(handoff, verify, ctx.server)
            settled = handoff.get("outcome") == "adopted" and verify.get("ok") and on_task.get("outcome") == "adopted"
            if why or settled:
                facts["reason"] = why or "adopted"
                break
            if time.monotonic() >= deadline:
                stage = ("authorization" if not handoff["authorized"] else "settlement" if not handoff.get("outcome")
                         else "restart_verify" if not verify else "task_record")
                unconfirmed = handoff.get("unconfirmed") if stage == "settlement" else None
                facts["reason"] = f"adoption_unconfirmed:{unconfirmed[-1]}" if unconfirmed else f"timeout:{stage}"
                break
            time.sleep(poll)
        facts["qualification"] = qualification_fact(facts["handoff"].get("tests"), sha, tree)
        if facts["reason"] == "adopted":
            remaining = lambda: max(1.0, deadline - time.monotonic())  # noqa: E731
            healthy = bool(ctx.server.wait_for_health(timeout=remaining()))
            state_sha = ctx.h.wait_until(lambda: ctx.server.current_sha() == sha[:8], remaining())
            version = _git(ctx.clone, "show", f"{sha}:VERSION")
            try:
                attestation = runtime_attestation(ctx.server.base_url, ctx.clone, expected_version=version)
            except Exception as exc:  # noqa: BLE001 - it refuses by raising; a refusal is this check's answer
                attestation = {"ok": False, "refused": f"{type(exc).__name__}: {exc}"[:500]}
            facts["served"] = {"healthy": healthy, "state_sha": ctx.server.current_sha(), "state_sha_matches": bool(state_sha),
                               "attestation": {key: attestation.get(key) for key in (
                                   "ok", "refused", "reason", "runtime_version", "repo_version", "repo_head") if key in attestation},
                               "worker_booted_commit": any(row.get("git_sha") == sha for row in ctx.oracle.events("worker_boot")),
                               "stylesheets": served_stylesheets(ctx.server.base_url, ctx.clone, sha, stylesheet, routes)}
    facts["serving"] = _serving_fact(ctx.clone, sha)
    return facts


def adoption_checks(facts: dict) -> dict[str, bool]:
    """The verdicts of SM1's direct adoption, each from its own stage; every observation stays in ``facts``."""
    gate, candidate, request, handoff = facts["gate"], facts["candidate"], facts["request"], facts["handoff"]
    verify, on_task, served, serving = facts["restart_verify"], facts["on_task"], facts["served"], facts["serving"]
    sha = str(gate.get("commit_sha") or "")
    attestation = served.get("attestation") or {}
    stylesheets = served.get("stylesheets") or []
    return {
        "adoption_requested_for_reviewed_commit": bool(sha) and candidate.get("reviewed_head") == sha
        and request.get("accepted") is True and names_commit(request.get("adopt_commit"), sha) and handoff.get("authorized") is True,
        "adoption_qualified_by_real_tests": facts["qualification"].get("bound") is True,
        "adoption_settled_and_restart_verified": handoff.get("outcome") == "adopted" and verify.get("ok") is True
        and verify.get("observed_sha") == sha,
        "adoption_recorded_on_task": on_task.get("outcome") == "adopted" and on_task.get("commit_sha") == sha,
        "adoption_serves_commit": bool(served.get("healthy")) and served.get("state_sha_matches") is True
        and attestation.get("ok") is True and attestation.get("repo_head") == sha and served.get("worker_booted_commit") is True
        and serving.get("at_commit") is True and serving.get("branch") == handoff.get("branch") and serving.get("clean") is True,
        "adoption_serves_committed_stylesheets": bool(stylesheets) and all(row.get("matches") for row in stylesheets),
    }
