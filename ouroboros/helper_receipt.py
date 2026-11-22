"""What reached a helper: one receipt of one shape for every kind of helper (owner, 21.11.2026, item 50).

A native child, a configured session under its nanny and a direct ``delegate_start`` session all answer the
delegating mind with the same receipt: first what did not arrive, differs from the request or is not known
yet, then one line per standing of fact, never claiming more than the host saw: ``requested`` (what the call
asked), ``prepared`` (what the host made ready before queueing or sending: retained copies, a frozen range, the
text sent) and ``observed`` (seen after the start). What is not known yet says so; an attempt without a record
says "no record". The differences are facts, not advice: what they mean for the work is the mind's judgment.

The receipt is rendered, never stored: ``build`` reads the call's own facts and that attempt's manifest, and
nothing here enters a child's work order or its fingerprint. ``result_line`` repeats only the differences
where the delegating work judges the result (``[SUBTASK_OUTCOME]``, the compact ``wait_tasks`` row, the
terminal ``delegate_wait`` payload). No call rebuilds the full receipt later, so that line names no pointer.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, NamedTuple, Optional

REQUESTED, PREPARED, OBSERVED = "requested", "prepared", "observed"
PENDING = "pending engine evidence"  # the engine reports applied access after a run, or never
NO_RECORD = "no record"
# My human's later words reach a run as frames (``delegate_message_store``); this outcome is the ordinary one.
ORDINARY_DELIVERY = "delivered"


class Receipt(NamedTuple):
    differences: List[str]  # each ``"<fact> [<stage>]"``
    summary: List[tuple]  # ``(stage, "subject: text")``


def _plural(count: Any, word: str, many: str = "") -> str:
    return f"{count} " + (word if count == 1 else many or word + "s")


def _outside(count: int, past: bool = False, when: str = "") -> str:
    verb = ("lay" if past else "lies" if count == 1 else "lie")
    return (f"{_plural(count, 'retained input copy', 'retained input copies')} {verb} outside the session's folder"
            f"{when}; reading {'it' if count == 1 else 'them'} is not confirmed [{PREPARED}]")


def _range_changes(rows: List[Mapping[str, Any]], spec: Mapping[str, Any]) -> str:
    """The changed-file count of a frozen range, from the host's retained ``range.json`` only when its bytes still
    match the size and sha256 its preparer recorded (one ``changes`` record per line, as ``delegate_git_range``
    writes it, counted while the digest is taken); otherwise unknown, never zero."""
    import hashlib

    for row in rows:
        copy = row.get("retained") or row.get("local")
        if row.get("git_range") == spec and row.get("path") == "range.json" and copy:
            digest, size, count = hashlib.sha256(), 0, 0
            try:
                with open(copy, "rb") as manifest:
                    for item in manifest:
                        digest.update(item)
                        size += len(item)
                        count += item.startswith(b'{"status"')
            except OSError:
                break
            if size == row.get("size") and digest.hexdigest() == row.get("sha256"):
                return _plural(count, "changed file")
            break
    return "changed files unknown"


def inputs_text(rows: List[Mapping[str, Any]]) -> str:
    """Count and bytes of selected copies; each frozen range by base, head, changed files and patch size."""
    plain = [row for row in rows if not row.get("git_range")]
    parts = [f"{_plural(len(plain), 'file')}, {sum(int(row.get('size') or 0) for row in plain)} bytes"] if plain else []
    for row in rows:
        spec = row.get("git_range")
        if spec and row.get("path") == "range.patch":
            scope = f" {spec['path']}" if spec.get("path") else ""
            parts.append(f"range {str(spec.get('base'))[:12]}..{str(spec.get('head'))[:12]}{scope}: "
                         f"{_range_changes(rows, spec)}, patch {int(row.get('size') or 0)} bytes")
    return ", ".join(parts) or "none selected ([])"


def memory_view_differences(view: Any, role: str, *, session_leaf: bool) -> List[str]:
    """A ``memory_view`` the view does not know, named with the role default that is used instead
    (``memory_view.view_spec_for_task``); a session leaf never receives one: its start composes its own selection
    (``subagent_work_order.session_memory_references``)."""
    if view is None:
        return []
    from ouroboros.memory_view import _explicit_problem

    out = []
    if problem := _explicit_problem(view):
        out.append(f"memory_view {view!r} not used ({problem}): the {role} default stands [{REQUESTED}]")
    if session_leaf and view:
        out.append("memory_view shapes only the nanny's own view; the session leaf's start composes its own "
                   f"[{REQUESTED}]")
    return out


def _memory_text(view: Any, role: str) -> str:
    from ouroboros.memory_view import _explicit_problem

    if not view or not isinstance(view, Mapping) or _explicit_problem(view):
        return f"{role} default"
    return f"{role} default with " + ", ".join(f"{key}={value}" for key, value in sorted(view.items()))


def owner_words_text(task: Mapping[str, Any], *, session: bool) -> str:
    """How many rows of my human's words the work carries by value: never a reply target (another author's words)
    nor a press or card input without words (their act, carried as such). A session's work order carries its rows,
    one per identity
    (``subagent_work_order.work_order_owner_rows``); a native child reads its room's, origin and marked rows at
    its own start, so those are named as read there."""
    from ouroboros.owner_words import OWNER_CARD_INPUT, OWNER_PRESS, REPLY_TARGET, task_governing_words
    from ouroboros.subagent_work_order import work_order_owner_rows

    carried = work_order_owner_rows(task) if session else (task_governing_words(task)[0], "", True)
    rows = [row for row in (carried or ([],))[0] if isinstance(row, Mapping)
            and row.get("source") not in (REPLY_TARGET, OWNER_PRESS, OWNER_CARD_INPUT)]
    told = (_plural(len(rows), "row") + " by value") if rows else (
        "none by value" if carried is not None else "none: declared inputs omit them")
    return told + ("" if session else "; room, origin and marked rows read at its start")


def build(facts: Mapping[str, Any]) -> Receipt:
    """The one receipt builder, from facts the caller already holds (each optional): ``access_ignored``
    (``(requested, row)``: an API child's ignored ``access``), ``access_lowering`` (``(asked, row)``: a session
    row's access a call could not widen), ``inputs_map``, ``session`` (whose work order carries the words),
    ``folder``, ``write_root``, ``access`` (``(subject, text, applied)`` grants; ``applied`` None is not known
    yet, False says nothing about it), ``inputs`` (prepared rows; None: nothing selected), ``inputs_outside``
    (how many of them lie outside a session's folder), ``inputs_when``, ``inputs_note``, ``lineage`` (automatic
    copies: ``(count, bytes)``), ``memory_view`` (``view``, ``role``, ``session_leaf``), ``owner_words`` (the
    record's fields, or a measured line) and ``assignment`` (``(text, stage)``: what was measured)."""
    differences: List[str] = []
    if ignored := facts.get("access_ignored"):
        differences.append(f"access={ignored[0]!r} ignored for subagent_id={ignored[1]!r} (api_model); "
                           f"write_surface controls read/write authority [{REQUESTED}]")
    if lowering := facts.get("access_lowering"):
        differences.append(f"access={lowering[0]!r} asked; the row's {lowering[1]} stands (access only lowers) "
                           f"[{REQUESTED}]")
    rows = facts.get("inputs")
    if rows and facts.get("inputs_outside"):
        differences.append(_outside(int(facts["inputs_outside"])))
    view = facts.get("memory_view") or {}
    if view:
        differences += memory_view_differences(view.get("view"), view["role"],
                                               session_leaf=bool(view.get("session_leaf")))
    summary = [(REQUESTED, "folder: " + facts["folder"])] if facts.get("folder") else []
    if facts.get("write_root"):
        summary.append((REQUESTED, "write root: " + facts["write_root"]))
    for subject, text, applied in facts.get("access") or ():
        shown = "" if applied is False else f" (applied: {applied or PENDING})"
        summary.append((REQUESTED, f"{subject}: {text}{shown}"))
    if rows == [] and isinstance(facts.get("lineage"), tuple):
        differences.append(f"input_files [] selected nothing; automatic lineage was copied instead [{PREPARED}]")
    if rows is not None:
        where = f", map {facts['inputs_map']}" if facts.get("inputs_map") else ""
        when = facts.get("inputs_when", "retained before queueing")
        summary.append((PREPARED, f"inputs: {inputs_text(rows)}, {when} (prepared is not read){where}"))
    if facts.get("inputs_note"):
        summary.append((REQUESTED, "inputs: " + ("" if rows is not None else "none selected. ") + facts["inputs_note"]))
    if lineage := facts.get("lineage"):
        counted = NO_RECORD if isinstance(lineage, str) else f"{_plural(lineage[0], 'file')}, {lineage[1]} bytes"
        summary.append((PREPARED, "inputs: automatic lineage, " + counted))
    if view:
        summary.append((REQUESTED, "memory view: " + _memory_text(view.get("view"), view["role"])))
    if "owner_words" in facts:
        words = facts["owner_words"]
        told = words if isinstance(words, str) else owner_words_text(words or {}, session=bool(facts.get("session")))
        summary.append((PREPARED, "my human's words: " + told))
    if assignment := facts.get("assignment"):
        summary.append((assignment[1], f"{assignment[2] if len(assignment) > 2 else 'assignment'}: {assignment[0]}"))
    return Receipt(differences, summary)


def render(receipt: Receipt) -> str:
    """The differences first, when any is known (no line claims there is none), then one line per standing of
    fact; "" for a receipt with nothing to say. A refusal carries none: its cause and rows are its answer."""
    if not (receipt.differences or receipt.summary):
        return ""
    head = ["receipt:", *receipt.differences]
    stages = [stage for stage in (REQUESTED, PREPARED, OBSERVED) if any(item[0] == stage for item in receipt.summary)]
    return "\n".join([*head, *(f"{stage}: " + "; ".join(text for kind, text in receipt.summary if kind == stage)
                               for stage in stages)])


def start_facts(request: Mapping[str, Any], lineage: Any = None) -> Dict[str, Any]:
    """What one ``delegate_start`` sent (the prompt and host instructions it measured) and the automatic lineage
    copies its read-only scope received (``ReadonlyInputs.lineage``: count and bytes as the copies were made;
    ``NO_RECORD`` for a replay that holds no such fact). My human's rows the host appends are not counted here:
    nothing ties a count to this invocation's own tail cheaply, and the prompt's own stamps are not the host's."""
    prompt, instructions = str(request.get("prompt") or ""), str(request.get("instructions") or "")
    facts: Dict[str, Any] = {"assignment": (f"prompt {len(prompt)} + host instructions {len(instructions)} chars",
                                            PREPARED, "sent")}
    if lineage:
        facts["lineage"] = lineage
    return facts


def result_line(differences: List[str]) -> str:
    """The one-line repeat where the delegating work judges the result: the differences and gaps observed, ""
    when none is (the caller then adds no line: nothing here assures). It restates no ``capability_delta`` and
    points at no call: none rebuilds the full receipt."""
    return "; ".join(differences)


def task_result_line(data: Mapping[str, Any]) -> str:
    """The repeat for a child task's result, from its saved fields only (never a live read): its scheduling-time
    facts as such (an API child's ignored ``access``, ``memory_view``, copies placed outside the session's folder;
    a later start may replace that selection). Nothing about its runs' applied access: one task's runs may each
    request differently (a plan turn, then its implementation), and the task record cannot attribute them; each
    run's own ``delegate_wait`` line compares its request with what it got (``run_result_line``)."""
    configured = data.get("configured_subagent") if isinstance(data.get("configured_subagent"), Mapping) else {}
    session = str((configured.get("route") or {}).get("kind") or "") == "agent_session"
    differences = memory_view_differences(data.get("memory_view"), "nanny" if session else "child",
                                          session_leaf=session)
    asked = data.get("requested_access")
    if not session and configured and asked is not None:
        differences.append(f"access={asked!r} ignored (api_model); write_surface controls read/write authority "
                           f"[{REQUESTED}]")
    row_access = str(configured.get("access") or "")
    if session and asked not in (None, "inherit", row_access) and row_access:
        differences.append(f"access={asked!r} asked; the row's {row_access} stood (access only lowers) [{REQUESTED}]")
    rows = data.get("prepared_input_files")
    intent = data.get("resource_intent") if isinstance(data.get("resource_intent"), Mapping) else {}
    if session and rows and intent.get("kind") != "explicit_none":
        differences.append(_outside(len(rows), past=True, when=" at scheduling"))
    return result_line(differences)


def run_result_line(ctx: Any, entry: Any, access_evidence: Optional[Mapping[str, Any]], *,
                    looked_up: bool = True) -> str:
    """The repeat on a finished run's terminal ``delegate_wait`` payload, from that run's custody rows: the applied
    access against what THIS run's own recorded request asked, copies outside its folder, and my human's later
    words whose last outcome is not ``delivered`` (``accepted`` is taken by the engine, not read). ``looked_up``
    False: the custody lookup itself failed, which proves nothing about the record."""
    if entry is None:
        return f"no custody record of this run [{OBSERVED}]" if looked_up else (
            f"this run's custody record was unavailable at this read [{OBSERVED}]")
    differences: List[str] = []
    evidence = access_evidence or {}
    effective = str(evidence.get("effective") or "")
    request = (getattr(entry, "request_event", None) or {}).get("request")
    # Only what this run recorded: its request body, else its own STARTED access. An older record holds neither, and
    # the evidence's ``requested`` may then be today's derivation, which nobody asked of this run.
    requested = str((request.get("access") if isinstance(request, Mapping) else "") or getattr(entry, "access", "")
                    or "")
    if not effective:
        differences.append(f"access applied: not disclosed by the engine (requested {requested or 'not on record'}) "
                           f"[{OBSERVED}]")
    elif not requested:
        differences.append(f"access applied {effective}; this run's original request is not on record [{OBSERVED}]")
    elif effective != requested:
        differences.append(f"access applied {effective}, requested {requested} [{OBSERVED}]")
    rows = (getattr(entry, "request_event", None) or {}).get("input_files") or []
    if outside := sum(1 for row in rows if isinstance(row, Mapping) and row.get("carrier") == "outside_scope_path"):
        differences.append(_outside(outside, past=True))
    try:
        from ouroboros.delegate_message_store import placed_words

        last = {str(row["frame"]["owner_words"].get("address") or ""): str(row.get("outcome") or "")
                for row in placed_words(ctx, entry.run_id)}
    except Exception:  # noqa: BLE001 — an unreadable journal is named, never read as delivered
        differences.append(f"my human's later words: delivery records unreadable [{OBSERVED}]")
        last = {}
    unusual = sorted(outcome for outcome in last.values() if outcome != ORDINARY_DELIVERY)
    if unusual:
        differences.append("my human's later words: " + ", ".join(
            f"{unusual.count(outcome)} {outcome}"
            + (" (taken by the engine, not read)" if outcome == "accepted" else "")
            for outcome in sorted(set(unusual))) + f" [{OBSERVED}]")
    return result_line(differences)
