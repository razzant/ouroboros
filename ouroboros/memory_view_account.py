"""How my story shows an account I wrote across rooms, and the records it tells in their place.

An account (``chronicle_store.publish_account``) is my own text over exact source versions of
several rooms. It enters ``## My story`` only through my selection (``select_account``): the
acting selection says whether it is shown and which named records of the story it tells
instead (``told_by``). The resident block keeps its period, counts and one exact composition
reader instead of a row per source or replaced record; the room keeps its detail
(``memory_view._capture_room``). New corrections and rejections stay whole beside the
unchanged account until a newly authored account cites their source versions. Mere
correction or re-selection of an old account does not update its frozen edges. The host
reports these facts, draft decisions, folds and gaps; it never rewrites meaning.

Only facts of the records, never a reason to read them. Nothing here reads a file.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping

from ouroboros import memory_inventory
from ouroboros.chronicle_store import ChronicleStore, draft_signer
from ouroboros.memory_view_legacy import indented


def _date(value: Any) -> str:
    return str(value or "")[:10] or "date not recorded"


def source_change_lines(src: Mapping[str, Any]) -> List[str]:
    """Signed changes below a cited source; shared by the story and exact account reader.

    These facts do not rewrite any narrative. A part has no recorded member revision, unlike an
    account's frozen source edge, so its member changes carry no invented incorporation claim.
    """
    lines = []
    for change in src.get("nested_changes") or ():
        event, ident = change["event"], change["source_id"]
        revision = change.get("revision")
        basis = (f"cited revision {revision}; later change not in that source account" if revision else
                 "part member; member revision was not recorded")
        kind = event["kind"] if event["kind"] == "correction" else "acceptance" if event["accepted"] else "rejection"
        lines.append(f"- {kind} of nested source {change['kind']} {ident} (room {change['room_id']}) "
                     f"via {' -> '.join(change['via'])}; {basis}; original account text unchanged")
        lines.append(f"  {event['kind']} {event['id']} by mind ({draft_signer(event.get('author'))}) "
                     f"at {event.get('ts') or 'time not recorded'}; memory_read(node_id='{event['id']}')")
        lines.append(indented(str(event.get("text") if event["kind"] == "correction" else event.get("reason") or "")))
        args = f", revision='{revision}'" if revision else ""
        lines.append(f"  source: memory_read(node_id='{ident}'{args})")
    return lines


def _source_lines(entry: Mapping[str, Any]) -> List[str]:
    """Only changed or unavailable source facts; unchanged composition lives in the account reader."""
    lines = []
    for src in entry.get("sources") or ():
        where = src.get("room_label") or f"room {src.get('room_id')}"
        status = f", a {src['status']} then" if src.get("status") and src.get("status") != "final" else ""
        head = f"- source: {src.get('kind')} {src.get('id')} of {where}, revision {src.get('revision')}{status}"
        if src.get("missing"):
            lines.append(f"{head}; not in this chronicle: its words are not here")
            continue
        then, now = src.get("status"), src.get("status_now")
        if (not src.get("revision_known") or src.get("later_fixes") or then != now
                or src.get("folded_into") or src.get("nested_changes")):
            lines.append(f"{head}; memory_read(node_id='{src['id']}', revision='{src['revision']}')")
        if not src.get("revision_known"):
            lines.append(f"  that revision of {src['id']} is not in this chronicle; its current one is {src.get('current_revision')}")
        for fix in src.get("later_fixes") or ():
            lines.append(f"- later correction of its source {src['id']} (not in this account):\n{indented(fix)}")
        if now == "rejected" and then != "rejected":
            lines.append(f"- my later rejection of its source {src['id']} (not in this account):\n"
                         + indented(src.get("rejection") or "reason not read"))
        elif then == "draft" and now == "accepted":
            lines.append(f"- its source {src['id']}, a draft then, I have since accepted")
        if src.get("folded_into"):
            lines.append(f"- its source {src['id']} has since been folded into part {src['folded_into']} (not in this account)")
        lines.extend(source_change_lines(src))
    return lines


def account_lines(entry: Mapping[str, Any]) -> List[str]:
    """My words and a bounded composition pointer, followed by pending source changes in full."""
    count = len(entry.get("sources") or ())
    return [indented(entry["text"]),
            f"- written {entry.get('written') or 'date not recorded'} by me ({entry.get('signer')}) from {count} "
            f"source{'' if count == 1 else 's'}; their rows span the period above, not every event in it",
            f"- exact composition, source versions and selections: memory_read(node_id='{entry['id']}'); "
            f"shown by selection {entry['selection']}",
            *_source_lines(entry)]


def accounts_line(status: Mapping[str, Any]) -> str:
    """The story's line about my accounts, present from the first one on (an install without any says nothing)."""
    shown = status.get("accounts_shown", 0)
    return (f"My accounts across rooms: {status.get('accounts', 0)} written, {shown} in the common view"
            f"{'' if shown == status.get('accounts', 0) else ' (the rest one memory_read away)'}.")


def account_entries(store: ChronicleStore, label: Callable[..., str], units: Mapping[str, memory_inventory.LegacyUnit],
                    fixes: Mapping[str, List[Dict[str, Any]]],
                    period_text: Callable[[memory_inventory.Period], str]) -> List[Dict[str, Any]]:
    """Every account shown by its acting selection, as a story entry placed like a page (``first``, sequence).

    ``fixes`` is the story's map of corrections and rejections by target (``memory_view._story_pages``);
    a source's later corrections are read from it by id, so the words shown are the correction's.
    ``period_text`` prints a ``Period`` the way the story prints every record's.
    """
    entries = []
    for record in store.accounts():
        selection = record.get("selection") or {}
        if not selection.get("shown"):
            continue
        period = memory_inventory.record_period(store, record, units)
        sources = []
        for src in record.get("sources") or ():
            later = set(src.get("later_corrections") or ())
            own = fixes.get(str(src.get("id")), ())
            sources.append({**src, "room_label": label(src.get("room_id")),
                            "later_fixes": [f["text"] for f in own if f["kind"] == "correction" and f.get("id") in later],
                            "rejection": next((f["reason"] for f in own if f["kind"] == "rejection"), "")})
        entries.append({"kind": "account", "id": record["id"], "room_id": str(record["room_id"]), "label": "My account across rooms",
                        "period": period_text(period), "source_period": period.span, "period_basis": period.source,
                        "first": period.first,
                        "text": str(record.get("current_text") or ""), "status": "",
                        "signer": draft_signer(record.get("author")), "stamp": "", "fixes": [], "quotes": [],
                        "written": _date(record.get("ts")), "sources": sources, "replaces": list(selection.get("replaces") or ()),
                        "selection": selection.get("id"), "selection_sequence": selection["sequence"],
                        "sequence": record["sequence"]})
    return entries


def told_map(entries: List[Dict[str, Any]]) -> Dict[str, str]:
    """``record id -> account id`` for every record a shown account tells in its place; the latest selection wins."""
    told: Dict[str, str] = {}
    # Re-selecting an older account is a new choice; its place in the story stays chronological.
    for entry in sorted(entries, key=lambda item: item["selection_sequence"]):
        told.pop(entry["id"], None)  # This later shown choice supersedes an older replacement of it.
        for ident in entry.get("replaces") or ():
            told[str(ident)] = entry["id"]
    return told
