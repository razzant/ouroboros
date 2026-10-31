"""How my story shows an account I wrote across rooms, and the records it tells in their place.

An account (``chronicle_store.publish_account``) is my own text over exact source versions of
several rooms. It enters ``## My story`` only through my selection (``select_account``): the
acting selection says whether it is shown and which named records of the story it tells
instead (``told_by``). The resident block keeps its period, counts and one exact composition
reader instead of a row per source or replaced record; with a story, a replaced record leaves its
own room's page too (``memory_view._capture_room``; a story-less nanny keeps it), where one line
(``told_here``) names how many of the room's records accounts tell, and which ones; each stays one
``memory_read`` away. The verified quotes I chose to keep stand under the account's words, as a page's do.
New corrections and rejections stay whole beside the unchanged account until a newly authored account
cites their source versions. A record it tells whose version none of its sources cites (through nested
accounts and parts) keeps every signed correction and rejection of mine beside it (``told_fixes``); a
cited one's are inside the version cited or reported as later changes. Reporting them does not compare
the account with its cited versions for omissions; a later correction may still name an old omission,
and leaving out what a cited version says is my own choice. Mere correction or re-selection of an old
account does not update its frozen edges. The host reports these facts, draft decisions, folds and
gaps; it never rewrites meaning.

Only facts of the records, never a reason to read them. Nothing here reads a file.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional

from ouroboros import memory_inventory
from ouroboros.chronicle_store import ChronicleStore, draft_signer, source_time_span
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


def quote_line(quote: Mapping[str, Any]) -> str:
    """A verified quote under a page's, part's or account's words: who spoke, the row's address, the words."""
    return f"- quote ({quote.get('speaker')}, {quote.get('address')}): {quote.get('text')}"


def _told_fix_lines(entry: Mapping[str, Any]) -> List[str]:
    """My signed changes of the records the account tells whose version none of its sources cites."""
    fixes = entry.get("told_fixes") or ()
    if not fixes:
        return []
    lines = [f"- {len(fixes)} signed change{'' if len(fixes) == 1 else 's'} of mine to records it tells whose "
             "version none of its sources cites, oldest first:"]
    for fix in fixes:
        verb = "correction by me of" if fix["kind"] == "correction" else "my rejection of the draft"
        lines.append(f"- {verb} {fix['target_kind']} {fix['target']} (room {fix['room_id']}, {_date(fix.get('ts'))}):\n"
                     + indented(str(fix.get("text") or fix.get("reason") or "")))
    return lines


def account_lines(entry: Mapping[str, Any]) -> List[str]:
    """My words and kept quotes, a bounded composition pointer, then my changes of what it tells and of its sources."""
    count = len(entry.get("sources") or ())
    return [indented(entry["text"]), *map(quote_line, entry.get("quotes") or ()),
            f"- written {entry.get('written') or 'date not recorded'} by me ({entry.get('signer')}) from {count} "
            f"source{'' if count == 1 else 's'}; their rows span the period above, not every event in it",
            f"- exact composition, source versions and selections: memory_read(node_id='{entry['id']}'); "
            f"shown by selection {entry['selection']}",
            *_told_fix_lines(entry), *_source_lines(entry)]


def accounts_line(status: Mapping[str, Any]) -> str:
    """The story's line about my accounts, present from the first one on (an install without any says nothing)."""
    shown = status.get("accounts_shown", 0)
    return (f"My accounts across rooms: {status.get('accounts', 0)} written, {shown} in the common view"
            f"{'' if shown == status.get('accounts', 0) else ' (the rest one memory_read away)'}.")


def account_entries(store: ChronicleStore, label: Callable[..., str], units: Mapping[str, memory_inventory.LegacyUnit],
                    fixes: Mapping[str, List[Dict[str, Any]]],
                    period_text: Callable[[memory_inventory.Period], str]) -> List[Dict[str, Any]]:
    """Every account shown by its acting selection, as a story entry placed like a part (``first``, sequence).

    ``fixes`` is the story's map of corrections and rejections by target (``memory_view._story_pages``);
    a source's later corrections are read from it by id, so the words shown are the correction's.
    ``period_text`` prints a ``Period`` the way the story prints every record's. An account no shown
    account tells carries ``told_fixes``: the fixes of what it tells (``told_records``) outside ``cited_records``.
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
                        "signer": draft_signer(record.get("author")), "stamp": "", "fixes": [],
                        "quotes": list(record.get("quotes") or ()),
                        "written": _date(record.get("ts")), "sources": sources, "replaces": list(selection.get("replaces") or ()),
                        "selection": selection.get("id"), "selection_sequence": selection["sequence"],
                        "sequence": record["sequence"]})
    told = told_map(entries)
    for entry in (entry for entry in entries if entry["id"] not in told):
        cited, entry["told_fixes"] = cited_records(store, entry["sources"], fixes), []
        for ident in told_records(store, entry["id"], told):
            found = None if ident in cited else store.get(ident)
            entry["told_fixes"] += [{**fix, "target": ident, "target_kind": found["kind"], "room_id": str(found["room_id"])}
                                    for fix in (fixes.get(ident, ()) if found else ())]
        entry["told_fixes"].sort(key=lambda fix: fix["sequence"])
    return entries


def fix_map(store: ChronicleStore) -> Dict[str, List[Dict[str, Any]]]:
    """My corrections and rejections of drafts by target, oldest first: the story's ``fixes``."""
    fixes: Dict[str, List[Dict[str, Any]]] = {}
    for record in store.records(kinds=("correction", "decision")):
        if record["kind"] == "correction" or record.get("accepted") is False:
            fixes.setdefault(str(record.get("target_id") or ""), []).append(
                {"kind": "correction" if record["kind"] == "correction" else "rejection", "id": record["id"],
                 "text": record.get("text"), "reason": record.get("reason"), "ts": record.get("ts"),
                 "sequence": record["sequence"]})
    return fixes


def selection_facts(store: ChronicleStore, account_id: str) -> Dict[str, Any]:
    """What an account tells as one capture reads it: ``told_records`` over every shown account's acting selection
    (``named``: by its own; ``teller``: who tells the account), what of it no version its sources cite holds
    (``cited_records``, as ``told_fixes``) with my changes of those, and their rooms. These are teller relations
    as they stand when read, never what a write displaced nor what any view shows."""
    accounts = {record["id"]: record for record in store.interpreted(kinds=("account",), sources=False)}
    told = told_map([{"id": ident, "replaces": [str(r) for r in record["selection"].get("replaces") or ()],
                      "selection_sequence": record["selection"]["sequence"]}
                     for ident, record in accounts.items() if (record.get("selection") or {}).get("shown")])
    now, fixes = told_records(store, account_id, told), fix_map(store)
    cited = cited_records(store, accounts[account_id].get("sources"), fixes)
    return {"tells": now, "named": [ident for ident in now if told.get(ident) == account_id], "teller": told.get(account_id),
            "uncited": [ident for ident in now if ident not in cited],
            "uncited_changes": sum(len(fixes.get(ident, ())) for ident in now if ident not in cited),
            "rooms": {ident: str((store.get(ident) or {}).get("room_id")) for ident in now}}


def told_records(store: ChronicleStore, account_id: str, told: Mapping[str, str]) -> List[str]:
    """Every record an account tells in the common view: those its selection names, through a told account those
    it tells in turn, and a told part's folded members; each once."""
    children: Dict[str, List[str]] = {}
    for ident, teller in told.items():
        children.setdefault(teller, []).append(ident)
    out, todo = [], [account_id]
    while todo:
        for ident in children.get(todo.pop(0), ()):
            if ident not in out and ident != account_id:
                out.append(ident)
                todo.append(ident)
                out += [member for member in store.folded_members(ident) if member not in out]
    return out


def cited_records(store: ChronicleStore, sources: Any, fixes: Mapping[str, List[Dict[str, Any]]]) -> set:
    """The records an account's sources cite at a version this chronicle holds, through nested accounts' frozen
    edges and parts' members (``ChronicleStore._nested_source_changes`` walks the same graph). A correction of one
    is inside the version cited or reported beside the account as a later change; a member, whose revision a
    part does not record, has every change reported."""
    cited, todo, expanded = set(), [(str(src.get("id")), src.get("revision"), True) for src in sources or ()], set()
    while todo:
        ident, revision, edge = todo.pop()
        record = store.get(ident)
        if record is None:
            continue
        revisions = [ident, *(fix["id"] for fix in fixes.get(ident, ()) if fix["kind"] == "correction")]
        if not edge or revision in revisions:
            cited.add(ident)
        if ident not in expanded:
            expanded.add(ident)
            covers = record.get("covers") if isinstance(record.get("covers"), dict) else {}
            todo += ([(str(ref.get("id")), ref.get("revision"), True) for ref in record.get("sources") or ()]
                     if record["kind"] == "account" else
                     [(str(member), None, False) for member in covers.get("member_ids") or ()] if record["kind"] == "part" else [])
    return cited


def told_here(story: List[Mapping[str, Any]], rooms: Any) -> List[Dict[str, Any]]:
    """Per room of ``rooms``, the story records of that room an account tells: their count, combined span and the
    accounts standing in my story that tell them (through accounts they tell in turn); the records stay at readers."""
    by_id, groups = {entry["id"]: entry for entry in story}, {}
    for entry in story:
        if not entry.get("told_by") or str(entry.get("room_id")) not in rooms:
            continue
        chain = [entry["told_by"]]
        while by_id.get(chain[-1], {}).get("told_by") and by_id[chain[-1]]["told_by"] not in chain:
            chain.append(by_id[chain[-1]]["told_by"])
        group = groups.setdefault(str(entry["room_id"]), {"room_id": str(entry["room_id"]), "accounts": [], "spans": []})
        group["accounts"] += [] if chain[-1] in group["accounts"] else [chain[-1]]
        group["spans"].append(entry.get("source_period") or {})
    return [{"room_id": group["room_id"], "accounts": group["accounts"], "count": len(group["spans"]),
             "span": source_time_span([s[key] for s in group["spans"] for key in ("start", "end") if s.get(key)],
                                      incomplete=any(not s or s.get("incomplete") for s in group["spans"]))}
            for group in groups.values()]


def told_by(store: ChronicleStore, ids: List[str]) -> Dict[str, Optional[str]]:
    """``record id -> account`` that tells it in the common view: ``told_map``, the story's own precedence, over
    only the acting selections that bear on each record (``ChronicleStore.telling_selections``, indexed)."""
    return {ident: told_map(entries).get(ident) for ident, entries in store.telling_selections(ids).items()}


def told_map(entries: List[Dict[str, Any]]) -> Dict[str, str]:
    """``record id -> account id`` for every record a shown account tells in its place; the latest selection wins."""
    told: Dict[str, str] = {}
    # Re-selecting an older account is a new choice; its place in the story stays chronological.
    for entry in sorted(entries, key=lambda item: item["selection_sequence"]):
        told.pop(entry["id"], None)  # This later shown choice supersedes an older replacement of it.
        for ident in entry.get("replaces") or ():
            told[str(ident)] = entry["id"]
    return told
