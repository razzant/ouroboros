"""Ordinary host facts for place rooms with arrivals but no recorded conversation.

Authorship comes from connector facts, never a provider name. Only the small
source openings used by event forms survive capture; readers retain the words.
"""
from ouroboros import chat_chain
from ouroboros.dialogue_provenance import recorded_person, recorded_speaker


def starts_conversation(author):
    """A recorded living person or an outward reply; an own-account arrival is neither."""
    return (recorded_person(author) and not author.get("is_bot") or
            author.get("kind") == "ouroboros" and author.get("via") != "own_account")


def capture(spoken):
    """Select event samples; arrivals without a recorded author are counted here too (owner choice A)."""
    arrivals = [(entry, cls["author"]) for entry, cls in spoken if cls["lane"] == 1 and entry[1].get("direction") == "in"]
    known = [(e, a) for e, a in arrivals if recorded_speaker(e[1])]
    unknown = [(e, a) for e, a in arrivals if not recorded_speaker(e[1])]
    texts = [(e, a) for e, a in arrivals if e[1].get("text_chars")]
    samples = {"first": min(texts, key=lambda p: p[0][2], default=None),
               "last": max(known, key=lambda p: p[0][2], default=None),
               "unknown_first": min(unknown, key=lambda p: p[0][2], default=None)}
    return {"count": len(arrivals), "unknown_count": len(unknown),
            "unknown_chars": sum(int(e[1].get("text_chars") or 0) for e, _a in unknown),
            "unknown_last": str(max(unknown, key=lambda p: p[0][2])[0][1].get("ts") or "") if unknown else "",
            "_samples": samples}


def text_entries(event):
    return [pair[0] for pair in event["_samples"].values() if pair is not None]


def finish_capture(event, texts):
    """Replace transient metadata references with the scalar facts printed by the forms."""
    for name, pair in event.pop("_samples").items():
        if pair is None:
            event[name] = None
            continue
        entry, author = pair
        row = texts.get(entry[0]["row_sha256"], {})
        text = str(row.get("text") or "").strip()
        label = ("author not recorded" if author.get("unattributed") else author.get("label", "author not recorded"))
        if author.get("via") == "own_account":
            label += " (account not confirmed as my human's)"
        event[name] = {"address": chat_chain.format_address(entry[0]), "pos": entry[2],
                       "ts": str(entry[1].get("ts") or ""), "author": label,
                       "opening": " ".join(text.splitlines()[:1])[:160]}


def _sample(sample, *, opening=False):
    if sample is None:
        return "author not recorded"
    return f"[{sample['address']}; {sample['author']}]" + (f" «{sample['opening']}»" if opening else "")


def _unknown(event):
    if not event["unknown_count"]:
        return ""
    # Keep this one line even when the source's first 160 characters contain newlines.
    opening = " ".join((event["unknown_first"] or {}).get("opening", "").split())
    return (f"; author not recorded: {event['unknown_count']} arrivals, {event['unknown_chars']} chars, "
            f"last {event['unknown_last']}; earliest opening «{opening}»")


def room_line(room):
    from ouroboros.memory_view import _minute
    event = room["events"]
    if not event["count"]:
        last = f"; last {_minute(latest)}" if (latest := room["facts"].get("latest")) else ""
        return f"- {room['label']}; no arrivals{last}; memory_read(room_id='{room['room_id']}', rows=true)"
    return (f"- {room['label']}; Event: {event['count']} arrivals; first {_sample(event['first'], opening=True)}; "
            f"last {_sample(event['last'])}{_unknown(event)}; memory_read(room_id='{room['room_id']}', rows=true)")


def totals(rooms):
    events = [r["events"] for r in rooms if r.get("events")]
    return len(events), sum(e["count"] for e in events)


def count_line(group):
    events = [r["events"] for r in group["rooms"] if r.get("events")]
    if not events:
        return ""
    known = [e["last"] for e in events if e["last"]]
    unknown = [e["unknown_first"] for e in events if e["unknown_first"]]
    combined = {"unknown_count": sum(e["unknown_count"] for e in events),
                "unknown_chars": sum(e["unknown_chars"] for e in events),
                "unknown_last": max(e["unknown_last"] for e in events),
                "unknown_first": min(unknown, key=lambda s: s["pos"], default=None)}
    last = max(known, key=lambda s: s["pos"], default=None)
    return (f"Events: {len(events)} rooms, {sum(e['count'] for e in events)} arrivals; "
            f"last {_sample(last, opening=True)}{_unknown(combined)}; "
            f"memory_read(room_id='*', start={group['catalogue_start']})")


def catalogue_lines(group):
    """The exact unwindowed room catalogue, shared with its character-offset reader."""
    from ouroboros import memory_view as mv
    from ouroboros.memory_view_conversation_forms import identity_line
    def presence(room):
        return ("a person spoke here; no word of my human or reply of mine" if room.get("overheard") else
                "a person or I spoke here" if room["engaged"] else "no recorded conversation here")
    return [identity_line(group), *(
        f"  - {room['label']}; {presence(room)}; "
        f"{mv._facts_text(room['facts'], 'that room')}; memory_read(room_id='{room['room_id']}', rows=true)"
        for room in group["rooms"])]


def locate_catalogue(groups):
    """Offsets include the focus's conversation and room; filter that room only afterwards."""
    offset = 0
    for group in groups:
        group["catalogue_start"] = offset
        offset += sum(len(line) + 1 for line in catalogue_lines(group))
