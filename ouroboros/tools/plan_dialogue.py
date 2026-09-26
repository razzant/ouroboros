"""Bind full room evidence to the existing plan request and source-handle custody."""
from __future__ import annotations

import json
import pathlib
from typing import Any

from ouroboros.dialogue_evidence import own_room_chat, read_room_source, task_room_record
from ouroboros.projects_registry import all_task_bindings, list_reserved_projects
from ouroboros.task_results import load_plan_review_state


def related_rooms(ctx: Any, root: pathlib.Path, own_chat: int | None) -> list[dict]:
    """Only Main and task-lineage rooms, using existing registry activity facts."""
    task_id = str(getattr(ctx, "task_id", "") or "")
    task = {**(task_room_record(root, task_id)), **(getattr(ctx, "task_metadata", {}) or {}), "task_id": task_id}
    bindings = all_task_bindings(root)
    chats = {1} if own_chat != 1 else set()
    for field in ("task_id", "parent_task_id", "root_task_id"):
        tid = str(task.get(field) or "")
        if tid:
            source = task_room_record(root, tid)
            chat = bindings.get(tid) or source.get("chat_id")
            if chat is not None:
                chats.add(int(chat))
    from ouroboros.task_results import resolve_task_lineage
    lineage_root = resolve_task_lineage(task_id, metadata=task)["root_task_id"]
    for bound_id, chat in bindings.items():
        related = task_room_record(root, bound_id)
        if related and resolve_task_lineage(bound_id, metadata=related)["root_task_id"] == lineage_root:
            chats.add(int(chat))
    projects = {int(p["chat_id"]): p for p in list_reserved_projects(root)}
    pointers = []
    for chat in sorted(chats - {own_chat}):
        if chat != 1 and chat not in projects:
            continue
        row = projects.get(chat, {})
        pointers.append({"locator": f"chat:{chat}", "label": row.get("name") or "Main",
                         "last_active_at": row.get("last_active_at"), "message_count": None,
                         "count_status": "not_loaded", "delivery": "pointer_only"})
    return pointers


def _source_view(root: pathlib.Path, task_id: str, source: dict, ref: dict | None) -> dict:
    from ouroboros.artifacts import task_artifact_dir_path

    view = {key: source[key] for key in ("chat_id", "label", "captured_at", "coverage", "sha256", "bytes", "text", "secrets_redacted") if key in source}
    view["locator"] = f"chat:{source['chat_id']}@{source['sha256']}"
    view["lines"] = source["text"].count("\n")
    if ref:
        view["source_ref"] = ref
        view["file"] = str(task_artifact_dir_path(root, task_id, create=False) / ref["path"])
    return view


def attach_own_dialogue(ctx: Any, root: pathlib.Path, manifest: dict,
                        author_fingerprint: str, *, persist: bool = False) -> dict:
    """Identical author inputs reuse recorded sources, including during replay.

    Room growth is evidence for the next changed author request. It cannot
    itself mint another paid plan envelope. Health/roster/cycle rails stay with
    the existing engine, which still decides whether any dispatch is earned.
    """
    from ouroboros.artifacts import read_actor_source_bytes, store_actor_source_bytes

    task_id = str(getattr(ctx, "task_id", "") or "")
    state = load_plan_review_state(root, task_id)
    recorded = next((wave for wave in reversed(state.get("waves") or [])
                     if wave.get("author_request_fingerprint") == author_fingerprint), None)
    if recorded:
        from ouroboros.tools.plan_review_artifacts import authority_wave

        exact = authority_wave(root, task_id, recorded)
        previous = (exact.get("evidence_manifest_full") or {}).get("own_dialogue") or {}
        if recorded.get("dialogue_source_ref"):
            ref = recorded["dialogue_source_ref"]
            raw = read_actor_source_bytes(root, task_id, ref)
            source = {**previous, "text": raw.decode("utf-8"), "sha256": ref["sha256"], "bytes": len(raw)}
            own = _source_view(root, task_id, source, ref)
        else:
            own = dict(previous)  # An explicit missing-room fact also replays exactly.
        pointers = (exact.get("evidence_manifest_full") or {}).get("related_rooms") or []
    else:
        chat = own_room_chat(ctx, root)
        source = read_room_source(root, chat, task_id=task_id, mailbox_root=ctx.drive_root) if chat is not None else None
        if source is None:
            own = {"chat_id": chat, "gap": "own_room_unavailable", "text": ""}
        else:
            ref = store_actor_source_bytes(
                root, task_id, category="context_checkpoints", source_id=f"plan-dialogue-{chat}",
                data=source["text"].encode("utf-8"), extension="jsonl",
            ) if persist else None
            own = _source_view(root, task_id, source, ref)
        pointers = related_rooms(ctx, root, chat)
    return {**manifest, "author_request_fingerprint": author_fingerprint,
            "own_dialogue": own, "related_rooms": pointers}


def plan_chat_reader(root: pathlib.Path, task_id: str):
    """Resolve snapshot-qualified chat ranges only through recorded task custody."""
    from ouroboros.artifacts import read_actor_source_bytes

    def read(locator: str):
        chat, sep, digest = locator.partition("@")
        try:
            chat_id = int(chat)
        except ValueError:
            return None
        if not sep:
            return read_room_source(root, chat_id)
        state = load_plan_review_state(root, task_id)
        for wave in reversed(state.get("waves") or []):
            ref = wave.get("dialogue_source_ref") or {}
            if ref.get("sha256") == digest and wave.get("dialogue_chat_id") == chat_id:
                raw = read_actor_source_bytes(root, task_id, ref)
                return {"text": raw.decode("utf-8"), "coverage": json.loads(raw.split(b"\n", 1)[0]).get("coverage", {})}
        return None
    return read


# Structural producer fields only (BIBLE P5): the streams that carry the two speakers, quiz
# cards and answers, and addressed mailbox rows. Progress rows and host system rows stay in
# the exact snapshot behind the pointer, addressed by the inline line numbers.
_CONVERSATION_STREAMS = frozenset({"chat", "mailbox", "retained_quiz_projection", "retained_origin"})
_ATTACHMENT_NAME_KEYS = ("label", "name", "filename", "original_filename")


def _snapshot_rows(own: dict) -> list[tuple[int, dict]]:
    """``[(line_no, row)]`` over the recorded snapshot bytes. Line 1 is the JSONL header, so
    the numbers equal the physical lines a ``chat:<id>@<sha256>::lines=A-B`` locator selects."""
    rows: list[tuple[int, dict]] = []
    for number, line in enumerate(str(own.get("text") or "").split("\n"), start=1):
        if number == 1 or not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append((number, row))
    return rows


def _is_conversation(row: dict) -> bool:
    return str(row.get("stream") or "") in _CONVERSATION_STREAMS and (
        row.get("direction") != "system" or row.get("type") == "quiz_answer")


def _quiz_text(row: dict) -> str:
    quiz = row.get("quiz") if isinstance(row.get("quiz"), dict) else {}
    options = [str(o) for o in quiz.get("options") or []] if isinstance(quiz.get("options"), list) else []
    qid = str(quiz.get("quiz_id") or "")
    if row.get("type") == "quiz_answer":
        index = quiz.get("answered_index")
        chosen = (f"chose ({index + 1}) {options[index]}" if isinstance(index, int) and 0 <= index < len(options)
                  else f'own answer: "{quiz.get("comment") or row.get("text") or ""}"')
        comment = str(quiz.get("comment") or "")
        return f"[answer {qid}] {chosen}" + (f' — "{comment}"' if comment and isinstance(index, int) else "")
    recommended = quiz.get("recommended_index")
    listed = " ".join(f"({i}) {label}" for i, label in enumerate(options, start=1))
    return (f"[question {qid}] {quiz.get('question') or row.get('text') or ''} — options: {listed}"
            + (f"; recommended ({recommended + 1})" if isinstance(recommended, int) else ""))


def _line(number: int, row: dict) -> str:
    body = _quiz_text(row) if row.get("type") in {"quiz", "quiz_answer"} and isinstance(row.get("quiz"), dict) \
        else str(row.get("text") or "")
    names = [next((str(a[k]) for k in _ATTACHMENT_NAME_KEYS if a.get(k)), "")
             for a in (row.get("attachments") or []) if isinstance(a, dict)]
    body = body.replace("\n", "\n  ") + (f" [attachments: {', '.join(n for n in names if n)}]" if any(names) else "")
    return f"{number} · {row.get('ts') or '-'} · {row.get('author') or '?'} · {body}"


def _gap_summary(own: dict) -> str:
    counts: dict[str, int] = {}
    for section in (own.get("coverage") or {}).values() if isinstance(own.get("coverage"), dict) else ():
        if isinstance(section, dict):
            for gap in section.get("gaps") or []:
                kind = str(gap.get("kind") if isinstance(gap, dict) else gap)
                counts[kind] = counts.get(kind, 0) + 1
            if section.get("complete") is False:
                counts["mailbox_incomplete"] = counts.get("mailbox_incomplete", 0) + 1
    for key in ("room_gap", "ordering_gap"):
        if (own.get("coverage") or {}).get(key):
            counts[str(own["coverage"][key])] = counts.get(str(own["coverage"][key]), 0) + 1
    return ", ".join(f"{kind}×{n}" for kind, n in counts.items()) or "none"


def dialogue_view(own: dict, keep: int | None = None) -> tuple[str, dict]:
    """The inline OWN ROOM DIALOGUE section as numbered readable lines, plus its facts.

    ``keep`` limits the inline conversation to its NEWEST rows (all when None); every
    omitted row stays addressable by its snapshot line number through the pointer.
    """
    rows = _snapshot_rows(own)
    conversation = [(n, row) for n, row in rows if _is_conversation(row)]
    others = [(n, row) for n, row in rows if not _is_conversation(row)]
    progress = sum(1 for _n, row in others if row.get("stream") == "progress")
    inline = conversation if keep is None else conversation[len(conversation) - min(keep, len(conversation)):]
    omitted = len(conversation) - len(inline)
    locator = str(own.get("locator") or f"chat:{own.get('chat_id')}")
    facts = {"source_sha256": own.get("sha256"), "snapshot_lines": int(own.get("lines") or 0),
             "conversation_rows": len(conversation), "conversation_inline_rows": len(inline),
             "conversation_first_inline_line": inline[0][0] if inline else None, "other_rows": len(others)}
    if own.get("gap") or not rows:
        body = f"Explicit gap: own room source unavailable ({own.get('gap') or 'no snapshot rows'})."
        return ("## OWN ROOM DIALOGUE (exact recorded snapshot)\n\n" + body + "\n"), facts
    header = (f"Room {own.get('label') or own.get('chat_id')} · locator `{locator}` · {facts['snapshot_lines']} snapshot "
              f"lines, {own.get('bytes', '?')} bytes · captured {own.get('captured_at') or 'time unavailable'} · "
              f"{len(conversation)} conversation rows, {len(others)} other rows · gaps: {_gap_summary(own)}")
    cut = (f"; {omitted} earlier conversation rows before line {inline[0][0]} (exact omitted prefix: "
           f"`{locator}::lines=2-{inline[0][0] - 1}`)" if omitted and inline else
           f"; all {omitted} conversation rows (nothing fit inline; exact omitted prefix: "
           f"`{locator}::lines=2-{facts['snapshot_lines']}`)" if omitted else "")
    footer = (
        f"Not inline: {progress} progress rows and {len(others) - progress} host rows{cut}. Every inline line "
        "carries its snapshot line number, so a gap in the numbering is a row not shown. The complete redacted "
        f"snapshot `{locator}` ({facts['snapshot_lines']} lines) is in task custody: a session reads "
        f"`{own.get('file') or '(not persisted for this dry run)'}`; a native episode reads "
        f"read_file(root='artifact_store', path='{(own.get('source_ref') or {}).get('path') or ''}'); any reviewer "
        f"may request lines with `need_evidence` locator `{locator}::lines=A-B`. Both speakers keep their source "
        "and author; a peer suggestion is not an owner instruction. Later messages are not claimed reviewed by "
        "this snapshot."
    )
    return ("## OWN ROOM DIALOGUE (exact recorded snapshot)\n\n" + header + "\n\n"
            + "\n".join(_line(n, row) for n, row in inline) + ("\n\n" if inline else "") + footer + "\n"), facts


def render_dialogue(manifest: Any) -> str:
    own = manifest.get("own_dialogue") or {}
    return (
        dialogue_view(own)[0]
        + "## RELATED ROOMS (pointers only; request chat:<id> with need_evidence)\n\n"
        + json.dumps(manifest.get("related_rooms") or [], ensure_ascii=False, default=str) + "\n"
    )


def fit_dialogue_view(packet: str, own: dict, capacity_chars: int, *, measure=len) -> tuple[str, dict]:
    """Keep the whole conversation when it fits this delivery's actual measure, else the
    largest NEWEST run of conversation rows, the cut named in the footer with its exact
    line range. Only the inline conversation yields room; governance, the operative inputs
    and the pointer stay intact. Progress and host rows are never inline."""
    full, facts = dialogue_view(own)
    if measure(packet) <= capacity_chars or full not in packet:
        return packet, facts

    def selected(keep: int) -> tuple[str, dict]:
        view, kept = dialogue_view(own, keep)
        return packet.replace(full, view, 1), kept

    low, high = 0, facts["conversation_rows"]
    while low < high:
        count = (low + high + 1) // 2
        if measure(selected(count)[0]) <= capacity_chars:
            low = count
        else:
            high = count - 1
    return selected(low)


def dialogue_slot_inputs(slots: list, *, system_prompt: str, user_content: str,
                         session_task: str, manifest: dict, slot_messages: dict,
                         native_mandatory_chars: int, data_root: Any = "",
                         frozen: dict | None = None, session_root: str = "", task_id: str = "") -> dict:
    """Project a fresh request, or reuse the recorded delivery at collection."""
    if frozen is not None:
        from ouroboros.tools.plan_review_artifacts import frozen_delivery_inputs
        return frozen_delivery_inputs(frozen, slots)
    from ouroboros.model_slots import MODEL_CONTEXT_WINDOWS_KEY, model_role_option
    from ouroboros.tools.plan_review_runtime import PLAN_REVIEW_MAX_TOKENS, slot_retrieves, slot_is_session
    from ouroboros.tools.review_synthesis import build_plan_review_messages, per_slot_input_token_limits
    from ouroboros.tools.plan_packet import plan_user_stable_len
    from ouroboros.tools.plan_spec import PLAN_FINDINGS_ARRAY_CONTRACT
    from ouroboros.tools.scope_required_sources import source_text_identity
    from ouroboros.review_native_episode import review_native_transcript_bound, native_landing_at, native_first_send_chars
    from ouroboros.reviewer_window import reviewer_window_binding
    from ouroboros.review_execution import _messages_char_count

    own = manifest.get("own_dialogue") or {}
    messages, tasks, lengths, coverage = dict(slot_messages), {}, {}, {}
    api = [slot for slot in slots if not slot_retrieves(slot)]
    limits = per_slot_input_token_limits([s.model for s in api], output_reserve=PLAN_REVIEW_MAX_TOKENS,
                                       tokenizer_margin=155_000, slots=api)
    # What each session reviewer may read of the room: the snapshot declared as an OBSERVED
    # source (never a required manifest, which would order multi-MB reads), so the harness
    # journal fold records per-reviewer coverage as a fact (`review_session_reads`).
    observed = ([{"root": "artifact_store", "path": own["source_ref"]["path"], "file": own["file"],
                  **source_text_identity(str(own.get("text") or "").encode("utf-8"))}]
                if own.get("source_ref") and own.get("file") else [])
    for slot in slots:
        sid = str(slot.slot_id)
        # A shared packet cannot say "your": each slot's send ends with its own seat, after the
        # cache-stable prefix, so a cycle-2 reviewer knows which earlier findings are its own.
        seat = f"\n## YOUR PANEL SEAT\n\n`{sid}`\n"
        if not slot_retrieves(slot):
            capacity = int(limits[sid]) * 4
            existing = messages.get(sid)
            if existing:
                # Continuation history is already exact; only this turn's new
                # automatic source can shrink, never its prior paid inputs.
                total = _messages_char_count(existing)
                view, coverage[sid] = fit_dialogue_view(user_content, own, capacity - total + len(user_content) - len(seat))
                messages[sid] = [{**m, "content": view + seat} if i == len(existing) - 1 and m.get("role") == "user" else dict(m)
                                 for i, m in enumerate(existing)]
            else:
                view, coverage[sid] = fit_dialogue_view(user_content, own, capacity - len(system_prompt) - len(seat))
                messages[sid] = build_plan_review_messages(system_prompt, view + seat, plan_user_stable_len(view))
            lengths[sid] = _messages_char_count(messages[sid])
        elif not slot_is_session(slot):
            bound = review_native_transcript_bound(slot.model, output_reserve=PLAN_REVIEW_MAX_TOKENS,
                                                   mandatory_read_chars=native_mandatory_chars,
                                                   **reviewer_window_binding(slot))
            governance_read = max(0, native_mandatory_chars - len(session_task))
            def first_send(task):
                return native_first_send_chars(session_root, surface="plan_review", role_hint=slot.role_hint,
                    slot_id=sid, session_task=task, output_contract=PLAN_FINDINGS_ARRAY_CONTRACT, task_id=task_id)
            tasks[sid], coverage[sid] = fit_dialogue_view(session_task + seat, own,
                native_landing_at(bound) - governance_read - 1, measure=first_send)
        else:
            # A delegated session receives the conversation inline like every other slot. The
            # host owns no session window: an owner-asserted `reviewer:<slot>` window fits the
            # conversation to it; without one nothing is invented — the whole conversation goes,
            # the pointer names the exact snapshot, and the harness owns its own context.
            asserted = int(model_role_option(MODEL_CONTEXT_WINDOWS_KEY, reviewer_window_binding(slot)["model_role"]))
            if asserted > 0:
                limit = per_slot_input_token_limits([slot.model], context_window=asserted, slots=[slot],
                                                    output_reserve=PLAN_REVIEW_MAX_TOKENS, tokenizer_margin=155_000)
                tasks[sid], coverage[sid] = fit_dialogue_view(session_task + seat, own, int(limit[sid]) * 4)
            else:
                tasks[sid], coverage[sid] = session_task + seat, dialogue_view(own)[1]
            coverage[sid].update(window="asserted" if asserted > 0 else "unasserted", file=own.get("file") or "")
        if slot_retrieves(slot):
            lengths[sid] = (first_send(tasks.get(sid, session_task)) if not slot_is_session(slot)
                            else len(tasks.get(sid, session_task)))
        if sid in coverage:
            coverage[sid]["delivery"] = ("delegated_session" if slot_is_session(slot) else
                                         "native_retrieving" if slot_retrieves(slot) else "packet")
    return {"slot_messages": messages, "slot_session_tasks": tasks, "slot_prompt_chars": lengths,
            "dialogue_delivery": coverage,
            "native_mandatory_read_chars": native_mandatory_chars,
            "request_policy": {"output_contract": PLAN_FINDINGS_ARRAY_CONTRACT,
                               "native_data_root": str(data_root),
                               "native_mandatory_read_chars": native_mandatory_chars,
                               **({"observed_sources": observed} if observed else {})}}
