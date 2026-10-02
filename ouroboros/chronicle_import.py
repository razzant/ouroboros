"""Retain legacy bytes and interpretations without guessing missing coverage.

Activation and its scan boundary share one transaction under the old writer's
lock. Invalid sources become explicit, source-addressed gaps beside valid
evidence. A later downgrade's changed files are captured for reconciliation;
they never overwrite the new biography or rewind its active scan frontier.
"""
from __future__ import annotations

import hashlib
import json
import os
from types import SimpleNamespace


def _digest(value):
    data = json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _legacy_paths(store):
    memory = store.data_root / "memory"
    return {"blocks": memory / "dialogue_blocks.json", "meta": memory / "dialogue_meta.json",
            "flat": memory / "dialogue_summary.md"}


def _source_stats(paths):
    facts = {}
    for name, path in paths.items():
        try:
            stat = path.stat()
            facts[name] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                           "device": stat.st_dev, "inode": stat.st_ino}
        except FileNotFoundError:
            facts[name] = {"missing": True}
        except OSError as exc:
            facts[name] = {"unavailable": type(exc).__name__}
    return facts


def _last_sources(store, activation):
    checkpoints = store.records(kinds=["legacy_reconciliation"])
    return (checkpoints[-1] if checkpoints else activation).get("metadata", {})


def import_legacy(store, *, already_locked=False):
    """Fence legacy publication without waiting for an in-flight paid writer."""
    active = store.activation()
    if active and _last_sources(store, active).get("source_stats") == _source_stats(_legacy_paths(store)):
        return active
    if already_locked:
        return _import_locked(store)
    from ouroboros.platform_layer import file_lock_exclusive_nb, file_unlock
    from ouroboros.utils import assert_test_data_path

    lock_path = store.data_root / "memory" / ".consolidation.lock"
    assert_test_data_path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_WRONLY, 0o644)
    held = False
    try:
        try:
            file_lock_exclusive_nb(fd)
            held = True
        except (OSError, BlockingIOError):
            # A deferred downgrade reconciliation does not deactivate already
            # published memory or make new authored episodes disappear.
            return store.activation() or {"kind": "import_pending", "reason": "legacy_writer_active"}
        return _import_locked(store)
    finally:
        if held:
            file_unlock(fd)
        os.close(fd)


def _import_locked(store):
    active = store.activation()
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.consolidator import retain_memory_source
    from ouroboros.room_consolidation import room_sections

    context = SimpleNamespace(drive_root=store.data_root, task_id="chronicle-import")
    paths, raw, refs, errors = _legacy_paths(store), {}, {}, {}
    # Copy every readable source BEFORE interpreting any one of them. A corrupt
    # component cannot prevent another component's bytes from being retained.
    for name, path in paths.items():
        try:
            raw[name] = path.read_bytes()
        except FileNotFoundError:
            continue
        except OSError as exc:
            errors[name] = f"source unreadable: {type(exc).__name__}"
            continue
        refs[name] = retain_memory_source(context, "legacy_" + name, raw[name], "md" if name == "flat" else "json")
    stats = _source_stats(paths)
    if active:
        previous = _last_sources(store, active)
        prior_refs = previous.get("source_refs", {})
        changed = [name for name in paths if (prior_refs.get(name, {}).get("sha256") != refs.get(name, {}).get("sha256")
                   or name in errors)]
        checkpoint = {"id": "legacy-reconciliation-" + _digest([refs, stats]), "kind": "legacy_reconciliation",
                      "room_id": "legacy", "metadata": {"source_refs": refs, "source_stats": stats,
                          "changed_sources": changed, "read_errors": errors, "paid_calls": 0}}
        records = []
        if changed:
            text = ("[MEMORY GAP] Legacy memory sources changed after activation, for example during a downgrade. "
                    "Their old and new exact snapshots remain available. New authored memory and its scan frontier "
                    "are unchanged; overlap and changed meaning require source-based reconciliation, not guessed replacement.")
            records.append({"id": "legacy-change-gap-" + _digest([prior_refs, refs, changed]), "kind": "legacy",
                "room_id": "legacy", "text": text, "author": {"kind": "host", "attribution": "source-change fact"},
                "source_refs": list(prior_refs.values()) + list(refs.values()),
                "metadata": {"view_root": True, "legacy_type": "reconciliation_gap", "coverage": "unknown",
                             "source_gap": text, "changed_sources": changed, "read_errors": errors}})
        store.publish([*records, checkpoint])  # Never reset the active scan on re-upgrade.
        return active

    blocks, meta, flat = [], {}, ""
    try:
        blocks = json.loads(raw.get("blocks", b"[]"))
        if not isinstance(blocks, list):
            raise ValueError("legacy blocks are not a list")
    except (ValueError, UnicodeError) as exc:
        errors["blocks"], blocks = str(exc), []
    try:
        from ouroboros.memory_nomination_receipts import _pending
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate cursor key: {key}")
                result[key] = value
            return result
        meta = json.loads(raw.get("meta", b"{}"), object_pairs_hook=unique)
        if not isinstance(meta, dict):
            raise ValueError("legacy cursor is not an object")
        offset = meta.get("last_consolidated_offset", 0)
        if type(offset) is not int or offset < 0:
            raise ValueError("legacy cursor offset is invalid")
        if not isinstance(meta.get("chat_log_signature", {}), dict):
            raise ValueError("legacy cursor generation is invalid")
        _pending(meta)
    except (ValueError, UnicodeError) as exc:
        errors["meta"], meta = str(exc), {}
    try:
        flat = raw.get("flat", b"").decode("utf-8")
    except UnicodeError as exc:
        errors["flat"] = str(exc)

    records, texts = [], set()
    seen_sources = set()

    def gap(name, reason, source_refs, location=""):
        text = f"[MEMORY GAP] Legacy {name} cannot establish complete memory at {location or 'this source'}: {reason}. " \
               "Available evidence is retained; this gap is not a claim that the missing history was summarized."
        records.append({"id": "legacy-gap-" + _digest([name, reason, source_refs, location]), "kind": "legacy",
            "room_id": "legacy", "text": text, "author": {"kind": "host", "attribution": "source-gap fact"},
            "source_refs": source_refs, "metadata": {"view_root": True, "legacy_type": "gap", "coverage": "unknown",
                "source_gap": reason, "legacy_path": str(paths.get(name, "")), "location": location}})

    def visit(block, location, source, parent_id=""):
        node_id = "legacy-" + _digest({"source": source.get("sha256"), "location": location, "block": block})
        children, gap = [], None
        ref = block.get("source_ref")
        if block.get("type") == "era" and isinstance(ref, dict):
            identity = _digest(ref)
            if identity in seen_sources:
                gap = "repeated era source; original reference retained"
            else:
                seen_sources.add(identity)
                try:
                    raw = read_actor_source_bytes(store.data_root, str(ref.get("task_id") or "consolidation"), ref)
                    old = json.loads(raw)
                    if not isinstance(old, list) or not all(isinstance(item, dict) for item in old):
                        raise ValueError("era source is not a block list")
                    for n, child in enumerate(old):
                        children.extend(visit(child, f"{location}/source/{n}", ref, node_id + "-0"))
                except (OSError, ValueError, TypeError) as exc:
                    gap = f"era source unavailable: {exc}"
                finally:
                    seen_sources.remove(identity)
        ids = []
        for n, room in enumerate(room_sections(block)):
            record_id = node_id + f"-{n}"
            content = room["content"]
            legacy_gap = bool(block.get("gap_id") or block.get("type") == "gap" or "[MEMORY GAP]" in content)
            texts.add(content)
            records.append({"id": record_id, "kind": "gap" if legacy_gap else "legacy_source" if parent_id else "legacy", "room_id": room["room_id"],
                            "text": content, "author": {"kind": "legacy", "attribution": "unknown"},
                            "source_refs": [{**source, "location": location}],
                            "metadata": {"view_root": not bool(parent_id), "parent_id": parent_id,
                                         "children": children, "range": block.get("range", "unknown"),
                                         "legacy_message_count": block.get("message_count"),
                                         "label": room["label"], "legacy_type": block.get("type", "summary"),
                                         "source_gap": gap or (content if legacy_gap else None),
                                         "legacy_gap_id": block.get("gap_id", ""), "legacy_source_ref": ref}})
            ids.append(record_id)
        return ids

    for n, block in enumerate(blocks):
        try:
            if not isinstance(block, dict):
                raise ValueError("legacy block is not an object")
            visit(block, str(n), refs["blocks"])
        except (ValueError, TypeError, KeyError) as exc:
            gap("blocks", str(exc), [refs["blocks"]], str(n))
    # Exact duplication is knowable without a model. Partial overlap is not:
    # retain it with honest unknown coverage instead of guessing it away.
    if flat and flat not in texts and flat not in {str(b.get("content", "")) for b in blocks if isinstance(b, dict)}:
        records.append({"id": "legacy-flat-" + hashlib.sha256(raw["flat"]).hexdigest(), "kind": "legacy",
                        "room_id": "legacy", "text": flat,
                        "author": {"kind": "legacy", "attribution": "unknown"},
                        "source_refs": [refs["flat"]],
                        "metadata": {"view_root": True, "coverage": "unknown", "legacy_type": "flat"}})
    for name, reason in errors.items():
        gap(name, reason, [refs[name]] if name in refs else [])
    if "meta" in errors:
        meta = _new_experience_boundary(store, context, records, refs.get("meta"))
    receipt = {"id": "legacy-import-" + _digest(refs), "kind": "activation", "room_id": "",
               "metadata": {"source_refs": refs, "source_stats": stats, "imported_records": len(records),
                            "paid_calls": 0, "legacy_files_unchanged": True}}
    # The scanner and the imported memory become visible together. A concurrent
    # importer is harmless: all identities and timestamps replay idempotently.
    return store.publish([*records, receipt], scan_state=meta)[-1]


def _new_experience_boundary(store, context, records, cursor_source):
    """Keep an explicit historical gap, then start only after a verified raw boundary."""
    from ouroboros import consolidator as c
    source = store.data_root / "logs/chat.jsonl"
    chain = c._ordered_chat_generation_paths(source)
    anchor = source if source.exists() and source.stat().st_size else next(
        (path for path in reversed(chain[:-1]) if path.exists()), source)
    segments = [anchor, source] if anchor != source else [source]
    scan = {"last_consolidated_offset": 0, "chat_log_signature": c._chat_log_signature(anchor)}
    try:
        captured = c._capture_generation_window(store.data_root / "memory/dialogue_meta.json", source,
            segments, 0, load_cursor=lambda: scan)
    except (OSError, ValueError, UnicodeError, AttributeError):
        captured = None
    refs = [cursor_source] if cursor_source else []
    if captured is None:
        scan["raw_scan_unavailable"] = {"kind": "activation_boundary_unavailable", "source_refs": refs}
    else:
        segments, signatures, entries, all_rows, _offset = captured
        refs.append(c.retain_memory_source(context, "activation_raw_boundary",
            json.dumps({"generations": signatures, "rows": all_rows}, ensure_ascii=False).encode(), "json"))
        if len(segments) > 1 and not entries[-1]:
            segments, signatures, entries = segments[:-1], signatures[:-1], entries[:-1]
        c._advance_cursor(scan, segments, signatures, entries, len(all_rows))
    text = ("[MEMORY GAP] The old cursor is unreadable, so historical unconsolidated coverage is unknown. "
            "Existing valid memories remain visible. Automatic maintenance starts only after the captured activation boundary; "
            "it has not rebuilt or summarized the earlier raw history. Read logs/chat.jsonl and archive/chat_*.jsonl "
            "for the surviving original generations and the retained boundary source before deciding on historical repair.")
    records.append({"id": "legacy-cursor-gap-" + _digest(refs), "kind": "legacy", "room_id": "legacy", "text": text,
        "author": {"kind": "host", "attribution": "coverage-gap fact"}, "source_refs": refs,
        "metadata": {"view_root": True, "legacy_type": "cursor_gap", "coverage": "unknown", "source_gap": text}})
    return scan
