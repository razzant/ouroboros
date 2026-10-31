"""Typed runtime reads with exact evidence exports, never model-supplied paths.

Selectors use dotted keys, [n] and [*]; wildcards preserve array nesting. JSONL
filters select whole records before tail and projection. A bounded tail window
discloses incomplete scanning. Store documents and multi-id bundles are canonical
JSON reconstructions; single file/blob records retain their original bytes.

Every result names its data plane: ``caller`` (the task's drive), ``child`` (an
owned descendant's drive, selected through the caller's lineage), ``canonical``
(the installation owner) or a native reviewer's pinned ``native_pin``, which never
resolves to live data. The owner's default queue files, its money ledger and the
owner fallback of a blob read need the same actor's existing resource read
admission of each exact loose or pack file; tool availability is separate. Refusal grants nothing. Task documents, review
records, skill state and retained runs are read from the owner's state without
that check.
"""
from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import re

from ouroboros import query_data_sql as sql, store
from ouroboros.tool_capabilities import RESULT_VIEW_PARAMS
from ouroboros.tools.registry import ToolContext, ToolEntry
from ouroboros.tools.tool_result import ToolResult, _publish_tool_result

_FILES = {"review_ledger": "state/review_ledger/{id}.json", "advisory_review": "state/advisory_review.json",
          "queue_snapshot": "state/queue_snapshot.json", "queue_dispatch": "state/queue_dispatch.json",
          "skill_state": "state/skills/{id}/{state_file}.json",
          "run": "task_drives/{child}/delegated_runs/{id}.json",
          "verification_receipts": "task_results/artifacts/{id}/verification_receipts.jsonl",
          "storage": "state/storage_measurement.json",  # the last two finished counts (installation-wide) + removers
          "knowledge_history": "memory/knowledge_history.jsonl",  # global knowledge writes, bodies left out
          "update_letter": "state/update_letter.json",  # the description with its whole range qualification
          "update_material": "state/update_material.json",  # the offered range's commits, rows and read recipe
          **{name + "_log": f"logs/{name}.jsonl" for name in ("supervisor", "events", "tools", "chat")}}
_ID_KINDS = {"task_result", "run", "review_ledger", "skill_state", "verification_receipts"}
# op=blob reads the observability CAS by digest. The digest in a task-source handle's name is that file's own
# hash, read through the source's reader; a miss names that reader instead of another plane (BIBLE P1).
_BLOB_KIND_HINT = ("; op=blob reads observability CAS blobs only. A digest in a task-source handle name "
                   "(source_handles/...-<sha256>.<ext>) is that file's own hash: open it with the reader its mark or "
                   "result names, memory_read(source_ref={kind, root, path, size, sha256, task_id})")
# ``context_snapshot``: the skill's current declared context snapshot, the address its outdated note names.
_STATES = ("review", "review_job", "health", "enabled", "deps", "grants", "context_snapshot")
# Each knowledge history row holds both complete versions of its note, so the ordinary
# 8 MiB log window would cover hours of writes; this read budget covers far more of them.
_HISTORY_SCAN_BYTES = 256 * 1024 * 1024
_HISTORY_BODY_FIELDS = ("old_content", "new_content")
_SELECTOR = re.compile(r"(?:[^.\[\]]+|\[(?:\d+|\*)\])(?:\.[^.\[\]]+|\[(?:\d+|\*)\])*")


class SelectorError(ValueError):
    """A value-free selector diagnostic.

    The error describes structure only.  In particular it never includes a
    neighbouring value from the document: object failures disclose key names,
    and array failures disclose only the array length.
    """


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def selected_bytes(value) -> int:
    """The smallest ``max_bytes`` that returns ``value`` as a selection: the output check counts
    its canonical JSON, whatever ``format`` is (a producer that advertises a budget uses this)."""
    return len(_canonical(value))


def _name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", value) or value in {".", ".."}:
        raise ValueError("IDs and artifact names must be plain names, without path components")
    return value


def _arg(name, value):
    """``name=value`` as the call spelled it in JSON, bounded: a refusal names field and value."""
    text = json.dumps(value, ensure_ascii=False, default=str)
    return f"{name}={text if len(text) <= 64 else text[:60] + '...'}"


def _over_input(subject, size, cap, repair=""):
    """A read past the fixed per-call input budget, which every json/jsonl source spends before
    select or max_bytes apply. ``size`` is exact, or ``None`` when a bounded read only proved it
    larger than ``cap``; IDs share the budget, so after earlier IDs ``cap`` is what they left."""
    budget = (f"the fixed {sql.MAX_BYTES}-byte input budget" if cap >= sql.MAX_BYTES
              else f"the {cap} bytes earlier ids left of the {sql.MAX_BYTES}-byte input budget")
    hints = [repair] if repair else []
    if cap < sql.MAX_BYTES and (size is None or size <= sql.MAX_BYTES):
        hints.append("query fewer ids per call")
    elif not repair:
        hints.append("select and max_bytes apply after this read and cannot admit it")
    measured = f"is {size} bytes, over" if size is not None else "is larger than"
    return ValueError(f"{subject} {measured} {budget}; " + "; ".join(hints))


def _json_type(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (int, float)):
        return "number"
    return type(value).__name__


def _object_keys(value):
    names = sorted(str(key) for key in value)
    shown = names[:32]
    suffix = f", ... ({len(names) - len(shown)} more)" if len(names) > len(shown) else ""
    return "[" + ", ".join(json.dumps(name, ensure_ascii=False) for name in shown) + suffix + "]"


def _select(value, selector):
    if not selector:
        return value
    parts = re.findall(r"[^.\[\]]+|\[(?:\d+|\*)\]", selector)

    def walk(current, index, path):
        if index == len(parts):
            return current
        part = parts[index]
        if part == "[*]":
            if not isinstance(current, list):
                detail = f"; object keys={_object_keys(current)}" if isinstance(current, dict) else ""
                raise SelectorError(
                    f"selector {selector!r} failed at path {path!r}, segment '[*]': "
                    f"expected array, got {_json_type(current)}{detail}"
                )
            return [walk(item, index + 1, f"{path}[{item_index}]")
                    for item_index, item in enumerate(current)]
        if part.startswith("["):
            if not isinstance(current, list):
                detail = f"; object keys={_object_keys(current)}" if isinstance(current, dict) else ""
                raise SelectorError(
                    f"selector {selector!r} failed at path {path!r}, segment {part!r}: "
                    f"expected array, got {_json_type(current)}{detail}"
                )
            offset = int(part[1:-1])
            if offset >= len(current):
                raise SelectorError(
                    f"selector {selector!r} failed at path {path!r}, segment {part!r}: "
                    f"array length={len(current)}"
                )
            return walk(current[offset], index + 1, f"{path}[{offset}]")
        if not isinstance(current, dict):
            raise SelectorError(
                f"selector {selector!r} failed at path {path!r}, segment {part!r}: "
                f"expected object, got {_json_type(current)}"
            )
        if part not in current:
            raise SelectorError(
                f"selector {selector!r} failed at path {path!r}, segment {part!r}: "
                f"object keys={_object_keys(current)}"
            )
        child_path = part if path == "$" else f"{path}.{part}"
        return walk(current[part], index + 1, child_path)
    return walk(value, 0, "$")


def _path(ctx, root, relative):
    from ouroboros.tools.core_secret_paths import is_runtime_secret_target
    target = (root / relative).resolve()
    if is_runtime_secret_target(target, ctx.repo_dir, data_root=root, ctx=ctx, include_owner_state=False):
        raise PermissionError("runtime secret target refused")
    if not target.is_relative_to(root.resolve()):
        raise PermissionError("source escapes its host-selected root")
    return target


def _read_refusal(ctx, target):
    """The actor's resource read admission of this file; "" when admitted.

    This reuses read_file's resource rules, not its separate tool availability.

    Physical binding, the Presence resource ceiling and the runtime-data and
    protected-artifact rules decide; disk visibility or a digest never grants.
    """
    from ouroboros.presence_authority import presence_ceiling_allows_binding, presence_ceiling_from_context
    from ouroboros.protected_artifacts import block_reason_for_path
    from ouroboros.tool_access import build_resolved_resource_binding
    from ouroboros.tools.core_file_tools import _runtime_data_read_block
    from ouroboros.tools.tool_resolution import _root_containing_absolute_path

    try:
        root = _root_containing_absolute_path(ctx, "read_file", str(target)) or "user_files"
        binding = build_resolved_resource_binding(ctx, root=root, operation="read", path=str(target))
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        return str(exc) or type(exc).__name__
    ceiling = presence_ceiling_from_context(ctx)
    if ceiling is not None and not presence_ceiling_allows_binding(ceiling, binding):
        return "the Presence resource ceiling does not include this read"
    return _runtime_data_read_block(ctx, target, root=binding.root) or block_reason_for_path(
        ctx, target, "read_bytes", binding) or ""


def _plane_path(ctx, plane, root, relative, local):
    """A plane file (``relative`` to ``root``, or an absolute physical source). Off the selected
    local drive (the caller's, or the owned child's that ``_child_root`` admitted through lineage)
    it needs the actor's read admission."""
    path = _path(ctx, root, relative)
    if root != local and (refusal := _read_refusal(ctx, path)):
        raise PermissionError(f"the {plane} plane is not readable by this actor ({refusal}); no grant is inferred")
    return path


def _child_root(ctx, canonical, child):
    from ouroboros.task_custody import own_child_drives
    caller = _name(ctx.task_id)
    with store.open_store(canonical, write=False) as txn:
        owned = txn.conn.execute("""WITH RECURSIVE lineage(task_id) AS (
            SELECT task_id FROM tasks WHERE task_id=? UNION
            SELECT t.task_id FROM tasks t JOIN lineage p ON t.parent_task_id=p.task_id)
            SELECT task_id FROM lineage WHERE task_id=?""", (caller, child)).fetchone()
    if not owned:
        raise PermissionError("child drive is outside the caller's lineage")
    if child == caller:
        return Path(ctx.drive_root).resolve()
    roots = own_child_drives(canonical, child)
    if not roots:
        raise FileNotFoundError("owned child drive is unavailable")
    return roots[0]


def _jsonl(path, where, tail, cap, subject):
    rows, consumed, partial = deque(maxlen=tail or None), 0, False
    whole = bytearray() if not where and not tail else None
    with path.open("rb") as stream:
        if tail and path.stat().st_size > cap:
            stream.seek(-cap, 2)
            stream.readline(cap + 1)  # discard the possibly partial leading record
            partial = True
        while line := stream.readline(cap - consumed + 1):
            consumed += len(line)
            if consumed > cap:  # A positive tail can overrun too if the input grows during the scan.
                repair = ("retry after input growth settles" if tail else
                          "a nonzero tail scans only the newest bytes and reports scan_truncated")
                raise _over_input(f"{subject} scan (tail={tail})", None, cap,
                                  f"read at least {consumed} bytes; {repair}")
            if whole is not None:
                whole.extend(line)
            if not line.strip():
                continue
            value = json.loads(line)
            if where and not _matches(value, where):
                continue
            rows.append((line, value))
    raw = bytes(whole) if whole is not None else b"".join(row[0] for row in rows)
    return raw, [row[1] for row in rows], partial, consumed


def _matches(value, where):
    try:
        field = _select(value, where["field"])
    except SelectorError:
        return False
    if where["op"] == "eq":
        return field == where["value"]
    return isinstance(field, (str, list, dict)) and where["value"] in field


def _knowledge_history(path, where, tail):
    """The newest ``tail`` matching knowledge writes, each without its two note bodies.

    Facts a row lacks stay absent (old rows carry no publication stamp); the current
    text is ``knowledge_read``. A window past the read budget is said as truncated.
    """
    rows, partial = deque(maxlen=tail or None), False
    with path.open("rb") as stream:
        if path.stat().st_size > _HISTORY_SCAN_BYTES:
            stream.seek(-_HISTORY_SCAN_BYTES, 2)
            stream.readline()  # discard the possibly partial leading record
            partial = True
        for line in stream:
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                for field in _HISTORY_BODY_FIELDS:
                    value.pop(field, None)
            if where and not _matches(value, where):
                continue
            rows.append(value)
    value = list(rows)
    raw = _canonical(value)
    return raw, value, partial, len(raw)


def _retained_run(canonical, identity, child):
    """A run's terminal JSON retained once in its staging task's canonical store (#1573), named by
    the run's current custody ``output_source``; only a run the selected task (the caller or its
    owned descendant) staged qualifies, as its task-drive staging did. ``None``: the legacy path."""
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.delegate_custody_current import run

    entry = run(canonical, identity)
    source = entry.output_source if entry is not None else {}
    if not source or str(source.get("task_id") or "") != child:
        return None
    return read_actor_source_bytes(canonical, child, {key: source.get(key) for key in ("kind", "root", "path", "size", "sha256")})


def _record(ctx, root, kind, identity, child, state_file, op, where, tail, cap, canonical=None):
    if kind == "task_result":
        if not _path(ctx, root, store.STORE_REL).is_file():
            raise FileNotFoundError("task store is unavailable in the selected data root")
        with store.open_store(root, write=False) as txn:
            size = txn.conn.execute("SELECT length(CAST(body AS BLOB)) FROM task_bodies WHERE task_id=?",
                                    (identity,)).fetchone()
            if size is None:
                raise FileNotFoundError("task result not found")
            if size[0] > cap:
                raise _over_input(f"task_result {identity} as stored", size[0], cap)
            value = txn.load(identity)
        raw = _canonical(value)
        return raw, value, True, False, len(raw), False
    raw = _retained_run(canonical, identity, child) if kind == "run" else None
    if raw is not None:  # size and sha256 verified against the custody ref; read from the owner's store
        if len(raw) > cap:
            raise _over_input(f"run {identity}", len(raw), cap)
        return raw, json.loads(raw), False, False, len(raw), True
    if root is None:
        raise FileNotFoundError("owned child drive is unavailable")
    relative = _FILES[kind].format(id=identity, child=child, state_file=state_file)
    if kind == "skill_state" and state_file == "context_snapshot":
        from ouroboros.extension_liveness import live_context_publishers

        descriptor = live_context_publishers(root).get(identity)
        if not descriptor or not descriptor.get("eligible"):
            raise FileNotFoundError(f"{identity} publishes no live context snapshot")
        relative = (Path(descriptor["state_dir"]).resolve() / descriptor["path"]).relative_to(Path(root).resolve()).as_posix()
    if kind == "run" and root.parts[-2:] == ("task_drives", child):
        relative = f"delegated_runs/{identity}.json"
    path = _path(ctx, root, relative)
    subject = f"{kind} {identity}".strip()
    if kind == "knowledge_history":
        raw, value, partial, consumed = _knowledge_history(path, where, tail)
        return raw, value, True, partial, consumed, False
    if op == "jsonl":
        raw, value, partial, consumed = _jsonl(path, where, tail, cap, subject)
        return raw, value, False, partial, consumed, False
    try:
        with path.open("rb") as stream:
            raw = stream.read(cap + 1)
    except FileNotFoundError:
        if kind != "storage":
            raise
        raw = b"{}"  # no count yet: the removers below are read all the same
    if len(raw) > cap:  # the bounded read proves only a lower bound
        raise _over_input(subject, None, cap)
    if kind == "storage":  # the saved counts beside each place's removers, read now (``storage_removers``)
        from ouroboros.storage_removers import place_removers

        value = {**json.loads(raw), "removers": place_removers(root)}
        raw = _canonical(value)
        return raw, value, True, False, len(raw), False
    return raw, json.loads(raw), False, False, len(raw), False


def _query_data(ctx: ToolContext, *, op: str, db="current", query="", kind="", ids=None,
                child_task_id="", state_file="review_job", sha256="", blob_kind="json", select="",
                filter=None, tail=100, max_rows=100, max_bytes=65536, snapshot=None,
                timeout_sec=1, save_as="", format="json", view_head_chars=None, view_tail_chars=None) -> str:
    from ouroboros.artifacts import store_task_artifact_bytes
    from ouroboros.tool_access import canonical_data_root
    from ouroboros.observability import read_blob_ref

    result, plane = {}, ""
    try:
        if op not in {"sql", "json", "jsonl", "blob"}:
            raise ValueError(f"{_arg('op', op)}: use sql, json, jsonl or blob")
        if format not in {"json", "text"}:
            raise ValueError(f"{_arg('format', format)}: use json or text")
        for name, value, low, high in (("max_bytes", max_bytes, 256, sql.MAX_BYTES), ("max_rows", max_rows, 1, 10000),
                                       ("tail", tail, 0, 10000)):
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{_arg(name, value)}: use an integer from {low} to {high}")
        if type(timeout_sec) not in (int, float) or not 0 < timeout_sec <= 10:
            raise ValueError("timeout_sec must be greater than zero and at most 10")
        if snapshot is not None and type(snapshot) is not bool:
            raise ValueError("snapshot must be boolean")
        if filter is not None and op != "jsonl":
            raise ValueError(f"{_arg('filter', filter)} applies only to op=jsonl; omit it")
        if filter is not None and (not isinstance(filter, dict)
                or set(filter) != {"field", "op", "value"} or filter["op"] not in {"eq", "contains"}
                or not isinstance(filter["field"], str) or not filter["field"]):
            raise ValueError("filter requires field, op=eq|contains and value (JSONL only)")
        for selector in (select, filter["field"] if filter else ""):
            if selector and (not isinstance(selector, str) or len(selector) > 512 or not _SELECTOR.fullmatch(selector)):
                raise ValueError("select/filter fields accept dotted keys, [n] and [*]")
        # Another operation's or kind's parameter is refused only when it asks for something;
        # its default value takes the omitted path.
        rows = {"jsonl": "tail", "sql": "max_rows"}.get(op)
        for owner, applies, given in (
                ("op=sql", op == "sql", (("db", db, db != "current"), ("query", query, bool(query)),
                                         ("snapshot", snapshot, snapshot is not None),
                                         ("max_rows", max_rows, max_rows != 100), ("timeout_sec", timeout_sec, timeout_sec != 1))),
                ("op=json or op=jsonl", op in {"json", "jsonl"}, (("kind", kind, bool(kind)), ("ids", ids, ids is not None))),
                ("op=blob", op == "blob", (("sha256", sha256, bool(sha256)), ("blob_kind", blob_kind, blob_kind != "json"))),
                ("op=jsonl", op == "jsonl", (("tail", tail, tail != 100),)),
                ("kind=skill_state", kind == "skill_state", (("state_file", state_file, state_file != "review_job"),))):
            named = [(name, value) for name, value, used in given if used]
            if named and not applies:
                hint = f"; {rows} bounds rows for op={op}" if rows and {"tail", "max_rows"} & {n for n, _ in named} else ""
                raise ValueError(f"{', '.join(_arg(*item) for item in named)} {'applies' if len(named) == 1 else 'apply'} "
                                 f"only to {owner}; omit {'it' if len(named) == 1 else 'them'}{hint}")
        if kind in {"storage", "update_letter", "update_material"} and child_task_id:
            raise ValueError(f"{kind} is installation-wide; it has no child selector")
        if kind == "knowledge_history" and child_task_id:
            raise ValueError("knowledge_history is the installation's global knowledge; it has no child selector")
        native = (ctx.task_metadata or {}).get("native_data_root")
        canonical, owner = Path(native or canonical_data_root(ctx)).resolve(), "native_pin" if native else "canonical"
        caller = root = Path(ctx.drive_root).resolve()
        plane = "child" if child_task_id else "caller"
        if child_task_id:
            try:
                root = _child_root(ctx, canonical, _name(child_task_id))
            except FileNotFoundError:
                if kind != "run":
                    raise
                root = None  # lineage held; a retained run outlives its execution drive (#1573)
        source = {"op": op, "child_task_id": child_task_id}
        reconstructed, partial = False, False
        if op == "sql":
            if db not in {"current", "money", "packs"} or not isinstance(query, str) or not query.strip() or len(query) > 16384:
                raise ValueError("sql requires db=current|money|packs and one query of at most 16384 characters")
            use_snapshot = False if snapshot is None else snapshot
            if db == "current" and not use_snapshot and timeout_sec > sql.CURRENT_HOLD_SEC:
                raise ValueError(f"a direct db=current read stops at {sql.CURRENT_HOLD_SEC:g} s because task writes wait "
                                 "on it; use snapshot=true for a longer query")
            local = root
            if db == "money" and not child_task_id and root != canonical:  # one ledger, the owner's (USAGE_STORE §4)
                root, plane = canonical, owner
            path = _plane_path(ctx, plane, root, sql.database_path(root, db).relative_to(root), local)
            absent = not any(path.parent.glob("*.members.json")) if db == "packs" else not path.is_file()
            if absent:
                result = {"status": "unavailable", "code": "no_packs" if db == "packs" else "not_found", "source": {"db": db}}
            else:
                source.update(db=db, query=query, snapshot=use_snapshot)
                result = sql.run_sql(ctx, root, db=db, query=query, max_rows=max_rows, max_bytes=max_bytes,
                                     snapshot=use_snapshot, timeout_sec=timeout_sec)
            value = result.get("data")
            raw, reconstructed = _canonical(value), True
        elif op == "blob":
            if not re.fullmatch(r"[0-9a-f]{64}", sha256) or blob_kind not in {"json", "txt", "bin"}:
                raise ValueError("blob requires a lowercase sha256 and blob_kind=json|txt|bin")
            # Local (caller or selected child) bytes first; the owner plane only when they are proven
            # absent there, never after a denied or corrupt read. Each loose or pack file opened stays
            # in its plane; an owner-plane file is itself admitted, not the digest's logical name.
            missed = []
            for plane, plane_root in [(plane, root), *([(owner, canonical)] if canonical != root else [])]:
                try:
                    raw = read_blob_ref(plane_root, dict(sha256=sha256, kind=blob_kind, encoding="gzip"),
                                        expected_kind=blob_kind, max_bytes=sql.MAX_BYTES, raw_bytes=True,
                                        admit=lambda physical: _plane_path(ctx, plane, plane_root, physical, root))
                    break
                except FileNotFoundError:
                    missed.append(plane)
                except PermissionError as exc:
                    raise PermissionError("".join(f"blob not found in the {p} plane; " for p in missed) + str(exc)) from None
            else:
                raise FileNotFoundError("blob not found in the " + " or ".join(missed) + " plane" + _BLOB_KIND_HINT)
            value = json.loads(raw) if blob_kind == "json" else {"hex": raw.hex()} if blob_kind == "bin" else raw.decode("utf-8")
            source.update(sha256=sha256, blob_kind=blob_kind)
        else:
            if kind not in _FILES and kind != "task_result":
                raise ValueError(f"{_arg('kind', kind)}: op={op} needs one of task_result, {', '.join(_FILES)}")
            if state_file not in _STATES:
                raise ValueError(f"{_arg('state_file', state_file)}: use one of {', '.join(_STATES)}")
            is_lines = kind.endswith("_log") or kind in {"verification_receipts", "knowledge_history"}
            if (op == "jsonl") != is_lines:
                raise ValueError(f"kind={kind} requires op=" + ("jsonl" if is_lines else "json"))
            identities = ids if ids is not None else ([ctx.task_id] if kind == "verification_receipts" else [""])
            if not isinstance(identities, list) or not 1 <= len(identities) <= 32:
                raise ValueError("ids must be an array of 1..32 identities")
            if kind in _ID_KINDS:
                identities = [_name(identity) for identity in identities]
            elif identities != [""]:
                raise ValueError("this kind has no IDs; use child_task_id for a child's logs or queue")
            if len(set(identities)) != len(identities):
                raise ValueError("ids must be unique")
            if kind in {"task_result", "review_ledger", "skill_state"}:
                root, plane = canonical, owner
            elif kind in {"queue_snapshot", "queue_dispatch"} and not child_task_id:
                # The installation queue lives with its owner; child_task_id keeps child-local meaning.
                root, plane = canonical, owner
                _plane_path(ctx, plane, root, _FILES[kind], caller)
            elif kind in {"storage", "knowledge_history", "update_letter", "update_material"}:
                root, plane = canonical, owner
                _plane_path(ctx, plane, root, _FILES[kind], caller)
            values, remaining, retained = {}, sql.MAX_BYTES, 0
            for identity in identities:
                raw, value, rebuilt, clipped, consumed, owned = _record(
                    ctx, root, kind, identity, _name(child_task_id or ctx.task_id) if kind == "run" else "",
                    state_file, op, filter, tail, remaining, canonical)
                retained += owned
                if consumed > remaining:  # history rows and a rebuilt task document are measured only here
                    history = kind == "knowledge_history"
                    raise _over_input(f"{kind} {identity}".strip() + (" without note bodies" if history else " as canonical JSON"),
                                      consumed, remaining,
                                      "keep fewer rows with tail or filter; select applies only after this check" if history else "")
                remaining -= consumed
                if select:
                    value = _select(value, select)
                    raw = _canonical(value)
                values[identity] = value
                reconstructed |= rebuilt
                partial |= clipped
            if retained:  # a retained run's terminal JSON lives in the owner's store (#1573)
                plane = owner if retained == len(identities) else f"{plane}+{owner}"
            if len(values) > 1:
                value, reconstructed = values, True
                raw = _canonical(value)
            source.update(kind=kind, ids=identities, state_file=state_file if kind == "skill_state" else None)
        if not result or result["status"] == "ok":
            if select:
                if op in {"sql", "blob"}:
                    value = _select(value, select)
                    raw = _canonical(value)
                source["select"] = select
            if op == "jsonl":
                source.update(filter=filter, tail=tail, scan_truncated=partial)
            if format == "text":
                if not isinstance(value, str):
                    raise ValueError("format=text requires exactly one selected string")
                raw = value.encode("utf-8")
            if (need := max(len(raw), selected_bytes(value))) > max_bytes:
                # Both the exact bytes and their canonical JSON must fit; format=text strings differ.
                sizes = f"{len(raw)} bytes" + (f" ({len(_canonical(value))} as canonical JSON)" if need != len(raw) else "")
                raise_to = (f"raise max_bytes to at least {need} (at most {sql.MAX_BYTES})" if need <= sql.MAX_BYTES
                            else f"no max_bytes admits it (at most {sql.MAX_BYTES})")
                narrow = {"jsonl": "select, filter or tail", "sql": "select or the query",
                          "json": "select" + (" or fewer ids" if len(source.get("ids") or ()) > 1 else ""),
                          "blob": "select" if blob_kind == "json" else ""}[op]
                raise ValueError(f"selected result is {sizes}, over max_bytes={max_bytes}; {raise_to}"
                                 + (f", or narrow it with {narrow}" if narrow else ""))
            source["plane"] = plane
            result = dict(status="ok", data=value, source=source, reconstructed=reconstructed)
            if save_as:
                receipt = store_task_artifact_bytes(ctx.drive_root, _name(ctx.task_id), _name(save_as), raw)
                result["saved"] = dict(path=receipt["path"], sha256=receipt["sha256"], size=receipt["bytes"],
                                       source=source, reconstructed=reconstructed)
    except (ValueError, TypeError, KeyError, IndexError, OSError, RuntimeError, RecursionError) as exc:
        code = "access_denied" if isinstance(exc, PermissionError) else "not_found" if isinstance(exc, FileNotFoundError) else "query_failed"
        result = dict(status="blocked" if code == "access_denied" else "unavailable" if code == "not_found" else "error",
                      code=code, message=str(exc))
    if plane and result["status"] != "ok":  # a refusal or gap names the plane it concerns, like a result
        result.setdefault("plane", plane)
    text = _canonical(result).decode("utf-8")
    code = {"ok": "OK", "error": "TOOL_ERROR", "blocked": "ACCESS_BLOCKED", "unavailable": "CAPABILITY_UNAVAILABLE"}[result["status"]]
    return _publish_tool_result(ctx, ToolResult(status=result["status"], code=code, text=text))


def get_tools():
    return [ToolEntry("query_data", {"name": "query_data", "description": (
        "Read runtime facts, money, pack indexes, typed JSON/JSONL and verified blobs. No paths accepted. "
        "db, query, max_rows, snapshot and timeout_sec are op=sql only: omit them otherwise (another op refuses "
        "any snapshot and other non-default values); op=jsonl bounds rows with tail. "
        "Current SQL exposes tasks facts; use json/task_result for bodies. SQL defaults to guarded direct reads. "
        f"Current-store reads hold the lock task writes wait on: at most {sql.CURRENT_HOLD_SEC:g} s; longer: snapshot=true. "
        "One SQL operation budget spans acquisition, optional snapshot copy/check and query; deadlines name the phase. "
        "The supervised child bounds functions that outlast interrupt, with separate startup/cleanup allowance. "
        "save_as retains exact whole-record bytes in the calling task's artifact_store with path/sha256/size/source. "
        "Projections use sorted UTF-8 JSON; store documents and multi-ID bundles are flagged reconstructed. "
        "format=text saves one selected string exactly. Complete tool output is retained before any delivery cut. "
        "kind=knowledge_history lists global knowledge writes (when, topic, mode, writer, focus, task, summary, sizes, "
        "change delta) without the note bodies, about 1 KB a row (raise max_bytes or lower tail); knowledge_read gives the current text. "
        "Results name their data plane; the owner's default queue files, money ledger and blob fallback need the caller's existing read of each file."),
        "parameters": {"type": "object", "required": ["op"], "properties": {
            "op": {"enum": ["sql", "json", "jsonl", "blob"], "type": "string"},
            "db": {"enum": ["current", "money", "packs"], "type": "string", "default": "current",
                   "description": "op=sql only; another op accepts only the default current, as if omitted."},
            "query": {"type": "string", "maxLength": 16384, "description": "One read-only SQL statement; BLOB cells return {hex:...}."},
            "kind": {"type": "string", "enum": ["task_result", *_FILES]},
            "ids": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 32,
                    "description": "Task, run, review-record or skill names; one returns its value, several return an ID map."},
            "child_task_id": {"type": "string", "description": "Owned descendant's execution drive; default caller drive. Task documents/ledger/skills and the default queue files use the installation owner's state."},
            "state_file": {"type": "string", "enum": list(_STATES), "default": "review_job", "description": "kind=skill_state only."},
            "sha256": {"type": "string", "description": "Blob SHA-256; the host locates loose or packed bytes, locally first, then the owner's when absent and readable."},
            "blob_kind": {"type": "string", "enum": ["json", "txt", "bin"], "default": "json", "description": "op=blob only."},
            "select": {"type": "string", "description": "Dotted keys, [n], [*], e.g. primary_output.text or [*].tool. Applied per ID after filter/tail."},
            "filter": {"type": "object", "required": ["field", "op", "value"], "additionalProperties": False,
                       "description": "op=jsonl only; keeps whole rows before tail.",
                       "properties": {"field": {"type": "string"}, "op": {"type": "string", "enum": ["eq", "contains"]}, "value": {}}},
            "tail": {"type": "integer", "minimum": 0, "maximum": 10000, "default": 100,
                     "description": "JSONL: last matching rows; 0 reads from start. An 8 MiB scan window shared by ids (256 MiB for knowledge_history) reports scan_truncated."},
            "max_rows": {"type": "integer", "minimum": 1, "maximum": 10000, "default": 100, "description": (
                "op=sql only: rows returned before truncated=true. Another op accepts only the default 100, "
                "as if omitted; op=jsonl bounds rows with tail.")},
            "max_bytes": {"type": "integer", "minimum": 256, "maximum": sql.MAX_BYTES, "default": 65536, "description": (
                "Output budget after select/filter/tail (canonical JSON counts); op=sql stops fetching rows at it "
                f"(truncated=true) before select. The fixed {sql.MAX_BYTES >> 20} MiB input budget, shared by ids, "
                "is separate and precedes select.")},
            "snapshot": {"type": "boolean", "default": False, "description": "SQL: opt in to a checked disposable snapshot; default false for every db. Copy/check share the operation budget."},
            "timeout_sec": {"type": "number", "exclusiveMinimum": 0, "maximum": 10, "default": 1,
                            "description": "SQL operation seconds across acquisition/copy/check/query, not per phase. Executor startup/cleanup is separately bounded."},
            "save_as": {"type": "string", "description": "One plain immutable artifact filename in the CALLING task."},
            "format": {"type": "string", "enum": ["json", "text"], "default": "json"},
            **RESULT_VIEW_PARAMS,
        }}}, _query_data, timeout_sec=30)]
