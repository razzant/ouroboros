"""Identity, the world map and a project's workpad: one editor behind update_identity, update_world and workpad_write.

The three plain memory texts take knowledge_write's arguments for the same jobs through its one
parser (``knowledge_edits.normalize_arguments``/``check_arguments``): ``mode`` overwrite (the
default) replaces the whole text with ``content``; ``edit`` applies exact ``{old_str, new_str}``
items, each found once in the revision read, all or none (``old_str`` with ``content`` or ``new_str``
is the one-item spelling; a refusal names the item and its lines: ``compile_edit_items``); ``revert``
puts back an earlier ``version``, verified by its sha256; ``reason`` is kept in history, never in the
text. ``memory_publication.publish_text`` publishes under the text's lock. A ``TextSource`` names
what differs between them: the file and its journal, the writer and its own arguments, whether rows
store changes against a verified base (the workpad) or both texts whole (identity, the map), whether
an empty result is refused (identity, the map), whether an overwrite may be blind (the workpad), and
the host's measured block (the map: ``world_profiler.edited_map``/``reverted_map``).

Every answer is a typed result. A receipt (``OK``) names the version the write replaced (who wrote
it, when and why, or that no history row wrote it), the characters removed and added, the headings
and sections that left, and the calls that read and put back that version; the workpad's states the
page's size and that it enters every request of its project whole, a fact without a threshold. A
blind overwrite of a workpad version this task did not write, or that no row names, landed: its
receipt comes with the code ``LEGACY_WARNING``, so a finishing turn reads it. A stale revision gets
every change since the held version in whole lines when the journal's tail holds it, else the whole
current text, else, when that does not fit the result, the exact read; never a clipped diff.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Optional

from ouroboros import knowledge_versions
from ouroboros.knowledge_edits import EditRefused, check_arguments, compile_edit_items, edit_items, normalize_arguments
from ouroboros.memory_publication import Publication, Refused, publish_text
from ouroboros.tools.tool_result import ToolResult, _publish_tool_result

MODES = ("overwrite", "edit", "revert")


@dataclass(frozen=True)
class TextSource:
    """One plain memory text: what its writer, its history and its version reads need to know about it."""

    name: str  # memory_read's source address: identity, world or workpad:<project id>
    noun: str  # what the answers call it
    path: Path  # the text itself
    journal: Path  # its history journal
    tool: str  # its writer
    arguments: tuple = ()  # the writer's own arguments a call names beside mode/version (the workpad's project_id)
    topic: Optional[str] = None  # the rows' topic: the workpad's carry one; a journal of one text needs none
    changes: bool = False  # rows store the change against a verified base (knowledge_rows), else both texts whole
    nonempty: bool = True  # an empty or whitespace-only result is an argument error
    blind: bool = False  # an overwrite may replace an existing text without expected_revision
    host: Optional[tuple] = None  # the map's (repo_dir, data_dir): the measured block stays the host's
    mirror: Optional[Path] = None  # a forked drive's copy refreshed after each publication (the map)
    resident: str = ""  # where the text rides whole, said in every receipt (the workpad)
    footer: str = ""  # a line every answer ends with (identity's canonical reader)
    where: str = ""  # where the text lives, said by a refusal of a missing or stale revision (the map, beside a fork's copy)
    outside: str = "an outside edit, or a write before history"  # what changes the text without a row


@dataclass(frozen=True)
class Deferred:
    """A source located only after its call's arguments parse, so an argument error never depends on a root:
    ``name``, ``noun`` and ``nonempty`` are all the parse reads, ``build`` resolves the rest."""

    name: str
    noun: str
    nonempty: bool
    build: Any  # () -> TextSource


def identity_source(root: Path, footer: str = "") -> TextSource:
    from ouroboros.memory import Memory

    memory = Memory(drive_root=root)
    return TextSource("identity", "identity", memory.identity_path(), memory.identity_journal_path(), "update_identity",
                      footer=footer)


def world_source(root: Path, repo_dir: str = "", drive_root: Any = None) -> TextSource:
    from ouroboros.memory import Memory

    memory = Memory(drive_root=root)
    copy = Memory(drive_root=Path(drive_root)).world_path() if drive_root is not None else None
    mirror = copy if copy is not None and copy.resolve(strict=False) != memory.world_path().resolve(strict=False) else None
    return TextSource("world", "world map", memory.world_path(), memory.world_journal_path(), "update_world",
                      host=(repo_dir, str(root)), mirror=mirror,
                      where=f"The map lives at {memory.world_path()}; a forked drive holds only a copy of it.",
                      outside="a change outside the writer, such as the host's refresh of its measured block at "
                              "startup, which writes no row, or a write before history")


def workpad_source(project_id: str) -> TextSource:
    from ouroboros.project_facts import project_workpad_history_path, project_workpad_path

    return TextSource(f"workpad:{project_id}", "workpad", project_workpad_path(project_id),
                      project_workpad_history_path(project_id), "workpad_write", (("project_id", project_id),),
                      topic="workpad", changes=True, nonempty=False, blind=True,
                      resident=f"it enters every request of project {project_id} whole")


def edit_properties(noun: str, expected: str) -> dict:
    """A writer schema's parameters for the shared modes, in knowledge_write's spelling; the tool's lead explains them
    (parameter prose does not reach every route), so each says only which mode takes it."""
    text = {"type": "string"}
    return {
        "mode": {**text, "enum": list(MODES)},
        "edits": {"type": "array", "minItems": 1, "description": "mode=edit: exact {old_str, new_str} items, all or none.",
                  "items": {"type": "object", "additionalProperties": False, "required": ["old_str", "new_str"],
                            "properties": {"old_str": text, "new_str": text}}},
        "old_str": {**text, "description": "mode=edit, one pair: exact text found once."},
        "new_str": {**text, "description": "old_str's replacement, as content (empty deletes)."},
        "content": {**text, "description": f"overwrite: the whole new {noun}; edit with old_str: its replacement."},
        "version": {**text, "description": "mode=revert: the revision to put back."},
        "reason": {**text, "description": "Why: kept in history, never in the text."},
        "expected_revision": {**text, "description": expected},
    }


def revision_of(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def read_call(source: TextSource, version: str) -> str:
    """The exact read of one version (the current one included) of this text."""
    return f"memory_read(source={source.name!r}, version={version!r})"


def write_call(source: TextSource, **arguments: Any) -> str:
    return f"{source.tool}(" + ", ".join(f"{key}={value!r}" for key, value in (*source.arguments, *arguments.items())) + ")"


@dataclass(frozen=True)
class PlainText:
    """A text as ``knowledge_versions.changes_since`` reads one: its bytes, text, headings, revision and source."""

    raw: bytes
    text: str
    source: Any
    revision: str
    address: TextSource


def plain_text(source: TextSource, text: str) -> PlainText:
    from ouroboros.memory_publication import _parsed

    parsed = _parsed(text)
    return PlainText(parsed.raw, text, parsed.source, revision_of(parsed.raw), source)


def _prepare(source: TextSource, mode: str, content: str, items: list | None, version: str | None, target: Any):
    """The new text from the current one, under the lock: the edit compiled over the current bytes, the version put
    back, the result checked, then the map's measured block kept the host's."""

    def prepare(old: str) -> str:
        if mode == "edit":
            try:
                text = compile_edit_items(old.encode("utf-8"), 0, items).decode("utf-8")
            except EditRefused as exc:
                raise Refused("edit_refused", str(exc)) from None
        elif mode == "revert":
            if revision_of(old.encode("utf-8")) == version:
                text = old  # the current version reads even when no row names it
            elif isinstance(target, Exception):
                raise Refused("version_unavailable", str(target))
            else:
                text = target
        else:
            text = content
        if source.nonempty and not text.strip():  # judged before a generated host block could fill it
            raise Refused("empty_result", f"the {mode} leaves the {source.noun} empty or whitespace only; nothing was "
                                          f"written ({source.noun} is never empty: write what it should say)")
        if source.host is None:
            return text
        from ouroboros import world_profiler

        repo_dir, data_dir = source.host
        if mode == "edit":
            try:
                return world_profiler.edited_map(text, old, items, repo_dir, data_dir)
            except ValueError as exc:
                raise Refused("host_block", str(exc)) from None
        if mode == "revert":
            return world_profiler.reverted_map(text, old, repo_dir, data_dir)
        return world_profiler.with_measured_block(text, old, repo_dir, data_dir)

    return prepare


def _answer(ctx: Any, source: TextSource, text: str, *, status: str = "ok", code: str = "OK") -> str:
    footer = f"\n{source.footer}" if source.footer else ""
    return _publish_tool_result(ctx, ToolResult(status=status, code=code, text=text + footer))


def _size(source: TextSource, text: str) -> list[str]:
    if not source.resident:
        return []
    return [f"Size: {len(text)} chars, about {len(text) // 4} tokens (chars/4); {source.resident}."]


def _receipt(source: TextSource, mode: str, publication: Publication, version: str | None) -> list[str]:
    """What the write did: the version it replaced, what left and came, and the calls that read and put back."""
    if publication.reason == "unchanged":
        return [f"OK: {source.noun} unchanged ({mode}): the result is the current text, so nothing was written; "
                f"revision={publication.revision}."] + _size(source, publication.content)
    verb = "updated" if publication.existed else "created"
    lines = [f"OK: {source.noun} {verb} ({mode}, {len(publication.content)} chars); revision={publication.revision}."]
    if mode == "revert":
        lines.append(f"Put back version {version}." if publication.revision == version or source.host is None else
                     f"Put back version {version}: the map's own part of it with the host's current measured block, so "
                     "the new revision is not that version's.")
    lines += _size(source, publication.content)
    parts = knowledge_versions.delta_parts({"delta": publication.delta})
    if not publication.existed:
        return lines + ([f"Written: {' · '.join(parts)}."] if parts else [])
    old = publication.previous["revision"]
    lines += [f"Replaced {old}: {knowledge_versions.written_text(publication.previous, source.outside)}.",
              f"Changed: {' · '.join(parts)}.",
              f"Read the replaced version: {read_call(source, old)}. Put it back: "
              f"{write_call(source, mode='revert', version=old, expected_revision=publication.revision)}."]
    return lines


def _current(source: TextSource, publication: Publication, head: str) -> str:
    """A refusal that resends the current text: every change since a held version, else the whole text, else its read."""
    from ouroboros.tool_capabilities import tool_result_limit

    limit, current = tool_result_limit(source.tool), publication.revision
    if publication.held is not None:
        changes = head + "\n\n" + knowledge_versions.changes_since(
            plain_text(source, publication.content), publication.held,
            [f"The complete current text is {read_call(source, current)}."])
        if len(changes) <= limit:
            return changes
    whole = head + f"\n\nThe current {source.noun} (revision {current}, {len(publication.content)} chars) follows.\n\n"
    if len(whole) + len(publication.content) <= limit:
        return whole + publication.content
    return head + f" The current {source.noun} ({len(publication.content)} chars) does not fit this result: read it " \
                  f"with {read_call(source, current)}."


def _refusal(ctx: Any, source: TextSource, mode: str, publication: Publication) -> str:
    reason, current = publication.reason, publication.revision
    if reason in ("revision_required", "revision_conflict"):
        head = (f"⚠️ RESOURCE_CONSTRAINT_BLOCKED: {reason}; current_revision={current!r}. Nothing was written. "
                + ("An existing text is replaced only against the revision you read" if reason == "revision_required"
                   else "Reconcile your change with what came since") + f": pass expected_revision={current!r}."
                + (f" {source.where}" if source.where else ""))
        return _answer(ctx, source, _current(source, publication, head), status="blocked",
                       code="RESOURCE_CONSTRAINT_BLOCKED")
    if reason == "history_unavailable":
        return _answer(ctx, source, (
            f"⚠️ RESOURCE_CONSTRAINT_BLOCKED: history_unavailable; current_revision={current!r}. Its history journal "
            f"could not record this version, so the {source.noun} was not changed; the same call can be retried."),
            status="blocked", code="RESOURCE_CONSTRAINT_BLOCKED")
    if reason == "empty_result":
        return _answer(ctx, source, f"⚠️ TOOL_ARG_ERROR: {publication.detail}.", status="error", code="TOOL_ARG_ERROR")
    if reason.endswith("_source_missing"):
        text = (f"⚠️ TOOL_ERROR: {reason}: the {source.noun} does not exist, so there is nothing to {mode}; "
                f"mode=overwrite creates it. Nothing was written.")
    elif reason == "replace_failed":
        text = (f"⚠️ TOOL_ERROR: replace_failed ({publication.detail}): the {source.noun} keeps its previous bytes, "
                f"current_revision={current!r}; its history row names the version that did not land. The same call "
                "can be retried.")
    else:  # edit_refused, host_block, version_unavailable: the caller holds the current revision; it is not resent
        text = f"⚠️ TOOL_ERROR: {reason}: {publication.detail.rstrip('.')}. Nothing was written; current_revision={current!r}."
    return _answer(ctx, source, text, status="error", code="TOOL_REPORTED_FAILURE")


def _foreign(ctx: Any, publication: Publication) -> bool:
    """A replaced version this task did not write, or one no row in the journal's tail names: the host's focus identity
    of the row that wrote it against this task's, never a model-supplied author."""
    focus = (publication.writer or {}).get("focus")
    task = str(getattr(ctx, "task_id", "") or "")
    return not (task and isinstance(focus, dict) and str(focus.get("task_id") or "") == task)


def _mirror(source: TextSource, publication: Publication) -> None:
    """A forked drive's copy (what a child's read_file opens) follows what was just published."""
    if source.mirror is None or not source.mirror.exists():
        return
    try:
        from ouroboros.utils import write_bytes_atomic

        write_bytes_atomic(source.mirror, publication.content.encode("utf-8"))
    except OSError:  # the text already landed canonically; a stale copy only costs the next read a refusal
        pass


def write_text(ctx: Any, source: Any, *, content: Any = None, mode: Any = "overwrite",
               expected_revision: Any = None, old_str: Any = None, new_str: Any = None, edits: Any = None,
               version: Any = None, reason: Any = "", source_type: str = "task", notices: Any = None) -> str:
    """One write of a plain memory text, parsed, published and answered (the tool handlers delegate here).

    ``source`` is a ``TextSource`` or a ``Deferred`` one, located after the arguments parse.
    ``notices(mode, content, publication)`` adds the source's own lines to a receipt (the map's)."""
    notes: list[str] = []
    version, edits, old_str, reason = normalize_arguments(mode, version, expected_revision, edits, old_str, reason, notes)
    try:
        content, old_str, edits = check_arguments(mode, content, old_str, new_str, edits, version, reason, notes,
                                                  modes=MODES, version_reader=f"memory_read(source={source.name!r}, "
                                                                              "version=...)")
        if expected_revision is not None and not isinstance(expected_revision, str):
            raise ValueError("expected_revision must be the revision text a read, a receipt or a refusal returned")
        items = edit_items(edits) if edits is not None else [(old_str, content)] if mode == "edit" else None
        if mode == "overwrite" and source.nonempty and not content.strip():  # known from the arguments alone
            raise ValueError(f"content is empty or whitespace only; the {source.noun} is never empty: send what it "
                             "should say, or change parts of it with mode=edit")
    except ValueError as exc:
        said = f" ({'; '.join(notes)})" if notes else ""
        return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR",
                                                    text=f"⚠️ TOOL_ARG_ERROR: {exc}{said}"))
    if isinstance(source, Deferred):
        source = source.build()
    target: Any = None
    if mode == "revert":  # outside the lock: a recorded row never changes, and the text is verified by its hash
        try:
            target = knowledge_versions.find_version(source, version)[0]
        except knowledge_versions.VersionUnavailable as exc:
            target = exc
    from ouroboros.knowledge import focus_signature

    publication = publish_text(source, expected_revision, _prepare(source, mode, content, items, version, target),
                               author=focus_signature(ctx), source_type=source_type, mode=mode, reason=reason,
                               reverted_from=version if mode == "revert" else None)
    if not publication.ok:
        return _refusal(ctx, source, mode, publication)
    lines = _receipt(source, mode, publication, version)
    if notices is not None:
        lines += notices(mode, content, publication)
    lines += [f"[Argument note] {item}" for item in notes]
    if publication.reason == "saved":
        _mirror(source, publication)
    if (publication.reason == "saved" and mode == "overwrite" and expected_revision is None and publication.existed
            and _foreign(ctx, publication)):
        written = knowledge_versions.written_text(publication.previous, source.outside)
        warning = (f"⚠️ BLIND_OVERWRITE: this overwrite replaced a {source.noun} version this task did not write "
                   f"({written}) without expected_revision. The write landed; what left is below, with the calls "
                   "that read and put back the replaced version.")
        return _answer(ctx, source, "\n".join([warning, *lines]), code="LEGACY_WARNING")
    return _answer(ctx, source, "\n".join(lines))


# --- memory_read(source=...): the version door ------------------------------------------------------------------------

SOURCES = "identity, world, workpad (this task's project) or workpad:<project id>"


class _ScopeRefused(ValueError):
    """The workpad's read scope refuses this address: ``code`` is the resolver's own (``project_journal._project_scope``)."""

    def __init__(self, code: str, text: str) -> None:
        super().__init__(text)
        self.code = code


def resolve_source(ctx: Any, address: Any) -> TextSource:
    """A source address as memory_read names it: identity and the map on the canonical root; a workpad through the
    project read scope (``project_journal._project_scope(write=False)``), on the page's own root."""
    if address in ("identity", "world"):
        from ouroboros.tool_access import canonical_data_root

        root = canonical_data_root(ctx)
        return identity_source(root) if address == "identity" else world_source(root)
    if isinstance(address, str) and (address == "workpad" or address.startswith("workpad:")):
        from ouroboros.tools.project_journal import _project_scope

        pid, refusal, code = _project_scope(ctx, address[len("workpad:"):], write=False)
        if refusal:
            raise _ScopeRefused(code, refusal)
        if not pid:
            raise ValueError("source='workpad' is this task's project, and this task has none: name one as "
                             "workpad:<project id>")
        return workpad_source(pid)
    raise ValueError(f"source={address!r}: a memory text is {SOURCES}")


def _page(head: str, text: str, start: int, resume: Any) -> str:
    """One character window of a document; the second line names the continuation or completeness."""
    from ouroboros.tool_capabilities import tool_result_limit

    total = len(text)
    start = min(start, total)
    room = max(tool_result_limit("memory_read") - len(head) - 300, 1)  # the continuation line fits beside it
    end = min(start + room, total)
    cont = (f"chars {start}–{end} of {total}; next_start={end}; " + resume(end) if end < total
            else f"chars {start}–{end} of {total}; complete")
    return head + "\n" + cont + "\n" + text[start:end]


def read_versions(ctx: Any, address: Any, version: Any, start: Any, fingerprint: Any, others: list[str]) -> str:
    """``memory_read(source=...)``: one version of a memory text, or the list of its versions; never the chronicle.

    With ``version``: that version's exact text, verified by its sha256 (the current bytes read even when no row
    names them; an earlier one reads even when the file is gone), paged by ``start``. Without: the versions newest
    first (``knowledge_versions.render_listing``: who, when, mode, reason, what changed and the version each write
    replaced), paged by ``start`` with the ``fingerprint`` of the list a page continues; a list that changed between
    pages is refused, never joined. Any other selector beside ``source`` is refused, never silently resolved."""

    def failed(code: str, text: str, status: str = "error") -> str:
        return _publish_tool_result(ctx, ToolResult(status=status, code=code, text=text,
                                                    meta={"operation_outcome": "completed_no_effect"}))

    note = ""
    if isinstance(version, str) and not version.strip():  # a blank value names no revision: the list, said so
        from ouroboros.tools.arg_feedback import ignored_argument_note

        note, version = f"\n[Argument note] {ignored_argument_note('version', version, 'no revision named; the list of versions follows')}", None
    try:
        if others:
            raise ValueError(f"source reads a memory text's versions, with version, start and fingerprint only; drop "
                             f"{', '.join(others)}")
        if address is None:
            raise ValueError(f"version reads one version of a memory text: name its source ({SOURCES}); a chronicle "
                             "record's revision is revision= with node_id")
        if type(start) is not int or start < 0:
            raise ValueError("start is a non-negative integer")
        if version is not None and (not isinstance(version, str) or fingerprint):
            raise ValueError("version is a revision text; fingerprint continues the list of versions, read without one"
                             if fingerprint else "version is a revision text (the sha256 of that version)")
        source = resolve_source(ctx, address)
        raw = source.path.read_bytes() if source.path.exists() else None
    except _ScopeRefused as exc:
        return failed(exc.code, str(exc), "blocked" if exc.code == "ACCESS_BLOCKED" else "error")
    except ValueError as exc:
        return failed("TOOL_ARG_ERROR", f"⚠️ TOOL_ARG_ERROR: {exc}")
    except OSError as exc:
        return failed("TOOL_REPORTED_FAILURE", f"⚠️ TOOL_ERROR: {address} unreadable: {type(exc).__name__}")
    current = revision_of(raw) if raw is not None else ""
    shown = f"current revision {current}" if raw is not None else "the text does not exist now"
    if version is not None:
        try:
            text, row = knowledge_versions.find_version(source, version)
        except knowledge_versions.VersionUnavailable as exc:
            if version != current:
                return failed("TOOL_REPORTED_FAILURE", f"⚠️ TOOL_ERROR: version_unavailable: {exc}")
            text, row = raw.decode("utf-8"), {}
        which = "the current text" if version == current else "an earlier version, not the current text"
        written = knowledge_versions.written_text(knowledge_versions.row_facts(row), source.outside)
        head = f"[{source.name} version {version}: {which}; {written}; {shown}; {len(text)} chars]"
        return _publish_tool_result(ctx, ToolResult(status="ok", code="OK", text=_page(
            head, text, start, lambda end: f"memory_read(source={source.name!r}, version={version!r}, start={end})"),
            meta={"operation_outcome": "completed_no_effect"}))
    listing = knowledge_versions.render_listing(
        source, current or "none (the text does not exist now)", f"History of {source.name} ({source.noun})",
        read_call(source, "<revision>").replace("'<revision>'", "<revision>"),
        write_call(source, mode="revert", version="<revision>", expected_revision=current or "<revision>")
        .replace("'<revision>'", "<revision>"))
    digest = sha256(listing.encode("utf-8")).hexdigest()[:16]
    if (start or fingerprint) and fingerprint != digest:
        return failed("TOOL_REPORTED_FAILURE", "⚠️ TOOL_ERROR: position_mismatch: that page belongs to another list of "
                                               "versions (a write landed since): read it again from start=0.")
    head = f"[{source.name}: versions newest first; {shown}; fingerprint='{digest}']{note}"
    return _publish_tool_result(ctx, ToolResult(status="ok", code="OK", text=_page(
        head, listing, start, lambda end: f"memory_read(source={source.name!r}, start={end}, fingerprint='{digest}')"),
        meta={"operation_outcome": "completed_no_effect"}))
