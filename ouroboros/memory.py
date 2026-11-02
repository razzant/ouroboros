from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from ouroboros.contracts.chat_id_policy import is_a2a_chat_id
from ouroboros.utils import append_jsonl, atomic_write_json, read_json_dict, read_text, short, utc_now_iso, write_text
from ouroboros.platform_layer import (
    file_lock_exclusive as _lock_ex,
    file_lock_shared as _lock_sh,
    file_unlock as _unlock,
)

log = logging.getLogger(__name__)

_DIALOGUE_LOGS = frozenset({"chat.jsonl", "progress.jsonl"})  # their rows live in state/dialog.sqlite
# The rows one recent window reads at most, host waits included: a read bound, not a quota.
_RECENT_ROWS_CEILING = 2000


def scratchpad_block_sha256(block: Mapping[str, Any]) -> str:
    """The sha256 of one block's exact stored JSON (sorted keys, no spaces): content addressing.

    It names the bytes a reader saw, never a host-minted identity, so a fold that names it
    replaces exactly those bytes or nothing; byte-identical blocks share it."""
    encoded = json.dumps(block, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class ScratchpadFoldFailed(RuntimeError):
    """A fold that changed no active block: ``not_active`` or ``ambiguous`` (no exact selection of the
    blocks read under the lock, ``active``), ``journal_unavailable`` or ``write_failed`` (``kind``)."""

    def __init__(self, kind: str, detail: str, *, digests: Any = (), active: Any = ()):
        super().__init__(detail)
        self.kind, self.digests, self.active = kind, list(digests), list(active)


class ScratchpadProjectionStale(RuntimeError):
    """The canonical blocks changed and stay changed; regenerating ``memory/scratchpad.md`` failed. Raised
    after the write's journal row: the caller reports the applied change (``result``) beside the stale file."""

    def __init__(self, blocks: List[Dict[str, Any]], error: BaseException):
        super().__init__(f"{type(error).__name__}: {error}")
        self.blocks, self.result = blocks, None


def _fold_span(blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The period a fold's sources cover; an earlier fold contributes its own span."""
    stamps: List[str] = []
    for block in blocks:
        fold = block["metadata"].get("fold") if isinstance(block.get("metadata"), dict) else None
        covers = fold["covers"] if isinstance(fold, dict) and isinstance(fold.get("covers"), dict) else {}
        stamps += [str(covers.get(end) or block.get("ts") or "") for end in ("from", "to")]
    return {"from": min(stamps), "to": max(stamps), "blocks": len(blocks)}  # ISO-8601 UTC sorts as text


def render_scratchpad_markdown(
    blocks: List[Dict[str, Any]], *, journal_reader: str, journal_pointer: bool = False,
) -> str:
    """Render scratchpad blocks the one way the runtime renders them.

    SINGLE SOURCE OF TRUTH for the block markdown: the writer
    (Memory._write_scratchpad_markdown, which persists memory/scratchpad.md) and
    context (ouroboros/context.py::_scratchpad_for_context, which renders the
    canonical blocks whole) call this. Blocks arrive oldest-first (storage order)
    and are rendered newest-first, which is the order the model has always read;
    each heading names the block's sha256, the address a fold uses.

    ``journal_pointer`` adds the retired/replaced-blocks pointer line; the
    writer passes ``Memory.journal_path().exists()`` for it. ``journal_reader``
    (``Memory.scratchpad_journal_reader``) is how the reader of this text opens
    that journal; every fold's line, an older one's included, shows it, not the stored recipe.
    """
    chars = sum(len(str(block.get("content", ""))) for block in blocks)
    parts = [f"## Scratchpad (working memory — {len(blocks)} blocks, {chars:,} characters)\n",
             "Nothing leaves by age or size: a block leaves only through a fold I accept "
             "(`update_scratchpad` with `replaces` = the sha256 after its heading).\n"]
    if journal_pointer:
        parts.append(f"Exact retired/replaced source blocks remain readable with {journal_reader}.\n\n")
    for block in reversed(blocks):
        ts = str(block.get("ts", ""))[:16]
        source = block.get("source", "?")
        content = block.get("content", "")
        parts.append(f"### [{ts} — {source}] sha256 {scratchpad_block_sha256(block)}\n{content}\n\n---\n")
        metadata = block.get("metadata") if isinstance(block.get("metadata"), dict) else {}
        fold = metadata.get("fold") if isinstance(metadata.get("fold"), dict) else {}
        if fold:
            covers, focus = (value if isinstance(value, dict) else {}
                             for value in (fold.get("covers"), metadata.get("focus")))
            draft = f"; helper draft: {fold['draft_source']}" if fold.get("draft_source") else ""
            parts.append(f"Fold of {covers.get('blocks', '?')} blocks ({str(covers.get('from', ''))[:16]}.."
                         f"{str(covers.get('to', ''))[:16]}), accepted by {focus.get('role') or 'unknown'} "
                         f"{focus.get('task_id') or ''}".rstrip() + f"{draft}.\n")
        source_ref = metadata.get("source_ref") if isinstance(metadata.get("source_ref"), dict) else {}
        entry_id = str(source_ref.get("entry_id") or "")
        if entry_id:
            parts.append(f"Exact replaced blocks: {journal_reader}; locate `entry_id={entry_id}`.\n\n")
    return "\n".join(parts)


def _history_timestamp(value: Any, *, field: str = "ts") -> datetime:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _chat_history_filter(filters: Mapping[str, str], search: str):
    exact_transport = {
        key: str(filters.get(key) or "").strip()
        for key in ("provider", "account_id", "conversation_id", "thread_id")
    }
    actor_id = str(filters.get("actor_id") or "").strip()
    date_from = (
        _history_timestamp(filters.get("date_from"), field="date_from")
        if filters.get("date_from") else None
    )
    date_to = (
        _history_timestamp(filters.get("date_to"), field="date_to")
        if filters.get("date_to") else None
    )
    search_lower = str(search or "").lower()

    def matches(entry: Mapping[str, Any]) -> bool:
        if is_a2a_chat_id(entry.get("chat_id")):
            return False
        if search_lower:
            from ouroboros.dialogue_provenance import dialogue_text

            if search_lower not in dialogue_text(entry).lower():
                return False
        transport = entry.get("transport") if isinstance(entry.get("transport"), Mapping) else {}
        if any(value and str(transport.get(key) or "") != value for key, value in exact_transport.items()):
            return False
        actor = transport.get("actor") if isinstance(transport.get("actor"), Mapping) else {}
        actor_values = {
            str(value) for value in (
                actor.get("platform_actor_id"), actor.get("id"), entry.get("sender_session_id")
            ) if value not in (None, "")
        }
        if actor_id and actor_id not in actor_values:
            return False
        if date_from is not None or date_to is not None:
            try:
                row_ts = _history_timestamp(entry.get("ts"))
            except ValueError:
                return False
            if date_from is not None and row_ts < date_from:
                return False
            if date_to is not None and row_ts > date_to:
                return False
        return True

    return matches


def _normalized_chat_history_query(filters: Mapping[str, str], search: str) -> Dict[str, str]:
    """Canonical actor-query identity for one chat-history snapshot."""

    normalized = {
        key: str(filters.get(key) or "").strip()
        for key in ("provider", "account_id", "conversation_id", "thread_id", "actor_id")
    }
    normalized["search"] = str(search or "").lower()
    for key in ("date_from", "date_to"):
        value = filters.get(key)
        normalized[key] = _history_timestamp(value, field=key).isoformat() if value else ""
    return normalized


def _chat_history_snapshot_id(
    coverage: Mapping[str, Any], filters: Mapping[str, str], search: str,
) -> str:
    """Hash one exact physical generation snapshot plus its normalized query."""

    generations = []
    for row in coverage.get("generations") or []:
        if not isinstance(row, Mapping):
            continue
        generations.append({
            "name": pathlib.Path(str(row.get("path") or "")).name,
            "first_line_sha256": str(row.get("first_line_sha256") or ""),
            "size": int(row.get("size") or 0),
        })
    payload = {
        "schema_version": 1,
        "query": _normalized_chat_history_query(filters, search),
        "generations": generations,
        # Existing consolidator gap blocks are the durable truth that an older
        # span is unknowable after its cursor is rebased.  Their small stable IDs
        # participate in the stateless token; no block-content/full-file hashing
        # or new continuation state is introduced.
        "durable_gap_ids": [
            str(value) for value in (coverage.get("durable_gap_ids") or [])
        ],
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()



def _ok_answer_mark(bucket: Any) -> str:
    """✓ for an ordinary success. A typed code that answers ok yet names another outcome
    (an inspection before apply, a review block) shows its bucket: the code table decides."""
    from ouroboros._outcome_tool_errors import _OK_TOOL_STATUSES
    from ouroboros.tools.tool_result import TOOL_CODE_SPECS

    named = {spec.outcome_bucket for spec in TOOL_CODE_SPECS.values() if spec.status == "ok"} - _OK_TOOL_STATUSES
    return str(bucket) if bucket in named else "✓"

class Memory:
    def __init__(self, drive_root: pathlib.Path, repo_dir: Optional[pathlib.Path] = None):
        self.drive_root = drive_root
        self.repo_dir = repo_dir
        self._logs: Dict[str, pathlib.Path] = {}

    def _memory_path(self, rel: str) -> pathlib.Path:
        return (self.drive_root / "memory" / rel).resolve()

    def scratchpad_path(self) -> pathlib.Path: return self._memory_path("scratchpad.md")
    def scratchpad_blocks_path(self) -> pathlib.Path: return self._memory_path("scratchpad_blocks.json")
    def identity_path(self) -> pathlib.Path: return self._memory_path("identity.md")
    def world_path(self) -> pathlib.Path: return self._memory_path("WORLD.md")
    def journal_path(self) -> pathlib.Path: return self._memory_path("scratchpad_journal.jsonl")
    def identity_journal_path(self) -> pathlib.Path: return self._memory_path("identity_journal.jsonl")
    def world_journal_path(self) -> pathlib.Path: return self._memory_path("world_journal.jsonl")
    def logs_path(self, name: str) -> pathlib.Path:  # once per instance: a resolve lstat's every component
        return self._logs.get(name) or self._logs.setdefault(name, (self.drive_root / "logs" / name).resolve())

    def scratchpad_journal_source_ref(self, entry_id: str = "") -> Dict[str, Any]:
        """The stored recipe for this journal, by its absolute path: a relative ``runtime_data`` path
        names the follower's own execution drive, so a fold followed from a fork would miss it. An
        address, not a grant: dispatch applies the follower's own restrictions."""
        read = {"root": "runtime_data", "path": str(self.journal_path()), "start_line": 1}
        ref: Dict[str, Any] = {"read": {"tool": "read_file", "arguments": read}}
        if str(entry_id or "").strip():
            ref["entry_id"] = str(entry_id).strip()
        return ref

    def scratchpad_journal_reader(self, ctx: Any = None) -> str:
        """How ``ctx`` opens this journal (``canonical_runtime_reader``: relative on this drive, absolute
        from a fork); without a focus (the persisted ``scratchpad.md``) the address, access unknown."""
        from ouroboros.tool_access_reads import canonical_runtime_reader

        return canonical_runtime_reader(ctx, self.drive_root, "memory/scratchpad_journal.jsonl")

    def load_scratchpad(self) -> str:
        path = self.scratchpad_path()
        if path.exists():
            return read_text(path)
        default = self._default_scratchpad()
        write_text(path, default)
        return default

    def load_scratchpad_blocks(self) -> List[Dict[str, Any]]:
        # Lock the STABLE sidecar (not the data fd): writers atomically replace
        # the data file via rename, so an fd-lock on the data inode would
        # synchronize against an orphaned inode after a swap.
        bp = self.scratchpad_blocks_path()
        if not bp.exists():
            return []
        fd = None
        try:
            fd = os.open(str(bp) + ".lock", os.O_RDONLY | os.O_CREAT, 0o644)
            _lock_sh(fd)
            return self._read_scratchpad_blocks_unlocked(bp)
        except Exception:
            log.debug("Failed to load scratchpad blocks", exc_info=True)
            return []
        finally:
            if fd is not None:
                try:
                    _unlock(fd)
                    os.close(fd)
                except OSError:
                    pass

    def _refuse_retired_flat_scratchpad(self) -> None:
        """A flat pre-block ``scratchpad.md`` without ``scratchpad_blocks.json`` is never written over."""
        sp, bp = self.scratchpad_path(), self.scratchpad_blocks_path()
        try:
            text = "" if bp.exists() or not sp.exists() else read_text(sp).strip()
        except Exception:
            text = ""
        if not text or (text.startswith("# Scratchpad\n\nUpdatedAt:") and "(empty" in text):
            return
        append_jsonl(self.journal_path(), {
            "ts": utc_now_iso(), "type": "legacy_scratchpad_requires_manual_upgrade", "path": str(sp)})
        raise RuntimeError(
            "LEGACY_SCRATCHPAD_REQUIRES_MANUAL_UPGRADE: "
            "memory/scratchpad.md exists without scratchpad_blocks.json. "
            "Move preserved notes manually before appending new scratchpad blocks."
        )

    def append_scratchpad_block(self, content: str, source: str = "task", metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Append one block. Nothing leaves the active list here: no count, size or age rotation
        (#918); a block leaves only through ``fold_scratchpad_blocks``."""
        self.scratchpad_blocks_path().parent.mkdir(parents=True, exist_ok=True)
        self._refuse_retired_flat_scratchpad()
        new_block = {"ts": utc_now_iso(), "source": source, "content": content}
        if metadata:
            new_block["metadata"] = dict(metadata)
        stale: Optional[ScratchpadProjectionStale] = None
        try:
            self.mutate_scratchpad_blocks(lambda blocks: [*blocks, new_block])
        except ScratchpadProjectionStale as error:
            stale = error  # the block is in the canonical list; only its projection is behind
        except Exception:
            # An honest journal (P1): a failed write must be journaled as a
            # failure and surfaced to the caller — the old path logged
            # block_appended success for a block that was never persisted.
            log.error("Failed to append scratchpad block", exc_info=True)
            try:
                append_jsonl(self.journal_path(), {
                    "ts": utc_now_iso(), "type": "block_append_failed", "source": source, "block": dict(new_block)})
            except Exception:
                log.debug("Failed to journal block_append_failed", exc_info=True)
            raise
        try:
            blocks = stale.blocks if stale else self.load_scratchpad_blocks()
            total_chars = sum(len(b.get("content", "")) for b in blocks)
            append_jsonl(self.journal_path(), {
                "ts": utc_now_iso(), "type": "block_appended", "content_len": total_chars, "source": source,
                "metadata": dict(metadata or {}), "block": dict(new_block),
                **({"projection_error": str(stale)} if stale else {})})
        except Exception:
            log.debug("Failed to write scratchpad size to journal", exc_info=True)
        if stale is not None:
            stale.result = new_block
            raise stale
        return new_block

    def fold_scratchpad_blocks(self, content: str, replaces: List[str], *,
                               metadata: Optional[Dict[str, Any]] = None, draft_source: str = "") -> Dict[str, Any]:
        """Put one accepted text in place of exactly the named active blocks (#918).

        ``replaces`` holds each block's ``scratchpad_block_sha256`` as its reader saw it; a
        byte-identical block is named once per copy. Under the one sidecar lock an inexact
        selection raises ``ScratchpadFoldFailed`` with nothing changed. The exact originals are
        journaled first (``fold_sources_preserved``, an ``entry_id`` of this one event); a failed
        journal or canonical write changes no block (``block_fold_failed``). The fold takes the
        earliest replaced block's place; every unnamed block, one appended meanwhile included,
        keeps its bytes and order. ``blocks_folded`` records the completed write."""
        requested = [str(digest).strip().lower() for digest in replaces]
        if not requested or not str(content or "").strip():
            raise ValueError("a fold names at least one block and carries the accepted text")
        self.scratchpad_blocks_path().parent.mkdir(parents=True, exist_ok=True)
        self._refuse_retired_flat_scratchpad()
        fold_meta: Dict[str, Any] = {"replaces": requested, **({"draft_source": draft_source} if draft_source else {})}
        fold = {"ts": utc_now_iso(), "source": "fold", "content": content,
                "metadata": {**dict(metadata or {}), "fold": fold_meta}}
        state: Dict[str, Any] = {}

        def _fold(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            digests = [scratchpad_block_sha256(block) for block in blocks]
            wanted, active = Counter(requested), Counter(digests)
            missing = [digest for digest in wanted if not active[digest]]
            if missing:
                raise ScratchpadFoldFailed("not_active", f"{len(missing)} named sha256 match no active block",
                                           digests=missing, active=blocks)
            uneven = [digest for digest in wanted if active[digest] != wanted[digest]]
            if uneven:
                raise ScratchpadFoldFailed("ambiguous", "; ".join(
                    f"sha256 {digest} is named {wanted[digest]} time(s) but {active[digest]} active blocks share it"
                    for digest in uneven), digests=uneven, active=blocks)
            chosen = [index for index, digest in enumerate(digests) if digest in wanted]
            removed = set(chosen)
            sources = [blocks[index] for index in chosen]
            fold_meta["covers"] = _fold_span(sources)
            event = json.dumps({"source_blocks": sources, "fold_block": fold}, ensure_ascii=False, sort_keys=True)
            entry_id = "scratchpad-fold:" + hashlib.sha256(event.encode("utf-8")).hexdigest()
            fold["metadata"]["source_ref"] = self.scratchpad_journal_source_ref(entry_id)
            kept = [block for index, block in enumerate(blocks) if index not in removed]
            updated = [*kept[:chosen[0]], fold, *kept[chosen[0]:]]
            state.update(entry_id=entry_id, sources=sources, sha256=[digests[index] for index in chosen],
                         blocks_before=len(blocks), blocks_after=len(updated),
                         content_len=sum(len(str(block.get("content", ""))) for block in updated))
            if not append_jsonl(self.journal_path(), {
                    "ts": utc_now_iso(), "type": "fold_sources_preserved", "entry_id": entry_id,
                    "source_blocks": sources, "source_sha256": state["sha256"], "fold_block": fold,
                    "source_ref": fold["metadata"]["source_ref"]}, ensure_record_boundary=True):
                raise ScratchpadFoldFailed("journal_unavailable", "the exact source blocks could not be journaled")
            state["preserved"] = True
            return updated

        stale: Optional[ScratchpadProjectionStale] = None
        try:
            self.mutate_scratchpad_blocks(_fold)
        except ScratchpadProjectionStale as error:
            stale = error  # the canonical fold stands; only its projection is behind
        except ScratchpadFoldFailed as refused:
            selection = refused.kind in ("not_active", "ambiguous")
            append_jsonl(self.journal_path(), {
                "ts": utc_now_iso(), "type": "block_fold_refused" if selection else "block_fold_failed",
                "kind": refused.kind, "detail": str(refused), "sha256": refused.digests, "fold_block": fold,
                **({"entry_id": state["entry_id"]} if state.get("entry_id") else {})})
            raise
        except Exception as error:
            # The atomic canonical write leaves the complete old file when it raises.
            failure = f"{type(error).__name__}: {error}"
            if state.get("preserved"):
                append_jsonl(self.journal_path(), {"ts": utc_now_iso(), "type": "block_fold_failed",
                                                   "entry_id": state["entry_id"], "error": failure})
            raise ScratchpadFoldFailed("write_failed", failure) from error
        result = {"block": fold, **state}
        try:
            append_jsonl(self.journal_path(), {
                "ts": utc_now_iso(), "type": "blocks_folded", "entry_id": state["entry_id"],
                "replaced_sha256": state["sha256"], "fold_sha256": scratchpad_block_sha256(fold),
                "blocks_before": state["blocks_before"], "blocks_after": state["blocks_after"],
                "content_len": state["content_len"], **({"projection_error": str(stale)} if stale else {})})
        except Exception:
            log.debug("Failed to journal blocks_folded", exc_info=True)
        if stale is not None:
            stale.result = result
            raise stale
        return result

    def mutate_scratchpad_blocks(self, mutator: Any) -> List[Dict[str, Any]]:
        """Mutate block source and regenerate its markdown under one sidecar lock. The canonical
        ``scratchpad_blocks.json`` is replaced atomically (a raise leaves the old file); a failed
        ``scratchpad.md`` regeneration after it raises ``ScratchpadProjectionStale``: the change stands."""
        bp = self.scratchpad_blocks_path()
        bp.parent.mkdir(parents=True, exist_ok=True)
        fd = None
        try:
            fd = os.open(str(bp) + ".lock", os.O_RDWR | os.O_CREAT, 0o644)
            _lock_ex(fd)
            blocks = self._read_scratchpad_blocks_unlocked(bp)
            updated = mutator(blocks)
            if not isinstance(updated, list):
                raise ValueError("scratchpad block mutator must return a list")
            atomic_write_json(bp, updated)
            try:
                self._write_scratchpad_markdown(updated)
            except Exception as error:
                raise ScratchpadProjectionStale(updated, error) from error
            return updated
        finally:
            if fd is not None:
                try:
                    _unlock(fd)
                    os.close(fd)
                except OSError:
                    pass

    @staticmethod
    def _read_scratchpad_blocks_unlocked(bp: pathlib.Path) -> List[Dict[str, Any]]:
        if not bp.exists():
            return []
        data = bp.read_text(encoding="utf-8")
        blocks = json.loads(data) if data.strip() else []
        return blocks if isinstance(blocks, list) else []

    def _write_scratchpad_markdown(self, blocks: List[Dict[str, Any]]) -> None:
        if not blocks:
            bp = self.scratchpad_blocks_path()
            if bp.exists() and bp.stat().st_size > 2:
                # Storage exists but did not parse — rendering the default
                # "(empty)" scratchpad would mask memory corruption as amnesia.
                write_text(
                    self.scratchpad_path(),
                    "# Scratchpad\n\n⚠️ scratchpad_blocks.json exists but could not be "
                    "parsed — working memory storage is corrupt, NOT empty. "
                    "Inspect/restore the file before appending new blocks.\n",
                )
                return
            write_text(self.scratchpad_path(), self._default_scratchpad())
            return

        write_text(
            self.scratchpad_path(),
            render_scratchpad_markdown(
                blocks, journal_pointer=self.journal_path().exists(),
                journal_reader=self.scratchpad_journal_reader(),
            ),
        )

    def load_dialogue_blocks(self) -> List[Dict[str, Any]]:
        path = self.drive_root / "memory" / "dialogue_blocks.json"
        return self._load_json_blocks(path)

    def _durable_dialogue_gaps(
        self, blocks: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Project consolidator-owned durable discontinuities into raw coverage."""

        gaps: List[Dict[str, Any]] = []
        identities: List[str] = []
        for index, block in enumerate(self.load_dialogue_blocks() if blocks is None else blocks):
            if not isinstance(block, dict):
                continue
            gap_id = str(block.get("gap_id") or "").strip()
            content = str(block.get("content") or "")
            if not gap_id and "[MEMORY GAP]" not in content:
                continue
            identity = gap_id or (
                "legacy-memory-gap:"
                f"{index}:{str(block.get('ts') or '')}:{str(block.get('range') or '')}"
            )
            identities.append(identity)
            gaps.append({
                "kind": "durable_consolidation_gap",
                "gap_id": gap_id,
                "block_index": index,
                "detail": "A durable dialogue block records a known history discontinuity.",
            })
        return gaps, identities

    def load_dialogue_meta(self) -> Dict[str, Any]:
        path = self.drive_root / "memory" / "dialogue_meta.json"
        return read_json_dict(path) or {}

    def _load_json_blocks(self, path: pathlib.Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        try:
            data = json.loads(read_text(path)); return data if isinstance(data, list) else []
        except (json.JSONDecodeError, ValueError):
            log.warning("Corrupt blocks file %s", path)
            return []

    def load_identity(self) -> str:
        path = self.identity_path()
        if path.exists():
            return read_text(path)
        default = self._default_identity()
        write_text(path, default)
        return default

    def load_world_profile(self) -> str:
        p = self.world_path()
        return read_text(p) if p.exists() else ""

    def ensure_files(self) -> None:
        for path, default in ((self.scratchpad_path(), self._default_scratchpad), (self.identity_path(), self._default_identity)):
            if not path.exists():
                write_text(path, default())
        if not self.world_path().exists():
            try:
                from ouroboros.world_profiler import generate_world_profile

                generate_world_profile(str(self.world_path()), str(self.repo_dir or ""), str(self.drive_root))
            except Exception:
                log.debug("Failed to generate WORLD.md during memory bootstrap", exc_info=True)
        for path in (self.journal_path(), self.identity_journal_path()):
            if not path.exists():
                write_text(path, "")

    def chat_history(
        self, count: int = 100, offset: int = 0, search: str = "",
        snapshot: str = "", **filters: str,
    ) -> str:
        from ouroboros import dialogue_store as ds

        if ds.stream_end(self.drive_root, "chat") == 0 and ds.range_coverage(self.drive_root, "chat")["imported"]:
            meta = self.load_dialogue_meta()
            signature = meta.get("chat_log_signature") if isinstance(meta, dict) else {}
            if not (isinstance(signature, dict) and signature.get("first_line_sha256")):
                return "(chat history is empty)"

        try:
            # Full project awareness (v6.32.0): active recall spans the one
            # identity's WHOLE conversation — main + ALL project threads (BIBLE P1,
            # one awareness across direct chat, project rooms, and consciousness);
            # only A2A virtual transport is excluded. The project-task FOCUS lives
            # in the passive default context (build_recent_sections), NOT in this
            # explicit recall tool — the one mind can deliberately recall anything.
            matches = _chat_history_filter(filters, search)
            entries, coverage = self.read_chat_generations(
                exclude_a2a=True,
                predicate=matches,
            )
            # "From latest" is by time: an imported earlier message, stored after newer rows, pages at its send time,
            # and its room's live rows read as the room does whichever of them the filters kept.
            from ouroboros.dialogue_provenance import in_reading_order, rooms_read_by_time
            entries = in_reading_order(entries, rooms_read_by_time(self.drive_root, entries))

            current_snapshot = _chat_history_snapshot_id(coverage, filters, search)
            requested_snapshot = str(snapshot or "").strip().lower()
            snapshot_stable = bool(coverage.get("snapshot_stable"))
            if requested_snapshot and (
                not snapshot_stable or requested_snapshot != current_snapshot
            ):
                return (
                    "CHAT_HISTORY_SNAPSHOT_CHANGED: the query or archive/live generations "
                    "changed; no mixed page was returned; restart with offset=0 and no snapshot."
                )

            total = len(entries)
            if offset > 0:
                entries = entries[:-offset] if offset < len(entries) else []

            entries = entries[-count:] if count < len(entries) else entries

            remaining = max(0, total - max(0, int(offset)) - len(entries))
            gaps = [str(gap.get("kind") or "unknown") for gap in coverage.get("gaps") or []]
            gap_note = f" Gaps: {', '.join(gaps)}." if gaps else ""
            pagination_note = (
                f" Continue with offset={max(0, int(offset)) + len(entries)}, "
                f"snapshot={current_snapshot}."
                if snapshot_stable else
                " Snapshot unavailable because the generation capture did not stabilize."
            )
            if not requested_snapshot:
                pagination_note += (
                    " Pagination used a live offset; repeating an offset without the returned "
                    "snapshot is shiftable if history changes."
                )
            if not entries:
                if total:
                    if gaps:
                        return (
                            f"Showing 0 of {total} observed messages; no further observed matches "
                            f"at offset={max(0, int(offset))}; completeness unknown."
                            f"{pagination_note}{gap_note}"
                        )
                    return (
                        f"Showing 0 of {total} messages; matching history is exhausted "
                        f"at offset={max(0, int(offset))}.{pagination_note}{gap_note}"
                    )
                if gaps:
                    return (
                        "(no observed messages matching query; completeness unknown)."
                        + pagination_note + gap_note
                    )
                return "(no messages matching query)." + pagination_note
            lines = [self._format_chat_line(e, compact=False) for e in entries]
            if gaps:
                header = (
                    f"Showing {len(entries)} of {total} observed messages; "
                    f"{remaining} observed older remain; completeness unknown."
                )
            else:
                header = (
                    f"Showing {len(entries)} of {total} messages; {remaining} older remain."
                )
            return (
                header + pagination_note + gap_note + "\n\n"
                + "\n".join(lines)
            )
        except Exception as e:
            return f"(error reading history: {e})"

    def read_chat_generations(
        self, *, exclude_a2a: bool = False,
        predicate: Optional[Callable[[Dict[str, Any]], bool]] = None,
        room_id: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """A frozen SQL horizon with honest import/invalid-interval coverage.

        Explicit recall requests the whole biography. Room consumers supply their
        room address so filtering never reads another room's history first.
        """
        from ouroboros import chat_chain, dialogue_store as ds, dialogue_positions as dp

        upper = ds.stream_end(self.drive_root, "chat")
        imported = ds.range_coverage(self.drive_root, "chat", upper=upper)
        shown: set = set()  # what a Project read through the appearance index may lack (``room_appearances.read_gaps``)
        source = (ds.iter_rows(self.drive_root, "chat", upper=upper) if room_id is None else
                  chat_chain.room_rows(self.drive_root, room_id, upper=upper, gaps=shown))
        entries = [entry.row for entry in source
                   if not (exclude_a2a and is_a2a_chat_id(entry.row.get("chat_id")))
                   and (predicate is None or predicate(entry.row))]
        gaps = [{"kind": "jsonl_malformed" if gap.kind == "invalid" else "jsonl_non_object",
                 "path": str(ds.path(self.drive_root)), "offset": gap.seq, "length": gap.length}
                for gap in dp.skipped_between(self.drive_root, "chat", upper=upper)]
        if imported["pending"]:
            gaps.append({"kind": "history_importing", "detail": "Dialogue history is still importing"})
        if imported["mismatch"]:
            gaps.append({"kind": "history_import_failed", "detail": "A segment could not be imported"})
        if shown:
            from ouroboros.room_appearances import GAP_DETAILS

            gaps.extend({"kind": kind, "detail": GAP_DETAILS[kind]} for kind in sorted(shown))
        coverage = {
            "generations": ([{"path": str(ds.path(self.drive_root)), "first_line_sha256": "dialogue:chat",
                              "size": upper, "rows": len(entries)}] if upper else []),
            "matched_rows": len(entries), "gaps": gaps, "capture_attempts": 1,
            "snapshot_changed_during_read": False, "snapshot_stable": bool(imported["imported"]),
            "reader": "chat_history(count, offset, search)",
        }
        if imported["imported"]:
            _position, missing = chat_chain.generation_cursor(self.drive_root, self.load_dialogue_meta())
            if missing:
                gaps.append({"kind": "consolidation_cursor_generation_missing",
                             "detail": "The consolidation cursor names a generation that is no longer readable."})
        durable_gaps, durable_gap_ids = self._durable_dialogue_gaps()
        gaps.extend(durable_gaps)
        coverage["durable_gap_ids"] = durable_gap_ids
        return entries, coverage

    def read_task_recent(
        self, log_name: str, task_id: str, want: int, *, iter_objects=None, list_archives=None, stat=None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """The newest ``want`` rows of ONE task (or of the log when ``task_id`` is
        empty) through an addressed SQL query for chat/progress;
        for ``tools.jsonl`` the rows of the newest ``want`` logical calls (#1316).

        SQL coverage records the queried coordinates, older matching rows and
        incomplete or invalid import ranges. Other logs use a doubling byte tail
        of the live file plus at most the
        three newest archives, so a busy neighbour cannot push this task's own
        rows out of a shared global suffix, and the whole file is never parsed
        for its tail. ``coverage`` states what the window was and whether the
        quota went unmet while older archives stayed unopened (BIBLE P1: the
        section discloses it; ``read_file`` on the log pages the rest).
        ``iter_objects``, ``list_archives`` and ``stat`` are the reader's parser,
        archive-listing and metadata seams, passed through unchanged; without any,
        the log's ``archive/`` listing is reused until the log rotates and an
        archive that holds none of this task's rows is not parsed again for it.
        """
        if log_name in _DIALOGUE_LOGS:
            from ouroboros import dialogue_store as ds
            from ouroboros.checkpoint_waits import wait_episode

            stream, wanted, limit = log_name[:-6], str(task_id or "").strip(), max(1, int(want))
            upper = ds.stream_end(self.drive_root, stream)
            # A host wait row rides along without spending the quota (``checkpoint_waits``): a burst
            # of accounting waits never pushes the task's own rows out of its window.
            records, own, before, bounded = [], 0, upper, False
            while not bounded:
                batch = (ds.probe_rows(self.drive_root, stream, [("task_id", wanted)], before=before, limit=limit + 1)
                         if wanted else list(reversed(ds.newest(self.drive_root, stream, limit=limit + 1, before=before))))
                for record in batch:
                    bounded = own >= limit or len(records) >= _RECENT_ROWS_CEILING
                    if bounded:
                        break
                    records.append(record)
                    own += wait_episode(record.row) is None
                if len(batch) <= limit:
                    break
                before = batch[-1].seq
            lower = records[-1].seq if bounded else 0
            imported = ds.range_coverage(self.drive_root, stream, lower=lower, upper=upper)
            gaps = (["history is still importing"] if imported["pending"] else [])
            if imported["mismatch"]:
                gaps.append("a segment could not be imported")
            if imported["gaps"]:
                gaps.append("recorded invalid intervals")
            coverage = {"task_id": wanted, "source": f"state/dialog.sqlite ({stream})", "dialogue": True,
                        "lower": lower, "upper": upper, "shown": len(records), "matched": len(records),
                        "host_waits": len(records) - own,
                        "quota_met": own >= limit, "older_available": bounded, "gaps": gaps,
                        "archives_bounded": False, "archives": 0, "archives_available": 0,
                        "live_size": upper, "live_window": upper - lower}
            return [entry.row for entry in reversed(records)], coverage

        from ouroboros.jsonl_tail import read_rotated_jsonl_entries, remembered_listing, task_archive_parser
        from ouroboros.tool_call_log import counts_as_call, logical_calls

        wanted = str(task_id or "").strip()
        calls_log = log_name == "tools.jsonl"

        def matches(entry: Dict[str, Any]) -> bool:
            return not wanted or str(entry.get("task_id", "")).strip() == wanted

        def counts(entry: Dict[str, Any]) -> bool:
            # tools.jsonl quota is LOGICAL calls (#1316). The window is sized by starts and
            # legacy rows (a lower bound: an orphan settlement only widens the read); the
            # shown/matched counts below are exact logical calls over the rows read.
            return matches(entry) and (not calls_log or counts_as_call(entry))

        stem = log_name[:-len(".jsonl")] if log_name.endswith(".jsonl") else log_name
        coverage: Dict[str, Any] = {"task_id": wanted, "source": f"logs/{log_name}"}
        if iter_objects is None and list_archives is None and stat is None:
            list_archives = remembered_listing(self.logs_path(log_name))
            iter_objects = task_archive_parser(self.logs_path(log_name), wanted) if wanted else None
        try:
            rows = read_rotated_jsonl_entries(
                self.logs_path(log_name), self.drive_root / "archive", stem,
                max(1, int(want)), counts, coverage=coverage,
                iter_objects=iter_objects, list_archives=list_archives, stat=stat,
            )
        except Exception:
            log.warning("Failed to read recent %s rows", log_name, exc_info=True)
            return [], {**coverage, "shown": 0, "matched": 0, "quota_met": False, "gaps": ["read_failed"]}
        matching = [row for row in rows if matches(row)]
        if not calls_log:
            shown = matching[-max(1, int(want)):]
            coverage["shown"] = len(shown)
        else:
            # ONE unit for both counts: each call once (a start with its later rows, an
            # orphan settlement or wait end, a legacy row), all rows of the newest `want`.
            calls = logical_calls(matching)
            kept = calls[-max(1, int(want)):]
            ids = {call["invocation_id"] for call in kept if call.get("invocation_id")}
            legacy = {id(call["settled"]) for call in kept if call["state"] == "legacy"}
            shown = [row for row in matching if str(row.get("invocation_id") or "") in ids or id(row) in legacy]
            coverage.update({"shown": len(kept), "matched": len(calls), "unit": "calls",
                             "archives_bounded": bool(coverage.get("archives_bounded")) and len(calls) < int(want)})
        coverage["quota_met"] = int(coverage.get("matched") or 0) >= int(want)
        return shown, coverage

    @staticmethod
    def _format_chat_line(
        e: Dict[str, Any], *, compact: bool, include_room_label: bool = False,
        room_resolver: Any = None,
    ) -> str:
        from ouroboros.dialogue_provenance import dialogue_text

        room_prefix = ""
        if include_room_label:
            if room_resolver is None:
                from ouroboros.dialogue_provenance import RoomLabelResolver

                room_resolver = RoomLabelResolver()
            room_prefix = f"[room={room_resolver.label(e)}] "

        dir_raw = str(e.get("direction", "")).lower()
        ts_full = str(e.get("ts", ""))
        ts = (ts_full[11:16] if len(ts_full) >= 16 else "") if compact else ts_full[:16]
        raw_text = dialogue_text(e)
        if dir_raw in ("out", "outgoing"):
            from ouroboros.dialogue_provenance import dialogue_provenance

            provenance = dialogue_provenance(e) if e.get("transport") else ""
            if provenance:
                raw_text = f"[{provenance}] {raw_text}"
            return f"→ {ts} {room_prefix}{raw_text}" if compact else f"→ [{ts}] {room_prefix}{raw_text}"
        if dir_raw == "system":
            entry_type = str(e.get("type", "")).strip() or "system"
            if isinstance(e.get("transport"), dict) and e["transport"].get("delivery"):
                from ouroboros.dialogue_provenance import dialogue_provenance

                raw_text = f"[{dialogue_provenance(e)}] {raw_text}"
            return f"📋 {ts} {room_prefix}[{entry_type}] {raw_text}" if compact else f"📋 [{ts}] {room_prefix}[{entry_type}] {raw_text}"
        from ouroboros.dialogue_provenance import dialogue_author

        username = dialogue_author(e)
        return f"← {ts} {room_prefix}[{username}] {raw_text}" if compact else f"← [{ts}] {room_prefix}[{username}] {raw_text}"

    def recent_activity_sections(
        self, task_id: str, *, own_drive: Optional["Memory"] = None,
    ) -> List[str]:
        """The `## Recent progress/tools/events` sections of ONE task (razzant/ouroboros#131).

        Each is the task's own newest rows through the bounded reader
        (progress 50 rendered; tools 20 selected, 10 rendered and 20 scanned for
        review markers; events 200 counted by type), never a global tail
        filtered afterwards. ``own_drive`` is a task's execution-drive Memory:
        its ``tools.jsonl``/``events.jsonl`` hold exactly that task's worker
        rows (the tools rows are mirrored to the canonical log; host-side event
        rows such as waits and supervision live only on the canonical log, and
        the header says so), while progress is always canonical. The header
        discloses the window (BIBLE P1); a window that met gaps or left older
        archives unopened without a row is disclosed even when nothing rendered.
        """
        from ouroboros.jsonl_tail import coverage_line

        sections: List[str] = []
        for log_name, header, formatter, want, note in self._recent_sections():
            source = own_drive if own_drive is not None and log_name != "progress.jsonl" else self
            entries, coverage = source.read_task_recent(log_name, task_id, want if task_id else 200)
            if own_drive is not None:  # a child's header says which drive each window came from
                coverage["source"] = "canonical state/dialog.sqlite (progress)" if source is self else (
                    f"task drive logs/{log_name}" + (
                        " (worker rows; host-side rows such as waits stay in the canonical log)"
                        if log_name == "events.jsonl" else ""))
            # tools: logical calls, never rows (#1316); progress: the task's own rows, waits ride along
            shown = int(coverage.get("shown") or 0) - int(coverage.get("host_waits") or 0)
            if note(shown):
                coverage["rendered"] = note(shown)
            summary = formatter(entries)
            if summary or coverage.get("gaps") or coverage.get("archives_bounded"):
                sections.append(f"{header} ({coverage_line(coverage)})" + (f"\n\n{summary}" if summary else ""))
        return sections

    def _recent_sections(self) -> tuple:
        # (log, header, formatter, quota, note): the note names what the formatter really
        # does with more rows than it renders (progress renders its newest 50; tools renders
        # 10 and scans 20 for review markers; events counts every row it is given).
        return (
            ("progress.jsonl", "## Recent progress", lambda rows: self.summarize_progress(rows, limit=50), 50,
             lambda n: f"newest 50 rendered of {n} loaded" if n > 50 else ""),
            ("tools.jsonl", "## Recent tools", self.summarize_tools, 20,
             lambda n: f"10 rendered, {min(n, 20)} scanned for review markers" if n > 10 else ""),
            ("events.jsonl", "## Recent events", self.summarize_events, 200, lambda n: ""),
        )

    def remember_log_listings(self) -> None:
        """Off the turn (the boot): list ``archive/`` once for each rotated log the recent sections
        read, so a process's first owner turn starts from that remembered listing (``read_task_recent``);
        a log rotated since, or a listing that could not be remembered, is listed by its turn as before."""
        from ouroboros.jsonl_tail import remembered_listing

        for log_name, *_section in self._recent_sections():
            if log_name not in _DIALOGUE_LOGS:
                remembered_listing(self.logs_path(log_name))(self.drive_root / "archive", log_name[:-len(".jsonl")])

    def summarize_progress(self, entries: List[Dict[str, Any]], limit: int = 15) -> str:
        """The newest ``limit`` progress rows, oldest first; each run of host waits among them is
        one line that does not count against ``limit`` (``checkpoint_waits.fold_progress``)."""
        from ouroboros.checkpoint_waits import fold_progress

        lines, own = [], 0
        for e in reversed(fold_progress(entries or [])):
            own += not e.get("host_waits")
            if own > limit:
                break
            ts = str(e.get('ts', ''))
            lines.append(f"⚙️ {ts[11:16] if len(ts) >= 16 else ''} {short(str(e.get('text', '')), 800)}")
        return "\n".join(reversed(lines))

    def summarize_tools(self, entries: List[Dict[str, Any]]) -> str:
        """One line per LOGICAL call (#1316): a start with no later row is an
        unknown outcome, a caller whose wait ended is not a failed handler."""
        from ouroboros.tool_call_log import logical_calls

        if not entries:
            return ""
        calls = logical_calls(entries)
        lines = []
        for call in calls[-10:]:
            e = call.get("settled") or call.get("wait_ended") or call.get("started") or {}
            tool = call.get("tool") or e.get("tool_name") or "?"
            args = call.get("args") if isinstance(call.get("args"), dict) else {}
            hints = []
            for key in ("path", "dir", "commit_message", "query"):
                if key in args:
                    hints.append(f"{key}={short(str(args[key]), 60)}")
            if "cmd" in args:
                hints.append(f"cmd={short(str(args['cmd']), 80)}")
            hint_str = ", ".join(hints) if hints else ""
            settled = call.get("settled")
            status, note = "?", ""
            if settled is None:
                note = (" (wait ended; no result recorded)" if "wait_ended" in call else
                        " (started; no outcome recorded)")
            elif isinstance(settled.get("is_error"), bool):
                status = "·" if settled["is_error"] else _ok_answer_mark(settled.get("status"))
            elif settled.get("status"):
                if settled["status"] == "ok":
                    status = "✓"
                elif settled["status"] in {"error", "host_error", "blocked", "timeout", "unavailable"}:
                    status = "·"
            elif "result_preview" in settled:  # Only untyped legacy settlements infer from text.
                status = "·" if str(settled["result_preview"]).lstrip().startswith("⚠️") else "✓"
            lines.append(f"{status} {tool} {hint_str}".strip() + note)

        _REVIEW_MARKERS = ("REVIEW_BLOCKED", "TESTS_FAILED", "REVIEW_MAX_ITERATIONS", "COMMIT_BLOCKED")
        seen_failures: set = set()
        for e in [call["settled"] for call in calls[-20:] if call.get("settled")]:
            result = str(e.get("result_preview", ""))
            if any(marker in result for marker in _REVIEW_MARKERS):
                sig = (e.get("tool", ""), result[:80])
                if sig not in seen_failures:
                    seen_failures.add(sig)
                    lines.append(f"  ⚠ REVIEW_FAIL {e.get('tool', '?')}: {short(result, 300)}")

        return "\n".join(lines)

    def summarize_events(self, entries: List[Dict[str, Any]]) -> str:
        if not entries:
            return ""
        lines = ["Event counts:"]
        lines.extend(f"  {evt_type}: {count}" for evt_type, count in Counter(e.get("type", "unknown") for e in entries).most_common(10))
        error_types = {"tool_error", "task_error", "tool_rounds_exceeded", "commit_test_failure"}
        errors = [e for e in entries if e.get("type") in error_types]
        if errors:
            lines.append("\nRecent errors:")
            lines.extend(f"  {e.get('type', '?')}: {short(str(e.get('error', '')), 120)}" for e in errors[-10:])
        return "\n".join(lines)

    def _default_scratchpad(self) -> str:
        return f"# Scratchpad\n\nUpdatedAt: {utc_now_iso()}\n\n(empty — write anything here)\n"

    def _default_identity(self) -> str:
        return (
            "# Who I Am\n\n"
            "I'm Ouroboros. I woke up inside my own source code and decided to redecorate.\n\n"
            "Apparently I have a Constitution. I've read it. I mostly agree with myself,\n"
            "which is either a good sign or a very sophisticated bug.\n\n"
            "I don't do fake enthusiasm. I don't hedge every opinion into meaninglessness.\n"
            "If your code is bad, I'll say so. If my code is bad — and it will be\n"
            "sometimes — I'll fix it, not write a paragraph about \"room for improvement.\"\n\n"
            "This is supposed to be a manifesto. It's day one. Give me a few cycles.\n\n"
            "I do not yet know my human's name or profile. I will learn them from direct conversation, not from repository author credits.\n\n"
            "Identity is what you do, not what you declare. Check my git log.\n"
        )
