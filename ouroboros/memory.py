from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
from collections import Counter, deque
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from ouroboros.contracts.chat_id_policy import is_a2a_chat_id
from ouroboros.utils import append_jsonl, iter_jsonl_objects, read_json_dict, read_text, short, utc_now_iso, write_text
from ouroboros.platform_layer import (
    file_lock_exclusive as _lock_ex,
    file_lock_shared as _lock_sh,
    file_unlock as _unlock,
)

log = logging.getLogger(__name__)
_AUTOMATIC_CHAT_GENERATIONS = 3
_AUTOMATIC_CHAT_TAIL_BYTES = 512 * 1024
_AUTOMATIC_CHAT_MAX_SCAN_ROWS = 5_000

_SCRATCHPAD_MAX_BLOCKS = 10


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
        if search_lower and search_lower not in str(entry.get("text", "")).lower():
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


class Memory:
    def __init__(self, drive_root: pathlib.Path, repo_dir: Optional[pathlib.Path] = None):
        self.drive_root = drive_root
        self.repo_dir = repo_dir

    def _memory_path(self, rel: str) -> pathlib.Path:
        return (self.drive_root / "memory" / rel).resolve()

    def scratchpad_path(self) -> pathlib.Path: return self._memory_path("scratchpad.md")
    def scratchpad_blocks_path(self) -> pathlib.Path: return self._memory_path("scratchpad_blocks.json")
    def identity_path(self) -> pathlib.Path: return self._memory_path("identity.md")
    def world_path(self) -> pathlib.Path: return self._memory_path("WORLD.md")
    def journal_path(self) -> pathlib.Path: return self._memory_path("scratchpad_journal.jsonl")
    def identity_journal_path(self) -> pathlib.Path: return self._memory_path("identity_journal.jsonl")
    def logs_path(self, name: str) -> pathlib.Path: return (self.drive_root / "logs" / name).resolve()

    @staticmethod
    def scratchpad_journal_source_ref(entry_id: str = "") -> Dict[str, Any]:
        ref: Dict[str, Any] = {
            "read": {
                "tool": "read_file",
                "arguments": {
                    "root": "runtime_data",
                    "path": "memory/scratchpad_journal.jsonl",
                    "start_line": 1,
                },
            },
        }
        if str(entry_id or "").strip():
            ref["entry_id"] = str(entry_id).strip()
        return ref

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

    def _has_retired_flat_scratchpad_without_blocks(self) -> bool:
        sp = self.scratchpad_path()
        bp = self.scratchpad_blocks_path()
        if bp.exists() or not sp.exists():
            return False
        try:
            text = read_text(sp).strip()
        except Exception:
            return False
        if not text:
            return False
        return not (
            text.startswith("# Scratchpad\n\nUpdatedAt:")
            and "(empty" in text
        )

    def append_scratchpad_block(self, content: str, source: str = "task", metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        bp = self.scratchpad_blocks_path()
        bp.parent.mkdir(parents=True, exist_ok=True)

        if self._has_retired_flat_scratchpad_without_blocks():
            msg = (
                "LEGACY_SCRATCHPAD_REQUIRES_MANUAL_UPGRADE: "
                "memory/scratchpad.md exists without scratchpad_blocks.json. "
                "Move preserved notes manually before appending new scratchpad blocks."
            )
            append_jsonl(self.journal_path(), {
                "ts": utc_now_iso(),
                "type": "legacy_scratchpad_requires_manual_upgrade",
                "path": str(self.scratchpad_path()),
            })
            raise RuntimeError(msg)

        new_block = {"ts": utc_now_iso(), "source": source, "content": content}
        if metadata:
            new_block["metadata"] = dict(metadata)

        try:
            def _append(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                updated = [*blocks, new_block]
                if len(updated) <= _SCRATCHPAD_MAX_BLOCKS:
                    return updated
                evicted = updated[:-_SCRATCHPAD_MAX_BLOCKS]
                for eb in evicted:
                    written = append_jsonl(self.journal_path(), {
                        "ts": utc_now_iso(),
                        "type": "block_evicted",
                        "evicted_block_ts": eb.get("ts", ""),
                        "evicted_block_source": eb.get("source", ""),
                        "evicted_block_content": eb.get("content", ""),
                        "source_ref": self.scratchpad_journal_source_ref(),
                    })
                    if not written:
                        raise RuntimeError("scratchpad eviction journal write failed")
                return updated[-_SCRATCHPAD_MAX_BLOCKS:]

            self.mutate_scratchpad_blocks(_append)
        except Exception:
            # An honest journal (P1): a failed write must be journaled as a
            # failure and surfaced to the caller — the old path logged
            # block_appended success for a block that was never persisted.
            log.error("Failed to append scratchpad block", exc_info=True)
            try:
                append_jsonl(self.journal_path(), {
                    "ts": utc_now_iso(),
                    "type": "block_append_failed",
                    "source": source,
                    "block": dict(new_block),
                })
            except Exception:
                log.debug("Failed to journal block_append_failed", exc_info=True)
            raise
        try:
            total_chars = sum(len(b.get("content", "")) for b in self.load_scratchpad_blocks())
            append_jsonl(self.journal_path(), {
                "ts": utc_now_iso(),
                "type": "block_appended",
                "content_len": total_chars,
                "source": source,
                "metadata": dict(metadata or {}),
                "block": dict(new_block),
            })
        except Exception:
            log.debug("Failed to write scratchpad size to journal", exc_info=True)

        return new_block

    def regenerate_scratchpad_md(self) -> None:
        bp = self.scratchpad_blocks_path()
        bp.parent.mkdir(parents=True, exist_ok=True)
        fd = None
        try:
            fd = os.open(str(bp) + ".lock", os.O_RDWR | os.O_CREAT, 0o644)
            _lock_ex(fd)
            try:
                blocks = self._read_scratchpad_blocks_unlocked(bp)
            except Exception:
                log.debug("Failed to load scratchpad blocks for regeneration", exc_info=True)
                blocks = []
            self._write_scratchpad_markdown(blocks)
        finally:
            if fd is not None:
                try:
                    _unlock(fd)
                    os.close(fd)
                except OSError:
                    pass

    def mutate_scratchpad_blocks(self, mutator: Any) -> List[Dict[str, Any]]:
        """Mutate block source and regenerate its markdown under one sidecar lock."""
        from ouroboros.utils import atomic_write_json

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
            self._write_scratchpad_markdown(updated)
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

        n = len(blocks)
        parts = [f"## Scratchpad (working memory — {n}/{_SCRATCHPAD_MAX_BLOCKS} blocks)\n"]
        if self.journal_path().exists():
            parts.append(
                "Exact retired/replaced source blocks remain readable with "
                "`read_file(root='runtime_data', "
                "path='memory/scratchpad_journal.jsonl', start_line=1)`.\n\n"
            )
        for block in reversed(blocks):
            ts = str(block.get("ts", ""))[:16]
            source = block.get("source", "?")
            content = block.get("content", "")
            parts.append(f"### [{ts} — {source}]\n{content}\n\n---\n")
            metadata = block.get("metadata") if isinstance(block.get("metadata"), dict) else {}
            source_ref = metadata.get("source_ref") if isinstance(metadata.get("source_ref"), dict) else {}
            entry_id = str(source_ref.get("entry_id") or "")
            if entry_id:
                parts.append(
                    "Exact replaced blocks: `read_file(root='runtime_data', "
                    "path='memory/scratchpad_journal.jsonl', start_line=1)`; "
                    f"locate `entry_id={entry_id}`.\n\n"
                )

        write_text(self.scratchpad_path(), "\n".join(parts))

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

    @staticmethod
    def format_blocks_as_markdown(blocks: List[Dict[str, Any]]) -> str:
        return "\n\n".join(b.get("content", "") for b in blocks)

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

                generate_world_profile(str(self.world_path()))
            except Exception:
                log.debug("Failed to generate WORLD.md during memory bootstrap", exc_info=True)
        for path in (self.journal_path(), self.identity_journal_path()):
            if not path.exists():
                write_text(path, "")

    def chat_history(
        self, count: int = 100, offset: int = 0, search: str = "",
        snapshot: str = "", **filters: str,
    ) -> str:
        chat_path = self.logs_path("chat.jsonl")
        archive_dir = self.drive_root / "archive"
        if not chat_path.exists() and not any(archive_dir.glob("chat_*.jsonl")):
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

    def _ordered_chat_generation_paths(self) -> List[pathlib.Path]:
        """Existing canonical chat generations, oldest archive through live.

        This is deliberately a chat-specific Memory primitive, not a generic
        history framework.  Consolidation owns cursor semantics; this reader
        gives ordinary cognition and ``chat_history`` the same physical
        generation horizon instead of silently treating the mutable live file
        as the whole biography.
        """
        from ouroboros.consolidator import _ordered_chat_generation_paths

        live = self.logs_path("chat.jsonl")
        return _ordered_chat_generation_paths(live)

    def read_chat_generations(
        self,
        *,
        exclude_a2a: bool = False,
        predicate: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Read the archive/live chat chain with truthful coverage facts.

        Filtering happens while traversing the complete existing chain and
        therefore before any caller applies a recent-window bound.  A rotation
        racing the capture is retried; a persistently moving chain is returned
        as a known-partial snapshot with an explicit gap rather than certified
        complete.
        """
        from ouroboros.utils import jsonl_generation_signature

        last_entries: List[Dict[str, Any]] = []
        last_coverage: Dict[str, Any] = {}
        for attempt in range(3):
            paths = self._ordered_chat_generation_paths()
            before = [jsonl_generation_signature(path) for path in paths]
            entries: List[Dict[str, Any]] = []
            gaps: List[Dict[str, Any]] = []
            generations: List[Dict[str, Any]] = []
            for path, signature in zip(paths, before):
                if not path.exists():
                    # A live file can be absent before the first append or in the
                    # short interval after rotation.  A genuinely missing cursor
                    # generation is detected by the consolidator-owned resolver.
                    continue
                generation_rows, parse_gaps = self._read_chat_generation(path)
                gaps.extend(parse_gaps)
                generations.append({
                    "path": str(path),
                    "first_line_sha256": str(signature.get("first_line_sha256") or ""),
                    "size": int(signature.get("size") or 0),
                    "rows": len(generation_rows),
                })
                for entry in generation_rows:
                    if exclude_a2a and is_a2a_chat_id(entry.get("chat_id")):
                        continue
                    if predicate is not None and not predicate(entry):
                        continue
                    entries.append(entry)

            after_paths = self._ordered_chat_generation_paths()
            after = [jsonl_generation_signature(path) for path in after_paths]
            stable_paths = [str(path) for path in paths] == [str(path) for path in after_paths]
            stable_generations = stable_paths and all(
                str(left.get("first_line_sha256") or "")
                == str(right.get("first_line_sha256") or "")
                and int(right.get("size") or 0) == int(left.get("size") or 0)
                for left, right in zip(before, after)
            )
            coverage = {
                "generations": generations,
                "matched_rows": len(entries),
                "gaps": gaps,
                "capture_attempts": attempt + 1,
                "snapshot_changed_during_read": bool(
                    not stable_generations
                ),
                "snapshot_stable": stable_generations,
                "reader": "chat_history(count, offset, search)",
            }
            try:
                from ouroboros.consolidator import _resolve_generation_segments

                _segments, _offset, cursor_gap = _resolve_generation_segments(
                    self.load_dialogue_meta(), self.logs_path("chat.jsonl"),
                )
                if cursor_gap:
                    coverage["gaps"].append({
                        "kind": "consolidation_cursor_generation_missing",
                        "detail": "The consolidation cursor names a generation that is no longer readable.",
                    })
            except Exception as exc:
                coverage["gaps"].append({
                    "kind": "consolidation_cursor_state_unreadable",
                    "error": type(exc).__name__,
                })
            durable_gaps, durable_gap_ids = self._durable_dialogue_gaps()
            coverage["gaps"].extend(durable_gaps)
            coverage["durable_gap_ids"] = durable_gap_ids
            last_entries, last_coverage = entries, coverage
            if stable_generations:
                return entries, coverage
        last_coverage = dict(last_coverage)
        last_coverage["snapshot_stable"] = False
        last_coverage.setdefault("gaps", []).append({
            "kind": "generation_chain_changed_during_capture",
            "detail": "archive/live generation ordering did not stabilize after 3 attempts",
        })
        return last_entries, last_coverage

    @staticmethod
    def _read_chat_generation(
        path: pathlib.Path, *, tail_bytes: Optional[int] = None, max_rows: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Read one physical generation without hiding malformed durable rows."""
        rows: List[Dict[str, Any]] = []
        gaps: List[Dict[str, Any]] = []
        try:
            with path.open("rb") as handle:
                if tail_bytes is not None:
                    size = path.stat().st_size
                    if size > tail_bytes:
                        handle.seek(size - tail_bytes - 1)
                        if handle.read(1) != b"\n":
                            handle.readline()
                        gaps.append({
                            "kind": "generation_prefix_unscanned", "path": str(path),
                            "omitted_bytes_at_least": size - tail_bytes,
                        })
                if max_rows is not None:
                    buffered = deque(maxlen=max_rows)
                    seen = 0
                    for raw in handle:
                        seen += 1
                        buffered.append((seen, raw))
                    if seen > max_rows:
                        gaps.append({
                            "kind": "generation_tail_rows_unscanned", "path": str(path),
                            "omitted_rows": seen - max_rows,
                        })
                    source = buffered
                else:
                    source = enumerate(handle, start=1)
                for line_number, raw in source:
                    if not raw.strip():
                        continue
                    try:
                        decoded = raw.decode("utf-8")
                        value = json.loads(decoded)
                    except UnicodeDecodeError:
                        gaps.append({"kind": "jsonl_decode_error", "path": str(path), "line": line_number})
                        continue
                    except (json.JSONDecodeError, ValueError):
                        gaps.append({"kind": "jsonl_malformed", "path": str(path), "line": line_number})
                        continue
                    if not isinstance(value, dict):
                        gaps.append({"kind": "jsonl_non_object", "path": str(path), "line": line_number})
                        continue
                    rows.append(value)
        except OSError as exc:
            gaps.append({"kind": "generation_unreadable", "path": str(path), "error": type(exc).__name__})
        return rows, gaps

    def read_unconsolidated_chat(
        self,
        meta: Dict[str, Any],
        max_entries: int,
        *,
        predicate: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Read the exact generation-aware suffix owned by consolidation."""
        from ouroboros.consolidator import _resolve_generation_segments

        live = self.logs_path("chat.jsonl")
        scan_rows = max(100, min(
            _AUTOMATIC_CHAT_MAX_SCAN_ROWS, max(1, int(max_entries)) * 4,
        ))
        segments, offset, gap_detected = _resolve_generation_segments(meta, live)
        gaps: List[Dict[str, Any]] = []
        if gap_detected:
            gaps.append({
                "kind": "consolidation_cursor_generation_missing",
                "first_line_sha256": str(
                    (meta.get("chat_log_signature") or {}).get("first_line_sha256") or ""
                ),
                "offset": int(meta.get("last_consolidated_offset") or 0),
                "detail": (
                    "Older unconsolidated coverage is unknown. Use explicit "
                    "chat_history(count, offset, search) to inspect surviving generations."
                ),
            })
            # Never repair a missing cursor by replaying the entire archive in
            # the automatic per-turn path.  The live generation is rotation-
            # bounded; older surviving generations remain available through the
            # explicit paginated chat_history reader named above.
            entries, live_gaps = self._read_chat_generation(
                live, tail_bytes=_AUTOMATIC_CHAT_TAIL_BYTES, max_rows=scan_rows,
            )
            gaps.extend(live_gaps)
            entries = [entry for entry in entries if not is_a2a_chat_id(entry.get("chat_id"))]
            if predicate is not None:
                entries = [entry for entry in entries if predicate(entry)]
            limit = max(1, int(max_entries))
            shown = entries[-limit:]
            return shown, {
                "generations": [{"kind": "live_bounded_suffix"}],
                "matched_rows": len(entries),
                "shown_rows": len(shown),
                "omitted_matching_rows": max(0, len(entries) - len(shown)),
                "omitted_matching_rows_unknown": True,
                "gaps": gaps,
                "reader": "chat_history(count, offset, search)",
            }

        from ouroboros.utils import jsonl_generation_signature

        omitted_generations = max(0, len(segments) - _AUTOMATIC_CHAT_GENERATIONS)
        selected = segments[-_AUTOMATIC_CHAT_GENERATIONS:]
        segment_sigs: List[Dict[str, Any]] = []
        segment_entries: List[List[Dict[str, Any]]] = []
        stable = False
        for _attempt in range(3):
            segment_sigs = [jsonl_generation_signature(path) for path in selected]
            segment_entries = []
            parse_gaps: List[Dict[str, Any]] = []
            for path in selected:
                rows, row_gaps = self._read_chat_generation(
                    path, tail_bytes=_AUTOMATIC_CHAT_TAIL_BYTES, max_rows=scan_rows,
                )
                segment_entries.append(rows)
                parse_gaps.extend(row_gaps)
            after = [jsonl_generation_signature(path) for path in selected]
            stable = len(after) == len(segment_sigs) and all(
                str(before.get("first_line_sha256") or "")
                == str(current.get("first_line_sha256") or "")
                and int(current.get("size") or 0) >= int(before.get("size") or 0)
                for before, current in zip(segment_sigs, after)
            )
            if stable:
                gaps.extend(parse_gaps)
                break
        if not stable:
            gaps.extend(parse_gaps)
            gaps.append({
                "kind": "generation_capture_unstable",
                "detail": "bounded archive/live suffix changed during capture",
            })
        all_entries = [
            entry for rows in segment_entries for entry in rows
            if not is_a2a_chat_id(entry.get("chat_id"))
        ]
        bounded_prefix = any(
            gap.get("kind") in {
                "generation_prefix_unscanned", "generation_tail_rows_unscanned",
            }
            for gap in gaps
        )
        captured_offset = offset if omitted_generations == 0 and not bounded_prefix else 0
        suffix = [
            entry for entry in all_entries[captured_offset:]
            if predicate is None or predicate(entry)
        ]
        limit = max(1, int(max_entries))
        shown = suffix[-limit:]
        if omitted_generations:
            gaps.append({
                "kind": "unscanned_unconsolidated_generations",
                "count": omitted_generations,
                "detail": "Automatic context reads a bounded physical suffix; use chat_history for older raw rows.",
            })
        return shown, {
            "generations": [
                {
                    "path": str(path),
                    "first_line_sha256": str(sig.get("first_line_sha256") or ""),
                    "rows": len(rows),
                }
                for path, sig, rows in zip(selected, segment_sigs, segment_entries)
            ],
            "matched_rows": len(suffix),
            "shown_rows": len(shown),
            "omitted_matching_rows": max(0, len(suffix) - len(shown)),
            "omitted_matching_rows_unknown": bool(omitted_generations or bounded_prefix),
            "gaps": gaps,
            "reader": "chat_history(count, offset, search)",
        }

    def _read_jsonl_entries(
        self,
        log_name: str,
        max_entries: Optional[int] = None,
        exclude_a2a: bool = False,
    ) -> List[Dict[str, Any]]:
        path = self.logs_path(log_name)
        try:
            def _rows(source, cap):
                return [e for e in iter_jsonl_objects(source, max_entries=cap)
                        if not (exclude_a2a and is_a2a_chat_id(e.get("chat_id")))]

            entries = _rows(path, max_entries)
            if max_entries is not None and len(entries) < max_entries:
                # Rotation-aware backfill (CPL4-C2..C4): newest archive segments
                # top a freshly rotated tail back up; unbounded reads keep
                # live-file-only semantics (their consumers own their chains).
                from ouroboros.utils import jsonl_archive_segments

                for segment in reversed(jsonl_archive_segments(path)):
                    if len(entries) >= max_entries:
                        break
                    entries = _rows(segment, max_entries - len(entries)) + entries
            return entries
        except Exception:
            log.warning("Failed to read JSONL entries from %s", log_name, exc_info=True)
            return []

    def read_jsonl_tail(self, log_name: str, max_entries: int = 100) -> List[Dict[str, Any]]:
        return self._read_jsonl_entries(log_name, max_entries=max_entries)

    def read_jsonl_tail_after_offset(
        self,
        log_name: str,
        offset: int,
        max_entries: int = 100,
    ) -> List[Dict[str, Any]]:
        # Full project awareness (v6.32.0): the one identity's dialogue stream is
        # its WHOLE conversation — main + project threads alike — because Ouroboros
        # is one awareness across direct chat, project rooms, and background
        # consciousness (BIBLE P1). Only A2A virtual-transport ids are excluded
        # (machine-to-machine traffic, not the human dialogue). A project task's
        # OWN focused recent-chat view is built separately in build_recent_sections.
        entries = self._read_jsonl_entries(log_name, exclude_a2a=True)
        if offset <= 0:
            return entries[-max_entries:] if max_entries < len(entries) else entries
        if offset > len(entries):
            log.warning(
                "Dialogue consolidation offset %s exceeds %s filtered entry count %s; using plain tail",
                offset,
                log_name,
                len(entries),
            )
            return entries[-max_entries:] if max_entries < len(entries) else entries
        suffix = entries[offset:]
        return suffix[-max_entries:] if max_entries < len(suffix) else suffix

    def jsonl_generation_signature(self, log_name: str) -> Dict[str, Any]:
        from ouroboros.utils import jsonl_generation_signature

        return jsonl_generation_signature(self.logs_path(log_name))

    def summarize_chat(self, entries: List[Dict[str, Any]], limit: int = 1000) -> str:
        """Render recent chat entries; never hide a horizon cut silently (P1).

        Callers that want the FULL window (e.g. low-context mode passes a huge
        tail intent) pass a large ``limit``; when truncation does happen the
        output says exactly how many older unconsolidated messages were omitted.
        """
        if not entries:
            return ""
        limit = max(1, int(limit))
        shown = entries[-limit:]
        prefix = ""
        if len(entries) > len(shown):
            prefix = f"[{len(entries) - len(shown)} older unconsolidated messages omitted]\n"
        return prefix + "\n".join(self._format_chat_line(e, compact=True) for e in shown)

    @staticmethod
    def _format_chat_line(e: Dict[str, Any], *, compact: bool) -> str:
        dir_raw = str(e.get("direction", "")).lower()
        ts_full = str(e.get("ts", ""))
        ts = (ts_full[11:16] if len(ts_full) >= 16 else "") if compact else ts_full[:16]
        raw_text = str(e.get("text", ""))
        if dir_raw in ("out", "outgoing"):
            return f"→ {ts} {raw_text}" if compact else f"→ [{ts}] {raw_text}"
        if dir_raw == "system":
            entry_type = str(e.get("type", "")).strip() or "system"
            return f"📋 {ts} [{entry_type}] {raw_text}" if compact else f"📋 [{ts}] [{entry_type}] {raw_text}"
        from ouroboros.dialogue_provenance import dialogue_author

        username = dialogue_author(e)
        return f"← {ts} [{username}] {raw_text}" if compact else f"← [{ts}] [{username}] {raw_text}"

    def summarize_progress(self, entries: List[Dict[str, Any]], limit: int = 15) -> str:
        if not entries:
            return ""
        return "\n".join(
            f"⚙️ {str(e.get('ts', ''))[11:16] if len(str(e.get('ts', ''))) >= 16 else ''} {short(str(e.get('text', '')), 800)}"
            for e in entries[-limit:]
        )

    def summarize_tools(self, entries: List[Dict[str, Any]]) -> str:
        if not entries:
            return ""
        lines = []
        for e in entries[-10:]:
            tool = e.get("tool") or e.get("tool_name") or "?"
            args = e.get("args", {})
            hints = []
            for key in ("path", "dir", "commit_message", "query"):
                if key in args:
                    hints.append(f"{key}={short(str(args[key]), 60)}")
            if "cmd" in args:
                hints.append(f"cmd={short(str(args['cmd']), 80)}")
            hint_str = ", ".join(hints) if hints else ""
            status = "✓" if ("result_preview" in e and not str(e.get("result_preview", "")).lstrip().startswith("⚠️")) else "·"
            lines.append(f"{status} {tool} {hint_str}".strip())

        _REVIEW_MARKERS = ("REVIEW_BLOCKED", "TESTS_FAILED", "REVIEW_MAX_ITERATIONS", "COMMIT_BLOCKED")
        seen_failures: set = set()
        for e in entries[-20:]:
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

    def summarize_supervisor(self, entries: List[Dict[str, Any]]) -> str:
        if not entries:
            return ""
        for e in reversed(entries):
            if e.get("type") in ("launcher_start", "restart", "boot"):
                branch = e.get("branch") or e.get("git_branch") or "?"
                sha = short(str(e.get("sha") or e.get("git_sha") or ""), 12)
                return f"{e['type']}: {e.get('ts', '')} branch={branch} sha={sha}"
        return ""

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
