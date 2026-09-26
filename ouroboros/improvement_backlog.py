"""Durable actionable improvement backlog stored in the knowledge base."""

from __future__ import annotations

import hashlib
import pathlib
import re
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List

from ouroboros.platform_layer import file_lock_exclusive, file_lock_shared, file_unlock
from ouroboros.utils import utc_now_iso

BACKLOG_TOPIC = "improvement-backlog"
BACKLOG_REL_PATH = f"memory/knowledge/{BACKLOG_TOPIC}.md"
_BACKLOG_TITLE = "# Improvement Backlog"
_BACKLOG_PREAMBLE = (
    "This topic stores concrete, evidence-backed improvement items discovered during task execution.\n"
    "Items here are advisory backlog nominations, not auto-started work.\n"
    "Before implementation, run plan_task for non-trivial backlog items."
)
_DEFAULT_BACKLOG_TEXT = f"{_BACKLOG_TITLE}\n\n{_BACKLOG_PREAMBLE}\n"


def backlog_path(drive_root: Any) -> pathlib.Path:
    return pathlib.Path(drive_root) / BACKLOG_REL_PATH


@contextmanager
def _locked_text_file(path: pathlib.Path, mode: str, *, shared: bool = False) -> Iterator[Any]:
    fh = open(path, mode, encoding="utf-8")
    try:
        if shared:
            file_lock_shared(fh.fileno())
        else:
            file_lock_exclusive(fh.fileno())
        yield fh
    finally:
        try:
            file_unlock(fh.fileno())
        except Exception:
            pass
        fh.close()


def ensure_backlog_file(drive_root: Any) -> pathlib.Path:
    path = backlog_path(drive_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _locked_text_file(path, mode="a+") as fh:
        fh.seek(0)
        current = fh.read()
        if not current:
            fh.write(_DEFAULT_BACKLOG_TEXT)
            fh.flush()
    return path


def _stable_fingerprint(summary: str, category: str, source: str) -> str:
    key = " | ".join(
        re.sub(r"\s+", " ", str(value or "")).strip().lower()
        for value in (summary, category, source)
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


def _parse_backlog_items(text: str) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    current: Dict[str, str] | None = None
    raw_lines: List[str] = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if line.startswith("### "):
            if current is not None:
                current["_raw"] = "\n".join(raw_lines).rstrip()
                items.append(current)
            current = {"id": line[4:].strip()}
            raw_lines = [raw_line]
            continue
        if current is None:
            continue
        # Retain the verbatim block (incl. freeform/comment lines) so an unmodified
        # item can be re-serialized losslessly (BIBLE P1: no silent data loss).
        raw_lines.append(raw_line)
        if line.startswith("- ") and ": " in line:
            key, value = line[2:].split(": ", 1)
            current[key.strip()] = value.strip()

    if current is not None:
        current["_raw"] = "\n".join(raw_lines).rstrip()
        items.append(current)
    return items


def load_backlog_items(drive_root: Any) -> List[Dict[str, str]]:
    path = backlog_path(drive_root)
    if not path.exists():
        return []
    with _locked_text_file(path, mode="r", shared=True) as fh:
        text = fh.read()
    return _parse_backlog_items(text)


# Canonical entry field order for (re)serialization. Additive over the original
# schema: priority/count/last_seen/kind/closed_at were added in v6.23.2.
_ENTRY_KEYS = (
    "status",
    "priority",
    "kind",
    "created_at",
    "last_seen",
    "count",
    "source",
    "category",
    "task_id",
    "requires_plan_review",
    "fingerprint",
    "closed_at",
    "summary",
    "evidence",
    "context",
    "proposed_next_step",
)

_PRIORITY_RANK = {"high": 0, "med": 1, "medium": 1, "low": 2}
_VALID_PRIORITY = {"high", "med", "low"}


def _sanitize(value: Any, limit: int = 300) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    # Backlog values must stay on ONE line for the `- key: value` parser, so use a
    # single-line visible omission note — never a silent clip of a cognitive
    # artifact (BIBLE P1 / DEVELOPMENT.md).
    return text[:limit] + f" ⚠️ OMISSION NOTE: +{len(text) - limit} chars omitted"


def _identity_is_known_partial(summary: Any, category: Any, source: Any) -> bool:
    """Whether semantic comparison would see less than a canonical identity field."""
    for value, limit in ((summary, 260), (category, 60), (source, 60)):
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        if "⚠️ OMISSION NOTE:" in text or len(text) > limit:
            return True
    return False


def _priority(value: Any) -> str:
    raw = str(value or "med").strip().lower()
    raw = {"medium": "med"}.get(raw, raw)
    return raw if raw in _VALID_PRIORITY else "med"


def _count_of(item: Dict[str, Any]) -> int:
    try:
        return max(1, int(item.get("count") or 1))
    except (TypeError, ValueError):
        return 1


def _serialize_item(entry: Dict[str, Any]) -> str:
    # Unmodified items carry their verbatim source block (`_raw`): re-emit it
    # exactly, so a read-modify-write that only touches OTHER items never alters
    # or drops this one (incl. hand-added freeform/comment lines). Modified items
    # drop `_raw` (see append/close/groom) and are re-serialized canonically.
    raw = entry.get("_raw")
    if raw:
        return str(raw)
    block = [f"### {entry.get('id', 'ibl-?')}"]
    for key in _ENTRY_KEYS:
        value = entry.get(key)
        if value not in (None, ""):
            block.append(f"- {key}: {value}")
    # Preserve any hand-added fields outside the canonical schema.
    for key, value in entry.items():
        if key == "id" or key == "_raw" or key in _ENTRY_KEYS:
            continue
        if value not in (None, ""):
            block.append(f"- {key}: {value}")
    return "\n".join(block)


def _serialize_backlog(items: List[Dict[str, Any]]) -> str:
    head = f"{_BACKLOG_TITLE}\n\n{_BACKLOG_PREAMBLE}\n"
    if not items:
        return head
    return head + "\n" + "\n\n".join(_serialize_item(it) for it in items) + "\n"


def _rebuild_index(path: pathlib.Path) -> None:
    try:
        from ouroboros.consolidator import _rebuild_knowledge_index

        _rebuild_knowledge_index(path.parent)
    except Exception:
        pass


_DEDUP_CANDIDATE_CAP = 20


def _dedup_candidates(open_items: List[Dict[str, Any]], category: str, source: str) -> List[Dict[str, str]]:
    """Open backlog items sharing the new item's category or source, deterministically
    ranked (priority, recurrence count, recency) and capped — the candidate pool the
    semantic detector ranks a new item against."""
    same = [
        it for it in open_items
        if str(it.get("category") or "") == category or str(it.get("source") or "") == source
    ]
    same.sort(key=lambda it: str(it.get("last_seen") or it.get("created_at") or ""), reverse=True)
    same.sort(key=lambda it: (_PRIORITY_RANK.get(_priority(it.get("priority")), 1), -_count_of(it)))
    return [
        {"id": str(it.get("id") or ""), "text": str(it.get("summary") or "")}
        for it in same[:_DEDUP_CANDIDATE_CAP]
        if it.get("id") and it.get("summary") and not _identity_is_known_partial(
            it.get("summary"), it.get("category"), it.get("source")
        )
    ]


def _semantic_redirect_fingerprints(
    drive_root: Any, path: pathlib.Path, items: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Two-phase semantic dedup (C9.2): snapshot the open backlog under a shared
    lock, RELEASE it, then for each NEW item that is an exact-fingerprint MISS ask
    the shared detector whether it duplicates an existing open item of the same
    category/source. A high-confidence dup has its fingerprint REDIRECTED to the
    match, so the locked exact pass below simply bumps that item's count/last_seen
    instead of minting a new ibl-*. The LLM call stays OUTSIDE the file lock; any
    failure (or a concurrent change before the exclusive write) is fail-open — the
    item keeps its own fingerprint and lands as new. Never raises."""
    try:
        with _locked_text_file(path, mode="r", shared=True) as fh:
            existing = _parse_backlog_items(fh.read())
    except Exception:
        return items
    existing_fps = {str(it.get("fingerprint") or "") for it in existing if it.get("fingerprint")}
    open_items = [
        it for it in existing
        if str(it.get("status") or "open").lower() != "done" and it.get("fingerprint")
    ]
    if not open_items:
        return items

    from ouroboros.semantic_dedup import find_semantic_duplicate_id

    out: List[Dict[str, Any]] = []
    for item in items:
        raw_summary = str(item.get("summary") or "")
        raw_category = str(item.get("category") or "process")
        raw_source = str(item.get("source") or "task")
        summary = _sanitize(raw_summary, 260)
        if not summary:
            out.append(item)
            continue
        category = _sanitize(raw_category, 60) or "process"
        source = _sanitize(raw_source, 60) or "task"
        fingerprint = str(
            item.get("fingerprint")
            or _stable_fingerprint(raw_summary, raw_category, raw_source)
        )
        if fingerprint in existing_fps:
            out.append(item)  # exact hit — the locked pass bumps it, no LLM needed
            continue
        if _identity_is_known_partial(raw_summary, raw_category, raw_source):
            out.append(item)
            continue
        candidates = _dedup_candidates(open_items, category, source)
        if not candidates:
            out.append(item)
            continue
        dup_id = find_semantic_duplicate_id(
            summary, candidates,
            subject="backlog improvement item",
            call_type="backlog_dedup",
            drive_root=drive_root,
        )
        target = next((it for it in open_items if str(it.get("id") or "") == dup_id), None) if dup_id else None
        if target and target.get("fingerprint"):
            item = {**item, "fingerprint": str(target["fingerprint"])}  # redirect -> bump
        out.append(item)
    return out


def append_backlog_items(drive_root: Any, items: List[Dict[str, Any]]) -> int:
    """Add/refresh backlog items. Recurrence (A): a repeat of an existing item is
    NOT dropped — its ``count`` and ``last_seen`` are bumped in place (and a
    previously-closed item re-opens). Priority/kind (B/D) are persisted. A reworded
    restatement that misses the exact fingerprint is caught by the semantic dedup
    pre-pass (C9.2) and folded into the item it duplicates."""
    if not items:
        return 0

    path = ensure_backlog_file(drive_root)
    items = _semantic_redirect_fingerprints(drive_root, path, items)
    with _locked_text_file(path, mode="r+") as fh:
        existing_text = fh.read()
        existing = _parse_backlog_items(existing_text)
        # Preserve EVERY parsed item (read-modify-write), including parser-valid
        # entries that lack a fingerprint (e.g. hand-added) — they must survive.
        by_key: Dict[str, Dict[str, Any]] = {}
        order: List[str] = []
        fp_to_key: Dict[str, str] = {}
        for idx, it in enumerate(existing):
            fp = str(it.get("fingerprint") or "")
            key = fp or f"__nofp_{idx}_{it.get('id', '') or idx}"
            by_key[key] = it
            order.append(key)
            if fp:
                fp_to_key[fp] = key
        now = utc_now_iso()
        changed = 0

        for item in items:
            raw_summary = str(item.get("summary") or "")
            raw_category = str(item.get("category") or "process")
            raw_source = str(item.get("source") or "task")
            summary = _sanitize(raw_summary, 260)
            if not summary:
                continue
            category = _sanitize(raw_category, 60) or "process"
            source = _sanitize(raw_source, 60) or "task"
            fingerprint = str(
                item.get("fingerprint")
                or _stable_fingerprint(raw_summary, raw_category, raw_source)
            )
            if fingerprint in fp_to_key:
                ex = by_key[fp_to_key[fingerprint]]
                ex["count"] = str(_count_of(ex) + 1)
                ex["last_seen"] = now
                # A recurring item that was marked done is evidently not resolved.
                if str(ex.get("status") or "").lower() == "done":
                    ex["status"] = "open"
                    ex.pop("closed_at", None)
                ex.pop("_raw", None)  # modified -> re-serialize canonically
                changed += 1
                continue
            created = _sanitize(item.get("created_at", now), 40)
            entry = {
                "id": str(item.get("id") or f"ibl-{fingerprint}"),
                "status": _sanitize(item.get("status", "open"), 40) or "open",
                "priority": _priority(item.get("priority")),
                "kind": _sanitize(item.get("kind", "improvement"), 40) or "improvement",
                "created_at": created,
                "last_seen": created,
                "count": "1",
                "source": source,
                "category": category,
                "task_id": _sanitize(item.get("task_id", ""), 80),
                "requires_plan_review": "yes" if item.get("requires_plan_review", True) else "no",
                "fingerprint": fingerprint,
                "summary": summary,
                "evidence": _sanitize(item.get("evidence", ""), 260),
                "context": _sanitize(item.get("context", ""), 400),
                "proposed_next_step": _sanitize(item.get("proposed_next_step", ""), 260),
            }
            by_key[fingerprint] = entry
            order.append(fingerprint)
            fp_to_key[fingerprint] = fingerprint
            changed += 1

        if not changed:
            return 0

        new_text = _serialize_backlog([by_key[k] for k in order])
        fh.seek(0)
        fh.write(new_text)
        fh.truncate()
        fh.flush()

    _rebuild_index(path)
    return changed


def merge_backlog_text(drive_root: Any, text: str) -> int:
    """Non-destructively merge backlog items parsed from ``text`` into the ONE
    global backlog (C10.1 Fix A). Routes through ``append_backlog_items`` so the
    write is a UNION (existing items preserved, new items added, reworded restatements
    folded by the dedup pre-pass) — never a truncating overwrite that could wipe the
    immune backlog. Returns the number of items merged, or ``-1`` (fail-closed) when
    ``text`` carries no parseable item — an unparseable overwrite must leave the
    backlog intact, even if the model believes it wrote something. Never raises."""
    try:
        items = [
            it for it in _parse_backlog_items(text or "")
            if str(it.get("summary") or "").strip()
        ]
    except Exception:
        return -1
    if not items:
        return -1
    # Only items NOT already present (by exact fingerprint) are merged: a verbatim
    # re-write of the backlog — the natural read-edit-write pattern — must not bump
    # every existing item's recurrence count by +1 (the count drives ranking and the
    # recurrence heuristics). Existing items stay untouched (non-destructive); only
    # genuinely new items flow through the SSOT append (its own semantic pre-pass +
    # exact merge still applies to those).
    try:
        existing_fps = {
            str(it.get("fingerprint") or "")
            for it in load_backlog_items(drive_root)
            if it.get("fingerprint")
        }
    except Exception:
        existing_fps = set()
    fresh: List[Dict[str, Any]] = []
    for it in items:
        raw_summary = str(it.get("summary") or "")
        raw_category = str(it.get("category") or "process")
        raw_source = str(it.get("source") or "task")
        fingerprint = str(
            it.get("fingerprint")
            or _stable_fingerprint(raw_summary, raw_category, raw_source)
        )
        if fingerprint in existing_fps:
            continue
        fresh.append(it)
    if not fresh:
        return 0  # everything already present — backlog intact, no inflation
    return append_backlog_items(drive_root, fresh)


def close_backlog_items(drive_root: Any, *, task_id: Any = None, ids: Any = None) -> int:
    """Close-on-commit (C): flip matching open items to ``status: done`` with a
    ``closed_at`` stamp. Match by originating ``task_id`` and/or explicit ``ids``."""
    path = backlog_path(drive_root)
    if not path.exists():
        return 0
    want_task = str(task_id or "").strip()
    want_ids = {str(i).strip() for i in (ids or []) if str(i or "").strip()}
    if not want_task and not want_ids:
        return 0
    with _locked_text_file(path, mode="r+") as fh:
        text = fh.read()
        items = _parse_backlog_items(text)
        now = utc_now_iso()
        closed = 0
        for it in items:
            if str(it.get("status") or "").lower() == "done":
                continue
            if (want_task and str(it.get("task_id") or "") == want_task) or (str(it.get("id") or "") in want_ids):
                it["status"] = "done"
                it["closed_at"] = now
                it.pop("_raw", None)  # modified -> re-serialize canonically
                closed += 1
        if not closed:
            return 0
        fh.seek(0)
        fh.write(_serialize_backlog(items))
        fh.truncate()
        fh.flush()
    _rebuild_index(path)
    return closed


def format_backlog_digest(drive_root: Any, *, limit: int = 5, max_chars: int = 2500) -> str:
    items = [item for item in load_backlog_items(drive_root) if item.get("status", "open") == "open"]
    if not items:
        return ""

    # Sort (B): priority, then recurrence count, then recency — so an important
    # older item outranks a junk burst. Recency uses last_seen (bumped on
    # recurrence), falling back to created_at, so a recently-recurred old item
    # is not unfairly demoted.
    items.sort(key=lambda item: str(item.get("last_seen") or item.get("created_at", "")), reverse=True)
    items.sort(key=lambda item: (_PRIORITY_RANK.get(_priority(item.get("priority")), 1), -_count_of(item)))
    visible = items[:limit]
    lines = [
        "## Improvement Backlog",
        "",
        f"- open_items: {len(items)}",
        "- policy: advisory backlog only; run plan_task before implementation",
    ]
    for item in visible:
        bits = [f"[{item.get('id', '?')}]", item.get("summary", "(missing summary)")]
        meta = [f"priority={_priority(item.get('priority'))}"]
        count = _count_of(item)
        if count > 1:
            meta.append(f"count={count}")
        if item.get("kind"):
            meta.append(f"kind={item['kind']}")
        if item.get("category"):
            meta.append(f"category={item['category']}")
        if item.get("source"):
            meta.append(f"source={item['source']}")
        if item.get("task_id"):
            meta.append(f"task={item['task_id']}")
        line = "- " + " ".join(bits)
        if meta:
            line += " (" + ", ".join(meta) + ")"
        lines.append(line)
    omitted = len(items) - len(visible)
    if omitted > 0:
        lines.append(f"- ⚠️ OMISSION NOTE: {omitted} additional open backlog items not shown")

    text = "\n".join(lines)
    if len(text) > max_chars:
        return text[:max_chars] + f"\n⚠️ OMISSION NOTE: backlog digest truncated at {max_chars} chars; original length {len(text)}"
    return text


_GROOM_CAP = 30

_GROOM_PROMPT = """You are grooming Ouroboros's improvement backlog. Goals:
- Merge near-duplicate items (keep the clearest summary + the higher priority; set the survivor's count to the summed count).
- Mark items that are clearly already resolved/obsolete as status "done".
- Drop pure noise.
- Keep at most {cap} of the most valuable OPEN items, ranked by priority (high>med>low), then recurrence count, then recency.

Current backlog items (JSON):
{items_json}

Return ONLY a JSON array of the items to KEEP. Each object: {{"id","status","priority","kind","summary","category","source","task_id","requires_plan_review","count","created_at","fingerprint","evidence","context","proposed_next_step"}}. Preserve the id/fingerprint/created_at of every kept item. Do NOT invent new work. Do NOT drop a high-priority or high-count item just to hit the cap."""


def _needs_verbatim_grooming(item: Dict[str, Any]) -> bool:
    """True when a modelled overlay cannot preserve the stored block losslessly."""
    if any(key not in {"id", "_raw", *_ENTRY_KEYS} for key in item):
        return True
    seen = set()
    for line in str(item.get("_raw") or "").splitlines()[1:]:
        if not line or not line.startswith("- ") or ": " not in line:
            return True
        key = line[2:].split(": ", 1)[0].strip()
        if key not in _ENTRY_KEYS or key in seen:
            return True
        seen.add(key)
    return False


def groom_backlog(drive_root: Any, *, cap: int = _GROOM_CAP) -> int:
    """D: best-effort LLM grooming of AUTO-generated (fingerprinted) backlog items —
    merge near-dupes, mark resolved, re-rank, cap to <=cap. Hand-added items and
    fingerprinted blocks with unmodelled bytes pass through UNCHANGED. Runs on a
    size trigger and re-serializes through the locked parser-safe writer. Returns
    the number written, or 0 on a bad/empty/oversized reply or concurrent change;
    a failed model call raises to the caller (never the 0 of a pass not needed)."""
    import json as _json

    path = backlog_path(drive_root)
    if not path.exists():
        return 0
    try:
        with _locked_text_file(path, mode="r", shared=True) as fh:
            snapshot_text = fh.read()
    except Exception:
        return 0
    items = _parse_backlog_items(snapshot_text)
    if len(items) <= cap:
        return 0
    # Fingerprinted blocks with unknown fields/freeform bytes cannot be safely
    # overlaid from the model schema, so preserve them like hand-authored items.
    fingerprinted = [it for it in items if str(it.get("fingerprint") or "").strip()]
    fp_items = [it for it in fingerprinted if not _needs_verbatim_grooming(it)]
    passthrough_items = [
        it for it in items
        if not str(it.get("fingerprint") or "").strip() or _needs_verbatim_grooming(it)
    ]
    if not fp_items:
        return 0

    from ouroboros.config import get_light_model
    from ouroboros.llm import LLMClient
    from ouroboros.llm_observability import chat_observed

    # One destructive grooming call sees every complete stored record,
    # including evidence/context/custom fields and immutable manual items.
    # If this full prompt cannot be served, the raised failure preserves the
    # file; there is no smaller second call or prefix-authorized rewrite. The
    # caller's stage classifies it: an interruption stops later paid post-work.
    complete = [dict(it) for it in items]
    prompt = _GROOM_PROMPT.format(cap=cap, items_json=_json.dumps(complete, ensure_ascii=False))
    resp, usage = chat_observed(
        LLMClient(),
        drive_root=pathlib.Path(drive_root),
        task_id="backlog_groom",
        call_type="backlog_groom",
        model_role="light",
        messages=[{"role": "user", "content": prompt}],
        model=get_light_model(),
        reasoning_effort="low",
        max_tokens=8192,
    )
    if usage:
        try:
            from supervisor.state import update_budget_from_usage

            update_budget_from_usage(usage)
        except Exception:
            pass
    content = (resp.get("content") or "").strip()
    start, end = content.find("["), content.rfind("]")
    try:
        kept_raw = _json.loads(content[start:end + 1]) if 0 <= start < end else None
    except ValueError:
        return 0  # a bad reply, like an empty one: the file is preserved
    if not isinstance(kept_raw, list):
        return 0

    # Anti-wipe: every kept item MUST map to an existing fingerprinted item — the
    # model may merge/drop/re-rank but NEVER invent survivors — and grooming may
    # not over-drop the auto items in one pass.
    by_fp = {str(it.get("fingerprint") or ""): it for it in fp_items}
    by_id = {str(it.get("id") or ""): it for it in fp_items if it.get("id")}
    kept_fp: List[Dict[str, Any]] = []
    seen_fp = set()
    for obj in kept_raw:
        if not isinstance(obj, dict):
            continue
        summary = _sanitize(obj.get("summary", ""), 260)
        if not summary:
            continue
        ofp = str(obj.get("fingerprint") or "").strip()
        oid = str(obj.get("id") or "").strip()
        base = (by_fp.get(ofp) if ofp else None) or (by_id.get(oid) if oid else None)
        if not base:
            continue  # invented item — drop it (never wipe-by-invention)
        fingerprint = str(base.get("fingerprint") or "")
        if not fingerprint or fingerprint in seen_fp:
            continue
        seen_fp.add(fingerprint)
        category = _sanitize(obj.get("category", base.get("category", "process")), 60) or "process"
        source = _sanitize(obj.get("source", base.get("source", "task")), 60) or "task"
        merged_count = _count_of(obj) if str(obj.get("count") or "").isdigit() else _count_of(base)
        kept_fp.append({
            "id": str(base.get("id") or f"ibl-{fingerprint}"),
            "status": _sanitize(obj.get("status", base.get("status", "open")), 40) or "open",
            "priority": _priority(obj.get("priority", base.get("priority"))),
            "kind": _sanitize(obj.get("kind", base.get("kind", "improvement")), 40) or "improvement",
            "created_at": _sanitize(base.get("created_at", utc_now_iso()), 40),
            "last_seen": _sanitize(base.get("last_seen", utc_now_iso()), 40),
            "count": str(max(1, merged_count)),
            "source": source,
            "category": category,
            "task_id": _sanitize(base.get("task_id", ""), 80),
            "requires_plan_review": "no" if str(obj.get("requires_plan_review", base.get("requires_plan_review", "yes"))).lower() in ("no", "false") else "yes",
            "fingerprint": fingerprint,
            "summary": summary,
            "evidence": _sanitize(base.get("evidence", ""), 260),
            "context": _sanitize(base.get("context", ""), 400),
            "proposed_next_step": _sanitize(base.get("proposed_next_step", ""), 260),
        })
    if not kept_fp or len(kept_fp) > cap or len(kept_fp) < max(1, min(len(fp_items), cap) // 2):
        return 0

    final = kept_fp + passthrough_items  # unmodelled bytes always survive unchanged
    with _locked_text_file(path, mode="r+") as fh:
        # Abort if the backlog changed during the lock-free LLM call (a concurrent
        # append/close/recurrence) — never overwrite a concurrent update. The next
        # post-task tick will groom the fresh state.
        if fh.read() != snapshot_text:
            return 0
        fh.seek(0)
        fh.write(_serialize_backlog(final))
        fh.truncate()
        fh.flush()
    _rebuild_index(path)
    return len(final)
