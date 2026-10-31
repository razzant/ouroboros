"""The usage store: ``state/usage.sqlite``, the one monetary authority.

Contract: docs/USAGE_STORE.md. One ``attempts`` row per physical id, updated on transitions.
``summaries`` holds exact scope/key buckets, maintained transactionally by
``_usage_rows.summary_delta`` (subtract old, add new; Decimal arithmetic on decimal text).
``bindings`` holds earliest root/group caps; ``dirty_owners`` holds stale projections;
``one_shots`` holds single-row identities; ``meta`` holds schema, tier, import and publication facts.

Nothing on an ordinary path aggregates the ``attempts`` table: admission, sends,
displays and status read summary rows and addressed rows (one attempt by id,
one root/task/wave, the open set). ``read_usage_records`` (every row) exists for
explicit audits and the export only.

Locking: one configuration per installation, recorded at import. ``enforced``
uses SQLite's own file locks (the shared journal policy, ``synchronous=FULL``) under the
existing acquisition contract: ``hold`` is the money acquisition primitive the
sliced send waits (``_usage_wait``) retry, so Stop and deadlines are honoured
between slices; ``name`` (no kernel file locks: Drive/FUSE/NFS per the existing
probe) opens SQLite without locking and runs EVERY access inside the existing
name-protocol money lock. Transactions are short; nothing network-bound runs
inside one.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import pathlib
import sqlite3
import time
import uuid
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from ouroboros._usage_money import (
    LiteralFloat, billing_group_key, decimal_of, durable_literals, exact_money, exceeds_limit, monetary_scope_key,
    render_cash, rounded_cash,
)
from ouroboros._usage_rows import (
    REVIEW_ATTRIBUTION_KEYS, BindingIndex, Bucket, row_ts_epoch, summary_delta,
)
from ouroboros.usage_origin import bind_root_attribution
from ouroboros import store as _core
from ouroboros.store import (
    TIER_ENFORCED, TIER_NAME, busy as _busy, file_identity as _identity, fsync_directory as _fsync_directory,
    process_lock as _process_lock,
)
from ouroboros.store_capture import Captured, capture
from ouroboros.usage_ledger import (
    LEDGER_REL, LOCK_REL, QUARANTINE_REL, USAGE_LOCK_TIMEOUT_SEC, UsageAccountingError, UsageExportInProgress,
    UsageLedgerCorrupt, UsageLockUnavailable,
    _named_lock, _number, is_one_shot_kind, late_receipt_eligible, validate_row_fields, validate_transition,
)
from ouroboros.utils import append_jsonl, utc_now_iso

log = logging.getLogger(__name__)

STORE_REL = pathlib.Path("state/usage.sqlite")
IMPORT_LOCK_NAME = "usage_import.lock"
SCHEMA_VERSION = 1
# The header selects the lock protocol BEFORE opening (also recorded in meta.lock_tier).
# Connection, tier and transaction core are shared with the current store (ouroboros.store).
_APPLICATION_ID = {TIER_ENFORCED: 0x4F555345, TIER_NAME: 0x4F55534E}
_OPEN_STATES = ("reserved", "dispatched", "unresolved")
_OPEN_PREDICATE = ("(state IN ('reserved','dispatched','unresolved') "
                   "OR (state='settled' AND settle_reason='abandoned'))")
# ``Txn.task_attempt_page``: newest first by the immutable, unique ``seq_first`` (``seq`` moves on every write).
_TASK_PAGE_SQL = ("SELECT attempt_id, state, settle_reason, model, provider, ts_reserved, ts_dispatched, "
                  "ts_final, ts_last FROM attempts INDEXED BY attempts_task "
                  "WHERE task_id=? AND category=? AND seq_first<? ORDER BY seq_first DESC LIMIT ?")
_TASK_CURSOR_SQL = "SELECT seq_first FROM attempts WHERE attempt_id=? AND task_id=? AND category=?"

# One root's money rows of a recent window by range, never the root's whole life (the consciousness
# allowance, ``consciousness_allowance``); a store an older body created gains it at the lifespan boot.
ROOT_TIME_INDEX = "CREATE INDEX IF NOT EXISTS attempts_root_time ON attempts(root_task_id, ts_last_epoch)"
DIRTY_REVISION_INDEX = "CREATE INDEX IF NOT EXISTS dirty_owners_revision ON dirty_owners(revision)"

_SCHEMA = f"""
CREATE TABLE attempts (
  attempt_id TEXT PRIMARY KEY, revision INTEGER NOT NULL, seq INTEGER NOT NULL, seq_first INTEGER NOT NULL,
  kind TEXT, state TEXT, task_id TEXT, root_task_id TEXT, parent_task_id TEXT, billing_group_id TEXT,
  model TEXT, provider TEXT, category TEXT, source TEXT,
  ts_reserved TEXT, ts_dispatched TEXT, ts_final TEXT, ts_last TEXT, ts_last_epoch REAL,
  cost_usd TEXT, reservation_upper_bound_usd TEXT, cost_final INTEGER, pricing_known INTEGER,
  settle_reason TEXT, late_receipt INTEGER NOT NULL DEFAULT 0,
  prompt_tokens INTEGER, completion_tokens INTEGER, cached_tokens INTEGER, cache_write_tokens INTEGER,
  weight INTEGER NOT NULL DEFAULT 1,
  root_limit_usd TEXT, has_root_limit INTEGER NOT NULL DEFAULT 0,
  billing_group_limit_usd TEXT, has_group_limit INTEGER NOT NULL DEFAULT 0,
  billing_group_limit_source TEXT, billing_group_limit_revision TEXT,
  review_skill TEXT, review_wave_id TEXT, review_slot_id TEXT,
  subscription_route TEXT, subscription_reset_at TEXT,
  local_answer_owner_pid INTEGER, local_answer_consumer_id TEXT, owner_birth TEXT, task_attempt INTEGER,
  prompt_cache_ttl TEXT, extra TEXT NOT NULL DEFAULT '{{}}'
);
CREATE INDEX attempts_root ON attempts(root_task_id);
CREATE INDEX attempts_group ON attempts(billing_group_id);
CREATE INDEX attempts_task ON attempts(task_id);
CREATE INDEX attempts_open ON attempts(state) WHERE {_OPEN_PREDICATE};
CREATE INDEX attempts_category_time ON attempts(category, ts_last_epoch);
{ROOT_TIME_INDEX};
CREATE INDEX attempts_review ON attempts(review_skill, review_wave_id);
CREATE INDEX attempts_route ON attempts(subscription_route);
CREATE TABLE summaries (
  scope TEXT NOT NULL, key TEXT NOT NULL, rows INTEGER NOT NULL,
  settled TEXT NOT NULL, confirmed TEXT NOT NULL, estimated TEXT NOT NULL, reserved TEXT NOT NULL,
  unresolved TEXT NOT NULL, accounted_num REAL NOT NULL,
  attempt_counts TEXT NOT NULL, unknown_unmetered INTEGER NOT NULL, priced_rows INTEGER NOT NULL,
  tracked_nonfinal_rows INTEGER NOT NULL, accounting_open_rows INTEGER NOT NULL, non_final_rows INTEGER NOT NULL,
  subscription_sessions INTEGER NOT NULL, subscription_windows TEXT NOT NULL, processing TEXT NOT NULL,
  prompt_tokens INTEGER NOT NULL, prompt_tokens_present INTEGER NOT NULL,
  completion_tokens INTEGER NOT NULL, completion_tokens_present INTEGER NOT NULL,
  cached_tokens INTEGER NOT NULL, cached_tokens_present INTEGER NOT NULL,
  cache_write_tokens INTEGER NOT NULL, cache_write_tokens_present INTEGER NOT NULL,
  physical_calls INTEGER NOT NULL, prompt_cache_ttls TEXT NOT NULL, caps TEXT NOT NULL,
  PRIMARY KEY (scope, key)
) WITHOUT ROWID;
CREATE INDEX summaries_accounted ON summaries(scope, accounted_num);
CREATE TABLE bindings (scope TEXT NOT NULL, key TEXT NOT NULL, binding TEXT NOT NULL,
  PRIMARY KEY (scope, key)) WITHOUT ROWID;
CREATE TABLE dirty_owners (owner_id TEXT PRIMARY KEY, revision INTEGER NOT NULL) WITHOUT ROWID;
{DIRTY_REVISION_INDEX};
CREATE TABLE one_shots (attempt_id TEXT PRIMARY KEY, kind TEXT, task_id TEXT, root_task_id TEXT,
  subscription_route TEXT, source TEXT, category TEXT, identity TEXT NOT NULL) WITHOUT ROWID;
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID;
"""

# ---- row codec: one attempt row <-> one ``attempts`` row, lossless ----------
_TEXT = {key: key for key in (
    "attempt_id", "kind", "state", "task_id", "root_task_id", "parent_task_id", "billing_group_id", "model",
    "provider", "category", "source", "settle_reason", "billing_group_limit_source",
    "billing_group_limit_revision", *REVIEW_ATTRIBUTION_KEYS, "subscription_route", "subscription_reset_at",
    "local_answer_consumer_id", "prompt_cache_ttl")}
_TEXT.update(ts="ts_last", local_answer_owner_birth="owner_birth")
_INT = {key: key for key in ("seq", "prompt_tokens", "completion_tokens", "cached_tokens", "cache_write_tokens",
                             "local_answer_owner_pid")}
_INT["local_answer_task_attempt"] = "task_attempt"
_BOOL = {"cost_final": "cost_final", "pricing_known": "pricing_known"}
_MONEY = {key: key for key in ("cost_usd", "reservation_upper_bound_usd", "root_limit_usd", "billing_group_limit_usd")}
_FIELDS = {**_TEXT, **_INT, **_BOOL, **_MONEY}
_NONE_KEY = "__none__"  # column-mapped keys present with an explicit None
_DERIVED = ("revision", "seq_first", "ts_reserved", "ts_dispatched", "ts_final", "ts_last_epoch", "late_receipt",
            "weight", "has_root_limit", "has_group_limit", "extra")
_COLUMNS = tuple(dict.fromkeys((*_FIELDS.values(), *_DERIVED)))
_ATTEMPT_CAPTURE = ("extra", "revision", *dict.fromkeys(_FIELDS.values()))  # what ``_decode`` reads: no REAL column


def _encode(row: Dict[str, Any]) -> Dict[str, Any]:
    """Column values of a materialized row; whatever does not fit a typed
    column (and every other field) stays verbatim in ``extra``."""
    columns: Dict[str, Any] = {column: None for column in _FIELDS.values()}
    extra: Dict[str, Any] = {}
    nones: List[str] = []
    for key, value in row.items():
        if key == "revision":
            continue
        column = _FIELDS.get(key)
        if column is None:
            extra[key] = value
        elif value is None:
            nones.append(key)
        elif key in _TEXT and isinstance(value, str):
            columns[column] = value
        elif key in _INT and type(value) is int:
            columns[column] = value
        elif key in _BOOL and isinstance(value, bool):
            columns[column] = int(value)
        elif key in _MONEY and isinstance(value, (int, float)) and not isinstance(value, bool):
            columns[column] = str(decimal_of(value))
        else:
            extra[key] = value  # an unusual type keeps its exact shape
    if nones:
        extra[_NONE_KEY] = nones
    columns["extra"] = json.dumps(durable_literals(extra), ensure_ascii=False, sort_keys=True,
                                  separators=(",", ":"))
    return columns


def decode_attempts(captured: Captured) -> List[Dict[str, Any]]:
    """``Txn.attempts``' rows from ``Txn.attempt_rows``, decoded after the read closed, in write order (``seq``)."""
    return [_decode(record) for record in captured.rows(key=lambda record: record["seq"])]


def _decode(record: Any) -> Dict[str, Any]:
    row = json.loads(record["extra"], parse_float=LiteralFloat)
    for key in row.pop(_NONE_KEY, ()):
        row[key] = None
    for key, column in _FIELDS.items():
        value = record[column]
        if value is None:
            continue
        row[key] = bool(value) if key in _BOOL else LiteralFloat(value) if key in _MONEY else value
    row["revision"] = record["revision"]
    return row


# ---- summary scopes ---------------------------------------------------------
_AXES = (("model", "model"), ("provider", "provider"), ("category", "category"))
_LEGACY_UNATTRIBUTED = frozenset({"legacy_metadata", "legacy_delta"})


def _cap_text(value: Any) -> Optional[str]:
    return None if _number(value) is None else str(decimal_of(value))


def summary_keys(row: Dict[str, Any]) -> List[Tuple[str, str, Optional[str]]]:
    """Every ``(scope, key, cap literal)`` the row contributes to.

    ``global``; ``root``/``group`` (every key, the empty one included, with the
    root cap / group cap multiset); ``task``; ``kind``; the breakdown axes
    model/provider/category (``unattributed`` when empty or legacy
    metadata/delta, exactly as ``usage_breakdown`` groups); within a root
    ``root_<axis>`` / ``root_task`` / ``root_delegated``, within a task
    ``task_<axis>`` / ``task_root`` / ``task_delegated`` (key ``"<id>|<value>"``,
    an empty value = unattributed inside that address)."""
    root, group = monetary_scope_key(row), billing_group_key(row)
    task, kind = str(row.get("task_id") or ""), str(row.get("kind") or "")
    legacy = kind in _LEGACY_UNATTRIBUTED
    group_cap = row.get("billing_group_limit_usd", row.get("root_limit_usd"))
    keys = [("global", "", None), ("root", root, _cap_text(row.get("root_limit_usd"))),
            ("group", group, _cap_text(group_cap)), ("kind", kind, None)]
    if task:
        keys.append(("task", task, None))
    for axis, field in _AXES:
        value = "" if legacy else str(row.get(field) or "")
        keys.append((axis, value, None) if value else ("unattributed", axis, None))
        if root:
            keys.append((f"root_{axis}", f"{root}|{value}", None))
        if task:
            keys.append((f"task_{axis}", f"{task}|{value}", None))
    if legacy or not task:
        keys.append(("unattributed", "task", None))
    if legacy or not root:
        keys.append(("unattributed", "root", None))
    if root:
        keys.append(("root_task", f"{root}|{'' if legacy else task}", None))
    if task:
        keys.append(("task_root", f"{task}|{'' if legacy else root}", None))
    if kind == "subscription_session":
        keys.extend((name, owner, None) for name, owner in (("root_delegated", root), ("task_delegated", task)) if owner)
    return keys


_TOKENS = ("prompt_tokens", "completion_tokens", "cached_tokens", "cache_write_tokens")
_SUMMARY_COLUMNS = (
    "scope", "key", "rows", "settled", "confirmed", "estimated", "reserved", "unresolved", "accounted_num",
    "attempt_counts", "unknown_unmetered", "priced_rows", "tracked_nonfinal_rows", "accounting_open_rows",
    "non_final_rows", "subscription_sessions", "subscription_windows", "processing",
    *(name for token in _TOKENS for name in (token, f"{token}_present")),
    "physical_calls", "prompt_cache_ttls", "caps")


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _decimal_text(value: Any) -> str:
    """Canonical decimal text: the same amount is the same text whatever
    sequence of additions produced it (exact; positional, never exponent).
    ``normalize`` rounds to the context precision, so it runs in the money
    context, never the ambient 28 digits."""
    amount = decimal_of(value)
    if amount == 0:
        return "0"
    with exact_money():
        return format(amount.normalize(), "f")


def _bucket_record(scope: str, key: str, bucket: Bucket) -> tuple:
    processing = {name: [_decimal_text(pair[0]), pair[1]] if isinstance(pair, list) else
                  {label: list(count) for label, count in pair.items()} for name, pair in bucket.processing.items()}
    tokens = [value for token in _TOKENS for value in (bucket.tokens.get(token) or [0, 0])]
    return (scope, key, bucket.rows, *(_decimal_text(value) for value in bucket.cash),
            float(rounded_cash(bucket.cash)[-1]),
            _json(bucket.counts), bucket.unknown, bucket.priced, bucket.tracked_nonfinal, bucket.accounting_open,
            bucket.non_final, bucket.sessions, _json(bucket.windows), _json(processing), *tokens,
            bucket.physical, _json(bucket.ttls), _json(bucket.caps))


_BUCKET_CAPTURE = tuple(column for column in _SUMMARY_COLUMNS if column != "accounted_num")  # REAL: ordering only


class SummaryRows:
    """``Txn.bucket``/``Txn.buckets`` over the rows ``Txn.summary_rows`` captured, decoded after the read closed."""

    def __init__(self, captured: Captured) -> None:
        self._rows = {(record["scope"], record["key"]): _bucket_of(record) for record in captured.rows()}

    def bucket(self, scope: str, key: str = "") -> Bucket:
        return self._rows.get((scope, str(key))) or Bucket()

    def buckets(self, scope: str, prefix: Optional[str] = None) -> Dict[str, Bucket]:
        under = "" if prefix is None else f"{prefix}|"  # every key, or the keys under "<prefix>|" without it
        return {key[len(under):]: bucket for (held, key), bucket in sorted(self._rows.items())
                if held == scope and key.startswith(under)}


def _bucket_of(record: Any) -> Bucket:
    from decimal import Decimal

    processing: Dict[str, Any] = {}
    for name, pair in json.loads(record["processing"]).items():
        if isinstance(pair, list):
            processing[name] = [Decimal(pair[0]) if name.endswith("_usd") else int(pair[0]), int(pair[1])]
        else:
            processing[name] = {label: [int(count[0]), int(count[1])] for label, count in pair.items()}
    return Bucket(
        rows=record["rows"],
        cash=tuple(Decimal(record[name]) for name in ("settled", "confirmed", "estimated", "reserved", "unresolved")),
        counts=json.loads(record["attempt_counts"]), unknown=record["unknown_unmetered"],
        priced=record["priced_rows"], tracked_nonfinal=record["tracked_nonfinal_rows"],
        accounting_open=record["accounting_open_rows"], non_final=record["non_final_rows"],
        sessions=record["subscription_sessions"], windows=json.loads(record["subscription_windows"]),
        processing=processing,
        tokens={token: [record[token], record[f"{token}_present"]] for token in _TOKENS
                if record[f"{token}_present"]},
        physical=record["physical_calls"], ttls=json.loads(record["prompt_cache_ttls"]),
        caps=json.loads(record["caps"]))


# One-shot identity: the fields a replay must repeat (today's rule): a
# subscription session's observed model and token counts are not identity.
ONE_SHOT_IDENTITY = {
    "external_unmetered": ("kind", "model", "provider", "task_id", "root_task_id", "parent_task_id",
                           "category", "source", "prompt_tokens", "completion_tokens",
                           "external_dispatch_id_sha256"),
    "subscription_session": ("kind", "provider", "task_id", "root_task_id", "parent_task_id", "category",
                             "source", *REVIEW_ATTRIBUTION_KEYS, "subscription_route", "session_id_sha256"),
}
_LEGACY_IDENTITY = ("kind", "provider", "task_id", "root_task_id", "category", "source")


def one_shot_identity(row: Dict[str, Any]) -> str:
    kind = str(row.get("kind") or "")
    keys = ONE_SHOT_IDENTITY.get(kind, _LEGACY_IDENTITY)
    # Rows written before physical_attempt_v1 omitted the review keys: missing
    # == explicit empty; a non-empty wave/slot conflicts with either.
    return _json({key: str(row.get(key) or "") if key in REVIEW_ATTRIBUTION_KEYS else
                  durable_literals(row.get(key), key) for key in keys})


def _derived(row: Dict[str, Any], previous: Optional[Dict[str, Any]], revision: int,
             seq_first: int) -> Dict[str, Any]:
    state, ts = str(row.get("state") or ""), str(row.get("ts") or "")
    kind = str(row.get("kind") or "")
    return {
        "revision": revision, "seq_first": seq_first,
        "ts_reserved": ts if state == "reserved" else (previous or {}).get("_ts_reserved"),
        "ts_dispatched": ts if state == "dispatched" else (previous or {}).get("_ts_dispatched"),
        "ts_final": ts if state in {"settled", "unresolved", "released"} else None,
        "ts_last_epoch": row_ts_epoch(row), "late_receipt": int(row.get("settle_reason") == "late_receipt"),
        "weight": max(1, int(row.get("folded_attempt_count") or 1)) if kind == "usage_baseline_group" else 1,
        "has_root_limit": int("root_limit_usd" in row), "has_group_limit": int("billing_group_limit_usd" in row),
    }


class Txn:
    """One store transaction: reads, the cash view the admission rules use
    (``summary``/``exceeds_limit``, the historical writer-view API) and writes.
    Summary buckets read or written here are cached for the transaction."""

    def __init__(self, conn: sqlite3.Connection, root: pathlib.Path, *, write: bool) -> None:
        self.conn, self.root, self.writable = conn, root, write
        self.committed = False
        self._handoff = False
        self._buckets: Dict[Tuple[str, str], Bucket] = {}
        self._marker: Optional[list] = None
        self._receipt_tasks: set[str] = set()

    # -- reads
    def attempt(self, attempt_id: str) -> Optional[Dict[str, Any]]:
        record = self.conn.execute("SELECT * FROM attempts WHERE attempt_id=?", (str(attempt_id),)).fetchone()
        return None if record is None else _decode(record)

    def attempts(self, where: str = "1", params: Sequence[Any] = ()) -> List[Dict[str, Any]]:
        return [_decode(record) for record in self.conn.execute(
            f"SELECT * FROM attempts WHERE {where} ORDER BY seq", tuple(params))]

    def attempt_rows(self, where: str = "1", params: Sequence[Any] = ()) -> Captured:
        """``attempts``' rows in ONE step (``store_capture``), for ``decode_attempts`` after the read closed: an
        addressed selection (a root, a task, the open set); ``attempts()`` of every row steps row by row."""
        return capture(self.conn, f"SELECT {', '.join(_ATTEMPT_CAPTURE)} FROM attempts WHERE {where}", params,
                       _ATTEMPT_CAPTURE)

    def open_attempts(self, *, root_task_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """The open set (partial index): reserved/dispatched/unresolved rows
        and abandoned settlements that keep their late-receipt right."""
        return decode_attempts(self.open_attempt_rows(root_task_id=root_task_id))

    def open_attempt_rows(self, *, root_task_id: Optional[str] = None) -> Captured:
        """``open_attempts``' rows captured (``attempt_rows``), for ``decode_attempts`` after the read closed."""
        if root_task_id is None:
            return self.attempt_rows(_OPEN_PREDICATE)
        return self.attempt_rows(f"{_OPEN_PREDICATE} AND root_task_id=?", (str(root_task_id),))

    def summary_rows(self, reads: Sequence[Tuple[str, str, str]]) -> Captured:
        """The summary rows ``reads`` names, in ONE step, for ``SummaryRows`` after the read closed: ``(scope,
        key, "key")`` one row, ``(scope, prefix, "prefix")`` the keys under ``"<prefix>|"``, ``(scope, "",
        "scope")`` every key of the scope (``bucket``/``buckets``' selections)."""
        terms, params = [], []
        for scope, key, how in dict.fromkeys(reads):
            if how == "key":
                terms.append("(scope=? AND key=?)")
                params += [scope, key]
            elif how == "prefix":
                terms.append("(scope=? AND key>=? AND key<?)")
                params += [scope, f"{key}|", f"{key}}}"]
            else:
                terms.append("scope=?")
                params.append(scope)
        return capture(self.conn, f"SELECT {', '.join(_BUCKET_CAPTURE)} FROM summaries "
                       f"WHERE {' OR '.join(terms) or '0'}", params, _BUCKET_CAPTURE)

    def task_attempt_page(self, task_id: str, category: str, *, after: str = "",
                          limit: int) -> Optional[List[Dict[str, Any]]]:
        """Lifecycle facts of one task's attempts in one category, newest first, ``limit`` rows older than ``after``.

        ``after`` must name an attempt of this task and category (else ``None``): its first
        record is the cursor, so older rows stay reachable while newer ones join a fresh list.
        The task index addresses only this task (``INDEXED BY`` refuses a plan that scans the
        ledger); ordering still sorts the task's matching rows before the bounded page returns.
        No money, usage or payload fields."""
        before = 2 ** 63 - 1
        if after:
            cursor = self.conn.execute(_TASK_CURSOR_SQL, (str(after), str(task_id), str(category))).fetchone()
            if cursor is None:
                return None
            before = cursor["seq_first"]
        return [dict(record) for record in self.conn.execute(_TASK_PAGE_SQL, (
            str(task_id), str(category), before, int(limit)))]

    def review_seat_costs(self, category: str, *, limit: int) -> List[Dict[str, Any]]:
        """What one reviewer seat cost in one wave: per (wave, row), the summed cost of every
        physical attempt — its turns and its verdict-extraction calls, whatever model served each —
        with the distinct models seen (``models``, comma-joined), newest wave first, at most ``limit``.

        ``priced`` is false when any attempt of the seat had no known price; the category/time
        index addresses only this category. No payload fields."""
        return [dict(record) for record in self.conn.execute(
            "SELECT review_wave_id AS wave_id, review_slot_id AS row_id, GROUP_CONCAT(DISTINCT model) AS models, "
            "SUM(CAST(cost_usd AS REAL)) AS usd, MIN(COALESCE(pricing_known, 0)) AS priced, "
            "MAX(ts_last_epoch) AS last_epoch FROM attempts INDEXED BY attempts_category_time "
            "WHERE category=? AND COALESCE(review_wave_id, '') != '' AND COALESCE(review_slot_id, '') != '' "
            "GROUP BY review_wave_id, review_slot_id ORDER BY last_epoch DESC LIMIT ?",
            (str(category), int(limit)))]

    def bucket(self, scope: str, key: str = "") -> Bucket:
        address = (scope, str(key))
        if address not in self._buckets:
            record = self.conn.execute("SELECT * FROM summaries WHERE scope=? AND key=?", address).fetchone()
            self._buckets[address] = Bucket() if record is None else _bucket_of(record)
        return self._buckets[address]

    def buckets(self, scope: str, prefix: Optional[str] = None) -> Dict[str, Bucket]:
        """Every key of one scope, or the keys under ``"<id>|"`` with the prefix removed."""
        if prefix is None:
            cursor = self.conn.execute("SELECT * FROM summaries WHERE scope=? ORDER BY key", (scope,))
        else:
            cursor = self.conn.execute("SELECT * FROM summaries WHERE scope=? AND key>=? AND key<? ORDER BY key",
                                       (scope, f"{prefix}|", f"{prefix}}}"))
        found = {}
        for record in cursor:
            address = (scope, record["key"])
            if address not in self._buckets:
                self._buckets[address] = _bucket_of(record)
            found[record["key"] if prefix is None else record["key"][len(prefix) + 1:]] = self._buckets[address]
        return found

    def dirty_owner_ids(self) -> List[str]:
        return [owner for owner, _revision in self.dirty_owners()]

    def dirty_owners(self, *, since: Optional[int] = None) -> List[Tuple[str, int]]:
        """The owners whose stored cost projection may be behind
        (``dirty_owners``) that own a non-review attempt or imported aggregate
        (rows under a review's own custody, ``_usage_rows.REVIEW_CUSTODY_KEYS``,
        never made an owner a candidate). Each owner's check is an indexed
        lookup of its own rows. ``since`` selects only newer writes in this epoch."""
        changed = "d.revision > ? AND " if since is not None else ""
        order = "revision, owner_id" if since is not None else "owner_id"
        return [(record["owner_id"], record["revision"]) for record in self.conn.execute(
            "SELECT owner_id, revision FROM dirty_owners AS d WHERE " + changed + "EXISTS (SELECT 1 FROM attempts AS a "
            "WHERE (a.task_id = d.owner_id OR a.root_task_id = d.owner_id) "
            "AND COALESCE(a.kind, 'attempt') IN ('attempt', 'usage_baseline_group') "
            "AND COALESCE(a.review_skill, '') = '' "
            "AND COALESCE(a.review_slot_id, '') = '') ORDER BY " + order, (() if since is None else (since,)))]

    def keys(self, scope: str) -> List[str]:
        """Every key of one summary scope (no bucket decoded)."""
        return [record["key"] for record in self.conn.execute(
            "SELECT key FROM summaries WHERE scope=? ORDER BY key", (scope,))]

    def top_keys(self, scope: str, limit: int) -> List[str]:
        """Keys of the costliest buckets (``accounted_num`` orders only; amounts
        are read from the exact decimal text)."""
        return [record["key"] for record in self.conn.execute(
            "SELECT key FROM summaries WHERE scope=? ORDER BY accounted_num DESC, key ASC LIMIT ?",
            (scope, max(0, int(limit))))]

    def binding(self, scope: str, key: str) -> Optional[Dict[str, Any]]:
        record = self.conn.execute("SELECT binding FROM bindings WHERE scope=? AND key=?", (scope, key)).fetchone()
        return None if record is None else json.loads(record["binding"], parse_float=LiteralFloat)

    def meta(self, key: str) -> Any:
        record = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return None if record is None else json.loads(record["value"])

    def set_meta(self, key: str, value: Any) -> None:
        """``None`` removes the key (the export mark, once the export is abandoned)."""
        if value is None:
            self.conn.execute("DELETE FROM meta WHERE key=?", (key,))
        else:
            self.conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, _json(value)))
        if key == "export" and value is not None:
            self._handoff = True

    def marker(self) -> list:
        """``[epoch, seq]``: continues the journal's ``[compaction_epoch, seq]``."""
        if self._marker is None:
            self._marker = list(self.meta("publication_marker") or [0, 0])
        return list(self._marker)

    # -- the historical writer-view API (admission rules)
    def totals(self, root_task_id: Optional[str] = None, billing_group_id: Optional[str] = None) -> tuple:
        if root_task_id is not None and billing_group_id is not None:
            raise ValueError("select one monetary axis")
        if billing_group_id is not None:
            return self.bucket("group", billing_group_id).cash
        return self.bucket("global").cash if root_task_id is None else self.bucket("root", root_task_id).cash

    def summary(self, root_task_id: Optional[str] = None, *, billing_group_id: Optional[str] = None) -> dict:
        return render_cash(self.totals(root_task_id, billing_group_id))

    def exceeds_limit(self, limit, *, root_task_id=None, billing_group_id=None) -> bool:
        return exceeds_limit(self.totals(root_task_id, billing_group_id), limit)

    # -- writes
    def record_evidence(self, attempt_id: str, fields: dict, *, expected_revision: int) -> bool:
        """CAS physical evidence only; money, state and projection debt stay unchanged."""
        if not self.writable or self.committed:
            raise UsageAccountingError("usage store transaction is not writable")
        if not fields or set(fields) - {"physical_failure", "provider_receipt_binding"}:
            raise ValueError("unsupported physical evidence fields")
        record = self.conn.execute("SELECT extra, revision FROM attempts WHERE attempt_id=?",
                                   (str(attempt_id),)).fetchone()
        if record is None or record["revision"] != expected_revision:
            return False
        extra = json.loads(record["extra"])
        if "physical_failure" in extra:
            fields = {key: value for key, value in fields.items() if key != "physical_failure"}
        if not fields or all(extra.get(key) == value for key, value in fields.items()):
            return True
        extra.update(fields)
        return self.conn.execute("UPDATE attempts SET extra=?, revision=revision+1 WHERE attempt_id=? AND revision=?",
                                 (_json(extra), str(attempt_id), expected_revision)).rowcount == 1

    def ack_dirty_owner(self, owner_id: str, revision: int) -> bool:
        """A concurrent receipt with a later revision keeps its projection debt."""
        if not self.writable or self.committed:
            raise UsageAccountingError("usage store transaction is not writable")
        return self.conn.execute("DELETE FROM dirty_owners WHERE owner_id=? AND revision=?",
                                 (owner_id, revision)).rowcount == 1

    def record_recovery(self, attempt_id: str, recovery: dict, *, expected_revision: int) -> bool:
        """Record custody evidence without changing money or the late-receipt right.

        The row revision also fences observations: a receipt that won the race
        remains authoritative. Summary buckets and dirty owners do not change.
        """
        if not self.writable or self.committed:
            raise UsageAccountingError("usage store transaction is not writable")
        return self.conn.execute(
            "UPDATE attempts SET extra=json_set(extra, '$.recovery', json(?)), revision=revision+1 "
            f"WHERE attempt_id=? AND revision=? AND {_OPEN_PREDICATE}",
            (_json(recovery), str(attempt_id), expected_revision)).rowcount == 1

    def write(self, row: Dict[str, Any], previous: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Insert a new attempt (``previous=None``) or replace ``previous`` by its
        next row, maintaining every summary, binding, dirty owner, one-shot
        identity and the marker in this transaction. Returns the stored row."""
        if not self.writable or self.committed:
            raise UsageAccountingError("usage store transaction is not writable")
        epoch, last = self.marker()
        seq = last + 1
        revision = int(previous["revision"]) + 1 if previous else 1
        materialized = durable_literals({**{key: value for key, value in row.items() if key != "revision"},
                                         "seq": seq, "ts": str(row.get("ts") or utc_now_iso())})
        from ouroboros.usage_origin import stamp_root_cause
        stamp_root_cause(self, materialized)
        validate_row_fields(materialized, seq)
        validate_transition(materialized, None if previous is None else str(previous.get("state") or ""),
                            late_receipt_eligible(previous), seq, previous_row=previous)
        columns = _encode(materialized)
        stamps = {} if previous is None else self.conn.execute(
            "SELECT seq_first, ts_reserved AS _ts_reserved, ts_dispatched AS _ts_dispatched FROM attempts "
            "WHERE attempt_id=?", (str(previous["attempt_id"]),)).fetchone()
        columns.update(_derived(materialized, dict(stamps) if stamps else None, revision,
                                stamps["seq_first"] if stamps else seq))
        attempt_id = str(materialized["attempt_id"])
        if previous is None:
            try:
                self.conn.execute(f"INSERT INTO attempts ({','.join(_COLUMNS)}) VALUES ({','.join('?' * len(_COLUMNS))})",
                                  tuple(columns[name] for name in _COLUMNS))
            except sqlite3.IntegrityError as exc:
                raise UsageAccountingError(f"usage attempt already recorded: {attempt_id}") from exc
        else:
            names = [name for name in _COLUMNS if name != "attempt_id"]
            updated = self.conn.execute(
                f"UPDATE attempts SET {','.join(f'{name}=?' for name in names)} WHERE attempt_id=? AND revision=?",
                (*(columns[name] for name in names), attempt_id, int(previous["revision"]))).rowcount
            if updated != 1:
                raise UsageAccountingError(f"usage attempt changed under its writer: {attempt_id}")
        stored = self.attempt(attempt_id)
        touched = self._apply(previous, -1) | self._apply(stored, 1)
        for scope, key in touched:
            bucket = self._buckets[(scope, key)]
            if bucket.rows:
                self.conn.execute(
                    f"INSERT OR REPLACE INTO summaries ({','.join(_SUMMARY_COLUMNS)}) "
                    f"VALUES ({','.join('?' * len(_SUMMARY_COLUMNS))})", _bucket_record(scope, key, bucket))
            else:
                self.conn.execute("DELETE FROM summaries WHERE scope=? AND key=?", (scope, key))
        bind_root_attribution(self.conn, stored)
        for owner in dict.fromkeys((str(stored.get("task_id") or ""), str(stored.get("root_task_id") or ""))):
            if owner:
                self._receipt_tasks.add(owner)
                self.conn.execute("INSERT INTO dirty_owners (owner_id, revision) VALUES (?, ?) ON CONFLICT(owner_id) "
                                  "DO UPDATE SET revision=excluded.revision", (owner, seq))
        if previous is None and is_one_shot_kind(str(stored.get("kind") or "")):
            self.conn.execute("INSERT INTO one_shots VALUES (?,?,?,?,?,?,?,?)", (
                attempt_id, *(str(stored.get(key) or "") for key in (
                    "kind", "task_id", "root_task_id", "subscription_route", "source", "category")),
                one_shot_identity(stored)))
        self._marker = [epoch, seq]
        self.conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('publication_marker', ?)",
                          (_json(self._marker),))
        return stored

    def _apply(self, row: Optional[Dict[str, Any]], sign: int) -> set:
        if row is None:
            return set()
        delta = summary_delta(row)
        touched = set()
        for scope, key, cap in summary_keys(row):
            self.bucket(scope, key).add(delta, sign, cap=cap if scope in {"root", "group"} else None)
            touched.add((scope, key))
        return touched

    def recorded_one_shot(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """The stored one-shot of ``row``'s id when its identity is identical;
        ``None`` when absent; a conflicting identity is refused."""
        attempt_id = str(row["attempt_id"])
        record = self.conn.execute("SELECT identity FROM one_shots WHERE attempt_id=?", (attempt_id,)).fetchone()
        if record is None:
            if self.attempt(attempt_id) is not None:
                raise UsageAccountingError(f"conflicting settled-row identity: {attempt_id}")
            return None
        if record["identity"] != one_shot_identity(row):
            raise UsageAccountingError(f"conflicting settled-row identity: {attempt_id}")
        return self.attempt(attempt_id)

    def commit(self, *, wait_sec: Optional[float] = None) -> None:
        if not self.committed:
            if wait_sec is not None:
                self.conn.execute(f"PRAGMA busy_timeout = {max(0, int(wait_sec * 1000))}")
            _commit(self.conn)
            self.committed = True


# ---- connection, lock tiers, holds (the SQLite core is ouroboros.store) ------
_READY: Dict[str, Tuple[tuple, str]] = {}  # store path -> (file identity, lock tier)


def _header_tier(path: pathlib.Path) -> Optional[str]:
    return _core.header_tier(path, _APPLICATION_ID)


def hand_off(root: pathlib.Path | str) -> bool:
    """``store.hand_off`` for the money store: a WAL file is checkpointed into DELETE mode."""
    path = pathlib.Path(root) / STORE_REL
    if not _core.wal_file(path):
        return False
    tier = _header_tier(path)
    if tier is None:
        raise UsageLedgerCorrupt(f"usage store unreadable at handoff: {path}")
    try:
        _core.prepare_handoff(path, tier, wait_sec=USAGE_LOCK_TIMEOUT_SEC)
    except _core.StoreBusy as exc:
        raise UsageLockUnavailable(str(exc), reason=exc.reason) from exc
    return True


def _connect(path: pathlib.Path, tier: str, *, create: bool = False, wait_sec: float = 0.0,
             shared: bool = False) -> sqlite3.Connection:
    """The shared connection with the money error vocabulary (``ouroboros.store.connect``)."""
    try:
        return _core.connect(path, tier, create=create, wait_sec=wait_sec, shared=shared)
    except sqlite3.Error as exc:
        raise _typed(exc) from exc


def _commit(conn: sqlite3.Connection) -> None:
    try:
        _core.commit(conn)
    except sqlite3.OperationalError as exc:
        if _busy(exc):
            error = UsageLockUnavailable(f"usage store commit did not complete: {exc}", reason="contention")
            error.rollback_confirmed = getattr(exc, "rollback_confirmed", False)
            raise error from exc
        raise UsageAccountingError(f"usage store commit failed: {exc}") from exc


def _remaining_wait(seconds, remaining_sec):
    """A caller-owned deadline narrows admission; ordinary waits keep their contract."""
    return min(seconds, remaining_sec()) if remaining_sec is not None else seconds


def store_tier(root: pathlib.Path, *, wait_sec: float, migrate: bool = True, remaining_sec=None) -> str:
    """The lock tier of the ready store. A missing store is migrated first
    (``migrate_from_journal``; ``migrate=False`` reports it unavailable
    instead). A caller that finds the import already running waits only its own
    budget. A store file that exists is always a completed import (imports are
    published by rename), so one that cannot be read is corruption, refused for
    every caller and never replaced."""
    path = pathlib.Path(root) / STORE_REL
    identity = _identity(path)
    cached = _READY.get(str(path))
    if cached is not None and identity is not None and cached[0] == identity:
        return cached[1]
    if cached is not None:
        forget(root)
    lock = _process_lock(path)
    if not lock.acquire(timeout=max(0.0, _remaining_wait(wait_sec, remaining_sec))):
        error = UsageLockUnavailable("usage store migration in progress", reason="contention")
        error.rollback_confirmed = True  # this caller acquired neither lock nor transaction
        raise error
    try:
        if _identity(path) is None:
            if not migrate:
                raise UsageLockUnavailable("usage store not migrated yet", reason="contention")
            migrate_from_journal(root)
        identity, tier = _identity(path), _header_tier(path)
        if identity is None or tier is None or (_stored_meta(path, tier, wait_sec, remaining_sec=remaining_sec).get("import") or {}).get("status") != "completed":
            raise UsageLedgerCorrupt(f"usage store unreadable or without a completed import: {path}")
        _READY[str(path)] = (identity, tier)
        return tier
    finally:
        lock.release()


def _stored_meta(path: pathlib.Path, tier: str, wait_sec: float, *, remaining_sec=None) -> Dict[str, Any]:
    """``meta.import`` and ``meta.export`` of a store file (``{}`` for one that cannot be read)."""
    with contextlib.ExitStack() as stack:
        if tier == TIER_NAME:
            stack.enter_context(_named_lock(path.parent.parent, LOCK_REL.name,
                                            timeout_sec=_remaining_wait(wait_sec, remaining_sec), stale_sec=90.0))
        if path.name == STORE_REL.name:
            conn = stack.enter_context(_core.connection(path, tier, wait_sec=_remaining_wait(wait_sec, remaining_sec), opener=_connect))
        else:  # retired copies must be closed before abandonment can rename them back
            conn = stack.enter_context(contextlib.closing(_connect(path, tier, wait_sec=_remaining_wait(wait_sec, remaining_sec))))
        if remaining_sec is not None:
            conn.execute(f"PRAGMA busy_timeout = {int(_remaining_wait(wait_sec, remaining_sec) * 1000)}")
        try:
            records = conn.execute("SELECT key, value FROM meta WHERE key IN ('import', 'export')").fetchall()
        except sqlite3.OperationalError as exc:
            if _busy(exc):
                error = UsageLockUnavailable(f"usage store busy: {exc}", reason="contention")
                error.rollback_confirmed = not conn.in_transaction
                raise error from exc
            return {}
        except sqlite3.DatabaseError:
            return {}
        return {record["key"]: json.loads(record["value"]) for record in records}


def _sha256_file(path: pathlib.Path) -> Optional[str]:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError:
        return None


@contextlib.contextmanager
def hold(root: pathlib.Path | str, *, timeout_sec: float = USAGE_LOCK_TIMEOUT_SEC,
         write: bool = True, migrate: bool = True, exporting: bool = False, remaining_sec=None) -> Iterator[Txn]:
    """Bounded money acquisition (``_usage_wait``); timeout raises UsageLockUnavailable.
    Writes use BEGIN IMMEDIATE and wait for readers at COMMIT. An export mark
    refuses writes with UsageExportInProgress; only export/abandonment use exporting=True.
    Receipt callbacks run after the transaction and name lock have closed.
    ``remaining_sec`` optionally narrows each acquisition to a caller's one deadline
    (query_data SQL); it never enlarges a wait or changes other callers' defaults.
    """
    root = pathlib.Path(root)
    tier = store_tier(root, wait_sec=timeout_sec, migrate=migrate, remaining_sec=remaining_sec)
    with contextlib.ExitStack() as stack:
        if tier == TIER_NAME:
            stack.enter_context(_named_lock(root, LOCK_REL.name,
                                            timeout_sec=_remaining_wait(timeout_sec, remaining_sec), stale_sec=90.0))
        conn = stack.enter_context(_core.connection(root / STORE_REL, tier,
                                                    wait_sec=_remaining_wait(timeout_sec, remaining_sec), opener=_connect))
        if remaining_sec is not None:
            conn.execute(f"PRAGMA busy_timeout = {int(_remaining_wait(timeout_sec, remaining_sec) * 1000)}")
        txn = Txn(conn, root, write=write)
        handoff = False
        try:  # the enforced tier's COMMIT waits for readers to drain
            with _core.transaction(conn, write=write, finish=txn.commit,
                                   commit_wait_sec=USAGE_LOCK_TIMEOUT_SEC if tier == TIER_ENFORCED else None):
                handoff = txn.meta("export") is not None
                if write and not exporting and handoff:
                    raise UsageExportInProgress(f"usage store is being exported for a switch; money writes are "
                                                f"refused until it completes or is abandoned: {root / STORE_REL}")
                yield txn
            handoff = handoff or txn._handoff
        except sqlite3.Error as exc:
            raise _typed(exc) from exc
        finally:
            if handoff:
                stack.close()
                with _named_lock(root, LOCK_REL.name, timeout_sec=timeout_sec, stale_sec=90.0) \
                        if tier == TIER_NAME else contextlib.nullcontext():
                    try:
                        _core.prepare_handoff(root / STORE_REL, tier, wait_sec=timeout_sec)
                    except _core.StoreBusy as exc:
                        raise UsageLockUnavailable(str(exc), reason=exc.reason) from exc
                    except sqlite3.Error as exc:
                        raise _typed(exc) from exc


    if txn._receipt_tasks:
        from ouroboros.swarm_receipt import usage_changed
        usage_changed(root, txn._receipt_tasks)


def _typed(exc: sqlite3.Error) -> UsageAccountingError:
    """The money error of a SQLite failure: contention is ``UsageLockUnavailable``
    (the waits' business), a damaged file is ``UsageLedgerCorrupt`` (unknown,
    never zero), anything else an ``UsageAccountingError``."""
    if isinstance(exc, sqlite3.OperationalError) and _busy(exc):
        error = UsageLockUnavailable(f"usage store busy: {exc}", reason="contention")
        if hasattr(exc, "rollback_confirmed"):
            error.rollback_confirmed = exc.rollback_confirmed
        return error
    if isinstance(exc, sqlite3.DatabaseError) and not isinstance(
            exc, (sqlite3.OperationalError, sqlite3.IntegrityError, sqlite3.ProgrammingError)):
        return UsageLedgerCorrupt(f"usage store unreadable: {exc}")
    return UsageAccountingError(f"usage store unavailable: {exc}")


def read(root: pathlib.Path | str, *, allow_stale: bool = False):
    """A read transaction: ``allow_stale`` is a display read (the short display wait, then
    the fact is reported unavailable). The server imports at lifespan start, on every door,
    before any request or worker. A display never imports: while a journal or a never-imported
    pre-ledger event chain waits for that job it reports the store unavailable (never zero);
    with neither it creates the empty store. A non-display reader that finds no store (a tool
    or a test without a server) runs the one import itself; a reader arriving while it runs
    waits only its own budget, then reports the store unavailable."""
    from ouroboros.runtime_limits import USAGE_DISPLAY_LOCK_TIMEOUT_SEC
    from ouroboros.usage_journal import completed_legacy_watermark

    if allow_stale:
        root, events = pathlib.Path(root), pathlib.Path(root) / "logs" / "events.jsonl"
        waiting = (root / LEDGER_REL).is_file() or (completed_legacy_watermark(root) is None
                                                    and events.is_file() and events.stat().st_size > 0)
        return hold(root, timeout_sec=USAGE_DISPLAY_LOCK_TIMEOUT_SEC, write=False,
                    migrate=_identity(root / STORE_REL) is not None or not waiting)
    return hold(root, timeout_sec=USAGE_LOCK_TIMEOUT_SEC, write=False)


def integrity_degraded(root: pathlib.Path) -> bool:
    """A journal row was once quarantined: that money is unknown, disclosed."""
    return (pathlib.Path(root) / QUARANTINE_REL).is_file()


def read_usage_records(root: pathlib.Path | str) -> List[Dict[str, Any]]:
    """EVERY attempt row in write order. Explicit audits and the export only;
    an ordinary path reads summaries and addressed rows (a test enforces it)."""
    with read(root) as txn:
        return txn.attempts()


# ---- migration: the one-time import of the retired journal ------------------

def migrate_from_journal(root: pathlib.Path | str, *, initialize_journal: bool = False) -> Dict[str, Any]:
    """Create the store from the live journal (a lifecycle job; writers quiescent).

    ``initialize_journal`` is the lifespan boot's alone: it brings a completed store to this
    runtime's journal (``store_journal``); the supervisor's revival and lazy readers keep it.

    Idempotent through ``meta.import.status``: a completed store is never
    re-imported (an interrupted journal rename is retried); an unfinished
    import (a sibling build file never published) is discarded and redone; a
    published store that cannot be read is corruption and is refused, never
    replaced. The live journal is read once with the
    validated reader; for every attempt id its LAST row is imported (open rows
    included), imported aggregates keep their weight, bindings are the
    ``BindingIndex`` fold of every live row, the marker continues the journal's
    ``[compaction_epoch, seq]``. An install whose pre-ledger telemetry import
    never completed imports that snapshot in the same job. The store is built
    in a sibling file and published by an atomic rename; the journal stays in
    place (evidence an older release reads after a rollback; never read again
    here). Timing and counts go to ``logs/supervisor.jsonl``."""
    from ouroboros.platform_layer import kernel_file_locks_enforced

    root = pathlib.Path(root)
    path = root / STORE_REL
    started = time.monotonic()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _process_lock(path):
        with _named_lock(root, IMPORT_LOCK_NAME, timeout_sec=600.0, stale_sec=600.0):
            for stale in path.parent.glob(f"{path.name}.import-*"):
                with contextlib.suppress(OSError):
                    stale.unlink()  # an unfinished import never published: redone from scratch
            if _identity(path) is not None:
                tier = _header_tier(path)
                if initialize_journal and tier and _core.wal_file(path):
                    _select_journal(root, path, tier)  # leave WAL this runtime must not share before reading it
                stored = _stored_meta(path, tier, USAGE_LOCK_TIMEOUT_SEC) if tier else {}
                status = stored.get("import")
                if not status or status.get("status") != "completed":
                    raise UsageLedgerCorrupt(f"usage store unreadable or without a completed import: {path}")
                report = {"status": "already_completed", **_retire_journal(root, status)}
                if initialize_journal and not stored.get("export"):  # an exporting store stays as it is handed off
                    report["journal_selected"] = _select_journal(root, path, tier)
                    report["window_index"] = _ensure_read_index(root, path, tier, "attempts_root_time", ROOT_TIME_INDEX)
                    report["dirty_owner_index"] = _ensure_read_index(root, path, tier, "dirty_owners_revision", DIRTY_REVISION_INDEX)
            else:
                tier = TIER_ENFORCED if kernel_file_locks_enforced(root / LOCK_REL) else TIER_NAME
                report = _import(root, path, tier)
        _READY.pop(str(path), None)
    report["duration_seconds"] = round(time.monotonic() - started, 3)
    try:
        append_jsonl(root / "logs" / "supervisor.jsonl",
                     {"ts": utc_now_iso(), "type": "usage_store_migration", "phase": report["status"], **report})
    except Exception:
        log.warning("usage store migration report not recorded", exc_info=True)
    return report


def _ensure_read_index(root: pathlib.Path, path: pathlib.Path, tier: str, name: str, ddl: str) -> str:
    """A read index on a store an older body created: built once at the lifespan boot while
    writers are quiescent, never by a reader; ``present`` costs one schema lookup. A failed build
    leaves readers on existing indexes (same answer, broader scan) until the next boot."""
    try:
        with contextlib.ExitStack() as stack:
            if tier == TIER_NAME:
                stack.enter_context(_named_lock(root, LOCK_REL.name, timeout_sec=USAGE_LOCK_TIMEOUT_SEC, stale_sec=90.0))
            conn = stack.enter_context(_core.connection(path, tier, wait_sec=USAGE_LOCK_TIMEOUT_SEC, opener=_connect))
            if conn.execute("SELECT 1 FROM sqlite_master WHERE type='index' AND name=?", (name,)).fetchone():
                return "present"
            with _core.transaction(conn, write=True, commit_wait_sec=USAGE_LOCK_TIMEOUT_SEC):
                conn.execute(ddl)
        return "created"
    except (sqlite3.Error, _core.StoreError, UsageAccountingError) as exc:
        log.warning("usage store index %s deferred to the next boot: %s", name, exc)
        return f"deferred: {type(exc).__name__}"


def _select_journal(root: pathlib.Path, path: pathlib.Path, tier: str) -> bool:
    """``store_journal.configure_journal`` for the lifespan boot, a name-tier file under the money lock."""
    from ouroboros.store_journal import configure_journal

    with _named_lock(root, LOCK_REL.name, timeout_sec=USAGE_LOCK_TIMEOUT_SEC, stale_sec=90.0) \
            if tier == TIER_NAME else contextlib.nullcontext():
        return configure_journal(path, tier, wait_sec=USAGE_LOCK_TIMEOUT_SEC)


def _import(root: pathlib.Path, path: pathlib.Path, tier: str) -> Dict[str, Any]:
    from ouroboros import usage_journal as journal

    # The pre-ledger source snapshot (I/O, archive copies) runs before the
    # journal lock is taken; only the journal read and rename hold it.
    snapshot = journal.legacy_snapshot(root) if journal.completed_legacy_watermark(root) is None else None
    with _named_lock(root, LOCK_REL.name, timeout_sec=USAGE_LOCK_TIMEOUT_SEC, stale_sec=90.0):
        journal_present = (root / LEDGER_REL).is_file()
        records, source = journal.read_journal_records(root)
        legacy_watermark = None
        if snapshot is not None:
            missing, legacy_watermark = journal.legacy_candidates(
                snapshot, {str(row.get("attempt_id") or "") for row in records})
            records = [*records, *({**row, "seq": len(records) + index, "ts": str(row.get("ts") or utc_now_iso())}
                                   for index, row in enumerate(missing, 1))]
        retired = None  # the journal stays in place: an older release still reads it after a rollback
        tmp = path.with_name(f"{path.name}.import-{os.getpid()}-{uuid.uuid4().hex[:8]}")
        try:
            counts = _build(tmp, tier, records, {
                "status": "completed", "imported_at": utc_now_iso(), "schema_version": SCHEMA_VERSION,
                "lock_tier": tier, "retired_as": retired, "legacy": legacy_watermark,
                "quarantine_present": (root / QUARANTINE_REL).is_file(),
                "source": {"path": LEDGER_REL.as_posix(), "present": journal_present, "size": len(source),
                           "sha256": hashlib.sha256(source).hexdigest(), "rows": len(records)}})
            os.replace(tmp, path)
        finally:
            with contextlib.suppress(FileNotFoundError):
                tmp.unlink()
        _fsync_directory(path.parent)
        if legacy_watermark is not None:
            from ouroboros.utils import atomic_write_json

            atomic_write_json(root / journal.IMPORT_REL, legacy_watermark, trailing_newline=True, fsync=True)
            append_jsonl(root / "logs" / "events.jsonl", {"type": "usage_import_completed", **legacy_watermark})
    return {"status": "completed", "lock_tier": tier, "journal_rows": len(records), "retired_as": retired, **counts}


def _build(tmp: pathlib.Path, tier: str, records: Sequence[Dict[str, Any]], provenance: Dict[str, Any]) -> Dict[str, int]:
    last: Dict[str, Dict[str, Any]] = {}
    first: Dict[str, int] = {}
    stamps: Dict[str, Dict[str, str]] = {}
    revisions: Dict[str, int] = {}
    bindings = BindingIndex()
    causes = {}
    header = None
    for row in records:
        bindings.fold(row)
        if row.get("root_task_id") and row.get("consciousness_cause"):
            causes.setdefault(str(row["root_task_id"]), {"cause": row["consciousness_cause"]})
        attempt_id, state = str(row["attempt_id"]), str(row.get("state") or "")
        if str(row.get("kind") or "") == "usage_baseline":
            header = row
            continue
        first.setdefault(attempt_id, int(row["seq"]))
        revisions[attempt_id] = revisions.get(attempt_id, 0) + 1
        if state in {"reserved", "dispatched"}:
            stamps.setdefault(attempt_id, {})[f"_ts_{state}"] = str(row.get("ts") or "")
        last[attempt_id] = row
    epoch = int((header or {}).get("compaction_epoch") or 0)
    last_seq = max((int(row["seq"]) for row in records), default=0)
    conn = _connect(tmp, tier, create=True)
    try:
        # BEGIN belongs inside executescript, which first commits any pending transaction.
        conn.executescript("BEGIN IMMEDIATE;\n" + _SCHEMA)
        conn.execute(f"PRAGMA application_id = {_APPLICATION_ID[tier]}")
        buckets: Dict[Tuple[str, str], Bucket] = {}
        for attempt_id, row in sorted(last.items(), key=lambda item: int(item[1]["seq"])):
            materialized = durable_literals(row)
            columns = _encode(materialized)
            columns.update(_derived(materialized, stamps.get(attempt_id), revisions[attempt_id], first[attempt_id]))
            conn.execute(f"INSERT INTO attempts ({','.join(_COLUMNS)}) VALUES ({','.join('?' * len(_COLUMNS))})",
                         tuple(columns[name] for name in _COLUMNS))
            stored = _decode(conn.execute("SELECT * FROM attempts WHERE attempt_id=?", (attempt_id,)).fetchone())
            delta = summary_delta(stored)
            for scope, key, cap in summary_keys(stored):
                buckets.setdefault((scope, key), Bucket()).add(delta, cap=cap if scope in {"root", "group"} else None)
            if is_one_shot_kind(str(stored.get("kind") or "")):
                conn.execute("INSERT INTO one_shots VALUES (?,?,?,?,?,?,?,?)", (
                    attempt_id, *(str(stored.get(key) or "") for key in (
                        "kind", "task_id", "root_task_id", "subscription_route", "source", "category")),
                    one_shot_identity(stored)))
        conn.executemany(f"INSERT INTO summaries ({','.join(_SUMMARY_COLUMNS)}) "
                         f"VALUES ({','.join('?' * len(_SUMMARY_COLUMNS))})",
                         [_bucket_record(scope, key, bucket) for (scope, key), bucket in buckets.items() if bucket.rows])
        bound = [(scope, key, _json(durable_literals(binding))) for scope, index in (
            ("root", bindings.roots), ("group", bindings.groups), ("consciousness", causes))
                 for key, binding in index.items()]
        conn.executemany("INSERT INTO bindings (scope, key, binding) VALUES (?, ?, ?)", bound)
        # The visible one-time import also seeds stale completed projections.
        from ouroboros.terminal_cost_reconciliation import imported_projection_debt

        owners = list(imported_projection_debt(tmp.parent.parent, buckets,
                                               degraded=bool(provenance.get("quarantine_present"))))
        conn.executemany("INSERT INTO dirty_owners (owner_id, revision) VALUES (?, ?)",
                         [(owner, last_seq) for owner in owners])
        counts = {"attempts": len(last), "summaries": sum(1 for bucket in buckets.values() if bucket.rows),
                  "bindings": len(bound), "dirty_owners": len(owners), "epoch": epoch, "last_seq": last_seq}
        meta = {"schema_version": SCHEMA_VERSION, "lock_tier": tier, "publication_marker": [epoch, last_seq],
                "import": {**provenance, "header": durable_literals(header) if header else None, "counts": counts}}
        conn.executemany("INSERT INTO meta (key, value) VALUES (?, ?)",
                         [(key, _json(value)) for key, value in meta.items()])
        _commit(conn)
    finally:
        conn.close()
    return counts


def _retire_journal(root: pathlib.Path, status: Dict[str, Any]) -> Dict[str, Any]:
    """After a completed import the journal stays in place as evidence an older release
    reads after a rollback; nothing here reads its bytes. A size other than the imported
    one means an older release appended after a rollback: disclosed, never merged (the
    export before a downgrade keeps money exact)."""
    journal = root / LEDGER_REL
    try:
        size = journal.stat().st_size
    except FileNotFoundError:
        return {}
    imported = (status.get("source") or {}).get("size")
    if size == imported:
        return {"journal": "kept"}
    return {"journal": "changed_after_import", "journal_size": size, "imported_size": imported}


def _export_rows(txn: Txn, provenance: Dict[str, Any]) -> Tuple[list, int, int]:
    """The journal rows an older release accepts, from the frozen store."""
    from ouroboros import usage_journal as journal

    records = txn.conn.execute("SELECT * FROM attempts ORDER BY seq_first, seq").fetchall()
    aggregates = [_decode(record) for record in sorted(records, key=lambda record: record["seq"])
                  if record["kind"] == "usage_baseline_group"]
    attempts = [(_decode(record), record["ts_reserved"], record["ts_dispatched"]) for record in records
                if record["kind"] != "usage_baseline_group"]
    bindings = {record["key"]: json.loads(record["binding"], parse_float=LiteralFloat) for record in
                txn.conn.execute("SELECT key, binding FROM bindings WHERE scope='root'")}
    rows = journal.export_rows(provenance.get("header"), aggregates, attempts, bindings)
    journal._validate_records(rows)
    return rows, len(attempts), len(aggregates)


def _retired_for(root: pathlib.Path, journal_sha256: Optional[str]) -> Optional[Tuple[pathlib.Path, Dict[str, Any]]]:
    """The newest retired store (``usage.sqlite.exported-*``) whose export mark planned the
    journal with ``journal_sha256``: proof the journal beside the absent store is that
    export's own. ``None`` otherwise (an older switch's copy proves nothing)."""
    path = root / STORE_REL
    if journal_sha256 is None:
        return None
    try:
        names = sorted((name for name in os.listdir(path.parent) if name.startswith(path.name + ".exported-")), reverse=True)
    except OSError:
        return None
    for name in names:
        retired = path.with_name(name)
        tier = _header_tier(retired)
        mark = _stored_meta(retired, tier, USAGE_LOCK_TIMEOUT_SEC).get("export") if tier else None
        if mark and (mark.get("journal") or {}).get("sha256") == journal_sha256:
            return retired, mark
    return None


def export_refusal(root: pathlib.Path | str, *, resume: bool = False) -> str:
    """Why ``export_journal`` would refuse now; ``""`` when it can run (or, ``resume``, complete
    an interrupted run whose rename already landed). A kept journal must be absent, the one the
    import kept unchanged (the import's recorded digest) or this export's own: the mark's digest,
    or a partial write of this export once the mark shows the kept one already set aside
    (``_own_partial``); sizes prove nothing."""
    root = pathlib.Path(root)
    path, kept = root / STORE_REL, root / LEDGER_REL
    kept_sha = _sha256_file(kept)
    if _identity(path) is None:
        if kept_sha is None:
            return f"no usage store to export: {path}"
        if resume and _retired_for(root, kept_sha):
            return ""
        return f"refusing to overwrite an existing journal without a store: {kept}"
    if kept_sha is None:
        return ""
    tier = _header_tier(path)
    if tier is None:
        return f"usage store unreadable: {path}"
    stored = _stored_meta(path, tier, USAGE_LOCK_TIMEOUT_SEC)
    mark = stored.get("export") or {}
    if kept_sha in {((stored.get("import") or {}).get("source") or {}).get("sha256"), (mark.get("journal") or {}).get("sha256")}:
        return ""
    if mark and _own_partial(root, mark):
        return ""
    return f"the money journal changed after the import; move it aside to export: {kept}"


def _own_partial(root: pathlib.Path, mark: Dict[str, Any]) -> bool:
    """Whether a kept journal that matches neither digest is this export's own interrupted write:
    the mark recorded no journal to set aside, or the one it named is already aside."""
    return not mark.get("pre_export_as") or (root / LEDGER_REL).with_name(mark["pre_export_as"]).exists()


def export_journal(root: pathlib.Path | str, *, resume: bool = False) -> Dict[str, Any]:
    """Write ``state/usage_attempts.jsonl`` from the store as a journal an older
    release accepts (``usage_journal.export_rows``), then retire the store.

    Offline, with the server stopped: the step before checking out an older
    release (a git revert alone is not a data rollback). The barrier is the
    export mark: ``meta.export`` (the planned journal's size, sha256 and marker,
    the retired and pre-export names, the ``state.json`` marker it replaces) is
    committed under the store's write lock BEFORE the journal is written, and
    from that commit every money write is refused typed (``hold``), so no late
    writer commits into the authority being exported. Then the files: the journal
    the import kept is set aside (``usage_attempts.jsonl.pre-export-<UTC>``), the
    journal is written, the legacy-import watermark the store recorded is
    restored, the ``state.json`` freshness marker is aligned with the journal's
    own ``[epoch, seq]``, the store is renamed ``usage.sqlite.exported-<UTC>`` so
    a later upgrade imports the journal again, including what the older release
    adds. Every step is idempotent through the mark: an interrupted run resumes
    (its own journal is recognised by the mark's digest, never by size) and, for
    the switch seam (``resume=True``), a store already renamed completes from the
    retired copy's mark. A journal that changed after the import, or one without
    a store, is never overwritten."""
    from ouroboros import usage_journal as journal
    from ouroboros.utils import atomic_write_json

    root = pathlib.Path(root)
    path, kept = root / STORE_REL, root / LEDGER_REL
    refusal = export_refusal(root, resume=resume)
    if refusal:
        raise UsageAccountingError(refusal)
    if _identity(path) is None:  # the rename already landed: complete from the retired copy's mark
        retired, mark = _retired_for(root, _sha256_file(kept))
        return {"status": "exported", **_export_result(mark, retired.name)}
    with hold(root, migrate=False, exporting=True) as txn:  # BEGIN IMMEDIATE: the last write this store takes
        provenance = txn.meta("import") or {}
        rows, attempts, aggregates = _export_rows(txn, provenance)
        payload = journal.journal_bytes(rows)
        mark = txn.meta("export")
        if mark is None:
            state = _state_document(root)
            mark = {"status": "writing", "started_at": utc_now_iso(),
                    "journal": {"size": len(payload), "sha256": hashlib.sha256(payload).hexdigest(), "rows": len(rows)},
                    "marker": journal.journal_marker(rows), "attempts": attempts, "aggregates": aggregates,
                    "pre_export_as": f"{kept.name}.pre-export-{_core.utc_stamp()}" if kept.exists() else None,
                    "store_retired_as": f"{path.name}.exported-{_core.utc_stamp()}", "legacy": provenance.get("legacy"),
                    "state_marker_before": (state or {}).get("usage_ledger_high_water_seq") if isinstance(state, dict) else None}
            txn.set_meta("export", mark)
        elif hashlib.sha256(payload).hexdigest() != mark["journal"]["sha256"]:
            raise UsageLedgerCorrupt(f"the frozen usage store no longer reproduces its export mark: {path}")
    kept_sha = _sha256_file(kept)
    if kept_sha is not None and kept_sha != mark["journal"]["sha256"]:
        if not _own_partial(root, mark):  # the import kept it, unchanged: set aside, never lost
            os.replace(kept, kept.with_name(mark["pre_export_as"]))
        kept_sha = None  # else this export's own interrupted write: rewritten whole below
    if kept_sha is None:
        written = journal.write_journal(root, rows)
        if written["sha256"] != mark["journal"]["sha256"]:
            raise UsageLedgerCorrupt(f"the exported journal does not match its mark: {kept}")
    if mark.get("legacy") and journal.completed_legacy_watermark(root) is None:
        atomic_write_json(root / journal.IMPORT_REL, mark["legacy"], trailing_newline=True, fsync=True)
    state = _state_document(root)
    if isinstance(state, dict) and "usage_ledger_high_water_seq" in state:
        atomic_write_json(root / "state" / "state.json", {**state, "usage_ledger_high_water_seq": mark["marker"]}, fsync=True)
    retired = path.with_name(mark["store_retired_as"])
    forget(root)  # Windows requires every process-owned cached handle closed before rename
    os.replace(path, retired)
    _fsync_directory(path.parent)
    forget(root)
    return {"status": "exported", **_export_result(mark, retired.name)}


def _export_result(mark: Dict[str, Any], retired_name: str) -> Dict[str, Any]:
    return {"journal": {"path": None, **mark["journal"]}, "marker": mark["marker"], "store_retired_as": retired_name,
            "pre_export_as": mark.get("pre_export_as"), "attempts": mark["attempts"], "aggregates": mark["aggregates"]}


def _state_document(root: pathlib.Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads((root / "state" / "state.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return None


def abandon_export(root: pathlib.Path | str, *, retired_as: Optional[str] = None) -> Dict[str, Any]:
    """Undo an interrupted export, idempotent and resumable at every step: the store back
    under its name from the retired copy (``retired_as`` from the switch receipt, else the
    newest copy whose mark planned the journal on disk), the journal the import kept back by
    ONE atomic replace over this export's own (no window without a journal), the
    ``state.json`` marker the export lowered put back, the mark cleared LAST so a death
    in between resumes here. Returns what moved (``{}`` when nothing of this export's is found)."""
    from ouroboros.utils import atomic_write_json

    root = pathlib.Path(root)
    path, kept = root / STORE_REL, root / LEDGER_REL
    moved: Dict[str, Any] = {}
    if _identity(path) is None:
        kept_sha = _sha256_file(kept)
        found = None
        if retired_as and path.with_name(retired_as).exists():
            found = (path.with_name(retired_as), None)
        elif kept_sha is not None:
            found = _retired_for(root, kept_sha)
        if found is None:
            return {}
        os.replace(found[0], path)
        _fsync_directory(path.parent)
        forget(root)
        moved["store"] = "restored"
    tier = _header_tier(path)
    mark = _stored_meta(path, tier, USAGE_LOCK_TIMEOUT_SEC).get("export") if tier else None
    if mark is None:
        return moved
    kept_sha = _sha256_file(kept)
    pre_export = kept.with_name(mark["pre_export_as"]) if mark.get("pre_export_as") else None
    if kept_sha is None or kept_sha == mark["journal"]["sha256"] or _own_partial(root, mark):  # absent or this export's own
        if pre_export is not None and pre_export.exists():
            os.replace(pre_export, kept)  # the kept one comes back over the own, one rename, no window without a journal
            moved["journal"] = "restored"
        elif kept_sha is not None and pre_export is None:
            kept.unlink()  # the import kept none: nothing to put back
            moved["journal"] = "removed"
        _fsync_directory(kept.parent)
    state = _state_document(root)
    if isinstance(state, dict) and state.get("usage_ledger_high_water_seq") == mark["marker"] \
            and mark.get("state_marker_before") is not None:
        atomic_write_json(root / "state" / "state.json", {**state, "usage_ledger_high_water_seq": mark["state_marker_before"]},
                          fsync=True)
        moved["state_marker"] = "restored"
    with hold(root, migrate=False, exporting=True) as txn:
        txn.set_meta("export", None)  # the mark LAST: the store takes writes again
    moved["mark"] = "cleared"
    return moved


def forget(root: pathlib.Path | str) -> None:
    """Close this process's cached connections and readiness memo (tests, export)."""
    _core.close_connections(pathlib.Path(root) / STORE_REL)
    _READY.pop(str(pathlib.Path(root) / STORE_REL), None)
