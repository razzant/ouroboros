"""The translation generator: the LIGHT model fills the install's translation memory.

Nothing here decides what the interface says. The English source is authored in the
code; this module asks the light model slot for the same strings in the install's
language (``OUROBOROS_UI_LANGUAGE``) and writes the answers into the memory
(``ouroboros/i18n_memory.py``) as ``generated`` entries, below every owner pin and
every imported pack.

What gets translated, and when:

* **Catalog codes** — the host's closed code→sentence tables (task headline words, cause
  sentences, question status, routing refusal causes; Telegram's own lines register
  through ``register_catalog``). They are enumerable, so a language choice, a regenerate
  or a boot re-check queues every code the memory lacks and every generated code whose
  English changed (``source_hash``), and an owner pin is never touched.
* **Misses** — strings the browser overlay or a host seam could not translate
  (``POST /api/ui/i18n/missing``): rendered chrome keyed by its English, ``code:`` keys
  with their English in the context, ``fmt`` templates with their placeholders. They
  sit in the bounded pending queue and are translated in batches as they arrive, so a
  self-modification that adds a label is translated the first time the label renders.

How: one worker thread per process (single-flight, woken by the gateway hooks), one
accounted light-model call per batch of up to ``BATCH_KEYS`` keys with a stable prefix
(rules, the language profile and its lexicon, the plural categories) so the provider's
prompt cache carries the fixed part. Every answer is validated before it is written: a
value may use only the placeholders its source has, never markup, and a plural key
answers one form per CLDR category the browser recorded. An invalid or missing answer
goes back to the queue with its attempt count; a key whose answers failed ``MAX_ATTEMPTS``
batches is refused durably (the memory's ``refused`` ledger, cleared by Regenerate) and
logged — a transport bound, not a judgement. A transport, provider, budget or output-limit
failure requeues the batch WITHOUT counting an attempt: an outage must not turn into
permanent English. A batch cut by the output budget is a failed batch, never a partially
trusted one. Each applied batch broadcasts
``ui_i18n_updated`` so every client repaints from the new revision.

A free-text language ("Quenya", "invent a language and translate everything into it")
becomes a language profile through ``resolve_language_request``: one light call answers
the tag (a real BCP-47 tag, or ``art-x-<slug>`` for an invented language), the label,
the direction, an instruction for the generator and — for a rare or invented language —
a lexicon every later batch reuses. Without a credentialed model the gateway says so
(``language_needs_model``) instead of guessing.

Cost: every call runs under ``UsageScope(category="ui_translation", non_task_operation=True)``
with the install's global budget, like the update letter; a budget refusal stops the
run and keeps the queue (attempts untouched).
"""

from __future__ import annotations

import json
import logging
import pathlib
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from ouroboros import i18n_memory as memory
from ouroboros.ui_language import invented_language_tag, is_english, normalize_language_tag
from ouroboros.utils import utc_now_iso

log = logging.getLogger(__name__)

SYSTEM_TASK_ID = "system:ui_translation"
USAGE_CATEGORY = "ui_translation"
# Output budget of ONE batch call (a hundred short strings with plural forms fit in a
# fraction of this; the stop marker, not the number, decides a cut).
UI_TRANSLATION_MAX_TOKENS = 16384
BATCH_KEYS = 100
MAX_ATTEMPTS = 3
_OUTPUT_LIMIT_STOPS = frozenset({"length", "max_tokens"})
_SLUG_RE = re.compile(r"[^a-z0-9]+")

CatalogTable = Callable[[], Dict[str, str]]
_EXTRA_CATALOGS: Dict[str, Tuple[CatalogTable, str]] = {}
# Keys taken from the queue for the batch at the model right now, per language: the
# gateway adds them to the pending count so the status never dips mid-batch.
_IN_FLIGHT: Dict[str, int] = {}


class LanguageResolveError(RuntimeError):
    """A free-text language could not become a profile; ``code`` and ``status`` are the gateway's."""

    def __init__(self, code: str, message: str, status: int = 502) -> None:
        super().__init__(message)
        self.code = code
        self.status = status


# ---------------------------------------------------------------------------
# the catalog: code → English, from the tables that own the English
# ---------------------------------------------------------------------------


def register_catalog(prefix: str, table: CatalogTable, note: str) -> None:
    """A host table of ``key → English sentence`` translated by code ``code:<prefix>.<key>``
    (Telegram's lines register here from the skill; the core tables are built in). A table
    registered after boot — a skill enabled later, a self-modification — is queued for the
    chosen language at once, so its rows never wait for the next language event."""
    _EXTRA_CATALOGS[str(prefix)] = (table, str(note))
    root = _WORKER.drive_root
    tag = memory.current_language()
    if root is None or not tag or is_english(tag):
        return
    try:
        if enqueue_catalog(root, tag, prefixes=(str(prefix),)):
            _WORKER.schedule(root)
    except Exception:
        log.debug("ui translation: late catalog %s not queued", prefix, exc_info=True)


def _core_catalogs() -> Dict[str, Tuple[Dict[str, str], str]]:
    from ouroboros import project_dialogue as pd

    return {
        "task.headline": ({**pd.OUTCOME_PHASE_HEADLINE, "quiet": pd.QUIET_RUN_HEADLINE},
                          "the one-word status headline of a task card (Working, Done, Failed…)"),
        "task.cause": (pd.TASK_CAUSE_PHRASES,
                       "one sentence under a finished task explaining why its answer was or was not signed off"),
        "question.status": (pd.QUESTION_STATUS,
                            "the status line of a question Ouroboros asked its human"),
        "routing.refusal": (pd.ROUTING_REFUSAL_CAUSES,
                            "the clause after 'Not started:' / 'Not moved:' when a message could not be routed"),
    }


def catalog_entries() -> Dict[str, Dict[str, Any]]:
    """``{"code:<table>.<key>": {"text": English, "context": {...}}}`` for every host table."""
    out: Dict[str, Dict[str, Any]] = {}
    tables: Dict[str, Tuple[Any, str]] = dict(_core_catalogs())
    for prefix, (table, note) in _EXTRA_CATALOGS.items():
        tables[prefix] = (table, note)
    for prefix, (table, note) in tables.items():
        rows = table() if callable(table) else table
        for key, text in dict(rows or {}).items():
            if not isinstance(text, str) or not text.strip():
                continue
            out[f"{memory.CODE_PREFIX}{prefix}.{key}"] = {"text": text, "context": {"table": prefix, "note": note}}
    return out


def catalog_source_hashes() -> Dict[str, str]:
    """The current English of every catalog code, hashed, for the memory's stale count."""
    return {key: memory.source_hash(item["text"]) for key, item in catalog_entries().items()}


def enqueue_catalog(drive_root: pathlib.Path, tag: str, *, prefixes: Optional[Tuple[str, ...]] = None) -> int:
    """Queue every catalog code the memory lacks, and every generated one whose English
    moved. Owner pins and imported entries are never queued. ``prefixes`` narrows the pass to
    the tables named (a late registration). Returns the count queued."""
    doc = memory.cached_memory(drive_root, tag)
    entries = (doc or {}).get("entries", {})
    items: List[Dict[str, Any]] = []
    for key, item in catalog_entries().items():
        if prefixes is not None and item["context"].get("table") not in prefixes:
            continue
        current = entries.get(key)
        if current is None:
            items.append({"key": key, "context": {"source": item["text"], **item["context"]}})
            continue
        if current.get("provenance") != "generated":
            continue
        if current.get("source_hash") and current["source_hash"] != memory.source_hash(item["text"]):
            items.append({"key": key, "context": {"source": item["text"], **item["context"], "stale": True}})
    if not items:
        return 0
    return memory.record_missing(drive_root, tag, items)["accepted"]


# ---------------------------------------------------------------------------
# the light model: route, prompt, call
# ---------------------------------------------------------------------------


def _light_route() -> Tuple[str, bool, bool]:
    """``(model, use_local, available)`` for the light slot (empty light → the first
    credentialed slot, as project naming resolves it; a local-only install uses local)."""
    from ouroboros.config import get_light_model, runtime_setting
    from ouroboros.provider_models import model_has_credentials, resolve_credentialed_model, review_model_uses_local

    model = resolve_credentialed_model(get_light_model())
    use_local = str(runtime_setting("USE_LOCAL_LIGHT", "") or "").strip().lower() in ("true", "1", "yes", "on") \
        or review_model_uses_local(model)
    return model, use_local, bool(model) and (use_local or model_has_credentials(model))


def _timeout_sec() -> float:
    from ouroboros.config import get_ui_translation_timeout_sec

    return get_ui_translation_timeout_sec()


def _usage_scope(drive_root: pathlib.Path):
    from ouroboros.settings_setup_contract import resolve_total_budget_usd
    from ouroboros.usage_accounting import UsageScope

    return UsageScope(drive_root=drive_root, task_id=SYSTEM_TASK_ID, root_task_id=SYSTEM_TASK_ID,
                      category=USAGE_CATEGORY, source=USAGE_CATEGORY, non_task_operation=True,
                      global_limit_usd=resolve_total_budget_usd())


def _call_light(drive_root: pathlib.Path, messages: List[Dict[str, Any]], *, client: Any = None,
                cache_affinity: str = "") -> Tuple[str, str]:
    """One accounted call; one deadline covers preparation, slot waiting and recovery.

    Returns ``(content, stop)``; raises on transport failure so the batch can requeue.
    """
    from ouroboros import model_concurrency
    from ouroboros.llm import LLMClient
    from ouroboros.llm_observability import chat_observed
    from ouroboros.usage_accounting import usage_scope

    model, use_local, available = _light_route()
    if not available:
        raise LanguageResolveError("language_needs_model", "no credentialed model is configured for the light slot", 400)
    deadline_ts = time.time() + _timeout_sec()
    with model_concurrency.model_call_scope(model, use_local, deadline_ts=deadline_ts):
        remaining = deadline_ts - time.time()
        if remaining <= 0:
            raise TimeoutError("the translation ceiling was spent waiting for a model slot")
        with usage_scope(_usage_scope(drive_root)):
            msg, usage = chat_observed(
                client or LLMClient(), drive_root=drive_root, task_id=SYSTEM_TASK_ID, call_type=USAGE_CATEGORY,
                messages=messages, model=model, tools=None, reasoning_effort="low", model_role="light",
                max_tokens=UI_TRANSLATION_MAX_TOKENS, use_local=use_local, timeout=remaining,
                caller_deadline_ts=deadline_ts,
                response_format={"type": "json_object"}, cache_affinity=cache_affinity)
    body_error = (usage or {}).get("provider_error")
    if isinstance(body_error, dict) and body_error:
        raise RuntimeError(f"provider body error (code={body_error.get('code')}): {body_error.get('message')}")
    stop = str((msg or {}).get("finish_reason") or (msg or {}).get("stop_reason")
               or (usage or {}).get("response_finish_reason") or "").strip().lower()
    return str((msg or {}).get("content") or ""), stop


def _json_object(content: str) -> Optional[Dict[str, Any]]:
    """The JSON object in a model answer: the whole text, or the outermost braces (the
    ``response_format`` hint is optional on some routes, so a text parse stays)."""
    text = str(content or "").strip()
    for candidate in (text, text[text.find("{"): text.rfind("}") + 1] if "{" in text and "}" in text else ""):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _profile_prefix(doc: Optional[Dict[str, Any]], tag: str) -> str:
    profile = (doc or {}).get("profile") or {}
    label = str(profile.get("label") or tag)
    lines = [
        "You translate the interface of Ouroboros, a self-modifying AI agent, from English into "
        f"the language tagged `{tag}` ({label}).",
        "Rules: translate meaning, not words; keep the register of a calm, precise product; keep "
        "product names (Ouroboros, Claudexor, Telegram, GitHub) and code identifiers as they are; "
        "keep every placeholder exactly as written — `{n}`, `{name}` and numbered inline slots such "
        "as `<1>…</1>` — and add none; never add markup; keep the length close to the English so "
        "the layout holds; a short label stays a short label; sentence-case where the English is.",
        "The strings come with their place in the interface (`context`): a button, a heading, a "
        "placeholder, a status word, a sentence under a task. Translate for that place.",
    ]
    categories = (doc or {}).get("plural_categories") or []
    if categories:
        lines.append("Plural forms: a string whose key is a template with `{n}` is answered as `forms`, one "
                     f"form per CLDR plural category of this language — exactly these: {', '.join(categories)}. "
                     "Every other string is answered as `text`.")
    else:
        lines.append("A string whose key is a template with `{n}` is answered as `forms` with the CLDR plural "
                     "categories this language uses (`other` always present); every other string as `text`.")
    direction = str(profile.get("direction") or "ltr")
    if direction == "rtl":
        lines.append("The language is written right to left; the layout mirrors, the text does not need markers.")
    instruction = str(profile.get("instruction") or "").strip()
    if instruction:
        lines.append("The owner's instruction for this language: " + instruction)
    lexicon = str(profile.get("lexicon") or "").strip()
    if lexicon:
        lines.append("Lexicon and rules of this language, which every translation must follow and reuse:\n" + lexicon)
    lines.append('Answer ONE JSON object: {"translations": [{"id": <id>, "text": "..."} or '
                 '{"id": <id>, "forms": {"<category>": "..."}}]}. Answer every id once; no commentary.')
    return "\n\n".join(lines)


def _batch_request(batch: List[Dict[str, Any]]) -> str:
    rows = [{"id": item["id"], "text": item["text"], "plural": item["plural"],
             "placeholders": sorted(memory.placeholders(item["text"])), "context": item["context"]}
            for item in batch]
    return "Translate these interface strings:\n" + json.dumps(rows, ensure_ascii=False, indent=1)


# ---------------------------------------------------------------------------
# validation and application of one batch
# ---------------------------------------------------------------------------


def _source_text(key: str, facts: Dict[str, Any], catalog: Dict[str, Dict[str, Any]]) -> str:
    """The English a queued key stands for: the host catalog's own text for a catalog code
    (never a client's possibly stale copy), the context's ``source`` for a browser-only code
    or a template, or — for rendered chrome — the key itself."""
    if key in catalog:
        return catalog[key]["text"]
    context = facts.get("context") if isinstance(facts.get("context"), dict) else {}
    source = memory.key_text(context.get("source"))  # as the browser keys it: whitespace collapsed
    if source:
        return source
    return "" if memory.is_code_key(key) else memory.split_scope(key)[0]


def _validate_answer(item: Dict[str, Any], answer: Dict[str, Any], categories: List[str]) -> Optional[Dict[str, Any]]:
    """``{"text"}`` or ``{"forms"}`` for a well-formed answer, else ``None``: the SAME placeholders
    as the source (specs included), no markup the source lacks, balanced inline slots, and for a
    plural key one form per recorded category."""
    allowed = memory.placeholders(item["text"])
    source_markup = memory.markup_tokens(item["text"])

    def _ok(value: Any) -> bool:
        if not isinstance(value, str) or not value.strip() or len(value) > memory.MAX_VALUE_CHARS:
            return False
        if memory.markup_tokens(value) - source_markup:  # no markup the source did not have
            return False
        return memory.placeholders(value) == allowed and memory.inline_slots_balanced(value)

    forms = answer.get("forms")
    if item["plural"]:
        if not isinstance(forms, dict) or not forms:
            return None
        cleaned = {str(k): v for k, v in forms.items() if _ok(v) and isinstance(k, str) and k and len(k) <= 16}
        if len(cleaned) != len(forms):
            return None
        if categories and set(cleaned) != set(categories):
            return None
        if not categories and "other" not in cleaned:
            return None
        return {"forms": cleaned}
    text = answer.get("text")
    if text is None and isinstance(forms, dict) and len(forms) == 1:
        text = next(iter(forms.values()))
    return {"text": text} if _ok(text) else None


def _broadcast(message: Dict[str, Any]) -> None:
    try:
        from ouroboros.gateway.ws import broadcast_ws_sync

        broadcast_ws_sync(message)
    except Exception:
        log.debug("ui translation broadcast skipped", exc_info=True)


def _refuse(drive_root: pathlib.Path, tag: str, keys: List[str], reason: str) -> None:
    """Remember keys that exhausted their attempts, so pages stop re-reporting them."""
    try:
        memory.update_memory(drive_root, tag, lambda doc: doc if memory.refuse_keys(doc, keys, reason=reason) else None, create=True)
    except Exception:
        log.debug("ui translation: refused keys not recorded", exc_info=True)


def translate_batch(drive_root: pathlib.Path, tag: str, *, client: Any = None,
                    limit: int = BATCH_KEYS) -> Dict[str, Any]:
    """Take one batch from the queue, translate it, write it. Returns the batch facts
    (``taken``, ``applied``, ``requeued``, ``dropped``, ``error``). Never raises."""
    facts: Dict[str, Any] = {"taken": 0, "applied": 0, "requeued": 0, "dropped": 0, "error": ""}
    taken = memory.take_pending(drive_root, tag, limit)
    facts["taken"] = len(taken)
    if not taken:
        return facts
    _IN_FLIGHT[tag] = len(taken)
    try:
        return _translate_taken(drive_root, tag, taken, facts, client=client)
    finally:
        _IN_FLIGHT.pop(tag, None)


def _translate_taken(drive_root: pathlib.Path, tag: str, taken: List[Tuple[str, Dict[str, Any]]],
                     facts: Dict[str, Any], *, client: Any = None) -> Dict[str, Any]:
    doc = memory.cached_memory(drive_root, tag)
    catalog = catalog_entries()
    entries = (doc or {}).get("entries", {})
    batch: List[Dict[str, Any]] = []
    back: List[Tuple[str, Dict[str, Any]]] = []
    refused: List[str] = []
    for index, (key, queued) in enumerate(taken):
        text = _source_text(key, queued, catalog)
        if not text:
            facts["dropped"] += 1  # a code nobody can read the English of
            continue
        current = entries.get(key)
        context_flags = queued.get("context") if isinstance(queued.get("context"), dict) else {}
        if current is not None and not context_flags.get("stale"):
            facts["satisfied"] = facts.get("satisfied", 0) + 1  # already in the memory: nothing to buy
            continue
        context = dict(queued.get("context") or {})
        context.pop("source", None)
        batch.append({"id": index, "key": key, "text": text, "plural": "{n}" in text,
                      "context": context, "queued": queued})
    if not batch:
        return facts

    def _requeue(items: List[Dict[str, Any]], *, error: str, count: bool = True) -> None:
        """Back to the queue. ``count`` is for an answer the model gave that no rule accepted —
        the only failure a key can exhaust. A transport, provider, budget or output-limit
        failure is not the key's doing and costs it no attempt."""
        for item in items:
            queued = dict(item["queued"])
            if count:
                queued["attempts"] = int(queued.get("attempts") or 0) + 1
            queued["last_error"] = error[:200]
            if count and queued["attempts"] >= MAX_ATTEMPTS:
                facts["dropped"] += 1
                refused.append(item["key"])
                log.warning("ui translation: dropping %r after %d failed batches (%s)", item["key"], queued["attempts"], error[:120])
            else:
                back.append((item["key"], queued))
                facts["requeued"] += 1

    messages = [{"role": "system", "content": _profile_prefix(doc, tag)},
                {"role": "user", "content": _batch_request(batch)}]
    try:
        content, stop = _call_light(drive_root, messages, client=client, cache_affinity=f"{USAGE_CATEGORY}:{tag}")
    except LanguageResolveError as exc:
        memory.requeue_pending(drive_root, tag, [(item["key"], item["queued"]) for item in batch])
        facts["requeued"] = len(batch)
        facts["error"] = exc.code
        return facts
    except Exception as exc:  # noqa: BLE001 — a failed batch is requeued with its cause, attempts untouched
        error = f"{type(exc).__name__}: {exc}"
        _requeue(batch, error=error, count=False)
        memory.requeue_pending(drive_root, tag, back)
        facts["error"] = error[:200]
        return facts
    if stop in _OUTPUT_LIMIT_STOPS:
        _requeue(batch, error=f"output budget hit ({stop})", count=False)
        memory.requeue_pending(drive_root, tag, back)
        facts["error"] = "output_truncated"
        return facts
    parsed = _json_object(content) or {}
    answers = parsed.get("translations")
    by_id: Dict[int, Dict[str, Any]] = {}
    if isinstance(answers, list):
        for answer in answers:
            if isinstance(answer, dict) and isinstance(answer.get("id"), int):
                by_id[answer["id"]] = answer
    elif isinstance(answers, dict):  # a model that keyed by id string
        for key, answer in answers.items():
            if str(key).isdigit() and isinstance(answer, dict):
                by_id[int(key)] = answer
    categories = list((doc or {}).get("plural_categories") or [])
    accepted: Dict[str, Dict[str, Any]] = {}
    rejected: List[Dict[str, Any]] = []
    for item in batch:
        value = _validate_answer(item, by_id.get(item["id"]) or {}, categories)
        if value is not None:
            try:  # the writer's own rule, per key: an answer it would refuse never poisons the batch
                memory.validate_candidate(item["key"], value, source=item["text"])
            except memory.MemoryFormatError as exc:
                log.info("ui translation: answer for %r refused by the memory rule: %s", item["key"], exc)
                value = None
        if value is None:
            rejected.append(item)
        else:
            accepted[item["key"]] = value
    if rejected:
        _requeue(rejected, error="no valid answer in the batch")
        memory.requeue_pending(drive_root, tag, back)
    if refused and not accepted:
        _refuse(drive_root, tag, refused, "no valid answer after the attempt bound")
    if accepted:
        hashes = {key: memory.source_hash(item["text"]) for item in batch for key in [item["key"]] if key in accepted}
        sources = {item["key"]: item["text"] for item in batch if item["key"] in accepted and memory.is_code_key(item["key"])}
        model, _use_local, _available = _light_route()
        applied = {"count": 0}

        def _apply(current: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            applied["count"] = memory.apply_generated(current, accepted, model=model, source_hashes=hashes, sources=sources)
            if refused:
                memory.refuse_keys(current, refused, reason="no valid answer after the attempt bound")
            return current if applied["count"] or refused else None

        try:
            written = memory.update_memory(drive_root, tag, _apply, create=True)
        except (memory.MemoryFormatError, TimeoutError) as exc:
            facts["error"] = f"{type(exc).__name__}: {exc}"[:200]
            memory.requeue_pending(drive_root, tag, [(k, item["queued"]) for item in batch for k in [item["key"]] if k in accepted])
            return facts
        facts["applied"] = applied["count"]
        if applied["count"]:
            _broadcast({"type": "ui_i18n_updated", "language": tag, "revision": written["revision"],
                        "applied": applied["count"]})
    return facts


# ---------------------------------------------------------------------------
# the worker: single-flight, woken by the gateway hooks
# ---------------------------------------------------------------------------


class _Worker:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.stop = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.drive_root: Optional[pathlib.Path] = None
        self.client_factory: Optional[Callable[[], Any]] = None
        self.status: Dict[str, Any] = {"state": "idle", "error": "", "updated_at": "", "applied": 0, "language": ""}
        self.batch_keys: Dict[str, int] = {}  # per language: the batch size the route's output bound allowed
        self.halt: Optional[threading.Event] = None  # the host's teardown event: no batch starts once it is set

    def schedule(self, drive_root: pathlib.Path) -> None:
        with self.lock:
            self.drive_root = pathlib.Path(drive_root)
            self.wake.set()
            if self.thread is None or not self.thread.is_alive():
                self.stop.clear()
                self.thread = threading.Thread(target=self._run, name="ui-translation", daemon=True)
                self.thread.start()

    def _set(self, state: str, error: str = "") -> None:
        self.status.update(state=state, error=error[:200], updated_at=utc_now_iso())

    def _stopping(self) -> bool:
        return self.stop.is_set() or (self.halt is not None and self.halt.is_set())

    def _run(self) -> None:
        while not self._stopping():
            if not self.wake.wait(timeout=0.5):
                with self.lock:
                    if not self.wake.is_set():
                        self.thread = None  # nothing more to do: the next hook starts a new thread
                        break
                continue
            self.wake.clear()
            root = self.drive_root
            tag = memory.current_language()
            if root is None or not tag or is_english(tag):
                self._set("idle")
                continue
            self._drain(root, tag)

    def _drain(self, root: pathlib.Path, tag: str) -> None:
        client = self.client_factory() if self.client_factory else None
        self.status["language"] = tag
        self._set("running")
        while not self._stopping() and memory.current_language() == tag:
            pending = memory.pending_count(root, tag)
            if not pending:
                self._set("idle")
                return
            facts = translate_batch(root, tag, client=client, limit=self.batch_keys.get(tag, BATCH_KEYS))
            self.status["applied"] = int(self.status.get("applied") or 0) + int(facts["applied"])
            if facts["error"] == "output_truncated" and int(facts["taken"]) > 1:
                # The answer did not fit this route's output bound: the queue is asked again in half
                # the batch, and the smaller size is kept for this language in this process.
                self.batch_keys[tag] = int(facts["taken"]) // 2
                continue
            if facts["error"] in ("language_needs_model",):
                self._set("no_model", facts["error"])
                return
            if facts["error"]:
                # A failed batch: the queue keeps it (with attempts); the next hook retries.
                self._set("failed", facts["error"])
                return
            if facts["taken"] == 0:
                self._set("idle")
                return
        self._set("idle")


_WORKER = _Worker()


def generator_status() -> Dict[str, Any]:
    """``{state, error, updated_at, applied, language, in_flight}`` for the gateway: the worker's
    state in this process and the size of the batch out at the model for its language."""
    status = dict(_WORKER.status)
    status["in_flight"] = int(_IN_FLIGHT.get(str(status.get("language") or ""), 0))
    return status


def set_client_factory(factory: Optional[Callable[[], Any]]) -> None:
    """Test seam: the LLM client the worker uses (default: ``LLMClient()``)."""
    _WORKER.client_factory = factory


def on_language_event(event: str, drive_root: pathlib.Path, tag: str) -> None:
    """The gateway hook (``register_language_hook``): a chosen, regenerated or imported
    language queues its catalog; every event wakes the worker."""
    if not tag or is_english(tag):
        return
    if event in ("language_set", "regenerate", "imported"):
        try:
            enqueue_catalog(pathlib.Path(drive_root), tag)
        except Exception:
            log.warning("ui translation: catalog enqueue failed for %s", tag, exc_info=True)
    _WORKER.schedule(pathlib.Path(drive_root))


def start_background(drive_root: pathlib.Path, halt: Optional[threading.Event] = None) -> None:
    """Server boot (fail-soft, no model call): register the gateway hook, remember the data
    root so a table registered later can queue itself, and re-check the chosen language —
    catalog codes the memory lacks or whose English moved, misses left in the queue. ``halt``
    is the host's own teardown event (the server lifespan's): once set, the worker starts no
    further batch, so a shutdown or restart never buys one it cannot keep."""
    try:
        from ouroboros.gateway.ui_i18n import register_language_hook

        register_language_hook(on_language_event)
        _WORKER.drive_root = pathlib.Path(drive_root)
        _WORKER.halt = halt
        tag = memory.current_language()
        if not tag or is_english(tag):
            return
        queued = enqueue_catalog(pathlib.Path(drive_root), tag)
        if queued or memory.pending_count(pathlib.Path(drive_root), tag):
            _WORKER.schedule(pathlib.Path(drive_root))
    except Exception:
        log.warning("ui translation: boot re-check failed", exc_info=True)


def stop_background(timeout: float = 2.0) -> None:
    """Stop the worker thread and wait for it (fail-soft): tests and an embedding host. The
    server lifespan does not wait — it hands ``start_background`` its teardown event."""
    try:
        _WORKER.stop.set()
        _WORKER.wake.set()
        thread = _WORKER.thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=timeout)
    except Exception:
        log.debug("ui translation: stop failed", exc_info=True)


# ---------------------------------------------------------------------------
# a free-text language → a profile
# ---------------------------------------------------------------------------

_RESOLVE_PROMPT = (
    "The owner of an Ouroboros install typed what interface language they want:\n\n{text}\n\n"
    "Answer ONE JSON object with these fields.\n"
    "- tag: the BCP-47 language tag. A real language gets its standard tag (ru, pt-BR, zh-Hant, "
    "ja); a constructed or fictional language with an ISO 639-3 code gets that code (qya for Quenya, "
    "tlh for Klingon, epo/eo for Esperanto); a language that must be invented gets `art-x-<slug>`, the "
    "slug lowercase letters and digits, at most 8 characters.\n"
    "- label: the language's own name (its endonym), or the name the owner gave it.\n"
    "- direction: `ltr` or `rtl`.\n"
    "- instruction: one paragraph for the translator — register, forms of address, anything the "
    "owner asked for (a dialect, a style, an era).\n"
    "- lexicon: empty for a well-known language. For a rare, fictional or invented language: the "
    "writing system, the phonology, the core grammar (word order, plurals, negation, questions) and "
    "a vocabulary of 150–300 interface words (settings, task, project, chat, model, review, start, stop, "
    "save, cancel, delete, error, done, waiting…) with their translations, so that every later "
    "translation batch is consistent with this one document. Plain text, under 20000 characters.\n"
    "No commentary outside the JSON."
)


def resolve_language_request(text: str, *, drive_root: Optional[pathlib.Path] = None,
                             client: Any = None) -> Dict[str, Any]:
    """``{"tag", "label", "direction", "instruction", "lexicon"}`` for a free-text language,
    or ``LanguageResolveError`` (``language_needs_model`` 400 without a credentialed light
    model; ``language_resolve_failed`` 502 when the model's answer is unusable)."""
    from ouroboros.config import DATA_DIR

    root = pathlib.Path(drive_root or DATA_DIR)
    request = str(text or "").strip()
    if not request:
        raise LanguageResolveError("language_not_a_tag", "the language is empty", 400)
    messages = [{"role": "user", "content": _RESOLVE_PROMPT.format(text=request[:2000])}]
    try:
        content, stop = _call_light(root, messages, client=client, cache_affinity=f"{USAGE_CATEGORY}:resolve")
    except LanguageResolveError:
        raise
    except Exception as exc:  # noqa: BLE001 — the gateway answers a typed 502
        raise LanguageResolveError("language_resolve_failed",
                                   f"the model could not be asked about this language ({type(exc).__name__})") from exc
    parsed = _json_object(content)
    if stop in _OUTPUT_LIMIT_STOPS and not parsed:
        raise LanguageResolveError("language_resolve_failed", "the model's answer was cut by the output budget")
    if not parsed or not isinstance(parsed.get("label"), str) or not parsed["label"].strip():
        # No profile in the answer: the host does not invent a language the model did not name.
        raise LanguageResolveError("language_resolve_failed", "the model's answer was not a language profile")
    label = parsed["label"][:120].strip()
    tag = normalize_language_tag(parsed.get("tag"))
    if not tag:
        # The model named a language without a usable tag: an invented one, carried as private use.
        slug = _SLUG_RE.sub("-", label.lower()).strip("-")[:8].strip("-") or "lang"
        tag = invented_language_tag(slug)
    if is_english(tag):
        tag = "en"
    direction = "rtl" if str(parsed.get("direction") or "").strip().lower() == "rtl" else "ltr"
    lexicon = str(parsed.get("lexicon") or "")
    if len(lexicon) > memory.MAX_LEXICON_CHARS:
        lexicon = lexicon[: memory.MAX_LEXICON_CHARS]
    return {"tag": tag, "label": label or tag, "direction": direction,
            "instruction": str(parsed.get("instruction") or "")[:4000].strip(), "lexicon": lexicon}


__all__ = [
    "UI_TRANSLATION_MAX_TOKENS", "BATCH_KEYS", "MAX_ATTEMPTS", "LanguageResolveError",
    "register_catalog", "catalog_entries", "catalog_source_hashes", "enqueue_catalog",
    "translate_batch", "generator_status", "set_client_factory", "on_language_event",
    "start_background", "stop_background", "resolve_language_request",
]
