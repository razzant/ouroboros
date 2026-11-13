"""A work's cache life, replayed end to end (the harness of ``tests/test_cache_optimization.py``'s life table).

One work runs through the production loop (``loop.run_llm_loop``): the start fit (``context._build_context_fit_plan``),
the route rebind, the cold restore (``owner_wait.resume_native_loop`` -> ``rebind_restored_route``), the pre-send check
(``llm_attempt.main_context_fit_refusal``), the refusal passes and the install's own queue (``model_concurrency``). Its
requests leave through the real Claudexor serializer to the fake gateway of ``tests/test_llm_claudexor.py``, or through
the real OpenAI lane to an in-memory SDK client. Behind both sits one stub provider (``Provider``): it keeps one cache
per place (an engine account, or one API key of one model string) and reads the longest run of byte-equal elements (the
tool array, each system block, each later row) of that place's previous request; it counts tokens at the place's own
density and enforces the place's bound the way the place does (``input``: an input above it is refused with its number;
``total``: input plus the reply allowance above it is refused, with its number, or, ``total_unnumbered``, without, or,
``total_worded``, in the words of a server that names ``max_tokens`` beside the window and the input, or,
``total_as_a_size``, as a size it will not take, with no number; ``cut``: nothing is refused and the reply is cut to
what is left); and it answers a scripted mind. It also notes how each part the mind
read came back: whole, or as a partial view (``tool_result_delivery.RESULT_VIEW_MARKER``).

A life is read back from the product's own record of each request (``physical_context.prefix`` ->
``transcript_prefix.prefix_class``), joined with the place that answered it (the observer keys a route by its model
string, so it cannot see an account), the free room the request left (stated window - reply reserve - input), the
pre-send refusals (``context_fit_refused``), the queue's refusals, the words the work's feed got and the facts line each
request carried. The law it is held to, and the gaps, live with the test.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import os
import pathlib
import threading
import uuid
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from ouroboros.tool_result_delivery import RESULT_VIEW_MARKER
from tests.test_llm_claudexor import MODEL as CX
from tests.test_llm_claudexor import ROUTE, result

OPUS = "anthropic/claude-opus-5.5"  # OpenRouter's built-in key
OPUS_NAMED = "openrouter@named::anthropic/claude-opus-5.5"  # a second direct key of the same model
START_CLASSES = frozenset({"first_request", "entered_model", "tools_changed", "system_changed", "paused"})
REPLY_ALLOWANCE = 65_536  # Main's reply ceiling (loop_llm_call.MAIN_LOOP_MAX_TOKENS): what a total-bound place adds
FAVOURABLE_HEADROOM = 8_192  # context_budget.NANO_MIN_HEADROOM_TOKENS: the most a stated window is read down by
DENSE = "DENSE-PART"


@dataclasses.dataclass(frozen=True)
class Place:
    """Where a request is answered: ``model`` as configured, an engine ``account`` ("" for a keyed provider), the
    ``window`` its evidence states, the ``bound`` it enforces (0: the stated window) in its ``kind``, and the
    tokens it counts per four characters (``density``) that the host learns only from its answers."""
    name: str
    model: str
    account: str = ""
    window: int = 300_000
    bound: int = 0
    kind: str = "input"
    density: float = 1.0
    dense: float = 1.0  # the density of a part marked ``DENSE`` (a script or a table the host cannot tell apart)
    marked: bool = False  # its cache keeps only prefixes the host's markers end (else: any prefix it answered)

    @property
    def limit(self) -> int:
        return self.bound or self.window


def _digest(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _unmarked(value: Any) -> Any:
    """A provider's cache identity ignores where the host puts its cache markers, and a row's text the same either as
    a string or as text blocks (a marker turns the one into the other)."""
    if isinstance(value, dict):
        out = {key: _unmarked(item) for key, item in value.items() if key != "cache_control"}
        content = out.get("content")
        if isinstance(content, list) and content and all(isinstance(b, dict) and set(b) == {"type", "text"}
                                                         and b["type"] == "text" for b in content):
            out["content"] = "".join(block["text"] for block in content)
        return out
    if isinstance(value, list):
        return [_unmarked(item) for item in value]
    return value


def _marked(value: Any) -> bool:
    if isinstance(value, dict):
        return "cache_control" in value or any(_marked(item) for item in value.values())
    return isinstance(value, list) and any(_marked(item) for item in value)


def _elements(tools: Any, rows: List[Dict[str, Any]]) -> Tuple[List[Tuple[str, Any]], int, List[bool]]:
    """``(elements, start, marked)``: the tool array, each system block, each later row; ``start`` counts the first
    two; ``marked`` says which carry the host's cache marker."""
    from ouroboros.send_clock import CLOCK_NOTE_PREFIX

    rows = list(rows or [])
    if rows and str(rows[-1].get("content") or "").startswith(CLOCK_NOTE_PREFIX):
        rows = rows[:-1]  # the request-local clock row is replaced every round by design
    raw: List[Tuple[str, Any]] = [("tools", tools or [])]
    for row in rows:
        content = row.get("content")
        if row.get("role") in ("system", "developer") and isinstance(content, list):
            raw += [("system", block) for block in content]
        else:
            raw.append(("row", row))
    return ([(kind, _unmarked(item)) for kind, item in raw], 1 + sum(1 for kind, _ in raw if kind == "system"),
            [_marked(item) for _, item in raw])


class Provider:
    """The far side of both transports: one answer per physical request, a cache per place, a log per request."""

    def __init__(self, places: Dict[str, Place], script: List[tuple], parts: Optional[Dict[str, int]] = None):
        self.places, self.script = places, list(script)
        self.parts = dict(parts or {})  # a readable part's name -> its characters
        self.asked: Dict[str, int] = {}  # a read's call id -> the characters of the part it asked for
        self.cache: Dict[str, Tuple[List[str], List[int]]] = {}
        self.marks: Dict[str, set] = {}  # key -> the prefixes (by digest) that a marked element of an answer ended
        self.log: List[Dict[str, Any]] = []
        self.engine_account = "account-a"  # who the engine serves an unpinned request with
        self.before: Dict[int, Any] = {}  # answered count -> a callback run inside that request's send
        self.lock = threading.Lock()

    def place_of(self, model: str, account: str = "") -> Place:
        for place in self.places.values():
            if place.model == model and place.account == account:
                return place
        raise AssertionError(f"no place for {model!r} {account!r}")

    def answer(self, place: Place, tools: Any, rows: List[Dict[str, Any]], reply: int) -> Dict[str, Any]:
        """``{"kind": "refused"|"lost"|"answered", ...}`` for one request at ``place``."""
        with self.lock:
            items, start, marked = _elements(tools, rows)
            digests = [_digest(item) for item in items]
            texts = [json.dumps(item, ensure_ascii=False, sort_keys=True, default=str) for item in items]
            sizes = [math.ceil(len(text) * (place.dense if DENSE in text else place.density) / 4) for text in texts]
            total = sum(sizes)
            answered = sum(1 for row in self.log if row.get("kind") == "answered")
            known = sum(math.ceil(len(text) * place.density / 4) for text in texts)  # what the host can learn
            entry = {"n": len(self.log) + 1, "round": answered + 1, "place": place.name, "tokens": total,
                     "known": known, "start_tokens": sum(sizes[:start]), "elements": len(items),
                     "start_elements": start, "digests": digests, "sizes": sizes,
                     "facts": next((text for text in reversed(texts) if "[CONTEXT_FACTS]" in text), "")}
            before = next((row for row in reversed(self.log) if row.get("kind") == "answered"), None)
            entry["reads"] = [  # each part that arrives in this request: its size, whether it came as a partial view
                {"chars": self.asked.pop(call), "cut": RESULT_VIEW_MARKER in text,
                 "before": before["tokens"] if before and before["place"] == place.name else None}
                for (kind, item), text in zip(items, texts)
                if kind == "row" and (call := str(item.get("tool_call_id") or "")) in self.asked]
            self.log.append(entry)
            limit, over = place.limit, total + (reply if place.kind != "input" else 0)
            if place.kind != "cut" and over > limit:
                entry["kind"] = "refused"
                entry["message"] = {
                    "input": f"prompt is too long: {total} tokens > {limit} maximum",
                    "total": f"input length and `max_tokens` exceed context limit: {total} + {reply} > {limit}",
                    "total_unnumbered": "The request exceeds the context length of this model.",
                    "total_as_a_size": "Requested tokens exceed the model limit: lower max_tokens or shorten the prompt.",
                    "total_worded": (f"This model's maximum context length is {limit} tokens. However, your request has "
                                     f"{total} input tokens. Please reduce the length of the input messages."
                                     if total > limit else
                                     f"'max_tokens' or 'max_completion_tokens' is too large: {reply}. This model's maximum "
                                     f"context length is {limit} tokens and your request has {total} input tokens "
                                     f"({reply} > {limit} - {total}).")}[place.kind]
                return entry
            entry["cut"] = place.kind == "cut" and over > limit
            hook = self.before.pop(answered + 1, None)
            if hook is not None:
                hook()
            action = self.script.pop(0) if self.script else ("final", "Done.")
            if action[0] == "lost":
                entry["kind"] = "lost"
                return entry
            key = place.name if not place.account else place.account
            old, common = self.cache.get(key), 0
            if old is not None:
                for a, b in zip(old[0], digests):
                    if a != b:
                        break
                    common += 1
            if place.marked:  # an explicit-marker cache reads to the last prefix a marker of this key ended
                held = self.marks.setdefault(key, set())
                common = max((k for k in range(1, len(digests) + 1) if _digest(digests[:k]) in held), default=0)
                held.update(_digest(digests[:k + 1]) for k, flag in enumerate(marked) if flag)
            read = sum(sizes[:common])
            entry.update(kind="answered", read=read, written=total - read, common=common, action=action[0],
                         where="first" if old is None else "append" if common == len(old[0]) else
                         "start" if common < start else "transcript")
            self.cache[key] = (digests, sizes)
            entry["message"] = _message(action, entry["n"])
            if action[0] == "read":
                self.asked[f"read-{entry['n']}"] = self.parts.get(action[-1], 0)
            return entry


def _message(action: tuple, n: int) -> Dict[str, Any]:
    kind = action[0]
    if kind == "final":
        return {"role": "assistant", "content": action[1]}
    calls = {"read": ("read_file", {"root": "task_drive", "path": action[-1]}),
             "fold": ("compact_context", {"working_note": "The parts read so far are summarised here; their exact "
                                                          "text stays readable.", "keep_unit_ids": []}),
             "switch": ("switch_model", {"model": action[-1]}),
             "return": ("switch_model", {"primary": "return"})}
    name, arguments = calls[kind]
    return {"role": "assistant", "content": f"Next: {name}.", "tool_calls": [{
        "id": f"{kind}-{n}", "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}]}


class _Refused(Exception):
    """An OpenAI-compatible provider's size refusal, as the SDK raises it."""
    status_code = 400

    def __init__(self, message: str):
        super().__init__(f"Error code: 400 - {{'error': {{'message': '{message}', 'code': 400}}}}")
        self.body = {"error": {"message": message, "code": 400}}


class _Response:
    def __init__(self, payload: Dict[str, Any]):
        self._payload = payload

    def model_dump(self) -> Dict[str, Any]:
        return copy.deepcopy(self._payload)


def wire_claudexor(gateway: Any, provider: Provider) -> None:
    """Answer every operation the fake Claudexor gateway accepts (the prototype's seam)."""
    gateway.results, gateway.dispatch = [], []
    create = gateway.create_model_operation

    def create_with_answer(ref, *, idempotency_key, **options):
        if idempotency_key not in gateway.accepted_operations:
            payload = gateway.uploads[-1][0]
            pinned = (payload.get("account") or {}).get("profileId")
            account = pinned or provider.engine_account
            entry = provider.answer(provider.place_of(CX, account), payload.get("tools"), payload.get("messages"),
                                    REPLY_ALLOWANCE)
            route = {**ROUTE, "credentialProfileId": account, "accountFingerprint": f"fingerprint-{account[-1]}"}
            if entry["kind"] == "refused":
                outcome = result(outcome="failed", route=route,
                                 problem={"code": "invalid_request", "message": entry["message"]})
            elif entry["kind"] == "lost":
                outcome = result(outcome="unknown", route=route)
            else:
                outcome = result(route=route)
                outcome["message"] = entry["message"]
                outcome["usage"] = {"input_tokens": entry["tokens"], "output_tokens": 400,
                                    "cached_input_tokens": entry["read"], "cache_write_tokens": None,
                                    "reasoning_tokens": 0}
            gateway.results.append(outcome)
            gateway.dispatch.append("unknown" if entry["kind"] == "lost" else "response_received")
        return create(ref, idempotency_key=idempotency_key, **options)

    gateway.create_model_operation = create_with_answer


def wire_remote(client: Any, provider: Provider, monkeypatch: Any) -> None:
    """The OpenAI lane's SDK client in memory: each candidate the real lane built reaches the stub whole."""
    from ouroboros import pricing
    from ouroboros import usage_accounting as ua
    from ouroboros.llm import LLMClient

    monkeypatch.setattr(pricing, "estimate_cost_optional", lambda *a, **kw: None)
    monkeypatch.setattr(ua, "estimate_cost_optional", lambda *a, **kw: None)
    monkeypatch.setattr(pricing, "_fetch_live_rows", lambda *a, **kw: {})
    monkeypatch.setattr("ouroboros.llm.transport_no_proxy", lambda: False)
    monkeypatch.setattr("ouroboros.llm_fallback.consume_stream", lambda response, **_kw: response)
    monkeypatch.setattr(LLMClient, "_SUPPORTED_PARAMS_FETCHED", True)
    monkeypatch.setattr(LLMClient, "_SUPPORTED_PARAMS_CACHE", {})
    monkeypatch.setattr(LLMClient, "_get_supported_parameters", lambda *a: None)

    def sdk(target):
        model = OPUS_NAMED if target.get("connection") else OPUS

        def create(**candidate):
            reply = int(candidate.get("max_tokens") or candidate.get("max_completion_tokens") or REPLY_ALLOWANCE)
            entry = provider.answer(provider.place_of(model), candidate.get("tools"), candidate.get("messages"), reply)
            if entry["kind"] != "answered":
                raise _Refused(entry.get("message") or "upstream lost the answer")
            message = entry["message"]
            finish = "tool_calls" if message.get("tool_calls") else "stop"
            details = {"cached_tokens": entry["read"], "cache_write_tokens": entry["written"]}
            return _Response({"id": f"gen-{entry['n']}", "model": target.get("resolved_model"),
                              "choices": [{"index": 0, "finish_reason": finish, "message": message}],
                              "usage": {"prompt_tokens": entry["tokens"], "completion_tokens": 400, "cost": 0.0,
                                        "prompt_tokens_details": details}})
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    monkeypatch.setattr(client, "_get_remote_client", sdk)


def hold_queue(provider: Provider, at: int, seconds: float, held: List[Any]) -> None:
    """Another holder takes the install's one slot of this route inside answered request ``at - 1`` (a slot file of a
    live process beside that send's own) and frees it ``seconds`` after request ``at`` joins the queue."""
    from ouroboros.config import resolve_data_dir

    stop = threading.Event()

    def take():
        [own] = list((resolve_data_dir() / "state" / "model_concurrency").glob("*/*.slot"))
        other = own.parent / f"{0:020d}-{os.getpid()}-{uuid.uuid4().hex}.slot"
        other.touch()

        def free():
            while not stop.is_set() and not list(other.parent.glob("*.ticket")):
                stop.wait(0.01)
            stop.wait(seconds)
            other.unlink(missing_ok=True)
        watcher = threading.Thread(target=free, name="work-life-queue-holder", daemon=True)
        held.append((stop, watcher, other))
        watcher.start()

    provider.before[at - 1] = take


def release_queue(held: List[Any]) -> None:
    for stop, watcher, path in held:
        stop.set()
        watcher.join(5)
        path.unlink(missing_ok=True)


def growth_files(drive_root: Any, sizes: List[int], tag: str = "part") -> List[str]:
    folder = pathlib.Path(drive_root) / "task_drives" / "task-one"
    folder.mkdir(parents=True, exist_ok=True)
    names = []
    for index, chars in enumerate(sizes):
        name = f"{tag}-{index:02d}.txt"
        line = f"{tag} {index:02d} line: the measured facts of this part, written out in full for the reader. "
        (folder / name).write_text((line * (chars // len(line) + 1))[:chars], encoding="utf-8")
        names.append(name)
    return names


class World:
    """``tests.test_context_economy_window._native`` for a life: its places, the production route question answered
    from them (``context_fit._context_route`` with the place's evidence), and a capture that a door can repeat."""

    def __init__(self, main_call: Any, monkeypatch: Any, tmp_path: Any, places: Dict[str, Place], *, primary: str,
                 fallbacks: Tuple[str, ...], light: str = "", mode: str = "max", memory: float = 1.0,
                 pinned: bool = True):
        from ouroboros import context, fallback_cooldown, loop_transport
        from ouroboros.memory_view import snapshot_json
        from ouroboros.model_slots import MODEL_ACCOUNTS_KEY
        from tests import _memory_scale_fixture as msf
        from tests._memory_view_context import world

        self.ctx, self.gateway, self.owner, self.events, _decide, _observations = main_call
        self.monkeypatch, self.mode, self.places = monkeypatch, mode, places
        self.primary = places[primary]
        chain = [places[name] for name in fallbacks]
        accounts = {"main": self.primary.account, "fallback": [p.account for p in chain if p.model == CX]}
        settings = {"OUROBOROS_CONTEXT_MODE": mode, "OUROBOROS_TASK_REVIEW_MODE": "off", "OUROBOROS_SAFETY_MODE": "off",
                    "OUROBOROS_MAX_ROUNDS": "40", "MCP_ENABLED": "false", "OUROBOROS_MODEL": self.primary.model,
                    "OUROBOROS_MODEL_FALLBACKS": ",".join(p.model for p in chain),
                    "OUROBOROS_MODEL_LIGHT": places[light].model if light else "",
                    "OPENROUTER_API_KEY": "test-only-key", "OUROBOROS_MODEL_MAX_CONCURRENCY": "1",
                    "OUROBOROS_CONNECTIONS": json.dumps([{"kind": "openrouter", "name": "named",
                                                          "api_key": "test-key-2"}])}
        if pinned:
            settings[MODEL_ACCOUNTS_KEY] = json.dumps({key: value for key, value in accounts.items() if value})
        for key, value in settings.items():
            monkeypatch.setenv(key, value)
        if not pinned:
            monkeypatch.delenv(MODEL_ACCOUNTS_KEY, raising=False)
        monkeypatch.delenv("USE_LOCAL_MAIN", raising=False)
        monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
        monkeypatch.setattr(fallback_cooldown, "is_cooling_down", lambda *_a: False)
        monkeypatch.setattr(loop_transport, "upstream_transport_reachable", lambda *_a, **_kw: {})
        monkeypatch.setattr(loop_transport, "interruptible_wait_sleep", lambda *_a: False)
        monkeypatch.setattr(context, "_context_fit_route", self.route)
        self.said: List[str] = []  # what the host told the work's feed
        self.tools = self.new_tools()
        self.env, self.memory, _rooms = world(tmp_path / "w")
        self.task = {"type": "task", "text": "Read the parts and report.", "id": "tmain", "chat_id": 1}
        self.snapshot = lambda extra=0: snapshot_json(msf.snapshot(rows=max(4, int(158 * memory)),
                                                                   pages=max(2, int(113 * memory)) + extra))
        self.provider: Optional[Provider] = None

    def evidence(self, place: Place) -> Any:
        from ouroboros import capability_evidence as ce

        options = (dict(source_id="codex", credential_profile_id=place.account,
                        account_fingerprint=f"fingerprint-{place.account[-1]}") if place.model == CX else {})
        provider = "claudexor" if place.model == CX else "openrouter"
        return SimpleNamespace(
            route_fp=ce.route_fingerprint(provider=provider, model=place.model, base_url="", options=options),
            status="confirmed", stale=False, window_tokens=place.window, window_kind="total", source="advertised",
            source_id="codex" if place.model == CX else "", credential_profile_id=place.account,
            account_fingerprint=f"fingerprint-{place.account[-1]}" if place.account else "")

    def route(self, task: Dict[str, Any], **_kw: Any) -> Tuple[Dict[str, Any], Any]:
        """The production route for ``task`` (settings, role, pin, carried account) with its place's evidence."""
        from ouroboros.context_fit import _context_route
        from ouroboros.context_window import route_observation_scope

        route, settings = _context_route(task)
        if route["provider"] == "claudexor":
            account = (route["options"].get("credential_profile_id") or route.get("account_pin")
                       or (self.provider.engine_account if self.provider else "account-a"))
            place = self.provider.place_of(CX, account) if self.provider else self.places["a"]
            evidence = self.evidence(place)
            route["options"] = {**route["options"], "source_id": "codex", "credential_profile_id": place.account,
                                "account_fingerprint": evidence.account_fingerprint}
            route["window_observation_fp"] = route_observation_scope(route, settings)
            return route, evidence
        model = str(task.get("model") or self.primary.model)
        return route, self.evidence(next(p for p in self.places.values() if p.model == model))

    def new_tools(self) -> Any:
        from tests.test_model_wait_controls import _loop_tools

        tools = _loop_tools(self.ctx, self.owner)
        tools._ctx.task_metadata = {"model_role": "main"}
        tools._ctx.model_wait_context = self.owner
        tools._ctx.task_model_override = self.primary.model
        tools._ctx.emit_progress_fn = lambda text, *_a, **_kw: self.said.append(str(text))
        return tools

    def plan(self, *, system_extra: str = "", pages_extra: int = 0, runtime_extra: str = "", task: Any = None) -> Any:
        from ouroboros import context
        from ouroboros.tool_policy import initial_tool_schemas

        task = task or self.task
        core = context._capture_context_core(self.env, self.memory, task, None, None)
        core = dataclasses.replace(core, memory_view_json=self.snapshot(pages_extra), base_prompt=core.base_prompt
                                   + system_extra, dynamic_text=core.dynamic_text + runtime_extra)
        schemas = initial_tool_schemas(self.tools, context_mode=self.mode)
        built = context._build_context_fit_plan(
            self.env, core, task, preferred_mode=self.mode, tool_schemas=schemas,
            route_resolver=lambda *_a, **_kw: self.route({**task, "model_role": "main"}))
        account = self.primary.account
        receipt = ({**ROUTE, "credentialProfileId": account, "accountFingerprint": f"fingerprint-{account[-1]}"}
                   if self.primary.model == CX else {})
        return dataclasses.replace(built, model_role="main", model_route=receipt)

    def run(self, plan: Any) -> Any:
        from tests.test_context_economy_window import _run

        return _run(self.ctx, self.tools, self.events, plan)

    def door(self, plan_after: Any, *, changed_tool: bool = False) -> Any:
        """A restart or an Update: the first process's real continuation state, a new registry, a fresh capture."""
        from ouroboros import continuation_tools, loop, owner_wait
        from ouroboros.tool_policy import initial_tool_schemas

        old = self.tools._ctx
        usage = dict(getattr(old, "_accumulated_usage", None) or {})
        trace = dict(getattr(old, "_execution_trace", None) or {"reasoning_notes": [], "tool_calls": []})
        schemas = initial_tool_schemas(self.tools, context_mode=self.mode)
        state = owner_wait.continuation_state(old, list(self.ctx.messages), trace, usage, 3, schemas,
                                              set(getattr(old, "_loop_mailbox_seen_ids", None) or ()))
        state.update(wait_id="w-door", quiz_id="", reason="owner")
        self.tools = self.new_tools()
        if changed_tool:
            def updated(tools, **kwargs):
                out = json.loads(json.dumps(initial_tool_schemas(tools, **kwargs)))
                for schema in out:
                    if schema.get("function", {}).get("name") == "read_file":
                        schema["function"]["description"] += " The reader names its source line range."
                return out
            self.monkeypatch.setattr(continuation_tools, "initial_tool_schemas", updated)
            self.monkeypatch.setattr(loop, "initial_tool_schemas", updated)
        self.tools._ctx.owner_wait_resume = {"wait_id": "w-door", "quiz_id": "", "reason": "owner"}
        self.tools._ctx.owner_wait_callback = lambda _ctx, _checkpoint: "answered"
        loaded = [state]
        self.monkeypatch.setattr(loop, "load_owner_wait", lambda _ctx, *_a, **_kw: loaded.pop() if loaded else {})
        return self.run(plan_after)


def drained(events: Any) -> List[Dict[str, Any]]:
    rows = []
    while not events.empty():
        rows.append(events.get_nowait().get("data") or {})
    return rows


def ledger(root: Any) -> List[Dict[str, Any]]:
    """Each attempt in start order: its state, ``prefix_class`` and the room facts of its physical context."""
    from ouroboros.transcript_prefix import prefix_class
    from tests._usage_store_testing import attempt_rows_in_start_order

    out = []
    for row in attempt_rows_in_start_order(root):
        physical = row.get("physical_context") or {}
        name, detail = prefix_class(physical.get("prefix"), row.get("prompt_cache_ttl") or "", None)
        fact = physical.get("prefix") if isinstance(physical.get("prefix"), dict) else {}
        out.append({"state": row.get("state"), "class": name, "detail": detail, "model": row.get("model"),
                    "origins": list(fact.get("origins") or ()),
                    "window": physical.get("capacity_total_tokens"), "reserve": physical.get("response_reserve_tokens"),
                    "mode": physical.get("rendered_mode")})
    return out


@dataclasses.dataclass(frozen=True)
class Life:
    """One work: its places (``primary`` first, then the configured ``fallbacks``; ``light`` is the model the mind may
    switch to), the characters of each part the mind reads (one per answered request), the mind's own acts inserted
    before the answered request they name (``acts``), and what happens to it: a ``door`` after that many answered
    requests (``update``: a changed tool and prompt), the install's queue held at request ``queue[0]`` for ``queue[1]``
    seconds, the engine serving another account from request ``rotate`` (``pinned``: the role pins its account)."""
    places: Tuple[Place, ...]
    primary: str = "a"
    fallbacks: Tuple[str, ...] = ()
    light: str = ""
    reads: Tuple[int, ...] = (8_000,) * 4
    dense: Tuple[int, ...] = ()  # the reads (by index) marked ``DENSE``
    acts: Tuple[Tuple[int, tuple], ...] = ()
    door: int = 0
    update: bool = False
    queue: Tuple[float, ...] = ()
    rotate: int = 0
    pinned: bool = True
    memory: float = 0.35  # of the owner-shaped memory of tests/_memory_scale_fixture.py
    mode: str = "max"
    second: bool = False  # after it, a second work's first request on the primary's key
    keeps_mode: bool = False  # nothing in this life is a reason to run in a lower mode than it started in


def run_life(life: Life, main_call: Any, monkeypatch: Any, tmp_path: Any) -> Dict[str, Any]:
    """Drive ``life`` and return what it did: the provider's log, the ledger, the loop's events, the final text."""
    places = {place.name: place for place in life.places}
    world = World(main_call, monkeypatch, tmp_path, places, primary=life.primary, fallbacks=life.fallbacks,
                  light=life.light, mode=life.mode, memory=life.memory, pinned=life.pinned)
    names = growth_files(world.ctx.drive_root, list(life.reads))
    for index in life.dense:
        path = pathlib.Path(world.ctx.drive_root) / "task_drives" / "task-one" / names[index]
        path.write_text(DENSE + " " + path.read_text(encoding="utf-8"), encoding="utf-8")
    script: List[tuple] = [("read", name) for name in names]
    for at, act in sorted(life.acts, reverse=True):
        script.insert(at - 1, act)
    first, second = (script[:life.door] + [("final", "Parked.")], script[life.door:]) if life.door else (script, [])
    provider = world.provider = Provider(places, first, dict(zip(names, life.reads)))
    wire_claudexor(world.gateway, provider)
    wire_remote(world.ctx.llm, provider, monkeypatch)
    held: List[Any] = []
    if life.queue:
        hold_queue(provider, int(life.queue[0]), float(life.queue[1]), held)
    if life.rotate:
        provider.before[life.rotate - 1] = lambda: setattr(provider, "engine_account", "account-b")
    try:
        built = world.plan()
        world.ctx.messages[:] = built.messages_for(built.initial_mode)
        text = world.run(built)[0]
        if life.door:
            provider.script = second
            after = world.plan(system_extra="\nOne new rule line of the updated prompt." if life.update else "",
                               pages_extra=1 if life.update else 0,
                               runtime_extra="\nCaptured again by the new process.")
            text = world.door(after, changed_tool=life.update)[0]
        record = {"text": text, "log": list(provider.log), "ledger": ledger(world.ctx.drive_root),
                  "events": drained(world.events), "windows": _window_refusals(), "said": list(world.said),
                  "world": world}
        record["second"] = _second_work(world) if life.second else None
    finally:
        release_queue(held)
    return record


def _window_refusals() -> List[Dict[str, Any]]:
    """The size refusals the product recorded for its routes (``context_window.record_refusal``), any route."""
    from ouroboros import capability_evidence as ce

    store = ce._read(ce.canonical_evidence_root()).get("main_windows") or {}
    return [row for record in store.values() for row in (record or {}).get("observations", []) if row.get("refused")]


def _second_work(world: World) -> Dict[str, Any]:
    """Another room's new work, its start captured and fitted by the same producers, sent once on the same key."""
    from ouroboros.tool_policy import initial_tool_schemas

    other = {"type": "task", "text": "Summarise the other room's open questions.", "id": "tother", "chat_id": 2}
    plan = world.plan(task=other)
    schemas = initial_tool_schemas(world.tools, context_mode=world.mode)
    world.provider.script = [("final", "Noted.")]
    world.ctx.llm.chat(plan.messages_for(plan.initial_mode), world.primary.model, tools=schemas, model_role="main")
    return world.provider.log[-1]


def answered_rows(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Each answered request: the product's class of it, the place that answered, its input and the room it left.

    The ledger's sent attempts (every state but ``released``) and the provider's log are the same requests in the
    same order; ``room`` is what the request left under the line it was measured against (the capacity its ledger row
    names, the admitted line of ``WindowRecord``, less the reply reserve and the input the place counted)."""
    sent = [row for row in record["ledger"] if row["state"] != "released"]
    assert len(sent) == len(record["log"]), ("harness: the ledger and the provider disagree on the requests sent",
                                             [row["state"] for row in sent], [entry["kind"] for entry in record["log"]])
    rows = []
    for row, entry in zip(sent, record["log"]):
        if entry["kind"] != "answered":
            continue
        room = (row["window"] - (row["reserve"] or 0) - entry["known"]) if row["window"] else None
        rows.append({**row, "place": entry["place"], "tokens": entry["tokens"], "known": entry["known"], "room": room,
                     "token": f"{row['class']}@{entry['place']}"})
    return rows


def breaches(record: Dict[str, Any], places: Dict[str, Place], *, sequence: str, details: Dict[str, str],
             refusals: int = 0, queued: bool = False, mode: str = "") -> List[str]:
    """Every clause of the law ``record`` breaks, each named by the mechanism that holds it (``tests/test_cache_
    optimization.py``): the sequence of (class, answering place); P4's room after a start write that narrowed the
    view for its window (a wall's pass, or a bind onto a place whose line the request answered before it did not fit:
    under the admitted line, measured by the place within half an eighth, since the host sizes by its own estimate)
    and its other side, that a restore of a view its window held takes nothing on a guess; P3's
    favourable reading before a send, unless the product recorded a refusal of the total kind no later than it, the
    same reading for a part the mind read (it comes back whole while the request with it whole would still go out),
    and its learning from a refusal or a cut reply; P5's queue that refuses nothing, tells the work's feed that a send
    waits in line and states the time on the mind's next facts line (``queued``: this life met a full queue; a life
    that met none reads neither); a start change that no seam of the host claims; and, where ``mode`` names one, that
    no request ran in another mode."""
    import re

    rows = answered_rows(record)
    found: List[str] = []
    tokens = " ".join(row["token"] for row in rows)
    if not re.fullmatch(sequence, tokens):
        found.append(f"sequence: {tokens!r} is not {sequence!r}")
    before = None  # the request answered before this one
    for row in rows:
        if row["class"] in details and row["detail"] != details[row["class"]]:
            found.append(f"detail: {row['token']} names {row['detail']!r}, not {details[row['class']]!r}")
        if row["class"] in ("system_changed", "tools_changed", "history_rewritten") and not row["origins"]:
            found.append(f"origin: {row['token']} changed the start with no seam of the host claiming it")
        if row["class"] in START_CLASSES - {"first_request"} and row["room"] is not None and before is not None:
            stated = places[row["place"]].window
            eighth = math.ceil(stated / 8)  # of the stated window; ``room`` is under the admitted line
            line = row["known"] + row["room"]  # the admitted line of this place, as its ledger row measured it
            # What the request answered before it counts here; the view was narrowed for the window when a wall's
            # pass wrote this start or that request did not fit this place's line.
            carried = math.ceil(before["known"] * places[row["place"]].density / places[before["place"]].density)
            walled = any(str(origin).startswith(("refusal_pass", "memory_coarsened")) for origin in row["origins"])
            if walled or carried > line:
                if row["room"] < eighth // 2:
                    found.append(f"P4 room: {row['token']} left {row['room']} of {stated} after its start write, "
                                 f"under the eighth {eighth}")
            elif before["place"] == row["place"] and carried - row["known"] > max(0, eighth - (line - carried)) // 2 > 0:
                # A restore of a view its window held: more than half the room it lacked is gone from the request.
                found.append(f"P4 guess: {row['token']} sent {row['known']} where {carried} fitted its line {line}: "
                             f"room was taken from a view the window held")
        before = row
        if mode and row["mode"] != mode:
            found.append(f"mode: {row['token']} ran in {row['mode']!r}, not {mode!r}")
    total = any(row.get("kind") == "total" for row in record["windows"])  # what the product learned, not the place
    total_refused = [entry["round"] for entry in record["log"] if entry["kind"] == "refused"
                     and places[entry["place"]].kind.startswith("total")] if total else []
    for event in record["events"]:
        if event.get("checkpoint_kind") == "context_fit_refused" and event.get("cause") == "known_window_prediction":
            # The window the route states, never the capacity frame the event carries (the admitted line plus R).
            stated = min((place.window for place in places.values() if place.model == event.get("model")),
                         default=int(event.get("capacity_total_tokens") or 0))
            size = int(event.get("estimated_input_tokens") or 0)
            standing = any(at <= int(event.get("round") or 0) for at in total_refused)
            if size <= stated - FAVOURABLE_HEADROOM and not standing:
                found.append(f"P3 reading: refused before sending an input of {size} the stated {stated} holds with "
                             f"{stated - size} to spare (no total refusal stands)")
    for entry in record["log"]:
        place = places[entry["place"]]
        line = place.window - (REPLY_ALLOWANCE if total else FAVOURABLE_HEADROOM)
        for read in entry.get("reads") or () if place.dense == 1.0 else ():  # a dense part: the host's estimate is off
            whole = math.ceil(read["chars"] * place.density / 4)
            # The unasked share is one eighth; the host sizes by its own estimate, so a tenth and 2,000 tokens of slack.
            if (read["cut"] and read["before"] is not None and whole <= math.ceil(place.window / 8)
                    and read["before"] + math.ceil(whole * 1.1) + 2_000 <= line):
                found.append(f"P3 delivery: request {entry['n']} got a part of {whole} tokens as a partial view, though "
                             f"{read['before']} + {whole} fits the {line} this place admits")
    refused = [entry for entry in record["log"] if entry["kind"] == "refused"]
    if len(refused) > refusals:
        found.append(f"P3 learning: the provider refused {len(refused)} requests (allowed {refusals}): "
                     f"{[entry['message'][:60] for entry in refused]}")
    cut = [entry["n"] for entry in record["log"] if entry.get("cut")]
    if len(cut) > 1:
        found.append(f"P3 cut reply: requests {cut} had their reply cut by the place's total (one teaches it)")
    queue = [event for event in record["events"] if event.get("type") == "llm_api_error"
             and event.get("provider_code") == "model_capacity_unavailable"]
    if queue:
        found.append(f"P5 queue: the install's own queue refused {len(queue)} requests (it keeps a place instead)")
    waits = [text for text in record["said"] if "waits in line" in text and "keeps its model" in text]
    told = [entry["n"] for entry in record["log"]
            if "in line under this install's Requests at once per model limit" in entry["facts"]]
    if queued != bool(waits) or queued != bool(told):
        found.append(f"P5 words: the life {'met' if queued else 'met no'} full queue, yet its feed read {len(waits)} "
                     f"waiting notes and the facts lines of requests {told} stated a time in line")
    return found
