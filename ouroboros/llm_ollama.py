"""Native Ollama transport for positively identified compatible connections.

The saved provider/address/model identity stays unchanged. Discovery reads only
the same root on private endpoints (GET /api/version and /api/ps, POST /api/show
for the route's model), without redirects or inference. Passive lookups may reuse
a negative verdict; a generation cannot trust the version of a previous server:
every physical send has its own read (``_admit`` before a call's first send,
``limit_compatible_send`` for each later one, ``_load_window`` natively), and a
saved identity never stands in for it. A refused, redirected or missing read and
a complete answer in another shape state nothing; a busy, unreachable or cut-short
one withholds the send. What follows a read that states nothing belongs to
``llm_ollama_guard``. The sender owns its client through settlement, including a
late answer after Pause.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import re
import time
from collections import OrderedDict
from threading import Lock
from urllib.parse import urlunsplit

import httpx

from ouroboros import llm_ollama_guard as guard
from ouroboros.deadline_utils import llm_transport_timeout_sec
from ouroboros.llm_ollama_policy import load_version, record_window, thinking_mode_from_show
from ouroboros.llm_ollama_policy import native_target_key as _identity
from ouroboros.llm_ollama_wire import (  # noqa: F401 - error classes are this module's public names
    OllamaIntegrityError, OllamaRouteUnsettledError, Reply, image_context, native_error_body, payload_for)
from ouroboros.llm_stream import ProviderStreamError
from ouroboros.server_window import _positive, ollama_load, ollama_model_maximum, private_host, server_root


def _client_options(target, no_proxy, timeout):
    from ouroboros.net_transport import verify_kwargs
    if not (target.get("api_key") or target.get("keyless")):
        from ouroboros.llm_attempt import ConnectionUnavailable
        raise ConnectionUnavailable(target.get("connection", ""), "missing_credentials")
    headers = dict(target.get("default_headers") or {})
    if target.get("api_key"):
        headers["Authorization"] = "Bearer " + target["api_key"]
    return dict(headers=headers, trust_env=not no_proxy, follow_redirects=False,
                timeout=llm_transport_timeout_sec(timeout), **verify_kwargs())


_rejected = OrderedDict()  # identity -> monotonic expiry of a "not Ollama" verdict
_identity_lock = Lock()
_REJECTION_TTL_SEC = 600.0
_REPLACED = {"protocol": "openai-compatible", "replaced": "ollama"}
_SHOW_FIELDS = {"details": dict, "model_info": dict, "capabilities": list, "modelfile": str, "template": str}
#: The ``/v1`` verdict a read left for the one physical send that follows it (``limit_compatible_send``).
_VERDICT = "route_verdict"
#: The accounting wait generation at preparation; a later wait entry expires the verdict.
_VERDICT_WAIT_GENERATION = "route_verdict_wait_generation"
log = logging.getLogger(__name__)


class OllamaProgramChanged(OllamaRouteUnsettledError):
    """A saved Ollama address now serves another program; its native route is retired."""
    code = "ollama_program_changed"


def _remember(target, record):
    from ouroboros import capability_evidence as ce
    if not guard.note_ollama(target, record["version"], record["source"]):  # a Settings Test keeps this much
        raise OllamaRouteUnsettledError("Ollama route evidence could not be saved; request not sent")
    if not target.get("settings_test"):
        identity, root = _identity(target), ce.canonical_evidence_root()
        previous = _REPLACED if target.get("route_notice") else target.get("ollama_record")
        if not ce._store_evidence(root, "native_protocols", identity, record, previous=previous):
            # Another discovery or refresh can publish the same model facts with a
            # different observation time. Adopt its saved row, not our unsaved date.
            # The unchanged prior row is not evidence of a successful refresh.
            held = (ce._read(root).get("native_protocols") or {}).get(identity)
            shown_at = held.get("shown_at") if isinstance(held, dict) else None
            if not (shown_at and record.get("shown_at") and held != previous
                    and held == {**record, "shown_at": shown_at}):
                raise OllamaRouteUnsettledError("Ollama route evidence could not be saved; request not sent")
            record = held
        with _identity_lock:
            _rejected.pop(identity, None)
    target["ollama_record"] = record
    target.pop("route_notice", None)


def _known(target, identity):
    from ouroboros import capability_evidence as ce
    record = ce._read(ce.canonical_evidence_root()).get("native_protocols", {}).get(identity, {})
    target.pop("ollama_record", None)
    target.pop("route_notice", None)
    if record.get("protocol") != "ollama":
        if record.get("replaced") == "ollama":
            target["route_notice"] = _replacement_notice(target)
        return False
    target["ollama_record"] = record
    with _identity_lock:
        _rejected.pop(identity, None)
    return True


def _rejected_now(identity):
    with _identity_lock:
        return _rejected.get(identity, 0) > time.monotonic()


def _reject(identity):
    with _identity_lock:
        _rejected[identity] = time.monotonic() + _REJECTION_TTL_SEC
        _rejected.move_to_end(identity)
        if len(_rejected) > 128:
            _rejected.popitem(last=False)


def forget_rejection(target, error=None):
    """A failed compatible send may mean the program behind the address changed: discover again next time.
    The caution an admitted send of the call carried still reaches the round, whatever the outcome; beside the
    words of a later send it withheld (``route_caution_used``), never in their place."""
    if isinstance(target, dict) and _discoverable(target):
        with _identity_lock:
            _rejected.pop(_identity(target), None)
        if error is not None and target.get("route_caution"):
            used = target["route_caution"], target.get("route_caution_key", "")
            if getattr(error, "route_caution", None):
                error.route_caution_used = used
            else:
                error.route_caution, error.route_caution_key = used


def route_notices(target):
    """The route facts an answered ``/v1`` send carries in its usage: the owner-facing words and, with a caution, the
    route it was minted for (``llm_ollama_guard.unidentified``)."""
    return {key: target[key] for key in ("route_notice", "route_caution", "route_caution_key") if target.get(key)}


def _discoverable(target):
    return target.get("provider") == "openai-compatible" and private_host(target.get("base_url"))


def _url(target, path):
    """Where this metadata read goes: the server root of the base URL (a trailing ``/v1`` is the client's own,
    never the server's root) with the native path, the base URL's query VERBATIM after it and its userinfo in
    the authority, while the words show neither (``llm_ollama_guard.shown_address``).

    The compatible send names the same connection: the SDK client of this route composes each request path
    before the same query bytes (``net_transport.openai_client_class``; the stock SDK folds the path into the
    query) and carries the same userinfo. The query token a gateway routes on travels intact and the path
    keeps the base's own prefix, so on a query- or path-routed gateway the read identifies the same tenant
    that then receives the history. The path is transport-normalized (``httpx.URL`` resolves dot segments
    before the request leaves)."""
    base = httpx.URL(str(target["base_url"]).strip())
    head, separator, query = base.raw_path.decode("ascii", "replace").partition("?")
    netloc = base.netloc.decode("ascii", "replace")
    userinfo = base.userinfo.decode("ascii", "replace")
    return urlunsplit((base.scheme, userinfo + "@" + netloc if userinfo else netloc, server_root(head) + path,
                       query if separator else "", ""))


def _address(target):
    """The server the words name; its path prefix stays out (``llm_ollama_guard.shown_address``)."""
    return guard.shown_address(target["base_url"])


def _set_verdict(target, verdict):
    """The ``/v1`` verdict of the read made just now and its accounting wait generation."""
    from ouroboros._usage_wait import presend_wait_generation

    target[_VERDICT], target[_VERDICT_WAIT_GENERATION] = verdict, presend_wait_generation()


def _pop_verdict(target, default=None):
    target.pop(_VERDICT_WAIT_GENERATION, None)
    return target.pop(_VERDICT, default)


def _document(response, method, path):
    """One metadata reply. A refusal, a redirect that is not followed, an unsupported path and a complete
    answer in another program's shape state nothing (``None``); a busy server or a cut-short reply raises."""
    status = response.status_code
    if status == 404 and path == "/api/show":
        try:
            value = response.json()
        except ValueError:
            return None
        if isinstance(value, dict) and isinstance(value.get("error"), str):
            return value
    if 300 <= status < 400 or status in (401, 403, 404, 405, 501):
        return None
    if status in (408, 429) or status >= 500:
        raise ValueError
    try:
        value = response.json()
    except ValueError:
        if response.content.lstrip()[:1] in (b"{", b"["):
            raise  # a JSON document that began and did not finish
        return None
    if status != 200 or not isinstance(value, dict) or not value or value.get("error") is not None:
        return None
    return value


def _version_number(doc):
    version = (doc or {}).get("version")
    if isinstance(version, str) and re.fullmatch(r"\d+\.\d+\.\d+(?:-[\w.-]+)?(?:\+[\w.-]+)?", version):
        return version
    return ""


class OllamaUpgradeRequired(OllamaRouteUnsettledError):
    code, kind, retryable = "ollama_guard_unavailable", "bad_request", False


# Wire fact (External facts, docs/development/02-naming-and-boundaries.md): the lower bound 0.12.6 comes
# from ollama/ollama#12582, the release whose /api/chat takes truncate:false and shift:false (sent by us).
# The dated link, api/types.go at v0.40.16 (https://github.com/ollama/ollama/blob/v0.40.16/api/types.go),
# read 2026-11-18, is documentation of the wire shape: not a new live observation and not proof for the
# exact server. Separately, a reply's prompt_eval_cached_count is read and validated when present
# (llm_ollama_wire.Reply._terminal_usage) but proves nothing about input being kept.
# Re-check: a native send refreshes version and /api/ps; _load_window re-reads /api/show when the version
# or digest changed, the digest is unconfirmed or the record is stale. Stale support risks silent input
# loss or a wrong window/reasoning mode; stale absence can refuse a working server.
GUARD_SINCE = "0.12.6"  # the first release whose /api/chat takes truncate:false and shift:false


def _supports_guard(version):
    release, since = (tuple(map(int, re.match(r"\d+\.\d+\.\d+", text)[0].split("."))) for text in (version, GUARD_SINCE))
    return release > since or release == since and "-" not in version.partition("+")[0]


def update_to(version):
    """The release an Ollama version read must reach for that guard: '' when it has it, or is no version."""
    return "" if not _version_number({"version": version}) or _supports_guard(version) else GUARD_SINCE


def _metadata(target, path, body=None):
    from ouroboros.transport_custody import is_loopback_base_url
    method = "POST" if body is not None else "GET"
    try:
        response = yield method, path, body
    except httpx.HTTPError as cause:
        error = OllamaRouteUnsettledError(
            f"Could not reach {_address(target)} for its {path} metadata; model request not sent")
        known_native = (target.get("ollama_record") or {}).get("protocol") == "ollama"
        if known_native or not is_loopback_base_url(target["base_url"]):
            error.kind = "transport_unavailable"
        else:
            error.kind = "provider_error"
        error.probe_reason = "Timed out" if isinstance(cause, httpx.TimeoutException) else "Could not reach provider"
        raise error from None
    try:
        return response.status_code, _document(response, method, path)
    except ValueError:
        error = OllamaRouteUnsettledError(
            f"Could not read the {path} metadata of {_address(target)} (HTTP {response.status_code}); "
            "model request not sent")
        error.status_code = response.status_code
        raise error from None


def _model_record(shown, digest, version):
    from ouroboros.utils import utc_now_iso
    return {"protocol": "ollama", "version": version, "source": "/api/version+/api/ps+/api/show",
            **({"thinking_mode": thinking_mode_from_show(shown), "maximum": ollama_model_maximum(shown),
                "digest": digest, "shown_at": utc_now_iso()} if shown is not None else {})}


def _missing_model(doc, target):
    name = re.escape(target["resolved_model"])
    return bool(re.fullmatch(rf"""model ['"]{name}['"] not found(?:, try pulling it first)?""",
                             str((doc or {}).get("error", ""))))


def _show(target):
    status, shown = yield from _metadata(target, "/api/show", {"model": target["resolved_model"]})
    if status == 404 and _missing_model(shown, target):
        return True, None
    return bool(status == 200 and shown and any(isinstance(shown.get(key), kind) and shown[key]
                                                for key, kind in _SHOW_FIELDS.items())), shown


def _other_program(target):
    """Another program's own documented metadata names it; an error or a missing path names nothing."""
    from ouroboros.server_window import PROGRAM_DOCUMENTS
    for name, path, shaped in PROGRAM_DOCUMENTS:
        status, doc = yield from _metadata(target, path)
        if status == 200 and shaped(doc):
            return name
    return ""


def _current_version(target):
    _, doc = yield from _metadata(target, "/api/version")
    version = _version_number(doc)
    if version and not _supports_guard(version):
        # A semver alone is not a program identity. /show also identifies an
        # older installation whose /ps endpoint is missing or unavailable.
        recognized, _ = yield from _show(target)
        if recognized:
            guard.note_ollama(target, version, "/api/version+/api/show")
            _note_check(target, "ollama", version=version)
            raise OllamaUpgradeRequired(
                f"Ollama {version} can silently truncate history. Upgrade to {GUARD_SINCE} or later.")
        _, ps = yield from _metadata(target, "/api/ps")
        if target.get("ollama_record") or isinstance((ps or {}).get("models"), list):
            raise OllamaRouteUnsettledError("Conflicting Ollama metadata; model request not sent")
        return ""
    if not version and target.get("ollama_record"):
        if doc is None:
            _, ps = yield from _metadata(target, "/api/ps")
            if ps is None:
                # Silence is no replacement: only another program's own answer retires the native route.
                if program := (yield from _other_program(target)):
                    _note_check(target, program)
                    _set_verdict(target, ("", ""))  # the replacement's own read admits its /v1 send
                    raise OllamaProgramChanged(_retire(target))
                _note_check(target, "")
                raise guard.held(_address(target))  # this connection read Ollama here
        raise OllamaRouteUnsettledError("Could not confirm the current Ollama version; request not sent")
    return version


def _discover(target):
    """``True`` for a supported Ollama. ``None`` when its version and model list answered but the model's own
    metadata did not: signs without proof. ``False`` when nothing names it."""
    version = yield from _current_version(target)
    if not version:
        return False
    _, ps = yield from _metadata(target, "/api/ps")
    if not isinstance((ps or {}).get("models"), list):
        return False
    recognized, shown = yield from _show(target)
    if not recognized:
        return None
    load = ollama_load(ps, {target["resolved_model"]})
    _remember(target, _model_record(shown, load.get("digest"), version))
    return True


def _admit(target):
    """This send's own read decides its protocol: ``True`` native Ollama, ``False`` ``/v1``.

    Signs of Ollama without its proof withhold the send, whatever else answers. Another identified program
    takes ordinary ``/v1``. A read that names nothing is withheld where Ollama was seen at the address;
    elsewhere it needs the connection's admission and carries a caution (``llm_ollama_guard.unidentified``).
    A ``/v1`` verdict decides only the physical send that follows this read (``limit_compatible_send``)."""
    target.pop(_VERDICT, None)
    found = yield from _discover(target)
    if found is None:
        raise OllamaRouteUnsettledError("Conflicting Ollama metadata; model request not sent")
    program = "ollama" if found else (yield from _other_program(target))
    _note_check(target, program, version=(target.get("ollama_record") or {}).get("version", "") if found else "")
    if not found:
        # Another program named itself: no caution. Else the connection's admission, with its caution and route,
        # or what withholds the send (``llm_ollama_guard.unidentified`` raises it).
        _set_verdict(target, ("", "") if program else guard.unidentified(target, _address(target)))
    return bool(found)


def _note_check(target, program, **observed):
    """What this connection's own check identified, for Settings; a Test records none."""
    if not target.get("settings_test"):
        guard.note_check(target, program or guard.UNIDENTIFIED, **observed)


def _load_window(target, requested, *, sizing_only=False):
    """Return (bounded num_ctx, observed load). Auto omits num_ctx; a tag can change since discovery."""
    from ouroboros.capability_evidence import _CONFIRMED_TTL_SEC
    from ouroboros.context_window import _window_age
    record = target.get("ollama_record") or {}
    version = yield from _current_version(target)
    if not version:
        raise OllamaRouteUnsettledError("Could not confirm the current Ollama version; request not sent")
    target["ollama_load_version"] = load_version(target)  # what this read may be published over (``_send_load_sequence``)
    _, ps = yield from _metadata(target, "/api/ps")
    if not isinstance((ps or {}).get("models"), list):
        raise OllamaRouteUnsettledError("Could not confirm the current Ollama model list; request not sent")
    _note_check(target, "ollama", version=version)  # this read found Ollama again, also after a check that did not
    same_version = record.get("version") == version
    load = ollama_load(ps, {target["resolved_model"]})
    loaded = _loaded_window(ps, target)
    if sizing_only and not requested and same_version:
        return 0, loaded
    digest = load.get("digest")
    fresh = bool(record.get("shown_at") and _window_age(record["shown_at"]) < _CONFIRMED_TTL_SEC)
    if not (same_version and digest and record.get("digest") == digest and fresh):
        recognized, shown = yield from _show(target)
        if not recognized:
            raise OllamaRouteUnsettledError("Ollama returned invalid model metadata; request not sent")
        updated = _model_record(shown, digest, version)
        if record and (any(record.get(key) != updated.get(key) for key in ("version", "digest", "maximum", "thinking_mode"))
                       or bool(shown) != bool(record.get("shown_at")) or shown is not None and not fresh):
            _remember(target, updated)
        record = updated
    if requested and not record.get("maximum"):
        error = OllamaRouteUnsettledError(
            "Ollama did not state the selected model's context maximum. Pull the model or use Auto in Settings.")
        error.code, error.kind, error.retryable = "ollama_window_unavailable", "bad_request", False
        raise error
    target["ollama_thinking_mode"] = record.get("thinking_mode")
    window = min(requested, record.get("maximum") or 0)
    return window, loaded


def _drive(steps, read):
    try:
        step = next(steps)
        while True:
            try:
                response = read(*step)
            except httpx.HTTPError as error:
                step = steps.throw(error)
            else:
                step = steps.send(response)
    except StopIteration as done:
        return done.value


async def _drive_async(steps, read):
    # Advancing the shared metadata protocol can publish under a cross-process
    # store lock. Join each step before releasing its client on cancellation.
    from ouroboros._usage_wait import presend_off_loop

    def advance(value):
        try:
            step = steps.throw(value) if isinstance(value, httpx.HTTPError) else steps.send(value)
        except StopIteration as done:
            return True, done.value
        return False, step

    value = None
    while True:
        done, step = await presend_off_loop(advance, value)
        if done:
            return step
        try:
            value = await read(*step)
        except httpx.HTTPError as error:
            value = error


def _gate(target, *, before_send, no_proxy=False):
    """``True``/``False`` when memory decides without a network read, else ``None``."""
    if not _discoverable(target) or not (target.get("api_key") or target.get("keyless")):
        return False
    if before_send:
        target["ollama_probe_no_proxy"] = no_proxy
    identity = _identity(target)
    known = _known(target, identity)
    if before_send and (known or target.get("route_notice")):
        # A native route saved before addresses were kept, or one retired since: Ollama was read here.
        guard.note_ollama(target, "", "saved native route")
    if known:
        return True
    return False if not target.get("settings_test") and not before_send and _rejected_now(identity) else None


def _discovery_result(target, found):
    if found or target.get("settings_test"):
        return bool(found)
    # Another worker may have identified this route while our metadata read was in flight.
    identity = _identity(target)
    _reject(identity)
    return _known(target, identity)


def is_ollama(target, no_proxy=False, *, before_send=False):
    known = _gate(target, before_send=before_send, no_proxy=no_proxy)
    if known is not None:
        return known
    with httpx.Client(**_client_options(target, no_proxy, 2.0)) as client:
        found = _drive((_admit if before_send else _discover)(target), _metadata_reader(client, target))
    return _discovery_result(target, found)


async def is_ollama_async(target, no_proxy=False, *, before_send=False):
    if not _discoverable(target):
        return False
    from ouroboros._usage_wait import presend_off_loop
    known = await presend_off_loop(_gate, target, before_send=before_send, no_proxy=no_proxy)
    if known is not None:
        return known
    async with httpx.AsyncClient(**_client_options(target, no_proxy, 2.0)) as client:
        found = await _drive_async((_admit if before_send else _discover)(target), _metadata_reader(client, target))
    return await presend_off_loop(_discovery_result, target, found)


def _replacement_notice(target):
    return (f"The program at {_address(target)} no longer answers as Ollama. This connection now uses "
            "OpenAI-compatible /v1. Its native protection against truncating the input is unavailable.")


def _retire(target):
    from ouroboros import capability_evidence as ce
    from ouroboros.llm_ollama_policy import forget_load
    notice = _replacement_notice(target)
    if target.get("settings_test"):
        target.pop("ollama_record", None)
        return notice
    identity, root = _identity(target), ce.canonical_evidence_root()
    try:
        with ce._store_transaction(root):
            data = ce._load(root)
            held = data.setdefault("native_protocols", {}).get(identity)
            if held and held.get("protocol") == "ollama" and held != target.get("ollama_record"):
                raise OllamaRouteUnsettledError("Ollama metadata changed during discovery; request not sent")
            if not held or held.get("protocol") == "ollama":
                data["native_protocols"][identity] = _REPLACED
                forget_load(data, target)
                if not ce._save(root, data):
                    raise OSError("could not save the replacement")
    except (OSError, TimeoutError) as exc:
        raise OllamaRouteUnsettledError("Could not retire the prior Ollama window; request not sent") from exc
    _reject(identity)
    target.pop("ollama_record", None)
    target.pop("requested_load_window", None)
    target["route_notice"] = notice
    log.warning("%s", notice)
    return notice


def _http_error(response):
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as error:
        # The shared reader consumes the adapter-normalized body; the original
        # response bytes and exact provider sentence remain available.
        error.args = (f"HTTP {response.status_code} from Ollama chat at {guard.shown_address(str(response.url))}",)
        error.status_code = response.status_code
        try:
            error.body = response.json()
            if response.status_code == 400:
                typed = native_error_body(error.body)
                # Accounting keeps the received envelope; the shared reader keeps this typed reading of it.
                error.adapter_typed_body = typed is not error.body
                error.body = typed
        except ValueError:
            error.body = {}
        from ouroboros._usage_response import provider_answer
        error.provider_message = provider_answer(error).innermost()[1] or response.text
        raise


def _loaded_window(ps, target):
    if not isinstance((ps or {}).get("models"), list):
        return None
    if any(not isinstance(row, dict) or not (row.get("name") or row.get("model")) for row in ps["models"]):
        return None
    load = ollama_load(ps, {target["resolved_model"]})
    return (_positive(load.get("context_length")) or None) if load else 0


def _read_window(client, target):
    try:
        response = client.get(_url(target, "/api/ps"), timeout=2.0)
        return _loaded_window(_document(response, "GET", "/api/ps"), target)
    except (httpx.HTTPError, ValueError):
        return None


async def _read_window_async(client, target):
    try:
        response = await client.get(_url(target, "/api/ps"), timeout=2.0)
        return _loaded_window(_document(response, "GET", "/api/ps"), target)
    except (httpx.HTTPError, ValueError):
        return None


def _prepare(target, messages, tools, reasoning_effort, max_tokens, tool_choice, temperature, stream, response_format, window=0):
    from dataclasses import replace
    from ouroboros.llm_attempt import _attempt_request, _finalized_physical_candidate
    image_urls = []
    payload = payload_for(target, messages, tools, reasoning_effort, max_tokens, tool_choice, temperature,
                          stream=stream, response_format=response_format, window=window, image_urls=image_urls)
    payload = _finalized_physical_candidate(target, payload, "ollama/chat", fresh_clock=True)
    target = {**target, "requested_reasoning_effort": reasoning_effort}
    request = _attempt_request(target, payload, image_context=image_context(payload, image_urls),
                               source="provider_test" if target.get("settings_test") else None)
    # The canonical sealed candidate is native JSON. Accounting still reserves
    # the actual native output allowance instead of interpreting absent max_tokens as zero.
    effort = dict(request.effort)
    if "think" in payload:
        effort.update(sent={"think": payload["think"]}, sent_state="explicit")
    request = replace(request, max_completion_tokens=payload["options"]["num_predict"], effort=effort,
                      prompt_count_basis="evaluated" if request.prompt_count_basis else "")
    return payload, request


def prospective_request(target, messages, tools, effort, max_tokens, stream, *, requested):
    """Price the same native wire the next generation would send, with no generation or load."""
    from ouroboros.utils import transport_no_proxy
    no_proxy = transport_no_proxy()
    if not is_ollama(target, no_proxy, before_send=True):
        return None
    with httpx.Client(**_client_options(target, no_proxy, 2.0)) as client:
        try:
            window, _ = _drive(_load_window(target, requested),
                               _metadata_reader(client, target))
        except OllamaProgramChanged:
            return None
    return _prepare(target, messages, tools, effort, max_tokens, "auto", None, stream, None, window)[1]


def _result(response, target):
    from ouroboros.llm_attempt import attach_processing_receipt
    from ouroboros.llm_local import split_message_reasoning
    split_message_reasoning(response["choices"][0]["message"])
    usage = dict(response["usage"])
    usage.update(response_finish_reason=response["choices"][0]["finish_reason"],
                 cached_tokens=usage.get("prompt_tokens_details", {}).get("cached_tokens"),
                 resolved_model=target.get("usage_model") or target["resolved_model"],
                 reported_model=response.get("model"), provider=target["provider"],
                 cost=None, cost_final=False)
    if "_stream_receipt" in response:
        usage["stream_receipt"] = response["_stream_receipt"]
    attach_processing_receipt(target, usage)
    return response["choices"][0]["message"], usage


def _metadata_reader(client, target):
    def read(method, path, body):
        return client.request(method, _url(target, path), timeout=2.0, **({"json": body} if body is not None else {}))
    return read


def limit_compatible_send(target, send, resolve=None):
    """Admit one physical ``/v1`` send to a private compatible connection by its own decision, after its model slot.

    The read its call made just before preparing it (``_admit``) decides its first send; a retry and a send whose
    slot waited read again (``_read_verdict``), with the consent its connection holds then: ``resolve``, the
    route's own resolver, reads it again, so a consent withdrawn meanwhile admits nothing. A send whose
    accounting waited (ledger contention, ``_usage_wait``) after the verdict's read re-reads too: the verdict
    expires with the wait it did not survive. A Settings Test keeps the form it checks. What withholds a send
    raises before it; what admits it stays with the call (``_admitted``). A connection no read decides (a
    public one) only takes its slot."""
    from ouroboros._usage_wait import presend_wait_generation
    from ouroboros.model_concurrency import limit_physical_send

    if "ollama_probe_no_proxy" not in target:
        return limit_physical_send(target, send)
    from ouroboros.llm_attempt import require_physical_dispatch_window
    prepared_generation = target.pop(_VERDICT_WAIT_GENERATION, None)
    prepared = [_pop_verdict(target)]  # made for this send, never for a later one
    waited = []  # this dispatch was decided right after its slot wait

    def take_prepared():
        verdict, prepared[0] = prepared[0], None
        return verdict if verdict is not None and prepared_generation == presend_wait_generation() else None

    def decide(verdict=None):
        _admitted(target, verdict if verdict is not None else _read_verdict(target, resolve))

    def after_wait():
        prepared[0] = None  # the wait outlived the read made before it
        decide()
        waited.append(True)

    async def admitted_send_async():
        from ouroboros._usage_wait import presend_read_off_loop
        if waited:
            waited.clear()
        elif (verdict := take_prepared()) is not None:
            decide(verdict)
        else:
            await presend_read_off_loop(decide)
            require_physical_dispatch_window()  # Pause may have arrived during the read
        return await send()

    def admitted_send():
        if waited:
            waited.clear()
        elif (verdict := take_prepared()) is not None:
            decide(verdict)
        else:
            decide()
            require_physical_dispatch_window()
        return send()

    return limit_physical_send(target, admitted_send_async if inspect.iscoroutinefunction(send) else admitted_send,
                               after_wait=after_wait)


def _read_verdict(target, resolve):
    """The read of a physical send no earlier read was made for: raises what withholds it, else its ``/v1`` verdict."""
    if resolve is not None and not target.get("settings_test"):
        _restate_consent(target, resolve)
    with httpx.Client(**_client_options(target, target["ollama_probe_no_proxy"], 2.0)) as client:
        reader = _metadata_reader(client, target)
        if _known(target, _identity(target)):
            try:
                _drive(_load_window(target, 0, sizing_only=True), reader)
            except OllamaProgramChanged:  # the replacement named itself (``_current_version``)
                return _pop_verdict(target, ("", ""))
        elif not _drive(_admit(target), reader):
            return _pop_verdict(target)
    raise OllamaRouteUnsettledError("This /v1 request now needs Ollama native chat; request not sent")


def _admitted(target, verdict):
    """The decision that admits this physical send: ``(caution, route)``, empty for a program that named itself.
    The call keeps the caution of its first send that was told one (the route at its first use), whatever later
    sends find."""
    words, route = verdict
    if words and not target.get("route_caution"):
        target["route_caution"], target["route_caution_key"] = words, route


def _restate_consent(target, resolve):
    """The connection's consent as Settings hold it now; one removed or disabled meanwhile holds none."""
    from ouroboros.provider_models import connection_id
    try:
        consent = resolve(f"{connection_id('openai-compatible', target.get('connection') or '')}::"
                          f"{target['resolved_model']}").get(guard.CONSENT_FIELD)
    except Exception:
        consent = None
    if consent:
        target[guard.CONSENT_FIELD] = consent
    else:
        target.pop(guard.CONSENT_FIELD, None)


def _native_send(target, send, no_proxy, requested, window, previous_load):
    from ouroboros.model_concurrency import limit_physical_send
    thinking = target.get("ollama_thinking_mode")

    def confirm():
        with httpx.Client(**_client_options(target, no_proxy, 2.0)) as client:
            selected, loaded = _drive(_load_window(target, requested), _metadata_reader(client, target))
        if (selected != window or target.get("ollama_thinking_mode") != thinking
                or not window and previous_load != loaded):
            raise OllamaRouteUnsettledError("The model load changed while waiting; prepared request not sent")

    return limit_physical_send(target, send, after_wait=confirm)


def _refusal_window(loaded, previous_load, window):
    if loaded is None and (not window or previous_load == window):
        loaded = previous_load
    return min(loaded or 0, window or loaded or 0)


def _send_load_sequence(target, requested, window, previous_load):
    """The order this send is dispatched under. An explicit load is what this request makes the runner hold. In
    Auto the statement is the ``/api/ps`` answer read before it, published only over the statement that read was
    made against: one overtaken by a different statement (a reload, an unload) is not written and gives no order."""
    from ouroboros.llm_attempt import require_physical_dispatch_window

    read = {} if window else {"read_sequence": target.get("ollama_load_version"), "replay": True}
    sequence = (None if target.get("settings_test") else
                record_window(target, window or previous_load, requested=requested, **read))
    require_physical_dispatch_window()  # publishing may wait for the store lock
    return sequence


def _post_load(target, window, previous_load, loaded, sequence):
    """``(load, order)`` a settled Auto send is bound to. A closing read that first shows another load is
    published for the next round, only over the statement this send went out under; the answer or refusal
    stays with that statement and its load. Only a cold send, whose own generation made the load it then
    reads, takes the new one."""
    if not window and sequence is not None and loaded is not None and loaded != previous_load:
        current = record_window(target, loaded, read_sequence=sequence)
        return (loaded, current) if current is not None and not previous_load else (previous_load, sequence)
    return loaded, sequence


def _load_statement(target, sequence):
    """The load a settled answer was dispatched under, ``_stamp_refusal``'s pair as a
    JSON-stable list (usage is persisted and compared after a checkpoint round trip).

    A native answer always carries one. ``sequence`` None (a Settings Test, an unpublished
    scope, a busy store) states no number: like an unnumbered refusal, the acceptance then
    binds nothing. Only an answer with no statement at all, an ordinary provider's, writes
    unconditionally (``context_anchor.record_input_anchor``)."""
    return [_identity(target).removeprefix("ollama:"), sequence]


def _stamp_refusal(error, target, window, previous_load, loaded, sequence):
    loaded, sequence = _post_load(target, window, previous_load, loaded, sequence)
    error.nominal_context_window = _refusal_window(loaded, previous_load, window)
    error.ollama_load_statement = (_identity(target).removeprefix("ollama:"), sequence)


def _completed(response, payload, window, statement):
    response["usage"]["ollama_context"] = {
        "requested": payload["options"].get("num_ctx"), "loaded": window or None,
        "read": "answered" if window is not None else "unavailable"}
    # A success keeps its dispatch load binding too (``context_anchor``): a late
    # answer from an old load is paid usage, never evidence for the current one.
    response["usage"]["ollama_load_statement"] = statement
    # The native guard refuses overflow. Usage counts the rendered template;
    # a character estimate cannot establish which words were consumed.
    actual = response["usage"]["prompt_tokens"]
    bounded = min((n for n in (window, payload["options"].get("num_ctx")) if n), default=0)
    response["usage"]["capacity_input_tokens"] = min(actual, bounded)
    excess = response["usage"]["total_tokens"] - bounded
    if bounded and excess > 0:
        response["usage"]["ollama_context"]["exceeded_by"] = excess
    return response


def chat_ollama(target, messages, tools, reasoning_effort, max_tokens, tool_choice, temperature=None, *,
                no_proxy=False, timeout=None, stream=False, response_format=None, model_role="", window=0):
    from ouroboros.deadline_utils import physical_dispatch_timeout
    from ouroboros.llm_attempt import _candidate_before_dispatch, _execute_candidate
    from ouroboros._usage_wait import close_after_model_send
    client = httpx.Client(**_client_options(target, no_proxy, timeout))
    try:
        requested = window
        window, previous_load = _drive(_load_window(target, window), _metadata_reader(client, target))
        payload, request = _prepare(target, messages, tools, reasoning_effort, max_tokens, tool_choice,
                                    temperature, stream, response_format, window)
        def send():
            sequence = _send_load_sequence(target, requested, window, previous_load)
            reply = Reply()
            try:
                with reply.stream(ndjson=stream) as feed:
                    with client.stream("POST", _url(target, "/api/chat"), json=payload,
                                       timeout=physical_dispatch_timeout(llm_transport_timeout_sec(timeout))) as response:
                        if response.status_code >= 300:
                            response.read()
                            _http_error(response)
                        for chunk in response.iter_bytes():
                            if feed(chunk):
                                break
            except (httpx.HTTPStatusError, ProviderStreamError) as error:
                if getattr(error, "status_code", None) == 400:
                    _stamp_refusal(error, target, window, previous_load, _read_window(client, target), sequence)
                raise
            loaded, current = _post_load(target, window, previous_load, _read_window(client, target), sequence)
            return _completed(reply.assembled, payload, loaded, _load_statement(target, current))

        response = _execute_candidate(request, _native_send(target, send, no_proxy, requested, window, previous_load),
                                      _candidate_before_dispatch(payload, request))
    finally:
        try:
            close_after_model_send(client)
        except httpx.RequestError:
            log.debug("Ollama client close failed after the send ended", exc_info=True)
    return _result(response, target)


async def chat_ollama_async(target, messages, tools, reasoning_effort, max_tokens, tool_choice, temperature=None, *,
                            no_proxy=False, timeout=None, stream=False, response_format=None, model_role="", window=0):
    from ouroboros.deadline_utils import physical_dispatch_timeout
    from ouroboros.llm_attempt import _candidate_before_dispatch, _execute_candidate_async
    from ouroboros._usage_wait import aclose_after_model_send, presend_off_loop
    client = httpx.AsyncClient(**_client_options(target, no_proxy, timeout))
    try:
        requested = window
        window, previous_load = await _drive_async(_load_window(target, window), _metadata_reader(client, target))
        payload, request = _prepare(target, messages, tools, reasoning_effort, max_tokens, tool_choice,
                                    temperature, stream, response_format, window)
        async def send():
            sequence = await presend_off_loop(_send_load_sequence, target, requested, window, previous_load)
            reply = Reply()
            try:
                with reply.stream(ndjson=stream) as feed:
                    async with client.stream("POST", _url(target, "/api/chat"), json=payload,
                                             timeout=physical_dispatch_timeout(llm_transport_timeout_sec(timeout))) as response:
                        if response.status_code >= 300:
                            await response.aread()
                            _http_error(response)
                        async for chunk in response.aiter_bytes():
                            if feed(chunk):
                                break
            except (httpx.HTTPStatusError, ProviderStreamError) as error:
                if getattr(error, "status_code", None) == 400:
                    loaded = await _read_window_async(client, target)
                    await presend_off_loop(_stamp_refusal, error, target, window, previous_load, loaded, sequence)
                raise
            except asyncio.CancelledError:
                # The caller's cancellation can overtake a complete, valid terminal
                # reply: buffered before EOF (nonstream) or already framed while the
                # response closes (NDJSON). The wire settled and retained exactly that
                # paid answer; return it so the joined sender's accounting keeps it,
                # while model_send_async still hands the cancellation to its caller.
                # A partial byte prefix stays under unknown-outcome custody.
                if reply.assembled is None:
                    raise
                return _completed(reply.assembled, payload, None, _load_statement(target, sequence))
            try:
                loaded = await _read_window_async(client, target)
                loaded, current = await presend_off_loop(_post_load, target, window, previous_load, loaded, sequence)
            except asyncio.CancelledError:
                # model_send_async still propagates its receiver's cancellation;
                # the joined sender must return the already completed paid answer.
                loaded, current = None, sequence
            return _completed(reply.assembled, payload, loaded, _load_statement(target, current))

        response = await _execute_candidate_async(request, _native_send(target, send, no_proxy, requested, window, previous_load),
                                                  _candidate_before_dispatch(payload, request))
    finally:
        try:
            await aclose_after_model_send(client)
        except httpx.RequestError:
            log.debug("Ollama client close failed after the send ended", exc_info=True)
    return _result(response, target)
