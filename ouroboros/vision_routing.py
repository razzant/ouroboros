"""One image projection policy and the route-scoped image-input evidence it reads.

A model name is not evidence. Whether a route accepts image input is a fact about
the exact resolved route (provider, endpoint, API surface, routing options and the
available account identity), recorded only from a catalog Ouroboros already reads
(OpenRouter ``/models``; an OpenAI-compatible gateway's ``/models`` read by the
task-start window probe) and valid for 24 hours from that catalog's response. No
record means unknown, and unknown never withholds the owner's image: the route
receives the pixels and answers for itself. A Claudexor route answers from its
account's catalog (``provider_models.supports_vision``).

``prepare_messages_for_send`` decides what each send carries; transport builders
encode it (a lane that cannot carry bytes marks them as its own limit). For a Main
send the owner's image mode decides:

* Auto: pixels unless the route's own metadata says no; then a caption from the
  explicit vision slot or an automatic candidate that is not confirmed-no, else a
  marker naming that metadata and its date.
* Inline: pixels even when metadata says no; the provider's refusal is shown.
* Caption: a caption, never pixels.
* Off: a marker, and no hidden image work.

Our own lanes that cannot carry image bytes (local llama.cpp, GigaChat) are named
as ours in the marker. A VLM or caption call (``purpose`` "vlm"/"caption") names
its model explicitly and receives the pixels; it never captions, so a caption
cannot recurse into another caption.

A provider's refusal of a request that carried images is an observation about
that request (size, count, format and content filters look the same), never a
fact about the route. ``retry_refused_image_round`` gives a Main round one
same-round retry with the refused images replaced, and the task remembers which
route refused which image (``REFUSED_IMAGES_KEY``) so that image is not resent to
that route; a VLM or caption refusal is remembered the same way, without a retry.
Forced finalization and the prospective wrap-up candidate read that memory
through this projection but do not recover a first refusal themselves.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from hashlib import sha256
import json
import logging
import pathlib
import threading
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

from ouroboros.config import get_image_input_mode, get_vision_caption_timeout_sec, get_vision_model, resolve_effort
from ouroboros.deadline_utils import owner_deadline_exhausted, transport_timeout_with_deadline
from ouroboros.observability import new_call_id, persist_call
from ouroboros.provider_models import provider_for_model, supports_vision
from ouroboros.utils import emit_cognitive_operation_event, utc_now_iso
from ouroboros.vision_image_limits import prepare_caption_image, prepare_route_images, refusal_identity, sent_caption_identity
from ouroboros.config import runtime_setting

log = logging.getLogger(__name__)


_CAPTION_PROMPT = (
    "Describe this image in detail for a coding/research agent that may not see pixels. "
    "Be objective and include visible text, UI state, diagrams, layout, and salient details. "
    "Do not infer hidden facts."
)

# The capability_evidence.json namespace of route-scoped image-input records.
EVIDENCE_NAMESPACE = "image_input"
# A catalog's statement holds for 24 hours from its response; older is unknown.
_EVIDENCE_FRESH_SEC = 24 * 3600
# Our own lanes that cannot carry image bytes: llama.cpp is launched without a
# vision handler, and the GigaChat lane flattens message content to text.
_OWN_LANES_WITHOUT_IMAGES = {"local": "local llama.cpp", "gigachat": "GigaChat"}
_IMAGE_TYPES = frozenset({"image_url", "image"})
_OFF_MARKER = "[image omitted: the image-input mode is Off]"
# A documented field present in a shape this parser does not read.
_MALFORMED = object()


def _vision_finalization_reserve() -> float:
    try:
        from ouroboros.config import get_finalization_grace_sec
        return float(get_finalization_grace_sec())
    except Exception:
        return 0.0


@dataclass
class VisionRoutingContext:
    model: str
    llm: Any
    accumulated_usage: Dict[str, Any]
    drive_root: pathlib.Path | None = None
    task_id: str = ""
    event_queue: Any = None
    use_local: bool = False
    task_attempt: Any = None
    deadline_ts: Any = None
    model_role: str = "main"
    model_account_override: str | None = None


# --- Evidence: one parser, one route scope, one store ---------------------------

def _modalities(value: Any) -> Any:
    if value is None or value == []:
        return None
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return "image" in {item.strip().lower() for item in value}
    return _MALFORMED


def _strict_flag(value: Any) -> Any:
    if value is None:
        return None
    return value if isinstance(value, bool) else _MALFORMED


def _supported_object(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict) and isinstance(value.get("supported"), bool):
        return value["supported"]
    return _MALFORMED


def image_input_from_row(row: Any) -> Optional[bool]:
    """Image input stated by one catalog row: True/False, or None when the row is unclear.

    Only a documented, unambiguous field counts: a non-empty input-modality list
    (``architecture.input_modalities``), a strict JSON bool in ``supports_vision``
    or ``capabilities.supports_vision``, or Anthropic's
    ``capabilities.image_input.supported``. A missing, null or empty field says
    nothing; any other shape, or fields that disagree, make the whole row unknown,
    so a catalog format this parser does not know can never become a "no".
    """
    if not isinstance(row, dict):
        return None
    architecture, capabilities = row.get("architecture"), row.get("capabilities")
    if any(part is not None and not isinstance(part, dict) for part in (architecture, capabilities)):
        return None
    architecture, capabilities = architecture or {}, capabilities or {}
    found = [
        _modalities(architecture.get("input_modalities")),
        _strict_flag(row.get("supports_vision")),
        _strict_flag(capabilities.get("supports_vision")),
        _supported_object(capabilities.get("image_input")),
    ]
    if any(value is _MALFORMED for value in found):
        return None
    verdicts = {value for value in found if value is not None}
    return verdicts.pop() if len(verdicts) == 1 else None


def image_route_scope(target: Mapping[str, Any]) -> str:
    """Key of the image-input namespace for one resolved route, or "" when its scope is unknown.

    Provider, endpoint, API surface, routing options and the available account
    identity: what one catalog response describes. The model is a member of the
    record, because one response observes every model of its endpoint at once.
    """
    provider = str(target.get("provider") or "").strip().lower()
    endpoint = str(target.get("base_url") or "").strip().rstrip("/").lower()
    if not provider or not endpoint:
        return ""
    routing: Dict[str, Any] = {}
    if provider == "openrouter":
        from ouroboros.llm_routing import _resolve_or_provider

        routing = _resolve_or_provider()
    scope = {
        "provider": provider, "endpoint": endpoint,
        "surface": "messages" if provider == "anthropic" else "chat.completions",
        "routing": routing, "account": str(target.get("account_fingerprint") or ""),
    }
    return sha256(json.dumps(scope, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:24]


def record_catalog_image_input(provider: str, base_url: str, rows: Any, *, source: str) -> None:
    """Persist one catalog response's image-input statements for its route scope.

    Called by the readers Ouroboros already runs, right after the response
    arrives, so the record's time is the catalog's observation time. The newest
    response replaces the scope's record whole: a model it no longer describes
    becomes unknown. Never raises; a lost write leaves the route unknown.
    """
    try:
        key = image_route_scope({"provider": provider, "base_url": base_url})
        if not key:
            return
        statements: Dict[str, bool] = {}
        for row in rows if isinstance(rows, list) else []:
            model = str(row.get("id") or row.get("name") or "") if isinstance(row, dict) else ""
            verdict = image_input_from_row(row)
            if model and verdict is not None:
                statements[model] = verdict
        from ouroboros import capability_evidence as evidence

        evidence._store_evidence(evidence.canonical_evidence_root(), EVIDENCE_NAMESPACE, key, {
            "source": source, "observed_at": utc_now_iso(), "models": statements,
        })
    except Exception:
        log.debug("image-input evidence write failed", exc_info=True)


@dataclass(frozen=True)
class ImageInputEvidence:
    """One route's image-input fact: ``verdict`` True, False or None (unknown)."""

    verdict: Optional[bool] = None
    source: str = ""
    observed_at: str = ""


def route_image_input(model: str) -> ImageInputEvidence:
    """The fresh catalog statement for the exact API route ``model`` resolves to, else unknown.

    Reads the shared store on every call and never fetches, so a worker or a
    cold VLM child sees what any process recorded under the canonical root.
    Claudexor routes answer from their account catalog instead; a local lane has
    no catalog.
    """
    name = str(model or "").strip()
    if not name or provider_for_model(name) in {"local", "claudexor"}:
        return ImageInputEvidence()
    try:
        from ouroboros import capability_evidence as evidence
        from ouroboros.llm import LLMClient

        target = LLMClient()._resolve_remote_target(name)
        key = image_route_scope(target)
        records = evidence._load(evidence.canonical_evidence_root()).get(EVIDENCE_NAMESPACE) or {}
        record = records.get(key) if key else None
        if not isinstance(record, dict):
            return ImageInputEvidence()
        verdict = (record.get("models") or {}).get(str(target.get("resolved_model") or ""))
        observed_at = str(record.get("observed_at") or "")
        if not isinstance(verdict, bool) or evidence._age_seconds(observed_at) > _EVIDENCE_FRESH_SEC:
            return ImageInputEvidence()
        return ImageInputEvidence(verdict, str(record.get("source") or ""), observed_at)
    except Exception:
        log.debug("image-input evidence read failed; the route stays unknown", exc_info=True)
        return ImageInputEvidence()


def own_lane_without_images(model: str, *, use_local: bool = False) -> str:
    """Our own lane that cannot carry image bytes, named; "" when the transport carries them.

    Decided by the resolved lane (``use_local``, our own provider namespace), never
    by what a model is called: the limit is ours, not evidence about the model.
    """
    return _OWN_LANES_WITHOUT_IMAGES.get("local" if use_local else provider_for_model(model), "")


def _metadata_says_no(model: str) -> str:
    fact = route_image_input(model)
    if fact.verdict is False and fact.source:
        return f"{fact.source}, observed {fact.observed_at[:10]} UTC, lists no image input for this model"
    if provider_for_model(model) == "claudexor":
        return "the Claudexor model catalog of this account lists no image input for this model"
    return "this route's model metadata lists no image input for this model"


def _image_input_verdict(model: str, **binding: Any) -> Optional[bool]:
    """``supports_vision`` where an unreadable fact is an unknown one.

    Waits and interruptions keep their owner (``propagate_model_error``); any
    other failure to read the route's evidence is not a "no".
    """
    try:
        return supports_vision(model, **binding)
    except Exception as error:
        from ouroboros.llm_claudexor import propagate_model_error

        propagate_model_error(error)
        log.debug("image-input evidence unreadable for %s; the route stays unknown", model, exc_info=True)
        return None


def _unique(models: Iterable[Any], seen: set) -> List[str]:
    texts = [str(model or "").strip() for model in models]
    return [text for text in texts if text and not (text in seen or seen.add(text))]  # order kept, first wins


def choose_image_model(explicit: Iterable[Any], automatic: Iterable[Any], *,
                       refused: Optional[Callable[[str], str]] = None) -> Tuple[str, List[Tuple[str, str]]]:
    """Pick the model that receives an image for a VLM or caption call.

    Returns ``(model, passed_over)``: ``model`` is "" when nothing can take the
    image, and ``passed_over`` pairs each skipped candidate with why. An explicit
    model (an owner-switched vision route, ``vlm_query model=``, the vision slot)
    is called even when its metadata says no; only our own lane's limit, or its
    route having refused this very image earlier in the task (``refused`` names
    why), skips it. Automatic candidates take a confirmed yes first, then an
    unknown; a confirmed no and a route that refused this image are skipped.
    """
    passed_over: List[Tuple[str, str]] = []
    seen: set = set()
    for model in _unique(explicit, seen):
        lane = own_lane_without_images(model)
        why = f"our {lane} transport lane cannot carry images" if lane else (refused(model) if refused else "")
        if not why:
            return model, passed_over
        passed_over.append((model, why))
    unknown = ""
    for model in _unique(automatic, seen):
        lane = own_lane_without_images(model)
        why = f"our {lane} transport lane cannot carry images" if lane else (refused(model) if refused else "")
        if why:
            passed_over.append((model, why))
            continue
        verdict = _image_input_verdict(model, model_role="vision")
        if verdict is True:
            return model, passed_over
        if verdict is None:
            unknown = unknown or model
        else:
            passed_over.append((model, ""))
    return unknown, passed_over


def describe_passed_over(passed_over: Iterable[Tuple[str, str]]) -> str:
    """Why each candidate was passed over, with the metadata's source and date."""
    return "; ".join(f"{model} ({why or _metadata_says_no(model)})" for model, why in passed_over)


def resolve_vision_caption_model(ctx: Any, llm: Any, *, use_local: bool = False,
                                 refused: Optional[Callable[[str], str]] = None) -> str:
    """The model that captions an image for a route that will not receive it, or "".

    An owner-switched vision route and the explicit vision slot are used even when
    their metadata says no, unless their route refused this image earlier in the
    task (``refused``); automatic candidates prefer a confirmed yes, then an
    unknown, and skip a confirmed no, such a route and our own lanes. A local Main
    route captions only through an explicit vision slot, so it starts no hidden
    remote image work.
    """
    from ouroboros.model_wait import current_model_wait

    wait = current_model_wait()
    override = wait.overrides.get("vision") if wait is not None else None
    if override:
        return "" if override.get("use_local") else choose_image_model([override["model"]], (), refused=refused)[0]
    explicit = str(runtime_setting("OUROBOROS_MODEL_VISION", "") or "").strip()
    if use_local and not explicit:
        return ""
    from ouroboros.model_slots import local_lane_label, slot_lane_label

    # Each candidate as it routes: a slot on our local lane is passed over by name.
    automatic = [
        slot_lane_label("vision", get_vision_model()),
        local_lane_label(getattr(ctx, "model", ""), bool(getattr(ctx, "use_local", False))),
        local_lane_label(getattr(ctx, "active_model", "") or getattr(ctx, "task_model_override", ""),
                         bool(getattr(ctx, "active_use_local", False))),
    ]
    try:
        from ouroboros.config import get_light_model, parse_fallback_chain

        automatic.append(slot_lane_label("light", get_light_model()))
        automatic.extend(slot_lane_label("fallback", model) for model in parse_fallback_chain())
    except Exception:
        pass
    try:
        automatic.append(slot_lane_label("main", llm.default_model()))
    except Exception:
        pass
    return choose_image_model([explicit], automatic, refused=refused)[0]


# --- The projection policy ---------------------------------------------------

def _image_url_from_block(block: Dict[str, Any]) -> str:
    """The image a block carries as one URL, in any shape a send copy or wire payload uses.

    Chat (``image_url``), Responses (``input_image``) and Anthropic ``source``
    blocks; an Anthropic base64 source reads back as the data URL it was split from.
    """
    image_url = block.get("image_url")
    if isinstance(image_url, dict):
        return str(image_url.get("url") or "")
    if isinstance(image_url, str) and image_url:
        return image_url
    source = block.get("source")
    if isinstance(source, dict):
        if source.get("type") == "base64":
            return f"data:{source.get('media_type') or 'image/png'};base64,{source.get('data') or ''}"
        return str(source.get("url") or "")
    return str(block.get("url") or "")


def _is_image(block: Any) -> bool:
    return isinstance(block, dict) and str(block.get("type") or "") in _IMAGE_TYPES


def _has_image(messages: List[Dict[str, Any]]) -> bool:
    return any(
        isinstance(msg, dict) and isinstance(msg.get("content"), list) and any(map(_is_image, msg["content"]))
        for msg in messages
    )


def _rewrite(messages: List[Dict[str, Any]],
             project: Callable[[Dict[str, Any]], Optional[str]]) -> List[Dict[str, Any]]:
    """Replace each image block for which ``project`` returns text; ``messages`` itself when none.

    The replacement keeps ``_source_path`` so a re-view hint survives; host
    metadata never reaches the wire (``_copy_messages_with_cache_policy``).
    """
    out: Optional[List[Dict[str, Any]]] = None
    for msg_index, msg in enumerate(messages):
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue
        for block_index, block in enumerate(content):
            if not _is_image(block):
                continue
            text = project(block)
            if text is None:
                continue
            if out is None:
                out = copy.deepcopy(messages)
            replacement: Dict[str, Any] = {"type": "text", "text": text}
            if block.get("_source_path"):
                replacement["_source_path"] = block["_source_path"]
            out[msg_index]["content"][block_index] = replacement
    return messages if out is None else out


def _url_digest(url: str) -> str:
    """The image key shared by the caption memo and the refused set: sha256 of the image URL."""
    return sha256(str(url or "").encode("utf-8", errors="replace")).hexdigest() if url else ""


def _image_digest(block: Dict[str, Any]) -> str:
    return _url_digest(_image_url_from_block(block))


# --- Refused images: what a physical candidate carried, the refusal, the task's memory ---

# Task memory of refused images in ``accumulated_usage`` ({route key: {digest: facts}}),
# and the same shape alive only while the one same-round retry runs.
REFUSED_IMAGES_KEY, _PENDING_REFUSALS_KEY = "_vision_refused_images", "_vision_refused_pending"
# A completed structural refusal: the classifier's typed kind at these HTTP statuses.
_REFUSAL_STATUSES = {"bad_request": frozenset({400, 422}), "provider_error": frozenset({404, 415})}
_PHYSICAL_IMAGE_TYPES = _IMAGE_TYPES | {"input_image"}
_CANDIDATE_IMAGES: Dict[str, frozenset] = {}
_CANDIDATE_IMAGES_LOCK = threading.Lock()
_CANDIDATE_IMAGES_KEPT, _REFUSAL_WORDS_MAX = 256, 500


def _physical_image_urls(value: Any, depth: int = 0) -> Iterator[str]:
    if isinstance(value, list) and depth < 8:
        for item in value:
            yield from _physical_image_urls(item, depth + 1)
    elif isinstance(value, dict):
        if str(value.get("type") or "") in _PHYSICAL_IMAGE_TYPES:
            yield _image_url_from_block(value)
        elif isinstance(value.get("content"), list):
            yield from _physical_image_urls(value["content"], depth + 1)


def note_candidate_images(raw_sha256: Optional[str], payload: Any) -> None:
    """Record which images one final physical candidate carries, under its raw sha256.

    Called where the candidate is bound (``llm_attempt._attempt_request``): a pure
    function of the candidate's bytes, read back by its identity alone (a dispatch
    predicate's ``AttemptRequest``, a failed attempt's capture). Bounded; never raises.
    """
    key = str(raw_sha256 or "")
    try:
        with _CANDIDATE_IMAGES_LOCK:
            if key in _CANDIDATE_IMAGES:  # a re-sent candidate (a 5xx series) becomes the newest entry again
                _CANDIDATE_IMAGES[key] = _CANDIDATE_IMAGES.pop(key)
            if not key or key in _CANDIDATE_IMAGES or not isinstance(payload, Mapping):
                return
        found = frozenset(_url_digest(url) for part in (payload.get("messages"), payload.get("input"))
                          for url in _physical_image_urls(part) if url)
        with _CANDIDATE_IMAGES_LOCK:
            _CANDIDATE_IMAGES[key] = found
            while len(_CANDIDATE_IMAGES) > _CANDIDATE_IMAGES_KEPT:
                _CANDIDATE_IMAGES.pop(next(iter(_CANDIDATE_IMAGES)))
    except Exception:
        log.debug("candidate image digests unavailable", exc_info=True)


def candidate_images(raw_sha256: Optional[str]) -> Optional[frozenset]:
    """The image digests a final candidate carried, or None when it was never recorded."""
    with _CANDIDATE_IMAGES_LOCK:
        return _CANDIDATE_IMAGES.get(str(raw_sha256 or ""))


def image_route_key(model: str, role: str = "vision", pin: Optional[str] = None) -> str:
    """The vision route a refusal belongs to: the resolved route's image scope, model and account.

    Another model, provider, endpoint, routing option or account is another route: a fallback
    never inherits a refusal, and equal model strings with different accounts are different
    routes (``model_slots.route_binding``: ``role``'s account unless ``pin`` names one; Auto
    and API routes have none). "" for our local lane.
    """
    name = str(model or "").strip()
    if not name or provider_for_model(name) == "local":
        return ""
    try:
        from ouroboros.llm import LLMClient
        from ouroboros.model_slots import route_binding
        from ouroboros.model_wait import current_model_wait

        target, waiter = LLMClient()._resolve_remote_target(name), current_model_wait()
        chosen = (waiter.overrides.get(role) or {}) if waiter is not None and pin is None else {}
        if chosen.get("model", name) == name and chosen.get("model_account_override") is not None:
            pin = chosen["model_account_override"]  # the owner's live choice binds the role, as the send does
        account = route_binding(name, False, role, overrides=None if pin is None else {
            role: {"model_account_override": pin}})[2]
    except Exception:
        return name
    return "|".join((image_route_scope(target) or str(target.get("provider") or ""), str(target.get("source") or ""),
                     str(target.get("resolved_model") or name), account))


def _route_refusals(accumulated_usage: Any, model: str, role: str = "vision",
                    pin: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    usage = accumulated_usage if isinstance(accumulated_usage, dict) else {}
    records = [record for record in (usage.get(REFUSED_IMAGES_KEY), usage.get(_PENDING_REFUSALS_KEY))
               if isinstance(record, dict) and record]
    route, merged = (image_route_key(model, role, pin) if records else ""), {}
    for record in records:
        merged.update(record.get(route) or {})
    return merged


def refused_images(routing: VisionRoutingContext) -> Dict[str, Dict[str, Any]]:
    """Images this route refused earlier in the task: digest -> the provider's answer.

    Such an image takes its mode's text instead of pixels (Auto and Caption: a caption
    from a route that did not refuse it, else a marker; Inline: a marker)."""
    return _route_refusals(routing.accumulated_usage, routing.model, routing.model_role or "main",
                           routing.model_account_override)


def _refusal_answer(facts: Mapping[str, Any]) -> str:
    """The provider's own answer: status, code and its words verbatim (bounded)."""
    status, code = facts.get("status"), str(facts.get("code") or "").strip()
    head = ", ".join(part for part in (f"HTTP {status}" if status else "",
                                       f"code {code}" if code and code != str(status) else "") if part)
    message = " ".join(str(facts.get("message") or "").split())
    if len(message) > _REFUSAL_WORDS_MAX:
        message = message[:_REFUSAL_WORDS_MAX].rstrip() + "…"
    said = f': "{message}"' if message else ""
    return f"{head or 'no status'}{said}"


def refusal_words(facts: Mapping[str, Any]) -> str:
    return f"this route refused this image earlier in the task ({_refusal_answer(facts)})"


def refusal_check(accumulated_usage: Any, digests: Iterable[str]) -> Optional[Callable[[str], str]]:
    """Why a candidate model may not receive these images (its route refused one), or None."""
    found, usage = [digest for digest in digests if digest], accumulated_usage
    if not found or not isinstance(usage, dict) or not (usage.get(REFUSED_IMAGES_KEY) or usage.get(_PENDING_REFUSALS_KEY)):
        return None

    def refused_by(model: str) -> str:
        refusals = _route_refusals(usage, model)
        facts = next((refusals[digest] for digest in found if digest in refusals), None)
        return refusal_words(facts) if facts is not None else ""

    return refused_by


def record_image_refusal(accumulated_usage: Any, model: str, digests: Iterable[str],
                         facts: Mapping[str, Any]) -> None:
    """Remember for the rest of the task that ``model``'s route refused these images.

    Task memory only, never ``capability_evidence.json``: a refusal observes one request."""
    route, found = image_route_key(model), [digest for digest in digests if digest]
    if route and found and isinstance(accumulated_usage, dict):
        record = accumulated_usage.setdefault(REFUSED_IMAGES_KEY, {}).setdefault(route, {})
        record.update({digest: dict(facts) for digest in found})


def query_image_url(image: Mapping[str, Any]) -> str:
    """The URL ``LLMClient.vision_query`` sends for one image (a ``url``, or ``base64`` + ``mime``)."""
    if "url" in image or "base64" not in image:
        return str(image.get("url") or "")
    return f"data:{image.get('mime', 'image/png')};base64,{image['base64']}"


def query_image_digests(images: Iterable[Any]) -> List[str]:
    """The refused-set keys of a ``vision_query`` call's images: the URLs it sends."""
    return [_url_digest(query_image_url(image)) for image in images if isinstance(image, Mapping)]


def completed_image_refusal(error: BaseException) -> Optional[Dict[str, Any]]:
    """A VLM or caption call's completed 400/422 or 404/415 refusal with the provider's words, else None.

    The Main trigger's typed facts without its 5xx case: an image sub-call runs no
    same-request retry series that could be spent."""
    try:
        from ouroboros.loop_llm_call import classify_llm_exception
        from ouroboros.loop_transport import owner_provider_message

        capture = getattr(error, "physical_attempt_capture", None)
        status = getattr(capture, "provider_status_code", None)
        if (not isinstance(error, Exception) or not isinstance(status, int) or isinstance(status, bool)
                or status not in _REFUSAL_STATUSES.get(classify_llm_exception(error).kind, ())):
            return None
        return {"status": status, "code": str(getattr(capture, "provider_code", "") or ""),
                "message": owner_provider_message(error) or str(getattr(capture, "provider_error", "") or "")}
    except Exception:
        log.debug("image refusal facts unreadable", exc_info=True)
        return None


def image_refusal(accumulated_usage: Mapping[str, Any], failed: Any) -> Optional[Dict[str, Any]]:
    """The facts of a completed refusal of a Main request that carried images, else None.

    From the failed attempt itself: its capture's HTTP status and physical images, and
    the kind the classifier stamped for it. A completed 400/422 (``bad_request``) or
    404/415 (``provider_error``), or a 5xx once the ordinary same-request retries used
    their whole attempt budget (not when a deadline or a refusal stopped them). Never
    auth, quota, rate limits, policy, overflow, a deadline or an unknown outcome (nor
    while the round holds an unresolved repeat); the ledger's money state is no outcome.
    """
    from ouroboros.loop_llm_call import RETRY_ATTEMPTS_SPENT_KEY, RETRY_WALL_EXHAUSTED_KEY, TRANSPORT_DEATHS_KEY

    status, kind = getattr(failed, "provider_status_code", None), str(accumulated_usage.get("_last_llm_error_kind") or "")
    if (not isinstance(status, int) or isinstance(status, bool) or accumulated_usage.get("_pending_transport_outcome")
            or isinstance(accumulated_usage.get(TRANSPORT_DEATHS_KEY), dict)):
        return None
    completed = status in _REFUSAL_STATUSES[kind] if kind in _REFUSAL_STATUSES else (
        kind == "provider_transient" and 500 <= status <= 599
        and accumulated_usage.get(RETRY_WALL_EXHAUSTED_KEY) is True
        and accumulated_usage.get(RETRY_ATTEMPTS_SPENT_KEY) is True)
    digests = candidate_images(getattr(failed, "candidate_raw_sha256", None)) if completed else None
    return {"status": status, "kind": kind, "code": str(getattr(failed, "provider_code", "") or ""),
            "message": str(accumulated_usage.get("_last_llm_provider_message") or getattr(failed, "provider_error", "") or ""),
            "digests": digests} if digests else None


def _caption_for_block(
    block: Dict[str, Any],
    *,
    ctx: Any,
    llm: Any,
    accumulated_usage: Dict[str, Any],
    drive_root: pathlib.Path | None = None,
    task_id: str = "",
    event_queue: Any = None,
) -> Tuple[str, str]:
    """Return (caption, failure), both empty if no route can accept these pixels.

    Each candidate's prepared-byte identity governs its refusal and memo lookup.
    """
    memo = accumulated_usage.setdefault("_vision_caption_memo", {})
    model, url, url_digest, preparation_note, failure = prepare_caption_image(block, ctx, llm, accumulated_usage)
    # Completed captions remain reusable after their generating account is unavailable.
    key = f"{url_digest}|{model}|v1"
    if memo.get(key):
        return str(memo[key]), ""
    if not model or not url:
        return "", failure
    call_id = new_call_id("vision_caption")
    prompt_ref = {}

    def operation(phase: str) -> None:
        emit_cognitive_operation_event(event_queue, task_id=task_id, operation_id=call_id, phase=phase,
                                       kind="vlm", task_attempt=getattr(ctx, "task_attempt", None))

    operation("started")
    # Receipts are BOOKKEEPING and live OUTSIDE the caption-producing try: a
    # persist_call failure used to jump into the failure arm below, REPLACE the
    # paid caption with a failure label and memoize it for the task.
    if drive_root is not None:
        try:
            prompt_ref = persist_call(
                drive_root,
                task_id=task_id,
                call_id=f"{call_id}_request",
                call_type="vision_caption_request",
                payload={"prompt": _CAPTION_PROMPT, "image_url": url, "model": model},
                manifest={"model": model},
            )
        except Exception:
            log.warning("vision caption request receipt failed", exc_info=True)
    try:
        reserve = _vision_finalization_reserve()
        if owner_deadline_exhausted(
            deadline_ts=getattr(ctx, "deadline_ts", None), reserve_sec=reserve,
        ):
            raise TimeoutError("owner deadline leaves no window for a vision caption")
        text, usage = llm.vision_query(
            _CAPTION_PROMPT,
            [{"url": url, "_original_image_url": block.get("_original_image_url") or _image_url_from_block(block)}],
            model=model,
            reasoning_effort=resolve_effort("task"),
            timeout=transport_timeout_with_deadline(
                get_vision_caption_timeout_sec(),
                deadline_ts=getattr(ctx, "deadline_ts", None),
                reserve_sec=reserve,
            ),
            purpose="caption",
        )
    except Exception as exc:
        from ouroboros.llm_claudexor import propagate_model_error
        propagate_model_error(exc)
        operation("failed")
        # NOT memoized: a memoized failure used to block every retry for this
        # image for the rest of the task. A completed refusal only takes this
        # route out of this image's caption candidates.
        model, refusal, digests = refusal_identity(exc, model, lambda _route: [url_digest])
        if refusal is not None:
            record_image_refusal(accumulated_usage, model, digests, {**refusal, "via": "caption", "model": model})
        return "", f"{type(exc).__name__}: {exc}"
    model, key, preparation_note = sent_caption_identity(usage, model, key, preparation_note)
    try:
        from ouroboros.llm import add_usage

        add_usage(accumulated_usage, usage)
    except Exception:
        pass
    try:
        from ouroboros.pricing import emit_llm_usage_event

        cost = float(usage["cost"]) if isinstance(usage, dict) and usage.get("cost") is not None else None
        emit_llm_usage_event(event_queue, task_id, model, usage, cost, category="task", source="vision_caption")
    except Exception:
        pass
    caption = str(text or "").strip()
    if caption and preparation_note:
        caption = preparation_note + " " + caption
    if drive_root is not None:
        try:
            persist_call(
                drive_root,
                task_id=task_id,
                call_id=f"{call_id}_response",
                call_type="vision_caption_response",
                payload={"caption": caption, "usage": usage, "prompt_ref": prompt_ref},
                manifest={"model": model},
            )
        except Exception:
            log.warning("vision caption response receipt failed", exc_info=True)
    operation("finished")
    if not caption:
        return "", f"{model} returned an empty caption"
    if key:
        memo[key] = caption
    return caption, ""


def _usable_existing_caption(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    # Browser/view_image producers use bracketed labels for eviction/re-view hints
    # (e.g. "[browser screenshot ...]" / "[image: file.png]"), not visual captions.
    if text.startswith("[") and text.endswith("]"):
        return ""
    return text


def _caption_or_marker(block: Dict[str, Any], routing: VisionRoutingContext, reason: str,
                       refusal: Optional[Dict[str, Any]] = None) -> str:
    """A caption (or a truthful marker) in place of the image; ``refusal`` also names the route's refusal."""
    existing = _usable_existing_caption(str(block.get("_caption") or ""))
    caption, failure = (existing, "") if existing else _caption_for_block(
        block,
        ctx=routing,
        llm=routing.llm,
        accumulated_usage=routing.accumulated_usage,
        drive_root=routing.drive_root,
        task_id=routing.task_id,
        event_queue=routing.event_queue,
    )
    if refusal is not None:
        refusal["projected"] = "caption" if caption else "marker"
        if caption:
            return f"[{reason}; a caption replaces it] [image caption: {caption}]"
        if failure:
            return f"[image omitted: {reason}; its caption failed: {failure}]"
    if caption:
        return f"[image caption: {caption}]"
    if failure:
        # A failure is reported as one, never presented as a caption.
        return f"[image caption unavailable: {failure}]"
    return f"[image omitted: {reason}; no caption route is available]"


def _lane_marker(lane: str, block: Dict[str, Any]) -> str:
    from ouroboros.llm_messages import own_lane_image_marker

    return own_lane_image_marker(lane, str(block.get("_caption") or "").strip())


def _refused_marker(block: Dict[str, Any], refused: Mapping[str, Dict[str, Any]]) -> Optional[str]:
    """Inline: a refused image becomes a marker quoting the provider; no caption call is made."""
    facts = refused.get(_image_digest(block))
    if facts is None:
        return None
    facts["projected"] = "marker"
    return (f"[image omitted: {refusal_words(facts)}; Inline mode keeps this image out of this "
            "route's requests for the rest of the task]")


def _refused_caption(block: Dict[str, Any], routing: VisionRoutingContext,
                     refused: Mapping[str, Dict[str, Any]]) -> Optional[str]:
    facts = refused.get(_image_digest(block))
    return None if facts is None else _caption_or_marker(block, routing, refusal_words(facts), facts)


def _project(messages: List[Dict[str, Any]], routing: VisionRoutingContext, purpose: str) -> List[Dict[str, Any]]:
    lane = own_lane_without_images(routing.model, use_local=routing.use_local)
    # A VLM or caption call names its model explicitly: Inline semantics, never a caption.
    mode = get_image_input_mode() if purpose == "main" else "inline"
    if mode == "off":
        return _rewrite(messages, lambda _block: _OFF_MARKER)
    # A VLM or caption model was chosen past the routes that refused its image.
    refused = refused_images(routing) if purpose == "main" and not lane else {}
    if mode == "inline":
        if lane:
            return _rewrite(messages, lambda block: _lane_marker(lane, block))
        messages = prepare_route_images(messages, routing.model)
        return _rewrite(messages, lambda block: _refused_marker(block, refused)) if refused else messages
    if mode == "auto" and not lane:
        verdict = _image_input_verdict(routing.model, model_role=routing.model_role,
                                       model_account_override=routing.model_account_override)
        if verdict is not False:
            messages = prepare_route_images(messages, routing.model)
            return _rewrite(messages, lambda block: _refused_caption(block, routing, refused)) if refused else messages
        reason = _metadata_says_no(routing.model)
    elif lane:
        reason = f"our {lane} transport lane cannot carry images"
    else:
        reason = "Caption mode sends a caption instead of the image"
    return _rewrite(messages, lambda block: _caption_or_marker(block, routing, reason))


def withhold_images(messages: List[Dict[str, Any]], error: BaseException, *,
                    purpose: str = "main") -> List[Dict[str, Any]]:
    """The safe projection after the projection itself failed.

    Unknown evidence may send pixels; a failure to carry out the owner's mode may
    not, except where the mode sends pixels whatever the evidence (Inline, or an
    explicitly named VLM/caption model, whose own lane still marks its images).
    The marker discloses the failure.
    """
    mode = get_image_input_mode() if purpose == "main" else "inline"
    if mode == "inline":
        return messages
    marker = (f"[image omitted: image preparation failed ({type(error).__name__}); "
              f"not sent under the {mode} image mode]")
    return _rewrite(messages, lambda _block: marker)


def prepare_messages_for_send(
    messages: List[Dict[str, Any]],
    *,
    routing: VisionRoutingContext,
    purpose: str = "main",
) -> List[Dict[str, Any]]:
    """Project image blocks for one send; the canonical transcript is never changed.

    Returns ``messages`` itself when every image goes as pixels. ``purpose`` is
    "main" (the owner's image mode applies), "vlm" or "caption" (an explicitly
    named model: pixels, never a caption, so a caption cannot recurse).
    """
    if not _has_image(messages):
        return messages
    try:
        return _project(messages, routing, purpose)
    except Exception as error:
        from ouroboros.llm_claudexor import propagate_model_error

        propagate_model_error(error)
        log.warning("image projection failed; images withheld per the image mode", exc_info=True)
        return withhold_images(messages, error, purpose=purpose)


# --- The one same-round retry after a Main image refusal ------------------------

# The round's error stamps a retry that never left the host gives back unchanged.
_ROUND_ERROR_KEYS = ("_last_llm_error", "_last_llm_error_kind", "_last_llm_retry_same_request", "_last_llm_status_code",
                     "_last_llm_provider_code", "_last_llm_provider_message", "_last_llm_provider_fields",
                     "_last_llm_provider_message_cut", "_last_llm_resource_refusal", "_last_llm_reset_at",
                     "execution_status", "reason_code")


def _round_capture(ctx: Any, failed: Any) -> bool:
    """Whether ``failed`` is this round's Main send, not a caption or helper call made in it."""
    context = getattr(failed, "physical_context", None)
    if context is not None:
        execution_id = str(ctx.accumulated_usage.get("execution_id") or "")
        return bool(execution_id) and context.round_id == f"{execution_id}:round:{ctx.round_idx}"
    try:
        from ouroboros.llm import LLMClient

        target = LLMClient()._resolve_remote_target(str(ctx.active_model or ""))
    except Exception:
        return False
    return str(getattr(failed, "model", "") or "") == str(target.get("usage_model") or target.get("resolved_model") or "")


def _without_refused_images(failed: Any, refused: frozenset) -> Callable[[Any], bool]:
    """Admit only the retry this refusal earned: no refused image in the physical candidate, the same
    model, route and account, round, and at most its reply allowance (a wait onto another route is not it)."""
    def predicate(request: Any) -> bool:
        sent = candidate_images(getattr(request, "candidate_raw_sha256", None))
        context, failed_context = request.physical_context, failed.physical_context
        same_round = (context is None and failed_context is None) or (
            context is not None and failed_context is not None
            and context.route_fp == failed_context.route_fp and context.round_id == failed_context.round_id)
        return bool(sent is not None and not sent & refused and same_round
                    and request.provider == failed.provider and request.model == failed.model
                    and request.max_completion_tokens <= failed.max_completion_tokens)  # the failed allowance caps it

    return predicate


def _disclose_refusal_retry(ctx: Any, facts: Mapping[str, Any], count: int, mode: str, outcome: str,
                            pending: Mapping[str, Mapping[str, Any]], retry: Optional[Dict[str, Any]]) -> None:
    from ouroboros import loop

    replaced = sorted({str(item.get("projected") or "") for item in pending.values()} - {""})
    loop._emit_checkpoint_event(ctx.event_queue, ctx.task_id, ctx.drive_logs, {
        "checkpoint_kind": "image_refusal_retry", "round": ctx.round_idx, "model": facts.get("model"),
        "image_mode": mode, "error_kind": facts.get("kind"), "status_code": facts.get("status"),
        "provider_code": facts.get("code"), "provider_message": str(facts.get("message") or "")[:_REFUSAL_WORDS_MAX],
        "refused_images": count, "outcome": outcome, "replaced_by": replaced, **({"retry_error": retry} if retry else {}),
    })
    emit = getattr(ctx, "emit_progress", None)
    if callable(emit) and outcome != "not_sent":
        head = f"🖼 {facts.get('model')} refused a request with an image ({_refusal_answer(facts)})"
        emit(f"{head}; this round was retried with {'a caption' if 'caption' in replaced else 'a note'} in its "
             "place, and that image stays out of this route's requests for the rest of the task." if outcome == "answered"
             else f"{head}; the retry without it failed too ({(retry or {}).get('error_kind') or 'an error'}).")


def retry_refused_image_round(ctx: Any, failed: Any) -> Optional[Tuple[Any, Any]]:
    """The one same-round retry after the route refused a Main request that carried images.

    ``ctx`` is the round's call context, ``failed`` the capture taken right after its
    dispatch. None when that is no such refusal (``image_refusal``) or the retry could
    not leave the host: the round then recovers exactly as before. Otherwise the retry's
    ``(message, cost)``. Auto and Inline only: Caption and Off never send pixels.
    The refused images are task memory, ephemeral while the retry runs: the projection
    replaces only them (Auto: a caption from a route that did not refuse them, else a
    marker; Inline: a marker) and quotes the provider. The retry re-measures the fit and
    dispatches one semantic attempt that ``_without_refused_images`` admits (request-wire
    recovery may still add physical attempts). Success keeps the memory under the route's
    key: the same model goes on, without a fallback route, a cooldown or any
    ``capability_evidence.json`` write. Failure keeps both errors and learns nothing.
    """
    usage = ctx.accumulated_usage
    mode = get_image_input_mode()
    refusal = image_refusal(usage, failed) if mode in ("auto", "inline") else None
    if refusal is None or not _round_capture(ctx, failed):
        return None
    from ouroboros import loop
    from ouroboros.loop_llm_call import RETRY_ATTEMPTS_SPENT_KEY, RETRY_WALL_EXHAUSTED_KEY
    from ouroboros.usage_accounting import PhysicalAttemptPreconditionFailed

    from ouroboros.model_slots import task_model_binding
    from ouroboros.model_wait import current_model_wait

    digests, waiter = refusal.pop("digests"), current_model_wait()  # the round's own role and account binding:
    role, pin = task_model_binding({"model_role": getattr(ctx, "model_role", ""), "task_metadata": getattr(  # as sent
        ctx.tools._ctx, "task_metadata", {})}, context_fit_plan=getattr(ctx, "context_fit_plan", None) or getattr(
        ctx.tools._ctx, "context_fit_plan", None), overrides=waiter.overrides if waiter else None)
    route, facts = image_route_key(ctx.active_model, role, pin), {**refusal, "via": "main", "model": str(ctx.active_model or "")}
    keys = (*_ROUND_ERROR_KEYS, RETRY_WALL_EXHAUSTED_KEY, RETRY_ATTEMPTS_SPENT_KEY)
    first = {key: usage[key] for key in keys if key in usage}  # the first error, before the retry overwrites it
    usage[_PENDING_REFUSALS_KEY] = {route: {digest: dict(facts) for digest in digests}}
    sent, retry = True, None
    try:
        fit = loop._measure_round_main_fit(ctx, automatic_pass_used=False)
        if fit is not None and (fit.measurement.route_fp, fit.measurement.round_id) in loop._context_reclaim_passes(
                ctx.tools._ctx):  # this round already reclaimed: measure its result, never reclaim again
            fit = loop._measure_after_reclaim(ctx)
        msg, cost = loop._dispatch_round_model(
            ctx, fit, attempt_cap=1, candidate_predicate=_without_refused_images(failed, digests),
            max_tokens=int(getattr(failed, "max_completion_tokens", 0) or 0) or None)
    except PhysicalAttemptPreconditionFailed:
        # A refused image was still there, or a wait moved the round to another route:
        # nothing left the host, so the first error stands as it was.
        msg, cost, sent = None, None, False
        for key in keys:
            usage.pop(key, None)
        usage.update(first)
    finally:
        pending = (usage.pop(_PENDING_REFUSALS_KEY, None) or {}).get(route) or {}
    if msg is not None:
        usage.setdefault(REFUSED_IMAGES_KEY, {}).setdefault(route, {}).update(pending)
    elif sent:
        capture = loop.last_physical_attempt_capture()
        own = capture is not None and getattr(capture, "attempt_id", None) != getattr(failed, "attempt_id", None)
        retry = {"error_kind": str(usage.get("_last_llm_error_kind") or ""),
                 "status_code": getattr(capture, "provider_status_code", None) if own else None,
                 "provider_message": str(usage.get("_last_llm_provider_message") or "")[:_REFUSAL_WORDS_MAX]}
    _disclose_refusal_retry(ctx, facts, len(digests), mode, "answered" if msg is not None else "failed" if sent
                            else "not_sent", pending, retry)
    return (msg, cost) if sent else None
