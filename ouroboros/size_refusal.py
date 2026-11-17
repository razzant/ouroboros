"""Numeric evidence layered on the upstream provider-answer reader and size recognition.

``_usage_response.provider_answer`` owns structural traversal; ``context_budget.size_refusal`` owns
recognition (final status, typed code, frozen wording, 413). FORMS only extract numbers after that
recognition; an unsupported sentence adds no recognition. Numeric evidence additionally excludes
output/body-only bounds and unrelated typed causes: they cannot teach a token window.
``meaning`` distinguishes total, input_plus_reply, input and unknown. Units stay explicit; a unitless
number never becomes a token window. No IO: ``window_learning`` owns recovery and ``context_window`` storage.

Recognition accepts client statuses 400/413/422 or an answer without a status; the local lane
reads its own server at any status. It uses a typed code on any layer, HTTP 413, a cited number
equal to a token limit declared for the route (``context_anchor.cited_limit_refusal``), or frozen
wording (``context_budget.size_refusal``). No phrase is added: other words go to minds with their
speaker (``failed_attempt_facts``). These are upstream's recognition rules, not extra readers.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

#: The usage key the failed call's parsed refusal is stamped under (``stamp``), cleared by the next failure.
USAGE_KEY = "_last_llm_size_refusal"
#: llama.cpp's typed size refusal (``ERROR_TYPE_EXCEED_CONTEXT_SIZE``, server-common.cpp).
LLAMA_CPP_TYPE = "exceed_context_size_error"
_N = r"([0-9][0-9,]*)"


@dataclass(frozen=True)
class Form:
    id: str
    pattern: str
    groups: Tuple[str, ...]  # what each captured number is: window | requested | reply | total
    meaning: str
    units: str = "tokens"


#: The sentences that name a size refusal, newest servers first. Order matters only where two forms overlap.
FORMS: Tuple[Form, ...] = (
    # llama.cpp b12421 / v1.0.0, server-context.cpp:4016 (stand body d-over-context) and :4005 (the !can_split branch).
    Form("llama.cpp/request-exceeds", rf"request \({_N} tokens\) exceeds (?:the )?available context size \({_N} tokens\)",
         ("requested", "window"), "total"),
    Form("llama.cpp/input-larger", rf"input \({_N} tokens\) is larger than the max context size \({_N} tokens\)",
         ("requested", "window"), "total"),
    # vLLM 919d input_processor.py:588 (prompt == window: the one output token does not fit) and :569.
    Form("vllm/decoder-plus-output", rf"decoder prompt \(length {_N}\) plus the number of requested output tokens"
         rf"[^.]*? is longer than the maximum model length of {_N}", ("requested", "window"), "input_plus_reply"),
    Form("vllm/decoder", rf"decoder prompt \(length {_N}\) is longer than the maximum model length of {_N}",
         ("requested", "window"), "total"),
    # vLLM 919d renderers/params.py:542 ("at least" is a lower bound of the input).
    Form("vllm/renderer", rf"maximum context length is {_N} tokens\. However, your prompt contains (?:at least )?{_N} "
         r"input tokens", ("window", "requested"), "total"),
    # vLLM v0.11.2 serving_engine.py:968 (input plus the reserved reply) and :960 (input alone).
    Form("vllm/max-tokens-beside-input", rf"is too large: {_N}\. This model's maximum context length is {_N} tokens "
         rf"and your request has {_N} input tokens", ("reply", "window", "requested"), "input_plus_reply"),
    Form("vllm/request-has-input", rf"maximum context length is {_N} tokens\. However, your request has {_N} input tokens",
         ("window", "requested"), "total"),
    # Anthropic (host regexes; fixture test_context_overflow_hint.py).
    Form("anthropic/input-plus-max-tokens", rf"input length and `?max_tokens`? exceed context limit: {_N} \+ {_N} > {_N}",
         ("requested", "reply", "window"), "input_plus_reply"),
    Form("anthropic/prompt-too-long", rf"prompt is too long: {_N} tokens > {_N} maximum", ("requested", "window"), "input"),
    # LM Studio: 0.4.x error stream (W2a §4, #2322) and 0.3.13 (#546).
    Form("lmstudio/input-does-not-fit", rf"input has {_N} tokens, but the context length only supports {_N} tokens",
         ("requested", "window"), "total"),
    Form("lmstudio/loaded-only", rf"loaded with context length of only {_N} tokens", ("window",), "total"),
    # Alibaba Model Studio through OpenRouter (#3518): the bound names no unit; its documentation ties it to input.
    Form("alibaba/input-range", rf"range of input length should be \[1, {_N}\]", ("window",), "input", units="unstated"),
    # Numberless forms: Ollama llama_server.go:2652 (truncate=false), LM Studio MLX (#2339).
    Form("ollama/prompt-longer", r"the prompt is longer than the context length currently available to the model", (),
         "unknown"),
    Form("lmstudio/mlx-exceeded", r"^context size has been exceeded", (), "unknown"),
)
_COMPILED = tuple((form, re.compile(form.pattern, re.I)) for form in FORMS)


@dataclass(frozen=True)
class SizeRefusal:
    window: int = 0      # the stated window in ``units``; 0 = unstated
    requested: int = 0   # the refused input as the server counted it; 0 = unstated
    reply: int = 0       # the reply the refused request reserved, where stated
    requested_total: int = 0  # input plus reply when only their total was stated
    meaning: str = "unknown"
    source: str = ""     # "structured:llama.cpp", "text:<form id>", "text:shared-vocabulary"
    units: str = "tokens"
    gateway: str = ""    # the gateway that carried an upstream provider's refusal (OpenRouter's provider_name)
    upstream: str = ""   # the sentence the numbers were read from where the host's owner message lacks it: a gateway's
                         # nested upstream error (#3518), or an error given as a bare string (LM Studio, Ollama native)

    @property
    def has_numbers(self) -> bool:
        return any((self.window, self.requested, self.reply, self.requested_total))

    @property
    def token_window(self) -> int:
        """The window a plan may be sized by: a number in tokens, else 0."""
        return self.window if self.units == "tokens" else 0

    @property
    def store_kind(self) -> str:
        """The window record's kind (``context_window.WindowLimit``)."""
        return "input" if self.meaning == "input" else "total"

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


def _int(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0 else 0


def excludes_size(status: Optional[int], answer: Any, message: str = "", *, codes=()) -> bool:
    """Typed non-context causes at any wrapper outrank a size quotation in its diagnostic."""
    from ouroboros.context_budget import CONTEXT_OVERFLOW_CODES, output_or_body_size_message
    from ouroboros.loop_llm_call import _RETRYABLE_PROVIDER_CODES, _provider_code_kind

    layers = [obj.value for obj in answer.objects]
    values = list(codes) + [layer.get(key) for layer in layers for key in ("code", "type", "limit_source")]
    size_codes = CONTEXT_OVERFLOW_CODES | {LLAMA_CPP_TYPE}
    typed_context = any(str(value or "").lower() in size_codes for value in values)
    generic = {"400", "badrequest", "badrequesterror", "invalidrequest", "invalidrequesterror", "error"}
    # A gateway's precise cause (e.g. in-flight funds) explains its generic HTTP code (402).
    for value in sorted(values, key=lambda value: str(value or "").isdigit()):
        code = str(value or "").strip().lower()
        if not code or code in size_codes or code.replace("_", "").replace("-", "").replace(" ", "") in generic:
            continue
        if code in _RETRYABLE_PROVIDER_CODES or code == "429":
            return True
        if code in {"authentication_error", "authorization_error", "permission_error", "permission_denied", "unauthenticated"}:
            return True
        if code in {"quota_exceeded", "quota_exhausted"}:
            return True
        if code in {"content_filter", "content_policy_violation", "policy_violation", "safety_violation"}:
            return True
        if "schema" in code or code in {"validation_error", "invalid_parameter", "invalid_parameters"}:
            return True
        if code in {"413", "request_too_large", "payload_too_large", "request_entity_too_large"}:
            return True
        kind = _provider_code_kind(code)
        if kind:
            return True
    if status in {401, 402, 403, 429}:
        return True
    texts = [message] + [str(layer.get(key) or "") for layer in layers for key in ("message", "error")
                         if not isinstance(layer.get(key), dict)]
    for text in texts:
        low = text.lower()
        # HTTP-body limits never become a token-window fact, even alongside an input+reply quotation.
        if "body too large" in low or "payload too large" in low or "request entity too large" in low:
            return True
        if typed_context:
            continue  # a typed context refusal outranks a mere output-wording heuristic
        form = form_numbers(text)
        output_text = low.replace("even without any output tokens", "") if form and form.source == "text:vllm/renderer" else text
        if output_or_body_size_message(output_text) and not (form and form.meaning == "input_plus_reply"):
            return True
        if any(pattern.search(text) and f.meaning == "input_plus_reply" for f, pattern in _COMPILED) and form is None:
            return True
    return False


def _structured(error: Dict[str, Any]) -> Optional[SizeRefusal]:
    if str(error.get("type") or "").strip().lower() != LLAMA_CPP_TYPE:
        return None
    window, requested = _int(error.get("n_ctx")), _int(error.get("n_prompt_tokens"))
    fallback = _from_text(str(error.get("message") or "")) if not (window and requested) else None
    return SizeRefusal(window=window or (fallback.token_window if fallback else 0),
                       requested=requested or (fallback.requested if fallback else 0),
                       meaning="total", source="structured:llama.cpp")


def form_numbers(text: str) -> Optional[SizeRefusal]:
    """The refusal one of ``FORMS`` states in ``text`` (the shared vocabulary's own patterns are not consulted)."""
    text = " ".join(str(text or "").split())
    for form, pattern in _COMPILED if text else ():
        match = pattern.search(text)
        if match is None:
            continue
        numbers = {name: int(value.replace(",", "")) for name, value in zip(form.groups, match.groups())}
        if (form.meaning == "input_plus_reply" and form.id != "vllm/max-tokens-beside-input"
                and numbers.get("reply", 0) >= numbers.get("window", 0) > 0):
            return None  # output-only forms; vLLM explicitly states the input + reply equation
        return SizeRefusal(window=numbers.get("window", 0), requested=numbers.get("requested", 0),
                           reply=numbers.get("reply", 0), meaning=form.meaning, source=f"text:{form.id}",
                           units=form.units)
    return None


def _from_text(text: str) -> Optional[SizeRefusal]:
    text = " ".join(str(text or "").split())
    if not text:
        return None
    if (found := form_numbers(text)) is not None or any(pattern.search(text) for _form, pattern in _COMPILED):
        return found  # a form that matched but names the reply alone stays None
    from ouroboros.capability_evidence import _parse_ctx_requested
    from ouroboros.context_budget import _parse_ctx_limit_number, context_overflow_message

    if not context_overflow_message(text):
        return None
    low = text.lower()
    requested, reply, total = _parse_ctx_requested(text)
    input_only = any(marker in low for marker in ("prompt is too long", "maximum input", "input token count"))
    return SizeRefusal(window=_parse_ctx_limit_number(text), requested=requested, reply=reply, requested_total=total,
                       meaning="input" if input_only else "total", source="text:shared-vocabulary")


def parse(status: Optional[int], body: Any, message: str = "") -> Optional[SizeRefusal]:
    """Normalize one answer through the same recognition used by every consumer."""
    from ouroboros._usage_response import provider_answer

    answer = provider_answer(body)
    return from_answer(answer, status, message)


def from_answer(answer, status, message="", typed=()):
    """Extract numeric evidence from an already-read answer; never extend recognition."""
    from dataclasses import replace

    from ouroboros.context_budget import size_refusal

    recognized = size_refusal(answer, status, typed, printed=message)
    if recognized is None or excludes_size(status, answer, message, codes=typed):
        return None
    basis, code, speaker, sentence = recognized
    if recognized == ("frozen", "", "", ""):
        # Printed-only recognition (e.g. vLLM's output wording): numbers still require
        # one unambiguous current provider sentence, never the exception display.
        current = [(who, text) for who, text, history in answer.said if not history]
        if len(current) == 1:
            speaker, sentence = current[0]
    found = next((fact for layer in answer.recognized_objects(speaker, sentence, code)
                  if (fact := _structured(layer)) is not None), None)
    # Printed exceptions can identify a refusal, but are not provider numeric evidence.
    found = found or _from_text(sentence)
    if found is None:
        found = SizeRefusal(units="unstated" if basis == "status_413" else "tokens", source=basis)
    return replace(found, gateway=speaker, upstream=sentence)


def from_exception(exc: Any) -> Optional[SizeRefusal]:
    """``parse`` over a provider exception as the host's classifier reads it (status, structured body)."""
    from ouroboros._usage_response import provider_answer, typed_values
    from ouroboros.context_budget import LocalContextTooLargeError
    from ouroboros.loop_llm_call import _exception_status_code

    answer = provider_answer(exc)
    status = None if isinstance(exc, LocalContextTooLargeError) else _exception_status_code(exc)
    return from_answer(answer, status, message=repr(exc), typed=typed_values(exc, answer))


def stamp(usage: Dict[str, Any], exc: Any) -> Optional[SizeRefusal]:
    """Replace the failed call's refusal fact in ``usage`` (cleared when this failure states none)."""
    usage.pop(USAGE_KEY, None)
    found = from_exception(exc)
    if found is not None:
        usage[USAGE_KEY] = found.to_json()
    return found


def stamp_failure(usage: Dict[str, Any], exc: Any, classification=None) -> Optional[SizeRefusal]:
    """Stamp the owner quotation and numeric evidence from the same failed attempt."""
    from ouroboros.loop_transport import stamp_owner_provider_message

    stamp_owner_provider_message(usage, exc, classification)
    return stamp(usage, exc)


def stamped(usage: Any) -> Optional[SizeRefusal]:
    row = usage.get(USAGE_KEY) if isinstance(usage, dict) else None
    try:
        return SizeRefusal(**row) if isinstance(row, dict) else None
    except TypeError:
        return None


def for_usage(usage: Any) -> SizeRefusal:
    """One normalized fact for recovery and storage; old checkpoints without a stamp use legacy text."""
    answer = (usage or {}).get("_last_llm_provider_answer") or {}
    found = stamped(usage)
    if found is None and isinstance(answer.get("size"), dict):
        found = SizeRefusal(**answer["size"])
    if found is not None and found.has_numbers:
        return found
    if answer.get("basis") in {"status_413", "number_equals_declared"}:
        return SizeRefusal(window=int(answer.get("limit") or 0), units="unstated",
                           meaning=str(answer.get("limit_kind") or "unknown"), source=str(answer["basis"]))
    return found or _from_text(refusal_sentence(usage or {})) or SizeRefusal()


def refusal_measure(usage: Dict[str, Any], reply_allowance: int) -> Tuple[Optional[int], int, int]:
    """The stated limit, reserved reply and provider-counted input for the recovery pass."""
    fact = for_usage(usage)
    cited = usage.get("_last_llm_provider_answer") or {}
    stated = fact.token_window or cited.get("limit") or None
    requested, reply, total = (fact.requested, fact.reply, fact.requested_total) if stated else (0, 0, 0)
    reply = 0 if cited.get("limit_kind") == "input" else reply or reply_allowance
    return stated, reply, requested or max(0, total - reply)


def refusal_sentence(usage: Dict[str, Any]) -> str:
    """The sentence the refusal's numbers are read from: the provider's own words (``_last_llm_provider_message``),
    or, when a gateway wrapped them, the upstream provider's sentence its stamp carries."""
    found = stamped(usage)
    if found is not None and found.upstream:
        return found.upstream
    return str((usage or {}).get("_last_llm_provider_message") or "")
