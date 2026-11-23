"""Numeric evidence layered on the upstream provider-answer reader and size recognition.

``_usage_response.provider_answer`` owns structural traversal; ``context_budget.size_refusal`` owns
recognition. FORMS only extract numbers after that recognition; an unsupported sentence adds no recognition.
One object is the refusal: the holder of a typed size code, else the object that said the recognised sentence.
Its subtree on the current path alone supplies structured counts, the sentence numbers are read from, and the
causes that exclude numbers: a code of another kind (the classifier's, ``OTHER_CAUSE_CODES``), an output or body
limit. An earlier upstream attempt, a sibling diagnostic and the exception's printed form supply none of these, and
recognised through them alone the refusal does not supply whole-request evidence for signed Chat (``own``).
A refusal the Z.ai 1261 pair typed (``paired_size_refusal``, #3858) reads its numbers from that one object alone
and sets ``retry_only`` when no token window is stated; Z.ai's 1214 parameter refusal (``cited_size_refusal``)
is no size refusal when size wording belongs only to what it cites or to an earlier attempt.
``meaning`` distinguishes total, input_plus_reply, input and unknown. Units stay explicit; a unitless
number never becomes a token window. No IO: ``window_learning`` owns recovery and ``context_window`` storage.
``send_counts`` and ``response_counts`` lay one refused send's two calculations side by side for Logs and the next
request (``counts``); they read the same answer again for display and teach nothing.
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, Optional, Tuple

#: The usage key the failed call's parsed refusal is stamped under (``stamp``), cleared by the next failure.
USAGE_KEY = "_last_llm_size_refusal"
#: llama.cpp's typed size refusal (``ERROR_TYPE_EXCEED_CONTEXT_SIZE``, server-common.cpp).
LLAMA_CPP_TYPE = "exceed_context_size_error"
#: Codes of another cause that the classifier's vocabulary (``loop_llm_call._NON_RETRYABLE_PROVIDER_MARKERS``) does
#: not name. An addition for numbers only: a refusal carrying one states no window; its retry class is unchanged.
OTHER_CAUSE_CODES = ("authentication_error", "authorization_error", "permission_error", "permission_denied",
                     "unauthenticated", "quota_exceeded", "content_filter", "content_policy_violation", "policy_violation",
                     "safety_violation", "invalid_schema", "validation_error", "invalid_parameter", "invalid_parameters",
                     "413", "payload_too_large", "request_entity_too_large")
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
    # Numberless forms: Ollama llama_server.go:2652 (truncate=false), LM Studio MLX (#2339), Z.ai (#3858).
    Form("ollama/prompt-longer", r"the prompt is longer than the context length currently available to the model", (),
         "unknown"),
    Form("lmstudio/mlx-exceeded", r"^context size has been exceeded", (), "unknown"),
    # Z.ai (#3858): HTTP 400, error code "1261", the whole message "Prompt too long" (live refusal 2026-11-20,
    # logs llm_912c35cf3f7c474688b85055e0d4fb0c) — no token numbers, so it teaches no window. Anchored to the whole
    # sentence: a wording that goes on to state a limit ("Prompt too long; maximum context length is 80000
    # tokens") falls through to the shared vocabulary, whose limit parser keeps the number. Recognition is the
    # code+message pair on one current, unquoted object (``paired_size_refusal``), never the words alone.
    Form("zai/prompt-too-long", r"^prompt too long\.?$", (), "unknown"),
)
_COMPILED = tuple((form, re.compile(form.pattern, re.I)) for form in FORMS)

#: Z.ai's documented 1261 prompt-length refusal (#3858): code "1261" beside "Prompt too long" (or a continuation
#: stating a context limit) on one current, non-inline object.
_ZAI_SHORT_PROMPT = re.compile(r"^prompt too long\.?$", re.I)
_ZAI_PROMPT_PREFIX = re.compile(r"^prompt too long\b", re.I)

#: Z.ai's documented 1214 parameter refusal (#3858): code "1214" beside ``Parameter <field> is invalid`` or
#: ``Invalid parameter`` on a current, non-inline object.
_ZAI_PARAMETER_CODE = "1214"
_ZAI_PARAMETER_MESSAGE = re.compile(r"^(?:parameter\s+\S+\s+is\s+invalid\b|invalid parameter\b)", re.I)

# A whole prompt's affirmative rejection, not the classifier's substring embedded in an example.
_WHOLE_REFUSAL = re.compile(
    r"(?:(?:the |your )?(?:prompt|input|request) (?:is too long|exceeds (?:the )?"
    r"(?:(?:max|maximum) length|context (?:length|limit|window)(?: of this model)?))|"
    r"(?:the |this model's )?(?:maximum )?context (?:length|limit|window) (?:(?:was|has been) )?exceeded)\.?", re.I)
# A closing single quote followed by a word character is internal (e.g. model's or model’s).
_QUOTED_CLAUSE = re.compile(r"""(?<!\w)(?:'(?:[^'\\]|\\.|'(?=\w))*'(?!\w)|"[^"]*"|"""
                            r"(?:‘(?:[^’]|’(?=\w))*’(?!\w))|“[^”]*”|«[^»]*»|‹[^›]*›)"
                            r"|(?<!`)(`+)(?=[^`]*\d)[^`]*\1(?!`)")
# Generic validation errors can name just one field in their own prose, with no structured param.
_INVALID_FIELD = re.compile(r"""^\s*invalid\s+(?:(?:field|parameter)\s+)?"""
                            r"""(?:['"`‘“][\w.\[\]-]+['"`’”]|[\w-]+[.\[][\w.\[\]-]+)\s*:""", re.I)


def zai_parameter_refusal(answer) -> bool:
    """True when a current, non-inline object carries Z.ai's documented 1214 parameter refusal (#3858)."""
    return any(
        not obj.history and not obj.inline
        and str(obj.value.get("code") or "").strip() == _ZAI_PARAMETER_CODE
        and bool(obj.message and _ZAI_PARAMETER_MESSAGE.match(obj.message))
        for obj in answer.objects
    )


def own_size_wording(answer) -> bool:
    """True when any current, non-inline object states a size refusal (1261 pair, typed size code, or frozen wording)."""
    from ouroboros.context_budget import CONTEXT_OVERFLOW_CODES, context_overflow_message

    if paired_size_refusal(answer) is not None:
        return True
    return any(
        not obj.history and not obj.inline
        and (any(code.lower() in CONTEXT_OVERFLOW_CODES for code in obj.codes)
             or any(context_overflow_message(text) for text in obj.sentences))
        for obj in answer.objects
    )


def cited_size_refusal(answer) -> bool:
    """True when Z.ai's 1214 parameter refusal explains the current answer and no current non-inline object states size."""
    return zai_parameter_refusal(answer) and not own_size_wording(answer)


def paired_size_refusal(answer) -> Optional[Tuple[int, str, str, str]]:
    """``(object index, speaker, code, sentence)`` for Z.ai's 1261 + ``Prompt too long`` pair on one current object."""
    from ouroboros.context_budget import context_overflow_message

    for index, obj in enumerate(answer.objects):
        if obj.history or obj.inline:
            continue
        code = str(obj.value.get("code") or "").strip()
        if code == "1261" and obj.message and (
                _ZAI_SHORT_PROMPT.fullmatch(obj.message)
                or (_ZAI_PROMPT_PREFIX.match(obj.message) and context_overflow_message(obj.message))):
            return index, obj.speaker, code, obj.message
    return None


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
    retry_only: bool = False  # numberless Z.ai 1261 pair (#3858): retries this round without storing a window
    # The direct error states this whole-request fact; looser evidence still sizes the round (``_read_numbers``).
    own: bool = False

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


def excludes_size(status: Optional[int], layers) -> bool:
    """A cause of another kind in ``layers`` (the recognised object's subtree; else the current path) outranks a
    size quotation: a code the classifier types otherwise (``loop_llm_call._provider_code_kind``, its retryable
    codes) or one of ``OTHER_CAUSE_CODES``, a body limit, or output wording beside no typed size code."""
    from ouroboros.context_budget import CONTEXT_OVERFLOW_CODES, output_or_body_size_message
    from ouroboros.loop_llm_call import _RETRYABLE_PROVIDER_CODES, _provider_code_kind

    values = [layer.get(key) for layer in layers for key in ("code", "type", "limit_source")]
    size_codes = CONTEXT_OVERFLOW_CODES | {LLAMA_CPP_TYPE}
    typed_context = any(str(value or "").lower() in size_codes for value in values)
    generic = {"400", "badrequest", "badrequesterror", "invalidrequest", "invalidrequesterror", "error"}
    for value in values:
        code = str(value or "").strip().lower()
        if not code or code in size_codes or code.replace("_", "").replace("-", "").replace(" ", "") in generic:
            continue
        if code in _RETRYABLE_PROVIDER_CODES or code == "429" or code in OTHER_CAUSE_CODES or _provider_code_kind(code):
            return True
    if status in {401, 402, 403, 429}:
        return True
    for text in (str(layer.get(key) or "") for layer in layers for key in ("message", "error")
                 if not isinstance(layer.get(key), dict)):
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


def _recognized_subtree(answer, basis: str, code: str, speaker: str, sentence: str):
    """The one recognised object and its subtree (``ProviderAnswer.recognized_subtree``): for a typed basis the
    current-path holder of the code, else the object that said the recognised sentence."""
    holder = next((who for who, value, history in answer.codes if value == code and not history), None)
    if basis == "typed" and holder is not None:
        return answer.recognized_subtree(holder, "", code)
    return answer.recognized_subtree(speaker, sentence, code) if sentence else []


def _own_whole_request(answer, subtree, sentence: str, fact: SizeRefusal, *, paired: bool) -> bool:
    """Require a current error carrier for the fact, then independently test whether it describes the whole send."""
    from ouroboros.context_budget import CONTEXT_OVERFLOW_CODES, INPUT_STRING_LENGTH_CODES

    if not answer.walk_complete or not subtree:
        return False
    for index in subtree:
        obj = answer.objects[index]
        direct = answer.direct_error(index)
        direct_text = direct and (sentence == obj.message or any(text == sentence and (
            field == "" or (field == "detail" and (paired or not obj.message))) for field, text in obj.error_text))
        forward = bool(sentence) and ("raw", sentence) in obj.error_text and answer.direct_error(index, raw_text=True)
        if obj.history or obj.inline or not (direct_text or forward):
            continue
        if fact.source == "structured:llama.cpp" and str(obj.value.get("type") or "").strip().lower() != LLAMA_CPP_TYPE:
            continue  # a numeric diagnostic child cannot speak for its parent
        codes = {code.lower() for code in obj.codes}
        if (codes & INPUT_STRING_LENGTH_CODES or _ZAI_PARAMETER_MESSAGE.match(obj.message)
                or _INVALID_FIELD.match(sentence)):
            return False
        # ``param: messages`` may label a whole Chat refusal, but only when its code or counts say so.
        fields = [obj.value[k] for k in ("param", "parameter", "field", "loc") if obj.value.get(k)]
        typed = bool(codes & (CONTEXT_OVERFLOW_CODES | {LLAMA_CPP_TYPE}))
        if fields and not (fields == ["messages"] and fact.token_window and (typed or (fact.requested and fact.reply))):
            return False
        if fact.token_window and fact.source.startswith("text:") and (plain := _QUOTED_CLAUSE.sub(" ", sentence)) != sentence:
            stated = _from_text(plain)
            if stated is None or (stated.window, stated.meaning, stated.requested, stated.reply, stated.requested_total) != (
                    fact.window, fact.meaning, fact.requested, fact.reply, fact.requested_total):
                return False  # the number belongs to quoted material, though the retry reader may still read it
        return bool(paired or typed or fact.token_window or _WHOLE_REFUSAL.fullmatch(sentence)
                    or any(not form.groups and pattern.match(sentence) for form, pattern in _COMPILED
                           if form.id != "zai/prompt-too-long"))
    return False


def _read_numbers(answer, status, message="", typed=()):
    """``(excluded, fact)`` for one answer: the recognised refusal's numbers, or why it may state none."""
    from dataclasses import replace

    from ouroboros.context_budget import size_refusal

    recognized = size_refusal(answer, status, typed, printed=message)
    if recognized is None:  # excluded only where its own words state numbers the text fallback would read
        stated = any((fact := _from_text(text)) is not None and fact.has_numbers
                     for _who, text, history in answer.said if not history)
        layers = [dict(obj.value, message=obj.message) for obj in answer.objects if not obj.history]
        return stated and excludes_size(status, layers), None
    basis, code, speaker, sentence = recognized
    if recognized == ("frozen", "", "", ""):
        # Printed-only recognition (e.g. vLLM's output wording): numbers still require
        # one unambiguous current provider sentence, never the exception display.
        current = [(who, text) for who, text, history in answer.said if not history]
        if len(current) == 1:
            speaker, sentence = current[0]
    subtree = _recognized_subtree(answer, basis, code, speaker, sentence)
    pair = paired_size_refusal(answer)
    paired = basis == "typed" and pair is not None and (code, speaker, sentence) == (pair[2], pair[1], pair[3])
    if paired:
        subtree = [pair[0]]  # the pair's own object alone, never a child or inline diagnostic
        for text in answer.objects[pair[0]].sentences:  # its own message or its own detail states a limit
            if (fact := _from_text(text)) is not None and (fact.token_window or fact.has_numbers):
                sentence = text
                if fact.token_window:
                    break
    layers = [dict(answer.objects[i].value, message=answer.objects[i].message) for i in subtree]
    if excludes_size(status, layers):
        return True, None
    found = next((fact for layer in layers if (fact := _structured(layer)) is not None), None)
    # The sentence counts only where the recognised object (or its child) said it.
    said = bool(sentence) and any(sentence in answer.objects[i].sentences for i in subtree)
    text_fact = _from_text(sentence) if said else None
    if found is not None and not found.token_window and text_fact is not None and text_fact.token_window:
        found = replace(found, window=text_fact.token_window)
    found = found or text_fact
    if found is None:
        found = SizeRefusal(units="unstated" if basis == "status_413" else "tokens", source=basis)
    return False, replace(found, gateway=speaker, upstream=sentence,
                          retry_only=bool(paired and not found.token_window),
                          own=_own_whole_request(answer, subtree, sentence, found, paired=paired))


def from_answer(answer, status, message="", typed=()):
    """Extract numeric evidence from an already-read answer; never extend recognition."""
    return _read_numbers(answer, status, message, typed)[1]


def recorded(excluded: bool, found: Optional[SizeRefusal]) -> Optional[SizeRefusal]:
    """The fact a failure's usage keeps: its numbers, a numberless ``excluded`` fact when it states numbers that
    teach no window (``for_usage`` then reads no text for it), else None."""
    return found if found is not None else SizeRefusal(source="excluded") if excluded else None


def answer_fact(answer, status) -> Optional[SizeRefusal]:
    """``recorded`` for an answer read without an exception (an HTTP 200 body's error)."""
    return recorded(*_read_numbers(answer, status))


def _exception_read(exc: Any):
    """``(answer, status, printed form, typed values)`` of a provider exception, as the host's classifier reads it."""
    from ouroboros._usage_response import provider_answer, typed_values
    from ouroboros.context_budget import LocalContextTooLargeError
    from ouroboros.loop_llm_call import _exception_status_code

    answer = provider_answer(exc)
    status = None if isinstance(exc, LocalContextTooLargeError) else _exception_status_code(exc)
    return answer, status, repr(exc), typed_values(exc, answer)


def _exception_numbers(exc: Any):
    return _read_numbers(*_exception_read(exc))


def from_exception(exc: Any) -> Optional[SizeRefusal]:
    """``parse`` over a provider exception as the host's classifier reads it (status, structured body)."""
    return _exception_numbers(exc)[1]


def stamp(usage: Dict[str, Any], exc: Any) -> Optional[SizeRefusal]:
    """Replace the failed call's refusal fact in ``usage`` (``recorded``; cleared when this failure states none)."""
    usage.pop(USAGE_KEY, None)
    excluded, found = _exception_numbers(exc)
    if (fact := recorded(excluded, found)) is not None:
        usage[USAGE_KEY] = fact.to_json()
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


# One refused send's two counts (#3669; owner, 18.11.2026: keep the context, and on a refusal show both calculations).
# Display only: the parser's reads above, ``recorded`` and ``window_learning`` are unchanged by anything below.

#: An HTTP-body limit's own codes and words: bytes, never a token count.
_BODY_LIMIT_CODES = frozenset({"413", "payload_too_large", "request_entity_too_large"})
_BODY_LIMIT_WORDS = re.compile(r"(?:body|payload|request entity) too large", re.I)
# Counts an already typed refusal stated on its own path. In particular a following unitless limit teaches nothing.
_REPORTED_INPUT = re.compile(
    rf"(?:^input too long:\s*|^you requested {_N} output tokens and your prompt contains )"
    rf"(at least )?{_N} input tokens\b", re.I)
#: How the host counted a sent request (``PhysicalAttemptContext.measurement_basis``); without one, the raw count.
_HOST_BASES = {"fresh_route_usage": "this route's measured density", "fresh_model_usage": "this model's measured density",
               "refused_input": "an earlier refusal's density", "cold_estimate": "cold estimate"}


def _stated_number(patterns, text: str) -> Optional[int]:
    """The first pattern's number in ``text``, a stated 0 kept; None when none states one."""
    for pattern in patterns:
        if match := pattern.search(text):
            return int(match.group(1).replace(",", ""))
    return None


def given_counts(fact: SizeRefusal) -> Dict[str, Any]:
    """What one refusal's answer gave, read again from the sentence its numbers came from (``fact.upstream``) with the
    parser's own patterns, never a new wording, so a stated 0 stays 0 and each count says how it was given: ``input``
    with ``input_given`` -- ``stated``, ``at_least`` (a lower bound), ``parts`` (the sum of its stated shares, quoted in
    ``input_parts``) or ``total_less_reply`` (the host's arithmetic on a stated total and reply, only where it states
    no input) --, a ``total`` (``total_about`` where it said "about"), the ``reply`` and the ``window``; a bound in no
    stated unit is ``bound``. A count the answer did not give is absent: no ``input`` means none was established."""
    from ouroboros.capability_evidence import (
        _CTX_INPUT_PART,
        _CTX_INPUT_PATTERNS,
        _CTX_REPLY_PATTERNS,
        _CTX_TOTAL_PATTERN,
    )

    if fact.units != "tokens":
        return {"bound": fact.window} if fact.window else {}
    text = " ".join(str(fact.upstream or "").split())
    given: Dict[str, Any] = {}
    if fact.source == "text:reported-input":
        given.update(input=fact.requested, input_given="at_least" if " at least " in text.lower() else "stated")
        if fact.reply:
            given["reply"] = fact.reply
        return given
    form, match = next(((candidate, found) for candidate, pattern in _COMPILED
                        if fact.source == f"text:{candidate.id}" and (found := pattern.search(text))), (None, None))
    if form is not None:
        numbers = {name: int(value.replace(",", "")) for name, value in zip(form.groups, match.groups())}
        given.update({key: numbers[key] for key in ("reply", "window") if key in numbers})
        if "requested" in numbers:
            given.update(input=numbers["requested"],
                         input_given="at_least" if "at least" in match.group(0).lower() else "stated")
    elif fact.source == "text:shared-vocabulary":
        whole, reply = _stated_number(_CTX_INPUT_PATTERNS, text), _stated_number(_CTX_REPLY_PATTERNS, text)
        total, parts = _CTX_TOTAL_PATTERN.search(text), list(_CTX_INPUT_PART.finditer(text))
        if reply is not None:
            given["reply"] = reply
        if total is not None:
            given["total"] = int(total.group(1).replace(",", ""))
            if re.search(r"\babout\b", total.group(0), re.I):
                given["total_about"] = True
        if whole is not None:
            given.update(input=whole, input_given="stated")
        elif parts:
            given.update(input=sum(int(part.group(1).replace(",", "")) for part in parts), input_given="parts")
        elif total is not None and reply is not None and given["total"] >= reply:
            given.update(input=given["total"] - reply, input_given="total_less_reply")
    else:  # a structured count or a numberless refusal: the parser's own fields, where it stated them
        given.update({key: value for key, value in (("input", fact.requested), ("reply", fact.reply),
                                                    ("total", fact.requested_total)) if value})
        if "input" in given:
            given["input_given"] = "stated"
    if fact.token_window and not given.get("window"):
        given["window"] = fact.token_window
    return given


def _status_notes(codes, http_413: bool) -> Tuple[str, ...]:
    """How a 413 stands in one answer: its received HTTP status, or that answer's own code under no such status."""
    return ("HTTP 413",) if http_413 else ("its answer's own code 413, not an HTTP status",) if "413" in codes else ()


def _other_cause(codes) -> bool:
    """Display-only precedence of exact typed causes, not a new recovery classification."""
    from ouroboros.loop_llm_call import _RETRYABLE_PROVIDER_CODES, NON_RETRYABLE_PROVIDER_MARKERS

    generic = {"400", "badrequest", "bad request", "badrequesterror", "invalidrequest", "invalidrequesterror", "error"}
    return any(code not in generic | _BODY_LIMIT_CODES and (
        code in OTHER_CAUSE_CODES or code in _RETRYABLE_PROVIDER_CODES or code == "429"
        or code in NON_RETRYABLE_PROVIDER_MARKERS or code in ("codex_content_policy_denied", "rate_limit_error")
        or any(code in markers for markers in NON_RETRYABLE_PROVIDER_MARKERS.values())) for code in codes)


def _answer_read(answer, status, message="", typed=(), *, http_413=False):
    """One current refusal's display fact, distinct from its window-learning stamp."""
    from ouroboros.context_budget import CONTEXT_OVERFLOW_CODES, size_refusal

    recognized = size_refusal(answer, status, typed, printed=message)
    basis, code, speaker, sentence = recognized or ("status_413", "", *answer.innermost())
    if basis == "status_413" and not http_413:
        return None  # the body's own 413 is not an HTTP status and, alone, gives no size diagnosis
    subtree = _recognized_subtree(answer, basis, code, speaker, sentence)
    pair = paired_size_refusal(answer)
    if pair and (basis, code, speaker, sentence) == ("typed", pair[2], pair[1], pair[3]):
        subtree = [pair[0]]  # the official paired refusal reads only its own object
    objects = [answer.objects[i] for i in subtree if not answer.objects[i].inline]
    codes = {str(value).strip().lower() for value in (*typed, *answer.own_codes())}
    if _other_cause(codes):
        return None
    texts = [text for obj in objects for text in obj.sentences]
    byte_limit = bool(codes & (_BODY_LIMIT_CODES - {"413"})) or any(_BODY_LIMIT_WORDS.search(text) for text in texts)
    notes = _status_notes(codes, http_413)
    if byte_limit or basis == "status_413":
        return (SizeRefusal(units="unstated"), notes + (("a byte limit",) if byte_limit else ())) if http_413 else None
    if not objects and code.lower() not in codes & (CONTEXT_OVERFLOW_CODES | {LLAMA_CPP_TYPE}):
        return None  # a cited/historical/printed-only sentence supplies no comparison
    found = from_answer(answer, status, message, typed)
    if found is None and "413" in codes and sentence in texts:
        found = _from_text(sentence)  # the token statement and its in-band code belong to this same answer
        if found is not None:
            found = replace(found, upstream=sentence)
    if basis == "typed" and (found is None or "input" not in given_counts(found)):
        for text in texts:
            if match := _REPORTED_INPUT.search(text):
                return SizeRefusal(requested=int(match[3].replace(",", "")), source="text:reported-input", upstream=text,
                                   reply=int((match[1] or "0").replace(",", ""))), notes
        for obj in objects:
            if LLAMA_CPP_TYPE in (value.lower() for value in obj.codes) and obj.value.get("n_prompt_tokens") == 0 \
                    and type(obj.value["n_prompt_tokens"]) is int:
                return replace(found or SizeRefusal(), source="text:reported-input", requested=0), notes
    return found or SizeRefusal(source=basis), notes


def send_counts(capture: Any, error: Any) -> Optional[Dict[str, Any]]:
    """``counts`` of one send that ended with ``error``, from that send alone: None for a send that never left the host
    or an answer whose own object names another cause. An actual HTTP 413 with no contrary typed cause yields the
    host's count beside an unknown provider answer (a body code 413 is not an HTTP status and keeps its typed token
    evidence through the ordinary read)."""
    from ouroboros.loop_llm_call import classify_llm_exception

    if not _observed(capture):
        return None
    answer, status, message, typed = _exception_read(error)
    http_413 = getattr(getattr(error, "response", None), "status_code", None) == 413  # a received HTTP status
    if classify_llm_exception(error).kind != "context_overflow":
        from ouroboros.context_budget import CONTEXT_OVERFLOW_CODES

        if getattr(error, "stream_rejected", False) and any(code.lower() in CONTEXT_OVERFLOW_CODES for code in typed):
            status = None  # rejected SSE frame; only its exact typed cause qualifies it for display
        elif not http_413:
            return None
    read = _answer_read(answer, status, message, typed, http_413=http_413)
    return None if read is None else counts(capture, *read)


def body_counts(capture: Any, body_error: Any) -> Optional[Dict[str, Any]]:
    """Read a received body's error, whose own codes are never the HTTP status."""
    from ouroboros._usage_response import provider_answer

    if not _observed(capture):
        return None
    answer = provider_answer({"error": body_error})
    read = _answer_read(answer, answer.status)
    return None if read is None else counts(capture, *read)


def _observed(capture: Any) -> bool:
    from ouroboros.physical_attempt_context import OBSERVED

    return bool(getattr(capture, "attempt_id", "")) and getattr(capture, "state", "") in OBSERVED


def response_counts(capture: Any, payload: dict) -> Optional[Dict[str, Any]]:
    """Read one retained response before an inner retry, or after its receiver left; no routing or settlement here."""
    from ouroboros._usage_response import (
        _first_chat_choice,
        provider_body_error,
        reply_usability,
        response_finish_reason,
    )
    from ouroboros.context_budget import CONTEXT_OVERFLOW_CODES

    if not _observed(capture):
        return None
    if capture.provider == "claudexor" and "outcome" in payload:
        from ouroboros.llm_claudexor import ClaudexorModelError
        from ouroboros.loop_llm_call import classify_llm_exception

        if payload.get("outcome") == "failed" and isinstance(problem := payload.get("problem"), dict):
            error = ClaudexorModelError(problem)
            if classify_llm_exception(error).basis == "typed":
                return send_counts(capture, error)
        return None
    if body := provider_body_error(payload):
        return body_counts(capture, body)
    first = _first_chat_choice(payload)
    finish = response_finish_reason({}, first or payload)[1]
    if str(finish or "").strip().lower() not in CONTEXT_OVERFLOW_CODES:
        return None
    msg = first.get("message") or {}
    if not first:  # native Messages: only text and tool-use blocks count as a reply
        blocks = [block for block in payload.get("content") or () if isinstance(block, dict)]
        msg = {"content": "".join(str(block.get("text") or "") for block in blocks if block.get("type") == "text"),
               "tool_calls": any(block.get("type") == "tool_use" for block in blocks)}
    text, tools = msg.get("content"), msg.get("tool_calls")
    if not tools and isinstance(text, str) and (
            capture.provider == "local" and not msg.get("reasoning_content") or capture.provider == "openai-compatible"
            and not any(msg.get(key) for key in ("reasoning_content", "reasoning", "reasoning_details"))):
        from ouroboros.llm_local import _split_leading_reasoning

        text = _split_leading_reasoning(text)[0]
    if reply_usability(finish, None, msg) == "unusable" or not (tools or str(text or "").strip()):
        return counts(capture, SizeRefusal(source="finish_reason"))  # the finish states no count
    return None


def counts(capture: Any, fact: SizeRefusal, notes: Tuple[str, ...] = ()) -> Dict[str, Any]:
    """One refused send's two calculations side by side: the host's count of the bytes THAT send carried beside what
    that send's own answer gave (``given_counts``, under ``provider_counts``) and how its status stands (``notes``).
    ``provider`` keeps the capture's route locality (``local``/loopback included), never the answer's numbers. The
    host count is the serving tokenizer's exact input/window when this capture bound one (a true 0
    kept), else its fit-estimator count at the density its round measured with; None where neither was recorded.
    Nothing comes from another send, a later answer or a guessed attempt id; two sends with equal numbers stay two
    rows."""
    try:
        sent = int(getattr(capture, "prompt_tokens_bounded_estimate", 0) or 0)
        reply = int(getattr(capture, "max_completion_tokens", 0) or 0)
    except (TypeError, ValueError):
        sent = reply = 0
    physical = getattr(capture, "physical_context", None)
    density = float(getattr(physical, "measurement_density", None) or 1.0)
    exact, window = getattr(capture, "prompt_tokens_exact", None), getattr(capture, "context_window_tokens_exact", None)
    if exact is not None:
        host = {"input": exact, "basis": "exact_local_count", "density": 1.0, **({"window": window} if window else {})}
    elif sent > 0:
        host = {"input": math.ceil(sent * density), "basis": str(getattr(physical, "measurement_basis", "") or "raw"),
                "density": density}
    else:
        host = None
    row = {"physical_attempt_id": str(getattr(capture, "attempt_id", "") or "") or None,
           "model": str(getattr(capture, "model", "") or ""), "provider": str(getattr(capture, "provider", "") or ""),
           "route_is_loopback": bool(getattr(capture, "route_is_loopback", False)),
           "host": host, **({"host_reply_allowance": reply} if reply > 0 else {}),
           "provider_counts": given_counts(fact), **({"notes": list(notes)} if notes else {})}
    return {**row, "text": counts_text(row)}


def counts_text(row: Dict[str, Any]) -> str:
    """``counts`` in one sentence, the same in Logs and in the next request; what the host or the answer lacks is said
    missing, never shown as 0."""
    host, given = row.get("host"), row.get("provider_counts") or {}
    if host:
        if host["basis"] == "exact_local_count":
            mine = f"host exact local count {host['input']:,} input tokens"
            if host.get("window"):
                mine += f" in a {host['window']:,}-token serving window"
        else:
            measured = f" {host['density']:.2f}" if host["basis"] in _HOST_BASES and host["basis"] != "cold_estimate" else ""
            mine = f"host estimate ~{host['input']:,} input tokens ({_HOST_BASES.get(host['basis'], 'raw count')}{measured})"
    else:
        mine = "no host estimate of its bytes on record"
    if row.get("host_reply_allowance"):
        mine += f", reply allowance {row['host_reply_allowance']:,}"
    how = given.get("input_given")
    if "input" not in given:
        words = ["input not established"]
    else:
        qualifier = "at least " if how == "at_least" else "about " if how == "total_less_reply" and given.get("total_about") else ""
        words = [f"input {qualifier}{given['input']:,}" + {
            "parts": " (sum of its stated parts)", "total_less_reply": " (derived: total less reply)"}.get(how, "")]
    if "total" in given:
        words.append(f"total {'about ' if given.get('total_about') else ''}{given['total']:,}")
    words += [f"{key} {given[key]:,}" for key in ("reply", "window") if key in given]
    words += [f"a bound of {given['bound']:,} in no stated unit"] if "bound" in given else []
    notes = f" ({', '.join(row['notes'])})" if row.get("notes") else ""
    attempt = row.get("physical_attempt_id")
    return (f"Refused send {attempt if attempt else '(attempt not observed)'}: {mine}; "
            f"provider answer{notes}: {', '.join(words)}.")
