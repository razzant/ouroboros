"""Shape-first classification of replayed reasoning artifacts on OpenRouter.

REASON-TO-CHANGE (why this is its own module and not part of ``llm.py``): it owns
MUTABLE EXTERNAL PROVIDER FACTS — which upstream families' opaque reasoning
artifacts survive a same-model cross-provider switch. That is an entry in the
registry of decaying provider facts (docs/DEVELOPMENT.md), re-probed when a
provider changes behavior; it does not change when the LLM client changes. Two
consumers in ``llm.py``: the proactive dispatch continuity pin and the reactive
same-model reroute.

THE QUESTION: replaying an assistant turn that carries provider-private reasoning
back to a DIFFERENT endpoint of the SAME model either works or 400s. It 400s only
when the artifact is SEALED — cryptographically bound to the endpoint that minted
it (a signature, an encrypted blob, a redacted block) AND not vouched as portable
for its family. Readable text and summaries carry no binding at all: they replay
anywhere. Pinning them to one endpoint costs same-model failover for nothing
(issue #468: a 64-lane run died on one provider's 429s with 30 healthy sibling
endpoints, holding a plain ``reasoning.text``).

OPAQUENESS IS DECIDED BY TYPE, never by the mere presence of a ``data`` key: a
sidecar ``data`` on a readable type must not seal the transcript, or #468 returns
in a new costume.

FAIL-CLOSED on shapes that CLAIM content we cannot read (unknown
``reasoning_details`` type, a TRUTHY non-list ``reasoning_details``, a truthy
non-string signature or flat reasoning field): sealed for a non-roster family.
A FALSY-present carrier (explicit null / "" / {} / 0) is the common JSON idiom
for "no reasoning at all" and is not an artifact — sealing it would pin
zero-reasoning transcripts, the response_id bug class. The reactive 400
strip-and-retry in ``llm.py`` is the safety net either way.
"""
import contextvars
from typing import Any, Dict, Iterable, List, Optional

# Families whose SEALED artifact forms are vouched to survive a same-model
# cross-provider switch on OpenRouter (live replay probe 2026-06: Anthropic
# across Anthropic/Bedrock/Vertex/Azure, Gemini across Vertex/AI-Studio).
# The roster vouches EVERY non-plain form for its families — signed, encrypted,
# redacted, unknown — which is exactly the exemption these families held before
# the shape-first classifier, so their failover behavior is unchanged.
#
# ``openai/`` is DELIBERATELY ABSENT despite the 2026-06 probe: in the field
# (2026-07, gpt-5.6-sol over 3x OpenAI + 2x Azure endpoints) replayed encrypted
# items answered "The encrypted content for item rs_... could not be ..." with a
# 400 after 429-driven reroutes and killed whole benchmark waves. Field evidence
# beats the probe. Its readable artifacts stay portable like everyone else's; only
# its encrypted/signed ones pin.
#
# Do not extend this roster by model-name resemblance — only by a fresh
# cross-provider replay probe of the exact family.
SIGNED_PORTABLE = ("anthropic/", "google/gemini-")

_READABLE_DETAIL_TYPES = frozenset({"reasoning.text", "reasoning.summary"})


def _roster_vouched(model: Any) -> bool:
    normalized = str(model or "").strip().lstrip("~").lower()
    return normalized.startswith(SIGNED_PORTABLE)


def _is_signed(value: Any) -> bool:
    """Falsy values (``None``, ``""``, whitespace) are how providers spell
    "unsigned" on a readable artifact. A truthy NON-STRING signature is a shape
    we do not recognize and seals (fail-closed), like every other malformed
    carrier here."""
    if isinstance(value, str):
        return bool(value.strip())
    return bool(value)


def _sealing_detail_label(details: Any) -> Optional[str]:
    if not isinstance(details, list):
        # TRUTHY non-list: an unreadable carrier that claims content — sealed.
        # FALSY non-list (explicit null / "" / {} / 0): the common JSON idiom for
        # "no reasoning at all" — not an artifact (sealing it would pin transcripts
        # with zero reasoning, the response_id bug class; parity with the presence
        # predicate's truthiness contract).
        return "reasoning_details_malformed" if details else None
    for entry in details:
        if not isinstance(entry, dict):
            return "unknown_type"
        dtype = str(entry.get("type") or "").strip().lower()
        if dtype == "reasoning.encrypted":
            return "encrypted"
        if dtype not in _READABLE_DETAIL_TYPES:
            return "unknown_type"
        if _is_signed(entry.get("signature")):
            # Any signed readable entry seals, summary included: a signature is an
            # endpoint binding wherever it rides (fail-closed, roast finding F6).
            return "signed_text"
    return None


def _sealing_block_label(content: Any) -> Optional[str]:
    if not isinstance(content, list):
        return None
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = str(block.get("type") or "").strip().lower()
        if btype == "redacted_thinking":
            return "redacted_thinking"
        if _is_signed(block.get("signature")):
            # Any signed block, not just a thinking one: a stray signature is the
            # endpoint binding wherever it rides (parity with the presence predicate).
            return "signed_block"
    return None


def sealed_reasoning_artifact(messages: Iterable[Any], model: Any) -> Optional[str]:
    """Label of the FIRST sealed reasoning artifact in ``messages``, else ``None``.

    The label is a bounded host-owned vocabulary (``encrypted``, ``signed_text``,
    ``signed_block``, ``redacted_thinking``, ``unknown_type``,
    ``reasoning_details_malformed``, ``reasoning_malformed``) safe to disclose in
    usage telemetry."""
    if _roster_vouched(model):
        # The roster vouches every form for its families: nothing here can seal.
        return None
    for msg in messages or ():
        if not isinstance(msg, dict):
            continue
        for flat_key in ("reasoning", "reasoning_content"):
            flat = msg.get(flat_key)
            if flat and not isinstance(flat, str):
                # A non-string truthy flat reasoning field is a shape we do not
                # recognize — fail-closed like the other malformed carriers.
                return "reasoning_malformed"
        label = _sealing_detail_label(msg.get("reasoning_details"))
        if label:
            return label
        label = _sealing_block_label(msg.get("content"))
        if label:
            return label
    return None


def transcript_has_sealed_reasoning(messages: Iterable[Any], model: Any) -> bool:
    """Whether replaying ``messages`` on ``model`` is endpoint-BOUND — i.e. at least
    one carried reasoning artifact is sealed. Flat ``reasoning``/``reasoning_content``
    strings and a bare ``response_id`` are never sealing (a readable string replays
    anywhere; an id is not an artifact at all)."""
    return sealed_reasoning_artifact(messages, model) is not None


def sealed_reasoning_pin_fact(messages: List[Any], model: Any) -> Optional[dict]:
    """The typed usage disclosure for a continuity pin, or ``None`` when the
    transcript is portable. Callers stamp it from the TERMINAL sent candidate."""
    label = sealed_reasoning_artifact(messages, model)
    return {"sealed": True, "artifact": label} if label else None


# Pin disclosure slot: a ContextVar isolates threads AND concurrent asyncio tasks.
# It lives beside the fact it carries, so the producer (llm_attempt, on send
# success) and the reader (llm_openai_compatible, on usage assembly) share it
# without either leaf importing the other.
_REASONING_PIN_CVAR: contextvars.ContextVar = contextvars.ContextVar(
    "ouroboros_reasoning_pin_note", default=None,
)


def pop_reasoning_pin_note() -> Optional[Dict[str, Any]]:
    """Take and clear this call's pin note, if the send staged one."""
    pending = _REASONING_PIN_CVAR.get()
    _REASONING_PIN_CVAR.set(None)
    return pending if isinstance(pending, dict) else None
