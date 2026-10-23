"""What a realtime voice provider declares and what the host gives it.

A provider plugin owns its wire: layout, captions, usage normalisation and when
its own session must be replaced. The host owns everything else: which context
is selected and how it fits, tools, history, money and Stop. The provider
declares its route identity and a typed catalog (models, voices, the settings
each model honours, and the facts of its route with their source); the host
fits, prices and accounts with them. Credential values, devices and context
policy never live here.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class RealtimeSetupError(ValueError):
    """A typed refusal; never permission to shorten or relabel the input."""

    def __init__(self, code: str, detail: str):
        self.code = code
        super().__init__(f"{code}: {detail}")


USAGE_AXES = (
    "text_input_tokens", "audio_input_tokens", "image_input_tokens", "video_input_tokens",
    "cached_input_tokens", "text_output_tokens", "audio_output_tokens", "thinking_tokens",
    "connected_seconds",
)
"""Host usage axes for every final response, including failed/interrupted ones.

A provider reports an explicit 0 for an axis its route never bills. An absent
axis means unknown, never zero. ``cached_input_tokens`` is part of the input
counts, whose modality split raw usage may witness. ``thinking_tokens`` is
billed IN ADDITION to the output axes, never a subset: a provider whose reasoning
count is already inside text output must not also report it as thinking.
``connected_seconds`` is a cumulative watermark for routes billed by connected
time: seconds since the physical connection opened, including idle and closing
tail; the host prices only its growth. ``billable_audio_seconds`` stays evidence.
"""

# Call settings a model may honour. A knob the selected model does not honour
# is not offered to the owner; the catalog says in one sentence why.
CALL_KNOBS = frozenset({"pause", "interruptions", "thinking", "search", "languages"})


def _text(name: str, value: Any, *, empty: bool = False) -> None:
    if not isinstance(value, str) or value != value.strip() or (not value and not empty):
        raise RealtimeSetupError("catalog", f"{name} must be a {'' if empty else 'nonblank '}trimmed string")


def _count(name: str, value: Any) -> None:
    if value is not None and (type(value) is not int or value < 1):
        raise RealtimeSetupError("catalog", f"{name} must be a positive integer or None (unknown)")


@dataclass(frozen=True)
class RealtimeRoute:
    """Route identity a provider registration declares; the host fits and accounts with it.

    ``accounting_provider`` keys capability evidence and the ledger, ``endpoint`` is
    the socket the provider opens, and ``model_prefix`` turns a native model id into
    host model identity.
    """

    accounting_provider: str
    endpoint: str
    model_prefix: str

    def __post_init__(self):
        for name in ("accounting_provider", "endpoint", "model_prefix"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise RealtimeSetupError("route", f"{name} must be a nonblank string")


@dataclass(frozen=True)
class RouteFacts:
    """What is known about one model's route, each fact labelled with where it came from.

    ``conversation_window_tokens`` is the window the ordinary fitter uses for this
    route: the largest input under which the model still held the conversation,
    measured or documented (``source``). ``accepted_input_tokens`` is evidence only:
    an input the endpoint once accepted. A larger accepted input raises the
    conversation window only when ``acceptance_can_raise`` explicitly says so. A window whose ``source`` names a retention measurement
    (spoken turns held under it) is bound by retention, below what the endpoint
    accepts; a route whose retention is not yet measured states its largest accepted
    and answered input as the window (a successful request is a lower bound, never a
    ceiling below it) and says so in ``source``. ``working_reserve_tokens`` is the room inside the window the
    conversation needs to grow in one session before the host renews it. ``basis``
    names the counter of these numbers: ``provider`` is the provider's own token
    count, which the host reaches through the route's calibrated density like any
    route; ``estimator`` is the host's own estimate before any density (four
    characters a token), for a route whose usage cannot calibrate it. None is
    unknown; an unknown window steps nothing and fits nothing.
    """

    conversation_window_tokens: int | None = None
    accepted_input_tokens: int | None = None
    working_reserve_tokens: int | None = None
    basis: str = "provider"
    source: str = ""
    observed: str = ""  # ISO date of the measurement or the documentation read

    acceptance_can_raise: bool = False  # retention and estimator windows are independent of accepted input

    def __post_init__(self):
        if type(self.acceptance_can_raise) is not bool or (self.acceptance_can_raise and self.basis == "estimator"):
            raise RealtimeSetupError("catalog", "only provider-token limits can be raised by accepted input")
        for name in ("conversation_window_tokens", "accepted_input_tokens", "working_reserve_tokens"):
            _count(name, getattr(self, name))
        if self.basis not in {"provider", "estimator"}:
            raise RealtimeSetupError("catalog", "basis must be 'provider' or 'estimator'")
        _text("source", self.source, empty=True)
        _text("observed", self.observed, empty=True)
        window, reserve = self.conversation_window_tokens, self.working_reserve_tokens
        if window is not None and reserve is not None and reserve >= window:
            raise RealtimeSetupError("catalog", "working reserve must leave room inside the window")


@dataclass(frozen=True)
class CatalogVoice:
    id: str
    label: str = ""

    def __post_init__(self):
        _text("voice id", self.id)
        _text("voice label", self.label, empty=True)


@dataclass(frozen=True)
class CatalogModel:
    """One model the provider offers, with what it can do and what is known of its route."""

    id: str
    label: str
    facts: RouteFacts = field(default_factory=RouteFacts)
    default: bool = False
    vision: bool = False  # accepts images and video frames
    voices: tuple[CatalogVoice, ...] = ()  # empty: the provider's own default voice only
    thinking_levels: tuple[str, ...] = ()  # empty: no thinking control
    honours: frozenset[str] = frozenset()  # subset of CALL_KNOBS
    # One plain sentence per knob in CALL_KNOBS it does not honour, shown where the knob would be.
    notes: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        _text("model id", self.id)
        _text("model label", self.label)
        if not isinstance(self.facts, RouteFacts):
            raise RealtimeSetupError("catalog", "facts must be RouteFacts")
        if not set(self.honours) <= CALL_KNOBS:
            raise RealtimeSetupError("catalog", f"unknown knob in honours: {sorted(set(self.honours) - CALL_KNOBS)}")
        if bool(self.thinking_levels) != ("thinking" in self.honours):
            raise RealtimeSetupError("catalog", "thinking levels and the thinking knob go together")
        if len({voice.id for voice in self.voices}) != len(self.voices):
            raise RealtimeSetupError("catalog", "voice ids must be unique")
        for knob, sentence in self.notes:
            if knob not in CALL_KNOBS or knob in self.honours:
                raise RealtimeSetupError("catalog", f"note for {knob!r} needs a knob the model does not honour")
            _text("note", sentence)

    def note(self, knob: str) -> str:
        return dict(self.notes).get(knob, "")


@dataclass(frozen=True)
class ProviderCatalog:
    """What the provider offers; the feature's settings and the host's fitting read it."""

    label: str  # the engine as the owner sees it, e.g. "Gemini · Google"
    models: tuple[CatalogModel, ...]
    credential_settings: tuple[str, ...]  # setting names whose grant opens a call, e.g. ("GEMINI_API_KEY",)
    privacy: str = ""  # one sentence: where the audio goes

    def __post_init__(self):
        _text("catalog label", self.label)
        _text("privacy", self.privacy, empty=True)
        if not self.models or sum(model.default for model in self.models) != 1:
            raise RealtimeSetupError("catalog", "a catalog has models and exactly one default")
        if len({model.id for model in self.models}) != len(self.models):
            raise RealtimeSetupError("catalog", "model ids must be unique")
        if not self.credential_settings:
            raise RealtimeSetupError("catalog", "name the credential setting a call needs")
        for name in self.credential_settings:
            _text("credential setting", name)

    def model(self, model_id: str = "") -> CatalogModel:
        """The named model, or the default for an empty id; an unknown id is refused."""
        for model in self.models:
            if (model.id == model_id) if model_id else model.default:
                return model
        raise RealtimeSetupError("model_unavailable", f"{model_id!r} is not in this engine's catalog")


@dataclass(frozen=True)
class RealtimeInput:
    """Immutable host-selected model input, detached from mutable caller dictionaries."""

    model: str
    messages_json: str
    tools_json: str
    output_allowance_tokens: int
    mode: str
    core_sha256: str
    memory_view_sha256: str

    @property
    def sha256(self) -> str:
        from dataclasses import asdict

        return sha256(canonical_json(asdict(self)))
