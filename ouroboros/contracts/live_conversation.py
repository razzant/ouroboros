"""Live Conversation's vocabulary between the host, the call feature and a provider.

The host admits a call inside the owner's ordinary turn and gives the feature's
handler a ``LiveHost`` port. The feature runs the call: it opens provider
sessions through the port, moves media and captions, and replaces a session when
the provider or the host asks. The host fits the room's ordinary context to the
route's window at every (re)open, runs tools, writes history, prices every
response against the shared limit and owns Stop and Panic. A provider session
(``LiveSession``) speaks its wire and nothing else: it never selects context,
dispatches tools, reserves money or silently drops input.

Provider events (``LiveEvent.kind`` and its ``data``):
  ready            {input_rate, output_rate}: PCM16 mono rates the session takes and returns
  audio            {item_id, rate, pcm}
  caption          {speaker: "owner"|"ouroboros", item_id, text, final, replace?, incomplete?}
  speech_started   {}: someone spoke over the answer (only where interruptions are on)
  input_started    {} / input_done {}: native input boundaries (VAD, transcription), the
                   only evidence that the input side is quiet; a sent frame is not
  response_started {response_id}
  response_done    {response_id, usage: {USAGE_AXES...}, raw_usage, status, continues}
                   status: completed | interrupted | cancelled | failed | incomplete;
                   continues: the provider's interaction goes on past this response
  tool_call        {call_id, name, arguments (JSON text)}
  resumable        {handle}: the latest handle a replacement session can resume with
  renew            {within_s, reason, resume}: replace this session within ``within_s``;
                   resume=True reopens with the latest handle (same server-side
                   conversation), False opens a fresh session the host refits
  closed           {reason, resumable}: the provider ended the connection
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping, Protocol, Sequence

from ouroboros.contracts.realtime import CatalogModel, RealtimeInput, RealtimeSetupError

PAUSES = ("short", "normal", "long")


def _languages(value: Any) -> None:
    if not isinstance(value, tuple) or any(not isinstance(tag, str) or not tag.strip() for tag in value):
        raise RealtimeSetupError("configuration", "languages must be a tuple of language tags")


@dataclass(frozen=True)
class LiveConfig:
    """The call feature's configuration as the host reads it (``configure()``).

    The owner's choices for the next call. No time limit and no per-call money: the
    shared spending limit is the one limit. An empty model or voice means the
    engine's default.
    """

    enabled: bool = False
    provider: str = ""  # "<skill>:<name>" of a registered realtime provider
    model: str = ""
    voice: str = ""
    instructions: str = ""  # the owner's instructions for calls
    pause: str = "normal"
    interruptions: bool = True
    thinking: str = ""
    search: bool = False

    def __post_init__(self):
        if type(self.enabled) is not bool or type(self.interruptions) is not bool or type(self.search) is not bool:
            raise RealtimeSetupError("configuration", "enabled, interruptions and search are booleans")
        for name in ("provider", "model", "voice", "thinking"):
            value = getattr(self, name)
            if not isinstance(value, str) or value != value.strip() or len(value) > 160:
                raise RealtimeSetupError("configuration", f"invalid {name}")
        if not isinstance(self.instructions, str) or len(self.instructions) > 8000:
            raise RealtimeSetupError("configuration", "instructions must be text up to 8000 characters")
        if self.pause not in PAUSES:
            raise RealtimeSetupError("configuration", f"pause must be one of {PAUSES}")
        if self.enabled and not self.provider:
            raise RealtimeSetupError("configuration", "choose an engine before enabling calls")


@dataclass(frozen=True)
class LiveOptions:
    """What one provider session is opened with, besides the host-selected input."""

    voice: str = ""
    # The feature's call instructions, then the owner's. A provider sends them
    # after the host system text, inside the same budgeted input.
    instructions: str = ""
    languages: tuple[str, ...] = ()  # BCP-47 tags recognition should expect, most spoken first
    pause: str = "normal"
    interruptions: bool = True
    thinking: str = ""  # one of the model's thinking levels; "" = the provider's default
    search: bool = False
    resume_handle: str = ""  # from a ``resumable`` event: continue that server-side conversation
    # Fresh exact-route response-limit evidence supplied by the host. This is
    # independent of the selected allowance; absent evidence is never a default.
    model_output_bound_tokens: int | None = None

    def __post_init__(self):
        _languages(self.languages)
        if self.pause not in PAUSES:
            raise RealtimeSetupError("options", f"pause must be one of {PAUSES}")
        bound = self.model_output_bound_tokens
        if bound is not None and (type(bound) is not int or bound < 1):
            raise RealtimeSetupError("options", "model_output_bound_tokens must be a positive integer or None")


@dataclass(frozen=True)
class LiveEvent:
    kind: str
    data: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Audio:
    pcm: bytes
    rate: int


@dataclass(frozen=True)
class Video:
    data: bytes
    mime: str


@dataclass(frozen=True)
class Interrupt:
    item_id: str = ""
    played_ms: int = 0


@dataclass(frozen=True)
class ToolOutputs:
    # Complete changes from the host's tool batch, including image blocks.
    messages: tuple[dict, ...]
    tools: tuple[dict, ...]
    # Tool call ids whose results open no spoken answer (a call tool marked silent).
    silent: tuple[str, ...] = ()


@dataclass(frozen=True)
class ContextUpdate:
    messages: tuple[dict, ...]


@dataclass(frozen=True)
class Respond:
    """Open one response that answers this host fact (the call's greeting).

    The fact is already in the selected input; a provider that needs content to
    open a response sends the fact as a host-labelled turn, never as owner words.
    """

    fact: str


class LiveSession(Protocol):
    selected: RealtimeInput
    wire: dict  # the physical setup before opening; the host seals it as evidence
    # Effective native output bound. None means unknown, including an unbounded
    # field with no fresh maximum.
    output_bound_tokens: int | None
    # Native input, transcription and playback work has reached a boundary where a
    # replacement session loses nothing.
    renewal_ready: bool

    def final_captions(self) -> list[LiveEvent]: ...
    async def open(self) -> None: ...
    async def receive(self) -> list[LiveEvent]: ...
    async def send(self, command: Audio | Video | Interrupt | ToolOutputs | ContextUpdate | Respond) -> None: ...
    async def close(self) -> None: ...


# A provider's session factory: session(selected, options, *, observe) -> LiveSession.
# ``observe(direction, raw)`` records wire evidence in the task. Direction "usage"
# additionally reports {"usage": {USAGE_AXES...}, "raw_usage": {...}} for this
# physical connection, including idle time and closing tail. The host settles
# connected_seconds cumulative growth across these samples and response_done;
# it never sums watermarks or creates a synthetic response. close() publishes its
# final sample before returning, even after Stop or exhausted admission: settlement
# records spent usage and admits nothing new. Response ids dedupe across the call,
# including resumed connections; each connection starts its own time watermark.
# billable_audio_seconds is evidence.
# A factory may take further keyword-only test hooks of its own.
SessionFactory = Callable[..., LiveSession]


@dataclass(frozen=True)
class CallTool:
    """A tool that exists only inside a call of the feature that registered it.

    The host runs it on the call's ordinary tool lane, records its call and result
    in history like any tool, and awaits ``handler(host, arguments)`` for the result
    text. A silent tool's result opens no spoken answer.
    """

    name: str
    description: str
    parameters: Mapping[str, Any]
    handler: Callable[..., Awaitable[str]]
    silent: bool = False


class ToolBatch(list):
    """History changes returned by LiveHost.tool_batch, with silent call ids.

    A list-compatible result keeps callers that only forward rows working; the
    feature forwards ``silent`` to ``ToolOutputs`` without inspecting tool names.
    """

    def __init__(self, rows=(), *, silent=()):
        super().__init__(rows)
        self.silent = tuple(silent)


class LiveHost(Protocol):
    """The admitted call's host port, given to the feature's handler and to its call tools."""

    hub: Any  # the call's peer hub (ouroboros.call_peers): media, captions and state to every peer
    config: LiveConfig  # the configuration this call was admitted with
    model: CatalogModel  # the selected model: vision, the knobs it honours, its route facts
    schemas: list  # the call's current tool schemas: the ordinary view plus the feature's call tools
    opening: str  # the host fact that opened this call; the greeting answers it once per call
    state: dict  # the feature's own per-call scratch; the host never reads it

    async def check(self) -> None:
        """Raise ``RealtimeSetupError`` when Stop, Panic, the shared limit or a disable ends the call."""

    async def session(self, consume: Callable[[LiveSession], Awaitable[Any]], options: LiveOptions) -> Any:
        """One physical connection: fit the current history to the route, open, ``consume``, close.

        The host refits the canonical history and tools at every call (never the
        first plan), adds its output-bound evidence to ``options`` and seals the
        provider's wire before dispatch. A known overflow the window ladder cannot
        fit refuses with a typed reason instead of cutting anything.
        """

    async def mailbox(self) -> list[dict]:
        """Rows that reached this room since the last read, already in history."""

    async def tool_batch(self, calls: Sequence[dict]) -> list[dict]:
        """Run ordinary and call tools on the serialized lane; return the history changes."""

    async def caption(self, speaker: str, text: str, item_id: str, *, heard: bool = False, played_ms: int = 0,
                      interrupted: bool = False, completed: bool = False, replace: bool = False) -> None:
        """Write one final caption to history: owner words, a heard participant, or spoken output."""
        # replace revises the same item's text, retaining identity and playback/provenance;
        # unchanged replays do nothing. A replacement before the first write creates one row.

    async def observe_context(self, row: dict) -> None:
        """Add a peer's observation to history as a labelled row, never as owner words."""

    def response_started(self, response_id: str) -> None:
        """Reserve this response once per call, across resumed connections; refusal ends the call."""

    def response_done(self, response_id: str, *, usage: Mapping[str, Any], raw_usage: Mapping[str, Any],
                      status: str) -> None:
        """Settle this call-lifetime response id once (unknown when unpriced, never zero)."""

    def renewal_due(self) -> bool:
        """Held input + pending history/schema growth + working reserve needs a refit.

        Pending means rows/schemas added since the last usage receipt (or the
        sealed selection before the first receipt), including mailbox, peer and
        tool changes. Counts use the route's declared units and retain all rows.
        Check after adding canonical context and before sending ContextUpdate or
        ToolOutputs; when due, refit/reopen instead of sending the overflow.
        """
