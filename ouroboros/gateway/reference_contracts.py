"""Additive stored-row identity shared by live chat and attachment frames."""
from typing import Any, Literal

try:
    from typing import NotRequired, TypedDict
except ImportError:  # Python 3.10
    from typing_extensions import NotRequired, TypedDict


class ReplyExcerpt(TypedDict):
    """Sender-reported rendered words and their zero-based UTF-16 position within the addressed body."""
    text: str
    start_utf16: int


class ChatInbound(TypedDict):
    """Inbound WS chat message. ``type`` and ``content`` are required."""

    type: Literal["chat"]
    content: str
    sender_session_id: NotRequired[str]
    client_message_id: NotRequired[str]
    force_plan: NotRequired[bool]  # Deprecated for one release; accepted and ignored.
    attachments: NotRequired[list]  # list[ChatAttachmentInbound] (additive, v6.26.0)
    # Multi-project chat routing: the owner stays user_id 1; chat_id selects the thread, project_id scopes memory.
    chat_id: NotRequired[int]
    project_id: NotRequired[str]
    reply_excerpt: NotRequired[ReplyExcerpt]
    reply_to: NotRequired[str]  # Host resolves the selected row/card at acceptance.
    # raw surface observables at send time; client_surface.normalize_client_surface keeps their closed key set
    client_surface: NotRequired[dict]
    # The data-profile scope the serving document proves (server_web.data_profile_scope digest, stamped as the
    # builtin index's ouroboros-chat-preview-scope meta and read by the browser from it). gateway/ws.py refuses a
    # chat frame whose claim positively differs from the serving connection's own confirmed scope (#3294);
    # absent or empty is no proof and is admitted, so legacy pages and unproven profiles keep working.
    profile_scope: NotRequired[str]
    # A Send again of a copy the page showed only on a newest history read's 503 word
    # (``ChatHistoryResponse.ordinary_room``): the room that word named. The ordered acceptance asks it again before
    # saving or handing over anything (gateway/ws.py ``_accept_on_room_word``); frames without it pass as before.
    ordinary_room: NotRequired[int]


class ConnectionProfileOutbound(TypedDict):
    """Issued once per socket at accept, before registration, when the serving connection proves a
    data profile (server_web.data_profile_scope; gateway/ws.py sends it directly — never per frame,
    poll or command). The browser adopts it before the generic emit and compares it with the profile
    its document proves (#3294). A direct accept-time issuance, not a broadcast envelope."""

    type: Literal["connection_profile"]
    profile_scope: str


class ChatReplyReference(TypedDict, total=False):
    """A ``reference_from_row`` reference as the host resolved it; an unresolved one has the base form
    ``{address, unresolved: true}``, and a browser ``task:`` address without a card also keeps ``work_task_id: ""``."""
    address: str
    row_sha256: str  # full digest of a resolved row; older references only carry the display address
    chat_id: int
    role: Literal["user", "assistant", "system"]
    quote_role: Literal["user"]  # ink of the owner's recorded card answer on its System row
    author_label: str
    kind: str
    program_source: Literal["agent_note", "late_result", "place_machine"]  # recorded source kind, not permission
    excerpt: ReplyExcerpt
    attachment_failures: list[dict[str, Any]]  # pathless facts from the exact source row
    snippet: str
    chars: int  # the quoted text's whole length, present when ``snippet`` cuts it
    ts: str
    task_id: str
    work_task_id: str  # addressed work, not the quoted author; empty means the work could not be resolved
    ordinary: Literal[True]  # the host proved the quoted row addresses no work; with neither key it could not tell
    target_home: Literal["work", "room", "none"]  # what decides the quoted row's room; an older reference lacks it
    derived_question: Literal[True]  # host-derived question quote on an owner answer bubble without its own Reply, built per room on read or single-room live frame; never stored, no owner Reply
    room_label: str  # only with derived_question: Main, the Project's display name, or Project when unnamed, when the question's home is another room
    room_label_kind: Literal["main", "project"]  # only with derived_question + room_label: built-in chrome (Main or unnamed Project); absent for a user-named Project
    telegram_room_chat_id: int | None  # the owner bot's Reply only: its Project's turn, 0 its chat's own, null refused
    unresolved: bool


class ChatReferenceFields(TypedDict, total=False):
    row_address: str
    row_client_message_id: str  # a row's own saved name (outgoing or host-fact ingress); never the owner's submission id
    reply_author_label: str  # whom a Reply to this row names: You, Ouroboros, System or the recorded speaker
    reply_to: ChatReplyReference
    relay: dict[str, Any]
    reply_to_relay: str
    kind: str
    room_revisions: dict[str, int]  # a row shown in several rooms: the revision it took in each room it counted in
    project_thread: bool  # server-stamped when chat_id is a reserved Project thread: Main never adopts it, even before the browser learns it


class SharedRowFrame(TypedDict, total=False):
    also_rooms: list[int]  # a live chat frame of a row other Projects show too: those rooms (``room_appearances.frame_rooms``)


class HostNotice(TypedDict, total=False):
    """A host notice on a System row (``ouroboros/host_notice.py``): the row's ``text`` is its one phrase."""
    code: str  # the notice's stable code
    step: str  # the owner's next step, a key of host_notice.STEPS; absent = no step
    details: str  # the raw facts behind Details
    instruction: str  # one line the owner acts on, as written (a Git command); a reader without Details says it under the line
    template: str  # a variable phrase's constant English with {name} slots; the template is what gets translated
    values: dict[str, str]  # the words that fill the template's slots, inserted as written, never translated
    line: str  # phrase + step in the install's language, added when the row is sent or read; Telegram and Copy say it
    phrase: str  # that line's phrase alone, sent beside it; a reader that sets the step under the phrase reads it


class HostNoticeField(TypedDict, total=False):
    notice: HostNotice  # a System row's host notice, the same on its live frame and its history row (DESIGN principle 7)
