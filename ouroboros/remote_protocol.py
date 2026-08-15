"""Strict dependency-light framing and prepared-call identity for execd."""

from __future__ import annotations

import hashlib
import json
import queue
import re
import struct
import unicodedata
from dataclasses import dataclass
from typing import Any, BinaryIO, Callable

from ouroboros.remote_contracts import (
    CONTRACT_SET_VERSION,
    ContractDriftError,
    refuse_unknown_members,
)

PROTOCOL_MAJOR = 1
# The wire minor IS the Home↔execd contract-set version, not a second number that
# tracks it (`remote_contracts` states the rationale). The wire already carries this
# value in the session preamble and in both handshake frames of every build ever
# shipped, so making it the compatibility carrier is what lets Home refuse a target
# that was installed before the check existed — such a target announces 0 without
# being asked. A NEW handshake field would have been absent on exactly those builds.
PROTOCOL_MINOR = CONTRACT_SET_VERSION
PREAMBLE_MAGIC = b"OUROBOROS-EXECD\0"
MAX_PREAMBLE_BYTES = 1024
MAX_CONTROL_BYTES = 1_048_576
MAX_BULK_BYTES = 65_536
MAX_REMOTE_EXTERNAL_ENVELOPE_BYTES = 8 * 1024 * 1024
MAX_UNACKNOWLEDGED_BULK_FRAMES = 1
MAX_JSON_DEPTH = 32
MAX_JSON_CONTAINER_ITEMS = 4096
MAX_JSON_STRING_BYTES = 262_144
MIN_JSON_INTEGER = -(1 << 63)
MAX_JSON_INTEGER = (1 << 63) - 1
MAX_LEASE_TTL_MS = 15_000

# The CLOSED channel registry (RWS v2 §3.2, Appendix C-1): every blob kind that may
# cross the Home/target boundary. It lives in the WIRE contract because both halves
# must agree on it and neither may import the other — the transport journal
# validates a durable intent against it, and the Home transfer service refuses an
# import whose channel is not in it. An unknown kind fails closed: a blob whose
# channel nobody declared has no export policy attached, so importing it would be an
# unpoliced write. Adding a channel is therefore a deliberate edit HERE, once.
IMPORT_CHANNELS = frozenset({"task_result_v1", "attachment_stage_v1"})
_HEADER = struct.Struct("!cI")
_KINDS = {b"J": "control", b"B": "bulk"}
_OPAQUE_ID_RE = re.compile(r"^[A-Za-z0-9_:@-](?:[A-Za-z0-9_.:@-]{0,254}[A-Za-z0-9_:@-])?$")
_TOOL_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_COMPLETION_STATES = frozenset({"not_started", "completed", "unknown"})
_COMMON_REQUIRED = frozenset({"kind", "seq"})
_CONTROL_SHAPES: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "handshake": (
        frozenset({"nonce", "protocol_major", "protocol_minor"}),
        frozenset({"client_build", "capability_hash"}),
    ),
    "handshake_ok": (
        frozenset({"protocol_major", "protocol_minor", "host_id", "server_generation"}),
        frozenset({"platform", "build", "capability_hash"}),
    ),
    "prepare": (
        frozenset({"request_id", "operation_id", "tool", "args"}),
        frozenset({"task_id", "workspace_id", "deadline_ms"}),
    ),
    "prepared": (
        frozenset({"request_id", "operation_id", "prepared_hash", "prepared"}),
        frozenset({"expires_ms"}),
    ),
    "continue": (
        frozenset({"request_id", "operation_id", "prepared_hash"}),
        frozenset(),
    ),
    "abort": (
        frozenset({"request_id", "operation_id"}),
        frozenset({"reason"}),
    ),
    "stdout": (
        frozenset({"request_id", "operation_id", "blob_id"}),
        frozenset({"eof"}),
    ),
    "stderr": (
        frozenset({"request_id", "operation_id", "blob_id"}),
        frozenset({"eof"}),
    ),
    "result": (
        frozenset({"request_id", "operation_id", "completion", "result"}),
        frozenset({"prepared_hash"}),
    ),
    "diagnostic": (
        frozenset({"request_id", "operation_id", "diagnostic"}),
        frozenset(),
    ),
    "blob_manifest": (
        frozenset({"request_id", "blob_id", "size", "sha256"}),
        frozenset({"operation_id", "media_type"}),
    ),
    "blob_ack": (
        frozenset({"request_id", "blob_id", "chunk_seq"}),
        frozenset({"operation_id", "complete"}),
    ),
    "blob_fetch": (
        frozenset({"request_id", "blob_id", "size"}),
        frozenset(),
    ),
    "lease": (
        frozenset({"server_generation", "lease_id", "ttl_ms"}),
        frozenset({"task_id", "service_id"}),
    ),
    "reconcile": (
        frozenset({"request_id", "operation_id", "prepared_hash"}),
        frozenset(),
    ),
    "reconcile_result": (
        frozenset({"request_id", "operation_id", "completion"}),
        frozenset({"result"}),
    ),
    "cancel": (
        frozenset({"request_id", "operation_id"}),
        frozenset({"task_id", "service_id"}),
    ),
    "panic": (
        frozenset({"server_generation"}),
        frozenset(),
    ),
    "ack": (
        frozenset({"ack_seq"}),
        frozenset({"request_id", "operation_id"}),
    ),
}


class ProtocolError(ValueError):
    pass


class ProtocolContractError(ProtocolError, ContractDriftError):
    """Wire-contract drift: a framing failure AND a typed contract refusal.

    Both bases are load-bearing. Every transport path already treats a
    ``ProtocolError`` as "this session cannot continue", and routing an unknown
    control kind through some other exception type would tear the session down by a
    different route than every other malformed frame. And the ``ContractDriftError``
    half is what turns "unknown control message kind" into a diagnostic that names
    the contract, the member and the owner's action.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        ContractDriftError.__init__(self, *args, **kwargs)


class ProtocolEOF(ProtocolError):
    """Typed transport termination; `partial` distinguishes truncation."""

    def __init__(self, *, partial: bool) -> None:
        self.partial = bool(partial)
        super().__init__("unexpected EOF inside frame" if partial else "protocol stream closed")


def _validate_json_value(value: Any, *, depth: int = 0) -> None:
    if depth > MAX_JSON_DEPTH:
        raise ProtocolError("JSON value exceeds maximum depth")
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, int):
        if value < MIN_JSON_INTEGER or value > MAX_JSON_INTEGER:
            raise ProtocolError("JSON integer exceeds signed 64-bit range")
        return
    if isinstance(value, float):
        raise ProtocolError("JSON floating-point values are not permitted")
    if isinstance(value, str):
        if unicodedata.normalize("NFC", value) != value:
            raise ProtocolError("JSON strings must use NFC Unicode normalization")
        if len(value.encode("utf-8")) > MAX_JSON_STRING_BYTES:
            raise ProtocolError("JSON string exceeds limit")
        return
    if isinstance(value, list):
        if len(value) > MAX_JSON_CONTAINER_ITEMS:
            raise ProtocolError("JSON array exceeds item limit")
        for item in value:
            _validate_json_value(item, depth=depth + 1)
        return
    if isinstance(value, dict):
        if len(value) > MAX_JSON_CONTAINER_ITEMS:
            raise ProtocolError("JSON object exceeds item limit")
        for key, item in value.items():
            if not isinstance(key, str):
                raise ProtocolError("JSON object keys must be strings")
            _validate_json_value(key, depth=depth + 1)
            _validate_json_value(item, depth=depth + 1)
        return
    raise ProtocolError(f"unsupported JSON value type: {type(value).__name__}")


def canonical_json(value: Any, *, _max_bytes: int | None = None) -> bytes:
    _validate_json_value(value)
    try:
        encoder = json.JSONEncoder(
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        payload = bytearray()
        for chunk in encoder.iterencode(value):
            encoded = chunk.encode("utf-8")
            if _max_bytes is not None and len(payload) + len(encoded) > _max_bytes:
                raise ProtocolError("control frame exceeds limit")
            payload.extend(encoded)
        return bytes(payload)
    except ProtocolError:
        raise
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"value is not canonical JSON: {exc}") from exc


def canonical_prepared_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def protocol_compatible(peer_major: Any, peer_minor: Any) -> bool:
    """A peer may be older within the same major, never newer or cross-major."""

    if (
        not isinstance(peer_major, int)
        or isinstance(peer_major, bool)
        or not isinstance(peer_minor, int)
        or isinstance(peer_minor, bool)
    ):
        return False
    return peer_major == PROTOCOL_MAJOR and 0 <= peer_minor <= PROTOCOL_MINOR


def _require_token(message: dict[str, Any], field: str) -> str:
    value = message.get(field)
    if not isinstance(value, str) or not _OPAQUE_ID_RE.fullmatch(value):
        raise ProtocolError(f"control field {field!r} must be a file-safe opaque ID")
    return value


def _require_integer(
    message: dict[str, Any],
    field: str,
    *,
    minimum: int = 0,
    maximum: int = MAX_JSON_INTEGER,
) -> int:
    value = message.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum or value > maximum:
        raise ProtocolError(f"control field {field!r} must be an integer in {minimum}..{maximum}")
    return value


def _validate_control_values(message: dict[str, Any], kind: str) -> None:
    _require_integer(message, "seq")
    for field in ("request_id", "operation_id", "blob_id", "server_generation", "lease_id"):
        if field in message:
            _require_token(message, field)
    for field in ("task_id", "workspace_id", "service_id"):
        if field in message:
            _require_token(message, field)
    for field in ("prepared_hash", "sha256", "capability_hash"):
        if field in message:
            value = message.get(field)
            if not isinstance(value, str) or not _HASH_RE.fullmatch(value):
                raise ProtocolError(f"control field {field!r} must be a lowercase SHA-256")
    for field in ("args", "prepared", "result", "diagnostic"):
        if field in message and not isinstance(message[field], dict):
            raise ProtocolError(f"control field {field!r} must be an object")
    for field in ("eof", "complete"):
        if field in message and not isinstance(message[field], bool):
            raise ProtocolError(f"control field {field!r} must be a boolean")
    for field in ("deadline_ms", "expires_ms", "size", "chunk_seq", "ack_seq"):
        if field in message:
            _require_integer(message, field)
    if kind in {"handshake", "handshake_ok"}:
        major = _require_integer(message, "protocol_major", maximum=65_535)
        minor = _require_integer(message, "protocol_minor", maximum=65_535)
        if not protocol_compatible(major, minor):
            raise ProtocolError(f"incompatible protocol version: {major}.{minor}")
    if kind == "handshake":
        nonce = message.get("nonce")
        if (
            not isinstance(nonce, str)
            or len(nonce) not in range(32, 129, 2)
            or re.fullmatch(r"[0-9a-f]+", nonce) is None
        ):
            raise ProtocolError("control field 'nonce' must be a 16..64 byte hex nonce")
    if kind == "handshake_ok":
        _require_token(message, "host_id")
    if kind == "prepare":
        tool = message.get("tool")
        if not isinstance(tool, str) or not _TOOL_RE.fullmatch(tool):
            raise ProtocolError("control field 'tool' must be a provider-safe tool name")
    if "completion" in message and message["completion"] not in _COMPLETION_STATES:
        raise ProtocolError("control completion state is invalid")
    if kind == "lease":
        _require_integer(message, "ttl_ms", minimum=1, maximum=MAX_LEASE_TTL_MS)
    for field in ("reason", "client_build", "platform", "build", "media_type"):
        if field in message and (not isinstance(message[field], str) or len(message[field].encode("utf-8")) > 4096):
            raise ProtocolError(f"control field {field!r} must be bounded text")


def validate_control_message(
    message: Any,
    *,
    peer_minor: int = PROTOCOL_MINOR,
) -> dict[str, Any]:
    """Validate one closed control shape; extensions live only in `optional`."""

    if not isinstance(message, dict):
        raise ProtocolError("control message must be an object")
    _validate_json_value(message)
    kind = message.get("kind")
    if not isinstance(kind, str) or kind not in _CONTROL_SHAPES:
        refuse_unknown_members(
            "wire_protocol",
            unknown=[kind],
            understood=_CONTROL_SHAPES,
            member="control message kinds",
            error_type=ProtocolContractError,
        )
    if not protocol_compatible(PROTOCOL_MAJOR, peer_minor):
        raise ProtocolError("incompatible control message minor version")
    required, known_optional = _CONTROL_SHAPES[kind]
    missing = (_COMMON_REQUIRED | required) - set(message)
    if missing:
        raise ProtocolError(f"control message {kind!r} is missing required fields: {sorted(missing)}")
    allowed = _COMMON_REQUIRED | required | known_optional | {"optional"}
    unknown = set(message) - allowed
    if unknown:
        refuse_unknown_members(
            "wire_protocol",
            unknown=unknown,
            understood=allowed,
            member=f"{kind} fields",
            error_type=ProtocolContractError,
        )
    optional = message.get("optional", {})
    if not isinstance(optional, dict):
        raise ProtocolError("control message optional field must be an object")
    _validate_control_values(message, kind)
    return dict(message)


def lease_answer_marker(lease_id: str, server_generation: str = "") -> dict[str, Any]:
    """The ``optional`` marker that makes an ``ack``/``diagnostic`` a LEASE answer.

    A lease renewal must be answerable — an accepted lease and a refused one look
    identical on a silent wire, and the refusal is the only observable form of "this
    generation does not own these process groups".  The answer reuses the EXISTING
    ack/diagnostic shapes rather than adding a control kind: an unknown kind fails the
    session by contract, while ``optional`` is the declared extension slot, so a peer
    that does not read this marker is unaffected by its presence.  It lives here, in
    the wire module, because both halves must agree on it and neither may import the
    other.
    """

    marker: dict[str, Any] = {"lease_id": str(lease_id)}
    if server_generation:
        marker["server_generation"] = str(server_generation)
    return {"lease": marker}


def lease_answer_id(message: Any) -> str:
    """The ``lease_id`` a control frame answers, or ``""`` when it answers no lease."""

    optional = message.get("optional") if isinstance(message, dict) else None
    lease = optional.get("lease") if isinstance(optional, dict) else None
    return str(lease.get("lease_id") or "") if isinstance(lease, dict) else ""


def session_preamble(
    nonce: bytes,
    *,
    protocol_major: int = PROTOCOL_MAJOR,
    protocol_minor: int = PROTOCOL_MINOR,
) -> bytes:
    if not isinstance(nonce, (bytes, bytearray, memoryview)):
        raise ProtocolError("session nonce must be bytes")
    nonce = bytes(nonce)
    if len(nonce) < 16 or len(nonce) > 64:
        raise ProtocolError("session nonce must be 16..64 bytes")
    if not protocol_compatible(protocol_major, protocol_minor):
        raise ProtocolError("session preamble protocol version is incompatible")
    return PREAMBLE_MAGIC + nonce.hex().encode("ascii") + f"\0{protocol_major}.{protocol_minor}\0".encode("ascii")


def parse_session_preamble(prefix: bytes, nonce: bytes) -> tuple[int, int, int]:
    if not isinstance(prefix, (bytes, bytearray, memoryview)):
        raise ProtocolError("execd preamble prefix must be bytes")
    if not isinstance(nonce, (bytes, bytearray, memoryview)):
        raise ProtocolError("session nonce must be bytes")
    nonce = bytes(nonce)
    if len(nonce) < 16 or len(nonce) > 64:
        raise ProtocolError("session nonce must be 16..64 bytes")
    bounded = bytes(prefix[:MAX_PREAMBLE_BYTES])
    first_magic = bounded.find(PREAMBLE_MAGIC)
    if first_magic < 0:
        raise ProtocolError("execd preamble not found within the bounded prefix")
    fixed = PREAMBLE_MAGIC + nonce.hex().encode("ascii") + b"\0"
    if not bounded.startswith(fixed, first_magic):
        raise ProtocolError("execd preamble nonce mismatch")
    version_start = first_magic + len(fixed)
    version_end = bounded.find(b"\0", version_start)
    if version_end < 0:
        raise ProtocolError("execd preamble exceeds the bounded prefix")
    try:
        major_text, minor_text = bounded[version_start:version_end].decode("ascii").split(".", 1)
        if not major_text.isdigit() or not minor_text.isdigit():
            raise ValueError("non-numeric version")
        major, minor = int(major_text), int(minor_text)
        if major_text != str(major) or minor_text != str(minor):
            raise ValueError("non-canonical version")
    except (UnicodeDecodeError, ValueError) as exc:
        raise ProtocolError("execd preamble protocol version is malformed") from exc
    if not protocol_compatible(major, minor):
        raise ProtocolError(f"execd preamble protocol version is incompatible: {major}.{minor}")
    consumed = version_end + 1
    if consumed > MAX_PREAMBLE_BYTES:
        raise ProtocolError("execd preamble exceeds the bounded prefix")
    return consumed, major, minor


def encode_control(message: dict[str, Any]) -> bytes:
    payload = canonical_json(
        validate_control_message(message),
        _max_bytes=MAX_CONTROL_BYTES,
    )
    return _HEADER.pack(b"J", len(payload)) + payload


def encode_bulk(payload: bytes) -> bytes:
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise ProtocolError("bulk frame payload must be bytes")
    size = payload.nbytes if isinstance(payload, memoryview) else len(payload)
    if size > MAX_BULK_BYTES:
        raise ProtocolError("bulk frame exceeds limit")
    data = bytes(payload)
    return _HEADER.pack(b"B", len(data)) + data


def read_frame(
    stream: BinaryIO,
    *,
    peer_minor: int = PROTOCOL_MINOR,
) -> tuple[str, dict[str, Any] | bytes]:
    header = _read_exact(stream, _HEADER.size, allow_clean_eof=True)
    kind, size = _HEADER.unpack(header)
    label = _KINDS.get(kind)
    if label is None:
        raise ProtocolError("unknown frame kind")
    limit = MAX_CONTROL_BYTES if label == "control" else MAX_BULK_BYTES
    if size > limit:
        raise ProtocolError(f"{label} frame exceeds limit")
    payload = _read_exact(stream, size)
    if label == "bulk":
        return label, payload
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_object_pairs,
            parse_constant=_reject_json_constant,
            parse_int=_parse_json_integer,
        )
    except ProtocolError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise ProtocolError(f"invalid control JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ProtocolError("control payload must decode to an object")
    if canonical_json(value) != payload:
        raise ProtocolError("control JSON must use the canonical encoding")
    return label, validate_control_message(value, peer_minor=peer_minor)


def _unique_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    raise ProtocolError(f"invalid JSON numeric constant: {value}")


def _parse_json_integer(value: str) -> int:
    if len(value.lstrip("-")) > 19:
        raise ProtocolError("JSON integer exceeds signed 64-bit range")
    parsed = int(value)
    if parsed < MIN_JSON_INTEGER or parsed > MAX_JSON_INTEGER:
        raise ProtocolError("JSON integer exceeds signed 64-bit range")
    return parsed


@dataclass
class BulkFrameCredit:
    """Non-blocking one-frame credit; transport owners decide how to await."""

    in_flight: tuple[str, int] | None = None

    def claim(self, blob_id: str, chunk_seq: int) -> None:
        if not isinstance(blob_id, str) or not _OPAQUE_ID_RE.fullmatch(blob_id):
            raise ProtocolError("bulk frame blob_id must be a file-safe opaque ID")
        if (
            not isinstance(chunk_seq, int)
            or isinstance(chunk_seq, bool)
            or chunk_seq < 0
            or chunk_seq > MAX_JSON_INTEGER
        ):
            raise ProtocolError("bulk frame chunk_seq must be a non-negative integer")
        if self.in_flight is not None:
            raise ProtocolError("bulk frame credit exhausted before ACK")
        self.in_flight = (blob_id, chunk_seq)

    def acknowledge(self, blob_id: str, chunk_seq: int) -> None:
        if (
            not isinstance(blob_id, str)
            or not _OPAQUE_ID_RE.fullmatch(blob_id)
            or not isinstance(chunk_seq, int)
            or isinstance(chunk_seq, bool)
            or chunk_seq < 0
            or chunk_seq > MAX_JSON_INTEGER
        ):
            raise ProtocolError("bulk ACK identity is invalid")
        expected = (blob_id, chunk_seq)
        if self.in_flight != expected:
            raise ProtocolError("bulk ACK does not match the in-flight frame")
        self.in_flight = None


@dataclass
class ControlSequence:
    """Receive-side monotonic sequence authority for one protocol direction."""

    next_seq: int = 0

    def observe(self, message: dict[str, Any]) -> None:
        seq = message.get("seq")
        if seq != self.next_seq:
            raise ProtocolError(f"control sequence mismatch: expected {self.next_seq}, got {seq!r}")
        self.next_seq += 1


def run_control_reader(
    stream: BinaryIO,
    *,
    on_control: Callable[[dict[str, Any]], None],
    bulk_queue: queue.Queue[bytes],
    sequence: ControlSequence | None = None,
    peer_minor: int = PROTOCOL_MINOR,
) -> None:
    """Keep control flowing while bulk consumers drain a separate one-slot queue."""

    if bulk_queue.maxsize != MAX_UNACKNOWLEDGED_BULK_FRAMES:
        raise ProtocolError("bulk reader queue must have exactly one frame of capacity")
    seq = sequence or ControlSequence()
    while True:
        label, payload = read_frame(stream, peer_minor=peer_minor)
        if label == "control":
            assert isinstance(payload, dict)
            seq.observe(payload)
            on_control(payload)
        else:
            assert isinstance(payload, bytes)
            try:
                bulk_queue.put_nowait(payload)
            except queue.Full as exc:
                raise ProtocolError("bulk reader queue exceeded one-frame credit") from exc


def _read_exact(
    stream: BinaryIO,
    size: int,
    *,
    allow_clean_eof: bool = False,
) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise ProtocolEOF(partial=bool(chunks) or not allow_clean_eof)
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)
