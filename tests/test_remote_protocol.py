import io
import os
import queue
import struct

import pytest

from ouroboros.remote_protocol import (
    MAX_BULK_BYTES,
    MAX_PREAMBLE_BYTES,
    MAX_UNACKNOWLEDGED_BULK_FRAMES,
    PREAMBLE_MAGIC,
    PROTOCOL_MAJOR,
    PROTOCOL_MINOR,
    BulkFrameCredit,
    ControlSequence,
    ProtocolEOF,
    ProtocolError,
    canonical_prepared_hash,
    encode_bulk,
    encode_control,
    parse_session_preamble,
    protocol_compatible,
    read_frame,
    run_control_reader,
    session_preamble,
)


def test_control_and_binary_frames_round_trip():
    control = {
        "kind": "prepare",
        "seq": 0,
        "request_id": "r1",
        "operation_id": "o1",
        "tool": "read_file",
        "args": {"path": "тест"},
    }
    assert read_frame(io.BytesIO(encode_control(control))) == ("control", control)
    payload = bytes(range(256)) * 128
    assert read_frame(io.BytesIO(encode_bulk(payload))) == ("bulk", payload)


def test_frame_limits_are_checked_before_payload_read():
    with pytest.raises(ProtocolError, match="bulk frame exceeds"):
        encode_bulk(b"x" * (MAX_BULK_BYTES + 1))
    hostile = struct.pack("!cI", b"B", MAX_BULK_BYTES + 1)
    with pytest.raises(ProtocolError, match="bulk frame exceeds"):
        read_frame(io.BytesIO(hostile))
    with pytest.raises(ProtocolError, match="payload must be bytes"):
        encode_bulk(3)  # type: ignore[arg-type]


def test_nonce_preamble_is_bounded_and_never_resynchronizes_after_false_magic():
    nonce = os.urandom(24)
    preamble = session_preamble(nonce)
    prefix = b"banner\n" + preamble + b"tail"
    assert parse_session_preamble(prefix, nonce)[0] == len(b"banner\n") + len(preamble)
    false = PREAMBLE_MAGIC + b"0" * (len(nonce) * 2) + b"\0"
    with pytest.raises(ProtocolError, match="nonce mismatch"):
        parse_session_preamble(false + preamble, nonce)
    with pytest.raises(ProtocolError, match="bounded prefix"):
        parse_session_preamble(b"x" * (MAX_PREAMBLE_BYTES + 1) + preamble, nonce)
    exact_banner = b"x" * (MAX_PREAMBLE_BYTES - len(preamble))
    assert parse_session_preamble(exact_banner + preamble, nonce)[0] == MAX_PREAMBLE_BYTES
    with pytest.raises(ProtocolError, match="bounded prefix"):
        parse_session_preamble(b"x" + exact_banner + preamble, nonce)
    assert parse_session_preamble(preamble, nonce)[1:] == (
        PROTOCOL_MAJOR,
        PROTOCOL_MINOR,
    )
    # Derived from the constants, never spelled: the wire minor IS
    # `remote_contracts.CONTRACT_SET_VERSION` and moves whenever a Home↔execd contract
    # changes shape, so a literal `1.0` here silently made this a no-op replace — the
    # non-canonical-version arm stopped being exercised at the first bump.
    version = f"{PROTOCOL_MAJOR}.{PROTOCOL_MINOR}".encode("ascii")
    padded = f"0{PROTOCOL_MAJOR}.0{PROTOCOL_MINOR}".encode("ascii")
    noncanonical = preamble.replace(b"\0" + version + b"\0", b"\0" + padded + b"\0")
    assert noncanonical != preamble, "the version substitution must really substitute"
    with pytest.raises(ProtocolError, match="malformed"):
        parse_session_preamble(noncanonical, nonce)
    with pytest.raises(ProtocolError, match="nonce must be bytes"):
        session_preamble("not-bytes")  # type: ignore[arg-type]


def test_prepared_hash_is_order_independent_but_value_sensitive():
    left = canonical_prepared_hash({"tool": "run_command", "args": {"b": 2, "a": 1}})
    right = canonical_prepared_hash({"args": {"a": 1, "b": 2}, "tool": "run_command"})
    assert left == right
    assert left != canonical_prepared_hash({"tool": "run_command", "args": {"a": 1, "b": 3}})
    assert MAX_UNACKNOWLEDGED_BULK_FRAMES == 1


def test_control_codec_rejects_noncanonical_duplicate_and_unbounded_values():
    noncanonical = b'{"seq":0, "kind":"panic","server_generation":"g"}'
    frame = struct.pack("!cI", b"J", len(noncanonical)) + noncanonical
    with pytest.raises(ProtocolError, match="canonical"):
        read_frame(io.BytesIO(frame))
    duplicate = b'{"kind":"panic","kind":"panic","seq":0,"server_generation":"g"}'
    frame = struct.pack("!cI", b"J", len(duplicate)) + duplicate
    with pytest.raises(ProtocolError, match="duplicate"):
        read_frame(io.BytesIO(frame))
    with pytest.raises(ProtocolError, match="floating-point"):
        encode_control({"kind": "panic", "seq": 0, "server_generation": 1.5})
    nested: object = "leaf"
    for _ in range(40):
        nested = [nested]
    with pytest.raises(ProtocolError, match="depth"):
        canonical_prepared_hash(nested)


def test_control_shapes_versions_sequences_and_bulk_credit_fail_closed():
    assert protocol_compatible(1, 0)
    assert not protocol_compatible(2, 0)
    assert not protocol_compatible("1", 0)
    assert not protocol_compatible(True, False)
    with pytest.raises(ProtocolError, match="unknown control message kind"):
        encode_control({"kind": "future_required", "seq": 0})
    # A wire-contract drift refusal now names the contract, the member and the owner's
    # action (`remote_contracts.refuse_unknown_members`) instead of only saying that
    # something was unrecognized. Still a `ProtocolError`, so the transport tears the
    # session down by exactly the same path as any other malformed frame.
    with pytest.raises(ProtocolError, match="unknown panic fields"):
        encode_control(
            {
                "kind": "panic",
                "seq": 0,
                "server_generation": "g",
                "must_understand": True,
            }
        )
    sequence = ControlSequence()
    sequence.observe({"seq": 0})
    with pytest.raises(ProtocolError, match="sequence mismatch"):
        sequence.observe({"seq": 2})
    credit = BulkFrameCredit()
    credit.claim("blob", 0)
    with pytest.raises(ProtocolError, match="exhausted"):
        credit.claim("blob", 1)
    with pytest.raises(ProtocolError, match="does not match"):
        credit.acknowledge("blob", 1)
    credit.acknowledge("blob", 0)
    assert credit.in_flight is None
    with pytest.raises(ProtocolError, match="file-safe opaque ID"):
        credit.claim(1, 0)  # type: ignore[arg-type]
    for hostile_id in (".", "..", "../op", "../../blob", "a/b"):
        with pytest.raises(ProtocolError, match="file-safe opaque ID"):
            BulkFrameCredit().claim(hostile_id, 0)
        with pytest.raises(ProtocolError, match="file-safe opaque ID"):
            encode_control(
                {
                    "kind": "prepare",
                    "seq": 0,
                    "request_id": hostile_id,
                    "operation_id": "op",
                    "tool": "read_file",
                    "args": {},
                }
            )


def test_control_size_budget_stops_streaming_encoding_before_full_materialization():
    huge_but_individually_valid = {
        "kind": "diagnostic",
        "seq": 0,
        "request_id": "request",
        "operation_id": "operation",
        "diagnostic": {f"k{index}": "x" * 262_144 for index in range(8)},
    }
    with pytest.raises(ProtocolError, match="control frame exceeds limit"):
        encode_control(huge_but_individually_valid)


@pytest.mark.parametrize(
    "message",
    [
        {"kind": "handshake", "seq": 0, "nonce": "00" * 16, "protocol_major": "1", "protocol_minor": 0},
        {"kind": "handshake", "seq": 0, "nonce": "00" * 16, "protocol_major": 1, "protocol_minor": -1},
        {"kind": "panic", "seq": 0, "server_generation": None},
        {"kind": "prepare", "seq": 0, "request_id": [], "operation_id": "o", "tool": "read_file", "args": {}},
        {"kind": "blob_manifest", "seq": 0, "request_id": "r", "blob_id": "b", "size": -1, "sha256": "no"},
        {"kind": "lease", "seq": 0, "server_generation": "g", "lease_id": "l", "ttl_ms": 15001},
        {"kind": "result", "seq": 0, "request_id": "r", "operation_id": "o", "completion": "maybe", "result": {}},
    ],
)
def test_control_security_fields_have_exact_types_and_ranges(message):
    with pytest.raises(ProtocolError):
        encode_control(message)


def test_decoder_wraps_huge_integer_and_depth_as_protocol_errors():
    huge = b'{"kind":"panic","seq":' + b"9" * 10_000 + b',"server_generation":"g"}'
    frame = struct.pack("!cI", b"J", len(huge)) + huge
    with pytest.raises(ProtocolError, match="integer"):
        read_frame(io.BytesIO(frame))
    deep = b'{"kind":"panic","optional":' + b"[" * 1100 + b"0" + b"]" * 1100 + b',"seq":0,"server_generation":"g"}'
    frame = struct.pack("!cI", b"J", len(deep)) + deep
    with pytest.raises(ProtocolError):
        read_frame(io.BytesIO(frame))


def test_reader_distinguishes_clean_eof_from_partial_and_does_not_block_panic_on_bulk():
    with pytest.raises(ProtocolEOF) as clean:
        read_frame(io.BytesIO(b""))
    assert clean.value.partial is False
    with pytest.raises(ProtocolEOF) as partial:
        read_frame(io.BytesIO(b"J\x00"))
    assert partial.value.partial is True

    panic = encode_control({"kind": "panic", "seq": 0, "server_generation": "generation-1"})
    stream = io.BytesIO(encode_bulk(b"backpressured") + panic)
    bulk = queue.Queue(maxsize=1)
    controls = []
    with pytest.raises(ProtocolEOF) as ended:
        run_control_reader(stream, on_control=controls.append, bulk_queue=bulk)
    assert ended.value.partial is False
    assert controls == [{"kind": "panic", "seq": 0, "server_generation": "generation-1"}]
    assert bulk.get_nowait() == b"backpressured"



# NOTE (RWS v2 Lane 1, stage A): the donor file additionally contains three
# OpenSSH transport-session tests —
#   test_reconnect_wire_reset_retains_operations_but_clears_session_state
#   test_reconnect_keeps_a_healthy_transport_and_only_reconciles
#   test_panic_pipe_write_is_portable_and_cleanup_is_unconditional
# — which exercise ouroboros.remote_ssh.  They transfer together with
# remote_ssh.py in the transport-session stage; the wire-protocol tests above
# are byte-identical to the donor's.
