"""Every bound in the transport/execd path reports what it dropped.

`export_policy_contract.export_disclosure_block` set the shape: an exact count
beside a bounded list, so the presence of a bound is never mistaken for the absence
of anything beyond it. Four places applied a bare slice instead, and each one turns
a shortened answer into a confident wrong one:

* `execd._bound_envelope` — `artifacts[:128]`. Home received 128 rows and no hint
  that a 129th existed, while the complete set sat in the very blob the envelope
  points at.
* `execd_state.safe_details` — `list(value.items())[:64]`. A diagnostic is read to
  explain a failure; a silently shortened one makes the reader think they have it.
* `remote_ssh` alias warnings — `warning_directives[:4]`. The owner is being told
  which forwarding directives their alias declared; four with no note reads as all.
* `remote_ssh.list_directories` — `rows[:1000]`. An owner browsing for a remote
  project folder concludes the 1001st directory does not exist.

The last one also had a broken consumer: `gateway/connections.py` already read
`result.get("truncated")`, a key the transport never produced, so the handler fell
back to re-deriving truncation from its own 500-row slice and could never see a cut
the transport had already made. Setting the key it was asking for is what makes the
disclosure reach the owner.
"""

from __future__ import annotations

from ouroboros.execd_state import _MAX_SAFE_DETAIL_KEYS, safe_details


def test_a_bounded_detail_map_names_its_full_size():
    details = {f"key_{index}": index for index in range(_MAX_SAFE_DETAIL_KEYS + 10)}

    result = safe_details(details)

    assert result["_details_total"] == _MAX_SAFE_DETAIL_KEYS + 10
    assert result["_details_truncated"] is True
    payload = {key: value for key, value in result.items() if not key.startswith("_details_")}
    assert len(payload) == _MAX_SAFE_DETAIL_KEYS


def test_an_unbounded_detail_map_claims_no_truncation():
    result = safe_details({"reason": "disk full", "path": "/srv/p"})

    assert "_details_total" not in result
    assert "_details_truncated" not in result
    assert result["reason"] == "disk full"


def test_a_bounded_artifact_row_list_names_the_produced_count():
    """The envelope bound: 128 rows listed, the true count disclosed, and the text
    the model reads says where the rest are."""

    from ouroboros.execd import _MAX_ENVELOPE_ARTIFACT_ROWS, _bound_envelope

    class _CAS:
        def put(self, payload: bytes) -> str:
            return "a" * 64

    produced = _MAX_ENVELOPE_ARTIFACT_ROWS + 7
    envelope = {
        "text": "x" * 200,
        # Large enough that the envelope is externalized rather than returned as-is.
        "artifacts": [
            {"name": f"out-{index}.bin", "blob_id": "b" * 64, "size": 1, "filler": "y" * 4000}
            for index in range(produced)
        ],
        "trace": {},
    }

    bound = _bound_envelope(envelope, _CAS())

    assert bound["trace"]["artifact_rows_total"] == produced
    assert bound["trace"]["artifact_rows_listed"] == _MAX_ENVELOPE_ARTIFACT_ROWS
    assert bound["trace"]["artifact_rows_truncated"] is True
    # +1 for the operation-envelope.json ref the bound always appends.
    assert len(bound["artifacts"]) == _MAX_ENVELOPE_ARTIFACT_ROWS + 1
    assert "ARTIFACT_ROWS_TRUNCATED" in bound["text"]
    assert str(produced) in bound["text"]


def test_an_unbounded_artifact_row_list_claims_no_truncation():
    from ouroboros.execd import _bound_envelope

    class _CAS:
        def put(self, payload: bytes) -> str:
            return "a" * 64

    # Over MAX_CONTROL_BYTES // 2, so the envelope really is externalized.
    envelope = {
        "text": "z" * 100,
        "artifacts": [{"name": "one.bin", "filler": "q" * 900_000}],
        "trace": {},
    }

    bound = _bound_envelope(envelope, _CAS())

    assert bound["trace"]["artifact_rows_total"] == 1
    assert bound["trace"]["artifact_rows_truncated"] is False
    assert "ARTIFACT_ROWS_TRUNCATED" not in bound["text"]


def test_alias_forwarding_warnings_carry_the_full_directive_count():
    """The warning is owner-facing, so a bounded list must not read as complete.

    Asserted against the PRODUCTION construction rather than a rebuilt copy: the
    warning is assembled in `OpenSSHExecdTransport.__init__`, which needs a live SSH
    session to reach, and a test that rebuilds the dict it is checking proves only
    that the test can build a dict.

    BOUNDARY: this pins the exact SPELLING of three lines, so it is brittle in the safe
    direction (a reflow or a rename fails loudly) and blind in one direction that matters
    — if the assembly MOVES out of `__init__` into a helper, the three lines vanish from
    this scope and the gate reports a pass over code it can no longer see. The disclosure
    BEHAVIOUR is covered by the bounded-envelope cases above, which read the built payload;
    this gate exists only to keep the production construction from drifting.
    """

    import inspect

    from ouroboros import remote_ssh

    source = inspect.getsource(remote_ssh.OpenSSHExecdTransport.__init__)
    assert '"directives": list(warning_directives[:_MAX_DISCLOSED_DIRECTIVES])' in source
    assert '"directives_total": len(warning_directives)' in source
    assert "len(warning_directives) > _MAX_DISCLOSED_DIRECTIVES" in source


def test_no_bare_slice_survives_in_the_disclosure_sites():
    """A regression fence over the exact four lines this change fixed.

    Cheap and specific on purpose: a general "no slicing anywhere" rule would be
    noise, while these four are the ones a reviewer found returning a shortened
    answer that read as a complete one.

    BOUNDARY: a four-line fence, not a class gate. `[:127]`, `[: 128]`, `[:LIMIT]`,
    `islice(...)` and any NEW bare slice elsewhere all pass here. The class is covered by
    `tests/test_disclosure_elision_gate.py`, which polices every bounded collection slice
    in the feature's modules; this file keeps the four founding regressions nailed down.
    """

    import pathlib

    from ouroboros import execd, execd_state, remote_ssh

    banned = {
        pathlib.Path(execd.__file__): 'or [])[:128]',
        pathlib.Path(execd_state.__file__): "list(value.items())[:64]",
    }
    for path, snippet in banned.items():
        assert snippet not in path.read_text(encoding="utf-8"), f"{path.name}: {snippet}"

    transport = pathlib.Path(remote_ssh.__file__).read_text(encoding="utf-8")
    assert "rows[:1000]" not in transport
    assert "warning_directives[:4]" not in transport


def test_the_transport_sets_the_truncated_key_the_owner_handler_reads():
    """Grep-proof: the handler's read and the transport's write must agree.

    `api_connection_dirs` asks for `result["truncated"]`. If the transport stops
    producing that exact spelling the read goes dead again — silently, because
    `.get()` returns None and the handler falls back to its own slice.

    BOUNDARY: agreement is checked as TEXT, so it holds only for the literal spelling.
    `result["truncated"]`, a key read through a constant, and a `**payload` merge are all
    invisible to it, and either side reformatting its line fails the gate without anything
    being wrong. Accepted deliberately: the failure this closes is a SILENT divergence
    between one writer and one reader, and a loud false alarm is the cheaper error.
    """

    import inspect
    import pathlib

    from ouroboros import remote_ssh

    transport = inspect.getsource(remote_ssh.OpenSSHExecdTransport.list_directories)
    assert '"truncated": len(rows) > _MAX_LISTED_DIRS' in transport
    assert '"dirs_total": len(rows)' in transport

    handler = pathlib.Path(remote_ssh.__file__).with_name("gateway") / "connections.py"
    assert 'result.get("truncated")' in handler.read_text(encoding="utf-8")
