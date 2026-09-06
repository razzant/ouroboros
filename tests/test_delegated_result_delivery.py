"""A large delegated result is delivered whole or declared partial.

Split verbatim out of ``tests/test_delegated_subagent_transport.py`` by theme. This
module owns the staged result artifact, the coverage acknowledgement bound to what
delivery actually hands the model, and the disclosure that follows an unread or
truncated output.
"""

from __future__ import annotations

import json
import pathlib

from tests._delegated_transport_shared import (  # noqa: F401  (autouse fixture applies on import)
    _LiveRunStub,
    _event_types,
    _nanny_ctx,
    _owned_gateway_uses_each_test_transport,
)


def test_a_large_delegated_result_is_delivered_whole_or_declared_partial(tmp_path, monkeypatch):
    """`final_summary`/`primary_output` carry the run's real work product and Claudexor
    returns up to 256 KiB. The 15k head-truncation cut it mid-string and destroyed the
    JSON, so a large review came back as an unparseable fragment that still looked like
    a verdict. The payload now bounds ITSELF and the remainder is a readable artifact."""
    from ouroboros.loop_tool_execution import _truncate_tool_result
    from ouroboros.tool_capabilities import tool_result_limit

    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways import claudexor as gw

    verdict = "V" * 120_000

    class _Stub(_LiveRunStub):
        def get_run(self, rid, **_kw):
            return {"lastSeq": 9, "primaryOutput": verdict,
                    "finalSummary": "S" * 60_000,
                    "outcomeBanner": "B" * 40_000,
                    "summary": {"state": "succeeded", "spendUsd": 0.0}}

    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _Stub())
    delegate._CUSTODY.clear()
    delegate._CUSTODY["run-1"] = delegate._RunCustody(
        run_id="run-1", task_id="t-a", route_id="r", model="m", project_id="p", project_owned=False)
    ctx = _nanny_ctx(tmp_path)
    raw = delegate._delegate_wait(ctx, "run-1", wait_sec=1)
    delegate._CUSTODY.clear()

    limit = tool_result_limit("delegate_wait")
    assert len(raw) <= limit, "the producer must fit the budget the truncator applies"
    assert _truncate_tool_result(raw, "delegate_wait", {}) == raw, "outer truncation must not fire"
    payload = json.loads(raw)          # the fatal symptom: this used to be unparseable

    delivery = payload["output_delivery"]
    assert delivery["complete"] is False and delivery["consumed"] is False
    assert "primary_output" not in payload, "a preview must not wear the whole field's name"
    assert payload["primary_output_preview"] and payload["primary_output_preview"] in verdict

    artifact = delivery["artifact"]
    assert artifact["root"] == "task_drive"
    staged = pathlib.Path(artifact["abs_path"]).read_text(encoding="utf-8")
    assert json.loads(staged)["primary_output"] == verdict, "the whole result must survive"
    assert delivery["read_next"]["tool"] == "read_file"

    # The advertised chunk read really works, with a stable cursor over an immutable
    # file — and it works for the READ-ONLY nanny, which is the common caller and the
    # one whose access policy could have made the whole contract unreachable.
    from ouroboros.tool_access import LOCAL_READONLY_SUBAGENT_MODE
    from ouroboros.tools.core import _read_file
    from ouroboros.contracts.task_constraint import TaskConstraint

    ctx.task_constraint = TaskConstraint(mode=LOCAL_READONLY_SUBAGENT_MODE)
    head = _read_file(ctx, path=artifact["path"], root="task_drive", start_line=1, max_lines=5)
    tail = _read_file(ctx, path=artifact["path"], root="task_drive",
                      start_line=artifact["lines"], max_lines=5)
    assert "BLOCKED" not in head and "NOT_FOUND" not in head and "ERROR" not in head
    assert head != tail, "start_line must be a real cursor, not a no-op"


def _read_artifact_whole(ctx, artifact, step=7):
    """Cover the staged artifact contiguously, like a real reader: line windows, plus
    the start_char sub-line cursor for any line longer than the delivery budget (a cut
    window only credits the delivered prefix)."""
    from ouroboros.tool_capabilities import tool_result_limit
    from ouroboros.tools.core import _read_file

    stride = tool_result_limit("read_file") - 5_000
    lines = pathlib.Path(artifact["abs_path"]).read_text(encoding="utf-8").splitlines(keepends=True)
    for line_no, line in enumerate(lines, start=1):
        offset = 0
        while offset == 0 or offset < len(line):
            _read_file(ctx, path=artifact["path"], root="task_drive",
                       start_line=line_no, max_lines=1, start_char=offset)
            offset += stride


def test_the_coverage_ack_binds_to_what_delivery_actually_hands_the_model(
        tmp_path, monkeypatch):
    """P34R.7 (scope reviewer, p34.part2 gate) claimed the ack credits characters the
    delivery layer cuts, because it runs before _annotate_reread and the 80K cap. The
    executed probe REFUTED it: the reread note is APPENDED and the outer truncator
    KEEPS THE HEAD (s[:limit]), so the note can only lose its own tail — it never
    displaces body characters — and the ack's budget math mirrors the real truncator
    to the character. This test PINS that equivalence on the real seam (tool ->
    annotation -> real _truncate_tool_result), so a future reordering — prepending
    the note, a tail-keep truncator, a second budget constant — cannot silently turn
    the rejected finding true: on every shape, the interval the ack credits must not
    exceed the window-body characters actually present in the delivered string."""
    import ouroboros.delegate_custody as dc
    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways import claudexor as gw
    from ouroboros.loop_tool_execution import _truncate_tool_result
    from ouroboros.tool_capabilities import tool_result_limit
    from ouroboros.tools.core import _read_file

    budget = tool_result_limit("read_file")

    class _Stub(_LiveRunStub):
        def get_run(self, rid, **_kw):
            return {"lastSeq": 9, "primaryOutput": "V" * (budget * 2),
                    "summary": {"state": "succeeded", "spendUsd": 0.0}}

    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _Stub())
    delegate._CUSTODY.clear()
    delegate._READ_COVERAGE.clear()
    dc.record_started(tmp_path, delegate._RunCustody(
        run_id="run-1", task_id="t-a", route_id="r", model="m",
        project_id="p", project_owned=False, root_task_id="t-a", ledger_root=str(tmp_path)))
    ctx = _nanny_ctx(tmp_path)
    artifact = json.loads(delegate._delegate_wait(ctx, "run-1", wait_sec=1)
                          )["output_delivery"]["artifact"]
    content = pathlib.Path(artifact["abs_path"]).read_text(encoding="utf-8")
    import hashlib as _hl
    identity = (f"{pathlib.Path(artifact['abs_path']).resolve()}|"
                f"{_hl.sha256(content.encode('utf-8', 'replace')).hexdigest()}")
    lines = content.splitlines(keepends=True)
    long_no, long_line = max(enumerate(lines, start=1), key=lambda p: len(p[1]))
    assert len(long_line) > budget + 1000

    def delivered_body(delivered, window_body, hdr):
        if hdr not in delivered:
            return 0
        after = delivered.split(hdr, 1)[1]
        lo, hi, best = 0, min(len(after), len(window_body)), 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if after.startswith(window_body[:mid]):
                best, lo = mid, mid + 1
            else:
                hi = mid - 1
        return best

    def call(start_char):
        before = sum(b - a for a, b in delegate._READ_COVERAGE.get(identity, []))
        result = _read_file(ctx, path=artifact["path"], root="task_drive",
                            start_line=long_no, max_lines=1, start_char=start_char)
        delivered = _truncate_tool_result(result, "read_file",
                                          {"path": artifact["path"], "root": "task_drive"})
        after = sum(b - a for a, b in delegate._READ_COVERAGE.get(identity, []))
        hdr = result.split("\n", 1)[0] + "\n"
        return result, delivered_body(delivered, long_line[start_char:], hdr), after - before

    # Shape A: rendering just under the budget; the repeat's appended note pushes the
    # annotated result over it — the rejected finding's exact scenario.
    offset = len(long_line) - (budget - 200)
    r1, d1, c1 = call(offset)
    assert len(r1) <= budget and c1 <= d1, (c1, d1)
    r2, d2, c2 = call(offset)
    assert len(r2) > budget, "the annotated repeat must exceed the budget here"
    assert c2 <= max(0, d2), (c2, d2)
    assert d2 == d1, "an appended note must never displace delivered body characters"

    # Shape B: the rendering alone exceeds the budget; ack == the truncator's cut.
    delegate._READ_COVERAGE.clear()
    r3, d3, c3 = call(0)
    assert len(r3) > budget and c3 == d3, (c3, d3)
    r4, d4, c4 = call(0)
    assert c4 <= max(0, d4) and d4 == d3, (c4, d4, d3)
    delegate._CUSTODY.clear()
    delegate._READ_COVERAGE.clear()


def test_reading_the_staged_artifact_whole_writes_the_canonical_acknowledgement(
        tmp_path, monkeypatch):
    """Owner doctrine D7: a delegated result is OBTAINED only after the artifact is
    read to EOF — meaning proven CONTINUOUS coverage from the first line to the last,
    not a cursor that merely touched the end. The canonical acknowledgement is a typed
    row written exactly when the windows have covered the whole artifact — carrying the
    byte length and hash of what was staged — written once, replayed across restarts,
    and surfaced on a re-wait. It gates NOTHING: partial reads still work, full reads
    still work, the only change is that the record can now tell the two apart."""
    import hashlib

    import ouroboros.delegate_custody as dc
    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways import claudexor as gw
    from ouroboros.tools.core import _read_file

    class _Stub(_LiveRunStub):
        def get_run(self, rid, **_kw):
            return {"lastSeq": 9, "primaryOutput": "V" * 120_000,
                    "summary": {"state": "succeeded", "spendUsd": 0.0}}

    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _Stub())
    delegate._CUSTODY.clear()
    delegate._READ_COVERAGE.clear()
    dc.record_started(tmp_path, delegate._RunCustody(
        run_id="run-1", task_id="t-a", route_id="r", model="m",
        project_id="p", project_owned=False, root_task_id="t-a", ledger_root=str(tmp_path)))
    ctx = _nanny_ctx(tmp_path)

    first = json.loads(delegate._delegate_wait(ctx, "run-1", wait_sec=1))
    artifact = first["output_delivery"]["artifact"]
    assert first["output_delivery"]["consumed"] is False
    assert "delegate_run_output_consumed" not in _event_types(tmp_path)
    spilled = [json.loads(l) for l in (tmp_path / "logs" / "events.jsonl").read_text().splitlines()
               if '"delegate_run_output_spilled"' in l]
    assert spilled and spilled[-1]["sha256"] == artifact["sha256"], \
        "the staged fact must durably carry what was staged"
    assert spilled[-1]["full_content"] is True

    # A head read is served in full and acknowledges nothing.
    head = _read_file(ctx, path=artifact["path"], root="task_drive", start_line=1, max_lines=5)
    assert "BLOCKED" not in head and "ERROR" not in head
    assert "delegate_run_output_consumed" not in _event_types(tmp_path)

    # THE NEGATIVE THAT DEFINES THE CONTRACT: a tail window whose end touches EOF, with
    # the middle never read, is NOT full reading and must not acknowledge. (The first
    # cut of this feature acknowledged exactly this shape.)
    tail = _read_file(ctx, path=artifact["path"], root="task_drive",
                      start_line=artifact["lines"], max_lines=5)
    assert "BLOCKED" not in tail and "ERROR" not in tail
    assert "delegate_run_output_consumed" not in _event_types(tmp_path), \
        "head+tail with a skipped middle must never acknowledge"
    gap_wait = json.loads(delegate._delegate_wait(ctx, "run-1", wait_sec=1))
    assert gap_wait["output_delivery"]["consumed"] is False

    # Filling the gap — contiguous coverage of every line — IS the acknowledgement.
    _read_artifact_whole(ctx, artifact)
    rows = [json.loads(l) for l in (tmp_path / "logs" / "events.jsonl").read_text().splitlines()
            if '"delegate_run_output_consumed"' in l]
    assert len(rows) == 1, "the acknowledgement is canonical: one row, not one per read"
    staged_bytes = pathlib.Path(artifact["abs_path"]).read_bytes()
    assert rows[0]["run_id"] == "run-1"
    assert rows[0]["bytes"] == len(staged_bytes) == artifact["bytes"]
    assert rows[0]["sha256"] == hashlib.sha256(staged_bytes).hexdigest() == artifact["sha256"]
    assert rows[0]["lines"] == artifact["lines"]

    # Reading it whole again does not write a second acknowledgement.
    _read_artifact_whole(ctx, artifact)
    assert sum(1 for t in _event_types(tmp_path) if t == "delegate_run_output_consumed") == 1

    # A re-wait on the terminal run now reports the durable fact in its disposition.
    second = json.loads(delegate._delegate_wait(ctx, "run-1", wait_sec=1))
    assert second["output_delivery"]["consumed"] is True

    # The fact survives a worker restart, like every other custody fact.
    delegate._CUSTODY.clear()
    replayed = dc.replay(tmp_path)["run-1"]
    assert replayed.output_consumed is True
    assert replayed.output_complete is True
    assert replayed.output_artifact == artifact["path"]
    delegate._CUSTODY.clear()


def test_the_staged_artifact_is_the_bytes_it_declares_even_under_a_translating_text_layer(
    tmp_path, monkeypatch,
):
    """The artifact's sha256 IS its identity — `custody.output_sha` — and the read
    receipt measures the file with `read_bytes`, so the declared bytes and the written
    bytes have to be one object. Staging used a TEXT write, whose `newline=None` layer
    translates every "\\n" to `os.linesep`: on Windows the payload (always
    `json.dumps(..., indent=2)`, so always multi-line) landed as CRLF while the
    published hash described the LF form, `record_output_consumed` refused on the
    mismatch, and the D7 acknowledgement could never be written for any delegated run —
    every result stayed "settled but NOT COLLECTED" forever.

    Runnable anywhere: the platform's text layer is emulated by a translating
    `Path.write_text`, which the fixed code simply never calls. Reverting to a text
    write brings the failure back on POSIX too, which is the point.
    """
    import hashlib

    import ouroboros.delegate_custody as dc
    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways import claudexor as gw

    class _Stub(_LiveRunStub):
        def get_run(self, rid, **_kw):
            return {"lastSeq": 9, "primaryOutput": "V" * 120_000,
                    "summary": {"state": "succeeded", "spendUsd": 0.0}}

    real_write_text = pathlib.Path.write_text

    def _windows_shaped_write_text(self, data, *args, **kwargs):
        return real_write_text(self, str(data).replace("\n", "\r\n"), *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "write_text", _windows_shaped_write_text)
    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _Stub())
    delegate._CUSTODY.clear()
    delegate._READ_COVERAGE.clear()
    dc.record_started(tmp_path, delegate._RunCustody(
        run_id="run-1", task_id="t-a", route_id="r", model="m",
        project_id="p", project_owned=False, root_task_id="t-a", ledger_root=str(tmp_path)))
    ctx = _nanny_ctx(tmp_path)

    out = json.loads(delegate._delegate_wait(ctx, "run-1", wait_sec=1))
    artifact = out["output_delivery"]["artifact"]
    on_disk = pathlib.Path(artifact["abs_path"]).read_bytes()
    assert b"\r\n" not in on_disk, "the staged payload is bytes, not translated text"
    assert len(on_disk) == artifact["bytes"]
    assert hashlib.sha256(on_disk).hexdigest() == artifact["sha256"]

    # ...and therefore the acknowledgement can actually land.
    _read_artifact_whole(ctx, artifact)
    assert sum(1 for t in _event_types(tmp_path) if t == "delegate_run_output_consumed") == 1
    assert dc.settled_unread_outputs(tmp_path) == []
    delegate._CUSTODY.clear()


def test_an_unread_result_is_a_loud_durable_fact_at_settlement(tmp_path, monkeypatch):
    """Owner directive: full-output consumption must be LOAD-BEARING before settlement.
    Until now the D7 acknowledgement was pure disclosure — the module said so in words,
    'nothing anywhere blocks on its absence' — so a delegated result could be paid for
    and never collected with nothing but a boolean field to notice it.

    WHY NOT A HARD GATE (the (a) option), proven by the call order right here:
    `delegate_wait` SETTLES and only then builds the payload that STAGES the artifact.
    Refusing to settle until the read happened would refuse the step that creates the
    thing to read, and would hold back the LEDGER ROW for money already spent; cancelled
    and failed runs commonly have no output at all and would strand in `open_runs`
    forever. So (b): the money settles immediately and the OMISSION becomes a typed
    durable fact on three surfaces — the settlement row, the parent's result, and the
    health invariants — self-clearing the moment the read lands."""
    import ouroboros.delegate_custody as dc
    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways import claudexor as gw

    class _Huge(_LiveRunStub):
        def get_run(self, rid, **_kw):
            return {"lastSeq": 9, "primaryOutput": "V" * 120_000,
                    "summary": {"state": "succeeded", "spendUsd": 0.0,
                                "effectiveAccess": "readonly"}}

    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _Huge())
    delegate._CUSTODY.clear()
    delegate._READ_COVERAGE.clear()
    dc.record_started(tmp_path, delegate._RunCustody(
        run_id="run-live", task_id="t-a", route_id="r", model="m",
        project_id="p", project_owned=False, root_task_id="t-a", ledger_root=str(tmp_path)))
    ctx = _nanny_ctx(tmp_path)

    first = json.loads(delegate._delegate_wait(ctx, "run-live", wait_sec=1))
    artifact = first["output_delivery"]["artifact"]

    # 1. The money settled — never held hostage to a disclosure.
    assert first["settlement"]["settled"] is True
    assert first["settlement"]["ledger_recorded"] is True
    # 2. ...and the omission is named, on the settlement row AND in words to the parent.
    assert "NOT COLLECTED" in first["result_not_collected"]
    assert "delegate_run_settled_unread" in _event_types(tmp_path)
    # 3. ...and it stays visible until the read happens.
    unread = dc.settled_unread_outputs(tmp_path)
    assert [c.run_id for c in unread] == ["run-live"]

    # ONCE PER RUN, not once per poll: a re-wait on an already settled run must not
    # append a second identical omission row (which would read as a second omission),
    # while still telling the parent the result is STILL not collected.
    repeat = json.loads(delegate._delegate_wait(ctx, "run-live", wait_sec=1))
    assert "NOT COLLECTED" in repeat["result_not_collected"]
    assert sum(1 for t in _event_types(tmp_path) if t == "delegate_run_settled_unread") == 1

    # It survives the worker that settled it: the fact is durable, not process-local —
    # and a restarted worker does not repeat the row either, because the flag replays.
    delegate._CUSTODY.clear()
    assert [c.run_id for c in dc.settled_unread_outputs(tmp_path)] == ["run-live"]
    restarted = json.loads(delegate._delegate_wait(ctx, "run-live", wait_sec=1))
    assert "NOT COLLECTED" in restarted["result_not_collected"]
    assert sum(1 for t in _event_types(tmp_path) if t == "delegate_run_settled_unread") == 1

    # THE READ CLEARS IT, on every surface, with no second settlement needed.
    _read_artifact_whole(ctx, artifact)
    assert dc.settled_unread_outputs(tmp_path) == []
    again = json.loads(delegate._delegate_wait(ctx, "run-live", wait_sec=1))
    assert "result_not_collected" not in again, "a collected result must stop nagging"
    assert again["output_delivery"]["consumed"] is True

    # NEGATIVE HALVES — the shapes that must never owe this, or the fact becomes noise
    # and legitimate flows deadlock on a warning they cannot discharge:
    #   (a) a run whose payload fit INLINE staged nothing;
    inline = dc.RunCustody(run_id="r-inline", task_id="t-a", settled=True)
    assert dc.settled_output_unread(inline) is False
    #   (b) a run whose staged content was only a PREVIEW was never acknowledgeable;
    preview = dc.RunCustody(run_id="r-prev", task_id="t-a", settled=True,
                            output_artifact="delegated_runs/r-prev.json",
                            output_complete=False)
    assert dc.settled_output_unread(preview) is False
    #   (c) a run that is not settled yet owes nothing here (it is still in flight).
    live = dc.RunCustody(run_id="r-live", task_id="t-a", settled=False,
                         output_artifact="delegated_runs/r-live.json",
                         output_complete=True)
    assert dc.settled_output_unread(live) is False
    delegate._CUSTODY.clear()
    delegate._READ_COVERAGE.clear()


def test_no_post_fires_when_the_start_request_row_did_not_land(tmp_path, monkeypatch):
    """Codex audit, claim 2, proven by run before fixing: with the event-log append
    failing, the POST still fired and the run started with NO durable request row --
    a worker death before record_started would leave a live overpowered run that
    nothing durable names. The POST is now conditional on the row landing: a broken
    event log refuses the start, typed, with the created registration retired."""
    import ouroboros.delegate_custody as dc
    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways import claudexor as gw

    posts = []

    class _Stub(_LiveRunStub):
        def start_run(self, request, *, idempotency_key=""):
            posts.append(idempotency_key)
            return {"runId": "run-1"}

    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "some-route=weak-model:low")
    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _Stub())
    real_append = dc.append_jsonl

    def broken_append(path, row):
        if row.get("type") == "delegate_run_start_requested":
            return False               # append_jsonl's own "did not land" signal
        return real_append(path, row)

    monkeypatch.setattr(dc, "append_jsonl", broken_append)
    delegate._CUSTODY.clear()
    out = json.loads(delegate._delegate_start(_nanny_ctx(tmp_path), "do the work"))
    delegate._CUSTODY.clear()
    assert out["status"] == "refused"
    assert out["reason"] == "start_request_row_unwritable"
    assert out["definitely_unrun"] is True
    assert posts == [], "POST is conditional on the durable request row"
    assert "delegate_run_started" not in _event_types(tmp_path)


def test_a_line_the_delivery_layer_cut_is_not_covered(tmp_path, monkeypatch):
    """Codex audit, claim 1: coverage must bind to what the DELIVERY layer actually
    hands the model, not to source-file line ranges. read_file's result is cut at
    tool_result_limit("read_file") by the outer truncator, so a single line longer
    than that budget renders a window the model only ever sees the head of. Crediting
    the whole line marked an artifact fully read while ~40K chars never reached the
    model. The cut remainder is reachable — and only creditable — through start_char,
    the sub-line cursor."""
    import ouroboros.delegate_custody as dc
    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways import claudexor as gw
    from ouroboros.tool_capabilities import UNTRUNCATED_TOOL_RESULTS, tool_result_limit
    from ouroboros.tools.core import _read_file

    # The premise the whole test rests on: these reads ARE outer-truncated.
    assert "read_file" not in UNTRUNCATED_TOOL_RESULTS
    budget = tool_result_limit("read_file")

    class _Stub(_LiveRunStub):
        def get_run(self, rid, **_kw):
            # One ~120K-char JSON line in the staged artifact: longer than any
            # deliverable read_file window.
            return {"lastSeq": 9, "primaryOutput": "V" * 120_000,
                    "summary": {"state": "succeeded", "spendUsd": 0.0}}

    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _Stub())
    delegate._CUSTODY.clear()
    delegate._READ_COVERAGE.clear()
    dc.record_started(tmp_path, delegate._RunCustody(
        run_id="run-1", task_id="t-a", route_id="r", model="m",
        project_id="p", project_owned=False, root_task_id="t-a", ledger_root=str(tmp_path)))
    ctx = _nanny_ctx(tmp_path)
    first = json.loads(delegate._delegate_wait(ctx, "run-1", wait_sec=1))
    artifact = first["output_delivery"]["artifact"]

    # THE NEGATIVE CODEX NAMES: a full line-window sweep — the pre-fix notion of
    # "whole file", no sub-line cursor — must NOT acknowledge, because the long
    # line's window is cut at delivery and the model never received its tail.
    line = 1
    while line <= artifact["lines"]:
        _read_file(ctx, path=artifact["path"], root="task_drive",
                   start_line=line, max_lines=7)
        line += 7
    assert "delegate_run_output_consumed" not in _event_types(tmp_path), \
        "a line the delivery layer cut is NOT covered"
    delegate._CUSTODY.clear()
    assert dc.replay(tmp_path)["run-1"].output_consumed is False

    # The remainder is reachable through the sub-line cursor, and only DELIVERED
    # chunks accumulate: advancing start_char across the long line completes coverage.
    staged_lines = pathlib.Path(artifact["abs_path"]).read_text(encoding="utf-8").splitlines(keepends=True)
    stride = budget - 5_000                     # safely below any delivered body size
    for line_no, line in enumerate(staged_lines, start=1):
        offset = 0
        while offset < len(line):
            view = _read_file(ctx, path=artifact["path"], root="task_drive",
                              start_line=line_no, max_lines=1, start_char=offset)
            if offset:
                assert f"(from char {offset} of this window)" in view.splitlines()[0], \
                    "the sub-line cursor must be disclosed in the header"
            offset += stride
    assert sum(1 for t in _event_types(tmp_path) if t == "delegate_run_output_consumed") == 1, \
        "delivered-chunk coverage of every character is the acknowledgement"
    delegate._CUSTODY.clear()


def test_a_restaged_different_artifact_does_not_inherit_the_old_acknowledgement(
        tmp_path, monkeypatch):
    """Codex audit, claim 5, proven by run before fixing: after a full read + ack of
    artifact A, a re-wait re-staged DIFFERENT bytes at the same path and the delivery
    still said consumed:true — the old ack transferred by PATH to content never read.
    The ack is hash-bound now: a re-stage with a different sha resets consumed (in
    process and in replay), the new content owes its own full read, and a second
    acknowledgement row for the new bytes is legitimate."""
    import ouroboros.delegate_custody as dc
    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways import claudexor as gw

    class _Stub(_LiveRunStub):
        payload = "A" * 30_000 + "\n" + ("x\n" * 200)
        def get_run(self, rid, **_kw):
            return {"lastSeq": 9, "primaryOutput": self.payload,
                    "summary": {"state": "succeeded", "spendUsd": 0.0}}

    stub = _Stub()
    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: stub)
    delegate._CUSTODY.clear()
    delegate._READ_COVERAGE.clear()
    dc.record_started(tmp_path, delegate._RunCustody(
        run_id="run-1", task_id="t-a", route_id="r", model="m",
        project_id="p", project_owned=False, root_task_id="t-a", ledger_root=str(tmp_path)))
    ctx = _nanny_ctx(tmp_path)

    first = json.loads(delegate._delegate_wait(ctx, "run-1", wait_sec=1))
    artifact = first["output_delivery"]["artifact"]
    _read_artifact_whole(ctx, artifact)
    acks = lambda: sum(1 for t in _event_types(tmp_path) if t == "delegate_run_output_consumed")
    assert acks() == 1

    # Identical re-stage keeps the acknowledgement: same bytes, same fact.
    same = json.loads(delegate._delegate_wait(ctx, "run-1", wait_sec=1))
    assert same["output_delivery"]["consumed"] is True

    # DIFFERENT content re-staged at the same path: the old ack must not transfer.
    stub.payload = "B" * 30_000 + "\n" + ("y\n" * 300)
    changed = json.loads(delegate._delegate_wait(ctx, "run-1", wait_sec=1))
    artifact2 = changed["output_delivery"]["artifact"]
    assert artifact2["sha256"] != artifact["sha256"]
    assert changed["output_delivery"]["consumed"] is False, \
        "an acknowledgement names bytes, never a path"
    delegate._CUSTODY.clear()
    assert dc.replay(tmp_path)["run-1"].output_consumed is False, \
        "the reset must survive a worker restart"

    # The new content earns its own acknowledgement by being read whole.
    _read_artifact_whole(ctx, artifact2)
    assert acks() == 2
    delegate._CUSTODY.clear()
    assert dc.replay(tmp_path)["run-1"].output_consumed is True
    delegate._CUSTODY.clear()


def test_a_truncated_primary_output_is_resolved_from_the_artifact_route(
        tmp_path, monkeypatch):
    """`primaryOutput.text` on the run detail is a bounded 256 KiB PREVIEW
    (control-api PRIMARY_OUTPUT_PREVIEW_BYTES) beside `bytes` and `truncated`. A
    truncated preview must never be staged or acknowledged as the result: the full
    file comes from GET /v2/runs/:id/artifacts/<path>, verified against the reported
    size before it may wear the plain name."""
    import ouroboros.delegate_custody as dc
    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways import claudexor as gw

    full_text = "W" * 120_000
    preview = full_text[:4_000]
    fetched_paths = []

    class _Stub(_LiveRunStub):
        def get_run(self, rid, **_kw):
            return {"lastSeq": 9,
                    "primaryOutput": {"kind": "answer", "path": "final/answer.md",
                                      "text": preview, "bytes": len(full_text),
                                      "truncated": True},
                    "summary": {"state": "succeeded", "spendUsd": 0.0}}
        def get_run_artifact(self, rid, path):
            fetched_paths.append((rid, path))
            return full_text.encode("utf-8")

    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _Stub())
    delegate._CUSTODY.clear()
    delegate._READ_COVERAGE.clear()
    dc.record_started(tmp_path, delegate._RunCustody(
        run_id="run-1", task_id="t-a", route_id="r", model="m",
        project_id="p", project_owned=False, root_task_id="t-a", ledger_root=str(tmp_path)))
    ctx = _nanny_ctx(tmp_path)

    out = json.loads(delegate._delegate_wait(ctx, "run-1", wait_sec=1))
    assert fetched_paths == [("run-1", "final/answer.md")], \
        "the full artifact must be fetched from the artifacts route, not trusted from the preview"
    delivery = out["output_delivery"]
    assert delivery["primary_output_full"]["fetched"] is True
    assert delivery["primary_output_full"]["verified"] == "size"
    artifact = delivery["artifact"]
    staged = json.loads(pathlib.Path(artifact["abs_path"]).read_text(encoding="utf-8"))
    assert staged["primary_output"]["text"] == full_text, "the STAGED result must be the full text"
    assert staged["primary_output"]["truncated"] is False

    # And the verified-full staging is what makes the acknowledgement reachable.
    _read_artifact_whole(ctx, artifact)
    assert sum(1 for t in _event_types(tmp_path) if t == "delegate_run_output_consumed") == 1
    delegate._CUSTODY.clear()


def test_an_unresolvable_truncated_output_is_disclosed_and_never_acknowledged(
        tmp_path, monkeypatch):
    """When the full artifact cannot be fetched — or fails size and preview-prefix
    verification — the result stays a PREVIEW: typed disclosure in the delivery, no
    acknowledgement ever (even after reading the staged file whole), and the custody
    replay says the staging was incomplete. Disclosure, not refusal: the preview is
    still delivered and readable."""
    import ouroboros.delegate_custody as dc
    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways import claudexor as gw

    class _FetchFails(_LiveRunStub):
        def get_run(self, rid, **_kw):
            return {"lastSeq": 9,
                    "primaryOutput": {"kind": "answer", "path": "final/answer.md",
                                      "text": "small preview", "bytes": 999_999,
                                      "truncated": True},
                    "summary": {"state": "succeeded", "spendUsd": 0.0}}
        def get_run_artifact(self, rid, path):
            raise gw.ClaudexorUnavailable("http_404", "no such artifact", status_code=404)

    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _FetchFails())
    delegate._CUSTODY.clear()
    delegate._READ_COVERAGE.clear()
    dc.record_started(tmp_path, delegate._RunCustody(
        run_id="run-1", task_id="t-a", route_id="r", model="m",
        project_id="p", project_owned=False, root_task_id="t-a", ledger_root=str(tmp_path)))
    ctx = _nanny_ctx(tmp_path)

    # Small payload -> the INLINE branch: even inline-fitting must not claim complete.
    out = json.loads(delegate._delegate_wait(ctx, "run-1", wait_sec=1))
    delivery = out["output_delivery"]
    assert delivery["complete"] is False and delivery["consumed"] is False
    assert delivery["primary_output_full"]["fetched"] is False
    assert "http_404" in delivery["primary_output_full"]["reason"]
    assert "INCOMPLETE AT THE SOURCE" in delivery["note"]

    # Large unverifiable payload -> the SPILL branch: staged as incomplete, unackable.
    big_preview = "P" * 120_000

    class _WrongBytes(_FetchFails):
        def get_run(self, rid, **_kw):
            return {"lastSeq": 9,
                    "primaryOutput": {"kind": "answer", "path": "final/answer.md",
                                      "text": big_preview, "bytes": 999_999,
                                      "truncated": True},
                    "summary": {"state": "succeeded", "spendUsd": 0.0}}
        def get_run_artifact(self, rid, path):
            return b"entirely different content"     # fails size AND prefix checks

    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _WrongBytes())
    dc.record_started(tmp_path, delegate._RunCustody(
        run_id="run-2", task_id="t-a", route_id="r", model="m",
        project_id="p", project_owned=False, root_task_id="t-a", ledger_root=str(tmp_path)))
    out2 = json.loads(delegate._delegate_wait(ctx, "run-2", wait_sec=1))
    delivery2 = out2["output_delivery"]
    assert delivery2["artifact"], "the preview is still delivered, staged and readable"
    assert delivery2["primary_output_full"]["fetched"] is True
    assert delivery2["primary_output_full"]["verified"] == ""
    assert "verification_failed" in delivery2["primary_output_full"]["reason"]
    spilled = [json.loads(l) for l in (tmp_path / "logs" / "events.jsonl").read_text().splitlines()
               if '"delegate_run_output_spilled"' in l]
    assert spilled[-1]["full_content"] is False

    # Reading the staged preview whole must NOT acknowledge: it is not the result.
    _read_artifact_whole(ctx, delivery2["artifact"])
    assert "delegate_run_output_consumed" not in _event_types(tmp_path)
    delegate._CUSTODY.clear()
    assert dc.replay(tmp_path)["run-2"].output_complete is False
    assert dc.replay(tmp_path)["run-2"].output_consumed is False
    delegate._CUSTODY.clear()


def test_a_reconciled_run_with_an_unread_artifact_is_visible_as_uncollected(
        tmp_path, monkeypatch):
    """The third "launched and never collected" recurrence, made structural: when the
    reconciler closes a run whose staged artifact has no EOF acknowledgement, its
    durable RECONCILED row says so — `staged_output_consumed: false` beside the
    artifact path — instead of the loss being inferable only from ledger discipline."""
    import ouroboros.delegate_custody as dc
    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways import claudexor as gw

    class _Stub(_LiveRunStub):
        def get_run(self, rid, **_kw):
            return {"lastSeq": 9, "primaryOutput": "V" * 120_000,
                    "summary": {"state": "succeeded", "spendUsd": 0.0}}
        def remove_project(self, pid): pass

    stub = _Stub()
    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: stub)
    # Decoupled: the run is held open for the sweep by an honest ledger
    # failure, not by a busy project retirement.
    import ouroboros.usage_accounting as ua
    ledger_blocked = {"now": True}
    real_record = ua.record_subscription_session

    def _flaky_ledger(*args, **kwargs):
        if ledger_blocked["now"]:
            raise ua.UsageAccountingError("usage accounting lock unavailable")
        return real_record(*args, **kwargs)

    monkeypatch.setattr(ua, "record_subscription_session", _flaky_ledger)
    delegate._CUSTODY.clear()
    dc.record_started(tmp_path, delegate._RunCustody(
        run_id="run-1", task_id="t-gone", route_id="r", model="m",
        project_id="prj", project_owned=True, root_task_id="t-gone", ledger_root=str(tmp_path)))

    # Nanny sees the staged preview; settlement cannot finish; the task dies
    # without reading the artifact to EOF.
    out = json.loads(delegate._delegate_wait(_nanny_ctx(tmp_path, "t-gone"), "run-1", wait_sec=1))
    assert out["output_delivery"]["artifact"], "this scenario is about a staged artifact"
    assert out["settlement"]["settled"] is False
    delegate._CUSTODY.clear()          # the worker is gone

    ledger_blocked["now"] = False
    results = dc.reconcile_orphaned_runs(tmp_path, {"t-alive"}, gateway_factory=lambda: stub)
    assert [r["run_id"] for r in results] == ["run-1"]
    assert results[0]["staged_output_consumed"] is False
    assert results[0]["staged_output"] == out["output_delivery"]["artifact"]["path"]
    reconciled = [json.loads(l) for l
                  in (tmp_path / "logs" / "events.jsonl").read_text().splitlines()
                  if '"delegate_run_reconciled"' in l]
    assert reconciled and reconciled[-1]["staged_output_consumed"] is False, \
        "the uncollected shape must be durable, not only returned"
    delegate._CUSTODY.clear()


def test_the_progress_payload_survives_a_verbose_harness_too(tmp_path, monkeypatch):
    """The sibling surface of the terminal payload: a harness-supplied timeline title is
    unbounded, and twelve long ones push the PROGRESS payload past the same cap, where
    head-truncation severs the same JSON."""
    from ouroboros.loop_tool_execution import _truncate_tool_result
    from ouroboros.tool_capabilities import tool_result_limit

    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways import claudexor as gw

    published = 30                   # what this harness puts on the timeline, in ONE batch

    class _Chatty(_LiveRunStub):
        def get_run(self, rid, *, timeout_sec=None):
            return {"lastSeq": 42, "summary": {"state": "running", "effectiveAccess": "readonly"},
                    "timeline": [{"type": "tool", "title": "T" * 20_000, "severity": "info"}
                                 for _ in range(published)]}

    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _Chatty())
    delegate._CUSTODY.clear()
    delegate._CUSTODY["run-1"] = delegate._RunCustody(
        run_id="run-1", task_id="t-a", route_id="r", model="m", project_id="p", project_owned=False)
    raw = delegate._delegate_wait(_nanny_ctx(tmp_path), "run-1", wait_sec=1, since_seq=1)
    delegate._CUSTODY.clear()
    assert len(raw) <= tool_result_limit("delegate_wait")
    assert _truncate_tool_result(raw, "delegate_wait", {}) == raw
    payload = json.loads(raw)
    assert payload["status"] == "progress"
    # P34R.5: the bound is the SHARED disclosed contract, not a hand-rolled slice —
    # every cut label carries the omission marker AND the original length.
    assert all("OMISSION NOTE" in row["title"] and "original length 20000" in row["title"]
               for row in payload["timeline_tail"])
    assert all(len(row["title"]) < 500 for row in payload["timeline_tail"])
    # Every advance accounts for its labels: kept + disclosed-shed == what the
    # harness ACTUALLY PUBLISHED, no kept label past the bound (shedding regime
    # itself is pinned in test_label_shedding_is_disclosed_...). The old
    # `== _TIMELINE_TAIL` pinned the defect: kept(12)+shed(0) passed while 18
    # observation-dropped rows went undisclosed — the tail is display width,
    # never arrival count.
    advances = payload["advances"]
    assert advances, payload
    for row in advances:
        assert "advances_omitted" not in row, row      # a head marker has no `events`
        kept, shed = row["events"], row.get("events_omitted", 0)
        assert len(kept) + shed == published, row
        assert kept or shed, row                       # never a silently empty row
        assert all("OMISSION NOTE" in event["title"] for event in kept), row
        assert all(len(event["title"]) < 500 for event in kept), row
