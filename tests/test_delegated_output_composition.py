"""#1573 retained terminal sources under the ONE actual-delivery D7 acknowledgement.

The retained payload lives in the staging task's canonical source store; Main's
loop credits only the ranges its final projection actually showed. Driven through
the real producer (``_delegate_wait``), the real ``run_llm_loop`` projection and the
real ``read_file`` stamp, never a helper call standing in for them:

* a whole delivery of the canonical source is one read to EOF; a measured head+tail
  credits both pieces and never the gap, which exact pages then complete;
* ``runtime_data`` reaches the same exact source where that root covers it;
* every read enters the acknowledgement once — no second hook, no prefix estimate;
* a rejected acknowledgement row blocks nothing and lands on a later read;
* bytes replaced before a deferred credit are not the source the read delivered;
* a retained engine preview is never called the full output.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from dataclasses import replace

import pytest

from ouroboros import delegate_custody as custody
from ouroboros import delegate_output as output
from tests._delegated_transport_shared import (  # noqa: F401  (autouse fixture applies on import)
    _LiveRunStub,
    _owned_gateway_uses_each_test_transport,
)
from tests.test_delegated_run_discovery import ROOT, _Engine, _read_whole, _receipts, _stage, _top
from tests.test_delegated_run_discovery import geo as _geo
from tests.test_main_authored_context import call
from tests.test_main_authored_context import main_loop as _main_loop

geo, main_loop = _geo, _main_loop
TASK = "authored-main"
BODY = "Ж🙂Ω" * 55_000  # multi-code-point text far above the read budget


def _retained_for_main(f, monkeypatch, run_id="run-canonical"):
    """The nanny's real terminal path on Main's own drive: settle, retain, deliver."""
    from ouroboros.gateways import claudexor as gw
    from ouroboros.tools import delegate

    class _Stub(_LiveRunStub):
        def get_run(self, rid, **_kw):
            return {"lastSeq": 9, "primaryOutput": BODY, "summary": {"state": "succeeded", "spendUsd": 0.0}}

    f.ctx.task_id = TASK
    f.ctx.task_metadata = {"root_task_id": TASK}
    output._READ_COVERAGE.clear()
    custody._CUSTODY.clear()
    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _Stub())
    drive = custody.custody_root(f.ctx)
    assert custody.record_started(drive, custody.RunCustody(
        run_id=run_id, task_id=TASK, route_id="r", model="m", project_id="p",
        project_owned=False, root_task_id=TASK, ledger_root=str(drive)))
    delivery = json.loads(delegate._delegate_wait(f.ctx, run_id, wait_sec=1))["output_delivery"]
    artifact = delivery["artifact"]
    assert artifact["root"] == "artifact_store" and artifact["source"]["task_id"] == TASK
    target = pathlib.Path(artifact["path"])
    return drive, artifact, target, target.read_bytes().decode("utf-8")


def _consumed(drive, run_id="run-canonical"):
    custody._CUSTODY.clear()
    return custody.replay(drive)[run_id].output_consumed


@pytest.mark.serial
@pytest.mark.parametrize("partial", [False, True], ids=["whole-large", "measured-head-tail"])
def test_main_credits_the_canonical_source_by_its_actual_projection(main_loop, monkeypatch, partial):
    from ouroboros.tools.core import _read_file
    from tests.test_main_result_ingress import _observe_measurements

    f = main_loop
    drive, artifact, target, content = _retained_for_main(f, monkeypatch)
    f.ctx.context_fit_plan = replace(f.ctx.context_fit_plan, window_tokens=120_000 if partial else 2_000_000)
    _observe_measurements(monkeypatch)
    answer, _, trace = f.run([
        call("read_file", {"root": artifact["root"], "path": artifact["path"]}, "read-retained"),
        {"content": "done"},
    ])
    assert answer == "done"
    row = next(row for row in trace["tool_calls"] if row["tool_call_id"] == "read-retained")
    read_view = row["tool_result_meta"]["read_view"]
    assert read_view["opened_root"] == "artifact_store"
    assert read_view["source_revision"] == artifact["sha256"]
    key = f"{target.resolve()}|{artifact['sha256']}"
    assert list(output._READ_COVERAGE) == [key], "one identity: no legacy/task-drive twin, no second hook"
    visible = next(m["content"] for m in f.inputs[-1]["messages"] if m.get("tool_call_id") == "read-retained")
    # read_file shows numbered rows (#1636); the display->source oracle is built from the content alone.
    numbered = "".join(f"{n:>6}\t{line}" for n, line in enumerate(content.splitlines(keepends=True), 1))
    header = read_view["body_start"]
    if not partial:
        assert numbered in visible
        assert output._READ_COVERAGE[key] == [[0, len(content)]]
        assert _consumed(drive)
    else:
        from tests.test_delegated_result_delivery import _credited, _numbered_display_oracle

        assert row["result_partial"] is True
        spans = row["result_source_view"]["shown_ranges"]
        oracle = _numbered_display_oracle(content, "#" * header + numbered, header,
                                          start_line=1, start_char=0, max_chars=None)
        expected = _credited(oracle, spans)
        assert output._READ_COVERAGE[key] == expected
        assert len(expected) == 2 and expected[0][0] == 0 and expected[1][1] == len(content)
        assert expected[0][1] < expected[1][0], "the omitted middle is a real gap"
        assert numbered[:spans[0][1] - header] in visible and numbered[spans[1][0] - header:] in visible
        assert not _consumed(drive), "head+tail never credits its gap"
        # An exact page over the gap completes the source; nothing else was needed.
        _read_file(f.ctx, root=artifact["root"], path=artifact["path"],
                   start_char=expected[0][1], max_chars=expected[1][0] - expected[0][1])
        assert output._READ_COVERAGE[key] == [[0, len(content)]]
        assert _consumed(drive)
    rows = [row for row in custody.custody_rows(drive) if row["type"] == custody.OUTPUT_CONSUMED]
    assert len(rows) == 1 and rows[0]["artifact"] == artifact["source"]["path"]
    assert rows[0]["sha256"] == artifact["sha256"] and rows[0]["chars"] == len(content)
    assert rows[0]["reader_task_id"] == TASK


@pytest.mark.serial
def test_main_runtime_data_read_of_the_same_exact_source_is_credited(main_loop, monkeypatch):
    from tests.test_main_result_ingress import _observe_measurements

    f = main_loop
    drive, artifact, target, content = _retained_for_main(f, monkeypatch)
    f.ctx.context_fit_plan = replace(f.ctx.context_fit_plan, window_tokens=2_000_000)
    _observe_measurements(monkeypatch)
    answer, _, trace = f.run([
        call("read_file", {"root": "runtime_data", "path": artifact["path"]}, "read-runtime"),
        {"content": "done"},
    ])
    assert answer == "done"
    row = next(row for row in trace["tool_calls"] if row["tool_call_id"] == "read-runtime")
    assert row["tool_result_meta"]["read_view"]["opened_root"] == "runtime_data"
    assert output._READ_COVERAGE[f"{target.resolve()}|{artifact['sha256']}"] == [[0, len(content)]]
    assert _consumed(drive)


def test_every_read_enters_the_one_acknowledgement_once(geo, monkeypatch):
    """No second hook, no prefix estimate: a direct read and an unrelated read each
    call the single acknowledgement owner once; only the exact source credits."""
    ctx = _top(geo, ROOT)
    read_next = _stage(geo, ctx, monkeypatch)["output_delivery"]["read_next"]
    calls = []
    real = output._acknowledge_staged_output
    monkeypatch.setattr(output, "_acknowledge_staged_output",
                        lambda *a, **k: calls.append(a[1].get("opened_root")) or real(*a, **k))
    from tests.test_delegated_run_discovery import _execute

    (geo.canonical / "notes.txt").write_text("plain\n", encoding="utf-8")
    assert _execute(ctx, "read_file", root="runtime_data", path="notes.txt").status == "ok"
    assert _execute(ctx, "read_file", root="artifact_store", path=read_next["path"],
                    start_line=1, max_lines=1).status == "ok"
    assert calls == ["runtime_data", "artifact_store"]
    assert _receipts(geo) == []


def test_a_rejected_acknowledgement_blocks_nothing_and_lands_later(geo, monkeypatch):
    ctx = _top(geo, ROOT)
    read_next = _stage(geo, ctx, monkeypatch)["output_delivery"]["read_next"]
    real = custody.emit
    monkeypatch.setattr(custody, "emit", lambda root, kind, payload, **kw: False
                        if kind == custody.OUTPUT_CONSUMED else real(root, kind, payload, **kw))
    _read_whole(ctx, read_next)  # every page is still served
    assert _receipts(geo) == []
    custody._CUSTODY.clear()
    assert custody.replay(geo.canonical)["run-1"].output_consumed is False
    from types import SimpleNamespace

    from ouroboros.context import build_health_invariants

    env = SimpleNamespace(drive_root=geo.canonical, drive_path=lambda rel="": geo.canonical / rel,
                          repo_path=lambda rel="": geo.repo / rel)
    assert "DELEGATED RESULT NEVER READ — run run-1" in build_health_invariants(env, task_id=ROOT)
    monkeypatch.setattr(custody, "emit", real)
    # The in-process proof of whole coverage survives; the next delivered page lands the row.
    from tests.test_delegated_run_discovery import _execute

    assert _execute(ctx, "read_file", root=read_next["root"], path=read_next["path"],
                    start_line=1, max_lines=1).status == "ok"
    assert [row["reader_task_id"] for row in _receipts(geo)] == [ROOT]


def test_bytes_replaced_before_a_deferred_credit_are_not_the_delivered_source(geo, monkeypatch):
    ctx = _top(geo, ROOT)
    read_next = _stage(geo, ctx, monkeypatch)["output_delivery"]["read_next"]
    from ouroboros.tools.registry import ToolRegistry

    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    with output.staged_read_delivery_scope():
        typed = registry.execute_result("read_file", {"root": read_next["root"], "path": read_next["path"]})
    target = pathlib.Path(read_next["path"])
    raw = target.read_bytes()
    target.write_bytes(raw.replace(b"complete", b"COMPLETE", 1))  # same size, other bytes
    output.acknowledge_staged_output_delivery(ctx, {"result": typed.text, "tool_result": typed},
                                              {"result": typed.text})
    assert output._READ_COVERAGE == {} and _receipts(geo) == []
    target.write_bytes(raw)
    output.acknowledge_staged_output_delivery(ctx, {"result": typed.text, "tool_result": typed},
                                              {"result": typed.text})
    assert [row["sha256"] for row in _receipts(geo)] == [hashlib.sha256(raw).hexdigest()]


def test_a_retained_engine_preview_is_never_called_the_full_output(geo, monkeypatch):
    """F1: the own-run notice keeps the address of an unverified preview but names it
    incomplete; the D7 NEVER-READ line (complete outputs only) stays silent for it."""
    from types import SimpleNamespace

    from ouroboros.context import build_health_invariants
    from ouroboros.gateways import claudexor as gw

    class _Preview(_Engine):
        def get_run(self, rid, *, timeout_sec=None):
            detail = super().get_run(rid)
            detail["primaryOutput"].update(bytes=10 ** 7, truncated=True)
            return detail

        def get_run_artifact(self, rid, path):
            raise gw.ClaudexorUnavailable("http_404", "no such artifact", status_code=404)

    out = _stage(geo, _top(geo, ROOT), monkeypatch, engine=_Preview())
    read_next = out["output_delivery"]["read_next"]
    assert read_next and custody.replay(geo.canonical)["run-1"].output_complete is False
    env = SimpleNamespace(drive_root=geo.canonical, drive_path=lambda rel="": geo.canonical / rel,
                          repo_path=lambda rel="": geo.repo / rel)
    health = build_health_invariants(env, task_id=ROOT, attempt_key="1")
    line = next(item for item in health.splitlines() if "OWN DELEGATED RUN ENDED — run run-1" in item)
    assert "retained full output" not in line
    assert f"retained INCOMPLETE payload (unverified engine preview, never acknowledgeable as read): " \
           f"read_file root='artifact_store' path='{read_next['path']}'" in line
    assert "(truncated preview)" in line
    assert "DELEGATED RESULT NEVER READ" not in health
    _read_whole(_top(geo, ROOT), read_next)
    assert _receipts(geo) == [], "an unverified preview is readable, never acknowledgeable"
