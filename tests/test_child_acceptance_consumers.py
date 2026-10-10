"""Explicit child/off review crosses the real slot executor and attempt ledger."""
import json
import re
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from tests.test_acceptance_delivery import (
    _CLEAN_VERDICT, _EpisodeLLM, _ROW_API, _ROW_NATIVE, _ROW_SESSION,
    _fake_session, _offline_env, _priced_offline_model, _real_panel, _roots, _tool_call,
)
from tests.test_acceptance_floor_admission import _KnownPricedLLM
from tests._usage_store_testing import ledger_rows


def _call(ctx, **kw):
    from ouroboros.tools.review import _handle_task_acceptance_review
    text = _handle_task_acceptance_review(ctx, claim="work delivered", **kw)
    match = re.search(r"<full_review>\s*(.*?)\s*</full_review>", text, re.S)
    return json.loads(match.group(1) if match else text)


def _ctx(tmp_path, workspace, governance, *, child=True, expired=False):
    from ouroboros.task_results import write_task_result, STATUS_RUNNING
    deadline = (datetime.now(timezone.utc) + timedelta(seconds=-100 if expired else 1200)).isoformat()
    task_id = "child" if child else "root"
    metadata = {"root_task_id": "root", "deadline_at": deadline,
                **({"parent_task_id": "root", "delegation_role": "subagent"} if child else {})}
    if child:
        # Real child admission starts under canonical root authority. A child
        # row alone cannot authorize inherited money or tree controls.
        write_task_result(tmp_path, "root", STATUS_RUNNING, root_task_id="root", deadline_at=deadline)
    write_task_result(tmp_path, task_id, STATUS_RUNNING, **metadata)
    return SimpleNamespace(task_id=task_id, drive_root=tmp_path, budget_drive_root=str(tmp_path),
                           task_metadata=metadata, task_contract={"objective": "read greeting"},
                           repo_dir=governance, workspace_root=workspace, workspace_mode="external",
                           pending_events=[])


# Each send settles a KNOWN $0.10. Known $0.10 below $0.15 admits the second send
# although its own bound would cross the cap; known $0.10 AT a $0.10 cap refuses it.
@pytest.mark.parametrize(("limit", "expired", "sends"),
                         [(1.0, False, 2), (0.15, False, 2), (0.1, False, 1), (1.0, True, 0)])
def test_child_one_native_reviewer_can_reason_again_but_inherits_money_and_deadline(monkeypatch, tmp_path, limit, expired, sends):
    from ouroboros import usage_accounting as ua

    _offline_env(monkeypatch, _ROW_API, _ROW_NATIVE, _ROW_SESSION)
    _priced_offline_model(monkeypatch)
    governance, workspace = _roots(tmp_path)
    ctx = _ctx(tmp_path, workspace, governance, expired=expired)
    llm = _KnownPricedLLM(tmp_path, [{"tool_calls": [_tool_call("read_file", {"path": "greeting.txt"})]},
                                   {"content": json.dumps(_CLEAN_VERDICT)}], scoped=True, reservation_usd=0.1)
    seen = _real_panel(monkeypatch, llm, stub_gate=False)
    with ua.usage_scope(ua.UsageScope(drive_root=tmp_path, task_id="child", root_task_id="root", root_limit_usd=limit)):
        result = _call(ctx, reviewer_slot_id="t_actor")
    assert len(llm.calls) == sends
    assert {r["slot_id"] for r in result["actors"]} == {"t_actor"}
    assert seen[0].deadline_at == ctx.task_metadata["deadline_at"]
    if sends == 2:
        assert any(m.get("role") == "tool" and "hello" in str(m) for m in llm.calls[-1]["messages"])
    if sends == 1:
        assert result["actors"][0]["usage"]["native_end_reason"] == "budget_exhausted"
    if sends:
        rows = ledger_rows(tmp_path)
        assert all(r["root_task_id"] == "root" and r["task_id"] == "child" for r in rows)
        assert ua.usage_projection(tmp_path, root_task_id="root")["settled_usd"] == pytest.approx(0.1 * sends)
    assert not (tmp_path / "state/child_review_cycles.json").exists()  # no extra cycle owner


def test_child_selected_session_and_explicit_root_off_panel_reach_real_consumers(monkeypatch, tmp_path):
    fake = _fake_session(monkeypatch)
    _offline_env(monkeypatch, _ROW_API, _ROW_NATIVE, _ROW_SESSION)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    governance, workspace = _roots(tmp_path)
    llm = _EpisodeLLM(tmp_path, [{"content": json.dumps(_CLEAN_VERDICT)}] * 2)
    seen = _real_panel(monkeypatch, llm, stub_gate=False)
    child = _call(_ctx(tmp_path, workspace, governance), reviewer_slot_id="t_sess")
    assert [r["slot_id"] for r in child["actors"]] == ["t_sess"]
    starts = [s for instance in fake.instances for s in instance.start_requests]
    assert len(starts) == 1 and starts[0]["scope"]["root"] == str(workspace)
    assert starts[0]["maxSeconds"] <= 1200 and not llm.calls
    root = _call(_ctx(tmp_path, workspace, governance, child=False))
    assert {r["slot_id"] for r in root["actors"]} == {"t_api", "t_actor", "t_sess"}
    assert len(llm.calls) == 2 and len(seen) == 2
