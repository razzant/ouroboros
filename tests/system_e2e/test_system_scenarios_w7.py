"""S28-S29 — serial addressed turns as a REAL consumer of the candidate (wave 7).

Owner 2026-09-23 21:12: the first delivery is the GENERAL capability of continuable
addressed participant turns — native children and session agents — with an explicit
end of participation, planning the first consumer, no fixed topology, participant
count or mandatory stage. Both scenarios drive the public tools on a real isolated
server (supervisor, workers, mailboxes, custody) with scripted stub models and the
fake engine, so they prove the HOST wiring only: what a live model or vendor session
does with the affordance is a separate, parent-owned live probe.

* S28 — NATIVE A→B→A, THEN THE PLANNER. A planning parent P in a file-less project
  room (advanced runtime mode, so `plan_task` is live) schedules two read-only
  native children on their own stub slots. Each announces itself to P (child → parent
  `peer_task` contribution), P hands A its sibling's id (ancestor steering), A opens
  with an interim position to B, B answers A, A replies, and each publishes ONE
  selected original to P. Every participant waits with `await_messages` (worker slot
  held, no model rounds) and ends its participation with a `FINAL:` answer only AFTER
  the reply it waited for landed — a contribution is not the end of participation.
  P then plans with `plan_task` declaring `task:<A>`/`task:<B>` evidence: the plan
  review request bodies and the durable wave carry the published originals and the
  children's finals, and NOT the sibling-only originals (negative control: A→B→A
  reaches the planner only when a participant addresses it to P).
* S29 — SESSION, SAME RUN, TWO QUESTIONS. P starts one `[FAKE:TURN]` run that pauses
  on two sequential questions. P re-waits on the question it already saw — which is
  the host's event-only sleep (`delegate_supervision.supervised_wait`): nothing wakes
  the model until a meaningful event — and a native child C's original, held back
  until the custody trail shows P parked there, IS that event: the addressed-message
  wake carries the window payload's `continuation: same_session` plus C's exact text
  under C's peer relation (`parent`). P relays it BYTE-EXACT through
  `delegate_answer(free_text=...)`, the run resumes in the SAME session (a new
  interaction id, the same run id), P answers the second question with its own
  original, and the run's terminal echoes both originals with their sha256. The
  codex-shaped twin (`[FAKE:INPUT_REQUIRED]`) ends needing input, its terminal states
  `continuation: new_physical_run`, and P continues it with a NEW `delegate_start`.

Default-lane tests pin the fake engine's two new run shapes against the REAL
gateway client, so drift between the fake and ``gateways/claudexor.py`` is a
named failure.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import uuid

import pytest

from tests.system_e2e.harness import (
    LANE_MOCK,
    ArtifactOracle,
    HeldStep,
    ReplayModel,
    body_text,
    canned_review_answer,
    classify_call,
    keyless_review_rows,
    keyless_settings,
    require_lane,
    start_server,
    wait_durable_result,
    wait_until,
)
from tests.system_e2e.interfaces import (
    FAKE_INPUT_REQUIRED_INPUTS,
    FAKE_INPUT_REQUIRED_MARKER,
    FAKE_TURN_MARKER,
    FAKE_TURN_QUESTIONS,
    FakeClaudexorDaemon,
)
from tests.system_e2e.test_system_scenarios_w2 import _s6_slot_binder
from tests.system_e2e.test_system_scenarios_w3a import _ROOT_TASK_ID_RE, _Again, _after_wave_settled
from tests.system_e2e.test_system_scenarios_w3b import _RUN_ID_RE, _SCOUT_ROW, _custody_rows, _roster
from tests.system_e2e.test_system_scenarios_w4 import _IID_RE

from devtools.benchmarks.common.server_runner import _api

# ===========================================================================
# Default lane: the fake engine's new run shapes, pinned with the REAL client.
# ===========================================================================


def test_fake_daemon_turn_run_pauses_twice_and_echoes_exact_free_text(tmp_path):
    from ouroboros.gateways.claudexor import ClaudexorGateway, discover_daemon_at, pending_interactions

    with FakeClaudexorDaemon(runs_dir=tmp_path / "runs") as daemon:
        daemon.install(tmp_path / "cx")
        with ClaudexorGateway(discover_daemon_at(tmp_path / "cx")) as gateway:
            gateway.handshake()
            request = {
                "prompt": FAKE_TURN_MARKER + " hold interim positions", "instructions": "i",
                "authPreference": "subscription", "mode": "ask",
                "scope": {"kind": "project", "root": str(tmp_path)},
                "harnesses": [daemon.harness_id], "primaryHarness": daemon.harness_id,
                "access": "readonly", "maxSeconds": 60,
            }
            run_id = str(gateway.start_run(request, idempotency_key="inv-turn-1")["runId"])
            originals = ["first original\nline two — exact bytes ✓", "second original"]
            seen_iids = []
            for ordinal, original in enumerate(originals, 1):
                detail = gateway.get_run(run_id)
                assert (detail.get("summary") or {}).get("state") == "running", detail
                [row] = pending_interactions(detail)
                assert row["interaction_id"] == f"turn-{run_id[:8]}-{ordinal}"
                assert row["interaction_id"] not in seen_iids
                seen_iids.append(row["interaction_id"])
                assert "INTERIM (not final)" in row["questions"][0]["question"]
                assert row["questions"][0]["options"] == [] and row["timeout_at"] is None
                answered = gateway.answer_interaction(run_id, row["interaction_id"], [
                    {"questionId": "q1", "selectedLabels": [], "freeText": original}])
                assert answered.get("status") == "delivered", answered
            assert len(seen_iids) == FAKE_TURN_QUESTIONS
            done = gateway.get_run(run_id)
            assert (done.get("summary") or {}).get("state") == "succeeded", done
            assert pending_interactions(done) == []
            text = str((done.get("primaryOutput") or {}).get("text") or "")
            assert text.startswith("FINAL:")
            for ordinal, original in enumerate(originals, 1):
                assert f"RELAYED[{ordinal}] sha256={hashlib.sha256(original.encode('utf-8')).hexdigest()}\n{original}\n" in text
            assert len(daemon.run_start_posts()) == 1  # one physical run, two turns


def test_fake_daemon_input_required_run_ends_needing_input(tmp_path):
    from ouroboros.gateways.claudexor import ClaudexorGateway, discover_daemon_at, pending_interactions
    from ouroboros.subagents import DelegatedRunShape
    from ouroboros.tools.delegate_terminal_evidence import _terminal_payload

    with FakeClaudexorDaemon(runs_dir=tmp_path / "runs") as daemon:
        daemon.install(tmp_path / "cx")
        with ClaudexorGateway(discover_daemon_at(tmp_path / "cx")) as gateway:
            gateway.handshake()
            request = {
                "prompt": FAKE_INPUT_REQUIRED_MARKER + " codex-shaped", "instructions": "i",
                "authPreference": "subscription", "mode": "ask",
                "scope": {"kind": "project", "root": str(tmp_path)},
                "harnesses": [daemon.harness_id], "primaryHarness": daemon.harness_id,
                "access": "readonly", "maxSeconds": 60,
            }
            run_id = str(gateway.start_run(request, idempotency_key="inv-inreq-1")["runId"])
            detail = gateway.get_run(run_id)
            assert (detail.get("summary") or {}).get("state") == "failed", detail
            assert pending_interactions(detail) == []
            assert detail["summary"]["outcomeFacts"] == {
                "reason": "input_required",
                "work_state": {"required_inputs": list(FAKE_INPUT_REQUIRED_INPUTS)}}
            # The real terminal projection reads it as the codex-shaped question.
            payload = _terminal_payload(run_id, detail, DelegatedRunShape(
                access="readonly", mode="ask", isolation="", delegated=False))
            assert payload["continuation"] == "new_physical_run"
            assert "input_required_note" in payload
            assert "INTERIM (not final)" in str((detail.get("primaryOutput") or {}).get("text") or "")


# ===========================================================================
# Shared pieces
# ===========================================================================

_CHILD_RECEIPT_RE = re.compile(r"Subagent request queued ([0-9a-f]{8}): (PEER [ABC])")
_SIBLING_PREFIX_RE = re.compile(r"\[Message from peer task ([0-9a-f]{8}) \(sibling\)\]\n")
_CHILD_PREFIX_RE = re.compile(r"\[Message from peer task ([0-9a-f]{8}) \(your child\)\]\n")
_HOLD_ROUNDS_MAX = 12
_AWAIT_SEC = 120
_PEER_ROWS = {
    "peer-a": "openai-compatible::mock-peer-a",
    "peer-b": "openai-compatible::mock-peer-b",
    "peer-c": "openai-compatible::mock-peer-c",
}
PARTICIPATION_RULES = (
    "You are one participant in an exchange of addressed turns: stating a position does "
    "not end your participation. Send interim positions or exact requests to your parent "
    "or a sibling with forward_to_worker, wait for the next addressed turn with "
    "await_messages, and end your participation with a final answer beginning 'FINAL:' "
    "only after the reply you waited for arrived."
)


_REFLECTION_MARKER = "Write the reflection now."  # ouroboros/reflection.py, the post-task reflection turn
_REFLECTION_ANSWER = ("Scripted scenario; nothing to persist.\nMEMORY_ACTIONS_JSON: []\n"
                      "BACKLOG_CANDIDATES_JSON: []")


class _PeerReplayModel(ReplayModel):
    """``ReplayModel`` whose host-side turns are answered canned without consuming the
    fixture: TOOL-LESS non-review calls (the supervisor's semantic duplicate probe on
    the light slot, and any other plain host probe) and the post-task REFLECTION turn
    (tool-bearing, after the final answer). The actors under test are the scripted
    tool-bearing rounds; host probe and reflection counts are not part of the wiring
    being proved."""

    def _answer(self, body: dict, seq: int) -> tuple[str, dict]:
        kind = classify_call(body)
        if canned_review_answer(kind) is None and not body.get("tools"):
            return "plain", {"role": "assistant", "content": "No existing task duplicates this request."}
        if canned_review_answer(kind) is None and _REFLECTION_MARKER in body_text(body):
            return "reflection", {"role": "assistant", "content": _REFLECTION_ANSWER}
        return super()._answer(body, seq)


def _peer_row(name: str) -> dict:
    return {"subagent_id": name, "recommended_use": f"Read-only exchange participant {name} (system_e2e w7).",
            "route": {"kind": "api_model", "target_id": _PEER_ROWS[name]}, "effort": "low"}


def _script_error(text: str) -> dict:
    return {"final": "E2E_SCRIPT_ERROR: " + text}


def _hold_until(visible, then, *, label: str, await_sec: int = _AWAIT_SEC, rounds_max: int = _HOLD_ROUNDS_MAX):
    """A callable step that serves ``then(body)`` once ``visible(text)`` holds and
    otherwise HOLDS the row on an ``await_messages`` wait — bounded, so a delivery
    that never arrives fails by name instead of running the task into its ceiling.
    ``visible`` may read evidence beyond the transcript (an oracle probe); a hold
    that polls such evidence passes a short ``await_sec`` and its own bound."""
    rounds = {"n": 0}

    def step(body: dict) -> dict:
        text = body_text(body)
        if visible(text):
            return then(body) if callable(then) else then
        rounds["n"] += 1
        if rounds["n"] > rounds_max:
            return _script_error(f"{label}: not visible after {rounds['n']} await rounds")
        return HeldStep({"tool": "await_messages", "arguments": {"timeout_sec": await_sec}})

    return step


def _parent_id(body: dict) -> str:
    ids = _ROOT_TASK_ID_RE.findall(body_text(body))
    return ids[-1] if ids else ""


def _forward(task_id: str, message: str) -> dict:
    return {"tool": "forward_to_worker", "arguments": {"task_id": task_id, "message": message}}


def _forward_to_parent(message: str):
    def step(body: dict) -> dict:
        parent = _parent_id(body)
        return _forward(parent, message) if parent else _script_error("no root_task_id visible")
    return step


def _child_ids(text: str) -> dict:
    return {label: tid for tid, label in _CHILD_RECEIPT_RE.findall(text)}


def _replay_hold(step_fn):
    """Adapt a wave-3a ``_Again`` hold (scripted stub) to the replay model's ``HeldStep``;
    a ``then`` that is itself a callable (dynamic ids) is resolved here, since the
    scripted-stub hold hands it back unresolved."""
    def step(body: dict):
        out = step_fn(body)
        if isinstance(out, _Again):
            return HeldStep(out.step)
        return out(body) if callable(out) else out
    return step


def _forwards(oracle: ArtifactOracle, sender: str) -> list:
    """The sender's durable ``forward_to_worker`` rows (server-level tools.jsonl): the
    exact message bytes ride ``args.message`` and the typed write receipt the preview.
    The mailbox FILES themselves are cleaned at the recipient's task_done by design,
    so they are not evidence a scenario may read after completion."""
    return [row for row in oracle.tools_rows() if str(row.get("type") or "tool_call") == "tool_call"
            and str(row.get("tool") or "") == "forward_to_worker" and str(row.get("task_id") or "") == sender]


def _injected(oracle: ArtifactOracle, recipient: str) -> list:
    """The recipient's durable ``task_message_injected`` events (round-top deliveries)."""
    return [row for row in oracle.events("task_message_injected") if str(row.get("task_id") or "") == recipient]


def _dump_misses(model: ReplayModel, root: pathlib.Path) -> None:
    """Keep the exact bodies of fixture MISSES beside the scenario root so a red
    integrity gate names the call the script did not expect."""
    with model._lock:
        misses = [body for kind, body in model.calls if kind == "replay_miss"]
    if misses:
        (root / "replay_misses.json").write_text(json.dumps(
            [{"model": body.get("model"), "tools": [t.get("function", {}).get("name") for t in body.get("tools") or []][:12],
              "tail": body_text(body)[-6000:]} for body in misses], ensure_ascii=False, indent=2), encoding="utf-8")


def _tool_rows(drive: ArtifactOracle, tool: str) -> list:
    # Result rows only (#1316): a call's start / wait-end rows share its invocation_id.
    return [row for row in drive.tools_rows()
            if str(row.get("type") or "tool_call") == "tool_call" and str(row.get("tool") or "") == tool]


def _isolation_settings(root: pathlib.Path) -> dict:
    """Keep every host-minted durable root of the scenario server under the scenario
    root: `config.py` otherwise resolves them from `$HOME`, which `IsolatedServer`
    does not override (the #1215/#1230 class)."""
    return {
        "OUROBOROS_SUBAGENT_WORKTREE_ROOT": str(root / "worktrees"),
        "OUROBOROS_SUBAGENT_PROJECTS_ROOT": str(root / "projects"),
        "OUROBOROS_DELIVERABLES_ROOT": str(root / "deliverables"),
    }


def _home_sentinel() -> dict:
    home = pathlib.Path(os.path.expanduser("~")) / "Ouroboros"
    return {name: sorted(os.listdir(home / name)) if (home / name).is_dir() else None
            for name in ("subagent_worktrees", "projects", "Deliverables")}


def _room_task(server, description: str, project_name: str) -> tuple[str, int]:
    """A root task addressed into a file-less project room (S21 idiom): without a
    room `own_room_chat` is None and the planner attaches no mailbox evidence."""
    project = (_api(server.base_url, "POST", "/api/projects", {"name": project_name}, timeout=60)
               .get("project") or {})
    project_id, chat_id = str(project.get("id") or ""), int(project.get("chat_id") or 0)
    assert project_id and chat_id, project
    created = _api(server.base_url, "POST", "/api/tasks", {
        "description": description, "project_id": project_id, "chat_id": chat_id,
        "memory_mode": "forked", "actor_id": "e2e-driver", "source": "e2e-driver",
        "metadata": {"source": "e2e-driver", "delegation_role": "root"},
    }, timeout=60)
    task_id = str(created.get("task_id") or "")
    assert task_id, created
    return task_id, chat_id


# ===========================================================================
# S28 — native A→B→A, then the planner
# ===========================================================================

S28_GOAL = "Decide the smoke note's shape from the participants' exchanged positions."


def _s28_fixture(n: dict) -> dict:
    """The three actors' scripts, keyed by (lineage, slot, attempt). Nonces ``n``
    make every original unique per run so transcripts and evidence are asserted on
    exact bytes."""
    ready_a, ready_b = f"PEER_READY A {n['ra']}", f"PEER_READY B {n['rb']}"
    sib_a1 = f"A1 interim position {n['a1']}: not my final. B, name your objection."
    sib_b1 = f"B1 interim objection {n['b1']}: not my final. A, revise your second point."
    sib_a2 = f"A2 revised interim {n['a2']}: still not my final."
    pub_a = f"PUBLISHED A {n['pa']}: selected original for the planner, A's position after B's objection."
    pub_b = f"PUBLISHED B {n['pb']}: selected original for the planner, B's objection and its ground."
    fin_a, fin_b = f"FINAL: A {n['fa']} — my participation ends.", f"FINAL: B {n['fb']} — my participation ends."
    n.update(ready_a=ready_a, ready_b=ready_b, sib_a1=sib_a1, sib_b1=sib_b1, sib_a2=sib_a2,
             pub_a=pub_a, pub_b=pub_b, fin_a=fin_a, fin_b=fin_b)

    def schedule(name: str, label: str) -> dict:
        return {"tool": "schedule_subagent", "arguments": {
            "subagent_id": name,
            "objective": f"PEER {label}: take part in the exchange. " + PARTICIPATION_RULES,
            "expected_output": "Interim contributions to your peers, one published original to your parent, then a FINAL line.",
        }}

    def hand_sibling_id(body: dict) -> dict:
        ids = _child_ids(body_text(body))
        if {"PEER A", "PEER B"} - set(ids):
            return _script_error(f"child receipts missing: {ids}")
        return _forward(ids["PEER A"], f"SIBLING_B_ID {ids['PEER B']}")

    def wait_children(body: dict) -> dict:
        ids = _child_ids(body_text(body))
        return {"tool": "wait_tasks", "arguments": {
            "task_ids": [ids["PEER A"], ids["PEER B"]], "timeout_sec": 240, "mode": "all_terminal"}}

    def dispose(label: str):
        def step(body: dict) -> dict:
            text = body_text(body)
            tid = _child_ids(text).get(label, "")
            match = re.search(rf'"task_id":\s*"{tid}".{{0,600}}?"child_result_sha256":\s*"([0-9a-f]{{64}})"', text, re.S)
            if not (tid and match):
                return _script_error(f"no exact result hash for {label} ({tid})")
            return {"tool": "tree_note", "arguments": {
                "kind": "decision", "text": f"Absorbed {label}'s final into the plan.",
                "payload": {"type": "child_result_disposition", "child_task_id": tid,
                            "disposition": "integrated", "child_result_sha256": match.group(1)}}}
        return step

    def plan(body: dict) -> dict:
        ids = _child_ids(body_text(body))
        return {"tool": "plan_task", "arguments": {
            "goal": S28_GOAL,
            "plan": "Weigh both published originals, draft the note, verify, finish.",
            "spec": {
                "in_scope": ["w7 serial-turns planning smoke"],
                "acceptance_claims": ["The plan cites both participants' published originals."],
                "affected_paths": [],
                "evidence": [f"task:{ids['PEER A']}", f"task:{ids['PEER B']}"],
            },
        }}

    def sibling_of(original: str):
        """The sibling id from the prefix of the row carrying ``original``."""
        def read(text: str) -> str:
            match = re.search(_SIBLING_PREFIX_RE.pattern + re.escape(original), text)
            return match.group(1) if match else ""
        return read

    def a_opens(body: dict) -> dict:
        match = re.search(r"SIBLING_B_ID ([0-9a-f]{8})", body_text(body))
        return _forward(match.group(1), sib_a1) if match else _script_error("no SIBLING_B_ID")

    def a_replies(body: dict) -> dict:
        sib = sibling_of(sib_b1)(body_text(body))
        return _forward(sib, sib_a2) if sib else _script_error("B1 prefix unreadable")

    def b_answers(body: dict) -> dict:
        sib = sibling_of(sib_a1)(body_text(body))
        return _forward(sib, sib_b1) if sib else _script_error("A1 prefix unreadable")

    return {
        # P — the planning parent
        ("root", "mock-model|tools", 1): schedule("peer-b", "B"),
        ("root", "mock-model|tools", 2): schedule("peer-a", "A"),
        ("root", "mock-model|tools", 3): _hold_until(lambda t: ready_a in t and ready_b in t, hand_sibling_id, label="both READY"),
        ("root", "mock-model|tools", 4): _hold_until(lambda t: pub_a in t and pub_b in t, wait_children, label="both PUBLISHED"),
        ("root", "mock-model|tools", 5): dispose("PEER A"),
        ("root", "mock-model|tools", 6): dispose("PEER B"),
        ("root", "mock-model|tools", 7): plan,
        ("root", "mock-model|tools", 8): _replay_hold(_after_wave_settled(1, plan)),
        ("root", "mock-model|tools", 9): {"final": f"S28_PARENT_FINAL {n['pf']}: planned with both published originals."},
        # A
        ("root", "mock-peer-a|tools", 1): _forward_to_parent(ready_a),
        ("root", "mock-peer-a|tools", 2): _hold_until(lambda t: "SIBLING_B_ID " in t, a_opens, label="sibling id from parent"),
        ("root", "mock-peer-a|tools", 3): _hold_until(lambda t: sib_b1 in t, a_replies, label="B1"),
        ("root", "mock-peer-a|tools", 4): _forward_to_parent(pub_a),
        ("root", "mock-peer-a|tools", 5): {"final": fin_a},
        # B
        ("root", "mock-peer-b|tools", 1): _forward_to_parent(ready_b),
        ("root", "mock-peer-b|tools", 2): _hold_until(lambda t: sib_a1 in t, b_answers, label="A1"),
        ("root", "mock-peer-b|tools", 3): _hold_until(lambda t: sib_a2 in t, lambda body: _forward_to_parent(pub_b)(body), label="A2"),
        ("root", "mock-peer-b|tools", 4): {"final": fin_b},
    }


@pytest.mark.integration
@pytest.mark.serial
def test_s28_native_peers_exchange_addressed_turns_and_the_planner_receives_selected_originals(
        e2e_clone, tmp_path_factory):
    require_lane(LANE_MOCK)
    root = tmp_path_factory.mktemp("s28")
    n = {key: uuid.uuid4().hex[:10] for key in ("ra", "rb", "a1", "b1", "a2", "pa", "pb", "fa", "fb", "pf")}
    model = _PeerReplayModel(_s28_fixture(n), slot_binder=_s6_slot_binder,
                             model_ids=["mock-model", "mock-peer-a", "mock-peer-b"])
    before = _home_sentinel()
    with model:
        # A scenario that passes its own catalog owns its review pool: the planner's
        # wave below seats the keyless reviewer rows beside the two exchange peers.
        settings = keyless_settings(
            model, OUROBOROS_RUNTIME_MODE="advanced",
            OUROBOROS_SUBAGENTS=_roster(_peer_row("peer-a"), _peer_row("peer-b"), *keyless_review_rows()),
            **_isolation_settings(pathlib.Path(root)))
        server = start_server(e2e_clone, root, settings)
        try:
            oracle = ArtifactOracle(server.data_root)
            parent_id, _chat = _room_task(server, "Run the exchange, then plan from it.", f"w7-s28-{n['pf']}")
            assert wait_until(lambda: parent_id in oracle.running_ids(), 120)
            result = server.wait_task(parent_id, timeout=900)
            assert result.get("status") == "completed", result
            stored = wait_durable_result(oracle, parent_id)
            assert f"S28_PARENT_FINAL {n['pf']}" in str(stored.get("result") or ""), stored
            assert str(stored.get("reason_code") or "") != "children_unabsorbed", stored

            children = {}
            for child_id in oracle.child_task_ids(parent_id):
                row = wait_durable_result(oracle, child_id)
                label = "A" if n["fin_a"] in str(row.get("result") or "") else "B" if n["fin_b"] in str(row.get("result") or "") else "?"
                children[label] = (child_id, row)
            assert set(children) == {"A", "B"}, children
            a_id, a_row = children["A"]
            b_id, b_row = children["B"]
            for row in (a_row, b_row):
                # (d) participation ended CLEAN on the participant's own FINAL, never on a rail.
                assert row.get("status") == "completed", row
                assert str(row.get("result") or "").startswith("FINAL:"), row
                assert str(row.get("reason_code") or "") not in {
                    "idle_timeout", "absolute_ceiling", "children_unabsorbed", "per_call_timeout"}, row
            p_drive, a_drive, b_drive = (oracle.task_drive(tid) for tid in (parent_id, a_id, b_id))
            assert len({d.data_root for d in (p_drive, a_drive, b_drive)}) == 3, "children share a drive"

            # (a) the writes and the deliveries, durably: every forward carries the exact bytes and
            # the typed peer receipt, and every recipient's injected event names provenance,
            # relation and sender (the mailbox files are cleaned at task_done by design).
            def sent(sender, recipient, text, receipt):
                rows = [r for r in _forwards(oracle, sender) if r["args"].get("task_id") == recipient and r["args"].get("message") == text]
                assert len(rows) == 1 and receipt in str(rows[0].get("result_preview") or ""), (sender, recipient, text, rows)
            for text in (n["sib_a1"], n["sib_a2"]):
                sent(a_id, b_id, text, "message from a peer task (your sibling")
            sent(b_id, a_id, n["sib_b1"], "message from a peer task (your sibling")
            for sender, text in ((a_id, n["ready_a"]), (b_id, n["ready_b"]), (a_id, n["pub_a"]), (b_id, n["pub_b"])):
                sent(sender, parent_id, text, "message from a peer task (your parent")
            sent(parent_id, a_id, f"SIBLING_B_ID {b_id}", "Message forwarded to task")
            assert [(r.get("provenance"), r.get("relation"), r.get("source_task_id"), r.get("text_preview")) for r in _injected(oracle, b_id)] == [
                ("peer_task", "sibling", a_id, n["sib_a1"]), ("peer_task", "sibling", a_id, n["sib_a2"])]
            assert [(r.get("provenance"), r.get("relation"), r.get("source_task_id"), r.get("text_preview")) for r in _injected(oracle, a_id)] == [
                ("ancestor_task", None, parent_id, f"SIBLING_B_ID {b_id}"), ("peer_task", "sibling", b_id, n["sib_b1"])]
            assert sorted((r.get("relation"), r.get("source_task_id"), r.get("text_preview")) for r in _injected(oracle, parent_id)
                          if r.get("provenance") == "peer_task") == sorted([
                ("parent", a_id, n["ready_a"]), ("parent", b_id, n["ready_b"]), ("parent", a_id, n["pub_a"]), ("parent", b_id, n["pub_b"])])

            # (b) transcripts: the exact prefix + body, never an ancestor or owner label for a peer.
            texts = {}
            for kind, body in model.calls:
                slot = _s6_slot_binder(body)
                texts.setdefault(slot, []).append(body_text(body))
            a_text, b_text, p_text = ("\n".join(texts.get(f"{slug}|tools", []))
                                      for slug in ("mock-peer-a", "mock-peer-b", "mock-model"))
            assert f"[Message from peer task {b_id} (sibling)]\n{n['sib_b1']}" in a_text
            assert f"[Message from ancestor task {parent_id}]\nSIBLING_B_ID {b_id}" in a_text
            assert f"[Message from peer task {a_id} (sibling)]\n{n['sib_a1']}" in b_text
            assert f"[Message from peer task {a_id} (sibling)]\n{n['sib_a2']}" in b_text
            for original in (n["pub_a"], n["pub_b"], n["ready_a"], n["ready_b"]):
                sender = a_id if original in (n["pub_a"], n["ready_a"]) else b_id
                assert f"[Message from peer task {sender} (your child)]\n{original}" in p_text
            for text in (a_text, b_text):
                # A peer's words are never delivered as the owner's or an ancestor's
                # (the identity prompt merely names those prefixes; deliveries are anchored).
                for original in (n["sib_a1"], n["sib_b1"], n["sib_a2"]):
                    assert f"[Message from my human]\n{original}" not in text
                    assert not re.search(r"\[Message from ancestor task [0-9a-f]{8}\]\n" + re.escape(original), text)

            # (c) await_messages rows: the wait held the slot and returned typed.
            for drive in (a_drive, b_drive, p_drive):
                rows = _tool_rows(drive, "await_messages")
                assert rows, f"no await_messages call in {drive.data_root}"
                assert any(json.loads(str(row.get("result_preview") or "{}")) .get("reason") ==
                           "owner_mailbox_pending" for row in rows), "only timeout polling, no mailbox wake"
                for row in rows:
                    out = json.loads(str(row.get("result_preview") or ""))
                    assert out["reason"] in {"owner_mailbox_pending", "timeout"} and out["slot"] == "held", out
                    assert out["window_sec"] <= _AWAIT_SEC and out["requested_sec"] == _AWAIT_SEC, out

            # (d)+(e) a contribution is not the end of participation: the awaited reply was
            # injected (server-level event carrying the RELATION) BEFORE the recipient's terminal.
            assert wait_until(lambda: any(str(row.get("task_id")) == parent_id
                                         for row in oracle.events("task_done")), 60)
            events = oracle.events()
            def _index(pred):
                return next(i for i, row in enumerate(events) if pred(row))
            def injected(task_id, preview_start, relation):
                return lambda row: (str(row.get("type")) == "task_message_injected" and str(row.get("task_id")) == task_id
                                    and str(row.get("text_preview") or "").startswith(preview_start)
                                    and row.get("relation") == relation)
            def done(task_id):
                return lambda row: str(row.get("type")) == "task_done" and str(row.get("task_id")) == task_id
            assert _index(injected(b_id, n["sib_a2"][:60], "sibling")) < _index(done(b_id))
            assert _index(injected(a_id, n["sib_b1"][:60], "sibling")) < _index(done(a_id))
            assert _index(injected(parent_id, n["pub_a"][:60], "parent")) < _index(done(parent_id))

            # (f) planner evidence: published originals and both finals reached the reviewers;
            # sibling-only originals did not (negative control, progress narration included).
            review_bodies = [body_text(body) for kind, body in model.calls if kind == "plan_review"]
            assert len(review_bodies) == 3, [k for k, _ in model.calls]
            for review in review_bodies:
                for present in (n["pub_a"], n["pub_b"], n["fin_a"], n["fin_b"], f"task:{a_id}", f"task:{b_id}"):
                    assert present in review, present
                for absent in (n["sib_a1"], n["sib_b1"], n["sib_a2"]):
                    assert absent not in review, absent
            state = stored.get("plan_review_state")
            assert isinstance(state, dict) and int(state.get("cycles_paid") or 0) == 1, state
            waves = [w for w in (state.get("waves") or []) if isinstance(w, dict)]
            assert waves and waves[-1].get("aggregate") == "GREEN" and waves[-1].get("closed") is True, waves
            from ouroboros.tools.plan_review_artifacts import read_wave
            wave = read_wave(oracle.data_root, parent_id, waves[-1]["wave_artifact"])
            manifest = wave.get("evidence_manifest_full") or {}
            own = manifest.get("own_dialogue") or {}
            assert own.get("text") and not own.get("gap"), own
            assert n["pub_a"] in own["text"] and n["pub_b"] in own["text"]
            assert n["sib_a1"] not in own["text"] and n["sib_b1"] not in own["text"]
            dialogue_rows = [json.loads(line) for line in own["text"].splitlines()[1:] if line.strip()]
            published = [r for r in dialogue_rows if n["pub_a"] in str(r.get("text")) or n["pub_b"] in str(r.get("text"))]
            assert {r.get("author") for r in published} == {"peer_task"}, published
            assert {r.get("source_task_id") for r in published} == {a_id, b_id}, published
            assert "mailbox_incomplete" not in json.dumps(manifest.get("omissions") or [])
            attached = {row.get("locator"): row for row in manifest.get("attached") or []}
            assert n["fin_a"] in str(attached[f"task:{a_id}"].get("text")) and n["fin_b"] in str(attached[f"task:{b_id}"].get("text"))

            # (g) exactly the scripted exchange, nothing missed, nothing left unserved.
            assert wait_until(lambda: parent_id not in oracle.running_ids(), 60)
            _dump_misses(model, pathlib.Path(root))
            model.assert_consumed()
        finally:
            server.stop()
    assert _home_sentinel() == before, "the isolated server touched the operator's durable roots"


# ===========================================================================
# S29 — session: one run, two questions, exact relayed bytes
# ===========================================================================

_C_ORIGINAL_RE = re.compile(r"(C_ORIGINAL_BEGIN [0-9a-f]{10}\n.*?\nC_ORIGINAL_END)", re.S)
# The addressed-message wake event as rendered into the parked nanny's transcript: a
# flat JSON object (indent=2), so the exact text rides JSON-escaped and is decoded back.
_WAKE_EVENT_RE = re.compile(r'\{\s*"type": "addressed_message".*?\}', re.S)
# C holds its original back until the custody trail shows P parked in its re-wait, so
# the original is the EVENT that wakes the parked nanny (S29's point): an observed
# fact, not a sleep guess (a 45 s delay lost the ordering on a slow runner). Each
# hold is a short await_messages window nothing addresses, re-polled until the
# observation holds — bounded, so a parent that never parks fails by name.
_C_POLL_SEC = 3
_C_HOLD_ROUNDS_MAX = 40


def _c_original_in(text: str) -> tuple[str, str]:
    """``(source_task_id, exact original)`` of C's contribution as the parent's model
    saw it: injected at a round top under its peer prefix, or carried by the
    supervised wait's addressed-message event (JSON-escaped). ``("", "")`` if absent."""
    match = re.search(_CHILD_PREFIX_RE.pattern + _C_ORIGINAL_RE.pattern, text, re.S)
    if match:
        return match.group(1), match.group(2)
    for candidate in _WAKE_EVENT_RE.findall(text):
        try:
            event = json.loads(candidate)
        except ValueError:
            continue
        original = _C_ORIGINAL_RE.search(str(event.get("text") or ""))
        if original and event.get("provenance") == "peer_task":
            return str(event.get("source_task_id") or ""), original.group(1)
    return "", ""


def _s29_fixture(n: dict, gate: dict) -> dict:
    """Actors' scripts; ``gate["parked"]`` is the test's observation (installed once the
    server is up) that P has entered its re-wait — C contributes only after it holds."""
    c_original = (f"C_ORIGINAL_BEGIN {n['c']}\nC's interim position: the note must name its marker line.\n"
                  f"  second line, indented, with unicode — ✓ {n['c']}\nC_ORIGINAL_END")
    exact2 = f"[Original from task P; authored by P]\nP_ORIGINAL {n['p']}\nP's own second original, two lines.\n"
    n.update(c_original=c_original, exact2=exact2)

    def relay_c(body: dict) -> dict:
        text = body_text(body)
        c_id, original = _c_original_in(text)
        run_ids, iids = _RUN_ID_RE.findall(text), _IID_RE.findall(text)
        parent = _parent_id(body)
        if not (c_id and original and run_ids and iids and parent):
            return _script_error("C's original, the run id, the interaction id or my own id is not visible")
        exact1 = f"[Original from peer task {c_id} (your child); relayed verbatim by {parent}]\n{original}"
        n["exact1"] = exact1
        return {"tool": "delegate_answer", "arguments": {
            "run_id": run_ids[-1], "interaction_id": iids[-1],
            "answers": [{"question_id": "q1", "free_text": exact1}]}}

    def answer_own(body: dict) -> dict:
        text = body_text(body)
        run_ids, iids = _RUN_ID_RE.findall(text), _IID_RE.findall(text)
        if not (run_ids and iids):
            return _script_error("no pending interaction visible for the second turn")
        return {"tool": "delegate_answer", "arguments": {
            "run_id": run_ids[-1], "interaction_id": iids[-1],
            "answers": [{"question_id": "q1", "free_text": exact2}]}}

    def wait(body: dict) -> dict:
        ids = _RUN_ID_RE.findall(body_text(body))
        return {"tool": "delegate_wait", "arguments": {"run_id": ids[-1]}} if ids else _script_error("no run id")

    def start(prompt: str) -> dict:
        return {"tool": "delegate_start", "arguments": {"subagent_id": "cx-scout", "prompt": prompt}}

    def wait_c(body: dict) -> dict:
        ids = _child_ids(body_text(body))
        return {"tool": "wait_tasks", "arguments": {"task_ids": [ids["PEER C"]], "timeout_sec": 240, "mode": "all_terminal"}}

    def dispose_c(body: dict) -> dict:
        text = body_text(body)
        tid = _child_ids(text).get("PEER C", "")
        match = re.search(rf'"task_id":\s*"{tid}".{{0,600}}?"child_result_sha256":\s*"([0-9a-f]{{64}})"', text, re.S)
        if not (tid and match):
            return _script_error("no exact result hash for C")
        return {"tool": "tree_note", "arguments": {
            "kind": "decision", "text": "Absorbed C's final.",
            "payload": {"type": "child_result_disposition", "child_task_id": tid,
                        "disposition": "integrated", "child_result_sha256": match.group(1)}}}

    return {
        ("root", "mock-model|tools", 1): {"tool": "schedule_subagent", "arguments": {
            "subagent_id": "peer-c",
            "objective": "PEER C: send your parent ONE original position with forward_to_worker, then finish with a FINAL line. " + PARTICIPATION_RULES,
            "expected_output": "One original to your parent, then a FINAL line."}},
        ("root", "mock-model|tools", 2): start(FAKE_TURN_MARKER + " Hold interim positions; each answer you receive is an addressed original. " + PARTICIPATION_RULES),
        ("root", "mock-model|tools", 3): wait,           # immediate waiting_on_user, turn 1
        ("root", "mock-model|tools", 4): wait,           # the RE-WAIT of a question already seen: parks until C's original wakes it
        ("root", "mock-model|tools", 5): _hold_until(lambda t: bool(_c_original_in(t)[1]), relay_c, label="C's original"),
        ("root", "mock-model|tools", 6): wait,           # waiting_on_user, turn 2 — same run, new interaction id
        ("root", "mock-model|tools", 7): answer_own,
        ("root", "mock-model|tools", 8): wait,           # terminal: FINAL + sha256 of both relayed originals
        ("root", "mock-model|tools", 9): start(FAKE_INPUT_REQUIRED_MARKER + " codex-shaped participant: no mid-run channel."),
        ("root", "mock-model|tools", 10): wait,          # terminal needing input: continuation=new_physical_run
        ("root", "mock-model|tools", 11): lambda body: start("NEW run carrying the prior turns: " + n.get("exact1", "") + "\n" + exact2),
        ("root", "mock-model|tools", 12): wait,          # the new run settles
        ("root", "mock-model|tools", 13): wait_c,
        ("root", "mock-model|tools", 14): dispose_c,
        ("root", "mock-model|tools", 15): {"final": f"S29_PARENT_FINAL {n['pf']}: two addressed turns in one session, then a new run."},
        # C holds its original until the custody trail shows P parked in the re-wait
        # (nothing addresses C, so each short hold times out), then contributes and
        # ends its participation.
        ("root", "mock-peer-c|tools", 1): _hold_until(
            lambda _text: bool(gate.get("parked")) and gate["parked"](), _forward_to_parent(c_original),
            label="P parked in its re-wait", await_sec=_C_POLL_SEC, rounds_max=_C_HOLD_ROUNDS_MAX),
        ("root", "mock-peer-c|tools", 2): {"final": f"FINAL: C {n['c']} — my participation ends."},
    }


@pytest.mark.integration
@pytest.mark.serial
def test_s29_session_run_resumes_twice_with_exact_relayed_originals_and_codex_twin_starts_new_run(
        e2e_clone, tmp_path_factory):
    require_lane(LANE_MOCK)
    root = tmp_path_factory.mktemp("s29")
    data_root = pathlib.Path(root) / "data"
    n = {key: uuid.uuid4().hex[:10] for key in ("c", "p", "pf")}
    gate: dict = {}
    model = _PeerReplayModel(_s29_fixture(n, gate), slot_binder=_s6_slot_binder,
                             model_ids=["mock-model", "mock-peer-c"])
    before = _home_sentinel()
    with FakeClaudexorDaemon() as daemon, model:
        daemon.install(data_root / "claudexor")
        settings = keyless_settings(
            model, OUROBOROS_SUBAGENTS=_roster(_SCOUT_ROW, _peer_row("peer-c")),
            OUROBOROS_DELEGATE_WAIT_SEC=5,  # the re-wait returns its window payload quickly
            **_isolation_settings(pathlib.Path(root)))
        server = start_server(e2e_clone, root, settings)
        try:
            oracle = ArtifactOracle(server.data_root)
            parent_id, _chat = _room_task(server, "Run one session through two addressed turns, then a new run.", f"w7-s29-{n['pf']}")

            def p_entered_rewait() -> list:
                """P's supervised waits so far, from the custody rows ``supervised_wait``
                appends SYNCHRONOUSLY before its first quiet window: the second one is the
                re-wait (the first is round 3's immediate waiting_on_user wait), so a message
                C writes once two exist is a WAKE of the parked wait, never a round-top drain."""
                return [row for row in oracle.events("delegate_supervision_wait_entered")
                        if str(row.get("task_id") or "") == parent_id]

            gate["parked"] = lambda: len(p_entered_rewait()) >= 2
            assert wait_until(lambda: parent_id in oracle.running_ids(), 120)
            result = server.wait_task(parent_id, timeout=900)
            assert result.get("status") == "completed", result
            stored = wait_durable_result(oracle, parent_id)
            assert f"S29_PARENT_FINAL {n['pf']}" in str(stored.get("result") or ""), stored
            [c_id] = oracle.child_task_ids(parent_id)
            c_row = wait_durable_result(oracle, c_id)
            assert c_row.get("status") == "completed" and str(c_row.get("result") or "").startswith("FINAL:"), c_row
            p_drive = oracle.task_drive(parent_id)

            # The retained original: the exact bytes C wrote (its durable forward row, typed as a
            # peer contribution to its parent) == the bytes P relayed under its attribution line.
            [c_forward] = _forwards(oracle, c_id)
            assert c_forward["args"] == {"task_id": parent_id, "message": n["c_original"]}, c_forward
            assert "message from a peer task (your parent" in str(c_forward.get("result_preview") or "")
            # Ordering, as observed: C published only after P's re-wait was entered.
            entered = p_entered_rewait()
            assert len(entered) >= 2 and str(c_forward.get("ts") or "") >= str(entered[1].get("ts") or ""), (
                [row.get("ts") for row in entered], c_forward.get("ts"))
            exact1 = n.get("exact1")
            assert exact1 and exact1.endswith(n["c_original"]) and exact1.startswith(
                f"[Original from peer task {c_id} (your child); relayed verbatim by {parent_id}]\n"), exact1

            # Wire truth: THREE physical starts (turn run, codex twin, its NEW run); the turn run
            # got exactly two answer POSTs on two DISTINCT interaction paths with byte-exact freeText.
            starts = daemon.run_start_posts()
            assert len(starts) == 3, starts
            assert FAKE_TURN_MARKER in starts[0]["body"]["prompt"]
            assert FAKE_INPUT_REQUIRED_MARKER in starts[1]["body"]["prompt"]
            assert exact1 in starts[2]["body"]["prompt"] and n["exact2"] in starts[2]["body"]["prompt"]
            turn_rid = next(rid for rid, run in daemon.runs.items() if FAKE_TURN_MARKER in run["body"]["prompt"])
            answer_posts = [row for row in daemon.calls("POST", f"/v2/runs/{turn_rid}/interactions/") if row["path"].endswith("/answer")]
            assert [row["path"] for row in answer_posts] == [
                f"/v2/runs/{turn_rid}/interactions/turn-{turn_rid[:8]}-{k}/answer" for k in (1, 2)], answer_posts
            assert answer_posts[0]["body"] == {"answers": [{"questionId": "q1", "selectedLabels": [], "freeText": exact1}]}
            assert answer_posts[1]["body"] == {"answers": [{"questionId": "q1", "selectedLabels": [], "freeText": n["exact2"]}]}
            assert daemon.runs[turn_rid]["state"] == "succeeded" and daemon.runs[turn_rid]["pending"] == []

            # Custody: one started row per run; the turn run answered twice (delivered, distinct
            # ids), settled succeeded, never cancelled; no fourth start anywhere.
            assert len(_custody_rows(oracle, "delegate_run_started")) == 3
            answered = _custody_rows(oracle, "delegate_interaction_answered", turn_rid)
            assert [row.get("status") for row in answered] == ["delivered", "delivered"], answered
            assert len({row.get("interaction_id") for row in answered}) == 2, answered
            settled = _custody_rows(oracle, "delegate_run_settled", turn_rid)
            assert settled and settled[-1].get("state") == "succeeded", settled
            assert _custody_rows(oracle, "delegate_run_cancel_outcome", turn_rid) == []

            # The model-visible route facts, read from the durable wake payloads the host handed
            # the model (tools.jsonl previews are cut at 2000 chars): the two immediate waiting
            # payloads AND the re-wait of the already-seen question — delivered as the
            # addressed-message WAKE that carried C's original — state continuation=same_session;
            # the codex terminal states new_physical_run beside the input_required note.
            wakes = oracle.events("delegate_supervision_wake_pending")
            turn_wakes = [row["payload"] for row in wakes if str(row.get("run_id") or "") == turn_rid]
            waiting = [p for p in turn_wakes if p.get("status") == "waiting_on_user"]
            rewaits = [p for p in turn_wakes if p.get("status") in {"progress", "no_progress"} and p.get("waiting_on_user") is True]
            assert len(waiting) == 2 and len(rewaits) == 1, [(p.get("status"), p.get("waiting_on_user")) for p in turn_wakes]
            for payload in waiting + rewaits:
                assert payload.get("continuation") == "same_session", payload
            assert all(payload.get("continuation_note") for payload in waiting)
            assert not any("contribution" in json.dumps(payload) for payload in turn_wakes)
            [rewait] = rewaits
            events_in_wake = rewait.get("wake_events") or []
            assert any(e.get("type") == "addressed_message" and e.get("source_task_id") == c_id
                       and e.get("provenance") == "peer_task" and e.get("relation") == "parent"
                       and e.get("text") == n["c_original"]
                       for e in events_in_wake), "C's original did not arrive as the parked re-wait's wake"
            inreq_rid = next(rid for rid, run in daemon.runs.items() if FAKE_INPUT_REQUIRED_MARKER in run["body"]["prompt"])
            inreq_terminal = [row["payload"] for row in wakes if str(row.get("run_id") or "") == inreq_rid][-1]
            assert inreq_terminal.get("status") == "terminal" and inreq_terminal.get("continuation") == "new_physical_run", inreq_terminal
            assert "input_required_note" in inreq_terminal and inreq_terminal["outcome_facts"]["reason"] == "input_required"
            # The tools.jsonl trail names every wait; the transcript the stub saw holds the full
            # results, so the fact is countable there too (2 immediate + 1 re-wait wake).
            assert len(_tool_rows(p_drive, "delegate_wait")) == 6
            p_text = "\n".join(body_text(body) for kind, body in model.calls if _s6_slot_binder(body) == "mock-model|tools")
            assert p_text.count('"continuation": "same_session"') >= 3
            assert '"continuation": "new_physical_run"' in p_text and "input_required_note" in p_text

            # The run's terminal carried the exact relayed bytes back: FINAL plus both sha256s.
            terminal_text = str((daemon._detail(daemon.runs[turn_rid]).get("primaryOutput") or {}).get("text") or "")
            assert terminal_text.startswith("FINAL:")
            for original in (exact1, n["exact2"]):
                assert hashlib.sha256(original.encode("utf-8")).hexdigest() in p_text
                assert f"sha256={hashlib.sha256(original.encode('utf-8')).hexdigest()}\n{original}\n" in terminal_text

            assert wait_until(lambda: parent_id not in oracle.running_ids(), 60)
            _dump_misses(model, pathlib.Path(root))
            model.assert_consumed()
        finally:
            server.stop()
    assert _home_sentinel() == before, "the isolated server touched the operator's durable roots"
