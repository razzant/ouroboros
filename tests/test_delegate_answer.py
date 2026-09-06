"""Poltergeist phase B (B4): the nanny answers its run's questions.

The engine has carried the ENTIRE interactive-question pipeline since 3.3.x —
durable store, full question text/options on the run detail
(``pendingInteractions``), a typed answer endpoint, a 15-minute benign-decline
timeout — while our ``delegate_wait`` read only ``bool(summary.waitingOnUser)``
and showed it at window expiry. A paused run therefore burned metered polling
for up to the whole engine timeout. Phase B: the wait returns IMMEDIATELY with
the typed question set, and the custody-gated ``delegate_answer`` delivers the
nanny's answer (owner decision 7=A: the nanny answers from task context;
above-authority questions ride the escalation verb up the task hierarchy
while the run waits out the engine timeout).
"""

import json
import queue as stdqueue

import pytest


@pytest.fixture(autouse=True)
def _owned_gateway_uses_each_test_transport(monkeypatch):
    from ouroboros import claudexor_daemon
    from ouroboros.gateways import claudexor as gateway_module

    monkeypatch.setattr(
        claudexor_daemon,
        "ensure_owned_gateway",
        lambda: gateway_module.ClaudexorGateway(),
    )


@pytest.fixture(autouse=True)
def _fresh_interaction_memo():
    from ouroboros.tools import delegate

    delegate._REPORTED_INTERACTIONS.clear()
    yield
    delegate._REPORTED_INTERACTIONS.clear()


def _pending_row(iid="int-1", question="Which port should the server use?"):
    return {
        "interactionId": iid,
        "runId": "run-1",
        "attemptId": "a01",
        "harnessId": "claude",
        "sourceTool": "AskUserQuestion",
        "questions": [{
            "id": "q1",
            "question": question,
            "header": "Port",
            "options": [{"label": "8080", "description": "the default"},
                        {"label": "9090", "description": None}],
            "multi_select": False,
        }],
        "requestedAt": "2026-08-11T10:00:00Z",
        "timeoutAt": "2026-08-11T10:15:00Z",
    }


# -- the gateway readers --------------------------------------------------------


def test_pending_interactions_normalizes_the_full_question_shape():
    from ouroboros.gateways.claudexor import pending_interactions

    rows = pending_interactions({"pendingInteractions": [
        _pending_row(),
        {"interactionId": "", "questions": []},   # unanswerable: dropped
        "junk",
    ]})
    assert len(rows) == 1
    row = rows[0]
    assert row["interaction_id"] == "int-1"
    assert row["source_tool"] == "AskUserQuestion"
    assert row["timeout_at"] == "2026-08-11T10:15:00Z"
    q = row["questions"][0]
    assert q["question_id"] == "q1"
    assert q["question"] == "Which port should the server use?"
    assert q["options"][0] == {"label": "8080", "description": "the default"}
    assert q["multi_select"] is False


def test_answer_interaction_returns_typed_statuses_at_any_http_code(monkeypatch):
    import httpx

    from ouroboros.gateways import claudexor as cx

    replies = {}

    class _Recorder:
        def request(self, method, path, **kwargs):
            replies["path"] = path
            replies["json"] = kwargs.get("json")
            return httpx.Response(replies["code"], json=replies["body"])

    gateway = cx.ClaudexorGateway(cx.DaemonEndpoint("127.0.0.1", 1, "secret"))
    gateway.close()
    gateway._client = _Recorder()

    replies.update(code=200, body={"accepted": True, "status": "delivered"})
    body = gateway.answer_interaction("run-1", "int-1", [
        {"questionId": "q1", "selectedLabels": ["8080"], "freeText": None}])
    assert body["status"] == "delivered"
    assert replies["path"] == "/v2/runs/run-1/interactions/int-1/answer"
    assert replies["json"] == {"answers": [
        {"questionId": "q1", "selectedLabels": ["8080"], "freeText": None}]}

    # A 409 with a typed body is an ANSWER, not an outage.
    replies.update(code=409, body={"accepted": False, "status": "already_resolved",
                                   "message": "resolved earlier"})
    assert gateway.answer_interaction("run-1", "int-1", [])["status"] == "already_resolved"

    # A bodyless 404 ("no such run") stays the typed refusal it is.
    replies.update(code=404, body={"error": "no such run"})
    with pytest.raises(cx.ClaudexorUnavailable) as exc:
        gateway.answer_interaction("run-gone", "int-1", [])
    assert exc.value.status_code == 404

    # 501: this engine build has no answer service.
    replies.update(code=501, body={"error": "interaction answers are not supported"})
    with pytest.raises(cx.ClaudexorUnavailable) as exc:
        gateway.answer_interaction("run-1", "int-1", [])
    assert exc.value.status_code == 501


# -- delegate_wait surfaces the question ----------------------------------------


def _wait_ctx(tmp_path):
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.tools.registry import ToolContext

    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    ctx = ToolContext(repo_dir=repo, drive_root=tmp_path,
                      task_constraint=TaskConstraint(mode="local_readonly_subagent"))
    ctx.task_id = "t-nanny"
    ctx.event_queue = stdqueue.Queue()
    return ctx


def _waiting_stub(monkeypatch, pending_rows):
    from ouroboros.gateways import claudexor as gw

    class _Stub:
        engine_version = "3.3.6"

        def handshake(self, **_kw): return {}
        def get_run(self, rid, *, timeout_sec=None):
            return {"lastSeq": 5,
                    "pendingInteractions": list(pending_rows),
                    "summary": {"state": "running", "effectiveAccess": "readonly",
                                "waitingOnUser": bool(pending_rows)}}
        def close(self): pass

    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _Stub())


def _own_run(delegate):
    delegate._CUSTODY.clear()
    delegate._CUSTODY["run-1"] = delegate._RunCustody(
        task_id="t-nanny", route_id="some-route", model="m",
        project_id="prj", project_owned=False,
    )


def test_a_new_question_returns_immediately_with_the_full_text(tmp_path, monkeypatch):
    import ouroboros.tools.delegate as delegate

    _waiting_stub(monkeypatch, [_pending_row()])
    ctx = _wait_ctx(tmp_path)
    _own_run(delegate)
    out = json.loads(delegate._delegate_wait(ctx, "run-1", wait_sec=600, since_seq=5))
    delegate._CUSTODY.clear()
    assert out["status"] == "waiting_on_user"
    assert out["last_seq"] == 5
    row = out["pending_interactions"][0]
    assert row["interaction_id"] == "int-1"
    assert row["questions"][0]["question"] == "Which port should the server use?"
    assert row["questions"][0]["options"][0]["label"] == "8080"
    assert "delegate_answer" in out["note"]
    assert "ABOVE your authority" in out["note"]


def test_the_same_question_does_not_busy_loop_the_next_wait(tmp_path, monkeypatch):
    """A nanny that escalated up the hierarchy and re-waits must HOLD its
    window, not spin: the known question rides the expiry payload instead."""
    import ouroboros.tools.delegate as delegate

    _waiting_stub(monkeypatch, [_pending_row()])
    ctx = _wait_ctx(tmp_path)
    _own_run(delegate)
    first = json.loads(delegate._delegate_wait(ctx, "run-1", wait_sec=600, since_seq=5))
    assert first["status"] == "waiting_on_user"
    second = json.loads(delegate._delegate_wait(ctx, "run-1", wait_sec=1, since_seq=5))
    delegate._CUSTODY.clear()
    assert second["status"] == "no_progress"
    assert second["waiting_on_user"] is True
    assert second["pending_interactions"][0]["interaction_id"] == "int-1"


def test_a_reask_with_a_new_interaction_id_is_news_again(tmp_path, monkeypatch):
    import ouroboros.tools.delegate as delegate

    rows = [_pending_row()]
    _waiting_stub(monkeypatch, rows)
    ctx = _wait_ctx(tmp_path)
    _own_run(delegate)
    assert json.loads(delegate._delegate_wait(
        ctx, "run-1", wait_sec=600, since_seq=5))["status"] == "waiting_on_user"
    rows[0] = _pending_row(iid="int-2")
    out = json.loads(delegate._delegate_wait(ctx, "run-1", wait_sec=600, since_seq=5))
    delegate._CUSTODY.clear()
    assert out["status"] == "waiting_on_user"
    assert out["pending_interactions"][0]["interaction_id"] == "int-2"


def test_an_oversized_question_set_spills_whole_with_a_receipt(tmp_path, monkeypatch):
    import ouroboros.tools.delegate as delegate

    big = [_pending_row(iid=f"int-{i}", question=("What about part %d? " % i) + "x" * 900)
           for i in range(24)]
    _waiting_stub(monkeypatch, big)
    ctx = _wait_ctx(tmp_path)
    _own_run(delegate)
    out = json.loads(delegate._delegate_wait(ctx, "run-1", wait_sec=600, since_seq=5))
    delegate._CUSTODY.clear()
    assert out["status"] == "waiting_on_user"
    delivery = out["interactions_delivery"]
    artifact = delivery["artifact"]
    assert artifact["sha256"] and artifact["bytes"] > 0
    assert artifact["path"].endswith(".interactions.json")
    staged = json.loads((tmp_path / "task_drive" / artifact["path"]).read_text(encoding="utf-8")
                        if (tmp_path / "task_drive" / artifact["path"]).exists()
                        else (tmp_path / artifact["path"]).read_text(encoding="utf-8")
                        if (tmp_path / artifact["path"]).exists()
                        else open(artifact["abs_path"], encoding="utf-8").read())
    assert len(staged["pending_interactions"]) == 24
    # The inline view is a COUNTED preview, and the whole payload respects the budget.
    from ouroboros.tool_capabilities import tool_result_limit

    assert len(json.dumps(out, ensure_ascii=False, indent=2)) <= tool_result_limit("delegate_wait")


# -- delegate_answer ------------------------------------------------------------


def _answer_ctx(tmp_path):
    return _wait_ctx(tmp_path)


def _answer_stub(monkeypatch, *, result=None, error=None, detail_pending=()):
    from ouroboros.gateways import claudexor as gw

    class _Stub:
        engine_version = "3.3.6"

        def handshake(self, **_kw): return {}
        def get_run(self, rid, *, timeout_sec=None):
            return {"lastSeq": 5, "pendingInteractions": list(detail_pending),
                    "summary": {"state": "running"}}
        def answer_interaction(self, rid, iid, answers):
            if error is not None:
                raise error
            return dict(result)
        def close(self): pass

    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _Stub())


def test_delivered_answer_relays_typed_and_writes_the_custody_row(tmp_path, monkeypatch):
    import ouroboros.tools.delegate as delegate

    _answer_stub(monkeypatch, result={"accepted": True, "status": "delivered"})
    ctx = _answer_ctx(tmp_path)
    _own_run(delegate)
    out = json.loads(delegate._delegate_answer(ctx, "run-1", "int-1", [
        {"question_id": "q1", "selected_labels": ["8080"]},
    ]))
    delegate._CUSTODY.clear()
    assert out["status"] == "delivered" and out["accepted"] is True
    assert "delegate_wait" in out["note"]
    events = [json.loads(line) for line in
              (tmp_path / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    answered = [e for e in events if e["type"] == "delegate_interaction_answered"]
    assert answered and answered[0]["interaction_id"] == "int-1"
    assert answered[0]["status"] == "delivered"


def test_already_resolved_tells_the_nanny_not_to_repost(tmp_path, monkeypatch):
    import ouroboros.tools.delegate as delegate

    _answer_stub(monkeypatch, result={"accepted": False, "status": "already_resolved",
                                      "message": "timed out"})
    ctx = _answer_ctx(tmp_path)
    _own_run(delegate)
    out = json.loads(delegate._delegate_answer(ctx, "run-1", "int-1", [
        {"question_id": "q1", "free_text": "9090"},
    ]))
    delegate._CUSTODY.clear()
    assert out["status"] == "already_resolved"
    assert "do NOT re-post" in out["note"]


def test_ambiguous_transport_becomes_delivery_unknown_with_a_reread(tmp_path, monkeypatch):
    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    # Re-read shows the interaction still pending: retry the SAME answer.
    _answer_stub(monkeypatch,
                 error=ClaudexorUnavailable("daemon_unreachable", "boom"),
                 detail_pending=[_pending_row()])
    ctx = _answer_ctx(tmp_path)
    _own_run(delegate)
    out = json.loads(delegate._delegate_answer(ctx, "run-1", "int-1", [
        {"question_id": "q1", "free_text": "9090"},
    ]))
    assert out["status"] == "delivery_unknown"
    assert out["still_pending"] is True
    assert "SAME answers" in out["note"]

    # Re-read shows it gone: never re-post, never a different answer.
    _answer_stub(monkeypatch,
                 error=ClaudexorUnavailable("daemon_unreachable", "boom"),
                 detail_pending=[])
    out = json.loads(delegate._delegate_answer(ctx, "run-1", "int-1", [
        {"question_id": "q1", "free_text": "9090"},
    ]))
    delegate._CUSTODY.clear()
    assert out["status"] == "delivery_unknown"
    assert out["still_pending"] is False
    assert "NEVER post a different answer" in out["note"]


def test_answers_are_validated_and_custody_gated(tmp_path, monkeypatch):
    import ouroboros.tools.delegate as delegate

    _answer_stub(monkeypatch, result={"accepted": True, "status": "delivered"})
    ctx = _answer_ctx(tmp_path)
    _own_run(delegate)

    out = json.loads(delegate._delegate_answer(ctx, "run-1", "int-1", []))
    assert out["status"] == "refused" and out["reason"] == "answers_required"

    out = json.loads(delegate._delegate_answer(ctx, "run-1", "int-1", [{"free_text": "x"}]))
    assert out["status"] == "refused" and out["reason"] == "answer_row_invalid"

    # Another task's run: custody refuses before any daemon call.
    delegate._CUSTODY["run-1"] = delegate._RunCustody(
        task_id="someone-else", route_id="r", model="m",
        project_id="prj", project_owned=False,
    )
    out = json.loads(delegate._delegate_answer(ctx, "run-1", "int-1", [
        {"question_id": "q1", "free_text": "x"},
    ]))
    delegate._CUSTODY.clear()
    assert out["status"] == "refused" and out["reason"] == "run_not_owned"


def test_unsupported_engine_build_is_a_typed_refusal(tmp_path, monkeypatch):
    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    _answer_stub(monkeypatch, error=ClaudexorUnavailable(
        "http_501", "interaction answers are not supported", status_code=501))
    ctx = _answer_ctx(tmp_path)
    _own_run(delegate)
    out = json.loads(delegate._delegate_answer(ctx, "run-1", "int-1", [
        {"question_id": "q1", "free_text": "x"},
    ]))
    delegate._CUSTODY.clear()
    assert out["status"] == "refused"
    assert out["reason"] == "interaction_answers_unsupported"
    assert "benign-decline" in out["detail"]


# -- fix batch: refusal mapping, memo, budget, validation ------------------------


def test_definite_4xx_maps_to_the_rejected_shape_not_delivery_unknown(tmp_path, monkeypatch):
    """F3 (races #1): a definite engine 4xx is an ANSWER about these bytes.
    Relaying it as `delivery_unknown` invited re-posting the same bytes; the
    rejected shape says fix the rows. `delivery_unknown` stays reserved for
    status 0 / 5xx / transport death."""
    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    ctx = _answer_ctx(tmp_path)

    # A 400 ControlProblem (typed refusal without an interaction status).
    _answer_stub(monkeypatch, error=ClaudexorUnavailable(
        "http_400", "ControlProblem: answers failed schema validation", status_code=400))
    _own_run(delegate)
    out = json.loads(delegate._delegate_answer(ctx, "run-1", "int-1", [
        {"question_id": "q1", "free_text": "x"},
    ]))
    assert out["status"] == "rejected", out
    assert out["accepted"] is False
    assert "do not re-post the same bytes" in out["note"]
    assert "HTTP 400" in out["note"]

    # A 409 whose body carried NO typed status (the untyped-conflict fake).
    _answer_stub(monkeypatch, error=ClaudexorUnavailable(
        "http_409", "conflict without a typed body", status_code=409))
    out = json.loads(delegate._delegate_answer(ctx, "run-1", "int-1", [
        {"question_id": "q1", "free_text": "x"},
    ]))
    assert out["status"] == "rejected", out
    assert "HTTP 409" in out["note"]

    # 413/422 are payload-semantic too (R2-1): still the rejected shape.
    for code in (413, 422):
        _answer_stub(monkeypatch, error=ClaudexorUnavailable(
            f"http_{code}", "typed refusal", status_code=code))
        out = json.loads(delegate._delegate_answer(ctx, "run-1", "int-1", [
            {"question_id": "q1", "free_text": "x"},
        ]))
        assert out["status"] == "rejected", out
        assert f"HTTP {code}" in out["note"]

    # A 503 stays ambiguous: the answer MAY have landed.
    _answer_stub(monkeypatch, error=ClaudexorUnavailable(
        "http_503", "bad gateway", status_code=503))
    out = json.loads(delegate._delegate_answer(ctx, "run-1", "int-1", [
        {"question_id": "q1", "free_text": "x"},
    ]))
    delegate._CUSTODY.clear()
    assert out["status"] == "delivery_unknown", out


def test_auth_and_rate_4xx_stay_delivery_unknown_not_rejected(tmp_path, monkeypatch):
    """R2-1 (two reviewers converged): only the payload-semantic codes
    (400/409/413/422) are a verdict about these bytes. A 401/403/408/429 says
    nothing about the rows, so it is the AMBIGUOUS shape — whose bounded
    re-read then correctly advises retrying the SAME answers while the row is
    still pending. The old blanket 4xx→rejected told the nanny to REWRITE
    answers an auth blip never even judged."""
    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    ctx = _answer_ctx(tmp_path)
    for code in (401, 403, 408, 429):
        _answer_stub(monkeypatch,
                     error=ClaudexorUnavailable(f"http_{code}", "no", status_code=code),
                     detail_pending=[_pending_row()])
        _own_run(delegate)
        out = json.loads(delegate._delegate_answer(ctx, "run-1", "int-1", [
            {"question_id": "q1", "free_text": "x"},
        ]))
        assert out["status"] == "delivery_unknown", (code, out)
        assert out["still_pending"] is True
        assert "SAME answers" in out["note"]
    delegate._CUSTODY.clear()


def test_a_spent_subscription_window_is_schedulable_not_flattened(tmp_path, monkeypatch):
    """R2-1: ClaudexorSubscriptionWindowExhausted keeps its own typed outcome
    carrying reset_at — a schedulable condition (review_execution plans against
    the same class), never flattened into `rejected` (which would tell the
    nanny to rewrite perfectly valid rows)."""
    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways.claudexor import ClaudexorSubscriptionWindowExhausted

    _answer_stub(monkeypatch, error=ClaudexorSubscriptionWindowExhausted(
        "window spent", reset_at="2026-08-11T22:00:00Z", status_code=429))
    ctx = _answer_ctx(tmp_path)
    _own_run(delegate)
    out = json.loads(delegate._delegate_answer(ctx, "run-1", "int-1", [
        {"question_id": "q1", "free_text": "x"},
    ]))
    delegate._CUSTODY.clear()
    assert out["status"] == "subscription_window_exhausted", out
    assert out["reset_at"] == "2026-08-11T22:00:00Z"
    assert out["accepted"] is False
    assert "SAME answers" in out["note"]


def test_a_delivered_answer_pops_the_reported_memo_so_the_next_wait_reports(tmp_path, monkeypatch):
    """F6 (gemini #2): after the engine resolves an interaction, the memo of
    already-shown questions is stale — a re-ask (or the rest of the set) must be
    news again on the very next wait, not held for a full window."""
    import ouroboros.tools.delegate as delegate

    _answer_stub(monkeypatch, result={"accepted": True, "status": "delivered"})
    ctx = _answer_ctx(tmp_path)
    _own_run(delegate)
    delegate._REPORTED_INTERACTIONS["run-1"] = frozenset({"int-1"})
    out = json.loads(delegate._delegate_answer(ctx, "run-1", "int-1", [
        {"question_id": "q1", "selected_labels": ["8080"]},
    ]))
    assert out["status"] == "delivered"
    assert "run-1" not in delegate._REPORTED_INTERACTIONS

    # already_resolved pops it too; a plain refusal does not.
    _answer_stub(monkeypatch, result={"accepted": False, "status": "already_resolved"})
    delegate._REPORTED_INTERACTIONS["run-1"] = frozenset({"int-1"})
    json.loads(delegate._delegate_answer(ctx, "run-1", "int-1", [
        {"question_id": "q1", "free_text": "x"},
    ]))
    assert "run-1" not in delegate._REPORTED_INTERACTIONS

    _answer_stub(monkeypatch, result={"accepted": False, "status": "rejected",
                                      "message": "bad rows"})
    delegate._REPORTED_INTERACTIONS["run-1"] = frozenset({"int-1"})
    json.loads(delegate._delegate_answer(ctx, "run-1", "int-1", [
        {"question_id": "q1", "free_text": "x"},
    ]))
    delegate._CUSTODY.clear()
    assert delegate._REPORTED_INTERACTIONS.get("run-1") == frozenset({"int-1"})


def test_an_unexpected_exception_becomes_typed_delivery_unknown_not_a_traceback(tmp_path, monkeypatch):
    """F7 (gemini #3): a broken gateway body (or any unexpected failure around
    the call) reaches the model as the typed ambiguous outcome, never as a raw
    traceback it can only retry blindly against."""
    import ouroboros.tools.delegate as delegate

    _answer_stub(monkeypatch, error=KeyError("malformed body surprise"))
    ctx = _answer_ctx(tmp_path)
    _own_run(delegate)
    out = json.loads(delegate._delegate_answer(ctx, "run-1", "int-1", [
        {"question_id": "q1", "free_text": "x"},
    ]))
    delegate._CUSTODY.clear()
    assert out["status"] == "delivery_unknown", out
    assert "NEVER post a different answer" in out["note"]


def test_an_exhausted_internal_budget_returns_typed_without_further_wire_calls(tmp_path, monkeypatch):
    """F8 (sol #5): the call runs under an internal monotonic deadline strictly
    below its 120s ToolEntry timeout; once it is spent, the POST and the re-read
    are both SKIPPED and the outcome is the typed delivery_unknown."""
    import ouroboros.delegate_interactions as interactions
    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways import claudexor as gw

    assert interactions._ANSWER_DEADLINE_SEC < 120

    wire_calls = []

    class _Stub:
        engine_version = "3.3.6"

        def handshake(self, **_kw): return {}
        def get_run(self, rid, *, timeout_sec=None):
            wire_calls.append(("get_run", rid))
            return {"lastSeq": 5, "summary": {"state": "running"}}
        def answer_interaction(self, rid, iid, answers):
            wire_calls.append(("answer", rid))
            raise AssertionError("the POST must not be sent on a spent budget")
        def close(self): pass

    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _Stub())
    # Spend the budget instantly: the deadline computes to "already passed".
    monkeypatch.setattr(interactions, "_ANSWER_DEADLINE_SEC", -1.0)
    ctx = _answer_ctx(tmp_path)
    _own_run(delegate)
    out = json.loads(delegate._delegate_answer(ctx, "run-1", "int-1", [
        {"question_id": "q1", "free_text": "x"},
    ]))
    delegate._CUSTODY.clear()
    assert out["status"] == "delivery_unknown", out
    assert out["still_pending"] is None
    assert "time budget" in out["transport_error"] or "time budget" in out["note"]
    assert wire_calls == [], "no wire calls after budget exhaustion"


def test_answer_rows_are_validated_strictly_before_the_post(tmp_path, monkeypatch):
    """F14 (sol #13): string-only labels, non-empty label-or-freeText per row,
    typed refusal on malformed input — no silent coercion that changes intent."""
    import ouroboros.tools.delegate as delegate

    sent = []

    def _capture_stub(monkeypatch):
        from ouroboros.gateways import claudexor as gw

        class _Stub:
            engine_version = "3.3.6"

            def handshake(self, **_kw): return {}
            def get_run(self, rid, *, timeout_sec=None):
                return {"lastSeq": 5, "summary": {"state": "running"}}
            def answer_interaction(self, rid, iid, answers):
                sent.append(answers)
                return {"accepted": True, "status": "delivered"}
            def close(self): pass

        monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _Stub())

    _capture_stub(monkeypatch)
    ctx = _answer_ctx(tmp_path)
    _own_run(delegate)

    # Non-string label: refused, nothing posted (8080 as an int is NOT "8080").
    out = json.loads(delegate._delegate_answer(ctx, "run-1", "int-1", [
        {"question_id": "q1", "selected_labels": [8080]},
    ]))
    assert out["status"] == "refused" and out["reason"] == "answer_row_invalid"

    # Non-string free_text: refused.
    out = json.loads(delegate._delegate_answer(ctx, "run-1", "int-1", [
        {"question_id": "q1", "free_text": 42},
    ]))
    assert out["status"] == "refused" and out["reason"] == "answer_row_invalid"

    # Empty row (no labels, no text): refused as empty, not posted as "an answer".
    out = json.loads(delegate._delegate_answer(ctx, "run-1", "int-1", [
        {"question_id": "q1", "selected_labels": [], "free_text": "  "},
    ]))
    assert out["status"] == "refused" and out["reason"] == "answer_row_empty"
    assert sent == [], "nothing malformed ever reached the wire"

    # A valid row still flows, uncoerced.
    out = json.loads(delegate._delegate_answer(ctx, "run-1", "int-1", [
        {"question_id": "q1", "selected_labels": ["8080"]},
    ]))
    delegate._CUSTODY.clear()
    assert out["status"] == "delivered"
    assert sent == [[{"questionId": "q1", "selectedLabels": ["8080"], "freeText": None}]]


# -- fix batch: expiry-path measurement, notes, spill identity, advances ---------


def _expiry_payload(pending_rows, *, advances=0, budget=None):
    from ouroboros.delegate_progress import WindowObservations, window_payload
    from ouroboros.gateways.claudexor import pending_interactions
    from ouroboros.tool_capabilities import tool_result_limit
    from ouroboros.tools.delegate import _bounded_interactions

    seen = WindowObservations()
    timeline = []
    for i in range(advances):
        timeline = timeline + [{"title": f"event {i}", "type": "tool"}]
        seen.record({"timeline": list(timeline)}, i + 1, i)
    pending = pending_interactions({"pendingInteractions": pending_rows})
    return window_payload(
        run_id="run-1", state="running", last_seq=max(5, advances),
        window=600, elapsed_seconds=600, max_seconds=1800,
        waiting_on_user=bool(pending), detail={"timeline": timeline}, seen=seen,
        pending_interactions=_bounded_interactions(pending) if pending else None,
        budget=budget if budget is not None else tool_result_limit("delegate_wait"))


def _giant_header_row(iid="int-1", header_chars=50_000):
    row = _pending_row(iid=iid)
    row["questions"][0]["header"] = "H" * header_chars
    return row


@pytest.mark.parametrize("n_rows", [2, 3])
def test_expiry_payload_with_giant_headers_fits_and_parses(n_rows):
    """F2 (two fable lenses + sol #4; probes 24 459 / 51 719 chars vs 15 000):
    harness-authored scalars beyond question/options — a 50k header — pushed the
    'bounded' expiry projection past the tool budget, where the EXTERNAL
    truncator severed the JSON mid-structure. Both branches (no_progress and
    progress) must ship a payload that fits whole and round-trips."""
    from ouroboros.loop_tool_execution import _truncate_tool_result
    from ouroboros.tool_capabilities import tool_result_limit

    rows = [_giant_header_row(iid=f"int-{i}") for i in range(n_rows)]
    limit = tool_result_limit("delegate_wait")

    # The early no_progress branch — the one that used to skip measurement.
    payload = _expiry_payload(rows, advances=0)
    raw = json.dumps(payload, ensure_ascii=False, indent=2)
    assert len(raw) <= limit, len(raw)
    assert _truncate_tool_result(raw, "delegate_wait", {}) == raw
    assert json.loads(raw) == payload
    assert payload["status"] == "no_progress"
    shown = len(payload.get("pending_interactions") or [])
    assert shown + int(payload.get("interactions_omitted") or 0) == n_rows

    # The progress branch, same rows plus a real advance sequence.
    payload = _expiry_payload(rows, advances=6)
    raw = json.dumps(payload, ensure_ascii=False, indent=2)
    assert len(raw) <= limit, len(raw)
    assert _truncate_tool_result(raw, "delegate_wait", {}) == raw
    assert json.loads(raw) == payload
    assert payload["status"] == "progress"
    assert payload["advances"], "the advance sequence survived beside the questions"
    shown = len(payload.get("pending_interactions") or [])
    assert shown + int(payload.get("interactions_omitted") or 0) == n_rows


def test_bounded_interactions_bounds_every_harness_authored_scalar():
    """F2: header, source tool and timestamps are harness-authored too. The
    ANSWER KEYS ride whole (R2-8) — they are echoed into delegate_answer, so a
    cut key is an id the engine never issued."""
    from ouroboros.gateways.claudexor import pending_interactions
    from ouroboros.tools.delegate import _bounded_interactions

    row = _pending_row()
    row["questions"][0]["header"] = "H" * 50_000
    row["sourceTool"] = "S" * 9_000
    row["requestedAt"] = "T" * 9_000
    row["timeoutAt"] = "T" * 9_000
    row["interactionId"] = "i" * 9_000
    row["questions"][0]["id"] = "q" * 9_000
    bounded = _bounded_interactions(pending_interactions({"pendingInteractions": [row]}))
    out = bounded[0]
    assert len(out["questions"][0]["header"]) <= 200
    assert len(out["source_tool"]) <= 200
    assert len(out["requested_at"]) <= 200
    assert len(out["timeout_at"]) <= 200
    assert out["interaction_id"] == "i" * 9_000  # a KEY, never truncated
    assert out["questions"][0]["question_id"] == "q" * 9_000
    assert "OMISSION NOTE" in out["questions"][0]["header"]


def test_answer_keys_ride_whole_through_the_inline_preview(tmp_path, monkeypatch):
    """R2-8: 200-char ids (over the old 160-char preview cut) must reach the
    model VERBATIM in the immediate waiting payload — they are echoed into
    delegate_answer, and a cut with an embedded marker yields engine
    not_found."""
    import ouroboros.tools.delegate as delegate

    iid = "I" * 200
    qid = "Q" * 200

    # The small-set path: full rows ride inline, ids untouched.
    row = _pending_row(iid=iid)
    row["questions"][0]["id"] = qid
    _waiting_stub(monkeypatch, [row])
    ctx = _wait_ctx(tmp_path)
    _own_run(delegate)
    out = json.loads(delegate._delegate_wait(ctx, "run-1", wait_sec=600, since_seq=5))
    assert out["status"] == "waiting_on_user"
    shown = out["pending_interactions"][0]
    assert shown["interaction_id"] == iid
    assert shown["questions"][0]["question_id"] == qid

    # The SPILLED path: the bounded preview cuts display fields, never the keys.
    big = _pending_row(iid="R" * 200, question="x" * 30_000)
    big["questions"][0]["id"] = "S" * 200
    _waiting_stub(monkeypatch, [big])
    delegate._REPORTED_INTERACTIONS.clear()
    out = json.loads(delegate._delegate_wait(ctx, "run-1", wait_sec=600, since_seq=5))
    delegate._CUSTODY.clear()
    assert out["status"] == "waiting_on_user"
    assert out["interactions_delivery"]["complete"] is False
    shown = out["pending_interactions"][0]
    assert shown["interaction_id"] == "R" * 200
    assert shown["questions"][0]["question_id"] == "S" * 200
    assert len(shown["questions"][0]["question"]) <= 600


def test_even_one_unfittable_row_yields_to_the_counted_marker():
    """F2: when even a single bounded row cannot fit, the rows yield entirely —
    a counted omission plus the recovery pointer, never an oversized payload."""
    payload = _expiry_payload([_giant_header_row()], advances=0, budget=700)
    raw = json.dumps(payload, ensure_ascii=False, indent=2)
    assert len(raw) <= 700 + 400, len(raw)  # marker itself is small and bounded
    assert "pending_interactions" not in payload
    assert payload["interactions_omitted"] == 1
    assert "waiting_on_user" in payload["interactions_note"] or \
        "PAUSED" in payload["interactions_note"]


def test_immediate_branch_drops_even_the_last_unfittable_row(tmp_path, monkeypatch):
    """R2-4 (delta, proven 15 210 > 15 000): the IMMEDIATE waiting_on_user
    branch's shed loop used to return the last bounded row even when it was
    still over budget after the advances yielded — the external truncator then
    severed the JSON mid-structure. The last row is now DROPPED too: what ships
    is the counted omission plus the artifact pointer, it passes the REAL
    truncator untouched, and it round-trips."""
    import ouroboros.delegate_interactions as interactions
    from ouroboros.gateways.claudexor import pending_interactions
    from ouroboros.loop_tool_execution import _truncate_tool_result
    from ouroboros.tool_capabilities import tool_result_limit

    # One max-shape row whose BOUNDED projection alone exceeds the budget:
    # three shown questions, each with a whole-riding 2 500-char question_id
    # (R2-8 keys are never cut), maxed question/header text and 12 options.
    row = _pending_row()
    row["questions"] = [{
        "id": f"q{i}-" + "K" * 2_500,
        "question": "Q" * 5_000,
        "header": "H" * 5_000,
        "options": [{"label": "L" * 400, "description": None} for _ in range(12)],
        "multi_select": False,
    } for i in range(3)]
    pending = pending_interactions({"pendingInteractions": [row]})
    ctx = _wait_ctx(tmp_path)
    raw = interactions._waiting_on_user_payload(ctx, "run-1", "running", 5, pending)
    limit = tool_result_limit("delegate_wait")
    assert len(raw) <= limit, len(raw)
    assert _truncate_tool_result(raw, "delegate_wait", {}) == raw
    out = json.loads(raw)
    assert out["status"] == "waiting_on_user"
    assert out["pending_interactions"] == []
    assert out["interactions_omitted"] == 1
    assert "read the staged artifact" in out["interactions_note"]
    # The full set is recoverable: staged whole with its receipt.
    artifact = out["interactions_delivery"]["artifact"]
    assert artifact and artifact["sha256"]
    staged = json.loads(open(artifact["abs_path"], encoding="utf-8").read())
    assert len(staged["pending_interactions"]) == 1


def test_waiting_notes_key_the_expiry_claim_on_timeout_at(tmp_path):
    """R2-7e: a null timeout_at means NO automatic expiry — every waiting note
    then says the run waits until answered instead of promising a benign
    decline that never comes; rows that DO carry timeout_at keep the
    benign-decline claim."""
    import ouroboros.delegate_interactions as interactions
    from ouroboros.gateways.claudexor import pending_interactions

    # Immediate payload, rows WITH timeout_at: the benign-decline claim stands.
    ctx = _wait_ctx(tmp_path)
    with_timeout = pending_interactions({"pendingInteractions": [_pending_row()]})
    note = json.loads(interactions._waiting_on_user_payload(
        ctx, "run-1", "running", 5, with_timeout))["note"]
    assert "benign-declines" in note

    # Immediate payload, timeout_at null: no expiry is promised.
    row = _pending_row()
    row["timeoutAt"] = ""
    without_timeout = pending_interactions({"pendingInteractions": [row]})
    note = json.loads(interactions._waiting_on_user_payload(
        ctx, "run-1", "running", 5, without_timeout))["note"]
    assert "benign-declines" not in note
    assert "waits until answered" in note

    # Both expiry-branch notes follow the same key.
    paused = _expiry_payload([_pending_row()], advances=0)
    assert "benign-declines" in paused["note"]
    no_expiry_row = _pending_row()
    no_expiry_row["timeoutAt"] = ""
    paused = _expiry_payload([no_expiry_row], advances=0)
    assert "benign-declines" not in paused["note"]
    assert "waits until answered" in paused["note"]
    paused_progress = _expiry_payload([no_expiry_row], advances=4)
    assert "benign-declines" not in paused_progress["note"]
    assert "waits until answered" in paused_progress["note"]
    with_progress = _expiry_payload([_pending_row()], advances=4)
    assert "benign-declines" in with_progress["note"]


def test_the_paused_expiry_note_never_hints_a_cancel(tmp_path):
    """F13 (owner 7=A): a run paused on its own question is not 'stuck' — the
    expiry note says answer / escalate / keep waiting, in BOTH branches, and the
    generic delegate_cancel hint appears only for a genuinely silent run."""
    paused = _expiry_payload([_pending_row()], advances=0)
    assert paused["waiting_on_user"] is True
    assert "delegate_answer" in paused["note"]
    assert "Do not cancel" in paused["note"]
    assert "delegate_cancel if it is stuck" not in paused["note"]

    silent = _expiry_payload([], advances=0)
    assert silent["waiting_on_user"] is False
    assert "delegate_cancel if it is stuck" in silent["note"]

    paused_progress = _expiry_payload([_pending_row()], advances=4)
    assert "do not cancel over it" in paused_progress["note"]

    silent_progress = _expiry_payload([], advances=4)
    assert "do not cancel over it" not in silent_progress["note"]


def test_interaction_spill_name_is_interaction_addressed_and_immutable(tmp_path, monkeypatch):
    """F15 (sol #14): a second, different pending set writes a DIFFERENT file —
    the first spill's sha256/size receipt keeps describing bytes that exist."""
    import ouroboros.tools.delegate as delegate

    rows = [_pending_row(iid=f"int-{i}", question=("Part %d? " % i) + "x" * 900)
            for i in range(24)]
    _waiting_stub(monkeypatch, rows)
    ctx = _wait_ctx(tmp_path)
    _own_run(delegate)
    first = json.loads(delegate._delegate_wait(ctx, "run-1", wait_sec=600, since_seq=5))
    first_artifact = first["interactions_delivery"]["artifact"]

    # A NEW question set (new ids) spills again — to a NEW name.
    rows2 = [_pending_row(iid=f"reask-{i}", question=("Again %d? " % i) + "y" * 900)
             for i in range(24)]
    _waiting_stub(monkeypatch, rows2)
    second = json.loads(delegate._delegate_wait(ctx, "run-1", wait_sec=600, since_seq=5))
    delegate._CUSTODY.clear()
    second_artifact = second["interactions_delivery"]["artifact"]

    assert first_artifact["path"].endswith(".interactions.json")
    assert second_artifact["path"] != first_artifact["path"]
    # BOTH files exist and BOTH receipts still verify.
    import hashlib as _hashlib
    import pathlib as _pathlib

    for artifact in (first_artifact, second_artifact):
        data = _pathlib.Path(artifact["abs_path"]).read_bytes()
        assert _hashlib.sha256(data).hexdigest() == artifact["sha256"]
        assert len(data) == artifact["bytes"]


def test_the_immediate_waiting_payload_carries_the_windows_advances(tmp_path, monkeypatch):
    """F17 (grok): a window cut short by a question must not lose the journal
    sequence it already observed — a compact `advances` list rides the immediate
    waiting_on_user return."""
    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways import claudexor as gw

    run_calls = {"n": 0}

    class _AdvancingThenAsking:
        engine_version = "3.3.6"

        def handshake(self, **_kw): return {}
        def get_run(self, rid, *, timeout_sec=None):
            run_calls["n"] += 1
            if run_calls["n"] == 1:
                return {"lastSeq": 5, "summary": {"state": "running",
                                                  "effectiveAccess": "readonly"},
                        "timeline": [{"type": "tool", "title": "step one"}]}
            return {"lastSeq": 6,
                    "pendingInteractions": [_pending_row()],
                    "summary": {"state": "running", "effectiveAccess": "readonly",
                                "waitingOnUser": True},
                    "timeline": [{"type": "tool", "title": "step one"},
                                 {"type": "tool", "title": "step two"}]}
        def close(self): pass

    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _AdvancingThenAsking())
    ctx = _wait_ctx(tmp_path)
    _own_run(delegate)
    out = json.loads(delegate._delegate_wait(ctx, "run-1", wait_sec=600, since_seq=5))
    delegate._CUSTODY.clear()
    assert out["status"] == "waiting_on_user"
    assert out["advances"], out
    assert any("step two" in json.dumps(row, ensure_ascii=False)
               for row in out["advances"])


# -- codex lane: the terminal question ------------------------------------------


def test_an_input_required_terminal_names_the_new_start_path():
    from ouroboros.subagents import DelegatedRunShape
    from ouroboros.tools.delegate import _terminal_payload

    detail = {"lastSeq": 9, "summary": {
        "state": "failed",
        "outcomeFacts": {"reason": "input_required",
                         "required_inputs": ["which database?"]},
    }}
    shape = DelegatedRunShape(access="readonly", mode="ask", isolation="", delegated=False)
    payload = _terminal_payload("run-1", detail, shape)
    assert "input_required_note" in payload
    assert "NEW delegate_start" in payload["input_required_note"]
    assert "rerun/decision" in payload["input_required_note"]


# -- the contract surfaces ------------------------------------------------------


def test_the_answer_verb_is_registered_on_every_contract_surface():
    from ouroboros.safety import TOOL_POLICY, POLICY_SKIP
    from ouroboros.tool_capabilities import (
        ACTING_SUBAGENT_TOOL_NAMES,
        LOCAL_READONLY_SUBAGENT_TOOL_NAMES,
    )
    from ouroboros.tools import delegate

    names = {entry.name for entry in delegate.get_tools()}
    assert "delegate_answer" in names
    assert "delegate_answer" in LOCAL_READONLY_SUBAGENT_TOOL_NAMES
    assert "delegate_answer" in ACTING_SUBAGENT_TOOL_NAMES
    assert TOOL_POLICY.get("delegate_answer") == POLICY_SKIP


def test_waiting_note_routes_above_authority_questions_to_escalate():
    """#204 (owner batch 3, decision 31): a harness question above the nanny's
    authority rides the escalation channel — the note names the escalate verb
    and the parent-first hierarchy, never a dead-end progress message."""
    from ouroboros.delegate_interactions import _waiting_on_user_note

    note = _waiting_on_user_note([{"interaction_id": "i1", "timeout_at": None}])
    assert "escalate(question, options, stake, assumption)" in note
    assert "PARENT task" in note
    assert "delegate_answer" in note and "delegate_wait" in note
    assert "progress message" not in note


def test_delegate_schemas_teach_the_escalation_verb():
    """sol finding: the LLM-facing tool descriptions are decision points — they
    must name the escalate verb, never the retired progress-message dead end."""
    from ouroboros.tools.delegate import get_tools

    schemas = {entry.schema["name"]: entry.schema["description"]
               for entry in get_tools()}
    for name in ("delegate_answer", "delegate_wait"):
        assert "escalate" in schemas[name]
        assert "surface it to your human via progress" not in schemas[name]
        assert "escalate to your human" not in schemas[name]


def test_expiry_notes_teach_the_escalation_verb():
    """Delta finding: the REPEAT delegate_wait rides the expiry path, whose
    notes are decision points too — both branches name the escalate verb and
    never the retired progress/your-human dead ends."""
    import inspect

    import ouroboros.delegate_progress as dp

    source = inspect.getsource(dp)
    assert "escalate an above-authority question with the" in source
    assert "raise it with the escalate verb (parent-first)" in source
    assert "escalate to your human via a progress message" not in source
    assert "escalate to your human," not in source
    # The dispatch charter is an LLM-facing user message too (agent.py appends
    # it): the same dead end must not survive there. The campaign owner of
    # dispatch_executor_note is agent_dispatch (subagent_dispatch_notes is the
    # historical import facade).
    import ouroboros.agent_dispatch as dn

    charter = inspect.getsource(dn)
    assert "escalated with the escalate verb (parent-first" in charter
    assert "goes to your human via progress" not in charter
    # No delegated-question surface may teach direct-to-human ESCALATION: the
    # hierarchy (parent-first) is the ONLY route (decision 31). Semantic
    # variants of the escalation phrasing, not just the exact retired lines
    # (the live progress STREAM legitimately mentions the human).
    import ouroboros.tools.delegate as dtool

    import ouroboros.delegate_interactions as di

    for module in (dp, dn, di, dtool):
        text = inspect.getsource(module)
        for phrase in ("escalated to its human", "escalated to your human",
                       "escalated a question to its human",
                       "goes to your human", "surface it to your human"):
            assert phrase not in text, (module.__name__, phrase)
