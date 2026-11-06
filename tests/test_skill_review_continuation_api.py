"""The Skills page continues its own stopped reviewer through ``POST /api/skills/{skill}/review``.

The real door, lifecycle job, review wave, ledger, custody and collector; only the engine is scripted
(the session's first run stops with continuable work, every later run answers clean). The ordinary call
with no body choice stays a new review. ``SkillReviewRequest`` names the page's own record, its reviewed
bytes and the seats its card row offered: the continuation runs as a lifecycle job, sends ONE successor
from the original start body, and projects the verdict onto those bytes. A malformed, partial or
unknown body is refused before any job; a foreign record is neither offered nor continued; a stale
choice is refused at $0 inside the job.
"""
from __future__ import annotations

import json


from ouroboros import review_ledger
from ouroboros.skill_loader import load_review_state
from tests.review_pool_rosters import pool_seat
from tests.test_skill_review_usage_events import server_bus as _server_bus, skill_door as _skill_door

server_bus, skill_door = _server_bus, _skill_door  # fixtures by name
URL = "/api/skills/weather/review"
OFFER_LEAD = "Continuable stopped seats (skill_review(skill, review_reference=that reference, continue_seats=[{seat_id, operation_id}])): "


def _engine(tmp_path, monkeypatch):
    from ouroboros import delegate_custody as custody
    from tests._review_session_route_shared import FakeGateway, _terminal_detail

    posts, runs = [], {}

    class Engine(FakeGateway):
        def start_run(self, request, *, idempotency_key=""):
            super().start_run(request, idempotency_key=idempotency_key)
            posts.append((dict(request), str(idempotency_key)))
            return {"runId": runs.setdefault(str(idempotency_key), f"run-{len(runs) + 1}"), "runDir": str(FakeGateway.run_dir)}

        def get_run(self, run_id, **_kw):
            self.run_gets.append(run_id)
            if run_id != "run-1":
                return _terminal_detail('{"findings": []}', conformance="passed")
            stopped = _terminal_detail("PARTIAL: two of five files read", state="failed")
            stopped["summary"].update(failure={"code": "credential_pool_exhausted", "safeMessage": "Account limit reached"},
                                      resumable={"cause": "pool_exhausted", "carriers": ["native"]})
            return stopped

    FakeGateway.reset()
    monkeypatch.setattr(FakeGateway, "run_dir", tmp_path / "review-run")
    monkeypatch.setattr("ouroboros.gateways.claudexor.ClaudexorGateway", Engine)
    monkeypatch.setattr("ouroboros.claudexor_daemon.ensure_owned_gateway", lambda: Engine())
    monkeypatch.setattr("ouroboros.claudexor_daemon.read_owned_gateway", lambda: Engine())
    custody._CUSTODY.clear()
    return posts


def _card_row(door, record_id):
    skills = door.client.get("/api/extensions").json()["skills"]
    history = next(skill for skill in skills if skill["name"] == "weather")["skill_review"]["history"]
    return next(row for row in history if row.get("record_id") == record_id)


def _set_actor(door, record_id, task_id):
    review_ledger.revise_record(door.drive, record_id, lambda record: {**record, "task_id": task_id})


def test_the_skills_page_continues_its_own_stopped_seat_on_the_reviewed_bytes_and_nothing_else(
        skill_door, monkeypatch, tmp_path):
    door = skill_door
    door.pool(pool_seat("review-s", "fake-review=fake-small", kind="agent_session", access="readonly"), api=False)
    posts = _engine(tmp_path, monkeypatch)
    for raw in (b"{not json", b"[]", b'{"continue_seat": []}', b'{"review_reference": {"record_id": "x"}}',
                b'{"review_reference": {"record_id": "x"}, "continue_seats": [{"seat_id": "s"}]}'):
        refused = door.client.post(URL, content=raw, headers={"Content-Type": "application/json"})
        assert refused.status_code == 400, (raw, refused.json())
    assert not posts and not review_ledger.recent_records(door.drive, "api_skill_review", limit=5), "no job, no send"

    first = door.client.post(URL, json={}).json()  # the ordinary call: a new review of the current bytes
    record_id, content_hash = first["record_id"], first["content_hash"]
    assert first["status"] != "clean" and len(posts) == 1, first
    offer = _card_row(door, record_id)["continuable_seats"]
    assert [seat["seat_id"] for seat in offer] == ["review-s"] and offer[0]["run_id"] == "run-1", offer
    choice = {"review_reference": {"record_id": record_id, "content_hash": content_hash},
              "continue_seats": [{"seat_id": "review-s", "operation_id": offer[0]["operation_id"]}]}

    _set_actor(door, record_id, "some-task")  # the same record as a task's own review: not the page's to continue
    assert "continuable_seats" not in _card_row(door, record_id)
    foreign = door.client.post(URL, json=choice).json()
    assert foreign["status"] == "pending" and "this author's review record" in foreign["error"], foreign
    _set_actor(door, record_id, "api_skill_review")
    stale = door.client.post(URL, json={**choice, "continue_seats": [{"seat_id": "review-s", "operation_id": "op-gone"}]}).json()
    assert stale["status"] == "pending" and "not_current" in stale["error"] and len(posts) == 1, stale

    continued = door.client.post(URL, json=choice).json()
    assert continued["status"] == "clean" and continued["record_id"] == record_id, continued
    (original, _key0), (successor, key) = posts
    assert successor["continueFrom"] == "run-1" and {k for k in set(original) | set(successor)
                                                      if original.get(k) != successor.get(k)} <= {"prompt", "maxSeconds", "continueFrom"}
    seat = review_ledger.load_record(door.drive, record_id)["rows"][0]
    assert seat["status"] == "responded" and seat["continuations"][0]["invocation_id"] == key
    review = load_review_state(door.drive, "weather")
    assert (review.status, review.record_id, review.content_hash) == ("clean", record_id, content_hash)
    assert "continuable_seats" not in _card_row(door, record_id)  # answered: nothing left to continue
    again = door.client.post(URL, json=choice).json()  # the same choice rejoins its answered successor at $0
    assert again["status"] == "clean" and len(posts) == 2, again
    assert json.loads(json.dumps(continued))["record_id"] == again["record_id"]


def test_the_skill_review_tool_answer_alone_lets_its_author_continue_the_stopped_seat(skill_door, monkeypatch, tmp_path):
    """The task-owned path (the Skills page lists only its own reviews): the author reads the sole offered tuple and the
    reference from the tool's own text, sends them back through the same tool, and gets one successor in the same wave."""
    import server as srv

    from ouroboros.tools import skill_exec
    from ouroboros.tools.registry import ToolContext
    from tests.test_skill_review_runner import _reset_queue

    door = skill_door
    door.pool(pool_seat("review-s", "fake-review=fake-small", kind="agent_session", access="readonly"), api=False)
    posts = _engine(tmp_path, monkeypatch)
    system = srv.app.app.state.repo_dir
    ctx = ToolContext(repo_dir=system, system_repo_dir=system, drive_root=door.drive, task_id="skill-author-task")
    _reset_queue()
    try:
        first = skill_exec._handle_review_skill(ctx, skill="weather")
        assert len(posts) == 1 and OFFER_LEAD in first, first
        offers = json.loads(first.split(OFFER_LEAD, 1)[1].split("\n", 1)[0])
        reference = json.loads(first.split("Review reference: ", 1)[1].split("\n", 1)[0])
        assert [offer["seat_id"] for offer in offers] == ["review-s"] and offers[0]["run_id"] == "run-1", offers
        choice = [{key: offers[0][key] for key in ("seat_id", "operation_id")}]
        records = review_ledger.recent_records(door.drive, "skill-author-task", limit=5)

        continued = skill_exec._handle_review_skill(ctx, skill="weather", review_reference=reference, continue_seats=choice)
        assert OFFER_LEAD not in continued, continued  # answered: nothing left to continue
        (original, _key0), (successor, key) = posts
        assert successor["continueFrom"] == "run-1" and {k for k in set(original) | set(successor)
                                                         if original.get(k) != successor.get(k)} <= {"prompt", "maxSeconds", "continueFrom"}
        assert json.loads(continued.split("Review reference: ", 1)[1].split("\n", 1)[0])["record_id"] == reference["record_id"]
        seat = review_ledger.load_record(door.drive, reference["record_id"])["rows"][0]
        assert seat["status"] == "responded" and seat["continuations"][0]["invocation_id"] == key
        assert load_review_state(door.drive, "weather").status == "clean"
        assert [row["record_id"] for row in review_ledger.recent_records(door.drive, "skill-author-task", limit=5)] == [
            row["record_id"] for row in records]  # the same wave: no fresh record
        skill_exec._handle_review_skill(ctx, skill="weather", review_reference=reference, continue_seats=choice)
        assert len(posts) == 2  # the repeat rejoins its answered successor at $0
    finally:
        _reset_queue()


def test_a_skill_continuation_waiting_on_its_reviewer_lends_the_lifecycle_lane(skill_door, monkeypatch, tmp_path):
    """As the ordinary skill wave does, a continuation lends the global lifecycle lane while its reviewer works: an
    unrelated lifecycle job runs while the continued reviewer is still held, and the projection follows only after
    the lane is taken back (the reviewed bytes and the job that owns the review are checked there, as before)."""
    import threading
    import time

    import server as srv

    from ouroboros import skill_lifecycle_queue
    from ouroboros.gateways import claudexor as gateway_module
    from ouroboros.tools import skill_exec
    from ouroboros.tools.registry import ToolContext
    from tests.test_skill_review_runner import _reset_queue

    door = skill_door
    door.pool(pool_seat("review-s", "fake-review=fake-small", kind="agent_session", access="readonly"), api=False)
    posts = _engine(tmp_path, monkeypatch)
    release, engine = threading.Event(), gateway_module.ClaudexorGateway

    class Held(engine):
        def get_run(self, run_id, **kw):
            if run_id != "run-1" and not release.is_set():  # the continued reviewer is still working
                return {"summary": {"state": "running"}, "lastSeq": 1}
            return super().get_run(run_id, **kw)
    monkeypatch.setattr("ouroboros.gateways.claudexor.ClaudexorGateway", Held)
    monkeypatch.setattr("ouroboros.claudexor_daemon.ensure_owned_gateway", lambda: Held())
    monkeypatch.setattr("ouroboros.claudexor_daemon.read_owned_gateway", lambda: Held())
    system = srv.app.app.state.repo_dir
    ctx = ToolContext(repo_dir=system, system_repo_dir=system, drive_root=door.drive, task_id="skill-lane-task")
    _reset_queue()
    continuing, out, unrelated = None, {}, {}
    try:
        first = skill_exec._handle_review_skill(ctx, skill="weather")
        offer = json.loads(first.split(OFFER_LEAD, 1)[1].split("\n", 1)[0])[0]
        reference = json.loads(first.split("Review reference: ", 1)[1].split("\n", 1)[0])
        continuing = threading.Thread(target=lambda: out.update(text=skill_exec._handle_review_skill(
            ctx, skill="weather", review_reference=reference,
            continue_seats=[{key: offer[key] for key in ("seat_id", "operation_id")}])), daemon=True)
        continuing.start()
        deadline = time.monotonic() + 30
        while len(posts) < 2 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert len(posts) == 2, "the continued reviewer never started"
        other = threading.Thread(target=lambda: unrelated.update(result=skill_lifecycle_queue.run_lifecycle_job_blocking(
            kind="toggle", target="an-unrelated-skill", runner=lambda: "ran")), daemon=True)
        other.start()
        other.join(15)
        assert unrelated.get("result") == "ran" and continuing.is_alive(), (unrelated, out)
    finally:
        release.set()
        if continuing is not None:
            continuing.join(60)
        _reset_queue()
    assert "Review reference" in out.get("text", "") and load_review_state(door.drive, "weather").status == "clean", out
