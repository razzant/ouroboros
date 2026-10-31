"""Saved review text through the authenticated task/record/seat door, not a path reader."""
from __future__ import annotations

import hashlib
import json
from urllib.parse import urlencode

import pytest
from starlette.applications import Starlette
from starlette.testclient import TestClient

from ouroboros import review_ledger as rl
from ouroboros.gateway.task_reviews import task_review_routes
from ouroboros.review_projection import compact_review_projection, task_ledger_records
from ouroboros.server_auth import NetworkAuthGate
from tests.test_review_record_card_projection import _gate_record, _terminal


def saved_review(root, task_id="review-reader", text="Whole answer\nTAIL", *, status="responded"):
    record = _gate_record(task_id)
    record.rows[0]["status"] = status
    source = rl.retain_text_source(root, task_id, record_id=record.record_id, seat_id="sol", role="response", part="change,coupling", text=text)
    record.rows[0]["source_refs"].append(source)
    return rl.write_record(root, record), source


def source_url(task_id, record, source, **changes):
    query = {"role": "response", "part": "change,coupling", "sha256": source["ref"]["sha256"], **changes}
    return f"/api/tasks/{task_id}/reviews/{record['record_id']}/sol?{urlencode(query)}"


def client_for(root, *, remote=False):
    app = Starlette(routes=task_review_routes())
    app.state.drive_root = root
    return TestClient(NetworkAuthGate(app), client=("10.11.12.13" if remote else "127.0.0.1", 1234))


def test_full_answer_tail_and_download_share_verified_bytes(tmp_path):
    body = "# Saved opinion\n" + "A real long finding.\n" * 1200 + "TAIL-935-LONG-END"
    record, source = saved_review(tmp_path, text=body)
    with client_for(tmp_path) as client:
        response = client.get(source_url("review-reader", record, source))
        assert response.status_code == 200
        assert response.content == body.encode()
        assert response.text.endswith("TAIL-935-LONG-END")
        assert response.headers["x-review-source-sha256"] == source["ref"]["sha256"]
        assert response.headers["x-review-export-sha256"] == hashlib.sha256(response.content).hexdigest()
        assert response.headers["content-disposition"].startswith("attachment;")
        assert response.headers["cache-control"] == "no-store"
        assert client.get(source_url("foreign-task", record, source)).status_code == 404
        assert client.get(source_url("review-reader", record, source, sha256="a" * 64)).status_code == 409
        assert client.get(source_url("review-reader", record, source, role="prompt")).status_code == 400
        assert client.get(source_url("review-reader", record, source).replace("/sol?", "/another-seat?")).status_code == 404


def test_export_redacts_without_claiming_original_sha(tmp_path):
    secret = "sk-ant-api03-" + "A1b2C3d4E5f6G7h8" * 4
    record, source = saved_review(tmp_path, text="provider refused " + secret + " TAIL")
    with client_for(tmp_path) as client:
        response = client.get(source_url("review-reader", record, source))
    assert response.status_code == 200 and secret not in response.text
    assert response.text.endswith(" TAIL")
    assert response.headers["x-review-export-sha256"] != response.headers["x-review-source-sha256"]


def test_network_auth_is_not_bypassed_by_review_source_or_history(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.server_auth.get_configured_network_password", lambda: "test-password")
    record, source = saved_review(tmp_path)
    with client_for(tmp_path, remote=True) as client:
        assert client.get(source_url("review-reader", record, source)).status_code == 401
        assert client.get("/api/tasks/review-reader/reviews").status_code == 401
        assert client.get("/api/tasks/review-reader/reviews/skill/job?skill=weather").status_code == 401
        login = client.post("/auth/login", data={"password": "test-password", "next": "/"}, follow_redirects=False)
        assert login.status_code == 303
        assert client.get(source_url("review-reader", record, source)).status_code == 200


@pytest.mark.parametrize("broken", ["json", "projection", "missing"])
def test_broken_record_does_not_hide_healthy_neighbor_or_terminal_carriers(tmp_path, broken):
    healthy, _source = saved_review(tmp_path, "neighbors")
    bad, _source = saved_review(tmp_path, "neighbors")
    path = rl.record_path(tmp_path, bad["record_id"])
    if broken == "json":
        path.write_text("{broken", encoding="utf-8")
    elif broken == "projection":
        value = json.loads(path.read_text())
        value["verdict"]["quorum"]["required"] = "not a count"
        path.write_text(json.dumps(value), encoding="utf-8")
    else:
        path.unlink()
    stored, event, history = _terminal(tmp_path, "neighbors")
    projection = stored["review_projection"]
    assert projection == event == history["review_projection"]
    assert [row["record_id"] for row in projection["panels"]] == [healthy["record_id"]]
    assert projection["gaps"][0]["record_id"] == bad["record_id"]
    assert projection["gaps"][0]["review_gap"] == {"json": "unreadable", "projection": "projection_unreadable", "missing": "missing"}[broken]
    with pytest.raises(ValueError) if broken == "json" else __import__("contextlib").nullcontext():
        rl.load_record(tmp_path, bad["record_id"])  # Strict reader contract is unchanged.


def test_only_gap_is_retained_and_plan_neighbor_survives_projection(tmp_path):
    bad, _source = saved_review(tmp_path, "only-gap")
    rl.record_path(tmp_path, bad["record_id"]).unlink()
    stored, event, history = _terminal(tmp_path, "only-gap")
    assert not stored["review_projection"]["panels"]
    assert stored["review_projection"]["gaps"] == event["gaps"] == history["review_projection"]["gaps"]
    records, _ = task_ledger_records({"id": "only-gap"}, tmp_path)
    plan = {"surface": "plan", "panel_id": "healthy-plan", "actors": [], "aggregate_signal": "UNKNOWN"}
    projection = compact_review_projection([plan], records)
    assert projection["panels"][0]["panel_id"] == "healthy-plan"
    assert projection["gaps"]


def test_explicit_history_reads_older_records_beyond_five_and_reports_local_gap(tmp_path):
    records = [saved_review(tmp_path, "history")[0] for _ in range(9)]
    rl.record_path(tmp_path, records[1]["record_id"]).unlink()
    with client_for(tmp_path) as client:
        first = client.get("/api/tasks/history/reviews?limit=4").json()
        second = client.get(f"/api/tasks/history/reviews?limit=5&offset={first['next_offset']}").json()
    assert first["next_offset"] == 4 and second["next_offset"] is None
    ids = [row["record_id"] for response in (first, second) for row in response["review_projection"]["panels"]]
    gaps = [row["record_id"] for response in (first, second) for row in response["review_projection"].get("gaps", [])]
    assert set(ids + gaps) == {row["record_id"] for row in records}
    assert gaps == [records[1]["record_id"]]


def test_changed_saved_source_fails_integrity_without_reading_arbitrary_path(tmp_path):
    record, source = saved_review(tmp_path)
    # Resolve only the legitimate source using the same artifact root binding.
    from ouroboros.artifacts import task_artifact_dir_path
    path = task_artifact_dir_path(tmp_path, "review-reader", create=False) / source["ref"]["path"]
    assert path.is_file()
    path.write_text("changed", encoding="utf-8")
    with client_for(tmp_path) as client:
        response = client.get(source_url("review-reader", record, source) + "&path=/etc/passwd")
    assert response.status_code == 404
    assert response.json()["reason_code"] == "review_source_unavailable"


def test_plan_reply_ref_uses_its_own_recorded_contract(tmp_path):
    from ouroboros.task_results import write_task_result
    from tests.test_plan_record import _record

    task_id = "plan-reader"
    source = rl.retain_text_source(tmp_path, task_id, record_id="ask-original", seat_id="participant", role="response", part="intention", text="Full plan contribution\nPLAN-TAIL")
    record = _record([{"seat_id": "participant", "status": "answered", "reply_ref": source}])
    record["asks"][0]["ask_id"] = "ask-original"
    write_task_result(tmp_path, task_id, "completed", plan_record=record)
    query = urlencode({"domain": "plan", "sha256": source["ref"]["sha256"]})
    with client_for(tmp_path) as client:
        response = client.get(f"/api/tasks/{task_id}/reviews/ask-original/participant?{query}")
        assert response.status_code == 200 and response.text.endswith("PLAN-TAIL")
        assert client.get(f"/api/tasks/{task_id}/reviews/newer-ask/participant?{query}").status_code == 409
        assert client.get(f"/api/tasks/{task_id}/reviews/ask-original/participant?{query}&role=prompt").status_code == 400
        # The same immutable physical selection must survive a retry alias.
        write_task_result(tmp_path, "plan-successor", "completed", plan_record=_record())
        write_task_result(tmp_path, task_id, "completed", plan_record=record, retry_task_id="plan-successor")
        assert client.get(f"/api/tasks/{task_id}/reviews/ask-original/participant?{query}").content == response.content
        assert client.get(f"/api/tasks/plan-successor/reviews/ask-original/participant?{query}").status_code == 409
        assert client.get(f"/api/tasks/{task_id}/reviews/ask-original/participant?{query}&role=failure").status_code == 409
        assert client.get(f"/api/tasks/{task_id}/reviews/ask-original/participant?{query}&part=foreign").status_code == 409


def test_archive_uncertainty_does_not_create_reviews_on_an_unreviewed_task(tmp_path, monkeypatch):
    archived, _ = saved_review(tmp_path, "actually-reviewed")
    rl.index_path(tmp_path).rename(rl.ledger_dir(tmp_path) / "index.20261020T000000.jsonl")
    real_iter, opened = rl.iter_jsonl_objects, []
    def read(path):
        opened.append(path.name)
        return real_iter(path)
    monkeypatch.setattr(rl, "iter_jsonl_objects", read)
    stored, event, history = _terminal(tmp_path, "never-reviewed")
    assert not stored.get("review_projection") and not event and not history.get("review_projection")
    assert not any(name != "index.jsonl" for name in opened), "hot finalization must not read archives"
    with client_for(tmp_path) as client:
        detail = client.get("/api/tasks/actually-reviewed/reviews").json()
    assert [p["record_id"] for p in detail["review_projection"]["panels"]] == [archived["record_id"]]
    assert detail["next_offset"] is None


def test_history_page_stops_after_exact_task_rows_and_one_lookahead(tmp_path, monkeypatch):
    # An unrelated old segment must remain unopened once this page is complete.
    saved_review(tmp_path, "old-unrelated")
    archive = rl.ledger_dir(tmp_path) / "index.20261020T000000.jsonl"
    rl.index_path(tmp_path).rename(archive)
    records = [saved_review(tmp_path, "page-owner")[0] for _ in range(4)]
    child = _gate_record("child", root_task_id="page-owner")
    rl.write_record(tmp_path, child)
    real_iter = rl.iter_jsonl_objects
    def read(path):
        assert path != archive, "page scanned history beyond the required lookahead"
        return real_iter(path)
    monkeypatch.setattr(rl, "iter_jsonl_objects", read)
    with client_for(tmp_path) as client:
        response = client.get("/api/tasks/page-owner/reviews?offset=1&limit=2")
    assert response.status_code == 200
    payload = response.json()
    assert payload["next_offset"] == 3
    assert [p["record_id"] for p in payload["review_projection"]["panels"]] == [r["record_id"] for r in records[1:3]]


@pytest.mark.parametrize("storage", ["legacy", "ledger"])
@pytest.mark.parametrize("owner_field", ["task_id", "origin_task_id", "presentation_owner_task_id", "group_id"])
def test_task_bound_saved_skill_export_matches_detail_and_bridge_prefix(tmp_path, storage, owner_field):
    import ast
    import pathlib
    import urllib.parse

    from ouroboros.gateway.extensions import _skill_review_history_detail_sync
    from ouroboros.skill_review_history import review_history_path
    from tests.test_review_ledger import _facts, _raw

    for job, owner in [("exact-job", "skill-owner"), ("another-job", "foreign-task")]:
        lifecycle = {"job_id": job, owner_field: f"task:{owner}:weather" if owner_field == "group_id" else owner,
                     "status": "clean", "job_status": "completed", "content_hash": "a" * 64}
        if storage == "legacy":
            path = review_history_path(tmp_path, "weather")
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as stream:
                stream.write(json.dumps(lifecycle) + "\n")
        else:
            record = rl.build_wave_record(_facts([_raw("s", "m", parts=("skill",))], task_id="physical-task"),
                                          surface="skill", drive_root=tmp_path)
            record.subject.update(kind="skill", skill="weather")
            record.brief["skill"] = lifecycle
            rl.write_record(tmp_path, record)
    url = "/api/tasks/skill-owner/reviews/skill/exact-job?skill=weather&format=text"
    # Execute the existing validator body, without starting or changing the app.
    launcher = pathlib.Path(__file__).resolve().parents[1] / "launcher.py"
    function = next(n for n in ast.walk(ast.parse(launcher.read_text()))
                    if isinstance(n, ast.FunctionDef) and n.name == "_resolve_bridge_file_url")
    namespace = {"actual_port": 8000, "urllib": urllib}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(launcher), "exec"), namespace)
    assert namespace["_resolve_bridge_file_url"](url) == "http://127.0.0.1:8000" + url
    with pytest.raises(ValueError):
        namespace["_resolve_bridge_file_url"]("/api/skills/weather/review-history/exact-job?format=text")
    with client_for(tmp_path) as client:
        response = client.get(url)
        assert response.status_code == 200
        assert response.text == _skill_review_history_detail_sync(tmp_path, "weather", "exact-job")["markdown"]
        assert response.headers["content-disposition"].startswith("attachment;")
        assert client.get(url.replace("skill-owner", "foreign-task")).status_code == 404
        assert client.get(url.replace("weather", "another-skill")).status_code == 404
        assert client.get(url.replace("exact-job", "another-job")).status_code == 404
        swapped = url.replace("exact-job", "another-job").replace("skill-owner", "foreign-task")
        assert client.get(swapped).status_code == 200  # The refused job exists, under its own task.


def test_history_contract_publishes_the_actual_response(tmp_path):
    from pathlib import Path
    from typing import get_type_hints

    from ouroboros.gateway import contracts
    from ouroboros.gateway.contracts import TaskReviewHistoryResponse
    from ouroboros.gateway.schema import contract_document, validate_ingress
    from tests.test_gateway_parity import _js_typedef_fields

    saved_review(tmp_path)
    with client_for(tmp_path) as client:
        response = client.get("/api/tasks/review-reader/reviews").json()
    assert validate_ingress(response, TaskReviewHistoryResponse) == []
    assert "TaskReviewHistoryResponse" in contract_document()["schemas"]
    assert "TaskReviewHistoryResponse" in contracts.__all__
    js = (Path(__file__).resolve().parents[1] / "web/modules/api_types.js").read_text()
    assert _js_typedef_fields(js, "TaskReviewHistoryResponse") == set(get_type_hints(TaskReviewHistoryResponse))


def test_saved_skill_download_through_real_launcher_and_http(tmp_path, monkeypatch):
    import pathlib
    import socket
    import threading
    import time

    import uvicorn

    from ouroboros.skill_review_history import review_history_detail, review_history_path
    from tests.test_ui_smoke_large_artifacts import _native_file_api

    path = review_history_path(tmp_path, "weather")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"job_id": "job", "task_id": "owner", "status": "clean",
                               "job_status": "completed", "content_hash": "b" * 64}) + "\n")
    app = Starlette(routes=task_review_routes())
    app.state.drive_root = tmp_path
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(NetworkAuthGate(app), log_level="warning"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(.01)
        assert server.started
        monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: tmp_path / "download-home"))
        sizes = []
        api, _ = _native_file_api(sock.getsockname()[1], sizes)
        old = api.download_file_to_downloads("/api/skills/weather/review-history/job?format=text", "old.txt")
        assert old["ok"] is False and "/api/tasks/" in old["error"]
        result = api.download_file_to_downloads("/api/tasks/owner/reviews/skill/job?skill=weather&format=text", "review.txt")
        assert result["ok"] is True
        # The export is UTF-8 bytes and the launcher copies them as they came: compare bytes, never a locale decode.
        expected = review_history_detail(tmp_path, "weather", "job")["markdown"].encode("utf-8")
        assert pathlib.Path(result["path"]).read_bytes() == expected
        assert sizes and all(0 < size <= 1024 * 1024 for size in sizes)
        foreign = api.download_file_to_downloads("/api/tasks/foreign/reviews/skill/job?skill=weather&format=text", "foreign.txt")
        assert foreign["ok"] is False and "404" in foreign["error"]
    finally:
        server.should_exit = True
        thread.join(10)
        sock.close()
        assert not thread.is_alive()


def test_saved_skill_export_is_the_same_normalized_text_as_the_json_detail(tmp_path):
    from starlette.routing import Route
    from ouroboros.gateway.extensions import api_skill_review_history_detail
    from ouroboros.skill_review_history import review_history_path

    # This is the existing saved-job contract, not a second reviewer dispatch.
    path = review_history_path(tmp_path, "weather")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"job_id": "exact-job", "task_id": "owner", "status": "clean", "content_hash": "a" * 64,
                    "job_status": "completed", "raw_actor_records": [{"model_id": "model", "status": "responded", "parsed_items": []}]}) + "\n", encoding="utf-8")
    app = Starlette(routes=[Route("/api/skills/{skill}/review-history/{job_id}", api_skill_review_history_detail),
                            *task_review_routes()])
    app.state.drive_root = tmp_path
    with TestClient(app) as client:
        original = client.get("/api/skills/weather/review-history/exact-job")
        exported = client.get("/api/tasks/owner/reviews/skill/exact-job?skill=weather")
        # The Chat card's JSON detail keeps its one contract; the reader's export is task-bound.
        assert client.get("/api/skills/weather/review-history/exact-job?format=text").json() == original.json()
    assert original.status_code == exported.status_code == 200
    assert exported.text == original.json()["markdown"]
    assert "clean" in exported.text.lower()
    assert exported.headers["cache-control"] == "no-store"


def test_full_safe_refusal_survives_substrate_parser_and_ledger(tmp_path):
    import queue
    from types import SimpleNamespace
    from ouroboros.gateways.claudexor import run_failure_error
    from ouroboros.review_substrate import ReviewCoordinator, ReviewRequest, ReviewSlot
    from ouroboros.tools.review_multi_model import _actor_query_result
    from ouroboros.tools.review import _parse_model_response
    from ouroboros.triad_review import parse_seat_answers
    from ouroboros.review_records import review_failure_summary
    from ouroboros.review_custody import _frozen_actor
    from tests.test_review_ledger import _facts

    tail = "TAIL: measured 940000 characters; limit 900000 characters"
    failure = {"safeMessage": "A saved engine refusal. " * 300 + tail, "code": "native_transcript_cap_exceeded",
               "measured": 940000, "limit": 900000, "unit": "characters"}
    class Refused:
        def chat(self, **_kwargs):
            raise run_failure_error("run-offline", "failed", failure)

    context = SimpleNamespace(drive_root=tmp_path, task_id="failure-reader", task_metadata={}, event_queue=queue.Queue(), pending_events=[])
    slot = ReviewSlot("failed-seat", "requested/model", native_retrieval_override=False)
    request = ReviewRequest(surface="multi_model_review", task_id="failure-reader", goal="offline refusal", policy={"parts": ["change", "coupling"]})
    run = ReviewCoordinator(llm=Refused(), drive_root=tmp_path, usage_ctx=context).run(request, [slot])
    actor = run.actors[0]
    assert tail in actor["error"] and actor["failure_details"]["limit"] == 900000
    from ouroboros.tools.plan_review_runtime import _plan_row_from_actor
    plan_row = _plan_row_from_actor(actor, slot)
    assert plan_row["failure_details"] == actor["failure_details"]
    model, payload, _ = _actor_query_result(slot.model, actor, slot.slot_id)
    parsed = parse_seat_answers({"results": [_parse_model_response(model, payload, None)]}, {})
    row = parsed.actor_records[0].to_dict()
    assert _frozen_actor(row, slot).failure_details["unit"] == "characters"
    assert tail in review_failure_summary(row)
    record = rl.build_wave_record(_facts([row], task_id="failure-reader"), surface="change", drive_root=tmp_path)
    stored = rl.write_record(tmp_path, record)
    refs = [source for source in stored["rows"][0]["source_refs"] if source["role"] == "failure"]
    assert refs and tail.encode() in rl.read_source(tmp_path, "failure-reader", refs[0])
