"""A task service's end, one fact from its stop to the answer and the card.

An unconfirmed stop stays an error in teardown, the author's disclosure, the delivery evidence and UI events; a
confirmed one stays recorded as ended when its log could not be read, with its outputs registered and the gap said.
"""
from __future__ import annotations

import json
import pathlib
import queue
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ouroboros import loop_forced_finalization, service_finalization
from ouroboros.tools.registry import ToolContext
from tests._launch_stop_shared import _author_evidence, _exit_context, _refuse_metadata_of


@pytest.mark.parametrize("live", [False, True])
def test_pending_stop_reaches_evidence_and_checkpoint_then_success_is_fresh(tmp_path, monkeypatch, live):
    from ouroboros.loop_delivery import delivery_evidence_fingerprint
    from ouroboros.tools import services

    tool_ctx = ToolContext(repo_dir=tmp_path / "repo", drive_root=tmp_path, task_id="task")
    ctx = SimpleNamespace(drive_root=tmp_path, task_id="task", tools=SimpleNamespace(_ctx=tool_ctx),
                          drive_logs=tmp_path / "logs", event_queue=queue.Queue() if live else None, llm_trace={})
    owed = [{"pid": 41000 + index, "birth": "Mon Nov 2 12:00:00 2026", "reason": "signal_denied"} for index in range(25)]
    pending = {"service_id": "task:svc", "name": "svc", "task_id": "task",
               "lifecycle": "cleanup_pending", "state": "exited", "stop_failed": True,
               "cleanup_dispatched": False, "stop_error": "unreadable /proc " + "x" * 2200,
               "process_cleanup": {"exit_confirmed": False, "census_failed": False, "survivors": [],
                                   "unverified": owed, "ended": [40999], "reach": ["group"]}}
    stopped = {"service_id": "task:svc", "name": "svc", "task_id": "task",
               "lifecycle": "stopped", "state": "stopped", "cleanup_dispatched": True}
    kept = {"service_id": "task:kept", "name": "kept", "lifecycle": "kept", "state": "running"}
    results = iter([[pending, kept], [dict(pending, uptime_sec=100), kept], [stopped]])
    monkeypatch.setattr(services, "stop_task_services", lambda _ctx: next(results))

    assert service_finalization.finalize(ctx, ctx.llm_trace) is True
    evidence = service_finalization.evidence(ctx.llm_trace)
    assert [event["kind"] for event in evidence] == ["service_finalization_error", "services_kept"]
    [row] = evidence[0]["services"]
    assert row["lifecycle"] == "cleanup_pending" and row["state"] == "exited"
    assert row["stop_failed"] is True and row["cleanup_dispatched"] is False
    assert len(row["stop_error"]) == 2000
    assert "OMISSION NOTE" in row["stop_error"]
    assert f"original length {len(pending['stop_error'])}" in row["stop_error"]
    assert row["process_cleanup"]["unverified"] == owed, "every owed identity, past the first twenty, by pid and birth"
    assert row["process_cleanup"]["exit_confirmed"] is False and "ended" not in row["process_cleanup"]
    before = delivery_evidence_fingerprint(tool_ctx, ctx.llm_trace)
    assert service_finalization.finalize(ctx, ctx.llm_trace) is False
    assert delivery_evidence_fingerprint(tool_ctx, ctx.llm_trace) == before
    assert service_finalization.finalize(ctx, ctx.llm_trace) is True
    assert delivery_evidence_fingerprint(tool_ctx, ctx.llm_trace) != before
    evidence = service_finalization.evidence(ctx.llm_trace)
    assert [event["kind"] for event in evidence] == ["service_finalization_error", "services_kept", "services_stopped"]
    assert evidence[-1]["services"][0]["cleanup_dispatched"] is True
    if live:
        events = []
        while not ctx.event_queue.empty():
            events.append(ctx.event_queue.get_nowait()["data"])
    else:
        events = [json.loads(line) for line in (ctx.drive_logs / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [event["checkpoint_kind"] for event in events] == [event["kind"] for event in evidence]
    assert [event["text"] for event in events] == [
        "Background service cleanup unconfirmed", "Background services kept", "Background services stopped"]
    assert all(event["system_type"] == "task_checkpoint" and "services" not in event for event in events), events


def test_one_lifecycle_reading_of_an_executor_stops_own_answer_and_no_service_dropped(tmp_path, monkeypatch):
    """The task end reads each stop's own answer once: a label the payload carries never outranks its facts, and a
    stop that raised is reported, as ended only when that launch's end was confirmed before it raised."""
    from ouroboros import workspace_executor
    from ouroboros.tools import services

    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path / "data", task_id="task")
    record = SimpleNamespace(service_id="task:svc", name="svc", task_id="task", keep_alive=False, finalizing=False)
    monkeypatch.setattr(services, "_SERVICES", {})
    monkeypatch.setattr(workspace_executor, "_services_snapshot", lambda: [record])
    monkeypatch.setattr(services, "executor_stop_service", lambda _ctx, _name, _finish=None, *, record: {
        "service_id": "task:svc", "lifecycle": "stopped", "state": "exited",
        "stop_failed": True, "cleanup_dispatched": False, "stop_error": "surviving members: 43"})
    [pending] = services.stop_task_services(ctx)
    assert pending["lifecycle"] == "cleanup_pending"
    assert pending["state"] == "exited" and pending["stop_error"] == "surviving members: 43"

    def raised(*_args, **_kwargs):
        raise OSError("census unreadable")

    monkeypatch.setattr(services, "executor_stop_service", raised)
    [unread] = services.stop_task_services(ctx)
    assert unread == {"service_id": "task:svc", "name": "svc", "task_id": "task", "stop_source": "task_end",
                      "lifecycle": "cleanup_pending", "stop_failed": True, "stop_error": "OSError: census unreadable",
                      "log_outcome": "not_attempted", "output_outcomes": []}
    record.finalizing = True
    record.outputs = ["declared.txt"]
    [ended] = services.stop_task_services(ctx)
    assert ended["lifecycle"] == "stopped" and ended["finalization_error"] == "OSError"
    assert ended["artifact_output_failed"] and ended["artifact_output_unknown"]
    assert "stop_failed" not in ended, "a confirmed end is not reported unconfirmed"


def _ordinary_completion(tmp_path, monkeypatch, rows, answer, *, previous=()):
    """Run the real ordinary rail: teardown holds the first answer before its result review
    and shows the author its evidence; the author selects again and that answer is reviewed at
    delivery and delivered. Returns the author's view and its own registry."""
    import ouroboros.loop as loop
    from ouroboros import service_outcomes
    from ouroboros.tools import services
    from tests.test_delivery_forced_finalization import _forced_test_context, _select_completion

    _loop, registry, ctx, trace = _forced_test_context(tmp_path)
    registry._ctx.is_direct_chat = False
    registry._ctx._execution_trace = trace
    for payload in previous:
        service_outcomes.observe_stop(registry._ctx, None, lambda: (payload, False), source="replacement")
    results = iter([rows])
    monkeypatch.setattr(services, "stop_task_services", lambda _ctx: next(results, []))
    reviewed = []
    monkeypatch.setattr("ouroboros.tools.review_result.result_review_at_delivery",
                        lambda _ctx, content, _trace, _hold: reviewed.append(content) or False)
    monkeypatch.setattr(loop, "_compute_subagent_handoff", lambda *_a, **_k: None)
    monkeypatch.setattr(loop, "_maybe_inject_finalization_nudges", lambda *_a, **_k: False)

    held = loop._no_tool_final_answer("The service is stopped.", ctx, trace, registry,
                                      queue.Queue(), set(), lambda _text: None)
    assert held is None and not reviewed and registry._ctx._delivery_control_required is True
    shown = [dict(message) for message in ctx.messages]
    _select_completion(registry, ctx, trace, answer=answer)
    text, _usage, _trace = loop._no_tool_final_answer(answer, ctx, trace, registry, queue.Queue(), set(),
                                                      lambda _text: None, explicit_candidate=True)
    assert text == answer  # Evidence for the author, not a finality veto.
    assert reviewed == [answer]
    return shown, registry


def test_ordinary_completion_shows_pending_cleanup_to_the_author_without_veto(tmp_path, monkeypatch):
    pending = {"service_id": "parent1:svc", "name": "svc", "task_id": "parent1", "lifecycle": "cleanup_pending",
               "state": "exited", "stop_failed": True, "cleanup_dispatched": False,
               "stop_error": "surviving process group members: 43"}
    messages, _registry = _ordinary_completion(
        tmp_path, monkeypatch, [pending], "Service cleanup is unconfirmed: process 43 survived the stop.")
    shown = str(messages)
    assert "[SERVICE_FINALIZATION_EVIDENCE]" in shown
    assert "cleanup_pending" in shown and "surviving process group members: 43" not in shown
    [row], _size = _author_evidence(messages)
    assert row["kind"] == "service_finalization_error"
    assert row["services"][0]["lifecycle"] == "cleanup_pending"
    assert row["services"][0]["stop_failed"] is True and "stop_error" not in row["services"][0]


@pytest.mark.parametrize("last_still_registered", [False, True])
def test_each_replacement_stop_reaches_the_ordinary_author(tmp_path, monkeypatch, last_still_registered):
    first = {"service_id": "parent1:preview", "name": "preview", "task_id": "parent1", "launch_id": "service-first",
             "state": "stopped", "cleanup_dispatched": True, "artifact_output_failed": True}
    second = {**first, "launch_id": "service-second"}
    previous = [first, dict(first, uptime_sec=42)]
    current = [{**second, "lifecycle": "stopped"}] if last_still_registered else []
    if not current:
        previous.append(second)
    shown, _ = _ordinary_completion(tmp_path, monkeypatch, current, "Both launches ended with a missing output.",
                                    previous=previous)
    events, _ = _author_evidence(shown)
    assert [[s["launch_id"] for s in event["services"]] for event in events] == [["service-first"], ["service-second"]]
    assert all(event["kind"] == "services_stopped" and event["services"][0]["artifact_output_failed"] for event in events)


def test_the_service_projection_appears_only_when_teardown_recorded_and_keeps_private_paths_out():
    projection = service_finalization.display
    assert projection({"tool_calls": []}) == []
    trace = {"tool_calls": [], "verification_events": [{"kind": "services_stopped", "services": [
        {"service_id": "task:svc", "lifecycle": "stopped", "state": "stopped", "pid": 7, "log_path": "/private",
         "artifact_outputs": "report.html is missing: " + "x" * 9000}]}]}
    shown = projection(trace)
    # Legacy artifact prose can contain paths. The typed view never forwards it.
    [row] = shown
    assert row["services"][0]["lifecycle"] == "stopped" and "artifact_outputs" not in row["services"][0]
    rendered = json.dumps(shown)
    assert "report.html is missing" not in rendered and "log_path" not in rendered


def test_only_a_host_attested_section_is_a_closure_edge():
    from ouroboros.review_source_closure import source_carrier

    for provenance, carrier in (("host_attested", "metadata"), ("agent_supplied", ""), (None, "")):
        evidence = {"service_finalization": [], "__provenance__": {"service_finalization": provenance}}
        assert source_carrier(evidence, "service_finalization", "evidence") == carrier


def test_service_disclosure_carries_and_fingerprints_full_projection():
    ctx = SimpleNamespace(messages=[], forced_service_evidence_fingerprint="")
    service = {"service_id": "task:svc", "lifecycle": "cleanup_pending",
               "artifact_outputs": "x" * 10000 + "a"}
    trace = {"verification_events": [{"kind": "service_finalization_error", "services": [service]}]}
    preview = service_finalization.display_text(trace)
    loop_forced_finalization._disclose_service_finalization(ctx, trace)
    first_fingerprint = ctx.forced_service_evidence_fingerprint
    service["artifact_outputs"] = "x" * 10000 + "b"
    assert service_finalization.display_text(trace) == preview  # the arbitrary detail is not delivered
    loop_forced_finalization._disclose_service_finalization(ctx, trace)
    assert ctx.forced_service_evidence_fingerprint != first_fingerprint
    assert str(ctx.messages).count("[SERVICE_FINALIZATION_EVIDENCE]") == 2


def test_cleanup_failure_after_long_successes_reaches_the_author_with_every_row(tmp_path):
    stopped = {"kind": "services_stopped", "services": [
        {"service_id": f"task:success-{i}", "lifecycle": "stopped", "artifact_outputs": "x" * 1000}
        for i in range(12)]}
    pending = {"kind": "service_finalization_error", "services": [
        {"service_id": "task:late-failure", "lifecycle": "cleanup_pending", "stop_failed": True,
         "cleanup_dispatched": False, "stop_error": "surviving members: 43"}]}
    trace = {"tool_calls": [], "verification_events": [stopped, pending]}
    ctx = SimpleNamespace(messages=[], forced_service_evidence_fingerprint="", drive_root=tmp_path, task_id="task")
    loop_forced_finalization._disclose_service_finalization(ctx, trace)
    assert "task:late-failure" in str(ctx.messages) and "surviving members: 43" not in str(ctx.messages)
    rows = service_finalization.display(trace)
    assert len(rows[0]["services"]) == 12
    assert rows[-1]["kind"] == pending["kind"]
    assert rows[-1]["services"] == [{key: value for key, value in pending["services"][0].items() if key != "stop_error"}]


def _finalized_log(drive_root, name, text):
    """Real local collection with its typed, already addressable log receipt."""
    from ouroboros.tools import services
    from tests._launch_stop_shared import _launch_log

    log = _launch_log(drive_root / "services" / "parent1", name, text.encode("utf-8"))
    return services._finalize_service_log_for_drive(drive_root, SimpleNamespace(secret_values=(), log_path=log))


def _chatty_rows(drive_root, monkeypatch, count=12):
    """Many complete logs, a safely bounded tail after archive failure, a size gap and a Docker row."""
    from ouroboros.tools import services

    def refuse_blob(*_args, **_kwargs):
        raise OSError("blob store full")

    mask = services._service_diagnostic

    def mask_complete(record, text):
        assert text.startswith("FAILED-HEAD\n") and text.endswith("\nFAILED-LAST\n")
        return mask(record, text)

    rows = [{"service_id": f"parent1:svc{i}", "name": f"svc{i}", "task_id": "parent1", "lifecycle": "stopped",
             "log_finalization": _finalized_log(drive_root, f"svc{i}", f"HEAD-{i}\n" + "x" * 90_000 + f"\nLAST-{i}\n")}
            for i in range(count)]
    with monkeypatch.context() as patch:
        patch.setattr(services, "_MAX_SERVICE_LOG_BLOB_BYTES", 1000)
        oversized = _finalized_log(drive_root, "oversized", "OVERSIZED-HEAD\n" + "y" * 4000 + "\nOVERSIZED-LAST\n")
    with monkeypatch.context() as patch:
        patch.setattr(services, "write_blob", refuse_blob)
        patch.setattr(services, "_service_diagnostic", mask_complete)
        failed = _finalized_log(drive_root, "failed", "FAILED-HEAD\n" + "z" * 90_000 + "\nFAILED-LAST\n")
    return rows + [
        {"service_id": "parent1:oversized", "name": "oversized", "task_id": "parent1", "lifecycle": "stopped",
         "log_finalization": oversized},
        {"service_id": "parent1:failed", "name": "failed", "task_id": "parent1", "lifecycle": "stopped",
         "log_finalization": failed},
        {"service_id": "parent1:docker", "name": "docker", "task_id": "parent1", "lifecycle": "stopped",
         "state": "stopped", "cleanup_dispatched": True, "executor": {"type": "docker_exec"},
         "backend_log_path": "/tmp/ouroboros-service-docker.log"},
    ]


def _assert_log_receipts_by_ref(read, rows, size):
    """Both full and tail scopes use the actor's actual reader, while a gap has no recipe."""
    shown = {service["service_id"]: service for row in rows for service in row["services"]}
    assert set(shown) == {f"parent1:svc{i}" for i in range(12)} | {"parent1:oversized", "parent1:failed", "parent1:docker"}
    for i in range(12):
        log = shown[f"parent1:svc{i}"]["log_finalization"]
        assert log["source_scope"] == "full" and log["deleted_live_log"] is True
        source = log["source_ref"]
        assert source["kind"] == "task_source" and source["read"]["tool"] == "read_file"
        assert source["size"] > 80_000
        # Readable by the actual reader, including the head that the 80k tail never held.
        text = read(source["read"]["arguments"])
        assert "lines 1–3 of 3" in text and f"HEAD-{i}" in text and f"LAST-{i}" in text
    oversized, failed = shown["parent1:oversized"]["log_finalization"], shown["parent1:failed"]["log_finalization"]
    assert oversized["source_unavailable"] == "tail_masking_unverified" and "source_ref" not in oversized
    assert failed["source_scope"] == "tail" and failed["error_count"] == 1
    assert failed["source_ref"]["size"] == 80_000
    tail = read(failed["source_ref"]["read"]["arguments"])
    assert "FAILED-LAST" in tail and "FAILED-HEAD" not in tail
    for gap in (oversized, failed):
        assert gap["live_log"] == "retained" and not {"full_log_ref", "tail", "retained_live_log_path", "errors"} & gap.keys()
    assert "log_finalization" not in shown["parent1:docker"]  # an in-container path is no retained source
    assert "ouroboros-service-docker.log" not in json.dumps(rows)
    assert size < 40_000  # twelve 80k tails would be 960k


def _tool_reader(registry):
    return lambda arguments: registry.execute("read_file", arguments)


def test_many_chatty_services_reach_the_ordinary_author_by_readable_ref(tmp_path, monkeypatch):
    rows = _chatty_rows(tmp_path, monkeypatch)
    assert sum(row["log_finalization"]["source_ref"]["size"] for row in rows[:12]) > 12 * 80_000
    messages, registry = _ordinary_completion(tmp_path, monkeypatch, rows,
                                              "Twelve previews stopped; two logs have gaps.")
    _assert_log_receipts_by_ref(_tool_reader(registry), *_author_evidence(messages))


def test_many_chatty_services_reach_the_forced_author_by_readable_ref(tmp_path, monkeypatch):
    from ouroboros.tools import services
    from tests.test_delivery_forced_finalization import _forced_test_context

    loop, registry, ctx, trace = _forced_test_context(tmp_path)
    rows = _chatty_rows(tmp_path, monkeypatch)
    monkeypatch.setattr(services, "stop_task_services", lambda _ctx: rows)
    sent = []
    monkeypatch.setattr(loop, "call_llm_with_retry", lambda _llm, messages, *_a, **_k: (
        sent.append([dict(message) for message in messages])
        or ({"role": "assistant", "content": "Twelve previews stopped; two logs have gaps."}, 0.0)))
    text, _usage, returned = loop._forced_final_answer(ctx, prompt="finalize", fallback_text="fallback",
                                                       reason_code="deadline_local")
    assert text == "Twelve previews stopped; two logs have gaps."
    _assert_log_receipts_by_ref(_tool_reader(registry), *_author_evidence(sent[0]))
    shown = service_finalization.display(returned)
    assert [row["log_finalization"] for row in returned["verification_events"][0]["services"][:12]] == [
        row["log_finalization"] for row in shown[0]["services"][:12]]
    _assert_log_receipts_by_ref(_tool_reader(registry), shown, len(json.dumps(shown)))


def test_display_fingerprints_the_recorded_log_without_reopening_it(tmp_path, monkeypatch):
    from ouroboros import artifacts

    def capture(text):
        log = _finalized_log(tmp_path, "svc", text)
        return {"verification_events": [{"kind": "services_stopped", "services": [
            {"service_id": "parent1:svc", "lifecycle": "stopped", "log_finalization": log}]}]}

    trace, changed = capture("Beginning A\nDiagnostic end\n"), capture("Beginning B\nDiagnostic end\n")
    monkeypatch.setattr(artifacts, "read_actor_source_bytes", lambda *a, **kw: pytest.fail("display read the source"))
    monkeypatch.setattr(artifacts, "persist_exact_text_source", lambda *a, **kw: pytest.fail("display wrote the source"))
    assert service_finalization.evidence(trace) != service_finalization.evidence(changed)
    ctx = SimpleNamespace(messages=[], forced_service_evidence_fingerprint="")
    loop_forced_finalization._disclose_service_finalization(ctx, trace)
    loop_forced_finalization._disclose_service_finalization(ctx, changed)
    assert str(ctx.messages).count("[SERVICE_FINALIZATION_EVIDENCE]") == 2
    for current in (trace, changed):
        [row] = json.loads(service_finalization.display_text(current))
        [service] = current["verification_events"][0]["services"]
        assert row["services"][0]["log_finalization"] == service["log_finalization"]
    assert "Diagnostic end" not in str(ctx.messages)


def test_ledger_download_and_passive_history_project_the_same_new_and_legacy_log_facts(tmp_path, monkeypatch):
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros import artifacts, outcomes
    from ouroboros.gateway.tasks import api_task_artifact, api_task_get
    from ouroboros.task_results import write_task_result

    log = _finalized_log(tmp_path, "preview", "Ordinary diagnostic\n")
    [archive] = [json.loads(line)["full_log_ref"] for line in (tmp_path / "logs/events.jsonl").read_bytes().splitlines()]
    old = {"full_log_ref": archive, "deleted_live_log": True, "tail": "Older diagnostic\n",
           "errors": ["older archive diagnostic"], "source_ref": log["source_ref"]}
    # The earlier receipt with a matching published full source stays useful. A historical
    # tail or a private archive pointer without a published source grants no new recipe.
    unpublished = {key: value for key, value in old.items() if key != "source_ref"}
    old_tail = {"deleted_live_log": False, "tail": "Earlier diagnostic sample\n",
                "retained_live_log_path": str(tmp_path / "private.log")}
    trace = {"verification_events": [{"kind": "services_stopped", "services": [
        {"name": f"preview-{i}", "task_id": "parent1", "lifecycle": "stopped", "log_finalization": fact}
        for i, fact in enumerate((log, old, unpublished, old_tail))]}]}
    original = json.dumps(trace)
    monkeypatch.setattr(artifacts, "read_actor_source_bytes", lambda *a, **kw: pytest.fail("passive read opened a log"))
    monkeypatch.setattr(artifacts, "persist_exact_text_source", lambda *a, **kw: pytest.fail("passive read saved a log"))
    [event] = service_finalization.display(trace)
    current, legacy_full, legacy_unpublished, legacy_tail = [row["log_finalization"] for row in event["services"]]
    assert current == log
    assert legacy_full == {**current, "error_count": 1}
    assert legacy_unpublished["source_unavailable"] == "legacy_source_not_published" and "source_ref" not in legacy_unpublished
    assert legacy_tail["source_unavailable"] == "tail_masking_unverified" and "source_ref" not in legacy_tail

    ledger = outcomes.build_verification_ledger(task={"id": "parent1"}, loop_outcome={}, llm_trace=trace, artifact_bundle={})
    assert ledger["entries"][-1] == event and json.dumps(trace) == original
    saved = outcomes.maybe_write_verification_artifact(tmp_path, "parent1", ledger, threshold_chars=0)
    mixed = {**ledger, "entries": [*ledger["entries"][:-1], trace["verification_events"][0]]}
    write_task_result(tmp_path, "parent1", "completed", verification_ledger=mixed,
                      artifacts=[saved["artifact"]], artifact_status="ready")
    app = Starlette(routes=[Route("/api/tasks/{task_id}", api_task_get),
        Route("/api/tasks/{task_id}/artifacts/{name}", api_task_artifact)])
    app.state.drive_root = tmp_path
    with TestClient(app) as client:
        history = client.get("/api/tasks/parent1")
        download = client.get("/api/tasks/parent1/artifacts/verification_ledger.json")
    assert history.status_code == download.status_code == 200
    assert history.json()["verification_ledger"]["entries"][-1]["services"] == event["services"]
    assert download.json() == ledger and download.content == pathlib.Path(saved["artifact"]["path"]).read_bytes()
    for response in (history, download):
        assert all(word not in response.text for word in (
            '"full_log_ref"', '"tail"', "private.log", "Older diagnostic", "Ordinary diagnostic", "older archive diagnostic"))


# --- A confirmed stop whose log could not be read (the e758 counterexample) ---------------------------------------

def _ended_service(root, monkeypatch, backend, task="t-svc"):
    """Task ``task``'s exited service ``paper`` on ``backend``: its log on disk, one declared output. Only the
    process-table pass (one confirmed exit) and the output registration are replaced; the stop, the log collector,
    the batch and the loop's record are the real ones. Returns the record, its log, the registry and a list that
    says, per registration, whether the stop still held the name."""
    from ouroboros import process_scope, workspace_executor
    from ouroboros.launch_custody import service_log_launch_id
    from ouroboros.tools import services, shell
    from tests._launch_stop_shared import _launch_log

    key = f"{task}:paper"
    log = _launch_log(root / "services" / task, "paper" if backend == "host" else "paper.executor",
                      b"HEAD-LINE of the run\nLAST-LINE\n")
    launch = service_log_launch_id(log.name)
    exited = SimpleNamespace(pid=None, poll=lambda: 0)
    registry: dict = {}
    monkeypatch.setattr(services, "_SERVICES", registry if backend == "host" else {})
    monkeypatch.setattr(workspace_executor, "_SERVICES", registry if backend != "host" else {})
    if backend == "host":
        record = services.ServiceRecord(name="paper", service_id=key, task_id=task, cmd=["paper"], cwd=str(root),
                                        log_path=log, proc=exited, outputs=["out.txt"], drive_root=str(root))
        record.record_id = launch
    else:
        record = workspace_executor._ExecutorService(
            service_id=key, task_id=task, name="paper", cmd=["paper"], host_cwd=root, backend_cwd=str(root),
            executor=SimpleNamespace(kind="local", executor_id="local", network="off"), cwd_root="task_drive",
            outputs=["out.txt"], before_outputs={}, local_proc=exited, backend_log_path=str(log), drive_root=root,
            record_id=launch if backend == "local_executor" else "service-" + launch.removeprefix("launch-"))
    registry[key] = record
    monkeypatch.setattr(process_scope, "end_held_processes", lambda held, **_kw: {
        item["key"]: {"facts": {"exit_confirmed": True}, "birth": "", "known": []} for item in held})
    held = []

    def capture(*_args, output_outcomes, **_kwargs):
        held.append(registry.get(key) is record)
        output_outcomes.append({"index": 1, "status": "saved", "registered": True,
                                "artifacts": [{"name": "out.txt", "sha256": "a" * 64}]})
        return "\n\n- out.txt: registered", False, True

    monkeypatch.setattr(shell, "_register_process_outputs", capture)
    monkeypatch.setattr(services, "_working_tree_changes", lambda _cwd: set())
    return record, log, registry, held


@pytest.mark.parametrize("backend", ["host", "local_executor", "docker"])
@pytest.mark.parametrize("second_status", ["absent", "unconfirmed"])
def test_explicit_stop_keeps_partial_output_receipts_and_is_collected_once(tmp_path, monkeypatch, backend, second_status):
    from ouroboros import workspace_executor
    from ouroboros.artifacts import registered_task_artifact
    from ouroboros.tools import services, shell, shell_outputs

    record, _log, registry, _held = _ended_service(tmp_path, monkeypatch, backend)
    record.outputs = ["saved.txt", "second.txt"]
    (tmp_path / "saved.txt").write_bytes(b"preview output")
    if second_status == "unconfirmed":
        (tmp_path / "second.txt").write_bytes(b"second output")
        copier = shell_outputs.copy_file_to_task_artifacts

        def fail_after_first(ctx, source, **kwargs):
            if source.name == "second.txt":
                raise RuntimeError("synthetic exporter fault")
            return copier(ctx, source, **kwargs)

        monkeypatch.setattr(shell_outputs, "copy_file_to_task_artifacts", fail_after_first)
    if backend == "docker":  # typed ordinary Docker completion, no Docker invocation in this test
        record.executor = SimpleNamespace(kind="docker_exec", executor_id="docker", network="none",
                                          container_name="fixture")
        record.backend_pid = "41234"
        record.backend_log_path = "/container/diagnostic.log"
        monkeypatch.setattr(workspace_executor, "subprocess", SimpleNamespace(
            PIPE=-1, run=lambda *_a, **_k: SimpleNamespace(returncode=0)))
        monkeypatch.setattr(workspace_executor, "_docker_observe_state", lambda *_a, **_k: "exited")
    monkeypatch.setattr(shell, "_register_process_outputs", shell_outputs._register_process_outputs)
    monkeypatch.setattr(shell_outputs, "_resolve_declared_output", lambda _ctx, name, cwd, **_kw: (cwd / name, ""))
    tool_ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id=record.task_id)
    ctx = _exit_context(tmp_path, tool_ctx, event_queue=queue.Queue())
    tool_ctx._execution_trace = ctx.llm_trace

    result = services._stop_answer(record.name, services._stop_by_name(tool_ctx, record.name))
    assert (result.status, result.code, dict(result.meta)) == ("error", "ARTIFACT_OUTPUT_ERROR", {
        "artifact_registered": True, "service_output_launch": {"task_id": record.task_id, "launch_id": record.record_id}})
    payload = json.loads(result.text.partition("\n\n")[2])
    assert bool(registry) is (second_status == "unconfirmed")
    if second_status == "absent":
        assert services._stop_service(tool_ctx, name=record.name) == "⚠️ SERVICE_NOT_FOUND: paper"
    artifact = registered_task_artifact(tmp_path, record.task_id, "saved.txt")
    assert pathlib.Path(artifact["path"]).read_bytes() == b"preview output"
    first, second = payload["output_outcomes"]
    version_ref = first["artifacts"][0]["version_ref"]
    version = registered_task_artifact(tmp_path, record.task_id, version_ref["path"])
    assert version["immutable"] is True and pathlib.Path(version["path"]).read_bytes() == b"preview output"
    assert (first["index"], first["status"], first["artifacts"]) == (
        1, "saved", [{"name": "saved.txt", "sha256": artifact["sha256"], "version_ref": version_ref}])
    assert second == {"index": 2, "status": second_status}
    assert payload["log_outcome"] == ("backend_only" if backend == "docker" else "saved")
    assert service_finalization.finalize(ctx, ctx.llm_trace)
    [event] = service_finalization.evidence(ctx.llm_trace)
    [stop] = event["services"]
    assert event["kind"] == "services_stopped" and stop["stop_source"] == "explicit"
    assert stop["output_outcomes"][0]["registered"] is True
    assert stop["output_outcomes"][1] == second and stop["log_outcome"] == payload["log_outcome"]
    frame = ctx.event_queue.get_nowait()["data"]
    second_words = "absent" if second_status == "absent" else "capture unconfirmed"
    assert f"output 1 saved: saved.txt · output 2 {second_words}" in frame["text"]
    assert str(tmp_path) not in frame["text"] and "/container" not in frame["text"]
    assert not service_finalization.finalize(ctx, ctx.llm_trace) and ctx.event_queue.empty()


def test_output_stat_refusal_preserves_the_sibling_capture(tmp_path, monkeypatch):
    from ouroboros.artifacts import registered_task_artifact
    from ouroboros.tools import shell_outputs

    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path / "data", task_id="capture")
    good, unreadable = tmp_path / "good.txt", tmp_path / "unreadable.txt"
    good.write_bytes(b"good output")
    unreadable.write_bytes(b"other output")
    monkeypatch.setattr(shell_outputs, "_resolve_declared_output", lambda ctx, name, cwd, **kw: (cwd / name, ""))
    _refuse_metadata_of(monkeypatch, unreadable)
    outcomes = []
    note, failed, registered = shell_outputs._register_process_outputs(
        ctx, [unreadable.name, good.name], tmp_path, output_outcomes=outcomes)
    assert failed and registered and "could not be inspected (PermissionError)" in note and "missing output" not in note
    saved = registered_task_artifact(ctx.drive_root, ctx.task_id, good.name)
    assert pathlib.Path(saved["path"]).read_bytes() == b"good output"
    version_ref = outcomes[1]["artifacts"][0]["version_ref"]
    version = registered_task_artifact(ctx.drive_root, ctx.task_id, version_ref["path"])
    assert version["immutable"] is True and pathlib.Path(version["path"]).read_bytes() == b"good output"
    assert outcomes == [{"index": 1, "status": "unconfirmed"}, {
        "index": 2, "status": "saved", "registered": True,
        "artifacts": [{"name": good.name, "sha256": saved["sha256"], "version_ref": version_ref}]}]


def test_each_output_version_is_readable_after_unchanged_reuse_and_copyback(tmp_path, monkeypatch):
    import zipfile

    from ouroboros import artifacts, service_outcomes
    from ouroboros.headless import copy_child_task_result
    from ouroboros.task_results import write_task_result
    from ouroboros.tools import shell_outputs
    from ouroboros.tools.registry import ToolRegistry

    child, work, parent = (tmp_path / name for name in ("child", "work", "parent"))
    for directory in (child, work, parent):
        directory.mkdir()
    ctx = ToolContext(repo_dir=work, drive_root=child, task_id="versions")
    monkeypatch.setattr(shell_outputs, "_resolve_declared_output", lambda _ctx, name, cwd, **_kw: (cwd / name, ""))
    report, bundle = work / "report.txt", work / "bundle"
    bundle.mkdir()
    member = bundle / "member.txt"
    outputs = [report.name, bundle.name]

    def capture():
        outcomes = []
        note, failed, registered = shell_outputs._register_process_outputs(
            ctx, outputs, work, output_outcomes=outcomes)
        assert not failed, note
        return outcomes, registered

    def versions(outcomes):
        return [item["version_ref"]["path"] for row in outcomes for item in row["artifacts"]]

    report.write_text("first", encoding="utf-8")
    member.write_text("first directory member", encoding="utf-8")
    first, first_registered = capture()
    assert first_registered is True
    report.write_text("second", encoding="utf-8")
    member.write_text("second directory member", encoding="utf-8")
    second, second_registered = capture()
    assert second_registered is True
    first_versions, second_versions = versions(first), versions(second)
    assert len(first_versions) == len(second_versions) == 3
    assert not set(first_versions) & set(second_versions)
    assert service_outcomes.public_outputs(first) == first
    assert service_outcomes.public_outputs(second) == second
    assert "source_path" not in json.dumps(first + second)

    before = {str(path): shell_outputs._fingerprint_output(path) for path in (report, bundle)}
    repeated, repeated_registered = shell_outputs._register_process_outputs(
        ctx, outputs, work, before_outputs=before, output_outcomes=(third := []))
    assert repeated_registered is False and "ARTIFACT_OUTPUT_NOTE" in repeated
    assert versions(third) == second_versions

    write_task_result(
        child, ctx.task_id, "completed", result="done",
        artifacts=artifacts.collect_task_artifact_records(child, ctx.task_id),
        verification_events=[
            {"kind": "services_stopped", "services": [{"task_id": ctx.task_id, "launch_id": "launch-one",
                                                       "output_outcomes": first}]},
            {"kind": "services_stopped", "services": [{"task_id": ctx.task_id, "launch_id": "launch-two",
                                                       "output_outcomes": second}]},
        ],
    )
    copied = copy_child_task_result(parent, {"id": ctx.task_id, "drive_root": str(child)})
    assert copied and copied["verification_events"]
    registry = ToolRegistry(repo_dir=work, drive_root=parent)
    registry._ctx.task_id = ctx.task_id

    def read(ref):
        return registry.execute("read_file", {"root": ref["root"], "path": ref["path"]})

    first_file, second_file = first[0]["artifacts"][0], second[0]["artifacts"][0]
    assert first_file["name"] == second_file["name"] == "report.txt"
    assert first_file["sha256"] != second_file["sha256"]
    assert first_file["version_ref"] != second_file["version_ref"]
    assert "first" in read(first_file["version_ref"])
    assert "second" in read(second_file["version_ref"])
    for outcomes, expected in ((first, "first directory member"), (second, "second directory member")):
        manifest, archive = outcomes[1]["artifacts"]
        assert '"path": "member.txt"' in read(manifest["version_ref"])
        with zipfile.ZipFile(artifacts.task_artifact_dir_path(parent, ctx.task_id) / archive["version_ref"]["path"]) as opened:
            assert opened.read("member.txt").decode("utf-8") == expected


def test_failed_immutable_version_is_not_a_readable_version(tmp_path, monkeypatch):
    from ouroboros.tools import shell_outputs

    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path / "data", task_id="versions")
    (tmp_path / "report.txt").write_text("current", encoding="utf-8")
    monkeypatch.setattr(shell_outputs, "_resolve_declared_output", lambda _ctx, name, cwd, **_kw: (cwd / name, ""))
    original = shell_outputs.copy_file_to_task_artifacts

    def refuse_version(*args, **kwargs):
        if kwargs.get("immutable"):
            raise OSError("controlled immutable refusal")
        return original(*args, **kwargs)

    monkeypatch.setattr(shell_outputs, "copy_file_to_task_artifacts", refuse_version)
    outcomes = []
    note, failed, registered = shell_outputs._register_process_outputs(
        ctx, ["report.txt"], tmp_path, output_outcomes=outcomes)
    assert failed and registered and "failed immutable output version" in note
    assert outcomes == [{"index": 1, "status": "unconfirmed"}]
    assert "version_ref" not in json.dumps(outcomes)


def test_public_output_accepts_only_a_matching_immutable_version_ref():
    from ouroboros import service_outcomes

    digest = "a" * 64
    valid = {"root": "artifact_store", "path": "report-" + digest + ".txt", "sha256": digest}
    row = {"index": 1, "status": "saved", "artifacts": [{"name": "report.txt", "sha256": digest,
                                                         "source_path": "/private/source", "version_ref": valid}]}
    assert service_outcomes.public_outputs([row]) == [{"index": 1, "status": "saved", "artifacts": [
        {"name": "report.txt", "sha256": digest, "version_ref": valid}]}]
    row["artifacts"][0]["version_ref"] = {**valid, "path": "../escape.txt"}
    assert service_outcomes.public_outputs([row]) == [{"index": 1, "status": "saved", "artifacts": [
        {"name": "report.txt", "sha256": digest}]}]


@pytest.mark.parametrize("backend", ["host", "local_executor", "docker"])
@pytest.mark.parametrize("retry", ["explicit", "reloaded_name", "automatic", "replacement", "retained", "missing"])
def test_ended_service_retries_only_its_owed_outputs(tmp_path, monkeypatch, backend, retry):
    from ouroboros import launch_custody, process_scope, workspace_executor
    from ouroboros.artifacts import registered_task_artifact
    from ouroboros.tools import kept_service_outputs, services, shell, shell_outputs

    record, _, registry, _ = _ended_service(tmp_path, monkeypatch, backend)
    record.outputs = ["saved.txt", "owed.txt"]
    data = tmp_path
    tool_ctx = ToolContext(repo_dir=data, drive_root=data, task_id=record.task_id)
    ctx = _exit_context(data, tool_ctx, event_queue=queue.Queue())
    tool_ctx._execution_trace = ctx.llm_trace
    if backend == "docker":
        record.executor = SimpleNamespace(kind="docker_exec", executor_id="fixture", network="none",
                                          container_name="b" * 64, mappings=(SimpleNamespace(host_path=data,
                                                                                            backend_path="/work"),))
        record.backend_pid = "41534"
        monkeypatch.setattr(workspace_executor, "subprocess", SimpleNamespace(
            PIPE=-1, run=lambda *_a, **_k: SimpleNamespace(returncode=0)))
        monkeypatch.setattr(workspace_executor, "_docker_observe_state", lambda *_a, **_k: "exited")
        # The transport preparations are not being tested on a physical backend here.
        monkeypatch.setattr("ouroboros.workspace_executor_paths.prepare_docker_outputs", lambda *_a, **_k: None)
    elif backend == "local_executor":
        record.executor = SimpleNamespace(kind="local", executor_id="fixture", network="host", mappings=(
            SimpleNamespace(host_path=data, backend_path="/work"),))

    saved, owed = (data / name for name in record.outputs)
    saved.write_bytes(b"first file from the service")
    owed.write_bytes(b"file the first copy missed")
    monkeypatch.setattr(shell, "_register_process_outputs", shell_outputs._register_process_outputs)
    monkeypatch.setattr(shell_outputs, "_resolve_declared_output", lambda _ctx, name, cwd, **_kw: (cwd / name, ""))
    real_copy = shell_outputs.copy_file_to_task_artifacts

    def blocked(ctx, source, **kwargs):
        if source == owed:
            raise OSError("synthetic temporary artifact store failure")
        return real_copy(ctx, source, **kwargs)

    monkeypatch.setattr(shell_outputs, "copy_file_to_task_artifacts", blocked)
    assert service_finalization.finalize(ctx, ctx.llm_trace)
    first = ctx.event_queue.get_nowait()["data"]
    assert first["checkpoint_kind"] == "services_stopped" and "output 2 capture unconfirmed" in first["text"]
    declaration = kept_service_outputs.declaration_path(data, record.task_id, record.record_id)
    assert declaration.is_file() and registry[record.service_id] is record

    saved.write_bytes(b"not from this launch")  # already registered: the retry must not reopen this declaration
    if retry == "missing":
        owed.unlink()
    monkeypatch.setattr(shell_outputs, "copy_file_to_task_artifacts", real_copy)
    monkeypatch.setattr(process_scope, "end_held_processes", lambda *_a, **_k: pytest.fail("second process stop"))
    monkeypatch.setattr(workspace_executor, "_stop_service_record", lambda *_a, **_k: pytest.fail("second executor stop"))
    if retry == "retained":
        registry.clear()  # a later process only has the declaration
        monkeypatch.setattr(launch_custody, "_kept_selection", lambda *_a, **_k: ([], True, ""))
        monkeypatch.setattr("ouroboros.owned_shutdown._stop_records", lambda *_a, **_k: pytest.fail("second retained stop"))
        emitted = []
        monkeypatch.setattr(service_finalization, "publish_retained_outputs", lambda _root, row: emitted.append(row))
        result = launch_custody.stop_kept_service(data, record.record_id, reason="save the remaining file", actor="owner:test")
        assert result["ok"] and result["outcome"] == "not_found" and len(emitted) == 1
        [updated] = service_finalization.checkpoints([{"kind": "service_outputs_updated", "services": emitted}])
    else:
        if retry in {"explicit", "reloaded_name"}:
            if retry == "reloaded_name":
                registry.clear()  # the file receipt, not a live process, now identifies the launch
            result = services._stop_answer(record.name, services._stop_by_name(tool_ctx, record.name))
            assert result.code == "OK" and result.meta["artifact_registered"] and result.meta["service_output_recovered"]
            assert result.meta["service_output_launch"] == {"task_id": record.task_id, "launch_id": record.record_id}
            row = json.loads(result.text)
            assert row["output_retry"] and [output["status"] for output in row["output_outcomes"]] == ["saved", "saved"]
        elif retry == "replacement":
            from contextlib import ExitStack

            with ExitStack() as admission:
                gap, _ = services._admit_named_start(tool_ctx, record.name, data, admission, kept_writer=False)
                assert gap == {}
        assert service_finalization.finalize(ctx, ctx.llm_trace)
        updated = ctx.event_queue.get_nowait()["data"]
        assert not service_finalization.finalize(ctx, ctx.llm_trace) and ctx.event_queue.empty()
    assert updated["checkpoint_kind"] == "service_outputs_updated" and updated["card_row_id"] != first["card_row_id"]
    assert ("output 2 absent" if retry == "missing" else "output 2 saved: owed.txt") in updated["text"]
    assert "stopped" not in updated["text"]
    assert not declaration.exists() and not registry
    expected = [(saved.name, b"first file from the service")]
    if retry != "missing":
        expected.append((owed.name, b"file the first copy missed"))
    else:
        assert registered_task_artifact(data, record.task_id, owed.name) is None
    for name, exact in expected:
        assert pathlib.Path(registered_task_artifact(data, record.task_id, name)["path"]).read_bytes() == exact


@pytest.mark.parametrize("backend", ["host", "local_executor"])
def test_retained_status_fault_does_not_drop_the_preceding_stop(tmp_path, monkeypatch, backend):
    from ouroboros import workspace_executor
    from ouroboros.tools import services

    record, _, registry, held = _ended_service(tmp_path, monkeypatch, backend)
    kept = replace(record, name="sidecar", service_id=f"{record.task_id}:sidecar", outputs=[], keep_alive=True)
    registry[kept.service_id] = kept

    def unavailable(*_args, **_kwargs):
        raise OSError("synthetic diagnostic problem")

    monkeypatch.setattr(services, "_status_payload", unavailable)
    monkeypatch.setattr(workspace_executor, "_service_payload", unavailable)
    first, second = services.stop_task_services(ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id=record.task_id))
    assert first["lifecycle"] == "stopped" and first["output_outcomes"][0]["status"] == "saved" and held == [True]
    assert (second["name"], second["lifecycle"], second["state"], second["status_unavailable"]) == (
        "sidecar", "kept", "unknown", True)


def test_service_tool_error_cannot_be_recovered_by_another_launch_or_generic_artifact():
    from ouroboros.outcomes import _classify_tool_errors

    launch = {"task_id": "producer", "launch_id": "launch-first"}
    failed = {"tool": "stop_service", "status": "artifact_output_error", "is_error": True,
              "args": {"name": "preview"}, "tool_result_meta": {"service_output_launch": launch}}
    other = {"tool": "stop_service", "status": "ok", "artifact_registered": True, "args": failed["args"],
             "tool_result_meta": {"service_output_launch": dict(launch, launch_id="launch-second"),
                                  "service_output_recovered": True}}
    trace = {"tool_calls": [failed, other, {"tool": "run_command", "status": "ok", "artifact_registered": True}]}
    assert len(_classify_tool_errors(trace)["unresolved"]) == 1
    trace["verification_events"] = [{"kind": "service_outputs_updated", "services": [
        {**launch, "artifact_output_failed": False, "output_outcomes": [{"index": 1, "status": "saved"}]}]}]
    classified = _classify_tool_errors(trace)
    assert not classified["unresolved"] and len(classified["recovered"]) == 1


def test_no_service_keeps_an_empty_trace_empty(tmp_path, monkeypatch):
    from ouroboros.tools import services

    ctx = _exit_context(tmp_path, ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="task"))
    monkeypatch.setattr(services, "stop_task_services", lambda _: [])
    assert not service_finalization.finalize(ctx, ctx.llm_trace)
    assert ctx.llm_trace == {}


def test_stop_errors_from_two_different_passes_preserve_the_selected_launch(tmp_path, monkeypatch):
    from ouroboros import service_outcomes
    from ouroboros.tools import services

    tool_ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="task")
    ctx = _exit_context(tmp_path, tool_ctx, event_queue=queue.Queue())
    tool_ctx._execution_trace = ctx.llm_trace
    record = SimpleNamespace(service_id="task:one", name="one", task_id="task", record_id="service-one")

    def unavailable(*_):
        tool_ctx._execution_trace = {}  # loop cleanup restores this binding before collecting its final trace
        raise OSError("observation unavailable")

    with pytest.raises(OSError):
        service_outcomes.observe_stop(tool_ctx, record, unavailable, source="replacement")
    assert tool_ctx._execution_trace == {}
    monkeypatch.setattr(services, "stop_task_services", unavailable)
    assert service_finalization.finalize(ctx, ctx.llm_trace)
    assert not service_finalization.finalize(ctx, ctx.llm_trace)
    previous, teardown = service_finalization.evidence(ctx.llm_trace)
    [row] = previous["services"]
    assert row["launch_id"] == record.record_id and row["lifecycle"] == "cleanup_pending" and row["stop_failed"]
    assert teardown == {"kind": "service_finalization_error", "services": [], "error": "OSError"}
    assert "Background service cleanup unconfirmed: one (one)" in ctx.event_queue.get_nowait()["data"]["text"]
    assert ctx.event_queue.get_nowait()["data"]["text"] == "Background service outcomes could not be collected"
    assert ctx.event_queue.empty()


def test_executor_task_end_never_stops_a_replacement_for_its_snapshot(tmp_path, monkeypatch):
    from ouroboros import workspace_executor
    from ouroboros.tools import services

    old, log, registry, captures = _ended_service(tmp_path, monkeypatch, "local_executor")
    replacement = replace(old, outputs=["new.txt"])
    registry[old.service_id] = replacement
    monkeypatch.setattr(workspace_executor, "_services_snapshot", lambda: [old])
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id=old.task_id)
    assert services.stop_task_services(ctx) == []
    assert registry[old.service_id] is replacement and not captures and log.exists()


@pytest.mark.parametrize("backend", ["host", "local_executor"])
@pytest.mark.parametrize("store_log", [False, True])
def test_unknown_declared_outputs_reach_author_card_and_outcome_beside_other_services(
        tmp_path, monkeypatch, backend, store_log):
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.outcomes import EXECUTION_DEGRADED, derive_loop_outcome
    from ouroboros.tools import services, shell

    record, log, registry, _ = _ended_service(tmp_path, monkeypatch, backend)
    for name, kept in (("other", False), ("keep", True)):
        path = log.with_name(f"{name}.log")
        path.write_text(name, encoding="utf-8")
        location = {"log_path": path} if backend == "host" else {"backend_log_path": str(path)}
        peer = replace(record, name=name, service_id=f"{record.task_id}:{name}", outputs=[], keep_alive=kept, **location)
        registry[peer.service_id] = peer

    def failed(*args, **kwargs):
        raise OSError(f"{tmp_path}/raw-output: PRIVATE-VALUE")

    monkeypatch.setattr(shell, "_register_process_outputs", failed)
    if not store_log:
        monkeypatch.setattr(services, "write_blob", failed)
    tool_ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id=record.task_id)
    ctx = _exit_context(tmp_path, tool_ctx, event_queue=queue.Queue())
    assert service_finalization.finalize(ctx, ctx.llm_trace)
    events = ctx.llm_trace["verification_events"]
    stopped, kept = events
    assert [event["kind"] for event in events] == ["services_stopped", "services_kept"]
    bad, good = stopped["services"]
    assert bad["process_cleanup"]["exit_confirmed"] and bad["lifecycle"] == "stopped" and "stop_failed" not in bad
    assert bad["artifact_output_unknown"] and bad["artifact_output_failed"]
    assert bad["log_finalization"]["archive_outcome"] == ("saved" if store_log else "not_saved")
    assert good["name"] == "other" and not good["artifact_output_failed"]
    assert kept["services"][0]["name"] == "keep" and set(registry) == {record.service_id, f"{record.task_id}:keep"}
    card = ctx.event_queue.get_nowait()["data"]["text"]
    assert card.startswith("Background service stopped: paper") and "output 1 capture unconfirmed" in card
    assert ctx.event_queue.get_nowait()["data"]["text"].startswith("Background service stopped: other")
    assert ctx.event_queue.get_nowait()["data"]["text"] == "Background services kept"
    shown = service_finalization.display(ctx.llm_trace)[0]["services"][0]
    shown_log = shown["log_finalization"]
    assert shown["artifact_output_unknown"] and shown_log["source_scope"] == "full"  # the small safe log fits whole
    assert read_actor_source_bytes(tmp_path, record.task_id, shown_log["source_ref"]) == b"HEAD-LINE of the run\nLAST-LINE\n"
    execution = derive_loop_outcome("The service ended.", {}, ctx.llm_trace)["outcome_axes"]["execution"]
    assert execution["status"] == EXECUTION_DEGRADED
    [gap] = execution["failure"]["verification_failures"]
    assert gap["service"] == "paper" and gap["status"] == "artifact_output_error" and "not be confirmed" in gap["reason"]
    for private in (str(tmp_path), "PRIVATE-VALUE", "HEAD-LINE", "LAST-LINE"):
        assert private not in json.dumps([card, shown, gap])


@pytest.mark.parametrize("output_unknown", [False, True])
def test_masked_tail_is_not_inline_when_its_source_cannot_be_stored(tmp_path, monkeypatch, output_unknown):
    from ouroboros import artifacts, service_outcomes
    from ouroboros.tools import services

    def refuse_archive(*_args):
        raise OSError("sample archive failure")

    monkeypatch.setattr(services, "_store_service_log", refuse_archive)
    monkeypatch.setattr(services, "_MAX_SERVICE_LOG_TAIL_CHARS", 16)
    monkeypatch.setattr(artifacts, "persist_exact_text_source", lambda *a, **kw: ("", {}, {"status": "unavailable"}))
    log = _finalized_log(tmp_path, "sample", "First ordinary line\nLast ordinary line\n")
    shown = service_outcomes.display({"artifact_output_unknown": output_unknown, "log_finalization": log})
    assert shown["artifact_output_unknown"] is output_unknown
    assert shown["log_finalization"] == log == {"archive_outcome": "not_saved", "deleted_live_log": False,
        "live_log": "retained", "error_count": 1, "source_unavailable": "masked_source_not_retained"}


@pytest.mark.parametrize("backend", ["host", "local_executor"])
@pytest.mark.parametrize("source_available", [True, False])
def test_stop_author_and_history_share_one_public_receipt(tmp_path, monkeypatch, backend, source_available):
    from ouroboros import artifacts, process_scope
    from ouroboros.outcomes import public_task_result
    from ouroboros.tools import services, shell

    record, log_path, _, _ = _ended_service(tmp_path, monkeypatch, backend)
    if backend == "local_executor":
        record.local_proc.returncode = 0  # the executor reads the saved Popen code after poll
    tool_ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id=record.task_id)
    ctx = _exit_context(tmp_path, tool_ctx, event_queue=queue.Queue())
    tool_ctx._execution_trace = ctx.llm_trace
    if not source_available:
        monkeypatch.setattr(artifacts, "persist_exact_text_source", lambda *a, **kw: (
            "", {}, {"reason": f"synthetic source write failure at {tmp_path}"}))

    end = process_scope.end_held_processes
    capture = shell._register_process_outputs
    cleanup = {"coverage": process_scope.COVERAGE_POSIX, "reach": ["group", "descendants", "owed"],
               "exit_confirmed": True, "ended": [12001], "already_gone": [12002]}

    def end_with_facts(*args, **kwargs):
        settled = end(*args, **kwargs)
        settled[record.service_id]["facts"].update(cleanup)
        return settled

    def capture_with_note(*args, **kwargs):
        _note, failed, registered = capture(*args, **kwargs)
        return f"Synthetic exporter note at {tmp_path / 'out.txt'}", failed, registered

    monkeypatch.setattr(process_scope, "end_held_processes", end_with_facts)
    monkeypatch.setattr(shell, "_register_process_outputs", capture_with_note)

    stopped_text = services._stop_service(tool_ctx, name=record.name)
    stopped = json.loads(stopped_text)
    assert stopped["lifecycle"] == "stopped" and stopped["returncode"] == 0
    assert stopped["process_cleanup"] == cleanup
    assert stopped["output_outcomes"] == [{"index": 1, "status": "saved", "registered": True,
                                           "artifacts": [{"name": "out.txt", "sha256": "a" * 64}]}]
    if backend == "local_executor":
        assert stopped["executor"] == {"type": "local"} and stopped["backend"] == "local_executor"

    assert service_finalization.finalize(ctx, ctx.llm_trace)
    [author_event] = service_finalization.display(ctx.llm_trace)
    assert author_event["services"] == [stopped]
    card = ctx.event_queue.get_nowait()["data"]["text"]
    assert "requested stop · log saved · output 1 saved: out.txt" in card
    [recorded] = ctx.llm_trace["verification_events"][0]["services"]
    with monkeypatch.context() as patch:
        patch.setattr(artifacts, "read_actor_source_bytes", lambda *a, **kw: pytest.fail("history read reopened a log"))
        patch.setattr(artifacts, "persist_exact_text_source", lambda *a, **kw: pytest.fail("history read saved a log"))
        historical = public_task_result({"services": [recorded, stopped]}, include_outcome_axes=False)["services"]
    assert historical == [stopped, stopped]

    shown_log = stopped["log_finalization"]
    assert recorded["log_finalization"] == shown_log
    assert shown_log["deleted_live_log"] and not log_path.exists()
    if source_available:
        source = shown_log["source_ref"]
        assert "source_unavailable" not in shown_log and shown_log["source_scope"] == "full"
        assert artifacts.read_actor_source_bytes(tmp_path, record.task_id, source) == b"HEAD-LINE of the run\nLAST-LINE\n"
    else:
        assert shown_log["source_unavailable"] == "masked_source_not_retained" and "source_ref" not in shown_log
    rendered = json.dumps([stopped, author_event, card, historical])
    for private in (str(tmp_path), "Synthetic exporter note", "HEAD-LINE", "LAST-LINE", "source write failure"):
        assert private not in rendered


@pytest.mark.parametrize("backend", ["host", "local_executor"])
def test_stop_tool_reports_unconfirmed_output_capture_without_diagnostics(tmp_path, monkeypatch, backend):
    from ouroboros.tools import shell
    from ouroboros.tools.registry import ToolRegistry

    record, _, entries, _ = _ended_service(tmp_path, monkeypatch, backend)

    def failed(*args, **kwargs):
        raise OSError(f"{tmp_path}/private-output PRIVATE-VALUE")

    monkeypatch.setattr(shell, "_register_process_outputs", failed)
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **kw: (True, ""))
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry.set_context(ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id=record.task_id))
    result = registry.execute_result("stop_service", {"name": "paper"})
    assert (result.status, result.code) == ("error", "ARTIFACT_OUTPUT_ERROR"), result.text
    shown = json.loads(result.text.partition("\n\n")[2])
    assert shown["artifact_output_unknown"] and shown["process_cleanup"]["exit_confirmed"] and entries[record.service_id] is record
    assert shown["log_finalization"]["source_ref"]["kind"] == "task_source"
    for private in (str(tmp_path), "PRIVATE-VALUE", "HEAD-LINE", "LAST-LINE"):
        assert private not in result.text


def test_executor_preserves_captured_outputs_and_log_on_late_bookkeeping_error(tmp_path, monkeypatch):
    from ouroboros import workspace_executor
    from ouroboros.tools import services

    record, _, entries, captures = _ended_service(tmp_path, monkeypatch, "local_executor")

    def failed(_path):
        raise OSError("private bookkeeping error")

    monkeypatch.setattr(workspace_executor, "_forget_process", failed)
    [row] = services.stop_task_services(ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id=record.task_id))
    assert row["lifecycle"] == "stopped" and row["finalization_error"] == "OSError"
    assert not row["artifact_output_failed"] and "artifact_output_unknown" not in row
    assert row["log_finalization"]["source_scope"] == "full" and "out.txt" in row["artifact_outputs"]
    assert captures == [True] and not entries


@pytest.mark.parametrize("backend", ["host", "local_executor"])
def test_an_ended_service_whose_log_cannot_be_read_stays_recorded_with_its_outputs(tmp_path, monkeypatch, backend):
    """The stop confirmed the end, then the log's metadata was refused. The end stays known, the declared output is
    still registered while the name is held, the log's fate is unknown (never deleted or absent, its bytes kept on
    disk), the name is free, and the evidence changes once; the card's row says it without a path or a tail."""
    from ouroboros.loop_delivery import delivery_evidence_fingerprint
    from ouroboros.outcomes import EXECUTION_OK, derive_loop_outcome

    record, log, registry, held = _ended_service(tmp_path, monkeypatch, backend)
    _refuse_metadata_of(monkeypatch, log)
    tool_ctx = ToolContext(repo_dir=tmp_path / "repo", drive_root=tmp_path, task_id="t-svc")
    ctx = _exit_context(tmp_path, tool_ctx)
    before = delivery_evidence_fingerprint(tool_ctx, ctx.llm_trace)

    assert service_finalization.finalize(ctx, ctx.llm_trace) is True
    [event] = ctx.llm_trace["verification_events"]
    assert event["kind"] == "services_stopped"
    [row] = event["services"]
    assert row["lifecycle"] == "stopped" and row["process_cleanup"]["exit_confirmed"] is True
    assert "stop_failed" not in row and row["artifact_output_failed"] is False
    final = row["log_finalization"]
    assert final == {"archive_outcome": "unconfirmed", "deleted_live_log": False, "live_log": "unknown",
                     "error_count": 2, "source_unavailable": "masked_source_not_verified"}
    assert held == [True], "the output registers after the log failed, with the name still held"
    assert record.service_id not in registry and record.finalizing is True
    assert log.read_text(encoding="utf-8") == "HEAD-LINE of the run\nLAST-LINE\n", "nothing deleted on an unknown"
    changed = delivery_evidence_fingerprint(tool_ctx, ctx.llm_trace)
    assert changed != before
    assert service_finalization.finalize(ctx, ctx.llm_trace) is False, "the cleanup's repeat finds nothing new"
    assert delivery_evidence_fingerprint(tool_ctx, ctx.llm_trace) == changed

    [shown] = [json.loads(line) for line in (tmp_path / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert shown["checkpoint_kind"] == "services_stopped" and shown["system_type"] == "task_checkpoint"
    assert shown["text"] == shown["content"] == (
        "Background service stopped: paper · at task end · log capture unconfirmed · output 1 saved: out.txt")
    assert not {"services", "error"} & set(shown)
    assert not any(word in json.dumps(shown) for word in ("HEAD-LINE", str(tmp_path), "injected metadata refusal"))
    assert derive_loop_outcome("The output is ready.", {}, ctx.llm_trace)["outcome_axes"]["execution"]["status"] == EXECUTION_OK


def test_a_confirmed_launch_changes_the_stop_evidence_only_once(tmp_path, monkeypatch):
    from ouroboros.loop_delivery import delivery_evidence_fingerprint
    from ouroboros.tools import services

    unread = {"deleted_live_log": False, "full_log_ref": {}, "tail": "", "unknown_live_log_path": "/data/paper.log",
              "errors": ["capture: PermissionError: refused", "inspect: PermissionError: refused"]}
    row = {"service_id": "t-svc:paper", "name": "paper", "task_id": "t-svc", "lifecycle": "stopped",
           "launch_id": "launch-one", "state": "exited", "log_finalization": unread}
    observed = iter([[dict(row)], [dict(row, artifact_output_failed=True, uptime_sec=2.0)]])
    monkeypatch.setattr(services, "stop_task_services", lambda _ctx: next(observed))
    tool_ctx = ToolContext(repo_dir=tmp_path / "repo", drive_root=tmp_path, task_id="t-svc")
    ctx = _exit_context(tmp_path, tool_ctx, event_queue=queue.Queue())
    assert service_finalization.finalize(ctx, ctx.llm_trace) is True
    once = delivery_evidence_fingerprint(tool_ctx, ctx.llm_trace)
    assert service_finalization.finalize(ctx, ctx.llm_trace) is False
    assert delivery_evidence_fingerprint(tool_ctx, ctx.llm_trace) == once
    assert [event["kind"] for event in ctx.llm_trace["verification_events"]] == ["services_stopped"]
    assert ctx.event_queue.qsize() == 1


def test_an_ordinary_success_and_a_kept_service_bind_only_the_recorded_evidence(tmp_path):
    """The new log fact is bound to the same ordinary and kept launch identities."""
    log = _finalized_log(tmp_path, "svc", "ready\n")
    stopped = {"service_id": "parent1:svc", "name": "svc", "task_id": "parent1", "lifecycle": "stopped",
               "state": "exited", "pid": 7, "pgid": 7, "uptime_sec": 3.5, "cwd": "/private", "log_path": "/private",
               "process_cleanup": {"exit_confirmed": True, "ended": [7], "reach": ["group"]},
               "artifact_output_failed": False, "log_finalization": log}
    kept = {"service_id": "parent1:dev", "name": "dev", "task_id": "parent1", "lifecycle": "kept",
            "state": "running", "pid": 8, "keep_alive": True, "uptime_sec": 9.0}
    trace = {"verification_events": [{"kind": "services_stopped", "services": [stopped]},
                                     {"kind": "services_kept", "services": [kept]}]}
    assert service_finalization.evidence(trace) == [
        {"kind": "services_stopped", "error": "", "services": [{
            "service_id": "parent1:svc", "name": "svc", "task_id": "parent1", "lifecycle": "stopped", "pid": 7,
            "artifact_output_failed": False, "log_finalization": log, "state": "exited",
            "process_cleanup": {"exit_confirmed": True}}]},
        {"kind": "services_kept", "error": "", "services": [{
            "service_id": "parent1:dev", "name": "dev", "task_id": "parent1", "lifecycle": "kept", "pid": 8,
            "state": "running"}]},
    ]
    assert log["deleted_live_log"] is True and log["source_scope"] == "full"
    assert set(log) == {"archive_outcome", "deleted_live_log", "source_ref", "source_scope"}
    assert [row["text"] for row in service_finalization.checkpoints(trace["verification_events"])] == [
        "Background services stopped", "Background services kept"]


def test_a_failed_declared_output_is_named_beside_an_unsaved_log_and_a_size_kept_log_is_no_loss():
    events = [{"kind": "services_stopped", "services": [
        {"name": "a", "artifact_output_failed": True, "log_finalization": {
            "deleted_live_log": False, "full_log_ref": {}, "tail": "x", "errors": ["capture: OSError: blob store full"],
            "retained_live_log_path": "/data/a.log"}},
        {"name": "b", "log_finalization": {"deleted_live_log": False, "full_log_ref": {}, "tail": "y", "errors": [],
                                           "full_log_omitted": "log exceeds 5000000 byte blob cap",
                                           "retained_live_log_path": "/data/b.log"}},
        {"name": "c", "log_finalization": {"deleted_live_log": True, "full_log_ref": {}, "tail": "",
                                           "errors": ["capture: OSError: blob store full"]}},  # it never existed
    ]}]
    [row] = service_finalization.checkpoints(events)
    assert row["text"] == "Background services stopped · log not saved: a · declared outputs not saved: a"
    many = [{"kind": "services_stopped", "services": [
        {"name": f"s{index}", "log_finalization": {"deleted_live_log": False, "full_log_ref": {}, "errors": ["x"],
                                                   "unknown_live_log_path": "/x"}} for index in range(5)]}]
    assert service_finalization.checkpoints(many)[0]["text"] == (
        "Background services stopped · log not saved: s0, s1, s2 and 2 more")
    [unavailable] = service_finalization.checkpoints([{"kind": "service_finalization_error", "services": [],
                                                       "error": "OSError: x"}])
    assert unavailable["text"] == "Background service outcomes could not be collected" and "OSError" not in str(unavailable)


# --- The row's way to the closed card: producer -> host -> the room's ordinary history read -> Chat ---------------

SERVICE_TASK = "svc-task"


def test_the_card_of_a_finished_task_reads_the_service_row_after_reload_and_reconnect(tmp_path, monkeypatch):
    """Two launches with the same name and stop time survive live emission and both real history reads."""
    import asyncio
    import os
    import shutil

    from ouroboros import utils
    from ouroboros.gateway import history
    from ouroboros.process_census import LAUNCH_SCOPE_PREFIX
    from ouroboros.store_import import ensure_current_store
    from supervisor.log_addressing import task_event_queue
    from tests.group_run import run_in_group

    monkeypatch.setattr("ouroboros.gateway.state._chat_activities_snapshot_safe", lambda *_, **__: [])
    started, stopped, answered = "2026-11-18T07:00:00Z", "2026-11-18T07:00:05Z", "2026-11-18T07:00:09Z"

    def write(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    write(tmp_path / "logs" / "tools.jsonl", [{"task_id": SERVICE_TASK, "ts": started, "type": fact,
                                               "tool": "start_service", "invocation_id": "start-1", "is_error": False}
                                              for fact in ("tool_call_started", "tool_call")])
    write(tmp_path / "task_results" / f"{SERVICE_TASK}.json", [{"_schema_version": 1, "task_id": SERVICE_TASK,
                                                              "status": "completed", "chat_id": 1}])
    write(tmp_path / "logs" / "chat.jsonl", [
        {"direction": "out", "task_id": SERVICE_TASK, "text": "The preview ran; its log could not be read.",
         "chat_id": 1, "ts": answered},
        {"direction": "system", "type": "task_summary", "task_id": SERVICE_TASK, "text": "Preview checked.",
         "chat_id": 1, "ts": answered, "tool_calls": 1, "tool_errors": 0}])
    ensure_current_store(tmp_path)

    # The producer: the real stop, collector, record and checkpoint writer, addressed by the task's own queue.
    record, log, registry, _ = _ended_service(tmp_path, monkeypatch, "host", task=SERVICE_TASK)
    record.proc.pid = 41007
    record.scope_marker = LAUNCH_SCOPE_PREFIX + "a" * 32
    _refuse_metadata_of(monkeypatch, log)
    monkeypatch.setattr(utils, "utc_now_iso", lambda: stopped)
    bus: "queue.Queue" = queue.Queue()
    tool_ctx = ToolContext(repo_dir=tmp_path / "repo", drive_root=tmp_path, task_id=SERVICE_TASK)
    ctx = _exit_context(tmp_path, tool_ctx, event_queue=task_event_queue(bus, {
        "id": SERVICE_TASK, "type": "task", "chat_id": 1, "root_task_id": SERVICE_TASK}))
    assert service_finalization.finalize(ctx, ctx.llm_trace)
    registry[record.service_id] = replace(record, finalizing=False, scope_marker=LAUNCH_SCOPE_PREFIX + "b" * 32)
    assert service_finalization.finalize(ctx, ctx.llm_trace)
    assert not service_finalization.finalize(ctx, ctx.llm_trace)
    frames = [bus.get_nowait()["data"] for _ in range(2)]
    assert bus.empty() and all(f["checkpoint_kind"] == "services_stopped" and f["chat_id"] == 1 for f in frames)
    assert len({f["card_row_id"] for f in frames}) == 2

    def room():
        response = asyncio.run(history.make_chat_history_endpoint(tmp_path)(
            SimpleNamespace(query_params={"chat_id": "1"})))
        assert response.status_code == 200
        return response.body

    first, again = room(), room()
    assert json.loads(first)["messages"] == json.loads(again)["messages"]
    rows = [message for message in json.loads(first)["messages"] if message.get("system_type") == "task_checkpoint"]
    assert [row["card_row_id"] for row in rows] == [f["card_row_id"] for f in frames]
    assert all((r["task_id"], r["checkpoint_kind"], r["text"], r["ts"]) == (
        SERVICE_TASK, "services_stopped", frame["text"], stopped) for r, frame in zip(rows, frames, strict=True))
    assert rows[0]["text"] != rows[1]["text"] and all("log capture unconfirmed" in r["text"] for r in rows)
    body = first.decode("utf-8")
    for private in ("HEAD-LINE", "LAST-LINE", str(tmp_path), "injected metadata refusal", "unknown_live_log_path"):
        assert private not in body, private
    produced = tmp_path / "service-history.json"
    produced.write_bytes(first)
    node = shutil.which("node")
    assert node, "Node is required for the real Chat consumer proof"
    result = run_in_group([node, "--test", "--test-reporter=tap",
                           "--test-name-pattern=a finished task reads its service row after reload and reconnect",
                           "web/tests/chat_activity_block.test.js"], cwd=pathlib.Path(__file__).resolve().parents[1],
                          env={**os.environ, "OURO_TEST_SERVICE_HISTORY": str(produced)}, timeout=60)
    assert result.returncode == 0 and "# pass 1" in result.stdout + result.stderr, result.stdout + result.stderr
