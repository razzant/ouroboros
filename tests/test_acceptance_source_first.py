"""#1329: a retrieving acceptance reviewer gets a delivery it can actually use.

Before this change the packet ceiling — sized for reviewers WITHOUT tools —
refused every row once the owner-requirement core overflowed, and a native row
whose first send outgrew its window was refused at $0. Now a retrieving row
gets a compact first send plus the complete packet as ONE exact source at an
address its own reader resolves; a row with no such address is a typed $0
refusal of that row alone. These tests drive the real panel, substrate and
native inspection registry (a scripted model; the Claudexor surface is the
offline fake) and check the source bytes the reviewer was pointed at.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from ouroboros import loop as loop_mod
from tests.test_acceptance_delivery import (
    _ACCEPTANCE_PACKET,
    _CLEAN_VERDICT,
    _ROW_API,
    _ROW_NATIVE,
    _ROW_SESSION,
    _acceptance_ctx,
    _EpisodeLLM,
    _fake_session,
    _offline_env,
    _real_panel,
    _roots,
    _tool_call,
)

_OVERFLOW = {**_ACCEPTANCE_PACKET, "__immutable_core_overflow__": {"reason": "owner requirements exceed the budget"}}
_SOURCE_PATH = re.compile(r'path="(source_handles/context_checkpoints/acceptance-packet-[0-9a-f]{64}\.json)"')


class _ReadingLLM(_EpisodeLLM):
    """A native reviewer that opens the packet source its work order names, then answers."""

    def _reply(self, kwargs):
        self.calls.append(dict(kwargs))
        if "tools" in kwargs and len([c for c in self.calls if "tools" in c]) == 1:
            prompt = kwargs["messages"][-1]["content"]
            path = _SOURCE_PATH.search(prompt).group(1)
            reply = {"tool_calls": [_tool_call("read_file", {"root": "artifact_store", "path": path})]}
        else:
            reply = {"content": json.dumps(_CLEAN_VERDICT)}
        return reply, {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.0}


def _panel(tmp_path, evidence, workspace, governance):
    ctx = _acceptance_ctx(tmp_path, evidence=evidence, repo_dir=str(governance),
                          workspace_root=str(workspace), workspace_mode="project")
    return loop_mod._execute_task_acceptance_panel(ctx)


def test_overflow_refuses_the_packet_row_and_the_native_row_reads_the_exact_source(monkeypatch, tmp_path):
    from ouroboros.artifacts import read_actor_source_bytes

    _offline_env(monkeypatch, _ROW_API, _ROW_NATIVE)
    governance, workspace = _roots(tmp_path)
    llm = _ReadingLLM(tmp_path, [])
    seen = _real_panel(monkeypatch, llm)
    result = _panel(tmp_path, dict(_OVERFLOW), workspace, governance)
    by_id = {actor["slot_id"]: actor for actor in result.actors}
    assert by_id["t_api"]["status"] == "not_dispatched" and "overflow" in str(by_id["t_api"]["error"])
    assert by_id["t_actor"]["parsed"]["verdict"] == "PASS"

    (request,) = seen
    delivery = request.slot_source_delivery["t_actor"]
    assert delivery["status"] == "paged" and "overflowed" in delivery["reason"]
    stored = read_actor_source_bytes(tmp_path, "root-delivery", {"kind": "task_source", **delivery["source"]})
    body = json.loads(stored)
    assert body["evidence"] == request.evidence and body["subject"] == request.subject
    order = request.slot_session_tasks["t_actor"]
    assert "TRAJECTORY-RESULT-3-passed" not in order and "MANDATORY" in order
    assert delivery["source"]["sha256"] in order and "- evidence:" in order
    # The real inspection registry resolved artifact_store and returned the FULL packet.
    native = [call for call in llm.calls if "tools" in call]
    assert len(native) == 2
    tool_result = json.dumps([m for m in native[1]["messages"] if m.get("role") == "tool"])
    assert "TRAJECTORY-RESULT-3-passed" in tool_result and "PREVIEW-BYTES-OF-THE-ARTIFACT" in tool_result


def test_a_native_row_over_its_own_window_gets_the_compact_send_without_overflow(monkeypatch, tmp_path):
    from ouroboros import review_native_episode

    _offline_env(monkeypatch, _ROW_NATIVE)
    governance, workspace = _roots(tmp_path)
    big = {**_ACCEPTANCE_PACKET, "repo_diff": "diff --git a/x b/x\n" + "+line\n" * 30_000}
    monkeypatch.setattr(review_native_episode, "native_episode_transcript_bound", lambda *_a, **_k: 120_000)
    llm = _EpisodeLLM(tmp_path, [{"content": json.dumps(_CLEAN_VERDICT)}])
    seen = _real_panel(monkeypatch, llm)
    result = _panel(tmp_path, big, workspace, governance)
    assert result.actors[0]["parsed"]["verdict"] == "PASS"
    (request,) = seen
    delivery = request.slot_source_delivery["t_actor"]
    assert delivery["status"] == "paged" and delivery["reason"] == "the whole first send does not fit this row"
    assert delivery["first_send_chars"] < delivery["first_send_ceiling"] < delivery["inline_first_send_chars"]
    assert "+line\n+line" not in json.dumps(llm.calls[0]["messages"])


def test_a_small_panel_keeps_its_inline_delivery(monkeypatch, tmp_path):
    _offline_env(monkeypatch, _ROW_NATIVE)
    governance, workspace = _roots(tmp_path)
    llm = _EpisodeLLM(tmp_path, [{"content": json.dumps(_CLEAN_VERDICT)}])
    seen = _real_panel(monkeypatch, llm)
    _panel(tmp_path, dict(_ACCEPTANCE_PACKET), workspace, governance)
    (request,) = seen
    assert request.slot_source_delivery["t_actor"]["status"] == "inline"
    assert "verification_receipts[0]" in request.slot_session_tasks["t_actor"]
    assert list(tmp_path.rglob("acceptance-packet-*"))  # identity is durable before any send, even inline
    assert request.slot_source_delivery["t_actor"]["source"]["sha256"]


def test_an_unwritable_source_refuses_the_row_typed_and_sends_nothing(monkeypatch, tmp_path):
    from ouroboros import artifacts

    _offline_env(monkeypatch, _ROW_API, _ROW_NATIVE)
    governance, workspace = _roots(tmp_path)
    monkeypatch.setattr(artifacts, "store_actor_source_bytes",
                        lambda *_a, **_k: (_ for _ in ()).throw(OSError("publication fenced")))
    llm = _EpisodeLLM(tmp_path, [])
    _real_panel(monkeypatch, llm)
    result = _panel(tmp_path, dict(_OVERFLOW), workspace, governance)
    by_id = {actor["slot_id"]: actor for actor in result.actors}
    assert by_id["t_actor"]["status"] == "not_dispatched"
    assert "review_source_closure_unavailable" in by_id["t_actor"]["error"] and "publication fenced" in by_id["t_actor"]["error"]
    assert by_id["t_api"]["operation_state"] == "not_dispatched"
    assert llm.calls == []


@pytest.mark.serial
@pytest.mark.parametrize("git_workspace", [False, True])
def test_session_consumer_reads_exact_retained_file_without_attachments(monkeypatch, tmp_path, git_workspace):
    """Real panel/coordinator/executor; offline engine physically reads the named file.

    This is consumer wiring evidence, not a subscription-harness read receipt.
    The opt-in canary exercises that last boundary with an actual session.
    """
    # Force JSON path escaping on POSIX too; on Windows this is a separator.
    tmp_path = tmp_path / "retained\\source-ñ"
    tmp_path.mkdir(parents=True)
    fake = _fake_session(monkeypatch)  # no attachmentInputs advertised
    original_start = fake.start_run
    observed = []

    def start(self, wire, **kw):
        from ouroboros.delegate_custody import START_REQUESTED
        from ouroboros.observability import read_blob_ref, read_call_payload

        path = json.loads(re.search(r'absolute path ("(?:[^"\\]|\\.)*") with', wire["prompt"]).group(1))
        raw = Path(path).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        assert digest in wire["prompt"]
        assert not Path(path).is_relative_to(workspace)
        assert wire["scope"] == {"kind": "project", "root": str(workspace)}
        assert (Path(wire["scope"]["root"]) / "greeting.txt").read_text(encoding="utf-8") == "hello native reviewer\n"
        assert wire["access"] == "readonly" and wire["mode"] == "ask"
        assert "attachments" not in wire
        assert "TRAJECTORY-RESULT-3-passed" not in wire["prompt"]
        # Before POST, BOTH coordinator prompt and invocation custody carry the source identity.
        rows = [json.loads(line) for line in (tmp_path / "logs/events.jsonl").read_text(encoding="utf-8").splitlines()]
        (invocation,) = [row for row in rows if row.get("type") == START_REQUESTED
                         and row.get("task_id") == "root-delivery" and row.get("slot_id") == "t_sess"]
        assert read_blob_ref(tmp_path, invocation["request_ref"]) == wire
        _, prompt, _ = read_call_payload(tmp_path, task_id="root-delivery",
                                         call_id=invocation["operation_id"] + "_prompt")
        assert prompt["request"]["task_id"] == "root-delivery" and prompt["slot"]["slot_id"] == "t_sess"
        deliveries = [prompt["request"]["slot_source_delivery"]["t_sess"], invocation["review_source_delivery"]]
        for delivery in deliveries:
            assert delivery["source_path"] == path
            assert delivery["custody_root"] == str(tmp_path)
            assert delivery["custody_source"]["sha256"] == digest
            assert delivery["custody_source"]["size"] == len(raw)
            assert delivery["custody_source"] == delivery["source"]
            assert delivery["reader"] == "filesystem" and delivery["external_ref_closure"] == "retained"
        assert deliveries[0]["reader_root"] == deliveries[1]["reader_root"]
        observed.append((raw, wire))
        return original_start(self, wire, **kw)

    monkeypatch.setattr(fake, "start_run", start)
    _offline_env(monkeypatch, _ROW_SESSION)
    governance, workspace = _roots(tmp_path)
    if git_workspace:
        subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
    seen = _real_panel(monkeypatch, _EpisodeLLM(tmp_path, []))
    result = _panel(tmp_path, dict(_OVERFLOW), workspace, governance)
    assert result.aggregate_signal == "PASS", json.dumps(result.actors, indent=2)
    (request,) = seen
    delivery = request.slot_source_delivery["t_sess"]
    assert delivery["status"] == "paged"
    (raw, wire), = observed
    body = json.loads(raw)
    for key in ("goal", "scope", "checklist", "subject", "evidence_refs", "evidence"):
        assert body[key] == getattr(request, key)
    assert delivery["source"] == delivery["custody_source"]
    assert delivery["source_path"] == str(tmp_path / "task_results" / "artifacts" / "root-delivery" / delivery["source"]["path"])
    assert delivery["first_send_chars"] == len(json.dumps(wire, ensure_ascii=False)) < delivery["first_send_ceiling"]
    assert delivery["first_send_bytes"] == len(json.dumps(wire, ensure_ascii=False).encode("utf-8"))
    (tmp_path / "consumer-evidence.json").write_text(json.dumps({"kind": "offline_real_consumer",
        "live_session": False, "source_delivery": delivery, "wire_request": wire}, ensure_ascii=False, indent=2), encoding="utf-8")
    assert not (workspace / ".review-drive").exists() and not (workspace / ".gitignore").exists()
    if git_workspace:
        assert subprocess.check_output(["git", "status", "--porcelain"], cwd=workspace, text=True) == "?? greeting.txt\n"


def test_complete_compact_send_must_fit_even_without_packet_overflow(monkeypatch, tmp_path):
    from ouroboros.tools import review_brief_coupling

    fake = _fake_session(monkeypatch)
    _offline_env(monkeypatch, _ROW_SESSION)
    governance, workspace = _roots(tmp_path)
    monkeypatch.setattr(review_brief_coupling, "SESSION_INLINE_DIFF_CEILING_CHARS", 1_000)
    _real_panel(monkeypatch, _EpisodeLLM(tmp_path, []))
    result = _panel(tmp_path, dict(_ACCEPTANCE_PACKET), workspace, governance)
    assert result.aggregate_signal != "PASS"
    assert "complete compact first send" in result.actors[0]["error"]
    assert all(not instance.start_requests for instance in fake.instances)



def test_wide_scope_checklist_index_and_refs_all_leave_the_compact_first_send(monkeypatch, tmp_path):
    from ouroboros import review_native_episode
    from ouroboros.acceptance_retrieving import acceptance_retrieving_work_order
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.review_records import ReviewRequest, ReviewSlot

    request = ReviewRequest(surface="task_acceptance", task_id="task", goal="goal" * 40_000,
                            scope="scope" * 40_000, checklist="check" * 40_000,
                            subject="subject" * 40_000, evidence_refs=["ref" * 40_000],
                            evidence={"key" * 50_000: "value"})
    slot = ReviewSlot(slot_id="native", model="m", native_retrieval_override=True)
    monkeypatch.setattr(review_native_episode, "native_episode_transcript_bound", lambda *a, **k: 120_000)
    acceptance_retrieving_work_order(request, [slot], session_root=str(tmp_path), data_root=tmp_path)
    delivery = request.slot_source_delivery[slot.slot_id]
    assert delivery["status"] == "paged"
    assert delivery["first_send_chars"] < delivery["first_send_ceiling"] < delivery["inline_first_send_chars"]
    source = json.loads(read_actor_source_bytes(tmp_path, "task", delivery["source"]))
    for key in ("goal", "scope", "checklist", "subject", "evidence_refs", "evidence"):
        assert source[key] == getattr(request, key)


def test_unwritable_saved_request_sends_no_reviewer(monkeypatch, tmp_path):
    from ouroboros import review_substrate

    _offline_env(monkeypatch, _ROW_NATIVE)
    governance, workspace = _roots(tmp_path)
    llm = _EpisodeLLM(tmp_path, [])
    _real_panel(monkeypatch, llm)
    monkeypatch.setattr(review_substrate, "persist_call", lambda *a, **k: (_ for _ in ()).throw(OSError("disk")))
    result = _panel(tmp_path, dict(_OVERFLOW), workspace, governance)
    assert result.aggregate_signal != "PASS" and not llm.calls
    assert list(tmp_path.rglob("acceptance-packet-*"))  # exact source remains for recovery


def _prepared_request(tmp_path, *, session=True, huge=False):
    from ouroboros.acceptance_retrieving import acceptance_retrieving_work_order
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_records import ReviewRequest, ReviewSlot
    from ouroboros.artifacts import task_artifact_dir_path
    from ouroboros.task_results import write_task_result

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "greeting.txt").write_text("workspace-marker-ñ", encoding="utf-8")
    author, canonical = tmp_path / "author-drive", tmp_path / "canonical"
    write_task_result(author, "source-reader", "running", result="original source-reader task")
    artifact = task_artifact_dir_path(author, "source-reader", create=True) / "report/summary.md"
    artifact.parent.mkdir(exist_ok=True)
    artifact.write_text(_ACCEPTANCE_PACKET["artifacts"][0]["preview"], encoding="utf-8")
    slot = ReviewSlot(slot_id="reader", model="fake-small", timeout_sec=30,
                      route=ReviewRouteKind.AGENT_SESSION if session else ReviewRouteKind.API_CHAT,
                      session_target="fake-review=fake-small" if session else "",
                      native_retrieval_override=not session)
    request = ReviewRequest(surface="task_acceptance", task_id="source-reader",
                            goal="goal" * (40_000 if huge else 1),
                            scope="scope" * (40_000 if huge else 1),
                            checklist="checklist" * (40_000 if huge else 1),
                            subject="subject" * (40_000 if huge else 1),
                            evidence_refs=["refs" * (40_000 if huge else 1)],
                            evidence=dict(_OVERFLOW), policy={"min_successful_slots": 1})
    acceptance_retrieving_work_order(request, [slot], session_root=str(workspace), data_root=author)
    return request, slot, author, canonical, workspace


def test_huge_fields_session_packet_and_named_closure_survive_cleanup(monkeypatch, tmp_path):
    from ouroboros.review_substrate import run_review_request

    fake = _fake_session(monkeypatch)
    request, slot, author, canonical, workspace = _prepared_request(tmp_path, huge=True)
    original_start = fake.start_run

    def start(self, wire, **kw):
        delivery = request.slot_source_delivery[slot.slot_id]
        # Delete ONLY this test's execution drive, after the real consumer retained the source.
        shutil.rmtree(author)
        raw = Path(delivery["source_path"]).read_bytes()
        assert Path(delivery["source_path"]).is_relative_to(canonical)
        assert json.dumps(delivery["source_path"], ensure_ascii=False) in wire["prompt"]
        body = json.loads(raw)
        for key in ("goal", "scope", "checklist", "subject", "evidence_refs", "evidence"):
            assert body[key] == getattr(request, key)
        assert wire["scope"]["root"] == str(workspace)
        assert "goalgoal" not in wire["prompt"] and "checklistchecklist" not in wire["prompt"]
        assert hashlib.sha256(raw).hexdigest() == delivery["custody_source"]["sha256"]
        return original_start(self, wire, **kw)

    monkeypatch.setattr(fake, "start_run", start)
    result = run_review_request(request, slots=[slot], drive_root=canonical, llm=_EpisodeLLM(canonical, []))
    assert result.aggregate_signal == "PASS", result.actors
    delivery = request.slot_source_delivery[slot.slot_id]
    assert delivery["first_send_chars"] < delivery["first_send_ceiling"] < delivery["inline_first_send_chars"]
    (wire,) = fake.instances[0].start_requests
    assert delivery["first_send_chars"] == len(json.dumps(wire, ensure_ascii=False))
    (tmp_path / "consumer-evidence.json").write_text(json.dumps({"kind": "offline_real_consumer",
        "live_session": False, "source_delivery": delivery, "wire_request": wire}, ensure_ascii=False, indent=2), encoding="utf-8")
    # The landed closure owner retained named snapshots before dispatch; packet
    # custody is now revalidated on that root, not the deleted original drive.
    from ouroboros.acceptance_retrieving import retain_review_source
    before = json.dumps(request.slot_source_delivery, sort_keys=True)
    retain_review_source(request, slot.slot_id, canonical)
    assert json.dumps(request.slot_source_delivery, sort_keys=True) == before
    assert request.policy["review_source_closure"]["read_root"] == delivery["reader_root"]
    assert delivery["external_ref_closure"] == "retained"
    assert not author.exists()


@pytest.mark.parametrize("damage", ["missing", "changed"])
def test_session_consumer_refuses_unavailable_canonical_bytes_without_start(monkeypatch, tmp_path, damage):
    from ouroboros import acceptance_retrieving
    from ouroboros.review_substrate import run_review_request

    fake = _fake_session(monkeypatch)
    request, slot, author, canonical, _workspace = _prepared_request(tmp_path)
    retain = acceptance_retrieving.retain_review_source

    def damaged(*args):
        retain(*args)
        path = Path(request.slot_source_delivery[slot.slot_id]["source_path"])
        if damage == "missing":
            path.unlink()
        else:
            path.write_bytes(b"changed source")

    monkeypatch.setattr(acceptance_retrieving, "retain_review_source", damaged)
    result = run_review_request(request, slots=[slot], drive_root=canonical, llm=_EpisodeLLM(canonical, []))
    assert result.aggregate_signal != "PASS"
    assert "degraded_source_unreachable" in result.actors[0]["error"]
    assert result.actors[0]["operation_state"] == "not_dispatched"
    assert not result.actors[0].get("late_result_pending")
    assert all(not instance.start_requests for instance in fake.instances)
    # A surviving execution copy cannot silently replace the missing canonical source.
    assert list(author.rglob("acceptance-packet-*"))


@pytest.mark.parametrize("fixture_encoding", ["utf-8", "cp1252"])
def test_native_consumer_reports_lost_retained_reader_without_switching_to_packet_custody(monkeypatch, tmp_path, fixture_encoding):
    from ouroboros.review_substrate import run_review_request

    write_text = Path.write_text

    def fixture_write(path, data, encoding=None, **kwargs):
        # Model the Windows default ONLY for the fixture's greeting write.
        if path == tmp_path / "workspace/greeting.txt" and encoding is None:
            encoding = fixture_encoding
        return write_text(path, data, encoding=encoding, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "write_text", fixture_write)
        request, slot, author, canonical, workspace = _prepared_request(tmp_path, session=False)

    class ReadingAfterCleanup(_EpisodeLLM):
        def _reply(self, kwargs):
            self.calls.append(dict(kwargs))
            if len(self.calls) == 1:
                from ouroboros.review_native_episode import _wire_size
                self.first_send_chars = _wire_size(kwargs["messages"], kwargs["tools"])
                shutil.rmtree(Path(request.policy["native_data_root"]))
                path = _SOURCE_PATH.search(kwargs["messages"][-1]["content"]).group(1)
                reply = {"tool_calls": [
                    _tool_call("read_file", {"root": "artifact_store", "path": path}, "source"),
                    _tool_call("read_file", {"path": "greeting.txt"}, "workspace")]}
            else:
                returned = json.dumps([m for m in kwargs["messages"] if m.get("role") == "tool"], ensure_ascii=False)
                assert "workspace-marker-ñ" in returned
                assert "TRAJECTORY-RESULT-3-passed" not in returned
                assert "PREVIEW-BYTES-OF-THE-ARTIFACT" not in returned
                self.returned = returned
                reply = {"content": json.dumps({**_CLEAN_VERDICT, "verdict": "DEGRADED",
                    "outcome_tier": "unknown", "criteria_used": [], "summary": "Original reader source unavailable"})}
            return reply, {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.0}

    llm = ReadingAfterCleanup(canonical, [])
    result = run_review_request(request, slots=[slot], drive_root=canonical, llm=llm)
    assert result.aggregate_signal == "DEGRADED", result.actors
    assert result.actors[0].get("parsed"), result.actors
    assert result.actors[0]["parsed"]["summary"] == "Original reader source unavailable"
    assert "⚠" in llm.returned or "error" in llm.returned.lower(), llm.returned
    assert request.session_root == str(workspace)
    assert llm.first_send_chars == request.slot_source_delivery[slot.slot_id]["first_send_chars"]
    receipts = result.actors[0]["usage"]["native_tool_receipts"]
    assert not any(r.get("opened_root") == "artifact_store" and r.get("eof") for r in receipts)
    assert request.policy["native_data_root"] != str(author)
    # Episode history may recreate its output directory; it must not recreate
    # the vanished input packet or borrow the separate canonical packet copy.
    from ouroboros.artifacts import task_artifact_dir_path
    assert not (task_artifact_dir_path(request.policy["native_data_root"], request.task_id)
                / request.slot_source_delivery[slot.slot_id]["source"]["path"]).exists()
    from ouroboros.artifacts import read_actor_source_bytes
    # The packet itself survived, but was not silently substituted for the lost reader plane.
    assert read_actor_source_bytes(canonical, request.task_id, request.slot_source_delivery[slot.slot_id]["custody_source"])


def test_session_rechecks_actual_request_after_packet_retention(monkeypatch, tmp_path):
    from ouroboros.review_substrate import run_review_request
    from ouroboros.tools import review_brief_coupling

    fake = _fake_session(monkeypatch)
    request, slot, _author, canonical, _workspace = _prepared_request(tmp_path)
    # Lower the available window only AFTER the measured preparation and
    # canonical retention. Actual-wire sizing must still prevent a POST.
    from ouroboros import acceptance_retrieving
    retain = acceptance_retrieving.retain_review_source
    def smaller_dispatch_window(*args):
        retain(*args)
        monkeypatch.setattr(review_brief_coupling, "SESSION_INLINE_DIFF_CEILING_CHARS", 1000)
    monkeypatch.setattr(acceptance_retrieving, "retain_review_source", smaller_dispatch_window)
    result = run_review_request(request, slots=[slot], drive_root=canonical, llm=_EpisodeLLM(canonical, []))
    assert result.aggregate_signal != "PASS"
    assert "Complete source-first request exceeds session send bound" in result.actors[0]["error"]
    delivery = request.slot_source_delivery[slot.slot_id]
    assert delivery["first_send_chars"] >= 1000
    assert Path(delivery["source_path"]).is_file()  # retained even though no dispatch fit
    assert all(not instance.start_requests for instance in fake.instances)


@pytest.mark.parametrize("session", [False, True])
def test_consumer_refuses_source_lost_before_retention(monkeypatch, tmp_path, session):
    from ouroboros.review_substrate import run_review_request

    fake = _fake_session(monkeypatch)
    request, slot, author, canonical, _workspace = _prepared_request(tmp_path, session=session)
    shutil.rmtree(author)
    llm = _EpisodeLLM(canonical, [])
    result = run_review_request(request, slots=[slot], drive_root=canonical, llm=llm)
    assert result.aggregate_signal != "PASS"
    assert "original_reader_root_unavailable" in result.actors[0]["error"]
    assert not llm.calls and all(not instance.start_requests for instance in fake.instances)


@pytest.mark.parametrize('fact,expected', [
    ('none', 'not_dispatched'), ('reserved', 'not_dispatched'), ('released', 'not_dispatched'),
    ('settled', 'settled'), ('dispatched', 'custody_lost'), ('unresolved', 'custody_lost'),
    ('malformed', 'custody_lost'), ('run', 'settled'), ('started', 'settled'),
    ('pending', 'custody_lost'), ('native_round', 'settled'),
])
def test_source_refusal_preserves_stronger_physical_custody(monkeypatch, fact, expected):
    from types import SimpleNamespace
    from ouroboros.review_custody import _ReviewAttemptHistory, _review_exception_projection
    from ouroboros.review_execution import ReviewRouteUnavailable

    # Each real slot starts a fresh capture scope; no prior test's send belongs here.
    monkeypatch.setattr('ouroboros.usage_accounting.last_physical_attempt_capture', lambda: None)
    error = ReviewRouteUnavailable('exact source missing', code='degraded_source_unreachable')
    if fact in {'reserved', 'released', 'settled', 'dispatched', 'unresolved', 'malformed'}:
        error.physical_attempt_capture = SimpleNamespace(state=fact)
    elif fact == 'run':
        error.delegated_run_id = 'synthetic-run'
    elif fact == 'started':
        error.delegated_run_started = True
    history = _ReviewAttemptHistory()
    history.observe(error)
    custody = {'native_rounds': 1} if fact == 'native_round' else {}
    retry = {'pending_invocation_id': 'synthetic-pending'} if fact == 'pending' else {}
    assert _review_exception_projection(error, custody, history, retry)[3] == expected
