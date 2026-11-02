"""#1657: real producers through the registry, plus the downstream trace contract.

External effects are replaced at their boundary. No provider, git mutation, real
service process, skill process or owner message is submitted by this module.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ouroboros.tools.registry import ToolEntry, ToolRegistry
from ouroboros.tools.tool_result import TOOL_CODE_SPECS, LegacyTextResultAdapter


@pytest.fixture
def invoke(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path / "data")

    serial = iter(range(1000))

    def call(handler, **kwargs):
        # The actual handler runs through the real sidecar/adaptation boundary.
        # A fixture name keeps unrelated route/authority gates out of this test.
        name = f"census_builtin_{next(serial)}"
        registry.register(ToolEntry(name, {
            "name": name, "description": "producer consumer test",
            "parameters": {"type": "object", "properties": {name: {} for name in kwargs}},
        }, handler))
        return registry.execute_result(name, kwargs)

    call.ctx = registry._ctx
    call.registered = registry.execute_result
    return call


@pytest.mark.parametrize(("identifier", "code"), [
    ("SEND_LINKS_NO_CHAT", "LEGACY_UNAVAILABLE"),
    ("SEND_LINKS_TOO_MANY", "TOOL_ARG_ERROR"),
    ("ESCALATE_PARENT_SETTLED", "LEGACY_UNAVAILABLE"),
    ("ESCALATE_UNWRITTEN", "TOOL_ERROR"),
    ("FILE_TOO_LARGE", "TOOL_ARG_ERROR"),
    ("SKILL_EXEC_EXTENSION", "TOOL_ARG_ERROR"),
    ("SKILL_EXEC_GRANT_REQUIRED", "LEGACY_BLOCKED"),
    ("SKILL_EXEC_ARGV_TOO_LARGE", "TOOL_ARG_ERROR"),
    ("SKILL_EXEC_OVERFLOW", "TOOL_ERROR"),
    ("SKILL_REPAIR_STALE", "LEGACY_BLOCKED"),
    ("SERVICE_ALREADY_RUNNING", "LEGACY_BLOCKED"),
    ("CHERRY_PICK_CONFLICT", "TOOL_ERROR"),
    ("RESTORE_PARTIAL", "TOOL_ERROR"),
    ("EVOLUTION_AUTHORITY_REVOKED", "LEGACY_BLOCKED"),
    ("EVOLUTION_COMMIT_ORPHANED", "TOOL_ERROR"),
    ("EVOLUTION_PUBLICATION_STOPPED", "LEGACY_BLOCKED"),
])
def test_confirmed_identifiers_reach_existing_trace_consumers(identifier, code):
    from ouroboros._outcome_tool_errors import _OK_TOOL_STATUSES
    from ouroboros.loop_tool_execution import _typed_execution_failure, _typed_result_metadata

    text = f"⚠️ {identifier}: source-confirmed result\nretained detail"
    result = LegacyTextResultAdapter.from_text("fixture_builtin", text)
    assert (result.code, result.status, result.text) == (code, TOOL_CODE_SPECS[code].status, text)
    assert dict(result.meta) == {}  # text never proves no effect or settles custody
    failed = _typed_execution_failure(True, result)
    facts = _typed_result_metadata("fixture_builtin", text, failed, result)
    assert failed and facts["status"] not in _OK_TOOL_STATUSES
    assert facts["status"] == TOOL_CODE_SPECS[code].outcome_bucket
    assert facts["tool_result_code"] == code


@pytest.mark.parametrize("marker", [
    "LIST_FILES_NOT_FOUND", "FILE_NOT_FOUND", "SEARCH_NOT_FOUND", "QUERY_CODE_TRUNCATED",
    "CHILD_RESULT_DISPOSITION_PARTIAL", "SERVICE_NOT_FOUND", "GIT_NO_CHANGES",
    "UNRECOGNIZED_REFUSED", "UNRECOGNIZED_TOO_MANY", "UNRECOGNIZED_OVERFLOW",
    # Coordinated review branches publish natively; their text is not a global verdict.
    "REVIEW_PENDING", "REVIEW_CUSTODY_LOST", "IDENTICAL_DIFF_REFUSED", "REVIEW_CYCLES_EXHAUSTED",
])
def test_warnings_and_branch_typed_review_markers_keep_their_meaning(marker):
    result = LegacyTextResultAdapter.from_text("builtin", f"⚠️ {marker}: retained")
    assert (result.status, result.code, dict(result.meta)) == ("ok", "LEGACY_WARNING", {})


def test_exact_refusal_identifiers_do_not_scan_payload_or_external_prose():
    text = "output was delivered\n⚠️ SKILL_EXEC_OVERFLOW: quoted payload"
    assert LegacyTextResultAdapter.from_text("builtin", text).code == "OK"
    assert LegacyTextResultAdapter.from_text("ext_4_demo_call", "⚠️ SKILL_EXEC_OVERFLOW: prose").code == "LEGACY_UNTYPED"
    assert LegacyTextResultAdapter.from_text("builtin", "⚠️ SKILL_EXEC_OVERFLOWED: unrelated").code == "LEGACY_WARNING"


@pytest.mark.parametrize("case", ["no_chat", "too_many", "success"])
def test_send_links_actual_validation_and_delivery_boundary(invoke, monkeypatch, case):
    from ouroboros.tools import core_artifacts as producer
    sent = []
    monkeypatch.setattr("ouroboros.tools.owner_delivery.deliver_owner_event", lambda ctx, event: sent.append(event) or "live")
    invoke.ctx.current_chat_id = None if case == "no_chat" else 123
    links = [{"label": "Source", "url": "https://example.com"}]
    if case == "too_many":
        links *= producer._MAX_LINK_ACTIONS + 1
    result = invoke(producer._send_links, links=links)
    expected = {"no_chat": "LEGACY_UNAVAILABLE", "too_many": "TOOL_ARG_ERROR", "success": "OK"}[case]
    assert result.code == expected
    assert len(sent) == (1 if case == "success" else 0)
    if case == "too_many":
        assert result.text.startswith("⚠️ SEND_LINKS_TOO_MANY:")
        assert result.text.endswith("No links were sent.")


@pytest.mark.parametrize("parent_alive", [False, True])
def test_escalation_dead_end_differs_from_failed_persistence(invoke, monkeypatch, parent_alive):
    invoke.ctx.task_id = "child"
    invoke.ctx.task_metadata = {"parent_task_id": "parent", "delegation_role": "subagent"}
    monkeypatch.setattr("ouroboros.task_status.load_effective_task_result", lambda *a: {"status": "running" if parent_alive else "completed"})
    monkeypatch.setattr("ouroboros.cancel_intents.cancel_pending", lambda *a: False)
    attempted = []
    monkeypatch.setattr("ouroboros.owner_mailbox.write_task_message", lambda *a, **k: attempted.append(k) or False)
    result = invoke.registered("escalate", {
        "question": "Which option?", "assumption": "Continue the independent work.",
    })
    assert result.code == ("TOOL_ERROR" if parent_alive else "LEGACY_UNAVAILABLE")
    assert len(attempted) == int(parent_alive)
    assert "operation_outcome" not in result.meta


@pytest.mark.parametrize("executor", [False, True])
def test_missing_service_result_depends_on_requested_operation(invoke, monkeypatch, executor):
    from ouroboros.tools import services
    monkeypatch.setattr(services, "_SERVICES", {})
    monkeypatch.setattr(services, "executor_ref_from_ctx", lambda ctx: object() if executor else None)
    for name in ("executor_service_logs", "executor_service_status", "executor_stop_service"):
        monkeypatch.setattr(services, name, lambda *a: None)
    logs = invoke(services._service_logs, name="absent")
    assert (logs.status, logs.code) == ("unavailable", "LEGACY_UNAVAILABLE")
    assert logs.text == "⚠️ SERVICE_NOT_FOUND: absent"
    assert LegacyTextResultAdapter.from_text("service_logs", logs.text).status == "ok"
    for handler in (services._service_status, services._stop_service):
        result = invoke(handler, name="absent")
        assert (result.status, result.code, result.text) == ("ok", "LEGACY_WARNING", logs.text)
        assert "operation_outcome" not in result.meta  # absence does not prove process settlement


def test_existing_service_refuses_new_start_without_releasing_prior_custody(invoke, monkeypatch):
    from ouroboros.tools import services
    invoke.ctx.task_id = "service-owner"
    record = SimpleNamespace(proc=SimpleNamespace(poll=lambda: None, pid=123))
    key = services._service_key(invoke.ctx, "demo")
    monkeypatch.setattr(services, "_SERVICES", {key: record})
    monkeypatch.setattr(services, "_panic_requested", False)
    result = invoke(services._start_service, name="demo", cmd=["different-command"])
    assert (result.status, result.code) == ("blocked", "LEGACY_BLOCKED")
    assert result.meta["operation_outcome"] == "completed_no_effect"  # this invocation only
    assert services._SERVICES[key] is record and record.proc.poll() is None


@pytest.mark.parametrize("overflow,exit_code,expected", [(True, -9, "TOOL_ERROR"), (False, 1, "LEGACY_TOOL_ERROR"), (False, 0, "OK")])
def test_skill_process_output_preserves_effects_and_failure(invoke, monkeypatch, overflow, exit_code, expected):
    from ouroboros.tools import skill_exec
    monkeypatch.setattr(skill_exec, "_emit_skill_lifecycle_event", lambda *a, **k: None)
    result = invoke(skill_exec._render_skill_exec_result,
                    payload={"skill": "test", "script": "run.py", "exit_code": exit_code},
                    stdout_bytes=b"file was written before output overflow", stderr_bytes=b"detail", overflowed=overflow)
    assert result.code == expected
    assert "file was written before output overflow" in result.text
    assert "operation_outcome" not in result.meta
    payload = json.loads(result.text.split("\n\n", 1)[1] if overflow or exit_code else result.text)
    assert payload["exit_code"] == exit_code and payload["output_overflow"] is overflow


@pytest.mark.parametrize("case", ["grant", "argv"])
def test_skill_preparation_refusal_never_starts_process(invoke, tmp_path, monkeypatch, case):
    from ouroboros.tools import skill_exec
    from tests.test_skill_exec import _build_skill, _mark_reviewed_and_enabled
    skills = tmp_path / "skills"
    directory = _build_skill(skills, "hello")
    _mark_reviewed_and_enabled(invoke.ctx.drive_root, directory, "hello")
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills))
    monkeypatch.setattr(skill_exec, "grant_status_for_skill", lambda *a: {"missing_keys": ["example"] if case == "grant" else []})
    monkeypatch.setattr(skill_exec, "_scrub_env", lambda **k: {})
    monkeypatch.setattr("ouroboros.argv_budget.argv_budget_excess", lambda *a, **k: "argv exceeds the limit")
    monkeypatch.setattr(skill_exec, "_run_skill_subprocess", lambda *a, **k: pytest.fail("refused skill started a process"))
    result = invoke(skill_exec._handle_skill_exec, skill="hello", script="scripts/hello.py")
    assert result.code == ("LEGACY_BLOCKED" if case == "grant" else "TOOL_ARG_ERROR")
    assert ("SKILL_EXEC_GRANT_REQUIRED" if case == "grant" else "SKILL_EXEC_ARGV_TOO_LARGE") in result.text


def test_oversized_media_refuses_before_pdf_extraction(invoke, tmp_path, monkeypatch):
    from ouroboros.tools import media, vision
    path = tmp_path / "large.pdf"
    path.write_bytes(b"1234")
    monkeypatch.setattr(vision, "_allowed_file_roots", lambda ctx: [tmp_path])
    monkeypatch.setattr(vision, "_read_file_parity_block", lambda *a: "")
    monkeypatch.setattr("ouroboros.protected_artifacts.block_reason_for_path", lambda *a: "")
    monkeypatch.setattr(media, "_OCR_PDF_MAX_BYTES", 3)
    result = invoke(media._ocr_pdf, path=str(path))
    assert (result.status, result.code) == ("error", "TOOL_ARG_ERROR")
    assert result.text == "⚠️ FILE_TOO_LARGE: 4 bytes (max 3)."


@pytest.mark.parametrize("earlier_applied", [False, True])
def test_cherry_pick_conflict_keeps_earlier_effects(invoke, monkeypatch, earlier_applied):
    from ouroboros.tools import git_pr
    first, conflict = "a" * 40, "b" * 40
    shas = [first, conflict] if earlier_applied else [conflict]
    monkeypatch.setattr(git_pr, "_validate_sha_list", lambda *a: shas)
    monkeypatch.setattr(git_pr, "_ouroboros_committer_env", lambda *a: {})
    monkeypatch.setattr(git_pr, "_acquire_git_lock", lambda *a: None)
    monkeypatch.setattr(git_pr, "_release_git_lock", lambda *a: None)
    calls = []

    def git(args, *a, **k):
        calls.append(args)
        stdout = git_pr._PR_BRANCH_PREFIX + "test" if args[0] == "rev-parse" else ""
        failed = args == ["cherry-pick", "--no-edit", conflict]
        return SimpleNamespace(returncode=int(failed), stdout=stdout, stderr="conflict" if failed else "")

    monkeypatch.setattr(git_pr, "_g", git)
    result = invoke(git_pr._cherry_pick_pr_commits, shas=shas)
    assert result.code == "TOOL_ERROR"
    assert (first[:12] in result.text) is earlier_applied
    assert ["cherry-pick", "--abort"] in calls
    assert "operation_outcome" not in result.meta


@pytest.mark.parametrize("paths", [None, ["changed.txt"]])
def test_restore_partial_keeps_checkout_effect_and_target_receipt(invoke, tmp_path, monkeypatch, paths):
    from ouroboros.tools import git, git_vcs_ops
    # repository_path is ResolvedResourceBinding's default: no nested repository was selected.
    binding = SimpleNamespace(root="active_workspace", base_path=tmp_path, repository_path=None)
    monkeypatch.setattr(git, "_vcs_binding", lambda *a, **k: binding)
    monkeypatch.setattr(git, "binding_targets_system_repo", lambda *a: False)
    monkeypatch.setattr("ouroboros.tools.review_helpers.list_changed_paths_from_git_status", lambda *a, **k: ["changed.txt"])
    calls = []

    def command(args, **kwargs):
        calls.append(args)
        if args[1] == "clean":
            raise RuntimeError("cleanup failed")
        assert args[1] == "checkout"
        return ""

    monkeypatch.setattr(git, "run_cmd", command)
    result = invoke(git_vcs_ops._restore_to_head, confirm=True, paths=paths, root="active_workspace")
    assert result.code == "TOOL_ERROR" and len(calls) == 2
    assert "RESTORE_PARTIAL" in result.text and "VCS target: root=active_workspace" in result.text
    assert "operation_outcome" not in result.meta


def test_recorded_historical_outcome_is_not_reclassified_from_its_text():
    from ouroboros._outcome_tool_errors import _classify_tool_errors
    trace = {"tool_calls": [{"tool": "skill_exec", "is_error": False, "status": "ok",
                              "result": "⚠️ SKILL_EXEC_OVERFLOW: historical recorded answer"}]}
    assert all(not rows for rows in _classify_tool_errors(trace).values())
    assert trace["tool_calls"][0]["status"] == "ok"


@pytest.mark.parametrize("action", ["click", "fill", "select", "evaluate", "wait", "unrecognized"])
def test_browser_plain_argument_refusals_survive_the_return_wrapper(invoke, monkeypatch, action):
    from ouroboros.tools import browser
    page = SimpleNamespace(set_default_timeout=lambda timeout: None)
    generation = invoke.ctx.browser_state
    monkeypatch.setattr(browser, "_readonly_subagent", lambda ctx: False)
    monkeypatch.setattr(browser, "_ensure_browser", lambda *a, **k: (page, generation))

    def navigation(*args):
        generation.request_block_reason = "retained browser route notice"
        return ""

    monkeypatch.setattr(browser, "_navigation_block_reason", navigation)
    result = invoke(browser._browser_action, action=action)
    assert (result.status, result.code) == ("error", "TOOL_ARG_ERROR")
    assert result.text.endswith("\n\nretained browser route notice")
    assert "operation_outcome" not in result.meta  # a browser could already have been opened


@pytest.mark.parametrize(("names", "code"), [
    ("", "TOOL_ARG_ERROR"), ("missing", "UNKNOWN_TOOL"), ("hidden", "ACCESS_BLOCKED"),
    ("hidden,missing", "ACCESS_BLOCKED"), ("known,missing", "OK"), ("known", "OK"),
])
def test_tool_enablement_distinguishes_total_refusal_from_partial_effect(invoke, names, code):
    from ouroboros.tools.tool_discovery import enable_tools
    catalog = SimpleNamespace(
        get_schema_by_name=lambda name: {"function": {"name": name}} if name == "known" else None,
        policy_hidden_reason=lambda name: "restricted" if name == "hidden" else "",
        callable_rows=lambda namespace="": [],
    )
    resident = []
    result = invoke(lambda ctx: enable_tools(catalog, ctx, names, resident))
    assert result.code == code
    assert len(resident) == int("known" in names)
    if "known,missing" == names:
        assert "Call schemas loaded" in result.text and "Not found: missing" in result.text


@pytest.mark.parametrize("handler", ["_list_available_tools", "_enable_tools"])
def test_unbound_discovery_is_unavailable(invoke, monkeypatch, handler):
    from ouroboros.tools import tool_discovery
    monkeypatch.setattr(tool_discovery, "_registry", None)
    assert invoke(getattr(tool_discovery, handler)).code == "LEGACY_UNAVAILABLE"


@pytest.mark.parametrize("outcome", ["failed", "unknown", "success", "no_route"])
def test_search_own_failure_object_reaches_registry_without_losing_custody(invoke, monkeypatch, outcome):
    from ouroboros.tools import search
    legs = [{"source": "openai", "backend": "openai_responses", "model": "fixture"}]
    monkeypatch.setattr(search, "resolve_web_search_route", lambda **k: {
        "source": "auto", "legs": [] if outcome == "no_route" else legs, "unavailable": [],
    })
    monkeypatch.setattr(search, "_web_search_deadline_exhausted", lambda ctx: False)
    monkeypatch.setattr("ouroboros.model_slots.resolve_processing_preference", lambda *a, **k: "standard")
    response = ({"answer": "found", "sources": []} if outcome == "success" else
                {"error": "provider did not complete", "ledger_attempt_ids": ["original-attempt"]})
    if outcome == "unknown":
        response["reason_code"] = "provider_outcome_unknown"
    calls = []
    monkeypatch.setattr(search, "_web_search_openai", lambda *a, **k: calls.append(k) or json.dumps(response))
    result = invoke(search._web_search, query="fixture")
    assert result.code == ("OK" if outcome == "success" else "TOOL_ERROR")
    payload = json.loads(result.text)
    assert len(calls) == int(outcome != "no_route")
    if outcome == "unknown":
        assert payload["reason_code"] == "provider_outcome_unknown"
        assert payload["ledger_attempt_ids"] == ["original-attempt"]
    assert "operation_outcome" not in result.meta


@pytest.mark.parametrize("saved", [False, True])
def test_evolution_stats_artifact_outcome_is_not_hidden_by_summary(invoke, monkeypatch, saved):
    from ouroboros.tools import evolution_stats
    from ouroboros import artifacts
    invoke.ctx.task_id = "local-evolution-export"
    monkeypatch.setattr(evolution_stats, "_collect_data", lambda: [
        {"ts": "2026-10-13", "py_lines": 1, "bible_bytes": 2, "system_bytes": 3}])
    if not saved:
        def fail(*args, **kwargs):
            raise OSError("fixture disk refused")
        monkeypatch.setattr(artifacts, "store_task_artifact_bytes", fail)
    result = invoke(evolution_stats.get_tools()[0].handler)
    assert result.code == ("OK" if saved else "TOOL_ERROR")
    assert ("py_lines=1 bible=2B system=3B" if saved else "fixture disk refused") in result.text
    assert "operation_outcome" not in result.meta


@pytest.mark.parametrize("case", ["stale", "unstable", "partial", "empty"])
def test_recent_tasks_snapshot_refusal_differs_from_partial_read(invoke, monkeypatch, case):
    from ouroboros.tools import recent_tasks

    serial = iter(range(10))
    monkeypatch.setattr(recent_tasks, "_task_inventory", lambda root: (
        [{"name": str(next(serial))}] if case == "unstable" else
        [{"name": "unreadable"}] if case == "partial" else []))
    monkeypatch.setattr(recent_tasks, "_recent_tasks_snapshot", lambda *a, **k: "current")
    monkeypatch.setattr(recent_tasks, "_task_record", lambda name, **k: (
        None, {"task_id": name, "error": "row could not be read"}))
    monkeypatch.setattr(recent_tasks, "_running_tasks", lambda *a: [])
    result = invoke(recent_tasks._handle_recent_tasks, snapshot="prior" if case == "stale" else "")
    payload = json.loads(result.text)
    assert result.code == ("LEGACY_UNAVAILABLE" if case in {"stale", "unstable"} else "OK")
    assert result.meta["operation_outcome"] == "completed_no_effect"  # existing read wrapper
    if case == "partial":
        assert payload["unreadable_tasks"][0]["task_id"] == "unreadable"
    if case in {"stale", "unstable"}:
        assert payload["tasks"] == [] and payload["error"]["code"].startswith("RECENT_TASKS_SNAPSHOT_CHANGED")


def _assert_consumers(result, code):
    from ouroboros._outcome_tool_errors import _OK_TOOL_STATUSES, _classify_tool_errors
    from ouroboros.loop_tool_execution import _typed_execution_failure, _typed_result_metadata

    assert result.code == code, result.text
    failed = _typed_execution_failure(True, result)
    facts = _typed_result_metadata("census", result.text, failed, result)
    row = {"tool": "census", "result": result.text, **facts, "is_error": failed}
    assert facts["status"] == TOOL_CODE_SPECS[code].outcome_bucket
    assert (row["status"] in _OK_TOOL_STATUSES) is not failed
    classified = _classify_tool_errors({"tool_calls": [row]})
    if TOOL_CODE_SPECS[code].status == "ok":
        assert not any(classified.values())
    elif code in {"LEGACY_BLOCKED", "LEGACY_UNAVAILABLE"}:
        assert len(classified["policy_denials"]) == 1 and not classified["unresolved"]
    else:
        assert len(classified["unresolved"]) == 1 and not classified["policy_denials"]
    assert result.meta.get("operation_outcome") != "completed_no_effect"


@pytest.mark.parametrize("phase", ["pre_tag", "post_receipt", "containment_failed", "orphaned"])
def test_evolution_retained_commit_publication_consumers(invoke, monkeypatch, phase):
    from ouroboros.tools import git, git_evolution
    invoke.ctx.current_task_type = "evolution"
    sha = "c" * 40
    records = []
    monkeypatch.setattr(git, "_evolution_commit_authority", lambda *a, **k: ({}, {"ok": False, "reason": "revoked"}))
    monkeypatch.setattr(git, "_record_commit_attempt", lambda *a, **k: records.append(k))
    if phase == "containment_failed":
        # Run the real exception branch that publishes the native fault.
        monkeypatch.setattr("supervisor.evolution_lifecycle.update_evolution_transaction", lambda *a, **k: True)
        def fail_containment(*a, **k):
            raise RuntimeError("retained commit cannot be contained")
        monkeypatch.setattr(git, "run_cmd", fail_containment)
    else:
        monkeypatch.setattr(git, "_preserve_evolution_orphan", lambda *a, **k:
                            f"Retained at refs/ouroboros/evolution-orphans/{sha}")
    monkeypatch.setattr("supervisor.evolution_lifecycle.record_evolution_commit", lambda **k: {"ok": False, "reason": "receipt unavailable"})
    if phase == "pre_tag":
        handler = lambda ctx: git_evolution._check_evolution_commit_stage(
            ctx, "reviewed change", 0, phase="pre_tag_authority", commit_sha=sha)[1]
    elif phase == "orphaned":
        handler = lambda ctx: git_evolution._record_evolution_commit_receipt(ctx, "reviewed change", 0, {}, sha)
    else:
        handler = lambda ctx: git_evolution._evolution_publication_stopped_result(
            ctx, "reviewed change", sha, " retained test warning")
    result = invoke(handler)
    _assert_consumers(result, "TOOL_ERROR" if phase in {"orphaned", "containment_failed"} else "LEGACY_BLOCKED")
    assert records and result.text == records[-1]["block_details"]
    assert (sha in result.text) or phase == "containment_failed"
    if phase in {"post_receipt", "containment_failed"}:
        assert result.text.endswith(" retained test warning")
        assert invoke.ctx.last_push_succeeded is False


@pytest.mark.parametrize("containment_failed", [False, True])
def test_pre_tag_publication_wrapper_preserves_containment_outcome(invoke, monkeypatch, containment_failed):
    from ouroboros.tools import git
    invoke.ctx.current_task_type = "evolution"
    invoke.ctx.task_id = "evolution-root"
    claim = {"campaign_id": "campaign", "transaction_id": "transaction", "task_id": "evolution-root"}
    sha = "c" * 40
    authority = iter([(claim, {"ok": True}), (claim, {"ok": True}), (claim, {"ok": False, "reason": "revoked"})])
    monkeypatch.setattr("supervisor.update_merge.managed_assisted_tx_for", lambda *a: ({}, ""))
    monkeypatch.setattr(git, "_commit_request_refusals", lambda *a, **k: (None, ""))
    monkeypatch.setattr(git, "_task_attributed_commit_paths", lambda *a: (None, None, "", None))
    monkeypatch.setattr(git, "_check_overlapping_review_attempt", lambda *a: "")
    monkeypatch.setattr(git, "_record_commit_attempt", lambda *a, **k: None)
    monkeypatch.setattr(git, "_acquire_git_lock", lambda *a: object())
    monkeypatch.setattr(git, "_release_git_lock", lambda *a: None)
    monkeypatch.setattr(git, "_prepare_review_commit_worktree", lambda *a: (False, ""))
    monkeypatch.setattr(git, "_evolution_commit_authority", lambda *a, **k: next(authority))
    monkeypatch.setattr(git, "_record_evolution_commit_intent", lambda *a: None)
    monkeypatch.setattr(git, "_run_reviewed_stage_cycle", lambda *a, **k: {
        "status": "passed", "pre_fingerprint": {"fingerprint": "same"},
        "post_fingerprint": {"fingerprint": "same", "binding": {}},
    })
    monkeypatch.setattr(git, "_commit_reviewed_candidate", lambda *a: (sha, ""))
    monkeypatch.setattr(git, "_verify_reviewed_commit_binding", lambda *a, **k: (True, ""))
    monkeypatch.setattr(git, "_managed_post_commit_tests_gate", lambda *a, **k: None)
    monkeypatch.setattr(git, "_auto_push", lambda *a, **k: pytest.fail("revoked publication pushed"))
    monkeypatch.setattr(git, "_auto_tag_on_version_bump", lambda *a, **k: pytest.fail("revoked publication tagged"))
    if containment_failed:
        monkeypatch.setattr("supervisor.evolution_lifecycle.update_evolution_transaction", lambda *a, **k: True)
        def fail_containment(*a, **k):
            raise RuntimeError("retained containment fault")
        monkeypatch.setattr(git, "run_cmd", fail_containment)
    else:
        monkeypatch.setattr(git, "_preserve_evolution_orphan", lambda *a, **k: "Retained private commit.")
    result = invoke(git._repo_commit_push, commit_message="reviewed candidate", skip_tests=True)
    _assert_consumers(result, "TOOL_ERROR" if containment_failed else "LEGACY_BLOCKED")
    assert result.text.startswith("⚠️ EVOLUTION_PUBLICATION_STOPPED:") and sha in result.text
    assert ("EVOLUTION_ORPHAN_CONTAINMENT_FAILED" in result.text) is containment_failed


@pytest.mark.parametrize("lost", [False, True])
def test_pending_review_observation_preserves_custody(invoke, monkeypatch, lost):
    from ouroboros.tools import git, git_review_cycle
    monkeypatch.setattr(git_review_cycle, "review_enforcement_blocks", lambda *a: True)
    records = []
    monkeypatch.setattr(git, "_record_commit_attempt", lambda *a, **k: records.append((a, k)))
    custody = SimpleNamespace(review_retry_key="paid-wave")
    invoke.ctx._pending_review_attempt = custody
    invoke.ctx._review_custody_lost = lost
    invoke.ctx._current_review_retry_key = "paid-wave"
    result = invoke(git_review_cycle._finalize_pending_review, commit_message="same candidate", commit_start=0,
                    pre_fingerprint={"fingerprint": "same"}, post_fingerprint={"fingerprint": "same"})
    _assert_consumers(result, "LEGACY_UNAVAILABLE" if lost else "OK")
    assert result.meta["review_pending"] is True
    assert result.meta["review_custody_lost"] is lost
    assert invoke.ctx._pending_review_attempt is custody
    assert records[0][0][2] == "reviewing" and records[0][1]["late_result_pending"] is True
    assert records[0][1]["review_retry_key"] == "paid-wave"


@pytest.mark.parametrize("case", ["content", "prompt", "empty_models", "invalid_models", "too_many", "exception"])
def test_multi_review_validation_and_execution_fault_consumers(invoke, monkeypatch, case):
    from ouroboros.tools import review_multi_model as producer
    args = {"content": "candidate", "prompt": "review", "models": ["fixture"]}
    if case == "content": args["content"] = ""
    if case == "prompt": args["prompt"] = ""
    if case == "empty_models": args["models"] = []
    if case == "invalid_models": args["models"] = 4
    if case == "too_many": args["models"] *= producer.MAX_MODELS + 1
    if case == "exception":
        async def fail(*a, **k):
            raise ValueError("retained execution fault")
        monkeypatch.setattr(producer, "_multi_model_review_async", fail)
    result = invoke(producer._handle_multi_model_review, **args)
    _assert_consumers(result, "TOOL_ERROR" if case == "exception" else "TOOL_ARG_ERROR")
    assert set(json.loads(result.text)) == {"error"}


@pytest.mark.parametrize("case", ["policy", "unavailable", "reference", "disposition", "preflight"])
def test_skill_author_finish_branch_outcomes_preserve_review_state(invoke, monkeypatch, case):
    from ouroboros.tools import skill_exec
    review_state = SimpleNamespace(status="pending" if case == "reference" else "failed", review_profile="",
                                  findings=[] if case == "reference" else ["retained finding"], raw_actor_records=[],
                                  raw_result="", reviewed_content_hash="reviewed", content_hash="current")
    loaded = SimpleNamespace(skill_dir=invoke.ctx.repo_dir, manifest=SimpleNamespace(entry="", scripts=[], type="skill"))
    binding = SimpleNamespace(state_drive_root=invoke.ctx.drive_root)
    monkeypatch.setattr("ouroboros.config.get_review_enforcement", lambda: "blocking" if case == "policy" else "advisory")
    monkeypatch.setattr("ouroboros.tools.review_helpers.review_enforcement_blocks", lambda mode: mode == "blocking")
    monkeypatch.setattr(skill_exec, "load_bound_skill", lambda *a: None if case == "unavailable" else loaded)
    monkeypatch.setattr("ouroboros.skill_loader.compute_content_hash", lambda *a, **k: "current")
    monkeypatch.setattr("ouroboros.skill_loader.load_review_state", lambda *a, **k: review_state)
    monkeypatch.setattr("ouroboros.skill_loader.save_review_state", lambda *a, **k: pytest.fail("refusal rewrote reviewer state"))
    monkeypatch.setattr("ouroboros.skill_review_runner._read_review_job", lambda *a: {})
    monkeypatch.setattr("ouroboros.skill_review_history.load_history", lambda *a, **k: [])
    monkeypatch.setattr("ouroboros.skill_review._run_deterministic_preflight", lambda *a, **k: SimpleNamespace(status="failed"))
    def handler(ctx):
        return skill_exec._author_finish_existing_skill_review(ctx, binding, "fixture",
            disposition="invalid" if case == "disposition" else "accepted", rationale="considered feedback")["error"]
    result = invoke(handler)
    _assert_consumers(result, {"policy": "LEGACY_BLOCKED", "unavailable": "LEGACY_UNAVAILABLE",
                              "reference": "TOOL_ARG_ERROR", "disposition": "TOOL_ARG_ERROR", "preflight": "TOOL_ERROR"}[case])
    assert result.text.startswith("SKILL_REVIEW_ERROR:")
    assert review_state.reviewed_content_hash == "reviewed"


@pytest.mark.parametrize("case", ["identical", "ceiling", "overlap", "advisory", "stopped"])
def test_review_gate_return_boundaries_reach_consumers(invoke, monkeypatch, case):
    from ouroboros.tools import git, git_review_cycle
    monkeypatch.setattr("ouroboros.config.get_review_enforcement", lambda: "advisory" if case == "advisory" else "blocking")
    monkeypatch.setattr(git, "review_enforcement_blocks", lambda mode: mode == "blocking")
    monkeypatch.setattr(git_review_cycle, "review_enforcement_blocks", lambda mode: mode == "blocking")
    monkeypatch.setattr(git, "commit_review_contract_fingerprint", lambda: "contract")
    monkeypatch.setattr(git, "resolve_root_task_id", lambda ctx: "root")
    monkeypatch.setattr(git, "_commit_review_retry_key", lambda *a, **k: "new-key")
    monkeypatch.setattr(git, "check_identical_verdict_refusal", lambda *a, **k:
                        "⚠️ IDENTICAL_DIFF_REFUSED: retained verdict" if case == "identical" else "")
    monkeypatch.setattr(git, "check_review_cycles_ceiling", lambda *a, **k:
                        {"message": "⚠️ REVIEW_CYCLES_EXHAUSTED: retained paid count", "cycles_paid": 3, "cap": 3}
                        if case in {"ceiling", "advisory"} else None)
    monkeypatch.setattr(git, "emit_review_cycles_exhausted", lambda *a, **k: None)
    monkeypatch.setattr(git, "record_commit_gate_refusal", lambda *a, **k: "free-refusal")
    monkeypatch.setattr(git, "run_cmd", lambda *a, **k: "")
    monkeypatch.setattr(git, "_authorized_managed_update_resolver", lambda *a: False)
    monkeypatch.setattr(git, "_stage_candidate_for_review", lambda *a, **k: (["x.py"], ["x.py"], None))
    monkeypatch.setattr(git, "protected_paths_in", lambda *a: [])
    monkeypatch.setattr(git, "_current_runtime_mode", lambda: "advanced")
    fingerprint = {"ok": True, "fingerprint": "same", "binding": {}}
    monkeypatch.setattr(git, "_fingerprint_staged_diff", lambda *a: fingerprint)
    monkeypatch.setattr(git, "_record_commit_attempt", lambda *a, **k: None)
    custody = SimpleNamespace(review_retry_key="original-key", review_contract_fingerprint="contract")
    invoke.ctx._pending_review_attempt = custody
    invoke.ctx._review_resume_pending = case == "overlap"
    invoke.ctx._review_advisory = []
    if case in {"stopped", "advisory"}:
        monkeypatch.setattr(git, "_review_binding_precondition_error", lambda *a, **k: "")
        monkeypatch.setattr(git, "_preflight_and_tests_gate", lambda *a, **k: None)
        monkeypatch.setattr("ouroboros.review_state.compute_snapshot_hash", lambda *a, **k: "same")
        monkeypatch.setattr("ouroboros.review_evidence.capture_commit_review_evidence", lambda *a: {})
        monkeypatch.setattr(git_review_cycle, "_revalidation_outcome", lambda *a, **k: None)
        monkeypatch.setattr(git_review_cycle, "_release_review_evidence_if_settled", lambda *a: None)
        monkeypatch.setattr(git_review_cycle, "settle_commit_review_ledger", lambda *a, **k: None)
        monkeypatch.setattr("ouroboros.tools.commit_gate.disclose_commit_review_replay", lambda *a: None)
        monkeypatch.setattr(git, "_install_paid_dispatch_stamp", lambda *a, **k: None)
        monkeypatch.setattr(git, "_run_parallel_review", lambda *a, **k: (None, None, "", []))
        monkeypatch.setattr(git, "_reconcile_and_clear_review_roster", lambda *a: None)
        monkeypatch.setattr(git, "_aggregate_review_verdict", lambda *a: (False, "", "", [], []))
        monkeypatch.setattr(git, "_review_custody_pending", lambda *a, counted_only=False: False)
        monkeypatch.setattr(git, "_subject_binding_mismatch_outcome", lambda *a: None)
        monkeypatch.setattr("ouroboros.review_custody.review_retry_cancelled", lambda *a: case == "stopped")
        monkeypatch.setattr("ouroboros.deadline_utils.owner_deadline_exhausted_for_context", lambda *a: False)
        monkeypatch.setattr(git, "classify_review_block", lambda **k: "infra")
        monkeypatch.setattr(git, "_finalize_blocked_review", lambda *a, **k: k["combined_msg"])
    else:
        monkeypatch.setattr(git, "_run_parallel_review", lambda *a, **k: pytest.fail("refusal dispatched paid review"))
    def handler(ctx):
        if case == "advisory":
            outcome = git._free_cycle_gate(ctx, "candidate", 0,
                pre_fingerprint=fingerprint, review_rebuttal="")
            assert "advisory_replay" in outcome
            return outcome["advisory_replay"]
        outcome = git_review_cycle._reviewed_stage_cycle(ctx, "candidate", 0)
        return outcome["message"] or "Author continuation remains available."
    result = invoke(handler)
    _assert_consumers(result, "LEGACY_WARNING" if case == "advisory" else "LEGACY_BLOCKED")
    assert invoke.ctx._pending_review_attempt is custody
    if case == "stopped":
        assert result.text.startswith("⚠️ REVIEW_STOPPED:")


@pytest.mark.parametrize("refused", [False, True])
def test_skill_cycle_refusal_preserves_pending_payload(invoke, monkeypatch, refused):
    from ouroboros.skill_review import SkillReviewOutcome, _outcome_from_result
    from ouroboros.skill_review_runner import _outcome_payload
    from ouroboros.tools import skill_exec
    binding = SimpleNamespace(state_drive_root=invoke.ctx.drive_root)
    if refused:
        # The wave's typed $0 refusal at the shared ceiling (review_change), as review_skill reads it.
        result = {"state": "settled", "aggregate": "NOT_DISPATCHED", "record_id": "rl-refused", "rows": [],
                  "findings": {}, "panel": {"seats": 1}, "cost": {}, "reused": False,
                  "dispatch_refusal": {"kind": "review_cycles_exhausted",
                                       "message": "⚠️ REVIEW_CYCLES_EXHAUSTED: 3 of 3 paid skill-review waves spent"}}
        outcome = _outcome_from_result(SimpleNamespace(name="fixture"), SimpleNamespace(tree_sha="same", delivery={}),
                                       result, rebuttal_sha="")
        assert outcome.dispatch_refusal_code == "LEGACY_BLOCKED" and outcome.paid is False
    else:
        # Quoted/recorded prose cannot fabricate a live dispatch refusal.
        outcome = SkillReviewOutcome(skill_name="fixture", status="pending", content_hash="same",
            error="⚠️ REVIEW_CYCLES_EXHAUSTED: retained observation, no live refusal")
    outcome.raw_actor_records = [{"operation_id": "paid-wave"}]
    payload = _outcome_payload(outcome, deps_status="not_run", deps_error="",
        extension_action=None, extension_reason=None, job_data={"job_id": "prior-job"})
    assert ("dispatch_refusal_code" in payload) is refused
    monkeypatch.setattr(skill_exec, "_skill_tool_preflight", lambda *a: "")
    monkeypatch.setattr("ouroboros.skill_review_runner.run_skill_review_lifecycle_blocking", lambda *a, **k: payload)
    # Keep the renderer's real content and append the existing job reference.
    result = invoke(lambda ctx: skill_exec._handle_review_skill(ctx, skill="fixture", _resolved_binding=binding))
    _assert_consumers(result, "LEGACY_BLOCKED" if refused else "OK")
    assert "REVIEW_CYCLES_EXHAUSTED" in result.text and "prior-job" in result.text
    assert payload["status"] == "pending" and payload["raw_actor_records"][0]["operation_id"] == "paid-wave"
