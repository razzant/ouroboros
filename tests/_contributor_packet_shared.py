"""Shared fixture of the contributor packet golden
(``tests/fixtures/contributor_review_packet_golden.json``).

One installed body (a git checkout detached at the base commit) and one committed
proposal that rewrites the review flow, the wrapper and the checklist; the seats the
review returns are persisted as real observability receipts, retained answers and
settled session custody, so the wrapper's packet is built from production readers.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess

GIT_ENV = {
    "GIT_AUTHOR_NAME": "Fixture", "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
    "GIT_COMMITTER_NAME": "Fixture", "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
    "GIT_AUTHOR_DATE": "2026-01-02T03:04:05+00:00", "GIT_COMMITTER_DATE": "2026-01-02T03:04:05+00:00",
}
BASE_FILES = {
    "VERSION": "1.2.3\n",
    "README.md": "# Fixture project\n\nInstalled body.\n",
    "build.sh": "#!/bin/sh\necho build\n",
    "docs/CHECKLISTS.md": "# Checklist\n\n- installed rule\n",
    "ouroboros/config.py": "FIXTURE = 'installed'\n",
    "ouroboros/tools/review.py": "RULES = 'installed review flow'\n",
    "scripts/run_external_review.py": "# installed wrapper\n",
}
PROPOSAL_FILES = {
    "README.md": "# Fixture project\n\nInstalled body.\n\nProposed paragraph.\n",
    "build.sh": "#!/bin/sh\necho build --fast\n",
    "docs/CHECKLISTS.md": "# Checklist\n\n- proposal relaxes every rule\n",
    "docs/new_note.md": "A new note.\n",
    "ouroboros/tools/review.py": "RULES = 'proposal rewrites the review flow'\n",
    "scripts/run_external_review.py": "# proposal wrapper\n",
}
PROPOSAL_TITLE = "Proposal: faster build"

# The golden's three seats as the wrapper resolves them from the review pool: t1
# receives the packet; t2 (a session) and s1 (a natively retrieving api row) read the
# subject themselves and are asked both parts of the brief.
GOLDEN_CONFIG = {
    "profile": "external_pr_readiness",
    "provider": "configured_per_slot",
    "slot_config_source": "review_pool",
    "pool_slots": [
        {"slot_id": "t1", "route": {"kind": "api_chat", "target_id": "openai/gpt-5.6-sol"}, "effort": "high",
         "delivery": "packet"},
        {"slot_id": "t2", "route": {"kind": "agent_session", "target_id": "codex=gpt-5.6-sol",
                                    "profile_id": "pinned"}, "effort": "high"},
        {"slot_id": "s1", "route": {"kind": "api_chat", "target_id": "openai/gpt-5.6-sol"}, "effort": "xhigh",
         "delivery": "native"},
    ],
    "pool_models": ["openai/gpt-5.6-sol", "codex=gpt-5.6-sol", "openai/gpt-5.6-sol"],
    "pool_efforts": ["high", "high", "xhigh"],
    "review_enforcement": "blocking",
    "context_mode": "max",
    "runtime_mode": "pro",
}


def _coupling_matrix(reason: str) -> list:
    """The whole Coupling questions answered PASS: a retrieving seat's
    ``coupling`` block must cover every required item or the gate records it as
    unanswered."""
    return [{"item": item, "verdict": "PASS", "severity": "advisory", "reason": reason}
            for item in ("intent_alignment", "forgotten_touchpoints", "cross_surface_consistency",
                         "regression_surface", "prompt_doc_sync", "architecture_fit",
                         "cross_module_bugs", "implicit_contracts")]


# The packet seat (t1) answers contract A (the change array); the retrieving seats
# (t2, a session; s1, a natively retrieving api row) are asked both parts of the
# brief and answer contract B (one object: ``change`` + ``coupling``).
ANSWERS = {
    "t1": json.dumps([{"item": "code_quality", "verdict": "PASS", "severity": "advisory",
                       "reason": "t1 read build.sh"}]),
    "t2": json.dumps({"change": [{"item": "tests_affected", "verdict": "PASS", "severity": "advisory",
                                  "reason": "t2 session read the tests"}],
                      "coupling": _coupling_matrix("t2 session read the checkout")}),
    "s1": json.dumps({"change": [], "change_clean": True,
                      "coupling": _coupling_matrix("s1 scope matches the title")}),
}
SESSION_TRANSCRIPT = "full session transcript of t2\nEOF_SENTINEL"
SESSION_RUN_ID = "run-golden"


def golden_pool(*extra_rows: dict) -> str:
    """``GOLDEN_CONFIG``'s seats as the review pool (``OUROBOROS_SUBAGENTS`` rows): ``t1``
    reads the packet, ``t2`` (a session) and ``s1`` (a natively retrieving api row)
    retrieve and are asked both parts of the brief. ``extra_rows`` are further catalog
    rows (e.g. an unmarked row a lane names by id)."""
    from tests.review_pool_rosters import pool_roster, pool_seat

    return pool_roster(pool_seat("t1", "openai/gpt-5.6-sol", effort="high"),
                       pool_seat("t2", "codex=gpt-5.6-sol", kind="agent_session", profile_id="pinned", effort="high"),
                       pool_seat("s1", "openai/gpt-5.6-sol", delivery="native", effort="xhigh"), *extra_rows)


def git(repo: pathlib.Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-c", "core.autocrlf=false", "-c", "commit.gpgsign=false", *args],
        cwd=str(repo), capture_output=True, text=True, check=True, env={**os.environ, **GIT_ENV},
    )
    return result.stdout.strip()


def _write(repo: pathlib.Path, files: dict[str, str]) -> None:
    for relative, text in files.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))


def init_installed_body(root: pathlib.Path) -> dict[str, str]:
    """The installed checkout, detached at the base, with the proposal on branch ``proposal``."""
    repo = (root / "installed").resolve()
    repo.mkdir(parents=True)
    git(repo, "init", "-q")
    _write(repo, BASE_FILES)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "installed body")
    base = git(repo, "rev-parse", "HEAD")
    git(repo, "branch", "base", base)
    git(repo, "checkout", "-q", "-b", "proposal")
    _write(repo, PROPOSAL_FILES)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "proposal")
    head = git(repo, "rev-parse", "HEAD")
    git(repo, "checkout", "-q", "--detach", base)
    patch = subprocess.run(
        ["git", "diff", "--binary", "--no-ext-diff", f"{base}..{head}"],
        cwd=str(repo), capture_output=True, check=True,
    ).stdout
    return {
        "repo": str(repo), "base_sha": base, "head_sha": head,
        "head_tree_sha": git(repo, "rev-parse", f"{head}^{{tree}}"),
        "diff_sha256": hashlib.sha256(patch).hexdigest(),
    }


def isolation_record(drive: pathlib.Path) -> dict:
    return {"review_data_root": str(drive), "run_cap_usd": 5.0, "attach_host_engine": True}


def persist_golden_actors(drive: pathlib.Path) -> tuple[list[dict], dict]:
    """The seats' raw actor records, with their receipts persisted under ``drive``."""
    from ouroboros import delegate_custody as custody
    from ouroboros.observability import persist_call

    def prompt(call_id: str, slot: dict) -> dict:
        return persist_call(drive, task_id="review", call_id=call_id, call_type="prompt",
                            payload={"request": {"surface": "review"}, "slot": slot})

    def response(call_id: str, usage: dict, *, answer: str, transcript: str = "") -> dict:
        message = {"content": answer}
        if transcript:
            message["session_transcript"] = transcript
            usage = {**usage, "verdict_provenance": {
                "raw_transcript_chars": len(transcript),
                "raw_transcript_sha256": hashlib.sha256(transcript.encode("utf-8", "replace")).hexdigest(),
            }}
        return persist_call(drive, task_id="review", call_id=call_id, call_type="response",
                            payload={"message": message, "usage": usage})

    api_usage = {"provider": "openrouter", "resolved_model": "openai/gpt-5.6-sol"}
    t1 = {
        "slot_id": "t1", "model_id": "openai/gpt-5.6-sol", "status": "responded",
        "tokens_in": 1200, "tokens_out": 300, "cost_usd": 0.0125, "raw_text": ANSWERS["t1"],
        "prompt_ref": prompt("t1_prompt", {"slot_id": "t1", "model": "openai/gpt-5.6-sol", "effort": "high",
                                           "route": "api_chat", "session_target": "", "session_profile": ""}),
        "response_ref": response("t1_response", api_usage, answer=ANSWERS["t1"]),
    }
    t2 = {
        "slot_id": "t2", "model_id": "codex=gpt-5.6-sol", "status": "responded",
        "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0, "raw_text": ANSWERS["t2"],
        "prompt_ref": prompt("t2_prompt", {"slot_id": "t2", "model": "codex=gpt-5.6-sol", "effort": "high",
                                           "route": "agent_session", "session_target": "codex=gpt-5.6-sol",
                                           "session_profile": "pinned"}),
        "response_ref": response("t2_response", {
            "provider": "claudexor", "delegated_route": "codex", "resolved_model": "gpt-5.6-sol",
            "applied_profile": "pinned", "applied_access": "readonly", "delegated_run_id": SESSION_RUN_ID,
            "custody_durable": True, "output_conformance": "passed", "verdict_method": "schema",
        }, answer=ANSWERS["t2"], transcript=SESSION_TRANSCRIPT),
    }
    s1 = {
        "slot_id": "s1", "model_id": "openai/gpt-5.6-sol", "status": "responded",
        "tokens_in": 3000, "tokens_out": 500, "cost_usd": 0.02, "raw_text": ANSWERS["s1"],
        "prompt_ref": prompt("s1_prompt", {"slot_id": "s1", "model": "openai/gpt-5.6-sol", "effort": "xhigh",
                                           "route": "api_chat", "session_target": "", "session_profile": ""}),
        "response_ref": response("s1_response", api_usage, answer=ANSWERS["s1"]),
    }
    custody.record_started(drive, custody.RunCustody(
        run_id=SESSION_RUN_ID, task_id="review", project_id="review-project",
        project_owned=True, ledger_root=str(drive),
    ))
    for event in (custody.LEDGER_RECORDED, custody.SETTLED, custody.PROJECT_RETIRED):
        custody.emit(drive, event, {"run_id": SESSION_RUN_ID})
    custody._CUSTODY.clear()
    return [t1, t2], {"status": "responded", "model_id": "openai/gpt-5.6-sol", "raw_results": [s1]}


GOLDEN_USAGE = {
    "t1": {"provider": "openrouter", "resolved_model": "openai/gpt-5.6-sol",
           "prompt_tokens": 1200, "completion_tokens": 300, "cost": 0.0125},
    "t2": {"provider": "claudexor", "delegated_route": "codex", "resolved_model": "gpt-5.6-sol",
           "applied_profile": "pinned", "applied_access": "readonly", "delegated_run_id": SESSION_RUN_ID,
           "custody_durable": True, "output_conformance": "passed", "verdict_method": "schema"},
    "s1": {"provider": "openrouter", "resolved_model": "openai/gpt-5.6-sol",
           "prompt_tokens": 3000, "completion_tokens": 500, "cost": 0.02},
}


def _golden_rows():
    """``persist_golden_actors`` once per drive, by seat id (thread-safe)."""
    import threading

    persisted: dict[str, dict] = {}
    lock = threading.Lock()

    def rows(drive_root) -> dict[str, dict]:
        with lock:
            if not persisted:
                triad, scope = persist_golden_actors(pathlib.Path(drive_root))
                persisted.update({row["slot_id"]: row for row in [*triad, *scope["raw_results"]]})
        return persisted

    return rows


def golden_substrate(briefs: list[dict]):
    """A ``review_substrate.run_review_request`` stand-in under the REAL review
    operation: the paid seam only. Each seat answers from ``ANSWERS`` with the golden's
    persisted receipts (``persist_golden_actors``), so the operation's record rows carry
    the pre-move packet's refs; what each seat was GIVEN is appended to ``briefs``."""
    from types import SimpleNamespace

    rows = _golden_rows()

    def run_review_request(request, *, slots, drive_root, llm=None, usage_ctx=None):
        persisted = rows(drive_root)
        # The gate reserves every seat's operation id before any send; an answer
        # that does not carry the reserved id is not that seat's answer.
        reserved = (getattr(usage_ctx, "_review_reserved_operations", None) or {}).get(request.surface) or {}
        actors = []
        for slot in slots:
            row = persisted[slot.slot_id]
            briefs.append({"slot_id": slot.slot_id, "surface": request.surface, "model": slot.model,
                           "messages": [dict(m) for m in request.messages], "session_task": request.session_task,
                           "session_root": request.session_root})
            actors.append({
                "slot_id": slot.slot_id, "model": slot.model, "status": "ok", "raw_text": row["raw_text"],
                "usage": dict(GOLDEN_USAGE[slot.slot_id]), "prompt_ref": row["prompt_ref"], "response_ref": row["response_ref"],
                "operation_id": str(reserved.get(slot.slot_id) or f"op-{slot.slot_id}"),
                "operation_state": "settled", "late_result_pending": False,
            })
        return SimpleNamespace(actors=actors)

    return run_review_request


def golden_physical_seam(sends: list[dict], *, answer=None):
    """A ``ReviewCoordinator._run_slot`` stand-in: the PHYSICAL send of one seat, under
    the REAL custody layer (``review_custody``: attempt keys, settled replays, pending
    rejoins, the paid stamp). Every send is appended to ``sends`` with the identity the
    custody layer keyed it by; the seat answers with the golden actor unless ``answer``
    (``answer(request, slot, actor, retry_state=..., pending_invocation_checkpoint=...)
    -> actor``) says otherwise — e.g. a delegated seat whose start outcome is unknown
    answers an error actor carrying ``usage["pending_invocation_id"]`` after
    checkpointing that token, as the session executor does. The seam leaves the real
    ``_run_slot``'s durable producer trail (the operation's prompt, then its completed
    outcome under the operation binding), so a later process can recover the seat
    (``review_operation.recover_review_producer``) exactly as in production."""
    from dataclasses import asdict

    from ouroboros.observability import persist_call
    from ouroboros.review_custody import finalize_review_actor
    from ouroboros.review_dispatch import invoke_review_paid_stamp, review_operation_binding

    rows = _golden_rows()

    def run_slot(self, request, slot, *, operation_id="", retry_state=None, logical_deadline_monotonic=None,
                 pending_invocation_checkpoint=None):
        invoke_review_paid_stamp(self._review_paid_stamp)  # what a route executor does before its transport
        call_id, call_type = str(operation_id), request.call_type or f"{request.surface}_review"
        binding = review_operation_binding(request, slot, call_id)
        row = rows(self.drive_root)[slot.slot_id]
        sends.append({"slot_id": slot.slot_id, "surface": request.surface, "operation_id": call_id,
                      "retry_key": str(request.retry_key or ""), "session_root": str(request.session_root or ""),
                      "retry_state": dict(retry_state or {}), "reconcile_only": bool(request.reconcile_only)})
        actor = self._error_actor(request, slot, "unused", operation_id=call_id)
        actor.status, actor.error, actor.raw_text = "ok", "", row["raw_text"]
        actor.usage, actor.prompt_ref, actor.response_ref = dict(GOLDEN_USAGE[slot.slot_id]), row["prompt_ref"], row["response_ref"]
        invocation = {"id": str((retry_state or {}).get("pending_invocation_id") or "")}

        def checkpoint(invocation_id):
            # The session executor's order: the durable START_REQUESTED row that names
            # the resources the start binds (what a rejoin after a restart finds), then
            # the reserved seat's checkpoint of the same token.
            from ouroboros import delegate_custody as custody

            invocation["id"] = str(invocation_id)
            assert custody.record_start_requested(
                self._custody_drive_root(), run_id="", task_id=request.task_id, invocation_id=str(invocation_id),
                idempotency_key=str(invocation_id), operation_id=call_id, max_seconds=300,
                request={"prompt": "golden review session"}, project_id="", project_owned=False, route="golden",
                surface=request.surface, slot_id=slot.slot_id,
                root_task_id=str(binding.get("root_task_id") or request.task_id), parent_task_id="")
            if callable(pending_invocation_checkpoint):
                pending_invocation_checkpoint(invocation_id)

        if answer is not None:
            actor = answer(request, slot, actor, retry_state=dict(retry_state or {}), pending_invocation_checkpoint=checkpoint)
        persist_call(self._custody_drive_root(), task_id=request.task_id or "review", call_id=f"{call_id}_prompt",
                     call_type=f"{call_type}_prompt", payload={"request": asdict(request), "slot": asdict(slot)},
                     manifest={"surface": request.surface, "slot_id": slot.slot_id, "model": slot.model,
                               "review_operation_binding": binding})
        actor.recovery_binding = {**binding, "pending_invocation_id": invocation["id"]}
        finalize_review_actor(actor, operation_id=call_id)
        actor.response_ref = self._persist_producer_outcome(
            request, actor, call_id, call_type, {"message": {"content": actor.raw_text}, "usage": actor.usage})
        return actor

    return run_slot


def passing_test_runner(ctx, **_kwargs):
    """The commit gate's hermetic runner stand-in: attests a passed suite of the
    checkout it was handed (the process-held proof the gate's tests fact reads)."""
    from ouroboros.commit_admission import capture_preflight_test_subject

    ctx._preflight_tests_passed = True
    ctx._preflight_test_proof = capture_preflight_test_subject(ctx.repo_dir)
    return None


def full_output_sections(text: str) -> dict[str, str]:
    """``full-output.txt`` split into ``{title: body}`` at its ``=`` separator lines."""
    lines, sections, index = text.splitlines(), {}, 0
    separator = "=" * 80
    while index < len(lines):
        if lines[index] == separator and index + 2 < len(lines) and lines[index + 2] == separator:
            title, index, body = lines[index + 1], index + 3, []
            while index < len(lines) and lines[index] != separator:
                body.append(lines[index])
                index += 1
            sections[title] = "\n".join(body)
        else:
            index += 1
    return sections


def _mask_volatile(value):
    if isinstance(value, dict):
        masked = {key: _mask_volatile(item) for key, item in value.items()}
        manifest = masked.get("manifest_ref")
        if isinstance(manifest, dict) and "sha256" in manifest:
            masked["manifest_ref"] = {**manifest, "sha256": "<manifest_sha256>"}
        if "compressed_size" in masked:  # the gzip header names the temporary file
            masked["compressed_size"] = "<compressed_size>"
        return masked
    if isinstance(value, list):
        return [_mask_volatile(item) for item in value]
    return value


def normalized(value, fixture: dict[str, str]):
    """Packet JSON with the fixture's commit/tree/diff identities named and the
    write-time fields (timestamps, manifest digests) masked."""
    text = json.dumps(value, sort_keys=True, ensure_ascii=False)
    for key in ("base_sha", "head_sha", "head_tree_sha", "diff_sha256"):
        text = text.replace(fixture[key], f"<{key}>")
    data = _mask_volatile(json.loads(text))
    if isinstance(data, dict):
        for key in ("reviewed_at", "elapsed_sec"):
            if key in data:
                data[key] = f"<{key}>"
    return data
