"""Prompt rendering and the single review execution seam.

Split by theme out of ``tests/test_review_substrate_v2.py``. This module owns
the rendered prompt: the byte-level pre-seam goldens of the api_chat executor,
one render per slot, the typed undeliverable-route refusal and the absolute
default drive root. (The task-acceptance outcome-tier/independence prompt rules
left with task acceptance.)
"""

import json

from ouroboros.review_substrate import ReviewRequest, ReviewSlot, run_review_request

from tests._review_substrate_shared import FakeLLM

# --- v6.87.11: the single review execution seam (Phase 5.1 / 5.2) -------------

# Byte-level golden for the api_chat prompt rendering, captured by running the
# generator below against the PRISTINE pre-seam substrate (v6.87.5, ca76d76).
# The seam refactor is a pure move: every digest must still match. Regenerate
# ONLY together with a deliberate, reviewed prompt change:
#     for request, slot in _seam_prompt_cases():
#         sha256(json.dumps(_request_messages(request, slot),
#                           ensure_ascii=False, sort_keys=True).encode())
# The task_acceptance case (its own re-pinned digests, once indexes 2-3) left with
# the acceptance surface (owner 17.10, Q2-A); the four remaining digests are the
# unchanged pre-seam bytes.
_PRE_SEAM_PROMPT_DIGESTS = [
    "0261c7c7fe477ad7f8901a28bee1ad23905d40c3c62825d2bc406ecd9ca37f82",
    "9cf4de6f66001c3b4cec7fdd3d8552ecf83fc886004a7020e98a4c28c022c4e3",
    "fca0f9401e544e371338f20effa6206db783e7098ff4d11ee2a980ebbe81ecb0",
    "fca0f9401e544e371338f20effa6206db783e7098ff4d11ee2a980ebbe81ecb0",
]


def _seam_prompt_cases():
    generic = ReviewRequest(
        surface="commit_review",
        goal="Judge the staged change.\nSecond line.",
        scope="ouroboros/review_substrate.py",
        subject="diff --git a/x b/x\n+1\n",
        evidence={"files": ["a.py", "b.py"], "nested": {"k": [1, 2, {"deep": "ünicode"}]}},
        evidence_refs=[{"kind": "blob", "sha256": "deadbeef"}],
        checklist="- one\n- two",
        policy={"hardness": "hard_gate", "min_successful_slots": 2},
        task_id="task-1",
    )
    prebuilt = ReviewRequest(
        surface="scope_review",
        goal="Review the staged change and context above. Output ONLY a JSON array.",
        messages=[
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "STABLE",
                     "cache_control": {"type": "ephemeral", "ttl": "1h"}},
                    {"type": "text", "text": "DYNAMIC"},
                ],
            },
            {"role": "user", "content": "Review the staged change and context above."},
        ],
        task_id="task-3",
        call_type="scope_review",
        max_tokens=64000,
        temperature=0.2,
        no_proxy=True,
    )
    slots = [
        ReviewSlot(slot_id="slot_1", model="anthropic/claude-x", effort="high", role_hint="commit reviewer"),
        ReviewSlot(slot_id="slot_2", model="openai/gpt-x", effort="medium", role_hint=""),
    ]
    for request in (generic, prebuilt):
        for slot in slots:
            yield request, slot


def test_api_chat_executor_renders_pre_seam_bytes_exactly():
    """5.2: moving prompt assembly behind the seam is a PURE move — the executor
    reproduces the pre-seam bytes and cache markers exactly."""
    import hashlib

    from ouroboros.review_execution import (
        ApiChatReviewExecutor,
        ReviewAssignment,
        _request_messages,
    )

    digests = []
    for request, slot in _seam_prompt_cases():
        messages = ApiChatReviewExecutor(ReviewAssignment(request=request, slot=slot)).messages
        # Same SSOT renderer, same bytes.
        assert messages == _request_messages(request, slot)
        blob = json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
        digests.append(hashlib.sha256(blob).hexdigest())
    assert digests == _PRE_SEAM_PROMPT_DIGESTS

    # Cache segmentation survives verbatim: exactly one marked governance block
    # and one marked task-stable block, mutable tail unmarked, slot label last.
    request, slot = next(iter(_seam_prompt_cases()))
    system_blocks = ApiChatReviewExecutor(
        ReviewAssignment(request=request, slot=slot)
    ).messages[0]["content"]
    assert [bool(block.get("cache_control")) for block in system_blocks] == [True, True]


def test_prompt_record_keeps_request_slot_messages_shape(tmp_path):
    """The durable prompt record still carries request/slot/messages, in order,
    with the route's own projection supplying the last key."""
    llm = FakeLLM()
    run_review_request(
        ReviewRequest(surface="scope", goal="g", task_id="prompt-shape"),
        slots=[ReviewSlot(slot_id="s1", model="m")],
        drive_root=tmp_path,
        llm=llm,
    )
    import gzip

    blobs = sorted((tmp_path / "observability" / "blobs").glob("*.json.gz"))
    payloads = [json.loads(gzip.open(path, "rb").read().decode("utf-8")) for path in blobs]
    prompt_payloads = [p for p in payloads if isinstance(p, dict) and "messages" in p]
    assert prompt_payloads
    assert list(prompt_payloads[0]) == ["messages", "request", "slot"]  # sorted on disk
    assert prompt_payloads[0]["slot"]["route"] == "api_chat"


def test_slot_prompt_is_rendered_once_per_slot(tmp_path, monkeypatch):
    """5.2: the prompt record and both permitted physical sends share ONE lazy
    rendering — the substrate never re-assembles the pack per attempt. The second
    send is the P3 rail's one same-slot retry of a transport failure (the acceptance
    format-repair resend that once supplied it left with task acceptance)."""
    import ouroboros.review_execution as rx

    calls = {"n": 0}
    real = rx._request_messages

    def _counted(request, slot):
        calls["n"] += 1
        return real(request, slot)

    # Patch the OWNER module: the api_chat executor renders through it.
    monkeypatch.setattr(rx, "_request_messages", _counted)

    class RetryLLM:
        def __init__(self):
            self.sends = 0

        def chat(self, **kwargs):
            self.sends += 1
            if self.sends == 1:
                raise TimeoutError("transient timeout")
            return {"content": json.dumps({"verdict": "PASS", "findings": [], "summary": "ok"})}, {}

    llm = RetryLLM()
    result = run_review_request(
        ReviewRequest(
            surface="multi_model_review", goal="g", subject="diff",
            policy={"min_successful_slots": 1}, task_id="lazy-render",
            call_type="multi_model_review",
        ),
        slots=[ReviewSlot(slot_id="s1", model="m")],
        drive_root=tmp_path,
        llm=llm,
    )
    assert llm.sends == 2        # the P3 retry resends once on the same slot
    assert result.actors[0]["status"] == "ok"
    assert calls["n"] == 1       # rendered once for the record AND both sends


def test_undeliverable_route_is_a_typed_refusal_not_a_fallback(tmp_path):
    """5.1: a route that cannot deliver THIS slot (here: an agent_session slot
    whose surface supplied no session root/task) refuses on its own slot. It
    never silently falls back to another transport, and it never reaches a
    chat client."""
    from ouroboros.review_execution import (
        ReviewAssignment,
        ReviewRouteKind,
        ReviewRouteUnavailable,
        _execute_slot_attempt,
    )

    request = ReviewRequest(surface="scope", goal="g", task_id="route")
    slot = ReviewSlot(slot_id="s1", model="m", timeout_sec=5, route=ReviewRouteKind.AGENT_SESSION)
    assignment = ReviewAssignment(request=request, slot=slot)
    llm = FakeLLM()
    try:
        _execute_slot_attempt(assignment, llm=llm)
    except ReviewRouteUnavailable as exc:
        assert "agent_session" in str(exc)
    else:  # pragma: no cover - the seam must refuse
        raise AssertionError("unimplemented route must raise ReviewRouteUnavailable")
    assert llm.calls == []

    # The refusal is contained before dispatch: the panel stays honest and free.
    result = run_review_request(request, slots=[slot], drive_root=tmp_path, llm=llm)
    assert result.aggregate_signal == "DEGRADED"
    assert result.actors[0]["status"] == "not_dispatched"
    assert llm.calls == []


def test_route_kinds_carry_no_harness_names():
    """Part IV: only api_chat and agent_session ever exist in the core."""
    from ouroboros.review_execution import ReviewRouteKind

    assert {kind.value for kind in ReviewRouteKind} == {"api_chat", "agent_session"}
    assert ReviewSlot(slot_id="s", model="m").route is ReviewRouteKind.API_CHAT


def test_default_drive_root_is_the_absolute_config_root_never_cwd_relative(tmp_path, monkeypatch):
    """ISO-DRIP regression: the coordinator's shipped default was the RELATIVE
    ``../data`` — with any cwd under a repo/ that names the live data root's
    sibling, so default-constructed coordinators dripped synthetic review
    records into live observability (or, on trees with the absolute-root
    guard, silently LOST them into empty refs). The default must resolve to
    the absolute config SSOT: records really land there, and nothing is ever
    created relative to the cwd."""
    import ouroboros.config as config

    apphome = tmp_path / "apphome"
    repo = apphome / "repo"
    repo.mkdir(parents=True)
    configured = tmp_path / "configured_data"
    monkeypatch.setattr(config, "DATA_DIR", configured)
    monkeypatch.chdir(repo)

    class OkLLM:
        def chat(self, **kwargs):
            return {"content": "[]"}, {"prompt_tokens": 2, "completion_tokens": 1}

    result = run_review_request(
        ReviewRequest(surface="multi_model_review", goal="iso-drip probe", task_id="iso-drip"),
        slots=[ReviewSlot(slot_id="slot_1", model="api/m", timeout_sec=10)],
        drive_root=None,  # the shipped default under test
        llm=OkLLM(),
    )
    actor = result.actors[0]
    # The records were REALLY written (not swallowed into empty refs) ...
    assert actor["prompt_ref"].get("manifest_ref", {}).get("path")
    assert actor["response_ref"].get("manifest_ref", {}).get("path")
    # ... into the configured absolute root ...
    assert (configured / "observability").is_dir()
    # ... and never cwd-relative: no ../data sibling, nothing under the cwd.
    assert not (apphome / "data").exists()
    assert list(repo.iterdir()) == []
