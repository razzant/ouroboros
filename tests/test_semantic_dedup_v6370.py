"""v6.37.0 guards for the semantic-dedup primitive (C9.6) and its consumers
(C9.1/C9.3/C9.5/C10.1/C10.2). The LLM is mocked everywhere — these assert the
structural contracts (fail-open, exact-id validation, fingerprint redirect, global
backlog routing, typed artifact field), not model quality. The backlog's own
semantic redirect, merge writer and ranked digest are retired with its host
writers: the backlog is one authored note."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


# --------------------------------------------------------------------------- #
# C9.6 — the primitive: fail-open, exact-id validation, confidence gate
# --------------------------------------------------------------------------- #

def _fake_llm(monkeypatch, content):
    import ouroboros.config as cfg
    import ouroboros.llm as llm_mod

    class FakeClient:
        def chat(self, **kw):
            return ({"content": content}, {})

    monkeypatch.setattr(llm_mod, "LLMClient", FakeClient)
    monkeypatch.setattr(cfg, "get_light_model", lambda: "fake-model")


def test_primitive_fail_open_without_candidates_or_text():
    from ouroboros.semantic_dedup import find_semantic_duplicate_id

    assert find_semantic_duplicate_id("x", [], subject="s", call_type="t") is None
    assert find_semantic_duplicate_id("", [{"id": "a", "text": "y"}], subject="s", call_type="t") is None


def test_primitive_returns_exact_high_confidence_match(monkeypatch):
    _fake_llm(monkeypatch, '{"duplicate_id":"obl-1","confidence":"high","reason":"same"}')
    from ouroboros.semantic_dedup import find_semantic_duplicate_id

    got = find_semantic_duplicate_id(
        "new bug", [{"id": "obl-1", "text": "old bug"}], subject="s", call_type="t", drive_root=None
    )
    assert got == "obl-1"


def test_primitive_rejects_low_confidence(monkeypatch):
    _fake_llm(monkeypatch, '{"duplicate_id":"obl-1","confidence":"medium","reason":"maybe"}')
    from ouroboros.semantic_dedup import find_semantic_duplicate_id

    assert find_semantic_duplicate_id("n", [{"id": "obl-1", "text": "o"}], subject="s", call_type="t", drive_root=None) is None


def test_primitive_rejects_id_not_in_candidates(monkeypatch):
    _fake_llm(monkeypatch, '{"duplicate_id":"obl-999","confidence":"high","reason":"hallucinated"}')
    from ouroboros.semantic_dedup import find_semantic_duplicate_id

    assert find_semantic_duplicate_id("n", [{"id": "obl-1", "text": "o"}], subject="s", call_type="t", drive_root=None) is None


def test_primitive_fail_open_on_unparseable(monkeypatch):
    _fake_llm(monkeypatch, "not json at all")
    from ouroboros.semantic_dedup import find_semantic_duplicate_id

    assert find_semantic_duplicate_id("n", [{"id": "obl-1", "text": "o"}], subject="s", call_type="t", drive_root=None) is None


@pytest.mark.parametrize("observed", [False, True])
@pytest.mark.parametrize("chosen", ["omitted", "", None, "claudexor::codex=explicit-model"])
def test_the_default_light_call_reaches_the_engine_with_the_light_role(tmp_path, monkeypatch, observed, chosen):
    """#2575: the default model IS the Light slot, so its role-owned account goes with it;
    an explicit model keeps its unlabelled (Auto) semantics. Real client, fake engine."""
    from ouroboros import llm_claudexor as transport, pricing, usage_accounting as ua
    from ouroboros.model_slots import MODEL_ACCOUNTS_KEY
    from ouroboros.semantic_dedup import find_semantic_duplicate_id
    from tests.test_llm_claudexor import MODEL, Gateway, result

    root = tmp_path / "data"
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(root / "settings.json"))
    monkeypatch.setenv("TOTAL_BUDGET", "100")
    monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", MODEL)
    monkeypatch.setenv(MODEL_ACCOUNTS_KEY, json.dumps({"main": "main-account", "light": "light-account"}))
    monkeypatch.setattr(transport.config, "CLAUDEXOR_MODEL_POLL_INTERVAL_SEC", 0.001)
    monkeypatch.setattr(pricing, "_fetch_live_rows", lambda *_: pytest.fail("Claudexor must not fetch API pricing"))
    verdict = result()
    verdict["message"] = {"role": "assistant",
                          "content": '{"duplicate_id":"obl-1","confidence":"high","reason":"same"}'}
    gateway = Gateway([verdict])
    monkeypatch.setattr(transport, "ensure_owned_gateway", lambda **_kw: gateway)

    with ua.usage_scope(ua.UsageScope(drive_root=root, task_id="dedup-owner", root_task_id="dedup-owner")):
        got = find_semantic_duplicate_id(
            "new bug", [{"id": "obl-1", "text": "old bug"}], subject="s", call_type="obligation_dedup",
            drive_root=root if observed else None, **({} if chosen == "omitted" else {"model": chosen}))

    assert got == "obl-1" and len(gateway.uploads) == 1
    sent = gateway.uploads[0][0]
    explicit = bool(chosen and chosen != "omitted")
    assert sent["account"] == ({"mode": "auto"} if explicit else {"mode": "pin", "profileId": "light-account"})
    assert sent["model"] == ("explicit-model" if explicit else "exact-model")
    assert bool(list((root / "observability" / "calls" / "semantic_dedup").glob("*_request.json"))) is observed


# --------------------------------------------------------------------------- #
# C10.1 Fix A — global routing of the one authored backlog note
# --------------------------------------------------------------------------- #

def test_project_scoped_backlog_write_routes_to_global_store(tmp_path, monkeypatch):
    import ouroboros.config as cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path, raising=False)
    from ouroboros.knowledge import read_knowledge_note, resolve_knowledge_address
    from ouroboros.tools import knowledge as kn

    ctx = SimpleNamespace(
        drive_root=tmp_path, budget_drive_root="", project_id="proj-x", task_id="t1",
        drive_path=lambda rel: tmp_path / rel,
    )
    msg = kn._knowledge_write(ctx, "improvement-backlog", "# Improvement Backlog\n\n- routed globally\n",
                              mode="overwrite")
    assert msg.startswith("✅ Knowledge 'improvement-backlog'")
    # Landed in the GLOBAL store, NOT projects/proj-x/knowledge.
    note = read_knowledge_note(resolve_knowledge_address(tmp_path, "improvement-backlog", "global"))
    assert "routed globally" in note.text
    assert not (tmp_path / "projects" / "proj-x" / "knowledge" / "improvement-backlog.md").exists()


# --------------------------------------------------------------------------- #
# C9.5 — outcomes recovery keys on the TYPED artifact_registered flag
# --------------------------------------------------------------------------- #

def test_outcomes_recovery_uses_typed_artifact_flag():
    from ouroboros.outcomes import _classify_tool_errors

    trace = {"tool_calls": [
        {"tool": "run_command", "is_error": True, "status": "artifact_output_error",
         "args": {"cmd": "build", "outputs": ["dist/app"]}, "result": "preview truncated, no marker here"},
        {"tool": "run_command", "is_error": False, "status": "ok",
         "args": {"cmd": "build", "outputs": ["dist/app"]}, "artifact_registered": True, "result": "ok"},
    ]}
    classified = _classify_tool_errors(trace)
    # The error is recovered via the typed flag despite the prose preview lacking the marker.
    assert not classified["unresolved"]
    assert classified["recovered"]


# --------------------------------------------------------------------------- #
# C9.3 — review_state obligation redirect (off-lock map applied under lock)
# --------------------------------------------------------------------------- #

def test_compute_obligation_redirects_empty_without_candidates(tmp_path):
    from ouroboros.review_state import AdvisoryReviewState, compute_obligation_semantic_redirects

    state = AdvisoryReviewState()
    findings = [{"verdict": "FAIL", "severity": "critical", "item": "bug_1", "reason": "boom"}]
    assert compute_obligation_semantic_redirects(state, findings, repo_key="r", drive_root=tmp_path) == {}


def test_obligation_redirect_folds_into_existing(monkeypatch):
    from ouroboros.review_state import AdvisoryReviewState, CommitAttemptRecord

    state = AdvisoryReviewState()
    a1 = CommitAttemptRecord(
        ts="2026-06-18T00:00:00Z", commit_message="m1", status="blocked", repo_key="r", blocked=True,
        critical_findings=[{"verdict": "FAIL", "severity": "critical", "item": "bug_login", "reason": "npe on submit"}],
    )
    state.record_attempt(a1)
    opens = state.get_open_obligations(repo_key="r")
    assert len(opens) == 1
    target = opens[0].obligation_id

    # A reworded restatement misses the exact fingerprint; the off-lock map redirects it.
    a2 = CommitAttemptRecord(
        ts="2026-06-18T00:01:00Z", commit_message="m2", status="blocked", repo_key="r", blocked=True,
        critical_findings=[{"verdict": "FAIL", "severity": "critical", "item": "bug_login", "reason": "null pointer when the form is submitted"}],
    )
    from ouroboros.review_state import _make_obligation_fingerprint
    fp = _make_obligation_fingerprint("bug_login", "null pointer when the form is submitted")
    state.record_attempt(a2, semantic_redirects={fp: target})
    assert len(state.get_open_obligations(repo_key="r")) == 1  # folded, not a new obligation


# --------------------------------------------------------------------------- #
# C10.2 Fix B — the complete authored backlog reaches the evolution CONTEXT, never a digest
# --------------------------------------------------------------------------- #

def test_evolution_task_text_carries_no_backlog_digest(monkeypatch, tmp_path):
    import supervisor.evolution_lifecycle as el
    import supervisor.queue as sq
    from ouroboros.knowledge import resolve_knowledge_address, write_knowledge_note

    assert write_knowledge_note(resolve_knowledge_address(tmp_path, "improvement-backlog", "global"),
                                "- ctx-only backlog item").ok
    monkeypatch.setattr(el, "_read_evolution_campaign", lambda: {"status": "active", "id": "camp", "objective": "improve"})
    # build_evolution_task_text does `from supervisor import queue` locally, so the real
    # module attribute (not el.queue) is what it reads — isolate the drive root there.
    monkeypatch.setattr(sq, "DRIVE_ROOT", tmp_path, raising=False)

    text = el.build_evolution_task_text(1)
    assert "Improvement Backlog" not in text and "ctx-only backlog item" not in text
