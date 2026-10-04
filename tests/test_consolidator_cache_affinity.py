"""Light memory calls share one routing affinity per data root.

Adapted from the PR #1449 candidate (Codex, 2026-10-03) to the base consolidator:
the affinity rides the existing ``cache_affinity`` parameter and changes nothing
else in the request.
"""
from copy import deepcopy

import pytest

from ouroboros import consolidator as c
from ouroboros.llm import LLMClient
from ouroboros.tools.registry import ToolContext
from tests import test_consolidator_context_fit as fit_helpers

fit = fit_helpers.fit
_LLM = fit_helpers._LLM


class _Actor(_LLM):
    """Records every Light send and answers with a summary."""

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return {"content": f"summary-{len(self.calls)}"}, dict(self.usage)


def _run(actor, root, *, task="preparation", source="first offer", knowledge=True, metadata=None):
    context = ToolContext(repo_dir=root, drive_root=root, task_id=task)
    if metadata:
        context.task_metadata = metadata
    reader = c.KnowledgeReadContext(context) if knowledge else None
    return c._call_consolidation_llm(actor, "Summarize this source.\n" + source, "Room summary", knowledge=reader)


def _remote(call, model="openai/gpt-5.6-sol", *, affinity=True):
    return LLMClient(api_key="unused")._build_remote_kwargs(
        {"provider": "openrouter", "resolved_model": model, "usage_model": model,
         "supports_openrouter_extensions": True},
        call["messages"], call["reasoning_effort"], call["max_tokens"], "auto", None,
        call["tools"], skip_capability_fetch=True,
        cache_affinity=call["cache_affinity"] if affinity else "",
    )


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setattr(LLMClient, "_SUPPORTED_PARAMS_FETCHED", True, raising=False)
    monkeypatch.setattr("ouroboros.pricing._fetch_live_rows", lambda *_a, **_kw: {})


def test_source_choices_keep_session_and_all_other_request_fields(tmp_path, fit):
    import json
    import httpx
    import openai

    actor = _Actor()
    for root, task, source in [(tmp_path, "preparation", "first offer"),
                               (tmp_path, "preparation", "different offer"),
                               (tmp_path / "other", "preparation", "first offer"),
                               (tmp_path, "other-task", "first offer")]:
        root.mkdir(parents=True, exist_ok=True)
        assert _run(actor, root, task=task, source=source)[0]
    originals = deepcopy(actor.calls)
    payloads = [_remote(call) for call in actor.calls]
    sessions = [payload["extra_body"]["session_id"] for payload in payloads]
    assert sessions[0] == sessions[1] == sessions[3], "one session per data root, not per task or source"
    assert sessions[0] != sessions[2]
    assert _remote(actor.calls[0], "openai/gpt-5.5")["extra_body"]["session_id"] != sessions[0]
    wire = []

    def respond(request):
        wire.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "fixture", "object": "chat.completion", "created": 0,
            "model": "openai/gpt-5.6-sol", "choices": [{"index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": "done"}}]})

    with openai.OpenAI(api_key="fixture", base_url="https://fixture.invalid/v1", max_retries=0,
                      http_client=httpx.Client(transport=httpx.MockTransport(respond))) as sdk:
        for call, payload in zip(actor.calls, payloads):
            sdk.chat.completions.create(**payload)
            sdk.chat.completions.create(**_remote(call, affinity=False))
    for index, session in enumerate(sessions):
        assert wire[index * 2].pop("session_id") == session
        assert wire[index * 2] == wire[index * 2 + 1], "affinity changes nothing but the session"
    for call, payload in zip(actor.calls, payloads):
        del payload["extra_body"]["session_id"]
        assert payload == _remote(call, affinity=False)
    assert actor.calls == originals


def test_direct_openai_uses_the_affinity_as_prompt_cache_key_without_a_system(tmp_path, fit):
    actor = _Actor()
    assert _run(actor, tmp_path)[0]
    call = actor.calls[0]
    target = {"provider": "openai", "resolved_model": "gpt-5.6-sol", "usage_model": "openai/gpt-5.6-sol"}
    client = LLMClient(api_key="unused")
    keyed = client._build_remote_kwargs(target, call["messages"], "low", 128, "auto", None, call["tools"],
                                        skip_capability_fetch=True, cache_affinity=call["cache_affinity"])
    plain = client._build_remote_kwargs(dict(target), call["messages"], "low", 128, "auto", None, call["tools"],
                                        skip_capability_fetch=True)
    assert keyed["prompt_cache_key"] == LLMClient._explicit_cache_affinity_identity(
        "openai/gpt-5.6-sol", call["cache_affinity"])
    assert "prompt_cache_key" not in plain, "no leading system and no affinity: still no key"
    assert "cache_control" not in str(keyed["messages"]), "direct Chat Completions carries no markers"
    # A leading system still owns the key; the affinity is only the fallback.
    system = [{"role": "system", "content": "stable policy"}, *call["messages"]]
    with_system = client._build_remote_kwargs(dict(target), system, "low", 128, "auto", None, None,
                                              skip_capability_fetch=True, cache_affinity=call["cache_affinity"])
    assert with_system["prompt_cache_key"] == LLMClient._prompt_cache_identity("openai/gpt-5.6-sol", system)


def test_missing_context_identity_keeps_affinity_absent(tmp_path, fit):
    actor = _Actor()
    assert _run(actor, tmp_path, knowledge=False)[0]
    for call in actor.calls:
        assert call["cache_affinity"] == ""
        assert "session_id" not in _remote(call)["extra_body"]


def test_canonical_memory_root_unifies_task_workspaces_without_task_identity(tmp_path, fit):
    actor = _Actor()
    for task in ("one", "two", ""):
        root = tmp_path / task if task else tmp_path
        root.mkdir(parents=True, exist_ok=True)
        assert _run(actor, root, task=task, metadata={"budget_drive_root": str(tmp_path)})[0]
    assert len({_remote(call)["extra_body"]["session_id"] for call in actor.calls}) == 1


def test_reprepare_keeps_affinity_and_a_reroute_changes_the_session(tmp_path, fit, monkeypatch):
    from contextlib import contextmanager
    from ouroboros import model_wait

    class Waiter:
        overrides = {}
        waits_allowed = True

        @contextmanager
        def register_reprepare(self, role, callback):
            self.prepare = callback
            yield

    waiter = Waiter()
    monkeypatch.setattr(model_wait, "current_model_wait", lambda: waiter)
    actor = _Actor()
    assert _run(actor, tmp_path)[0]
    sent = actor.calls[-1]
    assert isinstance(sent["messages"][0]["content"], str), "the Light prompt shape is unchanged"
    changed = waiter.prepare({**sent, "model": "openai/gpt-5.5"})
    assert changed["cache_affinity"] == sent["cache_affinity"]
    assert _remote(changed, changed["model"])["extra_body"]["session_id"] != _remote(sent)["extra_body"]["session_id"]


@pytest.mark.parametrize("use_local", [False, True])
def test_compaction_summarizer_shares_one_affinity_per_data_root(tmp_path, monkeypatch, use_local):
    import json
    from ouroboros import context_compaction as cc
    from ouroboros import llm_observability

    seen = []

    def chat_observed(_client, **kwargs):
        seen.append(kwargs)
        if kwargs.get("tools"):
            raise RuntimeError("structured lane unavailable in this fixture")
        return {"content": json.dumps({"summaries": [{"source_id": part.source_id, "summary": "s"}]})}, {}

    monkeypatch.setattr(llm_observability, "chat_observed", chat_observed)
    part = cc._part("unit-1", "tool output " * 20)
    spec = {"model": "openai/gpt-6-sol", "use_local": use_local, "effort": "low", "output_budget": 256}
    for task in ("task-a", "task-b"):
        result = cc._call_summarizer([part], drive_root=tmp_path, task_id=task, phase="map", spec=spec,
                                     summary_budgets={part.root_id: 64}, usage_total={})
        assert result == {part.source_id: "s"}
    affinities = {call["cache_affinity"] for call in seen}
    assert affinities == ({""} if use_local else {f"context_compaction:{tmp_path}"})
