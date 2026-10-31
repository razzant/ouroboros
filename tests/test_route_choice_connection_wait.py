"""A route whose connection is gone walks the configured routes, then opens the model-wait card.

``ConnectionUnavailable`` (``llm_attempt``) is a typed pre-dispatch refusal: nothing upstream answered
and nothing was paid. The round first tries the configured routes (the existing walk), then the task's
own model-wait card opens with the new reason ``connection_unavailable`` — the owner's call (switch the
role, retry once the key is back, or stop) — and never a task failure for that cause alone. A Presence
turn, which may not wait, keeps its typed temporary refusal as before.
"""

from __future__ import annotations

import queue

import pytest

import ouroboros.loop as loop_mod
from ouroboros import model_wait
from ouroboros import usage_accounting as ua
from ouroboros.llm_attempt import ConnectionUnavailable
from ouroboros.loop import run_llm_loop
from ouroboros.outcomes import RESULT_INFRA_FAILED
from ouroboros.tools.registry import ToolRegistry
from tests._shared import saved_settings

PRIMARY, FALLBACK, HEALTHY = "openrouter@credits::primary/model", "openai::fallback", "openai::healthy"


class _RouteLLM:
    """A route named in ``gone`` refuses before dispatch ($0); any other answers (no physical ledger here)."""

    def __init__(self, root, gone):
        self.root, self.sent, self.gone = root, [], set(gone)

    def default_model(self):
        return PRIMARY

    @model_wait.model_waitable
    def chat(self, **kwargs):
        model = kwargs["model"]
        self.sent.append(model)
        if model in self.gone:
            raise ConnectionUnavailable("openrouter@credits", "disabled")
        return {"role": "assistant", "content": f"answer from {model}"}, {"prompt_tokens": 1, "completion_tokens": 1}


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    root = tmp_path / "data"
    (root / "state").mkdir(parents=True)
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    monkeypatch.setenv("TOTAL_BUDGET", "100")
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setattr(loop_mod, "_rebind_context_fit_plan", lambda plan, *_a, **_kw: (plan, "max"))
    monkeypatch.setattr(loop_mod, "_server_web_allowed_by_task", lambda _ctx: False)
    from ouroboros import fallback_cooldown
    monkeypatch.setattr(fallback_cooldown, "is_cooling_down", lambda *_a: False)
    return root


def _run(tmp_path, llm, *, task_type="", notes):
    from ouroboros.task_results import write_task_result

    write_task_result(tmp_path, "t-1a", "running")  # the wait owner's task row: the owner-pause fence reads it
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    task = {"id": "t-1a", "_attempt": 1, "drive_root": str(tmp_path), **({"_presence_turn": True} if task_type == "presence" else {})}
    with ua.usage_scope(ua.UsageScope(drive_root=llm.root, task_id="t-1a", root_task_id="t-1a", global_limit_usd=100)), \
            model_wait.task_model_wait_scope(task=task, drive_root=tmp_path, event_queue=None, worker_slot_held=False,
                                             owner_control=lambda: "") as owner:
        registry._ctx.model_wait_context = owner
        owner.tool_context = registry._ctx
        text, usage, trace = run_llm_loop(
            messages=[{"role": "user", "content": "go"}], tools=registry, llm=llm, drive_logs=tmp_path,
            emit_progress=lambda text, *, incident=None, **_kw: notes.append(text), incoming_messages=queue.Queue(),
            task_id="t-1a", task_type=task_type, drive_root=tmp_path)
    return text, usage, trace, registry


def test_a_gone_connection_walks_the_chain_then_waits_for_the_owner_instead_of_failing(data_root, tmp_path, monkeypatch):
    saved_settings(monkeypatch, tmp_path, OUROBOROS_MODEL=PRIMARY, OUROBOROS_MODEL_FALLBACKS=FALLBACK, OPENAI_API_KEY="key")
    asked = []

    def owner_switches(self, llm, error, kwargs, caller_cancel=None):
        asked.append((model_wait.model_wait_reason(error), kwargs["model"], kwargs["model_role"], getattr(error, "reason", "")))
        self.overrides[kwargs["model_role"]] = {"model": HEALTHY, "use_local": False, "model_account_override": ""}
        return {**kwargs, **self.overrides[kwargs["model_role"]]}

    monkeypatch.setattr(model_wait.TaskModelWait, "wait", owner_switches)
    llm, notes = _RouteLLM(data_root, gone={PRIMARY, FALLBACK}), []
    text, usage, _trace, registry = _run(tmp_path, llm, notes=notes)

    assert text == f"answer from {HEALTHY}"
    assert llm.sent == [PRIMARY, FALLBACK, HEALTHY]  # the walk first, then the owner's choice re-sends the retained call
    assert asked == [("connection_unavailable", PRIMARY, "main", "disabled")]
    assert usage.get("execution_status") != RESULT_INFRA_FAILED and not usage.get("resource_refusal")
    assert registry._ctx.route_choice == {"kind": "explicit", "source": "wait_card"}
    assert any("Fallback:" in note and "refusal before any provider was reached" in note for note in notes)


def test_without_configured_routes_the_card_opens_on_the_refusal_itself(data_root, tmp_path, monkeypatch):
    saved_settings(monkeypatch, tmp_path, OUROBOROS_MODEL=PRIMARY, OUROBOROS_MODEL_FALLBACKS="", OPENAI_API_KEY="key")
    asked = []

    def owner_retries_after_re_enabling(self, llm, error, kwargs, caller_cancel=None):
        asked.append(model_wait.model_wait_reason(error))
        llm.gone.discard(kwargs["model"])  # the owner re-enabled the connection in Settings and pressed Try again
        return kwargs

    monkeypatch.setattr(model_wait.TaskModelWait, "wait", owner_retries_after_re_enabling)
    llm, notes = _RouteLLM(data_root, gone={PRIMARY}), []
    text, usage, _trace, registry = _run(tmp_path, llm, notes=notes)

    assert text == f"answer from {PRIMARY}" and llm.sent == [PRIMARY, PRIMARY]
    assert asked == ["connection_unavailable"] and usage.get("execution_status") != RESULT_INFRA_FAILED
    assert registry._ctx.route_choice == {"kind": "following", "role": "main"}  # a retry is not a choice


def test_a_presence_turn_keeps_its_typed_temporary_refusal(data_root, tmp_path, monkeypatch):
    saved_settings(monkeypatch, tmp_path, OUROBOROS_MODEL=PRIMARY, OUROBOROS_MODEL_FALLBACKS="", OPENAI_API_KEY="key")
    monkeypatch.setattr(model_wait.TaskModelWait, "wait",
                        lambda *_a, **_kw: pytest.fail("a Presence turn holds its conversation slot and never waits"))
    llm, notes = _RouteLLM(data_root, gone={PRIMARY}), []
    _text, usage, _trace, _registry = _run(tmp_path, llm, task_type="presence", notes=notes)
    assert llm.sent == [PRIMARY]
    assert usage["resource_refusal"]["reason"] == "connection_unavailable" and usage["resource_refusal"]["temporary"] is True
