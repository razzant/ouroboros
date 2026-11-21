"""Run the real model loop against a script of replies with configured fallback routes."""
import copy
from dataclasses import replace
import queue

import ouroboros.loop as loop_mod
import ouroboros.loop_transport as loop_transport
from ouroboros import usage_accounting as ua
from ouroboros.loop import run_llm_loop
from ouroboros.tools.registry import ToolRegistry
from tests._context_fit_plan import fit_plan


OK_RESPONSE = ({"role": "assistant", "content": "done"}, {"prompt_tokens": 1, "completion_tokens": 1})


class ScriptedLLM:
    """chat() returns the scripted replies in order (then OK), adopting a receipt when given one."""

    def __init__(self, *script, receipts=()):
        self.script, self.receipts, self.requests = list(script), list(receipts), []

    @property
    def calls(self):
        return len(self.requests)

    def default_model(self):
        return "test-model"

    def chat(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        if self.receipts:
            ua.adopt_physical_attempt_capture(self.receipts.pop(0))
        step = self.script.pop(0) if self.script else OK_RESPONSE
        if callable(step):
            raise step()
        return copy.deepcopy(step)


def loop_kwargs(tmp_path, llm, notes):
    return dict(messages=[{"role": "user", "content": "go"}], tools=ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path),
                llm=llm, drive_logs=tmp_path, emit_progress=lambda text, **_meta: notes.append(text),
                incoming_messages=queue.Queue(), task_id="t-exhaust-loop", drive_root=tmp_path)


def walk_loop(tmp_path, monkeypatch, *script, fallbacks="other/model", reachable=None, notes=None, seen=None):
    """The real loop with the real configured-route walk (a route switch keeps the captured core). ``seen``
    receives the loop's arguments, so a caller reads the registry's transcript after the walk."""
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", fallbacks)
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    monkeypatch.setattr(loop_transport, "interruptible_wait_sleep", lambda _sec, _wake: False)
    monkeypatch.setattr(loop_transport, "upstream_transport_reachable", lambda *_a, **_kw: reachable or {})

    def rebind(plan_, tools, _messages, *, model, use_local, **kwargs):
        rebound = replace(plan_, model=model, route_fp=f"route-{model}")
        tools._ctx.context_fit_plan = rebound
        return rebound, "max"

    monkeypatch.setattr(loop_mod, "_rebind_context_fit_plan", rebind)
    llm = ScriptedLLM(*script)
    kwargs = loop_kwargs(tmp_path, llm, [] if notes is None else notes)
    kwargs["tools"]._ctx.context_fit_plan = fit_plan(preferred="max", window=1_000_000, known=True)
    if seen is not None:
        seen.update(kwargs)
    return (llm, *run_llm_loop(**kwargs))
