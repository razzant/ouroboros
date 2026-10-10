"""A reviewer's inspection tool call runs under the loop's own timeout policy.

`review_native_episode` held the runtime's only unbounded `registry.execute`:
one wedged `query_code` kept a reviewer slot silent for 7 h 16 min while the
episode checked its clock only before a model send. The call is now bounded by
the SAME per-tool resolution every other tool call uses (`_get_tool_timeout`),
narrowed by whatever calendar/execution bound the reviewer inherited, and a
call that outlives its bound is ABANDONED: the reviewer is told in the standard
host text, the receipt is an error carrying `native_tool_abandoned`, and the
value the worker finally produces sources no receipt and no read coverage.
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pytest

import ouroboros.loop_tool_execution as loop_tool_execution
import ouroboros.review_native_episode as native_episode
from ouroboros.model_wait import execution_deadline_scope, monotonic_now
from ouroboros.review_execution import ReviewAssignment, ReviewRouteKind
from ouroboros.review_native_episode import NativeToolRoundReviewExecutor
from ouroboros.review_substrate import ReviewRequest, ReviewSlot
from tests.test_native_tool_round_executor import _VERDICT, _ScriptedLLM, _tool_call

_REAL_REGISTRY = native_episode.inspection_registry

# What the real reader stamps on the shared context after a read renders.
_STAMP = {
    "target": "/x/greeting.txt", "opened_path": "greeting.txt", "opened_root": "active_workspace",
    "first_line": 1, "end_line": 1, "total_lines": 1, "body_start": 0, "line_ends": [10],
}
_LATE_BODY = "late body from an abandoned read"


@pytest.fixture()
def repo(tmp_path):
    root = tmp_path / "subject"
    root.mkdir()
    (root / "greeting.txt").write_text("hello native reviewer\n", encoding="utf-8")
    (tmp_path / "custody").mkdir()
    return root


def _assignment(repo_dir):
    request = ReviewRequest(
        surface="multi_model_review", goal="review", task_id="t-native",
        session_root=str(repo_dir), session_task="Review the staged change; cite files.",
        policy={"output_contract": "JSON array of findings"}, no_proxy=True,
    )
    slot = ReviewSlot(slot_id="t1", model="openai/fake-reviewer", effort="low",
                      route=ReviewRouteKind.API_CHAT, subagent_id="api-critic")
    # A real custody root: the episode's own source store must succeed, or its
    # persistence gap would be the one this suite is reading.
    return ReviewAssignment(request=request, slot=slot, call_id="op-1",
                            custody_root=repo_dir.parent / "custody")


class _SlowRegistry:
    """The real inspection registry, with `read_file` held until released."""

    def __init__(self, inner, release, *, hold_sec):
        self._inner = inner
        self._release = release
        self._hold_sec = hold_sec
        self.returned = threading.Event()

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def execute_result(self, name, args):
        if name != "read_file":
            return self._inner.execute_result(name, args)
        self._release.wait(self._hold_sec)
        # A late worker stamps the shared reader view exactly as the real
        # reader would: the episode must not credit that stamp to anything.
        self._inner._ctx.last_read_view = dict(_STAMP)
        self.returned.set()
        from ouroboros.tools.tool_result import ToolResult
        return ToolResult(status="ok", code="OK", text=_LATE_BODY)


def _install_registry(monkeypatch, *, hold_sec):
    """Give the next episode a registry whose `read_file` hangs until released."""
    release, holder = threading.Event(), {}

    def factory(root, drive_root, task_id=""):
        registry, ctx, schemas = _REAL_REGISTRY(root, drive_root, task_id)
        holder["registry"], holder["ctx"] = _SlowRegistry(registry, release, hold_sec=hold_sec), ctx
        return holder["registry"], ctx, schemas

    monkeypatch.setattr(native_episode, "inspection_registry", factory)
    return release, holder


class _ReleasingLLM(_ScriptedLLM):
    """Frees the held tool the moment the model is asked a SECOND time, which can
    only happen after the first call was abandoned. An event, not a wall-clock
    guess: a timer that fired 0.3 s after the abandonment put the second call
    exactly on its own 0.3 s bound, and a slow CI runner lost that race."""

    def __init__(self, script, release):
        super().__init__(script)
        self._release = release

    def chat(self, **kwargs):
        if self.calls:
            self._release.set()
        return super().chat(**kwargs)


def _reading_script():
    return _ScriptedLLM([
        {"tool_calls": [_tool_call("read_file", {"path": "greeting.txt"})]},
        {"content": _VERDICT},
    ])


def _tool_messages(llm, call_index=1):
    return [m for m in llm.calls[call_index]["messages"] if m.get("role") == "tool"]


def test_wedged_reviewer_tool_is_abandoned_and_the_episode_continues(repo, monkeypatch):
    """The blocked case: a tool that outlives its bound never wedges the episode."""
    release, holder = _install_registry(monkeypatch, hold_sec=5.0)
    monkeypatch.setattr(loop_tool_execution, "_get_tool_timeout", lambda *_a, **_k: 0.3)
    llm = _reading_script()
    result = NativeToolRoundReviewExecutor(_assignment(repo), llm=llm).execute()

    assert result.raw_text == _VERDICT  # the episode CONTINUED past the wedged call
    usage = result.usage
    assert usage["native_rounds"] == 2
    receipt = usage["native_tool_receipts"][0]
    assert receipt["tool"] == "read_file"
    assert receipt["outcome"] == "error"
    assert receipt["source_gap"] == "native_tool_abandoned"
    assert "start_line" not in receipt and "opened_path" not in receipt
    assert usage["native_source_gap"] == "native_tool_abandoned"
    # The reviewer got the standard host text, not silence and not a verdict.
    assert _tool_messages(llm)[0]["content"].startswith("⚠️ TOOL_TIMEOUT (read_file)")

    release.set()
    assert holder["registry"].returned.wait(5), "the abandoned worker never settled"
    # The late value sources nothing: the receipts the episode published are final.
    assert usage["native_tool_receipts"][0]["outcome"] == "error"
    assert "text_sha256" not in usage["native_tool_receipts"][0]
    assert _LATE_BODY not in str(llm.calls[1]["messages"])


def test_fast_reviewer_tool_is_untouched_and_its_receipt_carries_duration(repo, monkeypatch):
    """The working case: the same bound leaves a returning read fully credited."""
    release, _holder = _install_registry(monkeypatch, hold_sec=5.0)
    release.set()  # the read returns at once
    monkeypatch.setattr(loop_tool_execution, "_get_tool_timeout", lambda *_a, **_k: 30)
    llm = _reading_script()
    result = NativeToolRoundReviewExecutor(_assignment(repo), llm=llm).execute()

    receipt = result.usage["native_tool_receipts"][0]
    assert receipt["outcome"] == "executed"
    assert "source_gap" not in receipt
    assert result.usage["native_source_gap"] == ""
    # One clock domain: an ISO wall stamp and the seconds elapsed on that clock.
    assert datetime.fromisoformat(receipt["started_at"]).tzinfo is not None
    assert isinstance(receipt["duration_sec"], float) and receipt["duration_sec"] >= 0.0
    assert _LATE_BODY in _tool_messages(llm)[0]["content"]


def test_tool_bound_is_the_loop_policy_for_this_registry_and_call(repo, monkeypatch):
    """The number comes from the main loop's own per-tool policy, resolved for
    THIS registry and THIS call's arguments — the episode mints none of its own."""
    release, holder = _install_registry(monkeypatch, hold_sec=5.0)
    release.set()
    seen = []
    real = loop_tool_execution._get_tool_timeout

    def spy(tools, tool_name, tool_args=None):
        seen.append((tools, tool_name, tool_args))
        return real(tools, tool_name, tool_args)

    monkeypatch.setattr(loop_tool_execution, "_get_tool_timeout", spy)
    NativeToolRoundReviewExecutor(_assignment(repo), llm=_reading_script()).execute()

    assert [(name, args) for _tools, name, args in seen] == [("read_file", {"path": "greeting.txt"})]
    assert seen[0][0] is holder["registry"]


def test_inherited_dispatch_deadline_narrows_the_tool_wait(repo, monkeypatch):
    """The inherited deadline narrows the tool wait, even if preparation spent it."""
    release, _holder = _install_registry(monkeypatch, hold_sec=3.0)
    llm = _reading_script()
    # The test owns and joins the actual executor even when an assertion fails.
    with ThreadPoolExecutor(max_workers=1) as worker, monkeypatch.context() as patch:
        patch.setattr(loop_tool_execution, "ThreadPoolExecutor", lambda **_kwargs: worker)
        try:
            with execution_deadline_scope(monotonic_now() + 1.0):
                result = NativeToolRoundReviewExecutor(_assignment(repo), llm=llm).execute()
            receipt = result.usage["native_tool_receipts"][0]
            assert receipt["outcome"] == "error" and receipt["source_gap"] == "native_tool_abandoned"
            bound = float(_tool_messages(llm)[0]["content"].split("exceeded ")[1].split("s limit")[0])
            # The episode prepares its registry after the inherited deadline starts.
            assert 0 <= bound <= 1.0 + 1e-9
        finally:
            release.set()

    release_again, _holder_again = _install_registry(monkeypatch, hold_sec=3.0)
    release_again.set()
    again = NativeToolRoundReviewExecutor(_assignment(repo), llm=_reading_script()).execute()
    assert again.usage["native_tool_receipts"][0]["outcome"] == "executed"


def test_abandoned_call_stops_the_episode_crediting_read_extents(repo, monkeypatch):
    """A worker loose on the shared reader view makes attribution unprovable:
    after an abandonment no further read extent is credited, so a late stamp
    can never become this episode's coverage."""
    release, _holder = _install_registry(monkeypatch, hold_sec=5.0)
    monkeypatch.setattr(loop_tool_execution, "_get_tool_timeout", lambda *_a, **_k: 0.3)
    llm = _ReleasingLLM([
        {"tool_calls": [_tool_call("read_file", {"path": "greeting.txt"}, "c1")]},
        {"tool_calls": [_tool_call("read_file", {"path": "greeting.txt"}, "c2")]},
        {"content": _VERDICT},
    ], release)
    result = NativeToolRoundReviewExecutor(_assignment(repo), llm=llm).execute()

    receipts = result.usage["native_tool_receipts"]
    assert receipts[0]["outcome"] == "error"
    assert receipts[1]["outcome"] == "executed"  # the second call DID return
    assert all("start_line" not in receipt for receipt in receipts)
    assert result.usage["native_source_gap"] == "native_tool_abandoned"
