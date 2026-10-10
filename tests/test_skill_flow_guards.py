"""Regression tests for skill authoring / repair guardrails."""

import hashlib
from types import SimpleNamespace
import queue

from ouroboros import loop as loop_mod
from ouroboros.contracts.task_constraint import TaskConstraint, normalize_task_constraint
from ouroboros.tool_access import active_tool_profile
from ouroboros.utils import sanitize_tool_args_for_log
from tests.test_subscription_main_wait import _without_context_facts
from tests.test_completion_selection import finish


def _completion_schema():
    from ouroboros.tools.control import get_tools
    entry = next(tool for tool in get_tools() if tool.name == "finish_task")
    return {"type": "function", "function": entry.schema}


def test_selected_skill_is_resource_context_not_a_reduced_profile():
    messages = [{"role": "user", "content": "Please run tests for task_constraint handling"}]
    plain = SimpleNamespace(messages=messages, task_constraint=None)

    constraint = TaskConstraint(mode="skill_repair", skill_name="target", payload_root="skills/external/target")
    assert constraint.has_selected_skill
    assert active_tool_profile(SimpleNamespace(messages=messages, task_constraint=constraint)) == active_tool_profile(plain)


def test_normalize_task_constraint_from_command_payload():
    constraint = normalize_task_constraint({
        "mode": "skill_repair",
        "skill_name": "target",
        "payload_root": "skills/external/target",
        "allow_enable": False,
    })
    assert constraint.mode == "skill_repair"
    assert constraint.skill_name == "target"
    assert constraint.payload_root == "skills/external/target"
    assert constraint.allow_enable is False


def test_normalize_task_constraint_strict_bool_and_local_readonly_canonicalization():
    repair = normalize_task_constraint({
        "mode": "skill_repair",
        "skill_name": "target",
        "payload_root": "skills/external/target",
        "allow_enable": "false",
        "allow_review": "0",
    })
    assert repair.allow_enable is False
    assert repair.allow_review is False

    readonly = normalize_task_constraint({
        "mode": "local_readonly_subagent",
        "skill_name": "ignored",
        "payload_root": "skills/external/ignored",
        "allow_enable": "true",
        "allow_review": "true",
    })
    assert readonly.mode == "local_readonly_subagent"
    assert readonly.skill_name == ""
    assert readonly.payload_root == ""
    assert readonly.allow_enable is False
    assert readonly.allow_review is False


def test_long_tool_args_log_as_placeholder_not_content_object():
    args = {"path": "skills/external/demo/plugin.py", "content": "x" * 4000}

    sanitized = sanitize_tool_args_for_log("write_file", args, threshold=100)

    assert isinstance(sanitized["content"], str)
    assert sanitized["content"].startswith("<TRUNCATED:content:")
    assert "content_len" not in sanitized


def test_skill_finalization_rearms_after_tool_round(monkeypatch, tmp_path):
    calls = iter([
        ({"content": "done", "tool_calls": []}, {}),
        ({"content": "", "tool_calls": [{"id": "c1", "function": {"name": "noop", "arguments": "{}"}}]}, {}),
        (finish("done again"), {}),
        (finish("final"), {}),
    ])
    progress = []
    seen_message_tails = []
    seen_messages = []

    class _Tools:
        CODE_TOOLS = set()

        def __init__(self):
            self._ctx = SimpleNamespace(
                event_queue=None,
                drive_root=tmp_path,
                task_id="task",
                messages=[],
                active_model_override=None,
                active_use_local_override=None,
                active_effort_override=None,
                _skill_finalization_injected=False,
            )

        def schemas(self):
            return [{"type": "function", "function": {"name": "noop", "description": "", "parameters": {}}}, _completion_schema()]

        def get_timeout(self, _name):
            return 1

        def execute(self, name, args):
            if name == "finish_task":
                from ouroboros.tools.control_runtime import _finish_task
                return _finish_task(self._ctx, **args)
            assert name == "noop"
            return "OK"

        def execute_result(self, name, args):
            # Typed dispatch seam (D02): adapt like the real registry.
            from ouroboros.tools.tool_result import LegacyTextResultAdapter

            return LegacyTextResultAdapter.from_text(name, self.execute(name, args))

        def override_handler(self, _name, _handler):
            return None

    class _LLM:
        def default_model(self):
            return "test-model"

    monkeypatch.setenv("OUROBOROS_MAX_ROUNDS", "6")
    monkeypatch.setattr(loop_mod, "_skill_finalization_message", lambda *_args, **_kwargs: "SKILL_NOT_FINALIZED")
    def fake_call(_llm, messages, *_args, **_kwargs):
        seen_message_tails.append([m.get("role") for m in messages[-3:]])
        seen_messages.append([dict(m) for m in messages])
        return next(calls)

    monkeypatch.setattr(loop_mod, "call_llm_with_retry", fake_call)

    result, _usage, trace = loop_mod.run_llm_loop(
        [{"role": "user", "content": "create skill"}],
        _Tools(),
        _LLM(),
        tmp_path,
        lambda text, *, incident=None: progress.append(text),
        queue.Queue(),
        task_id="task",
        drive_root=tmp_path,
    )

    assert result == "final"
    assert len(seen_messages) == 4
    assert progress.count("SKILL_NOT_FINALIZED") == 2
    assert trace["reasoning_notes"].count("SKILL_NOT_FINALIZED") == 2
    assert trace["delivery_candidate"]["revision"] == 3
    assert trace["delivery_candidate"]["finalization_control"] == "candidate"
    assert [row["tool"] for row in trace["tool_calls"]] == ["noop", "finish_task", "finish_task"]
    assert all(row["completion_control"] for row in trace["tool_calls"][1:])
    # Multiple host notices can follow one held response; the old last-two-role
    # assertion accidentally depended on the duplicated assistant row.
    for held, text in ((seen_messages[1], "done"), (seen_messages[3], "done again")):
        indices = [index for index, row in enumerate(held)
                   if row.get("role") == "assistant" and row.get("content") == text]
        assert len(indices) == 1
        notices = held[indices[0] + 1:]
        assert notices and all(row.get("role") == "user" for row in notices)
        assert any("latest whole held response answer_sha256=" + hashlib.sha256(text.encode()).hexdigest()
                   in str(row.get("content")) for row in notices)
    assert all(tail[-2:] != ["assistant", "system"] for tail in seen_message_tails)


def test_skill_action_and_effect_round_cannot_erase_complete_candidate(monkeypatch, tmp_path):
    original = "Complete skill delivery answer with all required details."
    responses = iter([
        ({"content": original, "tool_calls": []}, {}),
        ({"content": "", "tool_calls": [{
            "id": "finalize-1",
            "function": {"name": "finalize_skill", "arguments": "{}"},
        }]}, {}),
        ({"content": "Skill review completed.", "tool_calls": []}, {}),
        (finish(answer_sha256=hashlib.sha256(original.encode("utf-8")).hexdigest()), {}),
    ])
    finalized = {"value": False}
    seen_messages = []

    class _Tools:
        CODE_TOOLS = set()

        def __init__(self):
            self._ctx = SimpleNamespace(
                event_queue=None,
                drive_root=tmp_path,
                task_id="task",
                messages=[],
                active_model_override=None,
                active_use_local_override=None,
                active_effort_override=None,
                _skill_finalization_injected=False,
            )

        def schemas(self):
            return [{
                "type": "function",
                "function": {
                    "name": "finalize_skill",
                    "description": "",
                    "parameters": {},
                },
            }, _completion_schema()]

        def get_timeout(self, _name):
            return 1

        def execute(self, name, args):
            if name == "finish_task":
                from ouroboros.tools.control_runtime import _finish_task
                return _finish_task(self._ctx, **args)
            assert name == "finalize_skill"
            finalized["value"] = True
            return "OK"

        def execute_result(self, name, args):
            # Typed dispatch seam (D02): adapt like the real registry.
            from ouroboros.tools.tool_result import LegacyTextResultAdapter

            return LegacyTextResultAdapter.from_text(name, self.execute(name, args))

        def override_handler(self, _name, _handler):
            return None

    class _LLM:
        def default_model(self):
            return "test-model"

    def fake_call(_llm, messages, *_args, **_kwargs):
        seen_messages.append([dict(message) for message in messages])
        return next(responses)

    monkeypatch.setenv("OUROBOROS_MAX_ROUNDS", "7")
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setattr(
        loop_mod,
        "_skill_finalization_message",
        lambda *_args, **_kwargs: "" if finalized["value"] else "SKILL_NOT_FINALIZED",
    )
    monkeypatch.setattr(loop_mod, "call_llm_with_retry", fake_call)

    result, _usage, trace = loop_mod.run_llm_loop(
        [{"role": "user", "content": "create and finalize the skill"}],
        _Tools(),
        _LLM(),
        tmp_path,
        lambda _msg, *, incident=None, narration=False: None,
        queue.Queue(),
        task_id="task",
        drive_root=tmp_path,
    )

    assert finalized["value"] is True
    assert result == original
    assert len(seen_messages) == 4
    assert trace["delivery_candidate"]["revision"] == 1
    assert trace["delivery_candidate"]["content_sha256"] == hashlib.sha256(original.encode("utf-8")).hexdigest()
    assert trace["delivery_candidate"]["degraded"] is False
    assert trace["delivery_candidate"]["acceptance_binding"]["authoritative"] is False
    assert all(
        "[DELIVERY_FINALIZATION_CONTROL]" not in str(message.get("content") or "")
        for message in seen_messages[1]
    )
    assert any(row.get("role") == "tool" and row.get("content") == "OK" for row in seen_messages[2])
    assert seen_messages[3][:len(seen_messages[2])] == seen_messages[2]
    assert _without_context_facts(seen_messages[3])[-2] == {"role": "assistant", "content": "Skill review completed."}
    assert _without_context_facts(seen_messages[3])[-1]["role"] == "user"
    assert "No completion selection" in _without_context_facts(seen_messages[3])[-1]["content"]


def test_skill_finalization_empty_text_preserves_canonical_response_and_user_tail(monkeypatch, tmp_path):
    calls = iter([
        ({"content": "", "tool_calls": []}, {}),
        ({"content": "final", "tool_calls": []}, {}),
    ])
    seen_messages = []

    class _Tools:
        CODE_TOOLS = set()

        def __init__(self):
            self._ctx = SimpleNamespace(
                event_queue=None,
                drive_root=tmp_path,
                task_id="task",
                messages=[],
                active_model_override=None,
                active_use_local_override=None,
                active_effort_override=None,
                _skill_finalization_injected=False,
            )

        def schemas(self):
            return []

        def override_handler(self, _name, _handler):
            return None

    class _LLM:
        def default_model(self):
            return "test-model"

    def fake_call(_llm, messages, *_args, **_kwargs):
        seen_messages.append([dict(message) for message in messages])
        return next(calls)

    monkeypatch.setenv("OUROBOROS_MAX_ROUNDS", "3")
    monkeypatch.setattr(loop_mod, "_skill_finalization_message", lambda *_args, **_kwargs: "SKILL_NOT_FINALIZED")
    monkeypatch.setattr(loop_mod, "call_llm_with_retry", fake_call)

    result, _usage, _trace = loop_mod.run_llm_loop(
        [{"role": "user", "content": "create skill"}],
        _Tools(),
        _LLM(),
        tmp_path,
        lambda _msg, *, incident=None, narration=False: None,
        queue.Queue(),
        task_id="task",
        drive_root=tmp_path,
    )

    assert result == "final"
    assert len(seen_messages) == 2
    assert _without_context_facts(seen_messages[1])[-1]["role"] == "user"
    assert seen_messages[1][:len(seen_messages[0])] == seen_messages[0]
    assert _without_context_facts(seen_messages[1])[-2] == {"role": "assistant", "content": ""}
    assert "No completion selection" in _without_context_facts(seen_messages[1])[-1]["content"]
    assert hashlib.sha256(b"").hexdigest() in _without_context_facts(seen_messages[1])[-1]["content"]
