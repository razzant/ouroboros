import json
import os
import sys
import unittest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestLocalToolCallParsing(unittest.TestCase):
    def test_parses_pure_tool_call_blocks(self):
        from ouroboros.llm import LLMClient

        msg = {
            "content": """
<tool_call>
{"name": "read_file", "arguments": {"path": "README.md"}}
</tool_call>
<tool_call>
{"name": "write_file", "arguments": {"path": "notes.txt", "content": "hello"}}
</tool_call>
""",
            "tool_calls": [],
        }

        parsed = LLMClient._parse_tool_calls_from_content(
            msg,
            {"read_file", "write_file"},
        )

        self.assertEqual(len(parsed["tool_calls"]), 2)
        self.assertIsNone(parsed["content"])
        self.assertEqual(parsed["tool_calls"][0]["function"]["name"], "read_file")
        self.assertEqual(
            json.loads(parsed["tool_calls"][0]["function"]["arguments"]),
            {"path": "README.md"},
        )

    def test_rejects_mixed_prose_and_tool_calls(self):
        from ouroboros.llm import LLMClient

        msg = {
            "content": """
Sure, I will use the tool now.

<tool_call>
{"name": "read_file", "arguments": {"path": "README.md"}}
</tool_call>
""",
            "tool_calls": [],
        }

        parsed = LLMClient._parse_tool_calls_from_content(msg, {"read_file"})

        self.assertEqual(parsed, msg)

    def test_rejects_unknown_tool_names(self):
        from ouroboros.llm import LLMClient

        msg = {
            "content": """
<tool_call>
{"name": "repo_delete_everything", "arguments": {}}
</tool_call>
""",
            "tool_calls": [],
        }

        parsed = LLMClient._parse_tool_calls_from_content(msg, {"read_file"})

        self.assertEqual(parsed, msg)

    def test_rejects_non_object_arguments(self):
        from ouroboros.llm import LLMClient

        msg = {
            "content": """
<tool_call>
{"name": "read_file", "arguments": "README.md"}
</tool_call>
""",
            "tool_calls": [],
        }

        parsed = LLMClient._parse_tool_calls_from_content(msg, {"read_file"})

        self.assertEqual(parsed, msg)

    def test_parses_double_brace_qwen_format(self):
        """Qwen 3B copies Jinja2 template and outputs {{...}} instead of {...}."""
        from ouroboros.llm import LLMClient

        msg = {
            "content": '<tool_call>\n{{"name": "write_file", "arguments": {"content": "hello world", "path": "test.txt"}}}\n</tool_call>',
            "tool_calls": [],
        }

        parsed = LLMClient._parse_tool_calls_from_content(msg, {"write_file"})

        self.assertEqual(len(parsed["tool_calls"]), 1)
        self.assertIsNone(parsed["content"])
        self.assertEqual(parsed["tool_calls"][0]["function"]["name"], "write_file")
        args = json.loads(parsed["tool_calls"][0]["function"]["arguments"])
        self.assertEqual(args["content"], "hello world")
        self.assertEqual(args["path"], "test.txt")


class TestStripReasoningWrappers(unittest.TestCase):
    """Tests for LLMClient._strip_reasoning_wrappers."""

    def test_strips_think_block(self):
        from ouroboros.llm import LLMClient
        text = "<think>let me reason</think>\n<tool_call>{}</tool_call>"
        cleaned, reasoning = LLMClient._strip_reasoning_wrappers(text)
        self.assertNotIn("<think>", cleaned)
        self.assertEqual(reasoning, "let me reason")
        self.assertIn("<tool_call>", cleaned)

    def test_strips_reasoning_block(self):
        from ouroboros.llm import LLMClient
        text = "<reasoning>deep thoughts</reasoning>\n<tool_call>{}</tool_call>"
        cleaned, reasoning = LLMClient._strip_reasoning_wrappers(text)
        self.assertNotIn("<reasoning>", cleaned)
        self.assertEqual(reasoning, "deep thoughts")

    def test_no_wrapper_returns_unchanged(self):
        from ouroboros.llm import LLMClient
        text = "<tool_call>{}</tool_call>"
        cleaned, reasoning = LLMClient._strip_reasoning_wrappers(text)
        self.assertEqual(cleaned, text)
        self.assertEqual(reasoning, "")

    def test_multiple_think_blocks_concatenated(self):
        from ouroboros.llm import LLMClient
        text = "<think>first</think><think>second</think><tool_call>{}</tool_call>"
        cleaned, reasoning = LLMClient._strip_reasoning_wrappers(text)
        self.assertIn("first", reasoning)
        self.assertIn("second", reasoning)
        self.assertNotIn("<think>", cleaned)

    def test_empty_think_block(self):
        from ouroboros.llm import LLMClient
        text = "<think></think><tool_call>{}</tool_call>"
        cleaned, reasoning = LLMClient._strip_reasoning_wrappers(text)
        self.assertEqual(reasoning, "")
        self.assertIn("<tool_call>", cleaned)

    def test_case_insensitive(self):
        from ouroboros.llm import LLMClient
        text = "<THINK>reasoning</THINK><tool_call>{}</tool_call>"
        cleaned, reasoning = LLMClient._strip_reasoning_wrappers(text)
        self.assertEqual(reasoning, "reasoning")
        self.assertNotIn("<THINK>", cleaned)


class TestParseToolCallsWithThink(unittest.TestCase):
    """Tests for _parse_tool_calls_from_content with Qwen3 think blocks."""

    def test_parses_think_plus_tool_call(self):
        """Qwen3 canonical output: <think>...</think><tool_call>...</tool_call>."""
        from ouroboros.llm import LLMClient

        msg = {
            "content": (
                "<think>I need to read the file first.</think>\n"
                '<tool_call>\n{"name": "read_file", "arguments": {"path": "README.md"}}\n</tool_call>'
            ),
            "tool_calls": [],
        }
        parsed = LLMClient._parse_tool_calls_from_content(msg, {"read_file"})

        self.assertEqual(len(parsed["tool_calls"]), 1)
        self.assertEqual(parsed["tool_calls"][0]["function"]["name"], "read_file")
        # reasoning preserved in content
        self.assertEqual(parsed["content"], "I need to read the file first.")

    def test_reasoning_preserved_in_content(self):
        """content should be the think-text, not None."""
        from ouroboros.llm import LLMClient

        reasoning_text = "Let me check the path."
        msg = {
            "content": (
                f"<think>{reasoning_text}</think>"
                '<tool_call>\n{"name": "read_file", "arguments": {"path": "f.py"}}\n</tool_call>'
            ),
            "tool_calls": [],
        }
        parsed = LLMClient._parse_tool_calls_from_content(msg, {"read_file"})

        self.assertIsNotNone(parsed.get("content"))
        self.assertEqual(parsed["content"], reasoning_text)

    def test_empty_think_block_content_is_none(self):
        """Empty <think> block → content=None (falsy, same as before)."""
        from ouroboros.llm import LLMClient

        msg = {
            "content": (
                "<think></think>"
                '<tool_call>\n{"name": "read_file", "arguments": {"path": "f.py"}}\n</tool_call>'
            ),
            "tool_calls": [],
        }
        parsed = LLMClient._parse_tool_calls_from_content(msg, {"read_file"})

        self.assertEqual(len(parsed["tool_calls"]), 1)
        self.assertIsNone(parsed["content"])  # empty reasoning → None

    def test_mixed_prose_without_think_still_rejected(self):
        """Safety guard: prose without a think wrapper is NOT stripped → rejected."""
        from ouroboros.llm import LLMClient

        msg = {
            "content": (
                "Sure, I'll do that now.\n"
                '<tool_call>\n{"name": "read_file", "arguments": {"path": "f.py"}}\n</tool_call>'
            ),
            "tool_calls": [],
        }
        parsed = LLMClient._parse_tool_calls_from_content(msg, {"read_file"})

        # Must remain unchanged — safety guard in effect
        self.assertEqual(parsed, msg)

    def test_unknown_tool_inside_think_wrapper_rejected(self):
        """Unknown tool name inside a think-wrapped response is still rejected."""
        from ouroboros.llm import LLMClient

        msg = {
            "content": (
                "<think>I should delete everything.</think>\n"
                '<tool_call>\n{"name": "nuke_all", "arguments": {}}\n</tool_call>'
            ),
            "tool_calls": [],
        }
        parsed = LLMClient._parse_tool_calls_from_content(msg, {"read_file"})

        self.assertEqual(parsed, msg)

    def test_malformed_json_inside_think_wrapper_rejected(self):
        """Malformed JSON inside a think-wrapped tool_call is rejected."""
        from ouroboros.llm import LLMClient

        msg = {
            "content": (
                "<think>plan</think>\n"
                "<tool_call>\nnot valid json\n</tool_call>"
            ),
            "tool_calls": [],
        }
        parsed = LLMClient._parse_tool_calls_from_content(msg, {"read_file"})

        self.assertEqual(parsed, msg)

    def test_plain_tool_call_no_think_content_is_none(self):
        """Without think wrapper, content=None (original behaviour unchanged)."""
        from ouroboros.llm import LLMClient

        msg = {
            "content": '<tool_call>\n{"name": "read_file", "arguments": {"path": "f.py"}}\n</tool_call>',
            "tool_calls": [],
        }
        parsed = LLMClient._parse_tool_calls_from_content(msg, {"read_file"})

        self.assertEqual(len(parsed["tool_calls"]), 1)
        self.assertIsNone(parsed["content"])

    def test_literal_think_inside_tool_argument_not_stripped(self):
        """<think> and <reasoning> text that appears inside a JSON argument value MUST
        NOT be stripped — they are valid argument content, not model reasoning tags.
        This is a regression guard against the regex running over tool-call payloads.
        """
        from ouroboros.llm import LLMClient
        import json as _json

        # The argument value itself contains literal <think>...</think> text.
        arg_value = "<think>literal tag in arg</think>"
        msg = {
            "content": (
                f'<tool_call>\n{{"name": "write_file", "arguments": {{"content": "{arg_value}", "path": "out.txt"}}}}\n</tool_call>'
            ),
            "tool_calls": [],
        }
        parsed = LLMClient._parse_tool_calls_from_content(msg, {"write_file"})

        self.assertEqual(len(parsed["tool_calls"]), 1, "Tool call should be parsed")
        args = _json.loads(parsed["tool_calls"][0]["function"]["arguments"])
        self.assertEqual(
            args["content"],
            arg_value,
            "Literal <think> tag inside JSON argument must NOT be stripped",
        )

    def test_think_wrapper_then_literal_think_in_argument(self):
        """Reasoning wrapper before tool_call is stripped; literal <think> inside JSON arg is preserved."""
        from ouroboros.llm import LLMClient
        import json as _json

        arg_value = "<think>doc example</think>"
        msg = {
            "content": (
                "<think>model reasoning goes here</think>\n"
                f'<tool_call>\n{{"name": "write_file", "arguments": {{"content": "{arg_value}"}}}}\n</tool_call>'
            ),
            "tool_calls": [],
        }
        parsed = LLMClient._parse_tool_calls_from_content(msg, {"write_file"})

        self.assertEqual(len(parsed["tool_calls"]), 1)
        # Reasoning from the wrapper block is in content
        self.assertEqual(parsed["content"], "model reasoning goes here")
        # Argument value is untouched
        args = _json.loads(parsed["tool_calls"][0]["function"]["arguments"])
        self.assertEqual(args["content"], arg_value)


# The reply Qwen3.8-27B-UD-Q4_K_M gave on 12 November 2026 to the product's own tool-calling probe
# (local_model.test_tool_calling), served by ouroboros.local_model_server on llama-cpp-python 0.3.36: its template
# opens <think> in the prompt, so the reasoning only closes, and it writes the call in the XML form (#3255).
QWEN38_REPLY = ("The user is asking for the current time and specifically requesting that I use the get_time tool. "
                "Let me call it.\n</think>\n\n<tool_call>\n<function=get_time>\n</function>\n</tool_call>")
READ_SCHEMA = {"read_file": {"type": "object", "properties": {
    "path": {"type": "string"}, "limit": {"type": "integer"}, "follow": {"type": "boolean"},
    "options": {"type": "object"}}}}


class TestCurrentQwenToolCalls(unittest.TestCase):
    """Qwen3.5 and later: reasoning that only closes, and <function=…><parameter=…> calls."""

    def test_the_real_qwen38_reply_is_a_tool_call_and_its_reasoning_stays_reasoning(self):
        from ouroboros.llm import LLMClient

        parsed = LLMClient._parse_tool_calls_from_content({"content": QWEN38_REPLY, "tool_calls": []}, {"get_time"})
        self.assertEqual([(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in parsed["tool_calls"]],
                         [("get_time", {})])
        self.assertTrue(parsed["content"].startswith("The user is asking for the current time"))
        self.assertNotIn("</think>", parsed["content"])

    def test_xml_values_take_the_types_the_tool_declares(self):
        from ouroboros.llm import LLMClient

        reply = ("<tool_call>\n<function=read_file>\n<parameter=path>\n42\n</parameter>\n<parameter=limit>\n5\n</parameter>\n"
                 "<parameter=follow>\ntrue\n</parameter>\n<parameter=options>\n{\"encoding\": \"utf-8\"}\n</parameter>\n"
                 "</function>\n</tool_call>")
        parsed = LLMClient._parse_tool_calls_from_content({"content": reply}, {"read_file"}, READ_SCHEMA)
        self.assertEqual(json.loads(parsed["tool_calls"][0]["function"]["arguments"]),
                         {"path": "42", "limit": 5, "follow": True, "options": {"encoding": "utf-8"}})
        # Undeclared: a bare number or word stays the text the model wrote; only JSON structure is structure.
        bare = LLMClient._parse_tool_calls_from_content({"content": reply}, {"read_file"})
        self.assertEqual(json.loads(bare["tool_calls"][0]["function"]["arguments"]),
                         {"path": "42", "limit": "5", "follow": "true", "options": {"encoding": "utf-8"}})

    def test_an_xml_value_may_span_lines_as_a_written_file_does(self):
        from ouroboros.llm import LLMClient

        reply = ("<tool_call>\n<function=write_file>\n<parameter=path>\nnotes.md\n</parameter>\n<parameter=content>\n"
                 "# Notes\n\nFirst line.\nSecond line.\n\n</parameter>\n</function>\n</tool_call>")
        parsed = LLMClient._parse_tool_calls_from_content({"content": reply}, {"write_file"})
        # The template writes "\nVALUE\n" around a value: exactly one newline goes on each side, the rest is the file.
        self.assertEqual(json.loads(parsed["tool_calls"][0]["function"]["arguments"]),
                         {"path": "notes.md", "content": "# Notes\n\nFirst line.\nSecond line.\n"})

    def test_reasoning_that_only_closes_does_not_hide_a_json_call(self):
        from ouroboros.llm import LLMClient

        reply = "I should look first.\n</think>\n<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"a\"}}\n</tool_call>"
        parsed = LLMClient._parse_tool_calls_from_content({"content": reply}, {"read_file"})
        self.assertEqual(parsed["tool_calls"][0]["function"]["name"], "read_file")
        self.assertEqual(parsed["content"], "I should look first.")

    def test_prose_beside_or_between_calls_keeps_the_reply_text(self):
        from ouroboros.llm import LLMClient

        call = "<tool_call>\n<function=get_time>\n</function>\n</tool_call>"
        for reply in (f"Sure, here it is.\n{call}", f"{call}\nAnd then:\n{call}", f"{call} done"):
            msg = {"content": reply, "tool_calls": []}
            self.assertIs(LLMClient._parse_tool_calls_from_content(msg, {"get_time"}), msg, reply)
        self.assertEqual(len(LLMClient._parse_tool_calls_from_content(
            {"content": f"{call}\n{call}"}, {"get_time"})["tool_calls"]), 2)

    def test_an_unknown_tool_or_stray_text_inside_the_function_is_refused(self):
        from ouroboros.llm import LLMClient

        for reply in ("<tool_call>\n<function=rm_rf>\n</function>\n</tool_call>",
                      "<tool_call>\n<function=get_time>\nnow please\n</function>\n</tool_call>",
                      "<tool_call>\n<function=get_time>\n</tool_call>"):
            msg = {"content": reply, "tool_calls": []}
            self.assertIs(LLMClient._parse_tool_calls_from_content(msg, {"get_time"}), msg, reply)


    def test_a_json_argument_may_hold_the_closing_tag_itself(self):
        from ouroboros.llm import LLMClient

        body = {"name": "write_file", "arguments": {"path": "t.py", "content": "x = '</tool_call>'\n"}}
        reply = "<tool_call>\n" + json.dumps(body) + "\n</tool_call>"
        parsed = LLMClient._parse_tool_calls_from_content({"content": reply}, {"write_file"})
        self.assertEqual(json.loads(parsed["tool_calls"][0]["function"]["arguments"]), body["arguments"])
        msg = {"content": reply + " and done", "tool_calls": []}
        self.assertIs(LLMClient._parse_tool_calls_from_content(msg, {"write_file"}), msg)

    def test_a_parameter_that_may_be_a_string_keeps_the_text_written(self):
        from ouroboros.llm import LLMClient

        reply = '<tool_call>\n<function=note>\n<parameter=body>\n{"a": 1}\n</parameter>\n</function>\n</tool_call>'
        schemas = {"note": {"type": "object", "properties": {"body": {"anyOf": [{"type": "string"}, {"type": "null"}]}}}}
        parsed = LLMClient._parse_tool_calls_from_content({"content": reply}, {"note"}, schemas)
        self.assertEqual(json.loads(parsed["tool_calls"][0]["function"]["arguments"]), {"body": '{"a": 1}'})
        typed = {"note": {"type": "object", "properties": {"body": {"anyOf": [{"type": "object"}, {"type": "null"}]}}}}
        parsed = LLMClient._parse_tool_calls_from_content({"content": reply}, {"note"}, typed)
        self.assertEqual(json.loads(parsed["tool_calls"][0]["function"]["arguments"]), {"body": {"a": 1}})


class TestSplitLeadingReasoning(unittest.TestCase):

    def test_leading_reasoning_in_either_shape_is_split_off_and_other_text_is_left_whole(self):
        from ouroboros.llm_local import _split_leading_reasoning as split

        self.assertEqual(split("<think>plan</think>\n\nHello."), ("Hello.", "plan"))
        self.assertEqual(split("plan\n</think>\n\nHello."), ("Hello.", "plan"))
        self.assertEqual(split("Hello."), ("Hello.", ""))
        mention = "Tags look like <think>x</think> in Qwen."
        self.assertEqual(split(mention), (mention, ""))


def test_the_local_lane_answers_without_its_reasoning_and_calls_tools_the_new_way(tmp_path, monkeypatch):
    """Through the lane itself: a plain answer arrives without the model's reasoning (kept as reasoning_content),
    and a current-Qwen XML call with typed parameters becomes a structured tool call."""
    import copy
    from types import SimpleNamespace

    from ouroboros import llm_local, local_model, usage_accounting as ua
    from ouroboros.llm import LLMClient

    replies = ["I am greeting my human.\n</think>\n\nHello. I am Ouroboros.",
               "Read it.\n</think>\n<tool_call>\n<function=read_file>\n<parameter=path>\nREADME.md\n</parameter>\n"
               "<parameter=limit>\n3\n</parameter>\n</function>\n</tool_call>",
               # An empty reasoning block, written whole (Qwen3 told not to think) or only closed (Qwen3.5 and later).
               "<think>\n\n</think>\n\nHi.", "\n</think>\n\nHi again.", "\n</think>\n",
               # Reasoning the server already separated is the server's: the answer is left as it came.
               {"content": "Qwen closes it so:\n</think>\nThen it answers.", "reasoning_content": "plan"}]

    class Response:
        def __init__(self, content):
            fields = content if isinstance(content, dict) else {"content": content}
            self.body = {"choices": [{"message": {"role": "assistant", **fields}, "finish_reason": "stop"}],
                         "usage": {"prompt_tokens": 1, "completion_tokens": 1}}

        def model_dump(self):
            return copy.deepcopy(self.body)

    client = LLMClient(api_key="unused")
    monkeypatch.setattr(local_model, "get_manager", lambda: SimpleNamespace(serving_context_evidence=lambda: {}))
    monkeypatch.setattr(llm_local, "local_context_limits", lambda maximum: (0, maximum))
    monkeypatch.setattr(client, "_get_local_client", lambda: SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=lambda **payload: Response(replies.pop(0))))))
    tools = [{"type": "function", "function": {"name": "read_file", "description": "Read a file.",
                                                "parameters": READ_SCHEMA["read_file"]}}]
    with ua.usage_scope(ua.UsageScope(drive_root=tmp_path, task_id="local-qwen")):
        answer, _ = client.chat(messages=[{"role": "user", "content": "hi"}], model="local", use_local=True, max_tokens=64)
        call, _ = client.chat(messages=[{"role": "user", "content": "read"}], model="local", use_local=True,
                              max_tokens=64, tools=tools)
        quiet = [client.chat(messages=[{"role": "user", "content": "hi"}], model="local", use_local=True,
                             max_tokens=64)[0] for _ in range(4)]
    assert answer["content"] == "Hello. I am Ouroboros." and answer["reasoning_content"] == "I am greeting my human."
    # No bare tag reaches the owner; a reply that was only a closed empty block is empty, so the loop's recovery sees it.
    assert [m["content"] for m in quiet] == ["Hi.", "Hi again.", "", "Qwen closes it so:\n</think>\nThen it answers."]
    assert [m.get("reasoning_content") for m in quiet] == [None, None, None, "plan"]
    assert json.loads(call["tool_calls"][0]["function"]["arguments"]) == {"path": "README.md", "limit": 3}
    assert call["content"] == "Read it."


def test_the_wizard_presets_and_the_factory_window_start_a_fresh_install():
    """The presets name the current Qwen generation and open the factory window, which a fresh install's first turn
    fits in (about 34K tokens of input beside the reply, measured on Qwen3.8-27B on 12 November 2026); a window of
    16,384 refused that turn on every preset."""
    from ouroboros.local_model import RUNTIME_PROBE, local_model_settings
    from ouroboros.settings_defaults import SETTINGS_DEFAULTS
    from ouroboros.settings_setup_contract import _LOCAL_PRESETS

    window = SETTINGS_DEFAULTS["LOCAL_MODEL_CONTEXT_LENGTH"]
    assert window == 65536
    assert {preset["contextLength"] for preset in _LOCAL_PRESETS.values()} == {window}
    assert all(preset["source"].startswith("unsloth/Qwen3.") and preset["filename"].endswith(".gguf")
               for preset in _LOCAL_PRESETS.values())
    assert local_model_settings({})["LOCAL_MODEL_CONTEXT_LENGTH"] == window
    assert local_model_settings({"LOCAL_MODEL_CONTEXT_LENGTH": 0})["LOCAL_MODEL_CONTEXT_LENGTH"] == window
    assert local_model_settings({"LOCAL_MODEL_CONTEXT_LENGTH": 8192})["LOCAL_MODEL_CONTEXT_LENGTH"] == 8192
    # The runtime counts as installed only when the server it is started as can import.
    assert "llama_cpp.server.app" in RUNTIME_PROBE



class TestReplayedCallsAndReasoningShapes(unittest.TestCase):
    """The second round of a tool task, and the reasoning shapes both paths must read alike."""

    def test_a_replayed_call_renders_as_an_object_only_when_the_template_needs_one(self):
        from ouroboros.local_model_server import object_tool_arguments, render_tool_arguments_on_retry

        history = [{"role": "user", "content": "read it"},
                   {"role": "assistant", "content": "", "tool_calls": [{"id": "c0", "type": "function", "function": {
                       "name": "read_file", "arguments": '{"path": "README.md", "limit": 3}'}}]},
                   {"role": "tool", "tool_call_id": "c0", "content": "one"}]
        seen = []

        def iterating(_self, *, messages, **kwargs):  # a Qwen3.5+ template: tool_call.arguments|items
            arguments = messages[1]["tool_calls"][0]["function"]["arguments"]
            if not isinstance(arguments, dict):
                raise TypeError("Can only get item pairs from a mapping.")
            seen.append(arguments)
            return "rendered"

        def printing(_self, *, messages, **kwargs):  # a template that writes the text as it is
            seen.append(messages[1]["tool_calls"][0]["function"]["arguments"])
            return "rendered"

        self.assertEqual(render_tool_arguments_on_retry(iterating)(None, messages=history), "rendered")
        self.assertEqual(seen.pop(), {"path": "README.md", "limit": 3})
        self.assertEqual(render_tool_arguments_on_retry(printing)(None, messages=history), "rendered")
        self.assertEqual(seen.pop(), '{"path": "README.md", "limit": 3}')
        self.assertEqual(history[1]["tool_calls"][0]["function"]["arguments"], '{"path": "README.md", "limit": 3}')
        odd = [history[0], {"role": "assistant", "tool_calls": [{"function": {"name": "x", "arguments": "not json"}}]}]
        self.assertEqual(object_tool_arguments(odd), odd)
        with self.assertRaises(TypeError):  # a template error the conversion cannot change stays the error
            render_tool_arguments_on_retry(iterating)(None, messages=odd)

    def test_the_server_process_installs_the_retry_once(self):
        import sys
        import types
        from ouroboros.local_model_server import install_object_tool_arguments

        class Formatter:
            def __call__(self, *, messages, **kwargs):
                return messages

        package, module = types.ModuleType("llama_cpp"), types.ModuleType("llama_cpp.llama_chat_format")
        module.Jinja2ChatFormatter, package.llama_chat_format = Formatter, module
        saved = {name: sys.modules.get(name) for name in ("llama_cpp", "llama_cpp.llama_chat_format")}
        sys.modules.update({"llama_cpp": package, "llama_cpp.llama_chat_format": module})
        try:
            install_object_tool_arguments()
            first = Formatter.__call__
            install_object_tool_arguments()
            self.assertIs(Formatter.__call__, first)
            self.assertTrue(first._ouroboros_object_arguments)
        finally:
            for name, value in saved.items():
                if value is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = value

    def test_the_owned_server_installs_the_retry_before_it_serves(self):
        import sys
        import types
        from ouroboros import local_model_server

        class Formatter:
            def __call__(self, *, messages, **kwargs):
                return messages

        served = []
        package, chat_format = types.ModuleType("llama_cpp"), types.ModuleType("llama_cpp.llama_chat_format")
        server_package, server_main = types.ModuleType("llama_cpp.server"), types.ModuleType("llama_cpp.server.__main__")
        chat_format.Jinja2ChatFormatter, package.llama_chat_format = Formatter, chat_format
        server_main.create_app = lambda *a, **k: None
        server_main.main = lambda: served.append(getattr(Formatter.__call__, "_ouroboros_object_arguments", False))
        package.server, server_package.__main__ = server_package, server_main
        names = ("llama_cpp", "llama_cpp.llama_chat_format", "llama_cpp.server", "llama_cpp.server.__main__")
        saved = {name: sys.modules.get(name) for name in names}
        sys.modules.update(dict(zip(names, (package, chat_format, server_package, server_main))))
        try:
            local_model_server.main()
            self.assertEqual(served, [True], "the vendor server starts with the retry already in place")
        finally:
            for name, value in saved.items():
                if value is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = value

    def test_reasoning_that_mentions_a_call_still_ends_at_its_own_closing_line(self):
        from ouroboros.llm import LLMClient
        from ouroboros.llm_local import _split_leading_reasoning as split

        reply = ("I will answer with a <tool_call> block naming get_time.\n</think>\n\n"
                 "<tool_call>\n<function=get_time>\n</function>\n</tool_call>")
        parsed = LLMClient._parse_tool_calls_from_content({"content": reply}, {"get_time"})
        self.assertEqual(parsed["tool_calls"][0]["function"]["name"], "get_time")
        self.assertTrue(parsed["content"].startswith("I will answer with a <tool_call> block"))
        mention = "Qwen ends its reasoning with </think> and then answers."
        self.assertEqual(split(mention), (mention, ""))
        self.assertEqual(split("checking\n</think>\nIt is noon."), ("It is noon.", "checking"))

    def test_odd_schemas_never_raise_and_a_declared_boolean_reads_python_style(self):
        from ouroboros.llm import LLMClient

        reply = "<tool_call>\n<function=f>\n<parameter=flag>\nFalse\n</parameter>\n</function>\n</tool_call>"
        for schemas in ({"f": {"type": "object", "properties": ["flag"]}}, {"f": {"properties": {"flag": {"type": 5}}}},
                        {"f": {"properties": {"flag": {"anyOf": {"type": "string"}}}}}, ["not a mapping"]):
            parsed = LLMClient._parse_tool_calls_from_content({"content": reply}, {"f"}, schemas)
            self.assertEqual(json.loads(parsed["tool_calls"][0]["function"]["arguments"]), {"flag": "False"}, schemas)
        typed = {"f": {"type": "object", "properties": {"flag": {"type": "boolean"}}}}
        parsed = LLMClient._parse_tool_calls_from_content({"content": reply}, {"f"}, typed)
        self.assertEqual(json.loads(parsed["tool_calls"][0]["function"]["arguments"]), {"flag": False})
