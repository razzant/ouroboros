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


class TestDeepSeekDsmlParsing(unittest.TestCase):
    def _dsml(self, inner: str) -> str:
        from ouroboros.tool_call_markup import _DSML_MARK

        return f"<{_DSML_MARK}tool_calls>{inner}</{_DSML_MARK}tool_calls>"

    def test_well_formed_dsml_becomes_tool_calls(self):
        from ouroboros.llm import LLMClient
        from ouroboros.tool_call_markup import _DSML_MARK

        invoke = (
            f"<{_DSML_MARK}invoke name=\"read_file\">"
            f"<{_DSML_MARK}parameter name=\"path\" string=\"true\">README.md"
            f"</{_DSML_MARK}parameter>"
            f"</{_DSML_MARK}invoke>"
        )
        msg = {
            "content": self._dsml(invoke),
            "tool_calls": [],
        }
        parsed = LLMClient._parse_tool_calls_from_content(msg, {"read_file"})
        self.assertEqual(len(parsed["tool_calls"]), 1)
        self.assertEqual(parsed["tool_calls"][0]["function"]["name"], "read_file")
        self.assertEqual(
            json.loads(parsed["tool_calls"][0]["function"]["arguments"]),
            {"path": "README.md"},
        )

    def test_malformed_dsml_is_not_upgraded(self):
        from ouroboros.llm import LLMClient
        from ouroboros.tool_call_markup import _DSML_MARK, content_has_tool_markup

        broken = f"<{_DSML_MARK}tool_calls><{_DSML_MARK}invoke name=\"read_file\">broken"
        msg = {"content": broken, "tool_calls": []}
        self.assertTrue(content_has_tool_markup(broken))
        parsed = LLMClient._parse_tool_calls_from_content(msg, {"read_file"})
        self.assertFalse(parsed.get("tool_calls"))

    def test_valid_invoke_plus_unclosed_invoke_is_not_partially_upgraded(self):
        from ouroboros.llm import LLMClient

        content = (
            '<tool_calls><invoke name="read_file">'
            '<parameter name="path" string="true">README.md</parameter>'
            "</invoke>"
            '<invoke name="write_file"><parameter name="path" string="true">out.txt'
            "</tool_calls>"
        )
        msg = {"content": content, "tool_calls": []}

        parsed = LLMClient._parse_tool_calls_from_content(
            msg, {"read_file", "write_file"},
        )

        self.assertFalse(parsed.get("tool_calls"))
        self.assertEqual(parsed["content"], content)

    def test_valid_invoke_plus_truncated_tag_is_not_partially_upgraded(self):
        from ouroboros.llm import LLMClient

        valid = '<invoke name="read_file"></invoke>'
        for fragment in ("<invok", "</invok"):
            with self.subTest(fragment=fragment):
                content = f"<tool_calls>{valid}{fragment}</tool_calls>"
                msg = {"content": content, "tool_calls": []}

                parsed = LLMClient._parse_tool_calls_from_content(msg, {"read_file"})

                self.assertFalse(parsed.get("tool_calls"))
                self.assertEqual(parsed["content"], content)

    def test_mixed_tagged_and_plain_invoke_pair_is_not_upgraded(self):
        from ouroboros.llm import LLMClient
        from ouroboros.tool_call_markup import _DSML_MARK

        content = (
            f"<{_DSML_MARK}tool_calls>"
            f'<{_DSML_MARK}invoke name="read_file"></invoke>'
            f"</{_DSML_MARK}tool_calls>"
        )
        msg = {"content": content, "tool_calls": []}

        parsed = LLMClient._parse_tool_calls_from_content(msg, {"read_file"})

        self.assertFalse(parsed.get("tool_calls"))
        self.assertEqual(parsed["content"], content)

    def test_loop_wire_seam_promotes_well_formed_remote_dsml(self):
        from ouroboros.llm import LLMClient
        from ouroboros.tool_call_markup import _DSML_MARK, resolve_tool_markup

        invoke = (
            f"<{_DSML_MARK}invoke name=\"read_file\">"
            f"<{_DSML_MARK}parameter name=\"path\" string=\"true\">README.md"
            f"</{_DSML_MARK}parameter>"
            f"</{_DSML_MARK}invoke>"
        )
        client = LLMClient()
        target = {
            "provider": "openrouter",
            "usage_model": "deepseek/deepseek-v4-flash-0731",
            "resolved_model": "deepseek/deepseek-v4-flash-0731",
        }
        message, _usage = client._normalize_remote_response(
            {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": self._dsml(invoke),
                        "tool_calls": [],
                    }
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
            target,
            skip_cost_fetch=True,
        )
        self.assertFalse(message.get("tool_calls"))
        message, calls, _content, failure = resolve_tool_markup(
            message,
            [],
            message["content"],
            {},
            {"reasoning_notes": []},
            [{"type": "function", "function": {"name": "read_file"}}],
        )
        self.assertIsNone(failure)
        self.assertEqual(len(calls), 1)
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "read_file")

    def test_plain_dsml_preserves_literal_reasoning_tags_in_parameter(self):
        from ouroboros.llm import LLMClient

        literal = "<think>literal</think><reasoning>bytes</reasoning>"
        msg = {
            "content": (
                '<tool_calls><invoke name="write_file">'
                f'<parameter name="content" string="true">{literal}</parameter>'
                "</invoke></tool_calls>"
            ),
            "tool_calls": [],
        }
        parsed = LLMClient._parse_tool_calls_from_content(msg, {"write_file"})
        args = json.loads(parsed["tool_calls"][0]["function"]["arguments"])
        self.assertEqual(args["content"], literal)
        self.assertIsNone(parsed["content"])

    def test_prose_quoting_tool_markup_is_not_a_wire_envelope(self):
        from ouroboros.tool_call_markup import content_has_tool_markup

        self.assertFalse(
            content_has_tool_markup(
                "Document the literal <tool_call> tag without invoking anything."
            )
        )
        self.assertFalse(
            content_has_tool_markup(
                "Example: <tool_calls><invoke name=\"read_file\"></invoke></tool_calls>"
            )
        )

    def test_prefixed_executable_dsml_fails_closed_without_promotion(self):
        from ouroboros.tool_call_markup import (
            TOOL_MARKUP_PROTOCOL_FAIL_TEXT,
            resolve_tool_markup,
        )

        for markup in (
            '<tool_calls><invoke name="read_file"></invoke></tool_calls>',
            "<tool_calls><invok</tool_calls>",
        ):
            with self.subTest(markup=markup):
                content = f"I will use the tool now.\n{markup}"
                message = {"content": content, "tool_calls": []}
                resolved, calls, unchanged, failure = resolve_tool_markup(
                    message,
                    [],
                    content,
                    {},
                    {"reasoning_notes": []},
                    [{"type": "function", "function": {"name": "read_file"}}],
                )
                self.assertEqual(resolved, message)
                self.assertEqual(calls, [])
                self.assertEqual(unchanged, content)
                self.assertIsNotNone(failure)
                self.assertEqual(failure[0], TOOL_MARKUP_PROTOCOL_FAIL_TEXT)

    def test_prefixed_non_executable_tag_mention_remains_content(self):
        from ouroboros.tool_call_markup import resolve_tool_markup

        content = "Document the literal <tool_call> tag without invoking anything."
        message = {"content": content, "tool_calls": []}
        resolved, calls, unchanged, failure = resolve_tool_markup(
            message, [], content, {}, {"reasoning_notes": []}, [],
        )

        self.assertEqual(resolved, message)
        self.assertEqual(calls, [])
        self.assertEqual(unchanged, content)
        self.assertIsNone(failure)
