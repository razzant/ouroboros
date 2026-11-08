import os
import sys
import types
import unittest
from unittest.mock import patch


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class _FakeOpenAI:
    created = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        type(self).created.append(self)


class _FakeDefaultHttpxClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class TestLlmClientRefresh(unittest.TestCase):
    def setUp(self):
        _FakeOpenAI.created.clear()

    def test_construction_does_not_read_credentials(self):
        from ouroboros.llm import LLMClient

        with patch("ouroboros.llm.runtime_setting", side_effect=AssertionError("credential read")):
            LLMClient()

    def test_runtime_client_refreshes_when_env_key_changes(self):
        from ouroboros.llm import LLMClient

        fake_openai = types.SimpleNamespace(
            OpenAI=_FakeOpenAI, DefaultHttpxClient=_FakeDefaultHttpxClient,
        )
        with patch.dict(sys.modules, {"openai": fake_openai}):
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}, clear=False):
                client = LLMClient()
                first = client._get_client()

            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-new-key"}, clear=False):
                second = client._get_client()

        self.assertIsNot(first, second)
        self.assertEqual(len(_FakeOpenAI.created), 2)
        self.assertEqual(_FakeOpenAI.created[0].kwargs["api_key"], "")
        self.assertEqual(_FakeOpenAI.created[1].kwargs["api_key"], "sk-or-new-key")

    def test_explicit_api_key_does_not_track_env_changes(self):
        from ouroboros.llm import LLMClient

        fake_openai = types.SimpleNamespace(
            OpenAI=_FakeOpenAI, DefaultHttpxClient=_FakeDefaultHttpxClient,
        )
        with patch.dict(sys.modules, {"openai": fake_openai}):
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}, clear=False):
                client = LLMClient(api_key="explicit-key")
                first = client._get_client()

            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-new-key"}, clear=False):
                second = client._get_client()

        self.assertIs(first, second)
        self.assertEqual(len(_FakeOpenAI.created), 1)
        self.assertEqual(_FakeOpenAI.created[0].kwargs["api_key"], "explicit-key")
