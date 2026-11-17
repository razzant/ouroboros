"""First contact with a local llama.cpp-like server and no window in Settings (P1, 15.11.2026).

On the stand (llama.cpp b12421, Qwen3.5-9B, ``-c 80000``) a fresh install's first Main request carried the
whole Max view, the server answered HTTP 400 ``exceed_context_size_error`` and the person saw "no usable
response" 25 s later. Here the real loop, LLMClient, OpenAI SDK and physical ledger talk over loopback HTTP to a
fake server that counts a request's tokens from its bytes and answers the way llama.cpp does: the exact
refusal body over its window, an ordinary answer under it. The server publishes its window (``meta.n_ctx``) or
hides it, as an older build does.
"""
from __future__ import annotations

import json

import pytest

from tests._first_contact_server import FakeLlamaServer
from tests._first_contact import ANSWER, _episode, _run

N_CTX = 80_000


def _main_posts(server):
    return [post for post in server.posts if post["tools"]]  # Main carries tools; a Light helper call does not


def test_a_published_window_sizes_the_very_first_request(tmp_path, monkeypatch):
    with FakeLlamaServer(n_ctx=N_CTX, publish_window=True) as server:
        episode = _episode(tmp_path, monkeypatch, server)
        assert episode.plan.initial_mode == "nano"  # the server's own n_ctx, read at the start
        text, _usage, _trace = _run(episode)
    print("PUBLISHED_POSTS", json.dumps(server.posts), "GETS", json.dumps(server.gets))
    assert ANSWER in str(text)
    assert not any(post.get("refused") for post in server.posts)
    assert all(post["n_prompt"] < N_CTX for post in server.posts)


def test_a_hidden_window_is_learned_from_the_refusal_and_answered_in_the_same_turn(tmp_path, monkeypatch):
    from ouroboros import capability_evidence as ce

    with FakeLlamaServer(n_ctx=N_CTX, publish_window=False) as server:
        episode = _episode(tmp_path, monkeypatch, server)
        assert episode.plan.initial_mode == "max"  # nothing told the start the window
        text, usage, _trace = _run(episode)
    main = _main_posts(server)
    print("FIRST_CONTACT_POSTS", json.dumps(server.posts), "GETS", json.dumps(server.gets))
    assert main[0].get("refused") and main[0]["n_prompt"] > N_CTX  # the stand's first request, refused
    assert ANSWER in str(text), (text, server.posts)
    assert not main[-1].get("refused") and main[-1]["n_prompt"] < N_CTX  # the same turn's retry fitted
    assert episode.tools._ctx.active_context_mode == "nano"
    windows = ce._read(ce.canonical_evidence_root()).get("main_windows", {})
    assert any(row.get("refused") == N_CTX for record in windows.values() for row in record.get("observations", []))
    assert "/props" in server.gets  # the hidden window was looked for on the server's own path first


@pytest.mark.parametrize("status,payload,kind", [
    (400, {"error": {"code": 400, "message": "Prompt contains invalid tokens", "type": "invalid_request_error"}},
     "bad_request"),
    (429, {"error": {"code": 429, "message": "Too many requests", "type": "rate_limit_error"}}, "provider_transient"),
])
def test_another_refusal_of_the_first_request_is_not_a_window(tmp_path, monkeypatch, status, payload, kind):
    from ouroboros import capability_evidence as ce

    with FakeLlamaServer(n_ctx=N_CTX, publish_window=False, refuse_with=lambda _n: (status, payload)) as server:
        episode = _episode(tmp_path, monkeypatch, server)
        monkeypatch.setattr("ouroboros.loop_transport.interruptible_wait_sleep", lambda _seconds, wake: wake())
        text, usage, _trace = _run(episode)
    assert ANSWER not in str(text)
    assert all(post.get("refused") for post in _main_posts(server))
    if status == 429:
        assert len(_main_posts(server)) == 6
    windows = ce._read(ce.canonical_evidence_root()).get("main_windows", {})
    assert not any(row.get("refused") for record in windows.values() for row in record.get("observations", []))
    assert episode.tools._ctx.active_context_mode == "max"
