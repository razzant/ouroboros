"""CLI negotiation and consumption of additive task-event cursors."""
import io
import json
import urllib.error

import pytest

from ouroboros import cli


def cursor(seq):
    return {"v": 2, "seq": seq, "view": "view", "positions": {"root": {"progress": seq}}}


def event(seq, text="progress", **fields):
    return {"seq": seq, "type": "progress", "event_id": f"row-{seq}",
            "cursor": cursor(seq), "data": {"content": text}, **fields}


def terminal(seq):
    return event(seq, type="task_result", event_id="", data={"status": "completed", "result": "done"})


def http_error(code):
    failure = cli.CLIError(f"HTTP {code}")
    failure.__cause__ = urllib.error.HTTPError("http://test/events", code, "refused", {}, io.BytesIO())
    return failure


def test_client_stream_posts_json_without_cursor_in_url(monkeypatch):
    calls = []
    frame = event(1)
    def urlopen(request, timeout):
        calls.append((request, timeout))
        return io.BytesIO(("data: " + json.dumps(frame) + "\n\n").encode())
    monkeypatch.setattr(cli.urllib.request, "urlopen", urlopen)
    body = {"v": 2, "wait": 30, "cursor": cursor(0)}
    assert list(cli.OuroborosHTTPClient("http://test").stream_sse("/events", body=body)) == [frame]
    request, timeout = calls[0]
    assert request.get_method() == "POST"
    assert request.full_url == "http://test/events"
    assert json.loads(request.data) == body
    assert request.get_header("Content-type") == "application/json"


def test_watch_advances_consumed_checkpoint_and_deduplicates_view_replay(capsys):
    calls = []
    class Client:
        def stream_sse(self, path, timeout, *, body):
            calls.append(body)
            if len(calls) == 1:
                yield event(1, "once")
                yield event(1, type="cursor_checkpoint", event_id="", cursor=cursor(2), data={})
            else:
                assert body["cursor"] == cursor(2)
                yield event(2, type="cursor_replay", event_id="", data={})
                yield event(3, "once", event_id="row-1")
                yield event(4, "new")
                yield terminal(5)
    cli._watch_task(Client(), "task", jsonl=True, quiet=False, timeout_sec=0)
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [row["data"].get("content") for row in rows if row["type"] == "progress"] == ["once", "new"]
    assert rows[-1]["type"] == "task_result"


def test_watch_falls_back_only_after_first_connection_405():
    calls = []
    class Client:
        def stream_sse(self, path, timeout, **kwargs):
            calls.append((path, kwargs))
            if len(calls) == 1:
                raise http_error(405)
            yield {"seq": 1, "type": "task_result", "data": {"status": "completed"}}
    cli._watch_task(Client(), "task", jsonl=False, quiet=True, timeout_sec=0)
    assert calls[0][0].endswith("/events") and calls[0][1]["body"]["v"] == 2
    assert "cursor=0&wait=30" in calls[1][0] and not calls[1][1]


@pytest.mark.parametrize("code,after_frame", [(500, False), (405, True)])
def test_watch_does_not_fallback_on_other_failure_or_later_connection(code, after_frame):
    calls = []
    class Client:
        def stream_sse(self, path, timeout, *, body):
            calls.append(body)
            if after_frame and len(calls) == 1:
                yield event(1)
                return
            raise http_error(code)
    with pytest.raises(cli.CLIError, match=f"HTTP {code}"):
        cli._watch_task(Client(), "task", jsonl=False, quiet=True, timeout_sec=0)
    assert len(calls) == (2 if after_frame else 1)


def test_watch_refuses_unavailable_cursor_without_restarting():
    class Client:
        calls = 0
        def stream_sse(self, path, timeout, *, body):
            self.calls += 1
            yield event(1, type="error", error="shortened chain")
    client = Client()
    with pytest.raises(cli.CLIError, match="shortened chain"):
        cli._watch_task(client, "task", jsonl=False, quiet=True, timeout_sec=0)
    assert client.calls == 1


def test_watch_dedup_is_bounded_and_synthetic_results_are_always_consumed(capsys):
    class Client:
        def stream_sse(self, path, timeout, *, body):
            for seq in range(1, 4098):
                yield event(seq)
            yield event(4098, event_id="row-1")  # outside the recent window
            yield terminal(4099)
    cli._watch_task(Client(), "task", jsonl=True, quiet=False, timeout_sec=0)
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(rows) == 4099
    assert rows[-2]["event_id"] == "row-1" and rows[-1]["type"] == "task_result"
