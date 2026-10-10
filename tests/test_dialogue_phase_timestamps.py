"""Dialogue phase timestamps are durable observations, independent of scheduling."""
import json
import queue
from datetime import datetime
from types import SimpleNamespace

from ouroboros.agent import OuroborosAgent
from ouroboros.loop_llm_call import call_llm_with_retry
from supervisor import message_bus


def test_acceptance_stamp_is_the_durable_append_not_a_relabelled_input_time(tmp_path):
    row = message_bus.log_chat('in', 0, 1, 'Hello', ts='2000-01-01T00:00:00+00:00',
                               source='web', client_message_id='message', drive_root=tmp_path, require_write=True)
    stored = json.loads((tmp_path / 'logs/chat.jsonl').read_text(encoding='utf-8'))
    assert stored['message_accepted_at'] == row['message_accepted_at']
    assert stored['message_accepted_at'] > stored['ts']


def test_activity_stamp_is_carried_from_the_emitted_event():
    events = queue.Queue()
    host = SimpleNamespace(_current_chat_id=0, _event_queue=events)
    OuroborosAgent._emit_live_log(host, 'task_started', task_id='turn')
    row = events.get_nowait()['data']
    assert row['activity_emitted_at'] == row['ts'] == host._activity_emitted_at


def test_first_request_and_answer_survive_subsequent_rounds(tmp_path):
    class LLM:
        def chat(self, **kw):
            return {'content': 'answer', 'tool_calls': []}, {'provider': 'openai-compatible', 'cost': 0.0}
    logs = tmp_path / 'logs'
    logs.mkdir()
    usage = {}
    for round_index in (1, 2):
        call_llm_with_retry(LLM(), [{'role': 'user', 'content': 'hello'}],
                            'openai-compatible::test', None, 'medium', 1, logs, 'turn', round_index, queue.Queue(), usage)
    rows = [json.loads(line) for line in (logs / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
    rounds = [row for row in rows if row['type'] == 'llm_round']
    assert len(rounds) == 2
    for key in ('first_request_at', 'first_answer_at'):
        assert rounds[0][key] == rounds[1][key] == usage[key]
        datetime.fromisoformat(usage[key])
    assert usage['first_request_at'] <= usage['first_answer_at'] <= rounds[0]['ts']
