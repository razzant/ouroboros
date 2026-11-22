"""Scripted assistant replies shared by transport and delivery tests."""
import json


MIXED_REFUSAL_PARTS = [
    {"type": "text", "text": "First.\n"},
    {"type": "refusal", "refusal": "FINAL ANSWER: sample"},
    {"type": "text", "text": "\nLast."},
]
MIXED_REFUSAL_TEXT = "First.\nFINAL ANSWER: sample\nLast."


def finish(answer=None, *, action='finish', **kw):
    arguments = {'action': action, **kw}
    if answer is not None:
        arguments['answer'] = answer
    return {'role': 'assistant', 'content': None, 'tool_calls': [{
        'id': 'complete', 'type': 'function',
        'function': {'name': 'finish_task', 'arguments': json.dumps(arguments)},
    }]}
