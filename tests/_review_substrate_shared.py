"""The recording FakeLLM shared by the review-substrate suites.

Split out of ``tests/test_review_substrate_v2.py`` when that module was divided
by theme; the stub is verbatim, so every sibling suite drives the substrate
through the same recording transport it was written against.
"""

import json


class FakeLLM:
    def __init__(self):
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        body = {
            "verdict": "PASS",
            "findings": [],
            "summary": f"reviewed by {kwargs['model']}",
        }
        return {"content": json.dumps(body)}, {"prompt_tokens": 10, "completion_tokens": 5}
