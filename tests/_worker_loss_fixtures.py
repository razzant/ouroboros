"""Shared worker-loss double: the exception a scripted call raises to end its worker, which no ordinary handler absorbs.

Moved from ``tests/test_stream_timeout_recovery.py`` so test modules do not import each other
(Development 02, "Documentation contract").
"""


class WorkerLost(BaseException):
    pass
