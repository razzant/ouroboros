"""Forked workers must not rotate the supervisor's server.log.

CyberGym r8 (2026-09-04): 64 workers inherited the server's RotatingFileHandler
and rotated the same file on their own byte counts; the renames raced and the
log collapsed into 2 KB / 198 B / 739 B files, losing hours of forensics.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler, WatchedFileHandler

from supervisor.workers import _demote_inherited_rotating_log_handlers


class _Marker(logging.Filter):
    def filter(self, record):  # noqa: D401
        return True


def test_rotating_handler_is_replaced_by_a_watched_handler_on_the_same_file(tmp_path):
    root = logging.getLogger()
    before = list(root.handlers)
    path = tmp_path / "server.log"
    rotating = RotatingFileHandler(path, maxBytes=10, backupCount=2, encoding="utf-8")
    rotating.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    rotating.setLevel(logging.WARNING)
    marker = _Marker()
    rotating.addFilter(marker)
    stream = logging.StreamHandler()
    root.addHandler(rotating)
    root.addHandler(stream)
    try:
        _demote_inherited_rotating_log_handlers()

        assert not any(isinstance(h, RotatingFileHandler) for h in root.handlers)
        assert stream in root.handlers  # untouched
        watched = [h for h in root.handlers if isinstance(h, WatchedFileHandler)]
        assert len(watched) == 1
        handler = watched[0]
        assert handler.baseFilename == str(path)
        assert handler.level == logging.WARNING
        assert handler.formatter is rotating.formatter
        assert marker in handler.filters

        logging.getLogger("demotion-test").warning("x" * 100)
        handler.flush()
        # Far past maxBytes=10, yet nothing rotated: the worker never renames.
        assert path.exists() and not (tmp_path / "server.log.1").exists()
        assert "WARNING " + "x" * 100 in path.read_text(encoding="utf-8")
    finally:
        for h in list(root.handlers):
            if h not in before:
                root.removeHandler(h)
                h.close()


def test_noop_without_rotating_handlers():
    root = logging.getLogger()
    before = list(root.handlers)
    _demote_inherited_rotating_log_handlers()
    assert root.handlers == before
