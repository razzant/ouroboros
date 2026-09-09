"""
Ouroboros — self-modifying AI agent.

Philosophy: BIBLE.md
Architecture: agent.py (orchestrator), tools/ (plugin tools),
              llm.py (LLM client), memory.py (memory), review.py (deep review),
              utils.py (shared utilities).
"""

# IMPORTANT: Do NOT import agent/loop/llm/etc here!
# Eager imports here persist in worker processes as stale code,
# preventing hot-reload. Workers import make_agent directly.

__all__ = ['agent', 'tools', 'llm', 'memory', 'review', 'utils']

from .version import get_version

__version__ = get_version()
