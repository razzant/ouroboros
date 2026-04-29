"""
Ouroboros — LLM tool loop (Orchestrator).

Core loop: send messages to LLM, execute tool calls, repeat until final response.
Delegates low-level execution to ouroboros.execution.*
"""
from ouroboros.execution.loop import run_llm_loop

__all__ = ["run_llm_loop"]
