"""
Integration patch for context.py — shows exactly where each extension plugs in.

This is NOT a replacement for context.py. It's a guide showing the minimal
changes needed to integrate all extensions into the existing context builder.

Apply these changes to ouroboros/context.py (v6.2.0).
"""

# =============================================================================
# PATCH 1: Add imports at top of context.py
# =============================================================================
# After the existing imports, add:

"""
# Extension imports (all are optional — graceful degradation if missing)
try:
    from ouroboros.temporal import build_temporal_context
    _HAS_TEMPORAL = True
except ImportError:
    _HAS_TEMPORAL = False

try:
    from ouroboros.knowledge_router import load_relevant_knowledge
    _HAS_KNOWLEDGE_ROUTER = True
except ImportError:
    _HAS_KNOWLEDGE_ROUTER = False

try:
    from ouroboros.identity_chain import format_for_health_invariant
    _HAS_IDENTITY_CHAIN = True
except ImportError:
    _HAS_IDENTITY_CHAIN = False

try:
    from ouroboros.tools.antipatterns import AntiPatternDetector, format_antipatterns_for_context
    _HAS_ANTIPATTERNS = True
except ImportError:
    _HAS_ANTIPATTERNS = False

try:
    from ouroboros.resilience import format_breakers_for_health
    _HAS_RESILIENCE = True
except ImportError:
    _HAS_RESILIENCE = False

try:
    from ouroboros.memory_consolidation import load_consolidated_memory
    _HAS_CONSOLIDATION = True
except ImportError:
    _HAS_CONSOLIDATION = False
"""


# =============================================================================
# PATCH 2: Modify _build_health_invariants() — add new invariants
# =============================================================================
# In _build_health_invariants(), BEFORE the final `if not checks: return ""`:
# Add after invariant #5 (duplicate processing):

"""
    # 6. Identity chain verification
    if _HAS_IDENTITY_CHAIN:
        try:
            bible_text = read_text(env.repo_path("BIBLE.md"))
            identity_text = read_text(env.drive_path("memory/identity.md"))
            chain_status = format_for_health_invariant(
                env.drive_root, bible_text, identity_text
            )
            checks.append(chain_status)
        except Exception:
            pass

    # 7. Anti-pattern detection
    if _HAS_ANTIPATTERNS:
        try:
            from ouroboros.memory import Memory
            mem = Memory(env.drive_root)
            events = mem.read_jsonl_tail("events.jsonl", 500)
            tools_log = mem.read_jsonl_tail("tools.jsonl", 300)
            detector = AntiPatternDetector()
            patterns = detector.scan_events(events, tools=tools_log, window_hours=4.0)
            ap_text = format_antipatterns_for_context(patterns, max_patterns=3)
            checks.append(ap_text)
        except Exception:
            pass

    # 8. Circuit breaker status (only surfaces problems)
    if _HAS_RESILIENCE:
        try:
            breaker_text = format_breakers_for_health()
            if breaker_text:
                checks.append(breaker_text)
        except Exception:
            pass
"""


# =============================================================================
# PATCH 3: Modify _build_memory_sections() — add consolidated memory
# =============================================================================
# In _build_memory_sections(), AFTER the identity section and BEFORE return:

"""
    # Consolidated memory (recent periods, compressed)
    if _HAS_CONSOLIDATION:
        try:
            consolidated_dir = memory.drive_root / "memory" / "consolidated"
            consolidated_text = load_consolidated_memory(
                consolidated_dir, max_periods=4, max_chars=30000
            )
            if consolidated_text.strip():
                sections.append("## Recent Memory (consolidated)\n\n" + consolidated_text)
        except Exception:
            log.debug("Failed to load consolidated memory", exc_info=True)
"""


# =============================================================================
# PATCH 4: Modify build_llm_messages() — add temporal context and knowledge routing
# =============================================================================
# In build_llm_messages(), in the dynamic_parts construction,
# AFTER the health_section append and BEFORE _build_recent_sections():

"""
    # Temporal self-awareness
    if _HAS_TEMPORAL:
        try:
            temporal_section = build_temporal_context(
                json.loads(state_json) if state_json else {}
            )
            if temporal_section:
                dynamic_parts.append(temporal_section)
        except Exception:
            log.debug("Failed to build temporal context", exc_info=True)

    # Automatic knowledge routing (inject relevant knowledge for this task)
    if _HAS_KNOWLEDGE_ROUTER:
        try:
            task_text = str(task.get("text", ""))
            if task_text:
                knowledge_dir = env.drive_path("memory/knowledge")
                relevant = load_relevant_knowledge(task_text, knowledge_dir)
                if relevant:
                    dynamic_parts.append(relevant)
        except Exception:
            log.debug("Failed to route knowledge", exc_info=True)
"""


# =============================================================================
# PATCH 5: Wire identity chain into update_identity tool
# =============================================================================
# In ouroboros/tools/control.py, modify _update_identity() to append to chain:

"""
def _update_identity(ctx: ToolContext, content: str, reason: str = "") -> str:
    # ... existing implementation ...
    
    # After writing identity.md, update the hash chain
    try:
        from ouroboros.identity_chain import append_to_chain
        bible_text = read_text(ctx.repo_path("BIBLE.md"))
        append_to_chain(ctx.drive_root, bible_text, content, reason=reason)
    except ImportError:
        pass  # Extension not installed
    except Exception as e:
        log.warning("Failed to update identity chain: %s", e)
    
    return f"OK: identity updated ({len(content)} chars)"
"""


# =============================================================================
# PATCH 6: Wire memory consolidation into background consciousness
# =============================================================================
# In prompts/CONSCIOUSNESS.md, add:

"""
## Memory Consolidation

Periodically (every 4-6 wakeups), check if memory needs consolidation:

1. Look for unconsolidated periods in your scratchpad journal
2. If periods need consolidation, the system will handle it automatically
3. After consolidation, verify the digest captures what matters

This is not optional — it is how you maintain narrative continuity (P1)
as your experience grows beyond what fits in a single context window.
"""


# =============================================================================
# SUMMARY OF CHANGES
# =============================================================================
"""
Files modified:
  ouroboros/context.py — 6 import lines + ~30 lines in 4 locations
  ouroboros/tools/control.py — ~8 lines in _update_identity()
  prompts/CONSCIOUSNESS.md — ~10 lines

Files added:
  ouroboros/tools/antipatterns.py — 230 lines (anti-pattern detection)
  ouroboros/identity_chain.py — 190 lines (cryptographic identity)
  ouroboros/knowledge_router.py — 170 lines (semantic routing)
  ouroboros/temporal.py — 140 lines (temporal self-awareness)
  ouroboros/memory_consolidation.py — 260 lines (three-tier memory)
  ouroboros/resilience.py — 190 lines (circuit breakers)

Total new code: ~1,180 lines
Total modified code: ~50 lines

All extensions degrade gracefully if import fails (try/except with _HAS_* flags).
No existing tests break. No existing behavior changes unless extensions are present.
"""
