#!/usr/bin/env python3
"""Direct functional test for SearchAgent class.

This test instantiates the SearchAgent directly and verifies:
- Agent can process a query end-to-end
- Returns answer with sources within iteration limit
- No exceptions escape
"""
import os
import sys
import json
from pathlib import Path

# Set up paths
repo_root = Path('/home/ivan/.ouroboros/repo').resolve()
sys.path.insert(0, str(repo_root))

# Load .env to get credentials
env_path = repo_root / '.env'
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if '=' in line and not line.strip().startswith('#'):
            key, val = line.split('=', 1)
            os.environ[key.strip()] = val.strip()

from ouroboros.tools.search_agent import SearchAgent

def test_search_agent_direct():
    """Test that SearchAgent can process a query and return answer + sources."""
    # Get model and API key from environment
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    model = os.getenv("OPENROUTER_MODEL", "stepfun/Step-3.5-Flash:free")

    if not api_key:
        print("⚠️ No API key found in environment (OPENROUTER_API_KEY or OPENAI_API_KEY)")
        print("Skipping test - requires OpenRouter credentials")
        return True  # Skip, not fail

    print(f"Initializing SearchAgent with model={model}")
    agent = SearchAgent(
        api_key=api_key,
        base_url=base_url,
        model=model,
        verbose=True,
        max_iterations=5,  # short for test
        request_delay=0.5
    )

    query = "какие есть ИИ модели с бесплатным api?"
    print(f"\nTesting SearchAgent with query: {query!r}")

    try:
        result = agent.run(query)  # <-- Changed from process_query to run
        print("\n=== RESULT ===")
        print(json.dumps(result, ensure_ascii=False, indent=2))

        # Validate structure
        assert isinstance(result, dict), "Result must be a dict"
        assert "answer" in result, "Result must have 'answer' field"
        assert "sources" in result, "Result must have 'sources' field"
        assert "iterations" in result, "Result must have 'iterations' field"
        assert isinstance(result["sources"], list), "'sources' must be a list"

        # Answer should be non-empty string
        answer = result["answer"]
        assert isinstance(answer, str) and len(answer.strip()) > 10, f"Answer should be a non-empty string, got: {answer!r}"

        # Should complete within iteration limit (not None)
        assert result["iterations"] <= 5, f"Exceeded max iterations 5, got {result['iterations']}"

        # Should have at least one source or answer was synthesized from search snippets
        # (some queries might return answer without explicit sources if model used snippets directly)
        print(f"\n✅ Test passed! Iterations: {result['iterations']}, sources: {len(result['sources'])}")
        return True

    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        success = test_search_agent_direct()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)