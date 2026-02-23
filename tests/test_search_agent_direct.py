#!/usr/bin/env python3
"""Direct functional test for SearchAgent class.

This test instantiates the SearchAgent directly and verifies:
- Agent can process a query end-to-end
- Returns a non-empty answer (not the default failure message)
- Returns at least one source
- Completes within iteration limit
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
        max_search_results=5,
        request_delay=0.5
    )

    query = "что такое оuroboros система"
    print(f"\nTesting SearchAgent with query: {query!r}")

    try:
        result = agent.process_query(query, max_iterations=5)
        print("\n=== RESULT ===")
        print(json.dumps(result, ensure_ascii=False, indent=2))

        # Validate structure
        assert isinstance(result, dict), "Result must be a dict"
        assert "answer" in result, "Result must have 'answer' field"
        assert "sources" in result, "Result must have 'sources' field"
        assert "iterations" in result, "Result must have 'iterations' field"
        assert isinstance(result["sources"], list), "'sources' must be a list"

        answer = result["answer"]
        iterations = result["iterations"]
        sources = result["sources"]

        # The agent should produce a genuine answer, not the default failure message
        DEFAULT_FAILURE = "Не удалось получить ответ за допустимое число итераций."
        assert answer != DEFAULT_FAILURE, f"Agent failed to produce an answer. Got default failure message."

        # Answer should be non-empty and substantive
        assert isinstance(answer, str) and len(answer.strip()) > 20, f"Answer too short: {answer!r}"

        # Sources should include at least one URL
        assert len(sources) > 0, f"Expected at least one source, got {len(sources)}"

        # Should complete within iteration limit
        assert iterations <= 5, f"Exceeded max iterations 5, got {iterations}"

        print(f"\n✅ Test passed! Iterations: {iterations}, sources: {len(sources)}")
        print(f"Answer preview: {answer[:200]}...")
        return True

    except AssertionError as e:
        print(f"\n❌ Assertion failed: {e}")
        return False
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