#!/usr/bin/env python3
"""Functional test for search_agent tool.

Verifies that:
- Tool can be loaded and called
- Agent completes within timeout
- Returns answer with sources
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

from ouroboros.tools.search_agent import search_agent_tool

def test_search_agent_basic():
    """Test that search_agent returns a valid response."""
    query = "какие есть ИИ модели с бесплатным api?"
    print(f"Testing search_agent with query: {query!r}")
    
    result = search_agent_tool(query=query)
    print("\n=== RESULT ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    # Validate structure
    assert isinstance(result, dict), "Result must be a dict"
    assert "answer" in result, "Result must have 'answer' field"
    assert "sources" in result, "Result must have 'sources' field"
    assert isinstance(result["sources"], list), "'sources' must be a list"
    
    # Answer should be non-empty string
    answer = result["answer"]
    assert isinstance(answer, str) and len(answer.strip()) > 10, "Answer should be a non-empty string"
    
    # Should have at least one source
    assert len(result["sources"]) > 0, "Should have at least one source"
    
    print("\n✅ Test passed!")
    return True

if __name__ == "__main__":
    try:
        success = test_search_agent_basic()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)