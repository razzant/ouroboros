#!/usr/bin/env python3
"""Test search_agent tool directly."""
import os
import sys
from pathlib import Path

# Load .env
env_path = Path('/home/ivan/.env')
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if '=' in line and not line.strip().startswith('#'):
            key, val = line.split('=', 1)
            os.environ[key.strip()] = val.strip()

sys.path.insert(0, '/home/ivan/.ouroboros/repo')

from ouroboros.tools.search_agent import search_agent_tool

result = search_agent_tool(query='бесплатные ИИ модели API')
print("Result:", result)
