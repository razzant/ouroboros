#!/usr/bin/env python3
"""
Quick test to verify local llama server works with OpenRouter-compatible API.
"""

import json
import sys
import time

try:
    import requests
except ImportError:
    print("❌ requests not installed. Install: pip install requests")
    sys.exit(1)

BASE_URL = "http://localhost:8080/v1"
MODEL = "Qwen3.6-35B-A3B-Q4_K_M.gguf"

print(f"🔍 Testing local LLM server...")
print(f"   Base URL: {BASE_URL}")
print(f"   Model: {MODEL}")
print()

# Test 1: Check if server is running
print("1️⃣  Checking server connectivity...")
try:
    resp = requests.get(f"{BASE_URL}/models", timeout=5)
    if resp.status_code == 200:
        models = resp.json()
        print(f"   ✅ Server is running!")
        print(f"   📦 Available models: {len(models.get('data', []))}")
        for m in models.get('data', [])[:3]:
            print(f"      - {m.get('id', 'unknown')}")
    else:
        print(f"   ❌ Server returned {resp.status_code}")
        sys.exit(1)
except Exception as e:
    print(f"   ❌ Cannot connect to {BASE_URL}")
    print(f"   Error: {e}")
    print()
    print("   💡 Make sure your local llama server is running!")
    print("   💡 Typical commands: ollama serve, llm serve, etc.")
    sys.exit(1)

# Test 2: Simple chat completion
print()
print("2️⃣  Testing chat completion (simple greeting)...")
try:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "Hello, what is 2+2?"}
        ],
        "temperature": 0.5,
        "max_tokens": 100,
    }
    
    print(f"   📤 Sending request...")
    start = time.time()
    resp = requests.post(
        f"{BASE_URL}/chat/completions",
        json=payload,
        timeout=60
    )
    elapsed = time.time() - start
    
    if resp.status_code == 200:
        data = resp.json()
        content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
        print(f"   ✅ Got response in {elapsed:.1f}s")
        print(f"   📝 Content: {content[:100]}{'...' if len(content) > 100 else ''}")
    else:
        print(f"   ❌ Got status {resp.status_code}")
        print(f"   Response: {resp.text[:200]}")
        sys.exit(1)
        
except Exception as e:
    print(f"   ❌ Request failed: {e}")
    sys.exit(1)

print()
print("🎉 All tests passed! Your local LLM is ready for Ouroboros!")
print()
print("Next steps:")
print("  1. Start the bot: python local_launcher.py")
print("  2. The bot will use your local Qwen model via http://localhost:8080/v1")
print("  3. If local model fails, it will fallback to cloud models")
