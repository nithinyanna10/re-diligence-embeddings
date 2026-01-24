#!/usr/bin/env python3
"""Quick test script to verify Ollama client works."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from ollama_client import run_ollama, parse_json_response

def test_ollama():
    """Test Ollama client with a simple JSON request."""
    print("Testing Ollama client...")
    print("="*60)
    
    prompt = """Return STRICT JSON ONLY:
{
  "test": "success",
  "number": 42
}"""
    
    try:
        print(f"Model: gemini-3-flash-preview:cloud")
        print(f"Prompt length: {len(prompt)} chars")
        print("\nCalling Ollama...")
        
        response = run_ollama("gemini-3-flash-preview:cloud", prompt, timeout_s=60)
        print(f"\n✓ Raw response received ({len(response)} chars)")
        
        parsed = parse_json_response(response)
        print(f"✓ JSON parsed successfully: {parsed}")
        
        print("\n" + "="*60)
        print("✓ TEST PASSED - Ollama client is working!")
        return 0
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(test_ollama())
