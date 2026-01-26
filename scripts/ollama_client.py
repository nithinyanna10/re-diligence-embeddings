#!/usr/bin/env python3
"""
Ollama client for local LLM inference.
Handles subprocess calls to ollama run with JSON-only output enforcement.
"""

import subprocess
import json
import re
import time
from typing import Optional


def run_ollama(model: str, prompt: str, timeout_s: int = 240, max_retries: int = 2, system: str = None) -> str:
    """
    Run Ollama model with prompt and return raw output.
    
    Args:
        model: Ollama model name (e.g., "gemini-3-flash-preview:cloud")
        prompt: Full prompt string
        timeout_s: Timeout in seconds
        max_retries: Maximum retries on JSON parse failure
    
    Returns:
        Raw output string from Ollama
    
    Raises:
        subprocess.TimeoutExpired: If command times out
        subprocess.CalledProcessError: If command fails
        ValueError: If JSON extraction fails after retries
    """
    for attempt in range(max_retries + 1):
        try:
            # Build command: ollama run <model> "<prompt>"
            # Ollama CLI accepts prompt as a positional argument
            # For very long prompts, we use stdin as fallback
            # If system message provided, prepend it
            if system:
                full_prompt = f"{system}\n\n{prompt}"
            else:
                full_prompt = prompt
            
            if len(full_prompt) > 10000:  # Very long prompts via stdin
                cmd = ["ollama", "run", model]
                result = subprocess.run(
                    cmd,
                    input=full_prompt,
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                    check=True
                )
            else:
                # Normal prompts as positional argument
                cmd = ["ollama", "run", model, full_prompt]
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                    check=True
                )
            
            output = result.stdout.strip()
            
            # Return raw output - let caller handle JSON extraction
            # This allows for arrays, incomplete JSON, etc.
            return output
            
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"Ollama call timed out after {timeout_s}s")
        except subprocess.CalledProcessError as e:
            error_msg = f"Ollama command failed: {e.stderr}"
            if attempt < max_retries:
                time.sleep(2)
                continue
            raise RuntimeError(error_msg)
    
    raise ValueError("Max retries exceeded")


def parse_json_response(response: str) -> dict:
    """
    Parse JSON from Ollama response, handling markdown code blocks if present.
    
    Args:
        response: Raw response string
    
    Returns:
        Parsed JSON dict
    
    Raises:
        json.JSONDecodeError: If JSON is invalid
    """
    # Remove markdown code blocks if present
    response = re.sub(r'```json\s*', '', response)
    response = re.sub(r'```\s*', '', response)
    response = response.strip()
    
    # Extract JSON object
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        response = json_match.group(0)
    
    return json.loads(response)
