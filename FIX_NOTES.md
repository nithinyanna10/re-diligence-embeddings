# Fix Notes

## Issue: Ollama Command Error

### Problem
The script was failing with:
```
Error: unknown shorthand flag: 'p' in -p
```

### Root Cause
The original code was using:
```python
cmd = ["ollama", "run", model, "-p", prompt]
```

But Ollama CLI doesn't accept a `-p` flag. The prompt should be passed as a positional argument.

### Solution
Updated `scripts/ollama_client.py` to pass the prompt correctly:

1. **For normal prompts** (< 10,000 chars): Pass as positional argument
   ```python
   cmd = ["ollama", "run", model, prompt]
   ```

2. **For very long prompts** (> 10,000 chars): Use stdin
   ```python
   cmd = ["ollama", "run", model]
   result = subprocess.run(cmd, input=prompt, ...)
   ```

### Files Changed
- `scripts/ollama_client.py` - Fixed command construction (lines 34-56)

### Testing
Run the test script to verify:
```bash
python test_ollama.py
```

Or test directly:
```bash
python scripts/generate_dataset.py --companies 5 --target_chunks 250
```

The fix ensures Ollama receives the prompt correctly and can generate the dataset.
