# Improvements Made to JSON Extraction

## Problem
The `generate_better_pairs.py` script was failing to extract queries from Ollama responses because:
1. Ollama was returning "Thinking..." text before JSON
2. JSON arrays were sometimes incomplete
3. The `ollama_client.py` was trying to validate JSON objects `{}` instead of arrays `[]`
4. Extraction logic wasn't robust enough for partial responses

## Changes Made

### 1. `scripts/ollama_client.py`
- **Removed JSON validation**: Now returns raw output and lets the caller handle extraction
- This allows for arrays, incomplete JSON, and other formats
- Simpler and more flexible

### 2. `generate_better_pairs.py`
- **Improved extraction logic**:
  - More aggressive removal of "Thinking..." patterns
  - Better handling of incomplete JSON arrays
  - Multiple extraction strategies:
    1. Try to parse complete JSON array
    2. Extract complete quoted strings from partial JSON
    3. Fallback to default queries if all else fails
- **Better error handling**: 
  - Retries up to 3 times per chunk
  - Graceful fallback to default queries
  - Better logging of failures

## How It Works Now

1. **Request**: Sends prompt to Ollama requesting JSON array of 5 queries
2. **Clean**: Removes "Thinking..." text, markdown, code blocks
3. **Extract**: 
   - First tries to find complete JSON array `[...]`
   - If incomplete, extracts all quoted strings `"query"`
   - Builds array from extracted strings
4. **Validate**: Ensures exactly 5 queries (pads if needed)
5. **Fallback**: Uses default queries if extraction fails after 3 attempts

## Testing

To test the improvements:

```bash
# Test with a small batch first
python generate_better_pairs.py \
    --corpus data/corpus.jsonl \
    --output data/train_pairs_v2.jsonl \
    --model gemini-3-flash-preview:cloud \
    --batch_size 10
```

## Expected Results

- Should handle "Thinking..." text gracefully
- Should extract queries even from incomplete JSON
- Should generate ~15,000 pairs (5 queries × 3,010 chunks)
- Should have minimal fallback queries (only when extraction truly fails)

## Next Steps

1. Run the script and monitor for extraction failures
2. If still seeing many failures, consider:
   - Using a different Ollama model
   - Adjusting the prompt to be more explicit
   - Adding more aggressive text cleaning
3. Once pairs are generated, proceed with hard negative mining and retraining
