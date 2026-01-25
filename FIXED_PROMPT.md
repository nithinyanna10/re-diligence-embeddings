# Fixed: Ollama "Thinking..." Issue

## Problem
Ollama was returning "Thinking..." analysis text before JSON, breaking parsing:
```
Thinking...
**Analyzing Property Details**
I've just begun examining...
[actual JSON here]
```

## Solution

### 1. **Simplified Prompt**
- Removed verbose instructions
- Made prompt direct: "Return ONLY JSON array"
- Shorter chunk text (800 chars vs 1000)
- Clear example format

### 2. **Robust JSON Extraction**
- Multiple strategies to find JSON array:
  1. Regex search for `[...]` pattern
  2. Find first `[` and last `]`
  3. Fallback to any JSON array pattern
- Handles multi-line JSON
- Ignores "Thinking..." prefixes

### 3. **Better Error Handling**
- Falls back to default queries if parsing fails
- Continues processing other chunks
- Logs errors but doesn't stop

## Updated Code

**Prompt:**
```python
prompt = f"""Return ONLY a JSON array with 5 search queries. No thinking, no explanation, no analysis.

Chunk: {chunk_text[:800]}
Source: {source_type}, Topic: {topic}

Generate exactly 5 queries:
1. Short keyword (3-8 words)
2. Question format
3. Clause hunt (SNDA, estoppel, CAM, DSCR, WALE, etc.)
4. Financial/metrics query
5. Document-specific query

Return ONLY this format, nothing else:
["query 1", "query 2", "query 3", "query 4", "query 5"]"""
```

**JSON Extraction:**
```python
# Strategy 1: Regex search
json_match = re.search(r'(\[[\s\S]*?\])', response, re.MULTILINE)

# Strategy 2: Find brackets
first_bracket = response.find('[')
last_bracket = response.rfind(']')

# Strategy 3: Fallback pattern
json_match = re.search(r'\[.*?\]', response, re.DOTALL)
```

## Test

Run with small batch first:
```bash
python generate_better_pairs.py \
    --corpus data/corpus.jsonl \
    --output data/train_pairs_v2.jsonl \
    --model gemini-3-flash-preview:cloud \
    --batch_size 5
```

If it works, run full generation:
```bash
python generate_better_pairs.py \
    --corpus data/corpus.jsonl \
    --output data/train_pairs_v2.jsonl \
    --model gemini-3-flash-preview:cloud
```
