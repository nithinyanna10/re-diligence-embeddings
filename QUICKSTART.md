# Quick Start Guide

## Setup (One-time)

```bash
cd /Users/nithinyanna/Downloads/re-dd-embeddings
source venv/bin/activate

# Verify Ollama is installed and model is available
ollama list | grep gemini-3-flash-preview
```

## Generate Dataset (Full Pipeline)

### Step 1: Generate Corpus
```bash
python scripts/generate_dataset.py \
    --model gemini-3-flash-preview:cloud \
    --out_dir data \
    --companies 60 \
    --seed 7 \
    --target_chunks 3000
```

**Expected time**: ~2-4 hours (depends on Ollama speed)
**Output**: `data/corpus.jsonl` (~3,000 chunks)

### Step 2: Build Train/Eval Splits
```bash
python scripts/build_splits.py \
    --data_dir data \
    --target_train_pairs 15000 \
    --eval_queries 500 \
    --model gemini-3-flash-preview:cloud \
    --seed 42
```

**Expected time**: ~1-2 hours
**Output**: 
- `data/train_pairs.jsonl` (~15,000 pairs)
- `data/queries.jsonl` (500 queries)
- `data/qrels.jsonl` (relevance labels)

### Step 3: Validate
```bash
python scripts/validate_dataset.py --data_dir data
```

Should print: `✓ VALIDATION PASSED`

### Step 4: Sample Space Corpus (Optional)
```bash
python scripts/sample_space_corpus.py \
    --data_dir data \
    --k 300 \
    --seed 123
```

**Output**: `data/space_corpus.jsonl` (300 chunks for demo)

## Testing with Smaller Dataset

For faster testing:

```bash
# Generate 5 deals (~250 chunks)
python scripts/generate_dataset.py \
    --model gemini-3-flash-preview:cloud \
    --companies 5 \
    --target_chunks 250 \
    --seed 7

# Build smaller splits
python scripts/build_splits.py \
    --data_dir data \
    --target_train_pairs 1000 \
    --eval_queries 50 \
    --seed 42
```

## Troubleshooting

### Ollama not responding
```bash
# Check Ollama is running
ollama list

# Test model directly
ollama run gemini-3-flash-preview:cloud -p "Return JSON: {\"test\": 1}"
```

### JSON parse errors
- Scripts auto-retry up to 2 times
- If persistent, try a different Ollama model
- Check Ollama logs: `ollama logs`

### Out of disk space
- Each chunk is ~1-2 KB
- 3,000 chunks ≈ 3-6 MB
- Full dataset with splits ≈ 10-15 MB

## Next Steps

After dataset generation:
1. Train embedding model (see README.md)
2. Evaluate on `data/queries.jsonl` + `data/qrels.jsonl`
3. Deploy Space demo with `data/space_corpus.jsonl`
