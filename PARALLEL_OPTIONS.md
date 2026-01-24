# Parallel Execution Options

## ❌ Cannot Run Corpus + Splits in Parallel

**No, you cannot run `generate_dataset.py` and `build_splits.py` in parallel** because:

- `build_splits.py` **requires** `corpus.jsonl` to exist and be complete
- It reads the entire corpus to:
  - Generate queries for chunks
  - Find hard negatives from the corpus
  - Create eval queries and qrels

**Dependency Chain:**
```
corpus.jsonl (generate_dataset.py)
    ↓
    ↓ (must wait for completion)
    ↓
train_pairs.jsonl, queries.jsonl, qrels.jsonl (build_splits.py)
```

---

## ✅ What You CAN Do in Parallel

### Option 1: Generate Multiple Corpora in Parallel

Run multiple corpus generations with different output directories:

```bash
# Terminal 1: Generate corpus 1
python scripts/generate_dataset.py \
    --out_dir data/corpus1 \
    --companies 30 \
    --target_chunks 1500 \
    --seed 7

# Terminal 2: Generate corpus 2 (parallel)
python scripts/generate_dataset.py \
    --out_dir data/corpus2 \
    --companies 30 \
    --target_chunks 1500 \
    --seed 42

# Then combine them:
cat data/corpus1/corpus.jsonl data/corpus2/corpus.jsonl > data/corpus.jsonl
```

### Option 2: Generate Splits for Multiple Corpora in Parallel

If you have multiple complete corpora:

```bash
# Terminal 1: Build splits for corpus1
python scripts/build_splits.py \
    --data_dir data/corpus1 \
    --target_train_pairs 7500 \
    --eval_queries 250 \
    --model gpt-oss:120b-cloud \
    --seed 42

# Terminal 2: Build splits for corpus2 (parallel)
python scripts/build_splits.py \
    --data_dir data/corpus2 \
    --target_train_pairs 7500 \
    --eval_queries 250 \
    --model gpt-oss:120b-cloud \
    --seed 100
```

### Option 3: Use Background Jobs

Run corpus generation in background, then splits:

```bash
# Start corpus generation in background
python scripts/generate_dataset.py \
    --out_dir data \
    --companies 60 \
    --target_chunks 3000 \
    --seed 7 \
    > corpus.log 2>&1 &

# Check progress
tail -f corpus.log

# When corpus is done, run splits
python scripts/build_splits.py \
    --data_dir data \
    --target_train_pairs 15000 \
    --eval_queries 500 \
    --model gpt-oss:120b-cloud \
    --seed 42
```

---

## 🚀 Optimization: Incremental Processing (Advanced)

If you want to process splits as corpus is being generated, you'd need to modify the scripts to:

1. **Streaming approach**: Process chunks as they're written
2. **Checkpoint system**: Process batches of chunks incrementally

This would require code changes. Current scripts are designed for:
- Complete corpus → Complete splits

---

## ⚡ Recommended Workflow

**Best approach for speed:**

```bash
# Step 1: Generate corpus (single process, ~2-3 hours)
python scripts/generate_dataset.py \
    --out_dir data \
    --companies 60 \
    --target_chunks 3000 \
    --seed 7

# Step 2: While corpus generates, prepare split generation
# (Check corpus progress, set up monitoring, etc.)

# Step 3: Once corpus is done, generate splits (~1-2 hours)
python scripts/build_splits.py \
    --data_dir data \
    --target_train_pairs 15000 \
    --eval_queries 500 \
    --model gpt-oss:120b-cloud \
    --seed 42
```

**Total time: ~3-5 hours** (sequential, but most efficient)

---

## 📊 Time Breakdown

| Step | Time | Can Parallelize? |
|------|------|------------------|
| Corpus generation | ~2-3 hours | ✅ Yes (multiple corpora) |
| Split generation | ~1-2 hours | ✅ Yes (multiple splits) |
| **Total sequential** | **~3-5 hours** | ❌ No (dependency) |

---

## 💡 Pro Tip: Monitor Progress

While corpus generates, you can monitor progress:

```bash
# Watch corpus grow
watch -n 5 'wc -l data/corpus.jsonl'

# Check chunk count
tail -n 1 data/corpus.jsonl | python3 -c "import sys, json; print(json.load(sys.stdin)['chunk_id'])"

# When it reaches ~3000, start splits
```

---

## Summary

- ❌ **Cannot** run corpus + splits in parallel (dependency)
- ✅ **Can** run multiple corpus generations in parallel
- ✅ **Can** run multiple split generations in parallel (if you have multiple corpora)
- ✅ **Can** use background jobs to monitor progress
- ⚡ **Best**: Sequential execution (most reliable)

The scripts are designed for sequential execution, which is the most reliable approach.
