# Quick Start - Copy & Paste Commands

## 1. Navigate and Activate Environment

```bash
cd /Users/nithinyanna/Downloads/re-dd-embeddings
source venv/bin/activate
```

## 2. Quick Test (Recommended First - ~10-15 min)

### Generate Small Corpus
```bash
python scripts/generate_dataset.py \
    --model gemini-3-flash-preview:cloud \
    --out_dir data \
    --companies 5 \
    --seed 7 \
    --target_chunks 250
```

### Build Splits
```bash
python scripts/build_splits.py \
    --data_dir data \
    --target_train_pairs 1000 \
    --eval_queries 50 \
    --model gemini-3-flash-preview:cloud \
    --seed 42
```

### Validate
```bash
python scripts/validate_dataset.py --data_dir data
```

---

## 3. Full Dataset (Production - ~2-4 hours)

**Note**: You can use different models for corpus generation vs. query generation! See `MIXED_MODELS.md` for examples.

### Generate Full Corpus
```bash
python scripts/generate_dataset.py \
    --model gemini-3-flash-preview:cloud \
    --out_dir data \
    --companies 60 \
    --seed 7 \
    --target_chunks 3000
```

### Build Full Splits
```bash
python scripts/build_splits.py \
    --data_dir data \
    --target_train_pairs 15000 \
    --eval_queries 500 \
    --model gemini-3-flash-preview:cloud \
    --seed 42
```

### Validate Full Dataset
```bash
python scripts/validate_dataset.py --data_dir data
```

### Sample Space Corpus (Optional)
```bash
python scripts/sample_space_corpus.py \
    --data_dir data \
    --k 300 \
    --seed 123
```

---

## 4. Or Run Everything with One Script

```bash
chmod +x RUN_COMMANDS.sh
./RUN_COMMANDS.sh
```

---

## Expected Output Files

After running, check `data/` directory:

- ✅ `corpus.jsonl` - All document chunks
- ✅ `train_pairs.jsonl` - Training query-positive-negative pairs
- ✅ `queries.jsonl` - Evaluation queries
- ✅ `qrels.jsonl` - Relevance labels for eval
- ✅ `space_corpus.jsonl` - Demo corpus (if sampled)
- ✅ `index.json` - Metadata index

---

## Troubleshooting

**If Ollama errors:**
```bash
ollama list  # Check if model is available
ollama pull gemini-3-flash-preview:cloud  # Pull if missing
```

**If import errors:**
```bash
pip install tqdm numpy pandas
```

**To see progress:**
All scripts show real-time progress bars with:
- Percentage complete
- Items remaining
- Time estimates
