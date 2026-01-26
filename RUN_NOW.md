# What to Run Now - Quick Guide

## Current Status
✅ **15,050 training pairs** generated (`data/train_pairs_v2.jsonl`)

## Option 1: Skip Mining, Train Directly (FASTEST - Recommended)

If you want to start training immediately:

```bash
cd /Users/nithinyanna/Downloads/re-diligence-embeddings

python train_embeddings_final.py \
    --train_pairs data/train_pairs_v2.jsonl \
    --corpus data/corpus.jsonl \
    --epochs 5 \
    --learning_rate 2e-5 \
    --output_dir models/re-diligence-embeddings-v2
```

**Time:** ~30-60 minutes  
**Result:** Trained model with 15,050 pairs (8x more than before!)

---

## Option 2: Mine Hard Negatives First (BETTER QUALITY)

If the mining script finished, use the mined pairs:

```bash
cd /Users/nithinyanna/Downloads/re-diligence-embeddings

# Check if mining finished
ls -lh data/train_pairs_mined.jsonl

# If it exists, train with it:
python train_embeddings_final.py \
    --train_pairs data/train_pairs_mined.jsonl \
    --corpus data/corpus.jsonl \
    --epochs 5 \
    --learning_rate 2e-5 \
    --output_dir models/re-diligence-embeddings-v2
```

**If mining didn't finish yet**, you can:
- Wait for it to finish (check with `ps aux | grep mine_hard_negatives`)
- Or skip it and use Option 1 above

---

## Recommended: Start Training Now

**Just run this:**

```bash
cd /Users/nithinyanna/Downloads/re-diligence-embeddings

python train_embeddings_final.py \
    --train_pairs data/train_pairs_v2.jsonl \
    --corpus data/corpus.jsonl \
    --epochs 5 \
    --learning_rate 2e-5 \
    --output_dir models/re-diligence-embeddings-v2
```

This will:
- Train on 15,050 pairs (vs 1,915 before)
- Take ~30-60 minutes
- Create `models/re-diligence-embeddings-v2/`
- Expected improvement: MRR@10 from 7% → 15-20%

---

## After Training: Evaluate

Once training finishes, test the new model:

```bash
python test_model.py \
    --model_dir models/re-diligence-embeddings-v2 \
    --queries data/queries.jsonl \
    --qrels data/qrels.jsonl \
    --corpus data/corpus.jsonl
```

---

## Summary

**Right now, run:**
```bash
python train_embeddings_final.py \
    --train_pairs data/train_pairs_v2.jsonl \
    --corpus data/corpus.jsonl \
    --epochs 5 \
    --learning_rate 2e-5 \
    --output_dir models/re-diligence-embeddings-v2
```

That's it! Let it train and you'll have an improved model.
