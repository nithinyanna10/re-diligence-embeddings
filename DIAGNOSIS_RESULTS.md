# Evaluation Diagnosis Results

## ✅ Good News

1. **Coverage**: 100% of queries have at least 1 relevant doc ✓
2. **Qrels per query**: 3.84 average (target: 2-4) ✓
3. **No missing chunks**: All qrel doc_ids exist in corpus ✓
4. **Relevance distribution**: Good mix of relevance=1 and relevance=2 ✓

## ⚠️ Issues Found

### 1. **Fine-tuned Model is WORSE than Base Model!**

**Base Model:**
- MRR@10: 0.0829 (8.29%)
- Recall@10: 0.0425 (4.25%)

**Fine-tuned Model:**
- MRR@10: 0.0715 (7.15%) - **13.8% WORSE**
- Recall@10: 0.0335 (3.35%) - **21.2% WORSE**

**Diagnosis:** This indicates:
- Overfitting to limited training data (1,915 pairs)
- Misaligned training (queries don't match eval queries)
- Poor hard negatives (too easy or wrong type)

### 2. **Oracle Keyword Matching: 43.2%**

Only 43.2% of queries can find relevant docs via simple keyword matching.

**Diagnosis:** Suggests qrels/queries may be inconsistent or queries are too generic.

---

## Root Cause Analysis

The model is learning the wrong patterns because:

1. **Too few training pairs** (1,915 vs target 15,000)
2. **Query mismatch**: Training queries may not match eval query style
3. **Weak hard negatives**: Not challenging enough
4. **Limited diversity**: Only 5 queries per chunk, not diverse enough

---

## Action Plan

### Step 1: Generate Better Pairs ✅ (Script ready)
```bash
python generate_better_pairs.py \
    --corpus data/corpus.jsonl \
    --output data/train_pairs_v2.jsonl \
    --model gemini-3-flash-preview:cloud
```

**Target:** 5 queries per chunk = ~15,000 pairs
- 2 keyword queries
- 2 question queries  
- 1 clause-hunt query

### Step 2: Mine Hard Negatives ✅ (Script ready)
```bash
python mine_hard_negatives.py \
    --pairs data/train_pairs_v2.jsonl \
    --output data/train_pairs_mined.jsonl \
    --model models/re-diligence-embeddings
```

**Method:** Use current model to retrieve top-50, pick non-relevant as hard negatives.

### Step 3: Retrain with Better Data
```bash
python train_embeddings_final.py \
    --train_pairs data/train_pairs_mined.jsonl \
    --epochs 5 \
    --learning_rate 2e-5 \
    --output_dir models/re-diligence-embeddings-v2
```

**Expected:** MRR@10 should improve from 7% → 15-20%

### Step 4: Consider Base Model Upgrade
If still low, switch to:
- `sentence-transformers/all-mpnet-base-v2` (stronger)
- Or `BAAI/bge-base-en-v1.5` (retrieval-focused)

---

## Next Immediate Action

**Run this now:**
```bash
python generate_better_pairs.py
```

This will generate ~15,000 better training pairs with diverse query types.
