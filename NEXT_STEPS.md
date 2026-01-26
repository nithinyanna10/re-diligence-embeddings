# Next Steps After Generating Training Pairs

## ✅ Current Status
- **15,050 training pairs** generated successfully
- **5 queries per chunk** (perfect!)
- **0 failed chunks**
- Ready for training!

---

## Step 1: Mine Hard Negatives (Recommended)

Improve the hard negatives using your current model's retrieval capabilities.

```bash
cd /Users/nithinyanna/Downloads/re-diligence-embeddings

python mine_hard_negatives.py \
    --pairs data/train_pairs_v2.jsonl \
    --corpus data/corpus.jsonl \
    --output data/train_pairs_mined.jsonl \
    --model models/re-diligence-embeddings
```

**What this does:**
- Uses your current trained model to retrieve top-50 similar chunks for each query
- Picks non-relevant chunks as better hard negatives
- Creates `data/train_pairs_mined.jsonl` with improved negatives

**Time:** ~10-20 minutes (depends on model speed)

**Note:** If you want to skip this step, you can train directly with `train_pairs_v2.jsonl`, but mined negatives usually improve performance.

---

## Step 2: Train the Model

Train with the new, larger dataset (15,050 pairs vs previous 1,915).

### Option A: Train with Mined Negatives (Recommended)
```bash
python train_embeddings_final.py \
    --train_pairs data/train_pairs_mined.jsonl \
    --corpus data/corpus.jsonl \
    --epochs 5 \
    --learning_rate 2e-5 \
    --output_dir models/re-diligence-embeddings-v2
```

### Option B: Train with Original Pairs (Faster, Skip Mining)
```bash
python train_embeddings_final.py \
    --train_pairs data/train_pairs_v2.jsonl \
    --corpus data/corpus.jsonl \
    --epochs 5 \
    --learning_rate 2e-5 \
    --output_dir models/re-diligence-embeddings-v2
```

**What this does:**
- Trains on 15,050 pairs (8x more than before!)
- Uses triplet loss with hard negatives
- Saves to `models/re-diligence-embeddings-v2/`

**Time:** ~30-60 minutes (depends on GPU/CPU)

**Expected improvement:**
- Current MRR@10: 7.15%
- Expected MRR@10: 15-20% (2-3x improvement)

---

## Step 3: Evaluate the New Model

Test the improved model:

```bash
python test_model.py \
    --model_dir models/re-diligence-embeddings-v2 \
    --queries data/queries.jsonl \
    --qrels data/qrels.jsonl \
    --corpus data/corpus.jsonl
```

**What to look for:**
- MRR@10 should be **15-20%** (up from 7.15%)
- Recall@10 should be **10-15%** (up from 3.35%)
- NDCG@10 should improve significantly

---

## Quick Start (All Steps)

Run everything in sequence:

```bash
# Step 1: Mine hard negatives
python mine_hard_negatives.py \
    --pairs data/train_pairs_v2.jsonl \
    --corpus data/corpus.jsonl \
    --output data/train_pairs_mined.jsonl \
    --model models/re-diligence-embeddings

# Step 2: Train
python train_embeddings_final.py \
    --train_pairs data/train_pairs_mined.jsonl \
    --corpus data/corpus.jsonl \
    --epochs 5 \
    --learning_rate 2e-5 \
    --output_dir models/re-diligence-embeddings-v2

# Step 3: Evaluate
python test_model.py \
    --model_dir models/re-diligence-embeddings-v2 \
    --queries data/queries.jsonl \
    --qrels data/qrels.jsonl \
    --corpus data/corpus.jsonl
```

---

## Troubleshooting

### If mine_hard_negatives.py fails:
- Check that `models/re-diligence-embeddings` exists
- You can skip this step and train directly

### If training is slow:
- Reduce epochs: `--epochs 3`
- Use smaller batch size (check script defaults)

### If out of memory:
- Reduce batch size in training script
- Close other applications

---

## Expected Timeline

- **Mine negatives**: 10-20 min
- **Training**: 30-60 min
- **Evaluation**: 5-10 min
- **Total**: ~1-1.5 hours

---

## Success Criteria

After training, you should see:
- ✅ MRR@10 > 15% (target: 15-20%)
- ✅ Recall@10 > 10% (target: 10-15%)
- ✅ Model performs better than baseline (7.15% MRR)

If metrics are still low, consider:
- More training epochs
- Different learning rate
- Better base model (see DIAGNOSIS_RESULTS.md)
