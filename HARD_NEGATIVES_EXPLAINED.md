# Hard Negatives - What You Have vs What Mining Does

## ✅ You Already Have Hard Negatives!

Your `train_pairs_v2.jsonl` file **already contains hard negatives** for every pair!

Each pair has:
- 1 query
- 1 positive chunk (the correct answer)
- **3 hard negatives** (challenging wrong answers)

### How They Were Generated

The `generate_better_pairs.py` script uses the `find_hard_negatives()` function which finds hard negatives using **rule-based matching**:

1. **Same company, different topic** - e.g., same property but different document type
2. **Same source type, different topic** - e.g., both are CIMs but different sections
3. **Confusable pairs** - e.g., CAM vs taxes, Phase I vs PCA, DSCR vs debt yield

**Example from your data:**
```json
{
  "query": "Find the NOI for Summit Park Investments",
  "positive_chunk_id": "chunk_001",
  "hard_negatives": [
    {"chunk_id": "chunk_001979", "topic": "concessions", ...},
    {"chunk_id": "chunk_002470", "topic": "lease", ...},
    {"chunk_id": "chunk_001_6", "topic": "capex", ...}
  ]
}
```

---

## 🔍 What Mining Does (Optional Improvement)

The `mine_hard_negatives.py` script is an **optional improvement** that:

1. Uses your **trained model** to find semantically similar chunks
2. Retrieves top-50 most similar chunks for each query
3. Picks non-relevant ones as **better hard negatives**

**Why it's better:**
- Uses actual model similarity (not just rules)
- Finds chunks that the model might confuse
- More challenging for training

**Why it's optional:**
- Your current hard negatives are already good (rule-based)
- Mining takes 2-3 hours
- Training with 15,050 pairs will already improve performance significantly

---

## 📊 Comparison

| Aspect | Current (Rule-Based) | After Mining (Model-Based) |
|--------|---------------------|---------------------------|
| **Method** | Metadata matching | Semantic similarity |
| **Speed** | Already done ✅ | 2-3 hours |
| **Quality** | Good | Better |
| **Ready to train?** | Yes ✅ | After mining completes |

---

## 🎯 Recommendation

### Option 1: Train Now (Recommended)
Your pairs already have good hard negatives. Start training:

```bash
python train_embeddings_final.py \
    --train_pairs data/train_pairs_v2.jsonl \
    --corpus data/corpus.jsonl \
    --epochs 5 \
    --learning_rate 2e-5 \
    --output_dir models/re-diligence-embeddings-v2
```

**Why:** You have 15,050 pairs with hard negatives. This is 8x more data than before and will significantly improve performance.

### Option 2: Wait for Mining, Then Train
If mining is still running, you can:
1. Wait for it to finish (~2-3 hours)
2. Then train with `train_pairs_mined.jsonl`

**Why:** Slightly better hard negatives, but marginal improvement vs the huge data increase.

---

## ✅ Bottom Line

**You did NOT skip hard negatives!** Every pair has 3 hard negatives already.

The mining step is just an **optional improvement** that makes them slightly better. But training with your current 15,050 pairs (which already have hard negatives) will give you massive improvement over the previous 1,915 pairs.

**My recommendation:** Start training now. The 8x data increase is more important than the marginal improvement from mining.
