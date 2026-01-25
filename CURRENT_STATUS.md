# Current Status & Next Steps

## ✅ What's Complete

### 1. Corpus Generation ✅
- **3,010 chunks** generated with complete metadata
- **All 25 fields** present including `topic` field
- Validated and fixed

### 2. Evaluation Set ✅
- **500 queries** with qrels
- **1,935 relevance judgments**
- Evaluation diagnostics script created

### 3. Training Infrastructure ✅
- Training scripts (final, working, simple variants)
- Test/evaluation script
- Hard negative mining script
- All pushed to git

### 4. Documentation ✅
- Diagnosis results
- Training results
- Complete data guide
- Next steps guide
- All pushed to git

---

## ⚠️ Current Issue

### Training Pairs Generation
- **Target:** 15,000 pairs (5 queries × 3,010 chunks)
- **Current:** 1,915 pairs (only 12.8% complete)
- **Problem:** Ollama returns "Thinking..." text and incomplete JSON
- **Status:** Extraction logic improved, but may need testing

---

## 🎯 Next Steps

### Option 1: Test Improved Extraction (Recommended)
```bash
# Test with small batch first
python generate_better_pairs.py \
    --corpus data/corpus.jsonl \
    --output data/train_pairs_v2.jsonl \
    --model gemini-3-flash-preview:cloud \
    --batch_size 10
```

**What it does:**
- Generates 5 queries per chunk
- Extracts complete strings from incomplete JSON
- Uses `topic` field for better hard negatives
- Falls back to default queries if extraction fails

### Option 2: Use Existing Pairs + Mine Hard Negatives
```bash
# Mine better hard negatives using current model
python mine_hard_negatives.py \
    --pairs data/train_pairs.jsonl \
    --output data/train_pairs_mined.jsonl \
    --model models/re-diligence-embeddings
```

Then retrain:
```bash
python train_embeddings_final.py \
    --train_pairs data/train_pairs_mined.jsonl \
    --epochs 5 \
    --output_dir models/re-diligence-embeddings-v2
```

### Option 3: Generate More Corpus + Pairs
```bash
# Generate 10,000 chunks
python scripts/generate_dataset.py \
    --target_chunks 10000 \
    --model gemini-3-flash-preview:cloud

# Then generate pairs
python generate_better_pairs.py \
    --corpus data/corpus.jsonl \
    --output data/train_pairs_v2.jsonl
```

---

## 📊 Expected Improvements

**Current Performance:**
- MRR@10: 7.15%
- Recall@10: 3.35%
- **Problem:** Only 1,915 training pairs

**With 15,000 pairs (expected):**
- MRR@10: 15-20% (2-3x improvement)
- Recall@10: 10-15% (3-4x improvement)

**With better hard negatives:**
- Additional 5-10% improvement in MRR

---

## 🔧 Quick Commands

### Check Current Status
```bash
python check_dataset.py
python validate_corpus_completeness.py
python diagnose_eval.py
```

### Generate Better Pairs
```bash
python generate_better_pairs.py \
    --corpus data/corpus.jsonl \
    --output data/train_pairs_v2.jsonl \
    --model gemini-3-flash-preview:cloud
```

### Mine Hard Negatives
```bash
python mine_hard_negatives.py \
    --pairs data/train_pairs_v2.jsonl \
    --output data/train_pairs_mined.jsonl
```

### Train Model
```bash
python train_embeddings_final.py \
    --train_pairs data/train_pairs_mined.jsonl \
    --epochs 5 \
    --output_dir models/re-diligence-embeddings-v2
```

### Evaluate
```bash
python test_model.py \
    --model_dir models/re-diligence-embeddings-v2
```

---

## 📝 Notes

- All code is pushed to git
- Corpus is complete with all metadata
- Extraction logic handles incomplete JSON
- Fallback queries ensure we always get 5 queries per chunk
- Hard negative mining uses `topic` field for better selection

---

## 🚀 Recommended Action

**Start with Option 1** - Test the improved extraction:
```bash
python generate_better_pairs.py \
    --corpus data/corpus.jsonl \
    --output data/train_pairs_v2.jsonl \
    --model gemini-3-flash-preview:cloud \
    --batch_size 10
```

Monitor the output. If it's working (extracting queries successfully), let it run for all chunks. If not, we can try a different approach.
