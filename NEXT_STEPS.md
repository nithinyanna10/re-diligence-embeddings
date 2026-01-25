# Next Steps After Training

## ✅ What's Done

1. **Dataset Generated**
   - 3,010 corpus chunks
   - 1,915 training pairs
   - 500 eval queries with qrels

2. **Model Trained**
   - 3 epochs completed
   - Final loss: 0.0498
   - Model saved to `models/re-diligence-embeddings`

3. **Model Evaluated**
   - MRR@10: 7.15%
   - Recall@10: 3.35%
   - NDCG@10: 4.56%

---

## 🚀 Quick Actions

### Option 1: Test the Model Now
```bash
python test_model.py --test_only
```
See how it performs on example queries.

### Option 2: Generate More Training Data
```bash
# Generate 5,000 chunks instead of 3,000
python scripts/generate_dataset.py --target_chunks 5000

# Rebuild splits with more data
python scripts/build_splits.py

# Retrain
python train_embeddings_final.py --epochs 5
```

### Option 3: Improve Current Model
```bash
# Retrain with different hyperparameters
python train_embeddings_final.py \
  --epochs 10 \
  --learning_rate 1e-5 \
  --batch_size 16
```

---

## 📊 Performance Improvement Plan

### Phase 1: More Data (Recommended First)
```bash
# Generate 10,000 chunks
python scripts/generate_dataset.py --target_chunks 10000

# Build splits
python scripts/build_splits.py

# Train
python train_embeddings_final.py --epochs 5
```

**Expected Improvement:** MRR@10: 7% → 15-20%

### Phase 2: Better Training
```bash
# Use larger base model
python train_embeddings_final.py \
  --model sentence-transformers/all-mpnet-base-v2 \
  --epochs 10 \
  --learning_rate 2e-5
```

**Expected Improvement:** MRR@10: 15% → 25-30%

### Phase 3: Fine-tuning
- Experiment with different margins (0.3, 0.5, 0.7)
- Try different batch sizes (16, 32, 64)
- Add learning rate scheduling

---

## 🎯 Deployment Options

### 1. HuggingFace Space Demo
```bash
# Create demo corpus
python scripts/sample_space_corpus.py

# Deploy to HF Space
# (See README.md for instructions)
```

### 2. Local API Server
Create a FastAPI server:
```python
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI

app = FastAPI()
model = SentenceTransformer('models/re-diligence-embeddings')

@app.post("/search")
def search(query: str, top_k: int = 10):
    # Load corpus, encode, return top-k
    ...
```

### 3. Export to HuggingFace Hub
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('models/re-diligence-embeddings')
model.save_to_hub('your-username/re-diligence-embeddings')
```

---

## 📝 Documentation Tasks

- [ ] Create model card with metrics
- [ ] Document training procedure
- [ ] Add usage examples
- [ ] Create API documentation

---

## 🔍 Troubleshooting

### Low Performance?
- **More data**: Generate 10K+ chunks
- **Better negatives**: Review hard negative quality
- **Longer training**: Try 10 epochs
- **Larger model**: Use `all-mpnet-base-v2`

### Model Not Saving?
- Check disk space
- Verify write permissions
- Check model directory exists

### Evaluation Errors?
- Verify `queries.jsonl` and `qrels.jsonl` exist
- Check field names match (`qid` vs `query_id`)
- Ensure corpus chunks exist for all doc_ids in qrels

---

## 📈 Success Metrics

**Current Baseline:**
- MRR@10: 7.15%
- Recall@10: 3.35%

**Target (with improvements):**
- MRR@10: > 25%
- Recall@10: > 15%
- NDCG@10: > 20%

---

## 🎉 You're Ready!

Your model is trained and working. Choose your next step:
1. **Test it** - See how it performs
2. **Improve it** - Add more data and retrain
3. **Deploy it** - Create a demo or API

Good luck! 🚀
