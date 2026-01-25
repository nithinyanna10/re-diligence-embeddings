# Training Commands

## Quick Start (No datasets package needed)

```bash
cd /Users/nithinyanna/Downloads/re-diligence-embeddings
source venv/bin/activate

# Install only what's needed
pip install sentence-transformers torch

# Train with simple script (no datasets dependency)
python train_embeddings_simple.py
```

## Full Training Command

```bash
python train_embeddings_simple.py \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --train_pairs data/train_pairs.jsonl \
    --corpus data/corpus.jsonl \
    --output_dir models/re-diligence-embeddings \
    --epochs 3 \
    --batch_size 32 \
    --learning_rate 2e-5
```

## What This Does

1. ✅ Loads your 1,915 training pairs
2. ✅ Loads your 3,010 corpus chunks
3. ✅ Trains for 3 epochs (~5-10 minutes)
4. ✅ Saves model to `models/re-diligence-embeddings/`

## After Training

Test your model:
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('models/re-diligence-embeddings')
query = "SNDA status top tenants"
embedding = model.encode(query)
print(f"Embedding shape: {embedding.shape}")
```

---

**Run this now:**
```bash
python train_embeddings_simple.py
```
