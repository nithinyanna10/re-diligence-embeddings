# Start Training Your Embeddings

## Quick Start

### 1. Install Training Dependencies

```bash
cd /Users/nithinyanna/Downloads/re-diligence-embeddings
source venv/bin/activate

# Install sentence-transformers and PyTorch
pip install sentence-transformers torch
```

### 2. Train Your Model

```bash
# Basic training (recommended first)
python train_embeddings.py

# With custom settings
python train_embeddings.py \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --epochs 3 \
    --batch_size 32 \
    --output_dir models/re-diligence-embeddings
```

### 3. Training Options

**Small model (fast, good for testing):**
```bash
python train_embeddings.py \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --epochs 3 \
    --batch_size 32
```

**Better model (slower, better quality):**
```bash
python train_embeddings.py \
    --model sentence-transformers/all-mpnet-base-v2 \
    --epochs 3 \
    --batch_size 16
```

**Custom training:**
```bash
python train_embeddings.py \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --epochs 5 \
    --batch_size 64 \
    --learning_rate 1e-5 \
    --output_dir models/re-diligence-v2
```

## What Happens During Training

1. **Loads your dataset**:
   - 1,915 training pairs
   - 3,010 corpus chunks

2. **Creates training examples**:
   - Query + Positive chunk + Negative chunk

3. **Trains the model**:
   - Uses Multiple Negatives Ranking Loss
   - Learns to make queries closer to relevant chunks
   - Pushes queries away from irrelevant chunks

4. **Saves the model**:
   - Saved to `models/re-diligence-embeddings/`
   - Ready to use for inference

## Expected Training Time

- **all-MiniLM-L6-v2**: ~5-10 minutes (3 epochs)
- **all-mpnet-base-v2**: ~20-30 minutes (3 epochs)

## After Training

### Test Your Model

```python
from sentence_transformers import SentenceTransformer
import json

# Load your trained model
model = SentenceTransformer('models/re-diligence-embeddings')

# Test query
query = "SNDA status top tenants"
query_embedding = model.encode(query)

# Load corpus and find similar chunks
corpus = []
with open('data/corpus.jsonl', 'r') as f:
    for line in f:
        corpus.append(json.loads(line))

# Encode corpus
corpus_embeddings = model.encode([c['text'] for c in corpus])

# Find top matches
from sklearn.metrics.pairwise import cosine_similarity
similarities = cosine_similarity([query_embedding], corpus_embeddings)[0]
top_indices = similarities.argsort()[-5:][::-1]

print("Top 5 matches:")
for idx in top_indices:
    print(f"  {corpus[idx]['title']}: {similarities[idx]:.3f}")
```

### Evaluate on Eval Set

See `evaluate_model.py` (create this next) for full evaluation with MRR, NDCG, Recall@K.

## Troubleshooting

**Out of memory?**
- Reduce batch_size: `--batch_size 16`
- Use smaller model: `--model sentence-transformers/all-MiniLM-L6-v2`

**Training too slow?**
- Use smaller model
- Reduce epochs: `--epochs 2`

**Want better quality?**
- Use larger model: `--model sentence-transformers/all-mpnet-base-v2`
- More epochs: `--epochs 5`

## Next Steps After Training

1. ✅ **Evaluate** - Test on your 500 eval queries
2. ✅ **Deploy** - Create HuggingFace Space demo
3. ✅ **Use** - Integrate into your application
4. ✅ **Iterate** - Generate more training data if needed

---

**Ready to train? Run:**
```bash
python train_embeddings.py
```
