# Fixed Training Script

## The Fix

The issue was that PyTorch's default DataLoader can't batch `InputExample` objects. The fix uses `NoDuplicatesDataLoader` from sentence-transformers which handles this properly.

## Run Training

```bash
cd /Users/nithinyanna/Downloads/re-diligence-embeddings
source venv/bin/activate

python train_embeddings_simple.py
```

## What Changed

- ✅ Uses `NoDuplicatesDataLoader` instead of regular `DataLoader`
- ✅ Properly handles `InputExample` batching
- ✅ No datasets package required
- ✅ Manual training loop with warmup

## Expected Output

```
======================================================================
STARTING TRAINING
======================================================================

Epoch 1/3
Epoch 1: 100%|████████| 60/60 [XX:XX<00:00]
  Average loss: X.XXXX

Epoch 2/3
Epoch 2: 100%|████████| 60/60 [XX:XX<00:00]
  Average loss: X.XXXX

Epoch 3/3
Epoch 3: 100%|████████| 60/60 [XX:XX<00:00]
  Average loss: X.XXXX

✓ Model saved to: models/re-diligence-embeddings
```

---

**Run this command:**
```bash
python train_embeddings_simple.py
```
