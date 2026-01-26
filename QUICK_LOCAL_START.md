# Quick Start with Local Models

## 1. Install & Pull Model (One-Time Setup)

```bash
# Install Ollama (if not already installed)
# macOS: brew install ollama
# Or download from https://ollama.ai

# Pull the recommended model
ollama pull llama3.1:8b

# Verify it's installed
ollama list
```

## 2. Generate Training Pairs (Current Priority)

```bash
cd /Users/nithinyanna/Downloads/re-diligence-embeddings

# Generate better training pairs (uses local model by default)
python generate_better_pairs.py \
    --corpus data/corpus.jsonl \
    --output data/train_pairs_v2.jsonl \
    --model llama3.1:8b \
    --batch_size 10
```

This will:
- Generate ~15,000 training pairs (5 queries per chunk)
- Use improved JSON extraction (handles "Thinking..." text)
- Fall back gracefully if extraction fails

## 3. Monitor Progress

Watch for:
- ✅ Successful query extractions
- ⚠️ Fallback queries (should be minimal)
- 📊 Progress: Target is ~15,000 pairs

## 4. Next Steps After Pairs Generated

```bash
# Mine hard negatives (optional but recommended)
python mine_hard_negatives.py \
    --pairs data/train_pairs_v2.jsonl \
    --output data/train_pairs_mined.jsonl \
    --model models/re-diligence-embeddings

# Retrain model
python train_embeddings_final.py \
    --train_pairs data/train_pairs_mined.jsonl \
    --epochs 5 \
    --output_dir models/re-diligence-embeddings-v2
```

## Troubleshooting

**Model not found?**
```bash
ollama pull llama3.1:8b
```

**Too slow?**
- Use smaller model: `--model tinyllama`
- Reduce batch size: `--batch_size 5`

**Out of memory?**
- Use smaller model: `--model phi3:mini`
- Close other applications

**Many extraction failures?**
- Try different model: `--model mistral:7b`
- Check that Ollama is running: `ollama list`

## Model Options

- `llama3.1:8b` - **Recommended** (default, good balance)
- `mistral:7b` - Better JSON generation
- `tinyllama` - Fastest, lower quality
- `phi3:mini` - Small, good for testing

See [LOCAL_MODEL_SETUP.md](LOCAL_MODEL_SETUP.md) for full details.
