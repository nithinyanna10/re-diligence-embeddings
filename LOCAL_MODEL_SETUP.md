# Local Model Setup Guide

This guide helps you set up and use local Ollama models instead of cloud models (which require credits).

## Quick Start

### 1. Install Ollama

If you haven't already, install Ollama:
```bash
# macOS
brew install ollama

# Or download from: https://ollama.ai
```

### 2. Pull a Local Model

Recommended models for JSON generation and text generation:

**Option A: Llama 3.1 8B (Recommended - Good Balance)**
```bash
ollama pull llama3.1:8b
```
- **Size**: ~4.7 GB
- **Speed**: Fast
- **Quality**: Good for structured outputs
- **Best for**: General use, JSON generation

**Option B: Mistral 7B (Alternative)**
```bash
ollama pull mistral:7b
```
- **Size**: ~4.1 GB
- **Speed**: Fast
- **Quality**: Excellent for structured outputs
- **Best for**: JSON generation, function calling

**Option C: Llama 3.1 70B (Best Quality, Slower)**
```bash
ollama pull llama3.1:70b
```
- **Size**: ~40 GB
- **Speed**: Slower (requires more RAM/VRAM)
- **Quality**: Best
- **Best for**: When quality is critical and you have resources

**Option D: Smaller/Faster Models**
```bash
# TinyLlama (very fast, lower quality)
ollama pull tinyllama

# Phi-3 (Microsoft, good for structured outputs)
ollama pull phi3:mini
```

### 3. Verify Installation

Check that your model is available:
```bash
ollama list
```

You should see your model listed, e.g.:
```
NAME            ID              SIZE    MODIFIED
llama3.1:8b     abc123...       4.7GB   2 hours ago
```

### 4. Test the Model

Test that it works:
```bash
ollama run llama3.1:8b "Return a JSON array with 3 items: [\"test1\", \"test2\", \"test3\"]"
```

You should get a JSON array back.

## Using Local Models in Scripts

All scripts now default to `llama3.1:8b`. You can override with `--model`:

### Generate Corpus
```bash
python scripts/generate_dataset.py \
    --model llama3.1:8b \
    --target_chunks 3000
```

### Generate Training Pairs
```bash
python generate_better_pairs.py \
    --corpus data/corpus.jsonl \
    --output data/train_pairs_v2.jsonl \
    --model llama3.1:8b
```

### Build Splits
```bash
python scripts/build_splits.py \
    --data_dir data \
    --model llama3.1:8b
```

## Model Comparison

| Model | Size | Speed | JSON Quality | RAM Needed | Best For |
|-------|------|-------|--------------|------------|----------|
| `llama3.1:8b` | 4.7GB | ⚡⚡⚡ | ⭐⭐⭐⭐ | 8GB+ | **Recommended default** |
| `mistral:7b` | 4.1GB | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | 8GB+ | JSON generation |
| `llama3.1:70b` | 40GB | ⚡ | ⭐⭐⭐⭐⭐ | 48GB+ | Best quality |
| `tinyllama` | 0.6GB | ⚡⚡⚡⚡ | ⭐⭐ | 2GB+ | Testing/speed |
| `phi3:mini` | 2.3GB | ⚡⚡⚡ | ⭐⭐⭐ | 4GB+ | Balanced |

## Troubleshooting

### Model Not Found
If you get "model not found" errors:
```bash
# List available models
ollama list

# Pull the model again
ollama pull llama3.1:8b
```

### Out of Memory
If you get OOM errors:
1. Use a smaller model: `tinyllama` or `phi3:mini`
2. Close other applications
3. Reduce batch sizes in scripts (e.g., `--batch_size 5`)

### Slow Generation
- Use a smaller model
- Reduce `--target_chunks` for testing
- Use `--batch_size 5` instead of default 10

### JSON Extraction Issues
Local models may have different output formats. The extraction logic handles:
- "Thinking..." text
- Incomplete JSON
- Markdown code blocks

If you see many extraction failures, try:
1. A different model (e.g., `mistral:7b` is better at JSON)
2. Adjusting prompts in the scripts
3. Using a larger model for better compliance

## Performance Tips

1. **Start Small**: Test with `--target_chunks 100` first
2. **Monitor Resources**: Watch RAM/CPU usage
3. **Batch Processing**: Scripts process in batches to manage memory
4. **Use SSD**: Faster disk = faster model loading

## Switching Models

To switch models, just change the `--model` parameter:

```bash
# Use Mistral instead
python generate_better_pairs.py --model mistral:7b

# Use a larger model for better quality
python generate_better_pairs.py --model llama3.1:70b

# Use a smaller model for speed
python generate_better_pairs.py --model tinyllama
```

## Default Model

The default model is now `llama3.1:8b`. To change the default permanently, edit:
- `generate_better_pairs.py` (line ~295)
- `scripts/generate_dataset.py` (line ~349)
- `scripts/build_splits.py` (line ~182)

## Next Steps

1. Pull your chosen model: `ollama pull llama3.1:8b`
2. Test with a small batch: `python generate_better_pairs.py --batch_size 5`
3. Monitor for extraction failures
4. Scale up once working
