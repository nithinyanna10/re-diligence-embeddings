# Optimized Setup for M4 Pro with 24GB RAM

## Recommended Configuration

With **24GB RAM on M4 Pro**, you can run **8-12 parallel workers** for much faster processing.

### Fast Parallel Generation

```bash
cd /Users/nithinyanna/Downloads/re-diligence-embeddings

# Use 8 workers (recommended for 24GB RAM)
python generate_better_pairs_fast.py \
    --corpus data/corpus.jsonl \
    --output data/train_pairs_v2.jsonl \
    --model mistral:7b \
    --workers 8
```

### Performance Expectations

- **Sequential (original)**: ~5+ hours (62s per batch)
- **8 workers**: ~45-60 minutes (estimated)
- **12 workers**: ~30-45 minutes (if you want to push it)

### Resource Usage

- **Mistral 7B model**: ~4.1GB
- **8 workers**: ~8-12GB RAM (Ollama shares model weights efficiently)
- **12 workers**: ~12-16GB RAM
- **CPU**: M4 Pro handles this well with multiple cores

### Monitoring

Watch your system resources:
```bash
# In another terminal
top -o cpu  # or Activity Monitor on macOS
```

If you see high memory pressure, reduce workers:
```bash
--workers 6  # More conservative
```

### Recommended Settings

**For 24GB RAM + M4 Pro:**
- **Default**: `--workers 8` (good balance)
- **Aggressive**: `--workers 12` (if you want maximum speed)
- **Conservative**: `--workers 6` (if running other apps)

### Comparison

| Workers | Est. Time | RAM Usage | CPU Usage |
|---------|-----------|-----------|-----------|
| 1 (sequential) | ~5 hours | ~4GB | Low |
| 4 | ~1.5 hours | ~6-8GB | Medium |
| 8 | ~45-60 min | ~8-12GB | High |
| 12 | ~30-45 min | ~12-16GB | Very High |

### Tips

1. **Close other apps** when running with 12 workers
2. **Start with 8 workers** and monitor
3. **Increase to 12** if system handles it well
4. **Progress is saved incrementally** - safe to stop/resume

### If You Get Out of Memory

Reduce workers:
```bash
--workers 4  # More conservative
```

Or use a smaller model:
```bash
--model tinyllama  # Much smaller, faster
```
