# Using Different Models for Different Steps

**Yes, you can use different Ollama models for corpus generation vs. split generation!**

The scripts are independent - `generate_dataset.py` creates the corpus, and `build_splits.py` generates queries from that corpus. They don't need to use the same model.

## Why Use Different Models?

- **Faster model for corpus**: Use a faster model to generate lots of chunks quickly
- **Better model for queries**: Use a more capable model to generate high-quality analyst queries
- **Cost/Performance tradeoff**: Balance speed vs. quality per step

## Examples

### Example 1: Fast Corpus + Quality Queries

```bash
# Step 1: Generate corpus with fast model
python scripts/generate_dataset.py \
    --model llama3.2:3b \
    --out_dir data \
    --companies 60 \
    --target_chunks 3000

# Step 2: Generate queries with better model
python scripts/build_splits.py \
    --data_dir data \
    --target_train_pairs 15000 \
    --eval_queries 500 \
    --model gemini-3-flash-preview:cloud \
    --seed 42
```

### Example 2: Cloud Model for Corpus, Local for Queries

```bash
# Step 1: Use cloud model for corpus (if you have access)
python scripts/generate_dataset.py \
    --model gemini-3-flash-preview:cloud \
    --out_dir data \
    --companies 60 \
    --target_chunks 3000

# Step 2: Use local model for queries (faster, no API limits)
python scripts/build_splits.py \
    --data_dir data \
    --target_train_pairs 15000 \
    --eval_queries 500 \
    --model llama3.1:8b \
    --seed 42
```

### Example 3: Same Model (Default)

```bash
# Both use same model
python scripts/generate_dataset.py \
    --model gemini-3-flash-preview:cloud \
    --out_dir data \
    --companies 60 \
    --target_chunks 3000

python scripts/build_splits.py \
    --data_dir data \
    --target_train_pairs 15000 \
    --eval_queries 500 \
    --model gemini-3-flash-preview:cloud \
    --seed 42
```

## Available Ollama Models

Check what models you have:
```bash
ollama list
```

Pull a model if needed:
```bash
ollama pull llama3.2:3b
ollama pull llama3.1:8b
ollama pull mistral:7b
ollama pull gemini-3-flash-preview:cloud
```

## Considerations

1. **Consistency**: Different models may produce slightly different styles, but this is usually fine for training embeddings
2. **Quality**: Using a better model for query generation can improve the quality of your training pairs
3. **Speed**: Using a faster model for corpus generation can significantly reduce total time
4. **JSON Format**: All models should follow the same JSON output format (enforced by prompts)

## Recommended Combinations

| Use Case | Corpus Model | Query Model | Reason |
|----------|-------------|-------------|--------|
| **Fastest** | `llama3.2:3b` | `llama3.2:3b` | Both fast, good quality |
| **Balanced** | `llama3.1:8b` | `gemini-3-flash-preview:cloud` | Fast corpus, quality queries |
| **Best Quality** | `gemini-3-flash-preview:cloud` | `gemini-3-flash-preview:cloud` | Best for both |
| **Budget** | `llama3.2:3b` | `llama3.2:3b` | Free, local only |

## Full Example with Different Models

```bash
cd /Users/nithinyanna/Downloads/re-dd-embeddings
source venv/bin/activate

# Generate corpus with fast local model
python scripts/generate_dataset.py \
    --model llama3.2:3b \
    --out_dir data \
    --companies 60 \
    --seed 7 \
    --target_chunks 3000

# Generate splits with cloud model (better query quality)
python scripts/build_splits.py \
    --data_dir data \
    --target_train_pairs 15000 \
    --eval_queries 500 \
    --model gemini-3-flash-preview:cloud \
    --seed 42

# Validate (no model needed)
python scripts/validate_dataset.py --data_dir data
```

The corpus and queries will work together perfectly regardless of which models generated them!
