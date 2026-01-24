# Commands Using gpt-oss:120b-cloud for Query Generation

Use the high-quality `gpt-oss:120b-cloud` model specifically for generating the 15,000 train pairs.

## Option 1: If You Already Have a Corpus

If you already generated `data/corpus.jsonl`, just run:

```bash
cd /Users/nithinyanna/Downloads/re-dd-embeddings
source venv/bin/activate

# Build splits with gpt-oss:120b-cloud
python scripts/build_splits.py \
    --data_dir data \
    --target_train_pairs 15000 \
    --eval_queries 500 \
    --model gpt-oss:120b-cloud \
    --seed 42

# Validate
python scripts/validate_dataset.py --data_dir data
```

## Option 2: Full Pipeline (Corpus + Splits)

### Step 1: Generate Corpus (any model - fast is fine)
```bash
cd /Users/nithinyanna/Downloads/re-dd-embeddings
source venv/bin/activate

python scripts/generate_dataset.py \
    --model gemini-3-flash-preview:cloud \
    --out_dir data \
    --companies 60 \
    --seed 7 \
    --target_chunks 3000
```

### Step 2: Generate Queries with gpt-oss:120b-cloud
```bash
python scripts/build_splits.py \
    --data_dir data \
    --target_train_pairs 15000 \
    --eval_queries 500 \
    --model gpt-oss:120b-cloud \
    --seed 42
```

### Step 3: Validate
```bash
python scripts/validate_dataset.py --data_dir data
```

## Option 3: Run Everything with Script

```bash
cd /Users/nithinyanna/Downloads/re-dd-embeddings
./RUN_WITH_GPT_OSS.sh
```

## Why Use gpt-oss:120b-cloud for Queries?

- **Higher Quality**: 120B parameter model produces better analyst-grade queries
- **Better Understanding**: More nuanced understanding of RE diligence terminology
- **Query Diversity**: Generates more varied and realistic search queries
- **Better Hard Negatives**: More accurate hard negative query variants

## Expected Output

The `gpt-oss:120b-cloud` model will generate:
- **15,000 training pairs** with high-quality queries
- **500 eval queries** with realistic analyst search patterns
- All queries will be properly formatted and RE-diligence focused

## Notes

- The corpus can be generated with any model (fast is fine)
- Only the query generation step uses `gpt-oss:120b-cloud`
- This model is larger, so query generation may take longer but quality will be higher
- Make sure you have access to `gpt-oss:120b-cloud` in your Ollama setup

## Check Model Availability

```bash
ollama list | grep gpt-oss
```

If not available, you may need to pull it or check your Ollama cloud configuration.
