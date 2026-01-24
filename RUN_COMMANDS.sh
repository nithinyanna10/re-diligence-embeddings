#!/bin/bash
# Complete pipeline commands for RE-DD-Embeddings dataset generation

# ============================================================================
# SETUP (One-time, if not already done)
# ============================================================================
cd /Users/nithinyanna/Downloads/re-dd-embeddings
source venv/bin/activate

# Verify Ollama is working
echo "Testing Ollama..."
ollama run gemini-3-flash-preview:cloud "test" > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Ollama is working"
else
    echo "✗ Ollama not working. Make sure it's installed and model is pulled."
    exit 1
fi

# ============================================================================
# OPTION 1: QUICK TEST (5 deals, ~250 chunks) - ~10-15 minutes
# ============================================================================
echo ""
echo "======================================================================"
echo "QUICK TEST RUN (Recommended first)"
echo "======================================================================"
echo ""

# Step 1: Generate corpus
python scripts/generate_dataset.py \
    --model gemini-3-flash-preview:cloud \
    --out_dir data \
    --companies 5 \
    --seed 7 \
    --target_chunks 250

# Step 2: Build splits
python scripts/build_splits.py \
    --data_dir data \
    --target_train_pairs 1000 \
    --eval_queries 50 \
    --model gemini-3-flash-preview:cloud \
    --seed 42

# Step 3: Validate
python scripts/validate_dataset.py --data_dir data

# ============================================================================
# OPTION 2: FULL DATASET (60 deals, ~3,000 chunks) - ~2-4 hours
# ============================================================================
# Uncomment below to run full dataset generation

# echo ""
# echo "======================================================================"
# echo "FULL DATASET GENERATION"
# echo "======================================================================"
# echo ""
# 
# # Step 1: Generate corpus (~2-3 hours)
# python scripts/generate_dataset.py \
#     --model gemini-3-flash-preview:cloud \
#     --out_dir data \
#     --companies 60 \
#     --seed 7 \
#     --target_chunks 3000
# 
# # Step 2: Build splits (~1-2 hours)
# python scripts/build_splits.py \
#     --data_dir data \
#     --target_train_pairs 15000 \
#     --eval_queries 500 \
#     --model gemini-3-flash-preview:cloud \
#     --seed 42
# 
# # Step 3: Validate
# python scripts/validate_dataset.py --data_dir data
# 
# # Step 4: Sample Space corpus (optional)
# python scripts/sample_space_corpus.py \
#     --data_dir data \
#     --k 300 \
#     --seed 123

echo ""
echo "======================================================================"
echo "DONE!"
echo "======================================================================"
echo "Check data/ directory for generated files:"
echo "  - corpus.jsonl"
echo "  - train_pairs.jsonl"
echo "  - queries.jsonl"
echo "  - qrels.jsonl"
echo ""
