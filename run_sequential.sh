#!/bin/bash
# Sequential execution - recommended approach

cd /Users/nithinyanna/Downloads/re-dd-embeddings
source venv/bin/activate

echo "======================================================================"
echo "RE-DD-EMBEDDINGS SEQUENTIAL PIPELINE"
echo "======================================================================"
echo "Step 1: Generate corpus (~2-3 hours)"
echo "Step 2: Generate splits (~1-2 hours)"
echo "Total: ~3-5 hours"
echo "======================================================================"
echo ""

# Step 1: Generate corpus
echo "Starting corpus generation..."
python scripts/generate_dataset.py \
    --model gemini-3-flash-preview:cloud \
    --out_dir data \
    --companies 60 \
    --seed 7 \
    --target_chunks 3000

if [ $? -ne 0 ]; then
    echo "❌ Corpus generation failed!"
    exit 1
fi

echo ""
echo "✓ Corpus generation complete!"
echo ""

# Step 2: Generate splits
echo "Starting split generation..."
python scripts/build_splits.py \
    --data_dir data \
    --target_train_pairs 15000 \
    --eval_queries 500 \
    --model gpt-oss:120b-cloud \
    --seed 42

if [ $? -ne 0 ]; then
    echo "❌ Split generation failed!"
    exit 1
fi

echo ""
echo "✓ Split generation complete!"
echo ""

# Step 3: Validate
echo "Validating dataset..."
python scripts/validate_dataset.py --data_dir data

echo ""
echo "======================================================================"
echo "PIPELINE COMPLETE!"
echo "======================================================================"
echo "Files generated in data/:"
echo "  ✓ corpus.jsonl"
echo "  ✓ train_pairs.jsonl"
echo "  ✓ queries.jsonl"
echo "  ✓ qrels.jsonl"
echo ""
