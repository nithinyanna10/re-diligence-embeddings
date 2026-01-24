#!/bin/bash
# Run with gpt-oss:120b-cloud for query generation

cd /Users/nithinyanna/Downloads/re-dd-embeddings
source venv/bin/activate

echo "======================================================================"
echo "RE-DD-EMBEDDINGS DATASET GENERATION"
echo "======================================================================"
echo "Corpus Model: (use existing corpus.jsonl or generate with any model)"
echo "Query Model: gpt-oss:120b-cloud (for 15,000 train pairs)"
echo "======================================================================"
echo ""

# Check if corpus exists
if [ ! -f "data/corpus.jsonl" ]; then
    echo "⚠ No corpus found. Generate it first with:"
    echo "   python scripts/generate_dataset.py --model <any_model> --companies 60 --target_chunks 3000"
    echo ""
    read -p "Generate corpus now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Generating corpus..."
        python scripts/generate_dataset.py \
            --model gemini-3-flash-preview:cloud \
            --out_dir data \
            --companies 60 \
            --seed 7 \
            --target_chunks 3000
    else
        echo "Exiting. Generate corpus first."
        exit 1
    fi
fi

echo ""
echo "======================================================================"
echo "BUILDING SPLITS WITH GPT-OSS:120B-CLOUD"
echo "======================================================================"
echo "This will use the high-quality gpt-oss:120b-cloud model for query generation"
echo ""

# Build splits with gpt-oss:120b-cloud
python scripts/build_splits.py \
    --data_dir data \
    --target_train_pairs 15000 \
    --eval_queries 500 \
    --model gpt-oss:120b-cloud \
    --seed 42

echo ""
echo "======================================================================"
echo "VALIDATING DATASET"
echo "======================================================================"
python scripts/validate_dataset.py --data_dir data

echo ""
echo "======================================================================"
echo "DONE!"
echo "======================================================================"
echo "Generated files in data/:"
echo "  ✓ train_pairs.jsonl (15,000 pairs with gpt-oss:120b-cloud queries)"
echo "  ✓ queries.jsonl (500 eval queries)"
echo "  ✓ qrels.jsonl (relevance labels)"
echo ""
