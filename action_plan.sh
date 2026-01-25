#!/bin/bash
# Comprehensive action plan to improve model performance

set -e

echo "======================================================================"
echo "ACTION PLAN: Improve RE-Diligence Embeddings"
echo "======================================================================"
echo ""

# Step 1: Diagnose evaluation
echo "Step 1: Diagnosing evaluation..."
python diagnose_eval.py
echo ""

# Step 2: Generate better pairs (if needed)
read -p "Generate better training pairs? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Generating better pairs..."
    python generate_better_pairs.py \
        --corpus data/corpus.jsonl \
        --output data/train_pairs_v2.jsonl \
        --model gemini-3-flash-preview:cloud
    echo ""
fi

# Step 3: Mine hard negatives (if model exists)
if [ -d "models/re-diligence-embeddings" ]; then
    read -p "Mine hard negatives using current model? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Mining hard negatives..."
        python mine_hard_negatives.py \
            --pairs data/train_pairs_v2.jsonl \
            --output data/train_pairs_mined.jsonl \
            --model models/re-diligence-embeddings
        echo ""
    fi
fi

# Step 4: Train with better data
read -p "Train model with improved pairs? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    PAIRS_FILE="data/train_pairs_mined.jsonl"
    if [ ! -f "$PAIRS_FILE" ]; then
        PAIRS_FILE="data/train_pairs_v2.jsonl"
    fi
    if [ ! -f "$PAIRS_FILE" ]; then
        PAIRS_FILE="data/train_pairs.jsonl"
    fi
    
    echo "Training with: $PAIRS_FILE"
    python train_embeddings_final.py \
        --train_pairs "$PAIRS_FILE" \
        --epochs 5 \
        --learning_rate 2e-5 \
        --batch_size 32 \
        --output_dir models/re-diligence-embeddings-v2
    echo ""
fi

# Step 5: Evaluate
read -p "Evaluate new model? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    MODEL_DIR="models/re-diligence-embeddings-v2"
    if [ ! -d "$MODEL_DIR" ]; then
        MODEL_DIR="models/re-diligence-embeddings"
    fi
    
    echo "Evaluating: $MODEL_DIR"
    python test_model.py \
        --model_dir "$MODEL_DIR"
    echo ""
fi

echo "======================================================================"
echo "Action plan complete!"
echo "======================================================================"
