#!/bin/bash
# Run training with the working script

cd /Users/nithinyanna/Downloads/re-diligence-embeddings
source venv/bin/activate

echo "Starting training..."
python train_embeddings_working.py

echo ""
echo "Training complete! Check models/re-diligence-embeddings/"
