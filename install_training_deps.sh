#!/bin/bash
# Install all dependencies for training

cd /Users/nithinyanna/Downloads/re-diligence-embeddings
source venv/bin/activate

echo "Installing training dependencies..."
pip install sentence-transformers torch datasets scikit-learn

echo "✓ Dependencies installed!"
echo ""
echo "You can now run:"
echo "  python train_embeddings.py"
