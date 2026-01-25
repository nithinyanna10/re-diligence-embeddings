#!/usr/bin/env python3
"""
Working training script for RE-Diligence embeddings.
Uses sentence-transformers training API correctly.
"""

import json
import argparse
from pathlib import Path
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.datasets import NoDuplicatesDataLoader
from torch.optim import AdamW
from tqdm import tqdm
import torch


def load_corpus(corpus_file):
    """Load corpus into dictionary."""
    corpus = {}
    print(f"Loading corpus from {corpus_file}...")
    with open(corpus_file, 'r') as f:
        for line in tqdm(f, desc="Loading corpus"):
            if not line.strip():
                continue
            chunk = json.loads(line)
            corpus[chunk['chunk_id']] = chunk
    print(f"✓ Loaded {len(corpus)} chunks")
    return corpus


def load_train_pairs(pairs_file, corpus):
    """Load training pairs and create InputExamples for MultipleNegativesRankingLoss."""
    train_examples = []
    print(f"Loading train pairs from {pairs_file}...")
    
    with open(pairs_file, 'r') as f:
        for line in tqdm(f, desc="Loading pairs"):
            if not line.strip():
                continue
            
            pair = json.loads(line)
            query = pair['query']
            
            # Get positive chunk
            pos_chunk_id = pair['positive_chunk_id']
            if pos_chunk_id not in corpus:
                continue
            pos_chunk = corpus[pos_chunk_id]
            
            # For MultipleNegativesRankingLoss, we only need (anchor, positive) pairs
            # Negatives are automatically sampled from other positives in the batch
            train_examples.append(InputExample(
                texts=[query, pos_chunk['text']]
            ))
    
    print(f"✓ Loaded {len(train_examples)} training examples")
    return train_examples


def train_model(
    model_name='sentence-transformers/all-MiniLM-L6-v2',
    train_pairs_file='data/train_pairs.jsonl',
    corpus_file='data/corpus.jsonl',
    output_dir='models/re-diligence-embeddings',
    epochs=3,
    batch_size=32,
    learning_rate=2e-5
):
    """Train embedding model."""
    
    print("="*70)
    print("TRAINING RE-DILIGENCE EMBEDDINGS")
    print("="*70)
    print(f"Base model: {model_name}")
    print(f"Output: {output_dir}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}")
    print("="*70)
    print()
    
    # Load data
    corpus = load_corpus(corpus_file)
    train_examples = load_train_pairs(train_pairs_file, corpus)
    
    if not train_examples:
        print("❌ No training examples loaded!")
        return
    
    # Initialize model
    print(f"\nLoading base model: {model_name}...")
    model = SentenceTransformer(model_name)
    print("✓ Model loaded")
    
    # Define loss - MultipleNegativesRankingLoss for (anchor, positive) pairs
    print("\nSetting up training...")
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    # Create dataloader
    train_dataloader = NoDuplicatesDataLoader(
        train_examples,
        batch_size=batch_size
    )
    
    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # Training configuration
    warmup_steps = int(len(train_dataloader) * epochs * 0.1)
    
    print(f"\nTraining configuration:")
    print(f"  Training examples: {len(train_examples):,}")
    print(f"  Batches per epoch: {len(train_dataloader):,}")
    print(f"  Warmup steps: {warmup_steps}")
    print()
    
    # Training loop
    print("="*70)
    print("STARTING TRAINING")
    print("="*70)
    
    model.train()
    global_step = 0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        epoch_loss = 0
        num_batches = 0
        
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}"):
            # Warmup learning rate
            if global_step < warmup_steps:
                lr_scale = min(1.0, float(global_step + 1) / warmup_steps)
                for pg in optimizer.param_groups:
                    pg['lr'] = learning_rate * lr_scale
            
            optimizer.zero_grad()
            
            # Convert InputExamples to sentence features
            # smart_batching_collate for pairs returns nested structure
            # We need to flatten it: [[anchor, positive], ...] -> [anchor, positive, anchor2, positive2, ...]
            collated = model.smart_batching_collate(batch)
            
            # Flatten if nested (for pairs, it might be list of lists)
            if collated and isinstance(collated[0], list):
                sentence_features = [item for sublist in collated for item in sublist]
            else:
                sentence_features = collated
            
            # MultipleNegativesRankingLoss expects list of feature dicts
            # Format: [anchor_dict, positive_dict, anchor_dict2, positive_dict2, ...]
            loss_value = train_loss(sentence_features)
            
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss_value.item()
            num_batches += 1
            global_step += 1
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        print(f"  Average loss: {avg_loss:.4f}")
    
    # Save model
    print(f"\nSaving model to {output_dir}...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save(output_dir)
    
    print()
    print("="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"✓ Model saved to: {output_dir}")
    print()
    print("Next steps:")
    print("  1. Test your model")
    print("  2. Evaluate on eval set")
    print("  3. Deploy to HuggingFace Space")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Train RE-Diligence embeddings")
    parser.add_argument('--model', default='sentence-transformers/all-MiniLM-L6-v2', help='Base model name')
    parser.add_argument('--train_pairs', default='data/train_pairs.jsonl', help='Training pairs file')
    parser.add_argument('--corpus', default='data/corpus.jsonl', help='Corpus file')
    parser.add_argument('--output_dir', default='models/re-diligence-embeddings', help='Output directory')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    
    args = parser.parse_args()
    
    train_model(
        model_name=args.model,
        train_pairs_file=args.train_pairs,
        corpus_file=args.corpus,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )


if __name__ == "__main__":
    main()
