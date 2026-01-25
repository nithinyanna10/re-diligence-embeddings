#!/usr/bin/env python3
"""
Train embeddings model on RE-Diligence dataset.
Uses sentence-transformers for training.
"""

import json
import argparse
from pathlib import Path
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
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
    """Load training pairs and create InputExamples."""
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
            
            # Get negative chunks
            neg_chunks = []
            for neg in pair['hard_negatives']:
                neg_chunk_id = neg['chunk_id']
                if neg_chunk_id in corpus:
                    neg_chunks.append(corpus[neg_chunk_id]['text'])
            
            if not neg_chunks:
                continue
            
            # Create InputExample with query, positive, and first negative
            train_examples.append(InputExample(
                texts=[query, pos_chunk['text'], neg_chunks[0]]
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
    
    # Define loss
    print("\nSetting up training...")
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    # Create dataloader
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=batch_size
    )
    
    # Training arguments
    warmup_steps = int(len(train_dataloader) * epochs * 0.1)
    
    print(f"\nTraining configuration:")
    print(f"  Training examples: {len(train_examples):,}")
    print(f"  Batches per epoch: {len(train_dataloader):,}")
    print(f"  Warmup steps: {warmup_steps}")
    print()
    
    # Train using manual training loop (doesn't require datasets package)
    print("="*70)
    print("STARTING TRAINING")
    print("="*70)
    
    from torch.optim import AdamW
    from sentence_transformers.util import batch_to_device
    from sentence_transformers.trainer import SentenceTransformerTrainer
    from sentence_transformers.training_args import SentenceTransformerTrainingArguments
    
    # Create training arguments
    training_args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        warmup_steps=warmup_steps,
        learning_rate=learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
    )
    
    # Create trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_objectives=[(train_dataloader, train_loss)],
    )
    
    # Train
    trainer.train()
    
    # Save final model
    print(f"\nSaving final model to {output_dir}...")
    model.save(output_dir)
    
    print()
    print("="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"✓ Model saved to: {output_dir}")
    print()
    print("Next steps:")
    print("  1. Evaluate on eval set (see evaluate_model.py)")
    print("  2. Test on sample queries")
    print("  3. Deploy to HuggingFace Space")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Train RE-Diligence embeddings")
    parser.add_argument(
        '--model',
        default='sentence-transformers/all-MiniLM-L6-v2',
        help='Base model name'
    )
    parser.add_argument(
        '--train_pairs',
        default='data/train_pairs.jsonl',
        help='Training pairs file'
    )
    parser.add_argument(
        '--corpus',
        default='data/corpus.jsonl',
        help='Corpus file'
    )
    parser.add_argument(
        '--output_dir',
        default='models/re-diligence-embeddings',
        help='Output directory for trained model'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=2e-5,
        help='Learning rate'
    )
    
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
