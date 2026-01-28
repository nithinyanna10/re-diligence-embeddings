#!/usr/bin/env python3
"""
Final working training script - uses manual encoding approach.
"""

import json
import argparse
from pathlib import Path
from sentence_transformers import SentenceTransformer, losses
from torch.optim import AdamW
from tqdm import tqdm
import torch
import torch.nn.functional as F
import itertools


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
    """Load training pairs."""
    pairs = []
    skipped = 0
    print(f"Loading train pairs from {pairs_file}...")
    
    with open(pairs_file, 'r') as f:
        for line_num, line in enumerate(tqdm(f, desc="Loading pairs"), 1):
            if not line.strip():
                continue
            
            try:
                pair = json.loads(line)
            except json.JSONDecodeError as e:
                skipped += 1
                if skipped <= 5:  # Only print first 5 errors
                    print(f"⚠ Skipping malformed JSON on line {line_num}: {e}")
                continue
            
            query = pair.get('query')
            if not query:
                continue
            
            # Get positive chunk
            pos_chunk_id = pair.get('positive_chunk_id')
            if not pos_chunk_id or pos_chunk_id not in corpus:
                continue
            pos_chunk = corpus[pos_chunk_id]
            
            # Get negative chunks (keep all available hard negatives)
            neg_chunks = []
            hard_negatives = pair.get('hard_negatives', [])
            if not isinstance(hard_negatives, list):
                continue
            
            for neg in hard_negatives:
                if not isinstance(neg, dict):
                    continue
                neg_chunk_id = neg.get('chunk_id')
                if neg_chunk_id and neg_chunk_id in corpus:
                    neg_chunks.append(corpus[neg_chunk_id]['text'])
            
            if not neg_chunks:
                continue
            
            pairs.append({
                'query': query,
                'positive': pos_chunk['text'],
                # Store all negatives so we can use them with in-batch contrastive loss
                'negatives': neg_chunks
            })
    
    if skipped > 0:
        print(f"⚠ Skipped {skipped} malformed lines (out of {len(pairs) + skipped} total)")
    print(f"✓ Loaded {len(pairs)} training pairs")
    return pairs


def train_model(
    model_name='sentence-transformers/all-MiniLM-L6-v2',
    train_pairs_file='data/train_pairs.jsonl',
    corpus_file='data/corpus.jsonl',
    output_dir='models/re-diligence-embeddings',
    epochs=3,
    batch_size=32,
    learning_rate=2e-5
):
    """Train embedding model using in-batch contrastive loss with hard negatives."""
    
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
    train_pairs = load_train_pairs(train_pairs_file, corpus)
    
    if not train_pairs:
        print("❌ No training pairs loaded!")
        return
    
    # Initialize model
    print(f"\nLoading base model: {model_name}...")
    model = SentenceTransformer(model_name)
    print("✓ Model loaded")
    
    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # Training configuration
    num_batches = (len(train_pairs) + batch_size - 1) // batch_size
    warmup_steps = int(num_batches * epochs * 0.1)
    
    print(f"\nTraining configuration:")
    print(f"  Training pairs: {len(train_pairs):,}")
    print(f"  Batches per epoch: {num_batches:,}")
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
        batch_count = 0
        
        # Shuffle pairs
        import random
        random.shuffle(train_pairs)
        
        for i in tqdm(range(0, len(train_pairs), batch_size), desc=f"Epoch {epoch + 1}"):
            batch = train_pairs[i:i+batch_size]
            
            # Warmup learning rate
            if global_step < warmup_steps:
                lr_scale = min(1.0, float(global_step + 1) / warmup_steps)
                for pg in optimizer.param_groups:
                    pg['lr'] = learning_rate * lr_scale
            
            optimizer.zero_grad()
            
            # Encode queries and documents (positives + all hard negatives) with gradients enabled
            queries = [p['query'] for p in batch]
            positives = [p['positive'] for p in batch]
            # Flatten all negatives in the batch so each query sees:
            # - its own positive as the correct class
            # - all other positives + all negatives as contrasting examples
            negatives = list(itertools.chain.from_iterable(
                p.get('negatives', []) for p in batch
            ))
            docs = positives + negatives
            
            # Use model's tokenizer and forward pass to get embeddings with gradients
            # Access the tokenizer from the first module
            tokenizer = model._first_module().tokenizer
            
            # Tokenize all texts
            query_features = tokenizer(
                queries,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512,
            )
            doc_features = tokenizer(
                docs,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512,
            )
            
            # Move to device
            device = next(model.parameters()).device
            query_features = {k: v.to(device) for k, v in query_features.items()}
            doc_features = {k: v.to(device) for k, v in doc_features.items()}
            
            # Forward pass through model (with gradients)
            query_output = model(query_features)
            doc_output = model(doc_features)
            
            query_embeddings = query_output['sentence_embedding']
            doc_embeddings = doc_output['sentence_embedding']
            
            # Normalize embeddings
            query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
            doc_embeddings = F.normalize(doc_embeddings, p=2, dim=1)
            
            # In-batch contrastive loss:
            # - For batch size B, we treat the first B documents as the positives,
            #   aligned positionally with the B queries.
            # - All other documents (remaining positives and all negatives) act as
            #   implicit negatives.
            batch_size_actual = len(batch)
            logits = torch.matmul(query_embeddings, doc_embeddings.T)  # (B, D)
            
            # Temperature-scaled cross-entropy where the correct class for query i
            # is its own positive at index i
            temperature = 0.05
            logits = logits / temperature
            targets = torch.arange(batch_size_actual, device=device, dtype=torch.long)
            loss = F.cross_entropy(logits, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
            global_step += 1
        
        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
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
    parser.add_argument('--model', default='BAAI/bge-base-en-v1.5', help='Base model name')
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
