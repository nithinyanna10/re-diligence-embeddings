#!/usr/bin/env python3
"""
Mine hard negatives using current model retrieval.
For each (query, positive), retrieve top 50, pick non-relevant as hard negatives.
"""

import json
import argparse
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
from collections import defaultdict


def load_corpus(corpus_file):
    """Load corpus."""
    corpus = {}
    with open(corpus_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            chunk = json.loads(line)
            corpus[chunk['chunk_id']] = chunk
    return corpus


def load_pairs(pairs_file):
    """Load existing pairs."""
    pairs = []
    skipped = 0
    with open(pairs_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                pairs.append(json.loads(line))
            except json.JSONDecodeError as e:
                skipped += 1
                if skipped <= 5:  # Only print first 5 errors
                    print(f"⚠ Skipping malformed JSON on line {line_num}: {e}")
                    print(f"   Line preview: {line[:200]}...")
    if skipped > 0:
        print(f"⚠ Skipped {skipped} malformed lines (out of {len(pairs) + skipped} total)")
    return pairs


def mine_hard_negatives(model, corpus, pairs, top_k=50, num_negatives=3):
    """Mine hard negatives using model retrieval."""
    
    print("Encoding corpus...")
    chunk_ids = list(corpus.keys())
    chunk_texts = [corpus[cid]['text'] for cid in chunk_ids]
    chunk_embeddings = model.encode(chunk_texts, show_progress_bar=True, batch_size=32)
    chunk_embeddings = chunk_embeddings / np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
    
    print(f"\nMining hard negatives for {len(pairs)} pairs...")
    
    improved_pairs = []
    
    for pair in tqdm(pairs, desc="Mining negatives"):
        query = pair['query']
        positive_id = pair['positive_chunk_id']
        
        # Encode query
        query_emb = model.encode([query], show_progress_bar=False)[0]
        query_emb = query_emb / np.linalg.norm(query_emb)
        
        # Retrieve top-k
        similarities = np.dot(chunk_embeddings, query_emb)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_chunk_ids = [chunk_ids[idx] for idx in top_indices]
        
        # Filter out positive and existing negatives
        exclude = {positive_id}
        if 'hard_negatives' in pair:
            exclude.update([n['chunk_id'] for n in pair['hard_negatives']])
        
        # Get hard negatives from top results
        hard_negatives = []
        for cid in top_chunk_ids:
            if cid not in exclude and cid in corpus:
                hard_negatives.append({
                    'chunk_id': cid,
                    'topic': corpus[cid].get('topic', ''),
                    'source_type': corpus[cid].get('source_type', '')
                })
                if len(hard_negatives) >= num_negatives:
                    break
        
        # If we didn't find enough, keep existing or add random
        if len(hard_negatives) < num_negatives and 'hard_negatives' in pair:
            existing = pair['hard_negatives'][:num_negatives - len(hard_negatives)]
            hard_negatives.extend(existing)
        
        # Update pair
        pair['hard_negatives'] = hard_negatives[:num_negatives]
        improved_pairs.append(pair)
    
    return improved_pairs


def main():
    parser = argparse.ArgumentParser(description="Mine hard negatives using retrieval")
    parser.add_argument('--pairs', default='data/train_pairs.jsonl', help='Input pairs file')
    parser.add_argument('--corpus', default='data/corpus.jsonl', help='Corpus file')
    parser.add_argument('--model', default='models/re-diligence-embeddings', help='Model for retrieval')
    parser.add_argument('--output', default='data/train_pairs_mined.jsonl', help='Output file')
    parser.add_argument('--top_k', type=int, default=50, help='Retrieve top K for mining')
    parser.add_argument('--num_negatives', type=int, default=3, help='Number of hard negatives per pair')
    
    args = parser.parse_args()
    
    print("="*70)
    print("HARD NEGATIVE MINING")
    print("="*70)
    print(f"Pairs: {args.pairs}")
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    print("="*70)
    print()
    
    # Load data
    corpus = load_corpus(args.corpus)
    pairs = load_pairs(args.pairs)
    
    print(f"Loaded {len(corpus)} chunks, {len(pairs)} pairs")
    print()
    
    # Load model
    print(f"Loading model: {args.model}...")
    model = SentenceTransformer(args.model)
    print("✓ Model loaded")
    print()
    
    # Mine hard negatives
    improved_pairs = mine_hard_negatives(
        model, corpus, pairs,
        top_k=args.top_k,
        num_negatives=args.num_negatives
    )
    
    # Save
    print(f"\nSaving improved pairs to {args.output}...")
    with open(args.output, 'w') as f:
        for pair in improved_pairs:
            f.write(json.dumps(pair) + '\n')
    
    print(f"✓ Saved {len(improved_pairs)} pairs with mined hard negatives")
    print()
    print("Next: Train with these pairs:")
    print(f"  python train_embeddings_final.py --train_pairs {args.output}")


if __name__ == "__main__":
    main()
