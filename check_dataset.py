#!/usr/bin/env python3
"""Quick check of dataset readiness."""

import json
from pathlib import Path

def check_dataset():
    """Check dataset files and provide summary."""
    
    data_dir = Path("data")
    
    print("="*70)
    print("DATASET READINESS CHECK")
    print("="*70)
    print()
    
    # Check train pairs
    train_pairs_file = data_dir / "train_pairs.jsonl"
    if train_pairs_file.exists():
        with open(train_pairs_file, 'r') as f:
            train_pairs = sum(1 for _ in f)
        print(f"✅ Train pairs: {train_pairs:,} pairs")
        
        # Sample one
        with open(train_pairs_file, 'r') as f:
            sample = json.loads(f.readline())
            print(f"   Sample query: {sample['query'][:60]}...")
    else:
        print("❌ Train pairs: File not found")
    
    # Check eval queries
    queries_file = data_dir / "queries.jsonl"
    if queries_file.exists():
        with open(queries_file, 'r') as f:
            queries = sum(1 for _ in f)
        print(f"✅ Eval queries: {queries:,} queries")
    else:
        print("❌ Eval queries: File not found")
    
    # Check qrels
    qrels_file = data_dir / "qrels.jsonl"
    if qrels_file.exists():
        with open(qrels_file, 'r') as f:
            qrels = sum(1 for _ in f)
        print(f"✅ Eval qrels: {qrels:,} qrels")
        
        # Count qrels per query
        qrels_by_query = {}
        with open(qrels_file, 'r') as f:
            for line in f:
                qrel = json.loads(line)
                qid = qrel['qid']
                qrels_by_query[qid] = qrels_by_query.get(qid, 0) + 1
        
        avg_qrels = sum(qrels_by_query.values()) / len(qrels_by_query) if qrels_by_query else 0
        print(f"   Avg qrels per query: {avg_qrels:.1f}")
    else:
        print("❌ Eval qrels: File not found")
    
    # Check corpus
    corpus_file = data_dir / "corpus.jsonl"
    if corpus_file.exists():
        with open(corpus_file, 'r') as f:
            corpus_chunks = sum(1 for _ in f)
        print(f"✅ Corpus chunks: {corpus_chunks:,} chunks")
        
        # Check for unique chunk IDs
        chunk_ids = set()
        duplicates = []
        with open(corpus_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                chunk = json.loads(line)
                chunk_id = chunk.get("chunk_id", "")
                if chunk_id in chunk_ids:
                    duplicates.append((line_num, chunk_id))
                chunk_ids.add(chunk_id)
        
        if duplicates:
            print(f"⚠️  Corpus: {len(duplicates)} duplicate chunk IDs found")
        else:
            print(f"✅ Corpus: All chunk IDs are unique")
    else:
        print("❌ Corpus: File not found")
    
    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    
    all_good = (
        train_pairs_file.exists() and 
        queries_file.exists() and 
        qrels_file.exists() and 
        corpus_file.exists() and
        not duplicates
    )
    
    if all_good:
        print("✅ DATASET IS READY FOR TRAINING!")
        print()
        print("You have:")
        print(f"  • {train_pairs:,} training pairs")
        print(f"  • {queries:,} eval queries")
        print(f"  • {qrels:,} eval qrels")
        print(f"  • {corpus_chunks:,} corpus chunks")
        print()
        print("Start training your embeddings! 🚀")
    else:
        print("⚠️  Some files are missing or have issues")
        print("   Check the errors above")
    
    print("="*70)

if __name__ == "__main__":
    check_dataset()
