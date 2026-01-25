#!/usr/bin/env python3
"""
Test and evaluate the trained embedding model.
"""

import json
import argparse
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
from collections import defaultdict


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


def load_eval_queries(queries_file):
    """Load evaluation queries."""
    queries = {}
    print(f"Loading queries from {queries_file}...")
    with open(queries_file, 'r') as f:
        for line in tqdm(f, desc="Loading queries"):
            if not line.strip():
                continue
            query = json.loads(line)
            # Handle both 'query_id' and 'qid' field names
            query_id = query.get('query_id') or query.get('qid')
            queries[query_id] = query
    print(f"✓ Loaded {len(queries)} queries")
    return queries


def load_qrels(qrels_file):
    """Load relevance judgments."""
    qrels = defaultdict(dict)
    print(f"Loading qrels from {qrels_file}...")
    with open(qrels_file, 'r') as f:
        for line in tqdm(f, desc="Loading qrels"):
            if not line.strip():
                continue
            qrel = json.loads(line)
            # Handle both 'query_id' and 'qid' field names
            query_id = qrel.get('query_id') or qrel.get('qid')
            qrels[query_id][qrel['doc_id']] = qrel['relevance']
    print(f"✓ Loaded qrels for {len(qrels)} queries")
    return qrels


def evaluate_model(model, corpus, queries, qrels, top_k=10):
    """Evaluate model using MRR, Recall@K, and NDCG@K."""
    print("\n" + "="*70)
    print("EVALUATING MODEL")
    print("="*70)
    
    # Encode all corpus chunks
    print("\nEncoding corpus chunks...")
    chunk_ids = list(corpus.keys())
    chunk_texts = [corpus[cid]['text'] for cid in chunk_ids]
    chunk_embeddings = model.encode(chunk_texts, show_progress_bar=True, batch_size=32)
    chunk_embeddings = chunk_embeddings / np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)  # Normalize
    
    # Create chunk_id to index mapping
    chunk_id_to_idx = {cid: idx for idx, cid in enumerate(chunk_ids)}
    
    # Evaluate each query
    mrr_scores = []
    recall_at_k_scores = []
    ndcg_at_k_scores = []
    
    print(f"\nEvaluating {len(queries)} queries...")
    for query_id, query_data in tqdm(queries.items(), desc="Evaluating"):
        if query_id not in qrels:
            continue
        
        # Handle both 'query' and 'text' field names
        query_text = query_data.get('query') or query_data.get('text')
        relevant_docs = qrels[query_id]
        
        # Encode query
        query_embedding = model.encode([query_text], show_progress_bar=False)[0]
        query_embedding = query_embedding / np.linalg.norm(query_embedding)  # Normalize
        
        # Compute similarities
        similarities = np.dot(chunk_embeddings, query_embedding)
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_chunk_ids = [chunk_ids[idx] for idx in top_indices]
        
        # Compute MRR (Mean Reciprocal Rank)
        mrr = 0.0
        for rank, chunk_id in enumerate(top_chunk_ids, 1):
            if chunk_id in relevant_docs and relevant_docs[chunk_id] > 0:
                mrr = 1.0 / rank
                break
        mrr_scores.append(mrr)
        
        # Compute Recall@K
        relevant_found = sum(1 for cid in top_chunk_ids if cid in relevant_docs and relevant_docs[cid] > 0)
        total_relevant = len(relevant_docs)
        recall_at_k = relevant_found / total_relevant if total_relevant > 0 else 0.0
        recall_at_k_scores.append(recall_at_k)
        
        # Compute NDCG@K (simplified - using relevance scores)
        dcg = 0.0
        for rank, chunk_id in enumerate(top_chunk_ids, 1):
            if chunk_id in relevant_docs:
                relevance = relevant_docs[chunk_id]
                dcg += relevance / np.log2(rank + 1)
        
        # Ideal DCG (sort by relevance)
        ideal_relevances = sorted([r for r in relevant_docs.values() if r > 0], reverse=True)[:top_k]
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevances))
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_at_k_scores.append(ndcg)
    
    # Compute averages
    avg_mrr = np.mean(mrr_scores)
    avg_recall = np.mean(recall_at_k_scores)
    avg_ndcg = np.mean(ndcg_at_k_scores)
    
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"MRR@{top_k}:           {avg_mrr:.4f}")
    print(f"Recall@{top_k}:        {avg_recall:.4f}")
    print(f"NDCG@{top_k}:          {avg_ndcg:.4f}")
    print("="*70)
    
    return {
        'mrr': avg_mrr,
        'recall_at_k': avg_recall,
        'ndcg_at_k': avg_ndcg
    }


def test_example_queries(model, corpus, num_examples=5):
    """Test model with example queries."""
    print("\n" + "="*70)
    print("TESTING WITH EXAMPLE QUERIES")
    print("="*70)
    
    # Encode all corpus chunks
    print("\nEncoding corpus...")
    chunk_ids = list(corpus.keys())
    chunk_texts = [corpus[cid]['text'] for cid in chunk_ids]
    chunk_embeddings = model.encode(chunk_texts, show_progress_bar=True, batch_size=32)
    chunk_embeddings = chunk_embeddings / np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
    
    # Example queries
    example_queries = [
        "What is the occupancy rate of the property?",
        "What are the major tenants and their lease terms?",
        "What environmental issues were found during due diligence?",
        "What is the property's cap rate and NOI?",
        "What zoning restrictions apply to this property?"
    ]
    
    print(f"\nTesting {min(num_examples, len(example_queries))} example queries...\n")
    
    for i, query_text in enumerate(example_queries[:num_examples], 1):
        print(f"{'='*70}")
        print(f"Query {i}: {query_text}")
        print(f"{'='*70}")
        
        # Encode query
        query_embedding = model.encode([query_text], show_progress_bar=False)[0]
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Find top 3 results
        similarities = np.dot(chunk_embeddings, query_embedding)
        top_indices = np.argsort(similarities)[::-1][:3]
        
        for rank, idx in enumerate(top_indices, 1):
            chunk_id = chunk_ids[idx]
            chunk = corpus[chunk_id]
            similarity = similarities[idx]
            
            print(f"\n[Rank {rank}] Similarity: {similarity:.4f}")
            print(f"Chunk ID: {chunk_id}")
            print(f"Source: {chunk.get('source_type', 'N/A')} | Topic: {chunk.get('topic', 'N/A')}")
            print(f"Text preview: {chunk['text'][:200]}...")
        
        print()


def main():
    parser = argparse.ArgumentParser(description="Test and evaluate trained model")
    parser.add_argument('--model_dir', default='models/re-diligence-embeddings', help='Model directory')
    parser.add_argument('--corpus', default='data/corpus.jsonl', help='Corpus file')
    parser.add_argument('--queries', default='data/queries.jsonl', help='Eval queries file')
    parser.add_argument('--qrels', default='data/qrels.jsonl', help='Qrels file')
    parser.add_argument('--test_only', action='store_true', help='Only run example tests, skip evaluation')
    parser.add_argument('--top_k', type=int, default=10, help='Top K for evaluation')
    
    args = parser.parse_args()
    
    # Load model
    print("="*70)
    print("LOADING MODEL")
    print("="*70)
    print(f"Loading model from {args.model_dir}...")
    model = SentenceTransformer(args.model_dir)
    print("✓ Model loaded")
    
    # Load corpus
    corpus = load_corpus(args.corpus)
    
    # Test with example queries
    test_example_queries(model, corpus)
    
    # Evaluate on eval set
    if not args.test_only:
        if Path(args.queries).exists() and Path(args.qrels).exists():
            queries = load_eval_queries(args.queries)
            qrels = load_qrels(args.qrels)
            evaluate_model(model, corpus, queries, qrels, top_k=args.top_k)
        else:
            print(f"\n⚠ Eval files not found. Skipping evaluation.")
            print(f"  Queries: {args.queries}")
            print(f"  Qrels: {args.qrels}")
    
    print("\n" + "="*70)
    print("TESTING COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
