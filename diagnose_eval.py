#!/usr/bin/env python3
"""
Diagnose evaluation issues - check qrels coverage, oracle performance, baseline comparison.
"""

import json
from pathlib import Path
from collections import defaultdict, Counter
from sentence_transformers import SentenceTransformer
import numpy as np


def load_qrels(qrels_file):
    """Load qrels."""
    qrels = defaultdict(dict)
    with open(qrels_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            qrel = json.loads(line)
            query_id = qrel.get('query_id') or qrel.get('qid')
            qrels[query_id][qrel['doc_id']] = qrel['relevance']
    return qrels


def load_queries(queries_file):
    """Load queries."""
    queries = {}
    with open(queries_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            query = json.loads(line)
            query_id = query.get('query_id') or query.get('qid')
            queries[query_id] = query
    return queries


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


def check_coverage(qrels, queries, corpus):
    """Check qrels coverage sanity."""
    print("="*70)
    print("A. COVERAGE SANITY CHECK")
    print("="*70)
    
    # Check 1: Queries with at least 1 relevant doc
    queries_with_relevants = sum(1 for qid in queries if qid in qrels and len(qrels[qid]) > 0)
    total_queries = len(queries)
    coverage_pct = (queries_with_relevants / total_queries) * 100
    
    print(f"\n1. Queries with at least 1 relevant doc:")
    print(f"   {queries_with_relevants}/{total_queries} ({coverage_pct:.1f}%)")
    
    # Check 2: Average qrels per query
    qrels_per_query = [len(qrels[qid]) for qid in queries if qid in qrels]
    avg_qrels = np.mean(qrels_per_query) if qrels_per_query else 0
    
    print(f"\n2. Average qrels per query:")
    print(f"   {avg_qrels:.2f} (target: 2-4)")
    print(f"   Distribution: {dict(Counter(qrels_per_query))}")
    
    # Check 3: Missing chunk_ids in corpus
    all_doc_ids = set()
    for qid, docs in qrels.items():
        all_doc_ids.update(docs.keys())
    
    missing = [doc_id for doc_id in all_doc_ids if doc_id not in corpus]
    missing_pct = (len(missing) / len(all_doc_ids)) * 100 if all_doc_ids else 0
    
    print(f"\n3. Missing chunk_ids in corpus:")
    print(f"   {len(missing)}/{len(all_doc_ids)} ({missing_pct:.1f}%)")
    if missing:
        print(f"   Examples: {missing[:5]}")
    
    # Check 4: Relevance score distribution
    relevance_scores = []
    for qid, docs in qrels.items():
        relevance_scores.extend(docs.values())
    
    print(f"\n4. Relevance score distribution:")
    print(f"   {dict(Counter(relevance_scores))}")
    
    return {
        'coverage_pct': coverage_pct,
        'avg_qrels': avg_qrels,
        'missing_chunks': len(missing),
        'missing_pct': missing_pct
    }


def oracle_check(queries, qrels, corpus):
    """Oracle check - can we find relevant docs by keyword matching?"""
    print("\n" + "="*70)
    print("B. ORACLE CHECK (Keyword Matching)")
    print("="*70)
    
    oracle_found = 0
    oracle_total = 0
    
    # Key terms that should match
    key_terms = {
        'phase i': ['phase i', 'esa', 'environmental', 'rec'],
        'noi': ['noi', 'net operating income', 't12'],
        'occupancy': ['occupancy', 'occupied', 'vacancy'],
        'lease': ['lease', 'tenant', 'rental'],
        'zoning': ['zoning', 'zoning report', 'entitlement'],
        'alta': ['alta', 'survey', 'title'],
        'cap rate': ['cap rate', 'caprate', 'capitalization']
    }
    
    for qid, query_data in queries.items():
        if qid not in qrels or not qrels[qid]:
            continue
        
        query_text = (query_data.get('query') or query_data.get('text', '')).lower()
        relevant_doc_ids = set(qrels[qid].keys())
        
        # Check if any relevant doc contains keywords from query
        found_by_keyword = False
        for term_group in key_terms.values():
            if any(term in query_text for term in term_group):
                # Check if any relevant doc has this term
                for doc_id in relevant_doc_ids:
                    if doc_id in corpus:
                        doc_text = corpus[doc_id]['text'].lower()
                        if any(term in doc_text for term in term_group):
                            found_by_keyword = True
                            break
                if found_by_keyword:
                    break
        
        oracle_total += 1
        if found_by_keyword:
            oracle_found += 1
    
    oracle_pct = (oracle_found / oracle_total * 100) if oracle_total > 0 else 0
    
    print(f"\nOracle keyword matching:")
    print(f"   {oracle_found}/{oracle_total} queries ({oracle_pct:.1f}%)")
    print(f"   If this is low, qrels/queries may be inconsistent")
    
    return oracle_pct


def baseline_comparison(corpus, queries, qrels, base_model_name='sentence-transformers/all-MiniLM-L6-v2'):
    """Compare fine-tuned model vs base model."""
    print("\n" + "="*70)
    print("C. BASELINE COMPARISON")
    print("="*70)
    
    print(f"\nLoading base model: {base_model_name}...")
    base_model = SentenceTransformer(base_model_name)
    
    # Encode corpus with base model
    print("Encoding corpus with base model...")
    chunk_ids = list(corpus.keys())
    chunk_texts = [corpus[cid]['text'] for cid in chunk_ids]
    base_embeddings = base_model.encode(chunk_texts, show_progress_bar=True, batch_size=32)
    base_embeddings = base_embeddings / np.linalg.norm(base_embeddings, axis=1, keepdims=True)
    
    # Evaluate base model
    print("\nEvaluating base model...")
    mrr_scores = []
    recall_scores = []
    top_k = 10
    
    for qid, query_data in queries.items():
        if qid not in qrels or not qrels[qid]:
            continue
        
        query_text = query_data.get('query') or query_data.get('text')
        relevant_docs = qrels[qid]
        
        # Encode query
        query_emb = base_model.encode([query_text], show_progress_bar=False)[0]
        query_emb = query_emb / np.linalg.norm(query_emb)
        
        # Compute similarities
        similarities = np.dot(base_embeddings, query_emb)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_chunk_ids = [chunk_ids[idx] for idx in top_indices]
        
        # MRR
        mrr = 0.0
        for rank, chunk_id in enumerate(top_chunk_ids, 1):
            if chunk_id in relevant_docs and relevant_docs[chunk_id] > 0:
                mrr = 1.0 / rank
                break
        mrr_scores.append(mrr)
        
        # Recall
        relevant_found = sum(1 for cid in top_chunk_ids if cid in relevant_docs and relevant_docs[cid] > 0)
        total_relevant = len(relevant_docs)
        recall = relevant_found / total_relevant if total_relevant > 0 else 0.0
        recall_scores.append(recall)
    
    base_mrr = np.mean(mrr_scores)
    base_recall = np.mean(recall_scores)
    
    print(f"\nBase Model Performance:")
    print(f"   MRR@10:    {base_mrr:.4f}")
    print(f"   Recall@10: {base_recall:.4f}")
    
    # Load fine-tuned model
    print(f"\nLoading fine-tuned model...")
    try:
        ft_model = SentenceTransformer('models/re-diligence-embeddings')
        ft_embeddings = ft_model.encode(chunk_texts, show_progress_bar=True, batch_size=32)
        ft_embeddings = ft_embeddings / np.linalg.norm(ft_embeddings, axis=1, keepdims=True)
        
        ft_mrr_scores = []
        ft_recall_scores = []
        
        for qid, query_data in queries.items():
            if qid not in qrels or not qrels[qid]:
                continue
            
            query_text = query_data.get('query') or query_data.get('text')
            relevant_docs = qrels[qid]
            
            query_emb = ft_model.encode([query_text], show_progress_bar=False)[0]
            query_emb = query_emb / np.linalg.norm(query_emb)
            
            similarities = np.dot(ft_embeddings, query_emb)
            top_indices = np.argsort(similarities)[::-1][:top_k]
            top_chunk_ids = [chunk_ids[idx] for idx in top_indices]
            
            mrr = 0.0
            for rank, chunk_id in enumerate(top_chunk_ids, 1):
                if chunk_id in relevant_docs and relevant_docs[chunk_id] > 0:
                    mrr = 1.0 / rank
                    break
            ft_mrr_scores.append(mrr)
            
            relevant_found = sum(1 for cid in top_chunk_ids if cid in relevant_docs and relevant_docs[cid] > 0)
            total_relevant = len(relevant_docs)
            recall = relevant_found / total_relevant if total_relevant > 0 else 0.0
            ft_recall_scores.append(recall)
        
        ft_mrr = np.mean(ft_mrr_scores)
        ft_recall = np.mean(ft_recall_scores)
        
        print(f"\nFine-tuned Model Performance:")
        print(f"   MRR@10:    {ft_mrr:.4f}")
        print(f"   Recall@10: {ft_recall:.4f}")
        
        print(f"\nImprovement:")
        print(f"   MRR@10:    {ft_mrr - base_mrr:+.4f} ({((ft_mrr/base_mrr - 1) * 100):+.1f}%)")
        print(f"   Recall@10: {ft_recall - base_recall:+.4f} ({((ft_recall/base_recall - 1) * 100):+.1f}%)")
        
        if ft_mrr < base_mrr * 0.9:
            print(f"\n⚠️  WARNING: Fine-tuned model is WORSE than base!")
            print(f"   This suggests overfitting or misaligned training.")
        elif ft_mrr < base_mrr * 1.1:
            print(f"\n⚠️  Fine-tuned model is similar to base.")
            print(f"   Need more/better training data.")
        else:
            print(f"\n✓ Fine-tuned model is better than base.")
        
    except Exception as e:
        print(f"\n⚠️  Could not load fine-tuned model: {e}")
        print(f"   Base model results only.")
    
    return base_mrr, base_recall


def main():
    print("="*70)
    print("EVALUATION DIAGNOSTICS")
    print("="*70)
    
    corpus = load_corpus('data/corpus.jsonl')
    queries = load_queries('data/queries.jsonl')
    qrels = load_qrels('data/qrels.jsonl')
    
    print(f"\nLoaded:")
    print(f"  Corpus: {len(corpus)} chunks")
    print(f"  Queries: {len(queries)} queries")
    print(f"  Qrels: {len(qrels)} queries with judgments")
    
    # Run checks
    coverage = check_coverage(qrels, queries, corpus)
    oracle_pct = oracle_check(queries, qrels, corpus)
    base_mrr, base_recall = baseline_comparison(corpus, queries, qrels)
    
    # Summary
    print("\n" + "="*70)
    print("DIAGNOSIS SUMMARY")
    print("="*70)
    
    issues = []
    if coverage['coverage_pct'] < 90:
        issues.append(f"⚠️  Only {coverage['coverage_pct']:.1f}% of queries have relevant docs")
    if coverage['avg_qrels'] < 2:
        issues.append(f"⚠️  Average qrels per query is {coverage['avg_qrels']:.2f} (target: 2-4)")
    if coverage['missing_pct'] > 5:
        issues.append(f"⚠️  {coverage['missing_pct']:.1f}% of qrel doc_ids missing from corpus")
    if oracle_pct < 50:
        issues.append(f"⚠️  Oracle keyword matching only {oracle_pct:.1f}% (qrels may be inconsistent)")
    
    if issues:
        print("\nIssues found:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✓ No major issues detected in evaluation setup.")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
