#!/usr/bin/env python3
"""
Build train/eval splits from corpus.
Creates train_pairs.jsonl, queries.jsonl, and qrels.jsonl.
"""

import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Set, Any
from collections import defaultdict
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from ollama_client import run_ollama, parse_json_response


# Topic keyword mapping for tag derivation
TOPIC_KEYWORDS = {
    "lease": ["lease", "tenant", "rental", "lessee", "lessor"],
    "CAM": ["CAM", "common area", "maintenance", "operating expense"],
    "rent_roll": ["rent roll", "rentroll", "tenant list", "occupancy"],
    "T12": ["T-12", "T12", "trailing twelve", "trailing 12"],
    "NOI": ["NOI", "net operating income", "operating income"],
    "capex": ["capex", "capital expenditure", "deferred maintenance", "reserves"],
    "title": ["title", "title commitment", "chain of title"],
    "ALTA": ["ALTA", "survey", "boundary", "encroachment"],
    "PhaseI": ["Phase I", "PhaseI", "ESA", "environmental"],
    "RECs": ["REC", "recognized environmental condition", "contamination"],
    "zoning": ["zoning", "zoning classification", "FAR", "setback"],
    "permits": ["permit", "CO", "certificate of occupancy", "inspection"],
    "insurance": ["insurance", "coverage", "deductible", "loss run"],
    "debt_terms": ["debt", "loan", "term sheet", "lender"],
    "DSCR": ["DSCR", "debt service coverage", "coverage ratio"],
    "debt_yield": ["debt yield", "yield", "DY"],
    "TI_LC": ["TI", "LC", "tenant improvement", "lease commission"],
    "estoppel": ["estoppel", "estoppel certificate"],
    "SNDA": ["SNDA", "subordination", "non-disturbance"],
    "WALE": ["WALE", "weighted average lease"],
    "GPR": ["GPR", "gross potential rent", "potential rent"],
    "vacancy_loss": ["vacancy", "vacant", "vacancy loss"],
    "credit_loss": ["credit loss", "bad debt", "delinquent"],
    "concessions": ["concession", "free rent", "abatement"],
    "rollover": ["rollover", "lease expiration", "expiring", "renewal"],
    "PCA": ["PCA", "property condition", "physical condition"],
    "deferred_maintenance": ["deferred maintenance", "maintenance reserve"],
    "environmental": ["environmental", "contamination", "hazardous"],
    "survey": ["survey", "boundary", "encroachment"],
    "easements": ["easement", "right of way"],
    "zoning_compliance": ["zoning compliance", "nonconforming"],
    "insurance_coverage": ["coverage", "policy", "insured"],
    "debt_covenants": ["covenant", "covenants", "loan covenant"]
}


def derive_tags(text: str, existing_tags: List[str]) -> List[str]:
    """Derive topic tags from text content."""
    text_lower = text.lower()
    derived = set(existing_tags) if existing_tags else set()
    
    for tag, keywords in TOPIC_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            derived.add(tag)
    
    return sorted(list(derived))


def create_query_generation_prompt(chunks_batch: List[Dict], seed: int) -> str:
    """Create prompt for generating analyst queries for chunks."""
    chunks_text = []
    for chunk in chunks_batch:
        chunks_text.append(f"""
CHUNK_ID: {chunk['chunk_id']}
SOURCE_TYPE: {chunk['source_type']}
ASSET_TYPE: {chunk['asset_type']}
TAGS: {', '.join(chunk.get('tags', []))}
TEXT: {chunk['text'][:400]}...
""")
    
    prompt = f"""You are a real estate diligence analyst. For each chunk below, generate 4-6 realistic search queries that an analyst would type when looking for this information.

Queries must sound like what analysts actually search:
- Short keyword searches: "SNDA status top tenants", "Phase I RECs recommended actions"
- Specific questions: "what % of leases roll in next 12 months and mitigation?"
- Diagnostic: "why did NOI drop in Q2? utility expense spike?"
- Numbers: "top 5 tenants % of GPR", "DSCR covenant trigger level"
- Clause hunting: "CAM cap language", "go-dark clause", "co-tenancy anchor definition"
- Due diligence checklist: "estoppels received?", "ALTA encroachments?", "open permits?"

Use REAL RE diligence vocabulary. Do NOT copy sentences verbatim from chunks.

For each chunk, also generate 3 hard-negative query variants (queries that would match OTHER topics in the same asset but NOT this chunk).

Return STRICT JSON ONLY:
{{
  "items": [
    {{
      "chunk_id": "chunk_001",
      "queries": [
        {{"query": "...", "query_type": "direct|diagnostic|evidence|risk|numbers|clause_hunt"}},
        ...
      ],
      "hard_negative_query_variants": ["...", "...", "..."]
    }}
  ]
}}

No markdown, no code blocks, no explanation. Return STRICT JSON ONLY.

CHUNKS:
{''.join(chunks_text)}"""
    
    return prompt


def generate_queries_for_chunks(model: str, chunks: List[Dict], seed: int) -> Dict[str, Any]:
    """Generate queries for a batch of chunks."""
    prompt = create_query_generation_prompt(chunks, seed)
    
    try:
        response = run_ollama(model, prompt, timeout_s=300)
        result = parse_json_response(response)
        return result
    except Exception as e:
        print(f"Error generating queries: {e}")
        return {"items": []}


def find_hard_negatives(chunk: Dict, corpus_by_company: Dict[str, List[Dict]], 
                        exclude_chunk_ids: Set[str]) -> List[Dict[str, str]]:
    """Find 2 hard negative chunks from same company but different topic."""
    company = chunk["company"]
    chunk_tags = set(chunk.get("tags", []))
    chunk_source = chunk["source_type"]
    
    candidates = []
    for other_chunk in corpus_by_company.get(company, []):
        if other_chunk["chunk_id"] in exclude_chunk_ids:
            continue
        if other_chunk["chunk_id"] == chunk["chunk_id"]:
            continue
        
        other_tags = set(other_chunk.get("tags", []))
        other_source = other_chunk["source_type"]
        
        # Prefer different source type and minimal tag overlap
        tag_overlap = len(chunk_tags & other_tags)
        if other_source != chunk_source and tag_overlap <= 1:
            candidates.append({
                "chunk_id": other_chunk["chunk_id"],
                "doc_id": other_chunk["doc_id"],
                "overlap": tag_overlap,
                "different_source": other_source != chunk_source
            })
    
    # Sort by least overlap, prefer different source
    candidates.sort(key=lambda x: (x["overlap"], not x["different_source"]))
    
    return [{"chunk_id": c["chunk_id"], "doc_id": c["doc_id"]} 
            for c in candidates[:2]]


def main():
    parser = argparse.ArgumentParser(description="Build train/eval splits")
    parser.add_argument("--data_dir", default="data", help="Data directory")
    parser.add_argument("--target_train_pairs", type=int, default=15000, help="Target train pairs")
    parser.add_argument("--eval_queries", type=int, default=500, help="Number of eval queries")
    parser.add_argument("--model", default="gemini-3-flash-preview:cloud", help="Ollama model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    random.seed(args.seed)
    
    data_dir = Path(args.data_dir)
    corpus_file = data_dir / "corpus.jsonl"
    
    # Load corpus
    print("Loading corpus...")
    corpus = []
    corpus_by_company = defaultdict(list)
    corpus_by_chunk_id = {}
    
    with open(corpus_file, "r") as f:
        for line in f:
            chunk = json.loads(line)
            corpus.append(chunk)
            corpus_by_company[chunk["company"]].append(chunk)
            corpus_by_chunk_id[chunk["chunk_id"]] = chunk
    
    print(f"Loaded {len(corpus)} chunks from {len(corpus_by_company)} companies")
    
    # Derive tags if missing
    print("Deriving tags...")
    for chunk in corpus:
        if not chunk.get("tags"):
            chunk["tags"] = derive_tags(chunk["text"], [])
    
    # Generate train pairs
    print()
    print("="*70)
    print("BUILDING TRAIN/EVAL SPLITS")
    print("="*70)
    print(f"Target train pairs: {args.target_train_pairs:,}")
    print(f"Target eval queries: {args.eval_queries:,}")
    print("="*70)
    print()
    
    train_pairs_file = data_dir / "train_pairs.jsonl"
    
    batch_size = 8
    pair_id = 0
    used_chunk_ids = set()
    skipped_no_negatives = 0
    skipped_no_queries = 0
    
    with open(train_pairs_file, "w") as f:
        # Process in batches
        total_batches = (len(corpus) + batch_size - 1) // batch_size
        pbar = tqdm(
            range(0, len(corpus), batch_size),
            desc="Train Pairs",
            unit="batch",
            total=total_batches,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} batches [{elapsed}<{remaining}] | Pairs: {postfix}"
        )
        
        for i in pbar:
            if pair_id >= args.target_train_pairs:
                pbar.set_postfix({
                    "Generated": f"{pair_id:,}/{args.target_train_pairs:,}",
                    "Remaining": "0",
                    "Progress": "100.0%",
                    "Status": "TARGET REACHED"
                })
                break
            
            remaining_pairs = args.target_train_pairs - pair_id
            pct_complete = (pair_id / args.target_train_pairs * 100) if args.target_train_pairs > 0 else 0
            pbar.set_postfix({
                "Generated": f"{pair_id:,}/{args.target_train_pairs:,}",
                "Remaining": f"{remaining_pairs:,}",
                "Progress": f"{pct_complete:.1f}%"
            })
            
            batch = corpus[i:i+batch_size]
            batch = [c for c in batch if c["chunk_id"] not in used_chunk_ids]
            
            if not batch:
                continue
            
            # Generate queries
            query_result = generate_queries_for_chunks(args.model, batch, args.seed + i)
            
            for item in query_result.get("items", []):
                chunk_id = item["chunk_id"]
                chunk = corpus_by_chunk_id.get(chunk_id)
                if not chunk:
                    continue
                
                queries = item.get("queries", [])
                if not queries:
                    skipped_no_queries += 1
                    continue
                
                # Find hard negatives
                hard_negatives = find_hard_negatives(chunk, corpus_by_company, used_chunk_ids)
                if len(hard_negatives) < 2:
                    skipped_no_negatives += 1
                    continue
                
                # Create pairs for each query
                for q_item in queries[:5]:  # Up to 5 queries per chunk
                    if pair_id >= args.target_train_pairs:
                        break
                    
                    query_text = q_item.get("query", "")
                    if not query_text:
                        continue
                    
                    pair = {
                        "pair_id": f"pair_{pair_id:06d}",
                        "query": query_text,
                        "positive_chunk_id": chunk_id,
                        "positive_doc_id": chunk["doc_id"],
                        "hard_negatives": hard_negatives[:2],
                        "meta": {
                            "company": chunk["company"],
                            "sector": "Real Estate",
                            "source_type": chunk["source_type"],
                            "deal_stage": chunk["deal_stage"],
                            "date": chunk["date"],
                            "asset_type": chunk["asset_type"],
                            "deal_type": chunk["deal_type"],
                            "market": chunk["market"],
                            "region": chunk["region"]
                        }
                    }
                    
                    f.write(json.dumps(pair) + "\n")
                    pair_id += 1
                    used_chunk_ids.add(chunk_id)
    
    print()
    print(f"✓ Generated {pair_id:,} train pairs")
    if skipped_no_negatives > 0:
        print(f"⚠ Skipped {skipped_no_negatives} chunks (no hard negatives)")
    if skipped_no_queries > 0:
        print(f"⚠ Skipped {skipped_no_queries} chunks (no queries generated)")
    
    # Generate eval set
    print()
    print("="*70)
    print("GENERATING EVAL SET")
    print("="*70)
    queries_file = data_dir / "queries.jsonl"
    qrels_file = data_dir / "qrels.jsonl"
    
    # Sample chunks for eval (diverse by company, source_type, tags)
    print("Sampling eval chunks...")
    eval_chunks = []
    seen_companies = set()
    seen_sources = defaultdict(int)
    
    random.shuffle(corpus)
    for chunk in corpus:
        if len(eval_chunks) >= args.eval_queries:
            break
        
        company = chunk["company"]
        source = chunk["source_type"]
        
        # Prefer diversity
        if company not in seen_companies or seen_sources[source] < 50:
            eval_chunks.append(chunk)
            seen_companies.add(company)
            seen_sources[source] += 1
    
    print(f"✓ Sampled {len(eval_chunks)} chunks for eval")
    print()
    
    # Generate queries and qrels
    total_qrels = 0
    with open(queries_file, "w") as qf, open(qrels_file, "w") as rf:
        pbar = tqdm(
            enumerate(eval_chunks[:args.eval_queries]),
            desc="Eval Queries",
            total=min(len(eval_chunks), args.eval_queries),
            unit="query",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} queries [{elapsed}<{remaining}] | Qrels: {postfix}"
        )
        
        for qid, chunk in pbar:
            # Generate query for this chunk
            query_result = generate_queries_for_chunks(args.model, [chunk], args.seed + qid * 1000)
            
            if query_result.get("items"):
                queries = query_result["items"][0].get("queries", [])
                if queries:
                    query_text = queries[0]["query"]
                else:
                    query_text = f"Find information about {chunk.get('tags', ['property'])[0]} in {chunk['company']}"
            else:
                query_text = f"Find information about {chunk.get('tags', ['property'])[0]} in {chunk['company']}"
            
            # Write query
            query_obj = {
                "qid": f"q_{qid:04d}",
                "text": query_text,
                "company": chunk["company"],
                "sector": "Real Estate",
                "source_type": chunk["source_type"],
                "deal_stage": chunk["deal_stage"],
                "date": chunk["date"],
                "asset_type": chunk["asset_type"],
                "deal_type": chunk["deal_type"],
                "market": chunk["market"],
                "region": chunk["region"]
            }
            qf.write(json.dumps(query_obj) + "\n")
            
            # Find relevant chunks (2-4 total: 1 primary relevance=2, others=1)
            qid_str = query_obj["qid"]
            chunk_tags = set(chunk.get("tags", []))
            chunk_source = chunk["source_type"]
            chunk_company = chunk["company"]
            
            # Primary match (relevance 2)
            rf.write(json.dumps({
                "qid": qid_str,
                "doc_id": chunk["chunk_id"],
                "relevance": 2
            }) + "\n")
            total_qrels += 1
            
            # Supporting matches (relevance 1) - same company, related tags or source
            supporting = []
            for other_chunk in corpus_by_company.get(chunk_company, []):
                if other_chunk["chunk_id"] == chunk["chunk_id"]:
                    continue
                
                other_tags = set(other_chunk.get("tags", []))
                overlap = len(chunk_tags & other_tags)
                
                if overlap >= 1 or other_chunk["source_type"] == chunk_source:
                    supporting.append(other_chunk)
            
            # Select 1-3 supporting chunks
            random.shuffle(supporting)
            for supp_chunk in supporting[:3]:
                rf.write(json.dumps({
                    "qid": qid_str,
                    "doc_id": supp_chunk["chunk_id"],
                    "relevance": 1
                }) + "\n")
                total_qrels += 1
            
            # Update progress
            pbar.set_postfix({
                "Qrels": f"{total_qrels:,}",
                "Avg/Query": f"{total_qrels/(qid+1):.1f}"
            })
    
    print()
    print("="*70)
    print("SPLIT GENERATION COMPLETE")
    print("="*70)
    print(f"✓ Train pairs:         {pair_id:,}/{args.target_train_pairs:,} ({pair_id/args.target_train_pairs*100:.1f}%)")
    print(f"✓ Eval queries:        {len(eval_chunks):,}/{args.eval_queries:,} ({len(eval_chunks)/args.eval_queries*100:.1f}%)")
    print(f"✓ Eval qrels:          {total_qrels:,} (avg {total_qrels/len(eval_chunks):.1f} per query)")
    print(f"✓ Train pairs file:    {train_pairs_file}")
    print(f"✓ Queries file:        {queries_file}")
    print(f"✓ Qrels file:          {qrels_file}")
    print("="*70)


if __name__ == "__main__":
    main()
