#!/usr/bin/env python3
"""
Validate dataset integrity and quality.
Checks schemas, uniqueness, ranges, and PII safety.
"""

import json
import argparse
from pathlib import Path
from typing import Set, Dict, List
from collections import defaultdict, Counter


# Blacklist of real company names to catch PII leaks
COMPANY_BLACKLIST = {
    "apple", "google", "microsoft", "amazon", "meta", "facebook",
    "blackstone", "brookfield", "prologis", "simon", "welltower",
    "equity residential", "avalonbay", "essex", "mid-america",
    "public storage", "extra space", "life storage",
    "realty income", "agreement", "ventas", "hcp", "wellcare",
    "cbre", "jll", "cushman", "colliers", "marcus & millichap"
}


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def validate_corpus(corpus_file: Path) -> tuple[bool, List[str]]:
    """Validate corpus.jsonl."""
    errors = []
    chunk_ids: Set[str] = set()
    companies_seen = set()
    
    required_keys = {
        "doc_id", "chunk_id", "text", "title", "source_type", "company",
        "sector", "deal_stage", "date", "region", "doc_url", "tags",
        "confidentiality", "asset_type", "deal_type", "market", "vintage",
        "unit_count", "sqft", "occupancy_pct", "noi", "cap_rate", "ltv", "dscr"
    }
    
    valid_source_types = {
        "CIM", "IC_Memo", "Diligence_Report", "Lease_Abstract",
        "Rent_Roll_Summary", "T12_Operating_Statement", "PCA_Report",
        "PhaseI_ESA", "Title_Commitment", "ALTA_Survey_Summary",
        "Debt_Term_Sheet", "Insurance_Summary", "Zoning_Report",
        "Permit_Log", "Appraisal_Summary"
    }
    
    with open(corpus_file, "r") as f:
        for line_num, line in enumerate(f, 1):
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: Invalid JSON - {e}")
                continue
            
            # Check required keys
            missing = required_keys - set(chunk.keys())
            if missing:
                errors.append(f"Line {line_num}: Missing keys - {missing}")
            
            # Check chunk_id uniqueness
            chunk_id = chunk.get("chunk_id", "")
            if chunk_id in chunk_ids:
                errors.append(f"Line {line_num}: Duplicate chunk_id - {chunk_id}")
            chunk_ids.add(chunk_id)
            
            # Check word count
            text = chunk.get("text", "")
            word_count = count_words(text)
            if word_count < 120 or word_count > 260:
                errors.append(f"Line {line_num}: Word count {word_count} not in [120, 260]")
            
            # Check occupancy_pct
            occ = chunk.get("occupancy_pct")
            if occ is not None:
                try:
                    occ_val = float(occ) if isinstance(occ, str) else occ
                    if not (0 <= occ_val <= 1):
                        errors.append(f"Line {line_num}: occupancy_pct {occ_val} not in [0, 1]")
                except (ValueError, TypeError):
                    errors.append(f"Line {line_num}: Invalid occupancy_pct - {occ}")
            
            # Check source_type
            source_type = chunk.get("source_type", "")
            if source_type not in valid_source_types:
                errors.append(f"Line {line_num}: Invalid source_type - {source_type}")
            
            # Check PII
            company = chunk.get("company", "").lower()
            for blacklisted in COMPANY_BLACKLIST:
                if blacklisted in company:
                    errors.append(f"Line {line_num}: Potential PII - company name contains '{blacklisted}'")
            
            companies_seen.add(chunk.get("company", ""))
    
    print(f"✓ Corpus: {len(chunk_ids)} chunks, {len(companies_seen)} companies")
    return len(errors) == 0, errors


def validate_train_pairs(pairs_file: Path, corpus_chunk_ids: Set[str]) -> tuple[bool, List[str]]:
    """Validate train_pairs.jsonl."""
    errors = []
    pair_ids: Set[str] = set()
    
    required_keys = {
        "pair_id", "query", "positive_chunk_id", "positive_doc_id",
        "hard_negatives", "meta"
    }
    
    with open(pairs_file, "r") as f:
        for line_num, line in enumerate(f, 1):
            try:
                pair = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: Invalid JSON - {e}")
                continue
            
            # Check required keys
            missing = required_keys - set(pair.keys())
            if missing:
                errors.append(f"Line {line_num}: Missing keys - {missing}")
            
            # Check pair_id uniqueness
            pair_id = pair.get("pair_id", "")
            if pair_id in pair_ids:
                errors.append(f"Line {line_num}: Duplicate pair_id - {pair_id}")
            pair_ids.add(pair_id)
            
            # Check positive exists
            pos_chunk_id = pair.get("positive_chunk_id", "")
            if pos_chunk_id not in corpus_chunk_ids:
                errors.append(f"Line {line_num}: positive_chunk_id not in corpus - {pos_chunk_id}")
            
            # Check hard negatives
            hard_negatives = pair.get("hard_negatives", [])
            if len(hard_negatives) < 2:
                errors.append(f"Line {line_num}: Need at least 2 hard negatives")
            
            for neg in hard_negatives:
                neg_chunk_id = neg.get("chunk_id", "")
                if neg_chunk_id not in corpus_chunk_ids:
                    errors.append(f"Line {line_num}: hard_negative chunk_id not in corpus - {neg_chunk_id}")
                if neg_chunk_id == pos_chunk_id:
                    errors.append(f"Line {line_num}: hard_negative equals positive - {neg_chunk_id}")
    
    print(f"✓ Train pairs: {len(pair_ids)} pairs")
    return len(errors) == 0, errors


def validate_eval(queries_file: Path, qrels_file: Path, corpus_chunk_ids: Set[str]) -> tuple[bool, List[str]]:
    """Validate eval queries and qrels."""
    errors = []
    queries: Dict[str, Dict] = {}
    qrels_by_qid: Dict[str, List[Dict]] = defaultdict(list)
    
    # Load queries
    required_query_keys = {
        "qid", "text", "company", "sector", "source_type", "deal_stage",
        "date", "asset_type", "deal_type", "market", "region"
    }
    
    with open(queries_file, "r") as f:
        for line_num, line in enumerate(f, 1):
            try:
                query = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Queries line {line_num}: Invalid JSON - {e}")
                continue
            
            qid = query.get("qid", "")
            queries[qid] = query
            
            missing = required_query_keys - set(query.keys())
            if missing:
                errors.append(f"Queries line {line_num}: Missing keys - {missing}")
    
    # Load qrels
    with open(qrels_file, "r") as f:
        for line_num, line in enumerate(f, 1):
            try:
                qrel = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Qrels line {line_num}: Invalid JSON - {e}")
                continue
            
            qid = qrel.get("qid", "")
            doc_id = qrel.get("doc_id", "")
            relevance = qrel.get("relevance")
            
            if doc_id not in corpus_chunk_ids:
                errors.append(f"Qrels line {line_num}: doc_id not in corpus - {doc_id}")
            
            if relevance not in [1, 2]:
                errors.append(f"Qrels line {line_num}: relevance must be 1 or 2, got {relevance}")
            
            qrels_by_qid[qid].append(qrel)
    
    # Check each query has 2-4 relevant chunks with at least one relevance=2
    for qid, query in queries.items():
        qrels = qrels_by_qid[qid]
        if len(qrels) < 2 or len(qrels) > 4:
            errors.append(f"Query {qid}: Must have 2-4 qrels, got {len(qrels)}")
        
        relevances = [q["relevance"] for q in qrels]
        if 2 not in relevances:
            errors.append(f"Query {qid}: Must have at least one relevance=2")
    
    print(f"✓ Eval: {len(queries)} queries, {sum(len(v) for v in qrels_by_qid.values())} qrels")
    return len(errors) == 0, errors


def main():
    parser = argparse.ArgumentParser(description="Validate dataset")
    parser.add_argument("--data_dir", default="data", help="Data directory")
    
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    
    all_errors = []
    
    print("="*70)
    print("DATASET VALIDATION")
    print("="*70)
    
    # Load corpus chunk IDs
    corpus_file = data_dir / "corpus.jsonl"
    corpus_chunk_ids = set()
    
    if corpus_file.exists():
        print(f"\n📂 Loading corpus from {corpus_file.name}...")
        with open(corpus_file, "r") as f:
            for line in f:
                chunk = json.loads(line)
                corpus_chunk_ids.add(chunk["chunk_id"])
        print(f"✓ Loaded {len(corpus_chunk_ids):,} chunk IDs")
    else:
        print(f"\n✗ Corpus file not found: {corpus_file}")
        return
    
    # Validate corpus
    print("\n" + "="*70)
    print("VALIDATING CORPUS")
    print("="*70)
    valid, errors = validate_corpus(corpus_file)
    if not valid:
        all_errors.extend(errors)
        print(f"✗ Found {len(errors)} errors")
    else:
        print("✓ All checks passed")
    
    # Validate train pairs
    pairs_file = data_dir / "train_pairs.jsonl"
    if pairs_file.exists():
        print("\n" + "="*70)
        print("VALIDATING TRAIN PAIRS")
        print("="*70)
        valid, errors = validate_train_pairs(pairs_file, corpus_chunk_ids)
        if not valid:
            all_errors.extend(errors)
            print(f"✗ Found {len(errors)} errors")
        else:
            print("✓ All checks passed")
    else:
        print(f"\n⚠ Train pairs file not found: {pairs_file}")
    
    # Validate eval
    queries_file = data_dir / "queries.jsonl"
    qrels_file = data_dir / "qrels.jsonl"
    if queries_file.exists() and qrels_file.exists():
        print("\n" + "="*70)
        print("VALIDATING EVAL SET")
        print("="*70)
        valid, errors = validate_eval(queries_file, qrels_file, corpus_chunk_ids)
        if not valid:
            all_errors.extend(errors)
            print(f"✗ Found {len(errors)} errors")
        else:
            print("✓ All checks passed")
    else:
        print(f"\n⚠ Eval files not found: {queries_file}, {qrels_file}")
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    if all_errors:
        print(f"✗ VALIDATION FAILED: {len(all_errors)} total errors")
        print(f"\nError breakdown:")
        error_types = defaultdict(int)
        for err in all_errors:
            error_type = err.split(":")[0] if ":" in err else "Other"
            error_types[error_type] += 1
        for err_type, count in sorted(error_types.items(), key=lambda x: -x[1]):
            print(f"  {err_type}: {count}")
        print(f"\nFirst 20 errors:")
        for i, err in enumerate(all_errors[:20], 1):
            print(f"  {i:2d}. {err}")
        if len(all_errors) > 20:
            print(f"\n  ... and {len(all_errors) - 20} more errors")
        print("="*70)
        return 1
    else:
        print("✓ VALIDATION PASSED - All checks successful!")
        print("="*70)
        return 0


if __name__ == "__main__":
    exit(main())
