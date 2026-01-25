#!/usr/bin/env python3
"""
Validate that all corpus chunks have ALL required metadata fields.
"""

import json
from pathlib import Path
from collections import Counter
from tqdm import tqdm

REQUIRED_FIELDS = [
    "doc_id", "chunk_id", "text", "title", "source_type", "company",
    "sector", "deal_stage", "date", "region", "doc_url", "tags",
    "topic", "confidentiality", "asset_type", "deal_type", "market",
    "vintage", "unit_count", "sqft", "occupancy_pct", "noi", "cap_rate",
    "ltv", "dscr"
]


def validate_corpus(corpus_file):
    """Validate corpus completeness."""
    print("="*70)
    print("VALIDATING CORPUS COMPLETENESS")
    print("="*70)
    print(f"File: {corpus_file}")
    print()
    
    missing_fields = Counter()
    empty_fields = Counter()
    invalid_chunks = []
    total_chunks = 0
    
    with open(corpus_file, 'r') as f:
        for line_num, line in enumerate(tqdm(f, desc="Validating"), 1):
            if not line.strip():
                continue
            
            try:
                chunk = json.loads(line)
                total_chunks += 1
                
                # Check for missing fields
                for field in REQUIRED_FIELDS:
                    if field not in chunk:
                        missing_fields[field] += 1
                        invalid_chunks.append((line_num, chunk.get('chunk_id', 'unknown'), f"Missing: {field}"))
                    elif field in ['text', 'chunk_id', 'company'] and not chunk[field]:
                        empty_fields[field] += 1
                        invalid_chunks.append((line_num, chunk.get('chunk_id', 'unknown'), f"Empty: {field}"))
                
                # Validate text length
                text = chunk.get('text', '')
                if len(text) < 50:
                    invalid_chunks.append((line_num, chunk.get('chunk_id', 'unknown'), f"Text too short: {len(text)} chars"))
                
                # Validate topic exists
                if 'topic' not in chunk or not chunk['topic']:
                    invalid_chunks.append((line_num, chunk.get('chunk_id', 'unknown'), "Missing or empty topic"))
                
                # Validate tags
                tags = chunk.get('tags', [])
                if not tags or not isinstance(tags, list):
                    invalid_chunks.append((line_num, chunk.get('chunk_id', 'unknown'), "Missing or invalid tags"))
                
            except json.JSONDecodeError as e:
                invalid_chunks.append((line_num, 'unknown', f"JSON parse error: {e}"))
    
    # Report
    print()
    print("="*70)
    print("VALIDATION RESULTS")
    print("="*70)
    print(f"Total chunks: {total_chunks}")
    print()
    
    if missing_fields:
        print("⚠️  MISSING FIELDS:")
        for field, count in missing_fields.most_common():
            print(f"   {field}: {count} chunks ({count/total_chunks*100:.1f}%)")
        print()
    
    if empty_fields:
        print("⚠️  EMPTY FIELDS:")
        for field, count in empty_fields.most_common():
            print(f"   {field}: {count} chunks ({count/total_chunks*100:.1f}%)")
        print()
    
    if invalid_chunks:
        print(f"⚠️  INVALID CHUNKS: {len(invalid_chunks)}")
        print("\nFirst 10 issues:")
        for line_num, chunk_id, issue in invalid_chunks[:10]:
            print(f"   Line {line_num} ({chunk_id}): {issue}")
        if len(invalid_chunks) > 10:
            print(f"   ... and {len(invalid_chunks) - 10} more")
        print()
    
    # Summary
    issues = len(missing_fields) + len(empty_fields) + len(invalid_chunks)
    if issues == 0:
        print("✅ ALL CHUNKS ARE COMPLETE!")
        print("   All required fields present")
        print("   All fields populated")
        print("   Topic field present for hard negative mining")
    else:
        print(f"❌ FOUND {issues} ISSUES")
        print("   Run fix_corpus_metadata.py to fix missing fields")
    
    print("="*70)
    
    return issues == 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate corpus completeness")
    parser.add_argument('--corpus', default='data/corpus.jsonl', help='Corpus file')
    args = parser.parse_args()
    
    validate_corpus(args.corpus)
