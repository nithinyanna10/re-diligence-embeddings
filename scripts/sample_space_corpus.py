#!/usr/bin/env python3
"""
Sample balanced corpus for HuggingFace Space demo.
Creates space_corpus.jsonl with ~300 chunks balanced across categories.
"""

import json
import random
import argparse
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
from tqdm import tqdm


def sample_space_corpus(corpus_file: Path, k: int = 300) -> tuple[List[Dict], Dict]:
    """Sample k chunks balanced across source_type, asset_type, region, deal_stage."""
    # Load all chunks
    all_chunks = []
    with open(corpus_file, "r") as f:
        for line in f:
            all_chunks.append(json.loads(line))
    
    # Group by categories
    by_source = defaultdict(list)
    by_asset = defaultdict(list)
    by_region = defaultdict(list)
    by_stage = defaultdict(list)
    
    for chunk in all_chunks:
        by_source[chunk.get("source_type", "unknown")].append(chunk)
        by_asset[chunk.get("asset_type", "unknown")].append(chunk)
        by_region[chunk.get("region", "unknown")].append(chunk)
        by_stage[chunk.get("deal_stage", "unknown")].append(chunk)
    
    # Calculate target per category (roughly balanced)
    n_sources = len(by_source)
    n_assets = len(by_asset)
    n_regions = len(by_region)
    n_stages = len(by_stage)
    
    target_per_source = max(1, k // n_sources)
    target_per_asset = max(1, k // n_assets)
    target_per_region = max(1, k // n_regions)
    target_per_stage = max(1, k // n_stages)
    
    # Sample with diversity constraints
    sampled = []
    sampled_ids = set()
    
    # Track counts per category
    counts_source = defaultdict(int)
    counts_asset = defaultdict(int)
    counts_region = defaultdict(int)
    counts_stage = defaultdict(int)
    
    # Shuffle for randomness
    random.shuffle(all_chunks)
    
    for chunk in all_chunks:
        if len(sampled) >= k:
            break
        
        chunk_id = chunk["chunk_id"]
        if chunk_id in sampled_ids:
            continue
        
        source = chunk.get("source_type", "unknown")
        asset = chunk.get("asset_type", "unknown")
        region = chunk.get("region", "unknown")
        stage = chunk.get("deal_stage", "unknown")
        
        # Check if we need more from these categories
        need_source = counts_source[source] < target_per_source
        need_asset = counts_asset[asset] < target_per_asset
        need_region = counts_region[region] < target_per_region
        need_stage = counts_stage[stage] < target_per_stage
        
        # Prefer chunks that help balance multiple categories
        if need_source or need_asset or need_region or need_stage:
            sampled.append(chunk)
            sampled_ids.add(chunk_id)
            counts_source[source] += 1
            counts_asset[asset] += 1
            counts_region[region] += 1
            counts_stage[stage] += 1
    
    # Fill remaining slots randomly if needed
    remaining = [c for c in all_chunks if c["chunk_id"] not in sampled_ids]
    random.shuffle(remaining)
    
    while len(sampled) < k and remaining:
        chunk = remaining.pop()
        sampled.append(chunk)
        sampled_ids.add(chunk["chunk_id"])
        counts_source[chunk.get("source_type", "unknown")] += 1
        counts_asset[chunk.get("asset_type", "unknown")] += 1
        counts_region[chunk.get("region", "unknown")] += 1
        counts_stage[chunk.get("deal_stage", "unknown")] += 1
    
    # Build metadata
    meta = {
        "total_chunks": len(sampled),
        "by_source_type": dict(counts_source),
        "by_asset_type": dict(counts_asset),
        "by_region": dict(counts_region),
        "by_deal_stage": dict(counts_stage),
        "companies": len(set(c["company"] for c in sampled))
    }
    
    return sampled, meta


def main():
    parser = argparse.ArgumentParser(description="Sample Space corpus")
    parser.add_argument("--data_dir", default="data", help="Data directory")
    parser.add_argument("--k", type=int, default=300, help="Number of chunks to sample")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    
    args = parser.parse_args()
    random.seed(args.seed)
    
    data_dir = Path(args.data_dir)
    corpus_file = data_dir / "corpus.jsonl"
    output_file = data_dir / "space_corpus.jsonl"
    meta_file = data_dir / "space_meta.json"
    
    if not corpus_file.exists():
        print(f"✗ Error: Corpus file not found: {corpus_file}")
        return 1
    
    print("="*70)
    print("SPACE CORPUS SAMPLING")
    print("="*70)
    print(f"Target: {args.k} chunks")
    print(f"Source: {corpus_file}")
    print("="*70)
    print()
    
    print("Loading corpus...")
    sampled, meta = sample_space_corpus(corpus_file, args.k)
    
    print(f"✓ Sampled {len(sampled)} chunks")
    print()
    
    # Write sampled corpus
    print(f"Writing to {output_file.name}...")
    with open(output_file, "w") as f:
        for chunk in tqdm(sampled, desc="Writing chunks", unit="chunk"):
            f.write(json.dumps(chunk) + "\n")
    
    # Write metadata
    print(f"Writing metadata to {meta_file.name}...")
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)
    
    print()
    print("="*70)
    print("SAMPLING COMPLETE")
    print("="*70)
    print(f"✓ Total chunks:        {len(sampled):,}/{args.k:,} ({len(sampled)/args.k*100:.1f}%)")
    print(f"✓ Companies:           {meta['companies']}")
    print(f"✓ Source types:        {len(meta['by_source_type'])}")
    print(f"✓ Asset types:         {len(meta['by_asset_type'])}")
    print(f"✓ Regions:              {len(meta['by_region'])}")
    print(f"✓ Deal stages:          {len(meta['by_deal_stage'])}")
    print(f"✓ Output file:          {output_file}")
    print(f"✓ Metadata file:       {meta_file}")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    exit(main())
