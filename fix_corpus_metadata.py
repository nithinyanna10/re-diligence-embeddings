#!/usr/bin/env python3
"""
Fix missing metadata fields in corpus.jsonl - add topic, ensure all fields present.
"""

import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent / 'scripts'))
from generate_dataset import extract_primary_topic


def fix_corpus_metadata(corpus_file, output_file=None):
    """Fix missing metadata in corpus."""
    if output_file is None:
        output_file = corpus_file.replace('.jsonl', '_fixed.jsonl')
    
    print("="*70)
    print("FIXING CORPUS METADATA")
    print("="*70)
    print(f"Input:  {corpus_file}")
    print(f"Output: {output_file}")
    print()
    
    fixed_count = 0
    total_count = 0
    
    with open(corpus_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in tqdm(f_in, desc="Fixing chunks"):
            if not line.strip():
                continue
            
            try:
                chunk = json.loads(line)
                total_count += 1
                was_fixed = False
                
                # Ensure tags exist
                tags = chunk.get('tags', [])
                if not tags or not isinstance(tags, list):
                    # Generate default tags
                    source_type = chunk.get('source_type', 'CIM')
                    if 'CIM' in source_type:
                        tags = ['NOI', 'GPR']
                    elif 'Lease' in source_type:
                        tags = ['lease', 'WALE']
                    elif 'PhaseI' in source_type or 'ESA' in source_type:
                        tags = ['PhaseI', 'environmental']
                    elif 'ALTA' in source_type or 'Survey' in source_type:
                        tags = ['ALTA', 'survey']
                    elif 'Title' in source_type:
                        tags = ['title', 'easements']
                    elif 'T12' in source_type or 'Operating' in source_type:
                        tags = ['T12', 'NOI']
                    elif 'Debt' in source_type:
                        tags = ['debt_terms', 'DSCR']
                    else:
                        tags = ['general']
                    chunk['tags'] = tags
                    was_fixed = True
                
                # Add topic if missing
                if 'topic' not in chunk or not chunk.get('topic'):
                    chunk['topic'] = extract_primary_topic(tags)
                    was_fixed = True
                
                # Ensure all required fields with defaults
                defaults = {
                    'sector': 'Real Estate',
                    'confidentiality': 'public',
                    'doc_url': chunk.get('doc_url', ''),
                    'vintage': str(chunk.get('vintage', '2000')),
                    'unit_count': chunk.get('unit_count', 0),
                    'sqft': chunk.get('sqft', 0),
                    'occupancy_pct': chunk.get('occupancy_pct', 0.90),
                    'noi': chunk.get('noi', '$0M'),
                    'cap_rate': chunk.get('cap_rate', '5.0%'),
                    'ltv': chunk.get('ltv', '65.0%'),
                    'dscr': chunk.get('dscr', '1.40x')
                }
                
                for key, default_val in defaults.items():
                    if key not in chunk or chunk[key] is None:
                        chunk[key] = default_val
                        was_fixed = True
                
                if was_fixed:
                    fixed_count += 1
                
                f_out.write(json.dumps(chunk) + '\n')
                
            except Exception as e:
                print(f"\n⚠️  Error processing line: {e}")
                continue
    
    print()
    print("="*70)
    print("FIXING COMPLETE")
    print("="*70)
    print(f"Total chunks: {total_count}")
    print(f"Fixed chunks: {fixed_count} ({fixed_count/total_count*100:.1f}%)")
    print(f"Output file: {output_file}")
    print()
    print("Next: Replace original file:")
    print(f"  mv {output_file} {corpus_file}")
    print("="*70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fix corpus metadata")
    parser.add_argument('--corpus', default='data/corpus.jsonl', help='Corpus file')
    parser.add_argument('--output', help='Output file (default: corpus_fixed.jsonl)')
    args = parser.parse_args()
    
    fix_corpus_metadata(args.corpus, args.output)
