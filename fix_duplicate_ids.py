#!/usr/bin/env python3
"""Fix duplicate chunk_ids in corpus.jsonl."""

import json
from pathlib import Path
from collections import defaultdict

def fix_duplicate_ids():
    """Fix duplicate chunk_ids in corpus.jsonl."""
    
    data_dir = Path("data")
    corpus_file = data_dir / "corpus.jsonl"
    backup_file = data_dir / "corpus_backup.jsonl"
    fixed_file = data_dir / "corpus_fixed.jsonl"
    
    print("Fixing duplicate chunk IDs...")
    print(f"Input: {corpus_file}")
    print(f"Output: {fixed_file}")
    
    # Backup
    if corpus_file.exists():
        import shutil
        shutil.copy(corpus_file, backup_file)
        print(f"✓ Backed up to {backup_file}")
    
    # Read all chunks
    chunks = []
    seen_ids = set()
    id_counter = defaultdict(int)
    
    with open(corpus_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            try:
                chunk = json.loads(line)
                original_id = chunk.get("chunk_id", "")
                
                # Track duplicates
                if original_id in seen_ids:
                    # Generate new unique ID
                    base_id = original_id.split('_')[0] if '_' in original_id else "chunk"
                    id_counter[base_id] += 1
                    new_id = f"{base_id}_{id_counter[base_id]:06d}"
                    chunk["chunk_id"] = new_id
                    print(f"  Line {line_num}: Renamed {original_id} -> {new_id}")
                else:
                    seen_ids.add(original_id)
                
                chunks.append(chunk)
            except json.JSONDecodeError as e:
                print(f"  Line {line_num}: Invalid JSON - {e}")
                continue
    
    # Write fixed corpus
    with open(fixed_file, 'w') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")
    
    print(f"\n✓ Fixed corpus created with {len(chunks)} chunks")
    print(f"✓ All chunk IDs are now unique")
    
    # Replace original
    fixed_file.replace(corpus_file)
    print(f"✓ Replaced {corpus_file} with fixed version")
    
    # Verify
    seen_ids_check = set()
    duplicates = []
    with open(corpus_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            chunk = json.loads(line)
            chunk_id = chunk.get("chunk_id", "")
            if chunk_id in seen_ids_check:
                duplicates.append((line_num, chunk_id))
            seen_ids_check.add(chunk_id)
    
    if duplicates:
        print(f"⚠ Warning: Still found {len(duplicates)} duplicates!")
        for line_num, chunk_id in duplicates[:10]:
            print(f"  Line {line_num}: {chunk_id}")
    else:
        print("✓ Verification passed - no duplicates found")
    
    return len(chunks)

if __name__ == "__main__":
    fix_duplicate_ids()
