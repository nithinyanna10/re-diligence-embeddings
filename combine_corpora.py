#!/usr/bin/env python3
"""Combine multiple corpus files into one with unique chunk IDs."""

import json
from pathlib import Path

def combine_corpora():
    """Combine earlier_corpus.jsonl and corpus.jsonl into a single file with unique IDs."""

    data_dir = Path("data")
    earlier_file = data_dir / "earlier_corpus.jsonl"
    new_file = data_dir / "corpus.jsonl"
    combined_file = data_dir / "corpus_combined.jsonl"

    print("Combining corpora with unique chunk IDs...")
    print(f"Earlier corpus: {earlier_file}")
    print(f"New corpus: {new_file}")
    print(f"Output: {combined_file}")

    seen_chunk_ids = set()
    chunk_counter = 1
    total_chunks = 0

    # Read and combine with unique IDs
    with open(combined_file, 'w') as outfile:
        # First add the earlier corpus
        if earlier_file.exists():
            print("Adding earlier corpus...")
            with open(earlier_file, 'r') as infile:
                for line in infile:
                    if not line.strip():
                        continue
                    
                    chunk = json.loads(line)
                    original_id = chunk.get("chunk_id", "")
                    
                    # Make unique if duplicate
                    if original_id in seen_chunk_ids:
                        chunk["chunk_id"] = f"chunk_{chunk_counter:06d}"
                        chunk_counter += 1
                    else:
                        seen_chunk_ids.add(original_id)
                    
                    outfile.write(json.dumps(chunk) + "\n")
                    total_chunks += 1

        # Then add the new corpus with offset IDs
        if new_file.exists():
            print("Adding new corpus with unique IDs...")
            with open(new_file, 'r') as infile:
                for line in infile:
                    if not line.strip():
                        continue
                    
                    chunk = json.loads(line)
                    original_id = chunk.get("chunk_id", "")
                    
                    # Always make unique for new corpus
                    if original_id in seen_chunk_ids:
                        chunk["chunk_id"] = f"chunk_{chunk_counter:06d}"
                    else:
                        # Check if it conflicts, if so rename
                        if original_id in seen_chunk_ids:
                            chunk["chunk_id"] = f"chunk_{chunk_counter:06d}"
                        else:
                            seen_chunk_ids.add(original_id)
                    
                    chunk_counter += 1
                    outfile.write(json.dumps(chunk) + "\n")
                    total_chunks += 1

    print(f"✓ Combined corpus created with {total_chunks} chunks")
    print(f"✓ All chunk IDs are unique")
    print(f"✓ Saved to {combined_file}")

    return combined_file

if __name__ == "__main__":
    combined = combine_corpora()
    
    # Auto-replace main corpus
    data_dir = Path("data")
    backup_file = data_dir / "corpus_backup.jsonl"
    
    # Backup existing
    if (data_dir / "corpus.jsonl").exists():
        import shutil
        shutil.copy(data_dir / "corpus.jsonl", backup_file)
        print(f"✓ Backed up existing corpus to {backup_file}")
    
    # Replace
    combined.replace(data_dir / "corpus.jsonl")
    print("✓ Replaced main corpus.jsonl with deduplicated version")
