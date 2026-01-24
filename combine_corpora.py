#!/usr/bin/env python3
"""Combine multiple corpus files into one."""

import json
from pathlib import Path

def combine_corpora():
    """Combine earlier_corpus.jsonl and corpus.jsonl into a single file."""

    data_dir = Path("data")
    earlier_file = data_dir / "earlier_corpus.jsonl"
    new_file = data_dir / "corpus.jsonl"
    combined_file = data_dir / "corpus_combined.jsonl"

    print("Combining corpora...")
    print(f"Earlier corpus: {earlier_file}")
    print(f"New corpus: {new_file}")
    print(f"Output: {combined_file}")

    # Read and combine
    with open(combined_file, 'w') as outfile:
        # First add the earlier corpus
        if earlier_file.exists():
            print("Adding earlier corpus...")
            with open(earlier_file, 'r') as infile:
                for line in infile:
                    if line.strip():  # Skip empty lines
                        outfile.write(line)

        # Then add the new corpus
        if new_file.exists():
            print("Adding new corpus...")
            with open(new_file, 'r') as infile:
                for line in infile:
                    if line.strip():  # Skip empty lines
                        outfile.write(line)

    # Count chunks
    total_chunks = 0
    with open(combined_file, 'r') as f:
        for line in f:
            if line.strip():
                total_chunks += 1

    print(f"✓ Combined corpus created with {total_chunks} chunks")
    print(f"✓ Saved to {combined_file}")

    # Optionally replace the main corpus.jsonl
    response = input("\nReplace main corpus.jsonl with combined version? (y/n): ")
    if response.lower() in ['y', 'yes']:
        combined_file.replace(data_dir / "corpus.jsonl")
        print("✓ Replaced main corpus.jsonl")

    return combined_file

if __name__ == "__main__":
    combine_corpora()
