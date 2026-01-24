#!/usr/bin/env python3
"""Resume build_splits.py from where it stopped."""

import json
from pathlib import Path

def check_progress(data_dir="data"):
    """Check current progress of build_splits.py."""
    data_dir = Path(data_dir)

    # Check if train_pairs.jsonl exists and count lines
    train_pairs_file = data_dir / "train_pairs.jsonl"
    if train_pairs_file.exists():
        with open(train_pairs_file, 'r') as f:
            train_pairs_count = sum(1 for _ in f)
        print(f"✓ Train pairs: {train_pairs_count:,} generated")
    else:
        print("✗ Train pairs: 0 generated")
        return False

    # Check if eval files exist
    queries_file = data_dir / "queries.jsonl"
    qrels_file = data_dir / "qrels.jsonl"

    if queries_file.exists():
        with open(queries_file, 'r') as f:
            eval_queries_count = sum(1 for _ in f)
        print(f"✓ Eval queries: {eval_queries_count:,} generated")
    else:
        print("✗ Eval queries: 0 generated")

    if qrels_file.exists():
        with open(qrels_file, 'r') as f:
            qrels_count = sum(1 for _ in f)
        print(f"✓ Qrels: {qrels_count:,} generated")
    else:
        print("✗ Qrels: 0 generated")

    return True

def resume_splits(target_train_pairs=15000, target_eval_queries=500):
    """Resume build_splits.py with adjusted targets."""
    print("Resuming build_splits.py...")
    print("="*60)

    # Check current progress
    has_progress = check_progress()

    if has_progress:
        print("\nOptions:")
        print("1. Delete partial files and restart")
        print("2. Continue with remaining targets")
        print("3. Adjust targets based on current progress")

        choice = input("\nChoose (1-3): ").strip()

        if choice == "1":
            print("Deleting partial files...")
            import os
            for f in ["train_pairs.jsonl", "queries.jsonl", "qrels.jsonl"]:
                path = Path("data") / f
                if path.exists():
                    os.remove(path)
                    print(f"✓ Deleted {f}")
            print("Restart build_splits.py from beginning")

        elif choice == "2":
            print("Continue with remaining targets...")
            # This would require modifying build_splits.py to resume
            print("Need to implement resume logic in build_splits.py")

        elif choice == "3":
            print("Adjust targets...")
            print("Current targets: 15,000 train pairs, 500 eval queries")
            new_train = input("New train pairs target: ").strip() or "15000"
            new_eval = input("New eval queries target: ").strip() or "500"
            print(f"Use: --target_train_pairs {new_train} --eval_queries {new_eval}")

    else:
        print("No progress found - start fresh")

def main():
    """Main function."""
    print("Build Splits Progress Checker")
    print("="*60)

    resume_splits()

if __name__ == "__main__":
    main()
