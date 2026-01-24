# Progress Indicators & Status Display

All scripts now include comprehensive progress tracking with:
- **Real-time progress bars** (using tqdm)
- **Percentage complete** 
- **Countdown/remaining items**
- **Completion summaries**

## generate_dataset.py

### Progress Display:
```
======================================================================
RE-DD-EMBEDDINGS DATASET GENERATION
======================================================================
Target: 3,000 chunks across 60 deals
Model: gemini-3-flash-preview:cloud
======================================================================

Deals: 45/60 |████████████████░░░░░░| 75% [02:15<00:45] | Chunks: Generated: 2,340/3,000, Remaining: 660, Progress: 78.0%
```

### Completion Summary:
```
======================================================================
GENERATION COMPLETE
======================================================================
✓ Deals completed:     60/60
✓ Chunks generated:   3,045/3,000 (101.5%)
✓ Avg chunks/deal:    50.8
✓ Corpus file:         data/corpus.jsonl
✓ Index file:          data/index.json
======================================================================
```

## build_splits.py

### Train Pairs Progress:
```
======================================================================
BUILDING TRAIN/EVAL SPLITS
======================================================================
Target train pairs: 15,000
Target eval queries: 500
======================================================================

Train Pairs: 120/150 batches |████████████░░░░| 80% [05:30<01:22] | Pairs: Generated: 12,450/15,000, Remaining: 2,550, Progress: 83.0%
```

### Eval Set Progress:
```
======================================================================
GENERATING EVAL SET
======================================================================
Sampling eval chunks...
✓ Sampled 500 chunks for eval

Eval Queries: 450/500 queries |███████████████░░| 90% [03:20<00:22] | Qrels: 1,850, Avg/Query: 4.1
```

### Final Summary:
```
======================================================================
SPLIT GENERATION COMPLETE
======================================================================
✓ Train pairs:         15,000/15,000 (100.0%)
✓ Eval queries:        500/500 (100.0%)
✓ Eval qrels:          1,950 (avg 3.9 per query)
✓ Train pairs file:    data/train_pairs.jsonl
✓ Queries file:         data/queries.jsonl
✓ Qrels file:           data/qrels.jsonl
======================================================================
```

## validate_dataset.py

### Progress Display:
```
======================================================================
DATASET VALIDATION
======================================================================

📂 Loading corpus from corpus.jsonl...
✓ Loaded 3,045 chunk IDs

======================================================================
VALIDATING CORPUS
======================================================================
✓ Corpus: 3,045 chunks, 60 companies
✓ All checks passed

======================================================================
VALIDATING TRAIN PAIRS
======================================================================
✓ Train pairs: 15,000 pairs
✓ All checks passed

======================================================================
VALIDATING EVAL SET
======================================================================
✓ Eval: 500 queries, 1,950 qrels
✓ All checks passed

======================================================================
VALIDATION SUMMARY
======================================================================
✓ VALIDATION PASSED - All checks successful!
======================================================================
```

## sample_space_corpus.py

### Progress Display:
```
======================================================================
SPACE CORPUS SAMPLING
======================================================================
Target: 300 chunks
Source: data/corpus.jsonl
======================================================================

Loading corpus...
✓ Sampled 300 chunks

Writing to space_corpus.jsonl...
Writing chunks: 300/300 |████████████████████| 100% [00:02<00:00, 145.23chunk/s]

Writing metadata to space_meta.json...

======================================================================
SAMPLING COMPLETE
======================================================================
✓ Total chunks:        300/300 (100.0%)
✓ Companies:           45
✓ Source types:        15
✓ Asset types:         9
✓ Regions:             5
✓ Deal stages:         5
✓ Output file:         data/space_corpus.jsonl
✓ Metadata file:       data/space_meta.json
======================================================================
```

## Features

### Real-time Updates
- Progress bars update in real-time as chunks/pairs are generated
- Shows elapsed time and estimated time remaining
- Displays current count vs target

### Percentage Complete
- All progress bars show percentage completion
- Summary sections show final percentages

### Countdown/Remaining
- Shows remaining items to generate
- Updates dynamically as work progresses

### Error Tracking
- Failed deals/chunks are tracked and reported
- Skipped items (no hard negatives, etc.) are counted
- Validation errors are categorized and summarized

### Completion Summaries
- All scripts end with a formatted summary
- Shows what was generated vs targets
- Lists output files created
- Displays key statistics

## Example Output Flow

```
$ python scripts/generate_dataset.py --companies 60 --target_chunks 3000

======================================================================
RE-DD-EMBEDDINGS DATASET GENERATION
======================================================================
Target: 3,000 chunks across 60 deals
Model: gemini-3-flash-preview:cloud
======================================================================

Deals: 0/60 |░░░░░░░░░░░░░░░░░░░░| 0% [00:00<?] | Chunks: Generated: 0/3,000, Remaining: 3,000, Progress: 0.0%
Deals: 15/60 |████░░░░░░░░░░░░░░░░| 25% [12:30<37:30] | Chunks: Generated: 750/3,000, Remaining: 2,250, Progress: 25.0%
Deals: 30/60 |████████░░░░░░░░░░░░| 50% [25:00<25:00] | Chunks: Generated: 1,500/3,000, Remaining: 1,500, Progress: 50.0%
Deals: 45/60 |████████████░░░░░░░░| 75% [37:30<12:30] | Chunks: Generated: 2,250/3,000, Remaining: 750, Progress: 75.0%
Deals: 60/60 |████████████████████| 100% [50:00<00:00] | Chunks: Generated: 3,045/3,000, Remaining: 0, Progress: 101.5%, Status: TARGET REACHED

======================================================================
GENERATION COMPLETE
======================================================================
✓ Deals completed:     60/60
✓ Chunks generated:   3,045/3,000 (101.5%)
✓ Avg chunks/deal:    50.8
✓ Corpus file:         data/corpus.jsonl
✓ Index file:          data/index.json
======================================================================
```
