# Query Storage Structure

Queries are saved in **3 separate JSONL files** with different structures:

## 1. Training Queries: `data/train_pairs.jsonl`

**Purpose**: Training pairs with queries, positive chunks, and hard negatives

**Structure**: One JSON object per line (JSONL format)

```json
{
  "pair_id": "pair_000001",
  "query": "SNDA status top tenants",
  "positive_chunk_id": "chunk_001",
  "positive_doc_id": "doc_001",
  "hard_negatives": [
    {"chunk_id": "chunk_045", "doc_id": "doc_005"},
    {"chunk_id": "chunk_089", "doc_id": "doc_012"}
  ],
  "meta": {
    "company": "Riverside Garden Properties",
    "sector": "Real Estate",
    "source_type": "Lease_Abstract",
    "deal_stage": "Diligence",
    "date": "2024-01-15",
    "asset_type": "Multifamily",
    "deal_type": "Acquisition",
    "market": "Atlanta, GA",
    "region": "US-Southeast"
  }
}
```

**File**: `data/train_pairs.jsonl`  
**Count**: ~15,000 pairs (one per line)  
**Format**: JSONL (one JSON object per line)

---

## 2. Evaluation Queries: `data/queries.jsonl`

**Purpose**: Standalone eval queries for testing retrieval

**Structure**: One JSON object per line (JSONL format)

```json
{
  "qid": "q_0001",
  "text": "what % of leases roll in next 12 months and mitigation?",
  "company": "Riverside Garden Properties",
  "sector": "Real Estate",
  "source_type": "Rent_Roll_Summary",
  "deal_stage": "Diligence",
  "date": "2024-01-15",
  "asset_type": "Multifamily",
  "deal_type": "Acquisition",
  "market": "Atlanta, GA",
  "region": "US-Southeast"
}
```

**File**: `data/queries.jsonl`  
**Count**: 500 queries (one per line)  
**Format**: JSONL (one JSON object per line)

---

## 3. Relevance Labels: `data/qrels.jsonl`

**Purpose**: Maps eval queries to relevant chunks with relevance scores

**Structure**: One JSON object per line (JSONL format)

```json
{
  "qid": "q_0001",
  "doc_id": "chunk_001",
  "relevance": 2
}
```

**Relevance Scores**:
- `2` = Primary relevant chunk (most relevant)
- `1` = Supporting relevant chunk (related but less direct)

**File**: `data/qrels.jsonl`  
**Count**: ~1,500-2,000 qrels (2-4 per query)  
**Format**: JSONL (one JSON object per line)

---

## File Summary

| File | Purpose | Count | Structure |
|------|---------|-------|-----------|
| `train_pairs.jsonl` | Training data | ~15,000 | Query + positive + hard negatives |
| `queries.jsonl` | Eval queries | 500 | Query text + metadata |
| `qrels.jsonl` | Eval labels | ~1,500-2,000 | Query ID → chunk ID + relevance |

---

## How to Read the Files

### Python Example

```python
import json

# Read training pairs
with open('data/train_pairs.jsonl', 'r') as f:
    for line in f:
        pair = json.loads(line)
        print(f"Query: {pair['query']}")
        print(f"Positive: {pair['positive_chunk_id']}")
        print(f"Negatives: {pair['hard_negatives']}")

# Read eval queries
with open('data/queries.jsonl', 'r') as f:
    for line in f:
        query = json.loads(line)
        print(f"QID: {query['qid']}, Text: {query['text']}")

# Read qrels
with open('data/qrels.jsonl', 'r') as f:
    for line in f:
        qrel = json.loads(line)
        print(f"Query {qrel['qid']} → Chunk {qrel['doc_id']} (relevance={qrel['relevance']})")
```

### Command Line Examples

```bash
# Count training pairs
wc -l data/train_pairs.jsonl

# View first training pair
head -n 1 data/train_pairs.jsonl | python3 -m json.tool

# Count eval queries
wc -l data/queries.jsonl

# View first eval query
head -n 1 data/queries.jsonl | python3 -m json.tool

# Count qrels
wc -l data/qrels.jsonl

# View qrels for a specific query
grep "q_0001" data/qrels.jsonl | python3 -m json.tool
```

---

## Query Examples

### Training Query Examples (from train_pairs.jsonl)

```json
{"query": "SNDA status top tenants", ...}
{"query": "Phase I RECs recommended actions", ...}
{"query": "what % of leases roll in next 12 months and mitigation?", ...}
{"query": "why did NOI drop in Q2? utility expense spike?", ...}
{"query": "top 5 tenants % of GPR", ...}
{"query": "DSCR covenant trigger level", ...}
{"query": "CAM cap language", ...}
{"query": "go-dark clause", ...}
{"query": "estoppels received?", ...}
{"query": "ALTA encroachments?", ...}
```

### Eval Query Examples (from queries.jsonl)

Same style as training queries, but standalone (no positive/negative chunks attached).

---

## Notes

1. **JSONL Format**: Each file is JSONL (JSON Lines) - one JSON object per line, not a single JSON array
2. **No Duplicates**: Each query in `train_pairs.jsonl` is unique (different `pair_id`)
3. **Eval Queries**: Each query in `queries.jsonl` has a unique `qid`
4. **Qrels Mapping**: Multiple qrels can reference the same query (`qid`) but different chunks (`doc_id`)
5. **Chunk IDs**: All `chunk_id` and `doc_id` references must exist in `corpus.jsonl`
