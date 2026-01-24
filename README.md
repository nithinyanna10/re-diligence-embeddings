# Real Estate Diligence Semantic Search Embeddings

A complete synthetic dataset generator for training specialized embedding models for institutional real estate diligence document search.

## Overview

This repository generates a realistic, public-safe synthetic dataset that mimics institutional real estate diligence packs, including CIMs, IC memos, lease abstracts, rent rolls, T-12 statements, PCA reports, Phase I ESAs, title commitments, ALTA surveys, debt term sheets, insurance summaries, zoning reports, permits, and appraisals.

The dataset is designed for training embedding models that can semantically search through complex real estate diligence documents using analyst-grade queries.

## Requirements

- **Python 3.10+**
- **Ollama** installed and running locally
- **Model**: `gemini-3-flash-preview:cloud` (or another Ollama model)

### Setup

1. Install Ollama: https://ollama.ai

2. Pull the model:
```bash
ollama pull gemini-3-flash-preview:cloud
```

3. Install Python dependencies:
```bash
cd re-diligence-embeddings
source venv/bin/activate  # or: python3 -m venv venv && source venv/bin/activate
pip install tqdm numpy pandas
```

## Quickstart

### 1. Generate Corpus

Generate the full corpus (~3,000 chunks across 60 deals):

```bash
python scripts/generate_dataset.py \
    --model gemini-3-flash-preview:cloud \
    --out_dir data \
    --companies 60 \
    --seed 7 \
    --target_chunks 3000
```

This creates `data/corpus.jsonl` with all document chunks.

### 2. Build Train/Eval Splits

Create training pairs and evaluation set:

```bash
python scripts/build_splits.py \
    --data_dir data \
    --target_train_pairs 15000 \
    --eval_queries 500 \
    --model gemini-3-flash-preview:cloud \
    --seed 42
```

This creates:
- `data/train_pairs.jsonl` (~15,000 query-positive-negative triplets)
- `data/queries.jsonl` (500 eval queries)
- `data/qrels.jsonl` (relevance labels for eval)

### 3. Validate Dataset

Check dataset integrity:

```bash
python scripts/validate_dataset.py --data_dir data
```

### 4. Sample Space Corpus

Create a balanced subset for HuggingFace Space demo:

```bash
python scripts/sample_space_corpus.py \
    --data_dir data \
    --k 300 \
    --seed 123
```

Creates `data/space_corpus.jsonl` and `data/space_meta.json`.

## Dataset Schemas

### corpus.jsonl

One chunk per line:

```json
{
  "doc_id": "doc_001",
  "chunk_id": "chunk_001",
  "text": "120-260 word chunk text...",
  "title": "Confidential Information Memorandum - Riverside Garden Properties",
  "source_type": "CIM",
  "company": "Riverside Garden Properties",
  "sector": "Real Estate",
  "deal_stage": "Diligence",
  "date": "2024-01-15",
  "region": "US-Southeast",
  "asset_type": "Multifamily",
  "deal_type": "Acquisition",
  "market": "Atlanta, GA",
  "vintage": "2010",
  "unit_count": 250,
  "sqft": 275000,
  "occupancy_pct": 0.945,
  "noi": "$4.2M",
  "cap_rate": "5.4%",
  "ltv": "62%",
  "dscr": "1.48x",
  "tags": ["lease", "CAM", "rent_roll"],
  "confidentiality": "public"
}
```

### train_pairs.jsonl

One training pair per line:

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
    "source_type": "Lease_Abstract",
    "asset_type": "Multifamily",
    ...
  }
}
```

### queries.jsonl & qrels.jsonl

Eval queries and relevance labels:

```json
// queries.jsonl
{
  "qid": "q_0001",
  "text": "what % of leases roll in next 12 months and mitigation?",
  "company": "Riverside Garden Properties",
  ...
}

// qrels.jsonl
{
  "qid": "q_0001",
  "doc_id": "chunk_001",
  "relevance": 2
}
```

## Training Embeddings

After generating the dataset, train your embedding model:

```bash
# Placeholder - replace with your training script
python train_embeddings.py \
    --train_file data/train_pairs.jsonl \
    --corpus_file data/corpus.jsonl \
    --output_dir models/re-dd-embeddings \
    --model_name sentence-transformers/all-MiniLM-L6-v2 \
    --epochs 3 \
    --batch_size 32
```

Example training approaches:
- **Contrastive learning**: Use positive/negative pairs with in-batch negatives
- **Triplet loss**: Query-positive-negative triplets
- **Multiple negatives**: Use hard negatives from same asset

## Validation

The validation script checks:
- ✅ JSONL parseability
- ✅ Required schema keys
- ✅ Chunk ID uniqueness
- ✅ Word count (120-260 words)
- ✅ Occupancy percentage range [0, 1]
- ✅ Source type validity
- ✅ Train pair integrity (positives exist, hard negatives exist and differ)
- ✅ Eval set integrity (2-4 qrels per query, at least one relevance=2)
- ✅ PII safety (blacklist check for real company names)

## Space Demo Corpus

The `sample_space_corpus.py` script creates a balanced subset for HuggingFace Space demos:
- ~300 chunks balanced across source types, asset types, regions, and deal stages
- Includes metadata JSON for UI display

## Common Issues & Solutions

### JSON Parse Errors

**Issue**: Ollama returns markdown-wrapped JSON or extra commentary.

**Solution**: The `ollama_client.py` includes automatic JSON extraction and retry logic (up to 2 retries). If failures persist:
1. Check Ollama is running: `ollama list`
2. Try a different model
3. Increase timeout: `--timeout_s 300` in generation scripts

### Generation Too Slow

**Solutions**:
- Reduce `--companies` or `--target_chunks` for testing
- Use a faster Ollama model
- Batch size is already optimized (6-12 chunks per call)

### Out of Memory

**Solutions**:
- Scripts use streaming writes (JSONL), so memory should be minimal
- If issues persist, process in smaller batches by running `generate_dataset.py` multiple times with different `--seed` values

### Hard Negatives Not Found

**Issue**: Some chunks may not have enough hard negatives from the same company.

**Solution**: The script skips pairs without sufficient hard negatives. If many are skipped, increase `--companies` to get more chunks per company.

### Rate Limiting (429 Errors)

**Issue**: Ollama cloud models may have rate limits.

**Solutions**:
- Wait and retry (scripts auto-retry)
- Use local models instead of cloud models
- Reduce batch size or add delays between calls
- Generate in smaller chunks and combine later

## Dataset Statistics (V1 Target)

- **Corpus**: ~3,000 chunks
- **Train pairs**: ~15,000 pairs (≈5 queries per chunk)
- **Eval queries**: 500 queries
- **Eval qrels**: 2-4 relevant chunks per query (1 primary relevance=2, others=1)
- **Companies/Deals**: 60 fictional assets
- **Source types**: 15 document types
- **Asset types**: Multifamily, Industrial, Office, Retail, SelfStorage, Hotel, DataCenter, MixedUse, MHC
- **Regions**: US-Northeast, US-Southeast, US-Midwest, US-Southwest, US-West

## Real Estate Diligence Topics Covered

The dataset includes realistic content across:

1. **Lease & Tenant Diligence**: Rent rolls, lease abstracts, SNDA, estoppels, CAM, WALE, rollover
2. **Underwriting**: T-12 NOI, revenue/expense line items, NOI bridge, capex
3. **Debt/Capital**: Term sheets, DSCR/LTV covenants, debt yield, prepayment, recourse
4. **Physical/Engineering**: PCA reports, deferred maintenance, code compliance
5. **Environmental**: Phase I ESA, RECs, Phase II triggers
6. **Title/Survey**: Title commitments, ALTA surveys, easements, encroachments
7. **Zoning/Permits**: Zoning classification, permits, CO status, entitlements
8. **Insurance/Risk**: Coverage, deductibles, loss runs, COIs

## License

This dataset generator is provided as-is. Generated data is fictional and public-safe.

## Contributing

This is a specialized dataset generator. For improvements:
1. Enhance Ollama prompts for more realistic content
2. Add more source types or asset types
3. Improve hard negative selection strategies
4. Add more validation checks

## Citation

If you use this dataset, please cite:

```
RE-Diligence-Embeddings: Synthetic Real Estate Diligence Dataset Generator
https://github.com/your-username/re-diligence-embeddings
```
