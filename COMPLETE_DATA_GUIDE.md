# Complete Data Generation Guide

## ✅ What's Fixed

1. **All metadata fields now guaranteed:**
   - ✅ `topic` field added (critical for hard negative mining)
   - ✅ All 25 required fields always populated
   - ✅ Default values for missing fields
   - ✅ Validation before writing

2. **Updated generation script:**
   - `scripts/generate_dataset.py` now includes `extract_primary_topic()` function
   - `write_corpus_chunk()` ensures ALL fields with defaults
   - Validates text length and required fields

3. **Validation & Fixing tools:**
   - `validate_corpus_completeness.py` - Check for missing fields
   - `fix_corpus_metadata.py` - Auto-fix missing fields

---

## 📋 Complete Metadata Schema

Every chunk now has these **25 fields** (all guaranteed):

### Core Fields
- `doc_id` - Document identifier
- `chunk_id` - Unique chunk identifier
- `text` - Chunk text (120-260 words)
- `title` - Document title

### Source Metadata
- `source_type` - Document type (CIM, IC_Memo, Diligence_Report, etc.)
- `company` - Company/asset name
- `sector` - Always "Real Estate"
- `deal_stage` - Sourcing, LOI, Diligence, IC, PostClose
- `date` - Document date (YYYY-MM-DD)
- `region` - US-Northeast, US-Southeast, etc.
- `market` - City, State
- `doc_url` - URL (usually empty)
- `confidentiality` - Always "public"

### Asset Metadata
- `asset_type` - Multifamily, Industrial, Office, Retail, etc.
- `deal_type` - Acquisition, Refi, Development, JV, Disposition
- `vintage` - Year built
- `unit_count` - Number of units (0 for non-residential)
- `sqft` - Square footage
- `occupancy_pct` - Occupancy percentage (0.0-1.0)
- `noi` - Net Operating Income (e.g., "$2.8M")
- `cap_rate` - Cap rate (e.g., "5.1%")
- `ltv` - Loan-to-Value (e.g., "72.3%")
- `dscr` - Debt Service Coverage Ratio (e.g., "1.43x")

### Topic & Tags
- `tags` - Array of 2-6 topic tags (e.g., ["NOI", "GPR", "WALE"])
- `topic` - **Primary topic** extracted from tags (for hard negative mining)

---

## 🔧 Usage

### Generate New Corpus (with all fields)
```bash
python scripts/generate_dataset.py \
    --target_chunks 10000 \
    --model gemini-3-flash-preview:cloud
```

### Validate Existing Corpus
```bash
python validate_corpus_completeness.py --corpus data/corpus.jsonl
```

### Fix Missing Fields
```bash
python fix_corpus_metadata.py \
    --corpus data/corpus.jsonl \
    --output data/corpus_fixed.jsonl

# Replace original
mv data/corpus_fixed.jsonl data/corpus.jsonl
```

---

## 🎯 Topic Field Details

The `topic` field is **critical** for hard negative mining. It's extracted from tags using priority:

1. **Primary topics:** lease, CAM, rent_roll, T12, NOI, capex, title, ALTA
2. **Environmental:** PhaseI, RECs, environmental
3. **Financial:** DSCR, debt_yield, debt_terms, debt_covenants
4. **Legal:** estoppel, SNDA, zoning, permits, easements
5. **Operational:** WALE, GPR, vacancy_loss, credit_loss, concessions, rollover
6. **Physical:** PCA, deferred_maintenance, survey, insurance_coverage

**Fallback:** If no matching tag found, uses first tag or "general"

---

## ✅ Quality Checks

The generation script now validates:
- ✅ Text length (minimum 50 characters)
- ✅ Valid chunk_id
- ✅ All required fields present
- ✅ Tags array exists and is non-empty
- ✅ Topic field populated

---

## 📊 Example Chunk

```json
{
  "doc_id": "doc_001",
  "chunk_id": "chunk_001",
  "text": "Summit Park Investments represents...",
  "title": "Confidential Information Memorandum - Summit Park Investments",
  "source_type": "CIM",
  "company": "Summit Park Investments",
  "sector": "Real Estate",
  "deal_stage": "Sourcing",
  "date": "2024-01-15",
  "region": "US-Southwest",
  "doc_url": "",
  "tags": ["GPR", "NOI"],
  "topic": "NOI",
  "confidentiality": "public",
  "asset_type": "Industrial",
  "deal_type": "Refi",
  "market": "Dallas, TX",
  "vintage": "2008",
  "unit_count": 0,
  "sqft": 489563,
  "occupancy_pct": 0.977,
  "noi": "$2.8M",
  "cap_rate": "5.1%",
  "ltv": "72.3%",
  "dscr": "1.43x"
}
```

---

## 🚀 Next Steps

1. ✅ **Corpus fixed** - All chunks now have `topic` field
2. **Generate better pairs** - Use `generate_better_pairs.py` (needs `topic` field)
3. **Mine hard negatives** - Use `mine_hard_negatives.py`
4. **Retrain** - With complete, accurate data

---

## 📝 Notes

- All fields are **guaranteed** to exist (no nulls)
- Default values ensure backward compatibility
- Topic extraction is intelligent (priority-based)
- Validation catches issues before training
