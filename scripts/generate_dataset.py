#!/usr/bin/env python3
"""
Generate synthetic real estate diligence dataset using Ollama.
Creates corpus.jsonl with realistic RE diligence documents.
"""

import json
import random
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from ollama_client import run_ollama, parse_json_response


# Asset type distribution (realistic mix)
ASSET_TYPES = [
    "Multifamily", "Multifamily", "Multifamily",  # 30%
    "Industrial", "Industrial", "Industrial",  # 30%
    "Office", "Office",  # 20%
    "Retail",  # 10%
    "SelfStorage", "Hotel", "DataCenter", "MixedUse", "MHC"  # 10%
]

DEAL_TYPES = ["Acquisition", "Refi", "Development", "JV", "Disposition"]
DEAL_STAGES = ["Sourcing", "LOI", "Diligence", "IC", "PostClose"]

REGIONS = {
    "US-Northeast": ["Boston, MA", "New York, NY", "Philadelphia, PA", "Washington, DC"],
    "US-Southeast": ["Atlanta, GA", "Miami, FL", "Charlotte, NC", "Nashville, TN", "Tampa, FL"],
    "US-Midwest": ["Chicago, IL", "Detroit, MI", "Minneapolis, MN", "Columbus, OH"],
    "US-Southwest": ["Dallas, TX", "Houston, TX", "Phoenix, AZ", "Denver, CO", "Austin, TX"],
    "US-West": ["Los Angeles, CA", "San Francisco, CA", "Seattle, WA", "Portland, OR", "San Diego, CA"]
}

SOURCE_TYPES = [
    "CIM", "IC_Memo", "Diligence_Report", "Lease_Abstract", "Rent_Roll_Summary",
    "T12_Operating_Statement", "PCA_Report", "PhaseI_ESA", "Title_Commitment",
    "ALTA_Survey_Summary", "Debt_Term_Sheet", "Insurance_Summary", "Zoning_Report",
    "Permit_Log", "Appraisal_Summary"
]

# RE diligence topic tags
TOPIC_TAGS = [
    "lease", "CAM", "rent_roll", "T12", "NOI", "capex", "title", "ALTA",
    "PhaseI", "RECs", "zoning", "permits", "insurance", "debt_terms",
    "DSCR", "debt_yield", "TI_LC", "estoppel", "SNDA", "WALE", "GPR",
    "vacancy_loss", "credit_loss", "concessions", "rollover", "PCA",
    "deferred_maintenance", "environmental", "survey", "easements",
    "zoning_compliance", "permits", "insurance_coverage", "debt_covenants"
]


def generate_asset_metadata(asset_type: str, deal_type: str, region: str, market: str, seed: int) -> Dict[str, Any]:
    """Generate realistic asset metadata based on type."""
    random.seed(seed)
    
    # Realistic ranges by asset type
    if asset_type == "Multifamily":
        unit_count = random.randint(120, 450)
        sqft = unit_count * random.randint(750, 1200)
        occupancy_pct = round(random.uniform(0.88, 0.97), 3)
        noi_range = (unit_count * random.uniform(800, 1800) * 12) / 1_000_000
        cap_rate = round(random.uniform(4.5, 6.5), 2)
        ltv = round(random.uniform(55, 70), 1)
        dscr = round(random.uniform(1.35, 1.65), 2)
        vintage = random.randint(1985, 2020)
    elif asset_type == "Industrial":
        sqft = random.randint(150_000, 800_000)
        unit_count = 0
        occupancy_pct = round(random.uniform(0.92, 0.98), 3)
        noi_range = (sqft * random.uniform(4.5, 7.5)) / 1_000_000
        cap_rate = round(random.uniform(5.0, 7.0), 2)
        ltv = round(random.uniform(60, 75), 1)
        dscr = round(random.uniform(1.40, 1.70), 2)
        vintage = random.randint(1990, 2018)
    elif asset_type == "Office":
        sqft = random.randint(80_000, 350_000)
        unit_count = 0
        occupancy_pct = round(random.uniform(0.75, 0.92), 3)
        noi_range = (sqft * random.uniform(18, 32)) / 1_000_000
        cap_rate = round(random.uniform(5.5, 7.5), 2)
        ltv = round(random.uniform(55, 68), 1)
        dscr = round(random.uniform(1.30, 1.60), 2)
        vintage = random.randint(1980, 2015)
    elif asset_type == "Retail":
        sqft = random.randint(60_000, 250_000)
        unit_count = random.randint(8, 35)
        occupancy_pct = round(random.uniform(0.85, 0.95), 3)
        noi_range = (sqft * random.uniform(12, 22)) / 1_000_000
        cap_rate = round(random.uniform(5.0, 7.0), 2)
        ltv = round(random.uniform(58, 72), 1)
        dscr = round(random.uniform(1.35, 1.65), 2)
        vintage = random.randint(1985, 2015)
    else:  # SelfStorage, Hotel, etc.
        sqft = random.randint(50_000, 200_000)
        unit_count = random.randint(0, 150)
        occupancy_pct = round(random.uniform(0.82, 0.95), 3)
        noi_range = random.uniform(2.5, 12.0)
        cap_rate = round(random.uniform(5.5, 8.0), 2)
        ltv = round(random.uniform(55, 70), 1)
        dscr = round(random.uniform(1.30, 1.60), 2)
        vintage = random.randint(1990, 2020)
    
    return {
        "unit_count": unit_count,
        "sqft": sqft,
        "occupancy_pct": occupancy_pct,
        "noi": f"${noi_range:.1f}M",
        "cap_rate": f"{cap_rate}%",
        "ltv": f"{ltv}%",
        "dscr": f"{dscr}x",
        "vintage": str(vintage)
    }


def generate_company_name(asset_type: str, market: str, seed: int) -> str:
    """Generate fictional company/asset name."""
    random.seed(seed)
    prefixes = ["Riverside", "Parkview", "Summit", "Crestwood", "Highland", "Oakmont", "Pinecrest", "Meadowbrook"]
    suffixes = ["Properties", "Holdings", "Partners", "Investments", "Capital", "Group", "Ventures"]
    
    if asset_type == "Multifamily":
        names = ["Garden", "Manor", "Village", "Towers", "Heights", "Place", "Court", "Lane"]
    elif asset_type == "Industrial":
        names = ["Logistics", "Distribution", "Park", "Center", "Hub", "Complex"]
    elif asset_type == "Office":
        names = ["Plaza", "Tower", "Center", "Building", "Complex", "Square"]
    elif asset_type == "Retail":
        names = ["Shopping Center", "Plaza", "Marketplace", "Commons", "Village"]
    else:
        names = ["Facility", "Center", "Complex", "Park"]
    
    name_part = random.choice(names)
    prefix = random.choice(prefixes)
    suffix = random.choice(suffixes)
    
    return f"{prefix} {name_part} {suffix}"


def create_deal_pack_prompt(asset_metadata: Dict, seed: int) -> str:
    """Create prompt for generating a full diligence pack."""
    company = asset_metadata["company"]
    asset_type = asset_metadata["asset_type"]
    deal_type = asset_metadata["deal_type"]
    market = asset_metadata["market"]
    region = asset_metadata["region"]
    vintage = asset_metadata["vintage"]
    unit_count = asset_metadata["unit_count"]
    sqft = asset_metadata["sqft"]
    occupancy_pct = asset_metadata["occupancy_pct"]
    noi = asset_metadata["noi"]
    cap_rate = asset_metadata["cap_rate"]
    ltv = asset_metadata["ltv"]
    dscr = asset_metadata["dscr"]
    
    # Generate realistic date (within last 2 years)
    base_date = datetime.now() - timedelta(days=random.randint(0, 730))
    doc_date = base_date.strftime("%Y-%m-%d")
    
    prompt = f"""You are a real estate diligence analyst generating a complete institutional-grade diligence pack for a fictional {asset_type} property.

ASSET DETAILS:
- Company/Asset Name: {company}
- Asset Type: {asset_type}
- Deal Type: {deal_type}
- Market: {market}
- Region: {region}
- Vintage: {vintage}
- Unit Count: {unit_count}
- Square Feet: {sqft:,}
- Occupancy: {occupancy_pct:.1%}
- NOI: {noi}
- Cap Rate: {cap_rate}
- LTV: {ltv}
- DSCR: {dscr}

GENERATE A COMPLETE DILIGENCE PACK with the following documents. Each document should have multiple chunks (120-260 words each) covering different topics from the RE diligence taxonomy:

REQUIRED DOCUMENTS:
1. CIM (Confidential Information Memorandum) - 8-12 chunks covering: property overview, location, market fundamentals, investment highlights, financial summary, competitive positioning
2. IC_Memo (Investment Committee Memo) - 8-12 chunks covering: deal rationale, underwriting assumptions, risk factors, return profile, recommendation
3. Diligence_Report - 10-15 chunks covering: executive summary, lease analysis, financial performance, physical condition, legal, environmental, title, debt terms
4. Lease_Abstract - 6-10 chunks covering: key lease terms, base rent, escalations, free rent, TI/LC, renewal options, SNDA status, estoppels, exclusives, co-tenancy, go-dark clauses
5. Rent_Roll_Summary - 4-8 chunks covering: in-place vs market rent, tenant mix, WALE, top tenant concentration, delinquencies, bad debt, concessions, rollover schedule
6. T12_Operating_Statement - 4-8 chunks covering: trailing 12-month NOI, revenue line items (base rent, other income, parking, RUBS), expense line items (utilities, R&M, payroll, management fee, taxes, insurance), normalization adjustments, one-time expenses
7. PCA_Report (Property Condition Assessment) - 4-8 chunks covering: roof condition, HVAC systems, MEP, structural, façade, elevators, life safety, deferred maintenance with $ ranges and priorities, code compliance, ADA
8. PhaseI_ESA - 3-6 chunks covering: site history, RECs (Recognized Environmental Conditions), recommendations, Phase II triggers, asbestos/lead/PCBs/mold mentions
9. Title_Commitment - 3-6 chunks covering: easements, encroachments, exceptions, ALTA endorsements, access issues, chain of title
10. ALTA_Survey_Summary - 2-4 chunks covering: boundary, encroachments, parking ratio, setbacks, improvements
11. Debt_Term_Sheet - 2-4 chunks covering: spread, floors, IO period, amortization, debt yield, DSCR/LTV covenants, reserves, prepayment (yield maintenance/defeasance), stepdown, lockout, recourse carveouts
12. Insurance_Summary - 2-4 chunks covering: wind/flood zone, deductibles, exclusions, loss runs, required COIs, coverage gaps, litigation/claims
13. Zoning_Report - 2-4 chunks (if Development or value-add): zoning classification, nonconforming use, FAR, setbacks, entitlement risk, variances, CUPs
14. Permit_Log - 2-4 chunks (if Development): open permits, inspections, CO status, development agreements
15. Appraisal_Summary - 2-4 chunks: valuation methodology, comparable sales, income approach, cost approach, final opinion of value

CRITICAL REQUIREMENTS:
- Use REAL institutional RE diligence vocabulary: SNDA, estoppel, CAM, WALE, GPR, vacancy loss, credit loss, TI/LC, debt yield, SOFR floor, ALTA exceptions, RECs, Phase II, yield maintenance, defeasance, DSCR, LTV, NOI bridge, rollover, concessions, deferred maintenance, etc.
- Each chunk must be 120-260 words, crisp and professional
- Include realistic numbers, percentages, dollar amounts
- Each chunk must have 2-6 topic tags from: lease, CAM, rent_roll, T12, NOI, capex, title, ALTA, PhaseI, RECs, zoning, permits, insurance, debt_terms, DSCR, debt_yield, TI_LC, estoppel, SNDA, WALE, GPR, vacancy_loss, credit_loss, concessions, rollover, PCA, deferred_maintenance, environmental, survey, easements, zoning_compliance, insurance_coverage, debt_covenants
- Ensure different chunks cover different topics; avoid repetition
- Include occasional bullet-like lists using "-" lines within text
- All data must be FICTIONAL and PUBLIC-SAFE (no real companies, tenants, lenders)

Return STRICT JSON ONLY with this exact structure:
{{
  "company": "{company}",
  "docs": [
    {{
      "doc_id": "doc_001",
      "title": "Confidential Information Memorandum - {company}",
      "source_type": "CIM",
      "date": "2024-01-15",
      "region": "{region}",
      "tags": ["tag1", "tag2"],
      "chunks": [
        {{
          "chunk_id": "chunk_001",
          "tags": ["tag1", "tag2"],
          "text": "120-260 word chunk text here..."
        }}
      ]
    }}
  ]
}}

CRITICAL: Each chunk MUST have tags array with 2-6 topic tags. Tags are used for hard negative mining.
Tags must be from the RE diligence taxonomy: lease, CAM, rent_roll, T12, NOI, capex, title, ALTA, PhaseI, RECs, zoning, permits, insurance, debt_terms, DSCR, debt_yield, TI_LC, estoppel, SNDA, WALE, GPR, vacancy_loss, credit_loss, concessions, rollover, PCA, deferred_maintenance, environmental, survey, easements, zoning_compliance, insurance_coverage, debt_covenants

No markdown, no code blocks, no explanation. Return STRICT JSON ONLY."""
    
    return prompt


def generate_deal_pack(model: str, asset_metadata: Dict, seed: int) -> Dict[str, Any]:
    """Generate one complete deal pack using Ollama."""
    prompt = create_deal_pack_prompt(asset_metadata, seed)
    
    try:
        response = run_ollama(model, prompt, timeout_s=300)
        result = parse_json_response(response)
        return result
    except Exception as e:
        print(f"Error generating deal pack for {asset_metadata.get('company', 'unknown')}: {e}")
        raise


def extract_primary_topic(tags: List[str]) -> str:
    """Extract primary topic from tags for hard negative mining."""
    if not tags:
        return "general"
    
    # Priority order for topic extraction
    topic_priority = [
        "lease", "CAM", "rent_roll", "T12", "NOI", "capex", "title", "ALTA",
        "PhaseI", "RECs", "zoning", "permits", "insurance", "debt_terms",
        "DSCR", "debt_yield", "TI_LC", "estoppel", "SNDA", "WALE", "GPR",
        "vacancy_loss", "credit_loss", "concessions", "rollover", "PCA",
        "deferred_maintenance", "environmental", "survey", "easements",
        "zoning_compliance", "insurance_coverage", "debt_covenants"
    ]
    
    # Find first matching tag in priority order
    for priority_tag in topic_priority:
        for tag in tags:
            if priority_tag.lower() in tag.lower() or tag.lower() in priority_tag.lower():
                return priority_tag
    
    # Fallback to first tag
    return tags[0].lower() if tags else "general"


def write_corpus_chunk(out_file, chunk_data: Dict, doc_data: Dict, asset_metadata: Dict):
    """Write a single corpus chunk to JSONL with ALL metadata fields."""
    # Get tags
    tags = chunk_data.get("tags", doc_data.get("tags", []))
    if not tags:
        # Generate default tags based on source_type
        source_type = doc_data.get("source_type", "CIM")
        if "CIM" in source_type:
            tags = ["NOI", "GPR"]
        elif "Lease" in source_type:
            tags = ["lease", "WALE"]
        elif "PhaseI" in source_type or "ESA" in source_type:
            tags = ["PhaseI", "environmental"]
        elif "ALTA" in source_type or "Survey" in source_type:
            tags = ["ALTA", "survey"]
        elif "Title" in source_type:
            tags = ["title", "easements"]
        elif "T12" in source_type or "Operating" in source_type:
            tags = ["T12", "NOI"]
        elif "Debt" in source_type:
            tags = ["debt_terms", "DSCR"]
        else:
            tags = ["general"]
    
    # Extract primary topic
    topic = extract_primary_topic(tags)
    
    # Ensure ALL required fields are present with defaults
    chunk = {
        "doc_id": doc_data.get("doc_id", "doc_unknown"),
        "chunk_id": chunk_data.get("chunk_id", "chunk_unknown"),
        "text": chunk_data.get("text", ""),
        "title": doc_data.get("title", f"Document - {asset_metadata.get('company', 'Unknown')}"),
        "source_type": doc_data.get("source_type", "CIM"),
        "company": asset_metadata.get("company", "Unknown Company"),
        "sector": "Real Estate",
        "deal_stage": asset_metadata.get("deal_stage", "Sourcing"),
        "date": doc_data.get("date", datetime.now().strftime("%Y-%m-%d")),
        "region": asset_metadata.get("region", "US-Northeast"),
        "doc_url": doc_data.get("doc_url", ""),
        "tags": tags,
        "topic": topic,  # CRITICAL: Add topic field for hard negative mining
        "confidentiality": "public",
        "asset_type": asset_metadata.get("asset_type", "Multifamily"),
        "deal_type": asset_metadata.get("deal_type", "Acquisition"),
        "market": asset_metadata.get("market", "Boston, MA"),
        "vintage": str(asset_metadata.get("vintage", "2000")),
        "unit_count": asset_metadata.get("unit_count", 0),
        "sqft": asset_metadata.get("sqft", 0),
        "occupancy_pct": asset_metadata.get("occupancy_pct", 0.90),
        "noi": asset_metadata.get("noi", "$0M"),
        "cap_rate": asset_metadata.get("cap_rate", "5.0%"),
        "ltv": asset_metadata.get("ltv", "65.0%"),
        "dscr": asset_metadata.get("dscr", "1.40x")
    }
    
    # Validate critical fields
    if not chunk["text"] or len(chunk["text"]) < 50:
        raise ValueError(f"Chunk {chunk['chunk_id']} has invalid text (too short or empty)")
    
    if not chunk["chunk_id"] or chunk["chunk_id"] == "chunk_unknown":
        raise ValueError(f"Chunk missing valid chunk_id")
    
    out_file.write(json.dumps(chunk) + "\n")
    out_file.flush()


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic RE diligence dataset")
    parser.add_argument("--model", default="gemini-3-flash-preview:cloud", help="Ollama model name")
    parser.add_argument("--out_dir", default="data", help="Output directory")
    parser.add_argument("--companies", type=int, default=60, help="Number of companies/deals")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--target_chunks", type=int, default=3000, help="Target number of chunks")
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    
    corpus_file = out_dir / "corpus.jsonl"
    index_file = out_dir / "index.json"
    
    # Generate deals
    deals = []
    total_chunks = 0
    failed_deals = 0
    
    print("="*70)
    print("RE-DD-EMBEDDINGS DATASET GENERATION")
    print("="*70)
    print(f"Target: {args.target_chunks:,} chunks across {args.companies} deals")
    print(f"Model: {args.model}")
    print("="*70)
    print()
    
    with open(corpus_file, "w") as f:
        pbar = tqdm(
            range(args.companies),
            desc="Deals",
            unit="deal",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} deals [{elapsed}<{remaining}, {rate_fmt}] | Chunks: {postfix}"
        )
        
        for i in pbar:
            # Update progress bar with chunk count
            remaining_chunks = max(0, args.target_chunks - total_chunks)
            pct_complete = (total_chunks / args.target_chunks * 100) if args.target_chunks > 0 else 0
            pbar.set_postfix({
                "Generated": f"{total_chunks:,}/{args.target_chunks:,}",
                "Remaining": f"{remaining_chunks:,}",
                "Progress": f"{pct_complete:.1f}%"
            })
            
            # Select asset type, deal type, region
            asset_type = random.choice(ASSET_TYPES)
            deal_type = random.choice(DEAL_TYPES)
            region = random.choice(list(REGIONS.keys()))
            market = random.choice(REGIONS[region])
            
            # Generate metadata
            metadata = generate_asset_metadata(asset_type, deal_type, region, market, args.seed + i)
            company = generate_company_name(asset_type, market, args.seed + i * 100)
            
            asset_metadata = {
                "company": company,
                "asset_type": asset_type,
                "deal_type": deal_type,
                "deal_stage": random.choice(DEAL_STAGES),
                "region": region,
                "market": market,
                **metadata
            }
            
            # Generate deal pack
            try:
                pack = generate_deal_pack(args.model, asset_metadata, args.seed + i * 1000)
                
                chunks_this_deal = 0
                # Write chunks
                for doc in pack.get("docs", []):
                    # Use date from doc or generate realistic one
                    doc_date = doc.get("date")
                    if not doc_date or len(doc_date) != 10:
                        base_date = datetime.now() - timedelta(days=random.randint(0, 730))
                        doc_date = base_date.strftime("%Y-%m-%d")
                    doc["date"] = doc_date
                    
                    for chunk in doc.get("chunks", []):
                        write_corpus_chunk(f, chunk, doc, asset_metadata)
                        total_chunks += 1
                        chunks_this_deal += 1
                
                # Store for index
                deals.append({
                    "company": company,
                    "asset_metadata": asset_metadata,
                    "pack": pack
                })
                
            except Exception as e:
                failed_deals += 1
                tqdm.write(f"⚠ Skipping deal {i+1} ({company}) due to error: {e}")
                continue
            
            if total_chunks >= args.target_chunks:
                pbar.set_postfix({
                    "Generated": f"{total_chunks:,}/{args.target_chunks:,}",
                    "Remaining": "0",
                    "Progress": "100.0%",
                    "Status": "TARGET REACHED"
                })
                break
    
    # Save index
    with open(index_file, "w") as f:
        json.dump(deals, f, indent=2)
    
    # Final summary
    print()
    print("="*70)
    print("GENERATION COMPLETE")
    print("="*70)
    print(f"✓ Deals completed:     {len(deals):,}/{args.companies:,}")
    if failed_deals > 0:
        print(f"⚠ Deals failed:        {failed_deals}")
    print(f"✓ Chunks generated:   {total_chunks:,}/{args.target_chunks:,} ({total_chunks/args.target_chunks*100:.1f}%)")
    print(f"✓ Avg chunks/deal:     {total_chunks/len(deals):.1f}" if deals else "N/A")
    print(f"✓ Corpus file:         {corpus_file}")
    print(f"✓ Index file:          {index_file}")
    print("="*70)


if __name__ == "__main__":
    main()
