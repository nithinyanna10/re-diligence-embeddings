#!/usr/bin/env python3
"""
Generate copy/paste snippets for README and model card.
"""

import json
import argparse
from pathlib import Path
from collections import Counter


def load_corpus_stats(data_dir: Path) -> dict:
    """Load statistics from corpus."""
    corpus_file = data_dir / "corpus.jsonl"
    
    if not corpus_file.exists():
        return {}
    
    source_types = Counter()
    asset_types = Counter()
    companies = set()
    total_chunks = 0
    
    with open(corpus_file, "r") as f:
        for line in f:
            chunk = json.loads(line)
            source_types[chunk.get("source_type", "unknown")] += 1
            asset_types[chunk.get("asset_type", "unknown")] += 1
            companies.add(chunk.get("company", ""))
            total_chunks += 1
    
    return {
        "total_chunks": total_chunks,
        "total_companies": len(companies),
        "source_types": dict(source_types),
        "asset_types": dict(asset_types)
    }


def print_model_card_metrics(data_dir: Path):
    """Print model card metrics placeholders."""
    stats = load_corpus_stats(data_dir)
    
    print("\n" + "="*60)
    print("MODEL CARD METRICS (Placeholder)")
    print("="*60)
    print(f"""
## Dataset Statistics

- **Total Chunks**: {stats.get('total_chunks', 'N/A')}
- **Total Companies/Deals**: {stats.get('total_companies', 'N/A')}
- **Source Types**: {len(stats.get('source_types', {}))}
- **Asset Types**: {len(stats.get('asset_types', {}))}

## Training Configuration

- **Model Architecture**: [To be filled - e.g., sentence-transformers/all-MiniLM-L6-v2]
- **Training Method**: [To be filled - e.g., contrastive learning, in-batch negatives]
- **Epochs**: [To be filled]
- **Batch Size**: [To be filled]
- **Learning Rate**: [To be filled]
- **Max Sequence Length**: [To be filled]

## Evaluation Metrics

- **Retrieval@10**: [To be filled]
- **Retrieval@100**: [To be filled]
- **MRR@10**: [To be filled]
- **NDCG@10**: [To be filled]

## Training Data Distribution

### By Source Type:
""")
    for source, count in sorted(stats.get('source_types', {}).items()):
        print(f"- {source}: {count}")
    
    print("\n### By Asset Type:")
    for asset, count in sorted(stats.get('asset_types', {}).items()):
        print(f"- {asset}: {count}")
    
    print("\n" + "="*60)


def print_space_instructions(data_dir: Path):
    """Print HuggingFace Space setup instructions."""
    print("\n" + "="*60)
    print("HUGGINGFACE SPACE INSTRUCTIONS")
    print("="*60)
    print("""
## Setup Instructions

1. Create a new HuggingFace Space (Gradio app)

2. Upload `data/space_corpus.jsonl` to the Space

3. Use this Gradio app template:

```python
import gradio as gr
import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load model (replace with your trained model)
model = SentenceTransformer('your-username/re-dd-embeddings')

# Load corpus
corpus = []
with open('space_corpus.jsonl', 'r') as f:
    for line in f:
        corpus.append(json.loads(line))

# Build index
embeddings = model.encode([c['text'] for c in corpus])
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings.astype('float32'))

def search(query, top_k=5):
    query_emb = model.encode([query])
    scores, indices = index.search(query_emb.astype('float32'), top_k)
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        chunk = corpus[idx]
        results.append({
            'title': chunk['title'],
            'source': chunk['source_type'],
            'company': chunk['company'],
            'text': chunk['text'][:500] + '...',
            'score': float(score)
        })
    return results

iface = gr.Interface(
    fn=search,
    inputs=gr.Textbox(label="Search Query", placeholder="SNDA status top tenants"),
    outputs=gr.JSON(label="Results"),
    title="RE-DD-Embeddings Semantic Search",
    description="Search real estate diligence documents"
)
iface.launch()
```

4. Add requirements.txt:
```
sentence-transformers
faiss-cpu
gradio
numpy
```

5. Deploy and test!
""")


def print_example_queries(data_dir: Path):
    """Print example queries from eval set."""
    queries_file = data_dir / "queries.jsonl"
    
    if not queries_file.exists():
        print("\n⚠ Queries file not found. Generate eval set first.")
        return
    
    print("\n" + "="*60)
    print("EXAMPLE QUERIES (from eval set)")
    print("="*60)
    
    queries = []
    with open(queries_file, "r") as f:
        for line in f:
            queries.append(json.loads(line))
    
    print(f"\nTotal eval queries: {len(queries)}\n")
    print("Sample queries:\n")
    
    for i, q in enumerate(queries[:20], 1):
        print(f"{i}. {q['text']}")
        print(f"   [{q['source_type']} | {q['asset_type']} | {q['company']}]")
    
    if len(queries) > 20:
        print(f"\n... and {len(queries) - 20} more queries")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description="Generate README snippets")
    parser.add_argument("--data_dir", default="data", help="Data directory")
    parser.add_argument("--all", action="store_true", help="Print all snippets")
    
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    
    if args.all:
        print_model_card_metrics(data_dir)
        print_space_instructions(data_dir)
        print_example_queries(data_dir)
    else:
        print("Use --all to print all snippets, or modify script to print specific sections.")


if __name__ == "__main__":
    main()
