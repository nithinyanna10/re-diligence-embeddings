#!/usr/bin/env python3
"""
Generate better training pairs with diverse query types and proper hard negatives.
Target: 5 queries per chunk = ~15k pairs.
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm
import random
import sys

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent / 'scripts'))
from ollama_client import run_ollama


def load_corpus(corpus_file):
    """Load corpus."""
    corpus = {}
    with open(corpus_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            chunk = json.loads(line)
            corpus[chunk['chunk_id']] = chunk
    return corpus


def generate_queries_for_chunk(chunk, model='gemini-3-flash-preview:cloud'):
    """Generate 5 diverse queries for a chunk."""
    
    chunk_text = chunk['text']
    source_type = chunk.get('source_type', 'Unknown')
    topic = chunk.get('topic', 'General')
    
    system_msg = "You are a JSON generator. Return ONLY valid JSON arrays. No thinking, no analysis, no explanations, no markdown."
    
    prompt = f"""Return ONLY a JSON array with 5 search queries.

Chunk: {chunk_text[:800]}
Source: {source_type}, Topic: {topic}

Generate 5 queries:
1. Short keyword (3-8 words)
2. Question format  
3. Clause hunt (SNDA, estoppel, CAM, DSCR, WALE)
4. Financial/metrics query
5. Document-specific query

Format: ["query1", "query2", "query3", "query4", "query5"]"""

    try:
        response = run_ollama(model, prompt, max_retries=3, system=system_msg)
        
        # Parse JSON array - handle "Thinking..." and "...done thinking."
        import re
        response = response.strip()
        
        # Remove "Thinking..." and "...done thinking." patterns
        # Find the JSON array after any thinking text
        response = re.sub(r'Thinking[^\n]*\n', '', response, flags=re.IGNORECASE)
        response = re.sub(r'\.\.\.done thinking\.', '', response, flags=re.IGNORECASE)
        response = re.sub(r'\*\*[^\*]+\*\*', '', response)  # Remove markdown bold
        
        # Strategy 1: Find complete JSON array (balanced brackets)
        json_match = re.search(r'\[[^\]]*(?:\[[^\]]*\][^\]]*)*\]', response, re.DOTALL)
        if json_match:
            response = json_match.group(0)
        else:
            # Strategy 2: Find first [ and try to find matching ]
            first_bracket = response.find('[')
            if first_bracket >= 0:
                # Count brackets to find matching ]
                bracket_count = 0
                last_bracket = first_bracket
                for i, char in enumerate(response[first_bracket:], first_bracket):
                    if char == '[':
                        bracket_count += 1
                    elif char == ']':
                        bracket_count -= 1
                        if bracket_count == 0:
                            last_bracket = i
                            break
                
                if bracket_count == 0:
                    response = response[first_bracket:last_bracket + 1]
                else:
                    # Incomplete JSON - try to extract what we have
                    json_match = re.search(r'\[.*', response[first_bracket:], re.DOTALL)
                    if json_match:
                        partial = json_match.group(0)
                        # Try to close it
                        if partial.count('[') > partial.count(']'):
                            partial += ']' * (partial.count('[') - partial.count(']'))
                        response = partial
        
        response = response.strip()
        
        # Clean up any remaining non-JSON text
        response = re.sub(r'^[^\[]*', '', response)  # Remove text before [
        response = re.sub(r'[^\]]*$', '', response)  # Remove text after ]
        if not response.startswith('['):
            response = '[' + response
        if not response.endswith(']'):
            response = response + ']'
        
        queries = json.loads(response)
        
        if not isinstance(queries, list):
            queries = [queries] if isinstance(queries, str) else []
        
        # Ensure exactly 5 queries
        if len(queries) < 5:
            # Pad with variations
            while len(queries) < 5:
                base = queries[0] if queries else "query about this document"
                queries.append(f"{base} (variation {len(queries) + 1})")
        elif len(queries) > 5:
            queries = queries[:5]
        
        return queries
    
    except Exception as e:
        print(f"Error generating queries for {chunk.get('chunk_id', 'unknown')}: {e}")
        # Fallback queries
        return [
            f"query about {source_type}",
            f"what information about {topic}",
            f"details regarding this {source_type}",
            f"analysis of {topic}",
            f"{source_type} document information"
        ]


def find_hard_negatives(chunk_id, chunk, corpus, num_negatives=3):
    """Find hard negatives: same asset/company, different topic/source."""
    
    company = chunk.get('company', '')
    asset_type = chunk.get('asset_type', '')
    topic = chunk.get('topic', '')
    source_type = chunk.get('source_type', '')
    chunk_text = chunk.get('text', '')
    
    candidates = []
    
    for other_id, other_chunk in corpus.items():
        if other_id == chunk_id:
            continue
        
        # Hard negative criteria:
        # 1. Same company/asset but different topic/source
        # 2. Or same source type but different topic
        # 3. Or similar but different document type
        
        other_company = other_chunk.get('company', '')
        other_topic = other_chunk.get('topic', '')
        other_source = other_chunk.get('source_type', '')
        
        is_hard_negative = False
        
        # Same company, different topic
        if company and other_company == company and other_topic != topic:
            is_hard_negative = True
        
        # Same source type, different topic
        if source_type and other_source == source_type and other_topic != topic:
            is_hard_negative = True
        
        # Confusable pairs (CAM vs taxes, Phase I vs PCA, etc.)
        confusable_pairs = [
            ('CAM', 'taxes'), ('CAM', 'insurance'),
            ('Phase I', 'PCA'), ('Phase I', 'Phase II'),
            ('Title', 'ALTA'), ('Lease', 'Rent Roll'),
            ('DSCR', 'debt yield'), ('NOI', 'EBITDA')
        ]
        
        for term1, term2 in confusable_pairs:
            if (term1 in chunk_text and term2 in other_chunk['text']) or \
               (term2 in chunk_text and term1 in other_chunk['text']):
                is_hard_negative = True
                break
        
        if is_hard_negative:
            candidates.append({
                'chunk_id': other_id,
                'topic': other_topic,
                'source_type': other_source,
                'text_preview': other_chunk['text'][:100]
            })
    
    # Randomly sample
    if len(candidates) >= num_negatives:
        return random.sample(candidates, num_negatives)
    else:
        return candidates


def generate_pairs(corpus, output_file, model='gemini-3-flash-preview:cloud', batch_size=10):
    """Generate training pairs."""
    
    chunk_ids = list(corpus.keys())
    total_chunks = len(chunk_ids)
    
    print(f"Generating pairs for {total_chunks} chunks...")
    print(f"Target: ~{total_chunks * 5} pairs (5 queries per chunk)")
    print()
    
    pairs = []
    failed_chunks = 0
    
    with open(output_file, 'w') as f:
        for i in tqdm(range(0, total_chunks, batch_size), desc="Processing chunks"):
            batch = chunk_ids[i:i+batch_size]
            
            for chunk_id in batch:
                chunk = corpus[chunk_id]
                
                # Generate queries
                queries = generate_queries_for_chunk(chunk, model)
                
                # Find hard negatives
                hard_negatives = find_hard_negatives(chunk_id, chunk, corpus, num_negatives=3)
                
                if not hard_negatives:
                    failed_chunks += 1
                    continue
                
                # Create pairs (one per query)
                for query in queries:
                    pair = {
                        'query': query,
                        'positive_chunk_id': chunk_id,
                        'hard_negatives': hard_negatives[:3]  # Use up to 3
                    }
                    
                    pairs.append(pair)
                    f.write(json.dumps(pair) + '\n')
                    f.flush()
    
    print(f"\n✓ Generated {len(pairs)} training pairs")
    print(f"  Failed chunks (no hard negatives): {failed_chunks}")
    print(f"  Average queries per chunk: {len(pairs) / (total_chunks - failed_chunks):.2f}")
    
    return len(pairs)


def main():
    parser = argparse.ArgumentParser(description="Generate better training pairs")
    parser.add_argument('--corpus', default='data/corpus.jsonl', help='Corpus file')
    parser.add_argument('--output', default='data/train_pairs_v2.jsonl', help='Output pairs file')
    parser.add_argument('--model', default='gemini-3-flash-preview:cloud', help='Ollama model')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for processing')
    
    args = parser.parse_args()
    
    print("="*70)
    print("GENERATING BETTER TRAINING PAIRS")
    print("="*70)
    print(f"Corpus: {args.corpus}")
    print(f"Output: {args.output}")
    print(f"Model: {args.model}")
    print("="*70)
    print()
    
    # Load corpus
    corpus = load_corpus(args.corpus)
    print(f"Loaded {len(corpus)} chunks")
    print()
    
    # Generate pairs
    num_pairs = generate_pairs(corpus, args.output, args.model, args.batch_size)
    
    print()
    print("="*70)
    print("COMPLETE!")
    print("="*70)
    print(f"✓ Generated {num_pairs} training pairs")
    print(f"  File: {args.output}")
    print()
    print("Next: Train with these pairs:")
    print(f"  python train_embeddings_final.py --train_pairs {args.output}")


if __name__ == "__main__":
    main()
