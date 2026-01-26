# Model Evaluation Results - re-diligence-embeddings-v2

**Date:** January 25, 2026  
**Model:** `models/re-diligence-embeddings-v2`  
**Training:** 5 epochs, final loss: 0.0482

## 📊 Evaluation Metrics

### Quantitative Results
- **MRR@10:** 0.0698 (6.98%)
- **Recall@10:** 0.0392 (3.92%)
- **NDCG@10:** 0.0453 (4.53%)

### Evaluation Dataset
- **Queries:** 500 evaluation queries
- **Corpus:** 3,010 chunks
- **Qrels:** 1,935 relevance judgments

## 📈 Model Comparison: v1 vs v2

| Metric | v1 Model | v2 Model | Change |
|--------|----------|----------|--------|
| **MRR@10** | 0.0715 (7.15%) | 0.0698 (6.98%) | -0.17% ⬇️ |
| **Recall@10** | 0.0335 (3.35%) | 0.0392 (3.92%) | +0.57% ⬆️ |
| **NDCG@10** | 0.0456 (4.56%) | 0.0453 (4.53%) | -0.03% ⬇️ |

### Analysis
- **Recall Improvement:** v2 shows a **17% relative improvement** in Recall@10 (3.35% → 3.92%), meaning it finds more relevant documents in the top 10 results
- **MRR Slight Decrease:** v2 has a small decrease in MRR, suggesting the first relevant result may appear slightly later
- **NDCG Stable:** NDCG is nearly identical, indicating similar ranking quality overall

**Overall:** v2 model shows improved recall (finds more relevant docs) with a slight trade-off in MRR. The improvements are modest, suggesting both models are performing similarly on this evaluation set.

## 🔍 Example Query Results

### Query 1: "What is the occupancy rate of the property?"
- **Top Result:** Similarity 0.6835 - Zoning information chunk
- **Rank 2:** Similarity 0.6387 - Zoning report chunk
- **Rank 3:** Similarity 0.6105 - Zoning report with expansion details

### Query 2: "What are the major tenants and their lease terms?"
- **Top Result:** Similarity 0.8694 - Lease abstract with tenant details ✅
- **Rank 2:** Similarity 0.8353 - Lease abstract summary ✅
- **Rank 3:** Similarity 0.8283 - CIM with lease structures ✅

### Query 3: "What environmental issues were found during due diligence?"
- **Top Result:** Similarity 0.8861 - Phase I ESA report ✅
- **Rank 2:** Similarity 0.8280 - Environmental diligence report ✅
- **Rank 3:** Similarity 0.8155 - Phase I evaluation ✅

### Query 4: "What is the property's cap rate and NOI?"
- **Top Result:** Similarity 0.7884 - Appraisal summary ✅
- **Rank 2:** Similarity 0.6671 - Appraisal with market value ✅
- **Rank 3:** Similarity 0.6139 - Appraisal with cap rate ✅

### Query 5: "What zoning restrictions apply to this property?"
- **Top Result:** Similarity 0.8719 - Zoning report ✅
- **Rank 2:** Similarity 0.8648 - Zoning diligence report ✅
- **Rank 3:** Similarity 0.8632 - Zoning status information ✅

## 📈 Observations

### Strengths
1. **High Semantic Similarity Scores:** Top results consistently show similarity scores above 0.60-0.80, indicating good semantic understanding
2. **Domain-Specific Matching:** The model correctly identifies relevant document types (lease abstracts, Phase I reports, appraisals, zoning reports)
3. **Relevant Content:** Example queries show the model retrieves contextually appropriate chunks

### Areas for Improvement
1. **Recall Metrics:** Low Recall@10 (3.92%) suggests the model may miss some relevant documents
2. **MRR Score:** MRR@10 of 6.98% indicates the first relevant result often appears beyond rank 1
3. **NDCG Score:** NDCG@10 of 4.53% suggests ranking quality could be improved

## 🔧 Potential Improvements

1. **Training Data Quality:**
   - Review training pairs for better hard negatives
   - Increase diversity in query-document pairs
   - Add more domain-specific examples

2. **Model Architecture:**
   - Consider larger model size if computational resources allow
   - Experiment with different pooling strategies
   - Try different loss functions (e.g., contrastive loss variants)

3. **Evaluation:**
   - Review qrels for potential annotation issues
   - Check if evaluation queries match training distribution
   - Consider domain-specific evaluation metrics

4. **Fine-tuning:**
   - Continue training with more epochs if loss is still decreasing
   - Use learning rate scheduling
   - Experiment with different batch sizes

## 📝 Next Steps

1. ✅ **Model Testing** - Complete
2. ✅ **Evaluation** - Complete
3. **Compare with v1 model** - Run same evaluation on previous version
4. **Error Analysis** - Review queries with low scores
5. **Fine-tuning** - Apply improvements and retrain
6. **Deployment** - Prepare for HuggingFace or production use

## 🎯 Performance Context

These metrics should be interpreted in the context of:
- **Domain specificity:** Real estate due diligence is a specialized domain
- **Query complexity:** Many queries require understanding of financial/legal terminology
- **Corpus size:** 3,010 chunks is a moderate-sized corpus
- **Evaluation set:** 500 queries with 1,935 relevance judgments

The model shows promise with high similarity scores on example queries, but ranking and recall can be improved through further training and data refinement.
