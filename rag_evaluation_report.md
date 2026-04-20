# RAG Evaluation Report

## Setup
- Corpus: 8 insurance-domain documents in `data/` (policy, claims, appeals, pharmacy, network, pre-auth, billing)
- Chunking: 700 chars with 120 overlap
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Index: FAISS (`IndexFlatIP`, normalized embeddings)
- Generator: `mistral:7b-instruct` served with Ollama (local)

## Retrieval Accuracy (10 handcrafted queries)

| Query ID | Query (short) | Relevant retrieved? | Precision@3 | Notes |
|---|---|---|---|---|
| Q1 | policy A deductible | Yes | 0.67 | Correct doc ranked first |
| Q2 | policy A coinsurance | Yes | 0.67 | Correct value retrieved |
| Q3 | preventive care copay | Yes | 0.67 | In-network condition captured |
| Q4 | claim processing time | Yes | 0.67 | 7-10 days retrieved |
| Q5 | urgent claim turnaround | Yes | 0.33 | Mentioned but lower-ranked |
| Q6 | appeal timeline | Yes | 0.67 | 30-day timeline retrieved |
| Q7 | appeal package docs | Yes | 0.67 | Supporting docs identified |
| Q8 | emergency services coverage | Yes | 0.67 | Policy A source used |
| Q9 | unsupported dental benefits | No | 0.00 | Out-of-scope (expected) |
| Q10 | docs needed for claims | Yes | 0.67 | Member ID/invoice/notes |

Average Precision@3: **0.57**

## Chunking Strategy Comparison

| Config | Rationale | Avg Precision@3 (10 queries) | Observation |
|---|---|---|---|
| 256 / 64 overlap | Higher granularity | 0.50 | Better specificity, more context fragmentation |
| 512 / 96 overlap | Balanced candidate | 0.54 | Good compromise but some clause splitting |
| **700 / 120 overlap (chosen)** | Preserve policy clauses and timelines | **0.57** | Best grounding consistency in generated answers |
| 1024 / 128 overlap | Long context chunks | 0.51 | More noise, weaker ranking precision |

## Grounding Analysis
- Most answers remained grounded when retrieval returned at least one highly relevant chunk.
- Answers were strongest for factual fields (timelines, deductible, required documents).
- Out-of-scope questions were correctly answered with insufficient-information behavior when prompt constraints were followed.

## Hallucination / Failure Cases
- Case A (Q5): retrieval included partially relevant chunk but not optimal ranking; answer omitted urgency detail in one run.
- Case B (Q9): without strict prompt constraints, model attempted generic insurance advice.

## Error Attribution
- Retrieval failures: ranking misses for lower-frequency details (urgent review path).
- Generation failures: occasional over-generalization when context is weak.
- Combined issues: insufficiently strict prompt amplifies retrieval noise.

## Latency (representative)
- Retrieval latency: 0.01s - 0.06s
- Generation latency: 2.4s - 5.8s (hardware/model dependent)
- End-to-end latency: 2.5s - 5.9s

## Design Decisions
- 700/120 chunk config retained policy clause continuity while limiting semantic drift; tested against 256/64, 512/96, and 1024/128 alternatives.
- FAISS selected for local performance and low setup overhead.
- MiniLM chosen for speed/quality balance on a laptop environment.
