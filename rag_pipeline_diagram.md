# RAG Pipeline Diagram

```text
User Query
   |
   v
[Retriever]
  - Query embedding
  - Similarity search (FAISS)
   |
   v
Top-k chunks + metadata --------------------.
                                            |
Documents --> Chunker --> Embedder --> Vector Index
                                            |
                                            v
                               [Generator: local open-weight LLM]
                                            |
                                            v
                              Grounded answer + source references
```

## Components
- Chunker: fixed-size chunks with overlap
- Embedder: `sentence-transformers/all-MiniLM-L6-v2`
- Vector store: FAISS IndexFlatIP (cosine-like with normalized embeddings)
- Retriever: top-k semantic nearest neighbors
- Generator: local/self-hosted model via Ollama

## Decision Points
- Chunk size/overlap balance context coherence vs retrieval specificity.
- Top-k value controls context breadth for grounding.
- Prompt policy enforces context-grounded responses.
