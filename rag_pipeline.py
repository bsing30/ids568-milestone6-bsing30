"""
Milestone 6 - Part 1: RAG Pipeline

Usage:
  python rag_pipeline.py --query "What is covered in policy A?"

This script:
1) Ingests text documents from ./data
2) Chunks them with overlap
3) Embeds chunks using sentence-transformers
4) Indexes with FAISS
5) Retrieves top-k passages
6) Generates grounded response through local Ollama model
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer


@dataclass
class Chunk:
    doc_id: str
    chunk_id: int
    text: str


class RAGPipeline:
    def __init__(
        self,
        data_dir: str = "data",
        chunk_size: int = 700,
        chunk_overlap: int = 120,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: str = "mistral:7b-instruct",
        ollama_url: str = "http://localhost:11434/api/generate",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model_name = embedding_model
        self.llm_model = llm_model
        self.ollama_url = ollama_url

        self.embedder = SentenceTransformer(self.embedding_model_name)
        self.chunks: List[Chunk] = []
        self.index = None
        self.embedding_matrix = None

    def ingest_documents(self) -> List[Tuple[str, str]]:
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self._seed_sample_docs()
        docs = []
        for p in sorted(self.data_dir.glob("*.txt")):
            docs.append((p.name, p.read_text(encoding="utf-8")))
        return docs

    def _seed_sample_docs(self) -> None:
        sample_docs = {
            "policy_a.txt": (
                "Policy A covers inpatient hospitalization, emergency services, "
                "and basic diagnostics. It has a $500 annual deductible and "
                "20% coinsurance after deductible."
            ),
            "policy_b.txt": (
                "Policy B includes outpatient specialist visits, mental health "
                "consultations, and preventive care. Preventive care has zero "
                "copay when in-network."
            ),
            "claims_process.txt": (
                "Claim submission requires member ID, invoice, and provider notes. "
                "Claims are typically processed in 7-10 business days. Urgent claims "
                "can be escalated for 48-hour review."
            ),
            "appeals.txt": (
                "Denied claims can be appealed within 30 days. Appeal packets should "
                "include denial letter, supporting medical documents, and physician "
                "statement."
            ),
        }
        for name, content in sample_docs.items():
            (self.data_dir / name).write_text(content, encoding="utf-8")

    def chunk_documents(self, docs: List[Tuple[str, str]]) -> List[Chunk]:
        chunks: List[Chunk] = []
        for doc_id, text in docs:
            start = 0
            cid = 0
            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                piece = text[start:end].strip()
                if piece:
                    chunks.append(Chunk(doc_id=doc_id, chunk_id=cid, text=piece))
                    cid += 1
                if end == len(text):
                    break
                start = max(0, end - self.chunk_overlap)
        self.chunks = chunks
        return chunks

    def build_index(self) -> None:
        if not self.chunks:
            raise ValueError("No chunks available. Run chunk_documents first.")
        texts = [c.text for c in self.chunks]
        embeddings = self.embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        self.embedding_matrix = embeddings.astype("float32")
        dim = self.embedding_matrix.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embedding_matrix)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        if self.index is None:
            raise ValueError("Index is not built. Run build_index first.")
        q = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        scores, indices = self.index.search(q, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            c = self.chunks[int(idx)]
            results.append(
                {
                    "doc_id": c.doc_id,
                    "chunk_id": c.chunk_id,
                    "text": c.text,
                    "score": float(score),
                }
            )
        return results

    def generate_answer(self, query: str, retrieved: List[Dict], temperature: float = 0.1) -> str:
        context = "\n\n".join(
            [f"[{i+1}] ({r['doc_id']}#{r['chunk_id']}) {r['text']}" for i, r in enumerate(retrieved)]
        )
        prompt = (
            "You are an insurance support assistant. Answer ONLY with information "
            "grounded in the provided context. If missing, say you do not have enough information.\n\n"
            f"Question: {query}\n\n"
            f"Context:\n{context}\n\n"
            "Return a concise answer and list source ids used."
        )
        payload = {
            "model": self.llm_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        resp = requests.post(self.ollama_url, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()

    def run_query(self, query: str, top_k: int = 3) -> Dict:
        t0 = time.perf_counter()
        retrieved = self.retrieve(query, top_k=top_k)
        t1 = time.perf_counter()
        answer = self.generate_answer(query, retrieved)
        t2 = time.perf_counter()
        return {
            "query": query,
            "retrieval_latency_s": round(t1 - t0, 4),
            "generation_latency_s": round(t2 - t1, 4),
            "end_to_end_latency_s": round(t2 - t0, 4),
            "retrieved": retrieved,
            "answer": answer,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG pipeline query.")
    parser.add_argument("--query", type=str, required=True, help="User query.")
    parser.add_argument("--top_k", type=int, default=3, help="Top-k retrieval.")
    parser.add_argument("--output", type=str, default="", help="Optional output JSON file.")
    args = parser.parse_args()

    rag = RAGPipeline()
    docs = rag.ingest_documents()
    rag.chunk_documents(docs)
    rag.build_index()
    result = rag.run_query(args.query, top_k=args.top_k)

    print(json.dumps(result, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
