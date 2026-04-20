"""
Milestone 6 - Part 2: Multi-tool Agent Controller

Usage:
  python agent_controller.py --task "Summarize what policy B covers and mention claim turnaround time."
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List

from rag_pipeline import RAGPipeline


@dataclass
class TraceStep:
    step: int
    decision: str
    reason: str
    tool: str
    tool_input: Dict[str, Any]
    tool_output: Dict[str, Any]
    latency_s: float


class AgentController:
    def __init__(self) -> None:
        self.rag = RAGPipeline()
        docs = self.rag.ingest_documents()
        self.rag.chunk_documents(docs)
        self.rag.build_index()

    def summarize_tool(self, text: str) -> Dict[str, Any]:
        # Lightweight deterministic summarizer tool for transparent behavior.
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        summary = " ".join(sentences[:2]) if sentences and sentences[0] else text[:250]
        return {"summary": summary}

    def retrieval_tool(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        hits = self.rag.retrieve(query, top_k=top_k)
        return {"hits": hits}

    def should_retrieve(self, task: str) -> bool:
        trigger_words = ["policy", "claim", "appeal", "coverage", "deductible", "document", "timeline"]
        return any(w in task.lower() for w in trigger_words)

    def run(self, task: str) -> Dict[str, Any]:
        traces: List[TraceStep] = []
        step_id = 1
        context_blob = ""

        if self.should_retrieve(task):
            t0 = time.perf_counter()
            retrieval = self.retrieval_tool(task, top_k=3)
            t1 = time.perf_counter()
            context_blob = "\n".join([h["text"] for h in retrieval["hits"]])
            traces.append(
                TraceStep(
                    step=step_id,
                    decision="Use retrieval tool",
                    reason="Task asks for factual policy/claims knowledge.",
                    tool="retrieval_tool",
                    tool_input={"query": task, "top_k": 3},
                    tool_output=retrieval,
                    latency_s=round(t1 - t0, 4),
                )
            )
            step_id += 1

        t0 = time.perf_counter()
        summary = self.summarize_tool(context_blob or task)
        t1 = time.perf_counter()
        traces.append(
            TraceStep(
                step=step_id,
                decision="Use summarization tool",
                reason="Produce concise final answer from gathered context.",
                tool="summarize_tool",
                tool_input={"text": (context_blob or task)[:500]},
                tool_output=summary,
                latency_s=round(t1 - t0, 4),
            )
        )

        final_answer = summary["summary"]
        return {
            "task": task,
            "final_answer": final_answer,
            "trace": [asdict(t) for t in traces],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-tool agent task.")
    parser.add_argument("--task", type=str, required=True, help="Task for the agent.")
    parser.add_argument("--trace_dir", type=str, default="agent_traces", help="Directory to save trace.")
    args = parser.parse_args()

    agent = AgentController()
    result = agent.run(args.task)

    trace_dir = Path(args.trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_name = f"trace_{int(time.time())}.json"
    trace_path = trace_dir / trace_name
    trace_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(f"Saved trace to: {trace_path}")


if __name__ == "__main__":
    main()
