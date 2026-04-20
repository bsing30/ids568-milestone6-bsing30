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

import requests

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
        self.llm_model = self.rag.llm_model
        self.ollama_url = self.rag.ollama_url

    def summarize_tool(self, text: str) -> Dict[str, Any]:
        # Lightweight deterministic summarizer tool for transparent behavior.
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        summary = " ".join(sentences[:2]) if sentences and sentences[0] else text[:250]
        return {"summary": summary}

    def retrieval_tool(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        hits = self.rag.retrieve(query, top_k=top_k)
        return {"hits": hits}

    def extract_facts_tool(self, retrieved_hits: List[Dict[str, Any]]) -> Dict[str, Any]:
        facts = []
        for hit in retrieved_hits:
            sentences = re.split(r"(?<=[.!?])\s+", hit["text"])
            for s in sentences:
                cleaned = s.strip()
                if cleaned and len(cleaned) > 20:
                    facts.append({"source": f"{hit['doc_id']}#{hit['chunk_id']}", "fact": cleaned})
        return {"facts": facts[:8]}

    def llm_decide_tools(self, task: str) -> Dict[str, Any]:
        prompt = (
            "You are an agent planner. Decide which tools are needed for the task.\n"
            "Available tools:\n"
            "1) retrieval_tool: fetch factual policy/claims evidence.\n"
            "2) extract_facts_tool: extract concise evidence bullets from retrieval.\n"
            "3) summarize_tool: produce final short response.\n\n"
            "Return strict JSON with keys:\n"
            "need_retrieval (bool), need_fact_extraction (bool), reason (string).\n"
            f"Task: {task}"
        )
        payload = {
            "model": self.llm_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0},
        }
        try:
            resp = requests.post(self.ollama_url, json=payload, timeout=90)
            resp.raise_for_status()
            raw = resp.json().get("response", "").strip()
            start = raw.find("{")
            end = raw.rfind("}")
            if start >= 0 and end > start:
                parsed = json.loads(raw[start : end + 1])
                return {
                    "need_retrieval": bool(parsed.get("need_retrieval", True)),
                    "need_fact_extraction": bool(parsed.get("need_fact_extraction", True)),
                    "reason": str(parsed.get("reason", "LLM-selected tools based on task intent.")),
                    "planner_mode": "llm",
                }
        except Exception:
            pass
        return {
            "need_retrieval": True,
            "need_fact_extraction": True,
            "reason": "Fallback planner: retrieval-first for factual insurance tasks.",
            "planner_mode": "fallback",
        }

    def run(self, task: str) -> Dict[str, Any]:
        traces: List[TraceStep] = []
        step_id = 1
        context_blob = ""
        planner_start = time.perf_counter()
        plan = self.llm_decide_tools(task)
        planner_end = time.perf_counter()
        traces.append(
            TraceStep(
                step=step_id,
                decision="Plan tool sequence",
                reason=plan["reason"],
                tool="llm_planner",
                tool_input={"task": task},
                tool_output=plan,
                latency_s=round(planner_end - planner_start, 4),
            )
        )
        step_id += 1

        retrieval_hits: List[Dict[str, Any]] = []
        if plan["need_retrieval"]:
            t0 = time.perf_counter()
            retrieval = self.retrieval_tool(task, top_k=3)
            t1 = time.perf_counter()
            retrieval_hits = retrieval["hits"]
            context_blob = "\n".join([h["text"] for h in retrieval_hits])
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

        if plan["need_fact_extraction"] and retrieval_hits:
            t0 = time.perf_counter()
            facts = self.extract_facts_tool(retrieval_hits)
            t1 = time.perf_counter()
            fact_lines = [f"- {f['fact']} [{f['source']}]" for f in facts["facts"]]
            context_blob = "\n".join(fact_lines) if fact_lines else context_blob
            traces.append(
                TraceStep(
                    step=step_id,
                    decision="Extract evidence facts",
                    reason="Improve transparency before final summarization.",
                    tool="extract_facts_tool",
                    tool_input={"hit_count": len(retrieval_hits)},
                    tool_output=facts,
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
