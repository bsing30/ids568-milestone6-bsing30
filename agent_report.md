# Agent Controller Report

## Tool Selection Policy
The agent uses an LLM planner step to decide tool usage (`llm_planner` trace step). The planner chooses whether retrieval is needed and whether evidence extraction should run before summarization. If local model planning fails, a deterministic fallback policy is used and explicitly logged.

## Retrieval Integration
- Retriever tool is reused from Part 1 (`RAGPipeline.retrieve`).
- Retrieval returns structured records (doc ID, chunk ID, text, score).
- `extract_facts_tool` converts retrieved chunks into source-attributed evidence bullets.
- Summarizer consumes extracted evidence (or raw task text when retrieval is skipped).

## 10 Evaluation Tasks (overview)
1. Deductible and coinsurance lookup for Policy A  
2. Preventive care coverage check  
3. Claim document requirements extraction  
4. Claim processing timeline summary  
5. Urgent claim escalation behavior  
6. Appeal deadline and packet requirements  
7. Emergency-service coverage question  
8. Combined task: coverage + claim SLA  
9. Out-of-scope task to test abstention  
10. Multi-question task requiring merged evidence

## Performance and Observability
- Each task stores planner, retrieval, extraction, and summarization steps with reasons and latencies.
- Tool selection is LLM-driven for final workflow and transparent in traces.
- Trace artifacts are saved under `agent_traces/`.

## Failure Analysis
- Planner can occasionally over-call retrieval for procedural tasks that do not need external evidence.
- Retrieval limitations on sparse or low-frequency details can reduce final answer completeness.
- Summarizer can compress too aggressively; long context may lose secondary facts.

## Model Quality/Latency Tradeoffs
- 7B class models provide acceptable quality with lower latency for local runs.
- 14B class generally improves instruction-following and grounding at higher latency/cost.
- Better grounding typically comes from retrieval quality and strict prompting rather than model size alone.
