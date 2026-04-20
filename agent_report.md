# Agent Controller Report

## Tool Selection Policy
The agent uses a keyword-triggered policy for retrieval-first behavior on tasks involving policy coverage, claims, appeals, and timelines. After retrieval, it runs a summarization tool to produce concise output. This keeps decision points observable and deterministic.

## Retrieval Integration
- Retriever tool is reused from Part 1 (`RAGPipeline.retrieve`).
- Retrieval returns structured records (doc ID, chunk ID, text, score).
- Summarizer consumes retrieved context (or original task when retrieval is skipped).

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
- Each task stores a trace with explicit decisions, reasons, tool calls, and outputs.
- Retrieval selection is explainable and reproducible.
- Trace artifacts are saved under `agent_traces/`.

## Failure Analysis
- Wrong-tool risk for vague tasks that do not include trigger terms.
- Retrieval limitations on sparse or low-frequency details can reduce final answer completeness.
- Summarizer can compress too aggressively; long context may lose secondary facts.

## Model Quality/Latency Tradeoffs
- 7B class models provide acceptable quality with lower latency for local runs.
- 14B class generally improves instruction-following and grounding at higher latency/cost.
- Better grounding typically comes from retrieval quality and strict prompting rather than model size alone.
