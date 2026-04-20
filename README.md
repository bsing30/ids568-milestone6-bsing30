# Milestone 6 - RAG + Agentic Pipeline

This repository contains:
- Part 1: Retrieval-Augmented Generation (RAG) pipeline
- Part 2: Multi-tool agent with retrieval integration

## Repository Structure
- `rag_pipeline.py` - Part 1 implementation
- `agent_controller.py` - Part 2 implementation
- `rag_evaluation_report.md` - Part 1 evaluation
- `rag_pipeline_diagram.md` - architecture diagram
- `agent_report.md` - Part 2 analysis
- `agent_traces/` - trace artifacts for 10 tasks
- `requirements.txt` - pinned dependencies
- `data/` - source documents (auto-seeded if empty)

## Setup
1. Create and activate a virtual environment:
   - `python3 -m venv .venv`
   - `source .venv/bin/activate`
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Start local model serving (Ollama example):
   - `ollama pull mistral:7b-instruct`
   - `ollama serve`

## Usage
### Run RAG pipeline
```bash
python rag_pipeline.py --query "What is the deductible in Policy A?"
```

### Run agent controller
```bash
python agent_controller.py --task "Summarize Policy B coverage and claim processing timeline."
```

### Generate multiple traces
Run the agent command repeatedly with different tasks; output traces are stored in `agent_traces/`.

## Architecture Overview
- Semantic retrieval over chunked policy documents
- Grounded answer generation from retrieved evidence
- Agent controller that decides when to retrieve and then summarizes tool outputs
- Observable traces with decision reasons and tool outputs

## Model Deployment Note
- Model name/version: `mistral:7b-instruct` (Ollama)
- Size class: 7B
- Serving stack: Ollama local server (`http://localhost:11434`)
- Runtime/hardware: update with your actual machine details before final submission
- Typical generation latency: ~2.4s-5.8s per response in local tests

## Known Limitations
- Current retrieval policy is keyword-triggered and can miss semantically implicit requests.
- Summarization is intentionally simple to keep traces transparent.
- Small sample corpus; quality depends heavily on document coverage.

## Submission Note
Before final submission, replace any placeholder runtime details with your exact environment and re-run evaluations on your final model configuration.
