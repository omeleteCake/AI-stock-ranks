# Architecture

This document describes the layered architecture used for the AI Stock Ranks system.  The design enforces strict separation between data ingestion, LLM interaction and aggregation logic.  Adhering to these layers prevents architectural drift and enables autonomous agents to modify prompts and workflows without compromising the core logic.

## Layers

1. **Data**

   *Input:* CSV or table conforming to the `TickerRow` schema (see `docs/DATA_CONTRACTS.md`).

   *Output:* A list of `TickerRow` objects passed to the batching layer.  Data cleansing and type casting occur here.

2. **Batching**

   *Responsibility:* Randomly shuffle the ticker universe with a deterministic seed and slice into batches of fixed size.  Each batch is represented as a `BatchRequest` containing metadata (`run_id`, `trial_id`, `batch_id`) and the subset of tickers.

3. **LLM Ranking**

   *Responsibility:* For each `BatchRequest`, call the language model with the listwise prompt defined in `prompts/rank_listwise_v1.md`.  The model returns a `BatchResponse` JSON object ranking the tickers, providing confidence scores and optional red flags.  The LLM call is the only component that can return non‑deterministic output.

4. **Aggregation**

   *Responsibility:* Collect `BatchResponse` objects for all trials and compute aggregated metrics per ticker: mean rank, standard deviation of rank, appearance frequency in top 20 percent, and notes.  Generate the final `AggregateOutput` list sorted by mean rank.

5. **Evaluation and Filtering**

   *Responsibility:* Apply stability checks defined in `docs/EVAL.md`.  Filter out tickers failing top‑20 percent consistency.  Reject runs that do not satisfy variance thresholds.

6. **Persistence**

   *Responsibility:* Write run artefacts (raw batch responses, aggregated output) to `outputs/runs/` with a unique timestamp.  Artefacts are not tracked in version control but provide traceability for audits.

## Dependency rules

- Data → Batching → LLM Ranking → Aggregation → Evaluation.  Dependencies must only flow in this direction.  The LLM should never read files from the repository; it receives only the batch table via the orchestrator.
- Agents modifying prompts or evaluation logic must update the relevant documents in `docs/` and regenerate the orchestrator export.  They must not alter the core Python modules in `src/` without explicit instruction.

## Orchestrator integration

The no‑code orchestrator (e.g. Gumloop or Relevance AI) implements the runtime workflow.  It reads the repository prompts and data contracts, produces batches, calls the LLM, aggregates results and writes outputs.  The orchestrator’s export JSON should be stored under `workflows/<tool>/export.json` for reproducibility.