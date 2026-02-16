# AI‑Stock‑Ranks

This repository contains a framework for ranking publicly listed stocks using a listwise, relative‑comparison protocol and a large language model (LLM).  The design follows a first‑principles "batch and average" methodology: tickers are grouped into small batches, ranked relative to each other by an LLM, and aggregated over multiple trials to produce stable, context‑aware ratings.

The code in `src/` implements the core logic for creating random batches, calling an LLM to rank each batch, averaging the results and producing a final ordered list of tickers.  The `docs/` folder defines the product requirements, data contracts, prompts and evaluation procedures.  The `prompts/` directory holds the LLM prompts used during ranking and critique phases.

Use this framework as a starting point for building autonomous agents that score equities relative to their peers.  The project intentionally separates the execution harness (no‑code workflow orchestrator) from the system of record (this repository) to maximise reproducibility and allow AI agents to iterate on the specification without changing business logic.