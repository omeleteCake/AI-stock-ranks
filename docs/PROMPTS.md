# Prompt Templates

This document describes the prompts used by the AI Stock Ranks system.  Prompts are separated into individual files in the `prompts/` directory for ease of use by the orchestrator.  They follow the listwise ranking paradigm: the model must order a list of tickers relative to one another and output structured JSON only.

## Rank Listwise (prompts/rank_listwise_v1.md)

This system prompt instructs the model to act as a sector‑agnostic equity analyst.  The user message will contain a table of tickers and their metrics.  The model must rank the tickers from most to least attractive based on value, growth and risk.  It should output a JSON object with a `rankings` array, `rationale_bullets` (up to 5 bullets) and optional `red_flags`.

Key constraints:

- Do **not** invent or omit tickers.  All tickers must appear in the output exactly once.
- Provide a numeric `rank` for each ticker starting at 1.
- Provide a `confidence` score between 0 and 1 for each ranking entry.
- Write concise bullet points focusing on differences between the tickers.
- Output must be valid JSON conforming to `BatchResponse`.

## Critique (prompts/critique_v1.md)

The critic prompt is applied after the model produces its ranking.  It instructs a second LLM to review the `BatchResponse` and flag rule violations (e.g. missing tickers, invalid JSON).  The critic returns a short list of issues or an empty list if the response is valid.

## Normalize Output (prompts/normalize_output_v1.md)

This prompt instructs the model to take an arbitrary ranking result (which may include narrative or incorrect fields) and produce a clean JSON object conforming to the `BatchResponse` schema.  It is used as a post‑processor to salvage partially valid LLM responses.