# Product Requirements Document (PRD)

## Objective

Build a system that ranks a universe of publicly listed equities using a listwise, relative‑comparison protocol.  The system should ingest a list of tickers and associated financial metrics, produce small random batches, obtain relative rankings from a language model for each batch and aggregate these rankings over multiple trials.  The result is a stable, context‑aware ordering of the input tickers.

## Inputs

- **Ticker universe**: a CSV or table containing the following columns per row:
  - `ticker` – stock symbol (string)
  - `sector` – sector classification (string)
  - `market_cap` – market capitalisation (float, in USD)
  - `pe` – price/earnings ratio (float)
  - `ev_ebitda` – enterprise value / EBITDA ratio (float)
  - `debt_equity` – debt to equity ratio (float)
  - `rev_growth` – trailing twelve‑month revenue growth rate (float)
  - `fcf_margin` – free cash flow margin (float)
  - `price_momentum` – recent price momentum indicator (float)
  - `volatility` – price volatility measure (float)
  - `timestamp` – data timestamp (ISO 8601)
  - `source` – data source identifier (string)

- **Batch size**: number of tickers per batch (default 10–20).
- **Trials**: number of random batching trials (default 5–10).

## Process

1. **Random shuffle** the ticker universe with a deterministic seed and slice into batches of size *B*.
2. **Listwise ranking**: For each batch, send the batch table to an LLM with the prompt template defined in `prompts/rank_listwise_v1.md`.  The model must return a JSON object listing the tickers in descending order of relative attractiveness along with confidence scores and optional red flag bullets.
3. **Repeat** steps 1–2 for *T* trials with different seeds to average out batch bias.
4. **Aggregate**: Compute the mean rank and standard deviation for each ticker across all trials.  Derive a stability metric (e.g. rank standard deviation) and top‑k rate (how often the ticker appears in the top 20 percent of its batch).  Produce a final ordered list by ascending mean rank.
5. **Filter**: Exclude tickers that never appear in the top 20 percent across all trials.  Provide an exclusion reason in the final output.
6. **Output**: Write a table conforming to the `AggregateOutput` schema.

## Non‑negotiable rules

- The LLM must not generate new tickers or drop any tickers provided in the batch input.
- The LLM’s output must be valid JSON with fields exactly matching the data contract.
- All numeric columns must remain numeric; no conversion to strings except where specified.
- The aggregation logic must be deterministic given the same seed and input data.
- Narrative commentary from the LLM is prohibited; only structured JSON is accepted.

## Failure modes

Reject any run where:

1. The model returns malformed JSON.
2. A ticker from the input is missing from any batch response.
3. The aggregated ranking includes duplicates or omissions.
4. Variance thresholds (defined in `docs/EVAL.md`) are exceeded.
5. The data timestamp is inconsistent across rows.