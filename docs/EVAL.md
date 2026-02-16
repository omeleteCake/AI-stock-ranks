# Evaluation and Stability Criteria

Evaluation is critical for ensuring that the listwise batching protocol produces reliable rankings.  This document outlines the checks that must be performed on each run.

## Format Validity

* **JSON correctness**: Each batch response must parse as valid JSON.  Strict schema validation must be applied to ensure all required fields exist and no extraneous fields are present.
* **Complete coverage**: Every ticker from a batch must appear exactly once in the corresponding `rankings` array.
* **Numeric types**: `rank` must be an integer; `confidence` must be a float in `[0,1]`.

## Coverage

* **Universe coverage**: The aggregated output must include every ticker from the input dataset exactly once.  If a ticker is excluded, an `exclusion_reason` must be provided.

## Stability

* **Rank variance**: Compute the standard deviation of rank for each ticker across all trials.  For a run to be acceptable, `std_rank` for each ticker must not exceed `batch_size / 4`.  This threshold can be tuned in practice.
* **Top‑k consistency**: For each ticker, compute the fraction of trials in which it appears in the top 20 percent of its batch.  Tickers with `top20_rate` below 0.5 should be excluded unless there is a strong rationale.

## Reproducibility

Given the same seed, input data and model version, repeated runs must produce identical aggregated results.  Seeds must be logged and saved with the run artefacts.

## Golden sets

The `eval/golden_sets/` directory contains sample datasets against which the system should be evaluated regularly.  Pass/fail checklists for these datasets should be automated in the orchestrator.  If modifications to prompts or aggregation logic cause a regression on a golden set, the changes must be revisited.