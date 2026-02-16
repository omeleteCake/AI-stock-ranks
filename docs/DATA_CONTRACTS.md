# Data Contracts

This document defines the strict schemas for data structures used in the AI Stock Ranks system.  Agents and orchestrators must conform to these contracts.  Schemas are expressed informally in this document and should be enforced in code where applicable.

## TickerRow

Represents a single stock and its metrics.

| Field          | Type    | Description                              |
|---------------|---------|------------------------------------------|
| `ticker`      | string  | Stock symbol.                             |
| `sector`      | string  | Sector classification.                   |
| `market_cap`  | float   | Market capitalisation in USD.            |
| `pe`          | float   | Price/earnings ratio.                    |
| `ev_ebitda`   | float   | Enterprise value / EBITDA ratio.         |
| `debt_equity` | float   | Debt to equity ratio.                    |
| `rev_growth`  | float   | Trailing twelve‑month revenue growth.    |
| `fcf_margin`  | float   | Free cash flow margin.                   |
| `price_momentum` | float | Recent price momentum indicator.         |
| `volatility`  | float   | Price volatility measure.                |
| `timestamp`   | string  | ISO 8601 timestamp for the data snapshot.|
| `source`      | string  | Data source identifier.                  |

## BatchRequest

Request sent to the LLM when ranking a batch of tickers.  Contains minimal metadata and the list of tickers.

| Field       | Type    | Description                                     |
|------------|---------|-------------------------------------------------|
| `run_id`   | string  | Unique identifier for the overall ranking run.   |
| `trial_id` | integer | Index of the current trial (0‑indexed).          |
| `batch_id` | integer | Index of the batch within the trial (0‑indexed). |
| `tickers`  | array   | Array of `TickerRow` objects.                    |

## BatchResponse

Response returned by the LLM for a single batch.  Must be valid JSON with the following fields:

| Field            | Type                  | Description                                          |
|-----------------|-----------------------|------------------------------------------------------|
| `run_id`        | string                | Echoed from the request.                             |
| `trial_id`      | integer               | Echoed from the request.                             |
| `batch_id`      | integer               | Echoed from the request.                             |
| `rankings`      | array of objects      | List of ranked tickers.                              |
| `rationale_bullets` | array of strings | Optional bullet points explaining the ranking.       |
| `red_flags`     | array of strings      | Optional list of red flags.                          |

Each element of the `rankings` array must be an object with:

| Field      | Type   | Description                                            |
|-----------|--------|--------------------------------------------------------|
| `ticker`  | string | Ticker symbol.                                         |
| `rank`    | integer| Position in the ranking (1 is best, N is worst).       |
| `confidence` | float | Confidence score between 0 and 1.                     |

The `rankings` array must include all tickers from the request exactly once.  Additional fields are not allowed.

## AggregateOutput

The final output after aggregating multiple trials.

| Field            | Type      | Description                                             |
|-----------------|-----------|---------------------------------------------------------|
| `ticker`        | string    | Ticker symbol.                                          |
| `mean_rank`     | float     | Mean rank across all trials.                            |
| `std_rank`      | float     | Standard deviation of rank across trials.              |
| `top20_rate`    | float     | Fraction of trials in which the ticker ranked in top 20 percent. |
| `notes`         | string    | Concise notes summarising rationale or red flags.       |
| `exclusion_reason` | string | If excluded, reason for exclusion; otherwise empty.      |

The aggregated list must be sorted in ascending order of `mean_rank`.  Each ticker appears exactly once.  Removed tickers must still appear with an `exclusion_reason`.