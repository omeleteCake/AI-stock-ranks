"""AI Stock Ranks core logic.

This module provides functions to load a universe of stocks with their
financial metrics, split them into random batches, call a language
model to obtain listwise rankings for each batch, validate the
responses and aggregate the results across multiple trials.

The design follows the batch‑and‑average protocol described in the
project documentation.
"""

from __future__ import annotations

import csv
import json
import os
import random
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import openai  # type: ignore
except ImportError:
    openai = None  # optional dependency; fallback to random ranking


@dataclass
class BatchRequest:
    """Represents a request to rank a batch of tickers."""

    run_id: str
    trial_id: int
    batch_id: int
    tickers: pd.DataFrame


@dataclass
class BatchResponse:
    """Represents the model's response for a batch."""

    run_id: str
    trial_id: int
    batch_id: int
    rankings: List[Dict[str, Any]]
    rationale_bullets: List[str] = field(default_factory=list)
    red_flags: List[str] = field(default_factory=list)


def load_tickers(path: str) -> pd.DataFrame:
    """Load the ticker universe from a CSV file.

    The CSV must include the columns defined in the TickerRow schema.  Extra
    columns are ignored but preserved.

    Args:
        path: Path to the CSV file.

    Returns:
        A pandas DataFrame containing the ticker universe.
    """
    df = pd.read_csv(path)
    if 'ticker' not in df.columns:
        raise ValueError("Input CSV must include a 'ticker' column.")
    return df.copy()


def create_batches(df: pd.DataFrame, batch_size: int, seed: int) -> List[pd.DataFrame]:
    """Shuffle the ticker DataFrame and split it into batches.

    Args:
        df: Full DataFrame of tickers.
        batch_size: Desired number of tickers per batch.
        seed: Random seed for reproducibility.

    Returns:
        A list of DataFrames, each representing a batch.
    """
    rng = random.Random(seed)
    indices = list(range(len(df)))
    rng.shuffle(indices)
    batches: List[pd.DataFrame] = []
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i : i + batch_size]
        batches.append(df.iloc[batch_indices].reset_index(drop=True))
    return batches


def format_batch_table(batch: pd.DataFrame) -> str:
    """Format a DataFrame into a markdown table for the LLM prompt."""
    lines = []
    headers = list(batch.columns)
    lines.append('| ' + ' | '.join(headers) + ' |')
    lines.append('|' + '---|' * len(headers))
    for _, row in batch.iterrows():
        values = [str(row[h]) for h in headers]
        lines.append('| ' + ' | '.join(values) + ' |')
    return '\n'.join(lines)


def call_llm_rank_batch(
    batch_request: BatchRequest,
    prompt_path: str,
    model: str = 'gpt-4o',
    temperature: float = 0.0,
) -> BatchResponse:
    """Call a language model to rank a batch.

    This function reads the system prompt from `prompt_path`, constructs the
    user message containing the batch table and metadata, and attempts to call
    OpenAI's API if available.  If the OpenAI package is not installed or no
    API key is present, a fallback random ranking is produced.

    Args:
        batch_request: Metadata and tickers for the batch.
        prompt_path: Path to the system prompt file.
        model: Name of the OpenAI model to call.
        temperature: Sampling temperature.

    Returns:
        A BatchResponse object with rankings and optional notes.
    """
    # Read system prompt
    with open(prompt_path, 'r', encoding='utf-8') as f:
        system_prompt = f.read().strip()

    # Prepare user content: include batch metadata and table
    table_md = format_batch_table(batch_request.tickers)
    user_content = (
        f"run_id: {batch_request.run_id}\n"
        f"trial_id: {batch_request.trial_id}\n"
        f"batch_id: {batch_request.batch_id}\n"
        "Please rank the following stocks:\n\n"
        f"{table_md}"
    )

    # Attempt to call OpenAI API
    api_key = os.environ.get('OPENAI_API_KEY')
    if openai is not None and api_key:
        try:
            openai.api_key = api_key
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=1024,
            )
            content = response.choices[0].message['content']  # type: ignore
            # Extract JSON from the content
            response_json = extract_json_from_text(content)
            validate_batch_response(response_json, batch_request.tickers['ticker'].tolist())
            return BatchResponse(**response_json)
        except Exception:
            # Fall back to random ranking on any failure
            pass

    # Fallback: generate a random ranking
    rng = random.Random(
        hash((batch_request.run_id, batch_request.trial_id, batch_request.batch_id))
    )
    tickers = batch_request.tickers['ticker'].tolist()
    shuffled = tickers[:]
    rng.shuffle(shuffled)
    rankings = []
    for i, ticker in enumerate(shuffled, start=1):
        rankings.append(
            {
                "ticker": ticker,
                "rank": i,
                "confidence": float(rng.random()),
            }
        )
    return BatchResponse(
        run_id=batch_request.run_id,
        trial_id=batch_request.trial_id,
        batch_id=batch_request.batch_id,
        rankings=rankings,
        rationale_bullets=["fallback random ranking"],
        red_flags=[],
    )


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """Extract the first JSON object found within a text string.

    Args:
        text: Raw text potentially containing JSON.

    Returns:
        Parsed JSON object.

    Raises:
        ValueError: If no JSON object can be extracted.
    """
    import re
    pattern = re.compile(r'\{.*?\}', re.DOTALL)
    match = pattern.search(text)
    if not match:
        raise ValueError("No JSON object found in response.")
    json_str = match.group(0)
    return json.loads(json_str)


def validate_batch_response(response: Dict[str, Any], tickers: List[str]) -> None:
    """Validate a batch response against the data contract.

    Args:
        response: Parsed JSON object representing the model response.
        tickers: The list of tickers that were sent in the batch request.

    Raises:
        ValueError: If the response violates the schema.
    """
    required_top_fields = {
        'run_id': str,
        'trial_id': int,
        'batch_id': int,
        'rankings': list,
    }
    for key, typ in required_top_fields.items():
        if key not in response:
            raise ValueError(f"Missing field '{key}' in response.")
        if not isinstance(response[key], typ):
            raise ValueError(f"Field '{key}' must be of type {typ}.")

    rankings = response['rankings']
    if len(rankings) != len(tickers):
        raise ValueError("Number of rankings does not match number of tickers.")
    seen = set()
    for entry in rankings:
        if not isinstance(entry, dict):
            raise ValueError("Each ranking entry must be a dict.")
        for field in ['ticker', 'rank', 'confidence']:
            if field not in entry:
                raise ValueError(f"Ranking entry missing '{field}'.")
        t = entry['ticker']
        if t not in tickers:
            raise ValueError(f"Unknown ticker '{t}' in rankings.")
        if t in seen:
            raise ValueError(f"Duplicate ticker '{t}' in rankings.")
        seen.add(t)
        if not isinstance(entry['rank'], int) or entry['rank'] < 1:
            raise ValueError(f"Invalid rank for ticker '{t}'.")
        if not isinstance(entry['confidence'], (int, float)) or not (0.0 <= entry['confidence'] <= 1.0):
            raise ValueError(f"Invalid confidence for ticker '{t}'.")
    # Ensure ranks are contiguous
    ranks = sorted(entry['rank'] for entry in rankings)
    expected = list(range(1, len(tickers) + 1))
    if ranks != expected:
        raise ValueError("Ranks must be a permutation of 1..N.")


def aggregate_responses(
    responses: List[Tuple[BatchResponse, int]],
    batch_size: int,
) -> List[Dict[str, Any]]:
    """Aggregate batch responses across trials into final rankings.

    Args:
        responses: A list of tuples containing a BatchResponse and the size of
            the batch from which it was produced.  The batch size is needed to
            determine the top 20 percent threshold.
        batch_size: The nominal batch size used to generate requests.  This
            value is used when computing the top‑20 percent threshold.

    Returns:
        A list of dictionaries matching the AggregateOutput schema, sorted by
        ascending mean rank.
    """
    stats: Dict[str, Dict[str, Any]] = {}
    top_threshold = max(1, int(round(batch_size * 0.2)))
    for response, actual_size in responses:
        # Determine threshold for this batch: if actual_size differs, adjust
        threshold = max(1, int(round(actual_size * 0.2)))
        for entry in response.rankings:
            ticker = entry['ticker']
            rank = entry['rank']
            conf = entry['confidence']
            s = stats.setdefault(ticker, {'ranks': [], 'top_count': 0, 'notes': []})
            s['ranks'].append(rank)
            if rank <= threshold:
                s['top_count'] += 1
        # Append rationale bullets and red flags as notes
        notes = []
        if response.rationale_bullets:
            notes.extend(response.rationale_bullets)
        if response.red_flags:
            notes.extend([f"RED FLAG: {flag}" for flag in response.red_flags])
        for entry in response.rankings:
            ticker = entry['ticker']
            if notes:
                stats[ticker]['notes'].extend(notes)

    output: List[Dict[str, Any]] = []
    for ticker, s in stats.items():
        ranks = s['ranks']
        mean_rank = float(np.mean(ranks))
        std_rank = float(np.std(ranks, ddof=0))
        top20_rate = s['top_count'] / len(ranks) if ranks else 0.0
        notes = '; '.join(list(dict.fromkeys(s['notes'])))  # deduplicate order preserving
        exclusion_reason = ''
        if top20_rate < 0.5:
            exclusion_reason = 'low_top20_rate'
        output.append(
            {
                'ticker': ticker,
                'mean_rank': mean_rank,
                'std_rank': std_rank,
                'top20_rate': top20_rate,
                'notes': notes,
                'exclusion_reason': exclusion_reason,
            }
        )
    # Sort by mean rank
    output.sort(key=lambda x: x['mean_rank'])
    return output


def run_trials(
    csv_path: str,
    batch_size: int = 10,
    trials: int = 5,
    prompt_path: str = None,
    model: str = 'gpt-4o',
    temperature: float = 0.0,
    seed: Optional[int] = None,
    output_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Run multiple trials of batch ranking and return the aggregated results.

    Args:
        csv_path: Path to the input CSV of tickers.
        batch_size: Number of tickers per batch.
        trials: Number of random batching trials to perform.
        prompt_path: Path to the system prompt for ranking.  If None,
            defaults to `prompts/rank_listwise_v1.md` relative to this file.
        model: OpenAI model name.
        temperature: Sampling temperature for the LLM.
        seed: Optional global random seed.  If None, a random seed is generated.
        output_path: Optional path to write the aggregated output as CSV.

    Returns:
        A list of dictionaries conforming to `AggregateOutput`.
    """
    if prompt_path is None:
        prompt_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'prompts', 'rank_listwise_v1.md'
        )
    df = load_tickers(csv_path)
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    run_id = str(uuid.uuid4())
    all_responses: List[Tuple[BatchResponse, int]] = []
    for trial_id in range(trials):
        trial_seed = seed + trial_id
        batches = create_batches(df, batch_size, trial_seed)
        for batch_id, batch_df in enumerate(batches):
            request = BatchRequest(
                run_id=run_id,
                trial_id=trial_id,
                batch_id=batch_id,
                tickers=batch_df,
            )
            response = call_llm_rank_batch(
                request,
                prompt_path=prompt_path,
                model=model,
                temperature=temperature,
            )
            all_responses.append((response, len(batch_df)))
    aggregated = aggregate_responses(all_responses, batch_size=batch_size)
    if output_path:
        out_df = pd.DataFrame(aggregated)
        out_df.to_csv(output_path, index=False)
    return aggregated


def main():  # pragma: no cover
    import argparse
    parser = argparse.ArgumentParser(description='Run AI stock ranking trials.')
    parser.add_argument('csv_path', help='Path to input CSV of tickers and metrics')
    parser.add_argument('--batch-size', type=int, default=10, help='Number of tickers per batch')
    parser.add_argument('--trials', type=int, default=5, help='Number of random trials')
    parser.add_argument('--prompt', type=str, default=None, help='Path to system prompt file')
    parser.add_argument('--model', type=str, default='gpt-4o', help='OpenAI model name')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--output', type=str, default=None, help='Output CSV path for aggregated results')
    args = parser.parse_args()
    results = run_trials(
        csv_path=args.csv_path,
        batch_size=args.batch_size,
        trials=args.trials,
        prompt_path=args.prompt,
        model=args.model,
        temperature=args.temperature,
        seed=args.seed,
        output_path=args.output,
    )
    df_out = pd.DataFrame(results)
    print(df_out.to_string(index=False))


if __name__ == '__main__':  # pragma: no cover
    main()