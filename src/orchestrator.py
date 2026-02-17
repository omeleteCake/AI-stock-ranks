"""Runtime orchestrator for AI Stock Ranks.

This module wraps the lower-level ranking engine in `ranker.py` with a
single orchestration entrypoint that:
1) ingests either CSV metrics or a directory of per-ticker PDFs,
2) validates input data shape against the project contract,
3) runs multi-trial listwise ranking,
4) evaluates output stability and coverage,
5) persists run artifacts under `outputs/runs/`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    from .ranker import load_tickers, run_trials
except ImportError:  # pragma: no cover - enables running as a script
    from ranker import load_tickers, run_trials  # type: ignore

try:
    from .pdf_ingest import build_ticker_dataframe_from_pdf_root
except ImportError:  # pragma: no cover - enables running as a script
    from pdf_ingest import build_ticker_dataframe_from_pdf_root  # type: ignore


REQUIRED_TICKER_COLUMNS = [
    "ticker",
    "sector",
    "market_cap",
    "pe",
    "ev_ebitda",
    "debt_equity",
    "rev_growth",
    "fcf_margin",
    "price_momentum",
    "volatility",
    "timestamp",
    "source",
]


@dataclass
class OrchestratorConfig:
    """Configuration for one full ranking run."""

    input_path: str
    input_mode: str = "auto"
    batch_size: int = 10
    trials: int = 5
    prompt_path: Optional[str] = None
    model: str = "gpt-4o"
    temperature: float = 0.0
    seed: Optional[int] = None
    asof_timestamp: Optional[str] = None
    pdf_prompt_path: Optional[str] = None
    pdf_model: str = "gpt-4o-mini"
    pdf_temperature: float = 0.0
    max_pages_per_pdf: int = 40
    max_chars_per_ticker: int = 50000
    output_root: Optional[str] = None
    strict_eval: bool = False


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _default_output_root() -> str:
    return os.path.join(_project_root(), "outputs", "runs")


def _timestamp_label() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _resolve_input_mode(input_path: str, mode: str) -> str:
    normalized = mode.lower().strip()
    valid = {"auto", "csv", "pdf"}
    if normalized not in valid:
        raise ValueError(f"Invalid input mode '{mode}'. Expected one of {sorted(valid)}.")

    if normalized != "auto":
        return normalized

    if os.path.isfile(input_path) and input_path.lower().endswith(".csv"):
        return "csv"
    if os.path.isdir(input_path):
        return "pdf"
    raise ValueError(
        "Could not infer input mode from path. Use --input-mode csv|pdf explicitly."
    )


def validate_input_contract(df: pd.DataFrame) -> None:
    """Validate input rows against the repository data contract.

    Raises:
        ValueError: If contract checks fail.
    """
    missing = [c for c in REQUIRED_TICKER_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df["ticker"].isna().any():
        raise ValueError("Ticker column contains null values.")

    if not df["ticker"].is_unique:
        dupes = df["ticker"][df["ticker"].duplicated()].tolist()
        raise ValueError(f"Ticker column contains duplicates: {dupes}")

    timestamps = df["timestamp"].dropna().astype(str).unique().tolist()
    if len(timestamps) > 1:
        raise ValueError(
            "Input data has inconsistent timestamps. Expected exactly one timestamp value."
        )


def evaluate_run(
    aggregate_rows: List[Dict[str, Any]],
    input_tickers: List[str],
    batch_size: int,
) -> Dict[str, Any]:
    """Apply run-level checks derived from docs/EVAL.md."""
    issues: List[str] = []
    threshold = float(batch_size) / 4.0
    expected = set(input_tickers)

    output_tickers = [row["ticker"] for row in aggregate_rows]
    output_set = set(output_tickers)

    if len(output_tickers) != len(output_set):
        issues.append("Aggregated output contains duplicate tickers.")

    missing = sorted(expected - output_set)
    extra = sorted(output_set - expected)
    if missing:
        issues.append(f"Aggregated output is missing tickers: {missing}")
    if extra:
        issues.append(f"Aggregated output has unexpected tickers: {extra}")

    violators = [
        row["ticker"]
        for row in aggregate_rows
        if float(row.get("std_rank", 0.0)) > threshold
    ]
    if violators:
        issues.append(
            f"std_rank exceeded threshold {threshold:.3f} for tickers: {violators}"
        )

    return {
        "passed": len(issues) == 0,
        "std_rank_threshold": threshold,
        "issues": issues,
        "counts": {
            "input_tickers": len(input_tickers),
            "output_rows": len(aggregate_rows),
        },
    }


def run_orchestration(config: OrchestratorConfig) -> Dict[str, Any]:
    """Run one end-to-end ranking execution and persist artifacts."""
    input_path = os.path.abspath(config.input_path)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path not found: {input_path}")

    output_root = os.path.abspath(config.output_root or _default_output_root())
    run_dir = os.path.join(output_root, _timestamp_label())
    _mkdir(run_dir)

    input_mode = _resolve_input_mode(input_path, config.input_mode)
    source_csv_path = input_path
    derived_input_csv_path: Optional[str] = None
    input_df: pd.DataFrame

    if input_mode == "csv":
        if not os.path.isfile(input_path):
            raise ValueError(f"CSV mode requires a file path: {input_path}")
        input_df = load_tickers(input_path)
    else:
        if not os.path.isdir(input_path):
            raise ValueError(f"PDF mode requires a directory path: {input_path}")
        asof_timestamp = config.asof_timestamp or _iso_utc_now()
        input_df = build_ticker_dataframe_from_pdf_root(
            pdf_root=input_path,
            prompt_path=config.pdf_prompt_path,
            model=config.pdf_model,
            temperature=config.pdf_temperature,
            max_pages_per_pdf=config.max_pages_per_pdf,
            max_chars_per_ticker=config.max_chars_per_ticker,
            asof_timestamp=asof_timestamp,
        )
        input_df["timestamp"] = asof_timestamp
        derived_input_csv_path = os.path.join(run_dir, "input_from_pdf.csv")
        input_df.to_csv(derived_input_csv_path, index=False)
        source_csv_path = derived_input_csv_path

    if config.asof_timestamp:
        input_df["timestamp"] = config.asof_timestamp
        if input_mode == "csv":
            derived_input_csv_path = os.path.join(run_dir, "input_with_asof_timestamp.csv")
            input_df.to_csv(derived_input_csv_path, index=False)
            source_csv_path = derived_input_csv_path

    validate_input_contract(input_df)

    aggregate_csv_path = os.path.join(run_dir, "aggregate.csv")
    aggregate_json_path = os.path.join(run_dir, "aggregate.json")
    summary_json_path = os.path.join(run_dir, "summary.json")

    aggregate_rows = run_trials(
        csv_path=source_csv_path,
        batch_size=config.batch_size,
        trials=config.trials,
        prompt_path=config.prompt_path,
        model=config.model,
        temperature=config.temperature,
        seed=config.seed,
        output_path=aggregate_csv_path,
    )

    eval_report = evaluate_run(
        aggregate_rows=aggregate_rows,
        input_tickers=input_df["ticker"].tolist(),
        batch_size=config.batch_size,
    )

    with open(aggregate_json_path, "w", encoding="utf-8") as f:
        json.dump(aggregate_rows, f, indent=2)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_mode": input_mode,
        "input_path": input_path,
        "run_dir": run_dir,
        "artifacts": {
            "source_input_csv": source_csv_path,
            "derived_input_csv": derived_input_csv_path,
            "aggregate_csv": aggregate_csv_path,
            "aggregate_json": aggregate_json_path,
            "summary_json": summary_json_path,
        },
        "config": asdict(config),
        "evaluation": eval_report,
    }
    _write_json(summary_json_path, summary)

    if config.strict_eval and not eval_report["passed"]:
        raise RuntimeError(
            "Evaluation failed under --strict-eval. "
            f"Issues: {eval_report['issues']}"
        )

    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the AI Stock Ranks orchestration pipeline."
    )
    parser.add_argument(
        "input_path",
        help="Input CSV path or PDF root directory containing per-ticker documents.",
    )
    parser.add_argument(
        "--input-mode",
        type=str,
        default="auto",
        choices=["auto", "csv", "pdf"],
        help="Input mode: auto-detect, CSV, or PDF directory.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=10, help="Tickers per batch."
    )
    parser.add_argument(
        "--trials", type=int, default=5, help="Number of random batching trials."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Path to rank prompt; defaults to prompts/rank_listwise_v1.md.",
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o", help="Model name for ranking calls."
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Sampling temperature."
    )
    parser.add_argument("--seed", type=int, default=None, help="Global run seed.")
    parser.add_argument(
        "--asof-timestamp",
        type=str,
        default=None,
        help="Optional timestamp to enforce across all rows (ISO 8601 recommended).",
    )
    parser.add_argument(
        "--pdf-prompt",
        type=str,
        default=None,
        help="Prompt for extracting metrics from PDF text.",
    )
    parser.add_argument(
        "--pdf-model",
        type=str,
        default="gpt-4o-mini",
        help="Model name used for PDF-to-metrics extraction.",
    )
    parser.add_argument(
        "--pdf-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for PDF extraction calls.",
    )
    parser.add_argument(
        "--max-pages-per-pdf",
        type=int,
        default=40,
        help="Maximum number of pages to read from each PDF.",
    )
    parser.add_argument(
        "--max-chars-per-ticker",
        type=int,
        default=50000,
        help="Maximum combined extracted text size per ticker.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Output root for run artifacts (default: outputs/runs).",
    )
    parser.add_argument(
        "--strict-eval",
        action="store_true",
        help="Exit non-zero if evaluation checks fail.",
    )
    return parser


def main() -> None:  # pragma: no cover
    parser = _build_parser()
    args = parser.parse_args()
    config = OrchestratorConfig(
        input_path=args.input_path,
        input_mode=args.input_mode,
        batch_size=args.batch_size,
        trials=args.trials,
        prompt_path=args.prompt,
        model=args.model,
        temperature=args.temperature,
        seed=args.seed,
        asof_timestamp=args.asof_timestamp,
        pdf_prompt_path=args.pdf_prompt,
        pdf_model=args.pdf_model,
        pdf_temperature=args.pdf_temperature,
        max_pages_per_pdf=args.max_pages_per_pdf,
        max_chars_per_ticker=args.max_chars_per_ticker,
        output_root=args.output_root,
        strict_eval=args.strict_eval,
    )

    try:
        summary = run_orchestration(config)
    except Exception as exc:
        print(f"Orchestration failed: {exc}", file=sys.stderr)
        sys.exit(1)

    evaluation = summary["evaluation"]
    print(f"Run directory: {summary['run_dir']}")
    print(f"Evaluation passed: {evaluation['passed']}")
    if evaluation["issues"]:
        print("Issues:")
        for issue in evaluation["issues"]:
            print(f"- {issue}")


if __name__ == "__main__":  # pragma: no cover
    main()
